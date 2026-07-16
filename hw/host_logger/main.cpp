/*************************************************************************
* wave_logger - minimal SCAMP-5 packet logger for the travelling-wave
* autoencoder project (option B wrap-event readout).
*
* Connects to the vision system over USB (or to the Host GUI App / another
* interface app via its TCP proxy), decodes incoming packets with
* vs_packet_decoder, and appends every int32 data array to a binary log:
*
*   record = { int32 magic 'L5CS' (0x4C354353), int32 channel,
*              int32 rows, int32 cols, rows*cols x int32 payload }
*   little-endian throughout (ARM device and x86 host both are).
*
* Channels posted by src/scamp5_main.cpp (all vs_post_int32):
*   42  analog edge scan   { frame_id, 256 x packed 4 uint8 }
*   43  wrap events        { frame_id, count, count x ((x<<8)|y) }
*                          frame_id -1 = episode header, -2 = end marker
*   44  ground truth image { row_id, 64 x packed 4 uint8 } per row
*
* Parse the log with hw/scamp_log.py.
*
* Build: either copy the devkit example project "scamp5d_interface_app"
* (Codeblocks on Linux, Visual Studio on Windows) and replace its main.cpp
* with this file, or adjust DEVKIT in the Makefile next to this file.
*
* Usage:
*   ./wave_logger out.bin                     connect via USB
*   ./wave_logger out.bin 127.0.0.1 27888     connect via TCP (Host App proxy)
*
* Stop with Ctrl-C (the file is flushed after every record).
*************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <csignal>
#include <cstring>
#include <atomic>
#include <thread>
#include <chrono>

#include "scamp5d_usb.h"
#include "scamp5d_tcp.h"
#include "vs_packet_decoder.hpp"

static const int32_t LOG_MAGIC = 0x4C354353; // 'SC5L' little-endian

scamp5d_interface *box;
vs_packet_decoder *packet_switch;

std::atomic<bool> quit(false);
FILE *log_file = NULL;
uint64_t records_written = 0;

static void on_sigint(int){ quit = true; }

static void setup_packet_switch()
{
    // device printf's: episode progress, frame timing
    packet_switch->case_text([&](const char*text,size_t length){
        (void)length;
        printf("[device] %s",text);
    });

    // everything the wave kernel posts arrives here (channels 42/43/44)
    packet_switch->case_data_int32([&](const vs_array<int32_t>&data){
        int32_t channel = packet_switch->get_data_channel();
        int32_t rows = data.get_row_size();
        int32_t cols = data.get_col_size();
        int32_t hdr[4] = { LOG_MAGIC, channel, rows, cols };
        fwrite(hdr,sizeof(int32_t),4,log_file);
        for(int32_t r=0;r<rows;r++){
            for(int32_t c=0;c<cols;c++){
                int32_t v = data(r,c);
                fwrite(&v,sizeof(int32_t),1,log_file);
            }
        }
        fflush(log_file);
        records_written++;
        if((records_written & 1023) == 0)
            printf("[logger] %llu records\n",(unsigned long long)records_written);
    });
}

int main(int argc,char*argv[])
{
    if(argc < 2){
        fprintf(stderr,"usage: %s out.bin [tcp_ip [tcp_port]]\n",argv[0]);
        return -1;
    }

    log_file = fopen(argv[1],"wb");
    if(log_file == NULL){
        fprintf(stderr,"<Error: cannot open %s for writing>\n",argv[1]);
        return -1;
    }

    packet_switch = new vs_packet_decoder;
    setup_packet_switch();

    int r;
    if(argc >= 3){
        int port = (argc >= 4)? atoi(argv[3]) : 27888;
        box = new scamp5d_tcp();
        r = box->open(argv[2],port);
        printf("<Connecting via TCP %s:%d>\n",argv[2],port);
    }else{
        box = new scamp5d_usb();
        r = box->open("",-1);
        printf("<Connecting via USB>\n");
    }
    if(r){
        fprintf(stderr,"<Error: failed to open device!>\n");
        return -1;
    }
    printf("<Device Ready> logging to %s, Ctrl-C to stop\n",argv[1]);

    signal(SIGINT,on_sigint);

    box->on_receive_packet([&](const uint8_t*packet,size_t packet_size){
        packet_switch->decode_packet(packet,packet_size,box->get_packet_counter());
    });

    while(quit == false){
        box->routine();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    box->close();
    delete box;
    delete packet_switch;
    fclose(log_file);
    printf("\n[logger] done: %llu records -> %s\n",
           (unsigned long long)records_written,argv[1]);
    return 0;
}

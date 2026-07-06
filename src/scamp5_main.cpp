#include <scamp5.hpp>
#include <math.h>
using namespace SCAMP5_PE;

// ===========================================================================
// TRAVELLING WAVES on the 256x256 array  -  DIGITAL-STORAGE / ANALOG-COMPUTE
//
// The field state (displacement u, velocity v) is held DIGITALLY as 5-bit
// bit-planes in DREGs, so it does NOT decay between frames. Each frame we:
//    1. DAC the digital state into analog registers,
//    2. run ONE leapfrog wave step in the analog domain (fast, native),
//    3. inject a source,
//    4. ADC the result back to clean 5-bit digital.
// The ADC in step 4 re-quantises every frame, so analog noise / decay cannot
// accumulate - that is the whole point of going digital.
//
// Wave equation  u_tt = c^2 * Laplacian(u), leapfrog integrated:
//    L  = u_N + u_S + u_E + u_W - 4u          (5-point Laplacian)
//    v += k*L        (k = c^2*dt^2, here 1/8, safely within the CFL limit 1/2)
//    u += v
//
// Register map:
//   u : R1 R2 R3 R4 R5   (5-bit, bit4..bit0)   \ digital state, no decay
//   v : R6 R7 R8 R9 R10  (5-bit, bit4..bit0)   /
//   R11 : source-pixel mask     (R12,R0 free)
//   A : analog u , B : analog v , C/D/E/F : analog scratch  (NEWS used by movx)
// ===========================================================================

vs_stopwatch frame_timer;

// ---------------------------------------------------------------------------
// 5-bit ADC : analog "src" -> 5 DREG bit-planes (b4=MSB..b0=LSB).
// Direct extension of the library's adc_4bit SAR pattern with one more stage.
// Clobbers FLAG, t1, t2. Leaves src intact.
// ---------------------------------------------------------------------------
void adc_5bit(DREG b4,DREG b3,DREG b2,DREG b1,DREG b0, AREG src, AREG t1, AREG t2)
{
    scamp5_in(t2,-128);
    scamp5_kernel_begin();
        mov(t1,src);
        where(t1);  MOV(b4,FLAG);  add(t1,t1,t2);  all();
        sub(t1,t1,t2);
    scamp5_kernel_end();
    scamp5_in(t2,-64);
    scamp5_kernel_begin();
        where(t1,t2);  MOV(b3,FLAG);  add(t1,t1,t2);  all();
    scamp5_kernel_end();
    scamp5_in(t2,-32);
    scamp5_kernel_begin();
        where(t1,t2);  MOV(b2,FLAG);  add(t1,t1,t2);  all();
    scamp5_kernel_end();
    scamp5_in(t2,-16);
    scamp5_kernel_begin();
        where(t1,t2);  MOV(b1,FLAG);  add(t1,t1,t2);  all();
    scamp5_kernel_end();
    scamp5_in(t2,-8);
    scamp5_kernel_begin();
        where(t1,t2);  MOV(b0,FLAG);  all();
    scamp5_kernel_end();
}

// ---------------------------------------------------------------------------
// 5-bit DAC : 5 DREG bit-planes -> analog "dst". Extension of dac_4bit.
// Weighted sum: dst = -124 + 127*b4 + 64*b3 + 32*b2 + 16*b1 + 8*b0, so the
// mid code maps to ~analog 0 (the wave rest level). Uses t1 as scratch.
// ---------------------------------------------------------------------------
void dac_5bit(AREG dst, AREG t1, DREG b4,DREG b3,DREG b2,DREG b1,DREG b0)
{
    scamp5_in(dst,-124);
    scamp5_in(t1,127);  scamp5_kernel_begin(); WHERE(b4); add(dst,dst,t1); ALL(); scamp5_kernel_end();
    scamp5_in(t1,64);   scamp5_kernel_begin(); WHERE(b3); add(dst,dst,t1); ALL(); scamp5_kernel_end();
    scamp5_in(t1,32);   scamp5_kernel_begin(); WHERE(b2); add(dst,dst,t1); ALL(); scamp5_kernel_end();
    scamp5_in(t1,16);   scamp5_kernel_begin(); WHERE(b1); add(dst,dst,t1); ALL(); scamp5_kernel_end();
    scamp5_in(t1,8);    scamp5_kernel_begin(); WHERE(b0); add(dst,dst,t1); ALL(); scamp5_kernel_end();
}

int main()
{
    vs_init();

    //////////////////////////////////////////////////////////////////////////
    //SETUP IMAGE DISPLAYS
    int disp_size = 2;
    auto display_u = vs_gui_add_display("u  displacement",0,0,disp_size);
    auto display_v = vs_gui_add_display("v  velocity",0,disp_size,disp_size);

    // live scope: plots u at a probe pixel over time so you can watch the wave
    // arrive and pass through. scamp5_read_areg returns uint8_t (0..255).
    int graph_time_frame = 300;
    VS_GUI_DISPLAY_STYLE(style_plot,R"JSON(
    {
        "plot_palette": "plot_cmyw",
        "plot_palette_groups": 4
    }
    )JSON");
    auto display_scope = vs_gui_add_display("u @ probe (scope)",0,disp_size*2,disp_size,style_plot);
    // plot the value CENTRED on the rest level (measured at startup), so the wave
    // shows as a symmetric swing around 0 regardless of the readout's zero-offset.
    vs_gui_set_scope(display_scope,-127,127,graph_time_frame);   // (handle, min, max, time)

    //////////////////////////////////////////////////////////////////////////
    //SETUP GUI CONTROLS
    int source_r, source_c, drive_amp, drive_period, impulse, quiet;
    vs_gui_add_slider("source row: ",   0, 255, 128, &source_r);
    vs_gui_add_slider("source col: ",   0, 255, 128, &source_c);
    vs_gui_add_slider("drive amp: ",    0, 120, 100, &drive_amp);
    vs_gui_add_slider("drive period: ", 0, 200,  30, &drive_period); // frames/cycle, 0=off
    vs_gui_add_switch("impulse (ping)", 0, &impulse);
    vs_gui_add_switch("quiet (no drive)",0,&quiet);

    int probe_r, probe_c;
    vs_gui_add_slider("probe row: ", 0, 255, 128, &probe_r);
    vs_gui_add_slider("probe col: ", 0, 255, 128, &probe_c);   // start ON the source; move out to watch propagation

    //////////////////////////////////////////////////////////////////////////
    //INITIALISE the digital field to rest (u = 0, v = 0 everywhere)
    scamp5_in(A,0);
    scamp5_in(B,0);
    adc_5bit(R1,R2,R3,R4,R5, A, C, D);   // u := 0
    adc_5bit(R6,R7,R8,R9,R10,B, C, D);   // v := 0

    // measure the rest readout so the scope can be centred on it (A holds rest u)
    int rest_level = scamp5_read_areg(A,probe_r,probe_c);

    int impulse_prev = impulse;
    int t = 0;

    // Frame Loop
    while(1)
    {
        frame_timer.reset();
       	vs_disable_frame_trigger();
        vs_frame_loop_control();

        //////////////////////////////////////////////////////////////////////
        //1) DAC digital state -> analog  (A = u, B = v)
            dac_5bit(A, C, R1,R2,R3,R4,R5);
            dac_5bit(B, C, R6,R7,R8,R9,R10);

        //////////////////////////////////////////////////////////////////////
        //2) ONE leapfrog wave step in analog
        //   Laplacian is computed as the 4-neighbour AVERAGE (avg), which is
        //   Laplacian/4 = avg - u. Averaging keeps every intermediate in range:
        //   each neighbour is scaled to +/-30 (two halvings) BEFORE summing, so
        //   the running sum never overflows the +/-127 analog range.
            scamp5_kernel_begin();
                // C := average of the 4 neighbours of u(=A)
                movx(C,A,north);  divq(D,C);  divq(C,D);   // C = N/4
                movx(D,A,south);  divq(E,D);  divq(D,E);  add(C,C,D);  // += S/4
                movx(D,A,east);   divq(E,D);  divq(D,E);  add(C,C,D);  // += E/4
                movx(D,A,west);   divq(E,D);  divq(D,E);  add(C,C,D);  // C = avg

                // v += k*L  with k = 1/8  ->  k*L = (avg - u)/2
                sub(D,C,A);        // D = avg - u        (Laplacian/4)
                divq(E,D);         // E = (avg - u)/2 = k*L
                add(B,B,E);        // v += k*L

                // u += v
                add(A,A,B);        // u += v
            scamp5_kernel_end();

        //////////////////////////////////////////////////////////////////////
        //3) SOURCE: overwrite u at the source pixel with a drive value.
        //   Continuous sinusoid gives concentric travelling waves; the impulse
        //   switch drops a single full-amplitude ping.
            int drive = 0;
            bool inject = false;
            if(!quiet && drive_period > 0){
                float phase = 2.0f*3.14159265f*(float)t/(float)drive_period;
                drive = (int)(drive_amp*sinf(phase));
                inject = true;
            }
            if(impulse != impulse_prev){          // toggled -> one-frame ping
                impulse_prev = impulse;
                drive = drive_amp;
                inject = true;
            }
            if(inject){
                scamp5_load_point(R11,source_r,source_c);
                scamp5_in(C,drive);
                scamp5_kernel_begin();
                    WHERE(R11);
                        mov(A,C);                 // u(source) := drive
                    ALL();
                scamp5_kernel_end();
            }

        //////////////////////////////////////////////////////////////////////
        //4) ADC analog -> clean 5-bit digital (kills noise accumulation)
            adc_5bit(R1,R2,R3,R4,R5, A, C, D);    // store u
            adc_5bit(R6,R7,R8,R9,R10,B, C, D);    // store v

        //////////////////////////////////////////////////////////////////////
        //5) DISPLAY (A still holds u, B still holds v after the ADC)
            scamp5_output_image(A,display_u);
            scamp5_output_image(B,display_v);

            // plot u at the probe pixel, centred on the rest level (A still holds u)
            int16_t plotted_value[1];
            plotted_value[0] = (int16_t)(scamp5_read_areg(A,probe_r,probe_c) - rest_level);
            vs_post_set_channel(display_scope);
            vs_post_int16(plotted_value,1,1);

            int frame_us = frame_timer.get_usec();
            vs_post_text("t=%d  frame %d us  ~%d fps\n",t,frame_us,frame_us?1000000/frame_us:0);
            t++;
    }
    return 0;
}

#include <scamp5.hpp>
using namespace SCAMP5_PE;

// ===========================================================================
// COUPLED OSCILLATOR LATTICE on the 256x256 array  (multi-source interference)
//
// Every pixel is a mass connected by springs to its N/S/E/W neighbours AND
// (optionally) to ground - a coupled-pendulum / Klein-Gordon lattice:
//    L  = u_N + u_S + u_E + u_W - 4u          (5-point Laplacian = neighbour spring force)
//    v += k*L  -  Omega^2 * u                 (k=1/8 coupling; Omega^2 = ground stiffness)
//    u += v
// Ground spring OFF -> pure membrane (waves, but a pixel does not self-oscillate).
// Ground spring ON  -> every pixel is its own oscillator; that motion then
//                      spreads to its neighbours through the coupling.
//
// State (displacement u, velocity v) is held DIGITALLY as 5-bit bit-planes in
// DREGs so it does not decay. Each frame: DAC to analog -> one leapfrog step
// -> ADC back to clean digital (the ADC re-quantises, killing noise build-up).
//
// The lattice evolves freely every frame. There are FOUR disturbance sources;
// flipping a source's "fire" toggle pokes u at that location, launching a wave.
// Firing two or more sources makes their waves overlap into an interference
// pattern on the u display.
//
// Register map:
//   u : R1 R2 R3 R4 R5   (5-bit)   \ digital state, no decay
//   v : R6 R7 R8 R9 R10  (5-bit)   /
//   R11 : disturbance-region mask   (R12,R0 free)
//   A : analog u , B : analog v , C/D/E/F : analog scratch  (NEWS used by movx)
// ===========================================================================

vs_stopwatch frame_timer;

// 5-bit ADC : analog "src" -> 5 DREG bit-planes (b4=MSB..b0=LSB).
// Extension of the library adc_4bit SAR pattern. Clobbers FLAG,t1,t2; src intact.
void adc_5bit(DREG b4,DREG b3,DREG b2,DREG b1,DREG b0, AREG src, AREG t1, AREG t2)
{
    scamp5_in(t2,-128);
    scamp5_kernel_begin();
        mov(t1,src);
        where(t1);  MOV(b4,FLAG);  add(t1,t1,t2);  all();
        sub(t1,t1,t2);
    scamp5_kernel_end();
    scamp5_in(t2,-64);
    scamp5_kernel_begin(); where(t1,t2); MOV(b3,FLAG); add(t1,t1,t2); all(); scamp5_kernel_end();
    scamp5_in(t2,-32);
    scamp5_kernel_begin(); where(t1,t2); MOV(b2,FLAG); add(t1,t1,t2); all(); scamp5_kernel_end();
    scamp5_in(t2,-16);
    scamp5_kernel_begin(); where(t1,t2); MOV(b1,FLAG); add(t1,t1,t2); all(); scamp5_kernel_end();
    scamp5_in(t2,-8);
    scamp5_kernel_begin(); where(t1,t2); MOV(b0,FLAG); all(); scamp5_kernel_end();
}

// 5-bit DAC : 5 DREG bit-planes -> analog "dst". dst = -124 + 127*b4 + 64*b3 +
// 32*b2 + 16*b1 + 8*b0, so the mid code maps to ~analog 0 (rest). t1 is scratch.
void dac_5bit(AREG dst, AREG t1, DREG b4,DREG b3,DREG b2,DREG b1,DREG b0)
{
    scamp5_in(dst,-124);
    scamp5_in(t1,127); scamp5_kernel_begin(); WHERE(b4); add(dst,dst,t1); ALL(); scamp5_kernel_end();
    scamp5_in(t1,64);  scamp5_kernel_begin(); WHERE(b3); add(dst,dst,t1); ALL(); scamp5_kernel_end();
    scamp5_in(t1,32);  scamp5_kernel_begin(); WHERE(b2); add(dst,dst,t1); ALL(); scamp5_kernel_end();
    scamp5_in(t1,16);  scamp5_kernel_begin(); WHERE(b1); add(dst,dst,t1); ALL(); scamp5_kernel_end();
    scamp5_in(t1,8);   scamp5_kernel_begin(); WHERE(b0); add(dst,dst,t1); ALL(); scamp5_kernel_end();
}

// poke the analog u field (A) by "amp" over a square of half-width "hs" at (r,c)
void inject_disturbance(int r, int c, int hs, int amp)
{
    int r_top   = r-hs; if(r_top   < 0  ) r_top   = 0;
    int c_right = c+hs; if(c_right > 255) c_right = 255;
    int r_bot   = r+hs; if(r_bot   > 255) r_bot   = 255;
    int c_left  = c-hs; if(c_left  < 0  ) c_left  = 0;
    scamp5_load_rect(R11,r_top,c_left,r_bot,c_right);   // (row_min,col_min,row_max,col_max)
    scamp5_in(C,amp);
    scamp5_kernel_begin();
        WHERE(R11);
            mov(A,C);       // displace u at this source's region
        ALL();
    scamp5_kernel_end();
}

int main()
{
    vs_init();

    //////////////////////////////////////////////////////////////////////////
    //SETUP IMAGE DISPLAYS
    int disp_size = 2;
    auto display_u = vs_gui_add_display("u  displacement",0,0,disp_size);
    auto display_v = vs_gui_add_display("v  velocity",0,disp_size,disp_size);

    // scope: time-trace of u at ONE selected pixel (the probe), centred on rest
    // so the NEGATIVE half of the oscillation shows as a dip below the middle.
    int graph_time_frame = 300;
    VS_GUI_DISPLAY_STYLE(style_plot,R"JSON(
    {
        "plot_palette": "plot_cmyw",
        "plot_palette_groups": 4
    }
    )JSON");
    auto display_scope = vs_gui_add_display("u @ probe (scope)",0,disp_size*2,disp_size,style_plot);
    vs_gui_set_scope(display_scope,-127,127,graph_time_frame);

    //////////////////////////////////////////////////////////////////////////
    //SETUP GUI CONTROLS
    // shared poke strength/size
    int dist_amp, dist_size;
    vs_gui_add_slider("disturb amp: ",  0, 120, 120, &dist_amp);
    vs_gui_add_slider("disturb size: ", 0,  20,   6, &dist_size);   // half-width; 0 = a single pixel

    // four disturbance sources: each has a position and a "fire" toggle.
    // defaults: W / E / N / S of centre - fire two opposite ones for classic
    // two-source interference along their bisector.
    int src_r[4], src_c[4];
    vs_gui_add_slider("src1 row: ",0,255,128,&src_r[0]); vs_gui_add_slider("src1 col: ",0,255, 96,&src_c[0]);
    vs_gui_add_slider("src2 row: ",0,255,128,&src_r[1]); vs_gui_add_slider("src2 col: ",0,255,160,&src_c[1]);
    vs_gui_add_slider("src3 row: ",0,255, 96,&src_r[2]); vs_gui_add_slider("src3 col: ",0,255,128,&src_c[2]);
    vs_gui_add_slider("src4 row: ",0,255,160,&src_r[3]); vs_gui_add_slider("src4 col: ",0,255,128,&src_c[3]);

    int fire_all;
    vs_gui_add_switch("fire all",0,&fire_all);   // toggle -> all 4 sources fire in the same frame

    // ground spring: each pixel tied to rest, so it oscillates on its own.
    // Omega^2 = 2^-ground_shift  (bigger shift = softer spring = slower pixel bounce)
    int ground_on, ground_shift;
    vs_gui_add_switch("ground spring",  1, &ground_on);
    vs_gui_add_slider("ground shift: ", 0, 8, 3, &ground_shift);

    // the single pixel whose dynamics the scope plots.
    // default is ON src1 (128,96) so it actually sits on a disturbed pixel.
    int probe_r, probe_c;
    vs_gui_add_slider("probe row: ", 0, 255, 128, &probe_r);
    vs_gui_add_slider("probe col: ", 0, 255,  96, &probe_c);

    //////////////////////////////////////////////////////////////////////////
    //INITIALISE the digital field to rest (u = 0, v = 0)
    scamp5_in(A,0);
    scamp5_in(B,0);
    adc_5bit(R1,R2,R3,R4,R5, A, C, D);
    adc_5bit(R6,R7,R8,R9,R10,B, C, D);

    int rest_level = scamp5_read_areg(A,probe_r,probe_c);   // baseline for centring the scope
    int fire_all_prev = fire_all;
    int t = 0;

    // Frame Loop
    while(1)
    {
        frame_timer.reset();
       	vs_disable_frame_trigger();
        vs_frame_loop_control();

        //////////////////////////////////////////////////////////////////////
        //FREE EVOLUTION: one leapfrog step of the coupled lattice every frame
            dac_5bit(A, C, R1,R2,R3,R4,R5);      // A = u
            dac_5bit(B, C, R6,R7,R8,R9,R10);     // B = v

            scamp5_kernel_begin();
                // C := average of the 4 neighbours of u(=A); each neighbour is
                // scaled to +/-30 (two halvings) BEFORE summing to avoid overflow
                movx(C,A,north);  divq(D,C);  divq(C,D);              // C = N/4
                movx(D,A,south);  divq(E,D);  divq(D,E);  add(C,C,D); // += S/4
                movx(D,A,east);   divq(E,D);  divq(D,E);  add(C,C,D); // += E/4
                movx(D,A,west);   divq(E,D);  divq(D,E);  add(C,C,D); // C = avg

                // v += k*L  with k = 1/8  ->  k*L = (avg - u)/2
                sub(D,C,A);        // D = avg - u   (Laplacian/4 = neighbour spring force/4)
                divq(E,D);         // E = (avg-u)/2 = k*L
                add(B,B,E);        // v += k*L      (neighbour coupling only, so far)
            scamp5_kernel_end();

            // GROUND SPRING: v -= Omega^2 * u , with Omega^2 = 2^-ground_shift.
            // Applied to v BEFORE u += v (uses the old u, held in A).
            if(ground_on){
                scamp5_kernel_begin(); mov(C,A); scamp5_kernel_end();      // C = u
                for(int s=0;s<ground_shift;s++){
                    scamp5_kernel_begin(); divq(D,C); mov(C,D); scamp5_kernel_end();  // C /= 2
                }
                scamp5_kernel_begin(); sub(B,B,C); scamp5_kernel_end();    // v -= Omega^2 * u
            }

            scamp5_kernel_begin(); add(A,A,B); scamp5_kernel_end();        // u += v

        //////////////////////////////////////////////////////////////////////
        //DISTURB: fire any source whose toggle changed (injected after the step)
            if(fire_all != fire_all_prev){
                fire_all_prev = fire_all;
                for(int i=0;i<4;i++)
                    inject_disturbance(src_r[i],src_c[i],dist_size,dist_amp);
            }

            adc_5bit(R1,R2,R3,R4,R5, A, C, D);   // store u
            adc_5bit(R6,R7,R8,R9,R10,B, C, D);   // store v

        //////////////////////////////////////////////////////////////////////
        //DISPLAY (A holds u, B holds v)
            scamp5_output_image(A,display_u);
            scamp5_output_image(B,display_v);

            // time-trace of the probe pixel's displacement, centred on rest so
            // you can see it swing positive AND negative
            int16_t probe_val[1];
            probe_val[0] = (int16_t)(scamp5_read_areg(A,probe_r,probe_c) - rest_level);
            vs_post_set_channel(display_scope);
            vs_post_int16(probe_val,1,1);

            int frame_us = frame_timer.get_usec();
            vs_post_text("t=%d  running  frame %d us\n",t,frame_us);
            t++;
    }
    return 0;
}

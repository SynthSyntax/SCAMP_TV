#include <scamp5.hpp>
using namespace SCAMP5_PE;

// ===========================================================================
// PHASE-OSCILLATOR LATTICE (Kuramoto) - fully in the ANALOG domain
//
// Each pixel holds a PHASE theta (register A) that constantly ramps and wraps
// around - a rotating phasor e^(i*theta). Because only the ANGLE is stored and
// the phasor length is fixed at 1, there is no amplitude to decay: analog leak
// can only nudge the phase (a little frequency error), it can NEVER kill the
// oscillation. That is the whole point of this representation.
//
//   theta_i  +=  omega_i            (rotate; omega = base + intensity*gain)
//   theta_i  +=  K*(avg_neighbour_theta - theta_i)   (weak coupling -> sync)
//   wrap theta back into range when it passes the top
//
// omega depends on the light INTENSITY at each pixel (brighter -> faster), so
// this is the Kuramoto model with an intensity-set frequency map: similar-
// brightness regions lock into synchrony (oscillatory-correlation segmentation).
//
// Registers:
//   A : theta (phase state)         C,D,E,F : analog scratch
//   B : captured intensity (displayed early, then reused as the wrap
//       threshold in the coupling step)
//   NEWS : routing register, clobbered internally by every analog macro
//          (mov/add/sub/movx/diva) - never usable as a data buffer.
//   All divide-by-two use diva (accurate, in-place, needs 2 scratch regs)
//   rather than divq (quick but low precision).
// ===========================================================================

vs_stopwatch frame_timer;

// ===========================================================================
// EDGE READOUT (the "analog autoencoder" tap)
//
// Reading the full 256x256 array off-chip is the expensive operation SCAMP is
// built to avoid. Instead we read ONLY the 4 border lines of the phase field
// (4x256 = 1024 px = 1.6% of the array) every frame. The travelling waves set
// up by the Kuramoto coupling carry interior structure outward, so the border
// time-series is a temporal code for the whole image: a host-side RNN decoder
// (sim/wave_autoencoder.py) is trained to invert it back into the image.
//
// Packets go out on a dedicated raw channel: per frame, one int32 (frame id)
// followed by a 4x256 int8 array, rows = [north, south, west, east] border.
// scamp5_scan_areg returns uint8 (analog value + 128); we post it raw and let
// the host subtract 128.
// ===========================================================================
const uint32_t CH_EDGE_DATA = 42;      // raw-packet tag for the host logger
uint8_t edge_buf[4][256];

static void post_edges()
{
    scamp5_scan_areg(A, edge_buf[0],   0,   0,   0, 255, 1, 1);  // north row
    scamp5_scan_areg(A, edge_buf[1], 255,   0, 255, 255, 1, 1);  // south row
    scamp5_scan_areg(A, edge_buf[2],   0,   0, 255,   0, 1, 1);  // west  col
    scamp5_scan_areg(A, edge_buf[3],   0, 255, 255, 255, 1, 1);  // east  col
}

int main()
{
    vs_init();

    //////////////////////////////////////////////////////////////////////////
    //DISPLAYS
    int disp_size = 2;
    auto display_phase = vs_gui_add_display("theta (phase)",0,0,disp_size);
    auto display_int   = vs_gui_add_display("intensity",0,disp_size,disp_size);

    // scope of one pixel's phase over time: a SAWTOOTH (ramp up, snap down) =
    // the phasor rotating. Faster ramp = higher frequency = brighter pixel.
    VS_GUI_DISPLAY_STYLE(style_plot,R"JSON(
    {
        "plot_palette": "plot_cmyw",
        "plot_palette_groups": 4
    }
    )JSON");
    auto display_scope = vs_gui_add_display("theta @ probe",0,disp_size*2,disp_size,style_plot);
    vs_gui_set_scope(display_scope,0,255,300);

    //////////////////////////////////////////////////////////////////////////
    //CONTROLS
    int base_freq, freq_gain, couple, probe_r, probe_c, edge_readout;
    vs_gui_add_slider("base freq: ", 0, 30,  4, &base_freq);   // phase step/frame for a dark pixel
    vs_gui_add_slider("freq gain: ", 1,  8,  4, &freq_gain);   // intensity->freq (halvings; bigger=weaker); min 1 so the +offset fits in int8
    vs_gui_add_slider("coupling: ",  0,  6,  0, &couple);      // 0=off; diffusive sync strength (halvings)
    vs_gui_add_slider("probe row: ", 0,255,128, &probe_r);
    vs_gui_add_slider("probe col: ", 0,255,128, &probe_c);
    vs_gui_add_switch("edge readout", false, &edge_readout);   // stream border phase for the decoder

    //////////////////////////////////////////////////////////////////////////
    //INIT: all phases start at 0
    scamp5_kernel_begin(); res(A); scamp5_kernel_end();

    int t = 0;

    // Frame Loop
    while(1)
    {
        frame_timer.reset();
       	vs_disable_frame_trigger();
        vs_frame_loop_control();

        //////////////////////////////////////////////////////////////////////
        //1) capture light intensity into B, and display it NOW: step 4
        //   reuses B as a constant, so it must be shown before then
            scamp5_kernel_begin(); get_image(B,F); scamp5_kernel_end();
            scamp5_output_image(B,display_int);     // the intensity / frequency map

        //////////////////////////////////////////////////////////////////////
        //2) frequency omega -> C :  C = base_freq + (intensity+128)/2^freq_gain
        //   get_image gives SIGNED intensity (dark ~ -100), so shift it positive,
        //   AFTER the halving (halving first keeps the offset from saturating).
        //   This guarantees omega > 0: every pixel rotates FORWARD (a negative
        //   omega would ramp dark pixels down onto the rail and freeze them).
            scamp5_in(D,base_freq);
            scamp5_kernel_begin(); mov(C,B); scamp5_kernel_end();
            for(int s=0;s<freq_gain;s++){ scamp5_kernel_begin(); diva(C,E,F); scamp5_kernel_end(); }   // C /= 2 in place (diva > divq accuracy; E,F scratch)
            scamp5_in(E,128>>freq_gain);   // offset so darkest pixel maps to ~0
            scamp5_kernel_begin(); add(C,C,E); add(C,C,D); scamp5_kernel_end();   // C = omega > 0

        //////////////////////////////////////////////////////////////////////
        //3) advance the phase:  theta += omega
            scamp5_kernel_begin(); add(A,A,C); scamp5_kernel_end();

        //////////////////////////////////////////////////////////////////////
        //4) optional weak coupling: theta += K*avg_n( wrap(theta_n - theta) )
        //   Phase lives on a CIRCLE, so each neighbour difference must be
        //   wrapped into [-60,60) BEFORE use: +59 vs -59 are 2 apart, not
        //   -118. Coupling on the raw linear difference kicks wrap-straddling
        //   pixels the long way round, shearing the field into noise at every
        //   turn. Wrapped-linear = sawtooth coupling, the analog-friendly
        //   stand-in for Kuramoto's sin(). B is free here (already displayed)
        //   and holds the wrap threshold (-60); one turn (120) is applied as
        //   TWO adds of B, which keeps F free as scratch for diva (the
        //   accurate divide-by-two needs two scratch registers, unlike divq).
            if(couple>0){
                scamp5_in(B,-60);            // wrap threshold; 2x B = one turn
                scamp5_kernel_begin();
                    res(C);                  // C accumulates the mean wrapped difference

                    movx(E,A,north); sub(D,E,A);                // D = theta_n - theta
                    where(D,B); add(D,D,B); add(D,D,B); all();  // D > +60 : D -= 120
                    neg(E,D);
                    where(E,B); sub(D,D,B); sub(D,D,B); all();  // D < -60 : D += 120
                    diva(D,E,F); diva(D,E,F); add(C,C,D);       // C += D/4

                    movx(E,A,south); sub(D,E,A);
                    where(D,B); add(D,D,B); add(D,D,B); all();
                    neg(E,D);
                    where(E,B); sub(D,D,B); sub(D,D,B); all();
                    diva(D,E,F); diva(D,E,F); add(C,C,D);

                    movx(E,A,east);  sub(D,E,A);
                    where(D,B); add(D,D,B); add(D,D,B); all();
                    neg(E,D);
                    where(E,B); sub(D,D,B); sub(D,D,B); all();
                    diva(D,E,F); diva(D,E,F); add(C,C,D);

                    movx(E,A,west);  sub(D,E,A);
                    where(D,B); add(D,D,B); add(D,D,B); all();
                    neg(E,D);
                    where(E,B); sub(D,D,B); sub(D,D,B); all();
                    diva(D,E,F); diva(D,E,F); add(C,C,D);       // C = avg wrapped diff
                scamp5_kernel_end();
                for(int s=0;s<couple;s++){ scamp5_kernel_begin(); diva(C,D,E); scamp5_kernel_end(); }
                scamp5_kernel_begin(); add(A,A,C); scamp5_kernel_end();  // theta += K*avg wrapped diff
            }

        //////////////////////////////////////////////////////////////////////
        //5) WRAP both ends: one turn = 120 units, theta nominally in [-60,60).
        //   (turn = 120, not 128: constants at +/-128 sit on the DAC/rail limit
        //    and mis-load, causing phase slip at every wrap.)
        //   Analog registers SATURATE at the rail instead of wrapping, so wrap
        //   by hand - and in BOTH directions, because the coupling kick can
        //   also push theta below the bottom of the range.
            scamp5_in(D,-60);
            scamp5_in(E,-120);
            scamp5_kernel_begin();
                where(A,D);       // FLAG := theta > +60
                    add(A,A,E);   // theta -= 120  (wrap down)
                all();
            scamp5_kernel_end();
            scamp5_in(E,120);
            scamp5_kernel_begin();
                neg(F,A);         // F = -theta   (F is free after get_image)
                where(F,D);       // FLAG := -theta - 60 > 0  <=>  theta < -60
                    add(A,A,E);   // theta += 120  (wrap up)
                all();
            scamp5_kernel_end();

        //////////////////////////////////////////////////////////////////////
        //OUTPUT
            //edge readout: the decoder's input. Scan the 4 border lines of
            //theta and post them raw; the interior never leaves the chip.
            if(edge_readout){
                post_edges();
                vs_post_set_channel(CH_EDGE_DATA);
                int32_t frame_id = t;
                vs_post_int32(&frame_id,1,1);
                vs_post_int8((const int8_t*)edge_buf,4,256);
            }

            scamp5_output_image(A,display_phase);   // phase field (synced regions share a shade)

            int16_t pv[1];
            pv[0] = (int16_t)scamp5_read_areg(A,probe_r,probe_c);
            vs_post_set_channel(display_scope);
            vs_post_int16(pv,1,1);                  // sawtooth = the rotation

            int frame_us = frame_timer.get_usec();
            vs_post_text("t=%d  frame %d us\n",t,frame_us);
            t++;
    }
    return 0;
}

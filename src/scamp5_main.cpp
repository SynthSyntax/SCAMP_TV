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

// ===========================================================================
// EVENT READOUT (option B - spike-time coding, sim/wave_events.py)
//
// The wrap step already computes, in FLAG, exactly the pixels whose phase
// crosses +60 this frame - that flip IS the event. We latch FLAG into a
// DREG (R11), keep only the 4 border lines (interior mask R10, rebuilt every
// frame because DREGs leak), and read the set pixels out with the chip's
// sparse address-event scan. Nothing analog is read at all. Expected rate at
// default settings: background wrap period ~23 frames over 1020 border px
// -> ~44 events/frame ~ 90 bytes, vs 1024 analog samples for the edge scan;
// and the digital sparse scan is far cheaper than the per-pixel analog ADC.
//
// ALL packets on channels 42/43/44 are int32 arrays (vs_post_int32): the
// documented host-side decoder callback is case_data_int32 + get_data_channel
// (hw/host_logger); an int8 handler is not in the scamp5d interface docs.
// Byte payloads are packed 4-per-word big-endian-in-word, so wire size is
// unchanged.
//
// Channel 43, one packet per frame: { frame_id, count, count x ((x<<8)|y) }.
//   frame_id -1 = episode header {-1, n_frames, base_freq, freq_gain, couple, 0}
//   frame_id -2 = episode end marker
// Channel 44: ground truth = captured intensity B, row by row: { row_id,
// 64 x packed 4 pixels } (same right-to-left scan quirk as the edges),
// posted at BOTH episode start and end - the host discards episodes where
// the two differ (scene moved / slideshow flipped mid-episode).
// Channel 42 (analog edge scan): { frame_id, 256 x packed 4 samples } =
// the 4x256 border rows [north, south, west, east].
//
// One-time calibration on real silicon: scan_events' (x,y) convention and
// scan direction are unverified - record one episode with BOTH "edge
// readout" and "event readout" on, and match wraps in the analog trace to
// event coordinates.
// ===========================================================================
const uint32_t CH_EVENT_DATA = 43;
const uint32_t CH_GT_IMAGE   = 44;
#define EV_MAX 1024                    // worst case: the t=0 synchronized wrap fires all 1020 border px
uint8_t ev_buf[EV_MAX*2];              // raw (x,y) pairs from scan_events
int32_t ev_pkt[2+EV_MAX];              // packed packet: frame_id, count, coords
int32_t edge_pkt[1+256];               // packed analog edge packet

static inline int32_t pack4(const uint8_t*p)
{
    return ((int32_t)p[0]<<24)|((int32_t)p[1]<<16)|((int32_t)p[2]<<8)|(int32_t)p[3];
}

static void post_ground_truth()
{
    static uint8_t row_buf[256];
    static int32_t row_pkt[1+64];
    vs_post_set_channel(CH_GT_IMAGE);
    for(int r=0;r<256;r++){
        scamp5_scan_areg(B, row_buf, r, 0, r, 255, 1, 1);
        row_pkt[0] = r;
        for(int i=0;i<64;i++) row_pkt[1+i] = pack4(row_buf+4*i);
        vs_post_int32(row_pkt,1,65);
    }
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
    int event_readout, episode, auto_episodes, ep_frames;
    vs_gui_add_slider("base freq: ", 0, 30,  4, &base_freq);   // phase step/frame for a dark pixel
    vs_gui_add_slider("freq gain: ", 1,  8,  4, &freq_gain);   // intensity->freq (halvings; bigger=weaker); min 1 so the +offset fits in int8
    vs_gui_add_slider("coupling: ",  0,  6,  0, &couple);      // 0=off; diffusive sync strength (halvings)
    vs_gui_add_slider("probe row: ", 0,255,128, &probe_r);
    vs_gui_add_slider("probe col: ", 0,255,128, &probe_c);
    vs_gui_add_switch("edge readout", false, &edge_readout);   // stream border phase for the decoder
    vs_gui_add_switch("event readout", false, &event_readout); // stream border wrap events (option B), no episode framing
    vs_gui_add_switch("record episode", false, &episode);      // toggle (either way) = header + ground truth + phase reset + one episode
    vs_gui_add_switch("auto episodes", false, &auto_episodes); // restart back-to-back for bulk collection (starts as soon as it's on)
    vs_gui_add_slider("ep frames x256:", 1, 32, 16, &ep_frames); // episode length / 256 (default 16 -> 4096 frames, the n=256 budget)

    //////////////////////////////////////////////////////////////////////////
    //INIT: all phases start at 0
    scamp5_kernel_begin(); res(A); scamp5_kernel_end();

    int t = 0;
    int ep_prev = 0;      // edge-detect for the "record episode" toggle
    int ep_left = 0;      // frames remaining in the running episode (0 = idle)
    int pending_end = 0;  // 1 = post end marker + end ground truth next frame,
                          // once B holds a fresh capture again (the coupling
                          // step clobbers B mid-frame, so end GT can't be read
                          // in the OUTPUT section - it would be the -60 wrap
                          // constant, not the image)
    int pending_start = 0;// 1 = start the episode next frame. Exposure here is
                          // the inter-frame time (no frame trigger), and
                          // displays dominate it - so a GT captured on the
                          // toggle frame (long, displays on) is brighter than
                          // every in-episode frame (short, displays off).
                          // Arming for one display-less frame first makes the
                          // GT exposure match the episode's omega map.

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
            if(ep_left == 0 && !pending_start)      // displays off while recording (and
                scamp5_output_image(B,display_int); // while arming): image output
                                                    // dominates frame time = exposure

        //////////////////////////////////////////////////////////////////////
        //1b) episode control. MUST run here: B still holds the fresh capture
        //    (step 4 reuses B as the wrap constant). On a "record episode"
        //    toggle - or continuously in auto mode - post the config header
        //    and the ground-truth image, reset every phase to 0 (the decoder
        //    is trained on episodes that start from theta = 0, like the sim),
        //    and stream events for the next ep_frames*256 frames.
            if(pending_end){                         // close the PREVIOUS episode:
                int32_t end_hdr[2] = {-2, 0};        // end marker + a fresh end GT
                vs_post_set_channel(CH_EVENT_DATA);  // (B holds this frame's capture,
                vs_post_int32(end_hdr,1,2);          // so it's a real image again;
                post_ground_truth();                 // if the scene moved during the
                vs_post_text("episode done\n");      // episode, it won't match the start)
                pending_end = 0;
            }
            if(pending_start){                       // B was captured after a
                pending_start = 0;                   // display-less frame: exposure
                                                     // now matches the episode
                int32_t hdr[6] = {-1, ep_frames*256, base_freq, freq_gain, couple, 0};
                vs_post_set_channel(CH_EVENT_DATA);
                vs_post_int32(hdr,1,6);
                post_ground_truth();                 // the scene this episode encodes
                scamp5_kernel_begin(); res(A); scamp5_kernel_end();   // theta := 0
                t = 0;
                ep_left = ep_frames*256;
            }
            if(episode != ep_prev || (auto_episodes && ep_left == 0 && pending_end == 0)){
                ep_prev = episode;
                pending_start = 1;                   // arm: one display-less frame first
            }

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
        //   REPLICATE BOUNDARY (measured necessity, not a nicety): beyond the
        //   physical array edge, movx reads garbage/zero. With the episode-
        //   start reset synchronising the field, that phantom neighbour's
        //   pull cancels omega and PINS the whole border below the wrap
        //   threshold (hw trace: 0.45 wraps/pixel/256f instead of ~10). So
        //   each direction's edge line replaces the missing neighbour with
        //   the pixel itself (diff = 0, term drops out) - the same as the
        //   sim's --boundary replicate. R12 holds the line mask; the analog
        //   mov IS FLAG-gated (unlike DREG ops), so WHERE works here.
        //   If a trace still shows pinning after this, the movx direction
        //   convention is mirrored: swap row 0<->255 and col 0<->255 below.
        #define COUPLE_DIR(DIR, r0,c0,r1,c1) \
            scamp5_load_rect(R12, r0, c0, r1, c1); \
            scamp5_kernel_begin(); \
                movx(E,A,DIR); \
                WHERE(R12); mov(E,A); all();                /* replicate at the edge line */ \
                sub(D,E,A);                                 /* D = theta_n - theta */ \
                where(D,B); add(D,D,B); add(D,D,B); all();  /* D > +60 : D -= 120 */ \
                neg(E,D); \
                where(E,B); sub(D,D,B); sub(D,D,B); all();  /* D < -60 : D += 120 */ \
                diva(D,E,F); diva(D,E,F); add(C,C,D);       /* C += D/4 */ \
            scamp5_kernel_end();

            if(couple>0){
                scamp5_in(B,-60);            // wrap threshold; 2x B = one turn
                scamp5_kernel_begin();
                    res(C);                  // C accumulates the mean wrapped difference
                scamp5_kernel_end();
                COUPLE_DIR(north,   0,  0,   0,255)   // row 0 has no north neighbour
                COUPLE_DIR(south, 255,  0, 255,255)   // row 255 has no south neighbour
                COUPLE_DIR(east,    0,255, 255,255)   // col 255 has no east neighbour
                COUPLE_DIR(west,    0,  0, 255,  0)   // col 0 has no west neighbour
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
            //The forward wrap IS the event (option B): latch FLAG into R11
            //before wrapping down, then erase interior pixels so only the 4
            //border lines can fire. The interior mask R10 is REBUILT every
            //frame - DREGs leak/flip over time, and a mask loaded once at
            //startup silently corrupts the event stream minutes in. The
            //backward wrap below stays event-less on purpose: it matches the
            //sim's falling-edge detector (d < -HALF), which only sees
            //forward wraps.
            scamp5_load_rect(R10, 1, 1, 254, 254);   // interior = rows/cols 1..254
            scamp5_in(D,-60);
            scamp5_in(E,-120);
            scamp5_kernel_begin();
                all();
                CLR(R11);          // fresh latch (insurance for either gating semantics)
                where(A,D);        // FLAG := theta > +60
                    MOV(R11,FLAG); //   R11 := pixels wrapping THIS frame = the events
                    add(A,A,E);    //   theta -= 120  (wrap down; analog IS FLAG-gated)
                all();
                // Border masking with PURE LOGIC. Measured on silicon:
                // DREG writes DO NOT honour FLAG - a CLR(R11) under
                // WHERE(R10) cleared every pixel, killing all events (only a
                // stuck defective DREG cell survived). So compute
                // R11 := R11 AND NOT interior with unconditional ops:
                NOT(R12,R11);      // R12 := not-an-event
                NOR(R11,R12,R10);  // R11 := NOT(no-event OR interior) = event AND border
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
                edge_pkt[0] = t;
                const uint8_t*e = (const uint8_t*)edge_buf;
                for(int i=0;i<256;i++) edge_pkt[1+i] = pack4(e+4*i);
                vs_post_set_channel(CH_EDGE_DATA);
                vs_post_int32(edge_pkt,1,257);
            }

            //event readout: sparse address-event scan of the wrap latch.
            //Measured on silicon: scan_events ZERO-FILLS every unused buffer
            //slot (a prefilled sentinel does not survive), so the count is
            //recovered by trimming (0,0) pairs from the back. Caveat: a real
            //event at scan-coordinate (0,0) that lands adjacent to the filler
            //is absorbed - at most one corner event per frame, ignorable.
            if(event_readout || ep_left > 0){
                scamp5_scan_events(R11, ev_buf, EV_MAX);
                int cnt = EV_MAX;
                while(cnt > 0 && ev_buf[2*cnt-2]==0 && ev_buf[2*cnt-1]==0) cnt--;
                ev_pkt[0] = t;
                ev_pkt[1] = cnt;
                for(int i=0;i<cnt;i++)
                    ev_pkt[2+i] = ((int32_t)ev_buf[2*i]<<8)|(int32_t)ev_buf[2*i+1];
                vs_post_set_channel(CH_EVENT_DATA);
                vs_post_int32(ev_pkt,1,2+cnt);
            }

            //episode countdown; close with the end marker and a SECOND
            //ground truth, so the host can reject any episode whose scene
            //changed mid-run (start and end captures won't match).
            if(ep_left > 0){
                ep_left--;
                if((ep_left & 1023) == 0 && ep_left > 0)
                    vs_post_text("episode: %d frames left\n",ep_left);
                if(ep_left == 0) pending_end = 1;   // flush end marker + GT next
                                                    // frame (needs a fresh B)
            }

            if(ep_left == 0 && !pending_start){   // displays + probe off while
                scamp5_output_image(A,display_phase);   // recording or arming (see 1)

                int16_t pv[1];
                pv[0] = (int16_t)scamp5_read_areg(A,probe_r,probe_c);
                vs_post_set_channel(display_scope);
                vs_post_int16(pv,1,1);                  // sawtooth = the rotation

                int frame_us = frame_timer.get_usec();
                vs_post_text("t=%d  frame %d us\n",t,frame_us);
            }
            t++;
    }
    return 0;
}

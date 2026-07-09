#include <scamp5.hpp>
using namespace SCAMP5_PE;

// ===========================================================================
// PULSE-COUPLED (FIREFLY) OSCILLATOR LATTICE - fully digital state & coupling
//
// Each pixel is a firefly: a 7-BIT phase counter (0..127) in DREGs that ticks
// up at a light-set rate and FIRES (flashes) when it wraps 127 -> 0. The wrap
// carry-out IS the fire event - the hardware gives it to us for free.
//
// Coupling (Mirollo-Strogatz): when any N/S/E/W neighbour fired last frame,
// and my own phase is in the LATE half of the cycle (MSB set), I advance one
// extra tick - leaping toward my own firing. Late pixels get dragged into
// firing together -> avalanches -> visible PROPAGATING FIRING WAVES.
//
// Why this architecture is right for this chip:
//   * phase counters : digital -> exact forever (no decay, no noise, free wrap)
//   * fire events    : the increment's carry-out (digital, free)
//   * neighbour fired: DNEWS0 digital OR of 4 neighbours (no analog compare,
//                      no wrap-seam ambiguity - the flash IS the reference)
//   * analog is only used STATELESSLY: the intensity dither compare and the
//     display DAC. Nothing analog ever persists a frame -> nothing accumulates.
//
// Tick rate = frequency ∝ intensity, via temporal dither: TWO compares per
// frame against a 30-level threshold sweep (coprime order), plus a guaranteed
// base tick every "base period" frames so dark pixels never freeze.
//
// Register map (all 13 DREGs in use):
//   theta : R11 R10 R9 R8 R7 R6 R5   (7-bit counter, MSB..LSB)
//   R12 : "a neighbour fired last frame" memory (survives the frame)
//   R0  : tick mask, then fired mask (transient within the frame)
//   R1-R4 : scratch for the ripple adder; double as DNEWS direction selects
//   A : theta DAC'd for display/scope   B : intensity   C : compare constant
// ===========================================================================

vs_stopwatch frame_timer;

// ---------------------------------------------------------------------------
// One ripple-carry stage: BIT,carry(R2) -> BIT^carry, BIT&carry. Native ops
// only (NOT/NOR/MOV). Scratch R1,R3,R4. Every theta bit is rewritten every
// frame, which doubles as the DREG refresh.
// ---------------------------------------------------------------------------
#define INC_BIT(BIT)                          \
        NOT(R3,BIT);                          \
        NOT(R4,R2);                           \
        NOR(R1,R3,R4);      /* R1 = b&c   */  \
        NOR(R3,BIT,R2);     /* R3 = ~(b|c)*/  \
        NOR(BIT,R1,R3);     /* b  = b^c   */  \
        MOV(R2,R1);         /* carry= b&c */

// theta += 1 where carry-in R2 == 1, mod 128. Leaves fired mask in R2.
#define INC_THETA()                           \
        INC_BIT(R5);       /* LSB */          \
        INC_BIT(R6);                          \
        INC_BIT(R7);                          \
        INC_BIT(R8);                          \
        INC_BIT(R9);                          \
        INC_BIT(R10);                         \
        INC_BIT(R11);      /* MSB: carry-out in R2 = FIRED (wrapped) */

// ---------------------------------------------------------------------------
// DAC the 7-bit counter to analog A in [-63,+64] for display & the scope.
// ---------------------------------------------------------------------------
void dac_theta()
{
    scamp5_in(A,-63);
    scamp5_in(C,64); scamp5_kernel_begin(); WHERE(R11); add(A,A,C); ALL(); scamp5_kernel_end();
    scamp5_in(C,32); scamp5_kernel_begin(); WHERE(R10); add(A,A,C); ALL(); scamp5_kernel_end();
    scamp5_in(C,16); scamp5_kernel_begin(); WHERE(R9);  add(A,A,C); ALL(); scamp5_kernel_end();
    scamp5_in(C, 8); scamp5_kernel_begin(); WHERE(R8);  add(A,A,C); ALL(); scamp5_kernel_end();
    scamp5_in(C, 4); scamp5_kernel_begin(); WHERE(R7);  add(A,A,C); ALL(); scamp5_kernel_end();
    scamp5_in(C, 2); scamp5_kernel_begin(); WHERE(R6);  add(A,A,C); ALL(); scamp5_kernel_end();
    scamp5_in(C, 1); scamp5_kernel_begin(); WHERE(R5);  add(A,A,C); ALL(); scamp5_kernel_end();
}

int main()
{
    vs_init();

    //////////////////////////////////////////////////////////////////////////
    //DISPLAYS
    int disp_size = 2;
    auto display_phase = vs_gui_add_display("theta (phase)",0,0,disp_size);
    auto display_fire  = vs_gui_add_display("FIRED (flashes)",0,disp_size,disp_size);
    auto display_int   = vs_gui_add_display("intensity",0,disp_size*2,disp_size);

    VS_GUI_DISPLAY_STYLE(style_plot,R"JSON(
    {
        "plot_palette": "plot_cmyw",
        "plot_palette_groups": 4
    }
    )JSON");
    auto display_scope = vs_gui_add_display("theta @ probe",0,disp_size*3,disp_size,style_plot);
    vs_gui_set_scope(display_scope,0,255,300);

    //////////////////////////////////////////////////////////////////////////
    //CONTROLS
    int base_period, couple, probe_r, probe_c;
    vs_gui_add_slider("base period: ", 1, 64,  8, &base_period); // forced tick every N frames
    vs_gui_add_switch("coupling",      1, &couple);              // firefly pulse coupling
    vs_gui_add_slider("probe row: ",   0,255,128, &probe_r);
    vs_gui_add_slider("probe col: ",   0,255,128, &probe_c);

    //////////////////////////////////////////////////////////////////////////
    //INIT: counters, fire memory, masks all zero
    scamp5_kernel_begin();
        CLR(R5,R6,R7,R8);
        CLR(R9,R10,R11,R12);
        CLR(R0);
    scamp5_kernel_end();

    int t = 0;

    // Frame Loop
    while(1)
    {
        frame_timer.reset();
       	vs_disable_frame_trigger();
        vs_frame_loop_control();

        //////////////////////////////////////////////////////////////////////
        //1) capture intensity (fresh ground truth every frame)
            scamp5_kernel_begin(); get_image(B,F); scamp5_kernel_end();

        //////////////////////////////////////////////////////////////////////
        //2) TICK 1 mask -> R0 :  dither compare  OR  base tick  OR  coupling
        //   dither: threshold sweeps 30 levels over [-116,+116] in coprime
        //   order (step 11 mod 30 -> every level exactly once per 15 frames,
        //   spread maximally in time). Ticks/frame ∝ brightness.
            int thrA = -116 + 8*(((2*t  )*11)%30);
            int thrB = -116 + 8*(((2*t+1)*11)%30);

            scamp5_in(C,(int8_t)(-thrA));
            scamp5_kernel_begin();
                where(B,C);          // FLAG := intensity > thrA
                MOV(R0,FLAG);
                all();
            scamp5_kernel_end();

        //   base tick: every pixel ticks every "base period" frames (floor rate)
            if(t % base_period == 0){
                scamp5_kernel_begin(); SET(R0); scamp5_kernel_end(); }

        //   FIREFLY COUPLING: a neighbour fired last frame (R12) AND I am in
        //   the late half of my cycle (MSB R11 set) -> take an extra tick,
        //   leaping toward my own firing. Late pixels bunch up and fire
        //   together; the resulting avalanches propagate as firing waves.
            if(couple){
                scamp5_kernel_begin();
                    NOT(R1,R12);
                    NOT(R2,R11);
                    NOR(R3,R1,R2);   // R3 = nb_fired AND late_half
                    OR(R0,R0,R3);
                scamp5_kernel_end();
            }

        //////////////////////////////////////////////////////////////////////
        //3) increment 1:  theta += 1 where R0 ; capture who wrapped (fired)
            scamp5_kernel_begin();
                MOV(R2,R0);          // carry-in = tick mask
                INC_THETA();
                MOV(R0,R2);          // R0 = fired mask (tick mask now dead)
            scamp5_kernel_end();

        //////////////////////////////////////////////////////////////////////
        //4) TICK 2 (second dither compare) + increment 2; OR its fires into R0
            scamp5_in(C,(int8_t)(-thrB));
            scamp5_kernel_begin();
                where(B,C);
                MOV(R2,FLAG);        // carry-in = tick 2 mask, directly in R2
                all();
                INC_THETA();
                OR(R0,R0,R2);        // R0 = fired in either increment
            scamp5_kernel_end();

        //////////////////////////////////////////////////////////////////////
        //5) neighbour-fired memory for NEXT frame:
        //   R12 := OR of the 4 neighbours' fired (digital, boundary reads 0)
            scamp5_kernel_begin();
                SET(RS,RW,RN,RE);    // select all four directions (R1-R4)
                DNEWS0(R12,R0);      // R12 = any neighbour fired this frame
            scamp5_kernel_end();

        //////////////////////////////////////////////////////////////////////
        //OUTPUT
            dac_theta();                              // A = theta (display only)
            scamp5_output_image(A,display_phase);     // phase field
            scamp5_output_image(R0,display_fire);     // the flashes - watch waves here
            scamp5_output_image(B,display_int);       // intensity / frequency map

            int16_t pv[1];
            pv[0] = (int16_t)scamp5_read_areg(A,probe_r,probe_c);
            vs_post_set_channel(display_scope);
            vs_post_int16(pv,1,1);                    // staircase sawtooth

            int frame_us = frame_timer.get_usec();
            vs_post_text("t=%d frame %d us\n",t,frame_us);
            t++;
    }
    return 0;
}

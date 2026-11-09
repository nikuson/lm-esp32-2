// mv_pie3.S — matrix x vector kernels on the ESP32-S3 PIE (128-bit SIMD) unit.

    .section .iram1, "ax"
    .align  4

// void mv_s8_pie(const int8_t *W, const int8_t *x, int32_t *acc, int rows, int n16)
//   int8 rows of n16*16 bytes, one int32 dot product per row.
    .global mv_s8_pie
    .type   mv_s8_pie, @function
mv_s8_pie:
    entry   a1, 32
    beqz    a5, 9f
    addi    a6, a6, -1
1:
    mov     a7, a3
    ee.zero.accx
    ee.vld.128.ip   q0, a2, 16
    ee.vld.128.ip   q1, a7, 16
    loopnez a6, 2f
    ee.vmulas.s8.accx.ld.ip  q0, a2, 16, q0, q1
    ee.vld.128.ip   q1, a7, 16
2:
    ee.vmulas.s8.accx  q0, q1
    nop
    nop
    rur.accx_0  a8
    s32i    a8, a4, 0
    addi    a4, a4, 4
    addi    a5, a5, -1
    bnez    a5, 1b
9:
    retw.n
    .size   mv_s8_pie, . - mv_s8_pie

// void mv_q4_pie(const uint8_t *W, const int8_t *x, int32_t *gacc, int rows, int nblk, int bpg)
//   4-bit weights, 32 per 16-byte block: byte j holds w[j] (low nibble) and w[j+16] (high nibble).
//   No unpacking to int8: (b & 0xF0) is 16*w_hi as a signed byte, ((b << 4) & 0xF0) is 16*w_lo.
//   ee.vsl.32 shifts whole 32-bit lanes, the bits that leak into the neighbour byte are masked off.
//   One int32 per group of bpg blocks (the 16x factor is folded into the group scale).
    .global mv_q4_pie
    .type   mv_q4_pie, @function
mv_q4_pie:
    entry   a1, 48
    beqz    a5, 9f
    movi    a8, 0xF0                    // build 0xF0F0F0F0 and load it into all 4 lanes of q7
    slli    a9, a8, 8
    or      a8, a8, a9
    slli    a9, a8, 16
    or      a8, a8, a9
    s32i    a8, a1, 16
    s32i    a8, a1, 20
    s32i    a8, a1, 24
    s32i    a8, a1, 28
    addi    a9, a1, 16
    ee.vld.128.ip   q7, a9, 0
    movi    a8, 4
    wsr.sar a8
1:  // row
    mov     a10, a3                     // x
    mov     a11, a6                     // blocks left in the row
2:  // group
    ee.zero.accx
    loopnez a7, 3f
    ee.vld.128.ip   q0, a2, 16          // 32 packed weights
    ee.vld.128.ip   q1, a10, 16         // x[0..15]
    ee.vsl.32       q2, q0
    ee.andq         q3, q0, q7          // 16*w[16..31]
    ee.andq         q2, q2, q7          // 16*w[0..15]
    ee.vmulas.s8.accx  q2, q1
    ee.vld.128.ip   q1, a10, 16         // x[16..31]
    ee.vmulas.s8.accx  q3, q1
3:
    nop
    nop
    rur.accx_0  a8
    s32i    a8, a4, 0
    addi    a4, a4, 4
    sub     a11, a11, a7
    bnez    a11, 2b
    addi    a5, a5, -1
    bnez    a5, 1b
9:
    retw.n
    .size   mv_q4_pie, . - mv_q4_pie

// void mem_read_pie(const void *p, int n64)   — streams n64*64 bytes through the 128-bit load port
    .global mem_read_pie
    .type   mem_read_pie, @function
mem_read_pie:
    entry   a1, 32
    loopnez a3, 1f
    ee.vld.128.ip   q0, a2, 16
    ee.vld.128.ip   q1, a2, 16
    ee.vld.128.ip   q2, a2, 16
    ee.vld.128.ip   q3, a2, 16
1:
    retw.n
    .size   mem_read_pie, . - mem_read_pie

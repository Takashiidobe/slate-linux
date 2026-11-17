#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "div64.rs"]
mod __slate_unit_div64;
#[path = "gcd.rs"]
mod __slate_unit_gcd;
#[path = "lcm.rs"]
mod __slate_unit_lcm;
#[path = "int_log.rs"]
mod __slate_unit_int_log;
#[path = "int_pow.rs"]
mod __slate_unit_int_pow;
#[path = "int_sqrt.rs"]
mod __slate_unit_int_sqrt;
#[path = "reciprocal_div.rs"]
mod __slate_unit_reciprocal_div;
#[path = "rational.rs"]
mod __slate_unit_rational;
#[path = "crc16.rs"]
mod __slate_unit_crc16;
#[path = "crc_ccitt.rs"]
mod __slate_unit_crc_ccitt;
#[path = "crc32_main.rs"]
mod __slate_unit_crc32_main;
#[path = "datastore.rs"]
mod __slate_unit_datastore;
#[path = "inffast.rs"]
mod __slate_unit_inffast;
#[path = "inflate.rs"]
mod __slate_unit_inflate;
#[path = "infutil.rs"]
mod __slate_unit_infutil;
#[path = "inftrees.rs"]
mod __slate_unit_inftrees;
#[path = "inflate_syms.rs"]
mod __slate_unit_inflate_syms;
#[path = "deflate.rs"]
mod __slate_unit_deflate;
#[path = "deftree.rs"]
mod __slate_unit_deftree;
#[path = "deflate_syms.rs"]
mod __slate_unit_deflate_syms;
#[path = "lzo1x_compress.rs"]
mod __slate_unit_lzo1x_compress;
#[path = "lzo1x_compress_safe.rs"]
mod __slate_unit_lzo1x_compress_safe;
#[path = "lzo1x_decompress_safe.rs"]
mod __slate_unit_lzo1x_decompress_safe;
#[path = "lz4_compress.rs"]
mod __slate_unit_lz4_compress;
#[path = "lz4_decompress.rs"]
mod __slate_unit_lz4_decompress;
#[path = "zstd_decompress_module.rs"]
mod __slate_unit_zstd_decompress_module;
#[path = "huf_decompress.rs"]
mod __slate_unit_huf_decompress;
#[path = "zstd_ddict.rs"]
mod __slate_unit_zstd_ddict;
#[path = "zstd_decompress.rs"]
mod __slate_unit_zstd_decompress;
#[path = "zstd_decompress_block.rs"]
mod __slate_unit_zstd_decompress_block;
#[path = "zstd_common_module.rs"]
mod __slate_unit_zstd_common_module;
#[path = "debug.rs"]
mod __slate_unit_debug;
#[path = "entropy_common.rs"]
mod __slate_unit_entropy_common;
#[path = "error_private.rs"]
mod __slate_unit_error_private;
#[path = "fse_decompress.rs"]
mod __slate_unit_fse_decompress;
#[path = "zstd_common.rs"]
mod __slate_unit_zstd_common;
#[path = "xz_dec_syms.rs"]
mod __slate_unit_xz_dec_syms;
#[path = "xz_dec_stream.rs"]
mod __slate_unit_xz_dec_stream;
#[path = "xz_dec_lzma2.rs"]
mod __slate_unit_xz_dec_lzma2;
#[path = "xz_dec_bcj.rs"]
mod __slate_unit_xz_dec_bcj;
#[path = "dim.rs"]
mod __slate_unit_dim;
#[path = "net_dim.rs"]
mod __slate_unit_net_dim;
#[path = "rdma_dim.rs"]
mod __slate_unit_rdma_dim;
#[path = "fonts.rs"]
mod __slate_unit_fonts;
#[path = "font_8x16.rs"]
mod __slate_unit_font_8x16;

mod __slate_panic {
    unsafe extern "C" {
        fn panic(fmt: *const core::ffi::c_char, ...) -> !;
    }

    #[panic_handler]
    fn __slate_panic_handler(_: &core::panic::PanicInfo) -> ! {
        unsafe { panic(c"rust panic\n".as_ptr()) }
    }

    #[linkage = "weak"]
    #[unsafe(no_mangle)]
    extern "C" fn rust_eh_personality() {}
}


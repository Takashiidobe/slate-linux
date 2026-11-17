#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "hash_info.rs"]
mod __slate_unit_hash_info;
#[path = "memneq.rs"]
mod __slate_unit_memneq;
#[path = "utils.rs"]
mod __slate_unit_utils;
#[path = "aes.rs"]
mod __slate_unit_aes;
#[path = "arc4.rs"]
mod __slate_unit_arc4;
#[path = "blake2s.rs"]
mod __slate_unit_blake2s;
#[path = "chacha_block_generic.rs"]
mod __slate_unit_chacha_block_generic;
#[path = "gf128hash.rs"]
mod __slate_unit_gf128hash;
#[path = "md5.rs"]
mod __slate_unit_md5;
#[path = "sha1.rs"]
mod __slate_unit_sha1;
#[path = "sha256.rs"]
mod __slate_unit_sha256;
#[path = "sha512.rs"]
mod __slate_unit_sha512;
#[path = "generic_mpih_lshift.rs"]
mod __slate_unit_generic_mpih_lshift;
#[path = "generic_mpih_mul1.rs"]
mod __slate_unit_generic_mpih_mul1;
#[path = "generic_mpih_mul2.rs"]
mod __slate_unit_generic_mpih_mul2;
#[path = "generic_mpih_mul3.rs"]
mod __slate_unit_generic_mpih_mul3;
#[path = "generic_mpih_rshift.rs"]
mod __slate_unit_generic_mpih_rshift;
#[path = "generic_mpih_sub1.rs"]
mod __slate_unit_generic_mpih_sub1;
#[path = "generic_mpih_add1.rs"]
mod __slate_unit_generic_mpih_add1;
#[path = "mpicoder.rs"]
mod __slate_unit_mpicoder;
#[path = "mpi_add.rs"]
mod __slate_unit_mpi_add;
#[path = "mpi_bit.rs"]
mod __slate_unit_mpi_bit;
#[path = "mpi_cmp.rs"]
mod __slate_unit_mpi_cmp;
#[path = "mpi_sub_ui.rs"]
mod __slate_unit_mpi_sub_ui;
#[path = "mpi_div.rs"]
mod __slate_unit_mpi_div;
#[path = "mpi_mod.rs"]
mod __slate_unit_mpi_mod;
#[path = "mpi_mul.rs"]
mod __slate_unit_mpi_mul;
#[path = "mpih_cmp.rs"]
mod __slate_unit_mpih_cmp;
#[path = "mpih_div.rs"]
mod __slate_unit_mpih_div;
#[path = "mpih_mul.rs"]
mod __slate_unit_mpih_mul;
#[path = "mpi_pow.rs"]
mod __slate_unit_mpi_pow;
#[path = "mpiutil.rs"]
mod __slate_unit_mpiutil;

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


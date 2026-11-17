#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "api.rs"]
mod __slate_unit_api;
#[path = "cipher.rs"]
mod __slate_unit_cipher;
#[path = "algapi.rs"]
mod __slate_unit_algapi;
#[path = "scatterwalk.rs"]
mod __slate_unit_scatterwalk;
#[path = "proc.rs"]
mod __slate_unit_proc;
#[path = "aead.rs"]
mod __slate_unit_aead;
#[path = "geniv.rs"]
mod __slate_unit_geniv;
#[path = "lskcipher.rs"]
mod __slate_unit_lskcipher;
#[path = "skcipher.rs"]
mod __slate_unit_skcipher;
#[path = "seqiv.rs"]
mod __slate_unit_seqiv;
#[path = "echainiv.rs"]
mod __slate_unit_echainiv;
#[path = "ahash.rs"]
mod __slate_unit_ahash;
#[path = "shash.rs"]
mod __slate_unit_shash;
#[path = "akcipher.rs"]
mod __slate_unit_akcipher;
#[path = "sig.rs"]
mod __slate_unit_sig;
#[path = "rsa.rs"]
mod __slate_unit_rsa;
#[path = "rsa_helper.rs"]
mod __slate_unit_rsa_helper;
#[path = "rsa_pkcs1pad.rs"]
mod __slate_unit_rsa_pkcs1pad;
#[path = "rsassa_pkcs1.rs"]
mod __slate_unit_rsassa_pkcs1;
#[path = "acompress.rs"]
mod __slate_unit_acompress;
#[path = "scompress.rs"]
mod __slate_unit_scompress;
#[path = "algboss.rs"]
mod __slate_unit_algboss;
#[path = "testmgr.rs"]
mod __slate_unit_testmgr;
#[path = "cmac.rs"]
mod __slate_unit_cmac;
#[path = "hmac.rs"]
mod __slate_unit_hmac;
#[path = "sha1.rs"]
mod __slate_unit_sha1;
#[path = "sha256.rs"]
mod __slate_unit_sha256;
#[path = "sha512.rs"]
mod __slate_unit_sha512;
#[path = "ecb.rs"]
mod __slate_unit_ecb;
#[path = "cbc.rs"]
mod __slate_unit_cbc;
#[path = "cts.rs"]
mod __slate_unit_cts;
#[path = "ctr.rs"]
mod __slate_unit_ctr;
#[path = "gcm.rs"]
mod __slate_unit_gcm;
#[path = "ccm.rs"]
mod __slate_unit_ccm;
#[path = "aes.rs"]
mod __slate_unit_aes;
#[path = "camellia_generic.rs"]
mod __slate_unit_camellia_generic;
#[path = "authenc.rs"]
mod __slate_unit_authenc;
#[path = "authencesn.rs"]
mod __slate_unit_authencesn;
#[path = "krb5enc.rs"]
mod __slate_unit_krb5enc;
#[path = "lzo.rs"]
mod __slate_unit_lzo;
#[path = "lzo_rle.rs"]
mod __slate_unit_lzo_rle;
#[path = "lz4.rs"]
mod __slate_unit_lz4;
#[path = "asymmetric_type.rs"]
mod __slate_unit_asymmetric_type;
#[path = "restrict.rs"]
mod __slate_unit_restrict;
#[path = "signature.rs"]
mod __slate_unit_signature;
#[path = "public_key.rs"]
mod __slate_unit_public_key;
#[path = "x509_cert_parser.rs"]
mod __slate_unit_x509_cert_parser;
#[path = "x509_loader.rs"]
mod __slate_unit_x509_loader;
#[path = "x509_public_key.rs"]
mod __slate_unit_x509_public_key;
#[path = "pkcs7_parser.rs"]
mod __slate_unit_pkcs7_parser;
#[path = "pkcs7_trust.rs"]
mod __slate_unit_pkcs7_trust;
#[path = "pkcs7_verify.rs"]
mod __slate_unit_pkcs7_verify;
#[path = "krb5_kdf.rs"]
mod __slate_unit_krb5_kdf;
#[path = "krb5_api.rs"]
mod __slate_unit_krb5_api;
#[path = "rfc3961_simplified.rs"]
mod __slate_unit_rfc3961_simplified;
#[path = "rfc3962_aes.rs"]
mod __slate_unit_rfc3962_aes;
#[path = "rfc6803_camellia.rs"]
mod __slate_unit_rfc6803_camellia;
#[path = "rfc8009_aes2.rs"]
mod __slate_unit_rfc8009_aes2;

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


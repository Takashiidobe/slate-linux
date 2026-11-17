#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "tbxfroot.rs"]
mod __slate_unit_tbxfroot;
#[path = "utaddress.rs"]
mod __slate_unit_utaddress;
#[path = "utalloc.rs"]
mod __slate_unit_utalloc;
#[path = "utascii.rs"]
mod __slate_unit_utascii;
#[path = "utbuffer.rs"]
mod __slate_unit_utbuffer;
#[path = "utcksum.rs"]
mod __slate_unit_utcksum;
#[path = "utcopy.rs"]
mod __slate_unit_utcopy;
#[path = "utexcep.rs"]
mod __slate_unit_utexcep;
#[path = "utdebug.rs"]
mod __slate_unit_utdebug;
#[path = "utdecode.rs"]
mod __slate_unit_utdecode;
#[path = "utdelete.rs"]
mod __slate_unit_utdelete;
#[path = "uterror.rs"]
mod __slate_unit_uterror;
#[path = "uteval.rs"]
mod __slate_unit_uteval;
#[path = "utglobal.rs"]
mod __slate_unit_utglobal;
#[path = "uthex.rs"]
mod __slate_unit_uthex;
#[path = "utids.rs"]
mod __slate_unit_utids;
#[path = "utinit.rs"]
mod __slate_unit_utinit;
#[path = "utlock.rs"]
mod __slate_unit_utlock;
#[path = "utmath.rs"]
mod __slate_unit_utmath;
#[path = "utmisc.rs"]
mod __slate_unit_utmisc;
#[path = "utmutex.rs"]
mod __slate_unit_utmutex;
#[path = "utnonansi.rs"]
mod __slate_unit_utnonansi;
#[path = "utobject.rs"]
mod __slate_unit_utobject;
#[path = "utosi.rs"]
mod __slate_unit_utosi;
#[path = "utownerid.rs"]
mod __slate_unit_utownerid;
#[path = "utpredef.rs"]
mod __slate_unit_utpredef;
#[path = "utresdecode.rs"]
mod __slate_unit_utresdecode;
#[path = "utresrc.rs"]
mod __slate_unit_utresrc;
#[path = "utstate.rs"]
mod __slate_unit_utstate;
#[path = "utstring.rs"]
mod __slate_unit_utstring;
#[path = "utstrsuppt.rs"]
mod __slate_unit_utstrsuppt;
#[path = "utstrtoul64.rs"]
mod __slate_unit_utstrtoul64;
#[path = "utxface.rs"]
mod __slate_unit_utxface;
#[path = "utxfinit.rs"]
mod __slate_unit_utxfinit;
#[path = "utxferror.rs"]
mod __slate_unit_utxferror;
#[path = "utxfmutex.rs"]
mod __slate_unit_utxfmutex;

mod __slate_panic {
    #[panic_handler]
    fn __slate_panic_handler(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }

    #[linkage = "weak"]
    #[unsafe(no_mangle)]
    extern "C" fn rust_eh_personality() {}
}


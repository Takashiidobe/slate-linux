#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "dsargs.rs"]
mod __slate_unit_dsargs;
#[path = "dscontrol.rs"]
mod __slate_unit_dscontrol;
#[path = "dsdebug.rs"]
mod __slate_unit_dsdebug;
#[path = "dsfield.rs"]
mod __slate_unit_dsfield;
#[path = "dsinit.rs"]
mod __slate_unit_dsinit;
#[path = "dsmethod.rs"]
mod __slate_unit_dsmethod;
#[path = "dsmthdat.rs"]
mod __slate_unit_dsmthdat;
#[path = "dsobject.rs"]
mod __slate_unit_dsobject;
#[path = "dsopcode.rs"]
mod __slate_unit_dsopcode;
#[path = "dspkginit.rs"]
mod __slate_unit_dspkginit;
#[path = "dsutils.rs"]
mod __slate_unit_dsutils;
#[path = "dswexec.rs"]
mod __slate_unit_dswexec;
#[path = "dswload.rs"]
mod __slate_unit_dswload;
#[path = "dswload2.rs"]
mod __slate_unit_dswload2;
#[path = "dswscope.rs"]
mod __slate_unit_dswscope;
#[path = "dswstate.rs"]
mod __slate_unit_dswstate;
#[path = "evevent.rs"]
mod __slate_unit_evevent;
#[path = "evgpe.rs"]
mod __slate_unit_evgpe;
#[path = "evgpeblk.rs"]
mod __slate_unit_evgpeblk;
#[path = "evgpeinit.rs"]
mod __slate_unit_evgpeinit;
#[path = "evgpeutil.rs"]
mod __slate_unit_evgpeutil;
#[path = "evglock.rs"]
mod __slate_unit_evglock;
#[path = "evhandler.rs"]
mod __slate_unit_evhandler;
#[path = "evmisc.rs"]
mod __slate_unit_evmisc;
#[path = "evregion.rs"]
mod __slate_unit_evregion;
#[path = "evrgnini.rs"]
mod __slate_unit_evrgnini;
#[path = "evsci.rs"]
mod __slate_unit_evsci;
#[path = "evxface.rs"]
mod __slate_unit_evxface;
#[path = "evxfevnt.rs"]
mod __slate_unit_evxfevnt;
#[path = "evxfgpe.rs"]
mod __slate_unit_evxfgpe;
#[path = "evxfregn.rs"]
mod __slate_unit_evxfregn;
#[path = "exconcat.rs"]
mod __slate_unit_exconcat;
#[path = "exconfig.rs"]
mod __slate_unit_exconfig;
#[path = "exconvrt.rs"]
mod __slate_unit_exconvrt;
#[path = "excreate.rs"]
mod __slate_unit_excreate;
#[path = "exdebug.rs"]
mod __slate_unit_exdebug;
#[path = "exdump.rs"]
mod __slate_unit_exdump;
#[path = "exfield.rs"]
mod __slate_unit_exfield;
#[path = "exfldio.rs"]
mod __slate_unit_exfldio;
#[path = "exmisc.rs"]
mod __slate_unit_exmisc;
#[path = "exmutex.rs"]
mod __slate_unit_exmutex;
#[path = "exnames.rs"]
mod __slate_unit_exnames;
#[path = "exoparg1.rs"]
mod __slate_unit_exoparg1;
#[path = "exoparg2.rs"]
mod __slate_unit_exoparg2;
#[path = "exoparg3.rs"]
mod __slate_unit_exoparg3;
#[path = "exoparg6.rs"]
mod __slate_unit_exoparg6;
#[path = "exprep.rs"]
mod __slate_unit_exprep;
#[path = "exregion.rs"]
mod __slate_unit_exregion;
#[path = "exresnte.rs"]
mod __slate_unit_exresnte;
#[path = "exresolv.rs"]
mod __slate_unit_exresolv;
#[path = "exresop.rs"]
mod __slate_unit_exresop;
#[path = "exserial.rs"]
mod __slate_unit_exserial;
#[path = "exstore.rs"]
mod __slate_unit_exstore;
#[path = "exstoren.rs"]
mod __slate_unit_exstoren;
#[path = "exstorob.rs"]
mod __slate_unit_exstorob;
#[path = "exsystem.rs"]
mod __slate_unit_exsystem;
#[path = "extrace.rs"]
mod __slate_unit_extrace;
#[path = "exutils.rs"]
mod __slate_unit_exutils;
#[path = "hwacpi.rs"]
mod __slate_unit_hwacpi;
#[path = "hwesleep.rs"]
mod __slate_unit_hwesleep;
#[path = "hwgpe.rs"]
mod __slate_unit_hwgpe;
#[path = "hwregs.rs"]
mod __slate_unit_hwregs;
#[path = "hwsleep.rs"]
mod __slate_unit_hwsleep;
#[path = "hwvalid.rs"]
mod __slate_unit_hwvalid;
#[path = "hwxface.rs"]
mod __slate_unit_hwxface;
#[path = "hwxfsleep.rs"]
mod __slate_unit_hwxfsleep;
#[path = "hwpci.rs"]
mod __slate_unit_hwpci;
#[path = "nsaccess.rs"]
mod __slate_unit_nsaccess;
#[path = "nsalloc.rs"]
mod __slate_unit_nsalloc;
#[path = "nsarguments.rs"]
mod __slate_unit_nsarguments;
#[path = "nsconvert.rs"]
mod __slate_unit_nsconvert;
#[path = "nsdump.rs"]
mod __slate_unit_nsdump;
#[path = "nseval.rs"]
mod __slate_unit_nseval;
#[path = "nsinit.rs"]
mod __slate_unit_nsinit;
#[path = "nsload.rs"]
mod __slate_unit_nsload;
#[path = "nsnames.rs"]
mod __slate_unit_nsnames;
#[path = "nsobject.rs"]
mod __slate_unit_nsobject;
#[path = "nsparse.rs"]
mod __slate_unit_nsparse;
#[path = "nspredef.rs"]
mod __slate_unit_nspredef;
#[path = "nsprepkg.rs"]
mod __slate_unit_nsprepkg;
#[path = "nsrepair.rs"]
mod __slate_unit_nsrepair;
#[path = "nsrepair2.rs"]
mod __slate_unit_nsrepair2;
#[path = "nssearch.rs"]
mod __slate_unit_nssearch;
#[path = "nsutils.rs"]
mod __slate_unit_nsutils;
#[path = "nswalk.rs"]
mod __slate_unit_nswalk;
#[path = "nsxfeval.rs"]
mod __slate_unit_nsxfeval;
#[path = "nsxfname.rs"]
mod __slate_unit_nsxfname;
#[path = "nsxfobj.rs"]
mod __slate_unit_nsxfobj;
#[path = "psargs.rs"]
mod __slate_unit_psargs;
#[path = "psloop.rs"]
mod __slate_unit_psloop;
#[path = "psobject.rs"]
mod __slate_unit_psobject;
#[path = "psopcode.rs"]
mod __slate_unit_psopcode;
#[path = "psopinfo.rs"]
mod __slate_unit_psopinfo;
#[path = "psparse.rs"]
mod __slate_unit_psparse;
#[path = "psscope.rs"]
mod __slate_unit_psscope;
#[path = "pstree.rs"]
mod __slate_unit_pstree;
#[path = "psutils.rs"]
mod __slate_unit_psutils;
#[path = "pswalk.rs"]
mod __slate_unit_pswalk;
#[path = "psxface.rs"]
mod __slate_unit_psxface;
#[path = "rsaddr.rs"]
mod __slate_unit_rsaddr;
#[path = "rscalc.rs"]
mod __slate_unit_rscalc;
#[path = "rscreate.rs"]
mod __slate_unit_rscreate;
#[path = "rsdumpinfo.rs"]
mod __slate_unit_rsdumpinfo;
#[path = "rsinfo.rs"]
mod __slate_unit_rsinfo;
#[path = "rsio.rs"]
mod __slate_unit_rsio;
#[path = "rsirq.rs"]
mod __slate_unit_rsirq;
#[path = "rslist.rs"]
mod __slate_unit_rslist;
#[path = "rsmemory.rs"]
mod __slate_unit_rsmemory;
#[path = "rsmisc.rs"]
mod __slate_unit_rsmisc;
#[path = "rsserial.rs"]
mod __slate_unit_rsserial;
#[path = "rsutils.rs"]
mod __slate_unit_rsutils;
#[path = "rsxface.rs"]
mod __slate_unit_rsxface;
#[path = "tbdata.rs"]
mod __slate_unit_tbdata;
#[path = "tbfadt.rs"]
mod __slate_unit_tbfadt;
#[path = "tbfind.rs"]
mod __slate_unit_tbfind;
#[path = "tbinstal.rs"]
mod __slate_unit_tbinstal;
#[path = "tbprint.rs"]
mod __slate_unit_tbprint;
#[path = "tbutils.rs"]
mod __slate_unit_tbutils;
#[path = "tbxface.rs"]
mod __slate_unit_tbxface;
#[path = "tbxfload.rs"]
mod __slate_unit_tbxfload;

mod __slate_panic {
    #[panic_handler]
    fn __slate_panic_handler(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }

    #[linkage = "weak"]
    #[unsafe(no_mangle)]
    extern "C" fn rust_eh_personality() {}
}


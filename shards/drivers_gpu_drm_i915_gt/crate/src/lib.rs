#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "gen2_engine_cs.rs"]
mod __slate_unit_gen2_engine_cs;
#[path = "gen6_engine_cs.rs"]
mod __slate_unit_gen6_engine_cs;
#[path = "gen6_ppgtt.rs"]
mod __slate_unit_gen6_ppgtt;
#[path = "gen7_renderclear.rs"]
mod __slate_unit_gen7_renderclear;
#[path = "gen8_engine_cs.rs"]
mod __slate_unit_gen8_engine_cs;
#[path = "gen8_ppgtt.rs"]
mod __slate_unit_gen8_ppgtt;
#[path = "intel_breadcrumbs.rs"]
mod __slate_unit_intel_breadcrumbs;
#[path = "intel_context.rs"]
mod __slate_unit_intel_context;
#[path = "intel_context_sseu.rs"]
mod __slate_unit_intel_context_sseu;
#[path = "intel_engine_cs.rs"]
mod __slate_unit_intel_engine_cs;
#[path = "intel_engine_heartbeat.rs"]
mod __slate_unit_intel_engine_heartbeat;
#[path = "intel_engine_pm.rs"]
mod __slate_unit_intel_engine_pm;
#[path = "intel_engine_user.rs"]
mod __slate_unit_intel_engine_user;
#[path = "intel_execlists_submission.rs"]
mod __slate_unit_intel_execlists_submission;
#[path = "intel_ggtt.rs"]
mod __slate_unit_intel_ggtt;
#[path = "intel_ggtt_fencing.rs"]
mod __slate_unit_intel_ggtt_fencing;
#[path = "intel_gt.rs"]
mod __slate_unit_intel_gt;
#[path = "intel_gt_buffer_pool.rs"]
mod __slate_unit_intel_gt_buffer_pool;
#[path = "intel_gt_ccs_mode.rs"]
mod __slate_unit_intel_gt_ccs_mode;
#[path = "intel_gt_clock_utils.rs"]
mod __slate_unit_intel_gt_clock_utils;
#[path = "intel_gt_debugfs.rs"]
mod __slate_unit_intel_gt_debugfs;
#[path = "intel_gt_engines_debugfs.rs"]
mod __slate_unit_intel_gt_engines_debugfs;
#[path = "intel_gt_irq.rs"]
mod __slate_unit_intel_gt_irq;
#[path = "intel_gt_mcr.rs"]
mod __slate_unit_intel_gt_mcr;
#[path = "intel_gt_pm.rs"]
mod __slate_unit_intel_gt_pm;
#[path = "intel_gt_pm_debugfs.rs"]
mod __slate_unit_intel_gt_pm_debugfs;
#[path = "intel_gt_pm_irq.rs"]
mod __slate_unit_intel_gt_pm_irq;
#[path = "intel_gt_requests.rs"]
mod __slate_unit_intel_gt_requests;
#[path = "intel_gt_sysfs.rs"]
mod __slate_unit_intel_gt_sysfs;
#[path = "intel_gt_sysfs_pm.rs"]
mod __slate_unit_intel_gt_sysfs_pm;
#[path = "intel_gtt.rs"]
mod __slate_unit_intel_gtt;
#[path = "intel_llc.rs"]
mod __slate_unit_intel_llc;
#[path = "intel_lrc.rs"]
mod __slate_unit_intel_lrc;
#[path = "intel_migrate.rs"]
mod __slate_unit_intel_migrate;
#[path = "intel_mocs.rs"]
mod __slate_unit_intel_mocs;
#[path = "intel_ppgtt.rs"]
mod __slate_unit_intel_ppgtt;
#[path = "intel_rc6.rs"]
mod __slate_unit_intel_rc6;
#[path = "intel_region_lmem.rs"]
mod __slate_unit_intel_region_lmem;
#[path = "intel_renderstate.rs"]
mod __slate_unit_intel_renderstate;
#[path = "intel_reset.rs"]
mod __slate_unit_intel_reset;
#[path = "intel_ring.rs"]
mod __slate_unit_intel_ring;
#[path = "intel_ring_submission.rs"]
mod __slate_unit_intel_ring_submission;
#[path = "intel_rps.rs"]
mod __slate_unit_intel_rps;
#[path = "intel_sa_media.rs"]
mod __slate_unit_intel_sa_media;
#[path = "intel_sseu.rs"]
mod __slate_unit_intel_sseu;
#[path = "intel_sseu_debugfs.rs"]
mod __slate_unit_intel_sseu_debugfs;
#[path = "intel_timeline.rs"]
mod __slate_unit_intel_timeline;
#[path = "intel_tlb.rs"]
mod __slate_unit_intel_tlb;
#[path = "intel_wopcm.rs"]
mod __slate_unit_intel_wopcm;
#[path = "intel_workarounds.rs"]
mod __slate_unit_intel_workarounds;
#[path = "shmem_utils.rs"]
mod __slate_unit_shmem_utils;
#[path = "sysfs_engines.rs"]
mod __slate_unit_sysfs_engines;
#[path = "intel_ggtt_gmch.rs"]
mod __slate_unit_intel_ggtt_gmch;
#[path = "gen6_renderstate.rs"]
mod __slate_unit_gen6_renderstate;
#[path = "gen7_renderstate.rs"]
mod __slate_unit_gen7_renderstate;
#[path = "gen8_renderstate.rs"]
mod __slate_unit_gen8_renderstate;
#[path = "gen9_renderstate.rs"]
mod __slate_unit_gen9_renderstate;
#[path = "intel_gsc_fw.rs"]
mod __slate_unit_intel_gsc_fw;
#[path = "intel_gsc_proxy.rs"]
mod __slate_unit_intel_gsc_proxy;
#[path = "intel_gsc_uc.rs"]
mod __slate_unit_intel_gsc_uc;
#[path = "intel_gsc_uc_debugfs.rs"]
mod __slate_unit_intel_gsc_uc_debugfs;
#[path = "intel_gsc_uc_heci_cmd_submit.rs"]
mod __slate_unit_intel_gsc_uc_heci_cmd_submit;
#[path = "intel_guc.rs"]
mod __slate_unit_intel_guc;
#[path = "intel_guc_ads.rs"]
mod __slate_unit_intel_guc_ads;
#[path = "intel_guc_capture.rs"]
mod __slate_unit_intel_guc_capture;
#[path = "intel_guc_ct.rs"]
mod __slate_unit_intel_guc_ct;
#[path = "intel_guc_debugfs.rs"]
mod __slate_unit_intel_guc_debugfs;
#[path = "intel_guc_fw.rs"]
mod __slate_unit_intel_guc_fw;
#[path = "intel_guc_hwconfig.rs"]
mod __slate_unit_intel_guc_hwconfig;
#[path = "intel_guc_log.rs"]
mod __slate_unit_intel_guc_log;
#[path = "intel_guc_log_debugfs.rs"]
mod __slate_unit_intel_guc_log_debugfs;
#[path = "intel_guc_rc.rs"]
mod __slate_unit_intel_guc_rc;
#[path = "intel_guc_slpc.rs"]
mod __slate_unit_intel_guc_slpc;
#[path = "intel_guc_submission.rs"]
mod __slate_unit_intel_guc_submission;
#[path = "intel_huc.rs"]
mod __slate_unit_intel_huc;
#[path = "intel_huc_debugfs.rs"]
mod __slate_unit_intel_huc_debugfs;
#[path = "intel_huc_fw.rs"]
mod __slate_unit_intel_huc_fw;
#[path = "intel_uc.rs"]
mod __slate_unit_intel_uc;
#[path = "intel_uc_debugfs.rs"]
mod __slate_unit_intel_uc_debugfs;
#[path = "intel_uc_fw.rs"]
mod __slate_unit_intel_uc_fw;
#[path = "intel_gsc.rs"]
mod __slate_unit_intel_gsc;

mod __slate_panic {
    #[panic_handler]
    fn __slate_panic_handler(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }

    #[linkage = "weak"]
    #[unsafe(no_mangle)]
    extern "C" fn rust_eh_personality() {}
}


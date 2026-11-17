#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "i915_gem_busy.rs"]
mod __slate_unit_i915_gem_busy;
#[path = "i915_gem_clflush.rs"]
mod __slate_unit_i915_gem_clflush;
#[path = "i915_gem_context.rs"]
mod __slate_unit_i915_gem_context;
#[path = "i915_gem_create.rs"]
mod __slate_unit_i915_gem_create;
#[path = "i915_gem_dmabuf.rs"]
mod __slate_unit_i915_gem_dmabuf;
#[path = "i915_gem_domain.rs"]
mod __slate_unit_i915_gem_domain;
#[path = "i915_gem_execbuffer.rs"]
mod __slate_unit_i915_gem_execbuffer;
#[path = "i915_gem_internal.rs"]
mod __slate_unit_i915_gem_internal;
#[path = "i915_gem_lmem.rs"]
mod __slate_unit_i915_gem_lmem;
#[path = "i915_gem_mman.rs"]
mod __slate_unit_i915_gem_mman;
#[path = "i915_gem_object.rs"]
mod __slate_unit_i915_gem_object;
#[path = "i915_gem_object_frontbuffer.rs"]
mod __slate_unit_i915_gem_object_frontbuffer;
#[path = "i915_gem_pages.rs"]
mod __slate_unit_i915_gem_pages;
#[path = "i915_gem_panic.rs"]
mod __slate_unit_i915_gem_panic;
#[path = "i915_gem_phys.rs"]
mod __slate_unit_i915_gem_phys;
#[path = "i915_gem_pm.rs"]
mod __slate_unit_i915_gem_pm;
#[path = "i915_gem_region.rs"]
mod __slate_unit_i915_gem_region;
#[path = "i915_gem_shmem.rs"]
mod __slate_unit_i915_gem_shmem;
#[path = "i915_gem_shrinker.rs"]
mod __slate_unit_i915_gem_shrinker;
#[path = "i915_gem_stolen.rs"]
mod __slate_unit_i915_gem_stolen;
#[path = "i915_gem_throttle.rs"]
mod __slate_unit_i915_gem_throttle;
#[path = "i915_gem_tiling.rs"]
mod __slate_unit_i915_gem_tiling;
#[path = "i915_gem_ttm.rs"]
mod __slate_unit_i915_gem_ttm;
#[path = "i915_gem_ttm_move.rs"]
mod __slate_unit_i915_gem_ttm_move;
#[path = "i915_gem_ttm_pm.rs"]
mod __slate_unit_i915_gem_ttm_pm;
#[path = "i915_gem_userptr.rs"]
mod __slate_unit_i915_gem_userptr;
#[path = "i915_gem_wait.rs"]
mod __slate_unit_i915_gem_wait;
#[path = "intel_rom.rs"]
mod __slate_unit_intel_rom;
#[path = "intel_sbi.rs"]
mod __slate_unit_intel_sbi;
#[path = "intel_sprite.rs"]
mod __slate_unit_intel_sprite;
#[path = "intel_sprite_uapi.rs"]
mod __slate_unit_intel_sprite_uapi;
#[path = "intel_tc.rs"]
mod __slate_unit_intel_tc;
#[path = "intel_vblank.rs"]
mod __slate_unit_intel_vblank;
#[path = "intel_vga.rs"]
mod __slate_unit_intel_vga;
#[path = "intel_wm.rs"]
mod __slate_unit_intel_wm;
#[path = "skl_prefill.rs"]
mod __slate_unit_skl_prefill;
#[path = "skl_scaler.rs"]
mod __slate_unit_skl_scaler;
#[path = "skl_universal_plane.rs"]
mod __slate_unit_skl_universal_plane;
#[path = "skl_watermark.rs"]
mod __slate_unit_skl_watermark;
#[path = "vlv_clock.rs"]
mod __slate_unit_vlv_clock;
#[path = "vlv_sideband.rs"]
mod __slate_unit_vlv_sideband;
#[path = "intel_sdvo.rs"]
mod __slate_unit_intel_sdvo;
#[path = "intel_snps_hdmi_pll.rs"]
mod __slate_unit_intel_snps_hdmi_pll;
#[path = "intel_snps_phy.rs"]
mod __slate_unit_intel_snps_phy;
#[path = "intel_tv.rs"]
mod __slate_unit_intel_tv;
#[path = "intel_vdsc.rs"]
mod __slate_unit_intel_vdsc;
#[path = "intel_vrr.rs"]
mod __slate_unit_intel_vrr;
#[path = "vlv_dsi.rs"]
mod __slate_unit_vlv_dsi;
#[path = "vlv_dsi_pll.rs"]
mod __slate_unit_vlv_dsi_pll;
#[path = "intel_pxp.rs"]
mod __slate_unit_intel_pxp;
#[path = "intel_pxp_huc.rs"]
mod __slate_unit_intel_pxp_huc;
#[path = "intel_pxp_tee.rs"]
mod __slate_unit_intel_pxp_tee;

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


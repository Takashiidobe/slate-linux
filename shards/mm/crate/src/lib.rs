#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(asm_goto_with_outputs)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "filemap.rs"]
mod __slate_unit_filemap;
#[path = "mempool.rs"]
mod __slate_unit_mempool;
#[path = "oom_kill.rs"]
mod __slate_unit_oom_kill;
#[path = "fadvise.rs"]
mod __slate_unit_fadvise;
#[path = "maccess.rs"]
mod __slate_unit_maccess;
#[path = "page_writeback.rs"]
mod __slate_unit_page_writeback;
#[path = "folio_compat.rs"]
mod __slate_unit_folio_compat;
#[path = "readahead.rs"]
mod __slate_unit_readahead;
#[path = "folio.rs"]
mod __slate_unit_folio;
#[path = "truncate.rs"]
mod __slate_unit_truncate;
#[path = "vmscan.rs"]
mod __slate_unit_vmscan;
#[path = "shrinker.rs"]
mod __slate_unit_shrinker;
#[path = "shmem.rs"]
mod __slate_unit_shmem;
#[path = "util.rs"]
mod __slate_unit_util;
#[path = "mmzone.rs"]
mod __slate_unit_mmzone;
#[path = "vmstat.rs"]
mod __slate_unit_vmstat;
#[path = "backing_dev.rs"]
mod __slate_unit_backing_dev;
#[path = "mm_init.rs"]
mod __slate_unit_mm_init;
#[path = "percpu.rs"]
mod __slate_unit_percpu;
#[path = "slab_common.rs"]
mod __slate_unit_slab_common;
#[path = "compaction.rs"]
mod __slate_unit_compaction;
#[path = "show_mem.rs"]
mod __slate_unit_show_mem;
#[path = "interval_tree.rs"]
mod __slate_unit_interval_tree;
#[path = "list_lru.rs"]
mod __slate_unit_list_lru;
#[path = "workingset.rs"]
mod __slate_unit_workingset;
#[path = "debug.rs"]
mod __slate_unit_debug;
#[path = "gup.rs"]
mod __slate_unit_gup;
#[path = "mmap_lock.rs"]
mod __slate_unit_mmap_lock;
#[path = "vma_init.rs"]
mod __slate_unit_vma_init;
#[path = "highmem.rs"]
mod __slate_unit_highmem;
#[path = "memory.rs"]
mod __slate_unit_memory;
#[path = "mincore.rs"]
mod __slate_unit_mincore;
#[path = "mlock.rs"]
mod __slate_unit_mlock;
#[path = "mmap.rs"]
mod __slate_unit_mmap;
#[path = "mmu_gather.rs"]
mod __slate_unit_mmu_gather;
#[path = "mprotect.rs"]
mod __slate_unit_mprotect;
#[path = "mremap.rs"]
mod __slate_unit_mremap;
#[path = "msync.rs"]
mod __slate_unit_msync;
#[path = "page_vma_mapped.rs"]
mod __slate_unit_page_vma_mapped;
#[path = "pagewalk.rs"]
mod __slate_unit_pagewalk;
#[path = "pgtable_generic.rs"]
mod __slate_unit_pgtable_generic;
#[path = "rmap.rs"]
mod __slate_unit_rmap;
#[path = "vmalloc.rs"]
mod __slate_unit_vmalloc;
#[path = "vma.rs"]
mod __slate_unit_vma;
#[path = "vma_exec.rs"]
mod __slate_unit_vma_exec;
#[path = "process_vm_access.rs"]
mod __slate_unit_process_vm_access;
#[path = "mseal.rs"]
mod __slate_unit_mseal;
#[path = "page_alloc.rs"]
mod __slate_unit_page_alloc;
#[path = "page_frag_cache.rs"]
mod __slate_unit_page_frag_cache;
#[path = "init_mm.rs"]
mod __slate_unit_init_mm;
#[path = "memblock.rs"]
mod __slate_unit_memblock;
#[path = "slub.rs"]
mod __slate_unit_slub;
#[path = "madvise.rs"]
mod __slate_unit_madvise;
#[path = "page_io.rs"]
mod __slate_unit_page_io;
#[path = "swap_state.rs"]
mod __slate_unit_swap_state;
#[path = "swapfile.rs"]
mod __slate_unit_swapfile;
#[path = "dmapool.rs"]
mod __slate_unit_dmapool;
#[path = "hugetlb.rs"]
mod __slate_unit_hugetlb;
#[path = "hugetlb_sysfs.rs"]
mod __slate_unit_hugetlb_sysfs;
#[path = "hugetlb_sysctl.rs"]
mod __slate_unit_hugetlb_sysctl;
#[path = "hugetlb_vmemmap.rs"]
mod __slate_unit_hugetlb_vmemmap;
#[path = "mempolicy.rs"]
mod __slate_unit_mempolicy;
#[path = "sparse.rs"]
mod __slate_unit_sparse;
#[path = "sparse_vmemmap.rs"]
mod __slate_unit_sparse_vmemmap;
#[path = "mmu_notifier.rs"]
mod __slate_unit_mmu_notifier;
#[path = "migrate.rs"]
mod __slate_unit_migrate;
#[path = "memory_tiers.rs"]
mod __slate_unit_memory_tiers;
#[path = "page_counter.rs"]
mod __slate_unit_page_counter;
#[path = "hugetlb_cgroup.rs"]
mod __slate_unit_hugetlb_cgroup;
#[path = "early_ioremap.rs"]
mod __slate_unit_early_ioremap;
#[path = "numa.rs"]
mod __slate_unit_numa;
#[path = "numa_memblks.rs"]
mod __slate_unit_numa_memblks;
#[path = "secretmem.rs"]
mod __slate_unit_secretmem;
#[path = "memfd.rs"]
mod __slate_unit_memfd;
#[path = "ptdump.rs"]
mod __slate_unit_ptdump;
#[path = "execmem.rs"]
mod __slate_unit_execmem;

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


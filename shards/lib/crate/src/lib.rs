#![no_std]
#![no_builtins]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(asm_goto_with_outputs)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "lockref.rs"]
mod __slate_unit_lockref;
#[path = "bcd.rs"]
mod __slate_unit_bcd;
#[path = "sort.rs"]
mod __slate_unit_sort;
#[path = "parser.rs"]
mod __slate_unit_parser;
#[path = "debug_locks.rs"]
mod __slate_unit_debug_locks;
#[path = "random32.rs"]
mod __slate_unit_random32;
#[path = "bust_spinlocks.rs"]
mod __slate_unit_bust_spinlocks;
#[path = "kasprintf.rs"]
mod __slate_unit_kasprintf;
#[path = "bitmap.rs"]
mod __slate_unit_bitmap;
#[path = "scatterlist.rs"]
mod __slate_unit_scatterlist;
#[path = "list_sort.rs"]
mod __slate_unit_list_sort;
#[path = "uuid.rs"]
mod __slate_unit_uuid;
#[path = "iov_iter.rs"]
mod __slate_unit_iov_iter;
#[path = "clz_ctz.rs"]
mod __slate_unit_clz_ctz;
#[path = "bsearch.rs"]
mod __slate_unit_bsearch;
#[path = "find_bit.rs"]
mod __slate_unit_find_bit;
#[path = "llist.rs"]
mod __slate_unit_llist;
#[path = "lwq.rs"]
mod __slate_unit_lwq;
#[path = "memweight.rs"]
mod __slate_unit_memweight;
#[path = "kfifo.rs"]
mod __slate_unit_kfifo;
#[path = "percpu_refcount.rs"]
mod __slate_unit_percpu_refcount;
#[path = "rhashtable.rs"]
mod __slate_unit_rhashtable;
#[path = "base64.rs"]
mod __slate_unit_base64;
#[path = "once.rs"]
mod __slate_unit_once;
#[path = "refcount.rs"]
mod __slate_unit_refcount;
#[path = "rcuref.rs"]
mod __slate_unit_rcuref;
#[path = "usercopy.rs"]
mod __slate_unit_usercopy;
#[path = "errseq.rs"]
mod __slate_unit_errseq;
#[path = "bucket_locks.rs"]
mod __slate_unit_bucket_locks;
#[path = "generic_radix_tree.rs"]
mod __slate_unit_generic_radix_tree;
#[path = "bitmap_str.rs"]
mod __slate_unit_bitmap_str;
#[path = "string_helpers.rs"]
mod __slate_unit_string_helpers;
#[path = "hexdump.rs"]
mod __slate_unit_hexdump;
#[path = "kstrtox.rs"]
mod __slate_unit_kstrtox;
#[path = "iomap.rs"]
mod __slate_unit_iomap;
#[path = "iomap_copy.rs"]
mod __slate_unit_iomap_copy;
#[path = "devres.rs"]
mod __slate_unit_devres;
#[path = "check_signature.rs"]
mod __slate_unit_check_signature;
#[path = "interval_tree.rs"]
mod __slate_unit_interval_tree;
#[path = "assoc_array.rs"]
mod __slate_unit_assoc_array;
#[path = "bitrev.rs"]
mod __slate_unit_bitrev;
#[path = "xxhash.rs"]
mod __slate_unit_xxhash;
#[path = "genalloc.rs"]
mod __slate_unit_genalloc;
#[path = "percpu_counter.rs"]
mod __slate_unit_percpu_counter;
#[path = "syscall.rs"]
mod __slate_unit_syscall;
#[path = "errname.rs"]
mod __slate_unit_errname;
#[path = "nlattr.rs"]
mod __slate_unit_nlattr;
#[path = "cpu_rmap.rs"]
mod __slate_unit_cpu_rmap;
#[path = "dynamic_queue_limits.rs"]
mod __slate_unit_dynamic_queue_limits;
#[path = "glob.rs"]
mod __slate_unit_glob;
#[path = "strncpy_from_user.rs"]
mod __slate_unit_strncpy_from_user;
#[path = "strnlen_user.rs"]
mod __slate_unit_strnlen_user;
#[path = "net_utils.rs"]
mod __slate_unit_net_utils;
#[path = "sg_pool.rs"]
mod __slate_unit_sg_pool;
#[path = "stackdepot.rs"]
mod __slate_unit_stackdepot;
#[path = "asn1_decoder.rs"]
mod __slate_unit_asn1_decoder;
#[path = "oid_registry.rs"]
mod __slate_unit_oid_registry;
#[path = "ucs2_string.rs"]
mod __slate_unit_ucs2_string;
#[path = "sbitmap.rs"]
mod __slate_unit_sbitmap;
#[path = "group_cpus.rs"]
mod __slate_unit_group_cpus;
#[path = "fw_table.rs"]
mod __slate_unit_fw_table;
#[path = "argv_split.rs"]
mod __slate_unit_argv_split;
#[path = "bug.rs"]
mod __slate_unit_bug;
#[path = "buildid.rs"]
mod __slate_unit_buildid;
#[path = "clz_tab.rs"]
mod __slate_unit_clz_tab;
#[path = "cmdline.rs"]
mod __slate_unit_cmdline;
#[path = "cpumask.rs"]
mod __slate_unit_cpumask;
#[path = "ctype.rs"]
mod __slate_unit_ctype;
#[path = "dec_and_lock.rs"]
mod __slate_unit_dec_and_lock;
#[path = "decompress.rs"]
mod __slate_unit_decompress;
#[path = "decompress_bunzip2.rs"]
mod __slate_unit_decompress_bunzip2;
#[path = "decompress_inflate.rs"]
mod __slate_unit_decompress_inflate;
#[path = "decompress_unlz4.rs"]
mod __slate_unit_decompress_unlz4;
#[path = "decompress_unlzma.rs"]
mod __slate_unit_decompress_unlzma;
#[path = "decompress_unlzo.rs"]
mod __slate_unit_decompress_unlzo;
#[path = "decompress_unxz.rs"]
mod __slate_unit_decompress_unxz;
#[path = "decompress_unzstd.rs"]
mod __slate_unit_decompress_unzstd;
#[path = "dump_stack.rs"]
mod __slate_unit_dump_stack;
#[path = "earlycpio.rs"]
mod __slate_unit_earlycpio;
#[path = "extable.rs"]
mod __slate_unit_extable;
#[path = "flex_proportions.rs"]
mod __slate_unit_flex_proportions;
#[path = "idr.rs"]
mod __slate_unit_idr;
#[path = "iomem_copy.rs"]
mod __slate_unit_iomem_copy;
#[path = "irq_regs.rs"]
mod __slate_unit_irq_regs;
#[path = "is_single_threaded.rs"]
mod __slate_unit_is_single_threaded;
#[path = "klist.rs"]
mod __slate_unit_klist;
#[path = "kobject.rs"]
mod __slate_unit_kobject;
#[path = "kobject_uevent.rs"]
mod __slate_unit_kobject_uevent;
#[path = "logic_pio.rs"]
mod __slate_unit_logic_pio;
#[path = "maple_tree.rs"]
mod __slate_unit_maple_tree;
#[path = "memcat_p.rs"]
mod __slate_unit_memcat_p;
#[path = "nmi_backtrace.rs"]
mod __slate_unit_nmi_backtrace;
#[path = "objpool.rs"]
mod __slate_unit_objpool;
#[path = "plist.rs"]
mod __slate_unit_plist;
#[path = "radix_tree.rs"]
mod __slate_unit_radix_tree;
#[path = "ratelimit.rs"]
mod __slate_unit_ratelimit;
#[path = "rbtree.rs"]
mod __slate_unit_rbtree;
#[path = "seq_buf.rs"]
mod __slate_unit_seq_buf;
#[path = "siphash.rs"]
mod __slate_unit_siphash;
#[path = "string.rs"]
mod __slate_unit_string;
#[path = "sys_info.rs"]
mod __slate_unit_sys_info;
#[path = "timerqueue.rs"]
mod __slate_unit_timerqueue;
#[path = "union_find.rs"]
mod __slate_unit_union_find;
#[path = "vsprintf.rs"]
mod __slate_unit_vsprintf;
#[path = "win_minmax.rs"]
mod __slate_unit_win_minmax;
#[path = "xarray.rs"]
mod __slate_unit_xarray;

mod __slate_panic {
    #[panic_handler]
    fn __slate_panic_handler(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }

    #[linkage = "weak"]
    #[unsafe(no_mangle)]
    extern "C" fn rust_eh_personality() {}
}


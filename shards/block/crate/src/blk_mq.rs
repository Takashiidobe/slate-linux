
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! These attributes define new context lock (Clang: capability) types.
//! Internal only.
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! Turn individual warnings and errors on and off locally, depending
//! on version.
//! SPDX-License-Identifier: GPL-2.0
//!
//! Block multiqueue core code
//!
//! Copyright (C) 2013-2014 Jens Axboe
//! Copyright (C) 2013-2014 Christoph Hellwig
//! Optimization barrier
//!
//! The "volatile" is due to gcc bugs
//! __DECLARE_FLEX_ARRAY() - Declare a flexible array usable in a union
//!
//! @TYPE: The type of each flexible array element
//! @NAME: The name of the flexible array member
//!
//! In order to have a flexible array member in a union or alone in a
//! struct, it needs to be wrapped in an anonymous struct with at least 1
//! named member, but that member can be empty.
//! aligned_u64 should be used in defining kernel<->userspace ABIs to avoid
//! common 32/64-bit compat problems.
//! 64-bit values align to 4-byte boundaries on x86_32 (and possibly other
//! architectures) and to 8-byte boundaries on 64-bit architectures.  The new
//! aligned_64 type enforces 8-byte alignment so that structs containing
//! aligned_64 values have the same alignment on 32-bit and 64-bit architectures.
//! No conversions are necessary between 32-bit user-space and a 64-bit kernel.
//! These are the special 64-bit data types that are 8-byte aligned
//! Yes, this permits 64-bit accesses on 32-bit architectures. These will
//! actually be atomic in some cases (namely Armv7 + LPAE), but for others we
//! rely on the access being split into 2x32-bit accesses for a 32-bit quantity
//! (e.g. a virtual address) and a strong prevailing wind.
//! # chars in a file name
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
//! Force a compilation error if condition is true, but also produce a
//! result (of value 0 and type int), so the expression can be used
//! e.g. in a structure initializer (or where-ever else comma expressions
//! aren't permitted).
//!
//! Take an error message as an optional second argument. If omitted,
//! default to the stringification of the tested expression.
//! See linux/instrumentation.h
//! 64 bit
//! unused, was: #define EX_TYPE_COPY	 4
//! Begin/end of an instrumentation safe region
//! Call a function once. Similar to DO_ONCE(), but does not use jump label
//! patching via static keys.
//! These are for everybody (although not all archs will actually
//! discard it in modules)
//! error conditions
//! Internal kernel timer frequency
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! issue num suppressed message on exit
//! strlen("ratelimit") + 1
//! label
//! Prevent speculative execution past this barrier.
//! const_*() definitions provide good compile-time optimizations when
//! the passed arguments can be resolved at compile time.
//! Many architecture-specific non-atomic bitops contain inline asm code and due
//! to that the compiler can't optimize them to compile-time expressions or
//! constants. In contrary, generic_*() helpers are defined in pure C and
//! compilers optimize them just well.
//! Therefore, to make `unsigned long foo = 0; __set_bit(BAR, &foo)` effectively
//! equal to `unsigned long foo = BIT(BAR)`, pick the generic C alternative when
//! the arguments can be resolved at compile time. That expression itself is a
//! constant and doesn't bring any functional changes to the rest of cases.
//! The casts to `uintptr_t` are needed to mitigate `-Waddress` warnings when
//! passing a bitmap from .bss or .data (-> `!!addr` is always true).
//! This counts to 15. Any more, it will return 16th argument.
//! We do the locked ops that don't return the old value as
//! a mask operation on a byte.
//! Defines x86 CPU feature bits
//! Compile time versions of __arch_hweightN()
//! do_div - returns 2 values: calculate remainder and update new dividend
//! @n: uint64_t dividend (will be updated)
//! @base: uint32_t divisor
//!
//! Summary:
//! ``uint32_t remainder = n % base;``
//! ``n = n / base;``
//!
//! Return: (uint32_t)remainder
//!
//! NOTE: macro parameter @n is evaluated multiple times,
//! beware of side effects!
//! lower_32_bits - return bits 0-31 of a number
//! @n: the number we're accessing
//! These are non-NULL pointers that will result in page faults
//! under normal circumstances, used to verify that nobody uses
//! non-initialized list entries.
//! LIST_HEAD_INIT - initialize a &struct list_head's links to point to itself
//! @name: name of the list_head
//! DIV_U64_ROUND_UP - unsigned 64bit divide with 32bit divisor rounded up
//! @ll: unsigned 64bit dividend
//! @d: unsigned 32bit divisor
//!
//! Divide unsigned 64bit dividend by unsigned 32bit divisor
//! and round up.
//!
//! Return: dividend / divisor rounded up
//! Parameters used to convert the timespec values:
//! AUX clock support. AUXiliary clocks are dynamically configured by
//! enabling a clock ID. These clock can be steered independently of the
//! core timekeeper. The kernel can support up to 8 auxiliary clocks, but
//! the actual limit depends on eventual architecture constraints vs. VDSO.
//! Sign Flag
//! The __sme_pa() and __sme_pa_nodebug() macros are meant for use when
//! writing to or comparing values from the cr3 register.  Having the
//! encryption mask set in cr3 enables the PGD entry to be encrypted and
//! avoid special case handling of PGD allocations.
//! CR3's layout varies depending on several things.
//!
//! If CR4.PCIDE is set (64-bit only), then CR3[11:0] is the address space ID.
//! If PAE is enabled, then CR3[11:5] is part of the PDPT address
//! (i.e. it's 32-byte aligned, not page-aligned) and CR3[4:0] is ignored.
//! Otherwise (non-PAE, non-PCID), CR3[3] is PWT, CR3[4] is PCD, and
//! CR3[2:0] and CR3[11:5] are ignored.
//!
//! In all cases, Linux puts zeros in the low ignored bits and in PWT and PCD.
//!
//! CR3[63] is always read as zero.  If CR4.PCIDE is set, then CR3[63] may be
//! written as 1 to prevent the write to CR3 from flushing the TLB.
//!
//! On systems with SME, one bit (in a variable position!) is stolen to indicate
//! that the top-level paging structure is encrypted.
//!
//! On systemms with LAM, bits 61 and 62 are used to indicate LAM mode.
//!
//! All of the remaining bits indicate the physical address of the top-level
//! paging structure.
//!
//! CR3_ADDR_MASK is the mask used by read_cr3_pa().
//!
//! Mask off the address space ID and SME encryption bits.
//! Bit size and mask of CPU number stored in the per CPU data (and TSC_AUX)
//! PAGE_SHIFT determines the page size.
//!
//! Note: This definition is required because PAGE_SHIFT is used
//! in several places throughout the codebase.
//! The generic per-CPU infrastrucutre is not suitable for
//! reading const-qualified variables.
//! Maximum supported processors.  Setting this smaller saves quite a
//! bit of memory.  Use nr_cpu_ids instead of this except for static bitmaps.
//!
//! Places which use this should consider cpumask_var_t.
//! Base implementations of per-CPU variable declarations and definitions, where
//! the section in which the variable is to be placed is provided by the
//! 'sec' argument.  This may be used to affect the parameters governing the
//! variable's storage.
//!
//! NOTE!  The sections for the DECLARE and for the DEFINE must match, lest
//! linkage errors occur due the compiler generating the wrong code to access
//! that section.
//! Ensure that leaf 0x2 cache and TLB type values do not intersect,
//! since they share the same type field at struct cpuid_0x2_table.
//! virt_to_page(kaddr) returns a valid pointer if and only if
//! virt_addr_valid(kaddr) returns true.
//! writeable
//! Operation not permitted
//! This error code is special: arch syscall entry code will return
//! -ENOSYS if users try to call a syscall that doesn't exist.  To keep
//! failures of syscalls that really do exist distinguishable from
//! failures due to attempts to use a nonexistent syscall, syscall
//! implementations should refrain from returning -ENOSYS.
//! No ioctl command
//! Kernel pointers have redundant information, so we can use a
//! scheme where we can return either an error code or a normal
//! pointer with the same return value.
//!
//! This should be a per-architecture thing, to allow different
//! error and pointer decisions.
//! To support Context Analysis, we need to allow the compiler to see the
//! acquisition and release of the context lock. However, the "cleanup" helpers
//! wrap the lock in a struct passed through separate helper functions, which
//! hides the lock alias from the compiler (no inter-procedural analysis).
//!
//! To make it work, we introduce an explicit alias to the context lock instance
//! that is "cleaned" up with a separate cleanup helper. This helper is a dummy
//! function that does nothing at runtime, but has the "_unlock" attribute to
//! tell the compiler what happens at the end of the scope.
//!
//! To generalize the pattern, the WITH_LOCK_GUARD_1_ATTRS() macro should be used
//! to redefine the constructor, which then also creates the alias variable with
//! the right "cleanup" attribute, *after* DECLARE_LOCK_GUARD_1_ATTRS() has been
//! used.
//!
//! Example usage:
//!
//!   DECLARE_LOCK_GUARD_1_ATTRS(mutex, __acquires(_T), __releases(*(struct mutex **)_T))
//!   #define class_mutex_constructor(_T) WITH_LOCK_GUARD_1_ATTRS(mutex, _T)
//!
//! Note: To support the for-loop based scoped helpers, the auxiliary variable
//! must be a pointer to the "class" type because it is defined in the same
//! statement as the guard variable. However, we initialize it with the lock
//! pointer (despite the type mismatch, the compiler's alias analysis still works
//! as expected). The "_unlock" attribute receives a pointer to the auxiliary
//! variable (a double pointer to the class type), and must be cast and
//! dereferenced appropriately.
//! 64bit FS base
//! Use alternative patching for paravirt calls:
//! - For replacing an indirect call with a direct one, use the "normal"
//!   ALTERNATIVE() macro with the indirect call as the initial code sequence,
//!   which will be replaced with the related direct call by using the
//!   ALT_FLAG_DIRECT_CALL special case and the "always on" feature.
//! - In case the replacement is either a direct call or a short code sequence
//!   depending on a feature bit, the ALTERNATIVE_2() macro is being used.
//!   The indirect call is the initial code sequence again, while the special
//!   code sequence is selected with the specified feature bit. In case the
//!   feature is not active, the direct call is used as above via the
//!   ALT_FLAG_DIRECT_CALL special case and the "always on" feature.
//! strscpy - Copy a C-string into a sized buffer
//! @dst: Where to copy the string to
//! @src: Where to copy the string from
//! @...: Size of destination buffer (optional)
//!
//! Copy the source string @src, or as much of it as fits, into the
//! destination @dst buffer. The behavior is undefined if the string
//! buffers overlap. The destination @dst buffer is always NUL terminated,
//! unless it's zero-sized.
//!
//! The size argument @... is only required when @dst is not an array, or
//! when the copy needs to be smaller than sizeof(@dst).
//!
//! Preferred to strncpy() since it always returns a valid string, and
//! doesn't unnecessarily force the tail of the destination buffer to be
//! zero padded. If padding is desired please use strscpy_pad().
//!
//! Returns the number of characters copied in @dst (not including the
//! trailing %NUL) or -E2BIG if @size is 0 or the copy from @src was
//! truncated.
//! used to decode ioctl numbers..
//! The inline keyword gives the compiler room to decide to inline, or
//! not inline a function as it sees best.  However, as these functions
//! are called in both __init and non-__init functions, if they are not
//! inlined we will end up with a section mismatch error (of the type of
//! freeable items not being freed).  So we must use __always_inline here
//! to fix the problem.  If other functions in the future also end up in
//! this situation they will also need to be annotated as __always_inline
//! Wrap the arch provided IRQ routines to provide appropriate checks.
//! Instruction encoding provided for binutils backwards compatibility.
//! Defines VMX CPU feature bits
//! This is the default CPU features testing macro to use in code.
//!
//! It is for detection of features which need kernel infrastructure to be
//! used.  It may *not* directly test the CPU itself.  Use the cpu_has() family
//! if you want true runtime testing of CPU features, like in hypervisor code
//! where you are supporting a possible guest feature where host support for it
//! is not relevant.
//! Macros to retrieve the current execution context:
//!
//! in_nmi()		- We're in NMI context
//! in_hardirq()		- We're in hard IRQ context
//! in_serving_softirq()	- We're in softirq context
//! in_task()		- We're in task context
//! signal pending
//! New user namespace
//! These constants are defined in the MSan LLVM instrumentation pass.
//! ignore signal
//! bit-flags
//! If !LOCKDEP, the value is meaningless
//! Shortcut for the queued_spin_lock_slowpath() function that allows
//! virt to hijack it.
//!
//! Returns:
//!   true - lock has been negotiated, all done;
//!   false - queued_spin_lock_slowpath() will do its thing.
//! Remapping spinlock architecture specific functions to the corresponding
//! queued spinlock functions.
//! A writer holds the lock
//! Define the various spin_lock methods.  Note we define these
//! regardless of whether CONFIG_SMP or CONFIG_PREEMPTION are set. The
//! various methods are defined as nops in the case they are not
//! required.
//! Used in tsk->__state:
//! Instruction opcode for WRMSRNS supported in binutils >= 2.40
//! wait_queue_entry::flags
//! mutex_init - initialize the mutex
//! @mutex: the mutex to be initialized
//!
//! Initialize the mutex to unlocked state.
//!
//! It is not allowed to initialize an already locked mutex.
//! The seqlock seqcount_t interface does not prescribe a precise sequence of
//! read begin/retry/end. For readers, typically there is a call to
//! read_seqcount_begin() and read_seqcount_retry(), however, there are more
//! esoteric cases which do not follow this pattern.
//!
//! As a consequence, we take the following best-effort approach for raw usage
//! via seqcount_t under KCSAN: upon beginning a seq-reader critical section,
//! pessimistically mark the next KCSAN_SEQLOCK_REGION_MAX memory accesses as
//! atomics; if there is a matching read_seqcount_retry() call, no following
//! memory operations are considered atomic. Usage of the seqlock_t interface
//! is not affected.
//! __MAX_NR_ZONES
//! 'empty' nodes are nodes that are known not to be inserted in an rbtree
//! Allocated nodes are mutable until they have been inserted into the tree,
//! at which time they cannot change their type until they have been removed
//! from the tree and an RCU grace period has passed.
//!
//! Removed nodes have their ->parent set to point to themselves.  RCU readers
//! check ->parent before relying on the value that they loaded from the
//! slots array.  This lets us reuse the slots array for the RCU head.
//!
//! Nodes in the tree point to their parent unless bit 0 is set.
//!
//! 64bit sizes
//! The first word is the work queue pointer and the flags rolled into
//! one
//! Suits me
//! percpu_counter batch for local add or sub
//! page_folio - Converts from page to folio.
//! @p: The page.
//!
//! Every page is part of a folio.  This function cannot be called on a
//! NULL pointer.
//!
//! Context: No reference, nor lock is required on @page.  If the caller
//! does not hold a reference, this call may race with a folio split, so
//! it should re-check the folio still contains this page after gaining
//! a reference on the folio.
//! Return: The folio which contains this page.
//! local_lock_init - Runtime initialize a lock instance
//! @lock:	The lock variable
//! We do arithmetic on the LRU lists in various places in the code,
//! so it is important to keep the active lists LRU_ACTIVE higher in
//! the array than the corresponding inactive lists, and to keep
//! the *_FILE lists LRU_FILE higher than the corresponding _ANON lists.
//!
//! This has to be kept in sync with the statistics in zone_stat_item
//! above and the descriptions in vmstat_text in mm/vmstat.c
//! Helper macro to avoid gfp flags if they are the default one
//! Desired design of maximum size and alignment (see RFC2553)
//! return the ->next pointer of a list_head in an rcu safe
//! way, we must not access it directly
//! VMA locks do not behave like most ordinary locks found in the kernel, so we
//! cannot quite have full lockdep tracking in the way we would ideally prefer.
//!
//! Read locks act as shared locks which exclude an exclusive lock being
//! taken. We therefore mark these accordingly on read lock acquire/release.
//!
//! Write locks are acquired exclusively per-VMA, but released in a shared
//! fashion, that is upon vma_end_write_all(), we update the mmap's seqcount such
//! that write lock is released.
//!
//! We therefore cannot track write locks per-VMA, nor do we try. Mitigating this
//! is the fact that, of course, we do lockdep-track the mmap lock rwsem which
//! must be held when taking a VMA write lock.
//!
//! We do, however, want to indicate that during either acquisition of a VMA
//! write lock or detachment of a VMA that we require the lock held be exclusive,
//! so we utilise lockdep to do so.
//! MXCSR will be initialized
//! mprotect needs to preserve PAT and encryption bits when updating
//! vm_page_prot
//! 6 because 6 should be plenty and struct tlb_state will fit in two cache
//! lines.
//! This is the smallest unsigned integer type that can fit a value
//! (up to 'long long')
//! CAP_BPF allows the following BPF operations:
//! - Creating all types of BPF maps
//! - Advanced verifier features
//!   - Indirect variable access
//!   - Bounded loops
//!   - BPF to BPF function calls
//!   - Scalar precision tracking
//!   - Larger complexity limits
//!   - Dead code elimination
//!   - And potentially other features
//! - Loading BPF Type Format (BTF) data
//! - Retrieve xlated and JITed code of BPF programs
//! - Use bpf_spin_lock() helper
//!
//! CAP_PERFMON relaxes the verifier checks further:
//! - BPF progs can use of pointer-to-integer conversions
//! - speculation attack hardening measures are bypassed
//! - bpf_probe_read to read arbitrary kernel memory is allowed
//! - bpf_trace_printk to print kernel memory is allowed
//!
//! CAP_SYS_ADMIN is required to use bpf_probe_write_user.
//!
//! CAP_SYS_ADMIN is required to iterate system wide loaded
//! programs, maps, links, BTFs and convert their IDs to file descriptors.
//!
//! CAP_PERFMON and CAP_BPF are required to load tracing programs.
//! CAP_NET_ADMIN and CAP_BPF are required to load networking programs.
//! return the first ptr or next element in an RCU protected list
//! This hash multiplies the input by a large odd number and takes the
//! high bits.  Since multiplication propagates changes to the most
//! significant end only, it is essential that the high bits of the
//! product be used for the hash value.
//!
//! Chuck Lever verified the effectiveness of this technique:
//! http://www.citi.umich.edu/techreports/reports/citi-tr-00-1.pdf
//!
//! Although a random odd number will do, it turns out that the golden
//! ratio phi = (sqrt(5)-1)/2, or its negative, has particularly nice
//! properties.  (See Knuth vol 3, section 6.4, exercise 9.)
//!
//! These are the negative, (1 - phi) = phi**2 = (3 - sqrt(5))/2,
//! which is very slightly easier to multiply by and makes no
//! difference to the hash distribution.
//! Keep unconverted code working
//! used to be O_SYNC, see below
//! Valid hint values for F_{GET,SET}_RW_HINT. 0 is "not set", or can be
//! used to clear any hints previously set.
//! Gives us 8 prio classes with 13-bits of data for each class
//! per-IO O_SYNC
//! File supports atomic writes
//! The structure layout of user_xstateregs, used for exporting the
//! extended register state through ptrace and core-dump (NT_X86_XSTATE note)
//! interfaces will be same as the memory layout of xsave used by the processor
//! (except for the bytes 464..511, which can be used by the software) and hence
//! the size of this structure varies depending on the features supported by the
//! processor and OS. The size of the structure that users need to use can be
//! obtained by doing:
//!     cpuid_count(0xd, 0, &eax, &ptrace_xstateregs_struct_size, &ecx, &edx);
//! i.e., cpuid.(eax=0xd,ecx=0).ebx will be the size that user (debuggers, etc.)
//! need to use.
//!
//! For now, only the first 8 bytes of the software usable bytes[464..471] will
//! be used and will be set to OS enabled xstate mask (which is same as the
//! 64bit mask returned by the xgetbv's xCR0).  Users (analyzing core dump
//! remotely, etc.) can use this mask as well as the mask saved in the
//! xstate_hdr bytes and interpret what states the processor/OS supports
//! and what states are in modified/initialized conditions for the
//! particular process/thread.
//!
//! Also when the user modifies certain state FP/SSE/etc through the
//! ptrace interface, they must ensure that the header.xfeatures
//! bytes[512..519] of the memory layout are updated correspondingly.
//! i.e., for example when FP state is modified to a non-init state,
//! header.xfeatures's bit 0 must be set to '1', when SSE is modified to
//! non-init state, header.xfeatures's bit 1 must to be set to '1', etc.
//! number of env pointers
//! Generate SD flag indexes
//! Resource type
//! Must be a compile-time constant, so implement it as a macro
//! Bits set in the VMA until the stack is in its final location
//! kunmap_local - Unmap a page mapped via kmap_local_page().
//! @__addr: An address within the page mapped
//!
//! @__addr can be any address within the mapped page.  Commonly it is the
//! address return from kmap_local_page(), but it can also include offsets.
//!
//! Unmapping should be done in the reverse order of the mapping.  See
//! kmap_local_page() for details.
//! 31 pointers + header align the folio_batch structure to a power of two
//! Flags in backing_dev_info::capability
//!
//! BDI_CAP_WRITEBACK:		Supports dirty page writeback, and dirty pages
//! 				should contribute to accounting
//! BDI_CAP_STRICTLIMIT:		Keep number of dirty pages below bdi threshold
//! Maximum number of blkcg policies allowed to be registered concurrently.
//! Defined here to simplify include dependency.
//! See the comment in asm-generic/io.h about ioremap_np().
//! These macros should be used after a dma_map_sg call has been done
//! to get bus addresses of each of the SG entries and their lengths.
//! You should only work with the number of sg entries dma_map_sg
//! returns, or alternatively stop on the first sg_dma_len(sg) which
//! is 0.
//! Number of standard BARs
//! Number of reset methods used in pci_reset_fn_methods array in pci.c
//! DMA_ATTR_WRITE_COMBINE: Specifies that writes to the mapping may be
//! buffered to improve performance.
//! ***DANGEROUS*** don't add guard page
//! should be >= TASK_COMM_LEN
//! in linux/sched.h
//! hlist_nulls_first_rcu - returns the first element of the hash list.
//! @head: the head of the list.
//! define the enumeration of all cgroup subsystems
//! To avoid confusing the compiler (and generating warnings) with code
//! that attempts to access what would be a 0-element array (i.e. sized
//! to a potentially empty array when CGROUP_SUBSYS_COUNT == 0), this
//! constant expression can be added.
//! The microseconds/nanosecond delay multiplicators are used to convert a
//! constant microseconds/nanoseconds value to a value which can be used by the
//! architectures specific implementation to transform it into loops.
//! modes for vm.memfd_noexec sysctl
//! These values are stored in task->ptrace_message
//! by ptrace_stop to describe the current syscall-stop.
//! SEIZE used, enable new behavior
//! Number of chars for args
//! The per-cpu notes area is a list of notes terminated by a "NULL"
//! note header.  For kdump, the code in vmcore.c runs in the context
//! of the second kernel to combine them into one note.
//! The artificial cap on the number of segments passed to kexec_load.
//! For IS_ENABLED(CONFIG_CRASH_DUMP)
//! These are specified by iBCS2
//! Epoll event masks
//! macro tricks to expand the set_buffer_foo(), clear_buffer_foo()
//! and buffer_foo() functions.
//! To avoid reset buffer flags that are already set, because that causes
//! a costly cache line transition, check the flag first.
//! Currently, the max observed size in the kernel code is
//! JUMP_LABEL_NOP_SIZE/RELATIVEJUMP_SIZE, which are 5.
//! Raise it if needed.
//! 8237 DMA controllers
//! We are using top down, so it is safe to use 0 here
//! Runtime PM flag argument bits
// This header exists to force full rebuild when the compiler is upgraded.
//
// When fixdep scans this, it will find this string "CONFIG_CC_VERSION_TEXT"
// and add dependency on include/config/CC_VERSION_TEXT, which is touched
// by Kconfig when the version string from the compiler changes.
// Additional tree-wide dependencies start here.
// If any of the GCC plugins change, we need to rebuild everything that
// was built with them, as they may have changed their behavior and those
// behaviors may need to be synchronized across all translation units.
// If the randstruct seed itself changes (whether for GCC plugins or
// Clang), the entire tree needs to be rebuilt since the randomization of
// structures may change between compilation units if not.
// If any external changes affect Clang's integer wrapping sanitizer
// behavior, a full rebuild is needed as the coverage for wrapping types
// may have changed, which may impact the expected behaviors that should
// not differ between compilation units.
// The use of "&&" / "||" is limited in certain expressions.
// The following enable to calculate "and" / "or" with macro expansion only.
// Helper macros to use CONFIG_ options in C/CPP expressions. Note that
// these only work with boolean and tristate options.
// IS_BUILTIN(CONFIG_FOO) evaluates to 1 if CONFIG_FOO is set to 'y', 0
// otherwise. For boolean options, this is equivalent to
// IS_ENABLED(CONFIG_FOO).
// IS_MODULE(CONFIG_FOO) evaluates to 1 if CONFIG_FOO is set to 'm', 0
// otherwise.  CONFIG_FOO=m results in "#define CONFIG_FOO_MODULE 1" in
// autoconf.h.
// IS_REACHABLE(CONFIG_FOO) evaluates to 1 if the currently compiled
// code can call a function defined in code compiled based on CONFIG_FOO.
// This is similar to IS_ENABLED(), but returns false when invoked from
// built-in code when CONFIG_FOO is set to 'm'.
// __has_builtin is supported on gcc >= 10, clang >= 3 and icc >= 21.
// In the meantime, to support gcc < 10, we implement __has_builtin
// by hand.
// Indirect macros required for expanded argument pasting, eg. __LINE__.
// C23 introduces "auto" as a standard way to define type-inferred
// variables, but "auto" has been a (useless) keyword even since K&R C,
// so it has always been "namespace reserved."
//
// Until at some future time we require C23 support, we need the gcc
// extension __auto_type, but there is no reason to put that elsewhere
// in the source code.
// nothing
// The below are used to annotate code being checked. Internal only.
// The "assert_capability" attribute is a bit confusingly named. It does not
// generate a check. Instead, it tells the analysis to *assume* the capability
// is held. This is used for augmenting runtime assertions, that can then help
// with patterns beyond the compiler's static reasoning abilities.
// __guarded_by() - struct member and globals attribute, declares variable
//                  only accessible within active context
// @...: context lock instance pointer(s)
//
// Declares that the struct member or global variable is only accessible within
// the context entered by the given context lock(s). Read operations on the data
// require shared access to at least one of the context locks, while write
// operations require exclusive access to all listed context locks.
//
// .. code-block:: c
//
// 	struct some_state {
// 		spinlock_t lock;
// 		long counter __guarded_by(&lock);
// 	};
//
// 	struct some_state {
// 		spinlock_t lock1, lock2;
// 		long counter __guarded_by(&lock1, &lock2);
// 	};
// context_lock_struct() - declare or define a context lock struct
// @name: struct name
//
// Helper to declare or define a struct type that is also a context lock.
//
// .. code-block:: c
//
// 	context_lock_struct(my_handle) {
// 		int foo;
// 		long bar;
// 	};
//
// 	struct some_state {
// 		...
// 	};
// 	// ... declared elsewhere ...
// 	context_lock_struct(some_state);
//
// Note: The implementation defines several helper functions that can acquire
// and release the context lock.
// enable_context_analysis() - re-enables context analysis
//
// Re-enables context analysis. Must be paired with a prior
// disable_context_analysis().
// __no_context_analysis - function attribute, disables context analysis
//
// Function attribute denoting that context analysis is disabled for the
// whole function. Prefer use of `context_unsafe()` where possible.
// context_unsafe() - disable context checking for contained code
//
// Disables context checking for contained statements or expression.
//
// .. code-block:: c
//
// 	struct some_data {
// 		spinlock_t lock;
// 		int counter __guarded_by(&lock);
// 	};
//
// 	int foo(struct some_data *d)
// 	{
// 		// ...
// 		// other code that is still checked ...
// 		// ...
// 		return context_unsafe(d->counter);
// 	}
// __context_unsafe() - function attribute, disable context checking
// @comment: comment explaining why opt-out is safe
//
// Function attribute denoting that context analysis is disabled for the
// whole function. Forces adding an inline comment as argument.
// token_context_lock() - declare an abstract global context lock instance
// @name: token context lock name
//
// Helper that declares an abstract global context lock instance @name, but not
// backed by a real data structure (linker error if accidentally referenced).
// The type name is `__ctx_lock_@name`.
// token_context_lock_instance() - declare another instance of a global context lock
// @ctx: token context lock previously declared with token_context_lock()
// @name: name of additional global context lock instance
//
// Helper that declares an additional instance @name of the same token context
// lock class @ctx. This is helpful where multiple related token contexts are
// declared, to allow using the same underlying type (`__ctx_lock_@ctx`) as
// function arguments.
// __must_hold() - function attribute, caller must hold exclusive context lock
//
// Function attribute declaring that the caller must hold the given context
// lock instance(s) exclusively.
// __must_not_hold() - function attribute, caller must not hold context lock
//
// Function attribute declaring that the caller must not hold the given context
// lock instance(s).
// __acquires() - function attribute, function acquires context lock exclusively
//
// Function attribute declaring that the function acquires the given context
// lock instance(s) exclusively, but does not release them.
// __cond_acquires() - function attribute, function conditionally
//                     acquires a context lock exclusively
// @ret: abstract value returned by function if context lock acquired
// @x: context lock instance pointer
//
// Function attribute declaring that the function conditionally acquires the
// given context lock instance @x exclusively, but does not release it. The
// function return value @ret denotes when the context lock is acquired.
//
// @ret may be one of: true, false, nonzero, 0, nonnull, NULL.
// __releases() - function attribute, function releases a context lock exclusively
//
// Function attribute declaring that the function releases the given context
// lock instance(s) exclusively. The associated context(s) must be active on
// entry.
// __must_hold_shared() - function attribute, caller must hold shared context lock
//
// Function attribute declaring that the caller must hold the given context
// lock instance(s) with shared access.
// __acquires_shared() - function attribute, function acquires context lock shared
//
// Function attribute declaring that the function acquires the given
// context lock instance(s) with shared access, but does not release them.
// __cond_acquires_shared() - function attribute, function conditionally
//                            acquires a context lock shared
// @ret: abstract value returned by function if context lock acquired
// @x: context lock instance pointer
//
// Function attribute declaring that the function conditionally acquires the
// given context lock instance @x with shared access, but does not release it.
// The function return value @ret denotes when the context lock is acquired.
//
// @ret may be one of: true, false, nonzero, 0, nonnull, NULL.
// __releases_shared() - function attribute, function releases a
//                       context lock shared
//
// Function attribute declaring that the function releases the given context
// lock instance(s) with shared access. The associated context(s) must be
// active on entry.
// sparse defines __CHECKER__; see Documentation/dev-tools/sparse.rst
//
// address spaces
// other
// Attributes
// Note: do not use this directly. Instead, use __alloc_size() since it is conditionally
// available and includes other attributes. For GCC < 9.1, __alloc_size__ gets undefined
// in compiler-gcc.h, due to misbehaviors.
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-alloc_005fsize-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#alloc-size
// Note: users of __always_inline currently do not write "inline" themselves,
// which seems to be required by gcc to apply the attribute according
// to its docs (and also "warning: always_inline function might not be
// inlinable [-Wattributes]" is emitted).
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-always_005finline-function-attribute
// clang: mentioned
// The second argument is optional (default 0), so we use a variadic macro
// to make the shorthand.
//
// Beware: Do not apply this to functions which may return
// ERR_PTRs. Also, it is probably unwise to apply it to functions
// returning extra information in the low bits (but in that case the
// compiler should see some alignment anyway, when the return value is
// massaged by 'flags = ptr & 3; ptr &= ~3;').
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-assume_005faligned-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#assume-aligned
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-cleanup-variable-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#cleanup
// Note the long name.
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-const-function-attribute
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-format-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#format
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-malloc-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#malloc
// Note: deliberately not named '__nonnull', to avoid clashing with glibc's
// __nonnull() when kernel and userspace headers are combined.
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Attributes.html#index-nonnull
// clang: https://clang.llvm.org/docs/AttributeReference.html#nonnull
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-noreturn-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#noreturn
// clang: https://clang.llvm.org/docs/AttributeReference.html#id1
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-packed-type-attribute
// clang: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-packed-variable-attribute
// gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-pure-function-attribute
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-section-function-attribute
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-section-variable-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#section-declspec-allocate
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-warn_005funused_005fresult-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#nodiscard-warn-unused-result
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-cold-function-attribute
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Label-Attributes.html#index-cold-label-attribute
//
// When -falign-functions=N is in use, we must avoid the cold attribute as
// GCC drops the alignment for cold functions. Worse, GCC can implicitly mark
// callees of cold functions as cold themselves, so it's not sufficient to add
// __function_aligned here as that will not ensure that callees are correctly
// aligned.
//
// See:
//
//   https://lore.kernel.org/lkml/Y77%2FqVgvaJidFpYt@FVFF77S0Q05N
//   https://gcc.gnu.org/bugzilla/show_bug.cgi?id=88345#c9
// On x86-64 and arm64 targets, __preserve_most changes the calling convention
// of a function to make the code in the caller as unintrusive as possible. This
// convention behaves identically to the C calling convention on how arguments
// and return values are passed, but uses a different set of caller- and callee-
// saved registers.
//
// The purpose is to alleviates the burden of saving and recovering a large
// register set before and after the call in the caller.  This is beneficial for
// rarely taken slow paths, such as error-reporting functions that may be called
// from hot paths.
//
// Note: This may conflict with instrumentation inserted on function entry which
// does not use __preserve_most or equivalent convention (if in assembly). Since
// function tracing assumes the normal C calling convention, where the attribute
// is supported, __preserve_most implies notrace.  It is recommended to restrict
// use of the attribute to functions that should or already disable tracing.
//
// Optional: not supported by gcc.
//
// clang: https://clang.llvm.org/docs/AttributeReference.html#preserve-most
// Compiler specific macros.
// Severity used in pragma directives
// Some architectures need to provide custom definitions of macros provided
// by linux/compiler-*.h, and can do so using asm/compiler.h. We include that
// conditionally rather than using an asm-generic wrapper in order to avoid
// build failures if any C compilation, which will include this file via an
// -include argument in c_flags, occurs prior to the asm-generic wrappers being
// generated.
// it doesn't make sense on ARM (currently the only user of __naked)
// to trace naked functions because then mcount is called without
// stack and frame pointer being set up and there is no chance to
// restore the lr register to the value before mcount was called.
// Prefer gnu_inline, so that extern inline functions do not emit an
// externally visible function. This makes extern inline behave as per gnu89
// semantics rather than c99. This prevents multiple symbol definition errors
// of extern inline functions at link time.
// A lot of inline functions can cause havoc with function tracing.
// gcc provides both __inline__ and __inline as alternate spellings of
// the inline keyword, though the latter is undocumented. New kernel
// code should only use the inline spelling, but some existing code
// uses __inline__. Since we #define inline above, to ensure
// __inline__ has the same semantics, we need this #define.
//
// However, the spelling __inline is strictly reserved for referring
// to the bare keyword.
// GCC does not warn about unused static inline functions for -Wunused-function.
// Suppress the warning in clang as well by using __maybe_unused, but enable it
// for W=2 build. This will allow clang to find unused functions.
// Rather then using noinline to prevent stack consumption, use
// noinline_for_stack instead.  For documentation reasons.
// Use noinline_for_tracing for functions that should not be inlined.
// For tracing reasons.
// Sanitizer helper attributes: Because using __always_inline and
// __no_sanitize_* conflict, provide helper attributes that will either expand
// to __no_sanitize_* in compilation units where instrumentation is enabled
// (__SANITIZE_*__), or __always_inline in compilation units without
// instrumentation (__SANITIZE_*__ undefined).
//
// The assume attribute is used to indicate that a certain condition is
// assumed to be true. If this condition is violated at runtime, the behavior
// is undefined. Compilers may or may not use this indication to generate
// optimized code.
//
// Note that the clang documentation states that optimizers may react
// differently to this attribute, and this may even have a negative
// performance impact. Therefore this attribute should be used with care.
//
// Optional: only supported since gcc >= 13
// Optional: only supported since clang >= 19
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Statement-Attributes.html#index-assume-statement-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#id13
// Optional: only supported since gcc >= 15
// Optional: only supported since clang >= 18
//
//   gcc: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=108896
// clang: https://clang.llvm.org/docs/AttributeReference.html#counted-by-counted-by-or-null-sized-by-sized-by-or-null
//
// __bdos on clang < 19.1.2 can erroneously return 0:
// https://github.com/llvm/llvm-project/pull/110497
//
// __bdos on clang < 19.1.3 can be off by 4:
// https://github.com/llvm/llvm-project/pull/112636
// Runtime track number of objects pointed to by a pointer member for use by
// CONFIG_FORTIFY_SOURCE and CONFIG_UBSAN_BOUNDS.
//
// Optional: only supported since gcc >= 16
// Optional: only supported since clang >= 22
//
//   gcc: https://gcc.gnu.org/pipermail/gcc-patches/2025-April/681727.html
// clang: https://clang.llvm.org/docs/AttributeReference.html#counted-by-counted-by-or-null-sized-by-sized-by-or-null
// Optional: only supported since gcc >= 15
// Optional: not supported by Clang
//
// gcc: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=117178
// Apply __counted_by() when the Endianness matches to increase test coverage.
// This designates the minimum number of elements a passed array parameter must
// have. For example:
//
//     void some_function(u8 param[at_least 7]);
//
// If a caller passes an array with fewer than 7 elements, the compiler will
// emit a warning.
// Section for code which can't be instrumented at all
// The __cpuidle section is used twofold:
//
//  1) the original use -- identifying if a CPU is 'stuck' in idle state based
//     on it's instruction pointer. See cpu_in_idle().
//
//  2) supressing instrumentation around where cpuidle disables RCU; where the
//     function isn't strictly required for #1, this is interchangeable with
//     noinstr.
// The below symbols may be defined for one or more, but not ALL, of the above
// compilers. We don't consider that to be an error, so set them to nothing.
// For example, some of them are for compiler specific plugins.
// Any place that could be marked with the "alloc_size" attribute is also
// a place to be marked with the "malloc" attribute, except those that may
// be performing a _reallocation_, as that may alias the existing pointer.
// For these, use __realloc_size().
// When the size of an allocated object is needed, use the best available
// mechanism to find it. (For cases where sizeof() cannot be used.)
//
// Optional: only supported since gcc >= 12
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Object-Size-Checking.html
// clang: https://clang.llvm.org/docs/LanguageExtensions.html#evaluating-object-size
// Determine if an attribute has been applied to a variable.
// Using __annotated needs to check for __annotated being available,
// or negative tests may fail when annotation cannot be checked. For
// example, see the definition of __is_cstr().
// __flex_counter() - Get pointer to counter member for the given
//                    flexible array, if it was annotated with __counted_by()
// @FAM: Pointer to flexible array member of an addressable struct instance
//
// For example, with:
//
// 	struct foo {
// 		int counter;
// 		short array[] __counted_by(counter);
// 	} *p;
//
// __flex_counter(p->array) will resolve to &p->counter.
//
// Note that Clang may not allow this to be assigned to a separate
// variable; it must be used directly.
//
// If p->array is unannotated, this returns (void *)NULL.
// Some versions of gcc do not mark 'asm goto' volatile:
//
//  https://gcc.gnu.org/bugzilla/show_bug.cgi?id=103979
//
// We do it here by hand, because it doesn't hurt.
// Clang has trouble with constraints with multiple
// alternative behaviors ("g" , "rm" and "=rm").
// Use __typeof_unqual__() when available.
// Are two types/vars the same type (ignoring qualifiers)?
// __unqual_scalar_typeof(x) - Declare an unqualified scalar type, leaving
// 			       non-scalar types unchanged.
// __signed_scalar_typeof(x) - Declare a signed scalar type, leaving
// 			       non-scalar types unchanged.
// #ifdef __OPTIMIZE__ is only a good approximation; for instance "make
// CFLAGS_foo.o=-Og" defines __OPTIMIZE__, does not elide the conditional code
// and can break compilation with wrong error message(s). Combine with
// -U__OPTIMIZE__ when needed.
// 									 * __noreturn is needed to give the compiler enough			 * information to avoid certain possibly-uninitialized			 * warnings (regardless of the build failing).
// compiletime_assert - break build and emit msg if condition is false
// @condition: a compile-time constant condition to check
// @msg:       a message to emit if condition is false
//
// In tradition of POSIX assert, this macro will break the build if the
// supplied condition is *false*, emitting the supplied error message if the
// compiler has support to do so.
// Helpers for emitting diagnostics in pragmas.
// This version is i.e. to prevent dead stores elimination on @ptr
// where gcc and llvm may behave differently when otherwise using
// normal barrier(): while gcc behavior gets along with a normal
// barrier(), llvm needs an explicit input variable to be assumed
// clobbered. The issue is as follows: while the inline asm might
// access any memory it wants, the compiler could have fit all of
// @ptr into memory registers instead, and since @ptr never escaped
// from that, it proved that the inline asm wasn't touching any of
// it. This version works well with both compilers, i.e. we're telling
// the compiler that the inline asm absolutely may see the contents
// of @ptr. See also: https://llvm.org/bugs/show_bug.cgi?id=15495
// Make the optimizer believe the variable can be manipulated arbitrarily.
// Format: __UNIQUE_ID_<name>_<__COUNTER__>
// data_race - mark an expression as containing intentional data races
//
// This data_race() macro is useful for situations in which data races
// should be forgiven.  One example is diagnostic code that accesses
// shared variables but is not a part of the core synchronization design.
// For example, if accesses to a given variable are protected by a lock,
// except for diagnostic code, then the accesses under the lock should
// be plain C-language accesses and those in the diagnostic code should
// use data_race().  This way, KCSAN will complain if buggy lockless
// accesses to that variable are introduced, even if the buggy accesses
// are protected by READ_ONCE() or WRITE_ONCE().
//
// This macro *does not* affect normal code generation, but is a hint
// to tooling that data races here are to be ignored.  If the access must
// be atomic *and* KCSAN should ignore the access, use both data_race()
// and READ_ONCE(), for example, data_race(READ_ONCE(x)).
// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
// Force the compiler to emit 'sym' as a symbol, so that we can reference
// it from inline assembler. Necessary in case 'sym' could be inlined
// otherwise, or eliminated entirely due to lack of references that are
// visible to the compiler.
// This returns a constant expression while determining if an argument is
// a constant expression, most importantly without evaluating the argument.
// Glory to Martin Uecker <Martin.Uecker@med.uni-goettingen.de>
//
// Details:
// - sizeof() return an integer constant expression, and does not evaluate
//   the value of its operand; it only examines the type of its operand.
// - The results of comparing two integer constant expressions is also
//   an integer constant expression.
// - The first literal "8" isn't important. It could be any literal value.
// - The second literal "8" is to avoid warnings about unaligned pointers;
//   this could otherwise just be "1".
// - (long)(x) is used to avoid warnings about 64-bit types on 32-bit
//   architectures.
// - The C Standard defines "null pointer constant", "(void *)0", as
//   distinct from other void pointers.
// - If (x) is an integer constant expression, then the "* 0l" resolves
//   it into an integer constant expression of value 0. Since it is cast to
//   "void *", this makes the second operand a null pointer constant.
// - If (x) is not an integer constant expression, then the second operand
//   resolves to a void pointer (but not a null pointer constant: the value
//   is not an integer constant 0).
// - The conditional operator's third operand, "(int *)8", is an object
//   pointer (to type "int").
// - The behavior (including the return type) of the conditional operator
//   ("operand1 ? operand2 : operand3") depends on the kind of expressions
//   given for the second and third operands. This is the central mechanism
//   of the macro:
//   - When one operand is a null pointer constant (i.e. when x is an integer
//     constant expression) and the other is an object pointer (i.e. our
//     third operand), the conditional operator returns the type of the
//     object pointer operand (i.e. "int *"). Here, within the sizeof(), we
//     would then get:
//       sizeof(*((int *)(...))  == sizeof(int)  == 4
//   - When one operand is a void pointer (i.e. when x is not an integer
//     constant expression) and the other is an object pointer (i.e. our
//     third operand), the conditional operator returns a "void *" type.
//     Here, within the sizeof(), we would then get:
//       sizeof(*((void *)(...)) == sizeof(void) == 1
// - The equality comparison to "sizeof(int)" therefore depends on (x):
//     sizeof(int) == sizeof(int)     (x) was a constant expression
//     sizeof(int) != sizeof(void)    (x) was not a constant expression
// Similar to statically_true() but produces a constant expression
//
// To be used in conjunction with macros, such as BUILD_BUG_ON_ZERO(),
// which require their input to be a constant expression and for which
// statically_true() would otherwise fail.
//
// This is a trade-off: const_true() requires all its operands to be
// compile time constants. Else, it would always returns false even on
// the most trivial cases like:
//
//   true || non_const_var
//
// On the opposite, statically_true() is able to fold more complex
// tautologies and will return true on expressions such as:
//
//   !(non_const_var * 8 % 4)
//
// For the general case, statically_true() is better.
// DECLARE_FLEX_ARRAY() - Declare a flexible array usable in a union
//
// @TYPE: The type of each flexible array element
// @NAME: The name of the flexible array member
//
// In order to have a flexible array member in a union or alone in a
// struct, it needs to be wrapped in an anonymous struct with at least 1
// named member, but that member can be empty.
// # chars in a path name including nul
// BUILD_BUG_ON_INVALID() permits the compiler to check the validity of the
// expression but avoids the generation of any code, even if that expression
// has side-effects.
// BUILD_BUG_ON_MSG - break compile if a condition is true & emit supplied
// 		      error message.
// @cond: the condition which the compiler should know is false.
// @msg: build-time error message
//
// See BUILD_BUG_ON for description.
// BUILD_BUG_ON - break compile if a condition is true.
// @condition: the condition which the compiler should know is false.
//
// If you have some code which relies on certain constants being equal, or
// some other compile-time-evaluated condition, you should use BUILD_BUG_ON to
// detect if someone changes it.
// BUILD_BUG - break compile if used.
//
// If you have some code that you expect the compiler to eliminate at
// build time, you should use BUILD_BUG to detect if it is
// unexpectedly used.
// static_assert - check integer constant expression at build time
// @expr: expression to be checked
//
// static_assert() is a wrapper for the C11 _Static_assert, with a
// little macro magic to make the message optional (defaulting to the
// stringification of the tested expression).
//
// Contrary to BUILD_BUG_ON(), static_assert() can be used at global
// scope, but requires the expression to be an integer constant
// expression (i.e., it is not enough that __builtin_constant_p() is
// true for expr).
//
// Also note that BUILD_BUG_ON() fails the build if the condition is
// true, while static_assert() fails the build if the expression is
// false.
// objtool annotation to ignore the alternatives and only consider the original
// instruction(s).
// This should be used to refer to an instruction that is considered
// terminating, like a noreturn CALL or UD2 when we know they are not -- eg
// WARN using UD2.
// Annotate a special section entry.  This emables livepatch module generation
// to find and extract individual special section entries as needed.
// reg := -EIO
// reg := -EIO
// Because instrumentation_{begin,end}() can nest, objtool validation considers
// _begin() a +1 and _end() a -1 and computes a sum over the instructions.
// When the value is greater than 0, we consider instrumentation allowed.
//
// There is a problem with code like:
//
// noinstr void foo()
// {
// 	instrumentation_begin();
// 	...
// 	if (cond) {
// 		instrumentation_begin();
// 		...
// 		instrumentation_end();
// 	}
// 	bar();
// 	instrumentation_end();
// }
//
// If instrumentation_end() would be an empty label, like all the other
// annotations, the inner _end(), which is at the end of a conditional block,
// would land on the instruction after the block.
//
// If we then consider the sum of the !cond path, we'll see that the call to
// bar() is with a 0-value, even though, we meant it to happen with a positive
// value.
//
// To avoid this, have _end() be a NOP instruction, this ensures it will be
// part of the condition block and does not escape.
// Format: <modname>__<counter>_<line>_<fn>
// Format: __<prefix>__<iid><id>
// warning conditions
// informational
// __swahw32 - return a word-swapped 32-bit value
// @x: value to wordswap
//
// __swahw32(0x12340000) is 0x00001234
// __swahb32 - return a high and low byte-swapped 32-bit value
// @x: value to byteswap
//
// __swahb32(0x12345678) is 0x34127856
// printk - print a kernel message
// @fmt: format string
//
// This is printk(). It can be called from any context. We want it to work.
//
// If printk indexing is enabled, _printk() is called from printk_index_wrap.
// Otherwise, printk is simply #defined to _printk.
//
// We try to grab the console_lock. If we succeed, it's easy - we log the
// output and call the console drivers.  If we fail to get the semaphore, we
// place the output into the log buffer and return. The current holder of
// the console_sem will notice the new output in console_unlock(); and will
// send it to the consoles before releasing the lock.
//
// One effect of this deferred printing is that code which calls printk() and
// then changes console_loglevel may break. This is because console_loglevel
// is inspected when the actual printing occurs.
//
// See also:
// printf(3)
//
// See the vsnprintf() documentation for format string extensions over C99.
// pr_err - Print an error-level message
// @fmt: format string
// @...: arguments for the format string
//
// This macro expands to a printk with KERN_ERR loglevel. It uses pr_fmt() to
// generate the format string.
// pr_warn - Print a warning-level message
// @fmt: format string
// @...: arguments for the format string
//
// This macro expands to a printk with KERN_WARNING loglevel. It uses pr_fmt()
// to generate the format string.
// pr_info - Print an info-level message
// @fmt: format string
// @...: arguments for the format string
//
// This macro expands to a printk with KERN_INFO loglevel. It uses pr_fmt() to
// generate the format string.
// new instruction
// feature + flags
// source len
// replacement len
// replacement
// alternative assembly primitive:
// If @feature is set, patch in @newinstr_yes, otherwise @newinstr_no.
// Alternative instructions for different CPU types or capabilities.
//
// This allows to use optimized instructions even on generic binary
// kernels.
//
// length of oldinstr must be longer or equal the length of newinstr
// It can be padded with nops as needed.
//
// For non barrier like inlines please define new variants
// without volatile and memory clobber.
// Alternative inline assembly with input.
//
// Peculiarities:
// No memory clobber here.
// Argument numbers start with 1.
// Leaving an unused argument 0 to keep API compatibility.
// Like alternative_input, but with a single output argument
// Atomic operations are already serializing on x86
// The following macros are non-atomic versions of their non-underscored
// counterparts.
// Concatenate two parameters, but allow them to be expanded beforehand.
// N 32-bit words worth of info
// N 32-bit bug flags
// "pse" Page Size Extensions
// "tsc" Time Stamp Counter
// "apic" Onboard APIC
// "sse"
// "pdpe1gb" GB pages
// "rdtscp" RDTSCP
// "rep_good" REP microcode works well
// "popcnt" POPCNT instruction
// "3dnowprefetch" 3DNow prefetch instructions
// "pti" Kernel Page Table Isolation enabled
// "ibpb" Indirect Branch Prediction Barrier without a guaranteed RSB flush
// Intel-defined CPU features, CPUID level 0x00000007:0 (EBX), word 9
// "fsgsbase" RDFSBASE, WRFSBASE, RDGSBASE, WRGSBASE instructions
// "erms" Enhanced REP MOVSB/STOSB instructions
// "smap" Supervisor Mode Access Prevention
// "clflushopt" CLFLUSHOPT instruction
// "clwb" CLWB instruction
// IA32_TSC_DEADLINE and X2APIC MSRs need fencing
// "lass" Linear Address Space Separation
// Fast short REP STOSB
// Non-serializing WRMSR
// INVLPGB and TLBSYNC instructions supported
// "wbnoinvd" WBNOINVD instruction
// "ospke" OS Protection Keys Enable
// Shadow stack
// "la57" 5-level page tables
// "rdpid" RDPID instruction
// "fsrm" Fast Short Rep Mov
// "serialize" SERIALIZE instruction
// LFENCE always serializing / synchronizes RDTSC
// BUG word(s)
// "l1tf" CPU is affected by L1 Terminal Fault
// "div0" AMD DIV0 speculation bug
// Check that the bitops prototypes are sane
// assign_bit - Assign value to a bit in memory
// @nr: the bit to set
// @addr: the address to start counting from
// @value: the value to assign
// round_down - round down to next specified power of 2
// @x: the value to round
// @y: multiple to round down to (must be a power of 2)
//
// Rounds @x down to next multiple of @y (which must be a power of 2).
// To perform arbitrary rounding down, use rounddown() below.
// abs - return absolute value of an argument
// @x: the value.
//
// If it is unsigned type, @x is converted to signed type first.
// char is treated as if it was signed (regardless of whether it really is)
// but the macro's return type is preserved as char.
//
// NOTE, for signed type if @x is the minimum, the returned result is undefined
// as there is not enough bits to represent it as a positive number.
//
// Return: an absolute value of @x.
// min - return minimum of two values of the same or compatible types
// @x: first value
// @y: second value
// max - return maximum of two values of the same or compatible types
// @x: first value
// @y: second value
// max_t - return maximum of two values, using the specified type
// @type: data type to use
// @x: first value
// @y: second value
// in_range - Determine if a value lies within a range.
// @val: Value to test.
// @start: First value in range.
// @len: Number of values in range.
//
// This is more efficient than "if (start <= val && val < (start + len))".
// It also gives a different answer if @start + @len overflows the size of
// the type by a sufficient amount to encompass @val.  Decide for yourself
// which behaviour you want, or prove that start + len never overflow.
// Do not blindly replace one form with the other.
// Use these carefully: no type checking, and uses the arguments
// multiple times. Use for obvious constants only.
// net/core/page_pool.c
//
// page_pool uses additional free bits within this value to store data, see the
// definition of PP_DMA_INDEX_MASK in mm.h
// Overflow Flag
// early_idt_handler_array is an array of entry points referenced in the
// early IDT.  For simplicity, it's a real array with one entry point
// every nine bytes.  That leaves room for an optional 'push $0' if the
// vector has no error code (two bytes), a 'push $vector_number' (two
// bytes), and a jump to the common entry code (up to five bytes).
// Load a segment. Fall back on loading the zero segment if something goes
// wrong.  This variant assumes that loading zero fully clears the segment.
// This is always the case on Intel CPUs and, even on 64-bit AMD CPUs, any
// failure to fully clear the cached descriptor is only observable for
// FS and GS.
// Save a segment register away:
// Generate a per-CPU add to memory instruction and optimize code
// if one is added or subtracted.
// Add return operation
// this_cpu_read() makes the compiler load the per-CPU variable every time
// it is accessed while this_cpu_read_stable() allows the value to be cached.
// this_cpu_read_stable() is more efficient and can be used if its value
// is guaranteed to be valid across CPUs.  The current users include
// current_task and cpu_current_top_of_stack, both of which are
// actually per-thread variables implemented as per-CPU variables and
// thus stable for the duration of the respective task.
// No such file or directory
// I/O error
// No such device or address
// Argument list too long
// Try again
// Out of memory
// Bad address
// Device or resource busy
// Cross-device link
// No such device
// Invalid argument
// Not a typewriter
// Text file busy
// No space left on device
// Math result not representable
// Invalid system call number
// Operation would block
// Package not installed
// Value too large for defined data type
// Operation not supported on transport endpoint
// Memory page has hardware error
// restart by calling sys_restart_syscall
// Operation is not supported
// An exchange-type operation, which takes a value and a pointer, and
// returns the old value.
// Note: no "lock" prefix even on SMP: xchg always implies lock anyway.
// Since this is generally used to protect other memory information, we
// use "asm volatile" and "memory" clobbers to prevent gcc from moving
// information around.
// Atomic compare and exchange.  Compare OLD with MEM, if identical,
// store NEW in MEM.  Return the initial value in MEM.  Success is
// indicated by comparing RETURN with OLD.
// xadd() adds "inc" to "*ptr" and atomically returns the previous
// value of "*ptr".
//
// xadd() is locked when multiple CPUs are online
// Physical address zone modifiers (see linux/mmzone.h - low four bits)
//
// Do not put any conditional on these. If necessary modify the definitions
// without the underscores and use them consistently. The definitions here may
// be used in bit comparisons.
// ZONE_MOVABLE allowed
// DOC: Page mobility and placement hints
//
// Page mobility and placement hints
//
// These flags provide hints about how mobile the page is. Pages with similar
// mobility are placed within the same pageblocks to minimise problems due
// to external fragmentation.
//
// %__GFP_MOVABLE (also a zone modifier) indicates that the page can be
// moved by page migration during memory compaction or can be reclaimed.
//
// %__GFP_RECLAIMABLE is used for slab allocations that specify
// SLAB_RECLAIM_ACCOUNT and whose pages can be freed via shrinkers.
//
// %__GFP_WRITE indicates the caller intends to dirty the page. Where possible,
// these pages will be spread between local zones to avoid all the dirty
// pages being in one zone (fair zone allocation policy).
//
// %__GFP_HARDWALL enforces the cpuset memory allocation policy.
//
// %__GFP_THISNODE forces the allocation to be satisfied from the requested
// node with no fallbacks or placement policy enforcements.
//
// %__GFP_ACCOUNT causes the allocation to be accounted to the active
// cgroup context.
// DOC: Reclaim modifiers
//
// Reclaim modifiers
// Please note that all the following flags are only applicable to sleepable
// allocations (e.g. %GFP_NOWAIT and %GFP_ATOMIC will ignore them).
//
// %__GFP_IO can start physical IO.
//
// %__GFP_FS can call down to the low-level FS. Clearing the flag avoids the
// allocator recursing into the filesystem which might already be holding
// locks.
//
// %__GFP_DIRECT_RECLAIM indicates that the caller may enter direct reclaim.
// This flag can be cleared to avoid unnecessary delays when a fallback
// option is available.
//
// %__GFP_KSWAPD_RECLAIM indicates that the caller wants to wake kswapd when
// the low watermark is reached and have it reclaim pages until the high
// watermark is reached. A caller may wish to clear this flag when fallback
// options are available and the reclaim is likely to disrupt the system. The
// canonical example is THP allocation where a fallback is cheap but
// reclaim/compaction may cause indirect stalls.
//
// %__GFP_RECLAIM is shorthand to allow/forbid both direct and kswapd reclaim.
//
// The default allocator behavior depends on the request size. We have a concept
// of so-called costly allocations (with order > %PAGE_ALLOC_COSTLY_ORDER).
// !costly allocations are too essential to fail so they are implicitly
// non-failing by default (with some exceptions like OOM victims might fail so
// the caller still has to check for failures) while costly requests try to be
// not disruptive and back off even without invoking the OOM killer.
// The following three modifiers might be used to override some of these
// implicit rules. Please note that all of them must be used along with
// %__GFP_DIRECT_RECLAIM flag.
//
// %__GFP_NORETRY: The VM implementation will try only very lightweight
// memory direct reclaim to get some memory under memory pressure (thus
// it can sleep). It will avoid disruptive actions like OOM killer. The
// caller must handle the failure which is quite likely to happen under
// heavy memory pressure. The flag is suitable when failure can easily be
// handled at small cost, such as reduced throughput.
//
// %__GFP_RETRY_MAYFAIL: The VM implementation will retry memory reclaim
// procedures that have previously failed if there is some indication
// that progress has been made elsewhere.  It can wait for other
// tasks to attempt high-level approaches to freeing memory such as
// compaction (which removes fragmentation) and page-out.
// There is still a definite limit to the number of retries, but it is
// a larger limit than with %__GFP_NORETRY.
// Allocations with this flag may fail, but only when there is
// genuinely little unused memory. While these allocations do not
// directly trigger the OOM killer, their failure indicates that
// the system is likely to need to use the OOM killer soon.  The
// caller must handle failure, but can reasonably do so by failing
// a higher-level request, or completing it only in a much less
// efficient manner.
// If the allocation does fail, and the caller is in a position to
// free some non-essential memory, doing so could benefit the system
// as a whole.
//
// %__GFP_NOFAIL: The VM implementation _must_ retry infinitely: the caller
// cannot handle allocation failures. The allocation could block
// indefinitely but will never return with failure. Testing for
// failure is pointless.
// It _must_ be blockable and used together with __GFP_DIRECT_RECLAIM.
// It should _never_ be used in non-sleepable contexts.
// New users should be evaluated carefully (and the flag should be
// used only when there is no reasonable failure policy) but it is
// definitely preferable to use the flag rather than opencode endless
// loop around allocator.
// Allocating pages from the buddy with __GFP_NOFAIL and order > 1 is
// not supported. Please consider using kvmalloc() instead.
// Caller can reclaim
// DOC: Action modifiers
//
// Action modifiers
//
// %__GFP_NOWARN suppresses allocation failure reports.
//
// %__GFP_COMP address compound page metadata.
//
// %__GFP_ZERO returns a zeroed page on success.
//
// %__GFP_ZEROTAGS zeroes memory tags at allocation time. Setting memory tags at
// the same time as zeroing memory (e.g., with __GFP_ZERO) has minimal
// additional performance impact. However, __GFP_ZEROTAGS also zeroes the tags
// even if memory is not getting zeroed at allocation time (e.g.,
// with init_on_free).
//
// %__GFP_SKIP_KASAN makes KASAN skip unpoisoning on page allocation.
// Used for userspace and vmalloc pages; the latter are unpoisoned by
// kasan_unpoison_vmalloc instead. If passed to vmalloc, kasan_unpoison_vmalloc
// is skipped too. For userspace pages, results in poisoning being skipped as
// well, see should_skip_kasan_poison for details. Only effective in HW_TAGS mode.
// Disable lockdep for GFP context tracking
// DOC: Useful GFP flag combinations
//
// Useful GFP flag combinations
//
// Useful GFP flag combinations that are commonly used. It is recommended
// that subsystems start with one of these combinations and then set/clear
// %__GFP_FOO flags as necessary.
//
// %GFP_ATOMIC users can not sleep and need the allocation to succeed. A lower
// watermark is applied to allow access to "atomic reserves".
// The current implementation doesn't support NMI, nor contexts that disable
// preemption under PREEMPT_RT. This includes raw_spin_lock() and plain
// preempt_disable() - see "Memory allocation" in
// Documentation/core-api/real-time/differences.rst for more info.
//
// %GFP_KERNEL is typical for kernel-internal allocations. The caller requires
// %ZONE_NORMAL or a lower zone for direct access but can direct reclaim.
//
// %GFP_KERNEL_ACCOUNT is the same as GFP_KERNEL, except the allocation is
// accounted to the active cgroup context.
//
// %GFP_NOWAIT is for kernel allocations that should not stall for direct
// reclaim, start physical IO or use any filesystem callback.  It is very
// likely to fail to allocate memory, even for very small allocations.
// The same restrictions on calling contexts apply as for %GFP_ATOMIC.
//
// %GFP_NOIO will use direct reclaim to discard clean pages or slab pages
// that do not require the starting of any physical IO.
// Please try to avoid using this flag directly and instead use
// memalloc_noio_{save,restore} to mark the whole scope which cannot
// perform any IO with a short explanation why. All allocation requests
// will inherit GFP_NOIO implicitly.
//
// %GFP_NOFS will use direct reclaim but will not use any filesystem interfaces.
// Please try to avoid using this flag directly and instead use
// memalloc_nofs_{save,restore} to mark the whole scope which cannot/shouldn't
// recurse into the FS layer with a short explanation why. All allocation
// requests will inherit GFP_NOFS implicitly.
//
// %GFP_USER is for userspace allocations that also need to be directly
// accessibly by the kernel or hardware. It is typically used by hardware
// for buffers that are mapped to userspace (e.g. graphics) that hardware
// still must DMA to. cpuset limits are enforced for these allocations.
//
// %GFP_DMA exists for historical reasons and should be avoided where possible.
// The flags indicates that the caller requires that the lowest zone be
// used (%ZONE_DMA or 16M on x86-64). Ideally, this would be removed but
// it would require careful auditing as some users really require it and
// others use the flag to avoid lowmem reserves in %ZONE_DMA and treat the
// lowest zone as a type of emergency reserve.
//
// %GFP_DMA32 is similar to %GFP_DMA except that the caller requires a 32-bit
// address. Note that kmalloc(..., GFP_DMA32) does not return DMA32 memory
// because the DMA32 kmalloc cache array is not implemented.
// (Reason: there is no such user in kernel).
//
// %GFP_HIGHUSER is for userspace allocations that may be mapped to userspace,
// do not need to be directly accessible by the kernel but that cannot
// move once in use. An example may be a hardware allocation that maps
// data directly into userspace but has no addressing limitations.
//
// %GFP_HIGHUSER_MOVABLE is for userspace allocations that the kernel does not
// need direct access to but can use kmap() when access is required. They
// are expected to be movable via page reclaim or page migration. Typically,
// pages on the LRU would also be allocated with %GFP_HIGHUSER_MOVABLE.
//
// %GFP_TRANSHUGE and %GFP_TRANSHUGE_LIGHT are used for THP allocations. They
// are compound allocations that will generally fail quickly if memory is not
// available and will not wake kswapd/kcompactd on failure. The _LIGHT
// version does not attempt reclaim/compaction at all and is by default used
// in page fault path, while the non-light is used by khugepaged.
// No static inline type checking - see Subtlety (1) above.
// N 32-bit words worth of info
// The preempt_count offset needed for things like:
//
//  spin_lock_bh()
//
// Which need to disable both preemption (CONFIG_PREEMPT_COUNT) and
// softirqs, such that unlock sequences of:
//
//  spin_unlock();
//  local_bh_enable();
//
// Work as expected.
// signal notifications exist
// rescheduling necessary
// idle is polling for TIF_NEED_RESCHED
// re-arm the timer
// This function is needed by all SMP systems. It must _always_ be valid
// from the initial startup.
// SCHED_ISO: reserved but not implemented yet
// disable sas during sighandling
// For trivial one-depth nesting of a lock-class, the following
// global define can be used. (Subsystems with multiple levels
// of nesting should define their own lock-nesting subclasses.)
// Writer mask
// Remapping rwlock architecture specific functions to the corresponding
// queued rwlock functions.
// SEQCOUNT_LOCKNAME()	- Instantiate seqcount_LOCKNAME_t and helpers
// seqprop_LOCKNAME_*()	- Property accessors for seqcount_LOCKNAME_t
//
// @lockname:		"LOCKNAME" part of seqcount_LOCKNAME_t
// @locktype:		LOCKNAME canonical C data type
// @preemptible:	preemptibility of above locktype
// @lockbase:		prefix for associated lock/unlock
// 									 * Re-read the sequence counter since the (possibly			 * preempted) writer made progress.
// PREEMPT_RT relies on the above LOCK+UNLOCK
// __read_seqcount_begin() - begin a seqcount_t read section
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
//
// Return: count to be passed to read_seqcount_retry()
// raw_read_seqcount_begin() - begin a seqcount_t read section w/o lockdep
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
//
// Return: count to be passed to read_seqcount_retry()
// read_seqcount_begin() - begin a seqcount_t read critical section
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
//
// Return: count to be passed to read_seqcount_retry()
// raw_read_seqcount() - read the raw seqcount_t counter value
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
//
// raw_read_seqcount opens a read critical section of the given
// seqcount_t, without any lockdep checking, and without checking or
// masking the sequence counter LSB. Calling code is responsible for
// handling that.
//
// Return: count to be passed to read_seqcount_retry()
// raw_seqcount_try_begin() - begin a seqcount_t read critical section
//                            w/o lockdep and w/o counter stabilization
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
// @start: count to be passed to read_seqcount_retry()
//
// Similar to raw_seqcount_begin(), except it enables eliding the critical
// section entirely if odd, instead of doing the speculation knowing it will
// fail.
//
// Useful when counter stabilization is more or less equivalent to taking
// the lock and there is a slowpath that does that.
//
// If true, start will be set to the (even) sequence count read.
//
// Return: true when a read critical section is started.
// read_seqcount_retry() - end a seqcount_t read critical section
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
// @start: count, from read_seqcount_begin()
//
// read_seqcount_retry closes the read critical section of given
// seqcount_t.  If the critical section was invalid, it must be ignored
// (and typically retried).
//
// Return: true if a read section retry is required, else false
// seqlock_init() - dynamic initializer for seqlock_t
// @sl: Pointer to the seqlock_t instance
// read_seqlock_excl_irqsave() - start a non-interruptible seqlock_t
// 				 locking reader section
// @lock:  Pointer to seqlock_t
// @flags: Stack-allocated storage for saving caller's local interrupt
//         state, to be passed to read_sequnlock_excl_irqrestore().
//
// _irqsave variant of read_seqlock_excl(). Use this only if the seqlock_t
// write side section, *or other read sections*, can be invoked from a
// hardirq context.
// Same as rb_first(), but O(1)
// Don't call further
// folio_page - Return a page from a folio.
// @folio: The folio.
// @n: The page number to return.
//
// @n is relative to the start of the folio.  This function does not
// check that the page number lies within @folio; the caller is presumed
// to have a reference to the page.
// Page flags policies wrt compound pages
//
// PF_POISONED_CHECK
//     check if this struct page poisoned/uninitialized
//
// PF_ANY:
//     the page flag is relevant for small, head and tail pages.
//
// PF_HEAD:
//     for compound page all operations related to the page flag applied to
//     head page.
//
// PF_NO_TAIL:
//     modifications of the page flag must be done on small or head pages,
//     checks can be done on tail pages too.
//
// PF_NO_COMPOUND:
//     the page flag is not relevant for compound pages.
//
// PF_SECOND:
//     the page flag is stored in the first tail page.
// Which page is the flag stored in
// Macros to create function definitions for page flags
// On an anonymous folio mapped into a user virtual memory area,
// folio->mapping points to its anon_vma, not to a struct address_space;
// with the FOLIO_MAPPING_ANON bit set to distinguish it.  See rmap.h.
//
// On an anonymous folio in a VM_MERGEABLE area, if CONFIG_KSM is enabled,
// the FOLIO_MAPPING_ANON_KSM bit may be set along with the FOLIO_MAPPING_ANON
// bit; and then folio->mapping points, not to an anon_vma, but to a private
// structure which KSM associates with that merged folio.  See ksm.h.
//
// Please note that, confusingly, "folio_mapping" refers to the inode
// address_space which maps the folio from disk; whereas "folio_mapped"
// refers to user virtual address space into which the folio is mapped.
//
// For slab pages, since slab reuses the bits in struct page to store its
// internal states, the folio->mapping does not exist as such, nor do
// these flags below.  So in order to avoid testing non-existent bits,
// please make sure that folio_test_slab(folio) actually evaluates to
// false before calling the following functions (e.g., folio_test_anon).
// See mm/slab.h.
// local_lock - Acquire a per CPU local lock
// @lock:	The lock variable
// local_lock_irq - Acquire a per CPU local lock and disable interrupts
// @lock:	The lock variable
// local_lock_irqsave - Acquire a per CPU local lock, save and disable
// 			 interrupts
// @lock:	The lock variable
// @flags:	Storage for interrupt flags
// local_unlock - Release a per CPU local lock
// @lock:	The lock variable
// local_unlock_irq - Release a per CPU local lock and enable interrupts
// @lock:	The lock variable
// local_unlock_irqrestore - Release a per CPU local lock and restore
// 			      interrupt flags
// @lock:	The lock variable
// @flags:      Interrupt flags to restore
// local_trylock_init - Runtime initialize a lock instance
// @lock:	The lock variable
// Maximum number of zones on a zonelist
// zone_idx() returns 0 for the ZONE_DMA zone, 1 for the ZONE_NORMAL zone, etc.
// GFP_ZONE_BAD is a bitmap for all combinations of __GFP_DMA, __GFP_DMA32
// __GFP_HIGHMEM and __GFP_MOVABLE that are not permitted. One flag per
// entry starting with bit 0. Bit is set if the combination is not
// allowed.
// Implementation specific max size
// Currently stuck as a macro due to indirect forward reference to
// linux/mmzone.h's __section_mem_map_addr() definition:
// All top-level MITIGATION_PAGE_TABLE_ISOLATION page tables are order-1 pages
// (8k-aligned and 8k in size).  The kernel one is at the beginning 4k and
// the user one is in the last 4k.  To switch between them, you
// just need to flip the 12th bit in their addresses.
// Please ignore the name of this function.  It should be called
// switch_to_kernel_thread().
//
// enter_lazy_tlb() is a hint from the scheduler that we are entering a
// kernel thread or other context without an mm.  Acceptable implementations
// include doing nothing whatsoever, switching to init_mm, or various clever
// lazy tricks to try to minimize TLB flushes.
//
// The scheduler reserves the right to call enter_lazy_tlb() several times
// in a row.  It will notify us that we're going back to a real mm by
// calling switch_mm_irqs_off().
// Keep stack-allocated flush_tlb_info cacheline aligned, but cap the
// alignment to avoid excessive stack usage on large-cacheline systems.
// This is used for both get_user() and __get_user() to expand to
// the proper special function call that has odd calling conventions
// due to returning both a value and an error, and that depends on
// the size of the pointer passed in.
//
// Careful: we have to cast the result to the type of the pointer
// for sign reasons.
//
// The use of _ASM_DX as the register specifier is a bit of a
// simplification, as gcc only cares about it as the starting point
// and not size: for a 64-bit value it will use %ecx:%edx on 32 bits
// (%ecx being the next register in gcc's x86 register sequence), and
// %rdx on 64 bits.
//
// Clang/LLVM cares about the size of the register, but still wants
// the base register for something that ends up being a pair.
// get_user - Get a simple variable from user space.
// @x:   Variable to store result.
// @ptr: Source address, in user space.
//
// Context: User context only. This function may sleep if pagefaults are
//          enabled.
//
// This macro copies a single simple variable from user space to kernel
// space.  It supports simple types like char and int, but not larger
// data types like structures or arrays.
//
// @ptr must have pointer-to-simple-variable type, and the result of
// dereferencing @ptr must be assignable to @x without a cast.
//
// Return: zero on success, or -EFAULT on error.
// On error, the variable @x is set to zero.
// The bottom two bits of the slot determine how the remaining bits in the
// slot are interpreted:
//
// 00 - data pointer
// 10 - internal entry
// x1 - value entry
//
// The internal entry may be a pointer to the next level in the tree, a
// sibling entry, or an indicator that the entry in this slot has been moved
// to another location in the tree and the lookup should be restarted.  While
// NULL fits the 'data pointer' pattern, it means that there is no entry in
// the tree for this index (no matter what level of the tree it is found at).
// This means that storing a NULL entry in the tree is the same as deleting
// the entry from the tree.
// direct disk access hint
// Reserved kernel ranges [-100], [-10000, -40000].
// Special value for dirfd used to
// indicate openat should use the
// current working directory.
// Reserved for per-syscall flags	0xff.
// Do not follow symbolic
// links.
// Allow empty relative
// pathname to operate on dirfd
// directly.
// The 10 bits between the priority class and the priority level are used to
// optionally define I/O hints for any combination of I/O priority class and
// level. Depending on the kernel configuration, I/O scheduler being used and
// the target I/O device being used, hints can influence how I/Os are processed
// without affecting the I/O scheduling ordering defined by the I/O priority
// class and level.
// per-IO, return -EAGAIN if operation would block
// per-IO O_APPEND
// per-IO negation of O_APPEND
// Atomic Write
// buffered IO that drops the cache after reading or writing data
// mask of flags supported by the kernel
// Generate SD flag bits
// PCI/ISA I/O ports
// Resource extended types
// Userland may not map this resource
// No address assigned yet
// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
// We overload the LSB of the page pointer to indicate whether it's
// a valid sg entry, or whether it points to the start of a new scatterlist.
// Those low bits are there for everyone! (thanks mason :-)
// Device/Port type
// Root Port
// Link Control
// Root Control
// Link Control 2
// The PCI interface treats multi-function devices as independent
// devices.  The slot/function address of each device is encoded
// in a single byte as follows:
//
// 	7:3 = slot
// 	2:0 = function
//
// PCI_DEVFN(), PCI_SLOT(), and PCI_FUNC() are defined in uapi/linux/pci.h.
// In the interest of not exposing interfaces to user-space unnecessarily,
// the following kernel-only defines are being added here.
// Error values that may be returned by PCI functions
// Allow INTx interrupts
// Allow MSI interrupts
// Allow MSI-X interrupts
// These helpers provide future and backwards compatibility
// for accessing popular PCI BAR info
// DMA_ATTR_NO_WARN: This tells the DMA-mapping subsystem to suppress
// allocation failure reports (similarly to __GFP_NOWARN).
// DMA_ATTR_MMIO - Indicates memory-mapped I/O (MMIO) region for DMA mapping
//
// This attribute indicates the physical address is not normal system
// memory. It may not be used with kmap*()/phys_to_virt()/phys_to_page()
// functions, it may not be cacheable, and access using CPU load/store
// instructions may not be allowed.
//
// Usually this will be used to describe MMIO addresses, or other non-cacheable
// register addresses. When DMA mapping this sort of address we call
// the operation Peer to Peer as a one device is DMA'ing to another device.
// For PCI devices the p2pdma APIs must be used to determine if DMA_ATTR_MMIO
// is appropriate.
//
// For architectures that require cache flushing for DMA coherence
// DMA_ATTR_MMIO will not perform any cache flushing. The address
// provided must never be mapped cacheable into the CPU.
// A dma_addr_t can hold any valid DMA or bus address for the platform.  It can
// be given to a device to use as a DMA source or target.  It is specific to a
// given device and there may be a translation between the CPU physical address
// space and the bus address space.
//
// DMA_MAPPING_ERROR is the magic error code if a mapping failed.  It should not
// be used directly in drivers, but checked for using dma_mapping_error()
// instead.
// Use the high bit to mark if we used swiotlb for one or more ranges.
// hlist_nulls_next_rcu - returns the element of the list after @node.
// @node: element of the list.
// hlist_nulls_pprev_rcu - returns the dereferenced pprev of @node.
// @node: element of the list.
// task_css_check - obtain css for (task, subsys) w/ extra access conds
// @task: the target task
// @subsys_id: the target subsystem ID
// @__c: extra condition expression to be passed to rcu_dereference_check()
//
// Return the cgroup_subsys_state for the (@task, @subsys_id) pair.  The
// synchronization rules are the same as task_css_set_check().
// The maximum constant udelay/ndelay value picked out of thin air to prevent
// too long constant udelays/ndelays.
// PT_TRACE_* event enable flags
// test_set_buffer_foo() and test_clear_buffer_foo()
// 8 bit slave DMA, channels 0..3
// 16 bit master DMA, ch 4(=slave input)..7
// single-channel mask (w)
// mode register (w)
// clear pointer flip-flop (w)
// single-channel mask (w)
// mode register (w)
// clear pointer flip-flop (w)
// DMA page registers
// Skipped when running bindgen due to a libclang issue;
// see https://github.com/rust-lang/rust-bindgen/issues/2244.
// Annotating a function/variable with __retain tells the compiler to place
// the object in its own section and set the flag SHF_GNU_RETAIN. This flag
// instructs the linker to retain the object during garbage-cleanup or LTO
// phases.
//
// Note that the __used macro is also used to prevent functions or data
// being optimized out, but operates at the compiler/IR-level and may still
// allow unintended removal of objects during linking.
//
// Optional: only supported since gcc >= 11, clang >= 13
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-retain-function-attribute
// clang: https://clang.llvm.org/docs/AttributeReference.html#retain
// Optional: only supported since gcc >= 15, clang >= 19
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html#index-_005f_005fbuiltin_005fcounted_005fby_005fref
// clang: https://clang.llvm.org/docs/LanguageExtensions.html#builtin-counted-by-ref
/// @a is a power of 2 value
macro_rules! ALIGN__2 {
($x:expr, $a:expr) => {
__ALIGN_KERNEL__2!($x, $a)
};
}

macro_rules! BITS_PER_TYPE {
($type:ty) => {
(core::mem::size_of::<$type>() as u64).wrapping_mul(8i32 as $type)
};
}

macro_rules! BIT_MASK {
($nr:expr) => {
UL!() << ($nr) % 64
};
}

macro_rules! BIT_WORD {
($nr:expr) => {
($nr) / 64
};
}

macro_rules! BIT_WORD__2 {
($nr:expr) => {
($nr) / 64
};
}

macro_rules! BIT {
($nr:expr) => {
UL!() << $nr
};
}

macro_rules! BUGFLAG_TAINT {
($taint:expr) => {
($taint) << 8i32
};
}

macro_rules! COMPLETION_INITIALIZER_ONSTACK {
($work:expr) => {
{
let mut __v: *mut completion;
init_completion(core::ptr::addr_of_mut!($work));
__v = core::ptr::addr_of_mut!($work);
unsafe { *__v }
}
};
}

macro_rules! GENMASK_INPUT_CHECK {
() => {
core::mem::size_of::<__SlateRecord_find_h_67>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__2 {
() => {
core::mem::size_of::<__SlateRecord_find_h_207>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__3 {
() => {
core::mem::size_of::<__SlateRecord_find_h_316>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__4 {
() => {
core::mem::size_of::<__SlateRecord_find_h_97>() as i32
};
}

/// Generate a mask for the specified type @t. Additional checks are made to
/// guarantee the value returned fits in that type, relying on
/// -Wshift-count-overflow compiler check to detect incompatible arguments.
/// For example, all these create build errors or warnings:
///
/// - GENMASK(15, 20): wrong argument order
/// - GENMASK(72, 15): doesn't fit unsigned long
/// - GENMASK_U32(33, 15): doesn't fit in a u32
macro_rules! GENMASK_TYPE__2 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__2!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub($h))
};
}

/// Generate a mask for the specified type @t. Additional checks are made to
/// guarantee the value returned fits in that type, relying on
/// -Wshift-count-overflow compiler check to detect incompatible arguments.
/// For example, all these create build errors or warnings:
///
/// - GENMASK(15, 20): wrong argument order
/// - GENMASK(72, 15): doesn't fit unsigned long
/// - GENMASK_U32(33, 15): doesn't fit in a u32
macro_rules! GENMASK_TYPE {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub($h))
};
}

/// Generate a mask for the specified type @t. Additional checks are made to
/// guarantee the value returned fits in that type, relying on
/// -Wshift-count-overflow compiler check to detect incompatible arguments.
/// For example, all these create build errors or warnings:
///
/// - GENMASK(15, 20): wrong argument order
/// - GENMASK(72, 15): doesn't fit unsigned long
/// - GENMASK_U32(33, 15): doesn't fit in a u32
macro_rules! GENMASK_TYPE__3 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__4!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub($h))
};
}

/// Generate a mask for the specified type @t. Additional checks are made to
/// guarantee the value returned fits in that type, relying on
/// -Wshift-count-overflow compiler check to detect incompatible arguments.
/// For example, all these create build errors or warnings:
///
/// - GENMASK(15, 20): wrong argument order
/// - GENMASK(72, 15): doesn't fit unsigned long
/// - GENMASK_U32(33, 15): doesn't fit in a u32
macro_rules! GENMASK_TYPE__4 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__3!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub($h))
};
}

macro_rules! GENMASK {
($h:expr, $l:expr) => {
GENMASK_TYPE!(u64, $h, $l)
};
}

macro_rules! GENMASK__2 {
($h:expr, $l:expr) => {
GENMASK_TYPE__3!(u64, $h, $l)
};
}

macro_rules! GENMASK__3 {
($h:expr, $l:expr) => {
GENMASK_TYPE__2!(u64, $h, $l)
};
}

macro_rules! GENMASK__4 {
($h:expr, $l:expr) => {
GENMASK_TYPE__4!(u64, $h, $l)
};
}

macro_rules! IOPRIO_PRIO_CLASS {
($ioprio:expr) => {
(($ioprio) as i32) >> 13i32 & 7i32
};
}

/// IS_ENABLED(CONFIG_FOO) evaluates to 1 if CONFIG_FOO is set to 'y' or 'm',
/// 0 otherwise.  Note that CONFIG_FOO=y results in "#define CONFIG_FOO 1" in
/// autoconf.h, while CONFIG_FOO=m results in "#define CONFIG_FOO_MODULE 1".
macro_rules! IS_ENABLED__2 {
() => {
__or!()
};
}

/// IS_ENABLED(CONFIG_FOO) evaluates to 1 if CONFIG_FOO is set to 'y' or 'm',
/// 0 otherwise.  Note that CONFIG_FOO=y results in "#define CONFIG_FOO 1" in
/// autoconf.h, while CONFIG_FOO=m results in "#define CONFIG_FOO_MODULE 1".
macro_rules! IS_ENABLED {
() => {
__is_defined!()
};
}

/// IS_ERR_VALUE - Detect an error pointer.
/// @x: The pointer to check.
///
/// Like IS_ERR(), but does not generate a compiler warning if result is unused.
macro_rules! IS_ERR_VALUE {
($x:expr) => {
unlikely__2!(((($x) as *mut ()) as u64) >= (4095i32.wrapping_neg() as u64))
};
}

macro_rules! PASS_BUCKET_PARAMS {
($_size:expr) => {
$_size
};
}

macro_rules! PASS_KMALLOC_PARAMS {
($_size:expr) => {
PASS_BUCKET_PARAMS!($_size)
};
}

macro_rules! PASS_TOKEN_PARAMS {
($_size:expr) => {
$_size
};
}

macro_rules! PFN_PHYS {
($x:expr) => {
(($x) as u64) << 12i32
};
}

macro_rules! RB_CLEAR_NODE {
($node:expr) => {
unsafe {
(*$node).__rb_parent_color = ($node) as u64;
}
};
}

/// RCU_INITIALIZER() - statically initialize an RCU-protected global variable
/// @v: The value to statically initialize with.
macro_rules! RCU_INITIALIZER {
($v:expr) => {
$v
};
}

macro_rules! RCU_LOCKDEP_WARN {
($c:expr) => {
loop {
let __v: bool = false && $c;
if !__v {
break;
}
}
};
}

macro_rules! RELOC_HIDE {
($ptr:expr, $off:expr) => {
(($ptr) as u64).wrapping_add(($off) as u64) as *mut ()
};
}

macro_rules! SB_NR_TO_BIT {
($sb:expr, $bitnr:expr) => {
($bitnr) & (1u32 << unsafe { (*$sb).shift }).wrapping_sub(1)
};
}

macro_rules! SB_NR_TO_INDEX {
($sb:expr, $bitnr:expr) => {
($bitnr) >> unsafe { (*$sb).shift }
};
}

macro_rules! UL {
() => {
_UL!()
};
}

macro_rules! _AC {
() => {
1u64
};
}

macro_rules! _AC__2 {
() => {
16045481047390945280u64
};
}

macro_rules! _BITUL {
($x:expr) => {
_UL!() << $x
};
}

macro_rules! _UL {
() => {
_AC!()
};
}

macro_rules! __ALIGN_KERNEL_MASK {
($x:expr, $mask:expr) => {
$x.wrapping_add($mask) & !$mask
};
}

macro_rules! __ALIGN_KERNEL__2 {
($x:expr, $a:expr) => {
__ALIGN_KERNEL_MASK!($x, (($a) as u32).wrapping_sub(1))
};
}

macro_rules! __IRQ_MASK {
($x:expr) => {
(1u64 << $x).wrapping_sub(1)
};
}

macro_rules! __READ_ONCE {
($x:expr) => {
unsafe { core::ptr::read_volatile(core::ptr::addr_of!($x) as *const i32) }
};
}

macro_rules! __READ_ONCE__2 {
($x:expr) => {
unsafe { core::ptr::read_volatile(core::ptr::addr_of!($x) as *const i64) }
};
}

macro_rules! __WRITE_ONCE__13 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut u64, ($val) as u64) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE__8 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut u32, $val) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE__2 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut i32, $val) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut *mut list_head, $val) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE__12 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut *mut *mut blk_mq_hw_ctx, $val) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE__3 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut u64, $val) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE__4 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut u32, ($val) as u32) };
if true {
break;
}
}
};
}

macro_rules! ____is_defined {
() => {
0i32
};
}

macro_rules! ____or {
() => {
1i32
};
}

macro_rules! ___wait_cond_timeout {
($condition:expr, $__ret:expr) => {
{
let mut __v: bool;
let __v2: bool = $condition;
if __v2 && ($__ret) == 0 {
$__ret = 1;
}
__v = __v2 || ($__ret) == 0;
__v
}
};
}

/// __acquire() - function to acquire context lock exclusively
/// @x: context lock instance pointer
///
/// No-op function that acquires the given context lock instance @x exclusively.
macro_rules! __acquire {
($x:expr) => {
__acquire_ctx_lock__2(($x) as *const spinlock);
};
}

/// __acquire_shared() - function to acquire context lock shared
/// @x: context lock instance pointer
///
/// No-op function that acquires the given context lock instance @x with shared
/// access.
macro_rules! __acquire_shared {
($x:expr) => {
__acquire_shared_ctx_lock__6($x);
};
}

/// __acquire_shared() - function to acquire context lock shared
/// @x: context lock instance pointer
///
/// No-op function that acquires the given context lock instance @x with shared
/// access.
macro_rules! __acquire_shared__2 {
($x:expr) => {
__acquire_shared_ctx_lock__8(($x) as *const srcu_struct);
};
}

macro_rules! __alloc_percpu {
($_size:expr, $_align:expr) => {
alloc_hooks!(unsafe { pcpu_alloc_noprof($_size, $_align, false, 3264u32) })
};
}

/// run the code block in @dispatch_ops with rcu/srcu read lock held
macro_rules! __blk_mq_run_dispatch_ops {
($q:expr, $dispatch_ops:expr) => {
loop {
if (unsafe { (*(*$q).tag_set).flags }) & 16 != 0 {
let __tag_set: *mut blk_mq_tag_set = unsafe { (*$q).tag_set };
let mut srcu_idx: i32 = 0;
might_sleep_if__2!();
srcu_idx = srcu_read_lock(unsafe { (*__tag_set).srcu });
$dispatch_ops;
srcu_read_unlock(unsafe { (*__tag_set).srcu }, srcu_idx);
} else {
rcu_read_lock();
$dispatch_ops;
rcu_read_unlock();
}
if true {
break;
}
}
};
}

/// run the code block in @dispatch_ops with rcu/srcu read lock held
macro_rules! __blk_mq_run_dispatch_ops__2 {
($q:expr, $check_sleep:expr, $hctx:expr, $need_run:expr) => {
loop {
if (unsafe { (*(*$q).tag_set).flags }) & 16 != 0 {
let __tag_set: *mut blk_mq_tag_set = unsafe { (*$q).tag_set };
let mut srcu_idx: i32 = 0;
might_sleep_if!($check_sleep);
srcu_idx = srcu_read_lock(unsafe { (*__tag_set).srcu });
let __v: i32 = if true && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((core::ptr::addr_of_mut!((*(*$hctx).queue).queue_flags) as u64) != 0) } && ((unsafe { core::ptr::addr_of_mut!((*(*$hctx).queue).queue_flags) }) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(core::ptr::addr_of_mut!((*(*$hctx).queue).queue_flags) as *const u64)) } {
const_test_bit(9u64, (unsafe { core::ptr::addr_of_mut!((*(*$hctx).queue).queue_flags) }) as *const u64) as i32
} else {
_test_bit(9u64, (unsafe { core::ptr::addr_of_mut!((*(*$hctx).queue).queue_flags) }) as *const u64) as i32
};
let __v2: bool = __v == 0 && blk_mq_hctx_has_pending($hctx);
let __v3: bool = __v2;
$need_run = __v3;
srcu_read_unlock(unsafe { (*__tag_set).srcu }, srcu_idx);
} else {
rcu_read_lock();
let __v4: i32 = if true && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((core::ptr::addr_of_mut!((*(*$hctx).queue).queue_flags) as u64) != 0) } && ((unsafe { core::ptr::addr_of_mut!((*(*$hctx).queue).queue_flags) }) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(core::ptr::addr_of_mut!((*(*$hctx).queue).queue_flags) as *const u64)) } {
const_test_bit(9u64, (unsafe { core::ptr::addr_of_mut!((*(*$hctx).queue).queue_flags) }) as *const u64) as i32
} else {
_test_bit(9u64, (unsafe { core::ptr::addr_of_mut!((*(*$hctx).queue).queue_flags) }) as *const u64) as i32
};
let __v5: bool = __v4 == 0 && blk_mq_hctx_has_pending($hctx);
let __v6: bool = __v5;
$need_run = __v6;
rcu_read_unlock();
}
if true {
break;
}
}
};
}

/// run the code block in @dispatch_ops with rcu/srcu read lock held
macro_rules! __blk_mq_run_dispatch_ops__3 {
($q:expr, $check_sleep:expr, $dispatch_ops:expr) => {
loop {
if (unsafe { (*(*$q).tag_set).flags }) & 16 != 0 {
let __tag_set: *mut blk_mq_tag_set = unsafe { (*$q).tag_set };
let mut srcu_idx: i32 = 0;
might_sleep_if!($check_sleep);
srcu_idx = srcu_read_lock(unsafe { (*__tag_set).srcu });
$dispatch_ops;
srcu_read_unlock(unsafe { (*__tag_set).srcu }, srcu_idx);
} else {
rcu_read_lock();
$dispatch_ops;
rcu_read_unlock();
}
if true {
break;
}
}
};
}

/// run the code block in @dispatch_ops with rcu/srcu read lock held
macro_rules! __blk_mq_run_dispatch_ops__4 {
($q:expr, $rq:expr, $ret:expr) => {
loop {
if (unsafe { (*(*$q).tag_set).flags }) & 16 != 0 {
let __tag_set: *mut blk_mq_tag_set = unsafe { (*$q).tag_set };
let mut srcu_idx: i32 = 0;
might_sleep_if__2!();
srcu_idx = srcu_read_lock(unsafe { (*__tag_set).srcu });
let __v: u8 = blk_mq_request_issue_directly($rq, true);
$ret = __v;
srcu_read_unlock(unsafe { (*__tag_set).srcu }, srcu_idx);
} else {
rcu_read_lock();
let __v2: u8 = blk_mq_request_issue_directly($rq, true);
$ret = __v2;
rcu_read_unlock();
}
if true {
break;
}
}
};
}

macro_rules! __cmp_once {
($type:ty, $x:expr, $y:expr) => {
{
let mut __v: $type;
let __v2: $type = $x;
let __v3: $type = $y;
__v = if __v2 < __v3 { __v2 } else { __v3 };
__v
}
};
}

/// __ffs - find first set bit in word
/// @word: The word to search
///
/// Undefined if no bit exists, so code should check against 0 first.
macro_rules! __ffs {
($word:expr) => {
if unsafe { __slate_intrinsic_f217c4720f04f6db($word) } { ($word.trailing_zeros() as i32) as u64 } else { variable__ffs($word) }
};
}

/// Getting something that works in C and CPP for an arg that may or may
/// not be defined is tricky.  Here, if we have "#define CONFIG_BOOGER 1"
/// we match on the placeholder define, insert the "0," for arg1 and generate
/// the triplet (0, 1, 0).  Then the last step cherry picks the 2nd arg (a one).
/// When CONFIG_BOOGER is not defined, we generate a (... 1, 0) pair, and when
/// the last step cherry picks the 2nd arg, we get a zero.
macro_rules! __is_defined {
() => {
____is_defined!()
};
}

macro_rules! __is_nonneg {
($ux:expr) => {
statically_true!((($ux) as i64) >= 0)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word {
() => {
((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__16 {
() => {
((const { core::mem::size_of::<*mut llist_node>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut llist_node>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut llist_node>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut llist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__5 {
() => {
((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__4 {
() => {
((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__11 {
() => {
((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__3 {
() => {
((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__2 {
() => {
((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__10 {
() => {
((const { core::mem::size_of::<*mut *mut blk_mq_hw_ctx>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut *mut blk_mq_hw_ctx>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut *mut blk_mq_hw_ctx>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut *mut blk_mq_hw_ctx>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

macro_rules! __or {
() => {
____or!()
};
}

macro_rules! __ratelimit__5 {
($state:expr) => {
unsafe { ___ratelimit($state, c"blk_print_req_error".as_ptr() as *const u8) }
};
}

/// __release() - function to release context lock exclusively
/// @x: context lock instance pointer
///
/// No-op function that releases the given context lock instance @x.
macro_rules! __release {
($x:expr) => {
__release_ctx_lock__1(($x) as *const raw_spinlock);
};
}

/// __release_shared() - function to release context lock shared
/// @x: context lock instance pointer
///
/// No-op function that releases the given context lock instance @x with shared
/// access.
macro_rules! __release_shared__2 {
($x:expr) => {
__release_shared_ctx_lock__8(($x) as *const srcu_struct);
};
}

/// __release_shared() - function to release context lock shared
/// @x: context lock instance pointer
///
/// No-op function that releases the given context lock instance @x with shared
/// access.
macro_rules! __release_shared {
($x:expr) => {
__release_shared_ctx_lock__6($x);
};
}

/// This looks more complex than it should be. But we need to
/// get the type for the ~ right in round_down (it needs to be
/// as wide as the result!), and we want to evaluate the macro
/// arguments just once each.
macro_rules! __round_mask__3 {
($y:expr) => {
$y.wrapping_sub(1) as u64
};
}

/// __sign_use for integer expressions:
///   bit #0 set if ok for unsigned comparisons
///   bit #1 set if ok for signed comparisons
///
/// In particular, statically non-negative signed integer expressions
/// are ok for both.
///
/// NOTE! Unsigned types smaller than 'int' are implicitly converted to 'int'
/// in expressions, and are accepted for signed conversions for now.
/// This is debatable.
///
/// Note that 'x' is the original expression, and 'ux' is the unique variable
/// that contains the value.
///
/// We use 'ux' for pure type checking, and 'x' for when we need to look at the
/// value (but without evaluating it for side effects!
/// Careful to only ever evaluate it with sizeof() or __builtin_constant_p() etc).
///
/// Pointers end up being checked by the normal C type rules at the actual
/// comparison, and these expressions only need to be careful to not cause
/// warnings for pointer use.
macro_rules! __sign_use__2 {
($ux:expr) => {
if is_signed_type!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul((((const { core::mem::size_of::<u64>() }) as u64) < 4) as i32)) }
};
}

/// __sign_use for integer expressions:
///   bit #0 set if ok for unsigned comparisons
///   bit #1 set if ok for signed comparisons
///
/// In particular, statically non-negative signed integer expressions
/// are ok for both.
///
/// NOTE! Unsigned types smaller than 'int' are implicitly converted to 'int'
/// in expressions, and are accepted for signed conversions for now.
/// This is debatable.
///
/// Note that 'x' is the original expression, and 'ux' is the unique variable
/// that contains the value.
///
/// We use 'ux' for pure type checking, and 'x' for when we need to look at the
/// value (but without evaluating it for side effects!
/// Careful to only ever evaluate it with sizeof() or __builtin_constant_p() etc).
///
/// Pointers end up being checked by the normal C type rules at the actual
/// comparison, and these expressions only need to be careful to not cause
/// warnings for pointer use.
macro_rules! __sign_use__8 {
($ux:expr) => {
if is_signed_type__2!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul(((core::mem::size_of::<u32>() as u64) < 4) as i32)) }
};
}

/// __sign_use for integer expressions:
///   bit #0 set if ok for unsigned comparisons
///   bit #1 set if ok for signed comparisons
///
/// In particular, statically non-negative signed integer expressions
/// are ok for both.
///
/// NOTE! Unsigned types smaller than 'int' are implicitly converted to 'int'
/// in expressions, and are accepted for signed conversions for now.
/// This is debatable.
///
/// Note that 'x' is the original expression, and 'ux' is the unique variable
/// that contains the value.
///
/// We use 'ux' for pure type checking, and 'x' for when we need to look at the
/// value (but without evaluating it for side effects!
/// Careful to only ever evaluate it with sizeof() or __builtin_constant_p() etc).
///
/// Pointers end up being checked by the normal C type rules at the actual
/// comparison, and these expressions only need to be careful to not cause
/// warnings for pointer use.
macro_rules! __sign_use__3 {
($ux:expr) => {
if is_signed_type__3!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul(((core::mem::size_of::<i32>() as u64) < 4) as i32)) }
};
}

/// __sign_use for integer expressions:
///   bit #0 set if ok for unsigned comparisons
///   bit #1 set if ok for signed comparisons
///
/// In particular, statically non-negative signed integer expressions
/// are ok for both.
///
/// NOTE! Unsigned types smaller than 'int' are implicitly converted to 'int'
/// in expressions, and are accepted for signed conversions for now.
/// This is debatable.
///
/// Note that 'x' is the original expression, and 'ux' is the unique variable
/// that contains the value.
///
/// We use 'ux' for pure type checking, and 'x' for when we need to look at the
/// value (but without evaluating it for side effects!
/// Careful to only ever evaluate it with sizeof() or __builtin_constant_p() etc).
///
/// Pointers end up being checked by the normal C type rules at the actual
/// comparison, and these expressions only need to be careful to not cause
/// warnings for pointer use.
macro_rules! __sign_use {
($ux:expr) => {
if is_signed_type__2!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul((((const { core::mem::size_of::<u32>() }) as u64) < 4) as i32)) }
};
}

macro_rules! __smp_mb__after_atomic {
() => {
while false {
}
};
}

macro_rules! __timer_init {
($_timer:expr, $_fn:expr, $_flags:expr) => {
unsafe { timer_init_key($_timer, $_fn, ($_flags) as u32, core::ptr::null::<u8>(), core::ptr::null_mut::<lock_class_key>()) };
};
}

/// We need to compute the minimum and maximum values representable in a given
/// type. These macros may also be useful elsewhere. It would seem more obvious
/// to do something like:
///
/// #define type_min(T) (T)(is_signed_type(T) ? (T)1 << (8*sizeof(T)-1) : 0)
/// #define type_max(T) (T)(is_signed_type(T) ? ((T)1 << (8*sizeof(T)-1)) - 1 : ~(T)0)
///
/// Unfortunately, the middle expressions, strictly speaking, have
/// undefined behaviour, and at least some versions of gcc warn about
/// the type_max expression (but not if -fsanitize=undefined is in
/// effect; in that case, the warning is deferred to runtime...).
///
/// The slightly excessive casting in type_min is to make sure the
/// macros also produce sensible values for the exotic type _Bool. [The
/// overflow checkers only almost work for _Bool, but that's
/// a-feature-not-a-bug, since people shouldn't be doing arithmetic on
/// _Bools. Besides, the gcc builtins don't allow _Bool* as third
/// argument.]
///
/// Idea stolen from
/// https://mail-index.netbsd.org/tech-misc/2007/02/05/0000.html -
/// credit to Christian Biere.
macro_rules! __type_half_max {
() => {
1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((is_signed_type!() as i32) as u64)
};
}

macro_rules! __type_max {
() => {
__type_half_max!().wrapping_sub(1).wrapping_add(__type_half_max!())
};
}

macro_rules! __types_ok__4 {
($ux:expr, $uy:expr) => {
__sign_use!($ux) & __sign_use__2!($uy)
};
}

macro_rules! __types_ok {
($ux:expr, $uy:expr) => {
__sign_use!($ux) & __sign_use!($uy)
};
}

macro_rules! __types_ok__6 {
($ux:expr, $uy:expr) => {
__sign_use__8!($ux) & __sign_use__3!($uy)
};
}

macro_rules! __va {
($x:expr) => {
(($x) as u64).wrapping_add(unsafe { page_offset_base }) as *mut ()
};
}

/// __verify_pcpu_ptr() verifies @ptr is a percpu pointer without evaluating
/// @ptr and is invoked once before a percpu area is accessed by all
/// accessors and operations.  This is performed in the generic part of
/// percpu and arch overrides don't need to worry about it; however, if an
/// arch wants to implement an arch-specific percpu accessor or operation,
/// it may use __verify_pcpu_ptr() to verify the parameters.
///
/// + 0 is required in order to convert the pointer type from a
/// potential array type to a pointer to a single item of the array.
macro_rules! __verify_pcpu_ptr__2 {
() => {
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<u64>() as *const ();
if true {
break;
}
}
};
}

/// __verify_pcpu_ptr() verifies @ptr is a percpu pointer without evaluating
/// @ptr and is invoked once before a percpu area is accessed by all
/// accessors and operations.  This is performed in the generic part of
/// percpu and arch overrides don't need to worry about it; however, if an
/// arch wants to implement an arch-specific percpu accessor or operation,
/// it may use __verify_pcpu_ptr() to verify the parameters.
///
/// + 0 is required in order to convert the pointer type from a
/// potential array type to a pointer to a single item of the array.
macro_rules! __verify_pcpu_ptr__8 {
() => {
loop {
let __v: *const () = core::ptr::null_mut::<blk_mq_ctx>() as *const ();
if true {
break;
}
}
};
}

/// __verify_pcpu_ptr() verifies @ptr is a percpu pointer without evaluating
/// @ptr and is invoked once before a percpu area is accessed by all
/// accessors and operations.  This is performed in the generic part of
/// percpu and arch overrides don't need to worry about it; however, if an
/// arch wants to implement an arch-specific percpu accessor or operation,
/// it may use __verify_pcpu_ptr() to verify the parameters.
///
/// + 0 is required in order to convert the pointer type from a
/// potential array type to a pointer to a single item of the array.
macro_rules! __verify_pcpu_ptr__5 {
() => {
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<__SlateRecord_types_h_197>() as *const ();
if true {
break;
}
}
};
}

/// __verify_pcpu_ptr() verifies @ptr is a percpu pointer without evaluating
/// @ptr and is invoked once before a percpu area is accessed by all
/// accessors and operations.  This is performed in the generic part of
/// percpu and arch overrides don't need to worry about it; however, if an
/// arch wants to implement an arch-specific percpu accessor or operation,
/// it may use __verify_pcpu_ptr() to verify the parameters.
///
/// + 0 is required in order to convert the pointer type from a
/// potential array type to a pointer to a single item of the array.
macro_rules! __verify_pcpu_ptr__7 {
() => {
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<disk_stats>() as *const ();
if true {
break;
}
}
};
}

/// __verify_pcpu_ptr() verifies @ptr is a percpu pointer without evaluating
/// @ptr and is invoked once before a percpu area is accessed by all
/// accessors and operations.  This is performed in the generic part of
/// percpu and arch overrides don't need to worry about it; however, if an
/// arch wants to implement an arch-specific percpu accessor or operation,
/// it may use __verify_pcpu_ptr() to verify the parameters.
///
/// + 0 is required in order to convert the pointer type from a
/// potential array type to a pointer to a single item of the array.
macro_rules! __verify_pcpu_ptr__11 {
() => {
loop {
let __v: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
};
}

/// __verify_pcpu_ptr() verifies @ptr is a percpu pointer without evaluating
/// @ptr and is invoked once before a percpu area is accessed by all
/// accessors and operations.  This is performed in the generic part of
/// percpu and arch overrides don't need to worry about it; however, if an
/// arch wants to implement an arch-specific percpu accessor or operation,
/// it may use __verify_pcpu_ptr() to verify the parameters.
///
/// + 0 is required in order to convert the pointer type from a
/// potential array type to a pointer to a single item of the array.
macro_rules! __verify_pcpu_ptr__9 {
() => {
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<llist_head>() as *const ();
if true {
break;
}
}
};
}

/// __verify_pcpu_ptr() verifies @ptr is a percpu pointer without evaluating
/// @ptr and is invoked once before a percpu area is accessed by all
/// accessors and operations.  This is performed in the generic part of
/// percpu and arch overrides don't need to worry about it; however, if an
/// arch wants to implement an arch-specific percpu accessor or operation,
/// it may use __verify_pcpu_ptr() to verify the parameters.
///
/// + 0 is required in order to convert the pointer type from a
/// potential array type to a pointer to a single item of the array.
macro_rules! __verify_pcpu_ptr__20 {
() => {
loop {
let __v: *const () = core::ptr::null_mut::<__call_single_data>() as *const ();
if true {
break;
}
}
};
}

/// __verify_pcpu_ptr() verifies @ptr is a percpu pointer without evaluating
/// @ptr and is invoked once before a percpu area is accessed by all
/// accessors and operations.  This is performed in the generic part of
/// percpu and arch overrides don't need to worry about it; however, if an
/// arch wants to implement an arch-specific percpu accessor or operation,
/// it may use __verify_pcpu_ptr() to verify the parameters.
///
/// + 0 is required in order to convert the pointer type from a
/// potential array type to a pointer to a single item of the array.
macro_rules! __verify_pcpu_ptr__6 {
() => {
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i64>() as *const ();
if true {
break;
}
}
};
}

/// __verify_pcpu_ptr() verifies @ptr is a percpu pointer without evaluating
/// @ptr and is invoked once before a percpu area is accessed by all
/// accessors and operations.  This is performed in the generic part of
/// percpu and arch overrides don't need to worry about it; however, if an
/// arch wants to implement an arch-specific percpu accessor or operation,
/// it may use __verify_pcpu_ptr() to verify the parameters.
///
/// + 0 is required in order to convert the pointer type from a
/// potential array type to a pointer to a single item of the array.
macro_rules! __verify_pcpu_ptr {
() => {
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<*mut task_struct>() as *const ();
if true {
break;
}
}
};
}

/// __verify_pcpu_ptr() verifies @ptr is a percpu pointer without evaluating
/// @ptr and is invoked once before a percpu area is accessed by all
/// accessors and operations.  This is performed in the generic part of
/// percpu and arch overrides don't need to worry about it; however, if an
/// arch wants to implement an arch-specific percpu accessor or operation,
/// it may use __verify_pcpu_ptr() to verify the parameters.
///
/// + 0 is required in order to convert the pointer type from a
/// potential array type to a pointer to a single item of the array.
macro_rules! __verify_pcpu_ptr__3 {
() => {
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
};
}

/// __verify_pcpu_ptr() verifies @ptr is a percpu pointer without evaluating
/// @ptr and is invoked once before a percpu area is accessed by all
/// accessors and operations.  This is performed in the generic part of
/// percpu and arch overrides don't need to worry about it; however, if an
/// arch wants to implement an arch-specific percpu accessor or operation,
/// it may use __verify_pcpu_ptr() to verify the parameters.
///
/// + 0 is required in order to convert the pointer type from a
/// potential array type to a pointer to a single item of the array.
macro_rules! __verify_pcpu_ptr__14 {
() => {
loop {
let __v: *const () = core::ptr::null_mut::<llist_head>() as *const ();
if true {
break;
}
}
};
}

macro_rules! absolute_pointer {
($val:expr) => {
RELOC_HIDE!(($val) as *mut (), 0i32)
};
}

macro_rules! alloc_hooks__3 {
($_do_alloc:expr) => {
{
let mut __v: *mut page;
let mut __v2: *mut page;
let mut __v3: *mut page = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
__v3 = $_do_alloc;
alloc_tag_restore!();
} else {
__v3 = $_do_alloc;
}
__v2 = __v3;
__v = __v2;
__v
}
};
}

macro_rules! alloc_hooks {
($_do_alloc:expr) => {
{
let mut __v: *mut ();
let mut __v2: *mut ();
let mut __v3: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
__v3 = $_do_alloc;
alloc_tag_restore!();
} else {
__v3 = $_do_alloc;
}
__v2 = __v3;
__v = __v2;
__v
}
};
}

macro_rules! alloc_percpu {
($type:ty) => {
__alloc_percpu!(core::mem::size_of::<$type>() as u64, core::mem::align_of::<$type>() as u64) as *mut $type
};
}

macro_rules! alloc_tag_restore {
() => {
while false {
}
};
}

macro_rules! bio_iovec {
($bio:expr) => {
bio_iter_iovec!($bio, unsafe { (*$bio).bi_iter })
};
}

macro_rules! bio_iter_iovec {
($bio:expr, $iter:expr) => {
bvec_iter_bvec((unsafe { (*$bio).bi_io_vec }) as *const bio_vec, $iter)
};
}

macro_rules! bitmap_size {
($nbits:expr) => {
ALIGN__2!($nbits, 64i32) / 8
};
}

macro_rules! blk_mq_run_dispatch_ops {
($q:expr, $dispatch_ops:expr) => {
__blk_mq_run_dispatch_ops!($q, $dispatch_ops);
};
}

macro_rules! blk_mq_run_dispatch_ops__2 {
($q:expr, $dispatch_ops:expr) => {
__blk_mq_run_dispatch_ops__3!($q, true, $dispatch_ops);
};
}

macro_rules! blk_mq_run_dispatch_ops__3 {
($q:expr, $rq:expr, $ret:expr) => {
__blk_mq_run_dispatch_ops__4!($q, $rq, $ret);
};
}

macro_rules! blk_queue_init_done {
($q:expr) => {
test_bit!(6i32, unsafe { core::ptr::addr_of_mut!((*$q).queue_flags) })
};
}

macro_rules! blk_queue_io_stat {
($q:expr) => {
(unsafe { (*$q).limits.features }) & 16u32
};
}

macro_rules! blk_queue_nomerges {
($q:expr) => {
test_bit!(1i32, unsafe { core::ptr::addr_of_mut!((*$q).queue_flags) })
};
}

macro_rules! blk_queue_passthrough_stat {
($q:expr) => {
(unsafe { (*$q).limits.flags }) & 4u32
};
}

macro_rules! blk_queue_pm_only {
($q:expr) => {
atomic_read((unsafe { core::ptr::addr_of_mut!((*$q).pm_only) }) as *const __SlateRecord_types_h_190)
};
}

macro_rules! blk_queue_quiesced {
($q:expr) => {
test_bit!(9i32, unsafe { core::ptr::addr_of_mut!((*$q).queue_flags) })
};
}

macro_rules! blk_queue_registered {
($q:expr) => {
test_bit!(8i32, unsafe { core::ptr::addr_of_mut!((*$q).queue_flags) })
};
}

macro_rules! blk_queue_rq_alloc_time {
($q:expr) => {
test_bit!(10i32, unsafe { core::ptr::addr_of_mut!((*$q).queue_flags) })
};
}

macro_rules! blk_queue_skip_tagset_quiesce {
($q:expr) => {
(unsafe { (*$q).limits.features }) & 8192u32
};
}

macro_rules! blk_queue_sq_sched {
($q:expr) => {
test_bit!(12i32, unsafe { core::ptr::addr_of_mut!((*$q).queue_flags) })
};
}

macro_rules! cache_line_size {
() => {
unsafe { boot_cpu_data.x86_cache_alignment }
};
}

/// check_mul_overflow() - Calculate multiplication with overflow checking
/// @a: first factor
/// @b: second factor
/// @d: pointer to store product
///
/// Returns true on wrap-around, false otherwise.
///
/// *@d holds the results of the attempted multiplication, regardless of whether
/// wrap-around occurred.
macro_rules! check_mul_overflow {
($a:expr, $b:expr, $d:expr) => {
__must_check_overflow({
let __t: i128 = ($a) as i128;
let __t2: i128 = ($b) as i128;
let __t3: i128 = __t.wrapping_mul(__t2);
unsafe {
*$d = __t3 as u64;
}
__t.checked_mul(__t2).is_none() || ((__t3 as u64) as i128) != __t3
})
};
}

macro_rules! cond_resched__4 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/block/blk-mq.c".as_ptr() as *const u8, 1476i32, 0u32);
__v = _cond_resched();
};
}

macro_rules! cond_resched__8 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/block/blk-mq.c".as_ptr() as *const u8, 4826i32, 0u32);
__v = _cond_resched();
};
}

macro_rules! cond_resched__9 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/block/blk-mq.c".as_ptr() as *const u8, 4709i32, 0u32);
__v = _cond_resched();
};
}

/// container_of - cast a member of a structure out to the containing structure
/// @ptr:	the pointer to the member.
/// @type:	the type of the container struct this is embedded in.
/// @member:	the name of the member within the struct.
///
/// WARNING: any const qualifier of @ptr is lost.
/// Do not use container_of() in new code.
macro_rules! container_of__18 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__54!($type) as isize)) }) as *mut ()) as *mut $type;
__v
}
};
}

/// container_of - cast a member of a structure out to the containing structure
/// @ptr:	the pointer to the member.
/// @type:	the type of the container struct this is embedded in.
/// @member:	the name of the member within the struct.
///
/// WARNING: any const qualifier of @ptr is lost.
/// Do not use container_of() in new code.
macro_rules! container_of__11 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__13!($type) as isize)) }) as *mut ()) as *mut $type;
__v
}
};
}

/// container_of - cast a member of a structure out to the containing structure
/// @ptr:	the pointer to the member.
/// @type:	the type of the container struct this is embedded in.
/// @member:	the name of the member within the struct.
///
/// WARNING: any const qualifier of @ptr is lost.
/// Do not use container_of() in new code.
macro_rules! container_of__12 {
($ptr:expr) => {
{
let mut __v: *mut request_queue;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__4!() as isize)) }) as *mut ()) as *mut request_queue;
__v
}
};
}

/// container_of - cast a member of a structure out to the containing structure
/// @ptr:	the pointer to the member.
/// @type:	the type of the container struct this is embedded in.
/// @member:	the name of the member within the struct.
///
/// WARNING: any const qualifier of @ptr is lost.
/// Do not use container_of() in new code.
macro_rules! container_of__25 {
($ptr:expr) => {
{
let mut __v: *mut wait_queue_entry;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__29!() as isize)) }) as *mut ()) as *mut wait_queue_entry;
__v
}
};
}

/// container_of - cast a member of a structure out to the containing structure
/// @ptr:	the pointer to the member.
/// @type:	the type of the container struct this is embedded in.
/// @member:	the name of the member within the struct.
///
/// WARNING: any const qualifier of @ptr is lost.
/// Do not use container_of() in new code.
macro_rules! container_of__33 {
($ptr:expr) => {
{
let mut __v: *mut blk_mq_hw_ctx;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__36!() as isize)) }) as *mut ()) as *mut blk_mq_hw_ctx;
__v
}
};
}

/// container_of - cast a member of a structure out to the containing structure
/// @ptr:	the pointer to the member.
/// @type:	the type of the container struct this is embedded in.
/// @member:	the name of the member within the struct.
///
/// WARNING: any const qualifier of @ptr is lost.
/// Do not use container_of() in new code.
macro_rules! container_of__4 {
($ptr:expr) => {
{
let mut __v: *mut request;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof!() as isize)) }) as *mut ()) as *mut request;
__v
}
};
}

/// container_of - cast a member of a structure out to the containing structure
/// @ptr:	the pointer to the member.
/// @type:	the type of the container struct this is embedded in.
/// @member:	the name of the member within the struct.
///
/// WARNING: any const qualifier of @ptr is lost.
/// Do not use container_of() in new code.
macro_rules! container_of__36 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__57!($type) as isize)) }) as *mut ()) as *mut $type;
__v
}
};
}

/// container_of - cast a member of a structure out to the containing structure
/// @ptr:	the pointer to the member.
/// @type:	the type of the container struct this is embedded in.
/// @member:	the name of the member within the struct.
///
/// WARNING: any const qualifier of @ptr is lost.
/// Do not use container_of() in new code.
macro_rules! container_of__2 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof!() as isize)) }) as *mut ()) as *mut $type;
__v
}
};
}

/// container_of - cast a member of a structure out to the containing structure
/// @ptr:	the pointer to the member.
/// @type:	the type of the container struct this is embedded in.
/// @member:	the name of the member within the struct.
///
/// WARNING: any const qualifier of @ptr is lost.
/// Do not use container_of() in new code.
macro_rules! container_of__45 {
($ptr:expr) => {
{
let mut __v: *mut page;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__16!() as isize)) }) as *mut ()) as *mut page;
__v
}
};
}

/// container_of - cast a member of a structure out to the containing structure
/// @ptr:	the pointer to the member.
/// @type:	the type of the container struct this is embedded in.
/// @member:	the name of the member within the struct.
///
/// WARNING: any const qualifier of @ptr is lost.
/// Do not use container_of() in new code.
macro_rules! container_of__3 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__3!($type) as isize)) }) as *mut ()) as *mut $type;
__v
}
};
}

/// container_of - cast a member of a structure out to the containing structure
/// @ptr:	the pointer to the member.
/// @type:	the type of the container struct this is embedded in.
/// @member:	the name of the member within the struct.
///
/// WARNING: any const qualifier of @ptr is lost.
/// Do not use container_of() in new code.
macro_rules! container_of__60 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__55!($type) as isize)) }) as *mut ()) as *mut $type;
__v
}
};
}

/// container_of - cast a member of a structure out to the containing structure
/// @ptr:	the pointer to the member.
/// @type:	the type of the container struct this is embedded in.
/// @member:	the name of the member within the struct.
///
/// WARNING: any const qualifier of @ptr is lost.
/// Do not use container_of() in new code.
macro_rules! container_of__17 {
($ptr:expr) => {
{
let mut __v: *mut request;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__11!() as isize)) }) as *mut ()) as *mut request;
__v
}
};
}

/// cpumask_bits - get the bits in a cpumask
/// @maskp: the struct cpumask *
///
/// You should only assume nr_cpu_ids bits of this mask are valid.  This is
/// a macro so it's const-correct.
macro_rules! cpumask_bits__2 {
($maskp:expr) => {
unsafe { (*$maskp).bits.as_mut_ptr() as *mut u64 }
};
}

/// cpumask_bits - get the bits in a cpumask
/// @maskp: the struct cpumask *
///
/// You should only assume nr_cpu_ids bits of this mask are valid.  This is
/// a macro so it's const-correct.
macro_rules! cpumask_bits {
($maskp:expr) => {
unsafe { (*$maskp).bits.as_ptr() as *const u64 }
};
}

macro_rules! current_thread_info {
() => {
get_current() as *mut thread_info
};
}

/// delayed_work_pending - Find out whether a delayable work item is currently
/// pending
/// @w: The work item in question
macro_rules! delayed_work_pending {
($w:expr) => {
work_pending!(unsafe { core::ptr::addr_of_mut!((*$w).work) })
};
}

/// disable_context_analysis() - disables context analysis
///
/// Disables context analysis. Must be paired with a later
/// enable_context_analysis().
macro_rules! disable_context_analysis {
() => {
};
}

macro_rules! force_irqthreads {
() => {
static_branch_unlikely__11!(unsafe { core::ptr::addr_of_mut!(force_irqthreads_key) })
};
}

macro_rules! hlist_entry__5 {
($ptr:expr, $type:ty) => {
container_of__60!($ptr, $type)
};
}

macro_rules! hlist_entry__3 {
($ptr:expr, $type:ty) => {
container_of__11!($ptr, $type)
};
}

macro_rules! hlist_entry_safe__3 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
let __v2: *mut hlist_node = $ptr;
let __v3: *mut $type = if !__v2.is_null() {
hlist_entry__3!(__v2, $type)
} else {
core::ptr::null_mut::<blk_mq_hw_ctx>()
};
__v = __v3;
__v
}
};
}

macro_rules! hlist_entry_safe__2 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
let __v2: *mut hlist_node = $ptr;
let __v3: *mut $type = if !__v2.is_null() {
hlist_entry__5!(__v2, $type)
} else {
core::ptr::null_mut::<blk_mq_hw_ctx>()
};
__v = __v3;
__v
}
};
}

/// ilog2 - log base 2 of 32-bit or a 64-bit unsigned value
/// @n: parameter
///
/// constant-capable log of base 2 calculation
/// - this can be used to initialise global variables from constant data, hence
/// the massive ternary operator construction
///
/// selects the appropriately-sized optimised version depending on sizeof(n)
macro_rules! ilog2__2 {
($n:expr) => {
if true { if ($n) < 2 { 0i32 } else { 63i32.wrapping_sub((($n) as u64).leading_zeros() as i32) } } else { if (core::mem::size_of::<i32>() as u64) <= 4 { __ilog2_u32(($n) as u32) } else { __ilog2_u64(($n) as u64) } }
};
}

macro_rules! in_interrupt {
() => {
irq_count!()
};
}

macro_rules! in_task {
() => {
(preempt_count() as u64) & (__IRQ_MASK!(1i32.wrapping_add(3i32.wrapping_mul(IS_ENABLED__2!()))) << 28i32 | __IRQ_MASK!(4i32) << 24i32 | 256u64) == 0
};
}

macro_rules! irq_count {
() => {
(preempt_count() as u64) & (__IRQ_MASK!(1i32.wrapping_add(3i32.wrapping_mul(IS_ENABLED__2!()))) << 28i32 | __IRQ_MASK!(4i32) << 24i32 | __IRQ_MASK!(8i32) << 8i32)
};
}

macro_rules! irqs_disabled {
() => {
{
let mut __v: i32;
let mut __v2: u64 = 0;
raw_local_save_flags!(__v2);
let mut __v3: i32;
typecheck__3!(u64);
__v3 = arch_irqs_disabled_flags(__v2);
__v = __v3;
__v
}
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type__2 {
() => {
(1i32.wrapping_neg() as u32) < 1
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type__3 {
() => {
1i32.wrapping_neg() < 1
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type {
() => {
(1i32.wrapping_neg() as u64) < 1
};
}

macro_rules! kcsan_check_atomic_read {
($ptr:expr, $size:expr) => {
kcsan_check_access($ptr, $size, 4i32);
};
}

macro_rules! kcsan_check_atomic_read_write {
($ptr:expr, $size:expr) => {
kcsan_check_access($ptr, $size, 7i32);
};
}

macro_rules! kcsan_check_atomic_write {
($ptr:expr, $size:expr) => {
kcsan_check_access(absolute_pointer!($ptr) as *const (), $size, 5i32);
};
}

/// kcsan_check_read_write - check regular read-write access for races
///
/// @ptr: address of access
/// @size: size of access
macro_rules! kcsan_check_read_write {
($ptr:expr, $size:expr) => {
kcsan_check_access($ptr, $size, 3i32);
};
}

macro_rules! kcsan_mb {
() => {
while false {
}
};
}

macro_rules! kcsan_release {
() => {
while false {
}
};
}

/// kfree_rcu_mightsleep() - kfree an object after a grace period.
/// @ptr: pointer to kfree for single-argument invocations.
///
/// When it comes to head-less variant, only one argument
/// is passed and that is just a pointer which has to be
/// freed after a grace period. Therefore the semantic is
///
///     kfree_rcu_mightsleep(ptr);
///
/// where @ptr is the pointer to be freed by kvfree().
///
/// Please note, head-less way of freeing is permitted to
/// use from a context that has to follow might_sleep()
/// annotation. Otherwise, please switch and embed the
/// rcu_head structure within the type of @ptr.
macro_rules! kfree_rcu_mightsleep {
($ptr:expr) => {
kvfree_rcu_arg_1!($ptr);
};
}

macro_rules! kmalloc_index {
($s:expr) => {
__kmalloc_index($s, true)
};
}

/// Subtract two ktime_t variables. rem = lhs -rhs:
macro_rules! ktime_sub {
($lhs:expr, $rhs:expr) => {
$lhs.wrapping_sub($rhs)
};
}

macro_rules! kvfree_rcu_arg_1 {
($ptr:expr) => {
loop {
let ___p: *mut *mut blk_mq_hw_ctx = $ptr;
if !___p.is_null() {
unsafe { kvfree_call_rcu(core::ptr::null_mut::<kvfree_rcu_head>(), ___p as *mut ()) };
}
if true {
break;
}
}
};
}

macro_rules! likely {
($x:expr) => {
!(!$x) as i64
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__7 {
($ptr:expr) => {
container_of__25!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__3 {
($ptr:expr) => {
container_of__12!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry {
($ptr:expr) => {
container_of__4!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__10 {
($ptr:expr) => {
container_of__33!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__15 {
($ptr:expr) => {
container_of__45!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__2 {
($ptr:expr, $type:ty) => {
container_of__2!($ptr, $type)
};
}

/// list_entry_is_head - test if the entry points to the head of the list
/// @pos:	the type * to cursor
/// @head:	the head for your list.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry_is_head {
($pos:expr, $head:expr, $member:ident) => {
list_is_head((unsafe { core::ptr::addr_of_mut!((*$pos).$member) }) as *const list_head, ($head) as *const list_head)
};
}

/// list_entry_is_head - test if the entry points to the head of the list
/// @pos:	the type * to cursor
/// @head:	the head for your list.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry_is_head__2 {
($pos:expr, $head:expr, $member:ident) => {
list_is_head((unsafe { core::ptr::addr_of_mut!((*$pos).__slate_anon_12.$member) }) as *const list_head, ($head) as *const list_head)
};
}

/// list_entry_is_head - test if the entry points to the head of the list
/// @pos:	the type * to cursor
/// @head:	the head for your list.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry_is_head__3 {
($pos:expr, $head:expr, $member:ident) => {
list_is_head((unsafe { core::ptr::addr_of_mut!((*$pos).__slate_anon_1.__slate_anon_0.__slate_anon_0.$member) }) as *const list_head, ($head) as *const list_head)
};
}

macro_rules! list_entry_rq {
($ptr:expr) => {
{
let mut __v: *mut request;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof!() as isize)) }) as *mut ()) as *mut request;
__v
}
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__7 {
($ptr:expr) => {
list_entry__10!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__3 {
($ptr:expr) => {
list_entry!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry {
($ptr:expr) => {
list_entry__3!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__2 {
($ptr:expr, $type:ty) => {
list_entry__2!(unsafe { (*$ptr).next }, $type)
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__14 {
($ptr:expr) => {
list_entry__15!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__16 {
($ptr:expr) => {
list_entry__7!(unsafe { (*$ptr).next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry {
($pos:expr, $member:ident) => {
list_entry__3!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__6 {
($pos:expr, $member:ident) => {
list_entry__10!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__3 {
($pos:expr, $member:ident) => {
list_entry!(unsafe { (*$pos).__slate_anon_12.$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__12 {
($pos:expr, $member:ident) => {
list_entry__7!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__14 {
($pos:expr, $member:ident) => {
list_entry__15!(unsafe { (*$pos).__slate_anon_1.__slate_anon_0.__slate_anon_0.$member.next })
};
}

/// llist_entry - get the struct of this entry
/// @ptr:	the &struct llist_node pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the llist_node within the struct.
macro_rules! llist_entry {
($ptr:expr) => {
container_of__17!($ptr)
};
}

macro_rules! lock_acquire {
() => {
while false {
}
};
}

macro_rules! lock_acquire_shared {
() => {
lock_acquire!();
};
}

macro_rules! lock_release {
() => {
while false {
}
};
}

macro_rules! lockdep_assert_held__2 {
($l:expr) => {
__assume_ctx_lock__4(($l) as *const mutex);
};
}

macro_rules! lockdep_assert_held_write {
($l:expr) => {
__assume_ctx_lock__7(($l) as *const rw_semaphore);
};
}

/// member_address_is_nonnull - check whether the member address is not NULL
/// @ptr:	the object pointer (struct type * that contains the llist_node)
/// @member:	the name of the llist_node within the struct.
///
/// This macro is conceptually the same as
/// 	&ptr->member != NULL
/// but it works around the fact that compilers can decide that taking a member
/// address is never a NULL pointer.
///
/// Real objects that start at a high address and have a member at NULL are
/// unlikely to exist, but such pointers may be returned e.g. by the
/// container_of() macro.
macro_rules! member_address_is_nonnull__2 {
($ptr:expr) => {
(($ptr) as u64).wrapping_add(offsetof__11!()) != 0
};
}

macro_rules! might_sleep {
() => {
loop {
might_resched();
if true {
break;
}
}
};
}

macro_rules! might_sleep_if__2 {
() => {
loop {
might_sleep!();
if true {
break;
}
}
};
}

macro_rules! might_sleep_if {
($cond:expr) => {
loop {
if $cond {
might_sleep!();
}
if true {
break;
}
}
};
}

/// min_t - return minimum of two values, using the specified type
/// @type: data type to use
/// @x: first value
/// @y: second value
macro_rules! min_t {
($type:ty, $x:expr, $y:expr) => {
__cmp_once!($type, $x, $y)
};
}

macro_rules! offsetof__29 {
() => {
core::mem::offset_of!(wait_queue_entry, entry) as u64
};
}

macro_rules! offsetof__31 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, requeue_work.work) as u64
};
}

macro_rules! offsetof__13 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, cpuhp_dead) as u64
};
}

macro_rules! offsetof__36 {
() => {
core::mem::offset_of!(blk_mq_hw_ctx, hctx_list) as u64
};
}

macro_rules! offsetof__3 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, callback_head) as u64
};
}

macro_rules! offsetof__45 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, run_work.work) as u64
};
}

macro_rules! offsetof__16 {
() => {
8u64
};
}

macro_rules! offsetof {
() => {
72u64
};
}

macro_rules! offsetof__11 {
() => {
144u64
};
}

macro_rules! offsetof__54 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, dispatch_wait) as u64
};
}

macro_rules! offsetof__55 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, cpuhp_online) as u64
};
}

macro_rules! offsetof__57 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, timeout_work) as u64
};
}

macro_rules! offsetof__4 {
() => {
core::mem::offset_of!(request_queue, tag_set_list) as u64
};
}

macro_rules! page_address {
($page:expr) => {
lowmem_page_address(($page) as *const page)
};
}

macro_rules! page_to_virt {
($x:expr) => {
__va!(PFN_PHYS!(((unsafe { $x.offset_from((vmemmap_base as *mut page) as *const page) }) as i64) as u64))
};
}

macro_rules! per_cpu {
($var:expr, $cpu:expr) => {
unsafe { *per_cpu_ptr__11!(core::ptr::addr_of_mut!($var), $cpu) }
};
}

macro_rules! per_cpu_offset {
($x:expr) => {
unsafe { __per_cpu_offset[($x) as usize] }
};
}

macro_rules! per_cpu_ptr {
($ptr:expr, $cpu:expr) => {
{
let mut __v: *mut blk_mq_ctx;
__verify_pcpu_ptr__8!();
__v = (((($ptr) as u64) as *mut blk_mq_ctx) as u64).wrapping_add(per_cpu_offset!($cpu)) as *mut blk_mq_ctx;
__v
}
};
}

macro_rules! per_cpu_ptr__10 {
($ptr:expr, $cpu:expr) => {
{
let mut __v: *mut __call_single_data;
__verify_pcpu_ptr__20!();
__v = (((($ptr) as u64) as *mut __call_single_data) as u64).wrapping_add(per_cpu_offset!($cpu)) as *mut __call_single_data;
__v
}
};
}

macro_rules! per_cpu_ptr__2 {
($ptr:expr, $cpu:expr) => {
{
let mut __v: *mut llist_head;
__verify_pcpu_ptr__14!();
__v = (((($ptr) as u64) as *mut llist_head) as u64).wrapping_add(per_cpu_offset!($cpu)) as *mut llist_head;
__v
}
};
}

macro_rules! per_cpu_ptr__11 {
($ptr:expr, $cpu:expr) => {
{
let mut __v: *mut i32;
__verify_pcpu_ptr__11!();
__v = (((($ptr) as u64) as *mut i32) as u64).wrapping_add(per_cpu_offset!($cpu)) as *mut i32;
__v
}
};
}

macro_rules! preempt_count_add {
($val:expr) => {
__preempt_count_add($val);
};
}

macro_rules! preempt_count_dec_and_test {
() => {
__preempt_count_dec_and_test()
};
}

macro_rules! preempt_count_inc {
() => {
preempt_count_add!(1i32);
};
}

macro_rules! prefetch {
($x:expr) => {
{
let _ = ($x) as *const ();
};
};
}

macro_rules! raw_local_save_flags {
($flags:expr) => {
loop {
typecheck__3!(u64);
$flags = arch_local_save_flags();
if true {
break;
}
}
};
}

macro_rules! raw_spin_lock {
($lock:expr) => {
unsafe { _raw_spin_lock($lock) };
};
}

macro_rules! raw_spin_lock_irq {
($lock:expr) => {
unsafe { _raw_spin_lock_irq($lock) };
};
}

macro_rules! raw_spin_lock_irqsave {
($lock:expr, $flags:expr) => {
loop {
typecheck__2!(u64);
$flags = unsafe { _raw_spin_lock_irqsave($lock) };
if true {
break;
}
}
};
}

macro_rules! raw_spin_unlock {
($lock:expr) => {
unsafe { _raw_spin_unlock($lock) };
};
}

macro_rules! raw_spin_unlock_irq {
($lock:expr) => {
unsafe { _raw_spin_unlock_irq($lock) };
};
}

macro_rules! raw_spin_unlock_irqrestore {
($lock:expr, $flags:expr) => {
loop {
typecheck__2!(u64);
unsafe { _raw_spin_unlock_irqrestore($lock, $flags) };
if true {
break;
}
}
};
}

macro_rules! rcu_lock_acquire {
() => {
while false {
}
};
}

macro_rules! rcu_lock_release {
() => {
while false {
}
};
}

/// Optimized request reference counting. Ideally we'd make timeouts be more
/// clever, as that's the only reason we need references at all... But until
/// this happens, this is faster than using refcount_t. Also see:
///
/// abc54d634334 ("io_uring: switch to atomic_t for io_kiocb reference count")
macro_rules! req_ref_zero_or_close_to_overflow {
($req:expr) => {
(atomic_read((unsafe { core::ptr::addr_of_mut!((*$req).r#ref) }) as *const __SlateRecord_types_h_190) as u32).wrapping_add(127) <= 127
};
}

/// round_up - round up to next specified power of 2
/// @x: the value to round
/// @y: multiple to round up to (must be a power of 2)
///
/// Rounds @x up to next multiple of @y (which must be a power of 2).
/// To perform arbitrary rounding up, use roundup() below.
macro_rules! round_up__3 {
($x:expr, $y:expr) => {
($x.wrapping_sub(1) | __round_mask__3!($y)).wrapping_add(1)
};
}

macro_rules! rq_data_dir {
($rq:expr) => {
if op_is_write(req_op(($rq) as *const request) as u32) { 1i32 } else { 0i32 }
};
}

macro_rules! rwsem_acquire_read {
() => {
lock_acquire_shared!();
};
}

macro_rules! rwsem_release {
() => {
lock_release!();
};
}

/// small_const_nbits(n) is true precisely when it is known at compile-time
/// that BITMAP_SIZE(n) is 1, i.e. 1 <= n <= BITS_PER_LONG. This allows
/// various bit/bitmap APIs to provide a fast inline implementation. Bitmaps
/// of size 0 are very rare, and a compile-time-known-size 0 is most likely
/// a sign of error. They will be handled correctly by the bit/bitmap APIs,
/// but using the out-of-line functions, so that the inline implementations
/// can unconditionally dereference the pointer(s).
macro_rules! small_const_nbits__2 {
($nbits:expr) => {
(unsafe { __slate_intrinsic_1c5364fcd9da51fe($nbits) }) && ($nbits) <= 64 && ($nbits) > 0
};
}

/// small_const_nbits(n) is true precisely when it is known at compile-time
/// that BITMAP_SIZE(n) is 1, i.e. 1 <= n <= BITS_PER_LONG. This allows
/// various bit/bitmap APIs to provide a fast inline implementation. Bitmaps
/// of size 0 are very rare, and a compile-time-known-size 0 is most likely
/// a sign of error. They will be handled correctly by the bit/bitmap APIs,
/// but using the out-of-line functions, so that the inline implementations
/// can unconditionally dereference the pointer(s).
macro_rules! small_const_nbits {
($nbits:expr) => {
(unsafe { __slate_intrinsic_f217c4720f04f6db($nbits) }) && ($nbits) <= 64 && ($nbits) > 0
};
}

macro_rules! smp_mb__after_atomic {
() => {
loop {
kcsan_mb!();
__smp_mb__after_atomic!();
if true {
break;
}
}
};
}

macro_rules! spin_lock_irqsave {
($lock:expr, $flags:expr) => {
loop {
raw_spin_lock_irqsave!(spinlock_check($lock), $flags);
__release!(spinlock_check($lock));
__acquire!($lock);
if true {
break;
}
}
};
}

macro_rules! srcu_lock_acquire {
() => {
while false {
}
};
}

macro_rules! srcu_lock_release {
() => {
while false {
}
};
}

macro_rules! static_branch_unlikely__5 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const9(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__6 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const10(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const1(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__4 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const4(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__3 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const3(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__7 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const8(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__2 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const2(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__8 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const6(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__9 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const5(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__10 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const7(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__11 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const11(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

/// Useful shorthand for "is this condition known at compile-time?"
///
/// Note that the condition may involve non-constant values,
/// but the compiler may know enough about the details of the
/// values to determine that the condition is statically true.
macro_rules! statically_true {
($x:expr) => {
(unsafe { __slate_intrinsic_1cfe0293d0a2af8d($x) }) && $x
};
}

macro_rules! task_thread_info {
($task:expr) => {
unsafe { core::ptr::addr_of_mut!((*$task).thread_info) }
};
}

macro_rules! test_bit__3 {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if (unsafe { __slate_intrinsic_a39e427d3b6c2df4($nr) }) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
const_test_bit(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit(($nr) as u64, ($addr) as *const u64) as i32
};
__v
}
};
}

macro_rules! test_bit__2 {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if false && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*$addr) } {
const_test_bit(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit(($nr) as u64, ($addr) as *const u64) as i32
};
__v
}
};
}

macro_rules! test_bit {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if true && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
const_test_bit(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit(($nr) as u64, ($addr) as *const u64) as i32
};
__v
}
};
}

macro_rules! test_bit__5 {
($nr:expr, $addr:expr) => {
{
let __v: bool = (unsafe { __slate_intrinsic_1c5364fcd9da51fe($nr) }) && false && (($addr) as u64) != 0;
let __v2: i32 = if __v && false {
const_test_bit(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit(($nr) as u64, ($addr) as *const u64) as i32
};
__v2
}
};
}

/// time_after - returns true if the time a is after time b.
/// @a: first comparable as unsigned long
/// @b: second comparable as unsigned long
///
/// Do this with "<0" and ">=0" to only test the sign of the result. A
/// good compiler would generate better code (and a really good compiler
/// wouldn't care). Gcc is currently neither.
///
/// Return: %true is time a is after time b, otherwise %false.
macro_rules! time_after__2 {
($a:expr, $b:expr) => {
{
let __v: bool = typecheck!(u64) != 0 && typecheck__4!(u64) != 0;
__v && ($b.wrapping_sub($a) as i64) < 0
}
};
}

/// time_after - returns true if the time a is after time b.
/// @a: first comparable as unsigned long
/// @b: second comparable as unsigned long
///
/// Do this with "<0" and ">=0" to only test the sign of the result. A
/// good compiler would generate better code (and a really good compiler
/// wouldn't care). Gcc is currently neither.
///
/// Return: %true is time a is after time b, otherwise %false.
macro_rules! time_after {
($a:expr, $b:expr) => {
{
let __v: bool = typecheck!(u64) != 0 && typecheck!(u64) != 0;
__v && ($b.wrapping_sub($a) as i64) < 0
}
};
}

/// time_after_eq - returns true if the time a is after or the same as time b.
/// @a: first comparable as unsigned long
/// @b: second comparable as unsigned long
///
/// Return: %true is time a is after or the same as time b, otherwise %false.
macro_rules! time_after_eq__2 {
($a:expr, $b:expr) => {
{
let __v: bool = typecheck!(u64) != 0 && typecheck!(u64) != 0;
__v && ($a.wrapping_sub($b) as i64) >= 0
}
};
}

/// time_before - returns true if the time a is before time b.
/// @a: first comparable as unsigned long
/// @b: second comparable as unsigned long
///
/// Return: %true is time a is before time b, otherwise %false.
macro_rules! time_before {
($a:expr, $b:expr) => {
time_after__2!($b, $a)
};
}

macro_rules! type_max {
() => {
__type_max!()
};
}

/// Check at compile time that something is of a particular type.
/// Always evaluates to 1 so you may use it easily in comparisons.
macro_rules! typecheck__2 {
($type:ty) => {
let mut __v: i32;
let mut __dummy: $type = 0;
let mut __dummy2: $type = 0;
core::ptr::addr_of_mut!(__dummy) == core::ptr::addr_of_mut!(__dummy2);
__v = 1;
};
}

/// Check at compile time that something is of a particular type.
/// Always evaluates to 1 so you may use it easily in comparisons.
macro_rules! typecheck {
($type:ty) => {
{
let mut __v: i32;
let mut __v2: $type = 0;
let mut __v3: $type = 0;
core::ptr::addr_of_mut!(__v2) == core::ptr::addr_of_mut!(__v3);
__v = 1;
__v
}
};
}

/// Check at compile time that something is of a particular type.
/// Always evaluates to 1 so you may use it easily in comparisons.
macro_rules! typecheck__3 {
($type:ty) => {
let mut __v: i32;
let mut __v2: $type = 0;
let mut __v3: $type = 0;
core::ptr::addr_of_mut!(__v2) == core::ptr::addr_of_mut!(__v3);
__v = 1;
};
}

/// Check at compile time that something is of a particular type.
/// Always evaluates to 1 so you may use it easily in comparisons.
macro_rules! typecheck__4 {
($type:ty) => {
{
let mut __v: i32;
let mut __v2: $type = 0;
let mut __v3: $type = 0;
core::ptr::addr_of_mut!(__v2) == (core::ptr::addr_of_mut!(__v3) as *mut $type);
__v = 1;
__v
}
};
}

macro_rules! unlikely {
($x:expr) => {
(($x) != 0) as i64
};
}

macro_rules! unlikely__2 {
($x:expr) => {
!(!$x) as i64
};
}

macro_rules! unlikely__4 {
($x:expr) => {
(($x) != 0) as i64
};
}

macro_rules! unlikely__5 {
($x:expr) => {
(($x) != 0) as i64
};
}

macro_rules! wake_up_all {
($x:expr) => {
unsafe { __wake_up($x, 3u32, 0i32, core::ptr::null_mut::<()>()) };
};
}

/// work_pending - Find out whether a work item is currently pending
/// @work: The work item in question
macro_rules! work_pending {
($work:expr) => {
test_bit!(0i32, (unsafe { core::ptr::addr_of_mut!((*$work).data) }) as *mut u64)
};
}

unsafe extern "C" {
    static mut __SCK__WARN_trap: static_call_key;
    static mut kmsg_fops: file_operations;
    static mut __SCK__might_resched: static_call_key;
    static mut __per_cpu_offset: [u64; 64];
    static mut this_cpu_off: u64;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut page_offset_base: u64;
    static mut vmemmap_base: u64;
    static mut nr_cpu_ids: u32;
    static mut __cpu_possible_mask: cpumask;
    static mut __cpu_online_mask: cpumask;
    static mut boot_cpu_data: cpuinfo_x86;
    static mut cpuinfo_op: seq_operations;
    static mut __preempt_count: u64;
    static mut __SCK__preempt_schedule: static_call_key;
    static mut cpu_number: i32;
    static mut __SCK__cond_resched: static_call_key;
    static mut RCU: *const __ctx_lock_RCU;
    static mut jiffies: u64;
    static mut numa_node: i32;
    static mut sighand_cachep: *mut kmem_cache;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut fs_bio_set: bio_set;
    static mut force_irqthreads_key: static_key_false;
    static mut kstat: kernel_stat;
    static mut elfcorehdr_addr: u64;
    static mut tracepoint_srcu: srcu_struct;
    static mut __SCK__tp_func_block_rq_requeue: static_call_key;
    static mut __tracepoint_block_rq_requeue: tracepoint;
    static mut __SCK__tp_func_block_rq_complete: static_call_key;
    static mut __tracepoint_block_rq_complete: tracepoint;
    static mut __SCK__tp_func_block_rq_error: static_call_key;
    static mut __tracepoint_block_rq_error: tracepoint;
    static mut __SCK__tp_func_block_rq_insert: static_call_key;
    static mut __tracepoint_block_rq_insert: tracepoint;
    static mut __SCK__tp_func_block_rq_issue: static_call_key;
    static mut __tracepoint_block_rq_issue: tracepoint;
    static mut __SCK__tp_func_block_io_start: static_call_key;
    static mut __tracepoint_block_io_start: tracepoint;
    static mut __SCK__tp_func_block_io_done: static_call_key;
    static mut __tracepoint_block_io_done: tracepoint;
    static mut __SCK__tp_func_block_getrq: static_call_key;
    static mut __tracepoint_block_getrq: tracepoint;
    static mut __SCK__tp_func_block_plug: static_call_key;
    static mut __tracepoint_block_plug: tracepoint;
    static mut __SCK__tp_func_block_unplug: static_call_key;
    static mut __tracepoint_block_unplug: tracepoint;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn ___ratelimit(arg0: *mut ratelimit_state, arg1: *const u8) -> i32;
    fn _printk(arg0: *const u8, ...) -> i32;
    fn __cond_resched() -> i32;
    fn __SCT__might_resched() -> i32;
    fn __compiletime_assert_5();
    fn __compiletime_assert_6();
    fn __compiletime_assert_7();
    fn __compiletime_assert_10();
    fn __compiletime_assert_11();
    fn __compiletime_assert_15();
    fn __compiletime_assert_16();
    fn __compiletime_assert_17();
    fn __bad_size_call_parameter();
    fn __compiletime_assert_34();
    fn ____wrong_branch_error() -> bool;
    fn __xchg_wrong_size();
    fn __cmpxchg_wrong_size();
    fn _find_next_bit(arg0: *const u64, arg1: u64, arg2: u64) -> u64;
    fn _find_next_and_bit(arg0: *const u64, arg1: *const u64, arg2: u64, arg3: u64) -> u64;
    fn _find_first_bit(arg0: *const u64, arg1: u64) -> u64;
    fn _find_first_and_bit(arg0: *const u64, arg1: *const u64, arg2: u64) -> u64;
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn __compiletime_assert_63();
    fn llist_reverse_order(arg0: *mut llist_node) -> *mut llist_node;
    fn smp_call_function_single_async(arg0: i32, arg1: *mut __call_single_data) -> i32;
    fn _raw_spin_lock(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irqsave(arg0: *mut raw_spinlock) -> u64;
    fn _raw_spin_unlock(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irqrestore(arg0: *mut raw_spinlock, arg1: u64);
    fn delayed_work_timer_fn(arg0: *mut timer_list);
    fn schedule_timeout(arg0: i64) -> i64;
    fn schedule();
    fn __SCT__cond_resched() -> i32;
    fn pcpu_alloc_noprof(arg0: u64, arg1: u64, arg2: bool, arg3: u32) -> *mut ();
    fn __wake_up(arg0: *mut wait_queue_head, arg1: u32, arg2: i32, arg3: *mut ()) -> i32;
    fn init_wait_entry(arg0: *mut wait_queue_entry, arg1: i32);
    fn prepare_to_wait_event(arg0: *mut wait_queue_head, arg1: *mut wait_queue_entry, arg2: i32) -> i64;
    fn finish_wait(arg0: *mut wait_queue_head, arg1: *mut wait_queue_entry);
    fn mutex_init_generic(arg0: *mut mutex);
    fn mutex_lock(arg0: *mut mutex);
    fn mutex_unlock(arg0: *mut mutex);
    fn synchronize_rcu();
    fn __rcu_read_lock();
    fn __rcu_read_unlock();
    fn rcu_is_watching() -> bool;
    fn kvfree_call_rcu(arg0: *mut kvfree_rcu_head, arg1: *mut ());
    fn __init_rwsem(arg0: *mut rw_semaphore, arg1: *const u8, arg2: *mut lock_class_key);
    fn down_write(arg0: *mut rw_semaphore);
    fn up_write(arg0: *mut rw_semaphore);
    fn __init_swait_queue_head(arg0: *mut swait_queue_head, arg1: *const u8, arg2: *mut lock_class_key);
    fn wait_for_completion_io(arg0: *mut completion);
    fn wait_for_completion_io_timeout(arg0: *mut completion, arg1: u64) -> u64;
    fn completion_done(arg0: *mut completion) -> bool;
    fn complete(arg0: *mut completion);
    fn __msecs_to_jiffies(arg0: u32) -> u64;
    fn ktime_get() -> i64;
    fn ktime_get_mono_fast_ns() -> u64;
    fn timer_init_key(arg0: *mut timer_list, arg1: Option<unsafe extern "C" fn(*mut timer_list)>, arg2: u32, arg3: *const u8, arg4: *mut lock_class_key);
    fn mod_timer(arg0: *mut timer_list, arg1: u64) -> i32;
    fn cancel_delayed_work(arg0: *mut delayed_work) -> bool;
    fn cancel_delayed_work_sync(arg0: *mut delayed_work) -> bool;
    fn init_srcu_struct_generic(arg0: *mut srcu_struct) -> i32;
    fn __srcu_read_unlock(arg0: *mut srcu_struct, arg1: i32);
    fn __srcu_read_lock(arg0: *mut srcu_struct) -> i32;
    fn srcu_barrier(arg0: *mut srcu_struct);
    fn __compiletime_assert_167();
    fn __srcu_check_read_flavor(arg0: *mut srcu_struct, arg1: i32);
    fn call_srcu(arg0: *mut srcu_struct, arg1: *mut callback_head, arg2: Option<unsafe extern "C" fn(*mut callback_head)>);
    fn cleanup_srcu_struct(arg0: *mut srcu_struct);
    fn synchronize_srcu(arg0: *mut srcu_struct);
    fn alloc_pages_node_noprof(arg0: i32, arg1: u32, arg2: u32) -> *mut page;
    fn xa_load(arg0: *mut xarray, arg1: u64) -> *mut ();
    fn xa_destroy(arg0: *mut xarray);
    fn __compiletime_assert_228();
    fn __compiletime_assert_229();
    fn __compiletime_assert_230();
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_361();
    fn __compiletime_assert_386();
    fn percpu_ref_kill_and_confirm(arg0: *mut percpu_ref, arg1: Option<unsafe extern "C" fn(*mut percpu_ref)>);
    fn percpu_ref_resurrect(arg0: *mut percpu_ref);
    fn percpu_ref_is_zero(arg0: *mut percpu_ref) -> bool;
    fn __compiletime_assert_387();
    fn kfree(arg0: *const ());
    fn __compiletime_assert_392();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_node_noprof(arg0: u64, arg1: u32, arg2: i32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_cache_node_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: i32, arg3: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_large_node_noprof(arg0: u64, arg1: u32, arg2: i32) -> *mut ();
    fn kobject_put(arg0: *mut kobject);
    fn __module_get(arg0: *mut module);
    fn module_put(arg0: *mut module);
    fn cpus_equal_capacity(arg0: i32, arg1: i32) -> bool;
    fn cpus_share_cache(arg0: i32, arg1: i32) -> bool;
    fn __compiletime_assert_537();
    fn __compiletime_assert_540();
    fn __bio_advance(arg0: *mut bio, arg1: u32);
    fn bio_put(arg0: *mut bio);
    fn bio_alloc_clone(arg0: *mut block_device, arg1: *mut bio, arg2: u32, arg3: *mut bio_set) -> *mut bio;
    fn bio_endio(arg0: *mut bio);
    fn sbitmap_init_node(arg0: *mut sbitmap, arg1: u32, arg2: i32, arg3: u32, arg4: i32, arg5: bool, arg6: bool) -> i32;
    fn sbitmap_resize(arg0: *mut sbitmap, arg1: u32);
    fn sbitmap_any_bit_set(arg0: *const sbitmap) -> bool;
    fn sbitmap_queue_resize(arg0: *mut sbitmap_queue, arg1: u32);
    fn __sbitmap_queue_get(arg0: *mut sbitmap_queue) -> i32;
    fn blk_queue_flag_set(arg0: u32, arg1: *mut request_queue);
    fn blk_queue_flag_clear(arg0: u32, arg1: *mut request_queue);
    fn blk_queue_enter(arg0: *mut request_queue, arg1: u32) -> i32;
    fn blk_queue_exit(arg0: *mut request_queue);
    fn blk_sync_queue(arg0: *mut request_queue);
    fn blk_op_str(arg0: u32) -> *const u8;
    fn blk_queue_rq_timeout(arg0: *mut request_queue, arg1: u32);
    fn blk_get_queue(arg0: *mut request_queue) -> bool;
    fn blk_put_queue(arg0: *mut request_queue);
    fn kblockd_mod_delayed_work_on(arg0: i32, arg1: *mut delayed_work, arg2: u64) -> i32;
    fn __compiletime_assert_575();
    fn blk_mq_map_queues(arg0: *mut blk_mq_queue_map);
    fn __compiletime_assert_579();
    fn open_softirq(arg0: i32, arg1: Option<unsafe extern "C" fn()>);
    fn __raise_softirq_irqoff(arg0: u32);
    fn raise_softirq(arg0: u32);
    fn __bio_integrity_action(arg0: *mut bio) -> u32;
    fn __cpuhp_setup_state(arg0: i32, arg1: *const u8, arg2: bool, arg3: Option<unsafe extern "C" fn(u32) -> i32>, arg4: Option<unsafe extern "C" fn(u32) -> i32>, arg5: bool) -> i32;
    fn __cpuhp_state_add_instance(arg0: i32, arg1: *mut hlist_node, arg2: bool) -> i32;
    fn __cpuhp_state_remove_instance(arg0: i32, arg1: *mut hlist_node, arg2: bool) -> i32;
    fn pm_wakeup_pending() -> bool;
    fn msleep(arg0: u32);
    fn housekeeping_test_cpu(arg0: i32, arg1: u32) -> bool;
    fn __traceiter_block_rq_requeue(arg0: *mut (), arg1: *mut request) -> i32;
    fn __SCT__tp_func_block_rq_requeue(arg0: *mut (), arg1: *mut request) -> i32;
    fn __compiletime_assert_710();
    fn __traceiter_block_rq_complete(arg0: *mut (), arg1: *mut request, arg2: u8, arg3: u32) -> i32;
    fn __SCT__tp_func_block_rq_complete(arg0: *mut (), arg1: *mut request, arg2: u8, arg3: u32) -> i32;
    fn __compiletime_assert_717();
    fn __traceiter_block_rq_error(arg0: *mut (), arg1: *mut request, arg2: u8, arg3: u32) -> i32;
    fn __SCT__tp_func_block_rq_error(arg0: *mut (), arg1: *mut request, arg2: u8, arg3: u32) -> i32;
    fn __compiletime_assert_724();
    fn __traceiter_block_rq_insert(arg0: *mut (), arg1: *mut request) -> i32;
    fn __SCT__tp_func_block_rq_insert(arg0: *mut (), arg1: *mut request) -> i32;
    fn __compiletime_assert_738();
    fn __traceiter_block_rq_issue(arg0: *mut (), arg1: *mut request) -> i32;
    fn __SCT__tp_func_block_rq_issue(arg0: *mut (), arg1: *mut request) -> i32;
    fn __compiletime_assert_745();
    fn __traceiter_block_io_start(arg0: *mut (), arg1: *mut request) -> i32;
    fn __SCT__tp_func_block_io_start(arg0: *mut (), arg1: *mut request) -> i32;
    fn __compiletime_assert_759();
    fn __traceiter_block_io_done(arg0: *mut (), arg1: *mut request) -> i32;
    fn __SCT__tp_func_block_io_done(arg0: *mut (), arg1: *mut request) -> i32;
    fn __compiletime_assert_766();
    fn __traceiter_block_getrq(arg0: *mut (), arg1: *mut bio) -> i32;
    fn __SCT__tp_func_block_getrq(arg0: *mut (), arg1: *mut bio) -> i32;
    fn __compiletime_assert_801();
    fn __traceiter_block_plug(arg0: *mut (), arg1: *mut request_queue) -> i32;
    fn __SCT__tp_func_block_plug(arg0: *mut (), arg1: *mut request_queue) -> i32;
    fn __compiletime_assert_815();
    fn __traceiter_block_unplug(arg0: *mut (), arg1: *mut request_queue, arg2: u32, arg3: bool) -> i32;
    fn __SCT__tp_func_block_unplug(arg0: *mut (), arg1: *mut request_queue, arg2: u32, arg3: bool) -> i32;
    fn __compiletime_assert_822();
    fn __blk_crypto_rq_get_keyslot(arg0: *mut request) -> u8;
    fn __blk_crypto_rq_put_keyslot(arg0: *mut request);
    fn __blk_crypto_free_request(arg0: *mut request);
    fn __blk_crypto_rq_bio_prep(arg0: *mut request, arg1: *mut bio, arg2: u32) -> i32;
    fn is_flush_rq(arg0: *mut request) -> bool;
    fn blk_alloc_flush_queue(arg0: i32, arg1: i32, arg2: u32) -> *mut blk_flush_queue;
    fn blk_free_flush_queue(arg0: *mut blk_flush_queue);
    fn blk_status_to_str(arg0: u8) -> *const u8;
    fn blk_queue_start_drain(arg0: *mut request_queue) -> bool;
    fn __bio_queue_enter(arg0: *mut request_queue, arg1: *mut bio) -> i32;
    fn __compiletime_assert_870();
    fn __compiletime_assert_873();
    fn blk_add_timer(arg0: *mut request);
    fn blk_attempt_plug_merge(arg0: *mut request_queue, arg1: *mut bio, arg2: u32) -> bool;
    fn blk_insert_flush(arg0: *mut request) -> bool;
    fn elv_update_nr_hw_queues(arg0: *mut request_queue, arg1: *mut elv_change_ctx);
    fn elevator_set_none(arg0: *mut request_queue);
    fn bio_split_discard(arg0: *mut bio, arg1: *const queue_limits, arg2: *mut u32) -> *mut bio;
    fn bio_split_write_zeroes(arg0: *mut bio, arg1: *const queue_limits, arg2: *mut u32) -> *mut bio;
    fn bio_split_rw(arg0: *mut bio, arg1: *const queue_limits, arg2: *mut u32) -> *mut bio;
    fn bio_split_zone_append(arg0: *mut bio, arg1: *const queue_limits, arg2: *mut u32) -> *mut bio;
    fn blk_recalc_rq_segments(arg0: *mut request) -> u32;
    fn blk_try_merge(arg0: *mut request, arg1: *mut bio) -> u32;
    fn update_io_ticks(arg0: *mut block_device, arg1: u64, arg2: bool);
    fn __alloc_disk_node(arg0: *mut request_queue, arg1: i32, arg2: *mut lock_class_key) -> *mut gendisk;
    fn blk_alloc_queue(arg0: *mut queue_limits, arg1: i32) -> *mut request_queue;
    fn __compiletime_assert_881();
    fn __compiletime_assert_882();
    fn blk_integrity_prepare(arg0: *mut request);
    fn blk_integrity_complete(arg0: *mut request, arg1: u32);
    fn blk_stat_add(arg0: *mut request, arg1: u64);
    fn blk_mq_hw_queue_to_node(arg0: *mut blk_mq_queue_map, arg1: u32) -> i32;
    fn blk_mq_sysfs_init(arg0: *mut request_queue);
    fn blk_mq_sysfs_deinit(arg0: *mut request_queue);
    fn blk_mq_sysfs_register_hctxs(arg0: *mut request_queue) -> i32;
    fn blk_mq_sysfs_unregister_hctxs(arg0: *mut request_queue);
    fn blk_mq_hctx_kobj_init(arg0: *mut blk_mq_hw_ctx);
    fn blk_mq_init_tags(arg0: u32, arg1: u32, arg2: u32, arg3: i32) -> *mut blk_mq_tags;
    fn blk_mq_free_tags(arg0: *mut blk_mq_tag_set, arg1: *mut blk_mq_tags);
    fn blk_mq_get_tag(arg0: *mut blk_mq_alloc_data) -> u32;
    fn blk_mq_get_tags(arg0: *mut blk_mq_alloc_data, arg1: i32, arg2: *mut u32) -> u64;
    fn blk_mq_put_tag(arg0: *mut blk_mq_tags, arg1: *mut blk_mq_ctx, arg2: u32);
    fn blk_mq_put_tags(arg0: *mut blk_mq_tags, arg1: *mut i32, arg2: i32);
    fn blk_mq_tag_resize_shared_tags(arg0: *mut blk_mq_tag_set, arg1: u32);
    fn blk_mq_tag_update_sched_shared_tags(arg0: *mut request_queue, arg1: u32);
    fn blk_mq_tag_wakeup_all(arg0: *mut blk_mq_tags, arg1: bool);
    fn blk_mq_queue_tag_busy_iter(arg0: *mut request_queue, arg1: Option<unsafe extern "C" fn(*mut request, *mut ()) -> bool>, arg2: *mut ());
    fn blk_mq_all_tag_iter(arg0: *mut blk_mq_tags, arg1: Option<unsafe extern "C" fn(*mut request, *mut ()) -> bool>, arg2: *mut ());
    fn __blk_mq_tag_busy(arg0: *mut blk_mq_hw_ctx);
    fn __blk_mq_tag_idle(arg0: *mut blk_mq_hw_ctx);
    fn __compiletime_assert_885();
    fn __compiletime_assert_888();
    fn blk_mq_debugfs_register_hctxs(arg0: *mut request_queue);
    fn blk_mq_debugfs_unregister_hctxs(arg0: *mut request_queue);
    fn __compiletime_assert_889();
    fn elv_merge(arg0: *mut request_queue, arg1: *mut *mut request, arg2: *mut bio) -> u32;
    fn blk_mq_sched_bio_merge(arg0: *mut request_queue, arg1: *mut bio, arg2: u32) -> bool;
    fn blk_mq_sched_mark_restart_hctx(arg0: *mut blk_mq_hw_ctx);
    fn __blk_mq_sched_restart(arg0: *mut blk_mq_hw_ctx);
    fn blk_mq_sched_dispatch_requests(arg0: *mut blk_mq_hw_ctx);
    fn blk_mq_alloc_sched_res_batch(arg0: *mut xarray, arg1: *mut blk_mq_tag_set, arg2: u32) -> i32;
    fn blk_mq_alloc_sched_ctx_batch(arg0: *mut xarray, arg1: *mut blk_mq_tag_set) -> i32;
    fn blk_mq_free_sched_ctx_batch(arg0: *mut xarray);
    fn __rq_qos_cleanup(arg0: *mut rq_qos, arg1: *mut bio);
    fn __rq_qos_done(arg0: *mut rq_qos, arg1: *mut request);
    fn __rq_qos_issue(arg0: *mut rq_qos, arg1: *mut request);
    fn __rq_qos_requeue(arg0: *mut rq_qos, arg1: *mut request);
    fn __rq_qos_throttle(arg0: *mut rq_qos, arg1: *mut bio);
    fn __rq_qos_track(arg0: *mut rq_qos, arg1: *mut request, arg2: *mut bio);
    fn __compiletime_assert_905();
    fn __compiletime_assert_906();
    fn __compiletime_assert_908();
    fn __compiletime_assert_909();
    fn __compiletime_assert_912();
    fn __builtin_prefetch(arg0: *const (), ...);
    fn __compiletime_assert_921();
    fn __compiletime_assert_927();
    fn __compiletime_assert_934();
    fn __compiletime_assert_937();
    fn __compiletime_assert_942();
    fn __compiletime_assert_943();
    fn __compiletime_assert_956();
    fn __compiletime_assert_963();
    fn __compiletime_assert_989();
    fn __compiletime_assert_992();
    fn __compiletime_assert_1008();
    fn __compiletime_assert_1009();
    fn __compiletime_assert_1010();
    fn __compiletime_assert_1015();
    fn __compiletime_assert_1018();
    fn __compiletime_assert_1024();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_1c5364fcd9da51fe(arg0: u32) -> bool;
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_a39e427d3b6c2df4(arg0: i32) -> bool;
    #[link_name = "llvm.is.constant.i64"]
    fn __slate_intrinsic_e1c548fa55baa574(arg0: i64) -> bool;
    #[link_name = "llvm.is.constant.i64"]
    fn __slate_intrinsic_f217c4720f04f6db(arg0: u64) -> bool;
}

// generic data direction definitions
// The type of an index into the pagecache.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_types_h_190 {
    counter: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_types_h_197 {
    counter: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_types_h_202 {
    refcnt: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct list_head {
    next: *mut list_head,
    prev: *mut list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hlist_head {
    first: *mut hlist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hlist_node {
    next: *mut hlist_node,
    pprev: *mut *mut hlist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct callback_head {
    next: *mut callback_head,
    func: Option<unsafe extern "C" fn(*mut callback_head)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kvfree_rcu_head {
    next: *mut kvfree_rcu_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rcuwait {
    task: *mut task_struct,
}


fn kasan_check_read(p: *const (), size: u32) -> bool {
true
}

fn kasan_check_write(p: *const (), size: u32) -> bool {
true
}

fn kcsan_check_access(ptr: *const (), size: u64, r#type: i32) {
}

// kcsan_check_read - check regular read access for races
//
// @ptr: address of access
// @size: size of access
// kcsan_check_write - check regular write access for races
//
// @ptr: address of access
// @size: size of access
// ASSERT_EXCLUSIVE_WRITER - assert no concurrent writes to @var
//
// Assert that there are no concurrent writes to @var; other readers are
// allowed. This assertion can be used to specify properties of concurrent code,
// where violation cannot be detected as a normal data race.
//
// For example, if we only have a single writer, but multiple concurrent
// readers, to avoid data races, all these accesses must be marked; even
// concurrent marked writes racing with the single writer are bugs.
// Unfortunately, due to being marked, they are no longer data races. For cases
// like these, we can use the macro as follows:
//
// .. code-block:: c
//
// 	void writer(void) {
// 		spin_lock(&update_foo_lock);
// 		ASSERT_EXCLUSIVE_WRITER(shared_foo);
// 		WRITE_ONCE(shared_foo, ...);
// 		spin_unlock(&update_foo_lock);
// 	}
// 	void reader(void) {
// 		// update_foo_lock does not need to be held!
// 		... = READ_ONCE(shared_foo);
// 	}
//
// Note: ASSERT_EXCLUSIVE_WRITER_SCOPED(), if applicable, performs more thorough
// checking if a clear scope where no concurrent writes are expected exists.
//
// @var: variable to assert on
// ASSERT_EXCLUSIVE_ACCESS - assert no concurrent accesses to @var
//
// Assert that there are no concurrent accesses to @var (no readers nor
// writers). This assertion can be used to specify properties of concurrent
// code, where violation cannot be detected as a normal data race.
//
// For example, where exclusive access is expected after determining no other
// users of an object are left, but the object is not actually freed. We can
// check that this property actually holds as follows:
//
// .. code-block:: c
//
// 	if (refcount_dec_and_test(&obj->refcnt)) {
// 		ASSERT_EXCLUSIVE_ACCESS(*obj);
// 		do_some_cleanup(obj);
// 		release_for_reuse(obj);
// 	}
//
// Note:
//
// 1. ASSERT_EXCLUSIVE_ACCESS_SCOPED(), if applicable, performs more thorough
//    checking if a clear scope where no concurrent accesses are expected exists.
//
// 2. For cases where the object is freed, `KASAN <kasan.html>`_ is a better
//    fit to detect use-after-free bugs.
//
// @var: variable to assert on
// ASSERT_EXCLUSIVE_BITS - assert no concurrent writes to subset of bits in @var
//
// Bit-granular variant of ASSERT_EXCLUSIVE_WRITER().
//
// Assert that there are no concurrent writes to a subset of bits in @var;
// concurrent readers are permitted. This assertion captures more detailed
// bit-level properties, compared to the other (word granularity) assertions.
// Only the bits set in @mask are checked for concurrent modifications, while
// ignoring the remaining bits, i.e. concurrent writes (or reads) to ~mask bits
// are ignored.
//
// Use this for variables, where some bits must not be modified concurrently,
// yet other bits are expected to be modified concurrently.
//
// For example, variables where, after initialization, some bits are read-only,
// but other bits may still be modified concurrently. A reader may wish to
// assert that this is true as follows:
//
// .. code-block:: c
//
// 	ASSERT_EXCLUSIVE_BITS(flags, READ_ONLY_MASK);
// 	foo = (READ_ONCE(flags) & READ_ONLY_MASK) >> READ_ONLY_SHIFT;
//
// Note: The access that immediately follows ASSERT_EXCLUSIVE_BITS() is assumed
// to access the masked bits only, and KCSAN optimistically assumes it is
// therefore safe, even in the presence of data races, and marking it with
// READ_ONCE() is optional from KCSAN's point-of-view. We caution, however, that
// it may still be advisable to do so, since we cannot reason about all compiler
// optimizations when it comes to bit manipulations (on the reader and writer
// side). If you are sure nothing can go wrong, we can write the above simply
// as:
//
// .. code-block:: c
//
// 	ASSERT_EXCLUSIVE_BITS(flags, READ_ONLY_MASK);
// 	foo = (flags & READ_ONLY_MASK) >> READ_ONLY_SHIFT;
//
// Another example, where this may be used, is when certain bits of @var may
// only be modified when holding the appropriate lock, but other bits may still
// be modified concurrently. Writers, where other bits may change concurrently,
// could use the assertion as follows:
//
// .. code-block:: c
//
// 	spin_lock(&foo_lock);
// 	ASSERT_EXCLUSIVE_BITS(flags, FOO_MASK);
// 	old_flags = flags;
// 	new_flags = (old_flags & ~FOO_MASK) | (new_foo << FOO_SHIFT);
// 	if (cmpxchg(&flags, old_flags, new_flags) != old_flags) { ... }
// 	spin_unlock(&foo_lock);
//
// @var: variable to assert on
// @mask: only check for modifications to bits set in @mask
#[inline(always)]
fn __must_check_overflow(overflow: bool) -> bool {
unlikely__2!(overflow) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_static_call_types_h_65 {
    r#type: u64,
    mods: *mut static_call_mod,
    sites: *mut static_call_site,
}


// check_add_overflow() - Calculate addition with overflow checking
// @a: first addend
// @b: second addend
// @d: pointer to store sum
//
// Returns true on wrap-around, false otherwise.
//
// *@d holds the results of the attempted addition, regardless of whether
// wrap-around occurred.
// check_sub_overflow() - Calculate subtraction with overflow checking
// @a: minuend; value to subtract from
// @b: subtrahend; value to subtract from @a
// @d: pointer to store difference
//
// Returns true on wrap-around, false otherwise.
//
// *@d holds the results of the attempted subtraction, regardless of whether
// wrap-around occurred.
#[inline(always)]
fn size_mul(factor1: u64, factor2: u64) -> u64 {
let mut bytes: u64 = 0;
if check_mul_overflow!(factor1, factor2, core::ptr::addr_of_mut!(bytes)) {
return !0;
}
bytes
}

#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_site {
    addr: i32,
    key: i32,
}


// __ADDRESSABLE() is used to ensure the key symbol doesn't get stripped from
// the symbol table so that objtool can reference it when it generates the
// .static_call_sites section.
#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_key {
    func: *mut (),
    __slate_anon_1: __SlateRecord_static_call_types_h_65,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_qspinlock_types_h_44 {
    val: __SlateRecord_types_h_190,
    __slate_anon_1: __SlateRecord_qspinlock_types_h_44_2,
    __slate_anon_2: __SlateRecord_qspinlock_types_h_44_3,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_qspinlock_types_h_44_2 {
    locked: u8,
    pending: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_qspinlock_types_h_44_3 {
    locked_pending: u16,
    tail: u16,
}


#[inline(always)]
unsafe extern "C" fn __WARN_validate_printf(fmt: *const u8, __va_args: ...) {
}

// inhibit tail-call optimization
#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_qrwlock_types_h_27 {
    cnts: __SlateRecord_types_h_190,
    __slate_anon_1: __SlateRecord_qrwlock_types_h_27_2,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_qrwlock_types_h_27_2 {
    wlocked: u8,
    __lstate: [u8; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct qspinlock {
    __slate_anon_0: __SlateRecord_qspinlock_types_h_44,
}


// Bitfields in the atomic value:
//
// When NR_CPUS < 16K
//  0- 7: locked byte
//  8-15: pending byte
// 16-17: tail index
// 18-31: tail cpu (+1)
//
// When NR_CPUS >= 16K
//  0- 7: locked byte
//     8: pending
//  9-10: tail index
// 11-31: tail cpu (+1)
#[repr(C)]
#[derive(Clone, Copy)]
struct qrwlock {
    __slate_anon_0: __SlateRecord_qrwlock_types_h_27,
    wait_lock: qspinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct lock_class_key {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct lockdep_map {
}


#[inline(always)]
fn __acquire_ctx_lock__1(var: &raw_spinlock) {
}

#[inline(always)]
fn __acquire_shared_ctx_lock__1(var: &raw_spinlock) {
}

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
}


#[inline(always)]
fn __release_ctx_lock__1(var: *const raw_spinlock) {
}

#[inline(always)]
fn __release_shared_ctx_lock__1(var: &raw_spinlock) {
}

#[inline(always)]
fn __assume_ctx_lock__1(var: &raw_spinlock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct raw_spinlock {
    raw_lock: qspinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ratelimit_state {
    lock: raw_spinlock,
    interval: i32,
    burst: i32,
    rs_n_left: __SlateRecord_types_h_190,
    missed: __SlateRecord_types_h_190,
    flags: u32,
    begin: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_bitmap {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bug_entry {
    bug_addr_disp: i32,
    format_disp: i32,
    file_disp: i32,
    line: u16,
    flags: u16,
}


#[inline(always)]
fn const_test_bit(nr: u64, addr: *const u64) -> bool {
let p: *const u64 = (addr as *const u64).wrapping_offset(BIT_WORD!(nr) as isize);
let mask: u64 = BIT_MASK!(nr);
let val: u64 = unsafe { *p };
val & mask != 0
}

#[inline(always)]
fn arch_set_bit(mut nr: i64, mut addr: *mut u64) {
if unsafe { __slate_intrinsic_e1c548fa55baa574(nr) } {
unsafe {
core::arch::asm!("lock orb {1:l},({0:r})", in(reg) ((addr as *const u8).wrapping_offset((nr >> (3 as i32)) as isize) as *mut ()) as *mut u8, in(reg) ((1 as i32) << (nr & ((7 as i32) as i64))) as u32, options(nostack, att_syntax));
}
} else {
unsafe {
core::arch::asm!("lock  btsq  {1:r},({0:r})", in(reg) addr as *mut i64, in(reg) nr as u64, options(nostack, att_syntax));
}
}
}

#[inline(always)]
fn arch_clear_bit(mut nr: i64, mut addr: *mut u64) {
if unsafe { __slate_intrinsic_e1c548fa55baa574(nr) } {
unsafe {
core::arch::asm!("lock andb {1:l},({0:r})", in(reg) ((addr as *const u8).wrapping_offset((nr >> (3 as i32)) as isize) as *mut ()) as *mut u8, in(reg) !((1 as i32) << (nr & ((7 as i32) as i64))) as u32, options(nostack, att_syntax));
}
} else {
unsafe {
core::arch::asm!("lock  btrq  {1:r},({0:r})", in(reg) addr as *mut i64, in(reg) nr as u64, options(nostack, att_syntax));
}
}
}

#[inline(always)]
fn constant_test_bit(nr: i64, addr: *const u64) -> bool {
1u64 << (nr & 63i64) & unsafe { core::ptr::read_volatile(addr.wrapping_offset((nr >> 6i32) as isize)) } != 0
}

#[inline(always)]
fn variable_test_bit(mut nr: i64, mut addr: *const u64) -> bool {
let mut oldbit: bool = false;
unsafe {
let __t0: u8;
core::arch::asm!(" btq  {2:r},({1:r})\nsetc {0}", lateout(reg_byte) __t0, in(reg) addr as *mut u64, in(reg) nr as u64, options(nostack, att_syntax));
oldbit = (__t0 as u8) != 0;
}
oldbit
}

#[inline(always)]
fn arch_test_bit(nr: u64, addr: *const u64) -> bool {
let __v0: i32 = if unsafe { __slate_intrinsic_f217c4720f04f6db(nr) } {
constant_test_bit(nr as i64, addr) as i32
} else {
variable_test_bit(nr as i64, addr) as i32
};
__v0 != 0
}

#[inline(always)]
fn variable__ffs(mut word: u64) -> u64 {
unsafe {
let __t0: u64;
core::arch::asm!("tzcnt {1:r},{0:r}", lateout(reg) __t0, in(reg) word as u64, options(pure, nomem, nostack, att_syntax));
word = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
word
}

// ffz - find first zero bit in word
// @word: The word to search
//
// Undefined if no zero exists, so code should check against ~0UL first.
#[inline(always)]
fn fls(mut x: u32) -> i32 {
let mut r: i32 = 0;
if unsafe { __slate_intrinsic_1c5364fcd9da51fe(x) } {
return if x != 0 { 32i32.wrapping_sub(x.leading_zeros() as i32) } else { 0 };
}
// AMD64 says BSRL won't clobber the dest reg if x==0; Intel64 says the
// dest reg is undefined if x==0, but their CPU architect says its
// value is written to set it to the same as before, except that the
// top 32 bits will be cleared.
//
// We cannot do this on 32 bits because at the very least some
// 486 CPUs did not behave this way.
unsafe {
let __t0: u32;
core::arch::asm!("bsrl {1:e},{0:e}", inlateout(reg) (1 as i32).wrapping_neg() as u32 => __t0, in(reg) x as u32, options(pure, nomem, nostack, att_syntax));
r = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
}
r.wrapping_add(1)
}

#[inline(always)]
fn fls64(mut x: u64) -> i32 {
let mut bitpos: i32 = 1i32.wrapping_neg();
if unsafe { __slate_intrinsic_f217c4720f04f6db(x) } {
return if x != 0 { 64i32.wrapping_sub(x.leading_zeros() as i32) } else { 0 };
}
// AMD64 says BSRQ won't clobber the dest reg if x==0; Intel64 says the
// dest reg is undefined if x==0, but their CPU architect says its
// value is written to set it to the same as before.
unsafe {
let __t0: u32;
core::arch::asm!("bsrq {1:r},{0:r}", inlateout(reg) bitpos as u32 => __t0, in(reg) x as u64, options(pure, nomem, nostack, att_syntax));
bitpos = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
}
bitpos.wrapping_add(1)
}

#[inline(always)]
fn instrument_read_write(v: *const (), size: u64) {
kasan_check_write(v, size as u32);
kcsan_check_read_write!(v, size);
}

#[inline(always)]
fn instrument_atomic_check_alignment(v: *const (), size: u64) {
if IS_ENABLED!() != 0 {
let mut mask: u32 = size.wrapping_sub(1) as u32;
if IS_ENABLED!() != 0 {
mask = ((mask as u64) & (core::mem::size_of::<__SlateRecord_instrumented_h_66>() as u64).wrapping_sub(1)) as u32;
}
let mut __v2: i64;
let __ret_warn_on: i32 = ((v as u64) & (mask as u64) != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("2: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 2b - ., 3; .popsection\n# {0}", const 2 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_instrument_atomic_check_alignment_3, sym __slate_asm_string_instrument_atomic_check_alignment_4, const 67 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("3: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 3b - ., 4; .popsection\n# {0}", const 3 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v2 = (__ret_warn_on != 0) as i64;
}
}

#[inline(always)]
fn instrument_atomic_read(v: *const (), size: u64) {
kasan_check_read(v, size as u32);
kcsan_check_atomic_read!(v, size);
instrument_atomic_check_alignment(v, size);
}

#[inline(always)]
fn instrument_atomic_write(v: *const (), size: u64) {
kasan_check_write(v, size as u32);
kcsan_check_atomic_write!(v, size);
instrument_atomic_check_alignment(v, size);
}

#[inline(always)]
fn instrument_atomic_read_write(v: *const (), size: u64) {
kasan_check_write(v, size as u32);
kcsan_check_atomic_read_write!(v, size);
instrument_atomic_check_alignment(v, size);
}

// instrument_get_user() - add instrumentation to get_user()-like macros
// @to: destination variable, may not be address-taken
//
// get_user() and friends are fragile, so it may depend on the implementation
// whether the instrumentation happens before or after the data is copied from
// the userspace.
#[inline(always)]
fn set_bit(nr: i64, addr: *mut u64) {
instrument_atomic_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_set_bit(nr, addr);
}

#[inline(always)]
fn clear_bit(nr: i64, addr: *mut u64) {
instrument_atomic_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_clear_bit(nr, addr);
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_jump_label_h_95 {
    r#type: u64,
    entries: *mut jump_entry,
    next: *mut static_key_mod,
}


#[inline(always)]
fn _test_bit(nr: u64, addr: *const u64) -> bool {
instrument_atomic_read(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_bit(nr, addr)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct static_key_mod {
}


#[inline(always)]
fn __ilog2_u32(n: u32) -> i32 {
fls(n).wrapping_sub(1)
}

#[inline(always)]
fn __ilog2_u64(n: u64) -> i32 {
fls64(n).wrapping_sub(1)
}

// order_base_2 - calculate the (rounded up) base 2 order of the argument
// @n: parameter
//
// The first few values calculated by this routine:
//  ob2(0) = 0
//  ob2(1) = 0
//  ob2(2) = 1
//  ob2(3) = 2
//  ob2(4) = 2
//  ob2(5) = 3
//  ... and so on.
// bits_per - calculate the number of bits required for the argument
// @n: parameter
//
// This is constant-capable and can be used for compile time
// initializations, e.g bitfields.
//
// The first few values calculated by this routine:
// bf(0) = 1
// bf(1) = 1
// bf(2) = 2
// bf(3) = 2
// bf(4) = 3
// ... and so on.
#[inline(always)]
fn might_resched() {
let mut __v0: Option<unsafe extern "C" fn() -> i32>;
__v0 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn() -> i32>>(__SCT__might_resched as *const ()) };
unsafe { __v0.unwrap()() };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__might_resched_4: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__might_resched) }) as *mut ();

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_find_h_67 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_find_h_97 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_find_h_207 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_find_h_316 {
}


fn __might_resched(file: *const u8, line: i32, offsets: u32) {
}

// LIST_HEAD - definition of a &struct list_head with initialization values
// @name: name of the list_head
fn INIT_LIST_HEAD(list: *mut list_head) {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_5() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!((*list).next, list);
if true {
break;
}
}
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_6() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!((*list).prev, list);
if true {
break;
}
}
}

fn __list_add_valid(new: *mut list_head, prev: *mut list_head, next: *mut list_head) -> bool {
true
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_processor_h_111 {
    hw_cpu_type: u32,
    __slate_anon_1: __SlateRecord_processor_h_113,
    __slate_anon_2: __SlateRecord_processor_h_119,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_processor_h_113 {
    __slate_bits_0: __slate_bits::__SlateBits151U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_processor_h_119 {
    __slate_bits_0: __slate_bits::__SlateBits152U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_processor_h_136 {
    __slate_anon_0: __SlateRecord_processor_h_142,
    x86_vfm: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_processor_h_142 {
    x86_model: u8,
    x86: u8,
    x86_vendor: u8,
    x86_reserved: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_processor_h_154 {
    intel_platform_id: u8,
    amd_unused: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_processor_h_177 {
    x86_capability: [u32; 24],
    x86_capability_alignment: u64,
}


fn __list_del_entry_valid(entry: *mut list_head) -> bool {
true
}

#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
}


#[inline(always)]
fn __list_add(new: *mut list_head, prev: *mut list_head, next: *mut list_head) {
if !__list_add_valid(new, prev, next) {
return;
}
unsafe {
(*next).prev = new;
}
unsafe {
(*new).next = next;
}
unsafe {
(*new).prev = prev;
}
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_7() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!((*prev).next, new);
if true {
break;
}
}
}

#[inline(always)]
fn list_add(new: *mut list_head, head: *mut list_head) {
__list_add(new, head, unsafe { (*head).next });
}

fn list_add_tail(new: *mut list_head, head: *mut list_head) {
__list_add(new, unsafe { (*head).prev }, head);
}

fn __list_del(prev: *mut list_head, next: *mut list_head) {
unsafe {
(*next).prev = prev;
}
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_10() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!((*prev).next, next);
if true {
break;
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


fn __list_del_entry(entry: *mut list_head) {
if !__list_del_entry_valid(entry) {
return;
}
__list_del(unsafe { (*entry).prev }, unsafe { (*entry).next });
}

fn list_del_init(entry: *mut list_head) {
__list_del_entry(entry);
INIT_LIST_HEAD(entry);
}

fn list_is_head(list: *const list_head, head: *const list_head) -> i32 {
(list == head) as i32
}

fn list_empty(head: *const list_head) -> i32 {
let mut __v0: *mut list_head;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_11() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*head).next) as *const *mut list_head) };
(__v0 == (head as *mut list_head)) as i32
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_restart_block_h_29 {
    futex: __SlateRecord_restart_block_h_31,
    nanosleep: __SlateRecord_restart_block_h_40,
    poll: __SlateRecord_restart_block_h_50,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_restart_block_h_31 {
    uaddr: *mut u32,
    val: u32,
    flags: u32,
    bitset: u32,
    time: i64,
    uaddr2: *mut u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_restart_block_h_40 {
    clockid: i32,
    r#type: u32,
    __slate_anon_2: __SlateRecord_restart_block_h_43,
    expires: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_restart_block_h_43 {
    rmtp: *mut __kernel_timespec,
    compat_rmtp: *mut old_timespec32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_restart_block_h_50 {
    ufds: *mut pollfd,
    nfds: i32,
    has_timeout: i32,
    end_time: timespec64,
}


fn list_empty_careful(head: *const list_head) -> i32 {
let mut __v0: *mut list_head;
let mut __v1: *mut list_head;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_15() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*head).next) as *const *mut list_head) };
let ___p1: *mut list_head = __v1;
loop {
if !__native_word!() {
unsafe { __compiletime_assert_16() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
__v0 = ___p1;
let next: *mut list_head = __v0;
let __v4: bool = if list_is_head(next as *const list_head, head) != 0 {
let mut __v5: *mut list_head;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_17() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v5 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*head).prev) as *const *mut list_head) };
next == __v5
} else {
false
};
__v4 as i32
}

fn list_is_singular(head: *const list_head) -> i32 {
(list_empty(head) == 0 && (unsafe { (*head).next }) == unsafe { (*head).prev }) as i32
}

fn __list_splice(list: *const list_head, prev: *mut list_head, next: *mut list_head) {
let first: *mut list_head = unsafe { (*list).next };
let last: *mut list_head = unsafe { (*list).prev };
unsafe {
(*first).prev = prev;
}
unsafe {
(*prev).next = first;
}
unsafe {
(*last).next = next;
}
unsafe {
(*next).prev = last;
}
}

fn list_splice(list: *const list_head, head: *mut list_head) {
if list_empty(list) == 0 {
__list_splice(list, head, unsafe { (*head).next });
}
}

fn list_splice_init(list: *mut list_head, head: *mut list_head) {
if list_empty(list as *const list_head) == 0 {
__list_splice(list as *const list_head, head, unsafe { (*head).next });
INIT_LIST_HEAD(list);
}
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
}


fn list_splice_tail_init(list: *mut list_head, head: *mut list_head) {
if list_empty(list as *const list_head) == 0 {
__list_splice(list as *const list_head, unsafe { (*head).prev }, head);
INIT_LIST_HEAD(list);
}
}

// list_for_each	-	iterate over a list
// @pos:	the &struct list_head to use as a loop cursor.
// @head:	the head for your list.
// list_for_each_entry	-	iterate over list of given type
// @pos:	the type * to use as a loop cursor.
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
// list_for_each_entry_safe - iterate over list of given type safe against removal of list entry
// @pos:	the type * to use as a loop cursor.
// @n:		another type * to use as temporary storage
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
fn INIT_HLIST_NODE(h: *mut hlist_node) {
unsafe {
(*h).next = core::ptr::null_mut::<hlist_node>();
}
unsafe {
(*h).pprev = core::ptr::null_mut::<*mut hlist_node>();
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
}


fn hlist_unhashed(h: *const hlist_node) -> i32 {
unsafe { (*h).pprev }.is_null() as i32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct futex_pi_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct robust_list_head {
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct cacheline_padding {
    x: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __kernel_timespec {
    tv_sec: i64,
    tv_nsec: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_private_hash {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct timespec64 {
    tv_sec: i64,
    tv_nsec: i64,
}


// alpha modules require percpu variables to be defined as
// weak to force the compiler to generate GOT based external
// references for them.  This is necessary because percpu sections
// will be located outside of the usually addressable area.
//
// This definition puts the following two extra restrictions when
// defining percpu variables.
//
// 1. The symbol must be globally unique, even the static ones.
// 2. Static percpu variables cannot be defined inside a function.
//
// Archs which need weak percpu definitions should set
// CONFIG_ARCH_MODULE_NEEDS_WEAK_PER_CPU when necessary.
//
// Normal declaration and definition macros.
// Variant on the per-CPU variable declaration/definition theme used for
// ordinary per-CPU variables.
// Declaration/definition used for per-CPU variables that are frequently
// accessed and should be in a single cacheline.
//
// For use only by architecture and core code.  Only use scalar or pointer
// types to maximize density.
// Declaration/definition used for per-CPU variables that must be cacheline
// aligned under SMP conditions so that, whilst a particular instance of the
// data corresponds to a particular CPU, inefficiencies due to direct access by
// other CPUs are reduced by preventing the data from unnecessarily spanning
// cachelines.
//
// An example of this would be statistical data, where each CPU's set of data
// is updated by that CPU alone, but the data from across all CPUs is collated
// by a CPU processing a read from a proc file.
// Declaration/definition used for per-CPU variables that must be page aligned.
// Declaration/definition used for per-CPU variables that must be read mostly.
// Add an offset to a pointer.  Use RELOC_HIDE() to prevent the compiler
// from making incorrect assumptions about the pointer value.
#[inline(always)]
fn __this_cpu_preempt_check(op: *const u8) {
}

// Operations for contexts where we do not want to do any checks for
// preemptions.  Unless strictly necessary, always use [__]this_cpu_*()
// instead.
//
// If there is no other protection through preempt disable and/or disabling
// interrupts then one of these RMW operations can show unexpected behavior
// because the execution thread was rescheduled on another processor or an
// interrupt occurred and the same percpu variable was modified from the
// interrupt context.
// Operations for contexts that are safe from preemption/interrupts.  These
// operations verify that preemption is disabled.
// Operations with implied preemption/interrupt protection.  These
// operations can be used without worrying about preemption or interrupt.
#[inline(always)]
fn get_current() -> *mut task_struct {
if IS_ENABLED!() != 0 {
let mut __v0: *mut task_struct;
loop {
if true {
unsafe { __compiletime_assert_34() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = core::ptr::null_mut::<task_struct>();
return __v0;
}
let mut __v2: *mut task_struct;
let mut pscr_ret__: *mut task_struct = unsafe { core::mem::zeroed() };
__verify_pcpu_ptr!();
match (const { core::mem::size_of::<*mut task_struct>() }) as u64 {
1 => {
let mut __v4: *mut task_struct;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym current_task, options(pure, nomem, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v4 = (pfo_val__ as u64) as *mut task_struct;
pscr_ret__ = __v4;
}
2 => {
let mut __v6: *mut task_struct;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym current_task, options(pure, nomem, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v6 = (pfo_val__ as u64) as *mut task_struct;
pscr_ret__ = __v6;
}
4 => {
let mut __v7: *mut task_struct;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym current_task, options(pure, nomem, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v7 = (pfo_val__ as u64) as *mut task_struct;
pscr_ret__ = __v7;
}
8 => {
let mut __v8: *mut task_struct;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym current_task, options(pure, nomem, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v8 = pfo_val__ as *mut task_struct;
pscr_ret__ = __v8;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v2 = pscr_ret__;
__v2
}

#[repr(C, align(4))]
#[derive(Clone, Copy)]
struct leaf_0x0_0 {
    __slate_bits_0: __slate_bits::__SlateBits102U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sem_undo_list {
}


#[repr(C, align(4))]
#[derive(Clone, Copy)]
struct leaf_0x1_0 {
    __slate_bits_0: __slate_bits::__SlateBits103U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct leaf_parse_info {
    nr_entries: u32,
}


// __CPUID_LEAF() - Define a CPUID output and parse info entry
// @_name:	Struct type name of the CPUID leaf/subleaf (e.g. 'leaf_0x7_0'). Such
// 		types are defined at <cpuid/leaf_types.h> and follow the leaf_0xM_N
// 		format, where 0xM is the leaf and N is the subleaf.
// @_count:	Number of storage entries to allocate for this leaf/subleaf.
//
// For a given leaf/subleaf, define an array of CPUID storage entries and an associated
// query info structure.
//
// Use an array of storage entries to accommodate CPUID leaves with multiple subleaves
// having the same output format.  This is common for hierarchical enumeration; e.g.,
// CPUID(0x4), CPUID(0x12), and CPUID(0x8000001d).
// CPUID_LEAF() - Define a 'struct cpuid_leaves' storage entry
// @_leaf:	Leaf number, in compile-time 0xN format
// @_subleaf:	Subleaf number, in compile-time decimal format
//
// Convenience wrapper around __CPUID_LEAF().
#[repr(C)]
#[derive(Clone, Copy)]
struct cpuid_leaves {
    leaf_0x0_0: [leaf_0x0_0; 1],
    leaf_0x0_0_info: leaf_parse_info,
    leaf_0x1_0: [leaf_0x1_0; 1],
    leaf_0x1_0_info: leaf_parse_info,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cpuid_table {
    leaves: cpuid_leaves,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct range {
    start: u64,
    end: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_64_types_h_21 {
    pte: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_64_types_h_22 {
    pmd: u64,
}


// was written to (raised by CPU)
// on 4KB pages
// On 2MB or 1GB pages
// Protection Keys, bit 1/4
// Saved Dirty bit (leaf)
// The hardware requires shadow stack to be Write=0,Dirty=1. However,
// there are valid cases where the kernel might create read-only PTEs that
// are dirty (e.g., fork(), mprotect(), userfaultfd, soft-dirty tracking). In
// this case, the _PAGE_SAVED_DIRTY bit is used instead of the HW-dirty bit,
// to avoid creating a wrong "shadow stack" PTEs. Such PTEs have
// (Write=0,SavedDirty=1,Dirty=0) set.
// 4th level page in 5-level paging case
// 3rd level page
// PMD_SHIFT determines the size of the area a middle-level
// page table can map
// entries per page directory level
// We borrow bit 3 to remember PG_anon_exclusive.
// Extracts the PFN from a (pte|pmd|pud|pgd)val_t of a 4KB page
// Extracts the flags from a (pte|pmd|pud|pgd)val_t
// This includes the protection key value.
#[repr(C)]
#[derive(Clone, Copy)]
struct pgprot {
    pgprot: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_types_h_295 {
    pgd: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer_clock_base {
}


// No need to mask any bits for !PAE
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_types_h_368 {
    pud: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits127U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seccomp_filter {
}


#[inline(always)]
fn ERR_PTR(error: i64) -> *mut () {
error as *mut ()
}

#[inline(always)]
fn IS_ERR(ptr: *const ()) -> bool {
IS_ERR_VALUE!(ptr as u64) != 0
}

#[inline(always)]
fn ERR_CAST(ptr: *const ()) -> *mut () {
// cast away the const
ptr as *mut ()
}

#[repr(C)]
#[derive(Clone, Copy)]
struct static_key {
    enabled: __SlateRecord_types_h_190,
    __slate_anon_1: __SlateRecord_jump_label_h_95,
}


#[inline(always)]
fn arch_static_branch__const1(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_block_rq_requeue, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
}

#[inline(always)]
fn arch_static_branch__const2(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_block_rq_complete, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
}

#[inline(always)]
fn arch_static_branch__const3(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_block_rq_error, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_39 {
    _pid: i32,
    _uid: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_45 {
    _tid: i32,
    _overrun: i32,
    _sigval: sigval,
    _sys_private: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_53 {
    _pid: i32,
    _uid: u32,
    _sigval: sigval,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_60 {
    _pid: i32,
    _uid: u32,
    _status: i32,
    _utime: i64,
    _stime: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_69 {
    _addr: *mut (),
    __slate_anon_1: __SlateRecord_siginfo_h_74,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_siginfo_h_74 {
    _trapno: i32,
    _addr_lsb: i16,
    _addr_bnd: __SlateRecord_siginfo_h_83,
    _addr_pkey: __SlateRecord_siginfo_h_89,
    _perf: __SlateRecord_siginfo_h_94,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_83 {
    _dummy_bnd: [u8; 8],
    _lower: *mut (),
    _upper: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_89 {
    _dummy_pkey: [u8; 8],
    _pkey: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_94 {
    _data: u64,
    _type: u32,
    _flags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_103 {
    _band: i64,
    _fd: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siginfo_h_109 {
    _call_addr: *mut (),
    _syscall: i32,
    _arch: u32,
}


#[inline(always)]
fn arch_static_branch__const4(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_block_rq_insert, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_signal_types_h_14 {
    si_signo: i32,
    si_errno: i32,
    si_code: i32,
    _sifields: __sifields,
}


#[inline(always)]
fn arch_static_branch__const5(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_block_rq_issue, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
}

#[inline(always)]
fn arch_static_branch__const6(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_block_io_start, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
}

#[inline(always)]
fn arch_static_branch__const7(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_block_io_done, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
}

#[inline(always)]
fn arch_static_branch__const8(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_block_getrq, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
}

#[inline(always)]
fn arch_static_branch__const9(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_block_plug, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
}

#[inline(always)]
fn arch_static_branch__const10(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_block_unplug, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
}

#[inline(always)]
fn arch_static_branch__const11(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0} + {1} + 2 - . \n\t.popsection \n\t", sym force_irqthreads_key, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct jump_entry {
    code: i32,
    target: i32,
    key: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct static_key_false {
    key: static_key,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct orc_entry {
    sp_offset: i16,
    bp_offset: i16,
    __slate_bits_0: __slate_bits::__SlateBits134U0,
}


#[inline(always)]
fn native_save_fl() -> u64 {
let mut flags: u64 = 0;
// "=rm" is safe here, because "pop" adjusts the stack before
// it evaluates its effective address -- this is part of the
// documented behavior of the "pop" instruction.
unsafe {
let __t0: u64;
core::arch::asm!("# __raw_save_flags\n\tpushf ; pop {0:r}", lateout(reg) __t0, options(nostack, att_syntax));
flags = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
// no input
flags
}

#[inline(always)]
fn arch_atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
// Note for KASAN: we deliberately don't use READ_ONCE_NOCHECK() here,
// it's non-inlined function that increases binary size and stack usage.
__READ_ONCE!((*v).counter)
}

#[inline(always)]
fn arch_atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
__WRITE_ONCE__2!((*v).counter, i);
}

#[inline(always)]
fn arch_atomic_add(mut i: i32, mut v: *mut __SlateRecord_types_h_190) {
unsafe {
core::arch::asm!("lock addl {1:e}, ({0:r})", in(reg) core::ptr::addr_of_mut!((*v).counter), in(reg) i as u32, options(nostack, att_syntax));
}
}

#[inline(always)]
fn arch_atomic_sub(mut i: i32, mut v: *mut __SlateRecord_types_h_190) {
unsafe {
core::arch::asm!("lock subl {1:e}, ({0:r})", in(reg) core::ptr::addr_of_mut!((*v).counter), in(reg) i as u32, options(nostack, att_syntax));
}
}

#[inline(always)]
fn arch_atomic_inc(mut v: *mut __SlateRecord_types_h_190) {
unsafe {
core::arch::asm!("lock incl ({0:r})", in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct workqueue_struct {
}


#[inline(always)]
fn arch_atomic_dec(mut v: *mut __SlateRecord_types_h_190) {
unsafe {
core::arch::asm!("lock decl ({0:r})", in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
}


#[inline(always)]
fn arch_atomic_dec_and_test(mut v: *mut __SlateRecord_types_h_190) -> bool {
let mut __v0: bool;
let mut c: bool = false;
unsafe {
let __t0: u8;
core::arch::asm!("lock decl ({0:r})\nsete {1}", in(reg) core::ptr::addr_of_mut!((*v).counter), lateout(reg_byte) __t0, options(nostack, att_syntax));
c = (__t0 as u8) != 0;
}
__v0 = c;
__v0
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_rseq_types_h_38 {
    all: u64,
    __slate_anon_1: __SlateRecord_rseq_types_h_40,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_rseq_types_h_40 {
    __slate_anon_0: __SlateRecord_rseq_types_h_41,
    has_rseq: u8,
    __pad: u8,
    __slate_anon_3: __SlateRecord_rseq_types_h_52,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_rseq_types_h_41 {
    events: u32,
    __slate_anon_1: __SlateRecord_rseq_types_h_43,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_rseq_types_h_43 {
    sched_switch: u8,
    ids_changed: u8,
    user_irq: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_rseq_types_h_52 {
    error: u16,
    __slate_anon_1: __SlateRecord_rseq_types_h_54,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_rseq_types_h_54 {
    fatal: u8,
    slowpath: u8,
}


#[inline(always)]
fn arch_atomic_cmpxchg(v: *mut __SlateRecord_types_h_190, mut __old: i32, mut __new: i32) -> i32 {
let mut __v0: i32;
let mut __ret: i32 = 0;
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __ptr: *mut u8 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u8;
unsafe {
let __t0: u32;
core::arch::asm!("lock cmpxchgb {1:e}, ({0:r})", in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t0, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
}
}
2 => {
let mut __ptr: *mut u16 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u16;
unsafe {
let __t1: u32;
core::arch::asm!("lock cmpxchgw {1:e}, ({0:r})", in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t1, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t1 as u32) };
}
}
4 => {
let mut __ptr: *mut u32 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u32;
unsafe {
let __t2: u32;
core::arch::asm!("lock cmpxchgl {1:e}, ({0:r})", in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t2, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t2 as u32) };
}
}
8 => {
let mut __ptr: *mut u64 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u64;
unsafe {
let __t3: u32;
core::arch::asm!("lock cmpxchgq {1:e}, ({0:r})", in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t3, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t3 as u32) };
}
}
_ => {
unsafe { __cmpxchg_wrong_size() };
}
}
__v0 = __ret;
__v0
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_rseq_types_h_74 {
    cpu_cid: u64,
    __slate_anon_1: __SlateRecord_rseq_types_h_76,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_rseq_types_h_76 {
    cpu_id: u32,
    mm_cid: u32,
}


#[inline(always)]
fn arch_atomic64_read(v: *const __SlateRecord_types_h_197) -> i64 {
__READ_ONCE__2!((*v).counter)
}

#[inline(always)]
fn arch_atomic64_add(mut i: i64, mut v: *mut __SlateRecord_types_h_197) {
unsafe {
core::arch::asm!("lock addq {1:r}, ({0:r})\n# {2}", in(reg) core::ptr::addr_of_mut!((*v).counter), in(reg) i as u64, in(reg) core::ptr::addr_of!((*v).counter), options(nostack, att_syntax));
}
}

#[inline(always)]
fn arch_atomic64_sub_and_test(mut i: i64, mut v: *mut __SlateRecord_types_h_197) -> bool {
let mut __v0: bool;
let mut c: bool = false;
unsafe {
let __t0: u8;
core::arch::asm!("lock subq {2:r}, ({0:r})\nsete {1}", in(reg) core::ptr::addr_of_mut!((*v).counter), lateout(reg_byte) __t0, in(reg) i as u64, options(nostack, att_syntax));
c = (__t0 as u8) != 0;
}
__v0 = c;
__v0
}

#[inline(always)]
fn arch_atomic64_inc(mut v: *mut __SlateRecord_types_h_197) {
unsafe {
core::arch::asm!("lock incq ({0:r})\n# {1}", in(reg) core::ptr::addr_of_mut!((*v).counter), in(reg) core::ptr::addr_of!((*v).counter), options(nostack, att_syntax));
}
}

#[inline(always)]
fn arch_atomic64_try_cmpxchg(v: *mut __SlateRecord_types_h_197, old: &mut i64, mut __new: i64) -> bool {
let mut __v0: i64;
let mut success: bool = false;
let mut __old: i64 = unsafe { *old };
match (const { core::mem::size_of::<i64>() }) as u64 {
1 => {
let mut __ptr: *mut u8 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u8;
unsafe {
let __t0: u8;
let __t1: u64;
core::arch::asm!("lock cmpxchgb {2:r}, ({1:r})\nsetz {0}", lateout(reg_byte) __t0, in(reg) __ptr, in(reg) __new as u64, inlateout("ax") __old as u64 => __t1, options(nostack, att_syntax));
success = (__t0 as u8) != 0;
__old = unsafe { core::mem::transmute::<u64, i64>(__t1 as u64) };
}
}
2 => {
let mut __ptr: *mut u16 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u16;
unsafe {
let __t2: u8;
let __t3: u64;
core::arch::asm!("lock cmpxchgw {2:r}, ({1:r})\nsetz {0}", lateout(reg_byte) __t2, in(reg) __ptr, in(reg) __new as u64, inlateout("ax") __old as u64 => __t3, options(nostack, att_syntax));
success = (__t2 as u8) != 0;
__old = unsafe { core::mem::transmute::<u64, i64>(__t3 as u64) };
}
}
4 => {
let mut __ptr: *mut u32 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u32;
unsafe {
let __t4: u8;
let __t5: u64;
core::arch::asm!("lock cmpxchgl {2:r}, ({1:r})\nsetz {0}", lateout(reg_byte) __t4, in(reg) __ptr, in(reg) __new as u64, inlateout("ax") __old as u64 => __t5, options(nostack, att_syntax));
success = (__t4 as u8) != 0;
__old = unsafe { core::mem::transmute::<u64, i64>(__t5 as u64) };
}
}
8 => {
let mut __ptr: *mut u64 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u64;
unsafe {
let __t6: u8;
let __t7: u64;
core::arch::asm!("lock cmpxchgq {2:r}, ({1:r})\nsetz {0}", lateout(reg_byte) __t6, in(reg) __ptr, in(reg) __new as u64, inlateout("ax") __old as u64 => __t7, options(nostack, att_syntax));
success = (__t6 as u8) != 0;
__old = unsafe { core::mem::transmute::<u64, i64>(__t7 as u64) };
}
}
_ => {
unsafe { __cmpxchg_wrong_size() };
}
}
if unlikely__2!(!success) != 0 {
unsafe {
*old = __old;
}
}
__v0 = success as i64;
__v0 != 0
}

#[inline(always)]
fn raw_atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
arch_atomic_read(v)
}

#[inline(always)]
fn raw_atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
arch_atomic_set(v, i);
}

#[inline(always)]
fn raw_atomic_add(i: i32, v: *mut __SlateRecord_types_h_190) {
arch_atomic_add(i, v);
}

#[inline(always)]
fn raw_atomic_sub(i: i32, v: *mut __SlateRecord_types_h_190) {
arch_atomic_sub(i, v);
}

#[inline(always)]
fn raw_atomic_inc(v: *mut __SlateRecord_types_h_190) {
arch_atomic_inc(v);
}

#[inline(always)]
fn raw_atomic_dec(v: *mut __SlateRecord_types_h_190) {
arch_atomic_dec(v);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


#[inline(always)]
fn raw_atomic_cmpxchg(v: *mut __SlateRecord_types_h_190, old: i32, new: i32) -> i32 {
arch_atomic_cmpxchg(v, old, new)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct audit_context {
}


#[inline(always)]
fn raw_atomic_dec_and_test(v: *mut __SlateRecord_types_h_190) -> bool {
arch_atomic_dec_and_test(v)
}

#[inline(always)]
fn raw_atomic64_read(v: *const __SlateRecord_types_h_197) -> i64 {
arch_atomic64_read(v)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_net_context {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct capture_control {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cfs_rq {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_struct {
}


#[inline(always)]
fn raw_atomic64_add(i: i64, v: *mut __SlateRecord_types_h_197) {
arch_atomic64_add(i, v);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_task {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mempolicy {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct nameidata {
}


#[inline(always)]
fn raw_atomic64_inc(v: *mut __SlateRecord_types_h_197) {
arch_atomic64_inc(v);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event_context {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_ctx_data {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pipe_inode_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rcu_node {
}


#[inline(always)]
fn raw_atomic64_try_cmpxchg(v: *mut __SlateRecord_types_h_197, old: &mut i64, new: i64) -> bool {
arch_atomic64_try_cmpxchg(v, old, new)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rq {
}


#[inline(always)]
fn raw_atomic64_sub_and_test(i: i64, v: *mut __SlateRecord_types_h_197) -> bool {
arch_atomic64_sub_and_test(i, v)
}

#[inline(always)]
fn raw_atomic64_fetch_add_unless(v: *mut __SlateRecord_types_h_197, a: i64, u: i64) -> i64 {
let mut c: i64 = raw_atomic64_read(v as *const __SlateRecord_types_h_197);
while unlikely__2!(c == u) == 0 {
if !(!raw_atomic64_try_cmpxchg(v, unsafe { &mut (*core::ptr::addr_of_mut!(c)) }, c.wrapping_add(a))) {
break;
}
}
c
}

#[inline(always)]
fn raw_atomic64_add_unless(v: *mut __SlateRecord_types_h_197, a: i64, u: i64) -> bool {
raw_atomic64_fetch_add_unless(v, a, u) != u
}

#[inline(always)]
fn raw_atomic64_inc_not_zero(v: *mut __SlateRecord_types_h_197) -> bool {
raw_atomic64_add_unless(v, 1i64, 0i64)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct task_delay_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct task_exec_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct task_group {
}


#[inline(always)]
fn raw_atomic_long_add(i: i64, v: *mut __SlateRecord_types_h_197) {
raw_atomic64_add(i, v);
}

#[inline(always)]
fn raw_atomic_long_inc(v: *mut __SlateRecord_types_h_197) {
raw_atomic64_inc(v);
}

#[inline(always)]
fn raw_atomic_long_sub_and_test(i: i64, v: *mut __SlateRecord_types_h_197) -> bool {
raw_atomic64_sub_and_test(i, v)
}

#[inline(always)]
fn raw_atomic_long_add_unless(v: *mut __SlateRecord_types_h_197, a: i64, u: i64) -> bool {
raw_atomic64_add_unless(v, a, u)
}

#[inline(always)]
fn raw_atomic_long_inc_not_zero(v: *mut __SlateRecord_types_h_197) -> bool {
raw_atomic64_inc_not_zero(v)
}

#[inline(always)]
fn atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
instrument_atomic_read(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_read(v)
}

#[inline(always)]
fn atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
instrument_atomic_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_set(v, i);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rq_flags {
}


#[inline(always)]
fn atomic_add(i: i32, v: *mut __SlateRecord_types_h_190) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_add(i, v);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
}


#[inline(always)]
fn atomic_sub(i: i32, v: *mut __SlateRecord_types_h_190) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_sub(i, v);
}

#[inline(always)]
fn atomic_inc(v: *mut __SlateRecord_types_h_190) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_inc(v);
}

#[inline(always)]
fn atomic_dec(v: *mut __SlateRecord_types_h_190) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_dec(v);
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_sched_h_826 {
    ipi_mask_ptr: *mut cpumask,
    ipi_mask_val: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sched_class {
}


#[inline(always)]
fn atomic_cmpxchg(v: *mut __SlateRecord_types_h_190, old: i32, new: i32) -> i32 {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_cmpxchg(v, old, new)
}

#[inline(always)]
fn atomic_dec_and_test(v: *mut __SlateRecord_types_h_190) -> bool {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_dec_and_test(v)
}

#[inline(always)]
fn atomic_long_add(i: i64, v: *mut __SlateRecord_types_h_197) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_add(i, v);
}

#[inline(always)]
fn atomic_long_inc(v: *mut __SlateRecord_types_h_197) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_inc(v);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct files_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_restriction {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rt_mutex_waiter {
}


#[inline(always)]
fn atomic_long_sub_and_test(i: i64, v: *mut __SlateRecord_types_h_197) -> bool {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_sub_and_test(i, v)
}

#[inline(always)]
fn atomic_long_add_unless(v: *mut __SlateRecord_types_h_197, a: i64, u: i64) -> bool {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_add_unless(v, a, u)
}

#[inline(always)]
fn atomic_long_inc_not_zero(v: *mut __SlateRecord_types_h_197) -> bool {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_inc_not_zero(v)
}

#[inline(always)]
fn find_next_bit(addr: *const u64, size: u64, offset: u64) -> u64 {
if small_const_nbits!(size) {
let mut val: u64 = 0;
if unlikely__2!(offset >= size) != 0 {
return size;
}
val = (unsafe { *addr }) & GENMASK!(size.wrapping_sub(1), offset);
return if val != 0 { __ffs!(val) } else { size };
}
unsafe { _find_next_bit(addr, size, offset) }
}

#[inline(always)]
fn find_next_and_bit(addr1: *const u64, addr2: *const u64, size: u64, offset: u64) -> u64 {
if small_const_nbits!(size) {
let mut val: u64 = 0;
if unlikely__2!(offset >= size) != 0 {
return size;
}
val = (unsafe { *addr1 }) & unsafe { *addr2 } & GENMASK__2!(size.wrapping_sub(1), offset);
return if val != 0 { __ffs!(val) } else { size };
}
unsafe { _find_next_and_bit(addr1, addr2, size, offset) }
}

#[inline(always)]
fn find_first_bit(addr: *const u64, size: u64) -> u64 {
if small_const_nbits!(size) {
let val: u64 = (unsafe { *addr }) & GENMASK__3!(size.wrapping_sub(1), 0i32);
return if val != 0 { __ffs!(val) } else { size };
}
unsafe { _find_first_bit(addr, size) }
}

#[inline(always)]
fn find_first_and_bit(addr1: *const u64, addr2: *const u64, size: u64) -> u64 {
if small_const_nbits!(size) {
let val: u64 = (unsafe { *addr1 }) & unsafe { *addr2 } & GENMASK__4!(size.wrapping_sub(1), 0i32);
return if val != 0 { __ffs!(val) } else { size };
}
unsafe { _find_first_and_bit(addr1, addr2, size) }
}

#[inline(always)]
fn bitmap_zero(dst: *mut u64, nbits: u32) {
let len: u32 = bitmap_size!(nbits);
if small_const_nbits__2!(nbits) {
unsafe {
*dst = 0;
}
} else {
unsafe { memset(dst as *mut (), 0i32, len as u64) };
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct cpumask {
    bits: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_nodemask_types_h_17 {
    bits: [u64; 1],
}


#[inline(always)]
fn cpu_max_bits_warn(cpu: u32, bits: u32) {
}

#[inline(always)]
fn cpumask_check(cpu: u32) -> u32 {
cpu_max_bits_warn(cpu, 64u32);
cpu
}

#[inline(always)]
fn cpumask_first(srcp: *const cpumask) -> u32 {
find_first_bit(cpumask_bits!(srcp), 64u64) as u32
}

#[inline(always)]
fn cpumask_first_and(srcp1: *const cpumask, srcp2: *const cpumask) -> u32 {
find_first_and_bit(cpumask_bits!(srcp1), cpumask_bits!(srcp2), 64u64) as u32
}

#[inline(always)]
fn cpumask_next_and(n: i32, src1p: *const cpumask, src2p: *const cpumask) -> u32 {
// -1 is a legal arg here.
if n != 1i32.wrapping_neg() {
cpumask_check(n as u32);
}
find_next_and_bit(cpumask_bits!(src1p), cpumask_bits!(src2p), 64u64, n.wrapping_add(1) as u64) as u32
}

// for_each_cpu - iterate over every cpu in a mask
// @cpu: the (optionally unsigned) integer iterator
// @mask: the cpumask pointer
//
// After the loop, cpu is >= nr_cpu_ids.
#[inline(always)]
fn cpumask_set_cpu(cpu: u32, dstp: *mut cpumask) {
set_bit(cpumask_check(cpu) as i64, cpumask_bits__2!(dstp) as *mut u64);
}

#[inline(always)]
fn cpumask_clear_cpu(cpu: i32, dstp: *mut cpumask) {
clear_bit(cpumask_check(cpu as u32) as i64, cpumask_bits__2!(dstp) as *mut u64);
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[inline(always)]
fn cpumask_test_cpu(cpu: i32, cpumask: *const cpumask) -> bool {
test_bit__2!(cpumask_check(cpu as u32), unsafe { (*cpumask).bits.as_ptr() as *const u64 }) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
}


#[inline(always)]
fn cpumask_clear(dstp: *mut cpumask) {
bitmap_zero(cpumask_bits__2!(dstp), 64u32);
}

#[inline(always)]
fn zalloc_cpumask_var_node(mask: *mut [cpumask; 1], flags: u32, node: i32) -> bool {
cpumask_clear(unsafe { (*mask).as_mut_ptr() as *mut cpumask });
true
}

#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe {
}


#[inline(always)]
fn free_cpumask_var(mask: *mut cpumask) {
}

// to_cpumask - convert a NR_CPUS bitmap to a struct cpumask *
// @bitmap: the bitmap
//
// There are a few places where cpumask_var_t isn't appropriate and
// static cpumasks must be used (eg. very early boot), yet we don't
// expose the definition of 'struct cpumask'.
//
// This does the conversion, and can be used as a constant initializer.
#[inline(always)]
fn cpu_online(cpu: u32) -> bool {
cpumask_test_cpu(cpu as i32, (unsafe { core::ptr::addr_of_mut!(__cpu_online_mask) }) as *const cpumask)
}

#[inline(always)]
fn arch_local_save_flags() -> u64 {
native_save_fl()
}

#[inline(always)]
fn arch_irqs_disabled_flags(flags: u64) -> i32 {
(flags & _BITUL!(9i32) == 0) as i32
}

#[inline(always)]
fn native_pause() {
unsafe {
core::arch::asm!("pause", options(nostack, att_syntax));
}
}

#[inline(always)]
fn cpu_relax() {
native_pause();
}

#[repr(C)]
#[derive(Clone, Copy)]
struct cpuinfo_topology {
    apicid: u32,
    initial_apicid: u32,
    pkg_id: u32,
    die_id: u32,
    cu_id: u32,
    core_id: u32,
    logical_pkg_id: u32,
    logical_die_id: u32,
    logical_core_id: u32,
    amd_node_id: u32,
    llc_id: u32,
    l2c_id: u32,
    __slate_anon_12: __SlateRecord_processor_h_111,
    cpu_type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cpuinfo_x86 {
    __slate_anon_0: __SlateRecord_processor_h_136,
    x86_stepping: u8,
    __slate_anon_2: __SlateRecord_processor_h_154,
    x86_tlbsize: i32,
    vmx_capability: [u32; 5],
    x86_virt_bits: u8,
    x86_phys_bits: u8,
    extended_cpuid_level: u32,
    cpuid_level: i32,
    __slate_anon_9: __SlateRecord_processor_h_177,
    x86_vendor_id: [u8; 16],
    x86_model_id: [u8; 64],
    topo: cpuinfo_topology,
    cpuid: cpuid_table,
    x86_cache_size: u32,
    x86_cache_alignment: i32,
    x86_cache_max_rmid: i32,
    x86_cache_occ_scale: i32,
    x86_cache_mbm_width_offset: i32,
    x86_power: i32,
    loops_per_jiffy: u64,
    ppin: u64,
    x86_clflush_size: u16,
    booted_cores: u16,
    cpu_index: u16,
    smt_active: bool,
    microcode: u32,
    x86_cache_bits: u8,
    __slate_bits_0: __slate_bits::__SlateBits101U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct thread_struct {
    tls_array: [desc_struct; 3],
    sp: u64,
    es: u16,
    ds: u16,
    fsindex: u16,
    gsindex: u16,
    fsbase: u64,
    gsbase: u64,
    ptrace_bps: [*mut perf_event; 4],
    virtual_dr6: u64,
    ptrace_dr7: u64,
    cr2: u64,
    trap_nr: u64,
    error_code: u64,
    io_bitmap: *mut io_bitmap,
    iopl_emul: u64,
    __slate_bits_0: __slate_bits::__SlateBits135U0,
    pkru: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe_xol_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct codetag {
    flags: u32,
    lineno: u32,
    modname: *const u8,
    function: *const u8,
    filename: *const u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_uprobes_h_31 {
    insn: [u8; 16],
    ixol: [u8; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_uprobes_h_38 {
    branch: __SlateRecord_uprobes_h_39,
    defparam: __SlateRecord_uprobes_h_44,
    push: __SlateRecord_uprobes_h_48,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uprobes_h_39 {
    offs: i32,
    ilen: u8,
    opc1: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uprobes_h_44 {
    fixups: u8,
    ilen: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uprobes_h_48 {
    reg_offset: u8,
    ilen: u8,
}


// We use the MSB for PREEMPT_NEED_RESCHED mostly because it is available.
#[inline(always)]
fn preempt_count() -> i32 {
let mut __v0: u64;
let mut pfo_val__: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t0, sym __preempt_count, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
__v0 = pfo_val__;
(__v0 & !(!((1i64.wrapping_neg() as u64) >> 1i32))) as i32
}

#[inline(always)]
fn __preempt_count_add(val: i32) {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_a39e427d3b6c2df4(val) }) && (val == 1 || val == 1i32.wrapping_neg()) { val } else { 0 };
if false {
let pao_tmp__: u64 = val as u64;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incq %gs:{0}(%rip)", sym __preempt_count, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decq %gs:{0}(%rip)", sym __preempt_count, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u64 = val as u64;
if false {
let pto_tmp__: u64 = val as u64;
}
unsafe {
core::arch::asm!("addq {1:r}, %gs:{0}(%rip)", sym __preempt_count, in(reg) pto_val__ as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}

#[inline(always)]
fn __preempt_count_dec_and_test() -> bool {
let mut __v0: bool;
let mut c: bool = false;
unsafe {
let __t0: u8;
core::arch::asm!("decq %gs:{0}(%rip)\nsete {1}", sym __preempt_count, lateout(reg_byte) __t0, options(nostack, att_syntax));
c = (__t0 as u8) != 0;
}
__v0 = c;
__v0
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_uprobes_h_136 {
    __slate_anon_0: __SlateRecord_uprobes_h_137,
    __slate_anon_1: __SlateRecord_uprobes_h_142,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uprobes_h_137 {
    autask: arch_uprobe_task,
    vaddr: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uprobes_h_142 {
    dup_xol_work: callback_head,
    dup_xol_addr: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct llist_head {
    first: *mut llist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct llist_node {
    next: *mut llist_node,
}


fn init_llist_head(list: *mut llist_head) {
unsafe {
(*list).first = core::ptr::null_mut::<llist_node>();
}
}

// llist_for_each_entry_safe - iterate over some deleted entries of lock-less list of given type
// 			       safe against removal of list entry
// @pos:	the type * to use as a loop cursor.
// @n:		another type * to use as temporary storage
// @node:	the first entry of deleted list entries.
// @member:	the name of the llist_node with the struct.
//
// In general, some entries of the lock-less list can be traversed
// safely only after being removed from list, so start with an entry
// instead of list head.
//
// If being used on entries deleted from lock-less list directly, the
// traverse order is from the newest to the oldest added entry.  If
// you want to traverse from the oldest to the newest, you must
// reverse the order by yourself before traversing.
fn llist_add_batch(new_first: *mut llist_node, new_last: *mut llist_node, head: *mut llist_head) -> bool {
let mut first: *mut llist_node = unsafe { core::mem::zeroed() };
let mut __v10: *mut llist_node;
loop {
if !(__native_word__16!() || ((const { core::mem::size_of::<*mut llist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_63() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*head).first) as *const *mut llist_node) };
first = __v10;
loop {
unsafe {
(*new_last).next = first;
}
let mut __v13: i64;
let __v0: *mut *mut llist_node = unsafe { core::ptr::addr_of_mut!((*head).first) };
let __v1: *mut *mut llist_node = core::ptr::addr_of_mut!(first);
kcsan_mb!();
instrument_atomic_read_write(__v0 as *const (), core::mem::size_of::<*mut llist_node>() as u64);
instrument_read_write(__v1 as *const (), core::mem::size_of::<*mut llist_node>() as u64);
let mut __v15: i64;
let mut __v2: bool = false;
let mut __v4: *mut llist_node = unsafe { *__v1 };
let mut __v5: *mut llist_node = new_first;
match (const { core::mem::size_of::<*mut llist_node>() }) as u64 {
1 => {
let mut __v6: *mut u8 = __v0 as *mut u8;
unsafe {
let __t0: u8;
let __t1: u64;
core::arch::asm!("lock cmpxchgb {2:r}, ({1:r})\nsetz {0}", lateout(reg_byte) __t0, in(reg) __v6, in(reg) __v5 as u64, inlateout("ax") __v4 as u64 => __t1, options(nostack, att_syntax));
__v2 = (__t0 as u8) != 0;
__v4 = unsafe { core::mem::transmute::<u64, *mut llist_node>(__t1 as u64) };
}
}
2 => {
let mut __v7: *mut u16 = __v0 as *mut u16;
unsafe {
let __t2: u8;
let __t3: u64;
core::arch::asm!("lock cmpxchgw {2:r}, ({1:r})\nsetz {0}", lateout(reg_byte) __t2, in(reg) __v7, in(reg) __v5 as u64, inlateout("ax") __v4 as u64 => __t3, options(nostack, att_syntax));
__v2 = (__t2 as u8) != 0;
__v4 = unsafe { core::mem::transmute::<u64, *mut llist_node>(__t3 as u64) };
}
}
4 => {
let mut __v8: *mut u32 = __v0 as *mut u32;
unsafe {
let __t4: u8;
let __t5: u64;
core::arch::asm!("lock cmpxchgl {2:r}, ({1:r})\nsetz {0}", lateout(reg_byte) __t4, in(reg) __v8, in(reg) __v5 as u64, inlateout("ax") __v4 as u64 => __t5, options(nostack, att_syntax));
__v2 = (__t4 as u8) != 0;
__v4 = unsafe { core::mem::transmute::<u64, *mut llist_node>(__t5 as u64) };
}
}
8 => {
let mut __v9: *mut u64 = __v0 as *mut u64;
unsafe {
let __t6: u8;
let __t7: u64;
core::arch::asm!("lock cmpxchgq {2:r}, ({1:r})\nsetz {0}", lateout(reg_byte) __t6, in(reg) __v9, in(reg) __v5 as u64, inlateout("ax") __v4 as u64 => __t7, options(nostack, att_syntax));
__v2 = (__t6 as u8) != 0;
__v4 = unsafe { core::mem::transmute::<u64, *mut llist_node>(__t7 as u64) };
}
}
_ => {
unsafe { __cmpxchg_wrong_size() };
}
}
if unlikely__2!(!__v2) != 0 {
unsafe {
*__v1 = __v4;
}
}
__v15 = __v2 as i64;
__v13 = __v15;
if __v13 != 0 {
break;
}
}
first.is_null()
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ldt_struct {
}


fn llist_add(new: *mut llist_node, head: *mut llist_head) -> bool {
llist_add_batch(new, new, head)
}

fn llist_del_all(head: *mut llist_head) -> *mut llist_node {
let mut __v0: *mut llist_node;
let mut __ai_ptr: *mut *mut llist_node = unsafe { core::ptr::addr_of_mut!((*head).first) };
kcsan_mb!();
instrument_atomic_read_write(__ai_ptr as *const (), core::mem::size_of::<*mut llist_node>() as u64);
let mut __v2: *mut llist_node;
let mut __ret: *mut llist_node = core::ptr::null_mut::<llist_node>();
match (const { core::mem::size_of::<*mut llist_node>() }) as u64 {
1 => {
unsafe {
let __t0: u64;
core::arch::asm!("xchgb {0:l}, ({1:r})", inlateout(reg) __ret as u64 => __t0, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, *mut llist_node>(__t0 as u64) };
}
}
2 => {
unsafe {
let __t1: u64;
core::arch::asm!("xchgw {0:x}, ({1:r})", inlateout(reg) __ret as u64 => __t1, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, *mut llist_node>(__t1 as u64) };
}
}
4 => {
unsafe {
let __t2: u64;
core::arch::asm!("xchgl {0:r}, ({1:r})", inlateout(reg) __ret as u64 => __t2, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, *mut llist_node>(__t2 as u64) };
}
}
8 => {
unsafe {
let __t3: u64;
core::arch::asm!("xchgq {0:r}, ({1:r})", inlateout(reg) __ret as u64 => __t3, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, *mut llist_node>(__t3 as u64) };
}
}
_ => {
unsafe { __xchg_wrong_size() };
}
}
__v2 = __ret;
__v0 = __v2;
__v0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mem_cgroup {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __call_single_node {
    llist: llist_node,
    __slate_anon_1: __SlateRecord_smp_types_h_60,
    src: u16,
    dst: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_89 {
    __slate_anon_0: __SlateRecord_mm_types_h_90,
    __slate_anon_1: __SlateRecord_mm_types_h_118,
    __slate_anon_2: __SlateRecord_mm_types_h_129,
    __slate_anon_3: __SlateRecord_mm_types_h_132,
    callback_head: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_90 {
    __slate_anon_0: __SlateRecord_mm_types_h_96,
    mapping: *mut address_space,
    __slate_anon_2: __SlateRecord_mm_types_h_105,
    private: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_96 {
    lru: list_head,
    buddy_list: list_head,
    pcp_list: list_head,
    pcp_llist: llist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_105 {
    __folio_index: u64,
    share: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_118 {
    pp_magic: u64,
    pp: *mut page_pool,
    _pp_mapping_pad: u64,
    dma_addr: u64,
    pp_ref_count: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct page_pool {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_129 {
    compound_info: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_132 {
    _unused_pgmap_compound_info: *mut (),
    zone_device_data: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_155 {
    page_type: u32,
    _mapcount: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __call_single_data {
    node: __call_single_node,
    func: Option<unsafe extern "C" fn(*mut ())>,
    info: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_406 {
    __slate_anon_0: __SlateRecord_mm_types_h_407,
    page: page,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_407 {
    flags: __SlateRecord_mm_types_h_41,
    __slate_anon_1: __SlateRecord_mm_types_h_410,
    mapping: *mut address_space,
    __slate_anon_3: __SlateRecord_mm_types_h_425,
    __slate_anon_4: __SlateRecord_mm_types_h_429,
    _mapcount: __SlateRecord_types_h_190,
    _refcount: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_410 {
    lru: list_head,
    __slate_anon_1: __SlateRecord_mm_types_h_414,
    pgmap: *mut dev_pagemap,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_414 {
    __filler: *mut (),
    mlock_count: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct restart_block {
    arch_data: u64,
    r#fn: Option<unsafe extern "C" fn(*mut restart_block) -> i64>,
    __slate_anon_2: __SlateRecord_restart_block_h_29,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_425 {
    index: u64,
    share: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_429 {
    private: *mut (),
    swap: __SlateRecord_mm_types_h_288,
    migrate_info: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_451 {
    __slate_anon_0: __SlateRecord_mm_types_h_452,
    __page_1: page,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_452 {
    _flags_1: u64,
    _head_1: u64,
    __slate_anon_2: __SlateRecord_mm_types_h_455,
    _mapcount_1: __SlateRecord_types_h_190,
    _refcount_1: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_455 {
    __slate_anon_0: __SlateRecord_mm_types_h_456,
    _usable_1: [u64; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_456 {
    _large_mapcount: __SlateRecord_types_h_190,
    _nr_pages_mapped: __SlateRecord_types_h_190,
    _entire_mapcount: __SlateRecord_types_h_190,
    _pincount: __SlateRecord_types_h_190,
    _mm_id_mapcount: [i32; 2],
    __slate_anon_5: __SlateRecord_mm_types_h_465,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_465 {
    _mm_id: [u32; 2],
    _mm_ids: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_483 {
    __slate_anon_0: __SlateRecord_mm_types_h_484,
    __page_2: page,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_484 {
    _flags_2: u64,
    _head_2: u64,
    _deferred_list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_497 {
    __slate_anon_0: __SlateRecord_mm_types_h_498,
    __page_3: page,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_498 {
    _flags_3: u64,
    _head_3: u64,
    _hugetlb_subpool: *mut (),
    _hugetlb_cgroup: *mut (),
    _hugetlb_cgroup_rsvd: *mut (),
    _hugetlb_hwpoison: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct thread_info {
    flags: u64,
    syscall_work: u64,
    status: u32,
    cpu: u32,
}


// 32-bit address space on 64 bits
fn test_ti_thread_flag(ti: *mut thread_info, flag: i32) -> i32 {
test_bit__3!(flag, unsafe { core::ptr::addr_of_mut!((*ti).flags) })
}

#[inline(always)]
fn tif_test_bit(bit: i32) -> bool {
arch_test_bit(bit as u64, (unsafe { core::ptr::addr_of_mut!((*current_thread_info!()).flags) }) as *const u64)
}

#[inline(always)]
fn tif_need_resched() -> bool {
tif_test_bit(4i32)
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_830 {
    remap: __SlateRecord_mm_types_h_831,
    simple_ioremap: __SlateRecord_mm_types_h_837,
    map_kernel: __SlateRecord_mm_types_h_841,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_831 {
    start: u64,
    start_pfn: u64,
    size: u64,
    pgprot: pgprot,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_837 {
    start_phys_addr: u64,
    size: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_841 {
    start: u64,
    pages: *mut *mut page,
    nr_pages: u64,
    pgoff: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct alloc_tag_counters {
    bytes: u64,
    calls: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct alloc_tag {
    ct: codetag,
    counters: *mut alloc_tag_counters,
}


fn mem_alloc_profiling_enabled() -> bool {
false
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_926 {
    __slate_anon_0: __SlateRecord_mm_types_h_927,
    vm_freeptr: __SlateRecord_mm_types_h_701,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_927 {
    vm_start: u64,
    vm_end: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_947 {
    vm_flags: u64,
    flags: __SlateRecord_mm_types_h_871,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct anon_vma {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_1062 {
    rb: rb_node,
    rb_subtree_last: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct optimistic_spin_queue {
    tail: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kioctx_table {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_mm_data {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_1176 {
    __slate_anon_0: __SlateRecord_mm_types_h_1181,
    mm_mt: maple_tree,
    mmap_base: u64,
    mmap_legacy_base: u64,
    mmap_compat_base: u64,
    mmap_compat_legacy_base: u64,
    task_size: u64,
    pgd: *mut __SlateRecord_pgtable_types_h_295,
    membarrier_state: __SlateRecord_types_h_190,
    mm_users: __SlateRecord_types_h_190,
    mm_cid: mm_mm_cid,
    sched_cache_grp: *mut sched_cache_group,
    pgtables_bytes: __SlateRecord_types_h_197,
    map_count: i32,
    page_table_lock: spinlock,
    mmap_lock: rw_semaphore,
    mmlist: list_head,
    vma_writer_wait: rcuwait,
    mm_lock_seq: seqcount,
    futex: futex_mm_data,
    hiwater_rss: u64,
    hiwater_vm: u64,
    total_vm: u64,
    locked_vm: u64,
    pinned_vm: __SlateRecord_types_h_197,
    data_vm: u64,
    exec_vm: u64,
    stack_vm: u64,
    __slate_anon_28: __SlateRecord_mm_types_h_1289,
    write_protect_seq: seqcount,
    arg_lock: spinlock,
    start_code: u64,
    end_code: u64,
    start_data: u64,
    end_data: u64,
    start_brk: u64,
    brk: u64,
    start_stack: u64,
    arg_start: u64,
    arg_end: u64,
    env_start: u64,
    env_end: u64,
    saved_auxv: [u64; 56],
    rss_stat: [percpu_counter; 4],
    binfmt: *mut linux_binfmt,
    context: __SlateRecord_mmu_h_84,
    flags: __SlateRecord_mm_types_h_1171,
    ioctx_lock: spinlock,
    ioctx_table: *mut kioctx_table,
    exe_file: *mut file,
    notifier_subscriptions: *mut mmu_notifier_subscriptions,
    tlb_flush_pending: __SlateRecord_types_h_190,
    tlb_flush_batched: __SlateRecord_types_h_190,
    uprobes_state: uprobes_state,
    hugetlb_usage: __SlateRecord_types_h_197,
    async_put_work: work_struct,
    iommu_mm: *mut iommu_mm_data,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_1181 {
    mm_count: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_types_h_1289 {
    def_flags: u64,
    def_vma_flags: __SlateRecord_mm_types_h_871,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct linux_binfmt {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mmu_notifier_subscriptions {
}


#[inline(always)]
fn __acquire_ctx_lock__2(var: *const spinlock) {
}

#[inline(always)]
fn __acquire_shared_ctx_lock__2(var: &spinlock) {
}

#[inline(always)]
fn __release_ctx_lock__2(var: &spinlock) {
}

#[inline(always)]
fn __release_shared_ctx_lock__2(var: &spinlock) {
}

#[inline(always)]
fn __assume_ctx_lock__2(var: &spinlock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct spinlock {
    __slate_anon_0: __SlateRecord_spinlock_types_h_18,
}


#[inline(always)]
fn __acquire_ctx_lock__3(var: &rwlock) {
}

#[inline(always)]
fn __acquire_shared_ctx_lock__3(var: &rwlock) {
}

#[inline(always)]
fn __release_ctx_lock__3(var: &rwlock) {
}

#[inline(always)]
fn __release_shared_ctx_lock__3(var: &rwlock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_sysctl_h_254 {
    __slate_anon_0: __SlateRecord_sysctl_h_255,
    rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sysctl_h_255 {
    ctl_table: *const ctl_table,
    ctl_table_size: i32,
    used: i32,
    count: i32,
    nreg: i32,
}


#[inline(always)]
fn __assume_ctx_lock__3(var: &rwlock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rwlock {
    raw_lock: qrwlock,
}


#[inline(always)]
fn __acquire_ctx_lock__4(var: &mutex) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct folio_queue {
}


#[inline(always)]
fn __acquire_shared_ctx_lock__4(var: &mutex) {
}

#[inline(always)]
fn __release_ctx_lock__4(var: &mutex) {
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_uio_h_58 {
    __ubuf_iovec: iovec,
    __slate_anon_1: __SlateRecord_uio_h_65,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uio_h_65 {
    __slate_anon_0: __SlateRecord_uio_h_66,
    count: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_uio_h_66 {
    __iov: *const iovec,
    kvec: *const kvec,
    bvec: *const bio_vec,
    folioq: *const folio_queue,
    xarray: *mut xarray,
    ubuf: *mut (),
}


#[inline(always)]
fn __release_shared_ctx_lock__4(var: &mutex) {
}

#[inline(always)]
fn __assume_ctx_lock__4(var: *const mutex) {
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_uio_h_78 {
    nr_segs: u64,
    folioq_slot: u8,
    xarray_start: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sock {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mutex {
    owner: __SlateRecord_types_h_197,
    wait_lock: raw_spinlock,
    osq: optimistic_spin_queue,
    first_waiter: *mut mutex_waiter,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_sched_data {
    robust_list: *mut robust_list_head,
    compat_robust_list: *mut compat_robust_list_head,
    pi_state_list: list_head,
    pi_state_cache: *mut futex_pi_state,
    exit_mutex: mutex,
    state: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_mm_phash {
    lock: mutex,
    hash: *mut futex_private_hash,
    hash_new: *mut futex_private_hash,
    batches: u64,
    rcu: callback_head,
    atomic: __SlateRecord_types_h_197,
    r#ref: *mut u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_unlock_cs_range {
    start_ip: u64,
    len: u32,
    pop_size32: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_unlock_cs_ranges {
    cs_ranges: [futex_unlock_cs_range; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_mm_data {
    phash: futex_mm_phash,
    unlock: futex_unlock_cs_ranges,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sysv_sem {
    undo_list: *mut sem_undo_list,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sysv_shm {
    shm_clist: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct plist_node {
    prio: i32,
    prio_list: list_head,
    node_list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rb_node {
    __rb_parent_color: u64,
    rb_right: *mut rb_node,
    rb_left: *mut rb_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rb_node_linked {
    node: rb_node,
    prev: *mut rb_node_linked,
    next: *mut rb_node_linked,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rb_root {
    rb_node: *mut rb_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mnt_idmap {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rb_root_cached {
    rb_root: rb_root,
    rb_leftmost: *mut rb_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array_ptr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct net {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct timerqueue_head {
    rb_root: rb_root_cached,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key_type {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct timerqueue_linked_node {
    node: rb_node_linked,
    expires: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer {
    node: timerqueue_linked_node,
    base: *mut hrtimer_clock_base,
    is_queued: bool,
    is_rel: bool,
    is_soft: bool,
    is_hard: bool,
    is_lazy: bool,
    _softexpires: i64,
    function: Option<unsafe extern "C" fn(*mut hrtimer) -> u32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_key_h_117 {
    __slate_anon_0: __SlateRecord_key_h_118,
    x: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_key_h_118 {
    desc_len: u16,
    desc: [u8; 6],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct timer_list {
    entry: hlist_node,
    expires: u64,
    function: Option<unsafe extern "C" fn(*mut timer_list)>,
    flags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seccomp {
    mode: i32,
    filter_count: __SlateRecord_types_h_190,
    filter: *mut seccomp_filter,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_key_h_198 {
    graveyard_link: list_head,
    serial_node: rb_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key_user {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_key_h_208 {
    expiry: i64,
    revoked_at: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_key_h_246 {
    index_key: keyring_index_key,
    __slate_anon_1: __SlateRecord_key_h_248,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_key_h_248 {
    hash: u64,
    len_desc: u64,
    r#type: *mut key_type,
    domain_tag: *mut key_tag,
    description: *mut u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_key_h_261 {
    payload: key_payload,
    __slate_anon_1: __SlateRecord_key_h_263,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_key_h_263 {
    name_link: list_head,
    keys: assoc_array,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct refcount_struct {
    refs: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_cred_h_147 {
    non_rcu: i32,
    rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pidfs_attr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rlimit {
    rlim_cur: u64,
    rlim_max: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pid_h_62 {
    ino: u64,
    pidfs_hash: rhash_head,
    stashed: *mut dentry,
    attr: *mut pidfs_attr,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct task_cputime {
    stime: u64,
    utime: u64,
    sum_exec_runtime: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_signal_h_25 {
    sig: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
union sigval {
    sival_int: i32,
    sival_ptr: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
union __sifields {
    _kill: __SlateRecord_siginfo_h_39,
    _timer: __SlateRecord_siginfo_h_45,
    _rt: __SlateRecord_siginfo_h_53,
    _sigchld: __SlateRecord_siginfo_h_60,
    _sigfault: __SlateRecord_siginfo_h_69,
    _sigpoll: __SlateRecord_siginfo_h_103,
    _sigsys: __SlateRecord_siginfo_h_109,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_siginfo {
    __slate_anon_0: __SlateRecord_signal_types_h_14,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sigpending {
    list: list_head,
    signal: __SlateRecord_signal_h_25,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sigaction {
    sa_handler: Option<unsafe extern "C" fn(i32)>,
    sa_flags: u64,
    sa_restorer: Option<unsafe extern "C" fn()>,
    sa_mask: __SlateRecord_signal_h_25,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_audit_buf {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct k_sigaction {
    sa: sigaction,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tracepoint_func {
    func: *mut (),
    data: *mut (),
    prio: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tracepoint_ext {
    regfunc: Option<unsafe extern "C" fn() -> i32>,
    unregfunc: Option<unsafe extern "C" fn()>,
    __slate_bits_0: __slate_bits::__SlateBits246U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tracepoint {
    name: *const u8,
    key: static_key_false,
    static_call_key: *mut static_call_key,
    static_call_tramp: *mut (),
    iterator: *mut (),
    probestub: *mut (),
    funcs: *mut tracepoint_func,
    ext: *mut tracepoint_ext,
}


// If a tracepoint needs to be called from a header file, it is not
// recommended to call it directly, as tracepoints in header files
// may cause side-effects and bloat the kernel. Instead, use
// tracepoint_enabled() to test if the tracepoint is enabled, then if
// it is, call a wrapper function defined in a C file that will then
// call the tracepoint.
//
// For "trace_foo_bar()", you would need to create a wrapper function
// in a C file to call trace_foo_bar():
//   void do_trace_foo_bar(args) { trace_foo_bar(args); }
// Then in the header file, declare the tracepoint:
//   DECLARE_TRACEPOINT(foo_bar);
// And call your wrapper:
//   static inline void some_inlined_function() {
//            [..]
//            if (tracepoint_enabled(foo_bar))
//                    do_trace_foo_bar(args);
//            [..]
//   }
//
// Note: tracepoint_enabled(foo_bar) is equivalent to trace_foo_bar_enabled()
//   but is safe to have in headers, where trace_foo_bar_enabled() is not.
// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
#[inline(always)]
fn spinlock_check(lock: *mut spinlock) -> *mut raw_spinlock {
unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) }
}

#[inline(always)]
fn spin_lock(lock: *mut spinlock) {
raw_spin_lock!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
}


#[inline(always)]
fn spin_lock_irq(lock: *mut spinlock) {
raw_spin_lock_irq!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[inline(always)]
fn spin_unlock(lock: *mut spinlock) {
raw_spin_unlock!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[inline(always)]
fn spin_unlock_irq(lock: *mut spinlock) {
raw_spin_unlock_irq!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[inline(always)]
fn spin_unlock_irqrestore(lock: *mut spinlock, flags: u64) {
raw_spin_unlock_irqrestore!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) }, flags);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct syscall_user_dispatch {
    selector: *mut u8,
    offset: u64,
    len: u64,
    on_dispatch: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct arch_tlbflush_unmap_batch {
    cpumask: cpumask,
    unmapped_pages: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct page_frag {
    page: *mut page,
    offset: u32,
    size: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tlbflush_unmap_batch {
    arch: arch_tlbflush_unmap_batch,
    flush_required: bool,
    writable: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct task_io_accounting {
    rchar: u64,
    wchar: u64,
    syscr: u64,
    syscw: u64,
    read_bytes: u64,
    write_bytes: u64,
    cancelled_write_bytes: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_cputimer_base {
    nextevt: u64,
    tqhead: timerqueue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_cputimers {
    bases: [posix_cputimer_base; 3],
    timers_active: u32,
    expiry_active: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_cputimers_work {
    work: callback_head,
    mutex: mutex,
    scheduled: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_work {
    node: __call_single_node,
    func: Option<unsafe extern "C" fn(*mut irq_work)>,
    irqwait: rcuwait,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct work_struct {
    data: __SlateRecord_types_h_197,
    entry: list_head,
    func: Option<unsafe extern "C" fn(*mut work_struct)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq_event {
    __slate_anon_0: __SlateRecord_rseq_types_h_38,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct export_operations {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fsnotify_sb_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kstatfs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mount {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mtd_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct super_dev {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq_ids {
    __slate_anon_0: __SlateRecord_rseq_types_h_74,
    node_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xattr_handler {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fserror_event {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq_data {
    usrptr: *mut rseq,
    len: u32,
    sig: u32,
    event: rseq_event,
    ids: rseq_ids,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sched_mm_cid {
    active: u32,
    cid: u32,
    node: hlist_node,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct mm_cid_pcpu {
    cid: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_lockref_h_26 {
    lock_count: u64,
    __slate_anon_1: __SlateRecord_lockref_h_30,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_lockref_h_30 {
    lock: spinlock,
    count: i32,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct mm_mm_cid {
    pcpu: *mut mm_cid_pcpu,
    mode: u32,
    max_cids: u32,
    irq_work: irq_work,
    work: work_struct,
    lock: raw_spinlock,
    mutex: mutex,
    user_list: hlist_head,
    nr_cpus_allowed: u32,
    users: u32,
    pcpu_thrs: u32,
    update_deferred: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqcount {
    sequence: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_dcache_h_50 {
    __slate_anon_0: __SlateRecord_dcache_h_51,
    hash_len: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_dcache_h_51 {
    hash: u32,
    len: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqcount_spinlock {
    seqcount: seqcount,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct completion_list {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_dcache_h_99 {
    __d_name: qstr,
    d_name: qstr,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_dcache_h_126 {
    d_alias: hlist_node,
    d_in_lookup_hash: hlist_bl_node,
    d_rcu: callback_head,
    waiters: *mut completion_list,
}


#[inline(always)]
fn __acquire_ctx_lock__5(var: &seqlock) {
}

#[inline(always)]
fn __acquire_shared_ctx_lock__5(var: &seqlock) {
}

#[inline(always)]
fn __release_ctx_lock__5(var: &seqlock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_iocontext_h_83 {
    q_node: list_head,
    __rcu_icq_cache: *mut kmem_cache,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_iocontext_h_87 {
    ioc_node: hlist_node,
    __rcu_head: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_context {
}


#[inline(always)]
fn __release_shared_ctx_lock__5(var: &seqlock) {
}

#[inline(always)]
fn __assume_ctx_lock__5(var: &seqlock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct seqlock {
    seqcount: seqcount_spinlock,
    lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uidgid_types_h_9 {
    val: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uidgid_types_h_13 {
    val: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct unwind_cache {
    unwind_completed: u64,
    nr_entries: u32,
    entries: [u64; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
union unwind_task_id {
    __slate_anon_0: __SlateRecord_unwind_deferred_types_h_28,
    id: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct unwind_task_info {
    unwind_mask: __SlateRecord_types_h_197,
    cache: *mut unwind_cache,
    work: callback_head,
    id: unwind_task_id,
}


// Convenience macros for the sake of set_current_state:
// get_task_state():
// set_current_state() includes a barrier so that the write of current->__state
// is correctly serialised wrt the caller's subsequent test of whether to
// actually sleep:
//
//   for (;;) {
// 	set_current_state(TASK_UNINTERRUPTIBLE);
// 	if (CONDITION)
// 	   break;
//
// 	schedule();
//   }
//   __set_current_state(TASK_RUNNING);
//
// If the caller does not need such serialisation (because, for instance, the
// CONDITION test and condition change and wakeup are under the same lock) then
// use __set_current_state().
//
// The above is typically ordered against the wakeup, which does:
//
//   CONDITION = 1;
//   wake_up_state(p, TASK_UNINTERRUPTIBLE);
//
// where wake_up_state()/try_to_wake_up() executes a full memory barrier before
// accessing p->__state.
//
// Wakeup will do: if (@state & p->__state) p->__state = TASK_RUNNING, that is,
// once it observes the TASK_UNINTERRUPTIBLE store the waking CPU can issue a
// TASK_RUNNING store which can collide with __set_current_state(TASK_RUNNING).
//
// However, with slightly different timing the wakeup TASK_RUNNING store can
// also collide with the TASK_UNINTERRUPTIBLE store. Losing that store is not
// a problem either because that will result in one extra go around the loop
// and our @cond test will save the day.
//
// Also see the comments of try_to_wake_up().
// set_special_state() should be used for those states when the blocking task
// can not use the regular condition based wait-loop. In that case we must
// serialize against wakeups such that any possible in-flight TASK_RUNNING
// stores will not collide with our state change.
// may shadow
#[repr(C)]
#[derive(Clone, Copy)]
struct prev_cputime {
    utime: u64,
    stime: u64,
    lock: raw_spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sched_info {
    pcount: u64,
    run_delay: u64,
    max_run_delay: u64,
    min_run_delay: u64,
    last_arrival: u64,
    last_queued: u64,
    max_run_delay_ts: timespec64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct load_weight {
    weight: u64,
    inv_weight: u32,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct sched_avg {
    last_update_time: u64,
    load_sum: u64,
    runnable_sum: u64,
    util_sum: u32,
    period_contrib: u32,
    load_avg: u64,
    runnable_avg: u64,
    util_avg: u64,
    util_est: u32,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct sched_statistics {
    wait_start: u64,
    wait_max: u64,
    wait_count: u64,
    wait_sum: u64,
    iowait_count: u64,
    iowait_sum: u64,
    sleep_start: u64,
    sleep_max: u64,
    sum_sleep_runtime: i64,
    block_start: u64,
    block_max: u64,
    sum_block_runtime: i64,
    exec_max: i64,
    slice_max: u64,
    nr_failed_migrations_affine: u64,
    nr_failed_migrations_running: u64,
    nr_failed_migrations_hot: u64,
    nr_forced_migrations: u64,
    nr_wakeups: u64,
    nr_wakeups_sync: u64,
    nr_wakeups_migrate: u64,
    nr_wakeups_local: u64,
    nr_wakeups_remote: u64,
    nr_wakeups_affine: u64,
    nr_wakeups_affine_attempts: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fsnotify_mark_connector {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_parameter_spec {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct file_kattr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sched_entity {
    load: load_weight,
    h_load: load_weight,
    run_node: rb_node,
    deadline: u64,
    min_vruntime: u64,
    min_slice: u64,
    max_slice: u64,
    group_node: list_head,
    on_rq: u8,
    sched_delayed: u8,
    rel_deadline: u8,
    custom_slice: u8,
    exec_start: u64,
    sum_exec_runtime: u64,
    prev_sum_exec_runtime: u64,
    vruntime: u64,
    vlag: i64,
    vprot: u64,
    slice: u64,
    nr_migrations: u64,
    depth: i32,
    parent: *mut sched_entity,
    cfs_rq: *mut cfs_rq,
    my_q: *mut cfs_rq,
    runnable_weight: u64,
    avg: sched_avg,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_272 {
    ia_uid: __SlateRecord_uidgid_types_h_9,
    ia_vfsuid: __SlateRecord_mnt_idmapping_h_17,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_276 {
    ia_gid: __SlateRecord_uidgid_types_h_13,
    ia_vfsgid: __SlateRecord_mnt_idmapping_h_21,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sched_rt_entity {
    run_list: list_head,
    timeout: u64,
    watchdog_stamp: u64,
    time_slice: u32,
    on_rq: u16,
    on_list: u16,
    back: *mut sched_rt_entity,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sched_dl_entity {
    rb_node: rb_node,
    dl_runtime: u64,
    dl_deadline: u64,
    dl_period: u64,
    dl_bw: u64,
    dl_density: u64,
    runtime: i64,
    deadline: u64,
    flags: u32,
    __slate_bits_0: __slate_bits::__SlateBits309U0,
    dl_timer: hrtimer,
    inactive_timer: hrtimer,
    rq: *mut rq,
    server_pick_task: Option<unsafe extern "C" fn(*mut sched_dl_entity, *mut rq_flags) -> *mut task_struct>,
    pi_se: *mut sched_dl_entity,
}


#[repr(C)]
#[derive(Clone, Copy)]
union rcu_special {
    b: __SlateRecord_sched_h_791,
    s: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_acl {
}


// Number of contexts where an event can trigger:
//      task, softirq, hardirq, nmi.
#[repr(C)]
#[derive(Clone, Copy)]
struct wake_q_node {
    next: *mut wake_q_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kmap_ctrl {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_790 {
    i_nlink: u32,
    __i_nlink: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_834 {
    i_dentry: hlist_head,
    i_rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_846 {
    i_fop: *const file_operations,
    free_inode: Option<unsafe extern "C" fn(*mut inode)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct file_lock_context {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_852 {
    i_devices: list_head,
    i_linklen: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_856 {
    i_pipe: *mut pipe_inode_info,
    i_cdev: *mut cdev,
    i_link: *mut u8,
    i_dir_seq: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cdev {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct task_ipi_mask {
    __slate_anon_0: __SlateRecord_sched_h_826,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct task_struct {
    thread_info: thread_info,
    __state: u32,
    saved_state: u32,
    stack: *mut (),
    usage: refcount_struct,
    flags: u32,
    ptrace: u32,
    on_cpu: u8,
    on_rq: u8,
    is_blocked: u8,
    __pad: u8,
    wake_entry: __call_single_node,
    wakee_flips: u32,
    wakee_flip_decay_ts: u64,
    last_wakee: *mut task_struct,
    recent_used_cpu: i32,
    wake_cpu: i32,
    prio: i32,
    static_prio: i32,
    normal_prio: i32,
    rt_priority: u32,
    se: sched_entity,
    rt: sched_rt_entity,
    dl: sched_dl_entity,
    dl_server: *mut sched_dl_entity,
    sched_class: *const sched_class,
    sched_task_group: *mut task_group,
    stats: sched_statistics,
    btrace_seq: u32,
    policy: u32,
    max_allowed_capacity: u64,
    nr_cpus_allowed: i32,
    cpus_ptr: *const cpumask,
    user_cpus_ptr: *mut cpumask,
    cpus_mask: cpumask,
    migration_pending: *mut (),
    migration_disabled: u16,
    migration_flags: u16,
    rcu_read_lock_nesting: i32,
    rcu_read_unlock_special: rcu_special,
    rcu_node_entry: list_head,
    rcu_blocked_node: *mut rcu_node,
    rcu_tasks_nvcsw: u64,
    rcu_tasks_holdout: u8,
    rcu_tasks_idx: u8,
    rcu_tasks_idle_cpu: i32,
    rcu_tasks_holdout_list: list_head,
    rcu_tasks_exit_cpu: i32,
    rcu_tasks_exit_list: list_head,
    trc_reader_nesting: i32,
    trc_reader_scp: *mut srcu_ctr,
    sched_info: sched_info,
    tasks: list_head,
    pushable_tasks: plist_node,
    pushable_dl_tasks: rb_node,
    mm: *mut mm_struct,
    active_mm: *mut mm_struct,
    exec_state: *mut task_exec_state,
    exit_state: i32,
    exit_code: i32,
    exit_signal: i32,
    pdeath_signal: i32,
    jobctl: u64,
    personality: u32,
    __slate_bits_0: __slate_bits::__SlateBits69U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits69U1,
    atomic_flags: u64,
    restart_block: restart_block,
    pid: i32,
    tgid: i32,
    stack_canary: u64,
    real_parent: *mut task_struct,
    parent: *mut task_struct,
    children: list_head,
    sibling: list_head,
    group_leader: *mut task_struct,
    ptraced: list_head,
    ptrace_entry: list_head,
    thread_pid: *mut pid,
    pid_links: [hlist_node; 4],
    thread_node: list_head,
    vfork_done: *mut completion,
    set_child_tid: *mut i32,
    clear_child_tid: *mut i32,
    worker_private: *mut (),
    utime: u64,
    stime: u64,
    gtime: u64,
    prev_cputime: prev_cputime,
    nvcsw: u64,
    nivcsw: u64,
    start_time: u64,
    start_boottime: u64,
    min_flt: u64,
    maj_flt: u64,
    posix_cputimers: posix_cputimers,
    posix_cputimers_work: posix_cputimers_work,
    ptracer_cred: *const cred,
    real_cred: *const cred,
    cred: *const cred,
    cached_requested_key: *mut key,
    comm: [u8; 16],
    nameidata: *mut nameidata,
    sysvsem: sysv_sem,
    sysvshm: sysv_shm,
    real_fs: *mut fs_struct,
    fs: *mut fs_struct,
    files: *mut files_struct,
    io_uring: *mut io_uring_task,
    io_uring_restrict: *mut io_restriction,
    nsproxy: *mut nsproxy,
    signal: *mut signal_struct,
    sighand: *mut sighand_struct,
    blocked: __SlateRecord_signal_h_25,
    real_blocked: __SlateRecord_signal_h_25,
    saved_sigmask: __SlateRecord_signal_h_25,
    pending: sigpending,
    sas_ss_sp: u64,
    sas_ss_size: u64,
    sas_ss_flags: u32,
    task_works: *mut callback_head,
    audit_context: *mut audit_context,
    loginuid: __SlateRecord_uidgid_types_h_9,
    sessionid: u32,
    seccomp: seccomp,
    syscall_dispatch: syscall_user_dispatch,
    parent_exec_id: u64,
    self_exec_id: u64,
    alloc_lock: spinlock,
    pi_lock: raw_spinlock,
    wake_q: wake_q_node,
    pi_waiters: rb_root_cached,
    pi_top_task: *mut task_struct,
    pi_blocked_on: *mut rt_mutex_waiter,
    blocked_on: *mut mutex,
    blocked_lock: raw_spinlock,
    blocked_donor: *mut task_struct,
    journal_info: *mut (),
    bio_list: *mut bio_list,
    plug: *mut blk_plug,
    reclaim_state: *mut reclaim_state,
    io_context: *mut io_context,
    capture_control: *mut capture_control,
    ptrace_message: u64,
    last_siginfo: *mut kernel_siginfo,
    ioac: task_io_accounting,
    acct_rss_mem1: u64,
    acct_vm_mem1: u64,
    acct_timexpd: u64,
    mems_allowed: __SlateRecord_nodemask_types_h_17,
    mems_allowed_seq: seqcount_spinlock,
    cpuset_mem_spread_rotor: i32,
    cgroups: *mut css_set,
    cg_list: list_head,
    futex: futex_sched_data,
    perf_recursion: [u8; 4],
    perf_event_ctxp: *mut perf_event_context,
    perf_event_mutex: mutex,
    perf_event_list: list_head,
    perf_ctx_data: *mut perf_ctx_data,
    ipi_mask: task_ipi_mask,
    mempolicy: *mut mempolicy,
    il_prev: i16,
    il_weight: u8,
    pref_node_fork: i16,
    cache_work: callback_head,
    sched_cache_grp: *mut sched_cache_group,
    preferred_llc: i32,
    pref_llc_queued: i32,
    rseq: rseq_data,
    mm_cid: sched_mm_cid,
    tlb_ubc: tlbflush_unmap_batch,
    splice_pipe: *mut pipe_inode_info,
    task_frag: page_frag,
    delays: *mut task_delay_info,
    nr_dirtied: i32,
    nr_dirtied_pause: i32,
    dirty_paused_when: u64,
    timer_slack_ns: u64,
    default_timer_slack_ns: u64,
    trace_recursion: u64,
    throttle_disk: *mut gendisk,
    utask: *mut uprobe_task,
    kmap_ctrl: kmap_ctrl,
    rcu: callback_head,
    rcu_users: refcount_struct,
    pagefault_disabled: i32,
    oom_reaper_list: *mut task_struct,
    oom_reaper_timer: timer_list,
    stack_vm_area: *mut vm_struct,
    stack_refcount: refcount_struct,
    security: *mut (),
    bpf_net_context: *mut bpf_net_context,
    mce_vaddr: *mut (),
    mce_kflags: u64,
    mce_addr: u64,
    __slate_bits_2: __slate_bits::__SlateBits69U2,
    mce_kill_me: callback_head,
    mce_count: i32,
    kretprobe_instances: llist_head,
    rethooks: llist_head,
    l1d_flush_kill: callback_head,
    unwind_info: unwind_task_info,
    thread: thread_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_1267 {
    f_path: path,
    __f_path: path,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_1271 {
    f_pos_lock: mutex,
    f_pipe: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_h_1287 {
    f_task_work: callback_head,
    f_llist: llist_node,
    f_ra: file_ra_state,
    f_freeptr: __SlateRecord_mm_types_h_701,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct file_lock {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct file_lease {
}


// Per process flags
// I'm a virtual CPU
// I am an IDLE thread
// Allocating memory to free memory. See memalloc_noreclaim_save()
// Kernel thread cloned from userspace thread
// This thread should not be frozen
// I am kswapd
// All allocations inherit GFP_NOFS. See memalloc_nfs_save()
// All allocations inherit GFP_NOIO. See memalloc_noio_save()
// I am a kernel thread
// Userland is not allowed to meddle with cpus_mask
// Allocations constrained to zones which allow long term pinning.
// See memalloc_pin_save()
// plug has ts that needs updating
// Per-process atomic flags.
// May not gain new privileges.
// Spread page cache over cpuset
// Speculative Store Bypass disabled
// Speculative Store Bypass force disabled
// Indirect branch speculation restricted
// Indirect branch speculation permanently restricted
// Speculative Store Bypass clear on execve()
fn test_tsk_thread_flag(tsk: *mut task_struct, flag: i32) -> i32 {
test_ti_thread_flag(task_thread_info!(tsk), flag)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_cmd {
}


#[inline(always)]
fn _cond_resched() -> i32 {
let mut __v0: Option<unsafe extern "C" fn() -> i32>;
__v0 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn() -> i32>>(__SCT__cond_resched as *const ()) };
unsafe { __v0.unwrap()() }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__cond_resched_88: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__cond_resched) }) as *mut ();

#[inline(always)]
fn need_resched() -> bool {
unlikely__2!(tif_need_resched()) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct sched_cache_time {
    runtime: u64,
    epoch: u64,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct sched_cache_group {
    pcpu_sched: *mut sched_cache_time,
    lock: raw_spinlock,
    epoch: u64,
    nr_running_avg: u64,
    next_scan: u64,
    footprint: u64,
    cpu: i32,
    refcnt: refcount_struct,
    rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_open_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_iattrs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct old_timespec32 {
    tv_sec: i32,
    tv_nsec: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kstat {
    result_mask: u32,
    mode: u16,
    nlink: u32,
    blksize: u32,
    attributes: u64,
    attributes_mask: u64,
    ino: u64,
    dev: u32,
    rdev: u32,
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    size: i64,
    atime: timespec64,
    mtime: timespec64,
    ctime: timespec64,
    btime: timespec64,
    blocks: u64,
    mnt_id: u64,
    change_cookie: u64,
    subvol: u64,
    dio_mem_align: u32,
    dio_offset_align: u32,
    dio_read_offset_align: u32,
    atomic_write_unit_min: u32,
    atomic_write_unit_max: u32,
    atomic_write_unit_max_opt: u32,
    atomic_write_segments_max: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hlist_nulls_node {
    next: *mut hlist_nulls_node,
    pprev: *mut *mut hlist_nulls_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wait_queue_entry {
    flags: u32,
    private: *mut (),
    func: Option<unsafe extern "C" fn(*mut wait_queue_entry, u32, i32, *mut ()) -> i32>,
    entry: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wait_queue_head {
    lock: spinlock,
    head: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
}


fn init_waitqueue_func_entry(wq_entry: *mut wait_queue_entry, func: Option<unsafe extern "C" fn(*mut wait_queue_entry, u32, i32, *mut ()) -> i32>) {
unsafe {
(*wq_entry).flags = 0;
}
unsafe {
(*wq_entry).private = core::ptr::null_mut::<()>();
}
unsafe {
(*wq_entry).func = func;
}
}

fn __add_wait_queue(wq_head: *mut wait_queue_head, wq_entry: *mut wait_queue_entry) {
let mut head: *mut list_head = unsafe { core::ptr::addr_of_mut!((*wq_head).head) };
let mut wq: *mut wait_queue_entry = unsafe { core::mem::zeroed() };
wq = list_first_entry__16!(unsafe { core::ptr::addr_of_mut!((*wq_head).head) });
while list_entry_is_head!(wq, unsafe { core::ptr::addr_of_mut!((*wq_head).head) }, entry) == 0 {
if (unsafe { (*wq).flags }) & 16 == 0 {
break;
}
head = unsafe { core::ptr::addr_of_mut!((*wq).entry) };
wq = list_next_entry__12!(wq, entry);
}
list_add(unsafe { core::ptr::addr_of_mut!((*wq_entry).entry) }, head);
}

// explicit shadow
// wait_event - sleep until a condition gets true
// @wq_head: the waitqueue to wait on
// @condition: a C expression for the event to wait for
//
// The process is put to sleep (TASK_UNINTERRUPTIBLE) until the
// @condition evaluates to true. The @condition is checked each time
// the waitqueue @wq_head is woken up.
//
// wake_up() has to be called after changing any variable that could
// change the result of the wait condition.
// wait_event_timeout - sleep until a condition gets true or a timeout elapses
// @wq_head: the waitqueue to wait on
// @condition: a C expression for the event to wait for
// @timeout: timeout, in jiffies
//
// The process is put to sleep (TASK_UNINTERRUPTIBLE) until the
// @condition evaluates to true. The @condition is checked each time
// the waitqueue @wq_head is woken up.
//
// wake_up() has to be called after changing any variable that could
// change the result of the wait condition.
//
// Returns:
// 0 if the @condition evaluated to %false after the @timeout elapsed,
// 1 if the @condition evaluated to %true after the @timeout elapsed,
// or the remaining jiffies (at least 1) if the @condition evaluated
// to %true before the @timeout elapsed.
fn __mutex_init(lock: *mut mutex, name: *const u8, key: *mut lock_class_key) {
unsafe { mutex_init_generic(lock) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct kref {
    refcount: refcount_struct,
}


#[inline(always)]
fn __acquire_ctx_lock__6(var: &__ctx_lock_RCU) {
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_sysfs_h_107 {
    is_visible: Option<unsafe extern "C" fn(*mut kobject, *mut attribute, i32) -> u16>,
    is_visible_const: Option<unsafe extern "C" fn(*mut kobject, *const attribute, i32) -> u16>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_sysfs_h_118 {
    attrs: *mut *mut attribute,
    attrs_const: *const *const attribute,
}


#[inline(always)]
fn __acquire_shared_ctx_lock__6(var: *const __ctx_lock_RCU) {
}

#[inline(always)]
fn __release_ctx_lock__6(var: &__ctx_lock_RCU) {
}

#[inline(always)]
fn __release_shared_ctx_lock__6(var: *const __ctx_lock_RCU) {
}

#[inline(always)]
fn __assume_ctx_lock__6(var: &__ctx_lock_RCU) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __ctx_lock_RCU {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rcu_gp_seq {
    norm: u64,
    exp: u64,
}


// Defined as a macro as it is a very low level header included from
// areas that don't even know about current.  This gives the rcu_read_lock()
// nesting depth, but makes sense only if CONFIG_PREEMPT_RCU -- in other
// types of kernel builds, the rcu_read_lock() nesting depth is unknowable.
fn rcu_read_lock_held() -> i32 {
1
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
}


// Dependency order vs. p above.
// Dependency order vs. p above.
// rcu_assign_pointer() - assign to RCU-protected pointer
// @p: pointer to assign to
// @v: value to assign (publish)
//
// Assigns the specified value to the specified RCU-protected
// pointer, ensuring that any concurrent RCU readers will see
// any prior initialization.
//
// Inserts memory barriers on architectures that require them
// (which is most of them), and also prevents the compiler from
// reordering the code that initializes the structure after the pointer
// assignment.  More importantly, this call documents which pointers
// will be dereferenced by RCU read-side code.
//
// In some special cases, you may use RCU_INIT_POINTER() instead
// of rcu_assign_pointer().  RCU_INIT_POINTER() is a bit faster due
// to the fact that it does not constrain either the CPU or the compiler.
// That said, using RCU_INIT_POINTER() when you should have used
// rcu_assign_pointer() is a very bad thing that results in
// impossible-to-diagnose memory corruption.  So please be careful.
// See the RCU_INIT_POINTER() comment header for details.
//
// Note that rcu_assign_pointer() evaluates each of its arguments only
// once, appearances notwithstanding.  One of the "extra" evaluations
// is in typeof() and the other visible only to sparse (__CHECKER__),
// neither of which actually execute the argument.  As with most cpp
// macros, this execute-arguments-only-once property is important, so
// please be careful when making changes to rcu_assign_pointer() and the
// other macros that it invokes.
// rcu_replace_pointer() - replace an RCU pointer, returning its old value
// @rcu_ptr: RCU pointer, whose old value is returned
// @ptr: regular pointer
// @c: the lockdep conditions under which the dereference will take place
//
// Perform a replacement, where @rcu_ptr is an RCU-annotated
// pointer and @c is the lockdep argument that is passed to the
// rcu_dereference_protected() call used to read that pointer.  The old
// value of @rcu_ptr is returned, and @rcu_ptr is set to @ptr.
// rcu_access_pointer() - fetch RCU pointer with no dereferencing
// @p: The pointer to read
//
// Return the value of the specified RCU-protected pointer, but omit the
// lockdep checks for being in an RCU read-side critical section.  This is
// useful when the value of this pointer is accessed, but the pointer is
// not dereferenced, for example, when testing an RCU-protected pointer
// against NULL.  Within an RCU read-side critical section, there is little
// reason to use rcu_access_pointer().  Although rcu_access_pointer() may
// also be used in cases where update-side locks prevent the value of the
// pointer from changing, you should instead use rcu_dereference_protected()
// for this use case.  It is also permissible to use rcu_access_pointer()
// within lockless updaters to obtain the old value for an atomic operation,
// for example, for cmpxchg().
//
// It is usually best to test the rcu_access_pointer() return value
// directly in order to avoid accidental dereferences being introduced
// by later inattentive changes.  In other words, assigning the
// rcu_access_pointer() return value to a local variable results in an
// accident waiting to happen.
//
// It is also permissible to use rcu_access_pointer() when read-side
// access to the pointer was removed at least one grace period ago, as is
// the case in the context of the RCU callback that is freeing up the data,
// or after a synchronize_rcu() returns.  This can be useful when tearing
// down multi-linked structures after a grace period has elapsed.  However,
// rcu_dereference_protected() is normally preferred for this use case.
// rcu_dereference_check() - rcu_dereference with debug checking
// @p: The pointer to read, prior to dereferencing
// @c: The conditions under which the dereference will take place
//
// Do an rcu_dereference(), but check that the conditions under which the
// dereference will take place are correct.  Typically the conditions
// indicate the various locking conditions that should be held at that
// point.  The check should return true if the conditions are satisfied.
// An implicit check for being in an RCU read-side critical section
// (rcu_read_lock()) is included.
//
// For example:
//
// 	bar = rcu_dereference_check(foo->bar, lockdep_is_held(&foo->lock));
//
// could be used to indicate to lockdep that foo->bar may only be dereferenced
// if either rcu_read_lock() is held, or that the lock required to replace
// the bar struct at foo->bar is held.
//
// Note that the list of conditions may also include indications of when a lock
// need not be held, for example during initialisation or destruction of the
// target struct:
//
// 	bar = rcu_dereference_check(foo->bar, lockdep_is_held(&foo->lock) ||
// 					      atomic_read(&foo->usage) == 0);
//
// Inserts memory barriers on architectures that require them
// (currently only the Alpha), prevents the compiler from refetching
// (and from merging fetches), and, more importantly, documents exactly
// which pointers are protected by RCU and checks that the pointer is
// annotated as __rcu.
// rcu_dereference_protected() - fetch RCU pointer when updates prevented
// @p: The pointer to read, prior to dereferencing
// @c: The conditions under which the dereference will take place
//
// Return the value of the specified RCU-protected pointer, but omit
// the READ_ONCE().  This is useful in cases where update-side locks
// prevent the value of the pointer from changing.  Please note that this
// primitive does *not* prevent the compiler from repeating this reference
// or combining it with other references, so it should not be used without
// protection of appropriate locks.
//
// This function is only for update-side use.  Using this function
// when protected only by rcu_read_lock() will result in infrequent
// but very ugly failures.
// rcu_dereference() - fetch RCU-protected pointer for dereferencing
// @p: The pointer to read, prior to dereferencing
//
// This is a simple wrapper around rcu_dereference_check().
#[inline(always)]
fn rcu_read_lock() {
unsafe { __rcu_read_lock() };
__acquire_shared!(unsafe { RCU });
rcu_lock_acquire!();
RCU_LOCKDEP_WARN!(!unsafe { rcu_is_watching() });
}

fn rcu_read_unlock() {
RCU_LOCKDEP_WARN!(!unsafe { rcu_is_watching() });
rcu_lock_release!(); // Keep acq info for rls diags.
__release_shared!(unsafe { RCU });
unsafe { __rcu_read_unlock() };
}

// 256 bytes including ->parent
// 256 bytes
// 240 bytes
#[repr(C)]
#[derive(Clone, Copy)]
struct maple_tree {
    __slate_anon_0: __SlateRecord_maple_tree_h_261,
    ma_flags: u32,
    ma_root: *mut (),
}


#[inline(always)]
fn __acquire_ctx_lock__7(var: &rw_semaphore) {
}

#[inline(always)]
fn __acquire_shared_ctx_lock__7(var: &rw_semaphore) {
}

#[inline(always)]
fn __release_ctx_lock__7(var: &rw_semaphore) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct module_param_attrs {
}


#[inline(always)]
fn __release_shared_ctx_lock__7(var: &rw_semaphore) {
}

#[inline(always)]
fn __assume_ctx_lock__7(var: *const rw_semaphore) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rw_semaphore {
    count: __SlateRecord_types_h_197,
    owner: __SlateRecord_types_h_197,
    osq: optimistic_spin_queue,
    wait_lock: raw_spinlock,
    first_waiter: *mut rwsem_waiter,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swait_queue_head {
    lock: raw_spinlock,
    task_list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_symbol {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct module_sect_attrs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct module_notes_attrs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_call {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct completion {
    done: u32,
    wait: swait_queue_head,
}


fn init_completion(x: *mut completion) {
unsafe {
(*x).done = 0;
}
loop {
unsafe { __init_swait_queue_head(core::ptr::addr_of_mut!((*x).wait), c"&x->wait".as_ptr() as *const u8, core::ptr::addr_of_mut!(__key__87)) };
if true {
break;
}
}
}

static mut __key__87: lock_class_key = unsafe { core::mem::zeroed() };

// Change timeval to jiffies, trying to avoid the
// most obvious overflows..
//
// And some not so obvious.
//
// Note that we don't want to return LONG_MAX, because
// for various timeout reasons we often end up having
// to wait "jiffies+1" in order to guarantee that we wait
// at _least_ "jiffies" - so "jiffies+1" had better still
// be positive.
fn _msecs_to_jiffies(m: u32) -> u64 {
((m as i64).wrapping_add(1000 / 1000).wrapping_sub(1) / (1000 / 1000)) as u64
}

#[repr(C)]
#[derive(Clone, Copy)]
struct wake_irq {
}


#[inline(always)]
fn msecs_to_jiffies(m: u32) -> u64 {
if unsafe { __slate_intrinsic_1c5364fcd9da51fe(m) } {
if (m as i32) < 0 {
return 4611686018427387902u64;
}
return _msecs_to_jiffies(m);
} else {
return unsafe { __msecs_to_jiffies(m) };
}
}

// Add a ktime_t variable and a scalar nanosecond value.
// res = kt + nsval:
// Subtract a scalar nanosecod from a ktime_t variable
// res = kt - nsval:
// Map the ktime_t to timespec conversion to ns_to_timespec function
fn ktime_to_ns(kt: i64) -> i64 {
kt
}

#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_qos {
}


fn ktime_get_ns() -> u64 {
ktime_to_ns(unsafe { ktime_get() }) as u64
}

#[repr(C)]
#[derive(Clone, Copy)]
struct delayed_work {
    work: work_struct,
    timer: timer_list,
    wq: *mut workqueue_struct,
    cpu: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rcu_work {
    work: work_struct,
    rcu: callback_head,
    wq: *mut workqueue_struct,
}


fn __init_work(work: *mut work_struct, onstack: i32) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rcu_segcblist {
    head: *mut callback_head,
    tails: [*mut *mut callback_head; 4],
    gp_seq: [rcu_gp_seq; 4],
    len: i64,
    seglen: [i64; 4],
    flags: u8,
}


#[inline(always)]
fn __acquire_ctx_lock__8(var: &srcu_struct) {
}

#[inline(always)]
fn __acquire_shared_ctx_lock__8(var: *const srcu_struct) {
}

#[inline(always)]
fn __release_ctx_lock__8(var: &srcu_struct) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
}


#[inline(always)]
fn __release_shared_ctx_lock__8(var: *const srcu_struct) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct device_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_group {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_iommu {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_device_data {
}


#[inline(always)]
fn __assume_ctx_lock__8(var: &srcu_struct) {
}

fn __init_srcu_struct(ssp: *mut srcu_struct, name: *const u8, key: *mut lock_class_key) -> i32 {
unsafe { init_srcu_struct_generic(ssp) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct srcu_ctr {
    srcu_locks: __SlateRecord_types_h_197,
    srcu_unlocks: __SlateRecord_types_h_197,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct srcu_data {
    srcu_ctrs: [srcu_ctr; 2],
    srcu_reader_flavor: i32,
    __slate_pad_2: [u8; 28],
    lock: raw_spinlock,
    srcu_cblist: rcu_segcblist,
    srcu_gp_seq_needed: u64,
    srcu_gp_seq_needed_exp: u64,
    srcu_cblist_invoking: bool,
    delay_work: timer_list,
    work: work_struct,
    srcu_barrier_head: callback_head,
    srcu_ec_head: callback_head,
    srcu_ec_state: i32,
    mynode: *mut srcu_node,
    grpmask: u64,
    cpu: i32,
    ssp: *mut srcu_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_device_h_719 {
    name: *const u8,
    lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bus_dma_region {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_tlb_mem {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct srcu_node {
    lock: raw_spinlock,
    srcu_have_cbs: [u64; 4],
    srcu_data_have_cbs: [u64; 4],
    srcu_gp_seq_needed_exp: u64,
    srcu_parent: *mut srcu_node,
    grplo: i32,
    grphi: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct srcu_usage {
    node: *mut srcu_node,
    level: [*mut srcu_node; 3],
    srcu_size_state: i32,
    srcu_cb_mutex: mutex,
    lock: raw_spinlock,
    srcu_gp_mutex: mutex,
    srcu_gp_seq: u64,
    srcu_gp_seq_needed: u64,
    srcu_gp_seq_needed_exp: u64,
    srcu_gp_start: u64,
    srcu_last_gp_end: u64,
    srcu_size_jiffies: u64,
    srcu_n_lock_retries: u64,
    srcu_n_exp_nodelay: u64,
    sda_is_static: bool,
    srcu_barrier_seq: u64,
    srcu_barrier_mutex: mutex,
    srcu_barrier_completion: completion,
    srcu_barrier_cpu_cnt: __SlateRecord_types_h_190,
    reschedule_jiffies: u64,
    reschedule_count: u64,
    work: delayed_work,
    irq_work: irq_work,
    srcu_ssp: *mut srcu_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_memremap_h_143 {
    range: range,
    __slate_anon_1: __SlateRecord_memremap_h_145,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_memremap_h_145 {
    __empty_ranges: __SlateRecord_memremap_h_145_2,
    ranges: [range; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_memremap_h_145_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct srcu_struct {
    srcu_ctrp: *mut srcu_ctr,
    sda: *mut srcu_data,
    srcu_reader_flavor: u8,
    dep_map: lockdep_map,
    srcu_sup: *mut srcu_usage,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_h_754 {
    vma: *mut vm_area_struct,
    gfp_mask: u32,
    pgoff: u64,
    address: u64,
    real_address: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_h_768 {
    orig_pte: __SlateRecord_pgtable_64_types_h_21,
    orig_pmd: __SlateRecord_pgtable_64_types_h_22,
}


fn __srcu_read_lock_fast(ssp: *mut srcu_struct) -> *mut srcu_ctr {
let mut scp: *mut srcu_ctr = unsafe { core::mem::zeroed() };
let mut __v0: *mut srcu_ctr;
loop {
if !(__native_word__11!() || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_167() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*ssp).srcu_ctrp) as *const *mut srcu_ctr) };
scp = __v0;
if IS_ENABLED!() == 0 {
loop {
__verify_pcpu_ptr__6!();
match (const { core::mem::size_of::<i64>() }) as u64 {
1 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incb %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decb %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u8 = 1u8;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addb {1:l}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
2 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incw %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decw %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u16 = 1u16;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addw {1:x}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, in(reg) pto_val__ as u16, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
4 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incl %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decl %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u32 = 1u32;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addl {1:e}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, in(reg) pto_val__ as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
8 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incq %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decq %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u64 = 1;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addq {1:r}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, in(reg) pto_val__ as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
if true {
break;
}
}
} else {
let mut __v13: *mut __SlateRecord_types_h_197;
__verify_pcpu_ptr__5!();
let mut __v15: *mut __SlateRecord_types_h_197;
let mut tcp_ptr__: u64 = 0;
let mut __v16: u64;
let mut pfo_val__: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t0, sym this_cpu_off, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
__v16 = pfo_val__;
tcp_ptr__ = __v16;
let __v18: u64 = tcp_ptr__.wrapping_add((unsafe { core::ptr::addr_of_mut!((*scp).srcu_locks) }) as u64);
tcp_ptr__ = __v18;
__v15 = tcp_ptr__ as *mut __SlateRecord_types_h_197;
__v13 = __v15;
atomic_long_inc(__v13);
}
// Y, and implicit RCU reader.
// Y, and implicit RCU reader.
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
// Avoid leaking the critical section.
__acquire_shared__2!(ssp);
scp
}

fn __srcu_read_unlock_fast(ssp: *mut srcu_struct, mut scp: *mut srcu_ctr) {
__release_shared__2!(ssp);
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
// Avoid leaking the critical section.
if IS_ENABLED!() == 0 {
loop {
__verify_pcpu_ptr__6!();
match (const { core::mem::size_of::<i64>() }) as u64 {
1 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incb %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decb %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u8 = 1u8;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addb {1:l}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
2 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incw %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decw %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u16 = 1u16;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addw {1:x}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, in(reg) pto_val__ as u16, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
4 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incl %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decl %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u32 = 1u32;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addl {1:e}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, in(reg) pto_val__ as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
8 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incq %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decq %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u64 = 1;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addq {1:r}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, in(reg) pto_val__ as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
if true {
break;
}
}
} else {
let mut __v11: *mut __SlateRecord_types_h_197;
__verify_pcpu_ptr__5!();
let mut __v13: *mut __SlateRecord_types_h_197;
let mut tcp_ptr__: u64 = 0;
let mut __v14: u64;
let mut pfo_val__: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t0, sym this_cpu_off, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
__v14 = pfo_val__;
tcp_ptr__ = __v14;
let __v16: u64 = tcp_ptr__.wrapping_add((unsafe { core::ptr::addr_of_mut!((*scp).srcu_unlocks) }) as u64);
tcp_ptr__ = __v16;
__v13 = tcp_ptr__ as *mut __SlateRecord_types_h_197;
__v11 = __v13;
atomic_long_inc(__v11);
}
// Z, and implicit RCU reader.
// Z, and implicit RCU reader.
}

fn srcu_check_read_flavor(ssp: *mut srcu_struct, read_flavor: i32) {
if IS_ENABLED!() != 0 {
unsafe { __srcu_check_read_flavor(ssp, read_flavor) };
}
}

// Values for SRCU Tree srcu_data ->srcu_reader_flavor, but also used by rcutorture.
// srcu_read_lock().
// srcu_read_lock_nmisafe().
// 					0x4		// SRCU-lite is no longer with us.
// srcu_read_lock_fast(), also NMI-safe.
// srcu_read_lock_fast_updown().
fn srcu_read_lock(ssp: *mut srcu_struct) -> i32 {
let mut retval: i32 = 0;
srcu_check_read_flavor(ssp, 1i32);
retval = unsafe { __srcu_read_lock(ssp) };
srcu_lock_acquire!();
retval
}

fn srcu_read_lock_fast_notrace(ssp: *mut srcu_struct) -> *mut srcu_ctr {
srcu_check_read_flavor(ssp, 4i32);
let retval: *mut srcu_ctr = __srcu_read_lock_fast(ssp);
retval
}

fn srcu_read_unlock(ssp: *mut srcu_struct, idx: i32) {
let mut __v0: i64;
let __ret_warn_on: i32 = (idx & !1 != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("173: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 173b - ., 3; .popsection\n# {0}", const 173 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_srcu_read_unlock_1, sym __slate_asm_string_srcu_read_unlock_2, const 495 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("174: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 174b - ., 4; .popsection\n# {0}", const 174 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
srcu_check_read_flavor(ssp, 1i32);
srcu_lock_release!();
unsafe { __srcu_read_unlock(ssp, idx) };
}

fn srcu_read_unlock_fast_notrace(ssp: *mut srcu_struct, scp: *mut srcu_ctr) {
srcu_check_read_flavor(ssp, 4i32);
__srcu_read_unlock_fast(ssp, scp);
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_blk_types_h_239 {
    bi_cookie: u32,
    __bi_nr_segments: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blkcg_gq {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcu_h_645 {
    lock: *mut srcu_struct,
    scp: *mut srcu_ctr,
}


#[inline(always)]
fn class_srcu_fast_notrace_destructor(_T: *mut __SlateRecord_srcu_h_645) {
srcu_read_unlock_fast_notrace(unsafe { (*_T).lock }, unsafe { (*_T).scp });
}

#[repr(C)]
#[derive(Clone, Copy)]
struct bio_alloc_cache {
}


#[inline(always)]
fn class_srcu_fast_notrace_constructor(l: *mut srcu_struct) -> __SlateRecord_srcu_h_645 {
let mut _t: __SlateRecord_srcu_h_645 = {
let mut __t0: __SlateRecord_srcu_h_645 = unsafe { core::mem::zeroed() };
__t0.lock = l;
__t0
};
let _T: *mut __SlateRecord_srcu_h_645 = core::ptr::addr_of_mut!(_t);
unsafe {
(*_T).scp = srcu_read_lock_fast_notrace((*_T).lock);
}
return _t;
}

#[inline(always)]
fn __class_srcu_fast_notrace_cleanup_ctx(_T: *mut *mut __SlateRecord_srcu_h_645) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct arch_uprobe {
    __slate_anon_0: __SlateRecord_uprobes_h_31,
    ops: *const uprobe_xol_ops,
    __slate_anon_2: __SlateRecord_uprobes_h_38,
    flags: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct arch_uprobe_task {
    saved_scratch_register: u64,
    saved_trap_nr: u32,
    saved_tf: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hprobe {
    state: u32,
    srcu_scp: *mut srcu_ctr,
    uprobe: *mut uprobe,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe_task {
    state: u32,
    depth: u32,
    return_instances: *mut return_instance,
    ri_pool: *mut return_instance,
    ri_timer: timer_list,
    ri_seqcount: seqcount,
    __slate_anon_6: __SlateRecord_uprobes_h_136,
    active_uprobe: *mut uprobe,
    xol_vaddr: u64,
    signal_denied: bool,
    auprobe: *mut arch_uprobe,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct return_consumer {
    cookie: u64,
    id: u64,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct return_instance {
    hprobe: hprobe,
    func: u64,
    stack: u64,
    orig_ret_vaddr: u64,
    chained: bool,
    cons_cnt: i32,
    next: *mut return_instance,
    rcu: callback_head,
    consumer: return_consumer,
    extra_consumers: *mut return_consumer,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobes_state {
    xol_area: *mut xol_area,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pr_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_counter {
    lock: raw_spinlock,
    count: i64,
    list: list_head,
    counters: *mut i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hd_geometry {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_report_zones_args {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_queue_stats {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct disk_events {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct badblocks {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mmu_h_84 {
    ctx_id: u64,
    tlb_gen: __SlateRecord_types_h_197,
    next_trim_cpumask: u64,
    ldt_usr_sem: rw_semaphore,
    ldt: *mut ldt_struct,
    flags: u64,
    lock: mutex,
    vdso: *mut (),
    vdso_image: *const vdso_image,
    perf_rdpmc_allowed: __SlateRecord_types_h_190,
    pkey_allocation_map: u16,
    execute_only_pkey: i16,
    global_asid: u16,
    asid_transition: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_41 {
    f: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct timer_rand_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cdrom_device_info {
}


#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct page {
    flags: __SlateRecord_mm_types_h_41,
    __slate_anon_1: __SlateRecord_mm_types_h_89,
    __slate_anon_2: __SlateRecord_mm_types_h_155,
    _refcount: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_288 {
    val: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct folio {
    __slate_anon_0: __SlateRecord_mm_types_h_406,
    __slate_anon_1: __SlateRecord_mm_types_h_451,
    __slate_anon_2: __SlateRecord_mm_types_h_483,
    __slate_anon_3: __SlateRecord_mm_types_h_497,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_701 {
    v: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_userfaultfd_ctx {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pfnmap_track_ctx {
    kref: kref,
    pfn: u64,
    size: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mmap_action {
    __slate_anon_0: __SlateRecord_mm_types_h_830,
    r#type: u32,
    error_override: i32,
    __slate_bits_0: __slate_bits::__SlateBits447U0,
}


// Opaque type representing current VMA (vm_area_struct) flag state. Must be
// accessed via vma_flags_xxx() helper functions.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_871 {
    __vma_flags: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_area_desc {
    mm: *mut mm_struct,
    file: *mut file,
    start: u64,
    end: u64,
    pgoff: u64,
    vm_file: *mut file,
    vma_flags: __SlateRecord_mm_types_h_871,
    page_prot: pgprot,
    vm_ops: *const vm_operations_struct,
    private_data: *mut (),
    action: mmap_action,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct vm_area_struct {
    __slate_anon_0: __SlateRecord_mm_types_h_926,
    vm_mm: *mut mm_struct,
    vm_page_prot: pgprot,
    __slate_anon_3: __SlateRecord_mm_types_h_947,
    vm_lock_seq: u32,
    __vm_anon_pgoff_lo: u32,
    anon_vma_chain: list_head,
    anon_vma: *mut anon_vma,
    vm_ops: *const vm_operations_struct,
    vm_pgoff: u64,
    vm_file: *mut file,
    vm_private_data: *mut (),
    swap_readahead_info: __SlateRecord_types_h_197,
    vm_policy: *mut mempolicy,
    __slate_pad_14: [u8; 8],
    vm_refcnt: refcount_struct,
    __vm_anon_pgoff_hi: u32,
    shared: __SlateRecord_mm_types_h_1062,
    vm_userfaultfd_ctx: vm_userfaultfd_ctx,
    pfnmap_track_ctx: *mut pfnmap_track_ctx,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_blk_mq_h_125 {
    queuelist: list_head,
    rq_next: *mut request,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_blk_mq_h_182 {
    hash: hlist_node,
    ipi_list: llist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_blk_mq_h_193 {
    rb_node: rb_node,
    special_vec: bio_vec,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_blk_mq_h_202 {
    icq: *mut io_cq,
    r#priv: [*mut (); 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_blk_mq_h_207 {
    seq: u32,
    saved_end_io: Option<unsafe extern "C" fn(*mut request, u8, *const io_comp_batch) -> u32>,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct __SlateRecord_blk_mq_h_323 {
    lock: spinlock,
    dispatch: list_head,
    state: u64,
}


// Opaque type representing current mm_struct flag state. Must be accessed via
// mm_flags_xxx() helper functions.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_1171 {
    __mm_flags: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mm_struct {
    __slate_anon_0: __SlateRecord_mm_types_h_1176,
    flexible_array: [u8; 0],
}


// Clear the access bit
// Clear the dirty bit
// MMF_HAS_PINNED: Whether this mm has pinned any pages.  This can be either
// replaced in the future by mm.pinned_vm when it becomes stable, or grow into
// a counter on its own. We're aggresive on this bit for now: even if the
// pinned pages were unpinned later on, we'll still keep this bit set for the
// lifecycle of this mm, just for simplicity.
// FOLL_PIN has run, never cleared
#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_vmalloc_h_53 {
    next: *mut vm_struct,
    llnode: llist_node,
}


#[inline(always)]
fn __acquire_ctx_lock__9(var: &local_lock) {
}

#[inline(always)]
fn __acquire_shared_ctx_lock__9(var: &local_lock) {
}

#[inline(always)]
fn __release_ctx_lock__9(var: &local_lock) {
}

#[inline(always)]
fn __release_shared_ctx_lock__9(var: &local_lock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ipc_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mnt_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct proc_ns_operations {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct time_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uts_namespace {
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct __SlateRecord_ns_common_types_h_112 {
    __ns_ref: refcount_struct,
}


#[inline(always)]
fn __assume_ctx_lock__9(var: &local_lock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_user_namespace_h_26 {
    __slate_anon_0: __SlateRecord_user_namespace_h_27,
    __slate_anon_1: __SlateRecord_user_namespace_h_31,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_user_namespace_h_27 {
    extent: [uid_gid_extent; 5],
    nr_extents: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_user_namespace_h_31 {
    forward: *mut uid_gid_extent,
    reverse: *mut uid_gid_extent,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct binfmt_misc {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct local_lock {
}


#[inline(always)]
fn __acquire_ctx_lock__10(var: &local_trylock) {
}

#[inline(always)]
fn __acquire_shared_ctx_lock__10(var: &local_trylock) {
}

#[inline(always)]
fn __release_ctx_lock__10(var: &local_trylock) {
}

#[inline(always)]
fn __release_shared_ctx_lock__10(var: &local_trylock) {
}

#[inline(always)]
fn __assume_ctx_lock__10(var: &local_trylock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct local_trylock {
    acquired: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_taskset {
}


fn cpu_to_node(cpu: i32) -> i32 {
per_cpu!(numa_node, cpu)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ctl_table_poll {
    event: __SlateRecord_types_h_190,
    wait: wait_queue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ctl_table {
    procname: *const u8,
    data: *mut (),
    maxlen: i32,
    mode: u16,
    proc_handler: Option<unsafe extern "C" fn(*const ctl_table, i32, *mut (), *mut u64, *mut i64) -> i32>,
    poll: *mut ctl_table_poll,
    extra1: *mut (),
    extra2: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ctl_node {
    node: rb_node,
    header: *mut ctl_table_header,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ctl_table_header {
    __slate_anon_0: __SlateRecord_sysctl_h_254,
    unregistering: *mut completion,
    ctl_table_arg: *const ctl_table,
    root: *mut ctl_table_root,
    set: *mut ctl_table_set,
    parent: *mut ctl_dir,
    node: *mut ctl_node,
    inodes: hlist_head,
    r#type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_cgroup_defs_h_644 {
    __slate_anon_0: __SlateRecord_cgroup_defs_h_645,
    __slate_anon_1: __SlateRecord_cgroup_defs_h_646,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cgroup_defs_h_645 {
    __empty_ancestors: __SlateRecord_cgroup_defs_h_645_2,
    ancestors: [*mut cgroup; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cgroup_defs_h_645_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cgroup_defs_h_646 {
    _root_ancestor: *mut cgroup,
    __slate_anon_1: __SlateRecord_cgroup_defs_h_648,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cgroup_defs_h_648 {
    __empty__low_ancestors: __SlateRecord_cgroup_defs_h_648_2,
    _low_ancestors: [*mut cgroup; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cgroup_defs_h_648_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ctl_dir {
    header: ctl_table_header,
    root: rb_root,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_cluster_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ctl_table_set {
    is_seen: Option<unsafe extern "C" fn(*mut ctl_table_set) -> i32>,
    dir: ctl_dir,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_pin {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ctl_table_root {
    default_set: ctl_table_set,
    lookup: Option<unsafe extern "C" fn(*mut ctl_table_root) -> *mut ctl_table_set>,
    set_ownership: Option<unsafe extern "C" fn(*mut ctl_table_header, *mut __SlateRecord_uidgid_types_h_9, *mut __SlateRecord_uidgid_types_h_13)>,
    permissions: Option<unsafe extern "C" fn(*mut ctl_table_header, *const ctl_table) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rhash_head {
    next: *mut rhash_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iovec {
    iov_base: *mut (),
    iov_len: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kvec {
    iov_base: *mut (),
    iov_len: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iov_iter {
    iter_type: u8,
    nofault: bool,
    data_source: bool,
    iov_offset: u64,
    __slate_anon_4: __SlateRecord_uio_h_58,
    __slate_anon_5: __SlateRecord_uio_h_78,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct shrink_control {
    gfp_mask: u32,
    nid: i32,
    nr_to_scan: u64,
    nr_scanned: u64,
    memcg: *mut mem_cgroup,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct shrinker {
    count_objects: Option<unsafe extern "C" fn(*mut shrinker, *mut shrink_control) -> u64>,
    scan_objects: Option<unsafe extern "C" fn(*mut shrinker, *mut shrink_control) -> u64>,
    batch: i64,
    seeks: i32,
    flags: u32,
    refcount: refcount_struct,
    done: completion,
    rcu: callback_head,
    private_data: *mut (),
    list: list_head,
    nr_deferred: *mut __SlateRecord_types_h_197,
}


fn memalloc_flags_save(flags: u32) -> u32 {
let oldflags: u32 = !unsafe { (*get_current()).flags } & flags;
let __v0: *mut task_struct = get_current();
unsafe {
(*__v0).flags = (*__v0).flags | flags;
}
oldflags
}

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct __SlateRecord_blk_mq_h_20 {
    lock: spinlock,
    rq_lists: [list_head; 3],
}


fn memalloc_flags_restore(flags: u32) {
let __v0: *mut task_struct = get_current();
let __v2: u32 = (unsafe { (*__v0).flags }) & !flags;
unsafe {
(*__v0).flags = __v2;
}
}

fn memalloc_noio_save() -> u32 {
memalloc_flags_save(524288u32)
}

fn memalloc_noio_restore(flags: u32) {
memalloc_flags_restore(flags);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct xarray {
    xa_lock: spinlock,
    xa_flags: u32,
    xa_head: *mut (),
}


fn xa_init_flags(xa: &mut xarray, flags: u32) {
let mut __slate_compound_0: spinlock = unsafe { core::mem::zeroed() };
'__slate_break_1: loop {
spinlock_check(unsafe { core::ptr::addr_of_mut!((*(xa as *mut xarray)).xa_lock) });
unsafe {
*core::ptr::addr_of_mut!((*(xa as *mut xarray)).xa_lock) = *{
__slate_compound_0 = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = core::mem::zeroed();
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = core::mem::zeroed();
__t1.val = __SlateRecord_types_h_190 { counter: 0 };
__t1
} } };
__t0
} };
core::ptr::addr_of_mut!(__slate_compound_0)
};
}
if true {
break;
}
}
unsafe {
(*(xa as *mut xarray)).xa_flags = flags;
}
unsafe {
(*(xa as *mut xarray)).xa_head = core::ptr::null_mut::<()>();
}
}

fn xa_init(xa: &mut xarray) {
xa_init_flags(xa, 0u32);
}

// We encode errnos in the xas->xa_node.  If an error has happened, we need to
// drop the lock to fix it, and once we've done so the xa_state is invalid.
// XA_STATE() - Declare an XArray operation state.
// @name: Name of this operation state (usually xas).
// @array: Array to operate on.
// @index: Initial index of interest.
//
// Declare and initialise an xa_state on the stack.
// xas_for_each() - Iterate over a range of an XArray.
// @xas: XArray operation state.
// @entry: Entry retrieved from the array.
// @max: Maximum index to retrieve from array.
//
// The loop body will be executed for each entry present in the xarray
// between the current xas position and @max.  @entry will be set to
// the entry retrieved from the xarray.  It is safe to delete entries
// from the array in the loop body.  You should hold either the RCU lock
// or the xa_lock while iterating.  If you need to drop the lock, call
// xas_pause() first.
#[repr(C)]
#[derive(Clone, Copy)]
struct list_lru_one {
    list: list_head,
    nr_items: i64,
    lock: spinlock,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct list_lru_node {
    lru: list_lru_one,
    nr_items: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct list_lru {
    node: *mut list_lru_node,
}


#[inline(always)]
fn __acquire_ctx_lock__11(var: &__context_bitlock) {
}

#[inline(always)]
fn __acquire_shared_ctx_lock__11(var: &__context_bitlock) {
}

#[inline(always)]
fn __release_ctx_lock__11(var: &__context_bitlock) {
}

#[inline(always)]
fn __release_shared_ctx_lock__11(var: &__context_bitlock) {
}

#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

#[repr(C, align(32))]
struct __SlateAlign32<T>(T);

unsafe impl<T> Sync for __SlateAlign32<T> {
}

mod __slate_bits {
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits589U0 {
        #[bits(1)]
        pub for_kupdate: u32,
        #[bits(1)]
        pub for_background: u32,
        #[bits(1)]
        pub tagged_writepages: u32,
        #[bits(1)]
        pub range_cyclic: u32,
        #[bits(1)]
        pub for_sync: u32,
        #[bits(1)]
        pub unpinned_netfs_wb: u32,
        #[bits(1)]
        pub no_cgroup_owner: u32,
        #[bits(1, access = na)]
        pub __slate_pad_7: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits622U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits309U0 {
        #[bits(1)]
        pub dl_throttled: u32,
        #[bits(1)]
        pub dl_yielded: u32,
        #[bits(1)]
        pub dl_non_contending: u32,
        #[bits(1)]
        pub dl_overrun: u32,
        #[bits(1)]
        pub dl_server: u32,
        #[bits(1)]
        pub dl_server_active: u32,
        #[bits(1)]
        pub dl_defer: u32,
        #[bits(1)]
        pub dl_defer_armed: u32,
        #[bits(1)]
        pub dl_defer_running: u32,
        #[bits(1)]
        pub dl_defer_idle: u32,
        #[bits(1)]
        pub dl_bw_attached: u32,
        #[bits(5, access = na)]
        pub __slate_pad_11: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits69U0 {
        #[bits(1)]
        pub sched_reset_on_fork: u32,
        #[bits(1)]
        pub sched_contributes_to_load: u32,
        #[bits(1)]
        pub sched_migrated: u32,
        #[bits(1)]
        pub sched_task_hot: u32,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits69U1 {
        #[bits(1)]
        pub sched_remote_wakeup: u32,
        #[bits(1)]
        pub sched_rt_mutex: u32,
        #[bits(1)]
        pub in_execve: u32,
        #[bits(1)]
        pub in_iowait: u32,
        #[bits(1)]
        pub restore_sigmask: u32,
        #[bits(1)]
        pub no_cgroup_migration: u32,
        #[bits(1)]
        pub frozen: u32,
        #[bits(1)]
        pub use_memdelay: u32,
        #[bits(1)]
        pub in_eventfd: u32,
        #[bits(1)]
        pub pasid_activated: u32,
        #[bits(1)]
        pub reported_split_lock: u32,
        #[bits(1)]
        pub in_thrashing: u32,
        #[bits(1)]
        pub in_nf_duplicate: u32,
        #[bits(3, access = na)]
        pub __slate_pad_13: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits816U0 {
        #[bits(1)]
        pub flush_pending_idx: u32,
        #[bits(1)]
        pub flush_running_idx: u32,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits737U0 {
        #[bits(1)]
        pub can_wakeup: bool,
        #[bits(1)]
        pub async_suspend: bool,
        #[bits(1)]
        pub in_dpm_list: bool,
        #[bits(1)]
        pub is_prepared: bool,
        #[bits(1)]
        pub is_suspended: bool,
        #[bits(1)]
        pub is_noirq_suspended: bool,
        #[bits(1)]
        pub is_late_suspended: bool,
        #[bits(1)]
        pub no_pm: bool,
        #[bits(1)]
        pub early_init: bool,
        #[bits(1)]
        pub direct_complete: bool,
        #[bits(6, access = na)]
        pub __slate_pad_10: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits734U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits737U1 {
        #[bits(1)]
        pub wakeup_path: bool,
        #[bits(1)]
        pub syscore: bool,
        #[bits(1)]
        pub no_pm_callbacks: bool,
        #[bits(1)]
        pub smart_suspend: bool,
        #[bits(1)]
        pub must_resume: bool,
        #[bits(1)]
        pub may_skip_resume: bool,
        #[bits(1)]
        pub out_band_wakeup: bool,
        #[bits(1)]
        pub strict_midlayer: bool,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits737U2 {
        #[bits(3)]
        pub disable_depth: u32,
        #[bits(1)]
        pub idle_notification: bool,
        #[bits(1)]
        pub request_pending: bool,
        #[bits(1)]
        pub deferred_resume: bool,
        #[bits(1)]
        pub needs_force_resume: bool,
        #[bits(1)]
        pub runtime_auto: bool,
        #[bits(1)]
        pub ignore_children: bool,
        #[bits(1)]
        pub no_callbacks: bool,
        #[bits(1)]
        pub irq_safe: bool,
        #[bits(1)]
        pub use_autosuspend: bool,
        #[bits(1)]
        pub timer_autosuspends: bool,
        #[bits(1)]
        pub memalloc_noio: bool,
        #[bits(2, access = na)]
        pub __slate_pad_12: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits737U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits447U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits692U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits895U0 {
        #[bits(1)]
        pub early_init: bool,
        #[bits(1)]
        pub implicit_on_dfl: bool,
        #[bits(1)]
        pub threaded: bool,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits312U0 {
        #[bits(1)]
        pub is_child_subreaper: u32,
        #[bits(1)]
        pub has_child_subreaper: u32,
        #[bits(1)]
        pub autoreap: u32,
        #[bits(1)]
        pub timer_create_restore_ids: u32,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits69U2 {
        #[bits(1)]
        pub mce_ripv: u64,
        #[bits(1)]
        pub mce_whole_page: u64,
        #[bits(62)]
        pub __mce_reserved: u64,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits127U0 {
        #[bits(8)]
        pub base1: u16,
        #[bits(4)]
        pub r#type: u16,
        #[bits(1)]
        pub s: u16,
        #[bits(2)]
        pub dpl: u16,
        #[bits(1)]
        pub p: u16,
        #[bits(4)]
        pub limit1: u16,
        #[bits(1)]
        pub avl: u16,
        #[bits(1)]
        pub l: u16,
        #[bits(1)]
        pub d: u16,
        #[bits(1)]
        pub g: u16,
        #[bits(8)]
        pub base2: u16,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits135U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits635U0 {
        #[bits(1)]
        pub state_initialized: u32,
        #[bits(1)]
        pub state_in_sysfs: u32,
        #[bits(1)]
        pub state_add_uevent_sent: u32,
        #[bits(1)]
        pub state_remove_uevent_sent: u32,
        #[bits(1)]
        pub uevent_suppress: u32,
        #[bits(3, access = na)]
        pub __slate_pad_5: u8,
    }
    #[bitfields::bitfield([u8; 2], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits134U0 {
        #[bits(4)]
        pub sp_reg: u32,
        #[bits(4)]
        pub bp_reg: u32,
        #[bits(3)]
        pub r#type: u32,
        #[bits(1)]
        pub signal: u32,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits151U0 {
        #[bits(24)]
        pub intel_native_model_id: u32,
        #[bits(8)]
        pub intel_type: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits152U0 {
        #[bits(16)]
        pub amd_num_processors: u32,
        #[bits(8)]
        pub amd_power_eff_ranking: u32,
        #[bits(4)]
        pub amd_native_model_id: u32,
        #[bits(4)]
        pub amd_type: u32,
    }
    #[bitfields::bitfield([u8; 16], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits102U0 {
        #[bits(32)]
        pub max_std_leaf: u32,
        #[bits(32)]
        pub cpu_vendorid_0: u32,
        #[bits(32)]
        pub cpu_vendorid_2: u32,
        #[bits(32)]
        pub cpu_vendorid_1: u32,
    }
    #[bitfields::bitfield([u8; 16], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits103U0 {
        #[bits(4)]
        pub stepping: u32,
        #[bits(4)]
        pub base_model: u32,
        #[bits(4)]
        pub base_family_id: u32,
        #[bits(2)]
        pub cpu_type: u32,
        #[bits(2, access = na)]
        pub __slate_pad_4: u8,
        #[bits(4)]
        pub ext_model: u32,
        #[bits(8)]
        pub ext_family: u32,
        #[bits(4, access = na)]
        pub __slate_pad_7: u8,
        #[bits(8)]
        pub brand_id: u32,
        #[bits(8)]
        pub clflush_size: u32,
        #[bits(8)]
        pub n_logical_cpu: u32,
        #[bits(8)]
        pub local_apic_id: u32,
        #[bits(1)]
        pub sse3: u32,
        #[bits(1)]
        pub pclmulqdq: u32,
        #[bits(1)]
        pub dtes64: u32,
        #[bits(1)]
        pub monitor: u32,
        #[bits(1)]
        pub dscpl: u32,
        #[bits(1)]
        pub vmx: u32,
        #[bits(1)]
        pub smx: u32,
        #[bits(1)]
        pub est: u32,
        #[bits(1)]
        pub tm2: u32,
        #[bits(1)]
        pub ssse3: u32,
        #[bits(1)]
        pub cntxt_id: u32,
        #[bits(1)]
        pub sdbg: u32,
        #[bits(1)]
        pub fma: u32,
        #[bits(1)]
        pub cx16: u32,
        #[bits(1)]
        pub xtpr_update: u32,
        #[bits(1)]
        pub pdcm: u32,
        #[bits(1, access = na)]
        pub __slate_pad_28: u8,
        #[bits(1)]
        pub pcid: u32,
        #[bits(1)]
        pub dca: u32,
        #[bits(1)]
        pub sse4_1: u32,
        #[bits(1)]
        pub sse4_2: u32,
        #[bits(1)]
        pub x2apic: u32,
        #[bits(1)]
        pub movbe: u32,
        #[bits(1)]
        pub popcnt: u32,
        #[bits(1)]
        pub tsc_deadline_timer: u32,
        #[bits(1)]
        pub aes: u32,
        #[bits(1)]
        pub xsave: u32,
        #[bits(1)]
        pub osxsave: u32,
        #[bits(1)]
        pub avx: u32,
        #[bits(1)]
        pub f16c: u32,
        #[bits(1)]
        pub rdrand: u32,
        #[bits(1)]
        pub guest_status: u32,
        #[bits(1)]
        pub fpu: u32,
        #[bits(1)]
        pub vme: u32,
        #[bits(1)]
        pub de: u32,
        #[bits(1)]
        pub pse: u32,
        #[bits(1)]
        pub tsc: u32,
        #[bits(1)]
        pub msr: u32,
        #[bits(1)]
        pub pae: u32,
        #[bits(1)]
        pub mce: u32,
        #[bits(1)]
        pub cx8: u32,
        #[bits(1)]
        pub apic: u32,
        #[bits(1, access = na)]
        pub __slate_pad_54: u8,
        #[bits(1)]
        pub sep: u32,
        #[bits(1)]
        pub mtrr: u32,
        #[bits(1)]
        pub pge: u32,
        #[bits(1)]
        pub mca: u32,
        #[bits(1)]
        pub cmov: u32,
        #[bits(1)]
        pub pat: u32,
        #[bits(1)]
        pub pse36: u32,
        #[bits(1)]
        pub psn: u32,
        #[bits(1)]
        pub clflush: u32,
        #[bits(1, access = na)]
        pub __slate_pad_64: u8,
        #[bits(1)]
        pub ds: u32,
        #[bits(1)]
        pub acpi: u32,
        #[bits(1)]
        pub mmx: u32,
        #[bits(1)]
        pub fxsr: u32,
        #[bits(1)]
        pub sse: u32,
        #[bits(1)]
        pub sse2: u32,
        #[bits(1)]
        pub selfsnoop: u32,
        #[bits(1)]
        pub htt: u32,
        #[bits(1)]
        pub tm: u32,
        #[bits(1)]
        pub ia64: u32,
        #[bits(1)]
        pub pbe: u32,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits101U0 {
        #[bits(1)]
        pub initialized: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits246U0 {
        #[bits(1)]
        pub faultable: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
}

#[inline(always)]
fn __assume_ctx_lock__11(var: &__context_bitlock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __context_bitlock {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hlist_bl_node {
    next: *mut hlist_bl_node,
    pprev: *mut *mut hlist_bl_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uuid_h_21 {
    b: [u8; 16],
}


// The length of a UUID string ("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
// not including trailing NUL.
fn __list_add_rcu(new: *mut list_head, prev: *mut list_head, next: *mut list_head) {
if !__list_add_valid(new, prev, next) {
return;
}
unsafe {
(*new).next = next;
}
unsafe {
(*new).prev = prev;
}
disable_context_analysis!();
let _r_a_p__v: u64 = new as u64;
if false && _r_a_p__v == 0 {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_228() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!(*unsafe { core::ptr::addr_of_mut!((*prev).next) }, _r_a_p__v as *mut list_head);
if true {
break;
}
}
} else {
loop {
kcsan_release!();
loop {
loop {
if !__native_word!() {
unsafe { __compiletime_assert_229() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_230() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!(*unsafe { core::ptr::addr_of_mut!((*prev).next) }, RCU_INITIALIZER!(_r_a_p__v as *mut list_head));
if true {
break;
}
}
if true {
break;
}
}
if true {
break;
}
}
}
unsafe {
(*next).prev = new;
}
}

fn list_add_tail_rcu(new: *mut list_head, head: *mut list_head) {
__list_add_rcu(new, unsafe { (*head).prev }, head);
}

fn list_del_rcu(entry: *mut list_head) {
__list_del_entry(entry);
unsafe {
(*entry).prev = ((290i32 as *const u8).wrapping_offset(_AC__2!() as isize) as *mut ()) as *mut list_head;
}
}

// list_entry_rcu - get the struct for this entry
// @ptr:        the &struct list_head pointer.
// @type:       the type of the struct this is embedded in.
// @member:     the name of the list_head within the struct.
//
// This primitive may safely run concurrently with the _rcu list-mutation
// primitives such as list_add_rcu() as long as it's guarded by rcu_read_lock().
// list_next_or_null_rcu - get the next element from a list
// @head:	the head for the list.
// @ptr:        the list head to take the next element from.
// @type:       the type of the struct this is embedded in.
// @member:     the name of the list_head within the struct.
//
// Note that if the ptr is at the end of the list, NULL is returned.
//
// This primitive may safely run concurrently with the _rcu list-mutation
// primitives such as list_add_rcu() as long as it's guarded by rcu_read_lock().
// list_for_each_entry_rcu	-	iterate over rcu list of given type
// @pos:	the type * to use as a loop cursor.
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
// @cond:	optional lockdep expression if called from non-RCU protection.
//
// This list-traversal primitive may safely run concurrently with
// the _rcu list-mutation primitives such as list_add_rcu()
// as long as the traversal is guarded by rcu_read_lock().
// return the first or the next element in an RCU protected hlist
#[repr(C)]
#[derive(Clone, Copy)]
struct exception_table_entry {
    insn: i32,
    fixup: i32,
    data: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_capability_h_24 {
    val: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array {
    root: *mut assoc_array_ptr,
    nr_leaves_on_tree: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key_tag {
    rcu: callback_head,
    usage: refcount_struct,
    removed: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct keyring_index_key {
    hash: u64,
    __slate_anon_1: __SlateRecord_key_h_117,
    r#type: *mut key_type,
    domain_tag: *mut key_tag,
    description: *const u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union key_payload {
    rcu_data0: *mut (),
    data: [*mut (); 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key_restriction {
    check: Option<unsafe extern "C" fn(*mut key, *const key_type, *const key_payload, *mut key) -> i32>,
    key: *mut key,
    keytype: *mut key_type,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key {
    usage: refcount_struct,
    serial: i32,
    __slate_anon_2: __SlateRecord_key_h_198,
    sem: rw_semaphore,
    user: *mut key_user,
    security: *mut (),
    __slate_anon_6: __SlateRecord_key_h_208,
    last_used_at: i64,
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    perm: u32,
    quotalen: u16,
    datalen: u16,
    state: i16,
    flags: u64,
    __slate_anon_15: __SlateRecord_key_h_246,
    __slate_anon_16: __SlateRecord_key_h_261,
    restrict_link: *mut key_restriction,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct user_struct {
    __count: refcount_struct,
    epoll_watches: percpu_counter,
    unix_inflight: u64,
    pipe_bufs: __SlateRecord_types_h_197,
    uidhash_node: hlist_node,
    uid: __SlateRecord_uidgid_types_h_9,
    locked_vm: __SlateRecord_types_h_197,
    ratelimit: ratelimit_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct group_info {
    usage: refcount_struct,
    ngroups: i32,
    gid: [__SlateRecord_uidgid_types_h_13; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cred {
    usage: __SlateRecord_types_h_197,
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    suid: __SlateRecord_uidgid_types_h_9,
    sgid: __SlateRecord_uidgid_types_h_13,
    euid: __SlateRecord_uidgid_types_h_9,
    egid: __SlateRecord_uidgid_types_h_13,
    fsuid: __SlateRecord_uidgid_types_h_9,
    fsgid: __SlateRecord_uidgid_types_h_13,
    securebits: u32,
    cap_inheritable: __SlateRecord_capability_h_24,
    cap_permitted: __SlateRecord_capability_h_24,
    cap_effective: __SlateRecord_capability_h_24,
    cap_bset: __SlateRecord_capability_h_24,
    cap_ambient: __SlateRecord_capability_h_24,
    jit_keyring: u8,
    session_keyring: *mut key,
    process_keyring: *mut key,
    thread_keyring: *mut key,
    request_key_auth: *mut key,
    security: *mut (),
    user: *mut user_struct,
    user_ns: *mut user_namespace,
    ucounts: *mut ucounts,
    group_info: *mut group_info,
    __slate_anon_25: __SlateRecord_cred_h_147,
}


// current_cred - Access the current task's subjective credentials
//
// Access the subjective credentials of the current task.  RCU-safe,
// since nobody else can modify it.
#[repr(C)]
#[derive(Clone, Copy)]
struct upid {
    nr: i32,
    ns: *mut pid_namespace,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pid {
    count: refcount_struct,
    level: u32,
    lock: spinlock,
    __slate_anon_3: __SlateRecord_pid_h_62,
    tasks: [hlist_head; 4],
    inodes: hlist_head,
    wait_pidfd: wait_queue_head,
    rcu: callback_head,
    numbers: [upid; 0],
}


fn hrtimer_get_expires(timer: *const hrtimer) -> i64 {
unsafe { (*timer).node.expires }
}

fn hrtimer_get_softexpires(timer: *const hrtimer) -> i64 {
unsafe { (*timer)._softexpires }
}

fn hrtimer_start_expires(timer: *mut hrtimer, mode: u32) {
let soft: i64 = hrtimer_get_softexpires(timer as *const hrtimer);
let hard: i64 = hrtimer_get_expires(timer as *const hrtimer);
let delta: u64 = ktime_to_ns(ktime_sub!(hard, soft)) as u64;
unsafe { hrtimer_start_range_ns(timer, soft, delta, mode) };
}

fn hrtimer_restart(timer: *mut hrtimer) {
hrtimer_start_expires(timer, 0u32);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct sighand_struct {
    siglock: spinlock,
    count: refcount_struct,
    signalfd_wqh: wait_queue_head,
    action: [k_sigaction; 64],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pacct_struct {
    ac_flag: i32,
    ac_exitcode: i64,
    ac_mem: u64,
    ac_utime: u64,
    ac_stime: u64,
    ac_minflt: u64,
    ac_majflt: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cpu_itimer {
    expires: u64,
    incr: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct task_cputime_atomic {
    utime: __SlateRecord_types_h_197,
    stime: __SlateRecord_types_h_197,
    sum_exec_runtime: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct thread_group_cputimer {
    cputime_atomic: task_cputime_atomic,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct core_thread {
    task: *mut task_struct,
    next: *mut core_thread,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct core_state {
    nr_threads: __SlateRecord_types_h_190,
    dumper: core_thread,
    startup: completion,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct signal_struct {
    sigcnt: refcount_struct,
    live: __SlateRecord_types_h_190,
    nr_threads: i32,
    quick_threads: i32,
    thread_head: list_head,
    wait_chldexit: wait_queue_head,
    curr_target: *mut task_struct,
    shared_pending: sigpending,
    multiprocess: hlist_head,
    group_exit_code: i32,
    notify_count: i32,
    group_exec_task: *mut task_struct,
    group_stop_count: i32,
    flags: u32,
    core_state: *mut core_state,
    __slate_bits_0: __slate_bits::__SlateBits312U0,
    next_posix_timer_id: __SlateRecord_types_h_190,
    posix_timers: hlist_head,
    ignored_posix_timers: hlist_head,
    real_timer: hrtimer,
    it_real_incr: i64,
    it: [cpu_itimer; 2],
    cputimer: thread_group_cputimer,
    posix_cputimers: posix_cputimers,
    pids: [*mut pid; 4],
    tty_old_pgrp: *mut pid,
    leader: i32,
    tty: *mut tty_struct,
    stats_lock: seqlock,
    utime: u64,
    stime: u64,
    cutime: u64,
    cstime: u64,
    gtime: u64,
    cgtime: u64,
    prev_cputime: prev_cputime,
    nvcsw: u64,
    nivcsw: u64,
    cnvcsw: u64,
    cnivcsw: u64,
    min_flt: u64,
    maj_flt: u64,
    cmin_flt: u64,
    cmaj_flt: u64,
    inblock: u64,
    oublock: u64,
    cinblock: u64,
    coublock: u64,
    maxrss: u64,
    cmaxrss: u64,
    ioac: task_io_accounting,
    sum_sched_runtime: u64,
    rlim: [rlimit; 16],
    pacct: pacct_struct,
    stats: *mut taskstats,
    audit_tty: u32,
    tty_audit_buf: *mut tty_audit_buf,
    cgroup_threadgroup_rwsem: rw_semaphore,
    oom_flag_origin: bool,
    oom_score_adj: i16,
    oom_score_adj_min: i16,
    oom_mm: *mut mm_struct,
    cred_guard_mutex: mutex,
    exec_update_lock: rw_semaphore,
}


// group exit in progress
fn task_sigpending(p: *mut task_struct) -> i32 {
unlikely!(test_tsk_thread_flag(p, 1i32)) as i32
}

fn task_rlimit(task: *const task_struct, limit: u32) -> u64 {
let mut __v0: u64;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_361() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*((*(*task).signal).rlim.as_mut_ptr() as *mut rlimit).wrapping_offset(limit as isize)).rlim_cur) as *const u64) };
__v0
}

fn rlimit(limit: u32) -> u64 {
task_rlimit(get_current() as *const task_struct, limit)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rcu_sync {
    gp_state: i32,
    gp_count: i32,
    gp_wait: wait_queue_head,
    cb_head: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_rw_semaphore {
    rss: rcu_sync,
    read_count: *mut u32,
    writer: rcuwait,
    waiters: wait_queue_head,
    block: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_projid_h_24 {
    val: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kqid {
    __slate_anon_0: __SlateRecord_quota_h_69,
    r#type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mem_dqblk {
    dqb_bhardlimit: i64,
    dqb_bsoftlimit: i64,
    dqb_curspace: i64,
    dqb_rsvspace: i64,
    dqb_ihardlimit: i64,
    dqb_isoftlimit: i64,
    dqb_curinodes: i64,
    dqb_btime: i64,
    dqb_itime: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mem_dqinfo {
    dqi_format: *mut quota_format_type,
    dqi_fmt_id: i32,
    dqi_dirty_list: list_head,
    dqi_flags: u64,
    dqi_bgrace: u32,
    dqi_igrace: u32,
    dqi_max_spc_limit: i64,
    dqi_max_ino_limit: i64,
    dqi_priv: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dquot {
    dq_hash: hlist_node,
    dq_inuse: list_head,
    dq_free: list_head,
    dq_dirty: list_head,
    dq_lock: mutex,
    dq_dqb_lock: spinlock,
    dq_count: __SlateRecord_types_h_190,
    dq_sb: *mut super_block,
    dq_id: kqid,
    dq_off: i64,
    dq_flags: u64,
    dq_dqb: mem_dqblk,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct quota_format_ops {
    check_quota_file: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    read_file_info: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    write_file_info: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    free_file_info: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    read_dqblk: Option<unsafe extern "C" fn(*mut dquot) -> i32>,
    commit_dqblk: Option<unsafe extern "C" fn(*mut dquot) -> i32>,
    release_dqblk: Option<unsafe extern "C" fn(*mut dquot) -> i32>,
    get_next_id: Option<unsafe extern "C" fn(*mut super_block, *mut kqid) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dquot_operations {
    write_dquot: Option<unsafe extern "C" fn(*mut dquot) -> i32>,
    alloc_dquot: Option<unsafe extern "C" fn(*mut super_block, i32) -> *mut dquot>,
    destroy_dquot: Option<unsafe extern "C" fn(*mut dquot)>,
    acquire_dquot: Option<unsafe extern "C" fn(*mut dquot) -> i32>,
    release_dquot: Option<unsafe extern "C" fn(*mut dquot) -> i32>,
    mark_dirty: Option<unsafe extern "C" fn(*mut dquot) -> i32>,
    write_info: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    get_reserved_space: Option<unsafe extern "C" fn(*mut inode) -> *mut i64>,
    get_projid: Option<unsafe extern "C" fn(*mut inode, *mut __SlateRecord_projid_h_24) -> i32>,
    get_inode_usage: Option<unsafe extern "C" fn(*mut inode, *mut i64) -> i32>,
    get_next_id: Option<unsafe extern "C" fn(*mut super_block, *mut kqid) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct qc_dqblk {
    d_fieldmask: i32,
    d_spc_hardlimit: u64,
    d_spc_softlimit: u64,
    d_ino_hardlimit: u64,
    d_ino_softlimit: u64,
    d_space: u64,
    d_ino_count: u64,
    d_ino_timer: i64,
    d_spc_timer: i64,
    d_ino_warns: i32,
    d_spc_warns: i32,
    d_rt_spc_hardlimit: u64,
    d_rt_spc_softlimit: u64,
    d_rt_space: u64,
    d_rt_spc_timer: i64,
    d_rt_spc_warns: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct qc_type_state {
    flags: u32,
    spc_timelimit: u32,
    ino_timelimit: u32,
    rt_spc_timelimit: u32,
    spc_warnlimit: u32,
    ino_warnlimit: u32,
    rt_spc_warnlimit: u32,
    ino: u64,
    blocks: u64,
    nextents: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct qc_state {
    s_incoredqs: u32,
    s_state: [qc_type_state; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct qc_info {
    i_fieldmask: i32,
    i_flags: u32,
    i_spc_timelimit: u32,
    i_ino_timelimit: u32,
    i_rt_spc_timelimit: u32,
    i_spc_warnlimit: u32,
    i_ino_warnlimit: u32,
    i_rt_spc_warnlimit: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct quotactl_ops {
    quota_on: Option<unsafe extern "C" fn(*mut super_block, i32, i32, *const path) -> i32>,
    quota_off: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    quota_enable: Option<unsafe extern "C" fn(*mut super_block, u32) -> i32>,
    quota_disable: Option<unsafe extern "C" fn(*mut super_block, u32) -> i32>,
    quota_sync: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    set_info: Option<unsafe extern "C" fn(*mut super_block, i32, *mut qc_info) -> i32>,
    get_dqblk: Option<unsafe extern "C" fn(*mut super_block, kqid, *mut qc_dqblk) -> i32>,
    get_nextdqblk: Option<unsafe extern "C" fn(*mut super_block, *mut kqid, *mut qc_dqblk) -> i32>,
    set_dqblk: Option<unsafe extern "C" fn(*mut super_block, kqid, *mut qc_dqblk) -> i32>,
    get_state: Option<unsafe extern "C" fn(*mut super_block, *mut qc_state) -> i32>,
    rm_xquota: Option<unsafe extern "C" fn(*mut super_block, u32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct quota_format_type {
    qf_fmt_id: i32,
    qf_ops: *const quota_format_ops,
    qf_owner: *mut module,
    qf_next: *mut quota_format_type,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct quota_info {
    flags: u32,
    dqio_sem: rw_semaphore,
    files: [*mut inode; 3],
    info: [mem_dqinfo; 3],
    ops: [*const quota_format_ops; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sb_writers {
    frozen: u16,
    freeze_kcount: i32,
    freeze_ucount: i32,
    freeze_owner: *const (),
    rw_sem: [percpu_rw_semaphore; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct super_operations {
    alloc_inode: Option<unsafe extern "C" fn(*mut super_block) -> *mut inode>,
    destroy_inode: Option<unsafe extern "C" fn(*mut inode)>,
    free_inode: Option<unsafe extern "C" fn(*mut inode)>,
    dirty_inode: Option<unsafe extern "C" fn(*mut inode, i32)>,
    write_inode: Option<unsafe extern "C" fn(*mut inode, *mut writeback_control) -> i32>,
    sync_inode_metadata: Option<unsafe extern "C" fn(*mut inode, *mut writeback_control) -> i32>,
    drop_inode: Option<unsafe extern "C" fn(*mut inode) -> i32>,
    evict_inode: Option<unsafe extern "C" fn(*mut inode)>,
    put_super: Option<unsafe extern "C" fn(*mut super_block)>,
    sync_fs: Option<unsafe extern "C" fn(*mut super_block, i32) -> i32>,
    freeze_super: Option<unsafe extern "C" fn(*mut super_block, u32, *const ()) -> i32>,
    freeze_fs: Option<unsafe extern "C" fn(*mut super_block) -> i32>,
    thaw_super: Option<unsafe extern "C" fn(*mut super_block, u32, *const ()) -> i32>,
    unfreeze_fs: Option<unsafe extern "C" fn(*mut super_block) -> i32>,
    statfs: Option<unsafe extern "C" fn(*mut dentry, *mut kstatfs) -> i32>,
    umount_begin: Option<unsafe extern "C" fn(*mut super_block)>,
    show_options: Option<unsafe extern "C" fn(*mut seq_file, *mut dentry) -> i32>,
    show_devname: Option<unsafe extern "C" fn(*mut seq_file, *mut dentry) -> i32>,
    show_path: Option<unsafe extern "C" fn(*mut seq_file, *mut dentry) -> i32>,
    show_stats: Option<unsafe extern "C" fn(*mut seq_file, *mut dentry) -> i32>,
    quota_read: Option<unsafe extern "C" fn(*mut super_block, i32, *mut u8, u64, i64) -> i64>,
    quota_write: Option<unsafe extern "C" fn(*mut super_block, i32, *const u8, u64, i64) -> i64>,
    get_dquots: Option<unsafe extern "C" fn(*mut inode) -> *mut *mut dquot>,
    nr_cached_objects: Option<unsafe extern "C" fn(*mut super_block, *mut shrink_control) -> i64>,
    free_cached_objects: Option<unsafe extern "C" fn(*mut super_block, *mut shrink_control) -> i64>,
    remove_bdev: Option<unsafe extern "C" fn(*mut super_block, *mut block_device) -> i32>,
    shutdown: Option<unsafe extern "C" fn(*mut super_block)>,
    report_error: Option<unsafe extern "C" fn(*const fserror_event)>,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct super_block {
    s_list: list_head,
    s_dev: u32,
    s_super_dev: *mut super_dev,
    s_blocksize_bits: u8,
    s_blocksize: u64,
    s_maxbytes: i64,
    s_type: *mut file_system_type,
    s_op: *const super_operations,
    dq_op: *const dquot_operations,
    s_qcop: *const quotactl_ops,
    s_export_op: *const export_operations,
    s_flags: u64,
    s_iflags: u64,
    s_magic: u64,
    s_root: *mut dentry,
    s_umount: rw_semaphore,
    s_passive: refcount_struct,
    s_active: __SlateRecord_types_h_190,
    s_security: *mut (),
    s_xattr: *const *const xattr_handler,
    s_roots: hlist_head,
    s_roots_lock: spinlock,
    s_mounts: *mut mount,
    s_bdev: *mut block_device,
    s_bdev_file: *mut file,
    s_bdi: *mut backing_dev_info,
    s_mtd: *mut mtd_info,
    s_instances: hlist_node,
    s_quota_types: u32,
    s_dquot: quota_info,
    s_writers: sb_writers,
    s_fs_info: *mut (),
    s_time_gran: u32,
    s_time_min: i64,
    s_time_max: i64,
    s_fsnotify_mask: u32,
    s_fsnotify_info: *mut fsnotify_sb_info,
    s_id: [u8; 32],
    s_uuid: __SlateRecord_uuid_h_21,
    s_uuid_len: u8,
    s_sysfs_name: [u8; 37],
    s_max_links: u32,
    s_d_flags: u32,
    s_vfs_rename_mutex: mutex,
    s_subtype: *const u8,
    __s_d_op: *const dentry_operations,
    s_shrink: *mut shrinker,
    s_remove_count: __SlateRecord_types_h_197,
    s_readonly_remount: i32,
    s_wb_err: u32,
    s_dio_done_wq: *mut workqueue_struct,
    s_pins: hlist_head,
    s_user_ns: *mut user_namespace,
    s_dentry_lru: list_lru,
    s_inode_lru: list_lru,
    rcu: callback_head,
    destroy_work: work_struct,
    s_sync_lock: mutex,
    s_stack_depth: i32,
    __slate_pad_59: [u8; 12],
    s_inode_list_lock: spinlock,
    s_inodes: list_head,
    s_inode_wblist_lock: spinlock,
    s_inodes_wb: list_head,
    s_min_writeback_pages: i64,
    s_pending_errors: refcount_struct,
}


// sb->s_flags.  Note that these mirror the equivalent MS_* flags where
// represented in both.
// Mount read-only
// Update the on-disk [acm]times lazily
#[repr(C)]
#[derive(Clone, Copy)]
struct lockref {
    __slate_anon_0: __SlateRecord_lockref_h_26,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct qstr {
    __slate_anon_0: __SlateRecord_dcache_h_50,
    name: *const u8,
}


// 192 bytes
#[repr(C)]
#[derive(Clone, Copy)]
union shortname_store {
    string: [u8; 40],
    words: [u64; 5],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dentry {
    d_flags: u32,
    d_seq: seqcount_spinlock,
    d_hash: hlist_bl_node,
    d_parent: *mut dentry,
    __slate_anon_4: __SlateRecord_dcache_h_99,
    d_inode: *mut inode,
    d_shortname: shortname_store,
    d_op: *const dentry_operations,
    d_sb: *mut super_block,
    d_time: u64,
    d_fsdata: *mut (),
    d_lockref: lockref,
    d_lru: list_head,
    d_sib: hlist_node,
    d_children: hlist_head,
    __slate_anon_15: __SlateRecord_dcache_h_126,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct dentry_operations {
    d_revalidate: Option<unsafe extern "C" fn(*mut inode, *const qstr, *mut dentry, u32) -> i32>,
    d_weak_revalidate: Option<unsafe extern "C" fn(*mut dentry, u32) -> i32>,
    d_hash: Option<unsafe extern "C" fn(*const dentry, *mut qstr) -> i32>,
    d_compare: Option<unsafe extern "C" fn(*const dentry, u32, *const u8, *const qstr) -> i32>,
    d_delete: Option<unsafe extern "C" fn(*const dentry) -> i32>,
    d_init: Option<unsafe extern "C" fn(*mut dentry) -> i32>,
    d_release: Option<unsafe extern "C" fn(*mut dentry)>,
    d_prune: Option<unsafe extern "C" fn(*mut dentry)>,
    d_iput: Option<unsafe extern "C" fn(*mut dentry, *mut inode)>,
    d_dname: Option<unsafe extern "C" fn(*mut dentry, *mut u8, i32) -> *mut u8>,
    d_automount: Option<unsafe extern "C" fn(*mut path) -> *mut vfsmount>,
    d_manage: Option<unsafe extern "C" fn(*const path, bool) -> i32>,
    d_real: Option<unsafe extern "C" fn(*mut dentry, u32) -> *mut dentry>,
    d_unalias_trylock: Option<unsafe extern "C" fn(*const dentry) -> bool>,
    d_unalias_unlock: Option<unsafe extern "C" fn(*const dentry)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct path {
    mnt: *mut vfsmount,
    dentry: *mut dentry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct delayed_call {
    r#fn: Option<unsafe extern "C" fn(*mut ())>,
    arg: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_cq {
    q: *mut request_queue,
    ioc: *mut io_context,
    __slate_anon_2: __SlateRecord_iocontext_h_83,
    __slate_anon_3: __SlateRecord_iocontext_h_87,
    flags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_context {
    refcount: __SlateRecord_types_h_197,
    active_ref: __SlateRecord_types_h_190,
    ioprio: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vfsmount {
    mnt_root: *mut dentry,
    mnt_sb: *mut super_block,
    mnt_flags: i32,
    mnt_idmap: *mut mnt_idmap,
}


fn mnt_idmap(mnt: &vfsmount) -> *mut mnt_idmap {
// Pairs with smp_store_release() in do_idmap_mount().
let mut __v0: *mut mnt_idmap;
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_386() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*(mnt as *const vfsmount)).mnt_idmap) as *const *mut mnt_idmap) };
__v0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mnt_idmapping_h_17 {
    val: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mnt_idmapping_h_21 {
    val: u32,
}


// vfs{g,u}ids are created from k{g,u}ids.
// We don't allow them to be created from regular {u,g}id.
// Allow a vfs{g,u}id to be used as a k{g,u}id where we want to compare
// whether the mapped value is identical to value of a k{g,u}id.
#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_ref_data {
    count: __SlateRecord_types_h_197,
    release: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    confirm_switch: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    __slate_bits_0: __slate_bits::__SlateBits622U0,
    rcu: callback_head,
    r#ref: *mut percpu_ref,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_ref {
    percpu_count_ptr: u64,
    data: *mut percpu_ref_data,
}


fn percpu_ref_kill(r#ref: *mut percpu_ref) {
unsafe { percpu_ref_kill_and_confirm(r#ref, None) };
}

fn __ref_is_percpu(r#ref: *mut percpu_ref, percpu_countp: &mut *mut u64) -> bool {
let mut percpu_ptr: u64 = 0;
// The value of @ref->percpu_count_ptr is tested for
// !__PERCPU_REF_ATOMIC, which may be set asynchronously, and then
// used as a pointer.  If the compiler generates a separate fetch
// when using it as a pointer, __PERCPU_REF_ATOMIC may be set in
// between contaminating the pointer value, meaning that
// READ_ONCE() is required when fetching it.
//
// The dependency ordering from the READ_ONCE() pairs
// with smp_store_release() in __percpu_ref_switch_to_percpu().
let mut __v0: u64;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_387() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*r#ref).percpu_count_ptr) as *const u64) };
percpu_ptr = __v0;
// Theoretically, the following could test just ATOMIC; however,
// then we'd have to mask off DEAD separately as DEAD may be
// visible without ATOMIC if we race with percpu_ref_kill().  DEAD
// implies ATOMIC anyway.  Test them together.
if unlikely__4!(percpu_ptr & 3) != 0 {
return false;
}
unsafe {
*(percpu_countp as *mut *mut u64) = percpu_ptr as *mut u64;
}
true
}

fn percpu_ref_get_many(r#ref: *mut percpu_ref, nr: u64) {
let mut percpu_count: *mut u64 = unsafe { core::mem::zeroed() };
rcu_read_lock();
if __ref_is_percpu(r#ref, unsafe { &mut (*core::ptr::addr_of_mut!(percpu_count)) }) {
loop {
__verify_pcpu_ptr__2!();
match (const { core::mem::size_of::<u64>() }) as u64 {
1 => {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(nr) }) && (nr == 1 || nr == (1i32.wrapping_neg() as u64)) { nr as i32 } else { 0 };
if false {
let pao_tmp__: u64 = nr;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incb %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decb %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u8 = (nr & 255) as u8;
if false {
let pto_tmp__: u64 = nr;
}
unsafe {
core::arch::asm!("addb {1:l}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
2 => {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(nr) }) && (nr == 1 || nr == (1i32.wrapping_neg() as u64)) { nr as i32 } else { 0 };
if false {
let pao_tmp__: u64 = nr;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incw %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decw %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u16 = (nr & 65535) as u16;
if false {
let pto_tmp__: u64 = nr;
}
unsafe {
core::arch::asm!("addw {1:x}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) pto_val__ as u16, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
4 => {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(nr) }) && (nr == 1 || nr == (1i32.wrapping_neg() as u64)) { nr as i32 } else { 0 };
if false {
let pao_tmp__: u64 = nr;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incl %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decl %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u32 = (nr & 4294967295) as u32;
if false {
let pto_tmp__: u64 = nr;
}
unsafe {
core::arch::asm!("addl {1:e}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) pto_val__ as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
8 => {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(nr) }) && (nr == 1 || nr == (1i32.wrapping_neg() as u64)) { nr as i32 } else { 0 };
if false {
let pao_tmp__: u64 = nr;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incq %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decq %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u64 = nr;
if false {
let pto_tmp__: u64 = nr;
}
unsafe {
core::arch::asm!("addq {1:r}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) pto_val__ as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
if true {
break;
}
}
} else {
atomic_long_add(nr as i64, unsafe { core::ptr::addr_of_mut!((*(*r#ref).data).count) });
}
rcu_read_unlock();
}

fn percpu_ref_get(r#ref: *mut percpu_ref) {
percpu_ref_get_many(r#ref, 1u64);
}

fn percpu_ref_tryget_many(r#ref: *mut percpu_ref, nr: u64) -> bool {
let mut percpu_count: *mut u64 = unsafe { core::mem::zeroed() };
let mut ret: bool = false;
rcu_read_lock();
if __ref_is_percpu(r#ref, unsafe { &mut (*core::ptr::addr_of_mut!(percpu_count)) }) {
loop {
__verify_pcpu_ptr__2!();
match (const { core::mem::size_of::<u64>() }) as u64 {
1 => {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(nr) }) && (nr == 1 || nr == (1i32.wrapping_neg() as u64)) { nr as i32 } else { 0 };
if false {
let pao_tmp__: u64 = nr;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incb %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decb %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u8 = (nr & 255) as u8;
if false {
let pto_tmp__: u64 = nr;
}
unsafe {
core::arch::asm!("addb {1:l}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
2 => {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(nr) }) && (nr == 1 || nr == (1i32.wrapping_neg() as u64)) { nr as i32 } else { 0 };
if false {
let pao_tmp__: u64 = nr;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incw %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decw %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u16 = (nr & 65535) as u16;
if false {
let pto_tmp__: u64 = nr;
}
unsafe {
core::arch::asm!("addw {1:x}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) pto_val__ as u16, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
4 => {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(nr) }) && (nr == 1 || nr == (1i32.wrapping_neg() as u64)) { nr as i32 } else { 0 };
if false {
let pao_tmp__: u64 = nr;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incl %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decl %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u32 = (nr & 4294967295) as u32;
if false {
let pto_tmp__: u64 = nr;
}
unsafe {
core::arch::asm!("addl {1:e}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) pto_val__ as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
8 => {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(nr) }) && (nr == 1 || nr == (1i32.wrapping_neg() as u64)) { nr as i32 } else { 0 };
if false {
let pao_tmp__: u64 = nr;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incq %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decq %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u64 = nr;
if false {
let pto_tmp__: u64 = nr;
}
unsafe {
core::arch::asm!("addq {1:r}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) pto_val__ as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
if true {
break;
}
}
ret = true;
} else {
ret = atomic_long_add_unless(unsafe { core::ptr::addr_of_mut!((*(*r#ref).data).count) }, nr as i64, 0i64);
}
rcu_read_unlock();
ret
}

fn percpu_ref_tryget(r#ref: *mut percpu_ref) -> bool {
percpu_ref_tryget_many(r#ref, 1u64)
}

fn percpu_ref_tryget_live_rcu(r#ref: *mut percpu_ref) -> bool {
let mut percpu_count: *mut u64 = unsafe { core::mem::zeroed() };
let mut ret: bool = false;
let mut __v0: i64;
let __ret_warn_on: i32 = (rcu_read_lock_held() == 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("388: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 388b - ., 3; .popsection\n# {0}", const 388 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_percpu_ref_tryget_live_rcu_1, sym __slate_asm_string_percpu_ref_tryget_live_rcu_2, const 281 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("389: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 389b - ., 4; .popsection\n# {0}", const 389 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if likely!(__ref_is_percpu(r#ref, unsafe { &mut (*core::ptr::addr_of_mut!(percpu_count)) })) != 0 {
loop {
__verify_pcpu_ptr__2!();
match (const { core::mem::size_of::<u64>() }) as u64 {
1 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: u64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incb %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decb %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u8 = 1u8;
if false {
let pto_tmp__: u64 = 1;
}
unsafe {
core::arch::asm!("addb {1:l}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
2 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: u64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incw %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decw %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u16 = 1u16;
if false {
let pto_tmp__: u64 = 1;
}
unsafe {
core::arch::asm!("addw {1:x}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) pto_val__ as u16, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
4 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: u64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incl %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decl %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u32 = 1u32;
if false {
let pto_tmp__: u64 = 1;
}
unsafe {
core::arch::asm!("addl {1:e}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) pto_val__ as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
8 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: u64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incq %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decq %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u64 = 1;
if false {
let pto_tmp__: u64 = 1;
}
unsafe {
core::arch::asm!("addq {1:r}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) pto_val__ as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
if true {
break;
}
}
ret = true;
} else {
if (unsafe { (*r#ref).percpu_count_ptr }) & 2 == 0 {
ret = atomic_long_inc_not_zero(unsafe { core::ptr::addr_of_mut!((*(*r#ref).data).count) });
}
}
ret
}

fn percpu_ref_put_many(r#ref: *mut percpu_ref, nr: u64) {
let mut percpu_count: *mut u64 = unsafe { core::mem::zeroed() };
rcu_read_lock();
if __ref_is_percpu(r#ref, unsafe { &mut (*core::ptr::addr_of_mut!(percpu_count)) }) {
loop {
__verify_pcpu_ptr__2!();
match (const { core::mem::size_of::<u64>() }) as u64 {
1 => {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(nr.wrapping_neg()) }) && (nr.wrapping_neg() == 1 || nr.wrapping_neg() == (1i32.wrapping_neg() as u64)) { nr.wrapping_neg() as i32 } else { 0 };
if false {
let pao_tmp__: u64 = nr.wrapping_neg();
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incb %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decb %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u8 = (nr.wrapping_neg() & 255) as u8;
if false {
let pto_tmp__: u64 = nr.wrapping_neg();
}
unsafe {
core::arch::asm!("addb {1:l}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
2 => {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(nr.wrapping_neg()) }) && (nr.wrapping_neg() == 1 || nr.wrapping_neg() == (1i32.wrapping_neg() as u64)) { nr.wrapping_neg() as i32 } else { 0 };
if false {
let pao_tmp__: u64 = nr.wrapping_neg();
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incw %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decw %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u16 = (nr.wrapping_neg() & 65535) as u16;
if false {
let pto_tmp__: u64 = nr.wrapping_neg();
}
unsafe {
core::arch::asm!("addw {1:x}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) pto_val__ as u16, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
4 => {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(nr.wrapping_neg()) }) && (nr.wrapping_neg() == 1 || nr.wrapping_neg() == (1i32.wrapping_neg() as u64)) { nr.wrapping_neg() as i32 } else { 0 };
if false {
let pao_tmp__: u64 = nr.wrapping_neg();
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incl %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decl %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u32 = (nr.wrapping_neg() & 4294967295) as u32;
if false {
let pto_tmp__: u64 = nr.wrapping_neg();
}
unsafe {
core::arch::asm!("addl {1:e}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) pto_val__ as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
8 => {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(nr.wrapping_neg()) }) && (nr.wrapping_neg() == 1 || nr.wrapping_neg() == (1i32.wrapping_neg() as u64)) { nr.wrapping_neg() as i32 } else { 0 };
if false {
let pao_tmp__: u64 = nr.wrapping_neg();
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incq %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decq %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u64 = nr.wrapping_neg();
if false {
let pto_tmp__: u64 = nr.wrapping_neg();
}
unsafe {
core::arch::asm!("addq {1:r}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) pto_val__ as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
if true {
break;
}
}
} else {
if unlikely__2!(atomic_long_sub_and_test(nr as i64, unsafe { core::ptr::addr_of_mut!((*(*r#ref).data).count) })) != 0 {
unsafe { unsafe { (*(*r#ref).data).release }.unwrap()(r#ref) };
}
}
rcu_read_unlock();
}

fn percpu_ref_put(r#ref: *mut percpu_ref) {
percpu_ref_put_many(r#ref, 1u64);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_slab_h_532 {
}


// no-op
// kmem_cache_alloc and friends return pointers aligned to ARCH_SLAB_MINALIGN.
// kmalloc and friends return pointers aligned to both ARCH_KMALLOC_MINALIGN
// and ARCH_SLAB_MINALIGN, but here we only assume the former alignment.
// SLUB directly allocates requests fitting in to an order-1 page
// (PAGE_SIZE*2).  Larger requests are passed to the page allocator.
// Maximum size for which we actually use a slab cache
// Define gfp bits that should not be set for KMALLOC_NORMAL.
#[inline(always)]
fn kmalloc_type(flags: u32, token: __SlateRecord_slab_h_532) -> u32 {
// The most common case is KMALLOC_NORMAL, so test for it
// with a single branch for all the relevant flags.
if likely!(flags & ((BIT!(4i32) as u32) | if IS_ENABLED__2!() != 0 { BIT!(0i32) as u32 } else { 0 } | if IS_ENABLED!() != 0 { BIT!(22i32) as u32 } else { 0 }) == 0) != 0 {
return 0;
}
// At least one of the flags has to be set. Their priorities in
// decreasing order are:
//  1) __GFP_DMA
//  2) __GFP_RECLAIMABLE
//  3) __GFP_ACCOUNT
if IS_ENABLED__2!() != 0 && flags & (BIT!(0i32) as u32) != 0 {
return 2;
}
if IS_ENABLED!() == 0 || flags & (BIT!(4i32) as u32) != 0 {
return 1;
} else {
return 0;
}
}

#[inline(always)]
fn __kmalloc_index(size: u64, size_is_constant: bool) -> u32 {
if size == 0 {
return 0;
}
if size <= 8u64 {
return 3;
}
if 8i32 <= 32 && size > 64 && size <= 96 {
return 1;
}
if 8i32 <= 64 && size > 128 && size <= 192 {
return 2;
}
if size <= 8 {
return 3;
}
if size <= 16 {
return 4;
}
if size <= 32 {
return 5;
}
if size <= 64 {
return 6;
}
if size <= 128 {
return 7;
}
if size <= 256 {
return 8;
}
if size <= 512 {
return 9;
}
if size <= 1024 {
return 10;
}
if size <= 2048u64 {
return 11;
}
if size <= 4096u64 {
return 12;
}
if size <= 8192u64 {
return 13;
}
if size <= 16384u64 {
return 14;
}
if size <= 32768u64 {
return 15;
}
if size <= 65536u64 {
return 16;
}
if size <= 131072u64 {
return 17;
}
if size <= 262144u64 {
return 18;
}
if size <= 524288u64 {
return 19;
}
if size <= 1048576u64 {
return 20;
}
if size <= 2097152u64 {
return 21;
}
if IS_ENABLED!() == 0 && size_is_constant {
loop {
if true {
unsafe { __compiletime_assert_392() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
} else {
loop {
unsafe {
core::arch::asm!("393: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 393b - ., 3; .popsection\n# {0}", const 393 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string___kmalloc_index_1, sym __slate_asm_string___kmalloc_index_2, const 834 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe { core::hint::unreachable_unchecked() };
if true {
break;
}
}
}
// Will never be reached. Needed because the compiler may complain
1i32.wrapping_neg() as u32
}

#[inline(always)]
fn _kmalloc_noprof(size: u64, flags: u32, token: __SlateRecord_slab_h_532) -> *mut () {
if (unsafe { __slate_intrinsic_f217c4720f04f6db(size) }) && size != 0 {
let mut index: u32 = 0;
if size > 8192u64 {
return unsafe { __kmalloc_large_noprof(size, flags) };
}
index = kmalloc_index!(size);
return unsafe { __kmalloc_cache_noprof(kmalloc_caches[(kmalloc_type(flags, token) as u32) as usize][index as usize], flags, size) };
}
unsafe { __kmalloc_noprof(PASS_TOKEN_PARAMS!(size), flags) }
}

// __alloc_objs - Allocate objects of a given type using
// @KMALLOC: which size-based kmalloc wrapper to allocate with.
// @GFP: GFP flags for the allocation.
// @TYPE: type to allocate space for.
// @COUNT: how many @TYPE objects to allocate.
//
// Returns: Newly allocated pointer to (first) @TYPE of @COUNT-many
// allocated @TYPE objects, or NULL on failure.
// kmalloc_obj - Allocate a single instance of the given type
// @VAR_OR_TYPE: Variable or type to allocate.
// @...: optional GFP flags for the allocation (GFP_KERNEL when not specified).
//
// Returns: newly allocated pointer to a @VAR_OR_TYPE on success, or NULL
// on failure.
// All kzalloc aliases for kmalloc_(obj|objs|flex).
#[inline(always)]
fn _kmalloc_node_noprof(size: u64, flags: u32, node: i32, token: __SlateRecord_slab_h_532) -> *mut () {
if (unsafe { __slate_intrinsic_f217c4720f04f6db(size) }) && size != 0 {
let mut index: u32 = 0;
if size > 8192u64 {
return unsafe { __kmalloc_large_node_noprof(size, flags, node) };
}
index = kmalloc_index!(size);
return unsafe { __kmalloc_cache_node_noprof(kmalloc_caches[(kmalloc_type(flags, token) as u32) as usize][index as usize], flags, node, size) };
}
unsafe { __kmalloc_node_noprof(PASS_KMALLOC_PARAMS!(size), flags, node) }
}

fn _kmalloc_array_node_noprof(n: u64, size: u64, flags: u32, node: i32, token: __SlateRecord_slab_h_532) -> *mut () {
let mut bytes: u64 = 0;
if unlikely__2!(check_mul_overflow!(n, size, core::ptr::addr_of_mut!(bytes))) != 0 {
return core::ptr::null_mut::<()>();
}
if (unsafe { __slate_intrinsic_f217c4720f04f6db(n) }) && unsafe { __slate_intrinsic_f217c4720f04f6db(size) } {
return _kmalloc_node_noprof(bytes, flags, node, token);
}
unsafe { __kmalloc_node_noprof(PASS_KMALLOC_PARAMS!(bytes), flags, node) }
}

fn _kzalloc_noprof(size: u64, flags: u32, token: __SlateRecord_slab_h_532) -> *mut () {
_kmalloc_noprof(size, flags | (BIT!(8i32) as u32), token)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_file_ref_h_52 {
    refcnt: __SlateRecord_types_h_197,
}


// File is embedded in backing_file object
// File is capable of returning -EAGAIN if I/O will block
// The two FMODE_NONOTIFY* define which fsnotify events should not be generated
// for an open file. These are the possible values of
// (f->f_mode & FMODE_FSNOTIFY_MASK) and their meaning:
//
// FMODE_NONOTIFY - suppress all (incl. non-permission) events.
// FMODE_NONOTIFY_PERM - suppress permission (incl. pre-content) events.
// FMODE_NONOTIFY | FMODE_NONOTIFY_PERM - suppress only FAN_ACCESS_PERM.
// Whiteout is represented by a char device.  The following constants define the
// mode and device number to use.
#[repr(C)]
#[derive(Clone, Copy)]
struct iattr {
    ia_valid: u32,
    ia_mode: u16,
    __slate_anon_2: __SlateRecord_fs_h_272,
    __slate_anon_3: __SlateRecord_fs_h_276,
    ia_size: i64,
    ia_atime: timespec64,
    ia_mtime: timespec64,
    ia_ctime: timespec64,
    ia_file: *mut file,
}


// non-RWF related bits - start at 16
#[repr(C)]
#[derive(Clone, Copy)]
struct kiocb {
    ki_filp: *mut file,
    ki_pos: i64,
    ki_complete: Option<unsafe extern "C" fn(*mut kiocb, i64)>,
    private: *mut (),
    ki_flags: i32,
    ki_ioprio: u16,
    ki_write_stream: u8,
    ki_waitq: *mut wait_page_queue,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct address_space_operations {
    read_folio: Option<unsafe extern "C" fn(*mut file, *mut folio) -> i32>,
    writepages: Option<unsafe extern "C" fn(*mut address_space, *mut writeback_control) -> i32>,
    dirty_folio: Option<unsafe extern "C" fn(*mut address_space, *mut folio) -> bool>,
    readahead: Option<unsafe extern "C" fn(*mut readahead_control)>,
    write_begin: Option<unsafe extern "C" fn(*const kiocb, *mut address_space, i64, u32, *mut *mut folio, *mut *mut ()) -> i32>,
    write_end: Option<unsafe extern "C" fn(*const kiocb, *mut address_space, i64, u32, u32, *mut folio, *mut ()) -> i32>,
    bmap: Option<unsafe extern "C" fn(*mut address_space, u64) -> u64>,
    invalidate_folio: Option<unsafe extern "C" fn(*mut folio, u64, u64)>,
    release_folio: Option<unsafe extern "C" fn(*mut folio, u32) -> bool>,
    free_folio: Option<unsafe extern "C" fn(*mut folio)>,
    direct_IO: Option<unsafe extern "C" fn(*mut kiocb, *mut iov_iter) -> i64>,
    migrate_folio: Option<unsafe extern "C" fn(*mut address_space, *mut folio, *mut folio, u32) -> i32>,
    launder_folio: Option<unsafe extern "C" fn(*mut folio) -> i32>,
    is_partially_uptodate: Option<unsafe extern "C" fn(*mut folio, u64, u64) -> bool>,
    is_dirty_writeback: Option<unsafe extern "C" fn(*mut folio, *mut bool, *mut bool)>,
    error_remove_folio: Option<unsafe extern "C" fn(*mut address_space, *mut folio) -> i32>,
    swap_activate: Option<unsafe extern "C" fn(*mut swap_info_struct, *mut file, *mut u64) -> i32>,
    swap_deactivate: Option<unsafe extern "C" fn(*mut file)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct address_space {
    host: *mut inode,
    i_pages: xarray,
    invalidate_lock: rw_semaphore,
    gfp_mask: u32,
    i_mmap_writable: __SlateRecord_types_h_190,
    i_mmap: rb_root_cached,
    nrpages: u64,
    writeback_index: u64,
    a_ops: *const address_space_operations,
    flags: u64,
    wb_err: u32,
    i_private_lock: spinlock,
    i_mmap_rwsem: rw_semaphore,
}


// XArray tags, for tagging dirty and writeback pages in the pagecache.
#[repr(C)]
#[derive(Clone, Copy)]
struct inode_state_flags {
    __state: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct inode {
    i_mode: u16,
    i_opflags: u16,
    i_flags: u32,
    i_acl: *mut posix_acl,
    i_default_acl: *mut posix_acl,
    i_uid: __SlateRecord_uidgid_types_h_9,
    i_gid: __SlateRecord_uidgid_types_h_13,
    i_op: *const inode_operations,
    i_sb: *mut super_block,
    i_mapping: *mut address_space,
    i_security: *mut (),
    i_ino: u64,
    __slate_anon_12: __SlateRecord_fs_h_790,
    i_rdev: u32,
    i_size: i64,
    i_atime_sec: i64,
    i_mtime_sec: i64,
    i_ctime_sec: i64,
    i_atime_nsec: u32,
    i_mtime_nsec: u32,
    i_ctime_nsec: u32,
    i_generation: u32,
    i_lock: spinlock,
    i_bytes: u16,
    i_blkbits: u8,
    i_write_hint: u8,
    i_blocks: u64,
    i_state: inode_state_flags,
    i_rwsem: rw_semaphore,
    dirtied_when: u64,
    dirtied_time_when: u64,
    i_hash: hlist_node,
    i_io_list: list_head,
    i_lru: list_head,
    i_sb_list: list_head,
    i_wb_list: list_head,
    __slate_anon_36: __SlateRecord_fs_h_834,
    i_version: __SlateRecord_types_h_197,
    i_sequence: __SlateRecord_types_h_197,
    i_count: __SlateRecord_types_h_190,
    i_dio_count: __SlateRecord_types_h_190,
    i_writecount: __SlateRecord_types_h_190,
    i_readcount: __SlateRecord_types_h_190,
    __slate_anon_43: __SlateRecord_fs_h_846,
    i_flctx: *mut file_lock_context,
    i_data: address_space,
    __slate_anon_46: __SlateRecord_fs_h_852,
    __slate_anon_47: __SlateRecord_fs_h_856,
    i_fsnotify_mask: u32,
    i_fsnotify_marks: *mut fsnotify_mark_connector,
    i_private: *mut (),
}


// Get bit address from inode->i_state to use with wait_var_event()
// infrastructre.
#[repr(C)]
#[derive(Clone, Copy)]
struct fown_struct {
    file: *mut file,
    lock: rwlock,
    pid: *mut pid,
    pid_type: u32,
    uid: __SlateRecord_uidgid_types_h_9,
    euid: __SlateRecord_uidgid_types_h_9,
    signum: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct file_ra_state {
    start: u64,
    size: u32,
    async_size: u32,
    ra_pages: u32,
    order: u16,
    mmap_miss: u16,
    prev_pos: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct file {
    f_lock: spinlock,
    f_mode: u32,
    f_op: *const file_operations,
    f_mapping: *mut address_space,
    private_data: *mut (),
    f_inode: *mut inode,
    f_flags: u32,
    f_iocb_flags: u32,
    f_cred: *const cred,
    f_owner: *mut fown_struct,
    __slate_anon_10: __SlateRecord_fs_h_1267,
    __slate_anon_11: __SlateRecord_fs_h_1271,
    f_pos: i64,
    f_security: *mut (),
    f_wb_err: u32,
    f_sb_err: u32,
    f_ep: *mut hlist_head,
    __slate_anon_17: __SlateRecord_fs_h_1287,
    f_ref: __SlateRecord_file_ref_h_52,
}


// Multigrain timestamps
//
// Conditionally use fine-grained ctime and mtime timestamps when there
// are users actively observing them via getattr. The primary use-case
// for this is NFS clients that use the ctime to distinguish between
// different states of the file, and that are often fooled by multiple
// operations that occur in the same coarse-grained timer tick.
#[repr(C)]
#[derive(Clone, Copy)]
struct dir_context {
    actor: Option<unsafe extern "C" fn(*mut dir_context, *const u8, i32, i64, u64, u32) -> bool>,
    pos: i64,
    count: i32,
    dt_flags_mask: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct file_operations {
    owner: *mut module,
    fop_flags: u32,
    llseek: Option<unsafe extern "C" fn(*mut file, i64, i32) -> i64>,
    read: Option<unsafe extern "C" fn(*mut file, *mut u8, u64, *mut i64) -> i64>,
    write: Option<unsafe extern "C" fn(*mut file, *const u8, u64, *mut i64) -> i64>,
    read_iter: Option<unsafe extern "C" fn(*mut kiocb, *mut iov_iter) -> i64>,
    write_iter: Option<unsafe extern "C" fn(*mut kiocb, *mut iov_iter) -> i64>,
    iopoll: Option<unsafe extern "C" fn(*mut kiocb, *mut io_comp_batch, u32) -> i32>,
    iterate_shared: Option<unsafe extern "C" fn(*mut file, *mut dir_context) -> i32>,
    poll: Option<unsafe extern "C" fn(*mut file, *mut poll_table_struct) -> u32>,
    unlocked_ioctl: Option<unsafe extern "C" fn(*mut file, u32, u64) -> i64>,
    compat_ioctl: Option<unsafe extern "C" fn(*mut file, u32, u64) -> i64>,
    mmap: Option<unsafe extern "C" fn(*mut file, *mut vm_area_struct) -> i32>,
    open: Option<unsafe extern "C" fn(*mut inode, *mut file) -> i32>,
    flush: Option<unsafe extern "C" fn(*mut file, *mut ()) -> i32>,
    release: Option<unsafe extern "C" fn(*mut inode, *mut file) -> i32>,
    fsync: Option<unsafe extern "C" fn(*mut file, i64, i64, i32) -> i32>,
    fasync: Option<unsafe extern "C" fn(i32, *mut file, i32) -> i32>,
    lock: Option<unsafe extern "C" fn(*mut file, i32, *mut file_lock) -> i32>,
    get_unmapped_area: Option<unsafe extern "C" fn(*mut file, u64, u64, u64, u64) -> u64>,
    check_flags: Option<unsafe extern "C" fn(i32) -> i32>,
    flock: Option<unsafe extern "C" fn(*mut file, i32, *mut file_lock) -> i32>,
    splice_write: Option<unsafe extern "C" fn(*mut pipe_inode_info, *mut file, *mut i64, u64, u32) -> i64>,
    splice_read: Option<unsafe extern "C" fn(*mut file, *mut i64, *mut pipe_inode_info, u64, u32) -> i64>,
    splice_eof: Option<unsafe extern "C" fn(*mut file)>,
    setlease: Option<unsafe extern "C" fn(*mut file, i32, *mut *mut file_lease, *mut *mut ()) -> i32>,
    fallocate: Option<unsafe extern "C" fn(*mut file, i32, i64, i64) -> i64>,
    show_fdinfo: Option<unsafe extern "C" fn(*mut seq_file, *mut file)>,
    copy_file_range: Option<unsafe extern "C" fn(*mut file, i64, *mut file, i64, u64, u32) -> i64>,
    remap_file_range: Option<unsafe extern "C" fn(*mut file, i64, *mut file, i64, i64, u32) -> i64>,
    fadvise: Option<unsafe extern "C" fn(*mut file, i64, i64, i32) -> i32>,
    uring_cmd: Option<unsafe extern "C" fn(*mut io_uring_cmd, u32) -> i32>,
    uring_cmd_iopoll: Option<unsafe extern "C" fn(*mut io_uring_cmd, *mut io_comp_batch, u32) -> i32>,
    mmap_prepare: Option<unsafe extern "C" fn(*mut vm_area_desc) -> i32>,
}


// File system supports uncached read/write buffered IO
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct inode_operations {
    lookup: Option<unsafe extern "C" fn(*mut inode, *mut dentry, u32) -> *mut dentry>,
    get_link: Option<unsafe extern "C" fn(*mut dentry, *mut inode, *mut delayed_call) -> *const u8>,
    permission: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut inode, i32) -> i32>,
    get_inode_acl: Option<unsafe extern "C" fn(*mut inode, i32, bool) -> *mut posix_acl>,
    readlink: Option<unsafe extern "C" fn(*mut dentry, *mut u8, i32) -> i32>,
    create: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut inode, *mut dentry, u16) -> i32>,
    link: Option<unsafe extern "C" fn(*mut dentry, *mut inode, *mut dentry) -> i32>,
    unlink: Option<unsafe extern "C" fn(*mut inode, *mut dentry) -> i32>,
    symlink: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut inode, *mut dentry, *const u8) -> i32>,
    mkdir: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut inode, *mut dentry, u16) -> *mut dentry>,
    rmdir: Option<unsafe extern "C" fn(*mut inode, *mut dentry) -> i32>,
    mknod: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut inode, *mut dentry, u16, u32) -> i32>,
    rename: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut inode, *mut dentry, *mut inode, *mut dentry, u32) -> i32>,
    setattr: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut dentry, *mut iattr) -> i32>,
    getattr: Option<unsafe extern "C" fn(*mut mnt_idmap, *const path, *mut kstat, u32, u32) -> i32>,
    listxattr: Option<unsafe extern "C" fn(*mut dentry, *mut u8, u64) -> i64>,
    fiemap: Option<unsafe extern "C" fn(*mut inode, *mut fiemap_extent_info, u64, u64) -> i32>,
    update_time: Option<unsafe extern "C" fn(*mut inode, u32, u32) -> i32>,
    sync_lazytime: Option<unsafe extern "C" fn(*mut inode)>,
    atomic_open: Option<unsafe extern "C" fn(*mut inode, *mut dentry, *mut file, u32, u16) -> i32>,
    tmpfile: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut inode, *mut file, u16) -> i32>,
    get_acl: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut dentry, i32) -> *mut posix_acl>,
    set_acl: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut dentry, *mut posix_acl, i32) -> i32>,
    fileattr_set: Option<unsafe extern "C" fn(*mut mnt_idmap, *mut dentry, *mut file_kattr) -> i32>,
    fileattr_get: Option<unsafe extern "C" fn(*mut dentry, *mut file_kattr) -> i32>,
    get_offset_ctx: Option<unsafe extern "C" fn(*mut inode) -> *mut offset_ctx>,
}


// no suid or xattr security attributes
// Note that nosuid etc flags are inode-specific: setting some file-system
// flags just means all the inodes inherit those flags by default. It might be
// possible to override it selectively if you really wanted to with some
// ioctl() that is not currently implemented.
//
// Exception: SB_RDONLY is always applied to the entire file system.
//
// Unfortunately, it is possible to change a filesystems flags with it mounted
// with files in use.  This means that all of the inodes will not have their
// i_flags updated.  Hence, i_flags no longer inherit the superblock mount
// flags, so these have to be checked separately. -- rmk@arm.uk.linux.org
#[repr(C)]
#[derive(Clone, Copy)]
struct file_system_type {
    name: *const u8,
    fs_flags: i32,
    init_fs_context: Option<unsafe extern "C" fn(*mut fs_context) -> i32>,
    parameters: *const fs_parameter_spec,
    kill_sb: Option<unsafe extern "C" fn(*mut super_block)>,
    owner: *mut module,
    list: hlist_node,
    fs_supers: hlist_head,
    s_lock_key: lock_class_key,
    s_umount_key: lock_class_key,
    s_vfs_rename_key: lock_class_key,
    s_writers_key: [lock_class_key; 3],
    i_lock_key: lock_class_key,
    i_mutex_key: lock_class_key,
    invalidate_lock_key: lock_class_key,
    i_mutex_dir_key: lock_class_key,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct offset_ctx {
    mt: maple_tree,
    next_offset: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct compat_robust_list {
    next: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct compat_robust_list_head {
    list: compat_robust_list,
    futex_offset: i32,
    list_op_pending: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vdso_image {
    data: *mut (),
    size: u64,
    alt: u64,
    alt_len: u64,
    extable_base: u64,
    extable_len: u64,
    extable: *const (),
    sym___kernel_sigreturn: i64,
    sym___kernel_rt_sigreturn: i64,
    sym___kernel_vsyscall: i64,
    sym_int80_landing_pad: i64,
    sym_vdso32_sigreturn_landing_pad: i64,
    sym_vdso32_rt_sigreturn_landing_pad: i64,
    sym___futex_list64_try_unlock_cs_start: i64,
    sym___futex_list64_try_unlock_cs_end: i64,
    sym___futex_list32_try_unlock_cs_start: i64,
    sym___futex_list32_try_unlock_cs_end: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct elf64_sym {
    st_name: u32,
    st_info: u8,
    st_other: u8,
    st_shndx: u16,
    st_value: u64,
    st_size: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct idr {
    idr_rt: xarray,
    idr_base: u32,
    idr_next: u32,
}


// The IDR API does not expose the tagging functionality of the radix tree
// to users.  Use tag 0 to track whether a node has free space below it.
// Set the IDR flag and the IDR_FREE tag
#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_elem_dir {
    subdirs: u64,
    children: rb_root,
    root: *mut kernfs_root,
    rev: u64,
}


core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_blk_freeze_queue_start: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_freeze_queue_wait: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_freeze_queue_wait_timeout: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_freeze_queue_nomemsave: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_unfreeze_queue_nomemrestore: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_freeze_queue_start_non_owner: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_unfreeze_queue_non_owner: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_quiesce_queue_nowait: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {7} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_wait_quiesce_done: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {8} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_quiesce_queue: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {9} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_unquiesce_queue: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {10} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_quiesce_tagset: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {11} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_unquiesce_tagset: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {12} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_rq_init: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {13} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_alloc_request: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {14} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_alloc_request_hctx: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {15} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_free_request: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {16} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_dump_rq_flags: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {17} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_update_request: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {18} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol___blk_mq_end_request: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {19} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_end_request: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {20} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_end_request_batch: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {21} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_complete_request_remote: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {22} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_complete_request: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {23} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_start_request: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {24} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_execute_rq_nowait: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {25} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_rq_is_poll: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {26} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_execute_rq: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {27} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_requeue_request: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {28} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_kick_requeue_list: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {29} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_delay_kick_requeue_list: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {30} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_queue_inflight: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {31} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_delay_run_hw_queue: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {32} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_run_hw_queue: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {33} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_run_hw_queues: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {34} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_delay_run_hw_queues: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {35} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_stop_hw_queue: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {36} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_stop_hw_queues: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {37} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_start_hw_queue: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {38} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_start_hw_queues: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {39} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_start_stopped_hw_queue: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {40} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_start_stopped_hw_queues: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {41} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_insert_cloned_request: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {42} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_rq_unprep_clone: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {43} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_rq_prep_clone: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {44} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_steal_bios: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {45} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_alloc_queue: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {46} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_destroy_queue: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {47} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol___blk_mq_alloc_disk: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {48} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_alloc_disk_for_queue: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {49} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_init_allocated_queue: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {50} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_alloc_tag_set: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {51} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_alloc_sq_tag_set: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {52} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_free_tag_set: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {53} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_update_nr_hw_queues: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {54} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_rq_poll: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {55} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_mq_rq_cpu: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {56} ; .previous\n.section\t\".initcall4.init\", \"a\"\t\t\n__initcall__kmod_blk_mq__1032_5313_blk_mq_init4:\t\t\t\n.long\t{57} - .\t\n.previous\t\t\t\t\t\n", sym blk_freeze_queue_start, sym blk_mq_freeze_queue_wait, sym blk_mq_freeze_queue_wait_timeout, sym blk_mq_freeze_queue_nomemsave, sym blk_mq_unfreeze_queue_nomemrestore, sym blk_freeze_queue_start_non_owner, sym blk_mq_unfreeze_queue_non_owner, sym blk_mq_quiesce_queue_nowait, sym blk_mq_wait_quiesce_done, sym blk_mq_quiesce_queue, sym blk_mq_unquiesce_queue, sym blk_mq_quiesce_tagset, sym blk_mq_unquiesce_tagset, sym blk_rq_init, sym blk_mq_alloc_request, sym blk_mq_alloc_request_hctx, sym blk_mq_free_request, sym blk_dump_rq_flags, sym blk_update_request, sym __blk_mq_end_request, sym blk_mq_end_request, sym blk_mq_end_request_batch, sym blk_mq_complete_request_remote, sym blk_mq_complete_request, sym blk_mq_start_request, sym blk_execute_rq_nowait, sym blk_rq_is_poll, sym blk_execute_rq, sym blk_mq_requeue_request, sym blk_mq_kick_requeue_list, sym blk_mq_delay_kick_requeue_list, sym blk_mq_queue_inflight, sym blk_mq_delay_run_hw_queue, sym blk_mq_run_hw_queue, sym blk_mq_run_hw_queues, sym blk_mq_delay_run_hw_queues, sym blk_mq_stop_hw_queue, sym blk_mq_stop_hw_queues, sym blk_mq_start_hw_queue, sym blk_mq_start_hw_queues, sym blk_mq_start_stopped_hw_queue, sym blk_mq_start_stopped_hw_queues, sym blk_insert_cloned_request, sym blk_rq_unprep_clone, sym blk_rq_prep_clone, sym blk_steal_bios, sym blk_mq_alloc_queue, sym blk_mq_destroy_queue, sym __blk_mq_alloc_disk, sym blk_mq_alloc_disk_for_queue, sym blk_mq_init_allocated_queue, sym blk_mq_alloc_tag_set, sym blk_mq_alloc_sq_tag_set, sym blk_mq_free_tag_set, sym blk_mq_update_nr_hw_queues, sym blk_rq_poll, sym blk_mq_rq_cpu, sym blk_mq_init, options(att_syntax));

#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_elem_symlink {
    target_kn: *mut kernfs_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_elem_attr {
    ops: *const kernfs_ops,
    open: *mut kernfs_open_node,
    size: i64,
    notify_next: *mut kernfs_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_node {
    count: __SlateRecord_types_h_190,
    active: __SlateRecord_types_h_190,
    __parent: *mut kernfs_node,
    name: *const u8,
    rb: rb_node,
    ns: *const ns_common,
    hash: u32,
    flags: u16,
    mode: u16,
    __slate_anon_9: __SlateRecord_kernfs_h_221,
    id: u64,
    r#priv: *mut (),
    iattr: *mut kernfs_iattrs,
    rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_open_file {
    kn: *mut kernfs_node,
    file: *mut file,
    seq_file: *mut seq_file,
    r#priv: *mut (),
    mutex: mutex,
    prealloc_mutex: mutex,
    event: i32,
    list: list_head,
    prealloc_buf: *mut u8,
    atomic_write_len: u64,
    __slate_bits_0: __slate_bits::__SlateBits692U0,
    vm_ops: *const vm_operations_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_ops {
    open: Option<unsafe extern "C" fn(*mut kernfs_open_file) -> i32>,
    release: Option<unsafe extern "C" fn(*mut kernfs_open_file)>,
    seq_show: Option<unsafe extern "C" fn(*mut seq_file, *mut ()) -> i32>,
    seq_start: Option<unsafe extern "C" fn(*mut seq_file, *mut i64) -> *mut ()>,
    seq_next: Option<unsafe extern "C" fn(*mut seq_file, *mut (), *mut i64) -> *mut ()>,
    seq_stop: Option<unsafe extern "C" fn(*mut seq_file, *mut ())>,
    read: Option<unsafe extern "C" fn(*mut kernfs_open_file, *mut u8, u64, i64) -> i64>,
    atomic_write_len: u64,
    prealloc: bool,
    write: Option<unsafe extern "C" fn(*mut kernfs_open_file, *mut u8, u64, i64) -> i64>,
    poll: Option<unsafe extern "C" fn(*mut kernfs_open_file, *mut poll_table_struct) -> u32>,
    mmap: Option<unsafe extern "C" fn(*mut kernfs_open_file, *mut vm_area_struct) -> i32>,
    llseek: Option<unsafe extern "C" fn(*mut kernfs_open_file, i64, i32) -> i64>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kobj_ns_type_operations {
    r#type: u32,
    current_may_mount: Option<unsafe extern "C" fn() -> bool>,
    grab_current_ns: Option<unsafe extern "C" fn() -> *mut ns_common>,
    netlink_ns: Option<unsafe extern "C" fn(*mut sock) -> *const ns_common>,
    initial_ns: Option<unsafe extern "C" fn() -> *const ns_common>,
    drop_ns: Option<unsafe extern "C" fn(*mut ns_common)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct attribute {
    name: *const u8,
    mode: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct attribute_group {
    name: *const u8,
    __slate_anon_1: __SlateRecord_sysfs_h_107,
    is_bin_visible: Option<unsafe extern "C" fn(*mut kobject, *const bin_attribute, i32) -> u16>,
    bin_size: Option<unsafe extern "C" fn(*mut kobject, *const bin_attribute, i32) -> u64>,
    __slate_anon_4: __SlateRecord_sysfs_h_118,
    bin_attrs: *const *const bin_attribute,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bin_attribute {
    attr: attribute,
    size: u64,
    private: *mut (),
    f_mapping: Option<unsafe extern "C" fn() -> *mut address_space>,
    read: Option<unsafe extern "C" fn(*mut file, *mut kobject, *const bin_attribute, *mut u8, i64, u64) -> i64>,
    write: Option<unsafe extern "C" fn(*mut file, *mut kobject, *const bin_attribute, *mut u8, i64, u64) -> i64>,
    llseek: Option<unsafe extern "C" fn(*mut file, *mut kobject, *const bin_attribute, i64, i32) -> i64>,
    mmap: Option<unsafe extern "C" fn(*mut file, *mut kobject, *const bin_attribute, *mut vm_area_struct) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sysfs_ops {
    show: Option<unsafe extern "C" fn(*mut kobject, *mut attribute, *mut u8) -> i64>,
    store: Option<unsafe extern "C" fn(*mut kobject, *mut attribute, *const u8, u64) -> i64>,
}


// buffer for the variables
#[repr(C)]
#[derive(Clone, Copy)]
struct kobject {
    name: *const u8,
    entry: list_head,
    parent: *mut kobject,
    kset: *mut kset,
    ktype: *const kobj_type,
    sd: *mut kernfs_node,
    kref: kref,
    __slate_bits_0: __slate_bits::__SlateBits635U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kobj_type {
    release: Option<unsafe extern "C" fn(*mut kobject)>,
    sysfs_ops: *const sysfs_ops,
    default_groups: *mut *const attribute_group,
    child_ns_type: Option<unsafe extern "C" fn(*const kobject) -> *const kobj_ns_type_operations>,
    namespace: Option<unsafe extern "C" fn(*const kobject) -> *const ns_common>,
    get_ownership: Option<unsafe extern "C" fn(*const kobject, *mut __SlateRecord_uidgid_types_h_9, *mut __SlateRecord_uidgid_types_h_13)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kobj_uevent_env {
    argv: [*mut u8; 3],
    envp: [*mut u8; 64],
    envp_idx: i32,
    buf: [u8; 2048],
    buflen: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kset_uevent_ops {
    filter: Option<unsafe extern "C" fn(*const kobject) -> i32>,
    name: Option<unsafe extern "C" fn(*const kobject) -> *const u8>,
    uevent: Option<unsafe extern "C" fn(*const kobject, *mut kobj_uevent_env) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kset {
    list: list_head,
    list_lock: spinlock,
    kobj: kobject,
    uevent_ops: *const kset_uevent_ops,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_param_ops {
    flags: u32,
    set: Option<unsafe extern "C" fn(*const u8, *const kernel_param) -> i32>,
    get: Option<unsafe extern "C" fn(*mut u8, *const kernel_param) -> i32>,
    free: Option<unsafe extern "C" fn(*mut ())>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_param {
    name: *const u8,
    r#mod: *mut module,
    ops: *const kernel_param_ops,
    perm: u16,
    level: i8,
    flags: u8,
    __slate_anon_6: __SlateRecord_moduleparam_h_89,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kparam_string {
    maxlen: u32,
    string: *mut u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kparam_array {
    max: u32,
    elemsize: u32,
    num: *mut u32,
    ops: *const kernel_param_ops,
    elem: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct latch_tree_node {
    node: [rb_node; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct its_array {
    pages: *mut *mut (),
    num: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mod_arch_specific {
    num_orcs: u32,
    orc_unwind_ip: *mut i32,
    orc_unwind: *mut orc_entry,
    its_pages: its_array,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct module_kobject {
    kobj: kobject,
    r#mod: *mut module,
    drivers_dir: *mut kobject,
    mp: *mut module_param_attrs,
    kobj_completion: *mut completion,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct module_attribute {
    attr: attribute,
    show: Option<unsafe extern "C" fn(*const module_attribute, *mut module_kobject, *mut u8) -> i64>,
    store: Option<unsafe extern "C" fn(*const module_attribute, *mut module_kobject, *const u8, u64) -> i64>,
    setup: Option<unsafe extern "C" fn(*mut module, *const u8)>,
    test: Option<unsafe extern "C" fn(*mut module) -> i32>,
    free: Option<unsafe extern "C" fn(*mut module)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mod_tree_node {
    r#mod: *mut module,
    node: latch_tree_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct module_memory {
    base: *mut (),
    is_rox: bool,
    size: u32,
    mtn: mod_tree_node,
}


// Only touch one cacheline for common rbtree-for-core-layout case.
#[repr(C)]
#[derive(Clone, Copy)]
struct mod_kallsyms {
    symtab: *mut elf64_sym,
    num_symtab: u32,
    strtab: *mut u8,
    typetab: *mut u8,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct module {
    state: u32,
    list: list_head,
    name: [u8; 56],
    mkobj: module_kobject,
    modinfo_attrs: *mut module_attribute,
    version: *const u8,
    srcversion: *const u8,
    imported_namespaces: *const u8,
    holders_dir: *mut kobject,
    syms: *const kernel_symbol,
    crcs: *const u32,
    flagstab: *const u8,
    num_syms: u32,
    param_lock: mutex,
    kp: *mut kernel_param,
    num_kp: u32,
    using_gplonly_symbols: bool,
    async_probe_requested: bool,
    num_exentries: u32,
    extable: *mut exception_table_entry,
    init: Option<unsafe extern "C" fn() -> i32>,
    __slate_pad_21: [u8; 8],
    mem: [module_memory; 7],
    arch: mod_arch_specific,
    taints: u64,
    num_bugs: u32,
    bug_list: list_head,
    bug_table: *mut bug_entry,
    kallsyms: *mut mod_kallsyms,
    core_kallsyms: mod_kallsyms,
    sect_attrs: *mut module_sect_attrs,
    notes_attrs: *mut module_notes_attrs,
    percpu: *mut (),
    percpu_size: u32,
    noinstr_text_start: *mut (),
    noinstr_text_size: u32,
    num_tracepoints: u32,
    tracepoints_ptrs: *const i32,
    num_srcu_structs: u32,
    srcu_struct_ptrs: *mut *mut srcu_struct,
    jump_entries: *mut jump_entry,
    num_jump_entries: u32,
    num_trace_bprintk_fmt: u32,
    trace_bprintk_fmt_start: *mut *const u8,
    trace_events: *mut *mut trace_event_call,
    num_trace_events: u32,
    trace_evals: *mut *mut trace_eval_map,
    num_trace_evals: u32,
    kprobes_text_start: *mut (),
    kprobes_text_size: u32,
    kprobe_blacklist: *mut u64,
    num_kprobe_blacklist: u32,
    num_static_call_sites: i32,
    static_call_sites: *mut static_call_site,
    source_list: list_head,
    target_list: list_head,
    exit: Option<unsafe extern "C" fn()>,
    refcnt: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pm_message {
    event: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_ops {
    prepare: Option<unsafe extern "C" fn(*mut device) -> i32>,
    complete: Option<unsafe extern "C" fn(*mut device)>,
    suspend: Option<unsafe extern "C" fn(*mut device) -> i32>,
    resume: Option<unsafe extern "C" fn(*mut device) -> i32>,
    freeze: Option<unsafe extern "C" fn(*mut device) -> i32>,
    thaw: Option<unsafe extern "C" fn(*mut device) -> i32>,
    poweroff: Option<unsafe extern "C" fn(*mut device) -> i32>,
    restore: Option<unsafe extern "C" fn(*mut device) -> i32>,
    suspend_late: Option<unsafe extern "C" fn(*mut device) -> i32>,
    resume_early: Option<unsafe extern "C" fn(*mut device) -> i32>,
    freeze_late: Option<unsafe extern "C" fn(*mut device) -> i32>,
    thaw_early: Option<unsafe extern "C" fn(*mut device) -> i32>,
    poweroff_late: Option<unsafe extern "C" fn(*mut device) -> i32>,
    restore_early: Option<unsafe extern "C" fn(*mut device) -> i32>,
    suspend_noirq: Option<unsafe extern "C" fn(*mut device) -> i32>,
    resume_noirq: Option<unsafe extern "C" fn(*mut device) -> i32>,
    freeze_noirq: Option<unsafe extern "C" fn(*mut device) -> i32>,
    thaw_noirq: Option<unsafe extern "C" fn(*mut device) -> i32>,
    poweroff_noirq: Option<unsafe extern "C" fn(*mut device) -> i32>,
    restore_noirq: Option<unsafe extern "C" fn(*mut device) -> i32>,
    runtime_suspend: Option<unsafe extern "C" fn(*mut device) -> i32>,
    runtime_resume: Option<unsafe extern "C" fn(*mut device) -> i32>,
    runtime_idle: Option<unsafe extern "C" fn(*mut device) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pm_subsys_data {
    lock: spinlock,
    refcount: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_info {
    power_state: pm_message,
    __slate_bits_0: __slate_bits::__SlateBits737U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits737U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits737U2,
    links_count: u32,
    request: u32,
    runtime_status: i32,
    last_status: i32,
    runtime_error: i32,
    autosuspend_delay: i32,
    last_busy: u64,
    active_time: u64,
    suspended_time: u64,
    accounting_timestamp: u64,
    subsys_data: *mut pm_subsys_data,
    set_latency_tolerance: Option<unsafe extern "C" fn(*mut device, i32)>,
    qos: *mut dev_pm_qos,
    __slate_bits_3: __slate_bits::__SlateBits737U3,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_domain {
    ops: dev_pm_ops,
    start: Option<unsafe extern "C" fn(*mut device) -> i32>,
    detach: Option<unsafe extern "C" fn(*mut device, bool)>,
    activate: Option<unsafe extern "C" fn(*mut device) -> i32>,
    sync: Option<unsafe extern "C" fn(*mut device)>,
    dismiss: Option<unsafe extern "C" fn(*mut device)>,
    set_performance_state: Option<unsafe extern "C" fn(*mut device, u32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bus_type {
    name: *const u8,
    dev_name: *const u8,
    bus_groups: *const *const attribute_group,
    dev_groups: *const *const attribute_group,
    drv_groups: *const *const attribute_group,
    r#match: Option<unsafe extern "C" fn(*mut device, *const device_driver) -> i32>,
    uevent: Option<unsafe extern "C" fn(*const device, *mut kobj_uevent_env) -> i32>,
    probe: Option<unsafe extern "C" fn(*mut device) -> i32>,
    sync_state: Option<unsafe extern "C" fn(*mut device)>,
    remove: Option<unsafe extern "C" fn(*mut device)>,
    shutdown: Option<unsafe extern "C" fn(*mut device)>,
    irq_get_affinity: Option<unsafe extern "C" fn(*mut device, u32) -> *const cpumask>,
    online: Option<unsafe extern "C" fn(*mut device) -> i32>,
    offline: Option<unsafe extern "C" fn(*mut device) -> i32>,
    suspend: Option<unsafe extern "C" fn(*mut device, pm_message) -> i32>,
    resume: Option<unsafe extern "C" fn(*mut device) -> i32>,
    num_vf: Option<unsafe extern "C" fn(*mut device) -> i32>,
    dma_configure: Option<unsafe extern "C" fn(*mut device) -> i32>,
    dma_cleanup: Option<unsafe extern "C" fn(*mut device)>,
    pm: *const dev_pm_ops,
    driver_override: bool,
    need_parent_lock: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct class {
    name: *const u8,
    class_groups: *const *const attribute_group,
    dev_groups: *const *const attribute_group,
    dev_uevent: Option<unsafe extern "C" fn(*const device, *mut kobj_uevent_env) -> i32>,
    devnode: Option<unsafe extern "C" fn(*const device, *mut u16) -> *mut u8>,
    class_release: Option<unsafe extern "C" fn(*const class)>,
    dev_release: Option<unsafe extern "C" fn(*mut device)>,
    shutdown_pre: Option<unsafe extern "C" fn(*mut device) -> i32>,
    ns_type: *const kobj_ns_type_operations,
    namespace: Option<unsafe extern "C" fn(*const device) -> *const ns_common>,
    get_ownership: Option<unsafe extern "C" fn(*const device, *mut __SlateRecord_uidgid_types_h_9, *mut __SlateRecord_uidgid_types_h_13)>,
    pm: *const dev_pm_ops,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_device_id {
    id: [u8; 16],
    driver_data: u64,
    cls: u32,
    cls_msk: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct of_device_id {
    name: [u8; 32],
    r#type: [u8; 32],
    compatible: [u8; 128],
    data: *const (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_driver {
    name: *const u8,
    bus: *const bus_type,
    owner: *mut module,
    mod_name: *const u8,
    suppress_bind_attrs: bool,
    probe_type: u32,
    of_match_table: *const of_device_id,
    acpi_match_table: *const acpi_device_id,
    probe: Option<unsafe extern "C" fn(*mut device) -> i32>,
    sync_state: Option<unsafe extern "C" fn(*mut device)>,
    remove: Option<unsafe extern "C" fn(*mut device) -> i32>,
    shutdown: Option<unsafe extern "C" fn(*mut device)>,
    suspend: Option<unsafe extern "C" fn(*mut device, pm_message) -> i32>,
    resume: Option<unsafe extern "C" fn(*mut device) -> i32>,
    groups: *const *const attribute_group,
    dev_groups: *const *const attribute_group,
    pm: *const dev_pm_ops,
    coredump: Option<unsafe extern "C" fn(*mut device)>,
    p: *mut driver_private,
    p_cb: __SlateRecord_driver_h_126,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_archdata {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_type {
    name: *const u8,
    groups: *mut *const attribute_group,
    uevent: Option<unsafe extern "C" fn(*const device, *mut kobj_uevent_env) -> i32>,
    devnode: Option<unsafe extern "C" fn(*const device, *mut u16, *mut __SlateRecord_uidgid_types_h_9, *mut __SlateRecord_uidgid_types_h_13) -> *mut u8>,
    release: Option<unsafe extern "C" fn(*mut device)>,
    pm: *const dev_pm_ops,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_dma_parameters {
    max_segment_size: u32,
    min_align_mask: u32,
    segment_boundary_mask: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_links_info {
    suppliers: list_head,
    consumers: list_head,
    defer_sync: list_head,
    status: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_msi_info {
    domain: *mut irq_domain,
    data: *mut msi_device_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_physical_location {
    panel: u32,
    vertical_position: u32,
    horizontal_position: u32,
    dock: bool,
    lid: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device {
    kobj: kobject,
    parent: *mut device,
    p: *mut device_private,
    init_name: *const u8,
    r#type: *const device_type,
    bus: *const bus_type,
    driver: *mut device_driver,
    platform_data: *mut (),
    driver_data: *mut (),
    driver_override: __SlateRecord_device_h_719,
    mutex: mutex,
    links: dev_links_info,
    power: dev_pm_info,
    pm_domain: *mut dev_pm_domain,
    msi: dev_msi_info,
    dma_mask: *mut u64,
    coherent_dma_mask: u64,
    bus_dma_limit: u64,
    dma_range_map: *const bus_dma_region,
    dma_parms: *mut device_dma_parameters,
    dma_pools: list_head,
    dma_io_tlb_mem: *mut io_tlb_mem,
    archdata: dev_archdata,
    of_node: *mut device_node,
    fwnode: *mut fwnode_handle,
    numa_node: i32,
    devt: u32,
    id: u32,
    devres_lock: spinlock,
    devres_head: list_head,
    class: *const class,
    groups: *mut *const attribute_group,
    release: Option<unsafe extern "C" fn(*mut device)>,
    iommu_group: *mut iommu_group,
    iommu: *mut dev_iommu,
    physical_location: *mut device_physical_location,
    removable: u32,
    flags: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wakeup_source {
    name: *const u8,
    id: i32,
    entry: list_head,
    lock: spinlock,
    wakeirq: *mut wake_irq,
    timer: timer_list,
    timer_expires: u64,
    total_time: i64,
    max_time: i64,
    last_time: i64,
    start_prevent_time: i64,
    prevent_sleep_time: i64,
    event_count: u64,
    active_count: u64,
    relax_count: u64,
    expire_count: u64,
    wakeup_count: u64,
    dev: *mut device,
    __slate_bits_0: __slate_bits::__SlateBits734U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fprop_local_percpu {
    events: percpu_counter,
    period: u32,
    lock: raw_spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bdi_writeback {
    bdi: *mut backing_dev_info,
    state: u64,
    last_old_flush: u64,
    b_dirty: list_head,
    b_io: list_head,
    b_more_io: list_head,
    b_dirty_time: list_head,
    list_lock: spinlock,
    writeback_inodes: __SlateRecord_types_h_190,
    stat: [percpu_counter; 5],
    bw_time_stamp: u64,
    dirtied_stamp: u64,
    written_stamp: u64,
    write_bandwidth: u64,
    avg_write_bandwidth: u64,
    dirty_ratelimit: u64,
    balanced_dirty_ratelimit: u64,
    completions: fprop_local_percpu,
    dirty_exceeded: i32,
    start_all_reason: u32,
    work_lock: spinlock,
    work_list: list_head,
    dwork: delayed_work,
    bw_dwork: delayed_work,
    bdi_node: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct backing_dev_info {
    id: u64,
    rb_node: rb_node,
    bdi_list: list_head,
    ra_pages: u64,
    io_pages: u64,
    refcnt: kref,
    capabilities: u32,
    min_ratio: u32,
    max_ratio: u32,
    max_prop_frac: u32,
    tot_write_bandwidth: __SlateRecord_types_h_197,
    last_bdp_sleep: u64,
    wb: bdi_writeback,
    wb_list: list_head,
    wb_waitq: wait_queue_head,
    dev: *mut device,
    dev_name: [u8; 64],
    owner: *mut device,
    debug_dir: *mut dentry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vmem_altmap {
    base_pfn: u64,
    end_pfn: u64,
    reserve: u64,
    free: u64,
    align: u64,
    alloc: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pagemap_ops {
    folio_free: Option<unsafe extern "C" fn(*mut folio)>,
    migrate_to_ram: Option<unsafe extern "C" fn(*mut vm_fault) -> u32>,
    memory_failure: Option<unsafe extern "C" fn(*mut dev_pagemap, u64, u64, i32) -> i32>,
    folio_split: Option<unsafe extern "C" fn(*mut folio, *mut folio)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pagemap {
    altmap: vmem_altmap,
    r#ref: percpu_ref,
    done: completion,
    r#type: u32,
    flags: u32,
    vmemmap_shift: u64,
    ops: *const dev_pagemap_ops,
    owner: *mut (),
    nr_range: i32,
    __slate_anon_9: __SlateRecord_memremap_h_143,
}


// VMA basic access permission flags
// These flags can be updated atomically via VMA/mmap read lock.
#[repr(C)]
#[derive(Clone, Copy)]
struct vm_fault {
    __slate_anon_0: __SlateRecord_mm_h_754,
    flags: u32,
    pmd: *mut __SlateRecord_pgtable_64_types_h_22,
    pud: *mut __SlateRecord_pgtable_types_h_368,
    __slate_anon_4: __SlateRecord_mm_h_768,
    cow_page: *mut page,
    page: *mut page,
    pte: *mut __SlateRecord_pgtable_64_types_h_21,
    ptl: *mut spinlock,
    prealloc_pte: *mut page,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_operations_struct {
    open: Option<unsafe extern "C" fn(*mut vm_area_struct)>,
    close: Option<unsafe extern "C" fn(*mut vm_area_struct)>,
    mapped: Option<unsafe extern "C" fn(u64, u64, u64, *const file, *mut *mut ()) -> i32>,
    may_split: Option<unsafe extern "C" fn(*mut vm_area_struct, u64) -> i32>,
    mremap: Option<unsafe extern "C" fn(*mut vm_area_struct) -> i32>,
    mprotect: Option<unsafe extern "C" fn(*mut vm_area_struct, u64, u64, u64) -> i32>,
    fault: Option<unsafe extern "C" fn(*mut vm_fault) -> u32>,
    huge_fault: Option<unsafe extern "C" fn(*mut vm_fault, u32) -> u32>,
    map_pages: Option<unsafe extern "C" fn(*mut vm_fault, u64, u64) -> u32>,
    pagesize: Option<unsafe extern "C" fn(*mut vm_area_struct) -> u64>,
    page_mkwrite: Option<unsafe extern "C" fn(*mut vm_fault) -> u32>,
    pfn_mkwrite: Option<unsafe extern "C" fn(*mut vm_fault) -> u32>,
    access: Option<unsafe extern "C" fn(*mut vm_area_struct, u64, *mut (), i32, i32) -> i32>,
    name: Option<unsafe extern "C" fn(*mut vm_area_struct) -> *const u8>,
    set_policy: Option<unsafe extern "C" fn(*mut vm_area_struct, *mut mempolicy) -> i32>,
    get_policy: Option<unsafe extern "C" fn(*mut vm_area_struct, u64, *mut u64) -> *mut mempolicy>,
}


// Helper macro which bitwise-or combines the specified input flags into a
// vma_flags_t bitmap value. E.g.:
//
// vma_flags_t flags = mk_vma_flags(VMA_IO_BIT, VMA_PFNMAP_BIT,
//              VMA_DONTEXPAND_BIT, VMA_DONTDUMP_BIT);
//
// The compiler cleverly optimises away all of the work and this ends up being
// equivalent to aggregating the values manually.
// Test whether ALL specified VMA flags are set, e.g.:
//
// if (vma_flags_test_all(flags, VMA_READ_BIT, VMA_MAYREAD_BIT)) { ... }
// 127: arbitrary random number, small enough to assemble well
// GUP_PIN_COUNTING_BIAS, and the associated functions that use it, overload
// the page's refcount so that two separate items are tracked: the original page
// reference count, and also a new count of how many pin_user_pages() calls were
// made against the page. ("gup-pinned" is another term for the latter).
//
// With this scheme, pin_user_pages() becomes special: such pages are marked as
// distinct from normal pages. As such, the unpin_user_page() call (and its
// variants) must be used in order to release gup-pinned pages.
//
// Choice of value:
//
// By making GUP_PIN_COUNTING_BIAS a power of two, debugging of page reference
// counts with respect to pin_user_pages() and unpin_user_page() becomes
// simpler, due to the fact that adding an even power of two to the page
// refcount has the effect of using only the upper N bits, for the code that
// counts up using the bias value. This means that the lower bits are left for
// the exclusive use of the original code that increments and decrements by one
// (or at least, by much smaller values than the bias value).
//
// Of course, once the lower bits overflow into the upper bits (and this is
// OK, because subtraction recovers the original values), then visual inspection
// no longer suffices to directly view the separate counts. However, for normal
// applications that don't have huge page reference counts, this won't be an
// issue.
//
// Locking: the lockless algorithm described in folio_try_get_rcu()
// provides safe operation for get_user_pages(), folio_mkclean() and
// other calls that race to set up page table entries.
#[inline(always)]
fn lowmem_page_address(page: *const page) -> *mut () {
page_to_virt!(page)
}

// Mask used for checking in page_pool_page_is_pp() below. page->pp_magic is
// OR'ed with PP_SIGNATURE after the allocation in order to preserve bit 0 for
// the head page of compound page and bit 1 for pfmemalloc page, as well as the
// bits used for the DMA index. page_is_pfmemalloc() is checked in
// __page_pool_put_page() to avoid recycling the pfmemalloc page.
#[repr(C)]
#[derive(Clone, Copy)]
struct bio_vec {
    bv_page: *mut page,
    bv_len: u32,
    bv_offset: u32,
}


#[repr(C, packed(4))]
#[derive(Clone, Copy)]
struct bvec_iter {
    bi_sector: u64,
    bi_size: u32,
    bi_idx: u32,
    bi_offset: u32,
}


#[inline(always)]
fn __bvec_iter_bvec(bvecs: *const bio_vec, iter: bvec_iter) -> *const bio_vec {
bvecs.wrapping_offset(iter.bi_idx as isize)
}

#[inline(always)]
fn mp_bvec_iter_page(bvecs: *const bio_vec, iter: bvec_iter) -> *mut page {
unsafe { (*__bvec_iter_bvec(bvecs, iter)).bv_page }
}

#[inline(always)]
fn mp_bvec_iter_len(bvecs: *const bio_vec, iter: bvec_iter) -> u32 {
let mut __v0: u32;
let __UNIQUE_ID_x__535: u32 = unsafe { (*__bvec_iter_bvec(bvecs, iter)).bv_len }.wrapping_sub(iter.bi_offset);
let __UNIQUE_ID_y__536: u32 = iter.bi_size;
loop {
if __types_ok!(__UNIQUE_ID_x__535, __UNIQUE_ID_y__536) == 0 {
unsafe { __compiletime_assert_537() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__535 < __UNIQUE_ID_y__536 { __UNIQUE_ID_x__535 } else { __UNIQUE_ID_y__536 };
__v0
}

#[inline(always)]
fn mp_bvec_iter_offset(bvecs: *const bio_vec, iter: bvec_iter) -> u32 {
unsafe { (*__bvec_iter_bvec(bvecs, iter)).bv_offset }.wrapping_add(iter.bi_offset)
}

#[inline(always)]
fn mp_bvec_iter_page_idx(bvecs: *const bio_vec, iter: bvec_iter) -> u32 {
((mp_bvec_iter_offset(bvecs, iter) as u64) / (_AC!() << 12i32)) as u32
}

#[inline(always)]
fn bvec_iter_offset(bvecs: *const bio_vec, iter: bvec_iter) -> u32 {
((mp_bvec_iter_offset(bvecs, iter) as u64) % (_AC!() << 12i32)) as u32
}

#[inline(always)]
fn bvec_iter_len(bvecs: *const bio_vec, iter: bvec_iter) -> u32 {
let mut __v0: u64;
let __UNIQUE_ID_x__538: u32 = mp_bvec_iter_len(bvecs, iter);
let __UNIQUE_ID_y__539: u64 = (_AC!() << 12i32).wrapping_sub(bvec_iter_offset(bvecs, iter) as u64);
loop {
if __types_ok__4!(__UNIQUE_ID_x__538, __UNIQUE_ID_y__539) == 0 {
unsafe { __compiletime_assert_540() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if (__UNIQUE_ID_x__538 as u64) < __UNIQUE_ID_y__539 { __UNIQUE_ID_x__538 as u64 } else { __UNIQUE_ID_y__539 };
__v0 as u32
}

#[inline(always)]
fn bvec_iter_page(bvecs: *const bio_vec, iter: bvec_iter) -> *mut page {
mp_bvec_iter_page(bvecs, iter).wrapping_offset(mp_bvec_iter_page_idx(bvecs, iter) as isize)
}

#[inline(always)]
fn bvec_iter_bvec(bvecs: *const bio_vec, iter: bvec_iter) -> bio_vec {
let mut __slate_compound_0: bio_vec = unsafe { core::mem::zeroed() };
return unsafe { *{
__slate_compound_0 = bio_vec { bv_page: bvec_iter_page(bvecs, iter), bv_len: bvec_iter_len(bvecs, iter), bv_offset: bvec_iter_offset(bvecs, iter) };
core::ptr::addr_of_mut!(__slate_compound_0)
} };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct block_device {
    bd_start_sect: u64,
    bd_nr_sectors: u64,
    bd_disk: *mut gendisk,
    bd_queue: *mut request_queue,
    bd_stats: *mut disk_stats,
    bd_stamp: u64,
    __bd_flags: __SlateRecord_types_h_190,
    bd_dev: u32,
    bd_mapping: *mut address_space,
    bd_openers: __SlateRecord_types_h_190,
    bd_size_lock: spinlock,
    bd_claiming: *mut (),
    bd_holder: *mut (),
    bd_holder_ops: *const blk_holder_ops,
    bd_holder_lock: mutex,
    bd_holders: i32,
    bd_holder_dir: *mut kobject,
    bd_fsfreeze_count: __SlateRecord_types_h_190,
    bd_fsfreeze_mutex: mutex,
    bd_meta_info: *mut partition_meta_info,
    bd_writers: i32,
    bd_security: *mut (),
    bd_device: device,
}


// BLK_STS_AGAIN should only be returned if RQF_NOWAIT is set
// and the bio would block (cf bio_wouldblock_error())
// BLK_STS_DEV_RESOURCE is returned from the driver to the block layer if
// device related resources are unavailable, but the driver can guarantee
// that the queue will be rerun in the future once resources become
// available again. This is typically the case for device specific
// resources that are consumed for IO. If the driver fails allocating these
// resources, we know that inflight (or pending) IO will free these
// resource upon completion.
//
// This is different from BLK_STS_RESOURCE in that it explicitly references
// a device specific resource. For resources of wider scope, allocation
// failure can happen without having pending IO. This means that we can't
// rely on request completions freeing these resources, as IO may not be in
// flight. Examples of that are kernel memory allocations, DMA mappings, or
// any other system wide resources.
#[repr(C)]
#[derive(Clone, Copy)]
struct bio {
    bi_next: *mut bio,
    bi_bdev: *mut block_device,
    bi_opf: u32,
    bi_flags: u16,
    bi_ioprio: u16,
    bi_write_hint: u8,
    bi_write_stream: u8,
    bi_status: u8,
    bi_bvec_gap_bit: u8,
    __bi_remaining: __SlateRecord_types_h_190,
    bi_io_vec: *mut bio_vec,
    bi_iter: bvec_iter,
    __slate_anon_12: __SlateRecord_blk_types_h_239,
    bi_end_io: Option<unsafe extern "C" fn(*mut bio)>,
    bi_private: *mut (),
    bi_blkg: *mut blkcg_gq,
    issue_time_ns: u64,
    bi_iocost_cost: u64,
    bi_vcnt: u16,
    bi_max_vecs: u16,
    __bi_cnt: __SlateRecord_types_h_190,
    bi_pool: *mut bio_set,
}


fn bio_op(bio: *const bio) -> u32 {
((unsafe { (*bio).bi_opf }) & 255u32) as u32
}

fn op_is_write(op: u32) -> bool {
op & 1 != 0
}

fn op_is_flush(op: u32) -> bool {
op & 393216u32 != 0
}

fn op_is_sync(op: u32) -> bool {
op & 255u32 == 0 || op & 395264u32 != 0
}

fn op_is_discard(op: u32) -> bool {
op & 255u32 == 3
}

fn op_stat_group(op: u32) -> i32 {
if op_is_discard(op as u32) {
return 2;
}
op_is_write(op as u32) as i32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct folio_batch {
    nr: u8,
    i: u8,
    percpu_pvec_drained: bool,
    folios: [*mut folio; 31],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct writeback_control {
    nr_to_write: i64,
    pages_skipped: i64,
    range_start: i64,
    range_end: i64,
    sync_mode: u32,
    __slate_bits_0: __slate_bits::__SlateBits589U0,
    fbatch: folio_batch,
    index: u64,
    saved_err: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mempool {
    lock: spinlock,
    min_nr: i32,
    curr_nr: i32,
    elements: *mut *mut (),
    pool_data: *mut (),
    alloc: Option<unsafe extern "C" fn(u32, *mut ()) -> *mut ()>,
    free: Option<unsafe extern "C" fn(*mut (), *mut ())>,
    wait: wait_queue_head,
}


fn bio_flagged(bio: *const bio, bit: u32) -> bool {
(((unsafe { (*bio).bi_flags }) as i32) as u32) & 1u32 << bit != 0
}

fn bio_set_flag(__v0: *mut bio, bit: u32) {
unsafe {
(*__v0).bi_flags = ((((*__v0).bi_flags as i32) as u32) | 1u32 << bit) as u16;
}
}

fn bio_clear_flag(__v0: *mut bio, bit: u32) {
let __v2: u16 = ((((unsafe { (*__v0).bi_flags }) as i32) as u32) & !(1u32 << bit)) as u16;
unsafe {
(*__v0).bi_flags = __v2;
}
}

fn bio_has_data(bio: *mut bio) -> bool {
let __v0: bool = !bio.is_null() && (unsafe { (*bio).bi_iter.bi_size }) != 0 && (bio_op(bio as *const bio) as u32) != 3;
let __v1: bool = __v0 && (bio_op(bio as *const bio) as u32) != 5;
let __v2: bool = __v1 && (bio_op(bio as *const bio) as u32) != 9;
if __v2 {
return true;
}
false
}

fn bio_advance(bio: *mut bio, nbytes: u32) {
if nbytes == unsafe { (*bio).bi_iter.bi_size } {
unsafe {
(*bio).bi_iter.bi_size = 0;
}
return;
}
unsafe { __bio_advance(bio, nbytes) };
}

// iterate over multi-page bvec
fn bio_endio_status(bio: *mut bio, status: u8) {
unsafe {
(*bio).bi_status = status;
}
unsafe { bio_endio(bio) };
}

fn bio_io_error(bio: *mut bio) {
bio_endio_status(bio, 10u8);
}

fn bio_wouldblock_error(bio: *mut bio) {
bio_endio_status(bio, 12u8);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct bio_list {
    head: *mut bio,
    tail: *mut bio,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bio_set {
    bio_slab: *mut kmem_cache,
    front_pad: u32,
    cache: *mut bio_alloc_cache,
    bio_pool: mempool,
    bvec_pool: mempool,
    back_pad: u32,
    rescue_lock: spinlock,
    rescue_list: bio_list,
    rescue_work: work_struct,
    rescue_workqueue: *mut workqueue_struct,
    cpuhp_dead: hlist_node,
}


fn bio_is_zone_append(bio: *mut bio) -> bool {
if IS_ENABLED!() == 0 {
return false;
}
(bio_op(bio as *const bio) as u32) == 7 || bio_flagged(bio as *const bio, 12u32)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rq_list {
    head: *mut request,
    tail: *mut request,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_plug {
    mq_list: rq_list,
    cached_rqs: rq_list,
    cur_ktime: u64,
    nr_ios: u16,
    rq_count: u16,
    multiple_queues: bool,
    has_elevator: bool,
    cb_list: list_head,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct sbitmap_word {
    word: u64,
    __slate_pad_1: [u8; 56],
    cleared: u64,
    swap_lock: raw_spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sbitmap {
    depth: u32,
    shift: u32,
    map_nr: u32,
    round_robin: bool,
    map: *mut sbitmap_word,
    alloc_hint: *mut u32,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct sbq_wait_state {
    wait: wait_queue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sbitmap_queue {
    sb: sbitmap,
    wake_batch: u32,
    wake_index: __SlateRecord_types_h_190,
    ws: *mut sbq_wait_state,
    ws_active: __SlateRecord_types_h_190,
    min_shallow_depth: u32,
    completion_cnt: __SlateRecord_types_h_190,
    wakeup_cnt: __SlateRecord_types_h_190,
}


fn __map_depth(sb: *const sbitmap, index: i32) -> u32 {
if (index as u32) == unsafe { (*sb).map_nr }.wrapping_sub(1) {
return unsafe { (*sb).depth }.wrapping_sub((index << unsafe { (*sb).shift }) as u32);
}
1u32 << unsafe { (*sb).shift }
}

fn __sbitmap_for_each_set(sb: *mut sbitmap, mut start: u32, r#fn: Option<unsafe extern "C" fn(*mut sbitmap, u32, *mut ()) -> bool>, data: *mut ()) {
let mut depth: u32 = unsafe { core::mem::zeroed() };
let mut word: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut u64 = core::ptr::addr_of_mut!(word) as *mut u64;
let mut scanned: u32 = unsafe { core::mem::zeroed() };
let mut nr: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut u32 = core::ptr::addr_of_mut!(nr) as *mut u32;
let mut index: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut u32 = core::ptr::addr_of_mut!(index) as *mut u32;
unsafe {
{
let __slate_auto_bytes_1 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
scanned = 0;
if start >= (*sb).depth {
start = 0;
}
}
*__slate_slot_12 = SB_NR_TO_INDEX!(sb, start);
*__slate_slot_11 = SB_NR_TO_BIT!(sb, start);
while scanned < (*sb).depth {
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
depth = min_t!(u32, __map_depth(sb as *const sbitmap, *__slate_slot_12 as i32).wrapping_sub(*__slate_slot_11), unsafe { (*sb).depth }.wrapping_sub(scanned));
scanned = scanned.wrapping_add(depth);
*__slate_slot_9 = (*unsafe { (*sb).map }.wrapping_offset(*__slate_slot_12 as isize)).word & !(*unsafe { (*sb).map }.wrapping_offset(*__slate_slot_12 as isize)).cleared;
if *__slate_slot_9 != 0 {
// On the first iteration of the outer loop, we need to add the
// bit offset back to the size of the word for find_next_bit().
// On all other iterations, nr is zero, so this is a noop.
depth = depth.wrapping_add(*__slate_slot_11);
while true {
*__slate_slot_11 = find_next_bit(core::ptr::addr_of_mut!(*__slate_slot_9) as *const u64, depth as u64, *__slate_slot_11 as u64) as u32;
if *__slate_slot_11 >= depth {
break;
} else {
if !r#fn.unwrap()(sb, (*__slate_slot_12 << (*sb).shift).wrapping_add(*__slate_slot_11), data) {
return;
} else {
let __v3: u32 = *__slate_slot_11;
*__slate_slot_11 = __v3.wrapping_add(1);
}
}
}
}
*__slate_slot_11 = 0;
let __v1: u32 = *__slate_slot_12;
let __v0: u32 = __v1.wrapping_add(1);
*__slate_slot_12 = __v0;
if __v0 >= (*sb).map_nr {
*__slate_slot_12 = 0;
}
}
}
}

fn sbitmap_for_each_set(sb: *mut sbitmap, r#fn: Option<unsafe extern "C" fn(*mut sbitmap, u32, *mut ()) -> bool>, data: *mut ()) {
__sbitmap_for_each_set(sb, 0u32, r#fn, data);
}

fn __sbitmap_word(sb: *mut sbitmap, bitnr: u32) -> *mut u64 {
unsafe { core::ptr::addr_of_mut!((*unsafe { (*sb).map }.wrapping_offset(SB_NR_TO_INDEX!(sb, bitnr) as isize)).word) }
}

fn sbitmap_set_bit(sb: *mut sbitmap, bitnr: u32) {
set_bit(SB_NR_TO_BIT!(sb, bitnr) as i64, __sbitmap_word(sb, bitnr) as *mut u64);
}

fn sbitmap_clear_bit(sb: *mut sbitmap, bitnr: u32) {
clear_bit(SB_NR_TO_BIT!(sb, bitnr) as i64, __sbitmap_word(sb, bitnr) as *mut u64);
}

fn sbitmap_test_bit(sb: *mut sbitmap, bitnr: u32) -> i32 {
test_bit__5!(bitnr & (1u32 << unsafe { (*sb).shift }).wrapping_sub(1), __sbitmap_word(sb, bitnr))
}

fn sbq_index_inc(index: i32) -> i32 {
index.wrapping_add(1) & 7i32
}

fn sbq_index_atomic_inc(index: *mut __SlateRecord_types_h_190) {
let old: i32 = atomic_read(index as *const __SlateRecord_types_h_190);
let new: i32 = sbq_index_inc(old);
atomic_cmpxchg(index, old, new);
}

fn sbq_wait_ptr(sbq: *mut sbitmap_queue, wait_index: *mut __SlateRecord_types_h_190) -> *mut sbq_wait_state {
let ws: *mut sbq_wait_state = unsafe { (*sbq).ws }.wrapping_offset(atomic_read(wait_index as *const __SlateRecord_types_h_190) as isize);
sbq_index_atomic_inc(wait_index);
ws
}

// Enough for the string representation of any kind of UUID plus NULL.
// EFI UUID is 36 characters. MSDOS UUID is 11 characters.
#[repr(C)]
#[derive(Clone, Copy)]
struct partition_meta_info {
    uuid: [u8; 37],
    volname: [u8; 64],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_integrity {
    flags: u8,
    csum_type: u8,
    metadata_size: u8,
    pi_offset: u8,
    interval_exp: u8,
    tag_size: u8,
    pi_tuple_size: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gendisk {
    major: i32,
    first_minor: i32,
    minors: i32,
    disk_name: [u8; 32],
    events: u16,
    event_flags: u16,
    part_tbl: xarray,
    part0: *mut block_device,
    fops: *const block_device_operations,
    queue: *mut request_queue,
    private_data: *mut (),
    bio_split: bio_set,
    flags: i32,
    state: u64,
    open_mutex: mutex,
    open_partitions: u32,
    bdi: *mut backing_dev_info,
    queue_kobj: kobject,
    slave_dir: *mut kobject,
    slave_bdevs: list_head,
    random: *mut timer_rand_state,
    ev: *mut disk_events,
    cdi: *mut cdrom_device_info,
    node_id: i32,
    bb: *mut badblocks,
    lockdep_map: lockdep_map,
    diskseq: u64,
    open_mode: u32,
    ia_ranges: *mut blk_independent_access_ranges,
    rqos_state_mutex: mutex,
}


fn blk_op_is_passthrough(mut op: u32) -> bool {
op &= 255u32;
op == 34 || op == 35
}

// supports a volatile write cache
// supports passing on the FUA bit
// do disk/partitions IO accounting
// don't modify data until writeback is done
// always completes in submit context
// supports REQ_NOWAIT
// supports I/O polling
// is a zoned device
// do not send FLUSH/FUA commands despite advertising a write cache
#[repr(C)]
#[derive(Clone, Copy)]
struct queue_limits {
    features: u32,
    flags: u32,
    seg_boundary_mask: u64,
    virt_boundary_mask: u64,
    max_hw_sectors: u32,
    max_dev_sectors: u32,
    chunk_sectors: u32,
    max_sectors: u32,
    max_user_sectors: u32,
    max_segment_size: u32,
    max_fast_segment_size: u32,
    physical_block_size: u32,
    logical_block_size: u32,
    alignment_offset: u32,
    io_min: u32,
    io_opt: u32,
    max_discard_sectors: u32,
    max_hw_discard_sectors: u32,
    max_user_discard_sectors: u32,
    max_secure_erase_sectors: u32,
    max_write_zeroes_sectors: u32,
    max_wzeroes_unmap_sectors: u32,
    max_hw_wzeroes_unmap_sectors: u32,
    max_user_wzeroes_unmap_sectors: u32,
    max_hw_zone_append_sectors: u32,
    max_zone_append_sectors: u32,
    discard_granularity: u32,
    discard_alignment: u32,
    zone_write_granularity: u32,
    atomic_write_hw_max: u32,
    atomic_write_max_sectors: u32,
    atomic_write_hw_boundary: u32,
    atomic_write_boundary_sectors: u32,
    atomic_write_hw_unit_min: u32,
    atomic_write_unit_min: u32,
    atomic_write_hw_unit_max: u32,
    atomic_write_unit_max: u32,
    max_segments: u16,
    max_integrity_segments: u16,
    max_discard_segments: u16,
    max_write_streams: u16,
    write_stream_granularity: u32,
    max_open_zones: u32,
    max_active_zones: u32,
    dma_alignment: u32,
    dma_pad_mask: u32,
    integrity: blk_integrity,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_independent_access_range {
    kobj: kobject,
    sector: u64,
    nr_sectors: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_independent_access_ranges {
    kobj: kobject,
    sysfs_registered: bool,
    nr_ia_ranges: u32,
    ia_range: [blk_independent_access_range; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct request_queue {
    queuedata: *mut (),
    elevator: *mut elevator_queue,
    mq_ops: *const blk_mq_ops,
    queue_ctx: *mut blk_mq_ctx,
    queue_flags: u64,
    rq_timeout: u32,
    queue_depth: u32,
    refs: refcount_struct,
    nr_hw_queues: u32,
    queue_hw_ctx: *mut *mut blk_mq_hw_ctx,
    q_usage_counter: percpu_ref,
    io_lock_cls_key: lock_class_key,
    io_lockdep_map: lockdep_map,
    q_lock_cls_key: lock_class_key,
    q_lockdep_map: lockdep_map,
    last_merge: *mut request,
    queue_lock: spinlock,
    quiesce_depth: i32,
    disk: *mut gendisk,
    mq_kobj: *mut kobject,
    limits: queue_limits,
    dev: *mut device,
    rpm_status: i32,
    pm_only: __SlateRecord_types_h_190,
    stats: *mut blk_queue_stats,
    rq_qos: *mut rq_qos,
    rq_qos_mutex: mutex,
    id: i32,
    nr_requests: u32,
    async_depth: u32,
    timeout: timer_list,
    timeout_work: work_struct,
    nr_active_requests_shared_tags: __SlateRecord_types_h_190,
    sched_shared_tags: *mut blk_mq_tags,
    icq_list: list_head,
    blkcg_pols: [u64; 1],
    root_blkg: *mut blkcg_gq,
    blkg_list: list_head,
    blkcg_mutex: mutex,
    node: i32,
    requeue_lock: spinlock,
    requeue_list: list_head,
    requeue_work: delayed_work,
    blk_trace: *mut blk_trace,
    fq: *mut blk_flush_queue,
    flush_list: list_head,
    elevator_lock: mutex,
    sysfs_lock: mutex,
    limits_lock: mutex,
    unused_hctx_list: list_head,
    unused_hctx_lock: spinlock,
    mq_freeze_depth: i32,
    callback_head: callback_head,
    mq_freeze_wq: wait_queue_head,
    mq_freeze_lock: mutex,
    tag_set: *mut blk_mq_tag_set,
    tag_set_list: list_head,
    debugfs_dir: *mut dentry,
    sched_debugfs_dir: *mut dentry,
    rqos_debugfs_dir: *mut dentry,
    debugfs_mutex: mutex,
}


fn queue_is_mq(q: *mut request_queue) -> bool {
!unsafe { (*q).mq_ops }.is_null()
}

fn queue_rpm_status(q: *mut request_queue) -> i32 {
unsafe { (*q).rpm_status }
}

fn bdev_partno(bdev: *const block_device) -> u8 {
(atomic_read(unsafe { core::ptr::addr_of!((*bdev).__bd_flags) }) & 255) as u8
}

fn bio_needs_zone_write_plugging(bio: *mut bio) -> bool {
false
}

fn blk_zone_plug_bio(bio: *mut bio, nr_segs: u32) -> bool {
false
}

// only poll the hardware once, don't continue until a completion was found
fn bdev_get_queue(bdev: *mut block_device) -> *mut request_queue {
return unsafe { (*bdev).bd_queue }; // this is never NULL
}

fn bdev_is_partition(bdev: *mut block_device) -> bool {
(bdev_partno(bdev as *const block_device) as i32) != 0
}

fn queue_max_segments(q: *const request_queue) -> u16 {
unsafe { (*q).limits.max_segments }
}

fn queue_max_discard_segments(q: *const request_queue) -> u16 {
unsafe { (*q).limits.max_discard_segments }
}

fn queue_logical_block_size(q: *const request_queue) -> u32 {
unsafe { (*q).limits.logical_block_size }
}

fn bdev_logical_block_size(bdev: *mut block_device) -> u32 {
queue_logical_block_size(bdev_get_queue(bdev) as *const request_queue)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct block_device_operations {
    submit_bio: Option<unsafe extern "C" fn(*mut bio)>,
    poll_bio: Option<unsafe extern "C" fn(*mut bio, *mut io_comp_batch, u32) -> i32>,
    open: Option<unsafe extern "C" fn(*mut gendisk, u32) -> i32>,
    release: Option<unsafe extern "C" fn(*mut gendisk)>,
    ioctl: Option<unsafe extern "C" fn(*mut block_device, u32, u32, u64) -> i32>,
    compat_ioctl: Option<unsafe extern "C" fn(*mut block_device, u32, u32, u64) -> i32>,
    check_events: Option<unsafe extern "C" fn(*mut gendisk, u32) -> u32>,
    unlock_native_capacity: Option<unsafe extern "C" fn(*mut gendisk)>,
    getgeo: Option<unsafe extern "C" fn(*mut gendisk, *mut hd_geometry) -> i32>,
    set_read_only: Option<unsafe extern "C" fn(*mut block_device, bool) -> i32>,
    free_disk: Option<unsafe extern "C" fn(*mut gendisk)>,
    swap_slot_free_notify: Option<unsafe extern "C" fn(*mut block_device, u64)>,
    report_zones: Option<unsafe extern "C" fn(*mut gendisk, u64, u32, *mut blk_report_zones_args) -> i32>,
    devnode: Option<unsafe extern "C" fn(*mut gendisk, *mut u16) -> *mut u8>,
    get_unique_id: Option<unsafe extern "C" fn(*mut gendisk, *mut u8, u32) -> i32>,
    owner: *mut module,
    pr_ops: *const pr_ops,
    alternative_gpt_sector: Option<unsafe extern "C" fn(*mut gendisk, *mut u64) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_holder_ops {
    mark_dead: Option<unsafe extern "C" fn(*mut block_device, bool)>,
    sync: Option<unsafe extern "C" fn(*mut block_device)>,
    freeze: Option<unsafe extern "C" fn(*mut block_device) -> i32>,
    thaw: Option<unsafe extern "C" fn(*mut block_device) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_comp_batch {
    req_list: rq_list,
    need_ts: bool,
    complete: Option<unsafe extern "C" fn(*mut io_comp_batch)>,
    poll_ctx: *mut (),
}


// Maximum contiguous integrity buffer allocation.
#[repr(C)]
#[derive(Clone, Copy)]
struct fwnode_handle {
    secondary: *mut fwnode_handle,
    ops: *const fwnode_operations,
    dev: *mut device,
    suppliers: list_head,
    consumers: list_head,
    flags: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fwnode_endpoint {
    port: u32,
    id: u32,
    local_fwnode: *const fwnode_handle,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fwnode_reference_args {
    fwnode: *mut fwnode_handle,
    nargs: u32,
    args: [u64; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fwnode_operations {
    get: Option<unsafe extern "C" fn(*mut fwnode_handle) -> *mut fwnode_handle>,
    put: Option<unsafe extern "C" fn(*mut fwnode_handle)>,
    device_is_available: Option<unsafe extern "C" fn(*const fwnode_handle) -> bool>,
    device_get_match_data: Option<unsafe extern "C" fn(*const fwnode_handle, *const device) -> *const ()>,
    device_dma_supported: Option<unsafe extern "C" fn(*const fwnode_handle) -> bool>,
    device_get_dma_attr: Option<unsafe extern "C" fn(*const fwnode_handle) -> u32>,
    property_present: Option<unsafe extern "C" fn(*const fwnode_handle, *const u8) -> bool>,
    property_read_bool: Option<unsafe extern "C" fn(*const fwnode_handle, *const u8) -> bool>,
    property_read_int_array: Option<unsafe extern "C" fn(*const fwnode_handle, *const u8, u32, *mut (), u64) -> i32>,
    property_read_string_array: Option<unsafe extern "C" fn(*const fwnode_handle, *const u8, *mut *const u8, u64) -> i32>,
    get_name: Option<unsafe extern "C" fn(*const fwnode_handle) -> *const u8>,
    get_name_prefix: Option<unsafe extern "C" fn(*const fwnode_handle) -> *const u8>,
    get_parent: Option<unsafe extern "C" fn(*const fwnode_handle) -> *mut fwnode_handle>,
    get_next_child_node: Option<unsafe extern "C" fn(*const fwnode_handle, *mut fwnode_handle) -> *mut fwnode_handle>,
    get_named_child_node: Option<unsafe extern "C" fn(*const fwnode_handle, *const u8) -> *mut fwnode_handle>,
    get_reference_args: Option<unsafe extern "C" fn(*const fwnode_handle, *const u8, *const u8, u32, u32, *mut fwnode_reference_args) -> i32>,
    graph_get_next_endpoint: Option<unsafe extern "C" fn(*const fwnode_handle, *mut fwnode_handle) -> *mut fwnode_handle>,
    graph_get_remote_endpoint: Option<unsafe extern "C" fn(*const fwnode_handle) -> *mut fwnode_handle>,
    graph_get_port_parent: Option<unsafe extern "C" fn(*mut fwnode_handle) -> *mut fwnode_handle>,
    graph_parse_endpoint: Option<unsafe extern "C" fn(*const fwnode_handle, *mut fwnode_endpoint) -> i32>,
    iomap: Option<unsafe extern "C" fn(*mut fwnode_handle, i32) -> *mut ()>,
    irq_get: Option<unsafe extern "C" fn(*const fwnode_handle, u32) -> i32>,
    add_links: Option<unsafe extern "C" fn(*mut fwnode_handle) -> i32>,
}


// flags that prevent us from merging requests:
#[repr(C)]
#[derive(Clone, Copy)]
struct request {
    q: *mut request_queue,
    mq_ctx: *mut blk_mq_ctx,
    mq_hctx: *mut blk_mq_hw_ctx,
    cmd_flags: u32,
    rq_flags: u32,
    tag: i32,
    internal_tag: i32,
    timeout: u32,
    __data_len: u32,
    __sector: u64,
    bio: *mut bio,
    biotail: *mut bio,
    __slate_anon_12: __SlateRecord_blk_mq_h_125,
    part: *mut block_device,
    alloc_time_ns: u64,
    start_time_ns: u64,
    io_start_time_ns: u64,
    stats_sectors: u16,
    nr_phys_segments: u16,
    nr_integrity_segments: u16,
    phys_gap_bit: u8,
    state: u32,
    r#ref: __SlateRecord_types_h_190,
    deadline: u64,
    __slate_anon_24: __SlateRecord_blk_mq_h_182,
    __slate_anon_25: __SlateRecord_blk_mq_h_193,
    elv: __SlateRecord_blk_mq_h_202,
    flush: __SlateRecord_blk_mq_h_207,
    fifo_time: u64,
    end_io: Option<unsafe extern "C" fn(*mut request, u8, *const io_comp_batch) -> u32>,
    end_io_data: *mut (),
}


fn req_op(req: *const request) -> u32 {
((unsafe { (*req).cmd_flags }) & 255u32) as u32
}

fn blk_rq_is_passthrough(rq: *mut request) -> bool {
blk_op_is_passthrough(unsafe { (*rq).cmd_flags })
}

fn req_get_ioprio(req: *mut request) -> u16 {
if !unsafe { (*req).bio }.is_null() {
return unsafe { (*(*req).bio).bi_ioprio };
}
0
}

fn rq_list_empty(rl: *const rq_list) -> i32 {
unsafe { (*rl).head }.is_null() as i32
}

fn rq_list_add_tail(rl: *mut rq_list, rq: *mut request) {
unsafe {
(*rq).__slate_anon_12.rq_next = core::ptr::null_mut::<request>();
}
if !unsafe { (*rl).tail }.is_null() {
unsafe {
(*(*rl).tail).__slate_anon_12.rq_next = rq;
}
} else {
unsafe {
(*rl).head = rq;
}
}
unsafe {
(*rl).tail = rq;
}
}

fn rq_list_add_head(rl: *mut rq_list, rq: *mut request) {
unsafe {
(*rq).__slate_anon_12.rq_next = (*rl).head;
}
unsafe {
(*rl).head = rq;
}
if unsafe { (*rl).tail }.is_null() {
unsafe {
(*rl).tail = rq;
}
}
}

fn rq_list_pop(rl: *mut rq_list) -> *mut request {
let rq: *mut request = unsafe { (*rl).head };
if !rq.is_null() {
unsafe {
(*rl).head = (*(*rl).head).__slate_anon_12.rq_next;
}
if unsafe { (*rl).head }.is_null() {
unsafe {
(*rl).tail = core::ptr::null_mut::<request>();
}
}
unsafe {
(*rq).__slate_anon_12.rq_next = core::ptr::null_mut::<request>();
}
}
rq
}

fn rq_list_peek(rl: *mut rq_list) -> *mut request {
unsafe { (*rl).head }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_hw_ctx {
    __slate_anon_0: __SlateRecord_blk_mq_h_323,
    run_work: delayed_work,
    cpumask: [cpumask; 1],
    next_cpu: i32,
    next_cpu_batch: i32,
    flags: u64,
    sched_data: *mut (),
    queue: *mut request_queue,
    fq: *mut blk_flush_queue,
    driver_data: *mut (),
    ctx_map: sbitmap,
    dispatch_from: *mut blk_mq_ctx,
    dispatch_busy: u32,
    r#type: u16,
    nr_ctx: u16,
    ctxs: *mut *mut blk_mq_ctx,
    dispatch_wait_lock: spinlock,
    dispatch_wait: wait_queue_entry,
    wait_index: __SlateRecord_types_h_190,
    tags: *mut blk_mq_tags,
    sched_tags: *mut blk_mq_tags,
    numa_node: i32,
    queue_num: u32,
    nr_active: __SlateRecord_types_h_190,
    cpuhp_online: hlist_node,
    cpuhp_dead: hlist_node,
    kobj: kobject,
    debugfs_dir: *mut dentry,
    sched_debugfs_dir: *mut dentry,
    hctx_list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_queue_map {
    mq_map: *mut u32,
    nr_queues: u32,
    queue_offset: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_tag_set {
    ops: *const blk_mq_ops,
    map: [blk_mq_queue_map; 3],
    nr_maps: u32,
    nr_hw_queues: u32,
    queue_depth: u32,
    reserved_tags: u32,
    cmd_size: u32,
    numa_node: i32,
    timeout: u32,
    flags: u32,
    driver_data: *mut (),
    tags: *mut *mut blk_mq_tags,
    shared_tags: *mut blk_mq_tags,
    tag_list_lock: mutex,
    tag_list: list_head,
    srcu: *mut srcu_struct,
    tags_srcu: srcu_struct,
    update_nr_hwq_lock: rw_semaphore,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_queue_data {
    rq: *mut request,
    last: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_ops {
    queue_rq: Option<unsafe extern "C" fn(*mut blk_mq_hw_ctx, *const blk_mq_queue_data) -> u8>,
    commit_rqs: Option<unsafe extern "C" fn(*mut blk_mq_hw_ctx)>,
    queue_rqs: Option<unsafe extern "C" fn(*mut rq_list)>,
    get_budget: Option<unsafe extern "C" fn(*mut request_queue) -> i32>,
    put_budget: Option<unsafe extern "C" fn(*mut request_queue, i32)>,
    set_rq_budget_token: Option<unsafe extern "C" fn(*mut request, i32)>,
    get_rq_budget_token: Option<unsafe extern "C" fn(*mut request) -> i32>,
    timeout: Option<unsafe extern "C" fn(*mut request) -> u32>,
    poll: Option<unsafe extern "C" fn(*mut blk_mq_hw_ctx, *mut io_comp_batch) -> i32>,
    complete: Option<unsafe extern "C" fn(*mut request)>,
    init_hctx: Option<unsafe extern "C" fn(*mut blk_mq_hw_ctx, *mut (), u32) -> i32>,
    exit_hctx: Option<unsafe extern "C" fn(*mut blk_mq_hw_ctx, u32)>,
    init_request: Option<unsafe extern "C" fn(*mut blk_mq_tag_set, *mut request, u32, i32) -> i32>,
    exit_request: Option<unsafe extern "C" fn(*mut blk_mq_tag_set, *mut request, u32)>,
    cleanup_rq: Option<unsafe extern "C" fn(*mut request)>,
    busy: Option<unsafe extern "C" fn(*mut request_queue) -> bool>,
    map_queues: Option<unsafe extern "C" fn(*mut blk_mq_tag_set)>,
    show_rq: Option<unsafe extern "C" fn(*mut seq_file, *mut request)>,
}


#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_alloc_tag_set")]
extern "C" fn blk_mq_alloc_tag_set(set: *mut blk_mq_tag_set) -> i32 {
let mut __slate_compound_26: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_28: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_29: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_31: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_32: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_34: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut _old_1: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old_1) as *mut *mut alloc_tag;
let mut _res_1: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut () = core::ptr::addr_of_mut!(_res_1) as *mut *mut ();
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_y__1017: u32 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__1016: u32 = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_25: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
unsafe {
let __slate_auto_bytes_4 = __slate_slot_25 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
while false {
}
if (*set).nr_hw_queues == 0 {
return 22i32.wrapping_neg();
} else {
if (*set).queue_depth == 0 {
return 22i32.wrapping_neg();
} else {
if (*set).queue_depth < unsafe { (*set).reserved_tags }.wrapping_add(1) {
return 22i32.wrapping_neg();
} else {
if (*(*set).ops).queue_rq == None {
return 22i32.wrapping_neg();
} else {
if (((*(*set).ops).get_budget == None) as i32) ^ (((*(*set).ops).put_budget == None) as i32) != 0 {
return 22i32.wrapping_neg();
} else {
if (*set).queue_depth > 10240 {
while false {
}
_printk(c"\x016blk-mq: reduced tag depth to %u\n".as_ptr() as *const u8, 10240i32);
(*set).queue_depth = 10240;
}
if (*set).nr_maps == 0 {
(*set).nr_maps = 1;
} else {
if (*set).nr_maps > 3 {
return 22i32.wrapping_neg();
}
}
// If a crashdump is active, then we are potentially in a very
// memory constrained environment. Limit us to  64 tags to prevent
// using too much memory.
if is_kdump_kernel() {
__UNIQUE_ID_x__1016 = 64;
__UNIQUE_ID_y__1017 = (*set).queue_depth;
while false {
}
(*set).queue_depth = if __UNIQUE_ID_x__1016 < __UNIQUE_ID_y__1017 { __UNIQUE_ID_x__1016 } else { __UNIQUE_ID_y__1017 };
}
// There is no use for more h/w queues than cpus if we just have
// a single map
if (*set).nr_maps == 1 && (*set).nr_hw_queues > nr_cpu_ids {
(*set).nr_hw_queues = nr_cpu_ids;
}
'__join_2: {
if (*set).flags & 16 != 0 {
__obj_size = size_mul(core::mem::size_of::<srcu_struct>() as u64, 1u64);
let __slate_auto_bytes_3 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_14 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_15 = _kmalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_32 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_32)
});
alloc_tag_restore!();
} else {
*__slate_slot_15 = _kmalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_34 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_34)
});
}
(*set).srcu = *__slate_slot_15 as *mut srcu_struct;
if unsafe { (*set).srcu }.is_null() {
return 12i32.wrapping_neg();
} else {
let __v13: i32 = __init_srcu_struct((*set).srcu, c"set->srcu".as_ptr() as *const u8, core::ptr::addr_of_mut!(__srcu_key__4894));
*__slate_slot_24 = __v13;
if *__slate_slot_24 != 0 {
break '__join_2;
}
}
}
let __v12: i32 = __init_srcu_struct(core::ptr::addr_of_mut!((*set).tags_srcu), c"&set->tags_srcu".as_ptr() as *const u8, core::ptr::addr_of_mut!(__srcu_key__4898));
*__slate_slot_24 = __v12;
if *__slate_slot_24 == 0 {
loop {
__init_rwsem(core::ptr::addr_of_mut!((*set).update_nr_hwq_lock), c"&set->update_nr_hwq_lock".as_ptr() as *const u8, core::ptr::addr_of_mut!(__key__4902));
if true {
break;
}
}
*__slate_slot_24 = 12i32.wrapping_neg();
let __slate_auto_bytes_2 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_8 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_9 = _kmalloc_array_node_noprof((*set).nr_hw_queues as u64, core::mem::size_of::<*mut blk_mq_tags>() as u64, 3520u32, (*set).numa_node, *{
__slate_compound_29 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_29)
});
alloc_tag_restore!();
} else {
*__slate_slot_9 = _kmalloc_array_node_noprof((*set).nr_hw_queues as u64, core::mem::size_of::<*mut blk_mq_tags>() as u64, 3520u32, (*set).numa_node, *{
__slate_compound_31 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_31)
});
}
(*set).tags = *__slate_slot_9 as *mut *mut blk_mq_tags;
if !unsafe { (*set).tags }.is_null() {
*__slate_slot_25 = 0;
'__join_9: {
while (*__slate_slot_25 as u32) < (*set).nr_maps {
let __slate_auto_bytes_1 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_4 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_5 = _kmalloc_array_node_noprof(nr_cpu_ids as u64, core::mem::size_of::<u32>() as u64, 3520u32, (*set).numa_node, *{
__slate_compound_26 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_26)
});
alloc_tag_restore!();
} else {
*__slate_slot_5 = _kmalloc_array_node_noprof(nr_cpu_ids as u64, core::mem::size_of::<u32>() as u64, 3520u32, (*set).numa_node, *{
__slate_compound_28 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_28)
});
}
let __v7: *mut () = *__slate_slot_5;
(*((*set).map.as_mut_ptr() as *mut blk_mq_queue_map).wrapping_offset(*__slate_slot_25 as isize)).mq_map = __v7 as *mut u32;
if unsafe { (*((*set).map.as_mut_ptr() as *mut blk_mq_queue_map).wrapping_offset(*__slate_slot_25 as isize)).mq_map }.is_null() {
break '__join_9;
} else {
(*((*set).map.as_mut_ptr() as *mut blk_mq_queue_map).wrapping_offset(*__slate_slot_25 as isize)).nr_queues = (*set).nr_hw_queues;
let __v3: i32 = *__slate_slot_25;
*__slate_slot_25 = __v3.wrapping_add(1);
}
}
blk_mq_update_queue_map(set);
*__slate_slot_24 = blk_mq_alloc_set_map_and_rqs(set);
if *__slate_slot_24 == 0 {
loop {
__mutex_init(core::ptr::addr_of_mut!((*set).tag_list_lock), c"&set->tag_list_lock".as_ptr() as *const u8, core::ptr::addr_of_mut!(__key__4926));
if true {
break;
}
}
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*set).tag_list));
return 0;
}
}
*__slate_slot_25 = 0;
while (*__slate_slot_25 as u32) < (*set).nr_maps {
kfree((*((*set).map.as_mut_ptr() as *mut blk_mq_queue_map).wrapping_offset(*__slate_slot_25 as isize)).mq_map as *const ());
(*((*set).map.as_mut_ptr() as *mut blk_mq_queue_map).wrapping_offset(*__slate_slot_25 as isize)).mq_map = core::ptr::null_mut::<u32>();
let __v1: i32 = *__slate_slot_25;
*__slate_slot_25 = __v1.wrapping_add(1);
}
kfree((*set).tags as *const ());
(*set).tags = core::ptr::null_mut::<*mut blk_mq_tags>();
}
cleanup_srcu_struct(core::ptr::addr_of_mut!((*set).tags_srcu));
}
if (*set).flags & 16 != 0 {
cleanup_srcu_struct((*set).srcu);
}
}
if (*set).flags & 16 != 0 {
kfree((*set).srcu as *const ());
}
return *__slate_slot_24;
}
}
}
}
}
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_free_request")]
extern "C" fn blk_mq_free_request(rq: *mut request) {
let q: *mut request_queue = unsafe { (*rq).q };
blk_mq_finish_request(rq);
rq_qos_done(q, rq);
loop {
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_921() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__4!((*rq).state, 0i32);
if true {
break;
}
}
if req_ref_put_and_test(rq) {
__blk_mq_free_request(rq);
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_tags {
    nr_tags: u32,
    nr_reserved_tags: u32,
    active_queues: u32,
    bitmap_tags: sbitmap_queue,
    breserved_tags: sbitmap_queue,
    rqs: *mut *mut request,
    static_rqs: *mut *mut request,
    page_list: list_head,
    lock: spinlock,
    callback_head: callback_head,
}


fn blk_mq_rq_state(rq: *mut request) -> u32 {
let mut __v0: u32;
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_575() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*rq).state) as *const u32) };
__v0
}

fn blk_mq_request_started(rq: *mut request) -> i32 {
((blk_mq_rq_state(rq) as u32) != 0) as i32
}

fn blk_mq_request_completed(rq: *mut request) -> i32 {
((blk_mq_rq_state(rq) as u32) == 2) as i32
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_start_request")]
extern "C" fn blk_mq_start_request(rq: *mut request) {
let q: *mut request_queue = unsafe { (*rq).q };
trace_block_rq_issue(rq);
let __v0: bool = test_bit!(7i32, unsafe { core::ptr::addr_of_mut!((*q).queue_flags) }) != 0 && !blk_rq_is_passthrough(rq);
if __v0 {
unsafe {
(*rq).io_start_time_ns = blk_time_get_ns();
}
unsafe {
(*rq).stats_sectors = blk_rq_sectors(rq as *const request) as u16;
}
unsafe {
(*rq).rq_flags = (*rq).rq_flags | 2048u32;
}
rq_qos_issue(q, rq);
}
let mut __v5: i64;
let __ret_warn_on: i32 = ((blk_mq_rq_state(rq) as u32) != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("940: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 940b - ., 3; .popsection\n# {0}", const 940 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_blk_mq_start_request_6, sym __slate_asm_string_blk_mq_start_request_7, const 1364 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("941: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 941b - ., 4; .popsection\n# {0}", const 941 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v5 = (__ret_warn_on != 0) as i64;
unsafe { blk_add_timer(rq) };
loop {
while false {
}
__WRITE_ONCE__4!((*rq).state, 1i32);
if true {
break;
}
}
unsafe {
*unsafe { (*(*(*rq).mq_hctx).tags).rqs }.wrapping_offset((*rq).tag as isize) = rq;
}
let __v13: bool = blk_integrity_rq(rq as *const request) && (req_op(rq as *const request) as u32) == 1;
if __v13 {
unsafe { blk_integrity_prepare(rq) };
}
if !unsafe { (*rq).bio }.is_null() && (unsafe { (*(*rq).bio).bi_opf }) & 4194304u32 != 0 {
loop {
while false {
}
__WRITE_ONCE__8!((*unsafe { (*rq).bio }).__slate_anon_12.bi_cookie, unsafe { (*(*rq).mq_hctx).queue_num });
if true {
break;
}
}
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_end_request_batch")]
extern "C" fn blk_mq_end_request_batch(iob: *mut io_comp_batch) {
let mut tags: __SlateAlign16<[i32; 32]> = __SlateAlign16([0i32; 32]);
let mut nr_tags: i32 = 0;
let mut cur_hctx: *mut blk_mq_hw_ctx = core::ptr::null_mut::<blk_mq_hw_ctx>();
let mut rq: *mut request = unsafe { core::mem::zeroed() };
let mut now: u64 = 0;
if unsafe { (*iob).need_ts } {
now = blk_time_get_ns();
}
loop {
let __v7: *mut request = rq_list_pop(unsafe { core::ptr::addr_of_mut!((*iob).req_list) });
rq = __v7;
if __v7.is_null() {
break;
}
prefetch!(unsafe { (*rq).bio });
prefetch!(unsafe { (*rq).__slate_anon_12.rq_next });
blk_complete_request(rq);
if unsafe { (*iob).need_ts } {
__blk_mq_end_request_acct(rq, now);
}
blk_mq_finish_request(rq);
rq_qos_done(unsafe { (*rq).q }, rq);
// If end_io handler returns NONE, then it still has
// ownership of the request.
let __v0: bool = (unsafe { (*rq).end_io }) != None && ((unsafe { unsafe { (*rq).end_io }.unwrap()(rq, 0u8, iob as *const io_comp_batch) }) as u32) == 0;
if !__v0 {
loop {
while false {
}
__WRITE_ONCE__4!((*rq).state, 0i32);
if true {
break;
}
}
if req_ref_put_and_test(rq) {
blk_crypto_free_request(rq);
blk_pm_mark_last_busy(rq);
if nr_tags == 32 || cur_hctx != unsafe { (*rq).mq_hctx } {
if !cur_hctx.is_null() {
blk_mq_flush_tag_batch(cur_hctx, tags.0.as_mut_ptr() as *mut i32, nr_tags);
}
nr_tags = 0;
cur_hctx = unsafe { (*rq).mq_hctx };
}
let __v5: i32 = nr_tags;
nr_tags = __v5.wrapping_add(1);
tags.0[__v5 as usize] = unsafe { (*rq).tag };
}
}
}
if nr_tags != 0 {
blk_mq_flush_tag_batch(cur_hctx, tags.0.as_mut_ptr() as *mut i32, nr_tags);
}
}

fn blk_mq_need_time_stamp(rq: *mut request) -> bool {
(unsafe { (*rq).rq_flags }) & 2336u32 != 0
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_complete_request_remote")]
extern "C" fn blk_mq_complete_request_remote(rq: *mut request) -> bool {
loop {
while false {
}
__WRITE_ONCE__4!((*rq).state, 2i32);
if true {
break;
}
}
// For request which hctx has only one ctx mapping,
// or a polled request, always complete locally,
// it's pointless to redirect the completion.
let __v3: bool = if ((unsafe { (*(*rq).mq_hctx).nr_ctx }) as i32) == 1 {
let mut __v4: i32;
let mut pscr_ret__: i32 = 0;
__verify_pcpu_ptr__3!();
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v6: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v6 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v6;
}
2 => {
let mut __v8: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v8 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v8;
}
4 => {
let mut __v9: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v9 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v9;
}
8 => {
let mut __v10: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v10 = pfo_val__ as i32;
pscr_ret__ = __v10;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v4 = pscr_ret__;
(unsafe { (*(*rq).mq_ctx).cpu }) == (__v4 as u32)
} else {
false
};
if __v3 || (unsafe { (*rq).cmd_flags }) & 4194304u32 != 0 {
return false;
}
if blk_mq_complete_need_ipi(rq) {
blk_mq_complete_send_ipi(rq);
return true;
}
if (unsafe { (*(*rq).q).nr_hw_queues }) == 1 {
blk_mq_raise_softirq(rq);
return true;
}
false
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_quiesce_tagset")]
extern "C" fn blk_mq_quiesce_tagset(set: *mut blk_mq_tag_set) {
let mut q: *mut request_queue = unsafe { core::mem::zeroed() };
rcu_read_lock();
let mut __v0: *mut request_queue;
let mut __v1: *mut list_head;
loop {
if !(((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_905() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*set).tag_list)).next) as *const *mut list_head) };
__v0 = ((unsafe { (__v1 as *const u8).wrapping_offset(-(offsetof__4!() as isize)) }) as *mut ()) as *mut request_queue;
q = __v0;
while (unsafe { core::ptr::addr_of_mut!((*q).tag_set_list) }) != unsafe { core::ptr::addr_of_mut!((*set).tag_list) } {
if blk_queue_skip_tagset_quiesce!(q) == 0 {
blk_mq_quiesce_queue_nowait(q);
}
let mut __v5: *mut request_queue;
let mut __v6: *mut list_head;
loop {
if !(((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_906() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v6 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*q).tag_set_list.next) as *const *mut list_head) };
__v5 = ((unsafe { (__v6 as *const u8).wrapping_offset(-(offsetof__4!() as isize)) }) as *mut ()) as *mut request_queue;
q = __v5;
}
rcu_read_unlock();
blk_mq_wait_quiesce_done(set);
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_unquiesce_tagset")]
extern "C" fn blk_mq_unquiesce_tagset(set: *mut blk_mq_tag_set) {
let mut q: *mut request_queue = unsafe { core::mem::zeroed() };
rcu_read_lock();
let mut __v0: *mut request_queue;
let mut __v1: *mut list_head;
loop {
if !(((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_908() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*set).tag_list)).next) as *const *mut list_head) };
__v0 = ((unsafe { (__v1 as *const u8).wrapping_offset(-(offsetof__4!() as isize)) }) as *mut ()) as *mut request_queue;
q = __v0;
while (unsafe { core::ptr::addr_of_mut!((*q).tag_set_list) }) != unsafe { core::ptr::addr_of_mut!((*set).tag_list) } {
if blk_queue_skip_tagset_quiesce!(q) == 0 {
blk_mq_unquiesce_queue(q);
}
let mut __v5: *mut request_queue;
let mut __v6: *mut list_head;
loop {
if !(((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_909() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v6 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*q).tag_set_list.next) as *const *mut list_head) };
__v5 = ((unsafe { (__v6 as *const u8).wrapping_offset(-(offsetof__4!() as isize)) }) as *mut ()) as *mut request_queue;
q = __v5;
}
rcu_read_unlock();
}

fn blk_mq_freeze_queue(q: *mut request_queue) -> u32 {
let memflags: u32 = memalloc_noio_save();
blk_mq_freeze_queue_nomemsave(q);
memflags
}

fn blk_mq_unfreeze_queue(q: *mut request_queue, memflags: u32) {
blk_mq_unfreeze_queue_nomemrestore(q);
memalloc_noio_restore(memflags);
}

fn queue_hctx(q: *mut request_queue, id: i32) -> *mut blk_mq_hw_ctx {
rcu_read_lock();
let mut __v0: *mut *mut blk_mq_hw_ctx;
let mut __v1: *mut *mut blk_mq_hw_ctx;
loop {
if !(__native_word__10!() || ((const { core::mem::size_of::<*mut *mut blk_mq_hw_ctx>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_579() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*q).queue_hw_ctx) as *const *mut *mut blk_mq_hw_ctx) };
let __UNIQUE_ID_rcu_578: *mut *mut blk_mq_hw_ctx = __v1;
loop {
let __v4: bool = if false {
let __v5: bool = false || rcu_read_lock_held() != 0;
!__v5
} else {
false
};
if !__v4 {
break;
}
}
__v0 = __UNIQUE_ID_rcu_578;
let hctx: *mut blk_mq_hw_ctx = unsafe { *__v0.wrapping_offset(id as isize) };
rcu_read_unlock();
hctx
}

fn blk_rq_pos(rq: *const request) -> u64 {
unsafe { (*rq).__sector }
}

fn blk_rq_bytes(rq: *const request) -> u32 {
unsafe { (*rq).__data_len }
}

fn blk_rq_cur_bytes(rq: *const request) -> i32 {
let mut __slate_compound_0: bio_vec = unsafe { core::mem::zeroed() };
if unsafe { (*rq).bio }.is_null() {
return 0;
}
if !bio_has_data(unsafe { (*rq).bio }) {
return (unsafe { (*(*rq).bio).bi_iter.bi_size }) as i32;
}
// dataless requests such as discard
(unsafe { (*{
__slate_compound_0 = bio_iovec!(unsafe { (*rq).bio });
core::ptr::addr_of_mut!(__slate_compound_0)
}).bv_len }) as i32
}

fn blk_rq_sectors(rq: *const request) -> u32 {
blk_rq_bytes(rq) >> 9i32
}

fn blk_rq_cur_sectors(rq: *const request) -> u32 {
(blk_rq_cur_bytes(rq) >> 9i32) as u32
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_update_request")]
extern "C" fn blk_update_request(req: *mut request, error: u8, mut nr_bytes: u32) -> bool {
let is_flush: bool = (unsafe { (*req).rq_flags }) & 2u32 != 0;
let quiet: bool = (unsafe { (*req).rq_flags }) & 128u32 != 0;
let mut total_bytes: i32 = 0;
trace_block_rq_complete(req, error, nr_bytes);
if unsafe { (*req).bio }.is_null() {
return false;
}
let __v0: bool = blk_integrity_rq(req as *const request) && (req_op(req as *const request) as u32) == 0;
if __v0 && (error as i32) == 0 {
unsafe { blk_integrity_complete(req, nr_bytes) };
}
// Upper layers may call blk_crypto_evict_key() anytime after the last
// bio_endio().  Therefore, the keyslot must be released before that.
let __v1: bool = blk_crypto_rq_has_keyslot(req) && nr_bytes >= blk_rq_bytes(req as *const request);
if __v1 {
unsafe { __blk_crypto_rq_put_keyslot(req) };
}
let __v2: bool = error != 0 && !blk_rq_is_passthrough(req);
let __v3: bool = __v2 && !quiet && test_bit!(2i32, unsafe { core::ptr::addr_of_mut!((*(*(*req).q).disk).state) }) == 0;
if __v3 {
blk_print_req_error(req, error);
trace_block_rq_error(req, error, nr_bytes);
}
blk_account_io_completion(req, nr_bytes);
while !unsafe { (*req).bio }.is_null() {
let bio: *mut bio = unsafe { (*req).bio };
let mut bio_bytes: u32 = 0;
let mut __v5: u32;
let __UNIQUE_ID_x__925: u32 = unsafe { (*bio).bi_iter.bi_size };
let __UNIQUE_ID_y__926: u32 = nr_bytes;
while false {
}
__v5 = if __UNIQUE_ID_x__925 < __UNIQUE_ID_y__926 { __UNIQUE_ID_x__925 } else { __UNIQUE_ID_y__926 };
bio_bytes = __v5;
if unlikely__5!(error) != 0 {
unsafe {
(*bio).bi_status = error;
}
}
if bio_bytes == unsafe { (*bio).bi_iter.bi_size } {
unsafe {
(*req).bio = (*bio).bi_next;
}
} else {
if bio_is_zone_append(bio) && (error as i32) == 0 {
// Partial zone append completions cannot be supported
// as the BIO fragments may end up not being written
// sequentially.
unsafe {
(*bio).bi_status = 10;
}
}
}
// Completion has already been traced
bio_clear_flag(bio, 6u32);
if unlikely__2!(quiet) != 0 {
bio_set_flag(bio, 2u32);
}
bio_advance(bio, bio_bytes);
// Don't actually finish bio if it's part of flush sequence
if (unsafe { (*bio).bi_iter.bi_size }) == 0 {
if blk_req_bio_is_zone_append(req, bio) {
blk_zone_append_update_request_bio(req, bio);
}
if !is_flush {
unsafe { bio_endio(bio) };
}
}
total_bytes = (total_bytes as u32).wrapping_add(bio_bytes) as i32;
nr_bytes = nr_bytes.wrapping_sub(bio_bytes);
if nr_bytes == 0 {
break;
}
}
// completely done
if unsafe { (*req).bio }.is_null() {
// Reset counters so that the request stacking driver
// can find how many bytes remain in the request
// later.
unsafe {
(*req).__data_len = 0;
}
return false;
}
let __v13: u32 = unsafe { (*req).__data_len };
unsafe {
(*req).__data_len = __v13.wrapping_sub(total_bytes as u32);
}
// update sector only for requests with clear definition of sector
if !blk_rq_is_passthrough(req) {
let __v16: u64 = unsafe { (*req).__sector };
unsafe {
(*req).__sector = __v16.wrapping_add((total_bytes >> 9i32) as u64);
}
}
// mixed attributes always follow the first bio
if (unsafe { (*req).rq_flags }) & 4u32 != 0 {
let __v20: u32 = (unsafe { (*req).cmd_flags }) & 4294965503u32;
unsafe {
(*req).cmd_flags = __v20;
}
unsafe {
(*req).cmd_flags = (*req).cmd_flags | (*(*req).bio).bi_opf & 1792u32;
}
}
if (unsafe { (*req).rq_flags }) & 4096u32 == 0 {
// If total number of sectors is less than the first segment
// size, something has gone terribly wrong.
if blk_rq_bytes(req as *const request) < (blk_rq_cur_bytes(req as *const request) as u32) {
blk_dump_rq_flags(req, c"request botched".as_ptr() as *mut u8);
unsafe {
(*req).__data_len = blk_rq_cur_bytes(req as *const request) as u32;
}
}
// recalculate the number of segments
unsafe {
(*req).nr_phys_segments = blk_recalc_rq_segments(req) as u16;
}
}
true
}

fn blk_rq_passthrough_stats(rq: *mut request, q: *mut request_queue) -> bool {
let bio: *mut bio = unsafe { (*rq).bio };
if blk_queue_passthrough_stat!(q) == 0 {
return false;
}
// Requests without a bio do not transfer data.
if bio.is_null() {
return false;
}
// Stats are accumulated in the bdev, so must have one attached to a
// bio to track stats. Most drivers do not set the bdev for passthrough
// requests, but nvme is one that will set it.
if unsafe { (*bio).bi_bdev }.is_null() {
return false;
}
// We don't know what a passthrough command does, but we know the
// payload size and data direction. Ensuring the size is aligned to the
// block size filters out most commands with payloads that don't
// represent sector access.
if blk_rq_bytes(rq as *const request) & bdev_logical_block_size(unsafe { (*bio).bi_bdev }).wrapping_sub(1) != 0 {
return false;
}
true
}

#[repr(C)]
#[derive(Clone, Copy)]
struct bio_integrity_payload {
    bip_iter: bvec_iter,
    bip_vcnt: u16,
    bip_max_vcnt: u16,
    bip_flags: u16,
    app_tag: u16,
    bip_vec: *mut bio_vec,
}


fn bio_integrity(bio: *mut bio) -> *mut bio_integrity_payload {
core::ptr::null_mut::<bio_integrity_payload>()
}

fn bio_integrity_prep(bio: *mut bio, action: u32) {
}

fn blk_rq_count_integrity_sg(q: *mut request_queue, b: *mut bio) -> i32 {
0
}

fn blk_get_integrity(disk: *mut gendisk) -> *mut blk_integrity {
core::ptr::null_mut::<blk_integrity>()
}

fn queue_max_integrity_segments(q: *const request_queue) -> u16 {
0
}

fn blk_integrity_rq(rq: *const request) -> bool {
false
}

fn bio_integrity_action(bio: *mut bio) -> u32 {
if blk_get_integrity(unsafe { (*(*bio).bi_bdev).bd_disk }).is_null() {
return 0;
}
if !bio_integrity(bio).is_null() {
return 0;
}
unsafe { __bio_integrity_action(bio) }
}

// reset direct map and flush TLB on unmap, can't be freed in atomic context
#[repr(C)]
#[derive(Clone, Copy)]
struct vm_struct {
    __slate_anon_0: __SlateRecord_vmalloc_h_53,
    addr: *mut (),
    size: u64,
    flags: u64,
    pages: *mut *mut page,
    page_order: u32,
    nr_pages: u64,
    phys_addr: u64,
    caller: *const (),
    requested_size: u64,
}


fn kmemleak_alloc(ptr: *const (), size: u64, min_count: i32, gfp: u32) {
}

fn cpuhp_setup_state_nocalls(state: i32, name: *const u8, startup: Option<unsafe extern "C" fn(u32) -> i32>, teardown: Option<unsafe extern "C" fn(u32) -> i32>) -> i32 {
unsafe { __cpuhp_setup_state(state, name, false, startup, teardown, false) }
}

fn cpuhp_setup_state_multi(state: i32, name: *const u8, startup: Option<unsafe extern "C" fn(u32, *mut hlist_node) -> i32>, teardown: Option<unsafe extern "C" fn(u32, *mut hlist_node) -> i32>) -> i32 {
unsafe { __cpuhp_setup_state(state, name, false, unsafe { core::mem::transmute::<*mut (), Option<unsafe extern "C" fn(u32) -> i32>>(unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u32, *mut hlist_node) -> i32>, *mut ()>(startup) }) }, unsafe { core::mem::transmute::<*mut (), Option<unsafe extern "C" fn(u32) -> i32>>(unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u32, *mut hlist_node) -> i32>, *mut ()>(teardown) }) }, true) }
}

fn cpuhp_state_add_instance_nocalls(state: i32, node: *mut hlist_node) -> i32 {
unsafe { __cpuhp_state_add_instance(state, node, false) }
}

fn cpuhp_state_remove_instance_nocalls(state: i32, node: *mut hlist_node) -> i32 {
unsafe { __cpuhp_state_remove_instance(state, node, false) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct taskstats {
    version: u16,
    ac_exitcode: u32,
    ac_flag: u8,
    ac_nice: u8,
    cpu_count: u64,
    cpu_delay_total: u64,
    blkio_count: u64,
    blkio_delay_total: u64,
    swapin_count: u64,
    swapin_delay_total: u64,
    cpu_run_real_total: u64,
    cpu_run_virtual_total: u64,
    ac_comm: [u8; 32],
    ac_sched: u8,
    ac_pad: [u8; 3],
    __slate_pad_15: [u8; 4],
    ac_uid: u32,
    ac_gid: u32,
    ac_pid: u32,
    ac_ppid: u32,
    ac_btime: u32,
    ac_etime: u64,
    ac_utime: u64,
    ac_stime: u64,
    ac_minflt: u64,
    ac_majflt: u64,
    coremem: u64,
    virtmem: u64,
    hiwater_rss: u64,
    hiwater_vm: u64,
    read_char: u64,
    write_char: u64,
    read_syscalls: u64,
    write_syscalls: u64,
    read_bytes: u64,
    write_bytes: u64,
    cancelled_write_bytes: u64,
    nvcsw: u64,
    nivcsw: u64,
    ac_utimescaled: u64,
    ac_stimescaled: u64,
    cpu_scaled_run_real_total: u64,
    freepages_count: u64,
    freepages_delay_total: u64,
    thrashing_count: u64,
    thrashing_delay_total: u64,
    ac_btime64: u64,
    compact_count: u64,
    compact_delay_total: u64,
    ac_tgid: u32,
    ac_tgetime: u64,
    ac_exe_dev: u64,
    ac_exe_inode: u64,
    wpcopy_count: u64,
    wpcopy_delay_total: u64,
    irq_count: u64,
    irq_delay_total: u64,
    cpu_delay_max: u64,
    cpu_delay_min: u64,
    blkio_delay_max: u64,
    blkio_delay_min: u64,
    swapin_delay_max: u64,
    swapin_delay_min: u64,
    freepages_delay_max: u64,
    freepages_delay_min: u64,
    thrashing_delay_max: u64,
    thrashing_delay_min: u64,
    compact_delay_max: u64,
    compact_delay_min: u64,
    wpcopy_delay_max: u64,
    wpcopy_delay_min: u64,
    irq_delay_max: u64,
    irq_delay_min: u64,
    cpu_delay_max_ts: __kernel_timespec,
    blkio_delay_max_ts: __kernel_timespec,
    swapin_delay_max_ts: __kernel_timespec,
    freepages_delay_max_ts: __kernel_timespec,
    thrashing_delay_max_ts: __kernel_timespec,
    compact_delay_max_ts: __kernel_timespec,
    wpcopy_delay_max_ts: __kernel_timespec,
    irq_delay_max_ts: __kernel_timespec,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seq_file {
    buf: *mut u8,
    size: u64,
    from: u64,
    count: u64,
    pad_until: u64,
    index: i64,
    read_pos: i64,
    lock: mutex,
    op: *const seq_operations,
    poll_event: i32,
    file: *const file,
    private: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seq_operations {
    start: Option<unsafe extern "C" fn(*mut seq_file, *mut i64) -> *mut ()>,
    stop: Option<unsafe extern "C" fn(*mut seq_file, *mut ())>,
    next: Option<unsafe extern "C" fn(*mut seq_file, *mut (), *mut i64) -> *mut ()>,
    show: Option<unsafe extern "C" fn(*mut seq_file, *mut ()) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ns_tree_root {
    ns_rb: rb_root,
    ns_list_head: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ns_tree_node {
    ns_node: rb_node,
    ns_list_entry: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ns_tree {
    ns_id: u64,
    __ns_ref_active: __SlateRecord_types_h_190,
    ns_unified_node: ns_tree_node,
    ns_tree_node: ns_tree_node,
    ns_owner_node: ns_tree_node,
    ns_owner_root: ns_tree_root,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ns_common {
    __slate_anon_0: __SlateRecord_ns_common_types_h_112,
    ns_type: u32,
    stashed: *mut dentry,
    ops: *const proc_ns_operations,
    inum: u32,
    __slate_anon_5: ns_tree,
    ns_rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct nsproxy {
    count: refcount_struct,
    uts_ns: *mut uts_namespace,
    ipc_ns: *mut ipc_namespace,
    mnt_ns: *mut mnt_namespace,
    pid_ns_for_children: *mut pid_namespace,
    net_ns: *mut net,
    time_ns: *mut time_namespace,
    time_ns_for_children: *mut time_namespace,
    cgroup_ns: *mut cgroup_namespace,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uid_gid_extent {
    first: u32,
    lower_first: u32,
    count: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uid_gid_map {
    __slate_anon_0: __SlateRecord_user_namespace_h_26,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct user_namespace {
    uid_map: uid_gid_map,
    gid_map: uid_gid_map,
    projid_map: uid_gid_map,
    parent: *mut user_namespace,
    level: i32,
    owner: __SlateRecord_uidgid_types_h_9,
    group: __SlateRecord_uidgid_types_h_13,
    ns: ns_common,
    flags: u64,
    parent_could_setfcap: bool,
    keyring_name_list: list_head,
    user_keyring_register: *mut key,
    keyring_sem: rw_semaphore,
    work: work_struct,
    set: ctl_table_set,
    sysctls: *mut ctl_table_header,
    ucounts: *mut ucounts,
    ucount_max: [i64; 11],
    rlimit_max: [i64; 4],
    binfmt_misc: *mut binfmt_misc,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ucounts {
    node: hlist_nulls_node,
    ns: *mut user_namespace,
    uid: __SlateRecord_uidgid_types_h_9,
    rcu: callback_head,
    count: __SlateRecord_types_h_202,
    ucount: [__SlateRecord_types_h_197; 11],
    rlimit: [__SlateRecord_types_h_197; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_stat {
    irqs_sum: u64,
    softirqs: [u32; 10],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct u64_stats_sync {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_local_h_12 {
    a: __SlateRecord_types_h_197,
}


fn local_inc(mut l: *mut __SlateRecord_local_h_12) {
unsafe {
core::arch::asm!(" incq ({0:r})", in(reg) core::ptr::addr_of_mut!((*l).a.counter), options(nostack, att_syntax));
}
}

fn local_dec(mut l: *mut __SlateRecord_local_h_12) {
unsafe {
core::arch::asm!(" decq ({0:r})", in(reg) core::ptr::addr_of_mut!((*l).a.counter), options(nostack, att_syntax));
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_bpf {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct psi_group {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_file {
    kn: *mut kernfs_node,
    notified_at: u64,
    notify_timer: timer_list,
    lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_subsys_state {
    cgroup: *mut cgroup,
    ss: *mut cgroup_subsys,
    refcnt: percpu_ref,
    rstat_cpu: *mut css_rstat_cpu,
    sibling: list_head,
    children: list_head,
    id: i32,
    flags: u32,
    serial_nr: u64,
    online_cnt: __SlateRecord_types_h_190,
    destroy_work: work_struct,
    destroy_rwork: rcu_work,
    parent: *mut cgroup_subsys_state,
    nr_descendants: i32,
    nr_populated_csets: i32,
    nr_populated_children: i32,
    kill_finish_work: work_struct,
    rstat_flush_next: *mut cgroup_subsys_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct css_set {
    subsys: [*mut cgroup_subsys_state; 14],
    refcount: refcount_struct,
    dom_cset: *mut css_set,
    dfl_cgrp: *mut cgroup,
    nr_tasks: i32,
    tasks: list_head,
    mg_tasks: list_head,
    dying_tasks: list_head,
    task_iters: list_head,
    e_cset_node: [list_head; 14],
    threaded_csets: list_head,
    threaded_csets_node: list_head,
    hlist: hlist_node,
    cgrp_links: list_head,
    mg_src_preload_node: list_head,
    mg_dst_preload_node: list_head,
    mg_node: list_head,
    mg_src_cgrp: *mut cgroup,
    mg_dst_cgrp: *mut cgroup,
    mg_dst_cset: *mut css_set,
    dead: bool,
    callback_head: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_base_stat {
    cputime: task_cputime,
    ntime: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct css_rstat_cpu {
    updated_children: *mut cgroup_subsys_state,
    updated_next: *mut cgroup_subsys_state,
    lnode: llist_node,
    owner: *mut cgroup_subsys_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_rstat_base_cpu {
    bsync: u64_stats_sync,
    bstat: cgroup_base_stat,
    last_bstat: cgroup_base_stat,
    subtree_bstat: cgroup_base_stat,
    last_subtree_bstat: cgroup_base_stat,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_freezer_state {
    freeze: bool,
    e_freeze: bool,
    nr_frozen_descendants: i32,
    nr_frozen_tasks: i32,
    freeze_seq: seqcount_spinlock,
    freeze_start_nsec: u64,
    frozen_nsec: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup {
    self_: cgroup_subsys_state,
    flags: u64,
    level: i32,
    max_depth: i32,
    nr_descendants: i32,
    nr_dying_descendants: i32,
    max_descendants: i32,
    nr_populated_domain_children: i32,
    nr_populated_threaded_children: i32,
    nr_threaded_children: i32,
    kill_seq: u32,
    kn: *mut kernfs_node,
    procs_file: cgroup_file,
    events_file: cgroup_file,
    psi_files: [cgroup_file; 0],
    subtree_control: u32,
    subtree_ss_mask: u32,
    old_subtree_control: u32,
    old_subtree_ss_mask: u32,
    subsys: [*mut cgroup_subsys_state; 14],
    nr_dying_subsys: [i32; 14],
    root: *mut cgroup_root,
    cset_links: list_head,
    e_csets: [list_head; 14],
    dom_cgrp: *mut cgroup,
    old_dom_cgrp: *mut cgroup,
    rstat_base_cpu: *mut cgroup_rstat_base_cpu,
    _pad_: cacheline_padding,
    last_bstat: cgroup_base_stat,
    bstat: cgroup_base_stat,
    prev_cputime: prev_cputime,
    pidlists: list_head,
    pidlist_mutex: mutex,
    offline_waitq: wait_queue_head,
    release_agent_work: work_struct,
    psi: *mut psi_group,
    bpf: cgroup_bpf,
    freezer: cgroup_freezer_state,
    __slate_anon_38: __SlateRecord_cgroup_defs_h_644,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_root {
    kf_root: *mut kernfs_root,
    subsys_mask: u32,
    hierarchy_id: i32,
    root_list: list_head,
    rcu: callback_head,
    nr_cgrps: __SlateRecord_types_h_190,
    flags: u32,
    release_agent_path: [u8; 4096],
    name: [u8; 64],
    cgrp: cgroup,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cftype {
    name: [u8; 64],
    private: u64,
    max_write_len: u64,
    flags: u32,
    file_offset: u32,
    ss: *mut cgroup_subsys,
    node: list_head,
    kf_ops: *mut kernfs_ops,
    open: Option<unsafe extern "C" fn(*mut kernfs_open_file) -> i32>,
    release: Option<unsafe extern "C" fn(*mut kernfs_open_file)>,
    read_u64: Option<unsafe extern "C" fn(*mut cgroup_subsys_state, *mut cftype) -> u64>,
    read_s64: Option<unsafe extern "C" fn(*mut cgroup_subsys_state, *mut cftype) -> i64>,
    seq_show: Option<unsafe extern "C" fn(*mut seq_file, *mut ()) -> i32>,
    seq_start: Option<unsafe extern "C" fn(*mut seq_file, *mut i64) -> *mut ()>,
    seq_next: Option<unsafe extern "C" fn(*mut seq_file, *mut (), *mut i64) -> *mut ()>,
    seq_stop: Option<unsafe extern "C" fn(*mut seq_file, *mut ())>,
    write_u64: Option<unsafe extern "C" fn(*mut cgroup_subsys_state, *mut cftype, u64) -> i32>,
    write_s64: Option<unsafe extern "C" fn(*mut cgroup_subsys_state, *mut cftype, i64) -> i32>,
    write: Option<unsafe extern "C" fn(*mut kernfs_open_file, *mut u8, u64, i64) -> i64>,
    poll: Option<unsafe extern "C" fn(*mut kernfs_open_file, *mut poll_table_struct) -> u32>,
    lockdep_key: lock_class_key,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_subsys {
    css_alloc: Option<unsafe extern "C" fn(*mut cgroup_subsys_state) -> *mut cgroup_subsys_state>,
    css_online: Option<unsafe extern "C" fn(*mut cgroup_subsys_state) -> i32>,
    css_offline: Option<unsafe extern "C" fn(*mut cgroup_subsys_state)>,
    css_released: Option<unsafe extern "C" fn(*mut cgroup_subsys_state)>,
    css_free: Option<unsafe extern "C" fn(*mut cgroup_subsys_state)>,
    css_reset: Option<unsafe extern "C" fn(*mut cgroup_subsys_state)>,
    css_killed: Option<unsafe extern "C" fn(*mut cgroup_subsys_state)>,
    css_rstat_flush: Option<unsafe extern "C" fn(*mut cgroup_subsys_state, i32)>,
    css_extra_stat_show: Option<unsafe extern "C" fn(*mut seq_file, *mut cgroup_subsys_state) -> i32>,
    css_local_stat_show: Option<unsafe extern "C" fn(*mut seq_file, *mut cgroup_subsys_state) -> i32>,
    can_attach: Option<unsafe extern "C" fn(*mut cgroup_taskset) -> i32>,
    cancel_attach: Option<unsafe extern "C" fn(*mut cgroup_taskset)>,
    attach: Option<unsafe extern "C" fn(*mut cgroup_taskset)>,
    can_fork: Option<unsafe extern "C" fn(*mut task_struct, *mut css_set) -> i32>,
    cancel_fork: Option<unsafe extern "C" fn(*mut task_struct, *mut css_set)>,
    fork: Option<unsafe extern "C" fn(*mut task_struct)>,
    exit: Option<unsafe extern "C" fn(*mut task_struct)>,
    release: Option<unsafe extern "C" fn(*mut task_struct)>,
    bind: Option<unsafe extern "C" fn(*mut cgroup_subsys_state)>,
    __slate_bits_0: __slate_bits::__SlateBits895U0,
    id: i32,
    name: *const u8,
    legacy_name: *const u8,
    root: *mut cgroup_root,
    css_idr: idr,
    cfts: list_head,
    dfl_cftypes: *mut cftype,
    legacy_cftypes: *mut cftype,
    depends_on: u32,
    rstat_ss_lock: spinlock,
    lhead: *mut llist_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_namespace {
    ns: ns_common,
    user_ns: *mut user_namespace,
    ucounts: *mut ucounts,
    root_cset: *mut css_set,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wait_page_queue {
    folio: *mut folio,
    bit_nr: i32,
    wait: wait_queue_entry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct readahead_control {
    file: *mut file,
    mapping: *mut address_space,
    ra: *mut file_ra_state,
    _index: u64,
    _nr_pages: u32,
    _batch_count: u32,
    dropbehind: bool,
    _workingset: bool,
    _pflags: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct reclaim_state {
    reclaimed: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_sequential_cluster {
    next: [u32; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_info_struct {
    users: percpu_ref,
    flags: u64,
    prio: i16,
    list: plist_node,
    r#type: i8,
    max: u32,
    cluster_info: *mut swap_cluster_info,
    free_clusters: list_head,
    full_clusters: list_head,
    nonfull_clusters: [list_head; 1],
    frag_clusters: [list_head; 1],
    pages: u32,
    inuse_pages: __SlateRecord_types_h_197,
    global_cluster: *mut swap_sequential_cluster,
    global_cluster_lock: spinlock,
    swap_extent_root: rb_root,
    bdev: *mut block_device,
    swap_file: *mut file,
    comp: completion,
    lock: spinlock,
    discard_work: work_struct,
    reclaim_work: work_struct,
    discard_clusters: list_head,
    avail_list: plist_node,
    ops: *const swap_ops,
}


// MFD_EXEC implied if unset
#[repr(C)]
#[derive(Clone, Copy)]
struct pid_namespace {
    idr: idr,
    rcu: callback_head,
    pid_allocated: u32,
    memfd_noexec_scope: i32,
    set: ctl_table_set,
    sysctls: *mut ctl_table_header,
    child_reaper: *mut task_struct,
    pid_cachep: *mut kmem_cache,
    level: u32,
    pid_max: i32,
    parent: *mut pid_namespace,
    bacct: *mut fs_pin,
    user_ns: *mut user_namespace,
    ucounts: *mut ucounts,
    reboot: i32,
    ns: ns_common,
    work: work_struct,
}


fn is_kdump_kernel() -> bool {
(unsafe { elfcorehdr_addr }) != 1u64.wrapping_neg()
}

fn bio_has_crypt_ctx(bio: *mut bio) -> bool {
false
}

#[repr(C)]
#[derive(Clone, Copy)]
struct disk_stats {
    nsecs: [u64; 4],
    sectors: [u64; 4],
    ios: [u64; 4],
    merges: [u64; 4],
    io_ticks: u64,
    in_flight: [__SlateRecord_local_h_12; 2],
}


// Macros to operate on percpu disk statistics:
//
// part_stat_{add|sub|inc|dec}() modify the stat counters and should
// be called between part_stat_lock() and part_stat_unlock().
//
// part_stat_read() can be called at any time.
#[repr(C)]
#[derive(Clone, Copy)]
struct property {
    name: *mut u8,
    length: i32,
    value: *mut (),
    next: *mut property,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_node {
    name: *const u8,
    phandle: u32,
    full_name: *const u8,
    fwnode: fwnode_handle,
    properties: *mut property,
    deadprops: *mut property,
    parent: *mut device_node,
    child: *mut device_node,
    sibling: *mut device_node,
    _flags: u64,
    data: *mut (),
}


fn cpu_is_isolated(cpu: i32) -> bool {
!unsafe { housekeeping_test_cpu(cpu, 1u32) }
}

// The rest seem to be more-or-less nonstandard. Check them!
#[repr(C)]
#[derive(Clone, Copy)]
struct pollfd {
    fd: i32,
    events: i16,
    revents: i16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct poll_table_struct {
    _qproc: Option<unsafe extern "C" fn(*mut file, *mut wait_queue_head, *mut poll_table_struct)>,
    _key: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rchan_buf_stats {
    full_count: u32,
    big_count: u32,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct rchan_buf {
    start: *mut (),
    data: *mut (),
    offset: u64,
    subbufs_produced: u64,
    subbufs_consumed: u64,
    chan: *mut rchan,
    read_wait: wait_queue_head,
    wakeup_work: irq_work,
    dentry: *mut dentry,
    kref: kref,
    stats: rchan_buf_stats,
    page_array: *mut *mut page,
    page_count: u32,
    finalized: u32,
    padding: *mut u64,
    bytes_consumed: u64,
    early_bytes: u64,
    cpu: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rchan {
    version: u32,
    subbuf_size: u64,
    n_subbufs: u64,
    alloc_size: u64,
    cb: *const rchan_callbacks,
    kref: kref,
    private_data: *mut (),
    buf: *mut *mut rchan_buf,
    is_global: i32,
    list: list_head,
    parent: *mut dentry,
    has_base_filename: i32,
    base_filename: [u8; 255],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rchan_callbacks {
    subbuf_start: Option<unsafe extern "C" fn(*mut rchan_buf, *mut (), *mut ()) -> i32>,
    create_buf_file: Option<unsafe extern "C" fn(*const u8, *mut dentry, u16, *mut rchan_buf, *mut i32) -> *mut dentry>,
    remove_buf_file: Option<unsafe extern "C" fn(*mut dentry) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_trace {
    version: i32,
    trace_state: i32,
    rchan: *mut rchan,
    sequence: *mut u64,
    msg_data: *mut u8,
    act_mask: u64,
    start_lba: u64,
    end_lba: u64,
    pid: u32,
    dev: u32,
    dir: *mut dentry,
    running_list: list_head,
    dropped: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_mod {
    next: *mut static_call_mod,
    r#mod: *mut module,
    sites: *mut static_call_site,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_eval_map {
    system: *const u8,
    eval_string: *const u8,
    eval_value: u64,
}


// When a tracepoint is used, it's name is added to the __tracepoint_check
// section. This section is only used at build time to make sure all
// defined tracepoints are used. It is discarded after the build.
// Make sure the alignment of the structure in the __tracepoints section will
// not add unwanted padding between the beginning of the section and the
// structure. Force alignment to the same alignment as the section start.
//
// When lockdep is enabled, we make sure to always test if RCU is
// "watching" regardless if the tracepoint is enabled or not. Tracepoints
// require RCU to be active, and it should always warn at the tracepoint
// site if it is not watching, as it will need to be active when the
// tracepoint is enabled.
fn __do_trace_block_rq_requeue(rq: *mut request) {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v2 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v2;
}
2 => {
let mut __v4: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v4 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v4;
}
4 => {
let mut __v5: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
8 => {
let mut __v6: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v6 = pfo_val__ as i32;
pscr_ret__ = __v6;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v0 = pscr_ret__;
if cpu_online(__v0 as u32) {
let mut __UNIQUE_ID_guard_707: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_707) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_708: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_708) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_710() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_block_rq_requeue)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_709: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_709;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut request) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut request) -> i32>>(__SCT__tp_func_block_rq_requeue as *const ()) };
unsafe { __v12.unwrap()(__data, rq) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_block_rq_requeue: __SlateAlign16<[u8; 17]> = __SlateAlign16([98u8, 108u8, 111u8, 99u8, 107u8, 95u8, 114u8, 113u8, 95u8, 114u8, 101u8, 113u8, 117u8, 101u8, 117u8, 101u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_block_rq_requeue_711: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_block_rq_requeue) }) as *mut ();

fn trace_block_rq_requeue(rq: *mut request) {
if static_branch_unlikely!(unsafe { core::ptr::addr_of_mut!(__tracepoint_block_rq_requeue.key) }) != 0 {
__do_trace_block_rq_requeue(rq);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_712: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_block_rq_complete(rq: *mut request, error: u8, nr_bytes: u32) {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v2 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v2;
}
2 => {
let mut __v4: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v4 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v4;
}
4 => {
let mut __v5: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
8 => {
let mut __v6: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v6 = pfo_val__ as i32;
pscr_ret__ = __v6;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v0 = pscr_ret__;
if cpu_online(__v0 as u32) {
let mut __UNIQUE_ID_guard_714: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_714) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_715: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_715) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_717() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_block_rq_complete)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_716: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_716;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut request, u8, u32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut request, u8, u32) -> i32>>(__SCT__tp_func_block_rq_complete as *const ()) };
unsafe { __v12.unwrap()(__data, rq, error, nr_bytes) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_block_rq_complete: __SlateAlign16<[u8; 18]> = __SlateAlign16([98u8, 108u8, 111u8, 99u8, 107u8, 95u8, 114u8, 113u8, 95u8, 99u8, 111u8, 109u8, 112u8, 108u8, 101u8, 116u8, 101u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_block_rq_complete_718: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_block_rq_complete) }) as *mut ();

fn trace_block_rq_complete(rq: *mut request, error: u8, nr_bytes: u32) {
if static_branch_unlikely__2!(unsafe { core::ptr::addr_of_mut!(__tracepoint_block_rq_complete.key) }) != 0 {
__do_trace_block_rq_complete(rq, error, nr_bytes);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_719: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_block_rq_error(rq: *mut request, error: u8, nr_bytes: u32) {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v2 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v2;
}
2 => {
let mut __v4: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v4 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v4;
}
4 => {
let mut __v5: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
8 => {
let mut __v6: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v6 = pfo_val__ as i32;
pscr_ret__ = __v6;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v0 = pscr_ret__;
if cpu_online(__v0 as u32) {
let mut __UNIQUE_ID_guard_721: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_721) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_722: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_722) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_724() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_block_rq_error)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_723: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_723;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut request, u8, u32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut request, u8, u32) -> i32>>(__SCT__tp_func_block_rq_error as *const ()) };
unsafe { __v12.unwrap()(__data, rq, error, nr_bytes) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_block_rq_error: [u8; 15] = [98, 108, 111, 99, 107, 95, 114, 113, 95, 101, 114, 114, 111, 114, 0];

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_block_rq_error_725: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_block_rq_error) }) as *mut ();

fn trace_block_rq_error(rq: *mut request, error: u8, nr_bytes: u32) {
if static_branch_unlikely__3!(unsafe { core::ptr::addr_of_mut!(__tracepoint_block_rq_error.key) }) != 0 {
__do_trace_block_rq_error(rq, error, nr_bytes);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_726: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_block_rq_insert(rq: *mut request) {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v2 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v2;
}
2 => {
let mut __v4: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v4 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v4;
}
4 => {
let mut __v5: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
8 => {
let mut __v6: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v6 = pfo_val__ as i32;
pscr_ret__ = __v6;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v0 = pscr_ret__;
if cpu_online(__v0 as u32) {
let mut __UNIQUE_ID_guard_735: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_735) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_736: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_736) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_738() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_block_rq_insert)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_737: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_737;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut request) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut request) -> i32>>(__SCT__tp_func_block_rq_insert as *const ()) };
unsafe { __v12.unwrap()(__data, rq) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_block_rq_insert: __SlateAlign16<[u8; 16]> = __SlateAlign16([98u8, 108u8, 111u8, 99u8, 107u8, 95u8, 114u8, 113u8, 95u8, 105u8, 110u8, 115u8, 101u8, 114u8, 116u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_block_rq_insert_739: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_block_rq_insert) }) as *mut ();

fn trace_block_rq_insert(rq: *mut request) {
if static_branch_unlikely__4!(unsafe { core::ptr::addr_of_mut!(__tracepoint_block_rq_insert.key) }) != 0 {
__do_trace_block_rq_insert(rq);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_740: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_block_rq_issue(rq: *mut request) {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v2 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v2;
}
2 => {
let mut __v4: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v4 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v4;
}
4 => {
let mut __v5: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
8 => {
let mut __v6: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v6 = pfo_val__ as i32;
pscr_ret__ = __v6;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v0 = pscr_ret__;
if cpu_online(__v0 as u32) {
let mut __UNIQUE_ID_guard_742: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_742) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_743: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_743) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_745() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_block_rq_issue)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_744: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_744;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut request) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut request) -> i32>>(__SCT__tp_func_block_rq_issue as *const ()) };
unsafe { __v12.unwrap()(__data, rq) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_block_rq_issue: [u8; 15] = [98, 108, 111, 99, 107, 95, 114, 113, 95, 105, 115, 115, 117, 101, 0];

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_block_rq_issue_746: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_block_rq_issue) }) as *mut ();

fn trace_block_rq_issue(rq: *mut request) {
if static_branch_unlikely__9!(unsafe { core::ptr::addr_of_mut!(__tracepoint_block_rq_issue.key) }) != 0 {
__do_trace_block_rq_issue(rq);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_747: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_block_io_start(rq: *mut request) {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v2 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v2;
}
2 => {
let mut __v4: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v4 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v4;
}
4 => {
let mut __v5: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
8 => {
let mut __v6: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v6 = pfo_val__ as i32;
pscr_ret__ = __v6;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v0 = pscr_ret__;
if cpu_online(__v0 as u32) {
let mut __UNIQUE_ID_guard_756: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_756) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_757: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_757) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_759() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_block_io_start)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_758: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_758;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut request) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut request) -> i32>>(__SCT__tp_func_block_io_start as *const ()) };
unsafe { __v12.unwrap()(__data, rq) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_block_io_start: [u8; 15] = [98, 108, 111, 99, 107, 95, 105, 111, 95, 115, 116, 97, 114, 116, 0];

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_block_io_start_760: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_block_io_start) }) as *mut ();

fn trace_block_io_start(rq: *mut request) {
if static_branch_unlikely__8!(unsafe { core::ptr::addr_of_mut!(__tracepoint_block_io_start.key) }) != 0 {
__do_trace_block_io_start(rq);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_761: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_block_io_done(rq: *mut request) {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v2 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v2;
}
2 => {
let mut __v4: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v4 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v4;
}
4 => {
let mut __v5: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
8 => {
let mut __v6: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v6 = pfo_val__ as i32;
pscr_ret__ = __v6;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v0 = pscr_ret__;
if cpu_online(__v0 as u32) {
let mut __UNIQUE_ID_guard_763: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_763) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_764: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_764) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_766() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_block_io_done)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_765: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_765;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut request) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut request) -> i32>>(__SCT__tp_func_block_io_done as *const ()) };
unsafe { __v12.unwrap()(__data, rq) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_block_io_done: [u8; 14] = [98, 108, 111, 99, 107, 95, 105, 111, 95, 100, 111, 110, 101, 0];

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_block_io_done_767: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_block_io_done) }) as *mut ();

fn trace_block_io_done(rq: *mut request) {
if static_branch_unlikely__10!(unsafe { core::ptr::addr_of_mut!(__tracepoint_block_io_done.key) }) != 0 {
__do_trace_block_io_done(rq);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_768: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_block_getrq(bio: *mut bio) {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v2 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v2;
}
2 => {
let mut __v4: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v4 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v4;
}
4 => {
let mut __v5: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
8 => {
let mut __v6: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v6 = pfo_val__ as i32;
pscr_ret__ = __v6;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v0 = pscr_ret__;
if cpu_online(__v0 as u32) {
let mut __UNIQUE_ID_guard_798: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_798) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_799: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_799) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_801() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_block_getrq)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_800: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_800;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut bio) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut bio) -> i32>>(__SCT__tp_func_block_getrq as *const ()) };
unsafe { __v12.unwrap()(__data, bio) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_block_getrq: [u8; 12] = [98, 108, 111, 99, 107, 95, 103, 101, 116, 114, 113, 0];

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_block_getrq_802: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_block_getrq) }) as *mut ();

fn trace_block_getrq(bio: *mut bio) {
if static_branch_unlikely__7!(unsafe { core::ptr::addr_of_mut!(__tracepoint_block_getrq.key) }) != 0 {
__do_trace_block_getrq(bio);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_803: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_block_plug(q: *mut request_queue) {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v2 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v2;
}
2 => {
let mut __v4: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v4 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v4;
}
4 => {
let mut __v5: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
8 => {
let mut __v6: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v6 = pfo_val__ as i32;
pscr_ret__ = __v6;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v0 = pscr_ret__;
if cpu_online(__v0 as u32) {
let mut __UNIQUE_ID_guard_812: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_812) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_813: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_813) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_815() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_block_plug)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_814: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_814;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut request_queue) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut request_queue) -> i32>>(__SCT__tp_func_block_plug as *const ()) };
unsafe { __v12.unwrap()(__data, q) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_block_plug: [u8; 11] = [98, 108, 111, 99, 107, 95, 112, 108, 117, 103, 0];

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_block_plug_816: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_block_plug) }) as *mut ();

fn trace_block_plug(q: *mut request_queue) {
if static_branch_unlikely__5!(unsafe { core::ptr::addr_of_mut!(__tracepoint_block_plug.key) }) != 0 {
__do_trace_block_plug(q);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_817: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_block_unplug(q: *mut request_queue, depth: u32, explicit: bool) {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v2 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v2;
}
2 => {
let mut __v4: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v4 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v4;
}
4 => {
let mut __v5: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
8 => {
let mut __v6: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v6 = pfo_val__ as i32;
pscr_ret__ = __v6;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v0 = pscr_ret__;
if cpu_online(__v0 as u32) {
let mut __UNIQUE_ID_guard_819: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_819) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_820: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_820) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_822() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_block_unplug)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_821: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_821;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut request_queue, u32, bool) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut request_queue, u32, bool) -> i32>>(__SCT__tp_func_block_unplug as *const ()) };
unsafe { __v12.unwrap()(__data, q, depth, explicit) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_block_unplug: [u8; 13] = [98, 108, 111, 99, 107, 95, 117, 110, 112, 108, 117, 103, 0];

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_block_unplug_823: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_block_unplug) }) as *mut ();

fn trace_block_unplug(q: *mut request_queue, depth: u32, explicit: bool) {
if static_branch_unlikely__6!(unsafe { core::ptr::addr_of_mut!(__tracepoint_block_unplug.key) }) != 0 {
__do_trace_block_unplug(q, depth, explicit);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_824: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn blk_crypto_rq_set_defaults(rq: *mut request) {
}

fn blk_crypto_rq_is_encrypted(rq: *mut request) -> bool {
false
}

fn blk_crypto_rq_has_keyslot(rq: *mut request) -> bool {
false
}

fn blk_crypto_rq_get_keyslot(rq: *mut request) -> u8 {
if blk_crypto_rq_is_encrypted(rq) {
return unsafe { __blk_crypto_rq_get_keyslot(rq) };
}
0
}

fn blk_crypto_rq_put_keyslot(rq: *mut request) {
if blk_crypto_rq_has_keyslot(rq) {
unsafe { __blk_crypto_rq_put_keyslot(rq) };
}
}

fn blk_crypto_free_request(rq: *mut request) {
if blk_crypto_rq_is_encrypted(rq) {
unsafe { __blk_crypto_free_request(rq) };
}
}

fn blk_crypto_rq_bio_prep(rq: *mut request, bio: *mut bio, gfp_mask: u32) -> i32 {
if bio_has_crypt_ctx(bio) {
return unsafe { __blk_crypto_rq_bio_prep(rq, bio, gfp_mask) };
}
0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct blk_flush_queue {
    mq_flush_lock: spinlock,
    __slate_bits_0: __slate_bits::__SlateBits816U0,
    rq_status: u8,
    flush_pending_since: u64,
    flush_queue: [list_head; 2],
    flush_data_in_flight: u64,
    flush_rq: *mut request,
    callback_head: callback_head,
}


fn blk_try_enter_queue(q: *mut request_queue, pm: bool) -> bool {
let mut __v0: bool = unsafe { core::mem::zeroed() };
// The code that increments the pm_only counter must ensure that the
// counter is globally visible before the queue is unfrozen.
let mut __v1: bool = unsafe { core::mem::zeroed() };
unsafe {
rcu_read_lock();
if percpu_ref_tryget_live_rcu(core::ptr::addr_of_mut!((*q).q_usage_counter)) {
if blk_queue_pm_only!(q) != 0 {
if !pm {
__v0 = true;
} else {
__v0 = (queue_rpm_status(q) as i32) == 2;
}
__v1 = __v0;
} else {
__v1 = false;
}
if __v1 {
blk_queue_exit(q);
} else {
rcu_read_unlock();
return true;
}
}
rcu_read_unlock();
return false;
}
}

fn bio_queue_enter(bio: *mut bio) -> i32 {
let q: *mut request_queue = bdev_get_queue(unsafe { (*bio).bi_bdev });
if blk_try_enter_queue(q, false) {
rwsem_acquire_read!();
rwsem_release!();
return 0;
}
unsafe { __bio_queue_enter(q, bio) }
}

fn blk_wait_io(done: *mut completion) {
// Prevent hang_check timer from firing at us during very long I/O
let timeout: u64 = 0u64;
if timeout != 0 {
while (unsafe { wait_for_completion_io_timeout(done, timeout) }) == 0 {
}
} else {
unsafe { wait_for_completion_io(done) };
}
}

fn blk_rq_get_max_segments(rq: *mut request) -> u32 {
if (req_op(rq as *const request) as u32) == 3 {
return queue_max_discard_segments((unsafe { (*rq).q }) as *const request_queue) as u32;
}
queue_max_segments((unsafe { (*rq).q }) as *const request_queue) as u32
}

fn blk_queue_get_max_sectors(rq: *mut request) -> u32 {
let q: *mut request_queue = unsafe { (*rq).q };
let op: u32 = req_op(rq as *const request);
if unlikely__2!((op as u32) == 3) != 0 {
let mut __v0: u32;
let __UNIQUE_ID_x__868: u32 = unsafe { (*q).limits.max_discard_sectors };
let __UNIQUE_ID_y__869: u32 = 8388607u32;
loop {
if __types_ok!(__UNIQUE_ID_x__868, __UNIQUE_ID_y__869) == 0 {
unsafe { __compiletime_assert_870() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__868 < __UNIQUE_ID_y__869 { __UNIQUE_ID_x__868 } else { __UNIQUE_ID_y__869 };
return __v0;
}
if unlikely__2!((op as u32) == 5) != 0 {
let mut __v2: u32;
let __UNIQUE_ID_x__871: u32 = unsafe { (*q).limits.max_secure_erase_sectors };
let __UNIQUE_ID_y__872: u32 = 8388607u32;
loop {
if __types_ok!(__UNIQUE_ID_x__871, __UNIQUE_ID_y__872) == 0 {
unsafe { __compiletime_assert_873() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = if __UNIQUE_ID_x__871 < __UNIQUE_ID_y__872 { __UNIQUE_ID_x__871 } else { __UNIQUE_ID_y__872 };
return __v2;
}
if unlikely__2!((op as u32) == 9) != 0 {
return unsafe { (*q).limits.max_write_zeroes_sectors };
}
if (unsafe { (*rq).cmd_flags }) & 134217728u32 != 0 {
return unsafe { (*q).limits.atomic_write_max_sectors };
}
unsafe { (*q).limits.max_sectors }
}

// Plug flush limits
fn bio_may_need_split(bio: *mut bio, lim: *const queue_limits) -> bool {
if (unsafe { (*lim).chunk_sectors }) != 0 {
return true;
}
if unsafe { (*bio).bi_io_vec }.is_null() {
return true;
}
let bv: *const bio_vec = __bvec_iter_bvec((unsafe { (*bio).bi_io_vec }) as *const bio_vec, unsafe { (*bio).bi_iter });
if (unsafe { (*bio).bi_iter.bi_size }) > unsafe { (*bv).bv_len }.wrapping_sub(unsafe { (*bio).bi_iter.bi_offset }) {
return true;
}
if ((unsafe { (*bv).bv_offset }) | unsafe { (*bv).bv_len }) & unsafe { (*lim).dma_alignment } != 0 {
return true;
}
unsafe { (*bv).bv_len }.wrapping_add(unsafe { (*bv).bv_offset }) > unsafe { (*lim).max_fast_segment_size }
}

fn __bio_split_to_limits(bio: *mut bio, lim: *const queue_limits, nr_segs: *mut u32) -> *mut bio {
match bio_op(bio as *const bio) as u32 {
0 | 1 => {
if bio_may_need_split(bio, lim) {
return unsafe { bio_split_rw(bio, lim, nr_segs) };
}
unsafe {
*nr_segs = 1;
}
return bio;
}
7 => {
return unsafe { bio_split_zone_append(bio, lim, nr_segs) };
}
3 | 5 => {
return unsafe { bio_split_discard(bio, lim, nr_segs) };
}
9 => {
return unsafe { bio_split_write_zeroes(bio, lim, nr_segs) };
}
_ => {
unsafe {
*nr_segs = 0;
}
// other operations can't be split
return bio;
}
}
unsafe { core::mem::zeroed() }
}

fn bdev_inc_in_flight(bdev: *mut block_device, op: u32) {
let rw: bool = op_is_write(op as u32);
let mut __v0: *mut disk_stats;
__verify_pcpu_ptr__7!();
let mut __v2: i32;
__this_cpu_preempt_check(c"read".as_ptr() as *const u8);
let mut __v3: i32;
{
let mut pscr_ret__: i32 = 0;
__verify_pcpu_ptr__3!();
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v5: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
2 => {
let mut __v7: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v7 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v7;
}
4 => {
let mut __v8: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v8 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v8;
}
8 => {
let mut __v9: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v9 = pfo_val__ as i32;
pscr_ret__ = __v9;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v3 = pscr_ret__;
}
__v2 = __v3;
__v0 = ((((unsafe { (*bdev).bd_stats }) as u64) as *mut disk_stats) as u64).wrapping_add(unsafe { __per_cpu_offset[__v2 as usize] }) as *mut disk_stats;
local_inc(unsafe { (*__v0).in_flight.as_mut_ptr() as *mut __SlateRecord_local_h_12 }.wrapping_offset((rw as i32) as isize));
if bdev_is_partition(bdev) {
let mut __v10: *mut disk_stats;
__verify_pcpu_ptr__7!();
let mut __v12: i32;
__this_cpu_preempt_check(c"read".as_ptr() as *const u8);
let mut __v13: i32;
let mut pscr_ret__: i32 = 0;
__verify_pcpu_ptr__3!();
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v15: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t4: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t4, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t4 as u8) };
}
__v15 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v15;
}
2 => {
let mut __v17: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t5: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t5, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t5 as u16) };
}
__v17 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v17;
}
4 => {
let mut __v18: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t6: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t6, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t6 as u32) };
}
__v18 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v18;
}
8 => {
let mut __v19: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t7: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t7, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t7 as u64) };
}
__v19 = pfo_val__ as i32;
pscr_ret__ = __v19;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v13 = pscr_ret__;
__v12 = __v13;
__v10 = ((((unsafe { (*(*(*bdev).bd_disk).part0).bd_stats }) as u64) as *mut disk_stats) as u64).wrapping_add(unsafe { __per_cpu_offset[__v12 as usize] }) as *mut disk_stats;
local_inc(unsafe { (*__v10).in_flight.as_mut_ptr() as *mut __SlateRecord_local_h_12 }.wrapping_offset((rw as i32) as isize));
}
}

fn bdev_dec_in_flight(bdev: *mut block_device, op: u32) {
let rw: bool = op_is_write(op as u32);
let mut __v0: *mut disk_stats;
__verify_pcpu_ptr__7!();
let mut __v2: i32;
__this_cpu_preempt_check(c"read".as_ptr() as *const u8);
let mut __v3: i32;
{
let mut pscr_ret__: i32 = 0;
__verify_pcpu_ptr__3!();
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v5: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
2 => {
let mut __v7: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v7 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v7;
}
4 => {
let mut __v8: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v8 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v8;
}
8 => {
let mut __v9: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v9 = pfo_val__ as i32;
pscr_ret__ = __v9;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v3 = pscr_ret__;
}
__v2 = __v3;
__v0 = ((((unsafe { (*bdev).bd_stats }) as u64) as *mut disk_stats) as u64).wrapping_add(unsafe { __per_cpu_offset[__v2 as usize] }) as *mut disk_stats;
local_dec(unsafe { (*__v0).in_flight.as_mut_ptr() as *mut __SlateRecord_local_h_12 }.wrapping_offset((rw as i32) as isize));
if bdev_is_partition(bdev) {
let mut __v10: *mut disk_stats;
__verify_pcpu_ptr__7!();
let mut __v12: i32;
__this_cpu_preempt_check(c"read".as_ptr() as *const u8);
let mut __v13: i32;
let mut pscr_ret__: i32 = 0;
__verify_pcpu_ptr__3!();
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v15: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t4: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t4, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t4 as u8) };
}
__v15 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v15;
}
2 => {
let mut __v17: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t5: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t5, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t5 as u16) };
}
__v17 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v17;
}
4 => {
let mut __v18: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t6: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t6, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t6 as u32) };
}
__v18 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v18;
}
8 => {
let mut __v19: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t7: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t7, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t7 as u64) };
}
__v19 = pfo_val__ as i32;
pscr_ret__ = __v19;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v13 = pscr_ret__;
__v12 = __v13;
__v10 = ((((unsafe { (*(*(*bdev).bd_disk).part0).bd_stats }) as u64) as *mut disk_stats) as u64).wrapping_add(unsafe { __per_cpu_offset[__v12 as usize] }) as *mut disk_stats;
local_dec(unsafe { (*__v10).in_flight.as_mut_ptr() as *mut __SlateRecord_local_h_12 }.wrapping_offset((rw as i32) as isize));
}
}

fn bio_zone_write_plugging(bio: *mut bio) -> bool {
false
}

fn blk_req_bio_is_zone_append(req: *mut request, bio: *mut bio) -> bool {
false
}

fn blk_zone_write_plug_init_request(rq: *mut request) {
}

fn blk_zone_append_update_request_bio(rq: *mut request, bio: *mut bio) {
}

fn blk_zone_finish_request(rq: *mut request) {
}

fn should_fail_request(part: *mut block_device, bytes: u32) -> bool {
false
}

fn req_ref_put_and_test(req: *mut request) -> bool {
let mut __v0: i64;
let __ret_warn_on: i32 = !(!req_ref_zero_or_close_to_overflow!(req)) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("879: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 879b - ., 3; .popsection\n# {0}", const 879 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_req_ref_put_and_test_1, sym __slate_asm_string_req_ref_put_and_test_2, const 703 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("880: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 880b - ., 4; .popsection\n# {0}", const 880 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
atomic_dec_and_test(unsafe { core::ptr::addr_of_mut!((*req).r#ref) })
}

fn req_ref_set(req: *mut request, value: i32) {
atomic_set(unsafe { core::ptr::addr_of_mut!((*req).r#ref) }, value);
}

fn req_ref_read(req: *mut request) -> i32 {
atomic_read((unsafe { core::ptr::addr_of_mut!((*req).r#ref) }) as *const __SlateRecord_types_h_190)
}

fn blk_time_get_ns() -> u64 {
let plug: *mut blk_plug = unsafe { (*get_current()).plug };
let mut now: u64 = 0;
let __v0: bool = plug.is_null() || !in_task!();
if __v0 {
return ktime_get_ns();
}
// 0 could very well be a valid time, but rather than flag "this is
// a valid timestamp" separately, just accept that we'll do an extra
// ktime_get_ns() if we just happen to get 0 as the current time.
//
// cur_ktime can be zeroed by pre-emption the moment PF_BLOCK_TS is set.
let mut __v1: u64;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_881() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*plug).cur_ktime) as *const u64) };
now = __v1;
if now == 0 {
now = ktime_get_ns();
loop {
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_882() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__3!((*plug).cur_ktime, now);
if true {
break;
}
}
// Ensure PF_BLOCK_TS is set after cur_ktime.
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
let __v6: *mut task_struct = get_current();
unsafe {
(*__v6).flags = (*__v6).flags | 536870912;
}
}
now
}

fn blk_freeze_acquire_lock(q: *mut request_queue) {
}

fn blk_unfreeze_release_lock(q: *mut request_queue) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_ctxs {
    kobj: kobject,
    queue_ctx: *mut blk_mq_ctx,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_ctx {
    __slate_anon_0: __SlateRecord_blk_mq_h_20,
    cpu: u32,
    index_hw: [u16; 3],
    hctxs: [*mut blk_mq_hw_ctx; 3],
    queue: *mut request_queue,
    ctxs: *mut blk_mq_ctxs,
    kobj: kobject,
}


#[unsafe(no_mangle)]
extern "C" fn blk_mq_update_nr_requests(q: *mut request_queue, et: *mut elevator_tags, nr: u32) -> *mut elevator_tags {
let set: *mut blk_mq_tag_set = unsafe { (*q).tag_set };
let mut old_et: *mut elevator_tags = core::ptr::null_mut::<elevator_tags>();
let mut hctx: *mut blk_mq_hw_ctx = unsafe { core::mem::zeroed() };
let mut i: u64 = 0;
blk_mq_quiesce_queue(q);
if blk_mq_is_shared_tags(unsafe { (*set).flags }) {
// Shared tags, for sched tags, we allocate max initially hence
// tags can't grow, see blk_mq_alloc_sched_tags().
if !unsafe { (*q).elevator }.is_null() {
unsafe { blk_mq_tag_update_sched_shared_tags(q, nr) };
} else {
unsafe { blk_mq_tag_resize_shared_tags(set, nr) };
}
} else {
if unsafe { (*q).elevator }.is_null() {
// Non-shared hardware tags, nr is already checked from
// queue_requests_store() and tags can't grow.
i = 0;
loop {
let __v1: bool = if i < ((unsafe { (*q).nr_hw_queues }) as u64) {
let mut __v2: i32;
hctx = queue_hctx(q, i as i32);
__v2 = 1;
__v2 != 0
} else {
false
};
if !__v1 {
break;
}
if !unsafe { (*hctx).tags }.is_null() {
unsafe { sbitmap_queue_resize(core::ptr::addr_of_mut!((*(*hctx).tags).bitmap_tags), nr.wrapping_sub((*(*hctx).tags).nr_reserved_tags)) };
}
i = i.wrapping_add(1);
}
} else {
if nr <= unsafe { (*(*(*q).elevator).et).nr_requests } {
// Non-shared sched tags, and tags don't grow.
i = 0;
loop {
let __v6: bool = if i < ((unsafe { (*q).nr_hw_queues }) as u64) {
let mut __v7: i32;
hctx = queue_hctx(q, i as i32);
__v7 = 1;
__v7 != 0
} else {
false
};
if !__v6 {
break;
}
if !unsafe { (*hctx).sched_tags }.is_null() {
unsafe { sbitmap_queue_resize(core::ptr::addr_of_mut!((*(*hctx).sched_tags).bitmap_tags), nr.wrapping_sub((*(*hctx).sched_tags).nr_reserved_tags)) };
}
i = i.wrapping_add(1);
}
} else {
// Non-shared sched tags, and tags grow
i = 0;
loop {
let __v10: bool = if i < ((unsafe { (*q).nr_hw_queues }) as u64) {
let mut __v11: i32;
hctx = queue_hctx(q, i as i32);
__v11 = 1;
__v11 != 0
} else {
false
};
if !__v10 {
break;
}
unsafe {
(*hctx).sched_tags = *(core::ptr::addr_of_mut!((*et).tags) as *mut *mut blk_mq_tags).wrapping_offset(i as isize);
}
i = i.wrapping_add(1);
}
old_et = unsafe { (*(*q).elevator).et };
unsafe {
(*(*q).elevator).et = et;
}
}
}
}
// Preserve relative value, both nr and async_depth are at most 16 bit
// value, no need to worry about overflow.
let mut __v15: u32;
let __UNIQUE_ID_x__1022: u32 = unsafe { (*q).async_depth }.wrapping_mul(nr) / unsafe { (*q).nr_requests };
let __UNIQUE_ID_y__1023: i32 = 1;
loop {
if __types_ok__6!(__UNIQUE_ID_x__1022, __UNIQUE_ID_y__1023) == 0 {
unsafe { __compiletime_assert_1024() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v15 = if __UNIQUE_ID_x__1022 > (__UNIQUE_ID_y__1023 as u32) { __UNIQUE_ID_x__1022 } else { __UNIQUE_ID_y__1023 as u32 };
unsafe {
(*q).async_depth = __v15;
}
unsafe {
(*q).nr_requests = nr;
}
if !unsafe { (*q).elevator }.is_null() && (unsafe { (*(*(*q).elevator).r#type).ops.depth_updated }) != None {
unsafe { unsafe { (*(*(*q).elevator).r#type).ops.depth_updated }.unwrap()(q) };
}
blk_mq_unquiesce_queue(q);
old_et
}

fn blk_mq_map_queue_type(q: *mut request_queue, r#type: u32, cpu: u32) -> *mut blk_mq_hw_ctx {
queue_hctx(q, (unsafe { *unsafe { (*((*(*q).tag_set).map.as_mut_ptr() as *mut blk_mq_queue_map).wrapping_offset((r#type as u32) as isize)).mq_map }.wrapping_offset(cpu as isize) }) as i32)
}

fn blk_mq_get_hctx_type(opf: u32) -> u32 {
let mut r#type: u32 = 0;
// The caller ensure that if REQ_POLLED, poll must be enabled.
if opf & 4194304u32 != 0 {
r#type = 2;
} else {
if opf & 255u32 == 0 {
r#type = 1;
}
}
r#type
}

fn blk_mq_map_queue(opf: u32, ctx: *mut blk_mq_ctx) -> *mut blk_mq_hw_ctx {
unsafe { *((*ctx).hctxs.as_mut_ptr() as *mut *mut blk_mq_hw_ctx).wrapping_offset((blk_mq_get_hctx_type(opf) as u32) as isize) }
}

fn __blk_mq_get_ctx(q: *mut request_queue, cpu: u32) -> *mut blk_mq_ctx {
per_cpu_ptr!(unsafe { (*q).queue_ctx }, cpu)
}

fn blk_mq_get_ctx(q: *mut request_queue) -> *mut blk_mq_ctx {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
__verify_pcpu_ptr__3!();
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v2 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v2;
}
2 => {
let mut __v4: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v4 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v4;
}
4 => {
let mut __v5: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
8 => {
let mut __v6: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v6 = pfo_val__ as i32;
pscr_ret__ = __v6;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v0 = pscr_ret__;
__blk_mq_get_ctx(q, __v0 as u32)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_alloc_data {
    q: *mut request_queue,
    flags: u32,
    shallow_depth: u32,
    cmd_flags: u32,
    rq_flags: u32,
    nr_tags: u32,
    cached_rqs: *mut rq_list,
    ctx: *mut blk_mq_ctx,
    hctx: *mut blk_mq_hw_ctx,
}


fn bt_wait_ptr(bt: *mut sbitmap_queue, hctx: *mut blk_mq_hw_ctx) -> *mut sbq_wait_state {
if hctx.is_null() {
return unsafe { (*bt).ws }.wrapping_offset(0 as isize);
}
sbq_wait_ptr(bt, unsafe { core::ptr::addr_of_mut!((*hctx).wait_index) })
}

fn blk_mq_tag_busy(hctx: *mut blk_mq_hw_ctx) {
if (unsafe { (*hctx).flags }) & 2 != 0 {
unsafe { __blk_mq_tag_busy(hctx) };
}
}

fn blk_mq_tag_idle(hctx: *mut blk_mq_hw_ctx) {
if (unsafe { (*hctx).flags }) & 2 != 0 {
unsafe { __blk_mq_tag_idle(hctx) };
}
}

fn blk_mq_tag_is_reserved(tags: *mut blk_mq_tags, tag: u32) -> bool {
tag < unsafe { (*tags).nr_reserved_tags }
}

fn blk_mq_is_shared_tags(flags: u32) -> bool {
flags & 8 != 0
}

fn blk_mq_tags_from_data(data: *mut blk_mq_alloc_data) -> *mut blk_mq_tags {
if (unsafe { (*data).rq_flags }) & 16u32 != 0 {
return unsafe { (*(*data).hctx).sched_tags };
}
unsafe { (*(*data).hctx).tags }
}

fn blk_mq_hctx_stopped(hctx: *mut blk_mq_hw_ctx) -> bool {
// Fast path: hardware queue is not stopped most of the time.
if likely!(test_bit!(0i32, unsafe { core::ptr::addr_of_mut!((*hctx).__slate_anon_0.state) }) == 0) != 0 {
return false;
}
// This barrier is used to order adding of dispatch list before and
// the test of BLK_MQ_S_STOPPED below. Pairs with the memory barrier
// in blk_mq_start_stopped_hw_queue() so that dispatch code could
// either see BLK_MQ_S_STOPPED is cleared or dispatch list is not
// empty to avoid missing dispatching requests.
loop {
kcsan_mb!();
unsafe {
core::arch::asm!("lock addl $0,-4(%rsp)", options(nostack, att_syntax));
}
if true {
break;
}
}
test_bit!(0i32, unsafe { core::ptr::addr_of_mut!((*hctx).__slate_anon_0.state) }) != 0
}

fn blk_mq_hw_queue_mapped(hctx: *mut blk_mq_hw_ctx) -> bool {
(unsafe { (*hctx).nr_ctx }) != 0 && !unsafe { (*hctx).tags }.is_null()
}

fn blk_mq_put_dispatch_budget(q: *mut request_queue, budget_token: i32) {
if (unsafe { (*(*q).mq_ops).put_budget }) != None {
unsafe { unsafe { (*(*q).mq_ops).put_budget }.unwrap()(q, budget_token) };
}
}

fn blk_mq_get_dispatch_budget(q: *mut request_queue) -> i32 {
if (unsafe { (*(*q).mq_ops).get_budget }) != None {
return unsafe { unsafe { (*(*q).mq_ops).get_budget }.unwrap()(q) };
}
0
}

fn blk_mq_set_rq_budget_token(rq: *mut request, token: i32) {
if token < 0 {
return;
}
if (unsafe { (*(*(*rq).q).mq_ops).set_rq_budget_token }) != None {
unsafe { unsafe { (*(*(*rq).q).mq_ops).set_rq_budget_token }.unwrap()(rq, token) };
}
}

fn blk_mq_get_rq_budget_token(rq: *mut request) -> i32 {
if (unsafe { (*(*(*rq).q).mq_ops).get_rq_budget_token }) != None {
return unsafe { unsafe { (*(*(*rq).q).mq_ops).get_rq_budget_token }.unwrap()(rq) };
}
1i32.wrapping_neg()
}

fn __blk_mq_add_active_requests(hctx: *mut blk_mq_hw_ctx, val: i32) {
if blk_mq_is_shared_tags((unsafe { (*hctx).flags }) as u32) {
atomic_add(val, unsafe { core::ptr::addr_of_mut!((*(*hctx).queue).nr_active_requests_shared_tags) });
} else {
atomic_add(val, unsafe { core::ptr::addr_of_mut!((*hctx).nr_active) });
}
}

fn __blk_mq_inc_active_requests(hctx: *mut blk_mq_hw_ctx) {
__blk_mq_add_active_requests(hctx, 1i32);
}

fn __blk_mq_sub_active_requests(hctx: *mut blk_mq_hw_ctx, val: i32) {
if blk_mq_is_shared_tags((unsafe { (*hctx).flags }) as u32) {
atomic_sub(val, unsafe { core::ptr::addr_of_mut!((*(*hctx).queue).nr_active_requests_shared_tags) });
} else {
atomic_sub(val, unsafe { core::ptr::addr_of_mut!((*hctx).nr_active) });
}
}

fn __blk_mq_dec_active_requests(hctx: *mut blk_mq_hw_ctx) {
__blk_mq_sub_active_requests(hctx, 1i32);
}

fn blk_mq_add_active_requests(hctx: *mut blk_mq_hw_ctx, val: i32) {
if (unsafe { (*hctx).flags }) & 2 != 0 {
__blk_mq_add_active_requests(hctx, val);
}
}

fn blk_mq_inc_active_requests(hctx: *mut blk_mq_hw_ctx) {
if (unsafe { (*hctx).flags }) & 2 != 0 {
__blk_mq_inc_active_requests(hctx);
}
}

fn blk_mq_sub_active_requests(hctx: *mut blk_mq_hw_ctx, val: i32) {
if (unsafe { (*hctx).flags }) & 2 != 0 {
__blk_mq_sub_active_requests(hctx, val);
}
}

fn blk_mq_dec_active_requests(hctx: *mut blk_mq_hw_ctx) {
if (unsafe { (*hctx).flags }) & 2 != 0 {
__blk_mq_dec_active_requests(hctx);
}
}

fn __blk_mq_active_requests(hctx: *mut blk_mq_hw_ctx) -> i32 {
if blk_mq_is_shared_tags((unsafe { (*hctx).flags }) as u32) {
return atomic_read((unsafe { core::ptr::addr_of_mut!((*(*hctx).queue).nr_active_requests_shared_tags) }) as *const __SlateRecord_types_h_190);
}
atomic_read((unsafe { core::ptr::addr_of_mut!((*hctx).nr_active) }) as *const __SlateRecord_types_h_190)
}

fn __blk_mq_put_driver_tag(hctx: *mut blk_mq_hw_ctx, rq: *mut request) {
blk_mq_dec_active_requests(hctx);
unsafe { blk_mq_put_tag((*hctx).tags, (*rq).mq_ctx, (*rq).tag as u32) };
unsafe {
(*rq).tag = -1i32;
}
}

fn blk_mq_put_driver_tag(rq: *mut request) {
if ((unsafe { (*rq).tag }) as u32) == 4294967295 || ((unsafe { (*rq).internal_tag }) as u32) == 4294967295 {
return;
}
__blk_mq_put_driver_tag(unsafe { (*rq).mq_hctx }, rq);
}

fn blk_mq_get_driver_tag(rq: *mut request) -> bool {
let __v0: bool = ((unsafe { (*rq).tag }) as u32) == 4294967295 && !__blk_mq_alloc_driver_tag(rq);
if __v0 {
return false;
}
true
}

fn blk_mq_clear_mq_map(qmap: *mut blk_mq_queue_map) {
let mut cpu: i32 = 0;
loop {
cpu = find_next_bit(cpumask_bits!((unsafe { core::ptr::addr_of_mut!(__cpu_possible_mask) }) as *const cpumask), 64u64, cpu as u64) as i32;
if (cpu as u32) >= 64 {
break;
}
unsafe {
*unsafe { (*qmap).mq_map }.wrapping_offset(cpu as isize) = 0;
}
cpu = cpu.wrapping_add(1);
}
}

fn hctx_may_queue(hctx: *mut blk_mq_hw_ctx, bt: *mut sbitmap_queue) -> bool {
let mut depth: u32 = 0;
let mut users: u32 = 0;
if hctx.is_null() || (unsafe { (*hctx).flags }) & 2 == 0 {
return true;
}
// Don't try dividing an ant
if (unsafe { (*bt).sb.depth }) == 1 {
return true;
}
if blk_mq_is_shared_tags((unsafe { (*hctx).flags }) as u32) {
let q: *mut request_queue = unsafe { (*hctx).queue };
if test_bit!(11i32, unsafe { core::ptr::addr_of_mut!((*q).queue_flags) }) == 0 {
return true;
}
} else {
if test_bit!(1i32, unsafe { core::ptr::addr_of_mut!((*hctx).__slate_anon_0.state) }) == 0 {
return true;
}
}
let mut __v2: u32;
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_885() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*(*hctx).tags).active_queues) as *const u32) };
users = __v2;
if users == 0 {
return true;
}
// Allow at least some tags
let mut __v4: u32;
let __UNIQUE_ID_x__886: u32 = unsafe { (*bt).sb.depth }.wrapping_add(users).wrapping_sub(1) / users;
let __UNIQUE_ID_y__887: u32 = 4;
loop {
if __types_ok!(__UNIQUE_ID_x__886, __UNIQUE_ID_y__887) == 0 {
unsafe { __compiletime_assert_888() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v4 = if __UNIQUE_ID_x__886 > __UNIQUE_ID_y__887 { __UNIQUE_ID_x__886 } else { __UNIQUE_ID_y__887 };
depth = __v4;
(__blk_mq_active_requests(hctx) as u32) < depth
}

fn blk_mq_can_poll(q: *mut request_queue) -> bool {
(unsafe { (*q).limits.features }) & 512u32 != 0 && (unsafe { (*((*(*q).tag_set).map.as_mut_ptr() as *mut blk_mq_queue_map).wrapping_offset(2 as isize)).nr_queues }) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_debugfs_attr {
    name: *const u8,
    mode: u16,
    show: Option<unsafe extern "C" fn(*mut (), *mut seq_file) -> i32>,
    write: Option<unsafe extern "C" fn(*mut (), *const u8, u64, *mut i64) -> i64>,
    seq_ops: *const seq_operations,
}


// Request is asynchronous
// Increment/decrement the
// usage_count
// Use autosuspend_delay
// Succeed if runtime PM is disabled
fn pm_runtime_mark_last_busy(dev: *mut device) {
loop {
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_889() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__3!((*dev).power.last_busy, unsafe { ktime_get_mono_fast_ns() });
if true {
break;
}
}
}

fn blk_pm_mark_last_busy(rq: *mut request) {
if !unsafe { (*(*rq).q).dev }.is_null() && (unsafe { (*rq).rq_flags }) & 512u32 == 0 {
pm_runtime_mark_last_busy(unsafe { (*(*rq).q).dev });
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct elevator_tags {
    nr_hw_queues: u32,
    nr_requests: u32,
    tags: [*mut blk_mq_tags; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct elevator_resources {
    data: *mut (),
    et: *mut elevator_tags,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct elv_change_ctx {
    name: *const u8,
    no_uevent: bool,
    old: *mut elevator_queue,
    new: *mut elevator_queue,
    r#type: *mut elevator_type,
    res: elevator_resources,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct elevator_mq_ops {
    init_sched: Option<unsafe extern "C" fn(*mut request_queue, *mut elevator_queue) -> i32>,
    exit_sched: Option<unsafe extern "C" fn(*mut elevator_queue)>,
    init_hctx: Option<unsafe extern "C" fn(*mut blk_mq_hw_ctx, u32) -> i32>,
    exit_hctx: Option<unsafe extern "C" fn(*mut blk_mq_hw_ctx, u32)>,
    depth_updated: Option<unsafe extern "C" fn(*mut request_queue)>,
    alloc_sched_data: Option<unsafe extern "C" fn(*mut request_queue) -> *mut ()>,
    free_sched_data: Option<unsafe extern "C" fn(*mut ())>,
    allow_merge: Option<unsafe extern "C" fn(*mut request_queue, *mut request, *mut bio) -> bool>,
    bio_merge: Option<unsafe extern "C" fn(*mut request_queue, *mut bio, u32) -> bool>,
    request_merge: Option<unsafe extern "C" fn(*mut request_queue, *mut *mut request, *mut bio) -> i32>,
    request_merged: Option<unsafe extern "C" fn(*mut request_queue, *mut request, u32)>,
    requests_merged: Option<unsafe extern "C" fn(*mut request_queue, *mut request, *mut request)>,
    limit_depth: Option<unsafe extern "C" fn(u32, *mut blk_mq_alloc_data)>,
    prepare_request: Option<unsafe extern "C" fn(*mut request)>,
    finish_request: Option<unsafe extern "C" fn(*mut request)>,
    insert_requests: Option<unsafe extern "C" fn(*mut blk_mq_hw_ctx, *mut list_head, u32)>,
    dispatch_request: Option<unsafe extern "C" fn(*mut blk_mq_hw_ctx) -> *mut request>,
    has_work: Option<unsafe extern "C" fn(*mut blk_mq_hw_ctx) -> bool>,
    completed_request: Option<unsafe extern "C" fn(*mut request, u64)>,
    requeue_request: Option<unsafe extern "C" fn(*mut request)>,
    former_request: Option<unsafe extern "C" fn(*mut request_queue, *mut request) -> *mut request>,
    next_request: Option<unsafe extern "C" fn(*mut request_queue, *mut request) -> *mut request>,
    init_icq: Option<unsafe extern "C" fn(*mut io_cq)>,
    exit_icq: Option<unsafe extern "C" fn(*mut io_cq)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct elv_fs_entry {
    attr: attribute,
    show: Option<unsafe extern "C" fn(*mut elevator_queue, *mut u8) -> i64>,
    store: Option<unsafe extern "C" fn(*mut elevator_queue, *const u8, u64) -> i64>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct elevator_type {
    icq_cache: *mut kmem_cache,
    ops: elevator_mq_ops,
    icq_size: u64,
    icq_align: u64,
    elevator_attrs: *const elv_fs_entry,
    elevator_name: *const u8,
    elevator_alias: *const u8,
    elevator_owner: *mut module,
    queue_debugfs_attrs: *const blk_mq_debugfs_attr,
    hctx_debugfs_attrs: *const blk_mq_debugfs_attr,
    icq_cache_name: [u8; 22],
    list: list_head,
}


fn __elevator_get(e: *mut elevator_type) {
unsafe { __module_get((*e).elevator_owner) };
}

fn elevator_put(e: *mut elevator_type) {
unsafe { module_put((*e).elevator_owner) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct elevator_queue {
    r#type: *mut elevator_type,
    et: *mut elevator_tags,
    elevator_data: *mut (),
    kobj: kobject,
    sysfs_lock: mutex,
    flags: u64,
    hash: [hlist_head; 64],
}


fn blk_mq_sched_restart(hctx: *mut blk_mq_hw_ctx) {
if test_bit!(2i32, unsafe { core::ptr::addr_of_mut!((*hctx).__slate_anon_0.state) }) != 0 {
unsafe { __blk_mq_sched_restart(hctx) };
}
}

fn bio_mergeable(bio: *mut bio) -> bool {
(unsafe { (*bio).bi_opf }) & 409600u32 == 0
}

fn blk_mq_sched_completed_request(rq: *mut request, now: u64) {
if (unsafe { (*rq).rq_flags }) & 32u32 != 0 {
let e: *mut elevator_queue = unsafe { (*(*rq).q).elevator };
if (unsafe { (*(*e).r#type).ops.completed_request }) != None {
unsafe { unsafe { (*(*e).r#type).ops.completed_request }.unwrap()(rq, now) };
}
}
}

fn blk_mq_sched_requeue_request(rq: *mut request) {
if (unsafe { (*rq).rq_flags }) & 32u32 != 0 {
let q: *mut request_queue = unsafe { (*rq).q };
let e: *mut elevator_queue = unsafe { (*q).elevator };
if (unsafe { (*(*e).r#type).ops.requeue_request }) != None {
unsafe { unsafe { (*(*e).r#type).ops.requeue_request }.unwrap()(rq) };
}
}
}

fn blk_mq_sched_has_work(hctx: *mut blk_mq_hw_ctx) -> bool {
let e: *mut elevator_queue = unsafe { (*(*hctx).queue).elevator };
if !e.is_null() && (unsafe { (*(*e).r#type).ops.has_work }) != None {
return unsafe { unsafe { (*(*e).r#type).ops.has_work }.unwrap()(hctx) };
}
false
}

fn blk_mq_sched_needs_restart(hctx: *mut blk_mq_hw_ctx) -> bool {
test_bit!(2i32, unsafe { core::ptr::addr_of_mut!((*hctx).__slate_anon_0.state) }) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rq_qos {
    ops: *const rq_qos_ops,
    disk: *mut gendisk,
    id: u32,
    next: *mut rq_qos,
    debugfs_dir: *mut dentry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rq_qos_ops {
    throttle: Option<unsafe extern "C" fn(*mut rq_qos, *mut bio)>,
    track: Option<unsafe extern "C" fn(*mut rq_qos, *mut request, *mut bio)>,
    merge: Option<unsafe extern "C" fn(*mut rq_qos, *mut request, *mut bio)>,
    issue: Option<unsafe extern "C" fn(*mut rq_qos, *mut request)>,
    requeue: Option<unsafe extern "C" fn(*mut rq_qos, *mut request)>,
    done: Option<unsafe extern "C" fn(*mut rq_qos, *mut request)>,
    done_bio: Option<unsafe extern "C" fn(*mut rq_qos, *mut bio)>,
    cleanup: Option<unsafe extern "C" fn(*mut rq_qos, *mut bio)>,
    queue_depth_changed: Option<unsafe extern "C" fn(*mut rq_qos)>,
    exit: Option<unsafe extern "C" fn(*mut rq_qos)>,
    debugfs_attrs: *const blk_mq_debugfs_attr,
}


fn rq_qos_id(q: &request_queue, id: u32) -> *mut rq_qos {
let mut rqos: *mut rq_qos = unsafe { core::mem::zeroed() };
rqos = unsafe { (*(q as *const request_queue)).rq_qos };
while !rqos.is_null() {
if ((unsafe { (*rqos).id }) as u32) == (id as u32) {
break;
}
rqos = unsafe { (*rqos).next };
}
rqos
}

fn rq_qos_cleanup(q: *mut request_queue, bio: *mut bio) {
if test_bit!(15i32, unsafe { core::ptr::addr_of_mut!((*q).queue_flags) }) != 0 && !unsafe { (*q).rq_qos }.is_null() {
unsafe { __rq_qos_cleanup((*q).rq_qos, bio) };
}
}

fn rq_qos_done(q: *mut request_queue, rq: *mut request) {
let __v0: bool = test_bit!(15i32, unsafe { core::ptr::addr_of_mut!((*q).queue_flags) }) != 0 && !unsafe { (*q).rq_qos }.is_null() && !blk_rq_is_passthrough(rq);
if __v0 {
unsafe { __rq_qos_done((*q).rq_qos, rq) };
}
}

fn rq_qos_issue(q: *mut request_queue, rq: *mut request) {
if test_bit!(15i32, unsafe { core::ptr::addr_of_mut!((*q).queue_flags) }) != 0 && !unsafe { (*q).rq_qos }.is_null() {
unsafe { __rq_qos_issue((*q).rq_qos, rq) };
}
}

fn rq_qos_requeue(q: *mut request_queue, rq: *mut request) {
if test_bit!(15i32, unsafe { core::ptr::addr_of_mut!((*q).queue_flags) }) != 0 && !unsafe { (*q).rq_qos }.is_null() {
unsafe { __rq_qos_requeue((*q).rq_qos, rq) };
}
}

fn rq_qos_throttle(q: *mut request_queue, bio: *mut bio) {
if test_bit!(15i32, unsafe { core::ptr::addr_of_mut!((*q).queue_flags) }) != 0 && !unsafe { (*q).rq_qos }.is_null() {
bio_set_flag(bio, 8u32);
unsafe { __rq_qos_throttle((*q).rq_qos, bio) };
}
}

fn rq_qos_track(q: *mut request_queue, rq: *mut request, bio: *mut bio) {
if test_bit!(15i32, unsafe { core::ptr::addr_of_mut!((*q).queue_flags) }) != 0 && !unsafe { (*q).rq_qos }.is_null() {
unsafe { __rq_qos_track((*q).rq_qos, rq, bio) };
}
}

#[unsafe(link_section = ".data..percpu")]
static mut blk_cpu_done: llist_head = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".data..percpu")]
static mut blk_cpu_csd: __SlateAlign32<__call_single_data> = __SlateAlign32(unsafe { core::mem::zeroed() });

static mut blk_mq_cpuhp_lock: mutex = {
let mut __t0: mutex = unsafe { core::mem::zeroed() };
__t0.owner = __SlateRecord_types_h_197 { counter: 0i64 };
__t0.wait_lock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0.first_waiter = core::ptr::null_mut::<mutex_waiter>();
__t0
};

/// Check if any of the ctx, dispatch list or elevator
/// have pending work in this hardware queue.
fn blk_mq_hctx_has_pending(hctx: *mut blk_mq_hw_ctx) -> bool {
let __v0: bool = list_empty_careful((unsafe { core::ptr::addr_of_mut!((*hctx).__slate_anon_0.dispatch) }) as *const list_head) == 0 || unsafe { sbitmap_any_bit_set(core::ptr::addr_of_mut!((*hctx).ctx_map) as *const sbitmap) };
__v0 || blk_mq_sched_has_work(hctx)
}

/// Mark this ctx as having pending work in this hardware queue
fn blk_mq_hctx_mark_pending(hctx: *mut blk_mq_hw_ctx, ctx: *mut blk_mq_ctx) {
let bit: i32 = (unsafe { *((*ctx).index_hw.as_mut_ptr() as *mut u16).wrapping_offset(((*hctx).r#type as i32) as isize) }) as i32;
if sbitmap_test_bit(unsafe { core::ptr::addr_of_mut!((*hctx).ctx_map) }, bit as u32) == 0 {
sbitmap_set_bit(unsafe { core::ptr::addr_of_mut!((*hctx).ctx_map) }, bit as u32);
}
}

fn blk_mq_hctx_clear_pending(hctx: *mut blk_mq_hw_ctx, ctx: *mut blk_mq_ctx) {
let bit: i32 = (unsafe { *((*ctx).index_hw.as_mut_ptr() as *mut u16).wrapping_offset(((*hctx).r#type as i32) as isize) }) as i32;
sbitmap_clear_bit(unsafe { core::ptr::addr_of_mut!((*hctx).ctx_map) }, bit as u32);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mq_inflight {
    part: *mut block_device,
    inflight: [u32; 2],
}


#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_check_in_driver")]
extern "C" fn blk_mq_check_in_driver(rq: *mut request, r#priv: *mut ()) -> bool {
let __v0: bool = (unsafe { (*rq).rq_flags }) & 256u32 != 0 && (!bdev_is_partition(unsafe { (*(r#priv as *mut mq_inflight)).part }) || (unsafe { (*rq).part }) == unsafe { (*(r#priv as *mut mq_inflight)).part });
let __v1: bool = __v0 && (blk_mq_rq_state(rq) as u32) == 1;
if __v1 {
let __v2: *mut u32 = unsafe { (*(r#priv as *mut mq_inflight)).inflight.as_mut_ptr() as *mut u32 }.wrapping_offset(rq_data_dir!(rq) as isize);
let __v3: u32 = unsafe { *__v2 };
unsafe {
*__v2 = __v3.wrapping_add(1);
}
}
true
}

#[unsafe(no_mangle)]
extern "C" fn blk_mq_in_driver_rw(part: *mut block_device, inflight: *mut u32) {
let mut mi: mq_inflight = {
let mut __t0: mq_inflight = unsafe { core::mem::zeroed() };
__t0.part = part;
__t0
};
unsafe { blk_mq_queue_tag_busy_iter(bdev_get_queue(part), Some(blk_mq_check_in_driver), core::ptr::addr_of_mut!(mi) as *mut ()) };
unsafe {
*inflight.wrapping_offset(0 as isize) = mi.inflight[0];
}
unsafe {
*inflight.wrapping_offset(1 as isize) = mi.inflight[1];
}
}

fn blk_freeze_set_owner(q: *mut request_queue, owner: *mut task_struct) -> bool {
false
}

fn blk_unfreeze_check_owner(q: *mut request_queue) -> bool {
false
}

#[unsafe(no_mangle)]
extern "C" fn __blk_freeze_queue_start(q: *mut request_queue, owner: *mut task_struct) -> bool {
let mut freeze: bool = false;
unsafe { mutex_lock(core::ptr::addr_of_mut!((*q).mq_freeze_lock)) };
freeze = blk_freeze_set_owner(q, owner);
let __v1: i32 = unsafe { (*q).mq_freeze_depth };
let __v2: i32 = __v1.wrapping_add(1);
unsafe {
(*q).mq_freeze_depth = __v2;
}
if __v2 == 1 {
percpu_ref_kill(unsafe { core::ptr::addr_of_mut!((*q).q_usage_counter) });
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*q).mq_freeze_lock)) };
if queue_is_mq(q) {
blk_mq_run_hw_queues(q, false);
}
} else {
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*q).mq_freeze_lock)) };
}
freeze
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_freeze_queue_start")]
extern "C" fn blk_freeze_queue_start(q: *mut request_queue) {
if __blk_freeze_queue_start(q, get_current()) {
blk_freeze_acquire_lock(q);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_freeze_queue_start_890: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request_queue)>, *mut ()>(Some(blk_freeze_queue_start)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_freeze_queue_wait")]
extern "C" fn blk_mq_freeze_queue_wait(q: *mut request_queue) {
let mut __wq_entry: core::mem::MaybeUninit<wait_queue_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut wait_queue_entry = core::ptr::addr_of_mut!(__wq_entry) as *mut wait_queue_entry;
unsafe {
loop {
might_sleep!();
if percpu_ref_is_zero(core::ptr::addr_of_mut!((*q).q_usage_counter)) {
break;
} else {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 40usize);
init_wait_entry(core::ptr::addr_of_mut!(*__slate_slot_2), if false { 1i32 } else { 0i32 });
loop {
prepare_to_wait_event(core::ptr::addr_of_mut!((*q).mq_freeze_wq), core::ptr::addr_of_mut!(*__slate_slot_2), 2i32);
if percpu_ref_is_zero(core::ptr::addr_of_mut!((*q).q_usage_counter)) {
break;
} else {
schedule();
if percpu_ref_is_zero(core::ptr::addr_of_mut!((*q).q_usage_counter)) {
break;
}
}
}
finish_wait(core::ptr::addr_of_mut!((*q).mq_freeze_wq), core::ptr::addr_of_mut!(*__slate_slot_2));
if true {
break;
}
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_freeze_queue_wait_891: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request_queue)>, *mut ()>(Some(blk_mq_freeze_queue_wait)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_freeze_queue_wait_timeout")]
extern "C" fn blk_mq_freeze_queue_wait_timeout(q: *mut request_queue, timeout: u64) -> i32 {
let mut __cond: bool = unsafe { core::mem::zeroed() };
let mut __cond_1: bool = unsafe { core::mem::zeroed() };
let mut __ret: i64 = unsafe { core::mem::zeroed() };
let mut __wq_entry: core::mem::MaybeUninit<wait_queue_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut wait_queue_entry = core::ptr::addr_of_mut!(__wq_entry) as *mut wait_queue_entry;
let mut __ret_1: i64 = unsafe { core::mem::zeroed() };
unsafe {
__ret_1 = timeout as i64;
might_sleep!();
if !___wait_cond_timeout!(percpu_ref_is_zero(core::ptr::addr_of_mut!((*q).q_usage_counter)), __ret_1) {
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 40usize);
__ret = timeout as i64;
init_wait_entry(core::ptr::addr_of_mut!(*__slate_slot_6), if false { 1i32 } else { 0i32 });
loop {
prepare_to_wait_event(core::ptr::addr_of_mut!((*q).mq_freeze_wq), core::ptr::addr_of_mut!(*__slate_slot_6), 2i32);
__cond_1 = percpu_ref_is_zero(core::ptr::addr_of_mut!((*q).q_usage_counter));
if __cond_1 && __ret == 0 {
__ret = 1;
}
let __v3: bool = __cond_1 || __ret == 0;
if __v3 {
break;
} else {
__ret = schedule_timeout(__ret);
__cond = percpu_ref_is_zero(core::ptr::addr_of_mut!((*q).q_usage_counter));
if __cond && __ret == 0 {
__ret = 1;
}
if __cond || __ret == 0 {
break;
}
}
}
finish_wait(core::ptr::addr_of_mut!((*q).mq_freeze_wq), core::ptr::addr_of_mut!(*__slate_slot_6));
__ret_1 = __ret;
}
return __ret_1 as i32;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_freeze_queue_wait_timeout_892: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request_queue, u64) -> i32>, *mut ()>(Some(blk_mq_freeze_queue_wait_timeout)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_freeze_queue_nomemsave")]
extern "C" fn blk_mq_freeze_queue_nomemsave(q: *mut request_queue) {
blk_freeze_queue_start(q);
blk_mq_freeze_queue_wait(q);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_freeze_queue_nomemsave_893: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request_queue)>, *mut ()>(Some(blk_mq_freeze_queue_nomemsave)) };

#[unsafe(no_mangle)]
extern "C" fn __blk_mq_unfreeze_queue(q: *mut request_queue, force_atomic: bool) -> bool {
let mut unfreeze: bool = false;
unsafe { mutex_lock(core::ptr::addr_of_mut!((*q).mq_freeze_lock)) };
if force_atomic {
unsafe {
let __t0 = true;
(*(*q).q_usage_counter.data).__slate_bits_0.__set_force_atomic(__t0);
}
}
let __v1: i32 = unsafe { (*q).mq_freeze_depth };
unsafe {
(*q).mq_freeze_depth = __v1.wrapping_sub(1);
}
let mut __v3: i64;
let __ret_warn_on: i32 = ((unsafe { (*q).mq_freeze_depth }) < 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("894: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 894b - ., 3; .popsection\n# {0}", const 894 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___blk_mq_unfreeze_queue_4, sym __slate_asm_string___blk_mq_unfreeze_queue_5, const 219 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("895: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 895b - ., 4; .popsection\n# {0}", const 895 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v3 = (__ret_warn_on != 0) as i64;
if (unsafe { (*q).mq_freeze_depth }) == 0 {
unsafe { percpu_ref_resurrect(core::ptr::addr_of_mut!((*q).q_usage_counter)) };
wake_up_all!(unsafe { core::ptr::addr_of_mut!((*q).mq_freeze_wq) });
}
unfreeze = blk_unfreeze_check_owner(q);
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*q).mq_freeze_lock)) };
unfreeze
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_unfreeze_queue_nomemrestore")]
extern "C" fn blk_mq_unfreeze_queue_nomemrestore(q: *mut request_queue) {
if __blk_mq_unfreeze_queue(q, false) {
blk_unfreeze_release_lock(q);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_unfreeze_queue_nomemrestore_896: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request_queue)>, *mut ()>(Some(blk_mq_unfreeze_queue_nomemrestore)) };

/// non_owner variant of blk_freeze_queue_start
///
/// Unlike blk_freeze_queue_start, the queue doesn't need to be unfrozen
/// by the same task.  This is fragile and should not be used if at all
/// possible.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_freeze_queue_start_non_owner")]
extern "C" fn blk_freeze_queue_start_non_owner(q: *mut request_queue) {
__blk_freeze_queue_start(q, core::ptr::null_mut::<task_struct>());
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_freeze_queue_start_non_owner_897: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request_queue)>, *mut ()>(Some(blk_freeze_queue_start_non_owner)) };

/// non_owner variant of blk_mq_unfreeze_queue
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_unfreeze_queue_non_owner")]
extern "C" fn blk_mq_unfreeze_queue_non_owner(q: *mut request_queue) {
__blk_mq_unfreeze_queue(q, false);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_unfreeze_queue_non_owner_898: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request_queue)>, *mut ()>(Some(blk_mq_unfreeze_queue_non_owner)) };

/// FIXME: replace the scsi_internal_device_*block_nowait() calls in the
/// mpt3sas driver such that this function can be removed.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_quiesce_queue_nowait")]
extern "C" fn blk_mq_quiesce_queue_nowait(q: *mut request_queue) {
let mut flags: u64 = 0;
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*q).queue_lock) }, flags);
let __v4: i32 = unsafe { (*q).quiesce_depth };
unsafe {
(*q).quiesce_depth = __v4.wrapping_add(1);
}
if __v4 == 0 {
unsafe { blk_queue_flag_set(9u32, q) };
}
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*q).queue_lock) }, flags);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_quiesce_queue_nowait_899: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request_queue)>, *mut ()>(Some(blk_mq_quiesce_queue_nowait)) };

/// blk_mq_wait_quiesce_done() - wait until in-progress quiesce is done
/// @set: tag_set to wait on
///
/// Note: it is driver's responsibility for making sure that quiesce has
/// been started on or more of the request_queues of the tag_set.  This
/// function only waits for the quiesce on those request_queues that had
/// the quiesce flag set using blk_mq_quiesce_queue_nowait.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_wait_quiesce_done")]
extern "C" fn blk_mq_wait_quiesce_done(set: *mut blk_mq_tag_set) {
if (unsafe { (*set).flags }) & 16 != 0 {
unsafe { synchronize_srcu((*set).srcu) };
} else {
unsafe { synchronize_rcu() };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_wait_quiesce_done_900: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut blk_mq_tag_set)>, *mut ()>(Some(blk_mq_wait_quiesce_done)) };

/// blk_mq_quiesce_queue() - wait until all ongoing dispatches have finished
/// @q: request queue.
///
/// Note: this function does not prevent that the struct request end_io()
/// callback function is invoked. Once this function is returned, we make
/// sure no dispatch can happen until the queue is unquiesced via
/// blk_mq_unquiesce_queue().
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_quiesce_queue")]
extern "C" fn blk_mq_quiesce_queue(q: *mut request_queue) {
blk_mq_quiesce_queue_nowait(q);
// nothing to wait for non-mq queues
if queue_is_mq(q) {
blk_mq_wait_quiesce_done(unsafe { (*q).tag_set });
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_quiesce_queue_901: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request_queue)>, *mut ()>(Some(blk_mq_quiesce_queue)) };

/// blk_mq_unquiesce_queue() - counterpart of blk_mq_quiesce_queue()
/// @q: request queue.
///
/// This function recovers queue into the state before quiescing
/// which is done by blk_mq_quiesce_queue.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_unquiesce_queue")]
extern "C" fn blk_mq_unquiesce_queue(q: *mut request_queue) {
let mut flags: u64 = 0;
let mut run_queue: bool = false;
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*q).queue_lock) }, flags);
let mut __v3: i64;
let __ret_warn_on: i32 = ((unsafe { (*q).quiesce_depth }) <= 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("902: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 902b - ., 3; .popsection\n# {0}", const 902 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_blk_mq_unquiesce_queue_4, sym __slate_asm_string_blk_mq_unquiesce_queue_5, const 321 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("903: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 903b - ., 4; .popsection\n# {0}", const 903 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v3 = (__ret_warn_on != 0) as i64;
if __v3 == 0 {
let __v9: i32 = unsafe { (*q).quiesce_depth };
let __v10: i32 = __v9.wrapping_sub(1);
unsafe {
(*q).quiesce_depth = __v10;
}
if __v10 == 0 {
unsafe { blk_queue_flag_clear(9u32, q) };
run_queue = true;
}
}
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*q).queue_lock) }, flags);
// dispatch requests which are inserted during quiescing
if run_queue {
blk_mq_run_hw_queues(q, true);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_unquiesce_queue_904: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request_queue)>, *mut ()>(Some(blk_mq_unquiesce_queue)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_quiesce_tagset_907: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut blk_mq_tag_set)>, *mut ()>(Some(blk_mq_quiesce_tagset)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_unquiesce_tagset_910: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut blk_mq_tag_set)>, *mut ()>(Some(blk_mq_unquiesce_tagset)) };

#[unsafe(no_mangle)]
extern "C" fn blk_mq_wake_waiters(q: *mut request_queue) {
let mut hctx: *mut blk_mq_hw_ctx = unsafe { core::mem::zeroed() };
let mut i: u64 = 0;
loop {
let __v0: bool = if i < ((unsafe { (*q).nr_hw_queues }) as u64) {
let mut __v1: i32;
hctx = queue_hctx(q, i as i32);
__v1 = 1;
__v1 != 0
} else {
false
};
if !__v0 {
break;
}
if blk_mq_hw_queue_mapped(hctx) {
unsafe { blk_mq_tag_wakeup_all((*hctx).tags, true) };
}
i = i.wrapping_add(1);
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_rq_init")]
extern "C" fn blk_rq_init(q: *mut request_queue, rq: *mut request) {
unsafe { memset(rq as *mut (), 0i32, core::mem::size_of::<request>() as u64) };
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*rq).__slate_anon_12.queuelist) });
unsafe {
(*rq).q = q;
}
unsafe {
(*rq).__sector = 1i32.wrapping_neg() as u64;
}
unsafe {
(*rq).phys_gap_bit = 0;
}
INIT_HLIST_NODE(unsafe { core::ptr::addr_of_mut!((*rq).__slate_anon_24.hash) });
RB_CLEAR_NODE!(unsafe { core::ptr::addr_of_mut!((*rq).__slate_anon_25.rb_node) });
unsafe {
(*rq).tag = -1i32;
}
unsafe {
(*rq).internal_tag = -1i32;
}
unsafe {
(*rq).start_time_ns = blk_time_get_ns();
}
blk_crypto_rq_set_defaults(rq);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_rq_init_911: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request_queue, *mut request)>, *mut ()>(Some(blk_rq_init)) };

/// Set start and alloc time when the allocated request is actually used
fn blk_mq_rq_time_init(rq: *mut request, alloc_time_ns: u64) {
if blk_queue_rq_alloc_time!(unsafe { (*rq).q }) != 0 {
unsafe {
(*rq).alloc_time_ns = alloc_time_ns;
}
} else {
unsafe {
(*rq).alloc_time_ns = 0;
}
}
}

fn blk_mq_bio_issue_init(q: *mut request_queue, bio: *mut bio) {
if test_bit!(16i32, unsafe { core::ptr::addr_of_mut!((*q).queue_flags) }) != 0 {
unsafe {
(*bio).issue_time_ns = blk_time_get_ns();
}
}
}

fn blk_mq_rq_ctx_init(data: *mut blk_mq_alloc_data, tags: *mut blk_mq_tags, tag: u32) -> *mut request {
let ctx: *mut blk_mq_ctx = unsafe { (*data).ctx };
let hctx: *mut blk_mq_hw_ctx = unsafe { (*data).hctx };
let q: *mut request_queue = unsafe { (*data).q };
let rq: *mut request = unsafe { *unsafe { (*tags).static_rqs }.wrapping_offset(tag as isize) };
unsafe {
(*rq).q = q;
}
unsafe {
(*rq).mq_ctx = ctx;
}
unsafe {
(*rq).mq_hctx = hctx;
}
unsafe {
(*rq).cmd_flags = (*data).cmd_flags;
}
if (unsafe { (*data).flags }) & 4 != 0 {
unsafe {
(*data).rq_flags = (*data).rq_flags | 512u32;
}
}
unsafe {
(*rq).rq_flags = (*data).rq_flags;
}
if (unsafe { (*data).rq_flags }) & 16u32 != 0 {
unsafe {
(*rq).tag = -1i32;
}
unsafe {
(*rq).internal_tag = tag as i32;
}
} else {
unsafe {
(*rq).tag = tag as i32;
}
unsafe {
(*rq).internal_tag = -1i32;
}
}
unsafe {
(*rq).timeout = 0;
}
unsafe {
(*rq).part = core::ptr::null_mut::<block_device>();
}
unsafe {
(*rq).io_start_time_ns = 0;
}
unsafe {
(*rq).stats_sectors = 0;
}
unsafe {
(*rq).nr_phys_segments = 0;
}
unsafe {
(*rq).nr_integrity_segments = 0;
}
unsafe {
(*rq).end_io = None;
}
unsafe {
(*rq).end_io_data = core::ptr::null_mut::<()>();
}
blk_crypto_rq_set_defaults(rq);
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*rq).__slate_anon_12.queuelist) });
// tag was already set
loop {
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_912() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__13!((*rq).deadline, 0i32);
if true {
break;
}
}
req_ref_set(rq, 1i32);
rq
}

fn __blk_mq_alloc_requests_batch(data: *mut blk_mq_alloc_data) -> *mut request {
let mut tag: u32 = 0;
let mut tag_offset: u32 = 0;
let mut tag_mask: u64 = 0;
let mut i: i32 = 0;
let mut nr: i32 = 0;
loop {
tag_mask = unsafe { blk_mq_get_tags(data, unsafe { (*data).nr_tags }.wrapping_sub(nr as u32) as i32, core::ptr::addr_of_mut!(tag_offset)) };
if unlikely__2!(tag_mask == 0) != 0 {
if nr == 0 {
return core::ptr::null_mut::<request>();
}
break;
}
let tags: *mut blk_mq_tags = blk_mq_tags_from_data(data);
i = 0;
while tag_mask != 0 {
if tag_mask & 1u64 << i != 0 {
tag = tag_offset.wrapping_add(i as u32);
prefetch!(unsafe { *unsafe { (*tags).static_rqs }.wrapping_offset(tag as isize) });
let __v3: u64 = tag_mask & !(1u64 << i);
tag_mask = __v3;
let rq: *mut request = blk_mq_rq_ctx_init(data, tags, tag);
rq_list_add_head(unsafe { (*data).cached_rqs }, rq);
nr = nr.wrapping_add(1);
}
i = i.wrapping_add(1);
}
if (unsafe { (*data).nr_tags }) <= (nr as u32) {
break;
}
}
if (unsafe { (*data).rq_flags }) & 16u32 == 0 {
blk_mq_add_active_requests(unsafe { (*data).hctx }, nr);
}
// caller already holds a reference, add for remainder
percpu_ref_get_many(unsafe { core::ptr::addr_of_mut!((*(*data).q).q_usage_counter) }, nr.wrapping_sub(1) as u64);
let __v9: u32 = unsafe { (*data).nr_tags };
unsafe {
(*data).nr_tags = __v9.wrapping_sub(nr as u32);
}
rq_list_pop(unsafe { (*data).cached_rqs })
}

fn blk_op_bypass_sched(opf: u32) -> bool {
opf & 255u32 == 2 || blk_op_is_passthrough(opf)
}

fn blk_mq_limit_depth(data: *mut blk_mq_alloc_data) {
// If no I/O scheduler has been configured, don't limit requests
if unsafe { (*(*data).q).elevator }.is_null() {
blk_mq_tag_busy(unsafe { (*data).hctx });
return;
}
// All requests use scheduler tags when an I/O scheduler is
// enabled for the queue.
unsafe {
(*data).rq_flags = (*data).rq_flags | 16u32;
}
// Flush/passthrough requests are special and go directly to the
// dispatch list, they are not subject to the async_depth limit.
if blk_op_bypass_sched(unsafe { (*data).cmd_flags }) {
return;
}
let mut __v3: i64;
let __ret_warn_on: i32 = ((unsafe { (*data).flags }) & 2 != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("913: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 913b - ., 3; .popsection\n# {0}", const 913 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_blk_mq_limit_depth_4, sym __slate_asm_string_blk_mq_limit_depth_5, const 520 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("914: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 914b - ., 4; .popsection\n# {0}", const 914 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v3 = (__ret_warn_on != 0) as i64;
// By default, sync requests have no limit, and async requests are
// limited to async_depth.
let ops: *mut elevator_mq_ops = unsafe { core::ptr::addr_of_mut!((*(*(*(*data).q).elevator).r#type).ops) };
if (unsafe { (*ops).limit_depth }) != None {
unsafe { unsafe { (*ops).limit_depth }.unwrap()((*data).cmd_flags, data) };
}
}

/// Finish initializing a request once it has been claimed for an operation.
/// Cached requests are allocated before that operation is known.
fn blk_mq_rq_late_init(rq: *mut request, alloc_time_ns: u64) {
blk_mq_rq_time_init(rq, alloc_time_ns);
let __v0: bool = (unsafe { (*rq).rq_flags }) & 16u32 == 0 || (unsafe { (*rq).rq_flags }) & 32768u32 != 0 || blk_op_bypass_sched(unsafe { (*rq).cmd_flags });
if __v0 {
return;
}
unsafe {
(*rq).rq_flags = (*rq).rq_flags | 32u32;
}
INIT_HLIST_NODE(unsafe { core::ptr::addr_of_mut!((*rq).__slate_anon_24.hash) });
RB_CLEAR_NODE!(unsafe { core::ptr::addr_of_mut!((*rq).__slate_anon_25.rb_node) });
let e: *mut elevator_queue = unsafe { (*(*rq).q).elevator };
if (unsafe { (*(*e).r#type).ops.prepare_request }) != None {
unsafe { unsafe { (*(*e).r#type).ops.prepare_request }.unwrap()(rq) };
}
}

fn __blk_mq_alloc_requests(data: *mut blk_mq_alloc_data) -> *mut request {
let mut tag: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u32 = core::ptr::addr_of_mut!(tag) as *mut u32;
let mut rq: core::mem::MaybeUninit<*mut request> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut request = core::ptr::addr_of_mut!(rq) as *mut *mut request;
let mut alloc_time_ns: u64 = unsafe { core::mem::zeroed() };
let mut q: *mut request_queue = unsafe { core::mem::zeroed() };
unsafe {
q = (*data).q;
alloc_time_ns = 0;
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
// alloc_time includes depth and tag waits
if blk_queue_rq_alloc_time!(q) != 0 {
alloc_time_ns = blk_time_get_ns();
}
if (*data).cmd_flags & 2097152u32 != 0 {
(*data).flags = (*data).flags | 1;
}
loop {
(*data).ctx = blk_mq_get_ctx(q);
(*data).hctx = blk_mq_map_queue((*data).cmd_flags, (*data).ctx);
blk_mq_limit_depth(data);
if (*data).flags & 2 != 0 {
(*data).rq_flags = (*data).rq_flags | 32768u32;
}
// Try batched alloc if we want more than 1 tag.
if (*data).nr_tags > 1 {
*__slate_slot_7 = __blk_mq_alloc_requests_batch(data);
if !(*__slate_slot_7).is_null() {
blk_mq_rq_late_init(*__slate_slot_7, alloc_time_ns);
return *__slate_slot_7;
} else {
(*data).nr_tags = 1;
}
}
// Waiting allocations only fail because of an inactive hctx.  In that
// case just retry the hctx assignment and tag allocation as CPU hotplug
// should have migrated us to an online CPU by now.
*__slate_slot_6 = blk_mq_get_tag(data);
if *__slate_slot_6 == 4294967295 {
if (*data).flags & 1 != 0 {
return core::ptr::null_mut::<request>();
} else {
// Give up the CPU and sleep for a random short time to
// ensure that thread using a realtime scheduling class
// are migrated off the CPU, and thus off the hctx that
// is going away.
msleep(3u32);
}
} else {
break;
}
}
if (*data).rq_flags & 16u32 == 0 {
blk_mq_inc_active_requests((*data).hctx);
}
*__slate_slot_7 = blk_mq_rq_ctx_init(data, blk_mq_tags_from_data(data), *__slate_slot_6);
blk_mq_rq_late_init(*__slate_slot_7, alloc_time_ns);
return *__slate_slot_7;
}
}

fn blk_mq_rq_cache_fill(q: *mut request_queue, plug: *mut blk_plug, opf: u32, flags: u32) -> *mut request {
let mut data: blk_mq_alloc_data = {
let mut __t0: blk_mq_alloc_data = unsafe { core::mem::zeroed() };
__t0.q = q;
__t0.flags = flags;
__t0.shallow_depth = 0u32;
__t0.cmd_flags = opf;
__t0.rq_flags = 0u32;
__t0.nr_tags = (unsafe { (*plug).nr_ios }) as u32;
__t0.cached_rqs = unsafe { core::ptr::addr_of_mut!((*plug).cached_rqs) };
__t0.ctx = core::ptr::null_mut::<blk_mq_ctx>();
__t0.hctx = core::ptr::null_mut::<blk_mq_hw_ctx>();
__t0
};
if (unsafe { blk_queue_enter(q, flags) }) != 0 {
return core::ptr::null_mut::<request>();
}
unsafe {
(*plug).nr_ios = 1;
}
let rq: *mut request = __blk_mq_alloc_requests(core::ptr::addr_of_mut!(data));
if unlikely__2!(rq.is_null()) != 0 {
unsafe { blk_queue_exit(q) };
}
rq
}

fn blk_mq_alloc_cached_request(q: *mut request_queue, opf: u32, flags: u32) -> *mut request {
let plug: *mut blk_plug = unsafe { (*get_current()).plug };
let mut rq: *mut request = unsafe { core::mem::zeroed() };
if plug.is_null() {
return core::ptr::null_mut::<request>();
}
if rq_list_empty((unsafe { core::ptr::addr_of_mut!((*plug).cached_rqs) }) as *const rq_list) != 0 {
if ((unsafe { (*plug).nr_ios }) as i32) == 1 {
return core::ptr::null_mut::<request>();
}
rq = blk_mq_rq_cache_fill(q, plug, opf, flags);
} else {
rq = rq_list_peek(unsafe { core::ptr::addr_of_mut!((*plug).cached_rqs) });
if rq.is_null() || (unsafe { (*rq).q }) != q {
return core::ptr::null_mut::<request>();
}
if (blk_mq_get_hctx_type(opf) as u32) != (((unsafe { (*(*rq).mq_hctx).r#type }) as i32) as u32) {
return core::ptr::null_mut::<request>();
}
rq_list_pop(unsafe { core::ptr::addr_of_mut!((*plug).cached_rqs) });
unsafe {
(*rq).cmd_flags = opf;
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*rq).__slate_anon_12.queuelist) });
blk_mq_rq_late_init(rq, blk_time_get_ns());
}
rq
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_alloc_request")]
extern "C" fn blk_mq_alloc_request(q: *mut request_queue, opf: u32, flags: u32) -> *mut request {
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut data: core::mem::MaybeUninit<blk_mq_alloc_data> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut blk_mq_alloc_data = core::ptr::addr_of_mut!(data) as *mut blk_mq_alloc_data;
let mut rq: core::mem::MaybeUninit<*mut request> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut request = core::ptr::addr_of_mut!(rq) as *mut *mut request;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
*__slate_slot_2 = blk_mq_alloc_cached_request(q, opf, flags);
if (*__slate_slot_2).is_null() {
core::ptr::write(__slate_slot_1, {
let mut __t0: blk_mq_alloc_data = core::mem::zeroed();
__t0.q = q;
__t0.flags = flags;
__t0.shallow_depth = 0u32;
__t0.cmd_flags = opf;
__t0.rq_flags = 0u32;
__t0.nr_tags = 1u32;
__t0.cached_rqs = core::ptr::null_mut::<rq_list>();
__t0.ctx = core::ptr::null_mut::<blk_mq_ctx>();
__t0.hctx = core::ptr::null_mut::<blk_mq_hw_ctx>();
__t0
});
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_0 = blk_queue_enter(q, flags);
if *__slate_slot_0 != 0 {
return ERR_PTR(*__slate_slot_0 as i64) as *mut request;
} else {
*__slate_slot_2 = __blk_mq_alloc_requests(core::ptr::addr_of_mut!(*__slate_slot_1));
if (*__slate_slot_2).is_null() {
blk_queue_exit(q);
return ERR_PTR(11i32.wrapping_neg() as i64) as *mut request;
}
}
}
(*(*__slate_slot_2)).__data_len = 0;
(*(*__slate_slot_2)).phys_gap_bit = 0;
(*(*__slate_slot_2)).__sector = 1i32.wrapping_neg() as u64;
(*(*__slate_slot_2)).biotail = core::ptr::null_mut::<bio>();
(*(*__slate_slot_2)).bio = core::ptr::null_mut::<bio>();
return *__slate_slot_2;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_alloc_request_915: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request_queue, u32, u32) -> *mut request>, *mut ()>(Some(blk_mq_alloc_request)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_alloc_request_hctx")]
extern "C" fn blk_mq_alloc_request_hctx(q: *mut request_queue, opf: u32, flags: u32, hctx_idx: u32) -> *mut request {
let mut __ret_warn_on: i32 = unsafe { core::mem::zeroed() };
let mut __v7: bool = unsafe { core::mem::zeroed() };
let mut __ret_warn_on_1: i32 = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut tag: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut u32 = core::ptr::addr_of_mut!(tag) as *mut u32;
let mut cpu: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut u32 = core::ptr::addr_of_mut!(cpu) as *mut u32;
let mut rq: core::mem::MaybeUninit<*mut request> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut *mut request = core::ptr::addr_of_mut!(rq) as *mut *mut request;
let mut alloc_time_ns: u64 = unsafe { core::mem::zeroed() };
let mut data: core::mem::MaybeUninit<blk_mq_alloc_data> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut blk_mq_alloc_data = core::ptr::addr_of_mut!(data) as *mut blk_mq_alloc_data;
unsafe {
core::ptr::write(__slate_slot_16, {
let mut __t0: blk_mq_alloc_data = core::mem::zeroed();
__t0.q = q;
__t0.flags = flags;
__t0.shallow_depth = 0u32;
__t0.cmd_flags = opf;
__t0.rq_flags = 0u32;
__t0.nr_tags = 1u32;
__t0.cached_rqs = core::ptr::null_mut::<rq_list>();
__t0.ctx = core::ptr::null_mut::<blk_mq_ctx>();
__t0.hctx = core::ptr::null_mut::<blk_mq_hw_ctx>();
__t0
});
alloc_time_ns = 0;
let __slate_auto_bytes = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
// alloc_time includes depth and tag waits
if blk_queue_rq_alloc_time!(q) != 0 {
alloc_time_ns = blk_time_get_ns();
}
__ret_warn_on_1 = (flags & 1 == 0) as i32;
if unlikely!(__ret_warn_on_1) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("916: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 916b - ., 3; .popsection\n# {0}", const 916 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_blk_mq_alloc_request_hctx_19, sym __slate_asm_string_blk_mq_alloc_request_hctx_20, const 743 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("917: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 917b - ., 4; .popsection\n# {0}", const 917 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
// If the tag allocator sleeps we could get an allocation for a
// different hardware context.  No need to complicate the low level
// allocator for this for the rare use case of a command tied to
// a specific queue.
let __v10: i64 = (__ret_warn_on_1 != 0) as i64;
if __v10 != 0 {
__v7 = true;
} else {
__ret_warn_on = (flags & 2 == 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("918: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 918b - ., 3; .popsection\n# {0}", const 918 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_blk_mq_alloc_request_hctx_17, sym __slate_asm_string_blk_mq_alloc_request_hctx_18, const 744 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("919: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 919b - ., 4; .popsection\n# {0}", const 919 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
__v7 = __ret_warn_on != 0;
}
if __v7 {
return ERR_PTR(22i32.wrapping_neg() as i64) as *mut request;
} else {
if hctx_idx >= (*q).nr_hw_queues {
return ERR_PTR(5i32.wrapping_neg() as i64) as *mut request;
} else {
*__slate_slot_11 = blk_queue_enter(q, flags);
if *__slate_slot_11 != 0 {
return ERR_PTR(*__slate_slot_11 as i64) as *mut request;
} else {
// Check if the hardware context is actually mapped to anything.
// If not tell the caller that it should skip this queue.
*__slate_slot_11 = 18i32.wrapping_neg();
(*__slate_slot_16).hctx = *unsafe { (*q).queue_hw_ctx }.wrapping_offset(hctx_idx as isize);
if blk_mq_hw_queue_mapped((*__slate_slot_16).hctx) {
*__slate_slot_13 = cpumask_first_and(((*(*__slate_slot_16).hctx).cpumask.as_mut_ptr() as *mut cpumask) as *const cpumask, core::ptr::addr_of_mut!(__cpu_online_mask) as *const cpumask);
if !(*__slate_slot_13 >= nr_cpu_ids) {
(*__slate_slot_16).ctx = __blk_mq_get_ctx(q, *__slate_slot_13);
if !unsafe { (*q).elevator }.is_null() {
(*__slate_slot_16).rq_flags |= 16u32;
} else {
blk_mq_tag_busy((*__slate_slot_16).hctx);
}
if flags & 2 != 0 {
(*__slate_slot_16).rq_flags |= 32768u32;
}
*__slate_slot_11 = 11i32.wrapping_neg();
*__slate_slot_12 = blk_mq_get_tag(core::ptr::addr_of_mut!(*__slate_slot_16));
if *__slate_slot_12 != 4294967295 {
if (*__slate_slot_16).rq_flags & 16u32 == 0 {
blk_mq_inc_active_requests((*__slate_slot_16).hctx);
}
*__slate_slot_14 = blk_mq_rq_ctx_init(core::ptr::addr_of_mut!(*__slate_slot_16), blk_mq_tags_from_data(core::ptr::addr_of_mut!(*__slate_slot_16)), *__slate_slot_12);
blk_mq_rq_late_init(*__slate_slot_14, alloc_time_ns);
(*(*__slate_slot_14)).__data_len = 0;
(*(*__slate_slot_14)).phys_gap_bit = 0;
(*(*__slate_slot_14)).__sector = 1i32.wrapping_neg() as u64;
(*(*__slate_slot_14)).biotail = core::ptr::null_mut::<bio>();
(*(*__slate_slot_14)).bio = core::ptr::null_mut::<bio>();
return *__slate_slot_14;
}
}
}
blk_queue_exit(q);
return ERR_PTR(*__slate_slot_11 as i64) as *mut request;
}
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_alloc_request_hctx_920: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request_queue, u32, u32, u32) -> *mut request>, *mut ()>(Some(blk_mq_alloc_request_hctx)) };

fn blk_mq_finish_request(rq: *mut request) {
let q: *mut request_queue = unsafe { (*rq).q };
blk_zone_finish_request(rq);
if (unsafe { (*rq).rq_flags }) & 32u32 != 0 {
unsafe { unsafe { (*(*(*q).elevator).r#type).ops.finish_request }.unwrap()(rq) };
// For postflush request that may need to be
// completed twice, we should clear this flag
// to avoid double finish_request() on the rq.
let __v2: u32 = (unsafe { (*rq).rq_flags }) & 4294967263u32;
unsafe {
(*rq).rq_flags = __v2;
}
}
}

fn __blk_mq_free_request(rq: *mut request) {
let q: *mut request_queue = unsafe { (*rq).q };
let ctx: *mut blk_mq_ctx = unsafe { (*rq).mq_ctx };
let hctx: *mut blk_mq_hw_ctx = unsafe { (*rq).mq_hctx };
let sched_tag: i32 = unsafe { (*rq).internal_tag };
blk_crypto_free_request(rq);
blk_pm_mark_last_busy(rq);
unsafe {
(*rq).mq_hctx = core::ptr::null_mut::<blk_mq_hw_ctx>();
}
if ((unsafe { (*rq).tag }) as u32) != 4294967295 {
blk_mq_dec_active_requests(hctx);
unsafe { blk_mq_put_tag((*hctx).tags, ctx, (*rq).tag as u32) };
}
if (sched_tag as u32) != 4294967295 {
unsafe { blk_mq_put_tag((*hctx).sched_tags, ctx, sched_tag as u32) };
}
blk_mq_sched_restart(hctx);
unsafe { blk_queue_exit(q) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_free_request_922: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request)>, *mut ()>(Some(blk_mq_free_request)) };

#[unsafe(no_mangle)]
extern "C" fn blk_mq_free_plug_rqs(plug: *mut blk_plug) {
loop {
let __v0: *mut request = rq_list_pop(unsafe { core::ptr::addr_of_mut!((*plug).cached_rqs) });
let rq: *mut request = __v0;
if __v0.is_null() {
break;
}
blk_mq_free_request(rq);
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_dump_rq_flags")]
extern "C" fn blk_dump_rq_flags(rq: *mut request, msg: *mut u8) {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x016%s: dev %s: flags=%llx\n".as_ptr() as *const u8, msg, if !unsafe { (*(*rq).q).disk }.is_null() { (*(*(*rq).q).disk).disk_name.as_mut_ptr() as *mut u8 } else { c"?".as_ptr() as *mut u8 }, (*rq).cmd_flags as u64) };
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x016  sector %llu, nr/cnr %u/%u\n".as_ptr() as *const u8, blk_rq_pos(rq as *const request), blk_rq_sectors(rq as *const request), blk_rq_cur_sectors(rq as *const request)) };
let mut __v4: i32;
while false {
}
__v4 = unsafe { _printk(c"\x016  bio %p, biotail %p, len %u\n".as_ptr() as *const u8, (*rq).bio, (*rq).biotail, blk_rq_bytes(rq as *const request)) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_dump_rq_flags_923: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request, *mut u8)>, *mut ()>(Some(blk_dump_rq_flags)) };

fn blk_account_io_completion(mut req: *mut request, bytes: u32) {
if (unsafe { (*req).rq_flags }) & 256u32 != 0 {
let mut sgrp: i32 = op_stat_group(req_op(req as *const request));
loop {
preempt_count_inc!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
loop {
__this_cpu_preempt_check(c"add".as_ptr() as *const u8);
loop {
__verify_pcpu_ptr__2!();
match (const { core::mem::size_of::<u64>() }) as u64 {
1 => {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_1c5364fcd9da51fe(bytes >> 9i32) }) && (bytes >> 9i32 == 1 || bytes >> 9i32 == (1i32.wrapping_neg() as u32)) { (bytes >> 9i32) as i32 } else { 0 };
if false {
let pao_tmp__: u64 = (bytes >> 9i32) as u64;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incb %gs:({0:r})", in(reg) (((*(*(*req).part).bd_stats).sectors.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decb %gs:({0:r})", in(reg) (((*(*(*req).part).bd_stats).sectors.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u8 = ((bytes as u64) >> 9i32 & 255) as u8;
if false {
let pto_tmp__: u64 = (bytes >> 9i32) as u64;
}
unsafe {
core::arch::asm!("addb {1:l}, %gs:({0:r})", in(reg) (((*(*(*req).part).bd_stats).sectors.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
2 => {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_1c5364fcd9da51fe(bytes >> 9i32) }) && (bytes >> 9i32 == 1 || bytes >> 9i32 == (1i32.wrapping_neg() as u32)) { (bytes >> 9i32) as i32 } else { 0 };
if false {
let pao_tmp__: u64 = (bytes >> 9i32) as u64;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incw %gs:({0:r})", in(reg) (((*(*(*req).part).bd_stats).sectors.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decw %gs:({0:r})", in(reg) (((*(*(*req).part).bd_stats).sectors.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u16 = ((bytes as u64) >> 9i32 & 65535) as u16;
if false {
let pto_tmp__: u64 = (bytes >> 9i32) as u64;
}
unsafe {
core::arch::asm!("addw {1:x}, %gs:({0:r})", in(reg) (((*(*(*req).part).bd_stats).sectors.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, in(reg) pto_val__ as u16, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
4 => {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_1c5364fcd9da51fe(bytes >> 9i32) }) && (bytes >> 9i32 == 1 || bytes >> 9i32 == (1i32.wrapping_neg() as u32)) { (bytes >> 9i32) as i32 } else { 0 };
if false {
let pao_tmp__: u64 = (bytes >> 9i32) as u64;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incl %gs:({0:r})", in(reg) (((*(*(*req).part).bd_stats).sectors.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decl %gs:({0:r})", in(reg) (((*(*(*req).part).bd_stats).sectors.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u32 = ((bytes as u64) >> 9i32 & 4294967295) as u32;
if false {
let pto_tmp__: u64 = (bytes >> 9i32) as u64;
}
unsafe {
core::arch::asm!("addl {1:e}, %gs:({0:r})", in(reg) (((*(*(*req).part).bd_stats).sectors.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, in(reg) pto_val__ as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
8 => {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_1c5364fcd9da51fe(bytes >> 9i32) }) && (bytes >> 9i32 == 1 || bytes >> 9i32 == (1i32.wrapping_neg() as u32)) { (bytes >> 9i32) as i32 } else { 0 };
if false {
let pao_tmp__: u64 = (bytes >> 9i32) as u64;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incq %gs:({0:r})", in(reg) (((*(*(*req).part).bd_stats).sectors.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decq %gs:({0:r})", in(reg) (((*(*(*req).part).bd_stats).sectors.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u64 = (bytes >> 9i32) as u64;
if false {
let pto_tmp__: u64 = (bytes >> 9i32) as u64;
}
unsafe {
core::arch::asm!("addq {1:r}, %gs:({0:r})", in(reg) (((*(*(*req).part).bd_stats).sectors.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, in(reg) pto_val__ as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
if true {
break;
}
}
if bdev_is_partition(unsafe { (*req).part }) {
__this_cpu_preempt_check(c"add".as_ptr() as *const u8);
loop {
__verify_pcpu_ptr__2!();
match (const { core::mem::size_of::<u64>() }) as u64 {
1 => {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_1c5364fcd9da51fe(bytes >> 9i32) }) && (bytes >> 9i32 == 1 || bytes >> 9i32 == (1i32.wrapping_neg() as u32)) { (bytes >> 9i32) as i32 } else { 0 };
if false {
let pao_tmp__: u64 = (bytes >> 9i32) as u64;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incb %gs:({0:r})", in(reg) (((*(*(*(*(*req).part).bd_disk).part0).bd_stats).sectors.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decb %gs:({0:r})", in(reg) (((*(*(*(*(*req).part).bd_disk).part0).bd_stats).sectors.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u8 = ((bytes as u64) >> 9i32 & 255) as u8;
if false {
let pto_tmp__: u64 = (bytes >> 9i32) as u64;
}
unsafe {
core::arch::asm!("addb {1:l}, %gs:({0:r})", in(reg) (((*(*(*(*(*req).part).bd_disk).part0).bd_stats).sectors.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
2 => {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_1c5364fcd9da51fe(bytes >> 9i32) }) && (bytes >> 9i32 == 1 || bytes >> 9i32 == (1i32.wrapping_neg() as u32)) { (bytes >> 9i32) as i32 } else { 0 };
if false {
let pao_tmp__: u64 = (bytes >> 9i32) as u64;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incw %gs:({0:r})", in(reg) (((*(*(*(*(*req).part).bd_disk).part0).bd_stats).sectors.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decw %gs:({0:r})", in(reg) (((*(*(*(*(*req).part).bd_disk).part0).bd_stats).sectors.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u16 = ((bytes as u64) >> 9i32 & 65535) as u16;
if false {
let pto_tmp__: u64 = (bytes >> 9i32) as u64;
}
unsafe {
core::arch::asm!("addw {1:x}, %gs:({0:r})", in(reg) (((*(*(*(*(*req).part).bd_disk).part0).bd_stats).sectors.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, in(reg) pto_val__ as u16, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
4 => {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_1c5364fcd9da51fe(bytes >> 9i32) }) && (bytes >> 9i32 == 1 || bytes >> 9i32 == (1i32.wrapping_neg() as u32)) { (bytes >> 9i32) as i32 } else { 0 };
if false {
let pao_tmp__: u64 = (bytes >> 9i32) as u64;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incl %gs:({0:r})", in(reg) (((*(*(*(*(*req).part).bd_disk).part0).bd_stats).sectors.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decl %gs:({0:r})", in(reg) (((*(*(*(*(*req).part).bd_disk).part0).bd_stats).sectors.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u32 = ((bytes as u64) >> 9i32 & 4294967295) as u32;
if false {
let pto_tmp__: u64 = (bytes >> 9i32) as u64;
}
unsafe {
core::arch::asm!("addl {1:e}, %gs:({0:r})", in(reg) (((*(*(*(*(*req).part).bd_disk).part0).bd_stats).sectors.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, in(reg) pto_val__ as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
8 => {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_1c5364fcd9da51fe(bytes >> 9i32) }) && (bytes >> 9i32 == 1 || bytes >> 9i32 == (1i32.wrapping_neg() as u32)) { (bytes >> 9i32) as i32 } else { 0 };
if false {
let pao_tmp__: u64 = (bytes >> 9i32) as u64;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incq %gs:({0:r})", in(reg) (((*(*(*(*(*req).part).bd_disk).part0).bd_stats).sectors.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decq %gs:({0:r})", in(reg) (((*(*(*(*(*req).part).bd_disk).part0).bd_stats).sectors.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u64 = (bytes >> 9i32) as u64;
if false {
let pto_tmp__: u64 = (bytes >> 9i32) as u64;
}
unsafe {
core::arch::asm!("addq {1:r}, %gs:({0:r})", in(reg) (((*(*(*(*(*req).part).bd_disk).part0).bd_stats).sectors.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, in(reg) pto_val__ as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
if true {
break;
}
}
}
if true {
break;
}
}
loop {
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if unlikely__2!(preempt_count_dec_and_test!()) != 0 {
loop {
unsafe {
core::arch::asm!("call __SCT__preempt_schedule", options(att_syntax));
}
if true {
break;
}
}
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_924: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

fn blk_print_req_error(req: *mut request, status: u8) {
if __ratelimit__5!(unsafe { core::ptr::addr_of_mut!(_rs) }) != 0 {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x013%s error, dev %s, sector %llu op 0x%x:(%s) flags 0x%x phys_seg %u prio class %u\n".as_ptr() as *const u8, blk_status_to_str(status), if !unsafe { (*(*req).q).disk }.is_null() { (*(*(*req).q).disk).disk_name.as_mut_ptr() as *mut u8 } else { c"?".as_ptr() as *mut u8 }, blk_rq_pos(req as *const request), req_op(req as *const request) as u32, blk_op_str(req_op(req as *const request)), (*req).cmd_flags & 4294967040u32, (*req).nr_phys_segments as i32, IOPRIO_PRIO_CLASS!(req_get_ioprio(req))) };
}
}

static mut _rs: ratelimit_state = {
let mut __t0: ratelimit_state = unsafe { core::mem::zeroed() };
__t0.lock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0.interval = 5000i32;
__t0.burst = 10i32;
__t0.flags = 0u32;
__t0
};

/// Fully end IO on a request. Does not support partial completions, or
/// errors.
fn blk_complete_request(req: *mut request) {
let is_flush: bool = (unsafe { (*req).rq_flags }) & 2u32 != 0;
let total_bytes: i32 = blk_rq_bytes(req as *const request) as i32;
let mut bio: *mut bio = unsafe { (*req).bio };
trace_block_rq_complete(req, 0u8, total_bytes as u32);
if bio.is_null() {
return;
}
let __v0: bool = blk_integrity_rq(req as *const request) && (req_op(req as *const request) as u32) == 0;
if __v0 {
unsafe { blk_integrity_complete(req, total_bytes as u32) };
}
// Upper layers may call blk_crypto_evict_key() anytime after the last
// bio_endio().  Therefore, the keyslot must be released before that.
blk_crypto_rq_put_keyslot(req);
blk_account_io_completion(req, total_bytes as u32);
loop {
let next: *mut bio = unsafe { (*bio).bi_next };
// Completion has already been traced
bio_clear_flag(bio, 6u32);
if blk_req_bio_is_zone_append(req, bio) {
blk_zone_append_update_request_bio(req, bio);
}
if !is_flush {
unsafe { bio_endio(bio) };
}
bio = next;
if bio.is_null() {
break;
}
}
// Reset counters so that the request stacking driver
// can find how many bytes remain in the request
// later.
if (unsafe { (*req).end_io }) == None {
unsafe {
(*req).bio = core::ptr::null_mut::<bio>();
}
unsafe {
(*req).__data_len = 0;
}
}
}

// blk_update_request - Complete multiple bytes without completing the request
// @req:      the request being processed
// @error:    block status code
// @nr_bytes: number of bytes to complete for @req
//
// Description:
//     Ends I/O on a number of bytes attached to @req, but doesn't complete
//     the request structure even if @req doesn't have leftover.
//     If @req has leftover, sets it up for the next range of segments.
//
//     Passing the result of blk_rq_bytes() as @nr_bytes guarantees
//     %false return from this function.
//
// Note:
// 	The RQF_SPECIAL_PAYLOAD flag is ignored on purpose in this function
//      except in the consistency check at the end of this function.
//
// Return:
//     %false - this request doesn't have any more data
//     %true  - this request has more data
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_update_request_928: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request, u8, u32) -> bool>, *mut ()>(Some(blk_update_request)) };

fn blk_account_io_done(mut req: *mut request, now: u64) {
trace_block_io_done(req);
// Account IO completion.  flush_rq isn't accounted as a
// normal IO on queueing nor completion.  Accounting the
// containing request is enough.
if (unsafe { (*req).rq_flags }) & 258u32 == 256u32 {
let mut sgrp: i32 = op_stat_group(req_op(req as *const request));
loop {
preempt_count_inc!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe { update_io_ticks((*req).part, core::ptr::read_volatile(core::ptr::addr_of!(jiffies)), true) };
loop {
__this_cpu_preempt_check(c"add".as_ptr() as *const u8);
loop {
__verify_pcpu_ptr__2!();
match (const { core::mem::size_of::<u64>() }) as u64 {
1 => {
loop {
unsafe {
core::arch::asm!("incb %gs:({0:r})", in(reg) (((*(*(*req).part).bd_stats).ios.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
if true {
break;
}
}
}
2 => {
loop {
unsafe {
core::arch::asm!("incw %gs:({0:r})", in(reg) (((*(*(*req).part).bd_stats).ios.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
if true {
break;
}
}
}
4 => {
loop {
unsafe {
core::arch::asm!("incl %gs:({0:r})", in(reg) (((*(*(*req).part).bd_stats).ios.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
if true {
break;
}
}
}
8 => {
loop {
unsafe {
core::arch::asm!("incq %gs:({0:r})", in(reg) (((*(*(*req).part).bd_stats).ios.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
if true {
break;
}
}
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
if true {
break;
}
}
if bdev_is_partition(unsafe { (*req).part }) {
__this_cpu_preempt_check(c"add".as_ptr() as *const u8);
loop {
__verify_pcpu_ptr__2!();
match (const { core::mem::size_of::<u64>() }) as u64 {
1 => {
loop {
unsafe {
core::arch::asm!("incb %gs:({0:r})", in(reg) (((*(*(*(*(*req).part).bd_disk).part0).bd_stats).ios.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
if true {
break;
}
}
}
2 => {
loop {
unsafe {
core::arch::asm!("incw %gs:({0:r})", in(reg) (((*(*(*(*(*req).part).bd_disk).part0).bd_stats).ios.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
if true {
break;
}
}
}
4 => {
loop {
unsafe {
core::arch::asm!("incl %gs:({0:r})", in(reg) (((*(*(*(*(*req).part).bd_disk).part0).bd_stats).ios.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
if true {
break;
}
}
}
8 => {
loop {
unsafe {
core::arch::asm!("incq %gs:({0:r})", in(reg) (((*(*(*(*(*req).part).bd_disk).part0).bd_stats).ios.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
if true {
break;
}
}
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
if true {
break;
}
}
}
if true {
break;
}
}
loop {
__this_cpu_preempt_check(c"add".as_ptr() as *const u8);
loop {
__verify_pcpu_ptr__2!();
match (const { core::mem::size_of::<u64>() }) as u64 {
1 => {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(now.wrapping_sub((*req).start_time_ns)) }) && (now.wrapping_sub(unsafe { (*req).start_time_ns }) == 1 || now.wrapping_sub(unsafe { (*req).start_time_ns }) == (1i32.wrapping_neg() as u64)) { now.wrapping_sub(unsafe { (*req).start_time_ns }) as i32 } else { 0 };
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incb %gs:({0:r})", in(reg) (((*(*(*req).part).bd_stats).nsecs.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decb %gs:({0:r})", in(reg) (((*(*(*req).part).bd_stats).nsecs.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u8 = (now.wrapping_sub(unsafe { (*req).start_time_ns }) & 255) as u8;
unsafe {
core::arch::asm!("addb {1:l}, %gs:({0:r})", in(reg) (((*(*(*req).part).bd_stats).nsecs.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
2 => {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(now.wrapping_sub((*req).start_time_ns)) }) && (now.wrapping_sub(unsafe { (*req).start_time_ns }) == 1 || now.wrapping_sub(unsafe { (*req).start_time_ns }) == (1i32.wrapping_neg() as u64)) { now.wrapping_sub(unsafe { (*req).start_time_ns }) as i32 } else { 0 };
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incw %gs:({0:r})", in(reg) (((*(*(*req).part).bd_stats).nsecs.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decw %gs:({0:r})", in(reg) (((*(*(*req).part).bd_stats).nsecs.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u16 = (now.wrapping_sub(unsafe { (*req).start_time_ns }) & 65535) as u16;
unsafe {
core::arch::asm!("addw {1:x}, %gs:({0:r})", in(reg) (((*(*(*req).part).bd_stats).nsecs.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, in(reg) pto_val__ as u16, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
4 => {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(now.wrapping_sub((*req).start_time_ns)) }) && (now.wrapping_sub(unsafe { (*req).start_time_ns }) == 1 || now.wrapping_sub(unsafe { (*req).start_time_ns }) == (1i32.wrapping_neg() as u64)) { now.wrapping_sub(unsafe { (*req).start_time_ns }) as i32 } else { 0 };
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incl %gs:({0:r})", in(reg) (((*(*(*req).part).bd_stats).nsecs.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decl %gs:({0:r})", in(reg) (((*(*(*req).part).bd_stats).nsecs.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u32 = (now.wrapping_sub(unsafe { (*req).start_time_ns }) & 4294967295) as u32;
unsafe {
core::arch::asm!("addl {1:e}, %gs:({0:r})", in(reg) (((*(*(*req).part).bd_stats).nsecs.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, in(reg) pto_val__ as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
8 => {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(now.wrapping_sub((*req).start_time_ns)) }) && (now.wrapping_sub(unsafe { (*req).start_time_ns }) == 1 || now.wrapping_sub(unsafe { (*req).start_time_ns }) == (1i32.wrapping_neg() as u64)) { now.wrapping_sub(unsafe { (*req).start_time_ns }) as i32 } else { 0 };
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incq %gs:({0:r})", in(reg) (((*(*(*req).part).bd_stats).nsecs.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decq %gs:({0:r})", in(reg) (((*(*(*req).part).bd_stats).nsecs.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u64 = now.wrapping_sub(unsafe { (*req).start_time_ns });
unsafe {
core::arch::asm!("addq {1:r}, %gs:({0:r})", in(reg) (((*(*(*req).part).bd_stats).nsecs.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, in(reg) pto_val__ as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
if true {
break;
}
}
if bdev_is_partition(unsafe { (*req).part }) {
__this_cpu_preempt_check(c"add".as_ptr() as *const u8);
loop {
__verify_pcpu_ptr__2!();
match (const { core::mem::size_of::<u64>() }) as u64 {
1 => {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(now.wrapping_sub((*req).start_time_ns)) }) && (now.wrapping_sub(unsafe { (*req).start_time_ns }) == 1 || now.wrapping_sub(unsafe { (*req).start_time_ns }) == (1i32.wrapping_neg() as u64)) { now.wrapping_sub(unsafe { (*req).start_time_ns }) as i32 } else { 0 };
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incb %gs:({0:r})", in(reg) (((*(*(*(*(*req).part).bd_disk).part0).bd_stats).nsecs.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decb %gs:({0:r})", in(reg) (((*(*(*(*(*req).part).bd_disk).part0).bd_stats).nsecs.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u8 = (now.wrapping_sub(unsafe { (*req).start_time_ns }) & 255) as u8;
unsafe {
core::arch::asm!("addb {1:l}, %gs:({0:r})", in(reg) (((*(*(*(*(*req).part).bd_disk).part0).bd_stats).nsecs.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
2 => {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(now.wrapping_sub((*req).start_time_ns)) }) && (now.wrapping_sub(unsafe { (*req).start_time_ns }) == 1 || now.wrapping_sub(unsafe { (*req).start_time_ns }) == (1i32.wrapping_neg() as u64)) { now.wrapping_sub(unsafe { (*req).start_time_ns }) as i32 } else { 0 };
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incw %gs:({0:r})", in(reg) (((*(*(*(*(*req).part).bd_disk).part0).bd_stats).nsecs.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decw %gs:({0:r})", in(reg) (((*(*(*(*(*req).part).bd_disk).part0).bd_stats).nsecs.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u16 = (now.wrapping_sub(unsafe { (*req).start_time_ns }) & 65535) as u16;
unsafe {
core::arch::asm!("addw {1:x}, %gs:({0:r})", in(reg) (((*(*(*(*(*req).part).bd_disk).part0).bd_stats).nsecs.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, in(reg) pto_val__ as u16, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
4 => {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(now.wrapping_sub((*req).start_time_ns)) }) && (now.wrapping_sub(unsafe { (*req).start_time_ns }) == 1 || now.wrapping_sub(unsafe { (*req).start_time_ns }) == (1i32.wrapping_neg() as u64)) { now.wrapping_sub(unsafe { (*req).start_time_ns }) as i32 } else { 0 };
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incl %gs:({0:r})", in(reg) (((*(*(*(*(*req).part).bd_disk).part0).bd_stats).nsecs.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decl %gs:({0:r})", in(reg) (((*(*(*(*(*req).part).bd_disk).part0).bd_stats).nsecs.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u32 = (now.wrapping_sub(unsafe { (*req).start_time_ns }) & 4294967295) as u32;
unsafe {
core::arch::asm!("addl {1:e}, %gs:({0:r})", in(reg) (((*(*(*(*(*req).part).bd_disk).part0).bd_stats).nsecs.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, in(reg) pto_val__ as u32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
8 => {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(now.wrapping_sub((*req).start_time_ns)) }) && (now.wrapping_sub(unsafe { (*req).start_time_ns }) == 1 || now.wrapping_sub(unsafe { (*req).start_time_ns }) == (1i32.wrapping_neg() as u64)) { now.wrapping_sub(unsafe { (*req).start_time_ns }) as i32 } else { 0 };
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incq %gs:({0:r})", in(reg) (((*(*(*(*(*req).part).bd_disk).part0).bd_stats).nsecs.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decq %gs:({0:r})", in(reg) (((*(*(*(*(*req).part).bd_disk).part0).bd_stats).nsecs.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u64 = now.wrapping_sub(unsafe { (*req).start_time_ns });
unsafe {
core::arch::asm!("addq {1:r}, %gs:({0:r})", in(reg) (((*(*(*(*(*req).part).bd_disk).part0).bd_stats).nsecs.as_mut_ptr() as *mut u64).wrapping_offset(sgrp as isize) as u64) as *mut u64, in(reg) pto_val__ as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
}
}
if true {
break;
}
}
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
if true {
break;
}
}
}
if true {
break;
}
}
bdev_dec_in_flight(unsafe { (*req).part }, req_op(req as *const request));
loop {
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if unlikely__2!(preempt_count_dec_and_test!()) != 0 {
loop {
unsafe {
core::arch::asm!("call __SCT__preempt_schedule", options(att_syntax));
}
if true {
break;
}
}
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_929: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

fn blk_account_io_start(req: *mut request) {
trace_block_io_start(req);
if blk_queue_io_stat!(unsafe { (*req).q }) == 0 {
return;
}
let __v0: bool = blk_rq_is_passthrough(req) && !blk_rq_passthrough_stats(req, unsafe { (*req).q });
if __v0 {
return;
}
unsafe {
(*req).rq_flags = (*req).rq_flags | 256u32;
}
unsafe {
(*req).start_time_ns = blk_time_get_ns();
}
// All non-passthrough requests are created from a bio with one
// exception: when a flush command that is part of a flush sequence
// generated by the state machine in blk-flush.c is cloned onto the
// lower device by dm-multipath we can get here without a bio.
if !unsafe { (*req).bio }.is_null() {
unsafe {
(*req).part = (*(*req).bio).bi_bdev;
}
} else {
unsafe {
(*req).part = (*(*(*req).q).disk).part0;
}
}
loop {
preempt_count_inc!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe { update_io_ticks((*req).part, core::ptr::read_volatile(core::ptr::addr_of!(jiffies)), false) };
bdev_inc_in_flight(unsafe { (*req).part }, req_op(req as *const request));
loop {
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if unlikely__2!(preempt_count_dec_and_test!()) != 0 {
loop {
unsafe {
core::arch::asm!("call __SCT__preempt_schedule", options(att_syntax));
}
if true {
break;
}
}
}
if true {
break;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_930: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

fn __blk_mq_end_request_acct(rq: *mut request, now: u64) {
if (unsafe { (*rq).rq_flags }) & 2048u32 != 0 {
unsafe { blk_stat_add(rq, now) };
}
blk_mq_sched_completed_request(rq, now);
blk_account_io_done(rq, now);
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.__blk_mq_end_request")]
extern "C" fn __blk_mq_end_request(rq: *mut request, error: u8) {
if blk_mq_need_time_stamp(rq) {
__blk_mq_end_request_acct(rq, blk_time_get_ns());
}
blk_mq_finish_request(rq);
if (unsafe { (*rq).end_io }) != None {
rq_qos_done(unsafe { (*rq).q }, rq);
if ((unsafe { unsafe { (*rq).end_io }.unwrap()(rq, error, core::ptr::null::<io_comp_batch>()) }) as u32) == 1 {
blk_mq_free_request(rq);
}
} else {
blk_mq_free_request(rq);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___blk_mq_end_request_931: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request, u8)>, *mut ()>(Some(__blk_mq_end_request)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_end_request")]
extern "C" fn blk_mq_end_request(rq: *mut request, error: u8) {
if blk_update_request(rq, error, blk_rq_bytes(rq as *const request)) {
loop {
unsafe {
core::arch::asm!("932: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 932b - ., 3; .popsection\n# {0}", const 932 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_blk_mq_end_request_0, sym __slate_asm_string_blk_mq_end_request_1, const 1161 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe { core::hint::unreachable_unchecked() };
if true {
break;
}
}
}
__blk_mq_end_request(rq, error);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_end_request_933: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request, u8)>, *mut ()>(Some(blk_mq_end_request)) };

fn blk_mq_flush_tag_batch(hctx: *mut blk_mq_hw_ctx, tag_array: *mut i32, nr_tags: i32) {
let q: *mut request_queue = unsafe { (*hctx).queue };
blk_mq_sub_active_requests(hctx, nr_tags);
unsafe { blk_mq_put_tags((*hctx).tags, tag_array, nr_tags) };
percpu_ref_put_many(unsafe { core::ptr::addr_of_mut!((*q).q_usage_counter) }, nr_tags as u64);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_end_request_batch_935: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut io_comp_batch)>, *mut ()>(Some(blk_mq_end_request_batch)) };

fn blk_complete_reqs(list: *mut llist_head) {
let entry: *mut llist_node = unsafe { llist_reverse_order(llist_del_all(list)) };
let mut rq: *mut request = unsafe { core::mem::zeroed() };
let mut next: *mut request = unsafe { core::mem::zeroed() };
rq = llist_entry!(entry);
loop {
let __v1: bool = if member_address_is_nonnull__2!(rq) {
next = llist_entry!(unsafe { (*rq).__slate_anon_24.ipi_list.next });
true
} else {
false
};
if !__v1 {
break;
}
unsafe { unsafe { (*(*(*rq).q).mq_ops).complete }.unwrap()(rq) };
rq = next;
}
}

#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_done_softirq")]
extern "C" fn blk_done_softirq() {
let mut __v0: *mut llist_head;
__verify_pcpu_ptr__9!();
let mut __v2: *mut llist_head;
let mut tcp_ptr__: u64 = 0;
let mut __v3: u64;
let mut pfo_val__: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t0, sym this_cpu_off, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
__v3 = pfo_val__;
tcp_ptr__ = __v3;
let __v5: u64 = tcp_ptr__.wrapping_add((unsafe { core::ptr::addr_of_mut!(blk_cpu_done) }) as u64);
tcp_ptr__ = __v5;
__v2 = tcp_ptr__ as *mut llist_head;
__v0 = __v2;
blk_complete_reqs(__v0);
}

#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_softirq_cpu_dead")]
extern "C" fn blk_softirq_cpu_dead(cpu: u32) -> i32 {
blk_complete_reqs(per_cpu_ptr__2!(unsafe { core::ptr::addr_of_mut!(blk_cpu_done) }, cpu));
0
}

#[unsafe(link_section = ".text.slate_distinct.blk_mq.__blk_mq_complete_request_remote")]
extern "C" fn __blk_mq_complete_request_remote(data: *mut ()) {
unsafe { __raise_softirq_irqoff(4u32) };
}

fn blk_mq_complete_need_ipi(rq: *mut request) -> bool {
let mut cpu: i32 = 0;
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
__verify_pcpu_ptr__3!();
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v2 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v2;
}
2 => {
let mut __v4: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v4 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v4;
}
4 => {
let mut __v5: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
8 => {
let mut __v6: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v6 = pfo_val__ as i32;
pscr_ret__ = __v6;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v0 = pscr_ret__;
cpu = __v0;
let __v7: bool = IS_ENABLED__2!() == 0 || test_bit!(2i32, unsafe { core::ptr::addr_of_mut!((*(*rq).q).queue_flags) }) == 0;
if __v7 {
return false;
}
// With force threaded interrupts enabled, raising softirq from an SMP
// function call will always result in waking the ksoftirqd thread.
// This is probably worse than completing the request on a different
// cache domain.
if force_irqthreads!() != 0 {
return false;
}
// same CPU or cache domain and capacity?  Complete locally
let __v11: bool = if (cpu as u32) == unsafe { (*(*rq).mq_ctx).cpu } {
true
} else {
let __v12: bool = test_bit!(5i32, unsafe { core::ptr::addr_of_mut!((*(*rq).q).queue_flags) }) == 0 && unsafe { cpus_share_cache(cpu, (*(*rq).mq_ctx).cpu as i32) };
let __v14: bool = __v12 && unsafe { cpus_equal_capacity(cpu, (*(*rq).mq_ctx).cpu as i32) };
__v14
};
if __v11 {
return false;
}
// don't try to IPI to an offline CPU
cpu_online(unsafe { (*(*rq).mq_ctx).cpu })
}

fn blk_mq_complete_send_ipi(rq: *mut request) {
let cpu: u32 = unsafe { (*(*rq).mq_ctx).cpu };
if llist_add(unsafe { core::ptr::addr_of_mut!((*rq).__slate_anon_24.ipi_list) }, per_cpu_ptr__2!(unsafe { core::ptr::addr_of_mut!(blk_cpu_done) }, cpu)) {
unsafe { smp_call_function_single_async(cpu as i32, per_cpu_ptr__10!(core::ptr::addr_of_mut!(blk_cpu_csd.0), cpu)) };
}
}

fn blk_mq_raise_softirq(rq: *mut request) {
loop {
preempt_count_inc!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
let mut __v1: *mut llist_head;
__verify_pcpu_ptr__9!();
let mut __v3: *mut llist_head;
let mut tcp_ptr__: u64 = 0;
let mut __v4: u64;
let mut pfo_val__: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t0, sym this_cpu_off, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
__v4 = pfo_val__;
tcp_ptr__ = __v4;
let __v6: u64 = tcp_ptr__.wrapping_add((unsafe { core::ptr::addr_of_mut!(blk_cpu_done) }) as u64);
tcp_ptr__ = __v6;
__v3 = tcp_ptr__ as *mut llist_head;
__v1 = __v3;
let list: *mut llist_head = __v1;
if llist_add(unsafe { core::ptr::addr_of_mut!((*rq).__slate_anon_24.ipi_list) }, list) {
unsafe { raise_softirq(4u32) };
}
loop {
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if unlikely__2!(preempt_count_dec_and_test!()) != 0 {
loop {
unsafe {
core::arch::asm!("call __SCT__preempt_schedule", options(att_syntax));
}
if true {
break;
}
}
}
if true {
break;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_936: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_complete_request_remote_938: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request) -> bool>, *mut ()>(Some(blk_mq_complete_request_remote)) };

/// blk_mq_complete_request - end I/O on a request
/// @rq:		the request being processed
///
/// Description:
/// 	Complete a request by scheduling the ->complete_rq operation.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_complete_request")]
extern "C" fn blk_mq_complete_request(rq: *mut request) {
if !blk_mq_complete_request_remote(rq) {
unsafe { unsafe { (*(*(*rq).q).mq_ops).complete }.unwrap()(rq) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_complete_request_939: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request)>, *mut ()>(Some(blk_mq_complete_request)) };

// blk_mq_start_request - Start processing a request
// @rq: Pointer to request to be started
//
// Function used by device drivers to notify the block layer that a request
// is going to be processed now, so blk layer can do proper initializations
// such as starting the timeout timer.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_start_request_944: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request)>, *mut ()>(Some(blk_mq_start_request)) };

/// Allow 2x BLK_MAX_REQUEST_COUNT requests on plug queue for multiple
/// queues. This is important for md arrays to benefit from merging
/// requests.
fn blk_plug_max_rq_count(plug: *mut blk_plug) -> u16 {
if unsafe { (*plug).multiple_queues } {
return 64u16;
}
32
}

fn blk_add_rq_to_plug(plug: *mut blk_plug, rq: *mut request) {
let mut last: *mut request = rq_list_peek(unsafe { core::ptr::addr_of_mut!((*plug).mq_list) });
if (unsafe { (*plug).rq_count }) == 0 {
trace_block_plug(unsafe { (*rq).q });
} else {
let __v0: bool = ((unsafe { (*plug).rq_count }) as i32) >= (blk_plug_max_rq_count(plug) as i32) || blk_queue_nomerges!(unsafe { (*rq).q }) == 0 && blk_rq_bytes(last as *const request) >= 131072u32;
if __v0 {
blk_mq_flush_plug_list(plug, false);
last = core::ptr::null_mut::<request>();
trace_block_plug(unsafe { (*rq).q });
}
}
if !unsafe { (*plug).multiple_queues } && !last.is_null() && (unsafe { (*last).q }) != unsafe { (*rq).q } {
unsafe {
(*plug).multiple_queues = true;
}
}
// Any request allocated from sched tags can't be issued to
// ->queue_rqs() directly
if !unsafe { (*plug).has_elevator } && (unsafe { (*rq).rq_flags }) & 16u32 != 0 {
unsafe {
(*plug).has_elevator = true;
}
}
rq_list_add_tail(unsafe { core::ptr::addr_of_mut!((*plug).mq_list) }, rq);
unsafe {
(*plug).rq_count = ((*plug).rq_count as i32).wrapping_add(1) as u16;
}
}

/// blk_execute_rq_nowait - insert a request to I/O scheduler for execution
/// @rq:		request to insert
/// @at_head:    insert request at head or tail of queue
///
/// Description:
///    Insert a fully prepared request at the back of the I/O scheduler queue
///    for execution.  Don't wait for completion.
///
/// Note:
///    This function will invoke @done directly if the queue is dead.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_execute_rq_nowait")]
extern "C" fn blk_execute_rq_nowait(rq: *mut request, at_head: bool) {
let hctx: *mut blk_mq_hw_ctx = unsafe { (*rq).mq_hctx };
let mut __v0: i64;
{
let __ret_warn_on: i32 = (irqs_disabled!() != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("945: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 945b - ., 3; .popsection\n# {0}", const 945 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_blk_execute_rq_nowait_11, sym __slate_asm_string_blk_execute_rq_nowait_12, const 1432 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("946: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 946b - ., 4; .popsection\n# {0}", const 946 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
}
let mut __v15: i64;
let __ret_warn_on: i32 = !(!(!blk_rq_is_passthrough(rq))) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("947: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 947b - ., 3; .popsection\n# {0}", const 947 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_blk_execute_rq_nowait_16, sym __slate_asm_string_blk_execute_rq_nowait_17, const 1433 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("948: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 948b - ., 4; .popsection\n# {0}", const 948 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v15 = (__ret_warn_on != 0) as i64;
blk_account_io_start(rq);
if !unsafe { (*get_current()).plug }.is_null() && !at_head {
blk_add_rq_to_plug(unsafe { (*get_current()).plug }, rq);
return;
}
blk_mq_insert_request(rq, if at_head { 1u32 } else { 0u32 });
blk_mq_run_hw_queue(hctx, (unsafe { (*hctx).flags }) & 16 != 0);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_execute_rq_nowait_949: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request, bool)>, *mut ()>(Some(blk_execute_rq_nowait)) };

#[repr(C)]
#[derive(Clone, Copy)]
struct blk_rq_wait {
    done: completion,
    ret: u8,
}


#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_end_sync_rq")]
extern "C" fn blk_end_sync_rq(rq: *mut request, ret: u8, iob: *const io_comp_batch) -> u32 {
let wait: *mut blk_rq_wait = (unsafe { (*rq).end_io_data }) as *mut blk_rq_wait;
unsafe {
(*wait).ret = ret;
}
unsafe { complete(core::ptr::addr_of_mut!((*wait).done)) };
0
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_rq_is_poll")]
extern "C" fn blk_rq_is_poll(rq: *mut request) -> bool {
if unsafe { (*rq).mq_hctx }.is_null() {
return false;
}
if ((unsafe { (*(*rq).mq_hctx).r#type }) as i32) != 2 {
return false;
}
true
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_rq_is_poll_950: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request) -> bool>, *mut ()>(Some(blk_rq_is_poll)) };

fn blk_rq_poll_completion(rq: *mut request, wait: *mut completion) {
loop {
blk_hctx_poll(unsafe { (*rq).q }, unsafe { (*rq).mq_hctx }, core::ptr::null_mut::<io_comp_batch>(), 1u32);
cond_resched__4!();
if !(!unsafe { completion_done(wait) }) {
break;
}
}
}

/// blk_execute_rq - insert a request into queue for execution
/// @rq:		request to insert
/// @at_head:    insert request at head or tail of queue
///
/// Description:
///    Insert a fully prepared request at the back of the I/O scheduler queue
///    for execution and wait for completion.
/// Return: The blk_status_t result provided to blk_mq_end_request().
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_execute_rq")]
extern "C" fn blk_execute_rq(rq: *mut request, at_head: bool) -> u8 {
let hctx: *mut blk_mq_hw_ctx = unsafe { (*rq).mq_hctx };
let mut wait: blk_rq_wait = unsafe { core::mem::zeroed() };
wait = {
let mut __t0: blk_rq_wait = unsafe { core::mem::zeroed() };
__t0.done = COMPLETION_INITIALIZER_ONSTACK!(wait.done);
__t0
};
let mut __v1: i64;
{
let __ret_warn_on: i32 = (irqs_disabled!() != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("951: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 951b - ., 3; .popsection\n# {0}", const 951 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_blk_execute_rq_12, sym __slate_asm_string_blk_execute_rq_13, const 1497 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("952: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 952b - ., 4; .popsection\n# {0}", const 952 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
}
let mut __v16: i64;
let __ret_warn_on: i32 = !(!(!blk_rq_is_passthrough(rq))) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("953: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 953b - ., 3; .popsection\n# {0}", const 953 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_blk_execute_rq_17, sym __slate_asm_string_blk_execute_rq_18, const 1498 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("954: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 954b - ., 4; .popsection\n# {0}", const 954 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v16 = (__ret_warn_on != 0) as i64;
unsafe {
(*rq).end_io_data = core::ptr::addr_of_mut!(wait) as *mut ();
}
unsafe {
(*rq).end_io = Some(blk_end_sync_rq);
}
blk_account_io_start(rq);
blk_mq_insert_request(rq, if at_head { 1u32 } else { 0u32 });
blk_mq_run_hw_queue(hctx, false);
if blk_rq_is_poll(rq) {
blk_rq_poll_completion(rq, core::ptr::addr_of_mut!(wait.done));
} else {
blk_wait_io(core::ptr::addr_of_mut!(wait.done));
}
wait.ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_execute_rq_955: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request, bool) -> u8>, *mut ()>(Some(blk_execute_rq)) };

fn __blk_mq_requeue_request(rq: *mut request) {
let q: *mut request_queue = unsafe { (*rq).q };
blk_mq_put_driver_tag(rq);
trace_block_rq_requeue(rq);
rq_qos_requeue(q, rq);
if blk_mq_request_started(rq) != 0 {
loop {
while false {
}
__WRITE_ONCE__4!((*rq).state, 0i32);
if true {
break;
}
}
let __v5: u32 = (unsafe { (*rq).rq_flags }) & 4294950911u32;
unsafe {
(*rq).rq_flags = __v5;
}
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_requeue_request")]
extern "C" fn blk_mq_requeue_request(rq: *mut request, kick_requeue_list: bool) {
let q: *mut request_queue = unsafe { (*rq).q };
let mut flags: u64 = 0;
__blk_mq_requeue_request(rq);
// this request will be re-inserted to io scheduler queue
blk_mq_sched_requeue_request(rq);
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*q).requeue_lock) }, flags);
list_add_tail(unsafe { core::ptr::addr_of_mut!((*rq).__slate_anon_12.queuelist) }, unsafe { core::ptr::addr_of_mut!((*q).requeue_list) });
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*q).requeue_lock) }, flags);
if kick_requeue_list {
blk_mq_kick_requeue_list(q);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_requeue_request_957: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request, bool)>, *mut ()>(Some(blk_mq_requeue_request)) };

#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_requeue_work")]
extern "C" fn blk_mq_requeue_work(work: *mut work_struct) {
let mut q: *mut request_queue = unsafe { core::mem::zeroed() };
let mut __v0: *mut request_queue;
__v0 = ((unsafe { (work as *const u8).wrapping_offset(-(offsetof__31!(request_queue) as isize)) }) as *mut ()) as *mut request_queue;
q = __v0;
let mut rq_list: list_head = unsafe { core::mem::zeroed() };
rq_list = list_head { next: core::ptr::addr_of_mut!(rq_list), prev: core::ptr::addr_of_mut!(rq_list) };
let mut flush_list: list_head = unsafe { core::mem::zeroed() };
flush_list = list_head { next: core::ptr::addr_of_mut!(flush_list), prev: core::ptr::addr_of_mut!(flush_list) };
let mut rq: *mut request = unsafe { core::mem::zeroed() };
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*q).requeue_lock) });
list_splice_init(unsafe { core::ptr::addr_of_mut!((*q).requeue_list) }, core::ptr::addr_of_mut!(rq_list));
list_splice_init(unsafe { core::ptr::addr_of_mut!((*q).flush_list) }, core::ptr::addr_of_mut!(flush_list));
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*q).requeue_lock) });
while list_empty(core::ptr::addr_of_mut!(rq_list) as *const list_head) == 0 {
rq = list_entry__2!(rq_list.next, request);
list_del_init(unsafe { core::ptr::addr_of_mut!((*rq).__slate_anon_12.queuelist) });
// If RQF_DONTPREP is set, the request has been started by the
// driver already and might have driver-specific data allocated
// already.  Insert it into the hctx dispatch list to avoid
// block layer merges for the request.
if (unsafe { (*rq).rq_flags }) & 8u32 != 0 {
blk_mq_request_bypass_insert(rq, 0u32);
} else {
blk_mq_insert_request(rq, 1u32);
}
}
while list_empty(core::ptr::addr_of_mut!(flush_list) as *const list_head) == 0 {
rq = list_entry__2!(flush_list.next, request);
list_del_init(unsafe { core::ptr::addr_of_mut!((*rq).__slate_anon_12.queuelist) });
blk_mq_insert_request(rq, 0u32);
}
blk_mq_run_hw_queues(q, false);
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_kick_requeue_list")]
extern "C" fn blk_mq_kick_requeue_list(q: *mut request_queue) {
unsafe { kblockd_mod_delayed_work_on(64i32, core::ptr::addr_of_mut!((*q).requeue_work), 0u64) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_kick_requeue_list_958: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request_queue)>, *mut ()>(Some(blk_mq_kick_requeue_list)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_delay_kick_requeue_list")]
extern "C" fn blk_mq_delay_kick_requeue_list(q: *mut request_queue, msecs: u64) {
unsafe { kblockd_mod_delayed_work_on(64i32, core::ptr::addr_of_mut!((*q).requeue_work), msecs_to_jiffies(msecs as u32)) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_delay_kick_requeue_list_959: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request_queue, u64)>, *mut ()>(Some(blk_mq_delay_kick_requeue_list)) };

fn blk_is_flush_data_rq(rq: *mut request) -> bool {
(unsafe { (*rq).rq_flags }) & 2u32 != 0 && !unsafe { is_flush_rq(rq) }
}

#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_rq_inflight")]
extern "C" fn blk_mq_rq_inflight(rq: *mut request, r#priv: *mut ()) -> bool {
// If we find a request that isn't idle we know the queue is busy
// as it's checked in the iter.
// Return false to stop the iteration.
//
// In case of queue quiesce, if one flush data request is completed,
// don't count it as inflight given the flush sequence is suspended,
// and the original flush data request is invisible to driver, just
// like other pending requests because of quiesce
let __v0: bool = if blk_mq_request_started(rq) != 0 {
let __v1: bool = blk_queue_quiesced!(unsafe { (*rq).q }) != 0 && blk_is_flush_data_rq(rq);
let __v3: bool = __v1 && blk_mq_request_completed(rq) != 0;
!__v3
} else {
false
};
if __v0 {
unsafe {
*(r#priv as *mut bool) = true;
}
return false;
}
true
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_queue_inflight")]
extern "C" fn blk_mq_queue_inflight(q: *mut request_queue) -> bool {
let mut busy: bool = false;
unsafe { blk_mq_queue_tag_busy_iter(q, Some(blk_mq_rq_inflight), core::ptr::addr_of_mut!(busy) as *mut ()) };
busy
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_queue_inflight_960: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request_queue) -> bool>, *mut ()>(Some(blk_mq_queue_inflight)) };

fn blk_mq_rq_timed_out(req: *mut request) {
unsafe {
(*req).rq_flags = (*req).rq_flags | 16384u32;
}
if (unsafe { (*(*(*req).q).mq_ops).timeout }) != None {
let ret: u32 = unsafe { unsafe { (*(*(*req).q).mq_ops).timeout }.unwrap()(req) };
if (ret as u32) == 0 {
return;
}
let mut __v3: i64;
let __ret_warn_on: i32 = ((ret as u32) != 1) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("961: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 961b - ., 3; .popsection\n# {0}", const 961 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_blk_mq_rq_timed_out_4, sym __slate_asm_string_blk_mq_rq_timed_out_5, const 1648 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("962: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 962b - ., 4; .popsection\n# {0}", const 962 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v3 = (__ret_warn_on != 0) as i64;
}
unsafe { blk_add_timer(req) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct blk_expired_data {
    has_timedout_rq: bool,
    next: u64,
    timeout_start: u64,
}


fn blk_mq_req_expired(rq: *mut request, expired: *mut blk_expired_data) -> bool {
let mut deadline: u64 = 0;
if (blk_mq_rq_state(rq) as u32) != 1 {
return false;
}
if (unsafe { (*rq).rq_flags }) & 16384u32 != 0 {
return false;
}
let mut __v0: u64;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_963() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*rq).deadline) as *const u64) };
deadline = __v0;
if time_after_eq__2!(unsafe { (*expired).timeout_start }, deadline) {
return true;
}
if (unsafe { (*expired).next }) == 0 {
unsafe {
(*expired).next = deadline;
}
} else {
if time_after!(unsafe { (*expired).next }, deadline) {
unsafe {
(*expired).next = deadline;
}
}
}
false
}

#[unsafe(no_mangle)]
extern "C" fn blk_mq_put_rq_ref(rq: *mut request) {
if unsafe { is_flush_rq(rq) } {
if ((unsafe { unsafe { (*rq).end_io }.unwrap()(rq, 0u8, core::ptr::null::<io_comp_batch>()) }) as u32) == 1 {
blk_mq_free_request(rq);
}
} else {
if req_ref_put_and_test(rq) {
__blk_mq_free_request(rq);
}
}
}

#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_check_expired")]
extern "C" fn blk_mq_check_expired(rq: *mut request, r#priv: *mut ()) -> bool {
let expired: *mut blk_expired_data = r#priv as *mut blk_expired_data;
// blk_mq_queue_tag_busy_iter() has locked the request, so it cannot
// be reallocated underneath the timeout handler's processing, then
// the expire check is reliable. If the request is not expired, then
// it was completed and reallocated as a new request after returning
// from blk_mq_check_expired().
if blk_mq_req_expired(rq, expired) {
unsafe {
(*expired).has_timedout_rq = true;
}
return false;
}
true
}

#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_handle_expired")]
extern "C" fn blk_mq_handle_expired(rq: *mut request, r#priv: *mut ()) -> bool {
let expired: *mut blk_expired_data = r#priv as *mut blk_expired_data;
if blk_mq_req_expired(rq, expired) {
blk_mq_rq_timed_out(rq);
}
true
}

#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_timeout_work")]
extern "C" fn blk_mq_timeout_work(work: *mut work_struct) {
let q: *mut request_queue = container_of__36!(work, request_queue);
let mut expired: blk_expired_data = {
let mut __t0: blk_expired_data = unsafe { core::mem::zeroed() };
__t0.timeout_start = unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) };
__t0
};
let mut hctx: *mut blk_mq_hw_ctx = unsafe { core::mem::zeroed() };
let mut i: u64 = 0;
// A deadlock might occur if a request is stuck requiring a
// timeout at the same time a queue freeze is waiting
// completion, since the timeout code would not be able to
// acquire the queue reference here.
//
// That's why we don't use blk_queue_enter here; instead, we use
// percpu_ref_tryget directly, because we need to be able to
// obtain a reference even in the short window between the queue
// starting to freeze, by dropping the first reference in
// blk_freeze_queue_start, and the moment the last request is
// consumed, marked by the instant q_usage_counter reaches
// zero.
if !percpu_ref_tryget(unsafe { core::ptr::addr_of_mut!((*q).q_usage_counter) }) {
return;
}
// check if there is any timed-out request
unsafe { blk_mq_queue_tag_busy_iter(q, Some(blk_mq_check_expired), core::ptr::addr_of_mut!(expired) as *mut ()) };
if expired.has_timedout_rq {
// Before walking tags, we must ensure any submit started
// before the current time has finished. Since the submit
// uses srcu or rcu, wait for a synchronization point to
// ensure all running submits have finished
blk_mq_wait_quiesce_done(unsafe { (*q).tag_set });
expired.next = 0;
unsafe { blk_mq_queue_tag_busy_iter(q, Some(blk_mq_handle_expired), core::ptr::addr_of_mut!(expired) as *mut ()) };
}
if expired.next != 0 {
unsafe { mod_timer(core::ptr::addr_of_mut!((*q).timeout), expired.next) };
} else {
// Request timeouts are handled as a forward rolling timer. If
// we end up here it means that no requests are pending and
// also that no request has been pending for a while. Mark
// each hctx as idle.
i = 0;
loop {
let __v1: bool = if i < ((unsafe { (*q).nr_hw_queues }) as u64) {
let mut __v2: i32;
hctx = queue_hctx(q, i as i32);
__v2 = 1;
__v2 != 0
} else {
false
};
if !__v1 {
break;
}
// the hctx may be unmapped, so check it here
if blk_mq_hw_queue_mapped(hctx) {
blk_mq_tag_idle(hctx);
}
i = i.wrapping_add(1);
}
}
unsafe { blk_queue_exit(q) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct flush_busy_ctx_data {
    hctx: *mut blk_mq_hw_ctx,
    list: *mut list_head,
}


#[unsafe(link_section = ".text.slate_distinct.blk_mq.flush_busy_ctx")]
extern "C" fn flush_busy_ctx(sb: *mut sbitmap, bitnr: u32, data: *mut ()) -> bool {
let hctx: *mut blk_mq_hw_ctx = unsafe { (*(data as *mut flush_busy_ctx_data)).hctx };
let ctx: *mut blk_mq_ctx = unsafe { *unsafe { (*hctx).ctxs }.wrapping_offset(bitnr as isize) };
let r#type: u32 = (unsafe { (*hctx).r#type }) as u32;
spin_lock(unsafe { core::ptr::addr_of_mut!((*ctx).__slate_anon_0.lock) });
list_splice_tail_init(unsafe { (*ctx).__slate_anon_0.rq_lists.as_mut_ptr() as *mut list_head }.wrapping_offset((r#type as u32) as isize), unsafe { (*(data as *mut flush_busy_ctx_data)).list });
sbitmap_clear_bit(sb, bitnr);
spin_unlock(unsafe { core::ptr::addr_of_mut!((*ctx).__slate_anon_0.lock) });
true
}

/// Process software queues that have been marked busy, splicing them
/// to the for-dispatch
#[unsafe(no_mangle)]
extern "C" fn blk_mq_flush_busy_ctxs(hctx: *mut blk_mq_hw_ctx, list: *mut list_head) {
let mut data: flush_busy_ctx_data = flush_busy_ctx_data { hctx: hctx, list: list };
sbitmap_for_each_set(unsafe { core::ptr::addr_of_mut!((*hctx).ctx_map) }, Some(flush_busy_ctx), core::ptr::addr_of_mut!(data) as *mut ());
}

#[repr(C)]
#[derive(Clone, Copy)]
struct dispatch_rq_data {
    hctx: *mut blk_mq_hw_ctx,
    rq: *mut request,
}


#[unsafe(link_section = ".text.slate_distinct.blk_mq.dispatch_rq_from_ctx")]
extern "C" fn dispatch_rq_from_ctx(sb: *mut sbitmap, bitnr: u32, data: *mut ()) -> bool {
let hctx: *mut blk_mq_hw_ctx = unsafe { (*(data as *mut dispatch_rq_data)).hctx };
let ctx: *mut blk_mq_ctx = unsafe { *unsafe { (*hctx).ctxs }.wrapping_offset(bitnr as isize) };
let r#type: u32 = (unsafe { (*hctx).r#type }) as u32;
spin_lock(unsafe { core::ptr::addr_of_mut!((*ctx).__slate_anon_0.lock) });
if list_empty(unsafe { (*ctx).__slate_anon_0.rq_lists.as_mut_ptr() as *mut list_head }.wrapping_offset((r#type as u32) as isize) as *const list_head) == 0 {
unsafe {
(*(data as *mut dispatch_rq_data)).rq = list_entry_rq!(unsafe { (*((*ctx).__slate_anon_0.rq_lists.as_mut_ptr() as *mut list_head).wrapping_offset((r#type as u32) as isize)).next });
}
list_del_init(unsafe { core::ptr::addr_of_mut!((*(*(data as *mut dispatch_rq_data)).rq).__slate_anon_12.queuelist) });
if list_empty(unsafe { (*ctx).__slate_anon_0.rq_lists.as_mut_ptr() as *mut list_head }.wrapping_offset((r#type as u32) as isize) as *const list_head) != 0 {
sbitmap_clear_bit(sb, bitnr);
}
}
spin_unlock(unsafe { core::ptr::addr_of_mut!((*ctx).__slate_anon_0.lock) });
unsafe { (*(data as *mut dispatch_rq_data)).rq }.is_null()
}

#[unsafe(no_mangle)]
extern "C" fn blk_mq_dequeue_from_ctx(hctx: *mut blk_mq_hw_ctx, start: *mut blk_mq_ctx) -> *mut request {
let off: u32 = (if !start.is_null() { (unsafe { *((*start).index_hw.as_mut_ptr() as *mut u16).wrapping_offset(((*hctx).r#type as i32) as isize) }) as i32 } else { 0i32 }) as u32;
let mut data: dispatch_rq_data = dispatch_rq_data { hctx: hctx, rq: core::ptr::null_mut::<request>() };
__sbitmap_for_each_set(unsafe { core::ptr::addr_of_mut!((*hctx).ctx_map) }, off, Some(dispatch_rq_from_ctx), core::ptr::addr_of_mut!(data) as *mut ());
data.rq
}

#[unsafe(no_mangle)]
extern "C" fn __blk_mq_alloc_driver_tag(rq: *mut request) -> bool {
let mut bt: *mut sbitmap_queue = unsafe { core::ptr::addr_of_mut!((*(*(*rq).mq_hctx).tags).bitmap_tags) };
let mut tag_offset: u32 = unsafe { (*(*(*rq).mq_hctx).tags).nr_reserved_tags };
let mut tag: i32 = 0;
blk_mq_tag_busy(unsafe { (*rq).mq_hctx });
if blk_mq_tag_is_reserved(unsafe { (*(*rq).mq_hctx).sched_tags }, (unsafe { (*rq).internal_tag }) as u32) {
bt = unsafe { core::ptr::addr_of_mut!((*(*(*rq).mq_hctx).tags).breserved_tags) };
tag_offset = 0;
} else {
if !hctx_may_queue(unsafe { (*rq).mq_hctx }, bt) {
return false;
}
}
tag = unsafe { __sbitmap_queue_get(bt) };
if (tag as u32) == 4294967295 {
return false;
}
unsafe {
(*rq).tag = (tag as u32).wrapping_add(tag_offset) as i32;
}
blk_mq_inc_active_requests(unsafe { (*rq).mq_hctx });
true
}

#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_dispatch_wake")]
extern "C" fn blk_mq_dispatch_wake(wait: *mut wait_queue_entry, mode: u32, flags: i32, key: *mut ()) -> i32 {
let mut hctx: *mut blk_mq_hw_ctx = unsafe { core::mem::zeroed() };
hctx = container_of__18!(wait, blk_mq_hw_ctx);
spin_lock(unsafe { core::ptr::addr_of_mut!((*hctx).dispatch_wait_lock) });
if list_empty((unsafe { core::ptr::addr_of_mut!((*wait).entry) }) as *const list_head) == 0 {
let mut sbq: *mut sbitmap_queue = unsafe { core::mem::zeroed() };
list_del_init(unsafe { core::ptr::addr_of_mut!((*wait).entry) });
sbq = unsafe { core::ptr::addr_of_mut!((*(*hctx).tags).bitmap_tags) };
atomic_dec(unsafe { core::ptr::addr_of_mut!((*sbq).ws_active) });
}
spin_unlock(unsafe { core::ptr::addr_of_mut!((*hctx).dispatch_wait_lock) });
blk_mq_run_hw_queue(hctx, true);
1
}

/// Mark us waiting for a tag. For shared tags, this involves hooking us into
/// the tag wakeups. For non-shared tags, we can simply mark us needing a
/// restart. For both cases, take care to check the condition again after
/// marking us as waiting.
fn blk_mq_mark_tag_wait(hctx: *mut blk_mq_hw_ctx, rq: *mut request) -> bool {
let mut sbq: *mut sbitmap_queue = unsafe { core::mem::zeroed() };
let mut wq: *mut wait_queue_head = unsafe { core::mem::zeroed() };
let mut wait: *mut wait_queue_entry = unsafe { core::mem::zeroed() };
let mut ret: bool = false;
let __v0: bool = (unsafe { (*hctx).flags }) & 2 == 0 && !blk_mq_is_shared_tags((unsafe { (*hctx).flags }) as u32);
if __v0 {
unsafe { blk_mq_sched_mark_restart_hctx(hctx) };
// It's possible that a tag was freed in the window between the
// allocation failure and adding the hardware queue to the wait
// queue.
//
// Don't clear RESTART here, someone else could have set it.
// At most this will cost an extra queue run.
return blk_mq_get_driver_tag(rq);
}
wait = unsafe { core::ptr::addr_of_mut!((*hctx).dispatch_wait) };
if list_empty_careful((unsafe { core::ptr::addr_of_mut!((*wait).entry) }) as *const list_head) == 0 {
return false;
}
if blk_mq_tag_is_reserved(unsafe { (*(*rq).mq_hctx).sched_tags }, (unsafe { (*rq).internal_tag }) as u32) {
sbq = unsafe { core::ptr::addr_of_mut!((*(*hctx).tags).breserved_tags) };
} else {
sbq = unsafe { core::ptr::addr_of_mut!((*(*hctx).tags).bitmap_tags) };
}
wq = unsafe { core::ptr::addr_of_mut!((*bt_wait_ptr(sbq, hctx)).wait) };
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*wq).lock) });
spin_lock(unsafe { core::ptr::addr_of_mut!((*hctx).dispatch_wait_lock) });
if list_empty((unsafe { core::ptr::addr_of_mut!((*wait).entry) }) as *const list_head) == 0 {
spin_unlock(unsafe { core::ptr::addr_of_mut!((*hctx).dispatch_wait_lock) });
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*wq).lock) });
return false;
}
atomic_inc(unsafe { core::ptr::addr_of_mut!((*sbq).ws_active) });
let __v3: u32 = (unsafe { (*wait).flags }) & 4294967294u32;
unsafe {
(*wait).flags = __v3;
}
__add_wait_queue(wq, wait);
// Add one explicit barrier since blk_mq_get_driver_tag() may
// not imply barrier in case of failure.
//
// Order adding us to wait queue and allocating driver tag.
//
// The pair is the one implied in sbitmap_queue_wake_up() which
// orders clearing sbitmap tag bits and waitqueue_active() in
// __sbitmap_queue_wake_up(), since waitqueue_active() is lockless
//
// Otherwise, re-order of adding wait queue and getting driver tag
// may cause __sbitmap_queue_wake_up() to wake up nothing because
// the waitqueue_active() may not observe us in wait queue.
loop {
kcsan_mb!();
unsafe {
core::arch::asm!("lock addl $0,-4(%rsp)", options(nostack, att_syntax));
}
if true {
break;
}
}
// It's possible that a tag was freed in the window between the
// allocation failure and adding the hardware queue to the wait
// queue.
ret = blk_mq_get_driver_tag(rq);
if !ret {
spin_unlock(unsafe { core::ptr::addr_of_mut!((*hctx).dispatch_wait_lock) });
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*wq).lock) });
return false;
}
// We got a tag, remove ourselves from the wait queue to ensure
// someone else gets the wakeup.
list_del_init(unsafe { core::ptr::addr_of_mut!((*wait).entry) });
atomic_dec(unsafe { core::ptr::addr_of_mut!((*sbq).ws_active) });
spin_unlock(unsafe { core::ptr::addr_of_mut!((*hctx).dispatch_wait_lock) });
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*wq).lock) });
true
}

/// Update dispatch busy with the Exponential Weighted Moving Average(EWMA):
/// - EWMA is one simple way to compute running average value
/// - weight(7/8 and 1/8) is applied so that it can decrease exponentially
/// - take 4 as factor for avoiding to get too small(0) result, and this
///   factor doesn't matter because EWMA decreases exponentially
fn blk_mq_update_dispatch_busy(hctx: *mut blk_mq_hw_ctx, busy: bool) {
let mut ewma: u32 = unsafe { (*hctx).dispatch_busy };
if ewma == 0 && !busy {
return;
}
ewma = ewma.wrapping_mul(7u32);
if busy {
ewma = ewma.wrapping_add(16u32);
}
ewma /= 8;
unsafe {
(*hctx).dispatch_busy = ewma;
}
}

// ms units
fn blk_mq_handle_dev_resource(rq: *mut request, list: *mut list_head) {
list_add(unsafe { core::ptr::addr_of_mut!((*rq).__slate_anon_12.queuelist) }, list);
__blk_mq_requeue_request(rq);
}

fn blk_mq_prep_dispatch_rq(rq: *mut request, need_budget: bool) -> u32 {
let hctx: *mut blk_mq_hw_ctx = unsafe { (*rq).mq_hctx };
let mut budget_token: i32 = 1i32.wrapping_neg();
if need_budget {
budget_token = blk_mq_get_dispatch_budget(unsafe { (*rq).q });
if budget_token < 0 {
blk_mq_put_driver_tag(rq);
return 2;
}
blk_mq_set_rq_budget_token(rq, budget_token);
}
if !blk_mq_get_driver_tag(rq) {
// The initial allocation attempt failed, so we need to
// rerun the hardware queue when a tag is freed. The
// waitqueue takes care of that. If the queue is run
// before we add this entry back on the dispatch list,
// we'll re-run it below.
if !blk_mq_mark_tag_wait(hctx, rq) {
// All budgets not got from this function will be put
// together during handling partial dispatch
if need_budget {
blk_mq_put_dispatch_budget(unsafe { (*rq).q }, budget_token);
}
return 1;
}
}
0
}

/// release all allocated budgets before calling to blk_mq_dispatch_rq_list
fn blk_mq_release_budgets(q: *mut request_queue, list: *mut list_head) {
let mut rq: *mut request = unsafe { core::mem::zeroed() };
rq = list_first_entry__3!(list);
while list_entry_is_head__2!(rq, list, queuelist) == 0 {
let budget_token: i32 = blk_mq_get_rq_budget_token(rq);
if budget_token >= 0 {
blk_mq_put_dispatch_budget(q, budget_token);
}
rq = list_next_entry__3!(rq, queuelist);
}
}

/// blk_mq_commit_rqs will notify driver using bd->last that there is no
/// more requests. (See comment in struct blk_mq_ops for commit_rqs for
/// details)
/// Attention, we should explicitly call this in unusual cases:
///  1) did not queue everything initially scheduled to queue
///  2) the last attempt to queue a request failed
fn blk_mq_commit_rqs(hctx: *mut blk_mq_hw_ctx, queued: i32, from_schedule: bool) {
if (unsafe { (*(*(*hctx).queue).mq_ops).commit_rqs }) != None && queued != 0 {
trace_block_unplug(unsafe { (*hctx).queue }, queued as u32, !from_schedule);
unsafe { unsafe { (*(*(*hctx).queue).mq_ops).commit_rqs }.unwrap()(hctx) };
}
}

/// Returns true if we did some work AND can potentially do more.
#[unsafe(no_mangle)]
extern "C" fn blk_mq_dispatch_rq_list(hctx: *mut blk_mq_hw_ctx, list: *mut list_head, get_budget: bool) -> bool {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut __v2: bool = unsafe { core::mem::zeroed() };
let mut __v3: bool = unsafe { core::mem::zeroed() };
// For non-shared tags, the RESTART check will suffice
let mut no_tag: bool = unsafe { core::mem::zeroed() };
let mut needs_restart: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut bool = core::ptr::addr_of_mut!(needs_restart) as *mut bool;
let mut bd: core::mem::MaybeUninit<blk_mq_queue_data> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut blk_mq_queue_data = core::ptr::addr_of_mut!(bd) as *mut blk_mq_queue_data;
let mut needs_resource: bool = unsafe { core::mem::zeroed() };
let mut ret: u8 = unsafe { core::mem::zeroed() };
let mut queued: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut i32 = core::ptr::addr_of_mut!(queued) as *mut i32;
let mut rq: core::mem::MaybeUninit<*mut request> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut *mut request = core::ptr::addr_of_mut!(rq) as *mut *mut request;
let mut q: *mut request_queue = unsafe { core::mem::zeroed() };
let mut prep: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut u32 = core::ptr::addr_of_mut!(prep) as *mut u32;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
q = (*hctx).queue;
let __slate_auto_bytes_2 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
ret = 0;
needs_resource = false;
if list_empty(list as *const list_head) != 0 {
return false;
} else {
// Now process all the entries, sending them to the driver.
*__slate_slot_14 = 0;
'__join_26: {
'__join_29: {
loop {
let __slate_auto_bytes = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
*__slate_slot_15 = list_first_entry__2!(list, request);
let __ret_warn_on: i32 = (hctx != (*(*__slate_slot_15)).mq_hctx) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("964: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 964b - ., 3; .popsection\n# {0}", const 964 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_blk_mq_dispatch_rq_list_19, sym __slate_asm_string_blk_mq_dispatch_rq_list_20, const 2120 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("965: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 965b - ., 4; .popsection\n# {0}", const 965 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
*__slate_slot_17 = blk_mq_prep_dispatch_rq(*__slate_slot_15, get_budget);
if (*__slate_slot_17 as u32) != 0 {
break '__join_26;
} else {
list_del_init(core::ptr::addr_of_mut!((*(*__slate_slot_15)).__slate_anon_12.queuelist));
(*__slate_slot_11).rq = *__slate_slot_15;
(*__slate_slot_11).last = list_empty(list as *const list_head) != 0;
ret = unsafe { (*(*q).mq_ops).queue_rq }.unwrap()(hctx, core::ptr::addr_of_mut!(*__slate_slot_11) as *const blk_mq_queue_data);
let __t0: i32 = ret as i32;
if __t0 == 0 {
let __v7: i32 = *__slate_slot_14;
*__slate_slot_14 = __v7.wrapping_add(1);
} else {
if __t0 == 9 {
break;
} else {
if __t0 == 13 {
break '__join_29;
} else {
blk_mq_end_request(*__slate_slot_15, ret);
}
}
}
if list_empty(list as *const list_head) != 0 {
break '__join_26;
}
}
}
needs_resource = true;
}
blk_mq_handle_dev_resource(*__slate_slot_15, list);
}
if list_empty(list as *const list_head) == 0 || (ret as i32) != 0 {
blk_mq_commit_rqs(hctx, *__slate_slot_14, false);
}
// If we didn't flush the entire list, we could have told the driver
// there was more coming, but that turned out to be a lie.
// Any items that need requeuing? Stuff them into hctx->dispatch,
// that is where we will continue on next queue run.
if list_empty(list as *const list_head) == 0 {
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
if (*__slate_slot_17 as u32) == 1 {
if (*hctx).flags & 2 != 0 {
__v2 = true;
} else {
__v2 = blk_mq_is_shared_tags((*hctx).flags as u32);
}
__v3 = __v2;
} else {
__v3 = false;
}
no_tag = __v3;
// If the caller allocated budgets, free the budgets of the
// requests that have not yet been passed to the block driver.
if !get_budget {
blk_mq_release_budgets(q, list);
}
spin_lock(core::ptr::addr_of_mut!((*hctx).__slate_anon_0.lock));
list_splice_tail_init(list, core::ptr::addr_of_mut!((*hctx).__slate_anon_0.dispatch));
spin_unlock(core::ptr::addr_of_mut!((*hctx).__slate_anon_0.lock));
// Order adding requests to hctx->dispatch and checking
// SCHED_RESTART flag. The pair of this smp_mb() is the one
// in blk_mq_sched_restart(). Avoid restart code path to
// miss the new added requests to hctx->dispatch, meantime
// SCHED_RESTART is observed here.
loop {
kcsan_mb!();
core::arch::asm!("lock addl $0,-4(%rsp)", options(nostack, att_syntax));
if true {
break;
}
}
// If SCHED_RESTART was set by the caller of this function and
// it is no longer set that means that it was cleared by another
// thread and hence that a queue rerun is needed.
//
// If 'no_tag' is set, that means that we failed getting
// a driver tag with an I/O scheduler attached. If our dispatch
// waitqueue is no longer active, ensure that we run the queue
// AFTER adding our entries back to the list.
//
// If no I/O scheduler has been configured it is possible that
// the hardware queue got stopped and restarted before requests
// were pushed back onto the dispatch list. Rerun the queue to
// avoid starvation. Notes:
// - blk_mq_run_hw_queue() checks whether or not a queue has
//   been stopped before rerunning a queue.
// - Some but not all block drivers stop a queue before
//   returning BLK_STS_RESOURCE. Two exceptions are scsi-mq
//   and dm-rq.
//
// If driver returns BLK_STS_RESOURCE and SCHED_RESTART
// bit is set, run queue after a delay to avoid IO stalls
// that could otherwise occur if the queue is idle.  We'll do
// similar if we couldn't get budget or couldn't lock a zone
// and SCHED_RESTART is set.
*__slate_slot_5 = blk_mq_sched_needs_restart(hctx);
if (*__slate_slot_17 as u32) == 2 {
needs_resource = true;
}
if !(*__slate_slot_5) {
__v1 = true;
} else {
if no_tag {
__v0 = list_empty_careful(core::ptr::addr_of_mut!((*hctx).dispatch_wait.entry) as *const list_head) != 0;
} else {
__v0 = false;
}
__v1 = __v0;
}
if __v1 {
blk_mq_run_hw_queue(hctx, true);
} else {
if needs_resource {
blk_mq_delay_run_hw_queue(hctx, 3u64);
}
}
blk_mq_update_dispatch_busy(hctx, true);
return false;
} else {
blk_mq_update_dispatch_busy(hctx, false);
return true;
}
}
}
}

fn blk_mq_first_mapped_cpu(hctx: *mut blk_mq_hw_ctx) -> i32 {
let mut cpu: i32 = cpumask_first_and((unsafe { (*hctx).cpumask.as_mut_ptr() as *mut cpumask }) as *const cpumask, (unsafe { core::ptr::addr_of_mut!(__cpu_online_mask) }) as *const cpumask) as i32;
if (cpu as u32) >= unsafe { nr_cpu_ids } {
cpu = cpumask_first((unsafe { (*hctx).cpumask.as_mut_ptr() as *mut cpumask }) as *const cpumask) as i32;
}
cpu
}

/// ->next_cpu is always calculated from hctx->cpumask, so simply use
/// it for speeding up the check
fn blk_mq_hctx_empty_cpumask(hctx: *mut blk_mq_hw_ctx) -> bool {
((unsafe { (*hctx).next_cpu }) as u32) >= unsafe { nr_cpu_ids }
}

/// It'd be great if the workqueue API had a way to pass
/// in a mask and had some smarts for more clever placement.
/// For now we just round-robin here, switching for every
/// BLK_MQ_CPU_WORK_BATCH queued items.
fn blk_mq_hctx_next_cpu(hctx: *mut blk_mq_hw_ctx) -> i32 {
// Switch to unbound if no allowable CPUs in this hctx
let mut __v3: bool = unsafe { core::mem::zeroed() };
let mut next_cpu: i32 = unsafe { core::mem::zeroed() };
let mut tried: bool = unsafe { core::mem::zeroed() };
let mut __slate_state: usize = 13usize;
unsafe {
tried = false;
next_cpu = (*hctx).next_cpu;
if (*(*hctx).queue).nr_hw_queues == 1 {
__v3 = true;
} else {
__v3 = blk_mq_hctx_empty_cpumask(hctx);
}
if __v3 {
return 64;
} else {
let __v1: i32 = (*hctx).next_cpu_batch;
let __v0: i32 = __v1.wrapping_sub(1);
(*hctx).next_cpu_batch = __v0;
if __v0 <= 0 {
__slate_state = 7;
} else {
__slate_state = 4;
}
loop {
match __slate_state {
4 => {
// Do unbound schedule if we can't find a online CPU for this hctx,
// and it should only happen in the path of handling CPU DEAD.
if !cpu_online(next_cpu as u32) {
if !tried {
tried = true;
__slate_state = 7;
} else {
// Make sure to re-select CPU next time once after CPUs
// in hctx->cpumask become online again.
(*hctx).next_cpu = next_cpu;
(*hctx).next_cpu_batch = 1;
return 64;
}
} else {
break;
}
}
_ => {
next_cpu = cpumask_next_and(next_cpu, ((*hctx).cpumask.as_mut_ptr() as *mut cpumask) as *const cpumask, core::ptr::addr_of_mut!(__cpu_online_mask) as *const cpumask) as i32;
if (next_cpu as u32) >= nr_cpu_ids {
next_cpu = blk_mq_first_mapped_cpu(hctx);
}
(*hctx).next_cpu_batch = 8;
__slate_state = 4;
}
}
}
(*hctx).next_cpu = next_cpu;
return next_cpu;
}
}
}

/// blk_mq_delay_run_hw_queue - Run a hardware queue asynchronously.
/// @hctx: Pointer to the hardware queue to run.
/// @msecs: Milliseconds of delay to wait before running the queue.
///
/// Run a hardware queue asynchronously with a delay of @msecs.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_delay_run_hw_queue")]
extern "C" fn blk_mq_delay_run_hw_queue(hctx: *mut blk_mq_hw_ctx, msecs: u64) {
if unlikely__2!(blk_mq_hctx_stopped(hctx)) != 0 {
return;
}
unsafe { kblockd_mod_delayed_work_on(blk_mq_hctx_next_cpu(hctx), core::ptr::addr_of_mut!((*hctx).run_work), msecs_to_jiffies(msecs as u32)) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_delay_run_hw_queue_966: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut blk_mq_hw_ctx, u64)>, *mut ()>(Some(blk_mq_delay_run_hw_queue)) };

fn blk_mq_hw_queue_need_run(hctx: *mut blk_mq_hw_ctx) -> bool {
let mut need_run: bool = false;
// When queue is quiesced, we may be switching io scheduler, or
// updating nr_hw_queues, or other things, and we can't run queue
// any more, even blk_mq_hctx_has_pending() can't be called safely.
//
// And queue will be rerun in blk_mq_unquiesce_queue() if it is
// quiesced.
__blk_mq_run_dispatch_ops__2!(unsafe { (*hctx).queue }, false, hctx, need_run);
need_run
}

/// blk_mq_run_hw_queue - Start to run a hardware queue.
/// @hctx: Pointer to the hardware queue to run.
/// @async: If we want to run the queue asynchronously.
///
/// Check if the request queue is not in a quiesced state and if there are
/// pending requests to be sent. If this is true, run the queue to send requests
/// to hardware.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_run_hw_queue")]
extern "C" fn blk_mq_run_hw_queue(hctx: *mut blk_mq_hw_ctx, r#async: bool) {
let mut need_run: bool = false;
// We can't run the queue inline with interrupts disabled.
let mut __v0: i64;
let mut __ret_warn_on: i32 = 0;
let __v1: bool = !r#async && in_interrupt!() != 0;
__ret_warn_on = __v1 as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("967: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 967b - ., 3; .popsection\n# {0}", const 967 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_blk_mq_run_hw_queue_2, sym __slate_asm_string_blk_mq_run_hw_queue_3, const 2341 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("968: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 968b - ., 4; .popsection\n# {0}", const 968 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
might_sleep_if!(!r#async && (unsafe { (*hctx).flags }) & 16 != 0);
need_run = blk_mq_hw_queue_need_run(hctx);
if !need_run {
let mut flags: u64 = 0;
// Synchronize with blk_mq_unquiesce_queue(), because we check
// if hw queue is quiesced locklessly above, we need the use
// ->queue_lock to make sure we see the up-to-date status to
// not miss rerunning the hw queue.
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*(*hctx).queue).queue_lock) }, flags);
need_run = blk_mq_hw_queue_need_run(hctx);
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*(*hctx).queue).queue_lock) }, flags);
if !need_run {
return;
}
}
let __v11: bool = if r#async {
true
} else {
let mut __v12: i32;
let mut pscr_ret__: i32 = 0;
__verify_pcpu_ptr__3!();
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v14: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v14 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v14;
}
2 => {
let mut __v16: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v16 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v16;
}
4 => {
let mut __v17: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v17 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v17;
}
8 => {
let mut __v18: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v18 = pfo_val__ as i32;
pscr_ret__ = __v18;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v12 = pscr_ret__;
!cpumask_test_cpu(__v12, (unsafe { (*hctx).cpumask.as_mut_ptr() as *mut cpumask }) as *const cpumask)
};
if __v11 {
blk_mq_delay_run_hw_queue(hctx, 0u64);
return;
}
blk_mq_run_dispatch_ops__2!(unsafe { (*hctx).queue }, unsafe { blk_mq_sched_dispatch_requests(hctx) });
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_run_hw_queue_969: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut blk_mq_hw_ctx, bool)>, *mut ()>(Some(blk_mq_run_hw_queue)) };

/// Return prefered queue to dispatch from (if any) for non-mq aware IO
/// scheduler.
fn blk_mq_get_sq_hctx(q: *mut request_queue) -> *mut blk_mq_hw_ctx {
let ctx: *mut blk_mq_ctx = blk_mq_get_ctx(q);
// If the IO scheduler does not respect hardware queues when
// dispatching, we just don't bother with multiple HW queues and
// dispatch from hctx for the current CPU since running multiple queues
// just causes lock contention inside the scheduler and pointless cache
// bouncing.
let hctx: *mut blk_mq_hw_ctx = unsafe { *((*ctx).hctxs.as_mut_ptr() as *mut *mut blk_mq_hw_ctx).wrapping_offset(0 as isize) };
if !blk_mq_hctx_stopped(hctx) {
return hctx;
}
core::ptr::null_mut::<blk_mq_hw_ctx>()
}

/// blk_mq_run_hw_queues - Run all hardware queues in a request queue.
/// @q: Pointer to the request queue to run.
/// @async: If we want to run the queue asynchronously.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_run_hw_queues")]
extern "C" fn blk_mq_run_hw_queues(q: *mut request_queue, r#async: bool) {
let mut hctx: *mut blk_mq_hw_ctx = unsafe { core::mem::zeroed() };
let mut sq_hctx: *mut blk_mq_hw_ctx = unsafe { core::mem::zeroed() };
let mut i: u64 = 0;
sq_hctx = core::ptr::null_mut::<blk_mq_hw_ctx>();
if blk_queue_sq_sched!(q) != 0 {
sq_hctx = blk_mq_get_sq_hctx(q);
}
loop {
let __v3: bool = if i < ((unsafe { (*q).nr_hw_queues }) as u64) {
let mut __v4: i32;
hctx = queue_hctx(q, i as i32);
__v4 = 1;
__v4 != 0
} else {
false
};
if !__v3 {
break;
}
if !blk_mq_hctx_stopped(hctx) {
// Dispatch from this hctx either if there's no hctx preferred
// by IO scheduler or if it has requests that bypass the
// scheduler.
let __v2: bool = sq_hctx.is_null() || sq_hctx == hctx || list_empty_careful((unsafe { core::ptr::addr_of_mut!((*hctx).__slate_anon_0.dispatch) }) as *const list_head) == 0;
if __v2 {
blk_mq_run_hw_queue(hctx, r#async);
}
}
i = i.wrapping_add(1);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_run_hw_queues_970: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request_queue, bool)>, *mut ()>(Some(blk_mq_run_hw_queues)) };

/// blk_mq_delay_run_hw_queues - Run all hardware queues asynchronously.
/// @q: Pointer to the request queue to run.
/// @msecs: Milliseconds of delay to wait before running the queues.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_delay_run_hw_queues")]
extern "C" fn blk_mq_delay_run_hw_queues(q: *mut request_queue, msecs: u64) {
let mut hctx: *mut blk_mq_hw_ctx = unsafe { core::mem::zeroed() };
let mut sq_hctx: *mut blk_mq_hw_ctx = unsafe { core::mem::zeroed() };
let mut i: u64 = 0;
sq_hctx = core::ptr::null_mut::<blk_mq_hw_ctx>();
if blk_queue_sq_sched!(q) != 0 {
sq_hctx = blk_mq_get_sq_hctx(q);
}
loop {
let __v4: bool = if i < ((unsafe { (*q).nr_hw_queues }) as u64) {
let mut __v5: i32;
hctx = queue_hctx(q, i as i32);
__v5 = 1;
__v5 != 0
} else {
false
};
if !__v4 {
break;
}
if !blk_mq_hctx_stopped(hctx) {
// If there is already a run_work pending, leave the
// pending delay untouched. Otherwise, a hctx can stall
// if another hctx is re-delaying the other's work
// before the work executes.
if delayed_work_pending!(unsafe { core::ptr::addr_of_mut!((*hctx).run_work) }) == 0 {
// Dispatch from this hctx either if there's no hctx preferred
// by IO scheduler or if it has requests that bypass the
// scheduler.
let __v3: bool = sq_hctx.is_null() || sq_hctx == hctx || list_empty_careful((unsafe { core::ptr::addr_of_mut!((*hctx).__slate_anon_0.dispatch) }) as *const list_head) == 0;
if __v3 {
blk_mq_delay_run_hw_queue(hctx, msecs);
}
}
}
i = i.wrapping_add(1);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_delay_run_hw_queues_971: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request_queue, u64)>, *mut ()>(Some(blk_mq_delay_run_hw_queues)) };

/// This function is often used for pausing .queue_rq() by driver when
/// there isn't enough resource or some conditions aren't satisfied, and
/// BLK_STS_RESOURCE is usually returned.
///
/// We do not guarantee that dispatch can be drained or blocked
/// after blk_mq_stop_hw_queue() returns. Please use
/// blk_mq_quiesce_queue() for that requirement.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_stop_hw_queue")]
extern "C" fn blk_mq_stop_hw_queue(hctx: *mut blk_mq_hw_ctx) {
unsafe { cancel_delayed_work(core::ptr::addr_of_mut!((*hctx).run_work)) };
set_bit(0i64, (unsafe { core::ptr::addr_of_mut!((*hctx).__slate_anon_0.state) }) as *mut u64);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_stop_hw_queue_972: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut blk_mq_hw_ctx)>, *mut ()>(Some(blk_mq_stop_hw_queue)) };

/// This function is often used for pausing .queue_rq() by driver when
/// there isn't enough resource or some conditions aren't satisfied, and
/// BLK_STS_RESOURCE is usually returned.
///
/// We do not guarantee that dispatch can be drained or blocked
/// after blk_mq_stop_hw_queues() returns. Please use
/// blk_mq_quiesce_queue() for that requirement.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_stop_hw_queues")]
extern "C" fn blk_mq_stop_hw_queues(q: *mut request_queue) {
let mut hctx: *mut blk_mq_hw_ctx = unsafe { core::mem::zeroed() };
let mut i: u64 = 0;
loop {
let __v0: bool = if i < ((unsafe { (*q).nr_hw_queues }) as u64) {
let mut __v1: i32;
hctx = queue_hctx(q, i as i32);
__v1 = 1;
__v1 != 0
} else {
false
};
if !__v0 {
break;
}
blk_mq_stop_hw_queue(hctx);
i = i.wrapping_add(1);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_stop_hw_queues_973: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request_queue)>, *mut ()>(Some(blk_mq_stop_hw_queues)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_start_hw_queue")]
extern "C" fn blk_mq_start_hw_queue(hctx: *mut blk_mq_hw_ctx) {
clear_bit(0i64, (unsafe { core::ptr::addr_of_mut!((*hctx).__slate_anon_0.state) }) as *mut u64);
blk_mq_run_hw_queue(hctx, (unsafe { (*hctx).flags }) & 16 != 0);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_start_hw_queue_974: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut blk_mq_hw_ctx)>, *mut ()>(Some(blk_mq_start_hw_queue)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_start_hw_queues")]
extern "C" fn blk_mq_start_hw_queues(q: *mut request_queue) {
let mut hctx: *mut blk_mq_hw_ctx = unsafe { core::mem::zeroed() };
let mut i: u64 = 0;
loop {
let __v0: bool = if i < ((unsafe { (*q).nr_hw_queues }) as u64) {
let mut __v1: i32;
hctx = queue_hctx(q, i as i32);
__v1 = 1;
__v1 != 0
} else {
false
};
if !__v0 {
break;
}
blk_mq_start_hw_queue(hctx);
i = i.wrapping_add(1);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_start_hw_queues_975: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request_queue)>, *mut ()>(Some(blk_mq_start_hw_queues)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_start_stopped_hw_queue")]
extern "C" fn blk_mq_start_stopped_hw_queue(hctx: *mut blk_mq_hw_ctx, r#async: bool) {
if !blk_mq_hctx_stopped(hctx) {
return;
}
clear_bit(0i64, (unsafe { core::ptr::addr_of_mut!((*hctx).__slate_anon_0.state) }) as *mut u64);
// Pairs with the smp_mb() in blk_mq_hctx_stopped() to order the
// clearing of BLK_MQ_S_STOPPED above and the checking of dispatch
// list in the subsequent routine.
smp_mb__after_atomic!();
blk_mq_run_hw_queue(hctx, r#async);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_start_stopped_hw_queue_976: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut blk_mq_hw_ctx, bool)>, *mut ()>(Some(blk_mq_start_stopped_hw_queue)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_start_stopped_hw_queues")]
extern "C" fn blk_mq_start_stopped_hw_queues(q: *mut request_queue, r#async: bool) {
let mut hctx: *mut blk_mq_hw_ctx = unsafe { core::mem::zeroed() };
let mut i: u64 = 0;
loop {
let __v0: bool = if i < ((unsafe { (*q).nr_hw_queues }) as u64) {
let mut __v1: i32;
hctx = queue_hctx(q, i as i32);
__v1 = 1;
__v1 != 0
} else {
false
};
if !__v0 {
break;
}
blk_mq_start_stopped_hw_queue(hctx, r#async || (unsafe { (*hctx).flags }) & 16 != 0);
i = i.wrapping_add(1);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_start_stopped_hw_queues_977: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request_queue, bool)>, *mut ()>(Some(blk_mq_start_stopped_hw_queues)) };

#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_run_work_fn")]
extern "C" fn blk_mq_run_work_fn(work: *mut work_struct) {
let mut __v0: *mut blk_mq_hw_ctx;
__v0 = ((unsafe { (work as *const u8).wrapping_offset(-(offsetof__45!(blk_mq_hw_ctx) as isize)) }) as *mut ()) as *mut blk_mq_hw_ctx;
let hctx: *mut blk_mq_hw_ctx = __v0;
blk_mq_run_dispatch_ops__2!(unsafe { (*hctx).queue }, unsafe { blk_mq_sched_dispatch_requests(hctx) });
}

/// blk_mq_request_bypass_insert - Insert a request at dispatch list.
/// @rq: Pointer to request to be inserted.
/// @flags: BLK_MQ_INSERT_*
///
/// Should only be used carefully, when the caller knows we want to
/// bypass a potential IO scheduler on the target device.
fn blk_mq_request_bypass_insert(rq: *mut request, flags: u32) {
let hctx: *mut blk_mq_hw_ctx = unsafe { (*rq).mq_hctx };
spin_lock(unsafe { core::ptr::addr_of_mut!((*hctx).__slate_anon_0.lock) });
if flags & 1 != 0 {
list_add(unsafe { core::ptr::addr_of_mut!((*rq).__slate_anon_12.queuelist) }, unsafe { core::ptr::addr_of_mut!((*hctx).__slate_anon_0.dispatch) });
} else {
list_add_tail(unsafe { core::ptr::addr_of_mut!((*rq).__slate_anon_12.queuelist) }, unsafe { core::ptr::addr_of_mut!((*hctx).__slate_anon_0.dispatch) });
}
spin_unlock(unsafe { core::ptr::addr_of_mut!((*hctx).__slate_anon_0.lock) });
}

fn blk_mq_insert_requests(hctx: *mut blk_mq_hw_ctx, ctx: *mut blk_mq_ctx, list: *mut list_head, mut run_queue_async: bool) {
let mut r#type: u32 = unsafe { core::mem::zeroed() };
let mut rq: core::mem::MaybeUninit<*mut request> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut request = core::ptr::addr_of_mut!(rq) as *mut *mut request;
unsafe {
'__join_0: {
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
r#type = (*hctx).r#type as u32;
// Try to issue requests directly if the hw queue isn't busy to save an
// extra enqueue & dequeue to the sw queue.
if (*hctx).dispatch_busy == 0 && !run_queue_async {
blk_mq_run_dispatch_ops!(unsafe { (*hctx).queue }, blk_mq_try_issue_list_directly(hctx, list));
if list_empty(list as *const list_head) != 0 {
break '__join_0;
}
}
// preemption doesn't flush plug list, so it's possible ctx->cpu is
// offline now
*__slate_slot_1 = list_first_entry__3!(list);
'__join_8: {
while list_entry_is_head__2!(*__slate_slot_1, list, queuelist) == 0 {
loop {
if unlikely__2!((*(*__slate_slot_1)).mq_ctx != ctx) != 0 {
break '__join_8;
} else {
if true {
break;
}
}
}
trace_block_rq_insert(*__slate_slot_1);
if (*(*__slate_slot_1)).cmd_flags & 2097152u32 != 0 {
run_queue_async = true;
}
*__slate_slot_1 = list_next_entry__3!(*__slate_slot_1, queuelist);
}
spin_lock(core::ptr::addr_of_mut!((*ctx).__slate_anon_0.lock));
list_splice_tail_init(list, ((*ctx).__slate_anon_0.rq_lists.as_mut_ptr() as *mut list_head).wrapping_offset((r#type as u32) as isize));
blk_mq_hctx_mark_pending(hctx, ctx);
spin_unlock(core::ptr::addr_of_mut!((*ctx).__slate_anon_0.lock));
break '__join_0;
}
core::arch::asm!("978: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 978b - ., 3; .popsection\n# {0}", const 978 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_blk_mq_insert_requests_3, sym __slate_asm_string_blk_mq_insert_requests_4, const 2591 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
blk_mq_run_hw_queue(hctx, run_queue_async);
}
}

fn blk_mq_insert_request(rq: *mut request, flags: u32) {
let q: *mut request_queue = unsafe { (*rq).q };
let ctx: *mut blk_mq_ctx = unsafe { (*rq).mq_ctx };
let hctx: *mut blk_mq_hw_ctx = unsafe { (*rq).mq_hctx };
if blk_rq_is_passthrough(rq) {
// Passthrough request have to be added to hctx->dispatch
// directly.  The device may be in a situation where it can't
// handle FS request, and always returns BLK_STS_RESOURCE for
// them, which gets them added to hctx->dispatch.
//
// If a passthrough request is required to unblock the queues,
// and it is added to the scheduler queue, there is no chance to
// dispatch it given we prioritize requests in hctx->dispatch.
blk_mq_request_bypass_insert(rq, flags);
} else {
if (req_op(rq as *const request) as u32) == 2 {
// Firstly normal IO request is inserted to scheduler queue or
// sw queue, meantime we add flush request to dispatch queue(
// hctx->dispatch) directly and there is at most one in-flight
// flush request for each hw queue, so it doesn't matter to add
// flush request to tail or front of the dispatch queue.
//
// Secondly in case of NCQ, flush request belongs to non-NCQ
// command, and queueing it will fail when there is any
// in-flight normal IO request(NCQ command). When adding flush
// rq to the front of hctx->dispatch, it is easier to introduce
// extra time to flush rq's latency because of S_SCHED_RESTART
// compared with adding to the tail of dispatch queue, then
// chance of flush merge is increased, and less flush requests
// will be issued to controller. It is observed that ~10% time
// is saved in blktests block/004 on disk attached to AHCI/NCQ
// drive when adding flush rq to the front of hctx->dispatch.
//
// Simply queue flush rq to the front of hctx->dispatch so that
// intensive flush workloads can benefit in case of NCQ HW.
blk_mq_request_bypass_insert(rq, 1u32);
} else {
if !unsafe { (*q).elevator }.is_null() {
let mut list: list_head = unsafe { core::mem::zeroed() };
list = list_head { next: core::ptr::addr_of_mut!(list), prev: core::ptr::addr_of_mut!(list) };
let mut __v0: i64;
let __ret_warn_on: i32 = (((unsafe { (*rq).tag }) as u32) != 4294967295) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("979: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 979b - ., 3; .popsection\n# {0}", const 979 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_blk_mq_insert_request_1, sym __slate_asm_string_blk_mq_insert_request_2, const 2649 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("980: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 980b - ., 4; .popsection\n# {0}", const 980 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
list_add(unsafe { core::ptr::addr_of_mut!((*rq).__slate_anon_12.queuelist) }, core::ptr::addr_of_mut!(list));
unsafe { unsafe { (*(*(*q).elevator).r#type).ops.insert_requests }.unwrap()(hctx, core::ptr::addr_of_mut!(list), flags) };
} else {
trace_block_rq_insert(rq);
spin_lock(unsafe { core::ptr::addr_of_mut!((*ctx).__slate_anon_0.lock) });
if flags & 1 != 0 {
list_add(unsafe { core::ptr::addr_of_mut!((*rq).__slate_anon_12.queuelist) }, unsafe { (*ctx).__slate_anon_0.rq_lists.as_mut_ptr() as *mut list_head }.wrapping_offset(((unsafe { (*hctx).r#type }) as i32) as isize));
} else {
list_add_tail(unsafe { core::ptr::addr_of_mut!((*rq).__slate_anon_12.queuelist) }, unsafe { (*ctx).__slate_anon_0.rq_lists.as_mut_ptr() as *mut list_head }.wrapping_offset(((unsafe { (*hctx).r#type }) as i32) as isize));
}
blk_mq_hctx_mark_pending(hctx, ctx);
spin_unlock(unsafe { core::ptr::addr_of_mut!((*ctx).__slate_anon_0.lock) });
}
}
}
}

fn blk_mq_bio_to_request(rq: *mut request, bio: *mut bio, nr_segs: u32) {
let mut err: i32 = 0;
if (unsafe { (*bio).bi_opf }) & 524288u32 != 0 {
unsafe {
(*rq).cmd_flags = (*rq).cmd_flags | 1792u32;
}
}
let __v3: *mut bio = bio;
unsafe {
(*rq).biotail = __v3;
}
unsafe {
(*rq).bio = __v3;
}
unsafe {
(*rq).__sector = (*bio).bi_iter.bi_sector;
}
unsafe {
(*rq).__data_len = (*bio).bi_iter.bi_size;
}
unsafe {
(*rq).phys_gap_bit = (*bio).bi_bvec_gap_bit;
}
unsafe {
(*rq).nr_phys_segments = nr_segs as u16;
}
if !bio_integrity(bio).is_null() {
unsafe {
(*rq).nr_integrity_segments = blk_rq_count_integrity_sg((*rq).q, bio) as u16;
}
}
// This can't fail, since GFP_NOIO includes __GFP_DIRECT_RECLAIM.
err = blk_crypto_rq_bio_prep(rq, bio, (BIT!(10i32) | BIT!(11i32)) as u32);
let mut __v4: i64;
let __ret_warn_on: i32 = (err != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("981: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 981b - ., 3; .popsection\n# {0}", const 981 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_blk_mq_bio_to_request_5, sym __slate_asm_string_blk_mq_bio_to_request_6, const 2687 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("982: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 982b - ., 4; .popsection\n# {0}", const 982 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v4 = (__ret_warn_on != 0) as i64;
blk_account_io_start(rq);
}

fn __blk_mq_issue_directly(hctx: *mut blk_mq_hw_ctx, rq: *mut request, last: bool) -> u8 {
let q: *mut request_queue = unsafe { (*rq).q };
let mut bd: blk_mq_queue_data = {
let mut __t0: blk_mq_queue_data = unsafe { core::mem::zeroed() };
__t0.rq = rq;
__t0.last = last;
__t0
};
let mut ret: u8 = 0;
// For OK queue, we are done. For error, caller may kill it.
// Any other error (busy), just add it to our list as we
// previously would have done.
ret = unsafe { unsafe { (*(*q).mq_ops).queue_rq }.unwrap()(hctx, core::ptr::addr_of_mut!(bd) as *const blk_mq_queue_data) };
match ret as i32 {
0 => {
blk_mq_update_dispatch_busy(hctx, false);
}
9 | 13 => {
blk_mq_update_dispatch_busy(hctx, true);
__blk_mq_requeue_request(rq);
}
_ => {
blk_mq_update_dispatch_busy(hctx, false);
}
}
ret
}

fn blk_mq_get_budget_and_tag(rq: *mut request) -> bool {
let budget_token: i32 = blk_mq_get_dispatch_budget(unsafe { (*rq).q });
if budget_token < 0 {
return false;
}
blk_mq_set_rq_budget_token(rq, budget_token);
if !blk_mq_get_driver_tag(rq) {
blk_mq_put_dispatch_budget(unsafe { (*rq).q }, budget_token);
return false;
}
true
}

/// blk_mq_try_issue_directly - Try to send a request directly to device driver.
/// @hctx: Pointer of the associated hardware queue.
/// @rq: Pointer to request to be sent.
///
/// If the device has enough resources to accept a new request now, send the
/// request directly to device driver. Else, insert at hctx->dispatch queue, so
/// we can try send it another time in the future. Requests inserted at this
/// queue have higher priority.
fn blk_mq_try_issue_directly(hctx: *mut blk_mq_hw_ctx, rq: *mut request) {
let mut ret: u8 = 0;
let __v0: bool = blk_mq_hctx_stopped(hctx) || blk_queue_quiesced!(unsafe { (*rq).q }) != 0;
if __v0 {
blk_mq_insert_request(rq, 0u32);
blk_mq_run_hw_queue(hctx, false);
return;
}
let __v2: bool = (unsafe { (*rq).rq_flags }) & 32u32 != 0 || !blk_mq_get_budget_and_tag(rq);
if __v2 {
blk_mq_insert_request(rq, 0u32);
blk_mq_run_hw_queue(hctx, (unsafe { (*rq).cmd_flags }) & 2097152u32 != 0);
return;
}
ret = __blk_mq_issue_directly(hctx, rq, true);
match ret as i32 {
0 => {
}
9 | 13 => {
blk_mq_request_bypass_insert(rq, 0u32);
blk_mq_run_hw_queue(hctx, false);
}
_ => {
blk_mq_end_request(rq, ret);
}
}
}

fn blk_mq_request_issue_directly(rq: *mut request, last: bool) -> u8 {
let hctx: *mut blk_mq_hw_ctx = unsafe { (*rq).mq_hctx };
let __v0: bool = blk_mq_hctx_stopped(hctx) || blk_queue_quiesced!(unsafe { (*rq).q }) != 0;
if __v0 {
blk_mq_insert_request(rq, 0u32);
blk_mq_run_hw_queue(hctx, false);
return 0;
}
if !blk_mq_get_budget_and_tag(rq) {
return 9;
}
__blk_mq_issue_directly(hctx, rq, last)
}

fn blk_mq_issue_direct(rqs: *mut rq_list) {
let mut last: bool = unsafe { core::mem::zeroed() };
let mut ret: u8 = unsafe { core::mem::zeroed() };
let mut queued: i32 = unsafe { core::mem::zeroed() };
let mut rq: core::mem::MaybeUninit<*mut request> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut request = core::ptr::addr_of_mut!(rq) as *mut *mut request;
let mut hctx: *mut blk_mq_hw_ctx = unsafe { core::mem::zeroed() };
unsafe {
hctx = core::ptr::null_mut::<blk_mq_hw_ctx>();
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
queued = 0;
ret = 0;
'__join_2: {
loop {
let __v0: *mut request = rq_list_pop(rqs);
*__slate_slot_6 = __v0;
if !__v0.is_null() {
last = rq_list_empty(rqs as *const rq_list) != 0;
if hctx != (*(*__slate_slot_6)).mq_hctx {
if !hctx.is_null() {
blk_mq_commit_rqs(hctx, queued, false);
queued = 0;
}
hctx = (*(*__slate_slot_6)).mq_hctx;
}
ret = blk_mq_request_issue_directly(*__slate_slot_6, last);
let __t0: i32 = ret as i32;
if __t0 == 0 {
queued = queued.wrapping_add(1);
} else {
if __t0 == 9 {
break;
} else {
if __t0 == 13 {
break;
} else {
blk_mq_end_request(*__slate_slot_6, ret);
}
}
}
} else {
break '__join_2;
}
}
blk_mq_request_bypass_insert(*__slate_slot_6, 0u32);
blk_mq_run_hw_queue(hctx, false);
}
if (ret as i32) != 0 {
blk_mq_commit_rqs(hctx, queued, false);
}
}
}

fn __blk_mq_flush_list(q: *mut request_queue, rqs: *mut rq_list) {
if blk_queue_quiesced!(q) != 0 {
return;
}
unsafe { unsafe { (*(*q).mq_ops).queue_rqs }.unwrap()(rqs) };
}

fn blk_mq_extract_queue_requests(rqs: *mut rq_list, queue_rqs: &mut rq_list) -> u32 {
let mut rq: *mut request = rq_list_pop(rqs);
let this_q: *mut request_queue = unsafe { (*rq).q };
let mut prev: *mut *mut request = unsafe { core::ptr::addr_of_mut!((*rqs).head) };
let mut matched_rqs: rq_list = {
let mut __t0: rq_list = unsafe { core::mem::zeroed() };
__t0
};
let mut last: *mut request = core::ptr::null_mut::<request>();
let mut depth: u32 = 1;
rq_list_add_tail(core::ptr::addr_of_mut!(matched_rqs), rq);
loop {
let __v2: *mut request = unsafe { *prev };
rq = __v2;
if __v2.is_null() {
break;
}
if (unsafe { (*rq).q }) == this_q {
// move rq from rqs to matched_rqs
unsafe {
*prev = (*rq).__slate_anon_12.rq_next;
}
rq_list_add_tail(core::ptr::addr_of_mut!(matched_rqs), rq);
depth = depth.wrapping_add(1);
} else {
// leave rq in rqs
prev = unsafe { core::ptr::addr_of_mut!((*rq).__slate_anon_12.rq_next) };
last = rq;
}
}
unsafe {
(*rqs).tail = last;
}
unsafe {
*(queue_rqs as *mut rq_list) = matched_rqs;
}
depth
}

fn blk_mq_dispatch_queue_requests(rqs: *mut rq_list, depth: u32) {
let q: *mut request_queue = unsafe { (*rq_list_peek(rqs)).q };
trace_block_unplug(q, depth, true);
// Peek first request and see if we have a ->queue_rqs() hook.
// If we do, we can dispatch the whole list in one go.
// We already know at this point that all requests belong to the
// same queue, caller must ensure that's the case.
if (unsafe { (*(*q).mq_ops).queue_rqs }) != None {
blk_mq_run_dispatch_ops!(q, __blk_mq_flush_list(q, rqs));
if rq_list_empty(rqs as *const rq_list) != 0 {
return;
}
}
blk_mq_run_dispatch_ops!(q, blk_mq_issue_direct(rqs));
}

fn blk_mq_dispatch_list(rqs: *mut rq_list, from_sched: bool) {
let mut this_hctx: *mut blk_mq_hw_ctx = core::ptr::null_mut::<blk_mq_hw_ctx>();
let mut this_ctx: *mut blk_mq_ctx = core::ptr::null_mut::<blk_mq_ctx>();
let mut requeue_list: rq_list = {
let mut __t0: rq_list = unsafe { core::mem::zeroed() };
__t0
};
let mut depth: u32 = 0;
let mut is_passthrough: bool = false;
let mut list: list_head = unsafe { core::mem::zeroed() };
list = list_head { next: core::ptr::addr_of_mut!(list), prev: core::ptr::addr_of_mut!(list) };
loop {
'__slate_continue_1: {
let rq: *mut request = rq_list_pop(rqs);
if this_hctx.is_null() {
this_hctx = unsafe { (*rq).mq_hctx };
this_ctx = unsafe { (*rq).mq_ctx };
is_passthrough = blk_rq_is_passthrough(rq);
} else {
let __v0: bool = this_hctx != unsafe { (*rq).mq_hctx } || this_ctx != unsafe { (*rq).mq_ctx } || (is_passthrough as i32) != (blk_rq_is_passthrough(rq) as i32);
if __v0 {
rq_list_add_tail(core::ptr::addr_of_mut!(requeue_list), rq);
break '__slate_continue_1;
}
}
list_add_tail(unsafe { core::ptr::addr_of_mut!((*rq).__slate_anon_12.queuelist) }, core::ptr::addr_of_mut!(list));
depth = depth.wrapping_add(1);
}
if rq_list_empty(rqs as *const rq_list) != 0 {
break;
}
}
unsafe {
*rqs = requeue_list;
}
trace_block_unplug(unsafe { (*this_hctx).queue }, depth, !from_sched);
percpu_ref_get(unsafe { core::ptr::addr_of_mut!((*(*this_hctx).queue).q_usage_counter) });
// passthrough requests should never be issued to the I/O scheduler
if is_passthrough {
spin_lock(unsafe { core::ptr::addr_of_mut!((*this_hctx).__slate_anon_0.lock) });
list_splice_tail_init(core::ptr::addr_of_mut!(list), unsafe { core::ptr::addr_of_mut!((*this_hctx).__slate_anon_0.dispatch) });
spin_unlock(unsafe { core::ptr::addr_of_mut!((*this_hctx).__slate_anon_0.lock) });
blk_mq_run_hw_queue(this_hctx, from_sched);
} else {
if !unsafe { (*(*this_hctx).queue).elevator }.is_null() {
unsafe { unsafe { (*(*(*(*this_hctx).queue).elevator).r#type).ops.insert_requests }.unwrap()(this_hctx, core::ptr::addr_of_mut!(list), 0u32) };
blk_mq_run_hw_queue(this_hctx, from_sched);
} else {
blk_mq_insert_requests(this_hctx, this_ctx, core::ptr::addr_of_mut!(list), from_sched);
}
}
percpu_ref_put(unsafe { core::ptr::addr_of_mut!((*(*this_hctx).queue).q_usage_counter) });
}

fn blk_mq_dispatch_multiple_queue_requests(rqs: *mut rq_list) {
loop {
let mut queue_rqs: rq_list = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(queue_rqs) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let depth: u32 = blk_mq_extract_queue_requests(rqs, unsafe { &mut (*core::ptr::addr_of_mut!(queue_rqs)) });
blk_mq_dispatch_queue_requests(core::ptr::addr_of_mut!(queue_rqs), depth);
while rq_list_empty(core::ptr::addr_of_mut!(queue_rqs) as *const rq_list) == 0 {
blk_mq_dispatch_list(core::ptr::addr_of_mut!(queue_rqs), false);
}
if rq_list_empty(rqs as *const rq_list) != 0 {
break;
}
}
}

#[unsafe(no_mangle)]
extern "C" fn blk_mq_flush_plug_list(plug: *mut blk_plug, from_schedule: bool) {
let mut depth: u32 = 0;
// We may have been called recursively midway through handling
// plug->mq_list via a schedule() in the driver's queue_rq() callback.
// To avoid mq_list changing under our feet, clear rq_count early and
// bail out specifically if rq_count is 0 rather than checking
// whether the mq_list is empty.
if ((unsafe { (*plug).rq_count }) as i32) == 0 {
return;
}
depth = (unsafe { (*plug).rq_count }) as u32;
unsafe {
(*plug).rq_count = 0;
}
if !unsafe { (*plug).has_elevator } && !from_schedule {
if unsafe { (*plug).multiple_queues } {
blk_mq_dispatch_multiple_queue_requests(unsafe { core::ptr::addr_of_mut!((*plug).mq_list) });
return;
}
blk_mq_dispatch_queue_requests(unsafe { core::ptr::addr_of_mut!((*plug).mq_list) }, depth);
if rq_list_empty((unsafe { core::ptr::addr_of_mut!((*plug).mq_list) }) as *const rq_list) != 0 {
return;
}
}
loop {
blk_mq_dispatch_list(unsafe { core::ptr::addr_of_mut!((*plug).mq_list) }, from_schedule);
if rq_list_empty((unsafe { core::ptr::addr_of_mut!((*plug).mq_list) }) as *const rq_list) != 0 {
break;
}
}
}

fn blk_mq_try_issue_list_directly(hctx: *mut blk_mq_hw_ctx, list: *mut list_head) {
let mut rq: *mut request = unsafe { core::mem::zeroed() };
let mut ret: u8 = unsafe { core::mem::zeroed() };
let mut queued: i32 = unsafe { core::mem::zeroed() };
unsafe {
queued = 0;
ret = 0;
'__join_2: {
loop {
if list_empty(list as *const list_head) == 0 {
rq = list_first_entry__2!(list, request);
list_del_init(core::ptr::addr_of_mut!((*rq).__slate_anon_12.queuelist));
ret = blk_mq_request_issue_directly(rq, list_empty(list as *const list_head) != 0);
let __t0: i32 = ret as i32;
if __t0 == 0 {
queued = queued.wrapping_add(1);
} else {
if __t0 == 9 {
break;
} else {
if __t0 == 13 {
break;
} else {
blk_mq_end_request(rq, ret);
}
}
}
} else {
break '__join_2;
}
}
blk_mq_request_bypass_insert(rq, 0u32);
if list_empty(list as *const list_head) != 0 {
blk_mq_run_hw_queue(hctx, false);
}
}
if (ret as i32) != 0 {
blk_mq_commit_rqs(hctx, queued, false);
}
}
}

fn blk_mq_attempt_bio_merge(q: *mut request_queue, bio: *mut bio, nr_segs: u32) -> bool {
let __v0: bool = blk_queue_nomerges!(q) == 0 && bio_mergeable(bio);
if __v0 {
if unsafe { blk_attempt_plug_merge(q, bio, nr_segs) } {
return true;
}
if unsafe { blk_mq_sched_bio_merge(q, bio, nr_segs) } {
return true;
}
}
false
}

fn blk_mq_get_new_requests(q: *mut request_queue, plug: *mut blk_plug, bio: *mut bio) -> *mut request {
let mut data: blk_mq_alloc_data = {
let mut __t0: blk_mq_alloc_data = unsafe { core::mem::zeroed() };
__t0.q = q;
__t0.flags = 0u32;
__t0.shallow_depth = 0u32;
__t0.cmd_flags = unsafe { (*bio).bi_opf };
__t0.rq_flags = 0u32;
__t0.nr_tags = 1u32;
__t0.cached_rqs = core::ptr::null_mut::<rq_list>();
__t0.ctx = core::ptr::null_mut::<blk_mq_ctx>();
__t0.hctx = core::ptr::null_mut::<blk_mq_hw_ctx>();
__t0
};
rq_qos_throttle(q, bio);
if !plug.is_null() {
data.nr_tags = (unsafe { (*plug).nr_ios }) as u32;
unsafe {
(*plug).nr_ios = 1;
}
data.cached_rqs = unsafe { core::ptr::addr_of_mut!((*plug).cached_rqs) };
}
let rq: *mut request = __blk_mq_alloc_requests(core::ptr::addr_of_mut!(data));
if unlikely__2!(rq.is_null()) != 0 {
rq_qos_cleanup(q, bio);
}
rq
}

/// Check if there is a suitable cached request and return it.
fn blk_mq_get_cached_request(plug: *mut blk_plug, q: *mut request_queue, opf: u32) -> *mut request {
let r#type: u32 = blk_mq_get_hctx_type(opf);
if plug.is_null() {
return core::ptr::null_mut::<request>();
}
let rq: *mut request = rq_list_peek(unsafe { core::ptr::addr_of_mut!((*plug).cached_rqs) });
if rq.is_null() || (unsafe { (*rq).q }) != q {
return core::ptr::null_mut::<request>();
}
if (r#type as u32) != (((unsafe { (*(*rq).mq_hctx).r#type }) as i32) as u32) && ((r#type as u32) != 1 || ((unsafe { (*(*rq).mq_hctx).r#type }) as i32) != 0) {
return core::ptr::null_mut::<request>();
}
rq_list_pop(unsafe { core::ptr::addr_of_mut!((*plug).cached_rqs) });
rq
}

fn bio_unaligned(bio: *const bio, q: *mut request_queue) -> bool {
let bs_mask: u32 = queue_logical_block_size(q as *const request_queue).wrapping_sub(1);
// .bi_sector of any zero sized bio need to be initialized
if (unsafe { (*bio).bi_iter.bi_size }) & bs_mask != 0 || (unsafe { (*bio).bi_iter.bi_sector }) << 9i32 & (bs_mask as u64) != 0 {
return true;
}
false
}

/// blk_mq_submit_bio - Create and send a request to block device.
/// @bio: Bio pointer.
///
/// Builds up a request structure from @q and @bio and send to the device. The
/// request may not be queued directly to hardware if:
/// * This request can be merged with another one
/// * We want to place request at plug queue for possible future merging
/// * There is an IO scheduler active at this queue
///
/// It will not queue the request if there is an error with the bio, or at the
/// request creation.
#[unsafe(no_mangle)]
extern "C" fn blk_mq_submit_bio(mut bio: *mut bio) {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut u8 = core::ptr::addr_of_mut!(ret) as *mut u8;
let mut rq: core::mem::MaybeUninit<*mut request> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut request = core::ptr::addr_of_mut!(rq) as *mut *mut request;
let mut nr_segs: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u32 = core::ptr::addr_of_mut!(nr_segs) as *mut u32;
let mut hctx: core::mem::MaybeUninit<*mut blk_mq_hw_ctx> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut blk_mq_hw_ctx = core::ptr::addr_of_mut!(hctx) as *mut *mut blk_mq_hw_ctx;
let mut integrity_action: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u32 = core::ptr::addr_of_mut!(integrity_action) as *mut u32;
let mut is_sync: i32 = unsafe { core::mem::zeroed() };
let mut plug: *mut blk_plug = unsafe { core::mem::zeroed() };
let mut q: *mut request_queue = unsafe { core::mem::zeroed() };
unsafe {
'__join_3: {
q = bdev_get_queue((*bio).bi_bdev);
plug = (*get_current()).plug;
is_sync = op_is_sync((*bio).bi_opf) as i32;
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
// If the plug has a cached request for this queue, try to use it.
*__slate_slot_3 = blk_mq_get_cached_request(plug, q, (*bio).bi_opf);
// A BIO that was released from a zone write plug has already been
// through the preparation in this function, already holds a reference
// on the queue usage counter, and is the only write BIO in-flight for
// the target zone. Go straight to preparing a request for it.
if bio_zone_write_plugging(bio) {
*__slate_slot_4 = (*bio).__slate_anon_12.__bi_nr_segments;
if !(*__slate_slot_3).is_null() {
blk_queue_exit(q);
}
} else {
// The cached request already holds a q_usage_counter reference and we
// don't have to acquire a new one if we use it.
if (*__slate_slot_3).is_null() {
if unlikely!(bio_queue_enter(bio)) != 0 {
return;
}
}
// Device reconfiguration may change logical block size or reduce the
// number of poll queues, so the checks for alignment and poll support
// have to be done with queue usage counter held.
if unlikely__2!(bio_unaligned(bio as *const bio, q)) != 0 {
bio_io_error(bio);
break '__join_3;
} else {
if (*bio).bi_opf & 4194304u32 != 0 {
__v1 = !blk_mq_can_poll(q);
} else {
__v1 = false;
}
if __v1 {
bio_endio_status(bio, 1u8);
break '__join_3;
} else {
bio = __bio_split_to_limits(bio, core::ptr::addr_of_mut!((*q).limits) as *const queue_limits, core::ptr::addr_of_mut!(*__slate_slot_4));
if bio.is_null() {
break '__join_3;
} else {
*__slate_slot_6 = bio_integrity_action(bio);
if *__slate_slot_6 != 0 {
bio_integrity_prep(bio, *__slate_slot_6);
}
blk_mq_bio_issue_init(q, bio);
if blk_mq_attempt_bio_merge(q, bio, *__slate_slot_4) {
break '__join_3;
} else {
if bio_needs_zone_write_plugging(bio) {
if blk_zone_plug_bio(bio, *__slate_slot_4) {
break '__join_3;
}
}
}
}
}
}
}
if !(*__slate_slot_3).is_null() {
rq_qos_throttle((*(*__slate_slot_3)).q, bio);
(*(*__slate_slot_3)).cmd_flags = (*bio).bi_opf;
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*(*__slate_slot_3)).__slate_anon_12.queuelist));
blk_mq_rq_late_init(*__slate_slot_3, blk_time_get_ns());
} else {
*__slate_slot_3 = blk_mq_get_new_requests(q, plug, bio);
if unlikely__2!((*__slate_slot_3).is_null()) != 0 {
if (*bio).bi_opf & 2097152u32 != 0 {
bio_wouldblock_error(bio);
break '__join_3;
} else {
break '__join_3;
}
}
}
trace_block_getrq(bio);
rq_qos_track(q, *__slate_slot_3, bio);
blk_mq_bio_to_request(*__slate_slot_3, bio, *__slate_slot_4);
*__slate_slot_2 = blk_crypto_rq_get_keyslot(*__slate_slot_3);
if (*__slate_slot_2 as i32) != 0 {
bio_endio_status(bio, *__slate_slot_2);
blk_mq_free_request(*__slate_slot_3);
return;
} else {
if bio_zone_write_plugging(bio) {
blk_zone_write_plug_init_request(*__slate_slot_3);
}
if op_is_flush((*bio).bi_opf) {
__v0 = blk_insert_flush(*__slate_slot_3);
} else {
__v0 = false;
}
if __v0 {
return;
} else {
if !plug.is_null() {
blk_add_rq_to_plug(plug, *__slate_slot_3);
return;
} else {
*__slate_slot_5 = (*(*__slate_slot_3)).mq_hctx;
if (*(*__slate_slot_3)).rq_flags & 32u32 != 0 || (*(*__slate_slot_5)).dispatch_busy != 0 && ((*q).nr_hw_queues == 1 || is_sync == 0) {
blk_mq_insert_request(*__slate_slot_3, 0u32);
blk_mq_run_hw_queue(*__slate_slot_5, true);
} else {
blk_mq_run_dispatch_ops!(q, blk_mq_try_issue_directly(*__slate_slot_5, *__slate_slot_3));
}
return;
}
}
}
}
if (*__slate_slot_3).is_null() {
blk_queue_exit(q);
} else {
rq_list_add_head(core::ptr::addr_of_mut!((*plug).cached_rqs), *__slate_slot_3);
}
}
}

/// blk_insert_cloned_request - Helper for stacking drivers to submit a request
/// @rq: the request being queued
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_insert_cloned_request")]
extern "C" fn blk_insert_cloned_request(rq: *mut request) -> u8 {
let q: *mut request_queue = unsafe { (*rq).q };
let max_sectors: u32 = blk_queue_get_max_sectors(rq);
let max_segments: u32 = blk_rq_get_max_segments(rq);
let mut ret: u8 = 0;
if blk_rq_sectors(rq as *const request) > max_sectors {
// SCSI device does not have a good way to return if
// Write Same/Zero is actually supported. If a device rejects
// a non-read/write command (discard, write same,etc.) the
// low-level device driver will set the relevant queue limit to
// 0 to prevent blk-lib from issuing more of the offending
// operations. Commands queued prior to the queue limit being
// reset need to be completed with BLK_STS_NOTSUPP to avoid I/O
// errors being propagated to upper layers.
if max_sectors == 0 {
return 1;
}
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x013%s: over max size limit. (%u > %u)\n".as_ptr() as *const u8, c"blk_insert_cloned_request".as_ptr() as *mut u8, blk_rq_sectors(rq as *const request), max_sectors) };
return 10;
}
// The queue settings related to segment counting may differ from the
// original queue.
unsafe {
(*rq).nr_phys_segments = blk_recalc_rq_segments(rq) as u16;
}
if (((unsafe { (*rq).nr_phys_segments }) as i32) as u32) > max_segments {
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x013%s: over max segments limit. (%u > %u)\n".as_ptr() as *const u8, c"blk_insert_cloned_request".as_ptr() as *mut u8, (*rq).nr_phys_segments as i32, max_segments) };
return 10;
}
// Integrity segment counting depends on the same queue limits
// (virt_boundary_mask, seg_boundary_mask, max_segment_size) that
// vary across stacked queues, so recompute against the bottom
// queue just like nr_phys_segments above.
if blk_integrity_rq(rq as *const request) && !unsafe { (*rq).bio }.is_null() {
let max_int_segs: u16 = queue_max_integrity_segments(q as *const request_queue);
unsafe {
(*rq).nr_integrity_segments = blk_rq_count_integrity_sg((*rq).q, (*rq).bio) as u16;
}
if ((unsafe { (*rq).nr_integrity_segments }) as i32) > (max_int_segs as i32) {
let mut __v4: i32;
while false {
}
__v4 = unsafe { _printk(c"\x013%s: over max integrity segments limit. (%u > %u)\n".as_ptr() as *const u8, c"blk_insert_cloned_request".as_ptr() as *mut u8, (*rq).nr_integrity_segments as i32, max_int_segs as i32) };
return 10;
}
}
let __v6: bool = !unsafe { (*q).disk }.is_null() && should_fail_request(unsafe { (*(*q).disk).part0 }, blk_rq_bytes(rq as *const request));
if __v6 {
return 10;
}
ret = blk_crypto_rq_get_keyslot(rq);
if (ret as i32) != 0 {
return ret;
}
blk_account_io_start(rq);
// Since we have a scheduler attached on the top device,
// bypass a potential scheduler on the bottom device for
// insert.
blk_mq_run_dispatch_ops__3!(q, rq, ret);
if ret != 0 {
blk_account_io_done(rq, blk_time_get_ns());
}
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_insert_cloned_request_983: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request) -> u8>, *mut ()>(Some(blk_insert_cloned_request)) };

/// blk_rq_unprep_clone - Helper function to free all bios in a cloned request
/// @rq: the clone request to be cleaned up
///
/// Description:
///     Free all bios in @rq for a cloned request.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_rq_unprep_clone")]
extern "C" fn blk_rq_unprep_clone(rq: *mut request) {
loop {
let __v0: *mut bio = unsafe { (*rq).bio };
let bio: *mut bio = __v0;
if __v0.is_null() {
break;
}
unsafe {
(*rq).bio = (*bio).bi_next;
}
unsafe { bio_put(bio) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_rq_unprep_clone_984: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request)>, *mut ()>(Some(blk_rq_unprep_clone)) };

/// blk_rq_prep_clone - Helper function to setup clone request
/// @rq: the request to be setup
/// @rq_src: original request to be cloned
/// @bs: bio_set that bios for clone are allocated from
/// @gfp_mask: memory allocation mask for bio
/// @bio_ctr: setup function to be called for each clone bio.
///           Returns %0 for success, non %0 for failure.
/// @data: private data to be passed to @bio_ctr
///
/// Description:
///     Clones bios in @rq_src to @rq, and copies attributes of @rq_src to @rq.
///     Also, pages which the original bios are pointing to are not copied
///     and the cloned bios just point same pages.
///     So cloned bios must be completed before original bios, which means
///     the caller must complete @rq before @rq_src.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_rq_prep_clone")]
extern "C" fn blk_rq_prep_clone(rq: *mut request, rq_src: *mut request, mut bs: *mut bio_set, gfp_mask: u32, bio_ctr: Option<unsafe extern "C" fn(*mut bio, *mut bio, *mut ()) -> i32>, data: *mut ()) -> i32 {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut __v5: bool = unsafe { core::mem::zeroed() };
let mut bio: *mut bio = unsafe { core::mem::zeroed() };
let mut bio_src: core::mem::MaybeUninit<*mut bio> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut bio = core::ptr::addr_of_mut!(bio_src) as *mut *mut bio;
unsafe {
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
if bs.is_null() {
bs = core::ptr::addr_of_mut!(fs_bio_set);
}
'__join_0: {
'__join_7: {
if !unsafe { (*rq_src).bio }.is_null() {
*__slate_slot_7 = (*rq_src).bio;
loop {
if !(*__slate_slot_7).is_null() {
bio = bio_alloc_clone((*(*(*rq).q).disk).part0, *__slate_slot_7, gfp_mask, bs);
if bio.is_null() {
break '__join_0;
} else {
if bio_ctr != None {
__v5 = bio_ctr.unwrap()(bio, *__slate_slot_7, data) != 0;
} else {
__v5 = false;
}
if __v5 {
break;
} else {
if !unsafe { (*rq).bio }.is_null() {
(*(*rq).biotail).bi_next = bio;
(*rq).biotail = bio;
} else {
let __v4: *mut bio = bio;
(*rq).biotail = __v4;
(*rq).bio = __v4;
}
*__slate_slot_7 = (*(*__slate_slot_7)).bi_next;
}
}
} else {
break '__join_7;
}
}
bio_put(bio);
break '__join_0;
}
}
// Copy attributes of the original request to the clone request.
(*rq).__sector = blk_rq_pos(rq_src as *const request);
(*rq).__data_len = blk_rq_bytes(rq_src as *const request);
if (*rq_src).rq_flags & 4096u32 != 0 {
(*rq).rq_flags = (*rq).rq_flags | 4096u32;
(*rq).__slate_anon_25.special_vec = (*rq_src).__slate_anon_25.special_vec;
}
(*rq).nr_phys_segments = (*rq_src).nr_phys_segments;
(*rq).nr_integrity_segments = (*rq_src).nr_integrity_segments;
(*rq).phys_gap_bit = (*rq_src).phys_gap_bit;
if !unsafe { (*rq).bio }.is_null() {
__v0 = blk_crypto_rq_bio_prep(rq, (*rq).bio, gfp_mask) < 0;
} else {
__v0 = false;
}
if !__v0 {
return 0;
}
}
blk_rq_unprep_clone(rq);
return 12i32.wrapping_neg();
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_rq_prep_clone_985: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request, *mut request, *mut bio_set, u32, Option<unsafe extern "C" fn(*mut bio, *mut bio, *mut ()) -> i32>, *mut ()) -> i32>, *mut ()>(Some(blk_rq_prep_clone)) };

/// Steal bios from a request and add them to a bio list.
/// The request must not have been partially completed before.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_steal_bios")]
extern "C" fn blk_steal_bios(list: *mut bio_list, rq: *mut request) {
let mut bio: *mut bio = unsafe { core::mem::zeroed() };
bio = unsafe { (*rq).bio };
while !bio.is_null() {
if (unsafe { (*bio).bi_opf }) & 4194304u32 != 0 {
let __v2: u32 = (unsafe { (*bio).bi_opf }) & 4290772991u32;
unsafe {
(*bio).bi_opf = __v2;
}
unsafe {
(*bio).__slate_anon_12.bi_cookie = 1u32.wrapping_neg();
}
}
// The alternate request queue that we may end up submitting
// the bio to may be frozen temporarily, in this case REQ_NOWAIT
// will fail the I/O immediately with EAGAIN to the issuer.
// We are not in the issuer context which cannot block. Clear
// the flag to avoid spurious EAGAIN I/O failures.
let __v5: u32 = (unsafe { (*bio).bi_opf }) & 4292870143u32;
unsafe {
(*bio).bi_opf = __v5;
}
bio_clear_flag(bio, 8u32);
bio_clear_flag(bio, 9u32);
bio = unsafe { (*bio).bi_next };
}
if !unsafe { (*rq).bio }.is_null() {
if !unsafe { (*list).tail }.is_null() {
unsafe {
(*(*list).tail).bi_next = (*rq).bio;
}
} else {
unsafe {
(*list).head = (*rq).bio;
}
}
unsafe {
(*list).tail = (*rq).biotail;
}
unsafe {
(*rq).bio = core::ptr::null_mut::<bio>();
}
unsafe {
(*rq).biotail = core::ptr::null_mut::<bio>();
}
}
unsafe {
(*rq).__data_len = 0;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_steal_bios_986: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut bio_list, *mut request)>, *mut ()>(Some(blk_steal_bios)) };

fn order_to_size(order: u32) -> u64 {
_AC!() << 12i32 << order
}

/// called before freeing request pool in @tags
fn blk_mq_clear_rq_mapping(drv_tags: *mut blk_mq_tags, tags: *mut blk_mq_tags) {
let mut page: *mut page = unsafe { core::mem::zeroed() };
// There is no need to clear mapping if driver tags is not initialized
// or the mapping belongs to the driver tags.
if drv_tags.is_null() || drv_tags == tags {
return;
}
page = list_first_entry__14!(unsafe { core::ptr::addr_of_mut!((*tags).page_list) });
while list_entry_is_head__3!(page, unsafe { core::ptr::addr_of_mut!((*tags).page_list) }, lru) == 0 {
let start: u64 = page_address!(page) as u64;
let end: u64 = start.wrapping_add(order_to_size((unsafe { (*page).__slate_anon_1.__slate_anon_0.private }) as u32));
let mut i: i32 = 0;
while (i as u32) < unsafe { (*drv_tags).nr_tags } {
let rq: *mut request = unsafe { *unsafe { (*drv_tags).rqs }.wrapping_offset(i as isize) };
let rq_addr: u64 = rq as u64;
if rq_addr >= start && rq_addr < end {
let mut __v1: i64;
let __ret_warn_on: i32 = (req_ref_read(rq) != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("987: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 987b - ., 3; .popsection\n# {0}", const 987 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_blk_mq_clear_rq_mapping_2, sym __slate_asm_string_blk_mq_clear_rq_mapping_3, const 3470 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("988: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 988b - ., 4; .popsection\n# {0}", const 988 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
let mut __v6: *mut request;
let __ai_ptr: *mut *mut request = unsafe { (*drv_tags).rqs }.wrapping_offset(i as isize);
kcsan_mb!();
instrument_atomic_read_write(__ai_ptr as *const (), core::mem::size_of::<*mut request>() as u64);
let mut __v8: *mut request;
let mut __ret: *mut request = unsafe { core::mem::zeroed() };
let mut __old: *mut request = rq;
let mut __new: *mut request = core::ptr::null_mut::<request>();
match (const { core::mem::size_of::<*mut request>() }) as u64 {
1 => {
let mut __ptr: *mut u8 = __ai_ptr as *mut u8;
unsafe {
let __t0: u64;
core::arch::asm!("lock cmpxchgb {1:r}, ({0:r})", in(reg) __ptr, in(reg) __new as u64, inlateout("ax") __old as u64 => __t0, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, *mut request>(__t0 as u64) };
}
}
2 => {
let mut __ptr: *mut u16 = __ai_ptr as *mut u16;
unsafe {
let __t1: u64;
core::arch::asm!("lock cmpxchgw {1:r}, ({0:r})", in(reg) __ptr, in(reg) __new as u64, inlateout("ax") __old as u64 => __t1, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, *mut request>(__t1 as u64) };
}
}
4 => {
let mut __ptr: *mut u32 = __ai_ptr as *mut u32;
unsafe {
let __t2: u64;
core::arch::asm!("lock cmpxchgl {1:r}, ({0:r})", in(reg) __ptr, in(reg) __new as u64, inlateout("ax") __old as u64 => __t2, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, *mut request>(__t2 as u64) };
}
}
8 => {
let mut __ptr: *mut u64 = __ai_ptr as *mut u64;
unsafe {
let __t3: u64;
core::arch::asm!("lock cmpxchgq {1:r}, ({0:r})", in(reg) __ptr, in(reg) __new as u64, inlateout("ax") __old as u64 => __t3, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, *mut request>(__t3 as u64) };
}
}
_ => {
unsafe { __cmpxchg_wrong_size() };
}
}
__v8 = __ret;
__v6 = __v8;
}
i = i.wrapping_add(1);
}
page = list_next_entry__14!(page, lru);
}
}

#[unsafe(no_mangle)]
extern "C" fn blk_mq_free_rqs(set: *mut blk_mq_tag_set, tags: *mut blk_mq_tags, hctx_idx: u32) {
let mut drv_tags: *mut blk_mq_tags = unsafe { core::mem::zeroed() };
if list_empty((unsafe { core::ptr::addr_of_mut!((*tags).page_list) }) as *const list_head) != 0 {
return;
}
if blk_mq_is_shared_tags(unsafe { (*set).flags }) {
drv_tags = unsafe { (*set).shared_tags };
} else {
drv_tags = unsafe { *unsafe { (*set).tags }.wrapping_offset(hctx_idx as isize) };
}
if !unsafe { (*tags).static_rqs }.is_null() && (unsafe { (*(*set).ops).exit_request }) != None {
let mut i: i32 = 0;
while (i as u32) < unsafe { (*tags).nr_tags } {
let rq: *mut request = unsafe { *unsafe { (*tags).static_rqs }.wrapping_offset(i as isize) };
if !rq.is_null() {
unsafe { unsafe { (*(*set).ops).exit_request }.unwrap()(set, rq, hctx_idx) };
unsafe {
*unsafe { (*tags).static_rqs }.wrapping_offset(i as isize) = core::ptr::null_mut::<request>();
}
}
i = i.wrapping_add(1);
}
}
blk_mq_clear_rq_mapping(drv_tags, tags);
// Free request pages in SRCU callback, which is called from
// blk_mq_free_tags().
}

#[unsafe(no_mangle)]
extern "C" fn blk_mq_free_rq_map(set: *mut blk_mq_tag_set, tags: *mut blk_mq_tags) {
unsafe { kfree((*tags).rqs as *const ()) };
unsafe {
(*tags).rqs = core::ptr::null_mut::<*mut request>();
}
unsafe { kfree((*tags).static_rqs as *const ()) };
unsafe {
(*tags).static_rqs = core::ptr::null_mut::<*mut request>();
}
unsafe { blk_mq_free_tags(set, tags) };
}

fn hctx_idx_to_type(set: *mut blk_mq_tag_set, hctx_idx: u32) -> u32 {
let mut i: i32 = 0;
while (i as u32) < unsafe { (*set).nr_maps } {
let start: u32 = unsafe { (*((*set).map.as_mut_ptr() as *mut blk_mq_queue_map).wrapping_offset(i as isize)).queue_offset };
let end: u32 = start.wrapping_add(unsafe { (*((*set).map.as_mut_ptr() as *mut blk_mq_queue_map).wrapping_offset(i as isize)).nr_queues });
if hctx_idx >= start && hctx_idx < end {
break;
}
i = i.wrapping_add(1);
}
if (i as u32) >= unsafe { (*set).nr_maps } {
i = 0;
}
i as u32
}

fn blk_mq_get_hctx_node(set: *mut blk_mq_tag_set, hctx_idx: u32) -> i32 {
let r#type: u32 = hctx_idx_to_type(set, hctx_idx);
unsafe { blk_mq_hw_queue_to_node(((*set).map.as_mut_ptr() as *mut blk_mq_queue_map).wrapping_offset((r#type as u32) as isize), hctx_idx) }
}

fn blk_mq_alloc_rq_map(set: *mut blk_mq_tag_set, hctx_idx: u32, nr_tags: u32, reserved_tags: u32) -> *mut blk_mq_tags {
let mut __slate_compound_10: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_12: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_13: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_15: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut tags: core::mem::MaybeUninit<*mut blk_mq_tags> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut blk_mq_tags = core::ptr::addr_of_mut!(tags) as *mut *mut blk_mq_tags;
let mut node: i32 = unsafe { core::mem::zeroed() };
unsafe {
{
node = blk_mq_get_hctx_node(set, hctx_idx);
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
if node == 1i32.wrapping_neg() {
node = (*set).numa_node;
}
}
*__slate_slot_8 = blk_mq_init_tags(nr_tags, reserved_tags, (*set).flags, node);
if (*__slate_slot_8).is_null() {
return core::ptr::null_mut::<blk_mq_tags>();
} else {
let __slate_auto_bytes_2 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_4 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_5 = _kmalloc_array_node_noprof(nr_tags as u64, core::mem::size_of::<*mut request>() as u64, 77056u32, node, *{
__slate_compound_13 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_13)
});
alloc_tag_restore!();
} else {
*__slate_slot_5 = _kmalloc_array_node_noprof(nr_tags as u64, core::mem::size_of::<*mut request>() as u64, 77056u32, node, *{
__slate_compound_15 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_15)
});
}
(*(*__slate_slot_8)).rqs = *__slate_slot_5 as *mut *mut request;
if !unsafe { (*(*__slate_slot_8)).rqs }.is_null() {
let __slate_auto_bytes_1 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_0 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_1 = _kmalloc_array_node_noprof(nr_tags as u64, core::mem::size_of::<*mut request>() as u64, 77056u32, node, *{
__slate_compound_10 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_10)
});
alloc_tag_restore!();
} else {
*__slate_slot_1 = _kmalloc_array_node_noprof(nr_tags as u64, core::mem::size_of::<*mut request>() as u64, 77056u32, node, *{
__slate_compound_12 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_12)
});
}
(*(*__slate_slot_8)).static_rqs = *__slate_slot_1 as *mut *mut request;
if unsafe { (*(*__slate_slot_8)).static_rqs }.is_null() {
kfree((*(*__slate_slot_8)).rqs as *const ());
} else {
return *__slate_slot_8;
}
}
blk_mq_free_tags(set, *__slate_slot_8);
return core::ptr::null_mut::<blk_mq_tags>();
}
}
}

fn blk_mq_init_request(set: *mut blk_mq_tag_set, rq: *mut request, hctx_idx: u32, node: i32) -> i32 {
let mut ret: i32 = 0;
if (unsafe { (*(*set).ops).init_request }) != None {
ret = unsafe { unsafe { (*(*set).ops).init_request }.unwrap()(set, rq, hctx_idx, node) };
if ret != 0 {
return ret;
}
}
loop {
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_989() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__4!((*rq).state, 0i32);
if true {
break;
}
}
0
}

fn blk_mq_alloc_rqs(set: *mut blk_mq_tag_set, tags: *mut blk_mq_tags, hctx_idx: u32, depth: u32) -> i32 {
let mut __UNIQUE_ID_y__991: u32 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__990: u32 = unsafe { core::mem::zeroed() };
let mut __v14: bool = unsafe { core::mem::zeroed() };
let mut p: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut *mut () = core::ptr::addr_of_mut!(p) as *mut *mut ();
let mut to_do: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut i32 = core::ptr::addr_of_mut!(to_do) as *mut i32;
let mut page: core::mem::MaybeUninit<*mut page> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut *mut page = core::ptr::addr_of_mut!(page) as *mut *mut page;
let mut this_order: i32 = unsafe { core::mem::zeroed() };
let mut left: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut u64 = core::ptr::addr_of_mut!(left) as *mut u64;
let mut rq_size: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut u64 = core::ptr::addr_of_mut!(rq_size) as *mut u64;
let mut node: i32 = unsafe { core::mem::zeroed() };
let mut max_order: u32 = unsafe { core::mem::zeroed() };
let mut entries_per_page: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_25: *mut u32 = core::ptr::addr_of_mut!(entries_per_page) as *mut u32;
let mut j: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_26: *mut u32 = core::ptr::addr_of_mut!(j) as *mut u32;
let mut i: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_27: *mut u32 = core::ptr::addr_of_mut!(i) as *mut u32;
unsafe {
{
let __slate_auto_bytes_2 = __slate_slot_27 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_26 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_25 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
max_order = 4;
node = blk_mq_get_hctx_node(set, hctx_idx);
let __slate_auto_bytes_5 = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
if node == 1i32.wrapping_neg() {
node = (*set).numa_node;
}
}
// rq_size is the size of the request plus driver payload, rounded
// to the cacheline size
*__slate_slot_22 = round_up__3!((core::mem::size_of::<request>() as u64).wrapping_add((*set).cmd_size as u64), cache_line_size!());
*__slate_slot_21 = (*__slate_slot_22).wrapping_mul(depth as u64);
*__slate_slot_27 = 0;
'__join_0: {
'__join_5: {
'__loop_2: while *__slate_slot_27 < depth {
this_order = max_order as i32;
let __slate_auto_bytes_6 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
loop {
if this_order != 0 {
__v14 = *__slate_slot_21 < order_to_size(this_order.wrapping_sub(1) as u32);
} else {
__v14 = false;
}
if __v14 {
this_order = this_order.wrapping_sub(1);
} else {
break;
}
}
loop {
*__slate_slot_19 = alloc_hooks__3!(alloc_pages_node_noprof(node, 77056u32, this_order as u32));
if !(*__slate_slot_19).is_null() {
break;
} else {
let __v13: i32 = this_order;
this_order = __v13.wrapping_sub(1);
if __v13 == 0 {
break;
} else {
if order_to_size(this_order as u32) < *__slate_slot_22 {
break;
} else {
if false {
break;
}
}
}
}
}
if (*__slate_slot_19).is_null() {
break '__join_0;
} else {
(*(*__slate_slot_19)).__slate_anon_1.__slate_anon_0.private = this_order as u64;
list_add_tail(core::ptr::addr_of_mut!((*(*__slate_slot_19)).__slate_anon_1.__slate_anon_0.__slate_anon_0.lru), core::ptr::addr_of_mut!((*tags).page_list));
*__slate_slot_17 = page_address!(*__slate_slot_19);
// Allow kmemleak to scan these pages as they contain pointers
// to additional allocations like via ops->init_request().
kmemleak_alloc(*__slate_slot_17 as *const (), order_to_size(this_order as u32), 1i32, (BIT!(10i32) | BIT!(11i32)) as u32);
*__slate_slot_25 = (order_to_size(this_order as u32) / *__slate_slot_22) as u32;
__UNIQUE_ID_x__990 = *__slate_slot_25;
__UNIQUE_ID_y__991 = depth.wrapping_sub(*__slate_slot_27);
while false {
}
*__slate_slot_18 = (if __UNIQUE_ID_x__990 < __UNIQUE_ID_y__991 { __UNIQUE_ID_x__990 } else { __UNIQUE_ID_y__991 }) as i32;
let __v8: u64 = *__slate_slot_21;
*__slate_slot_21 = __v8.wrapping_sub((*__slate_slot_18 as u64).wrapping_mul(*__slate_slot_22));
*__slate_slot_26 = 0;
loop {
if *__slate_slot_26 < (*__slate_slot_18 as u32) {
let rq: *mut request = *__slate_slot_17 as *mut request;
*unsafe { (*tags).static_rqs }.wrapping_offset(*__slate_slot_27 as isize) = rq;
if blk_mq_init_request(set, rq, hctx_idx, node) != 0 {
break '__join_5;
} else {
*__slate_slot_17 = (*__slate_slot_17 as *const u8).wrapping_offset(*__slate_slot_22 as isize) as *mut ();
let __v3: u32 = *__slate_slot_27;
*__slate_slot_27 = __v3.wrapping_add(1);
let __v1: u32 = *__slate_slot_26;
*__slate_slot_26 = __v1.wrapping_add(1);
}
} else {
continue '__loop_2;
}
}
}
}
return 0;
}
*unsafe { (*tags).static_rqs }.wrapping_offset(*__slate_slot_27 as isize) = core::ptr::null_mut::<request>();
}
blk_mq_free_rqs(set, tags, hctx_idx);
return 12i32.wrapping_neg();
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rq_iter_data {
    hctx: *mut blk_mq_hw_ctx,
    has_rq: bool,
}


#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_has_request")]
extern "C" fn blk_mq_has_request(rq: *mut request, data: *mut ()) -> bool {
if (unsafe { (*rq).mq_hctx }) != unsafe { (*(data as *mut rq_iter_data)).hctx } {
return true;
}
unsafe {
(*(data as *mut rq_iter_data)).has_rq = true;
}
false
}

fn blk_mq_hctx_has_requests(hctx: *mut blk_mq_hw_ctx) -> bool {
let tags: *mut blk_mq_tags = if !unsafe { (*hctx).sched_tags }.is_null() { unsafe { (*hctx).sched_tags } } else { unsafe { (*hctx).tags } };
let mut data: rq_iter_data = {
let mut __t0: rq_iter_data = unsafe { core::mem::zeroed() };
__t0.hctx = hctx;
__t0
};
let srcu_idx: i32 = srcu_read_lock(unsafe { core::ptr::addr_of_mut!((*(*(*hctx).queue).tag_set).tags_srcu) });
unsafe { blk_mq_all_tag_iter(tags, Some(blk_mq_has_request), core::ptr::addr_of_mut!(data) as *mut ()) };
srcu_read_unlock(unsafe { core::ptr::addr_of_mut!((*(*(*hctx).queue).tag_set).tags_srcu) }, srcu_idx);
data.has_rq
}

fn blk_mq_hctx_has_online_cpu(hctx: *mut blk_mq_hw_ctx, this_cpu: u32) -> bool {
let r#type: u32 = (unsafe { (*hctx).r#type }) as u32;
// hctx->cpumask has to rule out isolated CPUs, but userspace still
// might submit IOs on these isolated CPUs, so use the queue map to
// check if all CPUs mapped to this hctx are offline
let mut cpu: i32 = 0;
loop {
cpu = find_next_bit(cpumask_bits!((unsafe { core::ptr::addr_of_mut!(__cpu_online_mask) }) as *const cpumask), 64u64, cpu as u64) as i32;
if (cpu as u32) >= 64 {
break;
}
let h: *mut blk_mq_hw_ctx = blk_mq_map_queue_type(unsafe { (*hctx).queue }, r#type, cpu as u32);
if h == hctx {
// this hctx has at least one online CPU
if this_cpu != (cpu as u32) {
return true;
}
}
cpu = cpu.wrapping_add(1);
}
false
}

#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_hctx_notify_offline")]
extern "C" fn blk_mq_hctx_notify_offline(cpu: u32, node: *mut hlist_node) -> i32 {
let hctx: *mut blk_mq_hw_ctx = hlist_entry_safe__2!(node, blk_mq_hw_ctx);
let mut ret: i32 = 0;
let __v4: bool = (unsafe { (*hctx).nr_ctx }) == 0 || blk_mq_hctx_has_online_cpu(hctx, cpu);
if __v4 {
return 0;
}
// Prevent new request from being allocated on the current hctx.
//
// The smp_mb__after_atomic() Pairs with the implied barrier in
// test_and_set_bit_lock in sbitmap_get().  Ensures the inactive flag is
// seen once we return from the tag allocator.
set_bit(3i64, (unsafe { core::ptr::addr_of_mut!((*hctx).__slate_anon_0.state) }) as *mut u64);
smp_mb__after_atomic!();
// Try to grab a reference to the queue and wait for any outstanding
// requests.  If we could not grab a reference the queue has been
// frozen and there are no requests.
if percpu_ref_tryget(unsafe { core::ptr::addr_of_mut!((*(*hctx).queue).q_usage_counter) }) {
while blk_mq_hctx_has_requests(hctx) {
// The wakeup capable IRQ handler of block device is
// not called during suspend. Skip the loop by checking
// pm_wakeup_pending to prevent the deadlock and improve
// suspend latency.
if unsafe { pm_wakeup_pending() } {
clear_bit(3i64, (unsafe { core::ptr::addr_of_mut!((*hctx).__slate_anon_0.state) }) as *mut u64);
ret = 16i32.wrapping_neg();
break;
}
unsafe { msleep(5u32) };
}
percpu_ref_put(unsafe { core::ptr::addr_of_mut!((*(*hctx).queue).q_usage_counter) });
}
ret
}

/// Check if one CPU is mapped to the specified hctx
///
/// Isolated CPUs have been ruled out from hctx->cpumask, which is supposed
/// to be used for scheduling kworker only. For other usage, please call this
/// helper for checking if one CPU belongs to the specified hctx
fn blk_mq_cpu_mapped_to_hctx(cpu: u32, hctx: *const blk_mq_hw_ctx) -> bool {
let mapped_hctx: *mut blk_mq_hw_ctx = blk_mq_map_queue_type(unsafe { (*hctx).queue }, (unsafe { (*hctx).r#type }) as u32, cpu);
mapped_hctx == (hctx as *mut blk_mq_hw_ctx)
}

#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_hctx_notify_online")]
extern "C" fn blk_mq_hctx_notify_online(cpu: u32, node: *mut hlist_node) -> i32 {
let hctx: *mut blk_mq_hw_ctx = hlist_entry_safe__2!(node, blk_mq_hw_ctx);
if blk_mq_cpu_mapped_to_hctx(cpu, hctx as *const blk_mq_hw_ctx) {
clear_bit(3i64, (unsafe { core::ptr::addr_of_mut!((*hctx).__slate_anon_0.state) }) as *mut u64);
}
0
}

/// 'cpu' is going away. splice any existing rq_list entries from this
/// software queue to the hw queue dispatch list, and ensure that it
/// gets run.
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_hctx_notify_dead")]
extern "C" fn blk_mq_hctx_notify_dead(cpu: u32, node: *mut hlist_node) -> i32 {
let mut hctx: *mut blk_mq_hw_ctx = unsafe { core::mem::zeroed() };
let mut ctx: *mut blk_mq_ctx = unsafe { core::mem::zeroed() };
let mut tmp: list_head = unsafe { core::mem::zeroed() };
tmp = list_head { next: core::ptr::addr_of_mut!(tmp), prev: core::ptr::addr_of_mut!(tmp) };
let mut r#type: u32 = 0;
hctx = hlist_entry_safe__3!(node, blk_mq_hw_ctx);
if !blk_mq_cpu_mapped_to_hctx(cpu, hctx as *const blk_mq_hw_ctx) {
return 0;
}
ctx = __blk_mq_get_ctx(unsafe { (*hctx).queue }, cpu);
r#type = (unsafe { (*hctx).r#type }) as u32;
spin_lock(unsafe { core::ptr::addr_of_mut!((*ctx).__slate_anon_0.lock) });
if list_empty(unsafe { (*ctx).__slate_anon_0.rq_lists.as_mut_ptr() as *mut list_head }.wrapping_offset((r#type as u32) as isize) as *const list_head) == 0 {
list_splice_init(unsafe { (*ctx).__slate_anon_0.rq_lists.as_mut_ptr() as *mut list_head }.wrapping_offset((r#type as u32) as isize), core::ptr::addr_of_mut!(tmp));
blk_mq_hctx_clear_pending(hctx, ctx);
}
spin_unlock(unsafe { core::ptr::addr_of_mut!((*ctx).__slate_anon_0.lock) });
if list_empty(core::ptr::addr_of_mut!(tmp) as *const list_head) != 0 {
return 0;
}
spin_lock(unsafe { core::ptr::addr_of_mut!((*hctx).__slate_anon_0.lock) });
list_splice_tail_init(core::ptr::addr_of_mut!(tmp), unsafe { core::ptr::addr_of_mut!((*hctx).__slate_anon_0.dispatch) });
spin_unlock(unsafe { core::ptr::addr_of_mut!((*hctx).__slate_anon_0.lock) });
blk_mq_run_hw_queue(hctx, true);
0
}

fn __blk_mq_remove_cpuhp(hctx: *mut blk_mq_hw_ctx) {
lockdep_assert_held__2!(unsafe { core::ptr::addr_of_mut!(blk_mq_cpuhp_lock) });
let __v0: bool = (unsafe { (*hctx).flags }) & 4 == 0 && hlist_unhashed((unsafe { core::ptr::addr_of_mut!((*hctx).cpuhp_online) }) as *const hlist_node) == 0;
if __v0 {
cpuhp_state_remove_instance_nocalls(149i32, unsafe { core::ptr::addr_of_mut!((*hctx).cpuhp_online) });
INIT_HLIST_NODE(unsafe { core::ptr::addr_of_mut!((*hctx).cpuhp_online) });
}
if hlist_unhashed((unsafe { core::ptr::addr_of_mut!((*hctx).cpuhp_dead) }) as *const hlist_node) == 0 {
cpuhp_state_remove_instance_nocalls(24i32, unsafe { core::ptr::addr_of_mut!((*hctx).cpuhp_dead) });
INIT_HLIST_NODE(unsafe { core::ptr::addr_of_mut!((*hctx).cpuhp_dead) });
}
}

fn blk_mq_remove_cpuhp(hctx: *mut blk_mq_hw_ctx) {
unsafe { mutex_lock(core::ptr::addr_of_mut!(blk_mq_cpuhp_lock)) };
__blk_mq_remove_cpuhp(hctx);
unsafe { mutex_unlock(core::ptr::addr_of_mut!(blk_mq_cpuhp_lock)) };
}

fn __blk_mq_add_cpuhp(hctx: *mut blk_mq_hw_ctx) {
lockdep_assert_held__2!(unsafe { core::ptr::addr_of_mut!(blk_mq_cpuhp_lock) });
let __v0: bool = (unsafe { (*hctx).flags }) & 4 == 0 && hlist_unhashed((unsafe { core::ptr::addr_of_mut!((*hctx).cpuhp_online) }) as *const hlist_node) != 0;
if __v0 {
cpuhp_state_add_instance_nocalls(149i32, unsafe { core::ptr::addr_of_mut!((*hctx).cpuhp_online) });
}
if hlist_unhashed((unsafe { core::ptr::addr_of_mut!((*hctx).cpuhp_dead) }) as *const hlist_node) != 0 {
cpuhp_state_add_instance_nocalls(24i32, unsafe { core::ptr::addr_of_mut!((*hctx).cpuhp_dead) });
}
}

fn __blk_mq_remove_cpuhp_list(head: *mut list_head) {
let mut hctx: *mut blk_mq_hw_ctx = unsafe { core::mem::zeroed() };
lockdep_assert_held__2!(unsafe { core::ptr::addr_of_mut!(blk_mq_cpuhp_lock) });
hctx = list_first_entry__7!(head);
while list_entry_is_head!(hctx, head, hctx_list) == 0 {
__blk_mq_remove_cpuhp(hctx);
hctx = list_next_entry__6!(hctx, hctx_list);
}
}

/// Unregister cpuhp callbacks from exited hw queues
///
/// Safe to call if this `request_queue` is live
fn blk_mq_remove_hw_queues_cpuhp(q: *mut request_queue) {
let mut hctx_list: list_head = unsafe { core::mem::zeroed() };
hctx_list = list_head { next: core::ptr::addr_of_mut!(hctx_list), prev: core::ptr::addr_of_mut!(hctx_list) };
spin_lock(unsafe { core::ptr::addr_of_mut!((*q).unused_hctx_lock) });
list_splice_init(unsafe { core::ptr::addr_of_mut!((*q).unused_hctx_list) }, core::ptr::addr_of_mut!(hctx_list));
spin_unlock(unsafe { core::ptr::addr_of_mut!((*q).unused_hctx_lock) });
unsafe { mutex_lock(core::ptr::addr_of_mut!(blk_mq_cpuhp_lock)) };
__blk_mq_remove_cpuhp_list(core::ptr::addr_of_mut!(hctx_list));
unsafe { mutex_unlock(core::ptr::addr_of_mut!(blk_mq_cpuhp_lock)) };
spin_lock(unsafe { core::ptr::addr_of_mut!((*q).unused_hctx_lock) });
list_splice(core::ptr::addr_of_mut!(hctx_list) as *const list_head, unsafe { core::ptr::addr_of_mut!((*q).unused_hctx_list) });
spin_unlock(unsafe { core::ptr::addr_of_mut!((*q).unused_hctx_lock) });
}

/// Register cpuhp callbacks from all hw queues
///
/// Safe to call if this `request_queue` is live
fn blk_mq_add_hw_queues_cpuhp(q: *mut request_queue) {
let mut hctx: *mut blk_mq_hw_ctx = unsafe { core::mem::zeroed() };
let mut i: u64 = 0;
unsafe { mutex_lock(core::ptr::addr_of_mut!(blk_mq_cpuhp_lock)) };
loop {
let __v0: bool = if i < ((unsafe { (*q).nr_hw_queues }) as u64) {
let mut __v1: i32;
hctx = queue_hctx(q, i as i32);
__v1 = 1;
__v1 != 0
} else {
false
};
if !__v0 {
break;
}
__blk_mq_add_cpuhp(hctx);
i = i.wrapping_add(1);
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!(blk_mq_cpuhp_lock)) };
}

/// Before freeing hw queue, clearing the flush request reference in
/// tags->rqs[] for avoiding potential UAF.
fn blk_mq_clear_flush_rq_mapping(tags: *mut blk_mq_tags, queue_depth: u32, flush_rq: *mut request) {
let mut i: i32 = 0;
// The hw queue may not be mapped yet
if tags.is_null() {
return;
}
let mut __v0: i64;
let __ret_warn_on: i32 = (req_ref_read(flush_rq) != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("993: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 993b - ., 3; .popsection\n# {0}", const 993 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_blk_mq_clear_flush_rq_mapping_1, sym __slate_asm_string_blk_mq_clear_flush_rq_mapping_2, const 3938 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("994: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 994b - ., 4; .popsection\n# {0}", const 994 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
while (i as u32) < queue_depth {
let mut __v5: *mut request;
let __ai_ptr: *mut *mut request = unsafe { (*tags).rqs }.wrapping_offset(i as isize);
kcsan_mb!();
instrument_atomic_read_write(__ai_ptr as *const (), core::mem::size_of::<*mut request>() as u64);
let mut __v7: *mut request;
let mut __ret: *mut request = unsafe { core::mem::zeroed() };
let mut __old: *mut request = flush_rq;
let mut __new: *mut request = core::ptr::null_mut::<request>();
match (const { core::mem::size_of::<*mut request>() }) as u64 {
1 => {
let mut __ptr: *mut u8 = __ai_ptr as *mut u8;
unsafe {
let __t0: u64;
core::arch::asm!("lock cmpxchgb {1:r}, ({0:r})", in(reg) __ptr, in(reg) __new as u64, inlateout("ax") __old as u64 => __t0, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, *mut request>(__t0 as u64) };
}
}
2 => {
let mut __ptr: *mut u16 = __ai_ptr as *mut u16;
unsafe {
let __t1: u64;
core::arch::asm!("lock cmpxchgw {1:r}, ({0:r})", in(reg) __ptr, in(reg) __new as u64, inlateout("ax") __old as u64 => __t1, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, *mut request>(__t1 as u64) };
}
}
4 => {
let mut __ptr: *mut u32 = __ai_ptr as *mut u32;
unsafe {
let __t2: u64;
core::arch::asm!("lock cmpxchgl {1:r}, ({0:r})", in(reg) __ptr, in(reg) __new as u64, inlateout("ax") __old as u64 => __t2, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, *mut request>(__t2 as u64) };
}
}
8 => {
let mut __ptr: *mut u64 = __ai_ptr as *mut u64;
unsafe {
let __t3: u64;
core::arch::asm!("lock cmpxchgq {1:r}, ({0:r})", in(reg) __ptr, in(reg) __new as u64, inlateout("ax") __old as u64 => __t3, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, *mut request>(__t3 as u64) };
}
}
_ => {
unsafe { __cmpxchg_wrong_size() };
}
}
__v7 = __ret;
__v5 = __v7;
i = i.wrapping_add(1);
}
}

#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_free_flush_queue_callback")]
extern "C" fn blk_free_flush_queue_callback(head: *mut callback_head) {
let fq: *mut blk_flush_queue = container_of__3!(head, blk_flush_queue);
unsafe { blk_free_flush_queue(fq) };
}

/// hctx->ctxs will be freed in queue's release handler
fn blk_mq_exit_hctx(q: *mut request_queue, set: *mut blk_mq_tag_set, hctx: *mut blk_mq_hw_ctx, hctx_idx: u32) {
let flush_rq: *mut request = unsafe { (*(*hctx).fq).flush_rq };
if blk_mq_hw_queue_mapped(hctx) {
blk_mq_tag_idle(hctx);
}
if blk_queue_init_done!(q) != 0 {
blk_mq_clear_flush_rq_mapping(unsafe { *unsafe { (*set).tags }.wrapping_offset(hctx_idx as isize) }, unsafe { (*set).queue_depth }, flush_rq);
}
if (unsafe { (*(*set).ops).exit_request }) != None {
unsafe { unsafe { (*(*set).ops).exit_request }.unwrap()(set, flush_rq, hctx_idx) };
}
if (unsafe { (*(*set).ops).exit_hctx }) != None {
unsafe { unsafe { (*(*set).ops).exit_hctx }.unwrap()(hctx, hctx_idx) };
}
unsafe { call_srcu(core::ptr::addr_of_mut!((*set).tags_srcu), core::ptr::addr_of_mut!((*(*hctx).fq).callback_head), Some(blk_free_flush_queue_callback)) };
unsafe {
(*hctx).fq = core::ptr::null_mut::<blk_flush_queue>();
}
spin_lock(unsafe { core::ptr::addr_of_mut!((*q).unused_hctx_lock) });
list_add(unsafe { core::ptr::addr_of_mut!((*hctx).hctx_list) }, unsafe { core::ptr::addr_of_mut!((*q).unused_hctx_list) });
spin_unlock(unsafe { core::ptr::addr_of_mut!((*q).unused_hctx_lock) });
}

fn blk_mq_exit_hw_queues(q: *mut request_queue, set: *mut blk_mq_tag_set, nr_queue: i32) {
let mut hctx: *mut blk_mq_hw_ctx = unsafe { core::mem::zeroed() };
let mut i: u64 = 0;
loop {
let __v1: bool = if i < ((unsafe { (*q).nr_hw_queues }) as u64) {
let mut __v2: i32;
hctx = queue_hctx(q, i as i32);
__v2 = 1;
__v2 != 0
} else {
false
};
if !__v1 {
break;
}
if i == (nr_queue as u64) {
break;
}
blk_mq_remove_cpuhp(hctx);
blk_mq_exit_hctx(q, set, hctx, i as u32);
i = i.wrapping_add(1);
}
}

fn blk_mq_init_hctx(q: *mut request_queue, set: *mut blk_mq_tag_set, hctx: *mut blk_mq_hw_ctx, hctx_idx: u32) -> i32 {
let mut __v0: bool = unsafe { core::mem::zeroed() };
unsafe {
let gfp: u32 = ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(13i32) as u32) | (BIT!(16i32) as u32);
(*hctx).fq = blk_alloc_flush_queue((*hctx).numa_node, (*set).cmd_size as i32, gfp);
if !unsafe { (*hctx).fq }.is_null() {
(*hctx).queue_num = hctx_idx;
(*hctx).tags = *unsafe { (*set).tags }.wrapping_offset(hctx_idx as isize);
if (*(*set).ops).init_hctx != None {
__v0 = unsafe { (*(*set).ops).init_hctx }.unwrap()(hctx, (*set).driver_data, hctx_idx) != 0;
} else {
__v0 = false;
}
if !__v0 {
if blk_mq_init_request(set, (*(*hctx).fq).flush_rq, hctx_idx, (*hctx).numa_node) != 0 {
if (*(*set).ops).exit_hctx != None {
unsafe { (*(*set).ops).exit_hctx }.unwrap()(hctx, hctx_idx);
}
} else {
return 0;
}
}
blk_free_flush_queue((*hctx).fq);
(*hctx).fq = core::ptr::null_mut::<blk_flush_queue>();
}
return 1i32.wrapping_neg();
}
}

fn blk_mq_alloc_hctx(q: *mut request_queue, set: *mut blk_mq_tag_set, mut node: i32) -> *mut blk_mq_hw_ctx {
let mut __slate_compound_10: spinlock = unsafe { core::mem::zeroed() };
let mut __slate_compound_11: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_13: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_14: spinlock = unsafe { core::mem::zeroed() };
let mut __slate_compound_15: __SlateRecord_types_h_197 = unsafe { core::mem::zeroed() };
let mut __slate_compound_16: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_18: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut gfp: u32 = unsafe { core::mem::zeroed() };
let mut hctx: core::mem::MaybeUninit<*mut blk_mq_hw_ctx> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut blk_mq_hw_ctx = core::ptr::addr_of_mut!(hctx) as *mut *mut blk_mq_hw_ctx;
unsafe {
let __slate_auto_bytes_2 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
gfp = ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(13i32) as u32) | (BIT!(16i32) as u32);
let __slate_auto_bytes_3 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_4 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_5 = _kmalloc_node_noprof(core::mem::size_of::<blk_mq_hw_ctx>() as u64, gfp | 256u32, node, *{
__slate_compound_16 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_16)
});
alloc_tag_restore!();
} else {
*__slate_slot_5 = _kmalloc_node_noprof(core::mem::size_of::<blk_mq_hw_ctx>() as u64, gfp | 256u32, node, *{
__slate_compound_18 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_18)
});
}
*__slate_slot_9 = *__slate_slot_5 as *mut blk_mq_hw_ctx;
if !(*__slate_slot_9).is_null() {
if zalloc_cpumask_var_node(core::ptr::addr_of_mut!((*(*__slate_slot_9)).cpumask), gfp, node) {
atomic_set(core::ptr::addr_of_mut!((*(*__slate_slot_9)).nr_active), 0i32);
if node == 1i32.wrapping_neg() {
node = (*set).numa_node;
}
(*(*__slate_slot_9)).numa_node = node;
loop {
__init_work(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*(*__slate_slot_9)).run_work)).work), 0i32);
(*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*(*__slate_slot_9)).run_work)).work)).data = *{
__slate_compound_15 = __SlateRecord_types_h_197 { counter: 4503599625273344i64 };
core::ptr::addr_of_mut!(__slate_compound_15)
};
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*(*__slate_slot_9)).run_work)).work)).entry));
(*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*(*__slate_slot_9)).run_work)).work)).func = Some(blk_mq_run_work_fn);
if true {
if true {
__timer_init!(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*(*__slate_slot_9)).run_work)).timer), unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut timer_list)>>(delayed_work_timer_fn as *const ()) }, 2097152i32);
if true {
break;
}
}
}
}
loop {
spinlock_check(core::ptr::addr_of_mut!((*(*__slate_slot_9)).__slate_anon_0.lock));
*core::ptr::addr_of_mut!((*(*__slate_slot_9)).__slate_anon_0.lock) = *{
__slate_compound_14 = spinlock { __slate_anon_0: {
let mut __t2: __SlateRecord_spinlock_types_h_18 = core::mem::zeroed();
__t2.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t3: __SlateRecord_qspinlock_types_h_44 = core::mem::zeroed();
__t3.val = __SlateRecord_types_h_190 { counter: 0 };
__t3
} } };
__t2
} };
core::ptr::addr_of_mut!(__slate_compound_14)
};
if true {
break;
}
}
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*(*__slate_slot_9)).__slate_anon_0.dispatch));
INIT_HLIST_NODE(core::ptr::addr_of_mut!((*(*__slate_slot_9)).cpuhp_dead));
INIT_HLIST_NODE(core::ptr::addr_of_mut!((*(*__slate_slot_9)).cpuhp_online));
(*(*__slate_slot_9)).queue = q;
(*(*__slate_slot_9)).flags = ((*set).flags & 4294967293u32) as u64;
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*(*__slate_slot_9)).hctx_list));
let __slate_auto_bytes_1 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_0 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_1 = _kmalloc_array_node_noprof(nr_cpu_ids as u64, core::mem::size_of::<*mut ()>() as u64, gfp, node, *{
__slate_compound_11 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_11)
});
alloc_tag_restore!();
} else {
*__slate_slot_1 = _kmalloc_array_node_noprof(nr_cpu_ids as u64, core::mem::size_of::<*mut ()>() as u64, gfp, node, *{
__slate_compound_13 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_13)
});
}
let __v2: *mut () = *__slate_slot_1;
// Allocate space for all possible cpus to avoid allocation at
// runtime
(*(*__slate_slot_9)).ctxs = __v2 as *mut *mut blk_mq_ctx;
if !unsafe { (*(*__slate_slot_9)).ctxs }.is_null() {
if sbitmap_init_node(core::ptr::addr_of_mut!((*(*__slate_slot_9)).ctx_map), nr_cpu_ids, ilog2__2!(8i32), gfp, node, false, false) != 0 {
kfree((*(*__slate_slot_9)).ctxs as *const ());
} else {
(*(*__slate_slot_9)).nr_ctx = 0;
loop {
spinlock_check(core::ptr::addr_of_mut!((*(*__slate_slot_9)).dispatch_wait_lock));
*core::ptr::addr_of_mut!((*(*__slate_slot_9)).dispatch_wait_lock) = *{
__slate_compound_10 = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = core::mem::zeroed();
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = core::mem::zeroed();
__t1.val = __SlateRecord_types_h_190 { counter: 0 };
__t1
} } };
__t0
} };
core::ptr::addr_of_mut!(__slate_compound_10)
};
if true {
break;
}
}
init_waitqueue_func_entry(core::ptr::addr_of_mut!((*(*__slate_slot_9)).dispatch_wait), Some(blk_mq_dispatch_wake));
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*(*__slate_slot_9)).dispatch_wait.entry));
blk_mq_hctx_kobj_init(*__slate_slot_9);
return *__slate_slot_9;
}
}
free_cpumask_var((*(*__slate_slot_9)).cpumask.as_mut_ptr() as *mut cpumask);
}
kfree(*__slate_slot_9 as *const ());
}
return core::ptr::null_mut::<blk_mq_hw_ctx>();
}
}

static mut __key__4047: lock_class_key = unsafe { core::mem::zeroed() };

fn blk_mq_init_cpu_queues(q: *mut request_queue, nr_hw_queues: u32) {
let mut __slate_compound_3: spinlock = unsafe { core::mem::zeroed() };
let set: *mut blk_mq_tag_set = unsafe { (*q).tag_set };
let mut i: u32 = 0;
let mut j: u32 = 0;
'__slate_break_11: loop {
i = find_next_bit(cpumask_bits!((unsafe { core::ptr::addr_of_mut!(__cpu_possible_mask) }) as *const cpumask), 64u64, i as u64) as u32;
if i >= 64 {
break;
}
let __ctx: *mut blk_mq_ctx = per_cpu_ptr!(unsafe { (*q).queue_ctx }, i);
unsafe {
(*__ctx).cpu = i;
}
'__slate_break_4: loop {
spinlock_check(unsafe { core::ptr::addr_of_mut!((*__ctx).__slate_anon_0.lock) });
unsafe {
*core::ptr::addr_of_mut!((*__ctx).__slate_anon_0.lock) = *{
__slate_compound_3 = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = core::mem::zeroed();
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = core::mem::zeroed();
__t1.val = __SlateRecord_types_h_190 { counter: 0 };
__t1
} } };
__t0
} };
core::ptr::addr_of_mut!(__slate_compound_3)
};
}
if true {
break;
}
}
for k in 0i32..3i32 {
INIT_LIST_HEAD(unsafe { (*__ctx).__slate_anon_0.rq_lists.as_mut_ptr() as *mut list_head }.wrapping_offset(k as isize));
}
unsafe {
(*__ctx).queue = q;
}
// Set local node, IFF we have more than one hw queue. If
// not, we remain on the home node of the device
j = 0;
while j < unsafe { (*set).nr_maps } {
let hctx: *mut blk_mq_hw_ctx = blk_mq_map_queue_type(q, j as u32, i);
if nr_hw_queues > 1 && (unsafe { (*hctx).numa_node }) == 1i32.wrapping_neg() {
unsafe {
(*hctx).numa_node = cpu_to_node(i as i32);
}
}
j = j.wrapping_add(1);
}
i = i.wrapping_add(1);
}
}

#[unsafe(no_mangle)]
extern "C" fn blk_mq_alloc_map_and_rqs(set: *mut blk_mq_tag_set, hctx_idx: u32, depth: u32) -> *mut blk_mq_tags {
let mut ret: i32 = 0;
let tags: *mut blk_mq_tags = blk_mq_alloc_rq_map(set, hctx_idx, depth, unsafe { (*set).reserved_tags });
if tags.is_null() {
return core::ptr::null_mut::<blk_mq_tags>();
}
ret = blk_mq_alloc_rqs(set, tags, hctx_idx, depth);
if ret != 0 {
blk_mq_free_rq_map(set, tags);
return core::ptr::null_mut::<blk_mq_tags>();
}
tags
}

fn __blk_mq_alloc_map_and_rqs(set: *mut blk_mq_tag_set, hctx_idx: i32) -> bool {
if blk_mq_is_shared_tags(unsafe { (*set).flags }) {
unsafe {
*unsafe { (*set).tags }.wrapping_offset(hctx_idx as isize) = (*set).shared_tags;
}
return true;
}
unsafe {
*unsafe { (*set).tags }.wrapping_offset(hctx_idx as isize) = blk_mq_alloc_map_and_rqs(set, hctx_idx as u32, (*set).queue_depth);
}
!unsafe { *unsafe { (*set).tags }.wrapping_offset(hctx_idx as isize) }.is_null()
}

#[unsafe(no_mangle)]
extern "C" fn blk_mq_free_map_and_rqs(set: *mut blk_mq_tag_set, tags: *mut blk_mq_tags, hctx_idx: u32) {
if !tags.is_null() {
blk_mq_free_rqs(set, tags, hctx_idx);
blk_mq_free_rq_map(set, tags);
}
}

fn __blk_mq_free_map_and_rqs(set: *mut blk_mq_tag_set, hctx_idx: u32) {
if !blk_mq_is_shared_tags(unsafe { (*set).flags }) {
blk_mq_free_map_and_rqs(set, unsafe { *unsafe { (*set).tags }.wrapping_offset(hctx_idx as isize) }, hctx_idx);
}
unsafe {
*unsafe { (*set).tags }.wrapping_offset(hctx_idx as isize) = core::ptr::null_mut::<blk_mq_tags>();
}
}

fn blk_mq_map_swqueue(q: *mut request_queue) {
let mut j: u32 = 0;
let mut hctx_idx: u32 = 0;
let mut i: u64 = 0;
let mut hctx: *mut blk_mq_hw_ctx = unsafe { core::mem::zeroed() };
let set: *mut blk_mq_tag_set = unsafe { (*q).tag_set };
loop {
let __v0: bool = if i < ((unsafe { (*q).nr_hw_queues }) as u64) {
let mut __v1: i32;
hctx = queue_hctx(q, i as i32);
__v1 = 1;
__v1 != 0
} else {
false
};
if !__v0 {
break;
}
cpumask_clear(unsafe { (*hctx).cpumask.as_mut_ptr() as *mut cpumask });
unsafe {
(*hctx).nr_ctx = 0;
}
unsafe {
(*hctx).dispatch_from = core::ptr::null_mut::<blk_mq_ctx>();
}
i = i.wrapping_add(1);
}
// Map software to hardware queues.
//
// If the cpu isn't present, the cpu is mapped to first hctx.
i = 0;
loop {
i = find_next_bit(cpumask_bits!((unsafe { core::ptr::addr_of_mut!(__cpu_possible_mask) }) as *const cpumask), 64u64, i);
if i >= 64 {
break;
}
let ctx: *mut blk_mq_ctx = per_cpu_ptr!(unsafe { (*q).queue_ctx }, i);
j = 0;
while j < unsafe { (*set).nr_maps } {
if (unsafe { (*((*set).map.as_mut_ptr() as *mut blk_mq_queue_map).wrapping_offset(j as isize)).nr_queues }) == 0 {
unsafe {
*((*ctx).hctxs.as_mut_ptr() as *mut *mut blk_mq_hw_ctx).wrapping_offset(j as isize) = blk_mq_map_queue_type(q, 0u32, i as u32);
}
} else {
hctx_idx = unsafe { *unsafe { (*((*set).map.as_mut_ptr() as *mut blk_mq_queue_map).wrapping_offset(j as isize)).mq_map }.wrapping_offset(i as isize) };
// unmapped hw queue can be remapped after CPU topo changed
let __v9: bool = unsafe { *unsafe { (*set).tags }.wrapping_offset(hctx_idx as isize) }.is_null() && !__blk_mq_alloc_map_and_rqs(set, hctx_idx as i32);
if __v9 {
// If tags initialization fail for some hctx,
// that hctx won't be brought online.  In this
// case, remap the current ctx to hctx[0] which
// is guaranteed to always have tags allocated
unsafe {
*unsafe { (*((*set).map.as_mut_ptr() as *mut blk_mq_queue_map).wrapping_offset(j as isize)).mq_map }.wrapping_offset(i as isize) = 0;
}
}
hctx = blk_mq_map_queue_type(q, j as u32, i as u32);
unsafe {
*((*ctx).hctxs.as_mut_ptr() as *mut *mut blk_mq_hw_ctx).wrapping_offset(j as isize) = hctx;
}
// If the CPU is already set in the mask, then we've
// mapped this one already. This can happen if
// devices share queues across queue maps.
if !cpumask_test_cpu(i as i32, (unsafe { (*hctx).cpumask.as_mut_ptr() as *mut cpumask }) as *const cpumask) {
cpumask_set_cpu(i as u32, unsafe { (*hctx).cpumask.as_mut_ptr() as *mut cpumask });
unsafe {
(*hctx).r#type = j as u16;
}
unsafe {
*((*ctx).index_hw.as_mut_ptr() as *mut u16).wrapping_offset(((*hctx).r#type as i32) as isize) = (*hctx).nr_ctx;
}
let __v11: u16 = unsafe { (*hctx).nr_ctx };
unsafe {
(*hctx).nr_ctx = (__v11 as i32).wrapping_add(1) as u16;
}
unsafe {
*unsafe { (*hctx).ctxs }.wrapping_offset((__v11 as i32) as isize) = ctx;
}
// If the nr_ctx type overflows, we have exceeded the
// amount of sw queues we can support.
loop {
if unlikely__2!((unsafe { (*hctx).nr_ctx }) == 0) != 0 {
loop {
unsafe {
core::arch::asm!("995: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 995b - ., 3; .popsection\n# {0}", const 995 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_blk_mq_map_swqueue_13, sym __slate_asm_string_blk_mq_map_swqueue_14, const 4233 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe { core::hint::unreachable_unchecked() };
if true {
break;
}
}
}
if true {
break;
}
}
}
}
j = j.wrapping_add(1);
}
while j < 3 {
unsafe {
*((*ctx).hctxs.as_mut_ptr() as *mut *mut blk_mq_hw_ctx).wrapping_offset(j as isize) = blk_mq_map_queue_type(q, 0u32, i as u32);
}
j = j.wrapping_add(1);
}
i = i.wrapping_add(1);
}
i = 0;
loop {
let __v35: bool = if i < ((unsafe { (*q).nr_hw_queues }) as u64) {
let mut __v36: i32;
hctx = queue_hctx(q, i as i32);
__v36 = 1;
__v36 != 0
} else {
false
};
if !__v35 {
break;
}
let mut cpu: i32 = 0;
// If no software queues are mapped to this hardware queue,
// disable it and free the request entries.
if (unsafe { (*hctx).nr_ctx }) == 0 {
// Never unmap queue 0.  We need it as a
// fallback in case of a new remap fails
// allocation
if i != 0 {
__blk_mq_free_map_and_rqs(set, i as u32);
}
unsafe {
(*hctx).tags = core::ptr::null_mut::<blk_mq_tags>();
}
} else {
unsafe {
(*hctx).tags = *unsafe { (*set).tags }.wrapping_offset(i as isize);
}
let mut __v27: i64;
let __ret_warn_on: i32 = unsafe { (*hctx).tags }.is_null() as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("996: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 996b - ., 3; .popsection\n# {0}", const 996 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_blk_mq_map_swqueue_28, sym __slate_asm_string_blk_mq_map_swqueue_29, const 4261 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("997: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 997b - ., 4; .popsection\n# {0}", const 997 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v27 = (__ret_warn_on != 0) as i64;
// Set the map size to the number of mapped software queues.
// This is more accurate and more efficient than looping
// over all possibly mapped software queues.
unsafe { sbitmap_resize(core::ptr::addr_of_mut!((*hctx).ctx_map), (*hctx).nr_ctx as u32) };
// Rule out isolated CPUs from hctx->cpumask to avoid
// running block kworker on isolated CPUs.
// FIXME: cpuset should propagate further changes to isolated CPUs
// here.
rcu_read_lock();
cpu = 0;
loop {
cpu = find_next_bit(cpumask_bits__2!(unsafe { (*hctx).cpumask.as_mut_ptr() as *mut cpumask }) as *const u64, 64u64, cpu as u64) as i32;
if (cpu as u32) >= 64 {
break;
}
if cpu_is_isolated(cpu) {
cpumask_clear_cpu(cpu, unsafe { (*hctx).cpumask.as_mut_ptr() as *mut cpumask });
}
cpu = cpu.wrapping_add(1);
}
rcu_read_unlock();
// Initialize batch roundrobin counts
unsafe {
(*hctx).next_cpu = blk_mq_first_mapped_cpu(hctx);
}
unsafe {
(*hctx).next_cpu_batch = 8;
}
}
i = i.wrapping_add(1);
}
}

/// Caller needs to ensure that we're either frozen/quiesced, or that
/// the queue isn't live yet.
fn queue_set_hctx_shared(q: *mut request_queue, shared: bool) {
let mut hctx: *mut blk_mq_hw_ctx = unsafe { core::mem::zeroed() };
let mut i: u64 = 0;
loop {
let __v6: bool = if i < ((unsafe { (*q).nr_hw_queues }) as u64) {
let mut __v7: i32;
hctx = queue_hctx(q, i as i32);
__v7 = 1;
__v7 != 0
} else {
false
};
if !__v6 {
break;
}
if shared {
unsafe {
(*hctx).flags = (*hctx).flags | 2;
}
} else {
blk_mq_tag_idle(hctx);
let __v5: u64 = (unsafe { (*hctx).flags }) & 18446744073709551613u64;
unsafe {
(*hctx).flags = __v5;
}
}
i = i.wrapping_add(1);
}
}

fn blk_mq_update_tag_set_shared(set: *mut blk_mq_tag_set, shared: bool) {
let mut q: *mut request_queue = unsafe { core::mem::zeroed() };
let mut memflags: u32 = 0;
lockdep_assert_held__2!(unsafe { core::ptr::addr_of_mut!((*set).tag_list_lock) });
q = list_first_entry!(unsafe { core::ptr::addr_of_mut!((*set).tag_list) });
while list_entry_is_head!(q, unsafe { core::ptr::addr_of_mut!((*set).tag_list) }, tag_set_list) == 0 {
memflags = blk_mq_freeze_queue(q);
queue_set_hctx_shared(q, shared);
blk_mq_unfreeze_queue(q, memflags);
q = list_next_entry!(q, tag_set_list);
}
}

fn blk_mq_del_queue_tag_set(q: *mut request_queue) {
let set: *mut blk_mq_tag_set = unsafe { (*q).tag_set };
unsafe { mutex_lock(core::ptr::addr_of_mut!((*set).tag_list_lock)) };
list_del_rcu(unsafe { core::ptr::addr_of_mut!((*q).tag_set_list) });
if list_is_singular((unsafe { core::ptr::addr_of_mut!((*set).tag_list) }) as *const list_head) != 0 {
// just transitioned to unshared
let __v2: u32 = (unsafe { (*set).flags }) & 4294967293u32;
unsafe {
(*set).flags = __v2;
}
// update existing queue
blk_mq_update_tag_set_shared(set, false);
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*set).tag_list_lock)) };
}

fn blk_mq_add_queue_tag_set(set: *mut blk_mq_tag_set, q: *mut request_queue) {
unsafe { mutex_lock(core::ptr::addr_of_mut!((*set).tag_list_lock)) };
// Check to see if we're transitioning to shared (from 1 to 2 queues).
if list_empty((unsafe { core::ptr::addr_of_mut!((*set).tag_list) }) as *const list_head) == 0 && (unsafe { (*set).flags }) & 2 == 0 {
unsafe {
(*set).flags = (*set).flags | 2;
}
// update existing queue
blk_mq_update_tag_set_shared(set, true);
}
if (unsafe { (*set).flags }) & 2 != 0 {
queue_set_hctx_shared(q, true);
}
list_add_tail_rcu(unsafe { core::ptr::addr_of_mut!((*q).tag_set_list) }, unsafe { core::ptr::addr_of_mut!((*set).tag_list) });
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*set).tag_list_lock)) };
}

/// All allocations will be freed in release handler of q->mq_kobj
fn blk_mq_alloc_ctxs(q: *mut request_queue) -> i32 {
let mut __slate_compound_19: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_21: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut cpu: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut i32 = core::ptr::addr_of_mut!(cpu) as *mut i32;
let mut ctxs: core::mem::MaybeUninit<*mut blk_mq_ctxs> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut blk_mq_ctxs = core::ptr::addr_of_mut!(ctxs) as *mut *mut blk_mq_ctxs;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
__obj_size = size_mul(core::mem::size_of::<blk_mq_ctxs>() as u64, 1u64);
let __slate_auto_bytes_3 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_3 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_4 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_19 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_19)
});
alloc_tag_restore!();
} else {
*__slate_slot_4 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_21 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_21)
});
}
*__slate_slot_10 = *__slate_slot_4 as *mut blk_mq_ctxs;
if (*__slate_slot_10).is_null() {
return 12i32.wrapping_neg();
} else {
(*(*__slate_slot_10)).queue_ctx = alloc_percpu!(blk_mq_ctx);
if unsafe { (*(*__slate_slot_10)).queue_ctx }.is_null() {
kfree(*__slate_slot_10 as *const ());
return 12i32.wrapping_neg();
} else {
*__slate_slot_9 = 0;
loop {
*__slate_slot_9 = find_next_bit(cpumask_bits!(core::ptr::addr_of_mut!(__cpu_possible_mask) as *const cpumask), 64u64, *__slate_slot_9 as u64) as i32;
if (*__slate_slot_9 as u32) < 64 {
let ctx: *mut blk_mq_ctx = per_cpu_ptr!(unsafe { (*(*__slate_slot_10)).queue_ctx }, *__slate_slot_9);
(*ctx).ctxs = *__slate_slot_10;
let __v1: i32 = *__slate_slot_9;
*__slate_slot_9 = __v1.wrapping_add(1);
} else {
break;
}
}
(*q).mq_kobj = core::ptr::addr_of_mut!((*(*__slate_slot_10)).kobj);
(*q).queue_ctx = (*(*__slate_slot_10)).queue_ctx;
return 0;
}
}
}
}

/// It is the actual release handler for mq, but we do it from
/// request queue's release handler for avoiding use-after-free
/// and headache because q->mq_kobj shouldn't have been introduced,
/// but we can't group ctx/kctx kobj without it.
#[unsafe(no_mangle)]
extern "C" fn blk_mq_release(q: *mut request_queue) {
let mut hctx: *mut blk_mq_hw_ctx = unsafe { core::mem::zeroed() };
let mut next: *mut blk_mq_hw_ctx = unsafe { core::mem::zeroed() };
let mut i: u64 = 0;
loop {
let __v6: bool = if i < ((unsafe { (*q).nr_hw_queues }) as u64) {
let mut __v7: i32;
hctx = queue_hctx(q, i as i32);
__v7 = 1;
__v7 != 0
} else {
false
};
if !__v6 {
break;
}
let mut __v0: i64;
let mut __ret_warn_on: i32 = 0;
let __v1: bool = !hctx.is_null() && list_empty((unsafe { core::ptr::addr_of_mut!((*hctx).hctx_list) }) as *const list_head) != 0;
__ret_warn_on = __v1 as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("998: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 998b - ., 3; .popsection\n# {0}", const 998 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_blk_mq_release_2, sym __slate_asm_string_blk_mq_release_3, const 4401 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("999: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 999b - ., 4; .popsection\n# {0}", const 999 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
i = i.wrapping_add(1);
}
// all hctx are in .unused_hctx_list now
hctx = list_first_entry__7!(unsafe { core::ptr::addr_of_mut!((*q).unused_hctx_list) });
next = list_next_entry__6!(hctx, hctx_list);
while list_entry_is_head!(hctx, unsafe { core::ptr::addr_of_mut!((*q).unused_hctx_list) }, hctx_list) == 0 {
list_del_init(unsafe { core::ptr::addr_of_mut!((*hctx).hctx_list) });
unsafe { kobject_put(core::ptr::addr_of_mut!((*hctx).kobj)) };
hctx = next;
next = list_next_entry__6!(next, hctx_list);
}
unsafe { kfree((*q).queue_hw_ctx as *const ()) };
// release .mq_kobj and sw queue's kobject now because
// both share lifetime with request queue.
unsafe { blk_mq_sysfs_deinit(q) };
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_alloc_queue")]
extern "C" fn blk_mq_alloc_queue(set: *mut blk_mq_tag_set, mut lim: *mut queue_limits, queuedata: *mut ()) -> *mut request_queue {
let mut default_lim: queue_limits = {
let mut __t0: queue_limits = unsafe { core::mem::zeroed() };
__t0
};
let mut ret: i32 = 0;
if lim.is_null() {
lim = core::ptr::addr_of_mut!(default_lim);
}
unsafe {
(*lim).features = (*lim).features | 144u32;
}
if (unsafe { (*set).nr_maps }) > 2 {
unsafe {
(*lim).features = (*lim).features | 512u32;
}
}
let q: *mut request_queue = unsafe { blk_alloc_queue(lim, (*set).numa_node) };
if IS_ERR(q as *const ()) {
return q;
}
unsafe {
(*q).queuedata = queuedata;
}
ret = blk_mq_init_allocated_queue(set, q);
if ret != 0 {
unsafe { blk_put_queue(q) };
return ERR_PTR(ret as i64) as *mut request_queue;
}
q
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_alloc_queue_1000: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut blk_mq_tag_set, *mut queue_limits, *mut ()) -> *mut request_queue>, *mut ()>(Some(blk_mq_alloc_queue)) };

/// blk_mq_destroy_queue - shutdown a request queue
/// @q: request queue to shutdown
///
/// This shuts down a request queue allocated by blk_mq_alloc_queue(). All future
/// requests will be failed with -ENODEV. The caller is responsible for dropping
/// the reference from blk_mq_alloc_queue() by calling blk_put_queue().
///
/// Context: can sleep
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_destroy_queue")]
extern "C" fn blk_mq_destroy_queue(q: *mut request_queue) {
let mut __v0: i64;
{
let __ret_warn_on: i32 = !(!(!queue_is_mq(q))) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("1001: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1001b - ., 3; .popsection\n# {0}", const 1001 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_blk_mq_destroy_queue_1, sym __slate_asm_string_blk_mq_destroy_queue_2, const 4456 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("1002: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1002b - ., 4; .popsection\n# {0}", const 1002 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
}
let mut __v5: i64;
let __ret_warn_on: i32 = (blk_queue_registered!(q) != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("1003: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1003b - ., 3; .popsection\n# {0}", const 1003 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_blk_mq_destroy_queue_7, sym __slate_asm_string_blk_mq_destroy_queue_8, const 4457 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("1004: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1004b - ., 4; .popsection\n# {0}", const 1004 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v5 = (__ret_warn_on != 0) as i64;
might_sleep!();
unsafe { blk_queue_flag_set(0u32, q) };
unsafe { blk_queue_start_drain(q) };
blk_mq_freeze_queue_wait(q);
unsafe { blk_sync_queue(q) };
blk_mq_cancel_work_sync(q);
blk_mq_exit_queue(q);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_destroy_queue_1005: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request_queue)>, *mut ()>(Some(blk_mq_destroy_queue)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.__blk_mq_alloc_disk")]
extern "C" fn __blk_mq_alloc_disk(set: *mut blk_mq_tag_set, lim: *mut queue_limits, queuedata: *mut (), lkclass: *mut lock_class_key) -> *mut gendisk {
let mut disk: *mut gendisk = unsafe { core::mem::zeroed() };
let q: *mut request_queue = blk_mq_alloc_queue(set, lim, queuedata);
if IS_ERR(q as *const ()) {
return ERR_CAST(q as *const ()) as *mut gendisk;
}
disk = unsafe { __alloc_disk_node(q, (*set).numa_node, lkclass) };
if disk.is_null() {
blk_mq_destroy_queue(q);
unsafe { blk_put_queue(q) };
return ERR_PTR(12i32.wrapping_neg() as i64) as *mut gendisk;
}
set_bit(6i64, (unsafe { core::ptr::addr_of_mut!((*disk).state) }) as *mut u64);
disk
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___blk_mq_alloc_disk_1006: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut blk_mq_tag_set, *mut queue_limits, *mut (), *mut lock_class_key) -> *mut gendisk>, *mut ()>(Some(__blk_mq_alloc_disk)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_alloc_disk_for_queue")]
extern "C" fn blk_mq_alloc_disk_for_queue(q: *mut request_queue, lkclass: *mut lock_class_key) -> *mut gendisk {
if !unsafe { blk_get_queue(q) } {
return core::ptr::null_mut::<gendisk>();
}
let disk: *mut gendisk = unsafe { __alloc_disk_node(q, 1i32.wrapping_neg(), lkclass) };
if disk.is_null() {
unsafe { blk_put_queue(q) };
}
disk
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_alloc_disk_for_queue_1007: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request_queue, *mut lock_class_key) -> *mut gendisk>, *mut ()>(Some(blk_mq_alloc_disk_for_queue)) };

/// Only hctx removed from cpuhp list can be reused
fn blk_mq_hctx_is_reusable(hctx: *mut blk_mq_hw_ctx) -> bool {
hlist_unhashed((unsafe { core::ptr::addr_of_mut!((*hctx).cpuhp_online) }) as *const hlist_node) != 0 && hlist_unhashed((unsafe { core::ptr::addr_of_mut!((*hctx).cpuhp_dead) }) as *const hlist_node) != 0
}

fn blk_mq_alloc_and_init_hctx(set: *mut blk_mq_tag_set, q: *mut request_queue, hctx_idx: i32, node: i32) -> *mut blk_mq_hw_ctx {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut tmp: core::mem::MaybeUninit<*mut blk_mq_hw_ctx> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut blk_mq_hw_ctx = core::ptr::addr_of_mut!(tmp) as *mut *mut blk_mq_hw_ctx;
let mut hctx: *mut blk_mq_hw_ctx = unsafe { core::mem::zeroed() };
unsafe {
hctx = core::ptr::null_mut::<blk_mq_hw_ctx>();
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
// reuse dead hctx first
spin_lock(core::ptr::addr_of_mut!((*q).unused_hctx_lock));
*__slate_slot_1 = list_first_entry__7!(core::ptr::addr_of_mut!((*q).unused_hctx_list));
'__join_8: {
loop {
if list_entry_is_head!(*__slate_slot_1, core::ptr::addr_of_mut!((*q).unused_hctx_list), hctx_list) == 0 {
if (*(*__slate_slot_1)).numa_node == node {
__v0 = blk_mq_hctx_is_reusable(*__slate_slot_1);
} else {
__v0 = false;
}
if __v0 {
break;
} else {
*__slate_slot_1 = list_next_entry__6!(*__slate_slot_1, hctx_list);
}
} else {
break '__join_8;
}
}
hctx = *__slate_slot_1;
}
if !hctx.is_null() {
list_del_init(core::ptr::addr_of_mut!((*hctx).hctx_list));
}
spin_unlock(core::ptr::addr_of_mut!((*q).unused_hctx_lock));
if hctx.is_null() {
hctx = blk_mq_alloc_hctx(q, set, node);
}
if !hctx.is_null() {
if blk_mq_init_hctx(q, set, hctx, hctx_idx as u32) != 0 {
kobject_put(core::ptr::addr_of_mut!((*hctx).kobj));
} else {
return hctx;
}
}
return core::ptr::null_mut::<blk_mq_hw_ctx>();
}
}

fn __blk_mq_realloc_hw_ctxs(set: *mut blk_mq_tag_set, q: *mut request_queue) {
let mut __slate_compound_2: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_4: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
let mut j: i32 = 0;
let mut end: i32 = 0;
let mut hctxs: *mut *mut blk_mq_hw_ctx = unsafe { (*q).queue_hw_ctx };
if (unsafe { (*q).nr_hw_queues }) < unsafe { (*set).nr_hw_queues } {
let mut __v0: *mut ();
let mut __v1: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_array_node_noprof((unsafe { (*set).nr_hw_queues }) as u64, core::mem::size_of::<*mut blk_mq_hw_ctx>() as u64, 3520u32, unsafe { (*set).numa_node }, unsafe { *{
__slate_compound_2 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_2)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_array_node_noprof((unsafe { (*set).nr_hw_queues }) as u64, core::mem::size_of::<*mut blk_mq_hw_ctx>() as u64, 3520u32, unsafe { (*set).numa_node }, unsafe { *{
__slate_compound_4 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_4)
} });
}
__v1 = _res;
__v0 = __v1;
let new_hctxs: *mut *mut blk_mq_hw_ctx = __v0 as *mut *mut blk_mq_hw_ctx;
if new_hctxs.is_null() {
return;
}
if !hctxs.is_null() {
unsafe { memcpy(new_hctxs as *mut (), hctxs as *const (), ((*q).nr_hw_queues as u64).wrapping_mul(core::mem::size_of::<*mut blk_mq_hw_ctx>() as u64)) };
}
disable_context_analysis!();
let _r_a_p__v: u64 = new_hctxs as u64;
if false && _r_a_p__v == 0 {
loop {
loop {
if !(__native_word__10!() || ((const { core::mem::size_of::<*mut *mut blk_mq_hw_ctx>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1008() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__12!((*q).queue_hw_ctx, _r_a_p__v as *mut *mut blk_mq_hw_ctx);
if true {
break;
}
}
} else {
loop {
kcsan_release!();
loop {
loop {
if !__native_word__10!() {
unsafe { __compiletime_assert_1009() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
loop {
loop {
if !(__native_word__10!() || ((const { core::mem::size_of::<*mut *mut blk_mq_hw_ctx>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1010() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__12!(*unsafe { core::ptr::addr_of_mut!((*q).queue_hw_ctx) }, RCU_INITIALIZER!(_r_a_p__v as *mut *mut blk_mq_hw_ctx));
if true {
break;
}
}
if true {
break;
}
}
if true {
break;
}
}
}
// Make sure reading the old queue_hw_ctx from other
// context concurrently won't trigger uaf.
kfree_rcu_mightsleep!(hctxs);
hctxs = new_hctxs;
}
while (i as u32) < unsafe { (*set).nr_hw_queues } {
let mut old_node: i32 = 0;
let node: i32 = blk_mq_get_hctx_node(set, i as u32);
let old_hctx: *mut blk_mq_hw_ctx = unsafe { *hctxs.wrapping_offset(i as isize) };
if !old_hctx.is_null() {
old_node = unsafe { (*old_hctx).numa_node };
blk_mq_exit_hctx(q, set, old_hctx, i as u32);
}
unsafe {
*hctxs.wrapping_offset(i as isize) = blk_mq_alloc_and_init_hctx(set, q, i, node);
}
if unsafe { *hctxs.wrapping_offset(i as isize) }.is_null() {
if old_hctx.is_null() {
break;
}
let mut __v17: i32;
while false {
}
__v17 = unsafe { _printk(c"\x014Allocate new hctx on node %d fails, fallback to previous one on node %d\n".as_ptr() as *const u8, node, old_node) };
unsafe {
*hctxs.wrapping_offset(i as isize) = blk_mq_alloc_and_init_hctx(set, q, i, old_node);
}
let mut __v19: i64;
let __ret_warn_on: i32 = unsafe { *hctxs.wrapping_offset(i as isize) }.is_null() as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("1011: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1011b - ., 3; .popsection\n# {0}", const 1011 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___blk_mq_realloc_hw_ctxs_20, sym __slate_asm_string___blk_mq_realloc_hw_ctxs_21, const 4594 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("1012: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1012b - ., 4; .popsection\n# {0}", const 1012 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v19 = (__ret_warn_on != 0) as i64;
}
i = i.wrapping_add(1);
}
// Increasing nr_hw_queues fails. Free the newly allocated
// hctxs and keep the previous q->nr_hw_queues.
if (i as u32) != unsafe { (*set).nr_hw_queues } {
j = (unsafe { (*q).nr_hw_queues }) as i32;
end = i;
} else {
j = i;
end = (unsafe { (*q).nr_hw_queues }) as i32;
unsafe {
(*q).nr_hw_queues = (*set).nr_hw_queues;
}
}
while j < end {
let hctx: *mut blk_mq_hw_ctx = unsafe { *hctxs.wrapping_offset(j as isize) };
if !hctx.is_null() {
blk_mq_exit_hctx(q, set, hctx, j as u32);
unsafe {
*hctxs.wrapping_offset(j as isize) = core::ptr::null_mut::<blk_mq_hw_ctx>();
}
}
j = j.wrapping_add(1);
}
}

fn blk_mq_realloc_hw_ctxs(set: *mut blk_mq_tag_set, q: *mut request_queue) {
__blk_mq_realloc_hw_ctxs(set, q);
// unregister cpuhp callbacks for exited hctxs
blk_mq_remove_hw_queues_cpuhp(q);
// register cpuhp for new initialized hctxs
blk_mq_add_hw_queues_cpuhp(q);
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_init_allocated_queue")]
extern "C" fn blk_mq_init_allocated_queue(set: *mut blk_mq_tag_set, q: *mut request_queue) -> i32 {
let mut __slate_compound_3: spinlock = unsafe { core::mem::zeroed() };
let mut __slate_compound_4: __SlateRecord_types_h_197 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_types_h_197 = unsafe { core::mem::zeroed() };
let mut __slate_compound_6: spinlock = unsafe { core::mem::zeroed() };
unsafe {
// mark the queue as mq asap
(*q).mq_ops = (*set).ops;
// ->tag_set has to be setup before initialize hctx, which cpuphp
// handler needs it for checking queue mapping
(*q).tag_set = set;
if blk_mq_alloc_ctxs(q) == 0 {
// init q->mq_kobj and sw queues' kobjects
blk_mq_sysfs_init(q);
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*q).unused_hctx_list));
loop {
spinlock_check(core::ptr::addr_of_mut!((*q).unused_hctx_lock));
*core::ptr::addr_of_mut!((*q).unused_hctx_lock) = *{
__slate_compound_6 = spinlock { __slate_anon_0: {
let mut __t2: __SlateRecord_spinlock_types_h_18 = core::mem::zeroed();
__t2.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t3: __SlateRecord_qspinlock_types_h_44 = core::mem::zeroed();
__t3.val = __SlateRecord_types_h_190 { counter: 0 };
__t3
} } };
__t2
} };
core::ptr::addr_of_mut!(__slate_compound_6)
};
if true {
break;
}
}
blk_mq_realloc_hw_ctxs(set, q);
if (*q).nr_hw_queues == 0 {
blk_mq_release(q);
} else {
loop {
__init_work(core::ptr::addr_of_mut!((*q).timeout_work), 0i32);
(*core::ptr::addr_of_mut!((*q).timeout_work)).data = *{
__slate_compound_5 = __SlateRecord_types_h_197 { counter: 4503599625273344i64 };
core::ptr::addr_of_mut!(__slate_compound_5)
};
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*q).timeout_work)).entry));
(*core::ptr::addr_of_mut!((*q).timeout_work)).func = Some(blk_mq_timeout_work);
if true {
if true {
break;
}
}
}
blk_queue_rq_timeout(q, if (*set).timeout != 0 { (*set).timeout } else { 30000u32 });
(*q).queue_flags = (*q).queue_flags | 4u64;
loop {
__init_work(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*q).requeue_work)).work), 0i32);
(*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*q).requeue_work)).work)).data = *{
__slate_compound_4 = __SlateRecord_types_h_197 { counter: 4503599625273344i64 };
core::ptr::addr_of_mut!(__slate_compound_4)
};
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*q).requeue_work)).work)).entry));
(*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*q).requeue_work)).work)).func = Some(blk_mq_requeue_work);
if true {
if true {
__timer_init!(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*q).requeue_work)).timer), unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut timer_list)>>(delayed_work_timer_fn as *const ()) }, 2097152i32);
if true {
break;
}
}
}
}
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*q).flush_list));
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*q).requeue_list));
loop {
spinlock_check(core::ptr::addr_of_mut!((*q).requeue_lock));
*core::ptr::addr_of_mut!((*q).requeue_lock) = *{
__slate_compound_3 = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = core::mem::zeroed();
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = core::mem::zeroed();
__t1.val = __SlateRecord_types_h_190 { counter: 0 };
__t1
} } };
__t0
} };
core::ptr::addr_of_mut!(__slate_compound_3)
};
if true {
break;
}
}
(*q).nr_requests = (*set).queue_depth;
(*q).async_depth = (*set).queue_depth;
blk_mq_init_cpu_queues(q, (*set).nr_hw_queues);
blk_mq_map_swqueue(q);
blk_mq_add_queue_tag_set(set, q);
return 0;
}
}
(*q).mq_ops = core::ptr::null::<blk_mq_ops>();
return 12i32.wrapping_neg();
}
}

static mut __key__4657: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__4662: lock_class_key = unsafe { core::mem::zeroed() };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_init_allocated_queue_1013: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut blk_mq_tag_set, *mut request_queue) -> i32>, *mut ()>(Some(blk_mq_init_allocated_queue)) };

/// tags can _not_ be used after returning from blk_mq_exit_queue
#[unsafe(no_mangle)]
extern "C" fn blk_mq_exit_queue(q: *mut request_queue) {
let set: *mut blk_mq_tag_set = unsafe { (*q).tag_set };
// Checks hctx->flags & BLK_MQ_F_TAG_QUEUE_SHARED.
blk_mq_exit_hw_queues(q, set, (unsafe { (*set).nr_hw_queues }) as i32);
// May clear BLK_MQ_F_TAG_QUEUE_SHARED in hctx->flags.
blk_mq_del_queue_tag_set(q);
}

fn __blk_mq_alloc_rq_maps(set: *mut blk_mq_tag_set) -> i32 {
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
unsafe {
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if blk_mq_is_shared_tags((*set).flags) {
(*set).shared_tags = blk_mq_alloc_map_and_rqs(set, 1u32.wrapping_neg(), (*set).queue_depth);
if unsafe { (*set).shared_tags }.is_null() {
return 12i32.wrapping_neg();
}
}
*__slate_slot_4 = 0;
'__join_3: {
while (*__slate_slot_4 as u32) < (*set).nr_hw_queues {
if !__blk_mq_alloc_map_and_rqs(set, *__slate_slot_4) {
break '__join_3;
} else {
cond_resched__9!();
let __v3: i32 = *__slate_slot_4;
*__slate_slot_4 = __v3.wrapping_add(1);
}
}
return 0;
}
loop {
let __v1: i32 = *__slate_slot_4;
let __v0: i32 = __v1.wrapping_sub(1);
*__slate_slot_4 = __v0;
if __v0 >= 0 {
__blk_mq_free_map_and_rqs(set, *__slate_slot_4 as u32);
} else {
break;
}
}
if blk_mq_is_shared_tags((*set).flags) {
blk_mq_free_map_and_rqs(set, (*set).shared_tags, 1u32.wrapping_neg());
}
return 12i32.wrapping_neg();
}
}

/// Allocate the request maps associated with this tag_set. Note that this
/// may reduce the depth asked for, if memory is tight. set->queue_depth
/// will be updated to reflect the allocated depth.
fn blk_mq_alloc_set_map_and_rqs(set: *mut blk_mq_tag_set) -> i32 {
let depth: u32 = unsafe { (*set).queue_depth };
let mut err: i32 = 0;
loop {
err = __blk_mq_alloc_rq_maps(set);
if err == 0 {
break;
}
unsafe {
(*set).queue_depth = (*set).queue_depth >> 1i32;
}
if (unsafe { (*set).queue_depth }) < unsafe { (*set).reserved_tags }.wrapping_add(1) {
err = 12i32.wrapping_neg();
break;
}
if (unsafe { (*set).queue_depth }) == 0 {
break;
}
}
if (unsafe { (*set).queue_depth }) == 0 || err != 0 {
let mut __v4: i32;
while false {
}
__v4 = unsafe { _printk(c"\x013blk-mq: failed to allocate request map\n".as_ptr() as *const u8) };
return 12i32.wrapping_neg();
}
if depth != unsafe { (*set).queue_depth } {
let mut __v6: i32;
while false {
}
__v6 = unsafe { _printk(c"\x016blk-mq: reduced tag depth (%u -> %u)\n".as_ptr() as *const u8, depth, (*set).queue_depth) };
}
0
}

fn blk_mq_update_queue_map(set: *mut blk_mq_tag_set) {
// blk_mq_map_queues() and multiple .map_queues() implementations
// expect that set->map[HCTX_TYPE_DEFAULT].nr_queues is set to the
// number of hardware queues.
if (unsafe { (*set).nr_maps }) == 1 {
unsafe {
(*((*set).map.as_mut_ptr() as *mut blk_mq_queue_map).wrapping_offset(0 as isize)).nr_queues = (*set).nr_hw_queues;
}
}
if (unsafe { (*(*set).ops).map_queues }) != None {
// transport .map_queues is usually done in the following
// way:
//
// for (queue = 0; queue < set->nr_hw_queues; queue++) {
// 	mask = get_cpu_mask(queue)
// 	for_each_cpu(cpu, mask)
// 		set->map[x].mq_map[cpu] = queue;
// }
//
// When we need to remap, the table has to be cleared for
// killing stale mapping since one CPU may not be mapped
// to any hw queue.
let mut i: i32 = 0;
while (i as u32) < unsafe { (*set).nr_maps } {
blk_mq_clear_mq_map(unsafe { (*set).map.as_mut_ptr() as *mut blk_mq_queue_map }.wrapping_offset(i as isize));
i = i.wrapping_add(1);
}
unsafe { unsafe { (*(*set).ops).map_queues }.unwrap()(set) };
} else {
loop {
if unlikely__2!((unsafe { (*set).nr_maps }) > 1) != 0 {
loop {
unsafe {
core::arch::asm!("1014: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1014b - ., 3; .popsection\n# {0}", const 1014 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_blk_mq_update_queue_map_3, sym __slate_asm_string_blk_mq_update_queue_map_4, const 4793 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe { core::hint::unreachable_unchecked() };
if true {
break;
}
}
}
if true {
break;
}
}
unsafe { blk_mq_map_queues(((*set).map.as_mut_ptr() as *mut blk_mq_queue_map).wrapping_offset(0 as isize)) };
}
}

fn blk_mq_prealloc_tag_set_tags(set: *mut blk_mq_tag_set, new_nr_hw_queues: i32) -> *mut *mut blk_mq_tags {
let mut __slate_compound_11: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_13: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut new_tags: core::mem::MaybeUninit<*mut *mut blk_mq_tags> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut *mut blk_mq_tags = core::ptr::addr_of_mut!(new_tags) as *mut *mut *mut blk_mq_tags;
unsafe {
let __slate_auto_bytes_2 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
if (*set).nr_hw_queues >= (new_nr_hw_queues as u32) {
return core::ptr::null_mut::<*mut blk_mq_tags>();
} else {
let __slate_auto_bytes_1 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_4 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_5 = _kmalloc_array_node_noprof(new_nr_hw_queues as u64, core::mem::size_of::<*mut blk_mq_tags>() as u64, 3520u32, (*set).numa_node, *{
__slate_compound_11 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_11)
});
alloc_tag_restore!();
} else {
*__slate_slot_5 = _kmalloc_array_node_noprof(new_nr_hw_queues as u64, core::mem::size_of::<*mut blk_mq_tags>() as u64, 3520u32, (*set).numa_node, *{
__slate_compound_13 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_13)
});
}
*__slate_slot_9 = *__slate_slot_5 as *mut *mut blk_mq_tags;
if (*__slate_slot_9).is_null() {
return ERR_PTR(12i32.wrapping_neg() as i64) as *mut *mut blk_mq_tags;
} else {
if !unsafe { (*set).tags }.is_null() {
memcpy(*__slate_slot_9 as *mut (), (*set).tags as *const (), ((*set).nr_hw_queues as u64).wrapping_mul(core::mem::size_of::<*mut blk_mq_tags>() as u64));
}
*__slate_slot_8 = (*set).nr_hw_queues as i32;
'__join_1: {
while *__slate_slot_8 < new_nr_hw_queues {
if blk_mq_is_shared_tags((*set).flags) {
*(*__slate_slot_9).wrapping_offset(*__slate_slot_8 as isize) = (*set).shared_tags;
} else {
*(*__slate_slot_9).wrapping_offset(*__slate_slot_8 as isize) = blk_mq_alloc_map_and_rqs(set, *__slate_slot_8 as u32, (*set).queue_depth);
if unsafe { *(*__slate_slot_9).wrapping_offset(*__slate_slot_8 as isize) }.is_null() {
break '__join_1;
}
}
cond_resched__8!();
let __v3: i32 = *__slate_slot_8;
*__slate_slot_8 = __v3.wrapping_add(1);
}
return *__slate_slot_9;
}
loop {
let __v1: i32 = *__slate_slot_8;
let __v0: i32 = __v1.wrapping_sub(1);
*__slate_slot_8 = __v0;
if (__v0 as u32) >= (*set).nr_hw_queues {
if !blk_mq_is_shared_tags((*set).flags) {
blk_mq_free_map_and_rqs(set, *(*__slate_slot_9).wrapping_offset(*__slate_slot_8 as isize), *__slate_slot_8 as u32);
}
} else {
break;
}
}
kfree(*__slate_slot_9 as *const ());
return ERR_PTR(12i32.wrapping_neg() as i64) as *mut *mut blk_mq_tags;
}
}
}
}

// Alloc a tag set to be associated with one or more request queues.
// May fail with EINVAL for various error conditions. May adjust the
// requested depth down, if it's too large. In that case, the set
// value will be stored in set->queue_depth.
static mut __srcu_key__4894: lock_class_key = unsafe { core::mem::zeroed() };

static mut __srcu_key__4898: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__4902: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__4926: lock_class_key = unsafe { core::mem::zeroed() };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_alloc_tag_set_1019: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut blk_mq_tag_set) -> i32>, *mut ()>(Some(blk_mq_alloc_tag_set)) };

/// allocate and initialize a tagset for a simple single-queue device
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_alloc_sq_tag_set")]
extern "C" fn blk_mq_alloc_sq_tag_set(set: *mut blk_mq_tag_set, ops: *const blk_mq_ops, queue_depth: u32, set_flags: u32) -> i32 {
unsafe { memset(set as *mut (), 0i32, core::mem::size_of::<blk_mq_tag_set>() as u64) };
unsafe {
(*set).ops = ops;
}
unsafe {
(*set).nr_hw_queues = 1;
}
unsafe {
(*set).nr_maps = 1;
}
unsafe {
(*set).queue_depth = queue_depth;
}
unsafe {
(*set).numa_node = 1i32.wrapping_neg();
}
unsafe {
(*set).flags = set_flags;
}
blk_mq_alloc_tag_set(set)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_alloc_sq_tag_set_1020: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut blk_mq_tag_set, *const blk_mq_ops, u32, u32) -> i32>, *mut ()>(Some(blk_mq_alloc_sq_tag_set)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_free_tag_set")]
extern "C" fn blk_mq_free_tag_set(set: *mut blk_mq_tag_set) {
let mut i: i32 = 0;
let mut j: i32 = 0;
while (i as u32) < unsafe { (*set).nr_hw_queues } {
__blk_mq_free_map_and_rqs(set, i as u32);
i = i.wrapping_add(1);
}
if blk_mq_is_shared_tags(unsafe { (*set).flags }) {
blk_mq_free_map_and_rqs(set, unsafe { (*set).shared_tags }, 1u32.wrapping_neg());
}
while (j as u32) < unsafe { (*set).nr_maps } {
unsafe { kfree((*((*set).map.as_mut_ptr() as *mut blk_mq_queue_map).wrapping_offset(j as isize)).mq_map as *const ()) };
unsafe {
(*((*set).map.as_mut_ptr() as *mut blk_mq_queue_map).wrapping_offset(j as isize)).mq_map = core::ptr::null_mut::<u32>();
}
j = j.wrapping_add(1);
}
unsafe { kfree((*set).tags as *const ()) };
unsafe {
(*set).tags = core::ptr::null_mut::<*mut blk_mq_tags>();
}
unsafe { srcu_barrier(core::ptr::addr_of_mut!((*set).tags_srcu)) };
unsafe { cleanup_srcu_struct(core::ptr::addr_of_mut!((*set).tags_srcu)) };
if (unsafe { (*set).flags }) & 16 != 0 {
unsafe { srcu_barrier((*set).srcu) };
unsafe { cleanup_srcu_struct((*set).srcu) };
unsafe { kfree((*set).srcu as *const ()) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_free_tag_set_1021: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut blk_mq_tag_set)>, *mut ()>(Some(blk_mq_free_tag_set)) };

/// Switch back to the elevator type stored in the xarray.
fn blk_mq_elv_switch_back(q: *mut request_queue, elv_tbl: *mut xarray) {
let ctx: *mut elv_change_ctx = (unsafe { xa_load(elv_tbl, (*q).id as u64) }) as *mut elv_change_ctx;
let mut __v0: i64;
let __ret_warn_on: i32 = ctx.is_null() as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("1025: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1025b - ., 3; .popsection\n# {0}", const 1025 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_blk_mq_elv_switch_back_1, sym __slate_asm_string_blk_mq_elv_switch_back_2, const 5064 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("1026: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1026b - ., 4; .popsection\n# {0}", const 1026 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
return;
}
// The elv_update_nr_hw_queues unfreezes the queue.
unsafe { elv_update_nr_hw_queues(q, ctx) };
// Drop the reference acquired in blk_mq_elv_switch_none.
if !unsafe { (*ctx).r#type }.is_null() {
elevator_put(unsafe { (*ctx).r#type });
}
}

/// Stores elevator name and type in ctx and set current elevator to none.
fn blk_mq_elv_switch_none(q: *mut request_queue, elv_tbl: *mut xarray) -> i32 {
lockdep_assert_held_write!(unsafe { core::ptr::addr_of_mut!((*(*q).tag_set).update_nr_hwq_lock) });
// Accessing q->elevator without holding q->elevator_lock is safe here
// because we're called from nr_hw_queue update which is protected by
// set->update_nr_hwq_lock in the writer context. So, scheduler update/
// switch code (which acquires the same lock in the reader context)
// can't run concurrently.
if !unsafe { (*q).elevator }.is_null() {
let ctx: *mut elv_change_ctx = (unsafe { xa_load(elv_tbl, (*q).id as u64) }) as *mut elv_change_ctx;
let mut __v0: i64;
let __ret_warn_on: i32 = ctx.is_null() as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("1027: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1027b - ., 3; .popsection\n# {0}", const 1027 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_blk_mq_elv_switch_none_1, sym __slate_asm_string_blk_mq_elv_switch_none_2, const 5094 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("1028: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1028b - ., 4; .popsection\n# {0}", const 1028 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
return 2i32.wrapping_neg();
}
unsafe {
(*ctx).name = (*(*(*q).elevator).r#type).elevator_name;
}
// Before we switch elevator to 'none', take a reference to
// the elevator module so that while nr_hw_queue update is
// running, no one can remove elevator module. We'd put the
// reference to elevator module later when we switch back
// elevator.
__elevator_get(unsafe { (*(*q).elevator).r#type });
// Store elevator type so that we can release the reference
// taken above later.
unsafe {
(*ctx).r#type = (*(*q).elevator).r#type;
}
unsafe { elevator_set_none(q) };
}
0
}

fn __blk_mq_update_nr_hw_queues(set: *mut blk_mq_tag_set, mut nr_hw_queues: i32) {
let mut i: i32 = unsafe { core::mem::zeroed() };
let mut queues_frozen: bool = unsafe { core::mem::zeroed() };
let mut new_tags: core::mem::MaybeUninit<*mut *mut blk_mq_tags> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut *mut blk_mq_tags = core::ptr::addr_of_mut!(new_tags) as *mut *mut *mut blk_mq_tags;
let mut elv_tbl: core::mem::MaybeUninit<xarray> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut xarray = core::ptr::addr_of_mut!(elv_tbl) as *mut xarray;
let mut i_1: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut i32 = core::ptr::addr_of_mut!(i_1) as *mut i32;
let mut memflags: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut u32 = core::ptr::addr_of_mut!(memflags) as *mut u32;
let mut prev_nr_hw_queues: i32 = unsafe { core::mem::zeroed() };
let mut q: core::mem::MaybeUninit<*mut request_queue> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut request_queue = core::ptr::addr_of_mut!(q) as *mut *mut request_queue;
unsafe {
let __slate_auto_bytes = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
prev_nr_hw_queues = (*set).nr_hw_queues as i32;
let __slate_auto_bytes_1 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 16usize);
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
queues_frozen = false;
lockdep_assert_held__2!(core::ptr::addr_of_mut!((*set).tag_list_lock));
if (*set).nr_maps == 1 && (nr_hw_queues as u32) > nr_cpu_ids {
nr_hw_queues = nr_cpu_ids as i32;
}
if nr_hw_queues < 1 {
return;
} else {
if (*set).nr_maps == 1 && (nr_hw_queues as u32) == (*set).nr_hw_queues {
return;
} else {
*__slate_slot_10 = memalloc_noio_save();
xa_init(&mut (*core::ptr::addr_of_mut!(*__slate_slot_8)));
if blk_mq_alloc_sched_ctx_batch(core::ptr::addr_of_mut!(*__slate_slot_8), set) >= 0 {
if blk_mq_alloc_sched_res_batch(core::ptr::addr_of_mut!(*__slate_slot_8), set, nr_hw_queues as u32) >= 0 {
*__slate_slot_12 = list_first_entry!(core::ptr::addr_of_mut!((*set).tag_list));
while list_entry_is_head!(*__slate_slot_12, core::ptr::addr_of_mut!((*set).tag_list), tag_set_list) == 0 {
blk_mq_debugfs_unregister_hctxs(*__slate_slot_12);
blk_mq_sysfs_unregister_hctxs(*__slate_slot_12);
*__slate_slot_12 = list_next_entry!(*__slate_slot_12, tag_set_list);
}
// Switch IO scheduler to 'none', cleaning up the data associated
// with the previous scheduler. We will switch back once we are done
// updating the new sw to hw queue mappings.
*__slate_slot_12 = list_first_entry!(core::ptr::addr_of_mut!((*set).tag_list));
'__join_11: {
while list_entry_is_head!(*__slate_slot_12, core::ptr::addr_of_mut!((*set).tag_list), tag_set_list) == 0 {
if blk_mq_elv_switch_none(*__slate_slot_12, core::ptr::addr_of_mut!(*__slate_slot_8)) != 0 {
break '__join_11;
} else {
*__slate_slot_12 = list_next_entry!(*__slate_slot_12, tag_set_list);
}
}
*__slate_slot_7 = blk_mq_prealloc_tag_set_tags(set, nr_hw_queues);
if !IS_ERR(*__slate_slot_7 as *const ()) {
*__slate_slot_12 = list_first_entry!(core::ptr::addr_of_mut!((*set).tag_list));
while list_entry_is_head!(*__slate_slot_12, core::ptr::addr_of_mut!((*set).tag_list), tag_set_list) == 0 {
blk_mq_freeze_queue_nomemsave(*__slate_slot_12);
*__slate_slot_12 = list_next_entry!(*__slate_slot_12, tag_set_list);
}
queues_frozen = true;
if !(*__slate_slot_7).is_null() {
kfree((*set).tags as *const ());
(*set).tags = *__slate_slot_7;
}
(*set).nr_hw_queues = nr_hw_queues as u32;
loop {
blk_mq_update_queue_map(set);
*__slate_slot_12 = list_first_entry!(core::ptr::addr_of_mut!((*set).tag_list));
loop {
if list_entry_is_head!(*__slate_slot_12, core::ptr::addr_of_mut!((*set).tag_list), tag_set_list) == 0 {
__blk_mq_realloc_hw_ctxs(set, *__slate_slot_12);
if (*(*__slate_slot_12)).nr_hw_queues != (*set).nr_hw_queues {
break;
} else {
blk_mq_map_swqueue(*__slate_slot_12);
*__slate_slot_12 = list_next_entry!(*__slate_slot_12, tag_set_list);
}
} else {
break '__join_11;
}
}
i = prev_nr_hw_queues;
while false {
}
_printk(c"\x014Increasing nr_hw_queues to %d fails, fallback to %d\n".as_ptr() as *const u8, nr_hw_queues, prev_nr_hw_queues);
while (i as u32) < (*set).nr_hw_queues {
__blk_mq_free_map_and_rqs(set, i as u32);
i = i.wrapping_add(1);
}
(*set).nr_hw_queues = prev_nr_hw_queues as u32;
}
}
}
*__slate_slot_12 = list_first_entry!(core::ptr::addr_of_mut!((*set).tag_list));
while list_entry_is_head!(*__slate_slot_12, core::ptr::addr_of_mut!((*set).tag_list), tag_set_list) == 0 {
// switch_back expects queue to be frozen
if !queues_frozen {
blk_mq_freeze_queue_nomemsave(*__slate_slot_12);
}
blk_mq_elv_switch_back(*__slate_slot_12, core::ptr::addr_of_mut!(*__slate_slot_8));
*__slate_slot_12 = list_next_entry!(*__slate_slot_12, tag_set_list);
}
// The blk_mq_elv_switch_back unfreezes queue for us.
*__slate_slot_12 = list_first_entry!(core::ptr::addr_of_mut!((*set).tag_list));
while list_entry_is_head!(*__slate_slot_12, core::ptr::addr_of_mut!((*set).tag_list), tag_set_list) == 0 {
blk_mq_sysfs_register_hctxs(*__slate_slot_12);
blk_mq_debugfs_register_hctxs(*__slate_slot_12);
blk_mq_remove_hw_queues_cpuhp(*__slate_slot_12);
blk_mq_add_hw_queues_cpuhp(*__slate_slot_12);
*__slate_slot_12 = list_next_entry!(*__slate_slot_12, tag_set_list);
}
}
}
blk_mq_free_sched_ctx_batch(core::ptr::addr_of_mut!(*__slate_slot_8));
xa_destroy(core::ptr::addr_of_mut!(*__slate_slot_8));
memalloc_noio_restore(*__slate_slot_10);
// Free the excess tags when nr_hw_queues shrink.
*__slate_slot_9 = (*set).nr_hw_queues as i32;
while *__slate_slot_9 < prev_nr_hw_queues {
__blk_mq_free_map_and_rqs(set, *__slate_slot_9 as u32);
let __v1: i32 = *__slate_slot_9;
*__slate_slot_9 = __v1.wrapping_add(1);
}
}
}
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_update_nr_hw_queues")]
extern "C" fn blk_mq_update_nr_hw_queues(set: *mut blk_mq_tag_set, nr_hw_queues: i32) {
unsafe { down_write(core::ptr::addr_of_mut!((*set).update_nr_hwq_lock)) };
unsafe { mutex_lock(core::ptr::addr_of_mut!((*set).tag_list_lock)) };
__blk_mq_update_nr_hw_queues(set, nr_hw_queues);
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*set).tag_list_lock)) };
unsafe { up_write(core::ptr::addr_of_mut!((*set).update_nr_hwq_lock)) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_update_nr_hw_queues_1029: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut blk_mq_tag_set, i32)>, *mut ()>(Some(blk_mq_update_nr_hw_queues)) };

fn blk_hctx_poll(q: *mut request_queue, hctx: *mut blk_mq_hw_ctx, iob: *mut io_comp_batch, flags: u32) -> i32 {
let mut ret: i32 = 0;
let timeout: u64 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) }.wrapping_add(2);
loop {
ret = unsafe { unsafe { (*(*q).mq_ops).poll }.unwrap()(hctx, iob) };
if ret > 0 {
return ret;
}
if task_sigpending(get_current()) != 0 {
return 1;
}
if ret < 0 || flags & 1u32 != 0 {
break;
}
cpu_relax();
let __v1: bool = !need_resched() && time_before!(unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) }, timeout);
if !__v1 {
break;
}
}
0
}

#[unsafe(no_mangle)]
extern "C" fn blk_mq_poll(q: *mut request_queue, cookie: u32, iob: *mut io_comp_batch, flags: u32) -> i32 {
if !blk_mq_can_poll(q) {
return 0;
}
blk_hctx_poll(q, unsafe { *unsafe { (*q).queue_hw_ctx }.wrapping_offset(cookie as isize) }, iob, flags)
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_rq_poll")]
extern "C" fn blk_rq_poll(rq: *mut request, iob: *mut io_comp_batch, poll_flags: u32) -> i32 {
let q: *mut request_queue = unsafe { (*rq).q };
let mut ret: i32 = 0;
if !blk_rq_is_poll(rq) {
return 0;
}
if !percpu_ref_tryget(unsafe { core::ptr::addr_of_mut!((*q).q_usage_counter) }) {
return 0;
}
ret = blk_hctx_poll(q, unsafe { (*rq).mq_hctx }, iob, poll_flags);
unsafe { blk_queue_exit(q) };
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_rq_poll_1030: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request, *mut io_comp_batch, u32) -> i32>, *mut ()>(Some(blk_rq_poll)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.blk_mq.blk_mq_rq_cpu")]
extern "C" fn blk_mq_rq_cpu(rq: *mut request) -> u32 {
unsafe { (*(*rq).mq_ctx).cpu }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_rq_cpu_1031: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request) -> u32>, *mut ()>(Some(blk_mq_rq_cpu)) };

#[unsafe(no_mangle)]
extern "C" fn blk_mq_cancel_work_sync(q: *mut request_queue) {
let mut hctx: *mut blk_mq_hw_ctx = unsafe { core::mem::zeroed() };
let mut i: u64 = 0;
unsafe { cancel_delayed_work_sync(core::ptr::addr_of_mut!((*q).requeue_work)) };
loop {
let __v0: bool = if i < ((unsafe { (*q).nr_hw_queues }) as u64) {
let mut __v1: i32;
hctx = queue_hctx(q, i as i32);
__v1 = 1;
__v1 != 0
} else {
false
};
if !__v0 {
break;
}
unsafe { cancel_delayed_work_sync(core::ptr::addr_of_mut!((*hctx).run_work)) };
i = i.wrapping_add(1);
}
}

#[unsafe(link_section = ".init.text")]
extern "C" fn blk_mq_init() -> i32 {
let mut __slate_compound_6: __call_single_data = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
loop {
i = find_next_bit(cpumask_bits!((unsafe { core::ptr::addr_of_mut!(__cpu_possible_mask) }) as *const cpumask), 64u64, i as u64) as i32;
if (i as u32) >= 64 {
break;
}
init_llist_head(per_cpu_ptr__2!(unsafe { core::ptr::addr_of_mut!(blk_cpu_done) }, i));
i = i.wrapping_add(1);
}
i = 0;
'__slate_break_11: loop {
i = find_next_bit(cpumask_bits!((unsafe { core::ptr::addr_of_mut!(__cpu_possible_mask) }) as *const cpumask), 64u64, i as u64) as i32;
if (i as u32) >= 64 {
break;
}
'__slate_break_10: loop {
unsafe {
*per_cpu_ptr__10!(core::ptr::addr_of_mut!(blk_cpu_csd.0), i) = *{
__slate_compound_6 = {
let mut __t0: __call_single_data = core::mem::zeroed();
__t0.func = Some(__blk_mq_complete_request_remote);
__t0.info = core::ptr::null_mut::<()>();
__t0
};
core::ptr::addr_of_mut!(__slate_compound_6)
};
}
if true {
break;
}
}
i = i.wrapping_add(1);
}
unsafe { open_softirq(4i32, Some(blk_done_softirq)) };
cpuhp_setup_state_nocalls(20i32, c"block/softirq:dead".as_ptr() as *const u8, None, Some(blk_softirq_cpu_dead));
cpuhp_setup_state_multi(24i32, c"block/mq:dead".as_ptr() as *const u8, None, Some(blk_mq_hctx_notify_dead));
cpuhp_setup_state_multi(149i32, c"block/mq:online".as_ptr() as *const u8, Some(blk_mq_hctx_notify_online), Some(blk_mq_hctx_notify_offline));
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_mq_init_1033: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i32>, *mut ()>(Some(blk_mq_init)) };

static __slate_asm_string___blk_mq_realloc_hw_ctxs_20: [u8; 1] = *b"\0";

static __slate_asm_string___blk_mq_realloc_hw_ctxs_21: [u8; 44] = *b"/home/takashi/c-corpus/linux/block/blk-mq.c\0";

static __slate_asm_string___blk_mq_unfreeze_queue_4: [u8; 1] = *b"\0";

static __slate_asm_string___blk_mq_unfreeze_queue_5: [u8; 44] = *b"/home/takashi/c-corpus/linux/block/blk-mq.c\0";

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_blk_execute_rq_12: [u8; 1] = *b"\0";

static __slate_asm_string_blk_execute_rq_13: [u8; 44] = *b"/home/takashi/c-corpus/linux/block/blk-mq.c\0";

static __slate_asm_string_blk_execute_rq_17: [u8; 1] = *b"\0";

static __slate_asm_string_blk_execute_rq_18: [u8; 44] = *b"/home/takashi/c-corpus/linux/block/blk-mq.c\0";

static __slate_asm_string_blk_execute_rq_nowait_11: [u8; 1] = *b"\0";

static __slate_asm_string_blk_execute_rq_nowait_12: [u8; 44] = *b"/home/takashi/c-corpus/linux/block/blk-mq.c\0";

static __slate_asm_string_blk_execute_rq_nowait_16: [u8; 1] = *b"\0";

static __slate_asm_string_blk_execute_rq_nowait_17: [u8; 44] = *b"/home/takashi/c-corpus/linux/block/blk-mq.c\0";

static __slate_asm_string_blk_mq_alloc_request_hctx_17: [u8; 1] = *b"\0";

static __slate_asm_string_blk_mq_alloc_request_hctx_18: [u8; 44] = *b"/home/takashi/c-corpus/linux/block/blk-mq.c\0";

static __slate_asm_string_blk_mq_alloc_request_hctx_19: [u8; 1] = *b"\0";

static __slate_asm_string_blk_mq_alloc_request_hctx_20: [u8; 44] = *b"/home/takashi/c-corpus/linux/block/blk-mq.c\0";

static __slate_asm_string_blk_mq_bio_to_request_5: [u8; 1] = *b"\0";

static __slate_asm_string_blk_mq_bio_to_request_6: [u8; 44] = *b"/home/takashi/c-corpus/linux/block/blk-mq.c\0";

static __slate_asm_string_blk_mq_clear_flush_rq_mapping_1: [u8; 1] = *b"\0";

static __slate_asm_string_blk_mq_clear_flush_rq_mapping_2: [u8; 44] = *b"/home/takashi/c-corpus/linux/block/blk-mq.c\0";

static __slate_asm_string_blk_mq_clear_rq_mapping_2: [u8; 1] = *b"\0";

static __slate_asm_string_blk_mq_clear_rq_mapping_3: [u8; 44] = *b"/home/takashi/c-corpus/linux/block/blk-mq.c\0";

static __slate_asm_string_blk_mq_destroy_queue_1: [u8; 1] = *b"\0";

static __slate_asm_string_blk_mq_destroy_queue_2: [u8; 44] = *b"/home/takashi/c-corpus/linux/block/blk-mq.c\0";

static __slate_asm_string_blk_mq_destroy_queue_7: [u8; 1] = *b"\0";

static __slate_asm_string_blk_mq_destroy_queue_8: [u8; 44] = *b"/home/takashi/c-corpus/linux/block/blk-mq.c\0";

static __slate_asm_string_blk_mq_dispatch_rq_list_19: [u8; 1] = *b"\0";

static __slate_asm_string_blk_mq_dispatch_rq_list_20: [u8; 44] = *b"/home/takashi/c-corpus/linux/block/blk-mq.c\0";

static __slate_asm_string_blk_mq_elv_switch_back_1: [u8; 1] = *b"\0";

static __slate_asm_string_blk_mq_elv_switch_back_2: [u8; 44] = *b"/home/takashi/c-corpus/linux/block/blk-mq.c\0";

static __slate_asm_string_blk_mq_elv_switch_none_1: [u8; 1] = *b"\0";

static __slate_asm_string_blk_mq_elv_switch_none_2: [u8; 44] = *b"/home/takashi/c-corpus/linux/block/blk-mq.c\0";

static __slate_asm_string_blk_mq_end_request_0: [u8; 1] = *b"\0";

static __slate_asm_string_blk_mq_end_request_1: [u8; 44] = *b"/home/takashi/c-corpus/linux/block/blk-mq.c\0";

static __slate_asm_string_blk_mq_insert_request_1: [u8; 1] = *b"\0";

static __slate_asm_string_blk_mq_insert_request_2: [u8; 44] = *b"/home/takashi/c-corpus/linux/block/blk-mq.c\0";

static __slate_asm_string_blk_mq_insert_requests_3: [u8; 1] = *b"\0";

static __slate_asm_string_blk_mq_insert_requests_4: [u8; 44] = *b"/home/takashi/c-corpus/linux/block/blk-mq.c\0";

static __slate_asm_string_blk_mq_limit_depth_4: [u8; 1] = *b"\0";

static __slate_asm_string_blk_mq_limit_depth_5: [u8; 44] = *b"/home/takashi/c-corpus/linux/block/blk-mq.c\0";

static __slate_asm_string_blk_mq_map_swqueue_13: [u8; 1] = *b"\0";

static __slate_asm_string_blk_mq_map_swqueue_14: [u8; 44] = *b"/home/takashi/c-corpus/linux/block/blk-mq.c\0";

static __slate_asm_string_blk_mq_map_swqueue_28: [u8; 1] = *b"\0";

static __slate_asm_string_blk_mq_map_swqueue_29: [u8; 44] = *b"/home/takashi/c-corpus/linux/block/blk-mq.c\0";

static __slate_asm_string_blk_mq_release_2: [u8; 1] = *b"\0";

static __slate_asm_string_blk_mq_release_3: [u8; 44] = *b"/home/takashi/c-corpus/linux/block/blk-mq.c\0";

static __slate_asm_string_blk_mq_rq_timed_out_4: [u8; 1] = *b"\0";

static __slate_asm_string_blk_mq_rq_timed_out_5: [u8; 44] = *b"/home/takashi/c-corpus/linux/block/blk-mq.c\0";

static __slate_asm_string_blk_mq_run_hw_queue_2: [u8; 1] = *b"\0";

static __slate_asm_string_blk_mq_run_hw_queue_3: [u8; 44] = *b"/home/takashi/c-corpus/linux/block/blk-mq.c\0";

static __slate_asm_string_blk_mq_start_request_6: [u8; 1] = *b"\0";

static __slate_asm_string_blk_mq_start_request_7: [u8; 44] = *b"/home/takashi/c-corpus/linux/block/blk-mq.c\0";

static __slate_asm_string_blk_mq_unquiesce_queue_4: [u8; 1] = *b"\0";

static __slate_asm_string_blk_mq_unquiesce_queue_5: [u8; 44] = *b"/home/takashi/c-corpus/linux/block/blk-mq.c\0";

static __slate_asm_string_blk_mq_update_queue_map_3: [u8; 1] = *b"\0";

static __slate_asm_string_blk_mq_update_queue_map_4: [u8; 44] = *b"/home/takashi/c-corpus/linux/block/blk-mq.c\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

static __slate_asm_string_percpu_ref_tryget_live_rcu_1: [u8; 1] = *b"\0";

static __slate_asm_string_percpu_ref_tryget_live_rcu_2: [u8; 61] = *b"/home/takashi/c-corpus/linux/include/linux/percpu-refcount.h\0";

static __slate_asm_string_req_ref_put_and_test_1: [u8; 1] = *b"\0";

static __slate_asm_string_req_ref_put_and_test_2: [u8; 41] = *b"/home/takashi/c-corpus/linux/block/blk.h\0";

static __slate_asm_string_srcu_read_unlock_1: [u8; 1] = *b"\0";

static __slate_asm_string_srcu_read_unlock_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/srcu.h\0";

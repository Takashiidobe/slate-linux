
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! context_unsafe() - disable context checking for contained code
//!
//! Disables context checking for contained statements or expression.
//!
//! .. code-block:: c
//!
//! 	struct some_data {
//! 		spinlock_t lock;
//! 		int counter __guarded_by(&lock);
//! 	};
//!
//! 	int foo(struct some_data *d)
//! 	{
//! 		// ...
//! 		// other code that is still checked ...
//! 		// ...
//! 		return context_unsafe(d->counter);
//! 	}
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! SPDX-License-Identifier: GPL-2.0-only
//!
//! Copyright (C) 2001, 2002 Sistina Software (UK) Limited.
//! Copyright (C) 2004-2008 Red Hat, Inc. All rights reserved.
//!
//! This file is released under the GPL.
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
//! Operation not permitted
//! Value too large for defined data type
//! Kernel pointers have redundant information, so we can use a
//! scheme where we can return either an error code or a normal
//! pointer with the same return value.
//!
//! This should be a per-architecture thing, to allow different
//! error and pointer decisions.
//! SCHED_ISO: reserved but not implemented yet
//! Force a compilation error if condition is true, but also produce a
//! result (of value 0 and type int), so the expression can be used
//! e.g. in a structure initializer (or where-ever else comma expressions
//! aren't permitted).
//!
//! Take an error message as an optional second argument. If omitted,
//! default to the stringification of the tested expression.
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
//! This counts to 15. Any more, it will return 16th argument.
//! # chars in a file name
//! See linux/instrumentation.h
//! 64 bit
//! unused, was: #define EX_TYPE_COPY	 4
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
//! Sign Flag
//! These are for everybody (although not all archs will actually
//! discard it in modules)
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
//! Begin/end of an instrumentation safe region
//! Call a function once. Similar to DO_ONCE(), but does not use jump label
//! patching via static keys.
//! warning conditions
//! Internal kernel timer frequency
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! issue num suppressed message on exit
//! strlen("ratelimit") + 1
//! label
//! Bit size and mask of CPU number stored in the per CPU data (and TSC_AUX)
//! PAGE_SHIFT determines the page size.
//!
//! Note: This definition is required because PAGE_SHIFT is used
//! in several places throughout the codebase.
//! Defines x86 CPU feature bits
//! Ensure that leaf 0x2 cache and TLB type values do not intersect,
//! since they share the same type field at struct cpuid_0x2_table.
//! virt_to_page(kaddr) returns a valid pointer if and only if
//! virt_addr_valid(kaddr) returns true.
//! Prevent speculative execution past this barrier.
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
//! We do the locked ops that don't return the old value as
//! a mask operation on a byte.
//! writeable
//! These should never be seen by user programs.  To return one of ERESTART*
//! codes, signal_pending() MUST be set.  Note that ptrace can observe these
//! at syscall exit tracing, but they will never be left for the debugged user
//! process to see.
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
//! @a is a power of 2 value
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
//! This looks more complex than it should be. But we need to
//! get the type for the ~ right in round_down (it needs to be
//! as wide as the result!), and we want to evaluate the macro
//! arguments just once each.
//! These are non-NULL pointers that will result in page faults
//! under normal circumstances, used to verify that nobody uses
//! non-initialized list entries.
//! LIST_HEAD_INIT - initialize a &struct list_head's links to point to itself
//! @name: name of the list_head
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
//! This is the default CPU features testing macro to use in code.
//!
//! It is for detection of features which need kernel infrastructure to be
//! used.  It may *not* directly test the CPU itself.  Use the cpu_has() family
//! if you want true runtime testing of CPU features, like in hypervisor code
//! where you are supporting a possible guest feature where host support for it
//! is not relevant.
//! signal pending
//! Macros to retrieve the current execution context:
//!
//! in_nmi()		- We're in NMI context
//! in_hardirq()		- We're in hard IRQ context
//! in_serving_softirq()	- We're in softirq context
//! in_task()		- We're in task context
//! These constants are defined in the MSan LLVM instrumentation pass.
//! ignore signal
//! bit-flags
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
//! GFP_ZONE_BAD is a bitmap for all combinations of __GFP_DMA, __GFP_DMA32
//! __GFP_HIGHMEM and __GFP_MOVABLE that are not permitted. One flag per
//! entry starting with bit 0. Bit is set if the combination is not
//! allowed.
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
//! Must be a compile-time constant, so implement it as a macro
//! Resource type
//! Bits set in the VMA until the stack is in its final location
//! number of env pointers
//! kunmap_local - Unmap a page mapped via kmap_local_page().
//! @__addr: An address within the page mapped
//!
//! @__addr can be any address within the mapped page.  Commonly it is the
//! address return from kmap_local_page(), but it can also include offsets.
//!
//! Unmapping should be done in the reverse order of the mapping.  See
//! kmap_local_page() for details.
//! Generate SD flag indexes
//! Desired design of maximum size and alignment (see RFC2553)
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
//! Generic info of form tag = "info"
//! Maximum number of blkcg policies allowed to be registered concurrently.
//! Defined here to simplify include dependency.
//! See the comment in asm-generic/io.h about ioremap_np().
//! These macros should be used after a dma_map_sg call has been done
//! to get bus addresses of each of the SG entries and their lengths.
//! You should only work with the number of sg entries dma_map_sg
//! returns, or alternatively stop on the first sg_dma_len(sg) which
//! is 0.
//! Size of the "software secret" which can be derived from a hardware-wrapped
//! key.  This is currently always 32 bytes.  Note, the choice of 32 bytes
//! assumes that the software secret is only used directly for algorithms that
//! don't require more than a 256-bit key to get the desired security strength.
//! If it were to be used e.g. directly as an AES-256-XTS key, then this would
//! need to be increased (which is possible if hardware supports it, but care
//! would need to be taken to avoid breaking users who need exactly 32 bytes).
//! These are specified by iBCS2
//! Epoll event masks
//! macro tricks to expand the set_buffer_foo(), clear_buffer_foo()
//! and buffer_foo() functions.
//! To avoid reset buffer flags that are already set, because that causes
//! a costly cache line transition, check the flag first.
//! Currently, the max observed size in the kernel code is
//! JUMP_LABEL_NOP_SIZE/RELATIVEJUMP_SIZE, which are 5.
//! Raise it if needed.
//! 31 pointers + header align the folio_batch structure to a power of two
//! Flags in backing_dev_info::capability
//!
//! BDI_CAP_WRITEBACK:		Supports dirty page writeback, and dirty pages
//! 				should contribute to accounting
//! BDI_CAP_STRICTLIMIT:		Keep number of dirty pages below bdi threshold
//! Suspend feature flags
//! Sizes of name fields. Unused at present.
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
// Note the missing underscores.
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-noinline-function-attribute
// clang: mentioned
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
// &a[0] degrades to a pointer: a different type from an array
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
// No such file or directory
// I/O error
// No such device or address
// Argument list too long
// Try again
// Out of memory
// Bad address
// Device or resource busy
// File exists
// Invalid argument
// Not a typewriter
// Text file busy
// No space left on device
// Operation not supported on transport endpoint
// Transport endpoint is not connected
// Memory page has hardware error
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
// Concatenate two parameters, but allow them to be expanded beforehand.
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
// check_mul_overflow() - Calculate multiplication with overflow checking
// @a: first factor
// @b: second factor
// @d: pointer to store product
//
// Returns true on wrap-around, false otherwise.
//
// *@d holds the results of the attempted multiplication, regardless of whether
// wrap-around occurred.
// objtool annotation to ignore the alternatives and only consider the original
// instruction(s).
// This should be used to refer to an instruction that is considered
// terminating, like a noreturn CALL or UD2 when we know they are not -- eg
// WARN using UD2.
// Annotate a special section entry.  This emables livepatch module generation
// to find and extract individual special section entries as needed.
// reg := -EIO
// reg := -EIO
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
// Overflow Flag
// Format: <modname>__<counter>_<line>_<fn>
// Format: __<prefix>__<iid><id>
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
// __swahw32 - return a word-swapped 32-bit value
// @x: value to wordswap
//
// __swahw32(0x12340000) is 0x00001234
// __swahb32 - return a high and low byte-swapped 32-bit value
// @x: value to byteswap
//
// __swahb32(0x12345678) is 0x34127856
// Dummy printk for disabled debugging statements to use whilst maintaining
// gcc's format checking.
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
// pr_crit - Print a critical-level message
// @fmt: format string
// @...: arguments for the format string
//
// This macro expands to a printk with KERN_CRIT loglevel. It uses pr_fmt() to
// generate the format string.
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
// N 32-bit words worth of info
// N 32-bit bug flags
// "tsc" Time Stamp Counter
// "apic" Onboard APIC
// "sse"
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
// "fsrm" Fast Short Rep Mov
// "serialize" SERIALIZE instruction
// LFENCE always serializing / synchronizes RDTSC
// BUG word(s)
// "l1tf" CPU is affected by L1 Terminal Fault
// "div0" AMD DIV0 speculation bug
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
// No ioctl command
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
// Use these carefully: no type checking, and uses the arguments
// multiple times. Use for obvious constants only.
// round_up - round up to next specified power of 2
// @x: the value to round
// @y: multiple to round up to (must be a power of 2)
//
// Rounds @x up to next multiple of @y (which must be a power of 2).
// To perform arbitrary rounding up, use roundup() below.
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
// net/core/page_pool.c
//
// page_pool uses additional free bits within this value to store data, see the
// definition of PP_DMA_INDEX_MASK in mm.h
// N 32-bit words worth of info
// signal notifications exist
// rescheduling necessary
// idle is polling for TIF_NEED_RESCHED
// re-arm the timer
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
// disable sas during sighandling
// This function is needed by all SMP systems. It must _always_ be valid
// from the initial startup.
// If !LOCKDEP, the value is meaningless
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
// return the first or the next element in an RCU protected hlist
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
// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
// Resource extended types
// No address assigned yet
// Generate SD flag bits
// Implementation specific max size
// We overload the LSB of the page pointer to indicate whether it's
// a valid sg entry, or whether it points to the start of a new scatterlist.
// Those low bits are there for everyone! (thanks mason :-)
// test_set_buffer_foo() and test_clear_buffer_foo()
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
/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__21 {
() => {
((core::mem::size_of::<[Option<unsafe extern "C" fn()>; 8]>() as u64) / (core::mem::size_of::<Option<unsafe extern "C" fn()>>() as u64)).wrapping_add(__must_be_array__17!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__34 {
() => {
((core::mem::size_of::<[Option<unsafe extern "C" fn() -> i32>; 8]>() as u64) / (core::mem::size_of::<Option<unsafe extern "C" fn() -> i32>>() as u64)).wrapping_add(__must_be_array__50!() as u64)
};
}

macro_rules! BITMAP_LAST_WORD_MASK {
($nbits:expr) => {
18446744073709551615u64 >> ($nbits.wrapping_neg() & 63u32)
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

macro_rules! GENMASK_INPUT_CHECK {
() => {
core::mem::size_of::<__SlateRecord_find_h_67>() as i32
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

macro_rules! GENMASK {
($h:expr, $l:expr) => {
GENMASK_TYPE!(u64, $h, $l)
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
unlikely!(((($x) as *mut ()) as u64) >= (4095i32.wrapping_neg() as u64))
};
}

macro_rules! MAJOR {
($dev:expr) => {
($dev) >> 20i32
};
}

macro_rules! MINOR {
($dev:expr) => {
($dev) & 1048575u32
};
}

macro_rules! MKDEV {
($ma:expr, $mi:expr) => {
($ma) << 20i32 | $mi
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

/// RCU_INITIALIZER() - statically initialize an RCU-protected global variable
/// @v: The value to statically initialize with.
macro_rules! RCU_INITIALIZER {
($v:expr) => {
$v
};
}

macro_rules! RCU_LOCKDEP_WARN__2 {
($c:expr) => {
while false && $c {
}
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

macro_rules! _UL {
() => {
_AC!()
};
}

macro_rules! __READ_ONCE {
($x:expr) => {
unsafe { core::ptr::read_volatile(core::ptr::addr_of!($x) as *const i32) }
};
}

macro_rules! __WRITE_ONCE__4 {
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

macro_rules! __WRITE_ONCE__8 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut *const cred, $val) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE__18 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut *mut (), $val) };
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

macro_rules! ___wait_is_interruptible {
($state:expr) => {
false || ($state) & (1 | 256) != 0
};
}

/// __acquire() - function to acquire context lock exclusively
/// @x: context lock instance pointer
///
/// No-op function that acquires the given context lock instance @x exclusively.
macro_rules! __acquire {
() => {
__acquire_ctx_lock!();
};
}

macro_rules! __acquire_ctx_lock {
() => {
while false {
}
};
}

/// __acquire_shared() - function to acquire context lock shared
/// @x: context lock instance pointer
///
/// No-op function that acquires the given context lock instance @x with shared
/// access.
macro_rules! __acquire_shared {
() => {
__acquire_shared_ctx_lock!();
};
}

macro_rules! __acquire_shared_ctx_lock {
() => {
while false {
}
};
}

macro_rules! __alloc_percpu {
($_size:expr, $_align:expr) => {
alloc_hooks!(unsafe { pcpu_alloc_noprof($_size, $_align, false, 3264u32) })
};
}

macro_rules! __assume_ctx_lock {
($var:expr) => {
loop {
$var;
if true {
break;
}
}
};
}

macro_rules! __cmp_once__3 {
($type:ty, $x:expr, $y:expr) => {
{
let mut __v: $type;
let __v2: $type = $x;
let __v3: $type = ($y) as $type;
__v = if __v2 < __v3 { __v2 } else { __v3 };
__v
}
};
}

macro_rules! __cmp_once {
($type:ty, $x:expr, $y:expr) => {
{
let mut __v: $type;
let __v2: $type = ($x) as $type;
let __v3: $type = $y;
__v = if __v2 < __v3 { __v2 } else { __v3 };
__v
}
};
}

macro_rules! __const_hweight16 {
($w:expr) => {
__const_hweight8!($w).wrapping_add(__const_hweight8!(($w) >> 8i32))
};
}

macro_rules! __const_hweight32 {
($w:expr) => {
__const_hweight16!($w).wrapping_add(__const_hweight16!(($w) >> 16i32))
};
}

macro_rules! __const_hweight64 {
($w:expr) => {
__const_hweight32!($w).wrapping_add(__const_hweight32!(($w) >> 32i32))
};
}

/// Compile time versions of __arch_hweightN()
macro_rules! __const_hweight8 {
($w:expr) => {
((($w) & 1u64 != 0) as i32).wrapping_add((($w) & 2u64 != 0) as i32).wrapping_add((($w) & 4u64 != 0) as i32).wrapping_add((($w) & 8u64 != 0) as i32).wrapping_add((($w) & 16u64 != 0) as i32).wrapping_add((($w) & 32u64 != 0) as i32).wrapping_add((($w) & 64u64 != 0) as i32).wrapping_add((($w) & 128u64 != 0) as i32) as u32
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

macro_rules! __must_be_array__17 {
() => {
core::mem::size_of::<__SlateRecord_dm_c_295>() as i32
};
}

macro_rules! __must_be_array__50 {
() => {
core::mem::size_of::<__SlateRecord_dm_c_271>() as i32
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__2 {
() => {
((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__3 {
() => {
((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__9 {
() => {
((const { core::mem::size_of::<*const cred>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*const cred>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*const cred>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*const cred>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__5 {
() => {
((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__8 {
() => {
((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__7 {
() => {
((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__4 {
() => {
((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__18 {
() => {
((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word {
() => {
((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

macro_rules! __or {
() => {
____or!()
};
}

macro_rules! __ratelimit__23 {
($state:expr) => {
unsafe { ___ratelimit($state, c"dm_submit_bio".as_ptr() as *const u8) }
};
}

macro_rules! __rcu_dereference_protected__5 {
($p:expr, $c:expr) => {
{
let mut __v: *mut ();
RCU_LOCKDEP_WARN__2!(($c) == 0);
__v = $p;
__v
}
};
}

macro_rules! __rcu_dereference_protected {
($p:expr, $c:expr) => {
{
let mut __v: *const cred;
RCU_LOCKDEP_WARN__2!(($c) == 0);
__v = $p;
__v
}
};
}

macro_rules! __rcu_dereference_protected__6 {
($p:expr, $c:expr) => {
{
let mut __v: *mut ();
RCU_LOCKDEP_WARN!(($c) == 0);
__v = $p;
__v
}
};
}

/// __release() - function to release context lock exclusively
/// @x: context lock instance pointer
///
/// No-op function that releases the given context lock instance @x.
macro_rules! __release {
() => {
__release_ctx_lock!();
};
}

macro_rules! __release_ctx_lock {
() => {
while false {
}
};
}

/// __release_shared() - function to release context lock shared
/// @x: context lock instance pointer
///
/// No-op function that releases the given context lock instance @x with shared
/// access.
macro_rules! __release_shared {
() => {
__release_shared_ctx_lock!();
};
}

macro_rules! __release_shared_ctx_lock {
() => {
while false {
}
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
if is_signed_type__2!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul((((const { core::mem::size_of::<u32>() }) as u64) < 4) as i32)) }
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
macro_rules! __sign_use__5 {
($ux:expr) => {
if is_signed_type__6!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul(((core::mem::size_of::<i64>() as u64) < 4) as i32)) }
};
}

macro_rules! __smp_mb__after_atomic {
() => {
while false {
}
};
}

/// Atomic operations are already serializing on x86
macro_rules! __smp_mb__before_atomic {
() => {
while false {
}
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

macro_rules! __types_ok {
($ux:expr, $uy:expr) => {
__sign_use__2!($ux) & __sign_use__2!($uy)
};
}

macro_rules! __types_ok__10 {
($ux:expr, $uy:expr) => {
__sign_use__5!($ux) & __sign_use__5!($uy)
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
macro_rules! __verify_pcpu_ptr__2 {
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
macro_rules! __verify_pcpu_ptr__3 {
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
macro_rules! __verify_pcpu_ptr__15 {
() => {
loop {
let __v: *const () = core::ptr::null_mut::<u64>() as *const ();
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
macro_rules! __verify_pcpu_ptr__5 {
() => {
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
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

macro_rules! arch_atomic_fetch_sub {
($i:expr, $v:expr) => {
arch_atomic_fetch_add($i.wrapping_neg(), $v)
};
}

/// Return the data direction, READ or WRITE.
macro_rules! bio_data_dir {
($bio:expr) => {
if op_is_write(bio_op(($bio) as *const bio) as u32) { 1i32 } else { 0i32 }
};
}

macro_rules! bio_end_sector {
($bio:expr) => {
bvec_iter_end_sector!((*$bio).bi_iter)
};
}

macro_rules! bio_sectors {
($bio:expr) => {
bvec_iter_sectors!((*$bio).bi_iter)
};
}

macro_rules! blk_queue_io_stat {
($q:expr) => {
(unsafe { (*$q).limits.features }) & 16u32
};
}

macro_rules! bvec_iter_end_sector {
($iter:expr) => {
unsafe { $iter.bi_sector }.wrapping_add(bvec_iter_sectors!($iter) as u64)
};
}

macro_rules! bvec_iter_sectors {
($iter:expr) => {
(unsafe { $iter.bi_size }) >> 9i32
};
}

macro_rules! cond_resched__7 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/drivers/md/dm.c".as_ptr() as *const u8, 1399i32, 0u32);
__v = _cond_resched();
};
}

macro_rules! cond_resched__11 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/drivers/md/dm.c".as_ptr() as *const u8, 1022i32, 0u32);
__v = _cond_resched();
};
}

macro_rules! cond_resched__13 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/drivers/md/dm.c".as_ptr() as *const u8, 1394i32, 0u32);
__v = _cond_resched();
};
}

macro_rules! cond_resched__15 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/drivers/md/dm.c".as_ptr() as *const u8, 2864i32, 0u32);
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
macro_rules! container_of__43 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__84!($type) as isize)) }) as *mut ()) as *mut $type;
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
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__19!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__63 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__20!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__74 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__134!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__13 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__5!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__19 {
($ptr:expr) => {
{
let mut __v: *mut dm_dev_internal;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__24!() as isize)) }) as *mut ()) as *mut dm_dev_internal;
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
macro_rules! container_of__111 {
($ptr:expr) => {
{
let mut __v: *mut table_device;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__109!() as isize)) }) as *mut ()) as *mut table_device;
__v
}
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

macro_rules! disk_to_dev {
($disk:expr) => {
unsafe { core::ptr::addr_of_mut!((*(*$disk).part0).bd_device) }
};
}

/// Sector offset taken relative to the start of the target instead of
/// relative to the start of the device.
macro_rules! dm_target_offset {
($ti:expr, $sector:expr) => {
$sector.wrapping_sub(unsafe { (*$ti).begin })
};
}

macro_rules! dm_target_supports_atomic_writes {
($type:expr) => {
(unsafe { (*$type).features }) & 1024
};
}

macro_rules! dm_target_supports_nowait {
($type:expr) => {
(unsafe { (*$type).features }) & 128
};
}

/// do_div - returns 2 values: calculate remainder and update new dividend
/// @n: uint64_t dividend (will be updated)
/// @base: uint32_t divisor
///
/// Summary:
/// ``uint32_t remainder = n % base;``
/// ``n = n / base;``
///
/// Return: (uint32_t)remainder
///
/// NOTE: macro parameter @n is evaluated multiple times,
/// beware of side effects!
macro_rules! do_div {
($n:expr, $base:expr) => {
{
let mut __v: u32;
let __v2: u32 = $base;
let __v3: u32 = ((($n) as u64) % (__v2 as u64)) as u32;
$n = (($n) as u64) / (__v2 as u64);
__v = __v3;
__v
}
};
}

macro_rules! flush_workqueue {
($wq:expr) => {
let _wq: *mut workqueue_struct = $wq;
if (unsafe { __slate_intrinsic_1cfe0293d0a2af8d(_wq == system_percpu_wq) }) && _wq == unsafe { system_percpu_wq } || (unsafe { __slate_intrinsic_1cfe0293d0a2af8d(_wq == system_highpri_wq) }) && _wq == unsafe { system_highpri_wq } || (unsafe { __slate_intrinsic_1cfe0293d0a2af8d(_wq == system_long_wq) }) && _wq == unsafe { system_long_wq } || (unsafe { __slate_intrinsic_1cfe0293d0a2af8d(_wq == system_dfl_long_wq) }) && _wq == unsafe { system_dfl_long_wq } || (unsafe { __slate_intrinsic_1cfe0293d0a2af8d(_wq == system_dfl_wq) }) && _wq == unsafe { system_dfl_wq } || (unsafe { __slate_intrinsic_1cfe0293d0a2af8d(_wq == system_freezable_wq) }) && _wq == unsafe { system_freezable_wq } || (unsafe { __slate_intrinsic_1cfe0293d0a2af8d(_wq == system_power_efficient_wq) }) && _wq == unsafe { system_power_efficient_wq } || (unsafe { __slate_intrinsic_1cfe0293d0a2af8d(_wq == system_freezable_power_efficient_wq) }) && _wq == unsafe { system_freezable_power_efficient_wq } {
unsafe { __warn_flushing_systemwide_wq() };
}
unsafe { __flush_workqueue(_wq) };
};
}

macro_rules! format_dev_t {
($buffer:expr, $dev:expr) => {
let mut __v: *mut u8;
unsafe { sprintf($buffer, c"%u:%u".as_ptr() as *const u8, MAJOR!($dev), MINOR!($dev)) };
__v = $buffer;
};
}

macro_rules! hweight32 {
($w:expr) => {
{
let __v: u32 = if unsafe { __slate_intrinsic_f217c4720f04f6db($w) } {
__const_hweight32!($w)
} else {
__arch_hweight32(($w) as u32)
};
__v
}
};
}

macro_rules! hweight64 {
($w:expr) => {
{
let __v: u64 = if unsafe { __slate_intrinsic_f217c4720f04f6db($w) } {
__const_hweight64!($w) as u64
} else {
__arch_hweight64($w)
};
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
macro_rules! is_signed_type__6 {
() => {
(1i32.wrapping_neg() as i64) < 1
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

macro_rules! kcsan_rmb {
() => {
while false {
}
};
}

macro_rules! kcsan_wmb {
() => {
while false {
}
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

macro_rules! kvmalloc_node {
($size:expr, $flags:expr, $node:expr) => {
alloc_hooks!(unsafe { __kvmalloc_node_noprof($size, 1u64, $flags, $node) })
};
}

macro_rules! kvzalloc_node {
($_size:expr, $_flags:expr, $_node:expr) => {
kvmalloc_node!($_size, ($_flags) | 256u32, $_node)
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
macro_rules! list_entry__18 {
($ptr:expr) => {
container_of__111!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__6 {
($ptr:expr) => {
container_of__19!($ptr)
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

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__21 {
($ptr:expr) => {
list_entry__18!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__18 {
($ptr:expr) => {
list_entry__6!(unsafe { (*$ptr).next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__28 {
($pos:expr, $member:ident) => {
list_entry__18!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__25 {
($pos:expr, $member:ident) => {
list_entry__6!(unsafe { (*$pos).$member.next })
};
}

/// list_prev_entry - get the prev element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_prev_entry {
($pos:expr, $member:ident) => {
list_entry__18!(unsafe { (*$pos).$member.prev })
};
}

macro_rules! lockdep_assert_held {
($l:expr) => {
__assume_ctx_lock!($l);
};
}

macro_rules! lockdep_init_map {
($name:expr, $key:expr) => {
loop {
$name;
$key;
if true {
break;
}
}
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

/// min_t - return minimum of two values, using the specified type
/// @type: data type to use
/// @x: first value
/// @y: second value
macro_rules! min_t__2 {
($type:ty, $x:expr, $y:expr) => {
__cmp_once!($type, $x, $y)
};
}

/// min_t - return minimum of two values, using the specified type
/// @type: data type to use
/// @x: first value
/// @y: second value
macro_rules! min_t {
($type:ty, $x:expr, $y:expr) => {
__cmp_once__3!($type, $x, $y)
};
}

macro_rules! mutex_lock_nested {
($lock:expr) => {
unsafe { mutex_lock($lock) };
};
}

macro_rules! nodes_weight {
($nodemask:expr) => {
__nodes_weight((unsafe { core::ptr::addr_of_mut!($nodemask) }) as *const __SlateRecord_nodemask_types_h_17, 64u32)
};
}

macro_rules! num_online_nodes {
() => {
num_node_state(1u32)
};
}

macro_rules! offsetof__84 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, dm_dev) as u64
};
}

macro_rules! offsetof__105 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, kobj_holder.kobj) as u64
};
}

macro_rules! offsetof__5 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, clone) as u64
};
}

macro_rules! offsetof__109 {
() => {
core::mem::offset_of!(table_device, list) as u64
};
}

macro_rules! offsetof__19 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, work) as u64
};
}

macro_rules! offsetof__24 {
() => {
core::mem::offset_of!(dm_dev_internal, list) as u64
};
}

macro_rules! offsetof__134 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, requeue_work) as u64
};
}

macro_rules! offsetof__20 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, tio) as u64
};
}

macro_rules! per_cpu_offset {
($x:expr) => {
unsafe { __per_cpu_offset[($x) as usize] }
};
}

macro_rules! per_cpu_ptr__7 {
($ptr:expr, $cpu:expr) => {
{
let mut __v: *mut u64;
__verify_pcpu_ptr__15!();
__v = (((($ptr) as u64) as *mut u64) as u64).wrapping_add(per_cpu_offset!($cpu)) as *mut u64;
__v
}
};
}

macro_rules! preempt_count_dec_and_test {
() => {
__preempt_count_dec_and_test()
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
typecheck!(u64);
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
typecheck!(u64);
unsafe { _raw_spin_unlock_irqrestore($lock, $flags) };
if true {
break;
}
}
};
}

/// rcu_dereference_protected() - fetch RCU pointer when updates prevented
/// @p: The pointer to read, prior to dereferencing
/// @c: The conditions under which the dereference will take place
///
/// Return the value of the specified RCU-protected pointer, but omit
/// the READ_ONCE().  This is useful in cases where update-side locks
/// prevent the value of the pointer from changing.  Please note that this
/// primitive does *not* prevent the compiler from repeating this reference
/// or combining it with other references, so it should not be used without
/// protection of appropriate locks.
///
/// This function is only for update-side use.  Using this function
/// when protected only by rcu_read_lock() will result in infrequent
/// but very ugly failures.
macro_rules! rcu_dereference_protected {
($p:expr, $c:expr) => {
__rcu_dereference_protected__6!($p, $c)
};
}

/// rcu_dereference_protected() - fetch RCU pointer when updates prevented
/// @p: The pointer to read, prior to dereferencing
/// @c: The conditions under which the dereference will take place
///
/// Return the value of the specified RCU-protected pointer, but omit
/// the READ_ONCE().  This is useful in cases where update-side locks
/// prevent the value of the pointer from changing.  Please note that this
/// primitive does *not* prevent the compiler from repeating this reference
/// or combining it with other references, so it should not be used without
/// protection of appropriate locks.
///
/// This function is only for update-side use.  Using this function
/// when protected only by rcu_read_lock() will result in infrequent
/// but very ugly failures.
macro_rules! rcu_dereference_protected__4 {
($p:expr, $c:expr) => {
__rcu_dereference_protected__5!($p, $c)
};
}

/// rcu_dereference_protected() - fetch RCU pointer when updates prevented
/// @p: The pointer to read, prior to dereferencing
/// @c: The conditions under which the dereference will take place
///
/// Return the value of the specified RCU-protected pointer, but omit
/// the READ_ONCE().  This is useful in cases where update-side locks
/// prevent the value of the pointer from changing.  Please note that this
/// primitive does *not* prevent the compiler from repeating this reference
/// or combining it with other references, so it should not be used without
/// protection of appropriate locks.
///
/// This function is only for update-side use.  Using this function
/// when protected only by rcu_read_lock() will result in infrequent
/// but very ugly failures.
macro_rules! rcu_dereference_protected__2 {
($p:expr, $c:expr) => {
__rcu_dereference_protected!($p, $c)
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

macro_rules! register_blkdev {
($major:expr, $name:expr) => {
unsafe { __register_blkdev($major, $name, None) }
};
}

macro_rules! sector_div {
($a:expr, $b:expr) => {
do_div!($a, $b)
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

macro_rules! smp_mb__before_atomic {
() => {
loop {
kcsan_mb!();
__smp_mb__before_atomic!();
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
__release!();
__acquire!();
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

macro_rules! static_branch_unlikely__7 {
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

macro_rules! static_branch_unlikely__8 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const14(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
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

macro_rules! static_branch_unlikely__10 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const12(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
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
__v2 = arch_static_branch__const13(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
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

macro_rules! static_branch_unlikely__12 {
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

macro_rules! static_branch_unlikely__6 {
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

macro_rules! static_branch_unlikely__5 {
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

macro_rules! static_branch_unlikely__13 {
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

macro_rules! static_branch_unlikely__14 {
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

macro_rules! test_bit__4 {
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

macro_rules! test_bit_acquire {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if (unsafe { __slate_intrinsic_a39e427d3b6c2df4($nr) }) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
generic_test_bit_acquire(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit_acquire(($nr) as u64, ($addr) as *const u64) as i32
};
__v
}
};
}

macro_rules! type_max {
() => {
__type_max!()
};
}

/// Check at compile time that something is of a particular type.
/// Always evaluates to 1 so you may use it easily in comparisons.
macro_rules! typecheck {
($type:ty) => {
let mut __v: i32;
let mut __dummy: $type = 0;
let mut __dummy2: $type = 0;
core::ptr::addr_of_mut!(__dummy) == core::ptr::addr_of_mut!(__dummy2);
__v = 1;
};
}

macro_rules! unlikely__3 {
($x:expr) => {
(($x) != 0) as i64
};
}

macro_rules! unlikely__2 {
($x:expr) => {
(($x) != 0) as i64
};
}

macro_rules! unlikely {
($x:expr) => {
!(!$x) as i64
};
}

macro_rules! unlikely__4 {
($x:expr) => {
(($x) != 0) as i64
};
}

macro_rules! wake_up {
($x:expr) => {
unsafe { __wake_up($x, 3u32, 1i32, core::ptr::null_mut::<()>()) };
};
}

unsafe extern "C" {
    static mut __per_cpu_offset: [u64; 64];
    static mut this_cpu_off: u64;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut __SCK__WARN_trap: static_call_key;
    static mut kmsg_fops: file_operations;
    static mut __SCK__might_resched: static_call_key;
    static mut node_states: [__SlateRecord_nodemask_types_h_17; 6];
    static mut __cpu_possible_mask: cpumask;
    static mut __cpu_online_mask: cpumask;
    static mut __preempt_count: u64;
    static mut __SCK__preempt_schedule: static_call_key;
    static mut cpu_number: i32;
    static mut __SCK__cond_resched: static_call_key;
    static mut jiffies: u64;
    static mut system_percpu_wq: *mut workqueue_struct;
    static mut system_highpri_wq: *mut workqueue_struct;
    static mut system_long_wq: *mut workqueue_struct;
    static mut system_dfl_wq: *mut workqueue_struct;
    static mut system_freezable_wq: *mut workqueue_struct;
    static mut system_power_efficient_wq: *mut workqueue_struct;
    static mut system_freezable_power_efficient_wq: *mut workqueue_struct;
    static mut system_dfl_long_wq: *mut workqueue_struct;
    static mut numa_node: i32;
    static mut sighand_cachep: *mut kmem_cache;
    static mut init_task: task_struct;
    static mut radix_tree_preloads: radix_tree_preload;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut param_ops_int: kernel_param_ops;
    static mut param_ops_uint: kernel_param_ops;
    static mut tracepoint_srcu: srcu_struct;
    static mut __SCK__tp_func_block_split: static_call_key;
    static mut __tracepoint_block_split: tracepoint;
    static mut __SCK__tp_func_block_bio_remap: static_call_key;
    static mut __tracepoint_block_bio_remap: tracepoint;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn __bad_size_call_parameter();
    fn __compiletime_assert_0();
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn ___ratelimit(arg0: *mut ratelimit_state, arg1: *const u8) -> i32;
    fn _printk(arg0: *const u8, ...) -> i32;
    fn __compiletime_assert_1();
    fn __compiletime_assert_2();
    fn ____wrong_branch_error() -> bool;
    fn __cmpxchg_wrong_size();
    fn __xadd_wrong_size();
    fn _find_next_bit(arg0: *const u64, arg1: u64, arg2: u64) -> u64;
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn __bitmap_weight(arg0: *const u64, arg1: u32) -> u32;
    fn sprintf(arg0: *mut u8, arg1: *const u8, ...) -> i32;
    fn snprintf(arg0: *mut u8, arg1: u64, arg2: *const u8, ...) -> i32;
    fn __cond_resched() -> i32;
    fn __SCT__might_resched() -> i32;
    fn __compiletime_assert_14();
    fn __compiletime_assert_15();
    fn __compiletime_assert_16();
    fn __compiletime_assert_19();
    fn __compiletime_assert_20();
    fn lockdep_is_held(arg0: *const ()) -> i32;
    fn _raw_spin_lock(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irqsave(arg0: *mut raw_spinlock) -> u64;
    fn _raw_spin_unlock(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irqrestore(arg0: *mut raw_spinlock, arg1: u64);
    fn schedule();
    fn io_schedule();
    fn __SCT__cond_resched() -> i32;
    fn pcpu_alloc_noprof(arg0: u64, arg1: u64, arg2: bool, arg3: u32) -> *mut ();
    fn free_percpu(arg0: *mut ());
    fn __compiletime_assert_103();
    fn ktime_get() -> i64;
    fn refcount_warn_saturate(arg0: *mut refcount_struct, arg1: u32);
    fn __init_waitqueue_head(arg0: *mut wait_queue_head, arg1: *const u8, arg2: *mut lock_class_key);
    fn __wake_up(arg0: *mut wait_queue_head, arg1: u32, arg2: i32, arg3: *mut ()) -> i32;
    fn init_wait_entry(arg0: *mut wait_queue_entry, arg1: i32);
    fn prepare_to_wait(arg0: *mut wait_queue_head, arg1: *mut wait_queue_entry, arg2: i32);
    fn prepare_to_wait_event(arg0: *mut wait_queue_head, arg1: *mut wait_queue_entry, arg2: i32) -> i64;
    fn finish_wait(arg0: *mut wait_queue_head, arg1: *mut wait_queue_entry);
    fn autoremove_wake_function(arg0: *mut wait_queue_entry, arg1: u32, arg2: i32, arg3: *mut ()) -> i32;
    fn __init_swait_queue_head(arg0: *mut swait_queue_head, arg1: *const u8, arg2: *mut lock_class_key);
    fn mutex_init_generic(arg0: *mut mutex);
    fn mutex_lock(arg0: *mut mutex);
    fn mutex_unlock(arg0: *mut mutex);
    fn __rcu_read_lock();
    fn __rcu_read_unlock();
    fn synchronize_rcu_expedited();
    fn rcu_is_watching() -> bool;
    fn alloc_workqueue_noprof(arg0: *const u8, arg1: u32, arg2: i32, ...) -> *mut workqueue_struct;
    fn destroy_workqueue(arg0: *mut workqueue_struct);
    fn queue_work_on(arg0: i32, arg1: *mut workqueue_struct, arg2: *mut work_struct) -> bool;
    fn __flush_workqueue(arg0: *mut workqueue_struct);
    fn __warn_flushing_systemwide_wq();
    fn init_srcu_struct_generic(arg0: *mut srcu_struct) -> i32;
    fn __srcu_read_unlock(arg0: *mut srcu_struct, arg1: i32);
    fn __srcu_read_lock(arg0: *mut srcu_struct) -> i32;
    fn __compiletime_assert_167();
    fn __srcu_check_read_flavor(arg0: *mut srcu_struct, arg1: i32);
    fn cleanup_srcu_struct(arg0: *mut srcu_struct);
    fn synchronize_srcu(arg0: *mut srcu_struct);
    fn capable(arg0: i32) -> bool;
    fn __compiletime_assert_339();
    fn __compiletime_assert_340();
    fn __compiletime_assert_341();
    fn __compiletime_assert_342();
    fn __compiletime_assert_343();
    fn __compiletime_assert_344();
    fn __compiletime_assert_345();
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_359();
    fn wake_up_bit(arg0: *mut u64, arg1: i32);
    fn out_of_line_wait_on_bit(arg0: *mut u64, arg1: i32, arg2: Option<unsafe extern "C" fn(*mut wait_bit_key, i32) -> i32>, arg3: u32) -> i32;
    fn bit_wait(arg0: *mut wait_bit_key, arg1: i32) -> i32;
    fn radix_tree_preload(arg0: u32) -> i32;
    fn down(arg0: *mut semaphore);
    fn up(arg0: *mut semaphore);
    fn __compiletime_assert_384();
    fn kfree(arg0: *const ());
    fn __compiletime_assert_390();
    fn __builtin_unreachable();
    fn __kmalloc_node_noprof(arg0: u64, arg1: u32, arg2: i32) -> *mut ();
    fn __kmalloc_cache_node_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: i32, arg3: u64) -> *mut ();
    fn __kmalloc_large_node_noprof(arg0: u64, arg1: u32, arg2: i32) -> *mut ();
    fn __kvmalloc_node_noprof(arg0: u64, arg1: u64, arg2: u32, arg3: i32) -> *mut ();
    fn kvfree(arg0: *const ());
    fn cache_get_priv_group(arg0: *mut cacheinfo) -> *const attribute_group;
    fn idr_preload(arg0: u32);
    fn idr_alloc(arg0: *mut idr, arg1: *mut (), arg2: i32, arg3: i32, arg4: u32) -> i32;
    fn idr_remove(arg0: *mut idr, arg1: u64) -> *mut ();
    fn idr_find(arg0: *const idr, arg1: u64) -> *mut ();
    fn idr_replace(arg0: *mut idr, arg1: *mut (), arg2: u64) -> *mut ();
    fn idr_destroy(arg0: *mut idr);
    fn kobject_uevent_env(arg0: *mut kobject, arg1: u32, arg2: *mut *mut u8) -> i32;
    fn try_module_get(arg0: *mut module) -> bool;
    fn module_put(arg0: *mut module);
    fn bio_trim(arg0: *mut bio, arg1: u64, arg2: u64);
    fn bioset_exit(arg0: *mut bio_set);
    fn bio_put(arg0: *mut bio);
    fn bio_alloc_clone(arg0: *mut block_device, arg1: *mut bio, arg2: u32, arg3: *mut bio_set) -> *mut bio;
    fn bio_endio(arg0: *mut bio);
    fn bio_init(arg0: *mut bio, arg1: *mut block_device, arg2: *mut bio_vec, arg3: u16, arg4: u32);
    fn bio_uninit(arg0: *mut bio);
    fn bio_associate_blkg(arg0: *mut bio);
    fn bio_clone_blkg_association(arg0: *mut bio, arg1: *mut bio);
    fn fput(arg0: *mut file);
    fn __fput_sync(arg0: *mut file);
    fn device_add_disk(arg0: *mut device, arg1: *mut gendisk, arg2: *mut *const attribute_group) -> i32;
    fn del_gendisk(arg0: *mut gendisk);
    fn put_disk(arg0: *mut gendisk);
    fn __blk_alloc_disk(arg0: *mut queue_limits, arg1: i32, arg2: *mut lock_class_key) -> *mut gendisk;
    fn __register_blkdev(arg0: u32, arg1: *const u8, arg2: Option<unsafe extern "C" fn(u32)>) -> i32;
    fn unregister_blkdev(arg0: u32, arg1: *const u8);
    fn set_capacity(arg0: *mut gendisk, arg1: u64);
    fn bd_link_disk_holder(arg0: *mut block_device, arg1: *mut gendisk) -> i32;
    fn bd_unlink_disk_holder(arg0: *mut block_device, arg1: *mut gendisk);
    fn submit_bio_noacct(arg0: *mut bio);
    fn bio_split_to_limits(arg0: *mut bio) -> *mut bio;
    fn bio_poll(arg0: *mut bio, arg1: *mut io_comp_batch, arg2: u32) -> i32;
    fn blk_mark_disk_dead(arg0: *mut gendisk);
    fn bdev_start_io_acct(arg0: *mut block_device, arg1: u32, arg2: u64) -> u64;
    fn bdev_end_io_acct(arg0: *mut block_device, arg1: u32, arg2: u32, arg3: u64);
    fn bdev_file_open_by_dev(arg0: u32, arg1: u32, arg2: *mut (), arg3: *const blk_holder_ops) -> *mut file;
    fn file_bdev(arg0: *mut file) -> *mut block_device;
    fn bdev_freeze(arg0: *mut block_device) -> i32;
    fn bdev_thaw(arg0: *mut block_device) -> i32;
    fn blk_mq_queue_inflight(arg0: *mut request_queue) -> bool;
    fn __traceiter_block_split(arg0: *mut (), arg1: *mut bio, arg2: u32) -> i32;
    fn __SCT__tp_func_block_split(arg0: *mut (), arg1: *mut bio, arg2: u32) -> i32;
    fn __compiletime_assert_756();
    fn __traceiter_block_bio_remap(arg0: *mut (), arg1: *mut bio, arg2: u32, arg3: u64) -> i32;
    fn __SCT__tp_func_block_bio_remap(arg0: *mut (), arg1: *mut bio, arg2: u32, arg3: u64) -> i32;
    fn __compiletime_assert_763();
    fn dm_table_destroy(arg0: *mut dm_table);
    fn dm_table_get_size(arg0: *mut dm_table) -> u64;
    fn dm_statistics_init() -> i32;
    fn dm_statistics_exit();
    fn dm_stats_init(arg0: *mut dm_stats) -> i32;
    fn dm_stats_cleanup(arg0: *mut dm_stats);
    fn dm_stats_account_io(arg0: *mut dm_stats, arg1: u64, arg2: u64, arg3: u32, arg4: bool, arg5: u64, arg6: *mut dm_stats_aux);
    fn dm_table_event_callback(arg0: *mut dm_table, arg1: Option<unsafe extern "C" fn(*mut ())>, arg2: *mut ());
    fn dm_table_find_target(arg0: *mut dm_table, arg1: u64) -> *mut dm_target;
    fn dm_table_has_no_data_devices(arg0: *mut dm_table) -> bool;
    fn dm_calculate_queue_limits(arg0: *mut dm_table, arg1: *mut queue_limits) -> i32;
    fn dm_table_set_restrictions(arg0: *mut dm_table, arg1: *mut request_queue, arg2: *mut queue_limits) -> i32;
    fn dm_table_get_devices(arg0: *mut dm_table) -> *mut list_head;
    fn dm_table_presuspend_targets(arg0: *mut dm_table);
    fn dm_table_presuspend_undo_targets(arg0: *mut dm_table);
    fn dm_table_postsuspend_targets(arg0: *mut dm_table);
    fn dm_table_resume_targets(arg0: *mut dm_table) -> i32;
    fn dm_table_get_type(arg0: *mut dm_table) -> u32;
    fn dm_table_get_immutable_target_type(arg0: *mut dm_table) -> *mut target_type;
    fn dm_table_get_immutable_target(arg0: *mut dm_table) -> *mut dm_target;
    fn dm_table_request_based(arg0: *mut dm_table) -> bool;
    fn dm_table_supports_size_change(arg0: *mut dm_table, arg1: u64, arg2: u64) -> bool;
    fn dm_zone_endio(arg0: *mut dm_io, arg1: *mut bio);
    fn dm_target_init() -> i32;
    fn dm_target_exit();
    fn dm_deferred_remove();
    fn dm_interface_init() -> i32;
    fn dm_interface_exit();
    fn dm_sysfs_init(arg0: *mut mapped_device) -> i32;
    fn dm_sysfs_exit(arg0: *mut mapped_device);
    fn linear_map(arg0: *mut dm_target, arg1: *mut bio) -> i32;
    fn dm_linear_init() -> i32;
    fn dm_linear_exit();
    fn stripe_map(arg0: *mut dm_target, arg1: *mut bio) -> i32;
    fn dm_stripe_init() -> i32;
    fn dm_stripe_exit();
    fn dm_request_based(arg0: *mut mapped_device) -> i32;
    fn dm_io_init() -> i32;
    fn dm_io_exit();
    fn dm_kcopyd_init() -> i32;
    fn dm_kcopyd_exit();
    fn dm_io_rewind(arg0: *mut dm_io, arg1: *mut bio_set);
    fn dm_mq_init_request_queue(arg0: *mut mapped_device, arg1: *mut dm_table) -> i32;
    fn dm_mq_cleanup_mapped_device(arg0: *mut mapped_device);
    fn dm_start_queue(arg0: *mut request_queue);
    fn dm_stop_queue(arg0: *mut request_queue);
    fn dax_get_private(arg0: *mut dax_device) -> *mut ();
    fn __bad_udelay();
    fn __udelay(arg0: u64);
    fn __const_udelay(arg0: u64);
    fn msleep(arg0: u32);
    fn usleep_range_state(arg0: u64, arg1: u64, arg2: u32);
    fn __compiletime_assert_810();
    fn __compiletime_assert_811();
    fn __compiletime_assert_812();
    fn __compiletime_assert_818();
    fn __compiletime_assert_820();
    fn __compiletime_assert_830();
    fn __compiletime_assert_836();
    fn __compiletime_assert_865();
    fn __compiletime_assert_866();
    fn __compiletime_assert_867();
    fn __compiletime_assert_868();
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
// kcsan_check_read_write - check regular read-write access for races
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
#[repr(C)]
#[derive(Clone, Copy)]
struct io_bitmap {
}


#[inline(always)]
fn ERR_PTR(error: i64) -> *mut () {
error as *mut ()
}

#[inline(always)]
fn PTR_ERR(ptr: *const ()) -> i64 {
ptr as i64
}

#[inline(always)]
fn IS_ERR(ptr: *const ()) -> bool {
IS_ERR_VALUE!(ptr as u64) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_static_call_types_h_65 {
    r#type: u64,
    mods: *mut static_call_mod,
    sites: *mut static_call_site,
}


#[inline(always)]
fn get_current() -> *mut task_struct {
if IS_ENABLED!() != 0 {
let mut __v0: *mut task_struct;
loop {
if true {
unsafe { __compiletime_assert_0() };
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

#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_site {
    addr: i32,
    key: i32,
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


#[inline(always)]
unsafe extern "C" fn __WARN_validate_printf(fmt: *const u8, __va_args: ...) {
}

// inhibit tail-call optimization
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
struct bug_entry {
    bug_addr_disp: i32,
    format_disp: i32,
    file_disp: i32,
    line: u16,
    flags: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct range {
    start: u64,
    end: u64,
}


#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
}


#[inline(always)]
fn generic_test_bit_acquire(nr: u64, addr: *const u64) -> bool {
let p: *mut u64 = (addr as *mut u64).wrapping_offset(BIT_WORD!(nr) as isize);
let mut __v0: u64;
let mut ___p1: u64 = 0;
let mut __v1: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(p as *const u64) };
___p1 = __v1;
loop {
if !__native_word!() {
unsafe { __compiletime_assert_2() };
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
1 & __v0 >> (nr & 63u64) != 0
}

// const_*() definitions provide good compile-time optimizations when
// the passed arguments can be resolved at compile time.
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
fn arch_test_and_clear_bit(mut nr: i64, mut addr: *mut u64) -> bool {
let mut __v0: bool;
let mut c: bool = false;
unsafe {
let __t0: u8;
core::arch::asm!("lock  btrq  {2:r}, ({0:r})\nsetc {1}", in(reg) addr, lateout(reg_byte) __t0, in(reg) nr as u64, options(nostack, att_syntax));
c = (__t0 as u8) != 0;
}
__v0 = c;
__v0
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

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_jump_label_h_95 {
    r#type: u64,
    entries: *mut jump_entry,
    next: *mut static_key_mod,
}


#[inline(always)]
fn arch_test_bit_acquire(nr: u64, addr: *const u64) -> bool {
(variable_test_bit(nr as i64, addr) as i32) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct static_key_mod {
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
fn __arch_hweight32(mut w: u32) -> u32 {
let mut res: u32 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("# ALT: oldinstr\n771:\n\tcall __sw_hweight32\n772:\n# ALT: padding\n.skip -(((775f-774f)-(772b-771b)) > 0) * ((775f-774f)-(772b-771b)),0x90\n773:\n.pushsection .altinstructions, \"aM\", @progbits, 14\n .long 771b - .\n .long 774f - .\n .4byte ( 4*32+23)\n .byte 773b-771b\n .byte 775f-774f\n.popsection\n.pushsection .altinstr_replacement, \"ax\"\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n# ALT: replacement\n774:\n\tpopcntl %edi, %eax\n775:\n.popsection\n", lateout("ax") __t0, in("di") w as u32, options(pure, nomem, att_syntax));
res = unsafe { core::mem::transmute::<u32, u32>(__t0 as u32) };
}
res
}

#[inline(always)]
fn __arch_hweight64(mut w: u64) -> u64 {
let mut res: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("# ALT: oldinstr\n771:\n\tcall __sw_hweight64\n772:\n# ALT: padding\n.skip -(((775f-774f)-(772b-771b)) > 0) * ((775f-774f)-(772b-771b)),0x90\n773:\n.pushsection .altinstructions, \"aM\", @progbits, 14\n .long 771b - .\n .long 774f - .\n .4byte ( 4*32+23)\n .byte 773b-771b\n .byte 775f-774f\n.popsection\n.pushsection .altinstr_replacement, \"ax\"\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n# ALT: replacement\n774:\n\tpopcntq %rdi, %rax\n775:\n.popsection\n", lateout("ax") __t0, in("di") w as u64, options(pure, nomem, att_syntax));
res = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
res
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
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("3: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 3b - ., 3; .popsection\n# {0}", const 3 as i32, options(nostack, att_syntax));
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
core::arch::asm!("4: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 4b - ., 4; .popsection\n# {0}", const 4 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v2 = (__ret_warn_on != 0) as i64;
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_find_h_67 {
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
#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
}


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

#[inline(always)]
fn test_and_clear_bit(nr: i64, addr: *mut u64) -> bool {
kcsan_mb!();
instrument_atomic_read_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_and_clear_bit(nr, addr)
}

#[inline(always)]
fn _test_bit(nr: u64, addr: *const u64) -> bool {
instrument_atomic_read(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_bit(nr, addr)
}

#[inline(always)]
fn _test_bit_acquire(nr: u64, addr: *const u64) -> bool {
instrument_atomic_read(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_bit_acquire(nr, addr)
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


// The following macros are non-atomic versions of their non-underscored
// counterparts.
// Check that the bitops prototypes are sane
#[inline(always)]
fn hweight_long(w: u64) -> u64 {
let __v0: u64 = if (core::mem::size_of::<u64>() as u64) == 4 {
hweight32!(w) as u64
} else {
hweight64!(w)
};
__v0
}

// assign_bit - Assign value to a bit in memory
// @nr: the bit to set
// @addr: the address to start counting from
// @value: the value to assign
#[inline(always)]
fn is_power_of_2(n: u64) -> bool {
n.wrapping_sub(1) < n ^ n.wrapping_sub(1)
}

// ilog2 - log base 2 of 32-bit or a 64-bit unsigned value
// @n: parameter
//
// constant-capable log of base 2 calculation
// - this can be used to initialise global variables from constant data, hence
// the massive ternary operator construction
//
// selects the appropriately-sized optimised version depending on sizeof(n)
// roundup_pow_of_two - round the given value up to nearest power of two
// @n: parameter
//
// round the given value up to the nearest power of two
// - the result is undefined when n == 0
// - this can be used to initialise global variables from constant data
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
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_64_types_h_21 {
    pte: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
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
struct mutex_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_types_h_295 {
    pgd: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_pi_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct robust_list_head {
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
    __slate_bits_0: __slate_bits::__SlateBits120U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_private_hash {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_block_split, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_block_bio_remap, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0} + {1} + 2 - . \n\t.popsection \n\t", sym zoned_enabled, const 0 as u8, label {
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
fn arch_static_branch__const4(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0} + {1} + 2 - . \n\t.popsection \n\t", sym stats_enabled, const 0 as u8, label {
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
fn arch_static_branch__const5(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0} + {1} + 2 - . \n\t.popsection \n\t", sym stats_enabled, const 0 as u8, label {
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
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pid_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sem_undo_list {
}


#[inline(always)]
fn arch_static_branch__const6(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0} + {1} + 2 - . \n\t.popsection \n\t", sym zoned_enabled, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0} + {1} + 2 - . \n\t.popsection \n\t", sym zoned_enabled, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0} + {1} + 2 - . \n\t.popsection \n\t", sym swap_bios_enabled, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0} + {1} + 2 - . \n\t.popsection \n\t", sym swap_bios_enabled, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0} + {1} + 2 - . \n\t.popsection \n\t", sym swap_bios_enabled, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0} + {1} + 2 - . \n\t.popsection \n\t", sym zoned_enabled, const 0 as u8, label {
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
fn arch_static_branch__const12(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0} + {1} + 2 - . \n\t.popsection \n\t", sym zoned_enabled, const 0 as u8, label {
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
fn arch_static_branch__const13(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0} + {1} + 2 - . \n\t.popsection \n\t", sym zoned_enabled, const 0 as u8, label {
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
fn arch_static_branch__const14(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0} + {1} + 2 - . \n\t.popsection \n\t", sym zoned_enabled, const 0 as u8, label {
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
struct hrtimer_clock_base {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct seccomp_filter {
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct orc_entry {
    sp_offset: i16,
    bp_offset: i16,
    __slate_bits_0: __slate_bits::__SlateBits127U0,
}


#[inline(always)]
fn arch_atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
// Note for KASAN: we deliberately don't use READ_ONCE_NOCHECK() here,
// it's non-inlined function that increases binary size and stack usage.
__READ_ONCE!((*v).counter)
}

#[inline(always)]
fn arch_atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
__WRITE_ONCE__4!((*v).counter, i);
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
fn arch_atomic_dec(mut v: *mut __SlateRecord_types_h_190) {
unsafe {
core::arch::asm!("lock decl ({0:r})", in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
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


#[repr(C)]
#[derive(Clone, Copy)]
struct ucounts {
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

#[inline(always)]
fn arch_atomic_add_return(i: i32, mut v: *mut __SlateRecord_types_h_190) -> i32 {
let mut __v0: i32;
let mut __ret: i32 = i;
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
unsafe {
let __t0: u32;
core::arch::asm!("lock xaddb {0:l}, ({1:r})", inlateout(reg) __ret as u32 => __t0, in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
}
}
2 => {
unsafe {
let __t1: u32;
core::arch::asm!("lock xaddw {0:x}, ({1:r})", inlateout(reg) __ret as u32 => __t1, in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t1 as u32) };
}
}
4 => {
unsafe {
let __t2: u32;
core::arch::asm!("lock xaddl {0:e}, ({1:r})", inlateout(reg) __ret as u32 => __t2, in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t2 as u32) };
}
}
8 => {
unsafe {
let __t3: u32;
core::arch::asm!("lock xaddq {0:r}, ({1:r})", inlateout(reg) __ret as u32 => __t3, in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t3 as u32) };
}
}
_ => {
unsafe { __xadd_wrong_size() };
}
}
__v0 = __ret;
i.wrapping_add(__v0)
}

#[inline(always)]
fn arch_atomic_fetch_add(mut __ret: i32, mut v: *mut __SlateRecord_types_h_190) -> i32 {
let mut __v0: i32;
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
unsafe {
let __t0: u32;
core::arch::asm!("lock xaddb {0:l}, ({1:r})", inlateout(reg) __ret as u32 => __t0, in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
}
}
2 => {
unsafe {
let __t1: u32;
core::arch::asm!("lock xaddw {0:x}, ({1:r})", inlateout(reg) __ret as u32 => __t1, in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t1 as u32) };
}
}
4 => {
unsafe {
let __t2: u32;
core::arch::asm!("lock xaddl {0:e}, ({1:r})", inlateout(reg) __ret as u32 => __t2, in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t2 as u32) };
}
}
8 => {
unsafe {
let __t3: u32;
core::arch::asm!("lock xaddq {0:r}, ({1:r})", inlateout(reg) __ret as u32 => __t3, in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t3 as u32) };
}
}
_ => {
unsafe { __xadd_wrong_size() };
}
}
__v0 = __ret;
__v0
}

#[inline(always)]
fn arch_atomic64_inc(mut v: *mut __SlateRecord_types_h_197) {
unsafe {
core::arch::asm!("lock incq ({0:r})\n# {1}", in(reg) core::ptr::addr_of_mut!((*v).counter), in(reg) core::ptr::addr_of!((*v).counter), options(nostack, att_syntax));
}
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
fn raw_atomic_add_return(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
arch_atomic_add_return(i, v)
}

#[inline(always)]
fn raw_atomic_fetch_add_relaxed(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
arch_atomic_fetch_add(i, v)
}

#[inline(always)]
fn raw_atomic_sub(i: i32, v: *mut __SlateRecord_types_h_190) {
arch_atomic_sub(i, v);
}

#[inline(always)]
fn raw_atomic_fetch_sub_release(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
arch_atomic_fetch_sub!(i, v)
}

#[inline(always)]
fn raw_atomic_inc(v: *mut __SlateRecord_types_h_190) {
arch_atomic_inc(v);
}

#[inline(always)]
fn raw_atomic_dec(v: *mut __SlateRecord_types_h_190) {
arch_atomic_dec(v);
}

#[inline(always)]
fn raw_atomic_dec_and_test(v: *mut __SlateRecord_types_h_190) -> bool {
arch_atomic_dec_and_test(v)
}

#[inline(always)]
fn raw_atomic64_inc(v: *mut __SlateRecord_types_h_197) {
arch_atomic64_inc(v);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct workqueue_struct {
}


#[inline(always)]
fn raw_atomic_long_inc(v: *mut __SlateRecord_types_h_197) {
raw_atomic64_inc(v);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
}


#[inline(always)]
fn atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
instrument_atomic_read(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_read(v)
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
fn atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
instrument_atomic_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_set(v, i);
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
fn atomic_add(i: i32, v: *mut __SlateRecord_types_h_190) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_add(i, v);
}

#[inline(always)]
fn atomic_add_return(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_add_return(i, v)
}

#[inline(always)]
fn atomic_fetch_add_relaxed(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_fetch_add_relaxed(i, v)
}

#[inline(always)]
fn atomic_sub(i: i32, v: *mut __SlateRecord_types_h_190) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_sub(i, v);
}

#[inline(always)]
fn atomic_fetch_sub_release(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
kcsan_release!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_fetch_sub_release(i, v)
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

#[inline(always)]
fn atomic_dec_and_test(v: *mut __SlateRecord_types_h_190) -> bool {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_dec_and_test(v)
}

#[inline(always)]
fn atomic_long_inc(v: *mut __SlateRecord_types_h_197) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_inc(v);
}

#[inline(always)]
fn find_next_bit(addr: *const u64, size: u64, offset: u64) -> u64 {
if small_const_nbits!(size) {
let mut val: u64 = 0;
if unlikely!(offset >= size) != 0 {
return size;
}
val = (unsafe { *addr }) & GENMASK!(size.wrapping_sub(1), offset);
return if val != 0 { __ffs!(val) } else { size };
}
unsafe { _find_next_bit(addr, size, offset) }
}

#[inline(always)]
fn bitmap_weight(src: *const u64, nbits: u32) -> u32 {
if small_const_nbits__2!(nbits) {
return hweight_long((unsafe { *src }) & BITMAP_LAST_WORD_MASK!(nbits)) as u32;
}
unsafe { __bitmap_weight(src, nbits) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cpumask {
    bits: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct audit_context {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_nodemask_types_h_17 {
    bits: [u64; 1],
}


#[inline(always)]
fn might_resched() {
let mut __v0: Option<unsafe extern "C" fn() -> i32>;
__v0 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn() -> i32>>(__SCT__might_resched as *const ()) };
unsafe { __v0.unwrap()() };
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


#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__might_resched_13: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__might_resched) }) as *mut ();

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


#[repr(C)]
#[derive(Clone, Copy)]
struct nsproxy {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct reclaim_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rq {
}


fn __might_resched(file: *const u8, line: i32, offsets: u32) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct seq_file {
}


// LIST_HEAD - definition of a &struct list_head with initialization values
// @name: name of the list_head
fn INIT_LIST_HEAD(list: *mut list_head) {
loop {
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_14() };
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
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_15() };
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


fn __list_del_entry_valid(entry: *mut list_head) -> bool {
true
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
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_16() };
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

fn __list_del(prev: *mut list_head, next: *mut list_head) {
unsafe {
(*next).prev = prev;
}
loop {
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_19() };
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

fn __list_del_entry(entry: *mut list_head) {
if !__list_del_entry_valid(entry) {
return;
}
__list_del(unsafe { (*entry).prev }, unsafe { (*entry).next });
}

fn list_del(entry: *mut list_head) {
__list_del_entry(entry);
unsafe {
(*entry).next = ((256i32 as *const u8).wrapping_offset(_AC__2!() as isize) as *mut ()) as *mut list_head;
}
unsafe {
(*entry).prev = ((290i32 as *const u8).wrapping_offset(_AC__2!() as isize) as *mut ()) as *mut list_head;
}
}

fn list_is_head(list: *const list_head, head: *const list_head) -> i32 {
(list == head) as i32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rq_flags {
}


fn list_empty(head: *const list_head) -> i32 {
let mut __v0: *mut list_head;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_20() };
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
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
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

fn list_splice_init(list: *mut list_head, head: *mut list_head) {
if list_empty(list as *const list_head) == 0 {
__list_splice(list as *const list_head, head, unsafe { (*head).next });
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
// list_for_each_entry_continue_reverse - iterate backwards from the given point
// @pos:	the type * to use as a loop cursor.
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
//
// Start to iterate over list of given type backwards, continuing after
// the current position.
// No static inline type checking - see Subtlety (1) above.
#[inline(always)]
fn __nodes_weight(srcp: *const __SlateRecord_nodemask_types_h_17, nbits: u32) -> i32 {
bitmap_weight(unsafe { (*srcp).bits.as_ptr() as *const u64 }, nbits) as i32
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
fn num_node_state(state: u32) -> i32 {
nodes_weight!(node_states[(state as u32) as usize])
}

#[inline(always)]
fn cpu_max_bits_warn(cpu: u32, bits: u32) {
}

#[inline(always)]
fn cpumask_check(cpu: u32) -> u32 {
cpu_max_bits_warn(cpu, 64u32);
cpu
}

// for_each_cpu - iterate over every cpu in a mask
// @cpu: the (optionally unsigned) integer iterator
// @mask: the cpumask pointer
//
// After the loop, cpu is >= nr_cpu_ids.
#[inline(always)]
fn cpumask_test_cpu(cpu: i32, cpumask: *const cpumask) -> bool {
test_bit__4!(cpumask_check(cpu as u32), unsafe { (*cpumask).bits.as_ptr() as *const u64 }) != 0
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


#[repr(C)]
#[derive(Clone, Copy)]
struct css_set {
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
    __slate_bits_0: __slate_bits::__SlateBits128U0,
    pkru: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __kernel_timespec {
    tv_sec: i64,
    tv_nsec: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct timespec64 {
    tv_sec: i64,
    tv_nsec: i64,
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

#[repr(C)]
#[derive(Clone, Copy)]
struct user_namespace {
}


// We use the MSB for PREEMPT_NEED_RESCHED mostly because it is available.
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
struct optimistic_spin_queue {
    tail: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct spinlock {
    __slate_anon_0: __SlateRecord_spinlock_types_h_18,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwlock {
    raw_lock: qrwlock,
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
struct mem_cgroup {
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
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_mm_data {
    phash: futex_mm_phash,
    unlock: futex_unlock_cs_ranges,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct llist_head {
    first: *mut llist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct llist_node {
    next: *mut llist_node,
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
struct uprobe_xol_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rb_root {
    rb_node: *mut rb_node,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct rb_root_cached {
    rb_root: rb_root,
    rb_leftmost: *mut rb_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct timerqueue_head {
    rb_root: rb_root_cached,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct timerqueue_linked_node {
    node: rb_node_linked,
    expires: i64,
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
struct xol_area {
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
struct refcount_struct {
    refs: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ldt_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rlimit {
    rlim_cur: u64,
    rlim_max: u64,
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
struct sigpending {
    list: list_head,
    signal: __SlateRecord_signal_h_25,
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
struct sigaction {
    sa_handler: Option<unsafe extern "C" fn(i32)>,
    sa_flags: u64,
    sa_restorer: Option<unsafe extern "C" fn()>,
    sa_mask: __SlateRecord_signal_h_25,
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
    __slate_bits_0: __slate_bits::__SlateBits224U0,
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


#[inline(always)]
fn spin_lock_irq(lock: *mut spinlock) {
raw_spin_lock_irq!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
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
struct device_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain {
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
union __SlateRecord_xarray_h_1175 {
    private_list: list_head,
    callback_head: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_xarray_h_1180 {
    tags: [[u64; 1]; 3],
    marks: [[u64; 1]; 3],
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
struct kmem_cache {
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
struct mnt_idmap {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq_ids {
    __slate_anon_0: __SlateRecord_rseq_types_h_74,
    node_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array_ptr {
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
struct key_type {
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
struct seqcount_spinlock {
    seqcount: seqcount,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_cred_h_147 {
    non_rcu: i32,
    rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqlock {
    seqcount: seqcount_spinlock,
    lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pidfs_attr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uidgid_types_h_9 {
    val: u32,
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
struct tty_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct taskstats {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_audit_buf {
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
    __slate_bits_0: __slate_bits::__SlateBits287U0,
    dl_timer: hrtimer,
    inactive_timer: hrtimer,
    rq: *mut rq,
    server_pick_task: Option<unsafe extern "C" fn(*mut sched_dl_entity, *mut rq_flags) -> *mut task_struct>,
    pi_se: *mut sched_dl_entity,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
}


#[repr(C)]
#[derive(Clone, Copy)]
union rcu_special {
    b: __SlateRecord_sched_h_791,
    s: u32,
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
    __slate_bits_0: __slate_bits::__SlateBits72U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits72U1,
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
    __slate_bits_2: __slate_bits::__SlateBits72U2,
    mce_kill_me: callback_head,
    mce_count: i32,
    kretprobe_instances: llist_head,
    rethooks: llist_head,
    l1d_flush_kill: callback_head,
    unwind_info: unwind_task_info,
    thread: thread_struct,
}


// Per process flags
// I'm a virtual CPU
// I am an IDLE thread
// Allocating memory to free memory. See memalloc_noreclaim_save()
// Kernel thread cloned from userspace thread
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

#[inline(always)]
fn _cond_resched() -> i32 {
let mut __v0: Option<unsafe extern "C" fn() -> i32>;
__v0 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn() -> i32>>(__SCT__cond_resched as *const ()) };
unsafe { __v0.unwrap()() }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__cond_resched_88: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__cond_resched) }) as *mut ();

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
struct codetag {
    flags: u32,
    lineno: u32,
    modname: *const u8,
    function: *const u8,
    filename: *const u8,
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
struct old_timespec32 {
    tv_sec: i32,
    tv_nsec: i32,
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
fn jiffies_to_usecs(j: u64) -> u32 {
// Hz usually doesn't go much further MSEC_PER_SEC.
// jiffies_to_usecs() and usecs_to_jiffies() depend on that.
loop {
if false {
unsafe { __compiletime_assert_103() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
1000u64.wrapping_mul(j) as u32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct xattr_handler {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fserror_event {
}


fn jiffies_to_nsecs(j: u64) -> u64 {
(jiffies_to_usecs(j) as u64).wrapping_mul(1000)
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


fn refcount_set(r: *mut refcount_struct, n: i32) {
atomic_set(unsafe { core::ptr::addr_of_mut!((*r).refs) }, n);
}

fn __refcount_add(i: i32, r: *mut refcount_struct, oldp: *mut i32) {
let old: i32 = atomic_fetch_add_relaxed(i, unsafe { core::ptr::addr_of_mut!((*r).refs) });
if !oldp.is_null() {
unsafe {
*oldp = old;
}
}
if unlikely!(old == 0) != 0 {
unsafe { refcount_warn_saturate(r, 2u32) };
} else {
if unlikely!(old < 0 || old.wrapping_add(i) < 0) != 0 {
unsafe { refcount_warn_saturate(r, 1u32) };
}
}
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


fn __refcount_inc(r: *mut refcount_struct, oldp: *mut i32) {
__refcount_add(1i32, r, oldp);
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


fn refcount_inc(r: *mut refcount_struct) {
__refcount_inc(r, core::ptr::null_mut::<i32>());
}

fn __refcount_sub_and_test(i: i32, r: *mut refcount_struct, oldp: *mut i32) -> bool {
let old: i32 = atomic_fetch_sub_release(i, unsafe { core::ptr::addr_of_mut!((*r).refs) });
if !oldp.is_null() {
unsafe {
*oldp = old;
}
}
if old > 0 && old == i {
loop {
kcsan_rmb!();
loop {
kcsan_rmb!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
if true {
break;
}
}
return true;
}
if unlikely!(old <= 0 || old.wrapping_sub(i) < 0) != 0 {
unsafe { refcount_warn_saturate(r, 3u32) };
}
false
}

fn __refcount_dec_and_test(r: *mut refcount_struct, oldp: *mut i32) -> bool {
__refcount_sub_and_test(1i32, r, oldp)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct semaphore_waiter {
}


fn refcount_dec_and_test(r: *mut refcount_struct) -> bool {
__refcount_dec_and_test(r, core::ptr::null_mut::<i32>())
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


fn waitqueue_active(wq_head: *mut wait_queue_head) -> i32 {
(list_empty((unsafe { core::ptr::addr_of_mut!((*wq_head).head) }) as *const list_head) == 0) as i32
}

fn wq_has_sleeper(wq_head: *mut wait_queue_head) -> bool {
// We need to be sure we are in sync with the
// add_wait_queue modifications to the wait queue.
//
// This memory barrier should be paired with one on the
// waiting side.
loop {
kcsan_mb!();
unsafe {
core::arch::asm!("lock addl $0,-4(%rsp)", options(nostack, att_syntax));
}
if true {
break;
}
}
waitqueue_active(wq_head) != 0
}

// explicit shadow
// wait_event_interruptible - sleep until a condition gets true
// @wq_head: the waitqueue to wait on
// @condition: a C expression for the event to wait for
//
// The process is put to sleep (TASK_INTERRUPTIBLE) until the
// @condition evaluates to true or a signal is received.
// The @condition is checked each time the waitqueue @wq_head is woken up.
//
// wake_up() has to be called after changing any variable that could
// change the result of the wait condition.
//
// The function will return -ERESTARTSYS if it was interrupted by a
// signal and 0 if @condition evaluated to true.
#[repr(C)]
#[derive(Clone, Copy)]
struct swait_queue_head {
    lock: raw_spinlock,
    task_list: list_head,
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


fn mutex_destroy(lock: *mut mutex) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
}


// mutex_init - initialize the mutex
// @mutex: the mutex to be initialized
//
// Initialize the mutex to unlocked state.
//
// It is not allowed to initialize an already locked mutex.
fn __mutex_init(lock: *mut mutex, name: *const u8, key: *mut lock_class_key) {
unsafe { mutex_init_generic(lock) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct kref {
    refcount: refcount_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rcu_gp_seq {
    norm: u64,
    exp: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_info_struct {
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
// rcu_dereference() - fetch RCU-protected pointer for dereferencing
// @p: The pointer to read, prior to dereferencing
//
// This is a simple wrapper around rcu_dereference_check().
#[inline(always)]
fn rcu_read_lock() {
unsafe { __rcu_read_lock() };
__acquire_shared!();
rcu_lock_acquire!();
RCU_LOCKDEP_WARN!(!unsafe { rcu_is_watching() });
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


fn rcu_read_unlock() {
RCU_LOCKDEP_WARN!(!unsafe { rcu_is_watching() });
rcu_lock_release!(); // Keep acq info for rls diags.
__release_shared!();
unsafe { __rcu_read_unlock() };
}

// RCU_INIT_POINTER() - initialize an RCU protected pointer
// @p: The pointer to be initialized.
// @v: The value to initialized the pointer to.
//
// Initialize an RCU-protected pointer in special cases where readers
// do not need ordering constraints on the CPU or the compiler.  These
// special cases are:
//
// 1.	This use of RCU_INIT_POINTER() is NULLing out the pointer *or*
// 2.	The caller has taken whatever steps are required to prevent
// 	RCU readers from concurrently accessing this pointer *or*
// 3.	The referenced data structure has already been exposed to
// 	readers either at compile time or via rcu_assign_pointer() *and*
//
// 	a.	You have not made *any* reader-visible changes to
// 		this structure since then *or*
// 	b.	It is OK for readers accessing this structure from its
// 		new location to see the old state of the structure.  (For
// 		example, the changes were to statistical counters or to
// 		other state where exact synchronization is not required.)
//
// Failure to follow these rules governing use of RCU_INIT_POINTER() will
// result in impossible-to-diagnose memory corruption.  As in the structures
// will look OK in crash dumps, but any concurrent RCU readers might
// see pre-initialized values of the referenced data structure.  So
// please be very careful how you use RCU_INIT_POINTER()!!!
//
// If you are creating an RCU-protected linked structure that is accessed
// by a single external-to-structure RCU-protected pointer, then you may
// use RCU_INIT_POINTER() to initialize the internal RCU-protected
// pointers, but you must use rcu_assign_pointer() to initialize the
// external-to-structure pointer *after* you have completely initialized
// the reader-accessible portions of the linked structure.
//
// Note that unlike rcu_assign_pointer(), RCU_INIT_POINTER() provides no
// ordering guarantees for either the CPU or the compiler.
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
struct posix_acl {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct delayed_work {
    work: work_struct,
    timer: timer_list,
    wq: *mut workqueue_struct,
    cpu: i32,
}


fn __init_work(work: *mut work_struct, onstack: i32) {
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


// alloc_ordered_workqueue - allocate an ordered workqueue
// @fmt: printf format for the name of the workqueue
// @flags: WQ_* flags (only WQ_FREEZABLE and WQ_MEM_RECLAIM are meaningful)
// @args: args for @fmt
//
// Allocate an ordered workqueue.  An ordered workqueue executes at
// most one work item at any given time in the queued order.  They are
// implemented as unbound workqueues with @max_active of one.
//
// RETURNS:
// Pointer to the allocated workqueue on success, %NULL on failure.
fn queue_work(wq: *mut workqueue_struct, work: *mut work_struct) -> bool {
unsafe { queue_work_on(64i32, wq, work) }
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


fn __init_srcu_struct(ssp: *mut srcu_struct, name: *const u8, key: *mut lock_class_key) -> i32 {
unsafe { init_srcu_struct_generic(ssp) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_cmd {
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
struct srcu_struct {
    srcu_ctrp: *mut srcu_ctr,
    sda: *mut srcu_data,
    srcu_reader_flavor: u8,
    dep_map: lockdep_map,
    srcu_sup: *mut srcu_usage,
}


fn __srcu_read_lock_fast(ssp: *mut srcu_struct) -> *mut srcu_ctr {
let mut scp: *mut srcu_ctr = unsafe { core::mem::zeroed() };
let mut __v0: *mut srcu_ctr;
loop {
if !(__native_word__8!() || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
__verify_pcpu_ptr__3!();
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
__verify_pcpu_ptr__2!();
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
__acquire_shared!();
scp
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


fn __srcu_read_unlock_fast(ssp: *mut srcu_struct, mut scp: *mut srcu_ctr) {
__release_shared!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
// Avoid leaking the critical section.
if IS_ENABLED!() == 0 {
loop {
__verify_pcpu_ptr__3!();
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
let mut __v12: *mut __SlateRecord_types_h_197;
__verify_pcpu_ptr__2!();
let mut __v14: *mut __SlateRecord_types_h_197;
let mut tcp_ptr__: u64 = 0;
let mut __v15: u64;
let mut pfo_val__: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t0, sym this_cpu_off, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
__v15 = pfo_val__;
tcp_ptr__ = __v15;
let __v17: u64 = tcp_ptr__.wrapping_add((unsafe { core::ptr::addr_of_mut!((*scp).srcu_unlocks) }) as u64);
tcp_ptr__ = __v17;
__v14 = tcp_ptr__ as *mut __SlateRecord_types_h_197;
__v12 = __v14;
atomic_long_inc(__v12);
}
// Z, and implicit RCU reader.
// Z, and implicit RCU reader.
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ns_common {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_open_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_iattrs {
}


fn srcu_check_read_flavor(ssp: *mut srcu_struct, read_flavor: i32) {
if IS_ENABLED!() != 0 {
unsafe { __srcu_check_read_flavor(ssp, read_flavor) };
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
}


// Values for SRCU Tree srcu_data ->srcu_reader_flavor, but also used by rcutorture.
// srcu_read_lock().
// srcu_read_lock_nmisafe().
// 					0x4		// SRCU-lite is no longer with us.
// srcu_read_lock_fast(), also NMI-safe.
// srcu_read_lock_fast_updown().
fn srcu_read_lock_held(ssp: *const srcu_struct) -> i32 {
1
}

fn __srcu_read_lock_must_hold(ssp: *const srcu_struct) {
}

// srcu_dereference_check - fetch SRCU-protected pointer for later dereferencing
// @p: the pointer to fetch and protect for later dereferencing
// @ssp: pointer to the srcu_struct, which is used to check that we
// 	really are in an SRCU read-side critical section.
// @c: condition to check for update-side use
//
// If PROVE_RCU is enabled, invoking this outside of an RCU read-side
// critical section will result in an RCU-lockdep splat, unless @c evaluates
// to 1.  The @c argument will normally be a logical expression containing
// lockdep_is_held() calls.
// srcu_dereference - fetch SRCU-protected pointer for later dereferencing
// @p: the pointer to fetch and protect for later dereferencing
// @ssp: pointer to the srcu_struct, which is used to check that we
// 	really are in an SRCU read-side critical section.
//
// Makes rcu_dereference_check() do the dirty work.  If PROVE_RCU
// is enabled, invoking this outside of an RCU read-side critical
// section will result in an RCU-lockdep splat.
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

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
}


fn srcu_read_unlock(ssp: *mut srcu_struct, idx: i32) {
let mut __v0: i64;
let __ret_warn_on: i32 = (idx & !1 != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
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

#[repr(C)]
#[derive(Clone, Copy)]
struct sock {
}


fn srcu_read_unlock_fast_notrace(ssp: *mut srcu_struct, scp: *mut srcu_ctr) {
srcu_check_read_flavor(ssp, 4i32);
__srcu_read_unlock_fast(ssp, scp);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcu_h_645 {
    lock: *mut srcu_struct,
    scp: *mut srcu_ctr,
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
fn class_srcu_fast_notrace_destructor(_T: *mut __SlateRecord_srcu_h_645) {
srcu_read_unlock_fast_notrace(unsafe { (*_T).lock }, unsafe { (*_T).scp });
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
struct percpu_counter {
    lock: raw_spinlock,
    count: i64,
    list: list_head,
    counters: *mut i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wake_irq {
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
struct dev_pm_qos {
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
    __slate_bits_0: __slate_bits::__SlateBits424U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct folio_queue {
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


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_uio_h_78 {
    nr_segs: u64,
    folioq_slot: u8,
    xarray_start: i64,
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


// Opaque type representing current mm_struct flag state. Must be accessed via
// mm_flags_xxx() helper functions.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_1171 {
    __mm_flags: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
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
struct local_lock {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct local_trylock {
    acquired: u8,
}


fn local_lock_release(l: *mut local_lock) {
}

fn numa_node_id() -> i32 {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
__verify_pcpu_ptr__5!();
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym numa_node, options(pure, readonly, nostack, att_syntax));
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
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym numa_node, options(pure, readonly, nostack, att_syntax));
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
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym numa_node, options(pure, readonly, nostack, att_syntax));
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
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym numa_node, options(pure, readonly, nostack, att_syntax));
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
__v0
}

fn memalloc_flags_save(flags: u32) -> u32 {
let oldflags: u32 = !unsafe { (*get_current()).flags } & flags;
let __v0: *mut task_struct = get_current();
unsafe {
(*__v0).flags = (*__v0).flags | flags;
}
oldflags
}

#[repr(C)]
#[derive(Clone, Copy)]
struct module_param_attrs {
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
struct xa_node {
    shift: u8,
    offset: u8,
    count: u8,
    nr_values: u8,
    parent: *mut xa_node,
    array: *mut xarray,
    __slate_anon_6: __SlateRecord_xarray_h_1175,
    slots: [*mut (); 64],
    __slate_anon_8: __SlateRecord_xarray_h_1180,
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
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct list_lru {
    node: *mut list_lru_node,
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
fn sigismember(set: *mut __SlateRecord_signal_h_25, _sig: i32) -> i32 {
let sig: u64 = _sig.wrapping_sub(1) as u64;
if 1i32 == 1 {
return (1 & (unsafe { *((*set).sig.as_mut_ptr() as *mut u64).wrapping_offset(0 as isize) }) >> sig) as i32;
} else {
return (1 & (unsafe { *((*set).sig.as_mut_ptr() as *mut u64).wrapping_offset((sig / 64) as isize) }) >> sig % 64) as i32;
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct exception_table_entry {
    insn: i32,
    fixup: i32,
    data: i32,
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
struct bio_alloc_cache {
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


fn kernel_cred() -> *const cred {
// shut up sparse
let mut __v0: *const cred;
let mut __v1: *const cred;
loop {
if !(__native_word__9!() || ((const { core::mem::size_of::<*const cred>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_339() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!(init_task.cred) as *const *const cred) };
let __UNIQUE_ID_rcu_338: *const cred = __v1;
__v0 = __UNIQUE_ID_rcu_338;
__v0
}

fn override_creds(override_cred: *const cred) -> *const cred {
let mut __v0: *const cred;
let __tmp: *const cred = rcu_dereference_protected__2!(unsafe { (*get_current()).cred }, 1i32);
let _r_a_p__v: u64 = override_cred as u64;
if false && _r_a_p__v == 0 {
loop {
loop {
if !(__native_word__9!() || ((const { core::mem::size_of::<*const cred>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_340() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__8!((*get_current()).cred, _r_a_p__v as *const cred);
if true {
break;
}
}
} else {
loop {
kcsan_release!();
loop {
loop {
if !__native_word__9!() {
unsafe { __compiletime_assert_341() };
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
if !(__native_word__9!() || ((const { core::mem::size_of::<*const cred>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_342() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__8!(*unsafe { core::ptr::addr_of_mut!((*get_current()).cred) }, RCU_INITIALIZER!(_r_a_p__v as *const cred));
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
__v0 = __tmp;
__v0
}

fn revert_creds(revert_cred: *const cred) -> *const cred {
let mut __v0: *const cred;
let __tmp: *const cred = rcu_dereference_protected__2!(unsafe { (*get_current()).cred }, 1i32);
let _r_a_p__v: u64 = revert_cred as u64;
if false && _r_a_p__v == 0 {
loop {
loop {
if !(__native_word__9!() || ((const { core::mem::size_of::<*const cred>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_343() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__8!((*get_current()).cred, _r_a_p__v as *const cred);
if true {
break;
}
}
} else {
loop {
kcsan_release!();
loop {
loop {
if !__native_word__9!() {
unsafe { __compiletime_assert_344() };
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
if !(__native_word__9!() || ((const { core::mem::size_of::<*const cred>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_345() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__8!(*unsafe { core::ptr::addr_of_mut!((*get_current()).cred) }, RCU_INITIALIZER!(_r_a_p__v as *const cred));
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
__v0 = __tmp;
__v0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct elevator_queue {
}


#[inline(always)]
fn class_override_creds_destructor(p: *mut *const cred) {
let _T: *const cred = unsafe { *p };
revert_creds(_T);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct blk_flush_queue {
}


#[inline(always)]
fn class_override_creds_constructor(override_cred: *const cred) -> *const cred {
let t: *const cred = override_creds(override_cred);
t
}

// current_cred - Access the current task's subjective credentials
//
// Access the subjective credentials of the current task.  RCU-safe,
// since nobody else can modify it.
#[repr(C)]
#[derive(Clone, Copy)]
struct rq_qos {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rhash_head {
    next: *mut rhash_head,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct timer_rand_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cdrom_device_info {
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

#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_ctx {
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
    __slate_bits_0: __slate_bits::__SlateBits290U0,
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
unlikely__2!(test_tsk_thread_flag(p, 1i32)) as i32
}

fn signal_pending(p: *mut task_struct) -> i32 {
// TIF_NOTIFY_SIGNAL isn't really a signal, but it requires the same
// behavior in terms of ensuring that we break out of wait loops
// so that notify signal callbacks can be processed.
if unlikely__2!(test_tsk_thread_flag(p, 2i32)) != 0 {
return 1;
}
task_sigpending(p)
}

fn __fatal_signal_pending(p: *mut task_struct) -> i32 {
unlikely__2!(sigismember(unsafe { core::ptr::addr_of_mut!((*p).pending.signal) }, 9i32)) as i32
}

fn fatal_signal_pending(p: *mut task_struct) -> i32 {
let __v0: bool = task_sigpending(p) != 0 && __fatal_signal_pending(p) != 0;
__v0 as i32
}

fn signal_pending_state(state: u32, p: *mut task_struct) -> i32 {
if state & 257u32 == 0 {
return 0;
}
if signal_pending(p) == 0 {
return 0;
}
let __v0: bool = state & 1 != 0 || __fatal_signal_pending(p) != 0;
__v0 as i32
}

fn task_rlimit(task: *const task_struct, limit: u32) -> u64 {
let mut __v0: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_359() };
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
struct dax_device {
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
struct dm_stats_last_position {
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
struct __SlateRecord_dm_c_271 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_dm_c_295 {
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
struct __SlateRecord_dm_c_3838 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_dm_c_3838_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_dm_c_3838_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_dm_c_3838_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_dm_c_3838_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_dm_c_3838_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_dm_c_3841 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_dm_c_3841_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_dm_c_3841_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_dm_c_3841_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_dm_c_3841_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_dm_c_3841_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_dm_c_3844 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_dm_c_3844_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_dm_c_3844_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_dm_c_3844_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_dm_c_3844_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_dm_c_3844_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_dm_c_3847 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_dm_c_3847_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_dm_c_3847_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_dm_c_3847_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_dm_c_3847_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_dm_c_3847_6 {
}


#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

#[repr(transparent)]
struct __SlateConst<T>(T);

unsafe impl<T> Sync for __SlateConst<T> {
}

mod __slate_bits {
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits287U0 {
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
    pub struct __SlateBits545U0 {
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
    pub struct __SlateBits582U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits683U0 {
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
    pub struct __SlateBits680U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits683U1 {
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
    pub struct __SlateBits683U2 {
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
    pub struct __SlateBits683U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits656U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits595U0 {
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
    pub struct __SlateBits127U0 {
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
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits424U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits72U0 {
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
    pub struct __SlateBits72U1 {
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
    pub struct __SlateBits290U0 {
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
    pub struct __SlateBits72U2 {
        #[bits(1)]
        pub mce_ripv: u64,
        #[bits(1)]
        pub mce_whole_page: u64,
        #[bits(62)]
        pub __mce_reserved: u64,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits120U0 {
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
    pub struct __SlateBits128U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits224U0 {
        #[bits(1)]
        pub faultable: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits841U0 {
        #[bits(1)]
        pub init_tio_pdu: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits840U0 {
        #[bits(1)]
        pub integrity_supported: bool,
        #[bits(1)]
        pub singleton: bool,
        #[bits(1)]
        pub flush_bypasses_map: bool,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits839U0 {
        #[bits(1)]
        pub flush_supported: bool,
        #[bits(1)]
        pub discards_supported: bool,
        #[bits(1)]
        pub zone_reset_all_supported: bool,
        #[bits(1)]
        pub max_discard_granularity: bool,
        #[bits(1)]
        pub limit_swap_bios: bool,
        #[bits(1)]
        pub emulate_zone_append: bool,
        #[bits(1)]
        pub accounts_remapped_io: bool,
        #[bits(1)]
        pub needs_bio_set_dev: bool,
        #[bits(1)]
        pub flush_bypasses_map: bool,
        #[bits(1)]
        pub mempool_needs_integrity: bool,
        #[bits(6, access = na)]
        pub __slate_pad_10: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits893U0 {
        #[bits(1)]
        pub is_abnormal_io: bool,
        #[bits(1)]
        pub submit_as_polled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
}

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
struct wait_bit_key {
    flags: *mut u64,
    bit_nr: i32,
    timeout: u64,
}


fn wait_on_bit(word: *mut u64, bit: i32, mode: u32) -> i32 {
might_sleep!();
if test_bit_acquire!(bit, word) == 0 {
return 0;
}
unsafe { out_of_line_wait_on_bit(word, bit, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut wait_bit_key, i32) -> i32>>(bit_wait as *const ()) }, mode) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct path {
    mnt: *mut vfsmount,
    dentry: *mut dentry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct radix_tree_preload {
    lock: local_lock,
    nr: u32,
    nodes: *mut xa_node,
}


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
#[repr(C)]
#[derive(Clone, Copy)]
struct semaphore {
    lock: raw_spinlock,
    count: u32,
    first_waiter: *mut semaphore_waiter,
}


fn sema_init(sem: *mut semaphore, val: i32) {
let mut __slate_compound_1: raw_spinlock = unsafe { core::mem::zeroed() };
let mut __slate_compound_0: semaphore = unsafe { core::mem::zeroed() };
unsafe {
*sem = *{
__slate_compound_0 = semaphore { lock: *{
__slate_compound_1 = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_qspinlock_types_h_44 = core::mem::zeroed();
__t0.val = __SlateRecord_types_h_190 { counter: 0 };
__t0
} } };
core::ptr::addr_of_mut!(__slate_compound_1)
}, count: val as u32, first_waiter: core::ptr::null_mut::<semaphore_waiter>() };
core::ptr::addr_of_mut!(__slate_compound_0)
};
}
lockdep_init_map!(c"semaphore->lock".as_ptr() as *mut u8, unsafe { core::ptr::addr_of_mut!(__key__51) });
}

static mut __key__51: lock_class_key = unsafe { core::mem::zeroed() };

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
if !(__native_word__3!() || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_384() };
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
    __slate_bits_0: __slate_bits::__SlateBits582U0,
    rcu: callback_head,
    r#ref: *mut percpu_ref,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_ref {
    percpu_count_ptr: u64,
    data: *mut percpu_ref_data,
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
unsafe { __compiletime_assert_390() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
} else {
loop {
unsafe {
core::arch::asm!("391: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 391b - ., 3; .popsection\n# {0}", const 391 as i32, options(nostack, att_syntax));
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


#[repr(C)]
#[derive(Clone, Copy)]
struct cacheinfo {
    id: u32,
    r#type: u32,
    level: u32,
    coherency_line_size: u32,
    number_of_sets: u32,
    ways_of_associativity: u32,
    physical_line_partition: u32,
    size: u32,
    shared_cpu_map: cpumask,
    attributes: u32,
    fw_token: *mut (),
    disable_sysfs: bool,
    r#priv: *mut (),
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
// Mask used for checking in page_pool_page_is_pp() below. page->pp_magic is
// OR'ed with PP_SIGNATURE after the allocation in order to preserve bit 0 for
// the head page of compound page and bit 1 for pfmemalloc page, as well as the
// bits used for the DMA index. page_is_pfmemalloc() is checked in
// __page_pool_put_page() to avoid recycling the pfmemalloc page.
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
// IDR_INIT() - Initialise an IDR.
// @name: Name of IDR.
//
// A freshly-initialised IDR contains no IDs.
// DEFINE_IDR() - Define a statically-allocated IDR.
// @name: Name of IDR.
//
// An IDR defined using this macro is ready for use with no additional
// initialisation required.  It contains no IDs.
fn idr_preload_end() {
loop {
__release!();
loop {
let mut __v1: *mut local_lock;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<local_lock>() as *const ();
if true {
break;
}
}
let mut __v3: *mut local_lock;
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
let __v6: u64 = tcp_ptr__.wrapping_add((unsafe { core::ptr::addr_of_mut!(radix_tree_preloads.lock) }) as u64);
tcp_ptr__ = __v6;
__v3 = tcp_ptr__ as *mut local_lock;
__v1 = __v3;
let __l: *mut local_lock = __v1;
local_lock_release(__l);
if true {
break;
}
}
loop {
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if unlikely!(preempt_count_dec_and_test!()) != 0 {
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
if true {
break;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_455: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_elem_dir {
    subdirs: u64,
    children: rb_root,
    root: *mut kernfs_root,
    rev: u64,
}


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
    __slate_bits_0: __slate_bits::__SlateBits656U0,
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


// Permissions on a sysfs file: you didn't miss the 0 prefix did you?
// USER_READABLE >= GROUP_READABLE >= OTHER_READABLE
// USER_WRITABLE >= GROUP_WRITABLE
// OTHER_WRITABLE?  Generally considered a bad idea.
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
    __slate_bits_0: __slate_bits::__SlateBits595U0,
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
    __slate_bits_0: __slate_bits::__SlateBits683U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits683U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits683U2,
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
    __slate_bits_3: __slate_bits::__SlateBits683U3,
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


// One for each parameter, describing how to use it.  Some files do
// multiple of these per line, so can't just use MODULE_INFO.
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


// module_param - typesafe helper for a module/cmdline parameter
// @name: the variable to alter, and exposed parameter name.
// @type: the type of the parameter
// @perm: visibility in sysfs.
//
// @name becomes the module parameter, or (prefixed by KBUILD_MODNAME and a
// ".") the kernel commandline parameter.  Note that - is changed to _, so
// the user can use "foo-bar=1" even for variable "foo_bar".
//
// @perm is 0 if the variable is not to appear in sysfs, or 0444
// for world-readable, 0644 for root-writable, etc.  Note that if it
// is writable, you may need to use kernel_param_lock() around
// accesses (esp. charp, which can be kfreed when it changes).
//
// The @type is simply pasted to refer to a param_ops_##type and a
// param_check_##type: for convenience many standard types are provided but
// you can create your own by defining those variables.
//
// Standard types are:
// 	byte, hexint, short, ushort, int, uint, long, ulong
// 	charp: a character pointer
// 	bool: a bool, values 0/1, y/n, Y/N.
// 	invbool: the above, only sense-reversed (N = true).
// module_param_named - typesafe helper for a renamed module/cmdline parameter
// @name: a valid C identifier which is the parameter name.
// @value: the actual lvalue to alter.
// @type: the type of the parameter
// @perm: visibility in sysfs.
//
// Usually it's a good idea to have variable names and user-exposed names the
// same, but that's harder if the variable must be non-static or is inside a
// structure.  This allows exposure under a different name.
// module_param_cb - general callback for a module/cmdline parameter
// @name: a valid C identifier which is the parameter name.
// @ops: the set & get operations for this parameter.
// @arg: args for @ops
// @perm: visibility in sysfs.
//
// The ops can have NULL set or get functions.
// This is the fundamental function for registering boot/module parameters.
// All the helper functions
//
// The macros to do compile-time type checking stolen from Jakub
// Jelinek, who IIRC came up with this idea for the 2.4 module init code.
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


core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_dm_per_bio_data: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_dm_bio_from_per_bio_data: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_dm_bio_get_target_bio_nr: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_dm_get_reserved_bio_based_ios: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_dm_start_time_ns_from_clone: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_dm_set_target_max_io_len: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_dm_accept_partial_bio: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_dm_submit_bio_remap: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {7} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_dm_get_md: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {8} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_dm_hold: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {9} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_dm_device_name: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {10} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_dm_put: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {11} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_dm_internal_suspend_noflush: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {12} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_dm_internal_resume: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {13} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_dm_internal_suspend_fast: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {14} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_dm_internal_resume_fast: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {15} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_dm_disk: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {16} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_dm_suspended: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {17} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_dm_post_suspending: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {18} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_dm_noflush_suspending: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {19} ; .previous\n.section\t\".initcall6.init\", \"a\"\t\t\n__initcall__kmod_dm_mod__889_3835_dm_init6:\t\t\t\n.long\t{20} - .\t\n.previous\t\t\t\t\t\n", sym dm_per_bio_data, sym dm_bio_from_per_bio_data, sym dm_bio_get_target_bio_nr, sym dm_get_reserved_bio_based_ios, sym dm_start_time_ns_from_clone, sym dm_set_target_max_io_len, sym dm_accept_partial_bio, sym dm_submit_bio_remap, sym dm_get_md, sym dm_hold, sym dm_device_name, sym dm_put, sym dm_internal_suspend_noflush, sym dm_internal_resume, sym dm_internal_suspend_fast, sym dm_internal_resume_fast, sym dm_disk, sym dm_suspended, sym dm_post_suspending, sym dm_noflush_suspending, sym dm_init, options(att_syntax));

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


// module_init() - driver initialization entry point
// @x: function to be run at kernel boot time or module insertion
//
// module_init() will either be called during do_initcalls() (if
// builtin) or at module insertion time (if a module).  There can only
// be one per module.
// module_exit() - driver exit entry point
// @x: function to be run when driver is removed
//
// module_exit() will wrap the driver clean-up code
// with cleanup_module() when used with rmmod when
// the driver is a module.  If the driver is statically
// compiled into the kernel, module_exit() has no effect.
// There can only be one per module.
// The following license idents are currently accepted as indicating free
// software modules
//
// 	"GPL"				[GNU Public License v2]
// 	"GPL v2"			[GNU Public License v2]
// 	"GPL and additional rights"	[GNU Public License v2 rights and more]
// 	"Dual BSD/GPL"			[GNU Public License v2
// 					 or BSD license choice]
// 	"Dual MIT/GPL"			[GNU Public License v2
// 					 or MIT license choice]
// 	"Dual MPL/GPL"			[GNU Public License v2
// 					 or Mozilla license choice]
//
// The following other idents are available
//
// 	"Proprietary"			[Non free products]
//
// Both "GPL v2" and "GPL" (the latter also in dual licensed strings) are
// merely stating that the module is licensed under the GPL v2, but are not
// telling whether "GPL v2 only" or "GPL v2 or later". The reason why there
// are two variants is a historic and failed attempt to convey more
// information in the MODULE_LICENSE string. For module loading the
// "only/or later" distinction is completely irrelevant and does neither
// replace the proper license identifiers in the corresponding source file
// nor amends them in any way. The sole purpose is to make the
// 'Proprietary' flagging work and to refuse to bind symbols which are
// exported with EXPORT_SYMBOL_GPL when a non free module is loaded.
//
// In the same way "BSD" is not a clear license information. It merely
// states, that the module is licensed under one of the compatible BSD
// license variants. The detailed and correct license information is again
// to be found in the corresponding source files.
//
// There are dual licensed components, but when running with Linux it is the
// GPL that is relevant so this is a non issue. Similarly LGPL linked with GPL
// is a GPL combined work.
//
// This exists for several reasons
// 1.	So modinfo can show license info for users wanting to vet their setup
// 	is free
// 2.	So the community can ignore bug reports including proprietary modules
// 3.	So vendors can do likewise based on their own policies
// Author(s), use "Name <email>" or just "Name", for multiple
// authors use multiple MODULE_AUTHOR() statements/lines.
// What your module does.
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
    __slate_bits_0: __slate_bits::__SlateBits680U0,
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


// hack for device mapper, don't use elsewhere:
// BLK_STS_AGAIN should only be returned if RQF_NOWAIT is set
// and the bio would block (cf bio_wouldblock_error())
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

fn op_is_zone_mgmt(op: u32) -> bool {
match (op as u32) & 255u32 {
11 | 13 | 15 | 17 | 19 => {
return true;
}
_ => {
return false;
}
}
unsafe { core::mem::zeroed() }
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

fn bio_set_dev(bio: *mut bio, bdev: *mut block_device) {
bio_clear_flag(bio, 10u32);
if (unsafe { (*bio).bi_bdev }) != bdev {
bio_clear_flag(bio, 5u32);
}
unsafe {
(*bio).bi_bdev = bdev;
}
unsafe { bio_associate_blkg(bio) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct bio_list {
    head: *mut bio,
    tail: *mut bio,
}


fn bio_list_add(bl: *mut bio_list, bio: *mut bio) {
unsafe {
(*bio).bi_next = core::ptr::null_mut::<bio>();
}
if !unsafe { (*bl).tail }.is_null() {
unsafe {
(*(*bl).tail).bi_next = bio;
}
} else {
unsafe {
(*bl).head = bio;
}
}
unsafe {
(*bl).tail = bio;
}
}

fn bio_list_add_head(bl: *mut bio_list, bio: *mut bio) {
unsafe {
(*bio).bi_next = (*bl).head;
}
unsafe {
(*bl).head = bio;
}
if unsafe { (*bl).tail }.is_null() {
unsafe {
(*bl).tail = bio;
}
}
}

fn bio_list_pop(bl: *mut bio_list) -> *mut bio {
let bio: *mut bio = unsafe { (*bl).head };
if !bio.is_null() {
unsafe {
(*bl).head = (*(*bl).head).bi_next;
}
if unsafe { (*bl).head }.is_null() {
unsafe {
(*bl).tail = core::ptr::null_mut::<bio>();
}
}
unsafe {
(*bio).bi_next = core::ptr::null_mut::<bio>();
}
}
bio
}

fn bio_inc_remaining(bio: *mut bio) {
bio_set_flag(bio, 3u32);
smp_mb__before_atomic!();
atomic_inc(unsafe { core::ptr::addr_of_mut!((*bio).__bi_remaining) });
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


fn bio_clear_polled(__v0: *mut bio) {
let __v2: u32 = (unsafe { (*__v0).bi_opf }) & 4290772991u32;
unsafe {
(*__v0).bi_opf = __v2;
}
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


fn disk_devt(disk: *mut gendisk) -> u32 {
MKDEV!(unsafe { (*disk).major }, unsafe { (*disk).first_minor }) as u32
}

// supports a volatile write cache
// supports passing on the FUA bit
// don't modify data until writeback is done
// always completes in submit context
// supports REQ_NOWAIT
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

fn blk_queue_is_zoned(q: *mut request_queue) -> bool {
IS_ENABLED!() != 0 && (unsafe { (*q).limits.features }) & 1024u32 != 0
}

fn add_disk(disk: *mut gendisk) -> i32 {
unsafe { device_add_disk(core::ptr::null_mut::<device>(), disk, core::ptr::null_mut::<*const attribute_group>()) }
}

fn bdev_nr_sectors(bdev: *mut block_device) -> u64 {
unsafe { (*bdev).bd_nr_sectors }
}

fn get_capacity(disk: *mut gendisk) -> u64 {
bdev_nr_sectors(unsafe { (*disk).part0 })
}

// blk_alloc_disk - allocate a gendisk structure
// @lim: queue limits to be used for this disk.
// @node_id: numa node to allocate on
//
// Allocate and pre-initialize a gendisk structure for use with BIO based
// drivers.
//
// Returns an ERR_PTR on error, else the allocated disk.
//
// Context: can sleep
// only poll the hardware once, don't continue until a completion was found
fn bdev_get_queue(bdev: *mut block_device) -> *mut request_queue {
return unsafe { (*bdev).bd_queue }; // this is never NULL
}

fn blk_boundary_sectors_left(mut offset: u64, boundary_sectors: u32) -> u32 {
if unlikely!(!is_power_of_2(boundary_sectors as u64)) != 0 {
return boundary_sectors.wrapping_sub(sector_div!(offset, boundary_sectors));
}
(boundary_sectors as u64).wrapping_sub(offset & (boundary_sectors.wrapping_sub(1) as u64)) as u32
}

fn blk_queue_disable_discard(q: *mut request_queue) {
unsafe {
(*q).limits.max_discard_sectors = 0;
}
}

fn blk_queue_disable_write_zeroes(q: *mut request_queue) {
unsafe {
(*q).limits.max_write_zeroes_sectors = 0;
}
unsafe {
(*q).limits.max_wzeroes_unmap_sectors = 0;
}
}

fn bdev_limits(bdev: *mut block_device) -> *mut queue_limits {
unsafe { core::ptr::addr_of_mut!((*bdev_get_queue(bdev)).limits) }
}

fn queue_max_sectors(q: *const request_queue) -> u32 {
unsafe { (*q).limits.max_sectors }
}

fn bdev_max_discard_sectors(bdev: *mut block_device) -> u32 {
unsafe { (*bdev_limits(bdev)).max_discard_sectors }
}

fn bdev_write_zeroes_sectors(bdev: *mut block_device) -> u32 {
unsafe { (*bdev_limits(bdev)).max_write_zeroes_sectors }
}

fn bdev_is_zoned(bdev: *mut block_device) -> bool {
blk_queue_is_zoned(bdev_get_queue(bdev))
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


fn req_op(req: &request) -> u32 {
((unsafe { (*(req as *const request)).cmd_flags }) & 255u32) as u32
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
fn __do_trace_block_split(bio: *mut bio, new_sector: u32) {
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
let mut __UNIQUE_ID_guard_753: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_753) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_754: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_754) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_756() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_block_split)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_755: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_755;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut bio, u32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut bio, u32) -> i32>>(__SCT__tp_func_block_split as *const ()) };
unsafe { __v12.unwrap()(__data, bio, new_sector) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_block_split: [u8; 12] = [98, 108, 111, 99, 107, 95, 115, 112, 108, 105, 116, 0];

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_block_split_757: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_block_split) }) as *mut ();

fn trace_block_split(bio: *mut bio, new_sector: u32) {
if static_branch_unlikely!(unsafe { core::ptr::addr_of_mut!(__tracepoint_block_split.key) }) != 0 {
__do_trace_block_split(bio, new_sector);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_758: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_block_bio_remap(bio: *mut bio, dev: u32, from: u64) {
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
let mut __UNIQUE_ID_guard_760: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_760) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_761: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_761) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_763() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_block_bio_remap)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_762: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_762;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut bio, u32, u64) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut bio, u32, u64) -> i32>>(__SCT__tp_func_block_bio_remap as *const ()) };
unsafe { __v12.unwrap()(__data, bio, dev, from) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_block_bio_remap: __SlateAlign16<[u8; 16]> = __SlateAlign16([98u8, 108u8, 111u8, 99u8, 107u8, 95u8, 98u8, 105u8, 111u8, 95u8, 114u8, 101u8, 109u8, 97u8, 112u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_block_bio_remap_764: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_block_bio_remap) }) as *mut ();

fn trace_block_bio_remap(bio: *mut bio, dev: u32, from: u64) {
if static_branch_unlikely__2!(unsafe { core::ptr::addr_of_mut!(__tracepoint_block_bio_remap.key) }) != 0 {
__do_trace_block_bio_remap(bio, dev, from);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_765: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[repr(C)]
#[derive(Clone, Copy)]
union map_info {
    ptr: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dm_dev {
    bdev: *mut block_device,
    bdev_file: *mut file,
    dax_dev: *mut dax_device,
    mode: u32,
    name: [u8; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct target_type {
    features: u64,
    name: *const u8,
    module: *mut module,
    version: [u32; 3],
    ctr: Option<unsafe extern "C" fn(*mut dm_target, u32, *mut *mut u8) -> i32>,
    dtr: Option<unsafe extern "C" fn(*mut dm_target)>,
    map: Option<unsafe extern "C" fn(*mut dm_target, *mut bio) -> i32>,
    clone_and_map_rq: Option<unsafe extern "C" fn(*mut dm_target, *mut request, *mut map_info, *mut *mut request) -> i32>,
    release_clone_rq: Option<unsafe extern "C" fn(*mut request, *mut map_info)>,
    end_io: Option<unsafe extern "C" fn(*mut dm_target, *mut bio, *mut u8) -> i32>,
    rq_end_io: Option<unsafe extern "C" fn(*mut dm_target, *mut request, u8, *mut map_info) -> i32>,
    presuspend: Option<unsafe extern "C" fn(*mut dm_target)>,
    presuspend_undo: Option<unsafe extern "C" fn(*mut dm_target)>,
    postsuspend: Option<unsafe extern "C" fn(*mut dm_target)>,
    preresume: Option<unsafe extern "C" fn(*mut dm_target) -> i32>,
    resume: Option<unsafe extern "C" fn(*mut dm_target)>,
    status: Option<unsafe extern "C" fn(*mut dm_target, u32, u32, *mut u8, u32)>,
    message: Option<unsafe extern "C" fn(*mut dm_target, u32, *mut *mut u8, *mut u8, u32) -> i32>,
    prepare_ioctl: Option<unsafe extern "C" fn(*mut dm_target, *mut *mut block_device, u32, u64, *mut bool) -> i32>,
    report_zones: Option<unsafe extern "C" fn(*mut dm_target) -> i32>,
    busy: Option<unsafe extern "C" fn(*mut dm_target) -> i32>,
    iterate_devices: Option<unsafe extern "C" fn(*mut dm_target, Option<unsafe extern "C" fn(*mut dm_target, *mut dm_dev, u64, u64, *mut ()) -> i32>, *mut ()) -> i32>,
    io_hints: Option<unsafe extern "C" fn(*mut dm_target, *mut queue_limits)>,
    direct_access: Option<unsafe extern "C" fn(*mut dm_target, u64, i64, u32, *mut *mut (), *mut u64) -> i64>,
    dax_zero_page_range: Option<unsafe extern "C" fn(*mut dm_target, u64, u64) -> i32>,
    dax_recovery_write: Option<unsafe extern "C" fn(*mut dm_target, u64, *mut (), u64, *mut iov_iter) -> u64>,
    list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dm_target {
    table: *mut dm_table,
    r#type: *mut target_type,
    begin: u64,
    len: u64,
    max_io_len: u32,
    num_flush_bios: u32,
    num_discard_bios: u32,
    num_secure_erase_bios: u32,
    num_write_zeroes_bios: u32,
    per_io_data_size: u32,
    private: *mut (),
    error: *mut u8,
    __slate_bits_0: __slate_bits::__SlateBits839U0,
}


// DM_ANY_MINOR chooses the next available minor number.
#[unsafe(no_mangle)]
extern "C" fn dm_get_live_table(md: *mut mapped_device, srcu_idx: *mut i32) -> *mut dm_table {
unsafe {
*srcu_idx = srcu_read_lock(core::ptr::addr_of_mut!((*md).io_barrier));
}
let mut __v0: *mut ();
__srcu_read_lock_must_hold((unsafe { core::ptr::addr_of_mut!((*md).io_barrier) }) as *const srcu_struct);
__acquire_shared_ctx_lock!();
let mut __v2: *mut ();
let mut __v3: *mut ();
loop {
if !(__native_word__7!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_818() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*md).map) as *const *mut ()) };
let __UNIQUE_ID_rcu_817: *mut () = __v3;
loop {
let __v6: bool = if false {
let __v7: bool = false || srcu_read_lock_held((unsafe { core::ptr::addr_of_mut!((*md).io_barrier) }) as *const srcu_struct) != 0;
!__v7
} else {
false
};
if !__v6 {
break;
}
}
__v2 = __UNIQUE_ID_rcu_817;
let __v: *mut () = __v2;
__release_shared_ctx_lock!();
__v0 = __v;
__v0 as *mut dm_table
}

// Macros.
// Definitions of return values from target end_io function.
// Definitions of return values from target map function.
fn to_bytes(n: u64) -> u64 {
n << 9i32
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
    __slate_bits_0: __slate_bits::__SlateBits545U0,
    fbatch: folio_batch,
    index: u64,
    saved_err: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hd_geometry {
    heads: u8,
    sectors: u8,
    cylinders: u16,
    start: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dm_stats {
    mutex: mutex,
    list: list_head,
    last: *mut dm_stats_last_position,
    precise_timestamps: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dm_stats_aux {
    merged: bool,
    duration_ns: u64,
}


fn dm_stats_used(st: *mut dm_stats) -> bool {
list_empty((unsafe { core::ptr::addr_of_mut!((*st).list) }) as *const list_head) == 0
}

fn dm_stats_record_start(stats: *mut dm_stats, aux: *mut dm_stats_aux) {
if unlikely!(unsafe { (*stats).precise_timestamps }) != 0 {
unsafe {
(*aux).duration_ns = ktime_to_ns(ktime_get()) as u64;
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct dm_dev_internal {
    list: list_head,
    count: refcount_struct,
    dm_dev: *mut dm_dev,
}


fn dm_is_zone_write(md: *mut mapped_device, bio: *mut bio) -> bool {
false
}

fn dm_ima_init(md: *mut mapped_device) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct dm_kobject_holder {
    kobj: kobject,
    completion: completion,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dm_md_mempools {
    bs: bio_set,
    io_bs: bio_set,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mapped_device {
    suspend_lock: mutex,
    table_devices_lock: mutex,
    table_devices: list_head,
    map: *mut (),
    flags: u64,
    type_lock: mutex,
    r#type: u32,
    numa_node_id: i32,
    queue: *mut request_queue,
    holders: __SlateRecord_types_h_190,
    open_count: __SlateRecord_types_h_190,
    immutable_target: *mut dm_target,
    immutable_target_type: *mut target_type,
    name: [u8; 16],
    disk: *mut gendisk,
    dax_dev: *mut dax_device,
    wait: wait_queue_head,
    pending_io: *mut u64,
    geometry: hd_geometry,
    wq: *mut workqueue_struct,
    work: work_struct,
    deferred_lock: spinlock,
    deferred: bio_list,
    requeue_work: work_struct,
    requeue_list: *mut dm_io,
    interface_ptr: *mut (),
    eventq: wait_queue_head,
    event_nr: __SlateRecord_types_h_190,
    uevent_seq: __SlateRecord_types_h_190,
    uevent_list: list_head,
    uevent_lock: spinlock,
    __slate_bits_0: __slate_bits::__SlateBits841U0,
    tag_set: *mut blk_mq_tag_set,
    stats: dm_stats,
    internal_suspend_count: u32,
    swap_bios: i32,
    swap_bios_semaphore: semaphore,
    swap_bios_lock: mutex,
    mempools: *mut dm_md_mempools,
    kobj_holder: dm_kobject_holder,
    io_barrier: srcu_struct,
}


// Bits for the flags field of struct mapped_device.
fn dm_get_size(md: *mut mapped_device) -> u64 {
get_capacity(unsafe { (*md).disk })
}

#[repr(C)]
#[derive(Clone, Copy)]
struct dm_table {
    md: *mut mapped_device,
    r#type: u32,
    depth: u32,
    counts: [u32; 16],
    index: [*mut u64; 16],
    num_targets: u32,
    num_allocated: u32,
    highs: *mut u64,
    targets: *mut dm_target,
    immutable_target_type: *mut target_type,
    __slate_bits_0: __slate_bits::__SlateBits840U0,
    mode: u32,
    devices: list_head,
    event_fn: Option<unsafe extern "C" fn(*mut ())>,
    event_context: *mut (),
    mempools: *mut dm_md_mempools,
}


fn dm_table_get_target(t: *mut dm_table, index: u32) -> *mut dm_target {
loop {
if unlikely!(index >= unsafe { (*t).num_targets }) != 0 {
loop {
unsafe {
core::arch::asm!("805: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 805b - ., 3; .popsection\n# {0}", const 805 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_dm_table_get_target_0, sym __slate_asm_string_dm_table_get_target_1, const 233 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
unsafe { (*t).targets }.wrapping_offset(index as isize)
}

// One of these is allocated per clone bio.
#[repr(C)]
#[derive(Clone, Copy)]
struct dm_target_io {
    magic: u16,
    flags: u16,
    target_bio_nr: u32,
    io: *mut dm_io,
    ti: *mut dm_target,
    len_ptr: *mut u32,
    old_sector: u64,
    clone: bio,
}


fn dm_tio_flagged(tio: *mut dm_target_io, bit: u32) -> bool {
(((unsafe { (*tio).flags }) as i32) as u32) & 1u32 << bit != 0
}

fn dm_tio_set_flag(__v0: *mut dm_target_io, bit: u32) {
unsafe {
(*__v0).flags = ((((*__v0).flags as i32) as u32) | 1u32 << bit) as u16;
}
}

fn dm_tio_is_normal(tio: *mut dm_target_io) -> bool {
dm_tio_flagged(tio, 0u32) && !dm_tio_flagged(tio, 1u32)
}

// One of these is allocated per original bio.
// It contains the first clone used for that original.
#[repr(C)]
#[derive(Clone, Copy)]
struct dm_io {
    magic: u16,
    flags: u16,
    lock: spinlock,
    start_time: u64,
    data: *mut (),
    next: *mut dm_io,
    stats_aux: dm_stats_aux,
    status: u8,
    requeue_flush_with_data: bool,
    io_count: __SlateRecord_types_h_190,
    md: *mut mapped_device,
    orig_bio: *mut bio,
    sector_offset: u32,
    sectors: u32,
    tio: dm_target_io,
}


fn dm_io_flagged(io: *mut dm_io, bit: u32) -> bool {
(((unsafe { (*io).flags }) as i32) as u32) & 1u32 << bit != 0
}

fn dm_io_set_flag(__v0: *mut dm_io, bit: u32) {
unsafe {
(*__v0).flags = ((((*__v0).flags as i32) as u32) | 1u32 << bit) as u16;
}
}

#[unsafe(no_mangle)]
extern "C" fn __dm_get_module_param(module_param: *mut u32, def: u32, max: u32) -> u32 {
let mut param: u32 = 0;
let mut __v0: u32;
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_812() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(module_param as *const u32) };
param = __v0;
let mut modified_param: u32 = 0;
if param == 0 {
modified_param = def;
} else {
if param > max {
modified_param = max;
}
}
if modified_param != 0 {
let mut __v2: u32;
let __ai_ptr: *mut u32 = module_param;
kcsan_mb!();
instrument_atomic_read_write(__ai_ptr as *const (), core::mem::size_of::<u32>() as u64);
let mut __v4: u32;
let mut __ret: u32 = 0;
let mut __old: u32 = param;
let mut __new: u32 = modified_param;
match (const { core::mem::size_of::<u32>() }) as u64 {
1 => {
let mut __ptr: *mut u8 = __ai_ptr as *mut u8;
unsafe {
let __t0: u32;
core::arch::asm!("lock cmpxchgb {1:e}, ({0:r})", in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t0, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, u32>(__t0 as u32) };
}
}
2 => {
let mut __ptr: *mut u16 = __ai_ptr as *mut u16;
unsafe {
let __t1: u32;
core::arch::asm!("lock cmpxchgw {1:e}, ({0:r})", in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t1, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, u32>(__t1 as u32) };
}
}
4 => {
let mut __ptr: *mut u32 = __ai_ptr as *mut u32;
unsafe {
let __t2: u32;
core::arch::asm!("lock cmpxchgl {1:e}, ({0:r})", in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t2, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
}
8 => {
let mut __ptr: *mut u64 = __ai_ptr as *mut u64;
unsafe {
let __t3: u32;
core::arch::asm!("lock cmpxchgq {1:e}, ({0:r})", in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t3, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, u32>(__t3 as u32) };
}
}
_ => {
unsafe { __cmpxchg_wrong_size() };
}
}
__v4 = __ret;
__v2 = __v4;
param = modified_param;
}
param
}

fn dm_uevent_init() -> i32 {
0
}

fn dm_uevent_exit() {
}

fn dm_send_uevents(events: &list_head, kobj: *mut kobject) {
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

fn bio_integrity_trim(bio: *mut bio) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct dax_operations {
    direct_access: Option<unsafe extern "C" fn(*mut dax_device, u64, i64, u32, *mut *mut (), *mut u64) -> i64>,
    zero_page_range: Option<unsafe extern "C" fn(*mut dax_device, u64, u64) -> i32>,
    recovery_write: Option<unsafe extern "C" fn(*mut dax_device, u64, *mut (), u64, *mut iov_iter) -> u64>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dax_holder_operations {
    notify_failure: Option<unsafe extern "C" fn(*mut dax_device, u64, u64, i32) -> i32>,
}


fn alloc_dax(private: *mut (), ops: *const dax_operations) -> *mut dax_device {
ERR_PTR(95i32.wrapping_neg() as i64) as *mut dax_device
}

fn put_dax(dax_dev: *mut dax_device) {
}

fn kill_dax(dax_dev: *mut dax_device) {
}

fn set_dax_nocache(dax_dev: *mut dax_device) {
}

fn set_dax_nomc(dax_dev: *mut dax_device) {
}

fn dax_add_host(dax_dev: *mut dax_device, disk: *mut gendisk) -> i32 {
0
}

fn dax_remove_host(disk: *mut gendisk) {
}

fn fs_dax_get_by_bdev(bdev: *mut block_device, start_off: &u64, holder: *mut (), ops: *const dax_holder_operations) -> *mut dax_device {
core::ptr::null_mut::<dax_device>()
}

// The microseconds/nanosecond delay multiplicators are used to convert a
// constant microseconds/nanoseconds value to a value which can be used by the
// architectures specific implementation to transform it into loops.
// The maximum constant udelay/ndelay value picked out of thin air to prevent
// too long constant udelays/ndelays.
#[inline(always)]
fn udelay(usec: u64) {
if unsafe { __slate_intrinsic_f217c4720f04f6db(usec) } {
if usec >= 20000 {
unsafe { __bad_udelay() };
} else {
unsafe { __const_udelay(usec.wrapping_mul(4295u64)) };
}
} else {
unsafe { __udelay(usec) };
}
}

fn usleep_range(min: u64, max: u64) {
unsafe { usleep_range_state(min, max, 2u32) };
}

static max_slack_shift: u32 = 2;

fn fsleep(usecs: u64) {
if usecs <= 10 {
udelay(usecs);
} else {
if usecs < (((1000000i64 << unsafe { max_slack_shift }) / 1000) as u64) {
usleep_range(usecs, usecs.wrapping_add(usecs >> unsafe { max_slack_shift }));
} else {
unsafe { msleep((usecs.wrapping_add(1000).wrapping_sub(1) / 1000) as u32) };
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pr_keys {
    generation: u32,
    num_keys: u32,
    keys: [u64; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pr_held_reservation {
    key: u64,
    generation: u32,
    r#type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pr_ops {
    pr_register: Option<unsafe extern "C" fn(*mut block_device, u64, u64, u32) -> i32>,
    pr_reserve: Option<unsafe extern "C" fn(*mut block_device, u64, u32, u32) -> i32>,
    pr_release: Option<unsafe extern "C" fn(*mut block_device, u64, u32) -> i32>,
    pr_preempt: Option<unsafe extern "C" fn(*mut block_device, u64, u64, u32, bool) -> i32>,
    pr_clear: Option<unsafe extern "C" fn(*mut block_device, u64) -> i32>,
    pr_read_keys: Option<unsafe extern "C" fn(*mut block_device, *mut pr_keys) -> i32>,
    pr_read_reservation: Option<unsafe extern "C" fn(*mut block_device, *mut pr_held_reservation) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_local_h_12 {
    a: __SlateRecord_types_h_197,
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


// Cookies are numeric values sent with CHANGE and REMOVE
// uevents while resuming, removing or renaming the device.
// For REQ_POLLED fs bio, this flag is set if we link mapped underlying
// dm_io into one list, and reuse bio->bi_private as the list head. Before
// ending this fs bio, we will recover its ->bi_private.
static mut _name: *const u8 = (b"device-mapper\0".as_ptr() as *mut u8) as *const u8;

static mut major: u32 = unsafe { core::mem::zeroed() };

static mut _major: u32 = unsafe { core::mem::zeroed() };

static mut _minor_idr: idr = idr { idr_rt: xarray { xa_lock: spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = unsafe { core::mem::zeroed() };
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0
} }, xa_flags: 16777220u32, xa_head: core::ptr::null_mut::<()>() }, idr_base: 0u32, idr_next: 0u32 };

static mut _minor_lock: spinlock = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = unsafe { core::mem::zeroed() };
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0
} };

static mut deferred_remove_work: work_struct = work_struct { data: __SlateRecord_types_h_197 { counter: 4503599625273344i64 }, entry: list_head { next: unsafe { core::ptr::addr_of_mut!(deferred_remove_work.entry) }, prev: unsafe { core::ptr::addr_of_mut!(deferred_remove_work.entry) } }, func: Some(do_deferred_remove) };

static mut deferred_remove_workqueue: *mut workqueue_struct = unsafe { core::mem::zeroed() };

#[unsafe(no_mangle)]
static mut dm_global_event_nr: __SlateRecord_types_h_190 = __SlateRecord_types_h_190 { counter: 0i32 };

#[unsafe(no_mangle)]
static mut dm_global_eventq: wait_queue_head = {
let mut __t0: wait_queue_head = unsafe { core::mem::zeroed() };
__t0.lock = spinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_spinlock_types_h_18 = unsafe { core::mem::zeroed() };
__t1.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t2: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t2.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t2
} } };
__t1
} };
__t0.head = list_head { next: unsafe { core::ptr::addr_of_mut!(dm_global_eventq.head) }, prev: unsafe { core::ptr::addr_of_mut!(dm_global_eventq.head) } };
__t0
};

#[unsafe(no_mangle)]
extern "C" fn dm_issue_global_event() {
atomic_inc(unsafe { core::ptr::addr_of_mut!(dm_global_event_nr) });
wake_up!(unsafe { core::ptr::addr_of_mut!(dm_global_eventq) });
}

#[unsafe(no_mangle)]
static mut stats_enabled: static_key_false = static_key_false { key: {
let mut __t0: static_key = unsafe { core::mem::zeroed() };
__t0.enabled = __SlateRecord_types_h_190 { counter: 0i32 };
__t0.__slate_anon_1 = __SlateRecord_jump_label_h_95 { r#type: 0u64 };
__t0
} };

#[unsafe(no_mangle)]
static mut swap_bios_enabled: static_key_false = static_key_false { key: {
let mut __t0: static_key = unsafe { core::mem::zeroed() };
__t0.enabled = __SlateRecord_types_h_190 { counter: 0i32 };
__t0.__slate_anon_1 = __SlateRecord_jump_label_h_95 { r#type: 0u64 };
__t0
} };

#[unsafe(no_mangle)]
static mut zoned_enabled: static_key_false = static_key_false { key: {
let mut __t0: static_key = unsafe { core::mem::zeroed() };
__t0.enabled = __SlateRecord_types_h_190 { counter: 0i32 };
__t0.__slate_anon_1 = __SlateRecord_jump_label_h_95 { r#type: 0u64 };
__t0
} };

/// One of these is allocated (on-stack) per original bio.
#[repr(C)]
#[derive(Clone, Copy)]
struct clone_info {
    map: *mut dm_table,
    bio: *mut bio,
    io: *mut dm_io,
    sector: u64,
    sector_count: u32,
    __slate_bits_0: __slate_bits::__SlateBits893U0,
}


fn clone_to_tio(clone: *mut bio) -> *mut dm_target_io {
container_of__13!(clone, dm_target_io)
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.dm.dm_per_bio_data")]
extern "C" fn dm_per_bio_data(bio: *mut bio, data_size: u64) -> *mut () {
if !dm_tio_flagged(clone_to_tio(bio), 0u32) {
return (bio as *mut u8).wrapping_offset(-(offsetof__5!(dm_target_io) as isize)).wrapping_offset(-(data_size as isize)) as *mut ();
}
(bio as *mut u8).wrapping_offset(-(offsetof__5!(dm_target_io).wrapping_add(offsetof__20!(dm_io)) as isize)).wrapping_offset(-(data_size as isize)) as *mut ()
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_dm_per_bio_data_806: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut bio, u64) -> *mut ()>, *mut ()>(Some(dm_per_bio_data)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.dm.dm_bio_from_per_bio_data")]
extern "C" fn dm_bio_from_per_bio_data(data: *mut (), data_size: u64) -> *mut bio {
let io: *mut dm_io = (data as *mut u8).wrapping_offset(data_size as isize) as *mut dm_io;
if ((unsafe { (*io).magic }) as i32) == 19577 {
return (io as *mut u8).wrapping_offset(offsetof__5!(dm_target_io).wrapping_add(offsetof__20!(dm_io)) as isize) as *mut bio;
}
loop {
if unlikely!(((unsafe { (*io).magic }) as i32) != 28714) != 0 {
loop {
unsafe {
core::arch::asm!("807: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 807b - ., 3; .popsection\n# {0}", const 807 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_dm_bio_from_per_bio_data_0, sym __slate_asm_string_dm_bio_from_per_bio_data_1, const 112 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
(io as *mut u8).wrapping_offset(offsetof__5!(dm_target_io) as isize) as *mut bio
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_dm_bio_from_per_bio_data_808: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), u64) -> *mut bio>, *mut ()>(Some(dm_bio_from_per_bio_data)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.dm.dm_bio_get_target_bio_nr")]
extern "C" fn dm_bio_get_target_bio_nr(bio: *const bio) -> u32 {
unsafe { (*container_of__13!(bio, dm_target_io)).target_bio_nr }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_dm_bio_get_target_bio_nr_809: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const bio) -> u32>, *mut ()>(Some(dm_bio_get_target_bio_nr)) };

static mut dm_numa_node: i32 = 1i32.wrapping_neg();

static mut swap_bios: i32 = 2048;

fn get_swap_bios() -> i32 {
let mut latch: i32 = 0;
let mut __v0: i32;
loop {
if !(__native_word__18!() || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_810() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!(swap_bios) as *const i32) };
latch = __v0;
if unlikely!(latch <= 0) != 0 {
latch = (8388608u64 / (_AC!() << 12i32)) as i32;
}
latch
}

#[repr(C)]
#[derive(Clone, Copy)]
struct table_device {
    list: list_head,
    count: refcount_struct,
    dm_dev: dm_dev,
}


// Bio-based DM's mempools' reserved IOs set by the user.
static mut reserved_bio_based_ios: u32 = 16;

fn __dm_get_module_param_int(module_param: *mut i32, min: i32, max: i32) -> i32 {
let mut param: i32 = 0;
let mut __v0: i32;
loop {
if !(__native_word__18!() || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_811() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(module_param as *const i32) };
param = __v0;
let mut modified_param: i32 = 0;
let mut modified: bool = true;
if param < min {
modified_param = min;
} else {
if param > max {
modified_param = max;
} else {
modified = false;
}
}
if modified {
let mut __v2: i32;
let __ai_ptr: *mut i32 = module_param;
kcsan_mb!();
instrument_atomic_read_write(__ai_ptr as *const (), core::mem::size_of::<i32>() as u64);
let mut __v4: i32;
let mut __ret: i32 = 0;
let mut __old: i32 = param;
let mut __new: i32 = modified_param;
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __ptr: *mut u8 = __ai_ptr as *mut u8;
unsafe {
let __t0: u32;
core::arch::asm!("lock cmpxchgb {1:e}, ({0:r})", in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t0, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
}
}
2 => {
let mut __ptr: *mut u16 = __ai_ptr as *mut u16;
unsafe {
let __t1: u32;
core::arch::asm!("lock cmpxchgw {1:e}, ({0:r})", in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t1, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t1 as u32) };
}
}
4 => {
let mut __ptr: *mut u32 = __ai_ptr as *mut u32;
unsafe {
let __t2: u32;
core::arch::asm!("lock cmpxchgl {1:e}, ({0:r})", in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t2, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t2 as u32) };
}
}
8 => {
let mut __ptr: *mut u64 = __ai_ptr as *mut u64;
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
__v4 = __ret;
__v2 = __v4;
param = modified_param;
}
param
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.dm.dm_get_reserved_bio_based_ios")]
extern "C" fn dm_get_reserved_bio_based_ios() -> u32 {
__dm_get_module_param(unsafe { core::ptr::addr_of_mut!(reserved_bio_based_ios) }, 16u32, 1024u32)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_dm_get_reserved_bio_based_ios_813: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> u32>, *mut ()>(Some(dm_get_reserved_bio_based_ios)) };

fn dm_get_numa_node() -> u32 {
__dm_get_module_param_int(unsafe { core::ptr::addr_of_mut!(dm_numa_node) }, 1i32.wrapping_neg(), num_online_nodes!().wrapping_sub(1)) as u32
}

#[unsafe(link_section = ".init.text")]
extern "C" fn local_init() -> i32 {
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut workqueue_struct> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut workqueue_struct = core::ptr::addr_of_mut!(_res) as *mut *mut workqueue_struct;
let mut r: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(r) as *mut i32;
unsafe {
let __slate_auto_bytes_2 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
*__slate_slot_4 = dm_uevent_init();
if *__slate_slot_4 != 0 {
return *__slate_slot_4;
} else {
let __slate_auto_bytes_1 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_0 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_1 = alloc_workqueue_noprof(c"kdmremove".as_ptr() as *const u8, 131074u32, 1i32);
alloc_tag_restore!();
} else {
*__slate_slot_1 = alloc_workqueue_noprof(c"kdmremove".as_ptr() as *const u8, 131074u32, 1i32);
}
deferred_remove_workqueue = *__slate_slot_1;
if unsafe { deferred_remove_workqueue }.is_null() {
*__slate_slot_4 = 12i32.wrapping_neg();
} else {
_major = major;
*__slate_slot_4 = register_blkdev!(unsafe { _major }, unsafe { _name });
if *__slate_slot_4 < 0 {
destroy_workqueue(deferred_remove_workqueue);
} else {
if _major == 0 {
_major = *__slate_slot_4 as u32;
}
return 0;
}
}
dm_uevent_exit();
return *__slate_slot_4;
}
}
}

#[unsafe(link_section = ".text.slate_distinct.dm.local_exit")]
extern "C" fn local_exit() {
unsafe { destroy_workqueue(deferred_remove_workqueue) };
unsafe { unregister_blkdev(_major, _name) };
dm_uevent_exit();
unsafe {
_major = 0;
}
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x016device-mapper: core: cleaned up\n".as_ptr() as *const u8) };
}

#[unsafe(link_section = ".init.data")]
static mut _inits: __SlateAlign16<[Option<unsafe extern "C" fn() -> i32>; 8]> = __SlateAlign16([Some(local_init), unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn() -> i32>>(dm_target_init as *const ()) }, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn() -> i32>>(dm_linear_init as *const ()) }, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn() -> i32>>(dm_stripe_init as *const ()) }, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn() -> i32>>(dm_io_init as *const ()) }, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn() -> i32>>(dm_kcopyd_init as *const ()) }, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn() -> i32>>(dm_interface_init as *const ()) }, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn() -> i32>>(dm_statistics_init as *const ()) }]);

static mut _exits: __SlateAlign16<[Option<unsafe extern "C" fn()>; 8]> = __SlateAlign16([Some(local_exit), unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn()>>(dm_target_exit as *const ()) }, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn()>>(dm_linear_exit as *const ()) }, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn()>>(dm_stripe_exit as *const ()) }, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn()>>(dm_io_exit as *const ()) }, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn()>>(dm_kcopyd_exit as *const ()) }, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn()>>(dm_interface_exit as *const ()) }, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn()>>(dm_statistics_exit as *const ()) }]);

#[unsafe(link_section = ".init.text")]
extern "C" fn dm_init() -> i32 {
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut r: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i32 = core::ptr::addr_of_mut!(r) as *mut i32;
let mut count: i32 = unsafe { core::mem::zeroed() };
unsafe {
count = ARRAY_SIZE__34!() as i32;
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_4 = 0;
'__join_1: {
while *__slate_slot_4 < count {
*__slate_slot_5 = unsafe { _inits.0[*__slate_slot_4 as usize] }.unwrap()();
if *__slate_slot_5 != 0 {
break '__join_1;
} else {
let __v3: i32 = *__slate_slot_4;
*__slate_slot_4 = __v3.wrapping_add(1);
}
}
return 0;
}
loop {
let __v1: i32 = *__slate_slot_4;
*__slate_slot_4 = __v1.wrapping_sub(1);
if __v1 != 0 {
unsafe { _exits.0[*__slate_slot_4 as usize] }.unwrap()();
} else {
break;
}
}
return *__slate_slot_5;
}
}

#[unsafe(link_section = ".exit.text")]
extern "C" fn dm_exit() {
let mut i: i32 = ARRAY_SIZE__21!() as i32;
loop {
let __v0: i32 = i;
i = __v0.wrapping_sub(1);
if __v0 == 0 {
break;
}
unsafe { unsafe { _exits.0[i as usize] }.unwrap()() };
}
// Should be empty by this point.
unsafe { idr_destroy(core::ptr::addr_of_mut!(_minor_idr)) };
}

/// Block device functions
#[unsafe(no_mangle)]
extern "C" fn dm_deleting_md(md: *mut mapped_device) -> i32 {
test_bit!(4i32, unsafe { core::ptr::addr_of_mut!((*md).flags) })
}

#[unsafe(link_section = ".text.slate_distinct.dm.dm_blk_open")]
extern "C" fn dm_blk_open(disk: *mut gendisk, mode: u32) -> i32 {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut md: core::mem::MaybeUninit<*mut mapped_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut mapped_device = core::ptr::addr_of_mut!(md) as *mut *mut mapped_device;
unsafe {
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
spin_lock(core::ptr::addr_of_mut!(_minor_lock));
*__slate_slot_1 = (*disk).private_data as *mut mapped_device;
if !(*__slate_slot_1).is_null() {
if test_bit!(3i32, core::ptr::addr_of_mut!((*(*__slate_slot_1)).flags)) != 0 {
__v0 = true;
} else {
__v0 = dm_deleting_md(*__slate_slot_1) != 0;
}
if __v0 {
*__slate_slot_1 = core::ptr::null_mut::<mapped_device>();
} else {
dm_get(*__slate_slot_1);
atomic_inc(core::ptr::addr_of_mut!((*(*__slate_slot_1)).open_count));
}
}
spin_unlock(core::ptr::addr_of_mut!(_minor_lock));
return if !(*__slate_slot_1).is_null() { 0 } else { 6i32.wrapping_neg() };
}
}

#[unsafe(link_section = ".text.slate_distinct.dm.dm_blk_close")]
extern "C" fn dm_blk_close(disk: *mut gendisk) {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut __ret_warn_on: i32 = unsafe { core::mem::zeroed() };
let mut md: core::mem::MaybeUninit<*mut mapped_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut mapped_device = core::ptr::addr_of_mut!(md) as *mut *mut mapped_device;
unsafe {
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
spin_lock(core::ptr::addr_of_mut!(_minor_lock));
*__slate_slot_4 = (*disk).private_data as *mut mapped_device;
__ret_warn_on = (*__slate_slot_4).is_null() as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("814: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 814b - ., 3; .popsection\n# {0}", const 814 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_dm_blk_close_6, sym __slate_asm_string_dm_blk_close_7, const 345 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("815: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 815b - ., 4; .popsection\n# {0}", const 815 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
let __v3: i64 = (__ret_warn_on != 0) as i64;
if __v3 == 0 {
if atomic_dec_and_test(core::ptr::addr_of_mut!((*(*__slate_slot_4)).open_count)) {
__v0 = test_bit!(6i32, core::ptr::addr_of_mut!((*(*__slate_slot_4)).flags)) != 0;
} else {
__v0 = false;
}
if __v0 {
queue_work(deferred_remove_workqueue, core::ptr::addr_of_mut!(deferred_remove_work));
}
dm_put(*__slate_slot_4);
}
spin_unlock(core::ptr::addr_of_mut!(_minor_lock));
}
}

#[unsafe(no_mangle)]
extern "C" fn dm_open_count(md: *mut mapped_device) -> i32 {
atomic_read((unsafe { core::ptr::addr_of_mut!((*md).open_count) }) as *const __SlateRecord_types_h_190)
}

/// Guarantees nothing is using the device before it's deleted.
#[unsafe(no_mangle)]
extern "C" fn dm_lock_for_deletion(md: *mut mapped_device, mark_deferred: bool, only_deferred: bool) -> i32 {
let mut r: i32 = 0;
spin_lock(unsafe { core::ptr::addr_of_mut!(_minor_lock) });
if dm_open_count(md) != 0 {
r = 16i32.wrapping_neg();
if mark_deferred {
set_bit(6i64, (unsafe { core::ptr::addr_of_mut!((*md).flags) }) as *mut u64);
}
} else {
let __v0: bool = only_deferred && test_bit!(6i32, unsafe { core::ptr::addr_of_mut!((*md).flags) }) == 0;
if __v0 {
r = 17i32.wrapping_neg();
} else {
set_bit(4i64, (unsafe { core::ptr::addr_of_mut!((*md).flags) }) as *mut u64);
}
}
spin_unlock(unsafe { core::ptr::addr_of_mut!(_minor_lock) });
r
}

#[unsafe(no_mangle)]
extern "C" fn dm_cancel_deferred_remove(md: *mut mapped_device) -> i32 {
let mut r: i32 = 0;
spin_lock(unsafe { core::ptr::addr_of_mut!(_minor_lock) });
if test_bit!(4i32, unsafe { core::ptr::addr_of_mut!((*md).flags) }) != 0 {
r = 16i32.wrapping_neg();
} else {
clear_bit(6i64, (unsafe { core::ptr::addr_of_mut!((*md).flags) }) as *mut u64);
}
spin_unlock(unsafe { core::ptr::addr_of_mut!(_minor_lock) });
r
}

#[unsafe(link_section = ".text.slate_distinct.dm.do_deferred_remove")]
extern "C" fn do_deferred_remove(w: *mut work_struct) {
unsafe { dm_deferred_remove() };
}

#[unsafe(link_section = ".text.slate_distinct.dm.dm_blk_getgeo")]
extern "C" fn dm_blk_getgeo(disk: *mut gendisk, geo: *mut hd_geometry) -> i32 {
let md: *mut mapped_device = (unsafe { (*disk).private_data }) as *mut mapped_device;
dm_get_geometry(md, geo)
}

fn dm_prepare_ioctl(md: *mut mapped_device, srcu_idx: *mut i32, bdev: *mut *mut block_device, cmd: u32, arg: u64, forward: *mut bool) -> i32 {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut r: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut i32 = core::ptr::addr_of_mut!(r) as *mut i32;
let mut map: core::mem::MaybeUninit<*mut dm_table> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut dm_table = core::ptr::addr_of_mut!(map) as *mut *mut dm_table;
let mut ti: core::mem::MaybeUninit<*mut dm_target> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut dm_target = core::ptr::addr_of_mut!(ti) as *mut *mut dm_target;
unsafe {
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
loop {
*__slate_slot_2 = 25i32.wrapping_neg();
*__slate_slot_3 = dm_get_live_table(md, srcu_idx);
if (*__slate_slot_3).is_null() {
__v1 = true;
} else {
__v1 = dm_table_get_size(*__slate_slot_3) == 0;
}
if __v1 {
return *__slate_slot_2;
} else {
// We only support devices that have a single target
if (*(*__slate_slot_3)).num_targets != 1 {
return *__slate_slot_2;
} else {
*__slate_slot_4 = dm_table_get_target(*__slate_slot_3, 0u32);
if (*(*(*__slate_slot_4)).r#type).prepare_ioctl == None {
return *__slate_slot_2;
} else {
if dm_suspended_md(md) != 0 {
return 11i32.wrapping_neg();
} else {
*__slate_slot_2 = unsafe { (*(*(*__slate_slot_4)).r#type).prepare_ioctl }.unwrap()(*__slate_slot_4, bdev, cmd, arg, forward);
if *__slate_slot_2 == 107i32.wrapping_neg() && *forward {
__v0 = fatal_signal_pending(get_current()) == 0;
} else {
__v0 = false;
}
if __v0 {
dm_put_live_table(md, *srcu_idx);
fsleep(10000u64);
} else {
break;
}
}
}
}
}
}
return *__slate_slot_2;
}
}

fn dm_unprepare_ioctl(md: *mut mapped_device, srcu_idx: i32) {
dm_put_live_table(md, srcu_idx);
}

#[unsafe(link_section = ".text.slate_distinct.dm.dm_blk_ioctl")]
extern "C" fn dm_blk_ioctl(mut bdev: *mut block_device, mode: u32, cmd: u32, arg: u64) -> i32 {
let mut forward: bool = unsafe { core::mem::zeroed() };
let mut srcu_idx: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut i32 = core::ptr::addr_of_mut!(srcu_idx) as *mut i32;
let mut r: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut i32 = core::ptr::addr_of_mut!(r) as *mut i32;
let mut md: *mut mapped_device = unsafe { core::mem::zeroed() };
unsafe {
'__join_0: {
md = (*(*bdev).bd_disk).private_data as *mut mapped_device;
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
forward = true;
*__slate_slot_3 = dm_prepare_ioctl(md, core::ptr::addr_of_mut!(*__slate_slot_2), core::ptr::addr_of_mut!(bdev), cmd, arg, core::ptr::addr_of_mut!(forward));
if !(!forward || *__slate_slot_3 < 0) {
if *__slate_slot_3 > 0 {
// Target determined this ioctl is being issued against a
// subset of the parent bdev; require extra privileges.
if !capable(17i32) {
*__slate_slot_3 = 515i32.wrapping_neg();
break '__join_0;
}
}
if (*(*(*bdev).bd_disk).fops).ioctl == None {
*__slate_slot_3 = 25i32.wrapping_neg();
} else {
*__slate_slot_3 = unsafe { (*(*(*bdev).bd_disk).fops).ioctl }.unwrap()(bdev, mode, cmd, arg);
}
}
}
dm_unprepare_ioctl(md, *__slate_slot_2);
return *__slate_slot_3;
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.dm.dm_start_time_ns_from_clone")]
extern "C" fn dm_start_time_ns_from_clone(bio: *mut bio) -> u64 {
jiffies_to_nsecs(unsafe { (*(*clone_to_tio(bio)).io).start_time })
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_dm_start_time_ns_from_clone_816: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut bio) -> u64>, *mut ()>(Some(dm_start_time_ns_from_clone)) };

fn dm_io_sectors(io: *mut dm_io, bio: *mut bio) -> u32 {
// If REQ_PREFLUSH set, don't account payload, it will be
// submitted (and accounted) after this flush completes.
if unsafe { (*io).requeue_flush_with_data } {
return 0;
}
if unlikely!(dm_io_flagged(io, 1u32)) != 0 {
return unsafe { (*io).sectors };
}
bio_sectors!(bio)
}

fn dm_io_acct(io: *mut dm_io, end: bool) {
let bio: *mut bio = unsafe { (*io).orig_bio };
if dm_io_flagged(io, 2u32) {
if !end {
unsafe { bdev_start_io_acct((*bio).bi_bdev, bio_op(bio as *const bio), (*io).start_time) };
} else {
unsafe { bdev_end_io_acct((*bio).bi_bdev, bio_op(bio as *const bio), dm_io_sectors(io, bio), (*io).start_time) };
}
}
let __v0: bool = static_branch_unlikely__4!(unsafe { core::ptr::addr_of_mut!(stats_enabled) }) != 0 && unlikely!(dm_stats_used(unsafe { core::ptr::addr_of_mut!((*(*io).md).stats) })) != 0;
if __v0 {
let sector: u64 = if unlikely!(dm_io_flagged(io, 1u32)) != 0 {
bio_end_sector!(bio).wrapping_sub((unsafe { (*io).sector_offset }) as u64)
} else {
unsafe { (*bio).bi_iter.bi_sector }
};
unsafe { dm_stats_account_io(core::ptr::addr_of_mut!((*(*io).md).stats), bio_data_dir!(bio) as u64, sector, dm_io_sectors(io, bio), end, (*io).start_time, core::ptr::addr_of_mut!((*io).stats_aux)) };
}
}

fn __dm_start_io_acct(io: *mut dm_io) {
dm_io_acct(io, false);
}

fn dm_start_io_acct(io: *mut dm_io, clone: *mut bio) {
// Ensure IO accounting is only ever started once.
if dm_io_flagged(io, 0u32) {
return;
}
// Expect no possibility for race unless DM_TIO_IS_DUPLICATE_BIO.
let __v0: bool = clone.is_null() || likely!(dm_tio_is_normal(clone_to_tio(clone))) != 0;
if __v0 {
dm_io_set_flag(io, 0u32);
} else {
let mut flags: u64 = 0;
// Can afford locking given DM_TIO_IS_DUPLICATE_BIO
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*io).lock) }, flags);
if dm_io_flagged(io, 0u32) {
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*io).lock) }, flags);
return;
}
dm_io_set_flag(io, 0u32);
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*io).lock) }, flags);
}
__dm_start_io_acct(io);
}

fn dm_end_io_acct(io: *mut dm_io) {
dm_io_acct(io, true);
}

fn alloc_io(mut md: *mut mapped_device, bio: *mut bio, gfp_mask: u32) -> *mut dm_io {
let mut __slate_compound_8: spinlock = unsafe { core::mem::zeroed() };
let mut io: *mut dm_io = unsafe { core::mem::zeroed() };
let clone: *mut bio = unsafe { bio_alloc_clone(core::ptr::null_mut::<block_device>(), bio, gfp_mask, core::ptr::addr_of_mut!((*(*md).mempools).io_bs)) };
if unlikely!(clone.is_null()) != 0 {
return core::ptr::null_mut::<dm_io>();
}
let tio: *mut dm_target_io = clone_to_tio(clone);
unsafe {
(*tio).flags = 0;
}
dm_tio_set_flag(tio, 0u32);
unsafe {
(*tio).io = core::ptr::null_mut::<dm_io>();
}
io = container_of__63!(tio, dm_io);
unsafe {
(*io).magic = 19577;
}
unsafe {
(*io).status = 0;
}
unsafe {
(*io).requeue_flush_with_data = false;
}
// one ref is for submission, the other is for completion
atomic_set(unsafe { core::ptr::addr_of_mut!((*io).io_count) }, 2i32);
loop {
__verify_pcpu_ptr__6!();
match (const { core::mem::size_of::<u64>() }) as u64 {
1 => {
loop {
unsafe {
core::arch::asm!("incb %gs:({0:r})", in(reg) ((*md).pending_io as u64) as *mut u64, options(nostack, att_syntax));
}
if true {
break;
}
}
}
2 => {
loop {
unsafe {
core::arch::asm!("incw %gs:({0:r})", in(reg) ((*md).pending_io as u64) as *mut u64, options(nostack, att_syntax));
}
if true {
break;
}
}
}
4 => {
loop {
unsafe {
core::arch::asm!("incl %gs:({0:r})", in(reg) ((*md).pending_io as u64) as *mut u64, options(nostack, att_syntax));
}
if true {
break;
}
}
}
8 => {
loop {
unsafe {
core::arch::asm!("incq %gs:({0:r})", in(reg) ((*md).pending_io as u64) as *mut u64, options(nostack, att_syntax));
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
unsafe {
(*io).orig_bio = bio;
}
unsafe {
(*io).md = md;
}
'__slate_break_9: loop {
spinlock_check(unsafe { core::ptr::addr_of_mut!((*io).lock) });
unsafe {
*core::ptr::addr_of_mut!((*io).lock) = *{
__slate_compound_8 = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = core::mem::zeroed();
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = core::mem::zeroed();
__t1.val = __SlateRecord_types_h_190 { counter: 0 };
__t1
} } };
__t0
} };
core::ptr::addr_of_mut!(__slate_compound_8)
};
}
if true {
break;
}
}
unsafe {
(*io).start_time = core::ptr::read_volatile(core::ptr::addr_of!(jiffies));
}
unsafe {
(*io).flags = 0;
}
if blk_queue_io_stat!(unsafe { (*md).queue }) != 0 {
dm_io_set_flag(io, 2u32);
}
let __v10: bool = static_branch_unlikely__5!(unsafe { core::ptr::addr_of_mut!(stats_enabled) }) != 0 && unlikely!(dm_stats_used(unsafe { core::ptr::addr_of_mut!((*md).stats) })) != 0;
if __v10 {
dm_stats_record_start(unsafe { core::ptr::addr_of_mut!((*md).stats) }, unsafe { core::ptr::addr_of_mut!((*io).stats_aux) });
}
io
}

fn free_io(io: *mut dm_io) {
unsafe { bio_put(core::ptr::addr_of_mut!((*io).tio.clone)) };
}

fn alloc_tio(ci: *mut clone_info, ti: *mut dm_target, target_bio_nr: u32, len: *mut u32, gfp_mask: u32) -> *mut bio {
let md: *mut mapped_device = unsafe { (*(*ci).io).md };
let mut tio: *mut dm_target_io = unsafe { core::mem::zeroed() };
let mut clone: *mut bio = unsafe { core::mem::zeroed() };
if unsafe { (*(*ci).io).tio.io }.is_null() {
// the dm_target_io embedded in ci->io is available
tio = unsafe { core::ptr::addr_of_mut!((*(*ci).io).tio) };
// alloc_io() already initialized embedded clone
clone = unsafe { core::ptr::addr_of_mut!((*tio).clone) };
} else {
clone = unsafe { bio_alloc_clone(core::ptr::null_mut::<block_device>(), (*ci).bio, gfp_mask, core::ptr::addr_of_mut!((*(*md).mempools).bs)) };
if clone.is_null() {
return core::ptr::null_mut::<bio>();
}
// REQ_DM_POLL_LIST shouldn't be inherited
let __v2: u32 = (unsafe { (*clone).bi_opf }) & 4261412863u32;
unsafe {
(*clone).bi_opf = __v2;
}
tio = clone_to_tio(clone);
unsafe {
(*tio).flags = 0;
}
// also clears DM_TIO_INSIDE_DM_IO
}
unsafe {
(*tio).magic = 28714;
}
unsafe {
(*tio).io = (*ci).io;
}
unsafe {
(*tio).ti = ti;
}
unsafe {
(*tio).target_bio_nr = target_bio_nr;
}
unsafe {
(*tio).len_ptr = len;
}
unsafe {
(*tio).old_sector = 0;
}
// Set default bdev, but target must bio_set_dev() before issuing IO
unsafe {
(*clone).bi_bdev = (*(*md).disk).part0;
}
if likely!(!ti.is_null()) != 0 && unlikely__2!((unsafe { (*ti).__slate_bits_0.__get_needs_bio_set_dev() }) as i32) != 0 {
bio_set_dev(clone, unsafe { (*(*md).disk).part0 });
}
if !len.is_null() {
unsafe {
(*clone).bi_iter.bi_size = to_bytes(*len as u64) as u32;
}
if !bio_integrity(clone).is_null() {
bio_integrity_trim(clone);
}
}
clone
}

fn free_tio(clone: *mut bio) {
if dm_tio_flagged(clone_to_tio(clone), 0u32) {
return;
}
unsafe { bio_put(clone) };
}

/// Add the bio to the list of deferred io.
fn queue_io(md: *mut mapped_device, bio: *mut bio) {
let mut flags: u64 = 0;
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*md).deferred_lock) }, flags);
bio_list_add(unsafe { core::ptr::addr_of_mut!((*md).deferred) }, bio);
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*md).deferred_lock) }, flags);
queue_work(unsafe { (*md).wq }, unsafe { core::ptr::addr_of_mut!((*md).work) });
}

// Everyone (including functions in this file), should use this
// function to access the md->map field, and make sure they call
// dm_put_live_table() when finished.
#[unsafe(no_mangle)]
extern "C" fn dm_put_live_table(md: *mut mapped_device, srcu_idx: i32) {
srcu_read_unlock(unsafe { core::ptr::addr_of_mut!((*md).io_barrier) }, srcu_idx);
}

#[unsafe(no_mangle)]
extern "C" fn dm_sync_table(md: *mut mapped_device) {
unsafe { synchronize_srcu(core::ptr::addr_of_mut!((*md).io_barrier)) };
unsafe { synchronize_rcu_expedited() };
}

// A fast alternative to dm_get_live_table/dm_put_live_table.
// The caller must not block between these two functions.
fn dm_get_live_table_fast(md: *mut mapped_device) -> *mut dm_table {
rcu_read_lock();
let mut __v0: *mut ();
let mut __v1: *mut ();
loop {
if !(__native_word__7!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_820() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*md).map) as *const *mut ()) };
let __UNIQUE_ID_rcu_819: *mut () = __v1;
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
__v0 = __UNIQUE_ID_rcu_819;
__v0 as *mut dm_table
}

fn dm_put_live_table_fast(md: *mut mapped_device) {
rcu_read_unlock();
}

static mut _dm_claim_ptr: *mut u8 = b"I belong to device-mapper\0".as_ptr() as *mut u8;

/// Open a table device so we can use it as a map destination.
fn open_table_device(md: *mut mapped_device, dev: u32, mode: u32) -> *mut table_device {
let mut __slate_compound_11: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_13: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
// Open the backing device with kernel rather than caller
// credentials. Otherwise the caller's credentials would be
// pinned in bdev_file->f_cred until the table device is closed.
// That would keep the caller's thread keyring alive long beyond the
// lifetime of the caller, breaking userspace expectation (e.g.
// cryptsetup(8) leaking the LUKS volume key).
let mut __UNIQUE_ID_label_821: core::mem::MaybeUninit<*const cred> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *const cred = core::ptr::addr_of_mut!(__UNIQUE_ID_label_821) as *mut *const cred;
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut r: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i32 = core::ptr::addr_of_mut!(r) as *mut i32;
let mut part_off: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u64 = core::ptr::addr_of_mut!(part_off) as *mut u64;
let mut bdev: core::mem::MaybeUninit<*mut block_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut block_device = core::ptr::addr_of_mut!(bdev) as *mut *mut block_device;
let mut bdev_file: core::mem::MaybeUninit<*mut file> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut file = core::ptr::addr_of_mut!(bdev_file) as *mut *mut file;
let mut td: core::mem::MaybeUninit<*mut table_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut table_device = core::ptr::addr_of_mut!(td) as *mut *mut table_device;
let __slate_cleanup_ptr_0 = __slate_slot_0;
let __slate_cleanup_armed_0 = core::cell::Cell::new(false);
let __slate_cleanup_0 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_0.get() {
unsafe { class_override_creds_destructor(__slate_cleanup_ptr_0 as *mut _) };
}
});
unsafe {
let __slate_auto_bytes_1 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_6 = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_1 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_2 = _kmalloc_node_noprof(core::mem::size_of::<table_device>() as u64, 3264u32, (*md).numa_node_id, *{
__slate_compound_11 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_11)
});
alloc_tag_restore!();
} else {
*__slate_slot_2 = _kmalloc_node_noprof(core::mem::size_of::<table_device>() as u64, 3264u32, (*md).numa_node_id, *{
__slate_compound_13 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_13)
});
}
*__slate_slot_9 = *__slate_slot_2 as *mut table_device;
if (*__slate_slot_9).is_null() {
return ERR_PTR(12i32.wrapping_neg() as i64) as *mut table_device;
} else {
refcount_set(core::ptr::addr_of_mut!((*(*__slate_slot_9)).count), 1i32);
core::ptr::write(__slate_slot_0, class_override_creds_constructor(kernel_cred()));
__slate_cleanup_armed_0.set(true);
if true {
*__slate_slot_8 = bdev_file_open_by_dev(dev, mode, _dm_claim_ptr as *mut (), core::ptr::null::<blk_holder_ops>());
}
'__join_0: {
__slate_cleanup_armed_0.set(false);
class_override_creds_destructor(__slate_slot_0 as *mut _);
if IS_ERR(*__slate_slot_8 as *const ()) {
*__slate_slot_5 = PTR_ERR(*__slate_slot_8 as *const ()) as i32;
} else {
*__slate_slot_7 = file_bdev(*__slate_slot_8);
// We can be called before the dm disk is added.  In that case we can't
// register the holder relation here.  It will be done once add_disk was
// called.
if !unsafe { (*(*md).disk).slave_dir }.is_null() {
*__slate_slot_5 = bd_link_disk_holder(*__slate_slot_7, (*md).disk);
if *__slate_slot_5 != 0 {
__fput_sync(*__slate_slot_8);
break '__join_0;
}
}
(*(*__slate_slot_9)).dm_dev.mode = mode;
(*(*__slate_slot_9)).dm_dev.bdev = *__slate_slot_7;
(*(*__slate_slot_9)).dm_dev.bdev_file = *__slate_slot_8;
(*(*__slate_slot_9)).dm_dev.dax_dev = fs_dax_get_by_bdev(*__slate_slot_7, &(*core::ptr::addr_of_mut!(*__slate_slot_6)), core::ptr::null_mut::<()>(), core::ptr::null::<dax_holder_operations>());
format_dev_t!((*(*__slate_slot_9)).dm_dev.name.as_mut_ptr() as *mut u8, dev);
list_add(core::ptr::addr_of_mut!((*(*__slate_slot_9)).list), core::ptr::addr_of_mut!((*md).table_devices));
return *__slate_slot_9;
}
}
kfree(*__slate_slot_9 as *const ());
return ERR_PTR(*__slate_slot_5 as i64) as *mut table_device;
}
}
}

/// Close a table device that we've been using.
fn close_table_device(td: *mut table_device, md: *mut mapped_device) {
if !unsafe { (*(*md).disk).slave_dir }.is_null() {
unsafe { bd_unlink_disk_holder((*td).dm_dev.bdev, (*md).disk) };
}
// Leverage async fput() if DMF_DEFERRED_REMOVE set
if unlikely__2!(test_bit!(6i32, unsafe { core::ptr::addr_of_mut!((*md).flags) })) != 0 {
unsafe { fput((*td).dm_dev.bdev_file) };
} else {
unsafe { __fput_sync((*td).dm_dev.bdev_file) };
}
put_dax(unsafe { (*td).dm_dev.dax_dev });
list_del(unsafe { core::ptr::addr_of_mut!((*td).list) });
unsafe { kfree(td as *const ()) };
}

fn find_table_device(l: *mut list_head, dev: u32, mode: u32) -> *mut table_device {
let mut td: *mut table_device = unsafe { core::mem::zeroed() };
td = list_first_entry__21!(l);
while list_entry_is_head!(td, l, list) == 0 {
if (unsafe { (*(*td).dm_dev.bdev).bd_dev }) == dev && (unsafe { (*td).dm_dev.mode }) == mode {
return td;
}
td = list_next_entry__28!(td, list);
}
core::ptr::null_mut::<table_device>()
}

#[unsafe(no_mangle)]
extern "C" fn dm_get_table_device(md: *mut mapped_device, dev: u32, mode: u32, result: *mut *mut dm_dev) -> i32 {
let mut td: *mut table_device = unsafe { core::mem::zeroed() };
unsafe { mutex_lock(core::ptr::addr_of_mut!((*md).table_devices_lock)) };
td = find_table_device(unsafe { core::ptr::addr_of_mut!((*md).table_devices) }, dev, mode);
if td.is_null() {
td = open_table_device(md, dev, mode);
if IS_ERR(td as *const ()) {
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*md).table_devices_lock)) };
return PTR_ERR(td as *const ()) as i32;
}
} else {
refcount_inc(unsafe { core::ptr::addr_of_mut!((*td).count) });
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*md).table_devices_lock)) };
unsafe {
*result = core::ptr::addr_of_mut!((*td).dm_dev);
}
0
}

#[unsafe(no_mangle)]
extern "C" fn dm_put_table_device(md: *mut mapped_device, d: *mut dm_dev) {
let td: *mut table_device = container_of__43!(d, table_device);
unsafe { mutex_lock(core::ptr::addr_of_mut!((*md).table_devices_lock)) };
if refcount_dec_and_test(unsafe { core::ptr::addr_of_mut!((*td).count) }) {
close_table_device(td, md);
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*md).table_devices_lock)) };
}

/// Get the geometry associated with a dm device
#[unsafe(no_mangle)]
extern "C" fn dm_get_geometry(md: *mut mapped_device, geo: *mut hd_geometry) -> i32 {
unsafe {
*geo = (*md).geometry;
}
0
}

/// Set the geometry of a device.
#[unsafe(no_mangle)]
extern "C" fn dm_set_geometry(md: *mut mapped_device, geo: *mut hd_geometry) -> i32 {
let sz: u64 = ((unsafe { (*geo).cylinders }) as u64).wrapping_mul(((unsafe { (*geo).heads }) as i32) as u64).wrapping_mul(((unsafe { (*geo).sectors }) as i32) as u64);
if (unsafe { (*geo).start }) > sz {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x013device-mapper: core: Start sector is beyond the geometry limits.\n".as_ptr() as *const u8) };
return 22i32.wrapping_neg();
}
unsafe {
(*md).geometry = *geo;
}
0
}

fn __noflush_suspending(md: *mut mapped_device) -> i32 {
test_bit!(5i32, unsafe { core::ptr::addr_of_mut!((*md).flags) })
}

fn dm_requeue_add_io(io: *mut dm_io, first_stage: bool) {
let md: *mut mapped_device = unsafe { (*io).md };
if first_stage {
let next: *mut dm_io = unsafe { (*md).requeue_list };
unsafe {
(*md).requeue_list = io;
}
unsafe {
(*io).next = next;
}
} else {
bio_list_add_head(unsafe { core::ptr::addr_of_mut!((*md).deferred) }, unsafe { (*io).orig_bio });
}
}

fn dm_kick_requeue(md: *mut mapped_device, first_stage: bool) {
if first_stage {
queue_work(unsafe { (*md).wq }, unsafe { core::ptr::addr_of_mut!((*md).requeue_work) });
} else {
queue_work(unsafe { (*md).wq }, unsafe { core::ptr::addr_of_mut!((*md).work) });
}
}

/// Return true if the dm_io's original bio is requeued.
/// io->status is updated with error if requeue disallowed.
fn dm_handle_requeue(io: *mut dm_io, first_stage: bool) -> bool {
let bio: *mut bio = unsafe { (*io).orig_bio };
let handle_requeue: bool = ((unsafe { (*io).status }) as i32) == 11;
let handle_polled_eagain: bool = ((unsafe { (*io).status }) as i32) == 12 && (unsafe { (*bio).bi_opf }) & 4194304u32 != 0;
let md: *mut mapped_device = unsafe { (*io).md };
let mut requeued: bool = false;
if handle_requeue || handle_polled_eagain {
let mut flags: u64 = 0;
if (unsafe { (*bio).bi_opf }) & 4194304u32 != 0 {
// Upper layer won't help us poll split bio
// (io->orig_bio may only reflect a subset of the
// pre-split original) so clear REQ_POLLED.
bio_clear_polled(bio);
}
// Target requested pushing back the I/O or
// polled IO hit BLK_STS_AGAIN.
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*md).deferred_lock) }, flags);
let __v5: bool = if __noflush_suspending(md) != 0 {
let mut __v6: i64;
let __ret_warn_on: i32 = !(!dm_is_zone_write(md, bio)) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("823: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 823b - ., 3; .popsection\n# {0}", const 823 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_dm_handle_requeue_7, sym __slate_asm_string_dm_handle_requeue_8, const 930 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("824: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 824b - ., 4; .popsection\n# {0}", const 824 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v6 = (__ret_warn_on != 0) as i64;
__v6 == 0
} else {
false
};
if __v5 || handle_polled_eagain || first_stage {
dm_requeue_add_io(io, first_stage);
requeued = true;
} else {
// noflush suspend was interrupted or this is
// a write to a zoned target.
unsafe {
(*io).status = 10;
}
}
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*md).deferred_lock) }, flags);
}
if requeued {
dm_kick_requeue(md, first_stage);
}
requeued
}

fn __dm_io_complete(io: *mut dm_io, first_stage: bool) {
let bio: *mut bio = unsafe { (*io).orig_bio };
let mut md: *mut mapped_device = unsafe { (*io).md };
let mut io_error: u8 = 0;
let mut requeued: bool = false;
let mut requeue_flush_with_data: bool = false;
requeued = dm_handle_requeue(io, first_stage);
if requeued && first_stage {
return;
}
io_error = unsafe { (*io).status };
if dm_io_flagged(io, 0u32) {
dm_end_io_acct(io);
} else {
if io_error == 0 {
// Must handle target that DM_MAPIO_SUBMITTED only to
// then bio_endio() rather than dm_submit_bio_remap()
__dm_start_io_acct(io);
dm_end_io_acct(io);
}
}
requeue_flush_with_data = unsafe { (*io).requeue_flush_with_data };
free_io(io);
loop {
kcsan_wmb!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
loop {
__verify_pcpu_ptr__6!();
match (const { core::mem::size_of::<u64>() }) as u64 {
1 => {
loop {
unsafe {
core::arch::asm!("decb %gs:({0:r})", in(reg) ((*md).pending_io as u64) as *mut u64, options(nostack, att_syntax));
}
if true {
break;
}
}
}
2 => {
loop {
unsafe {
core::arch::asm!("decw %gs:({0:r})", in(reg) ((*md).pending_io as u64) as *mut u64, options(nostack, att_syntax));
}
if true {
break;
}
}
}
4 => {
loop {
unsafe {
core::arch::asm!("decl %gs:({0:r})", in(reg) ((*md).pending_io as u64) as *mut u64, options(nostack, att_syntax));
}
if true {
break;
}
}
}
8 => {
loop {
unsafe {
core::arch::asm!("decq %gs:({0:r})", in(reg) ((*md).pending_io as u64) as *mut u64, options(nostack, att_syntax));
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
// nudge anyone waiting on suspend queue
if unlikely!(wq_has_sleeper(unsafe { core::ptr::addr_of_mut!((*md).wait) })) != 0 {
wake_up!(unsafe { core::ptr::addr_of_mut!((*md).wait) });
}
// Return early if the original bio was requeued
if requeued {
return;
}
if unlikely!(requeue_flush_with_data) != 0 {
// Preflush done for flush with data, reissue
// without REQ_PREFLUSH.
let __v11: u32 = (unsafe { (*bio).bi_opf }) & 4294705151u32;
unsafe {
(*bio).bi_opf = __v11;
}
queue_io(md, bio);
} else {
// done with normal IO or empty flush
if io_error != 0 {
unsafe {
(*bio).bi_status = io_error;
}
}
unsafe { bio_endio(bio) };
}
}

#[unsafe(link_section = ".text.slate_distinct.dm.dm_wq_requeue_work")]
extern "C" fn dm_wq_requeue_work(work: *mut work_struct) {
let md: *mut mapped_device = container_of__74!(work, mapped_device);
let mut flags: u64 = 0;
let mut io: *mut dm_io = unsafe { core::mem::zeroed() };
// reuse deferred lock to simplify dm_handle_requeue
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*md).deferred_lock) }, flags);
io = unsafe { (*md).requeue_list };
unsafe {
(*md).requeue_list = core::ptr::null_mut::<dm_io>();
}
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*md).deferred_lock) }, flags);
while !io.is_null() {
let next: *mut dm_io = unsafe { (*io).next };
unsafe { dm_io_rewind(io, core::ptr::addr_of_mut!((*(*md).disk).bio_split)) };
unsafe {
(*io).next = core::ptr::null_mut::<dm_io>();
}
__dm_io_complete(io, false);
io = next;
cond_resched__11!();
}
}

/// Two staged requeue:
///
/// 1) io->orig_bio points to the real original bio, and the part mapped to
///    this io must be requeued, instead of other parts of the original bio.
///
/// 2) io->orig_bio points to new cloned bio which matches the requeued dm_io.
fn dm_io_complete(io: *mut dm_io) {
// Only dm_io that has been split needs two stage requeue, otherwise
// we may run into long bio clone chain during suspend and OOM could
// be triggered.
//
// Also flush data dm_io won't be marked as DM_IO_WAS_SPLIT, so they
// also aren't handled via the first stage requeue.
__dm_io_complete(io, dm_io_flagged(io, 1u32));
}

/// Decrements the number of outstanding ios that a bio has been
/// cloned into, completing the original io if necc.
fn __dm_io_dec_pending(io: *mut dm_io) {
if atomic_dec_and_test(unsafe { core::ptr::addr_of_mut!((*io).io_count) }) {
dm_io_complete(io);
}
}

fn dm_io_set_error(io: *mut dm_io, error: u8) {
let mut flags: u64 = 0;
// Push-back supersedes any I/O errors
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*io).lock) }, flags);
let __v5: bool = ((unsafe { (*io).status }) as i32) == 11 && __noflush_suspending(unsafe { (*io).md }) != 0;
if !__v5 {
unsafe {
(*io).status = error;
}
}
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*io).lock) }, flags);
}

fn dm_io_dec_pending(io: *mut dm_io, error: u8) {
if unlikely__4!(error) != 0 {
dm_io_set_error(io, error);
}
__dm_io_dec_pending(io);
}

/// The queue_limits are only valid as long as you have a reference
/// count on 'md'. But _not_ imposing verification to avoid atomic_read(),
fn dm_get_queue_limits(md: *mut mapped_device) -> *mut queue_limits {
unsafe { core::ptr::addr_of_mut!((*(*md).queue).limits) }
}

fn swap_bios_limit(ti: *mut dm_target, bio: *mut bio) -> bool {
unlikely!((unsafe { (*bio).bi_opf }) & 16777216u32 != 0) != 0 && unlikely__2!((unsafe { (*ti).__slate_bits_0.__get_limit_swap_bios() }) as i32) != 0
}

#[unsafe(link_section = ".text.slate_distinct.dm.clone_endio")]
extern "C" fn clone_endio(bio: *mut bio) {
let mut error: u8 = unsafe { (*bio).bi_status };
let tio: *mut dm_target_io = clone_to_tio(bio);
let ti: *mut dm_target = unsafe { (*tio).ti };
let endio: Option<unsafe extern "C" fn(*mut dm_target, *mut bio, *mut u8) -> i32> = {
let __t0: Option<unsafe extern "C" fn(*mut dm_target, *mut bio, *mut u8) -> i32> = if likely!(!ti.is_null()) != 0 { unsafe { (*(*ti).r#type).end_io } } else { None };
__t0
};
let io: *mut dm_io = unsafe { (*tio).io };
let md: *mut mapped_device = unsafe { (*io).md };
if unlikely!((error as i32) == 5) != 0 {
let __v0: bool = (bio_op(bio as *const bio) as u32) == 3 && bdev_max_discard_sectors(unsafe { (*bio).bi_bdev }) == 0;
if __v0 {
blk_queue_disable_discard(unsafe { (*md).queue });
} else {
let __v1: bool = (bio_op(bio as *const bio) as u32) == 9 && bdev_write_zeroes_sectors(unsafe { (*bio).bi_bdev }) == 0;
if __v1 {
blk_queue_disable_write_zeroes(unsafe { (*md).queue });
}
}
}
let __v2: bool = static_branch_unlikely__6!(unsafe { core::ptr::addr_of_mut!(zoned_enabled) }) != 0 && unlikely!(bdev_is_zoned(unsafe { (*bio).bi_bdev })) != 0;
if __v2 {
unsafe { dm_zone_endio(io, bio) };
}
if endio != None {
let r: i32 = unsafe { endio.unwrap()(ti, bio, core::ptr::addr_of_mut!(error)) };
// Requeuing writes to a sequential zone of a zoned
// target will break the sequential write pattern:
// fail such IO.
// The target will handle the io
match r {
2 => {
if static_branch_unlikely__13!(unsafe { core::ptr::addr_of_mut!(zoned_enabled) }) != 0 {
// Requeuing writes to a sequential zone of a zoned
// target will break the sequential write pattern:
// fail such IO.
let mut __v8: i64;
let __ret_warn_on: i32 = !(!dm_is_zone_write(md, bio)) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("825: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 825b - ., 3; .popsection\n# {0}", const 825 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_clone_endio_9, sym __slate_asm_string_clone_endio_10, const 1125 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("826: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 826b - ., 4; .popsection\n# {0}", const 826 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v8 = (__ret_warn_on != 0) as i64;
if __v8 != 0 {
error = 10;
} else {
error = 11;
}
} else {
error = 11;
}
}
0 => {
}
1 => {
return; // The target will handle the io
}
_ => {
let mut __v13: i32;
while false {
}
__v13 = unsafe { _printk(c"\x012device-mapper: core: unimplemented target endio return value: %d\n".as_ptr() as *const u8, r) };
loop {
unsafe {
core::arch::asm!("827: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 827b - ., 3; .popsection\n# {0}", const 827 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_clone_endio_15, sym __slate_asm_string_clone_endio_16, const 1139 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
}
}
let __v19: bool = static_branch_unlikely__12!(unsafe { core::ptr::addr_of_mut!(swap_bios_enabled) }) != 0 && likely!(!ti.is_null()) != 0 && unlikely!(swap_bios_limit(ti, bio)) != 0;
if __v19 {
unsafe { up(core::ptr::addr_of_mut!((*md).swap_bios_semaphore)) };
}
free_tio(bio);
dm_io_dec_pending(io, error);
}

/// Return maximum size of I/O possible at the supplied sector up to the current
/// target boundary.
fn max_io_len_target_boundary(ti: *mut dm_target, target_offset: u64) -> u64 {
unsafe { (*ti).len }.wrapping_sub(target_offset)
}

fn __max_io_len(ti: *mut dm_target, sector: u64, max_granularity: u32, __v2: u32) -> u64 {
let target_offset: u64 = dm_target_offset!(ti, sector);
let len: u64 = max_io_len_target_boundary(ti, target_offset);
// Does the target need to split IO even further?
// - varied (per target) IO splitting is a tenet of DM; this
//   explains why stacked chunk_sectors based splitting via
//   bio_split_to_limits() isn't possible here.
if max_granularity == 0 {
return len;
}
let mut __v0: u64;
let __UNIQUE_ID_x__831: u64 = len;
let mut __UNIQUE_ID_y__832: u64 = 0;
let mut __v1: u32;
let mut __UNIQUE_ID_x__828: u32 = 0;
let __v3: u32 = if __v2 != 0 {
__v2
} else {
queue_max_sectors((unsafe { (*(*(*ti).table).md).queue }) as *const request_queue)
};
__UNIQUE_ID_x__828 = __v3;
let __UNIQUE_ID_y__829: u32 = blk_boundary_sectors_left(target_offset, max_granularity);
loop {
if __types_ok!(__UNIQUE_ID_x__828, __UNIQUE_ID_y__829) == 0 {
unsafe { __compiletime_assert_830() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = if __UNIQUE_ID_x__828 < __UNIQUE_ID_y__829 { __UNIQUE_ID_x__828 } else { __UNIQUE_ID_y__829 };
__UNIQUE_ID_y__832 = __v1 as u64;
__v0 = if __UNIQUE_ID_x__831 < __UNIQUE_ID_y__832 { __UNIQUE_ID_x__831 } else { __UNIQUE_ID_y__832 };
__v0
}

fn max_io_len(ti: *mut dm_target, sector: u64) -> u64 {
__max_io_len(ti, sector, unsafe { (*ti).max_io_len }, 0u32)
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.dm.dm_set_target_max_io_len")]
extern "C" fn dm_set_target_max_io_len(ti: *mut dm_target, len: u64) -> i32 {
if len > 4294967295u64 {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x013device-mapper: core: Specified maximum size of target IO (%llu) exceeds limit (%u)\n".as_ptr() as *const u8, len, 4294967295u32) };
unsafe {
(*ti).error = c"Maximum size of target IO is too large".as_ptr() as *mut u8;
}
return 22i32.wrapping_neg();
}
unsafe {
(*ti).max_io_len = len as u32;
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_dm_set_target_max_io_len_833: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut dm_target, u64) -> i32>, *mut ()>(Some(dm_set_target_max_io_len)) };

fn dm_dax_get_live_target(md: *mut mapped_device, sector: u64, srcu_idx: *mut i32) -> *mut dm_target {
let map: *mut dm_table = dm_get_live_table(md, srcu_idx);
if map.is_null() {
return core::ptr::null_mut::<dm_target>();
}
let ti: *mut dm_target = unsafe { dm_table_find_target(map, sector) };
if ti.is_null() {
return core::ptr::null_mut::<dm_target>();
}
ti
}

#[unsafe(link_section = ".text.slate_distinct.dm.dm_dax_direct_access")]
extern "C" fn dm_dax_direct_access(dax_dev: *mut dax_device, pgoff: u64, mut nr_pages: i64, mode: u32, kaddr: *mut *mut (), pfn: *mut u64) -> i64 {
let mut __UNIQUE_ID_y__835: i64 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__834: i64 = unsafe { core::mem::zeroed() };
let mut srcu_idx: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut i32 = core::ptr::addr_of_mut!(srcu_idx) as *mut i32;
let mut ret: i64 = unsafe { core::mem::zeroed() };
let mut len: core::mem::MaybeUninit<i64> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i64 = core::ptr::addr_of_mut!(len) as *mut i64;
let mut ti: core::mem::MaybeUninit<*mut dm_target> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut dm_target = core::ptr::addr_of_mut!(ti) as *mut *mut dm_target;
let mut sector: u64 = unsafe { core::mem::zeroed() };
let mut md: *mut mapped_device = unsafe { core::mem::zeroed() };
unsafe {
md = dax_get_private(dax_dev) as *mut mapped_device;
sector = pgoff.wrapping_mul(8u64);
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
ret = 5i32.wrapping_neg() as i64;
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_6 = dm_dax_get_live_target(md, sector, core::ptr::addr_of_mut!(*__slate_slot_3));
if !(*__slate_slot_6).is_null() {
if (*(*(*__slate_slot_6)).r#type).direct_access != None {
*__slate_slot_5 = (max_io_len(*__slate_slot_6, sector) / 8u64) as i64;
if *__slate_slot_5 >= 1 {
__UNIQUE_ID_x__834 = *__slate_slot_5;
__UNIQUE_ID_y__835 = nr_pages;
loop {
if __types_ok__10!(__UNIQUE_ID_x__834, __UNIQUE_ID_y__835) == 0 {
__compiletime_assert_836();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
nr_pages = if __UNIQUE_ID_x__834 < __UNIQUE_ID_y__835 { __UNIQUE_ID_x__834 } else { __UNIQUE_ID_y__835 };
ret = unsafe { (*(*(*__slate_slot_6)).r#type).direct_access }.unwrap()(*__slate_slot_6, pgoff, nr_pages, mode, kaddr, pfn);
}
}
}
dm_put_live_table(md, *__slate_slot_3);
return ret;
}
}

#[unsafe(link_section = ".text.slate_distinct.dm.dm_dax_zero_page_range")]
extern "C" fn dm_dax_zero_page_range(dax_dev: *mut dax_device, pgoff: u64, nr_pages: u64) -> i32 {
let mut __ret_warn_on: i32 = unsafe { core::mem::zeroed() };
let mut srcu_idx: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut i32 = core::ptr::addr_of_mut!(srcu_idx) as *mut i32;
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut ti: core::mem::MaybeUninit<*mut dm_target> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut dm_target = core::ptr::addr_of_mut!(ti) as *mut *mut dm_target;
let mut md: *mut mapped_device = unsafe { core::mem::zeroed() };
unsafe {
md = dax_get_private(dax_dev) as *mut mapped_device;
let sector: u64 = pgoff.wrapping_mul(8u64);
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
ret = 5i32.wrapping_neg();
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_5 = dm_dax_get_live_target(md, sector, core::ptr::addr_of_mut!(*__slate_slot_3));
if !(*__slate_slot_5).is_null() {
__ret_warn_on = ((*(*(*__slate_slot_5)).r#type).dax_zero_page_range == None) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("837: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 837b - ., 3; .popsection\n# {0}", const 837 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_dm_dax_zero_page_range_8, sym __slate_asm_string_dm_dax_zero_page_range_9, const 1260 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("838: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 838b - ., 4; .popsection\n# {0}", const 838 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
let __v2: i64 = (__ret_warn_on != 0) as i64;
if __v2 != 0 {
// ->zero_page_range() is mandatory dax operation. If we are
//  here, something is wrong.
} else {
ret = unsafe { (*(*(*__slate_slot_5)).r#type).dax_zero_page_range }.unwrap()(*__slate_slot_5, pgoff, nr_pages);
}
}
dm_put_live_table(md, *__slate_slot_3);
return ret;
}
}

#[unsafe(link_section = ".text.slate_distinct.dm.dm_dax_recovery_write")]
extern "C" fn dm_dax_recovery_write(dax_dev: *mut dax_device, pgoff: u64, addr: *mut (), bytes: u64, i: *mut iov_iter) -> u64 {
let mut ret: i64 = unsafe { core::mem::zeroed() };
let mut srcu_idx: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut i32 = core::ptr::addr_of_mut!(srcu_idx) as *mut i32;
let mut ti: core::mem::MaybeUninit<*mut dm_target> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut dm_target = core::ptr::addr_of_mut!(ti) as *mut *mut dm_target;
let mut md: *mut mapped_device = unsafe { core::mem::zeroed() };
unsafe {
md = dax_get_private(dax_dev) as *mut mapped_device;
let sector: u64 = pgoff.wrapping_mul(8u64);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
ret = 0;
*__slate_slot_2 = dm_dax_get_live_target(md, sector, core::ptr::addr_of_mut!(*__slate_slot_1));
if !((*__slate_slot_2).is_null() || (*(*(*__slate_slot_2)).r#type).dax_recovery_write == None) {
ret = unsafe { (*(*(*__slate_slot_2)).r#type).dax_recovery_write }.unwrap()(*__slate_slot_2, pgoff, addr, bytes, i) as i64;
}
dm_put_live_table(md, *__slate_slot_1);
return ret as u64;
}
}

/// A target may call dm_accept_partial_bio only from the map routine.  It is
/// allowed for all bio types except REQ_PREFLUSH, REQ_OP_ZONE_* zone management
/// operations, zone append writes (native with REQ_OP_ZONE_APPEND or emulated
/// with write BIOs flagged with BIO_EMULATES_ZONE_APPEND) and any bio serviced
/// by __send_duplicate_bios().
///
/// dm_accept_partial_bio informs the dm that the target only wants to process
/// additional n_sectors sectors of the bio and the rest of the data should be
/// sent in a next bio.
///
/// A diagram that explains the arithmetics:
/// |         1          |       2       |   3   |
///
/// <-------------- *tio->len_ptr --------------->
///                      <----- bio_sectors ----->
///                      <-- n_sectors -->
///
/// Region 1 was already iterated over with bio_advance or similar function.
/// 	(it may be empty if the target doesn't use bio_advance)
/// Region 2 is the remaining bio size that the target wants to process.
/// 	(it may be empty if region 1 is non-empty, although there is no reason
/// 	 to make it empty)
/// The target requires that region 3 is to be sent in the next bio.
///
/// If the target wants to receive multiple copies of the bio (via num_*bios, etc),
/// the partially processed part (the sum of regions 1+2) must be the same for all
/// copies of the bio.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.dm.dm_accept_partial_bio")]
extern "C" fn dm_accept_partial_bio(bio: *mut bio, n_sectors: u32) {
let tio: *mut dm_target_io = clone_to_tio(bio);
let io: *mut dm_io = unsafe { (*tio).io };
let bio_sectors: u32 = bio_sectors!(bio);
loop {
if unlikely!(dm_tio_flagged(tio, 1u32)) != 0 {
loop {
unsafe {
core::arch::asm!("839: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 839b - ., 3; .popsection\n# {0}", const 839 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_dm_accept_partial_bio_0, sym __slate_asm_string_dm_accept_partial_bio_1, const 1330 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
loop {
if unlikely!(bio_sectors > unsafe { *(*tio).len_ptr }) != 0 {
loop {
unsafe {
core::arch::asm!("840: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 840b - ., 3; .popsection\n# {0}", const 840 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_dm_accept_partial_bio_5, sym __slate_asm_string_dm_accept_partial_bio_6, const 1331 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
loop {
if unlikely!(n_sectors > bio_sectors) != 0 {
loop {
unsafe {
core::arch::asm!("841: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 841b - ., 3; .popsection\n# {0}", const 841 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_dm_accept_partial_bio_10, sym __slate_asm_string_dm_accept_partial_bio_11, const 1332 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
loop {
if unlikely__3!((unsafe { (*bio).bi_opf }) & 134217728u32) != 0 {
loop {
unsafe {
core::arch::asm!("842: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 842b - ., 3; .popsection\n# {0}", const 842 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_dm_accept_partial_bio_15, sym __slate_asm_string_dm_accept_partial_bio_16, const 1333 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
let __v20: bool = static_branch_unlikely__3!(unsafe { core::ptr::addr_of_mut!(zoned_enabled) }) != 0 && unlikely!(bdev_is_zoned(unsafe { (*bio).bi_bdev })) != 0;
if __v20 {
let op: u32 = bio_op(bio as *const bio);
loop {
if unlikely!(op_is_zone_mgmt(op)) != 0 {
loop {
unsafe {
core::arch::asm!("843: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 843b - ., 3; .popsection\n# {0}", const 843 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_dm_accept_partial_bio_23, sym __slate_asm_string_dm_accept_partial_bio_24, const 1339 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
loop {
if unlikely!((op as u32) == 1) != 0 {
loop {
unsafe {
core::arch::asm!("844: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 844b - ., 3; .popsection\n# {0}", const 844 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_dm_accept_partial_bio_28, sym __slate_asm_string_dm_accept_partial_bio_29, const 1340 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
loop {
if unlikely!((op as u32) == 9) != 0 {
loop {
unsafe {
core::arch::asm!("845: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 845b - ., 3; .popsection\n# {0}", const 845 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_dm_accept_partial_bio_33, sym __slate_asm_string_dm_accept_partial_bio_34, const 1341 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
loop {
if unlikely!((op as u32) == 7) != 0 {
loop {
unsafe {
core::arch::asm!("846: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 846b - ., 3; .popsection\n# {0}", const 846 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_dm_accept_partial_bio_38, sym __slate_asm_string_dm_accept_partial_bio_39, const 1342 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
let __v43: *mut u32 = unsafe { (*tio).len_ptr };
let __v44: u32 = unsafe { *__v43 };
unsafe {
*__v43 = __v44.wrapping_sub(bio_sectors.wrapping_sub(n_sectors));
}
unsafe {
(*bio).bi_iter.bi_size = n_sectors << 9i32;
}
// __split_and_process_bio() may have already saved mapped part
// for accounting but it is being reduced so update accordingly.
dm_io_set_flag(io, 1u32);
unsafe {
(*io).sectors = n_sectors;
}
unsafe {
(*io).sector_offset = bio_sectors!(unsafe { (*io).orig_bio });
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_dm_accept_partial_bio_847: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut bio, u32)>, *mut ()>(Some(dm_accept_partial_bio)) };

/// @clone: clone bio that DM core passed to target's .map function
/// @tgt_clone: clone of @clone bio that target needs submitted
///
/// Targets should use this interface to submit bios they take
/// ownership of when returning DM_MAPIO_SUBMITTED.
///
/// Target should also enable ti->accounts_remapped_io
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.dm.dm_submit_bio_remap")]
extern "C" fn dm_submit_bio_remap(clone: *mut bio, mut tgt_clone: *mut bio) {
let tio: *mut dm_target_io = clone_to_tio(clone);
let io: *mut dm_io = unsafe { (*tio).io };
// establish bio that will get submitted
if tgt_clone.is_null() {
tgt_clone = clone;
}
unsafe { bio_clone_blkg_association(tgt_clone, (*io).orig_bio) };
// Account io->origin_bio to DM dev on behalf of target
// that took ownership of IO with DM_MAPIO_SUBMITTED.
dm_start_io_acct(io, clone);
trace_block_bio_remap(tgt_clone, disk_devt(unsafe { (*(*io).md).disk }), unsafe { (*tio).old_sector });
unsafe { submit_bio_noacct(tgt_clone) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_dm_submit_bio_remap_848: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut bio, *mut bio)>, *mut ()>(Some(dm_submit_bio_remap)) };

#[inline(never)]
fn __set_swap_bios_limit(md: *mut mapped_device, latch: i32) {
unsafe { mutex_lock(core::ptr::addr_of_mut!((*md).swap_bios_lock)) };
while latch < unsafe { (*md).swap_bios } {
cond_resched__13!();
unsafe { down(core::ptr::addr_of_mut!((*md).swap_bios_semaphore)) };
let __v2: i32 = unsafe { (*md).swap_bios };
unsafe {
(*md).swap_bios = __v2.wrapping_sub(1);
}
}
while latch > unsafe { (*md).swap_bios } {
cond_resched__7!();
unsafe { up(core::ptr::addr_of_mut!((*md).swap_bios_semaphore)) };
let __v7: i32 = unsafe { (*md).swap_bios };
unsafe {
(*md).swap_bios = __v7.wrapping_add(1);
}
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*md).swap_bios_lock)) };
}

fn __map_bio(clone: *mut bio) {
let tio: *mut dm_target_io = clone_to_tio(clone);
let ti: *mut dm_target = unsafe { (*tio).ti };
let io: *mut dm_io = unsafe { (*tio).io };
let md: *mut mapped_device = unsafe { (*io).md };
let mut r: i32 = 0;
unsafe {
(*clone).bi_end_io = Some(clone_endio);
}
// Map the clone.
unsafe {
(*tio).old_sector = (*clone).bi_iter.bi_sector;
}
let __v0: bool = static_branch_unlikely__7!(unsafe { core::ptr::addr_of_mut!(swap_bios_enabled) }) != 0 && unlikely!(swap_bios_limit(ti, clone)) != 0;
if __v0 {
let latch: i32 = get_swap_bios();
if unlikely!(latch != unsafe { (*md).swap_bios }) != 0 {
__set_swap_bios_limit(md, latch);
}
unsafe { down(core::ptr::addr_of_mut!((*md).swap_bios_semaphore)) };
}
if likely!((unsafe { (*(*ti).r#type).map }) == unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut dm_target, *mut bio) -> i32>>(linear_map as *const ()) }) != 0 {
r = unsafe { linear_map(ti, clone) };
} else {
if (unsafe { (*(*ti).r#type).map }) == unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut dm_target, *mut bio) -> i32>>(stripe_map as *const ()) } {
r = unsafe { stripe_map(ti, clone) };
} else {
r = unsafe { unsafe { (*(*ti).r#type).map }.unwrap()(ti, clone) };
}
}
match r {
0 => {
if ((unsafe { (*ti).__slate_bits_0.__get_accounts_remapped_io() }) as i32) == 0 {
dm_start_io_acct(io, clone);
}
// target has assumed ownership of this io
}
1 => {
dm_submit_bio_remap(clone, core::ptr::null_mut::<bio>());
}
2 | 4 => {
let __v4: bool = static_branch_unlikely__9!(unsafe { core::ptr::addr_of_mut!(swap_bios_enabled) }) != 0 && unlikely!(swap_bios_limit(ti, clone)) != 0;
if __v4 {
unsafe { up(core::ptr::addr_of_mut!((*md).swap_bios_semaphore)) };
}
free_tio(clone);
if r == 4 {
dm_io_dec_pending(io, 10u8);
} else {
dm_io_dec_pending(io, 11u8);
}
}
_ => {
let mut __v7: i32;
while false {
}
__v7 = unsafe { _printk(c"\x012device-mapper: core: unimplemented target map return value: %d\n".as_ptr() as *const u8, r) };
loop {
unsafe {
core::arch::asm!("849: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 849b - ., 3; .popsection\n# {0}", const 849 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string___map_bio_9, sym __slate_asm_string___map_bio_10, const 1459 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
}
}

fn setup_split_accounting(ci: *mut clone_info, len: u32) {
let io: *mut dm_io = unsafe { (*ci).io };
if (unsafe { (*ci).sector_count }) > len {
// Split needed, save the mapped part for accounting.
// NOTE: dm_accept_partial_bio() will update accordingly.
dm_io_set_flag(io, 1u32);
unsafe {
(*io).sectors = len;
}
unsafe {
(*io).sector_offset = bio_sectors!(unsafe { (*ci).bio });
}
}
}

fn alloc_multiple_bios(blist: &mut bio_list, ci: *mut clone_info, ti: *mut dm_target, num_bios: u32, len: *mut u32) {
let mut bio: *mut bio = unsafe { core::mem::zeroed() };
for r#try in 0i32..2i32 {
let mut bio_nr: i32 = 0;
if r#try != 0 && num_bios > 1 {
unsafe { mutex_lock(core::ptr::addr_of_mut!((*(*(*ci).io).md).table_devices_lock)) };
}
while (bio_nr as u32) < num_bios {
bio = alloc_tio(ci, ti, bio_nr as u32, len, if r#try != 0 { (BIT!(10i32) | BIT!(11i32)) as u32 } else { (BIT!(11i32) as u32) | (BIT!(13i32) as u32) });
if bio.is_null() {
break;
}
bio_list_add(blist as *mut bio_list, bio);
bio_nr = bio_nr.wrapping_add(1);
}
if r#try != 0 && num_bios > 1 {
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*(*(*ci).io).md).table_devices_lock)) };
}
if (bio_nr as u32) == num_bios {
return;
}
loop {
let __v3: *mut bio = bio_list_pop(blist as *mut bio_list);
bio = __v3;
if __v3.is_null() {
break;
}
free_tio(bio);
}
}
}

fn __send_duplicate_bios(ci: *mut clone_info, ti: *mut dm_target, num_bios: u32, len: *mut u32) -> u32 {
let mut blist: bio_list = bio_list { head: core::ptr::null_mut::<bio>(), tail: core::ptr::null_mut::<bio>() };
let mut ret: u32 = 0;
let mut __v0: i64;
let __ret_warn_on: i32 = (num_bios == 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("850: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 850b - ., 3; .popsection\n# {0}", const 850 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___send_duplicate_bios_1, sym __slate_asm_string___send_duplicate_bios_2, const 1515 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("851: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 851b - ., 4; .popsection\n# {0}", const 851 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
return 0;
}
// num_bios = 0 is a bug in caller
// dm_accept_partial_bio() is not supported with shared tio->len_ptr
if !len.is_null() {
setup_split_accounting(ci, unsafe { *len });
}
// Using alloc_multiple_bios(), even if num_bios is 1, to consistently
// support allocating using GFP_NOWAIT with GFP_NOIO fallback.
alloc_multiple_bios(unsafe { &mut (*core::ptr::addr_of_mut!(blist)) }, ci, ti, num_bios, len);
loop {
let __v7: *mut bio = bio_list_pop(core::ptr::addr_of_mut!(blist));
let clone: *mut bio = __v7;
if __v7.is_null() {
break;
}
if num_bios > 1 {
dm_tio_set_flag(clone_to_tio(clone), 1u32);
}
__map_bio(clone);
ret = ret.wrapping_add(1);
}
ret
}

fn __send_empty_flush(ci: *mut clone_info) {
let t: *mut dm_table = unsafe { (*ci).map };
let mut flush_bio: bio = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(flush_bio) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 120usize);
}
let mut opf: u32 = 264193u32;
if (unsafe { (*(*(*ci).io).orig_bio).bi_opf }) & 34816u32 == 34816u32 {
opf |= 32768u32;
}
// Use an on-stack bio for this, it's safe since we don't
// need to reference it after submit. It's just used as
// the basis for the clone(s).
unsafe { bio_init(core::ptr::addr_of_mut!(flush_bio), (*(*(*(*ci).io).md).disk).part0, core::ptr::null_mut::<bio_vec>(), 0u16, opf) };
unsafe {
(*ci).bio = core::ptr::addr_of_mut!(flush_bio);
}
unsafe {
(*ci).sector_count = 0;
}
unsafe {
(*(*ci).io).tio.clone.bi_iter.bi_size = 0;
}
if ((unsafe { (*t).__slate_bits_0.__get_flush_bypasses_map() }) as i32) == 0 {
let mut i: u32 = 0;
while i < unsafe { (*t).num_targets } {
let mut bios: u32 = 0;
let ti: *mut dm_target = dm_table_get_target(t, i);
if unlikely!((unsafe { (*ti).num_flush_bios }) == 0) == 0 {
atomic_add((unsafe { (*ti).num_flush_bios }) as i32, unsafe { core::ptr::addr_of_mut!((*(*ci).io).io_count) });
bios = __send_duplicate_bios(ci, ti, unsafe { (*ti).num_flush_bios }, core::ptr::null_mut::<u32>());
atomic_sub(unsafe { (*ti).num_flush_bios }.wrapping_sub(bios) as i32, unsafe { core::ptr::addr_of_mut!((*(*ci).io).io_count) });
}
i = i.wrapping_add(1);
}
} else {
// Note that there's no need to grab t->devices_lock here
// because the targets that support flush optimization don't
// modify the list of devices.
let devices: *mut list_head = unsafe { dm_table_get_devices(t) };
let mut len: u32 = 0;
let mut dd: *mut dm_dev_internal = unsafe { core::mem::zeroed() };
dd = list_first_entry__18!(devices);
while list_entry_is_head!(dd, devices, list) == 0 {
// Note that the structure dm_target_io is not
// associated with any target (because the device may be
// used by multiple targets), so we set tio->ti = NULL.
// We must check for NULL in the I/O processing path, to
// avoid NULL pointer dereference.
let clone: *mut bio = alloc_tio(ci, core::ptr::null_mut::<dm_target>(), 0u32, core::ptr::addr_of_mut!(len), (BIT!(10i32) | BIT!(11i32)) as u32);
atomic_add(1i32, unsafe { core::ptr::addr_of_mut!((*(*ci).io).io_count) });
bio_set_dev(clone, unsafe { (*(*dd).dm_dev).bdev });
unsafe {
(*clone).bi_end_io = Some(clone_endio);
}
dm_submit_bio_remap(clone, core::ptr::null_mut::<bio>());
dd = list_next_entry__25!(dd, list);
}
}
// alloc_io() takes one extra reference for submission, so the
// reference won't reach 0 without the following subtraction
atomic_sub(1i32, unsafe { core::ptr::addr_of_mut!((*(*ci).io).io_count) });
unsafe { bio_uninit((*ci).bio) };
}

fn __send_abnormal_io(ci: *mut clone_info, ti: *mut dm_target, num_bios: u32, max_granularity: u32, max_sectors: u32) {
let mut len: u32 = min_t__2!(u64, unsafe { (*ci).sector_count }, __max_io_len(ti, unsafe { (*ci).sector }, max_granularity, max_sectors)) as u32;
let mut bios: u32 = 0;
atomic_add(num_bios as i32, unsafe { core::ptr::addr_of_mut!((*(*ci).io).io_count) });
bios = __send_duplicate_bios(ci, ti, num_bios, core::ptr::addr_of_mut!(len));
// alloc_io() takes one extra reference for submission, so the
// reference won't reach 0 without the following (+1) subtraction
atomic_sub(num_bios.wrapping_sub(bios).wrapping_add(1) as i32, unsafe { core::ptr::addr_of_mut!((*(*ci).io).io_count) });
let __v4: u64 = unsafe { (*ci).sector };
unsafe {
(*ci).sector = __v4.wrapping_add(len as u64);
}
let __v7: u32 = unsafe { (*ci).sector_count };
unsafe {
(*ci).sector_count = __v7.wrapping_sub(len);
}
}

fn is_abnormal_io(bio: *mut bio) -> bool {
match bio_op(bio as *const bio) as u32 {
0..=2 => {
return false;
}
3 | 5 | 9 | 19 => {
return true;
}
_ => {
return false;
}
}
unsafe { core::mem::zeroed() }
}

fn __process_abnormal_io(ci: *mut clone_info, ti: *mut dm_target) -> u8 {
let mut num_bios: u32 = 0;
let mut max_granularity: u32 = 0;
let mut max_sectors: u32 = 0;
let limits: *mut queue_limits = dm_get_queue_limits(unsafe { (*(*ti).table).md });
match bio_op((unsafe { (*ci).bio }) as *const bio) as u32 {
3 => {
num_bios = unsafe { (*ti).num_discard_bios };
max_sectors = unsafe { (*limits).max_discard_sectors };
if ((unsafe { (*ti).__slate_bits_0.__get_max_discard_granularity() }) as i32) != 0 {
max_granularity = max_sectors;
}
}
5 => {
num_bios = unsafe { (*ti).num_secure_erase_bios };
max_sectors = unsafe { (*limits).max_secure_erase_sectors };
}
9 => {
num_bios = unsafe { (*ti).num_write_zeroes_bios };
max_sectors = unsafe { (*limits).max_write_zeroes_sectors };
}
_ => {
}
}
// Even though the device advertised support for this type of
// request, that does not mean every target supports it, and
// reconfiguration might also have changed that since the
// check was performed.
if unlikely!(num_bios == 0) != 0 {
return 1;
}
__send_abnormal_io(ci, ti, num_bios, max_granularity, max_sectors);
0
}

/// Reuse ->bi_private as dm_io list head for storing all dm_io instances
/// associated with this bio, and this bio's bi_private needs to be
/// stored in dm_io->data before the reuse.
///
/// bio->bi_private is owned by fs or upper layer, so block layer won't
/// touch it after splitting. Meantime it won't be changed by anyone after
/// bio is submitted. So this reuse is safe.
fn dm_poll_list_head(bio: *mut bio) -> *mut *mut dm_io {
(unsafe { core::ptr::addr_of_mut!((*bio).bi_private) }) as *mut *mut dm_io
}

fn dm_queue_poll_io(bio: *mut bio, io: *mut dm_io) {
let head: *mut *mut dm_io = dm_poll_list_head(bio);
if (unsafe { (*bio).bi_opf }) & 33554432u32 == 0 {
unsafe {
(*bio).bi_opf = (*bio).bi_opf | 33554432u32;
}
// Save .bi_private into dm_io, so that we can reuse
// .bi_private as dm_io list head for storing dm_io list
unsafe {
(*io).data = (*bio).bi_private;
}
// tell block layer to poll for completion
unsafe {
(*bio).__slate_anon_12.bi_cookie = !1u32.wrapping_neg();
}
unsafe {
(*io).next = core::ptr::null_mut::<dm_io>();
}
} else {
// bio recursed due to split, reuse original poll list,
// and save bio->bi_private too.
unsafe {
(*io).data = (*(*head)).data;
}
unsafe {
(*io).next = *head;
}
}
unsafe {
*head = io;
}
}

/// Select the correct strategy for processing a non-flush bio.
fn __split_and_process_bio(ci: *mut clone_info) -> u8 {
let mut clone: *mut bio = unsafe { core::mem::zeroed() };
let mut len: u32 = 0;
let ti: *mut dm_target = unsafe { dm_table_find_target((*ci).map, (*ci).sector) };
if unlikely!(ti.is_null()) != 0 {
return 10;
}
if unlikely__2!((unsafe { (*ci).__slate_bits_0.__get_is_abnormal_io() }) as i32) != 0 {
return __process_abnormal_io(ci, ti);
}
// Only support bio polling for normal IO, and the target io is
// exactly inside the dm_io instance (verified in dm_poll_dm_io)
unsafe {
let __t0 = (*(*ci).bio).bi_opf & 4194304u32 != 0;
(*ci).__slate_bits_0.__set_submit_as_polled(__t0);
}
len = min_t!(u64, max_io_len(ti, unsafe { (*ci).sector }), unsafe { (*ci).sector_count }) as u32;
if (unsafe { (*(*ci).bio).bi_opf }) & 134217728u32 != 0 {
if unlikely!(dm_target_supports_atomic_writes!(unsafe { (*ti).r#type }) == 0) != 0 {
return 10;
}
if unlikely!(len != unsafe { (*ci).sector_count }) != 0 {
return 10;
}
}
setup_split_accounting(ci, len);
if unlikely__3!((unsafe { (*(*ci).bio).bi_opf }) & 2097152u32) != 0 {
if unlikely!(dm_target_supports_nowait!(unsafe { (*ti).r#type }) == 0) != 0 {
return 1;
}
clone = alloc_tio(ci, ti, 0u32, core::ptr::addr_of_mut!(len), (BIT!(11i32) as u32) | (BIT!(13i32) as u32));
if unlikely!(clone.is_null()) != 0 {
return 12;
}
} else {
clone = alloc_tio(ci, ti, 0u32, core::ptr::addr_of_mut!(len), (BIT!(10i32) | BIT!(11i32)) as u32);
}
__map_bio(clone);
let __v4: u64 = unsafe { (*ci).sector };
unsafe {
(*ci).sector = __v4.wrapping_add(len as u64);
}
let __v7: u32 = unsafe { (*ci).sector_count };
unsafe {
(*ci).sector_count = __v7.wrapping_sub(len);
}
0
}

fn init_clone_info(ci: &mut clone_info, io: *mut dm_io, map: *mut dm_table, bio: *mut bio, is_abnormal: bool) {
unsafe {
(*(ci as *mut clone_info)).map = map;
}
unsafe {
(*(ci as *mut clone_info)).io = io;
}
unsafe {
(*(ci as *mut clone_info)).bio = bio;
}
unsafe {
(*(ci as *mut clone_info)).__slate_bits_0.__set_is_abnormal_io(is_abnormal);
}
unsafe {
let __t0 = false;
(*(ci as *mut clone_info)).__slate_bits_0.__set_submit_as_polled(__t0);
}
unsafe {
(*(ci as *mut clone_info)).sector = (*bio).bi_iter.bi_sector;
}
unsafe {
(*(ci as *mut clone_info)).sector_count = bio_sectors!(bio);
}
// Shouldn't happen but sector_count was being set to 0 so...
let __v0: bool = if static_branch_unlikely__14!(unsafe { core::ptr::addr_of_mut!(zoned_enabled) }) != 0 {
let mut __v3: i64;
let __ret_warn_on: i32 = (op_is_zone_mgmt(bio_op(bio as *const bio)) && (unsafe { (*(ci as *mut clone_info)).sector_count }) != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("856: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 856b - ., 3; .popsection\n# {0}", const 856 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_init_clone_info_4, sym __slate_asm_string_init_clone_info_5, const 1790 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("857: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 857b - ., 4; .popsection\n# {0}", const 857 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v3 = (__ret_warn_on != 0) as i64;
__v3 != 0
} else {
false
};
if __v0 {
unsafe {
(*(ci as *mut clone_info)).sector_count = 0;
}
}
}

fn dm_zone_bio_needs_split(bio: *mut bio) -> bool {
false
}

fn dm_zone_plug_bio(md: *mut mapped_device, bio: *mut bio) -> bool {
false
}

fn __send_zone_reset_all(ci: &clone_info) -> u8 {
1
}

/// Entry point to split a bio into clones and submit them to the targets.
fn dm_split_and_process_bio(md: *mut mapped_device, map: *mut dm_table, mut bio: *mut bio) {
let mut __v0: bool = unsafe { core::mem::zeroed() };
// Use the block layer zone write plugging for mapped devices that
// need zone append emulation (e.g. dm-crypt).
let mut __v2: bool = unsafe { core::mem::zeroed() };
let mut __v3: bool = unsafe { core::mem::zeroed() };
let mut need_split: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut bool = core::ptr::addr_of_mut!(need_split) as *mut bool;
let mut is_abnormal: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut bool = core::ptr::addr_of_mut!(is_abnormal) as *mut bool;
let mut error: u8 = unsafe { core::mem::zeroed() };
let mut io: core::mem::MaybeUninit<*mut dm_io> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut dm_io = core::ptr::addr_of_mut!(io) as *mut *mut dm_io;
let mut ci: core::mem::MaybeUninit<clone_info> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut clone_info = core::ptr::addr_of_mut!(ci) as *mut clone_info;
unsafe {
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 40usize);
let __slate_auto_bytes_1 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
error = 0;
let __slate_auto_bytes_2 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 1usize);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
*__slate_slot_5 = is_abnormal_io(bio);
if static_branch_unlikely__10!(core::ptr::addr_of_mut!(zoned_enabled)) != 0 {
if *__slate_slot_5 {
__v3 = true;
} else {
__v3 = dm_zone_bio_needs_split(bio);
}
*__slate_slot_4 = __v3;
} else {
*__slate_slot_4 = *__slate_slot_5;
}
if unlikely!(*__slate_slot_4) != 0 {
// Use bio_split_to_limits() for abnormal IO (e.g. discard, etc)
// otherwise associated queue_limits won't be imposed.
// Also split the BIO for mapped devices needing zone append
// emulation to ensure that the BIO does not cross zone
// boundaries.
bio = bio_split_to_limits(bio);
if bio.is_null() {
return;
}
}
if static_branch_unlikely__11!(core::ptr::addr_of_mut!(zoned_enabled)) != 0 {
__v2 = dm_zone_plug_bio(md, bio);
} else {
__v2 = false;
}
if __v2 {
return;
} else {
// Only support nowait for normal IO
if unlikely__3!((*bio).bi_opf & 2097152u32) != 0 && !(*__slate_slot_5) {
// Don't support NOWAIT for FLUSH because it may allocate
// multiple bios and there's no easy way how to undo the
// allocations.
if (*bio).bi_opf & 262144u32 != 0 {
bio_wouldblock_error(bio);
return;
} else {
*__slate_slot_7 = alloc_io(md, bio, (BIT!(11i32) as u32) | (BIT!(13i32) as u32));
if unlikely!((*__slate_slot_7).is_null()) != 0 {
// Unable to do anything without dm_io.
bio_wouldblock_error(bio);
return;
}
}
} else {
*__slate_slot_7 = alloc_io(md, bio, (BIT!(10i32) | BIT!(11i32)) as u32);
}
'__join_5: {
'__join_12: {
init_clone_info(&mut (*core::ptr::addr_of_mut!(*__slate_slot_8)), *__slate_slot_7, map, bio, *__slate_slot_5);
if unlikely!((*bio).bi_opf & 262144u32 != 0) != 0 {
// The "flush_bypasses_map" is set on targets where it is safe
// to skip the map function and submit bios directly to the
// underlying block devices - currently, it is set for dm-linear
// and dm-stripe.
//
// If we have just one underlying device (i.e. there is one
// linear target or multiple linear targets pointing to the same
// device), we can send the flush with data directly to it.
if (*bio).bi_iter.bi_size != 0 && ((*map).__slate_bits_0.__get_flush_bypasses_map() as i32) != 0 {
let devices: *mut list_head = dm_table_get_devices(map);
if (*devices).next == (*devices).prev {
break '__join_12;
}
}
if (*bio).bi_iter.bi_size != 0 {
(*(*__slate_slot_7)).requeue_flush_with_data = true;
}
__send_empty_flush(core::ptr::addr_of_mut!(*__slate_slot_8));
// dm_io_complete submits any data associated with flush
break '__join_5;
}
}
if static_branch_unlikely__8!(core::ptr::addr_of_mut!(zoned_enabled)) != 0 {
__v0 = (bio_op(bio as *const bio) as u32) == 19;
} else {
__v0 = false;
}
if __v0 {
error = __send_zone_reset_all(&(*core::ptr::addr_of_mut!(*__slate_slot_8)));
} else {
error = __split_and_process_bio(core::ptr::addr_of_mut!(*__slate_slot_8));
if !(error != 0 || (*__slate_slot_8).sector_count == 0) {
// Remainder must be passed to submit_bio_noacct() so it gets handled
// *after* bios already submitted have been completely processed.
bio_trim(bio, (*(*__slate_slot_7)).sectors as u64, (*__slate_slot_8).sector_count as u64);
trace_block_split(bio, (*bio).bi_iter.bi_sector as u32);
bio_inc_remaining(bio);
submit_bio_noacct(bio);
}
}
}
if error != 0 || ((*__slate_slot_8).__slate_bits_0.__get_submit_as_polled() as i32) == 0 {
// In case of submission failure, the extra reference for
// submitting io isn't consumed yet
if error != 0 {
atomic_dec(core::ptr::addr_of_mut!((*(*__slate_slot_7)).io_count));
}
dm_io_dec_pending(*__slate_slot_7, error);
} else {
dm_queue_poll_io(bio, *__slate_slot_7);
}
// Drop the extra reference count for non-POLLED bio, and hold one
// reference for POLLED bio, which will be released in dm_poll_bio
//
// Add every dm_io instance into the dm_io list head which is stored
// in bio->bi_private, so that dm_poll_bio can poll them all.
}
}
}

#[unsafe(link_section = ".text.slate_distinct.dm.dm_submit_bio")]
extern "C" fn dm_submit_bio(bio: *mut bio) {
let mut map: core::mem::MaybeUninit<*mut dm_table> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut dm_table = core::ptr::addr_of_mut!(map) as *mut *mut dm_table;
let mut srcu_idx: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut i32 = core::ptr::addr_of_mut!(srcu_idx) as *mut i32;
let mut md: *mut mapped_device = unsafe { core::mem::zeroed() };
unsafe {
md = (*(*(*bio).bi_bdev).bd_disk).private_data as *mut mapped_device;
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_1 = dm_get_live_table(md, core::ptr::addr_of_mut!(*__slate_slot_2));
if unlikely!((*__slate_slot_1).is_null()) != 0 {
if __ratelimit__23!(core::ptr::addr_of_mut!(_rs)) != 0 {
while false {
}
_printk(c"\x013device-mapper: core: %s: mapping table unavailable, erroring io\n".as_ptr() as *const u8, dm_device_name(md));
}
bio_io_error(bio);
} else {
// If suspended, queue this IO for later
if unlikely__2!(test_bit!(0i32, core::ptr::addr_of_mut!((*md).flags))) != 0 {
if (*bio).bi_opf & 2097152u32 != 0 {
bio_wouldblock_error(bio);
} else {
if (*bio).bi_opf & 524288u32 != 0 {
bio_io_error(bio);
} else {
queue_io(md, bio);
}
}
} else {
dm_split_and_process_bio(md, *__slate_slot_1, bio);
}
}
dm_put_live_table(md, *__slate_slot_2);
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

fn dm_poll_dm_io(io: *mut dm_io, iob: *mut io_comp_batch, flags: u32) -> bool {
let mut __v0: i64;
let __ret_warn_on: i32 = !(!(!dm_tio_is_normal(unsafe { core::ptr::addr_of_mut!((*io).tio) }))) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("858: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 858b - ., 3; .popsection\n# {0}", const 858 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_dm_poll_dm_io_1, sym __slate_asm_string_dm_poll_dm_io_2, const 2107 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("859: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 859b - ., 4; .popsection\n# {0}", const 859 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
// don't poll if the mapped io is done
if atomic_read((unsafe { core::ptr::addr_of_mut!((*io).io_count) }) as *const __SlateRecord_types_h_190) > 1 {
// DM hides the target queues from the upper poller, which may
// decide it is safe to spin on a single stacked queue.  Do not
// pass that spinning policy down to a target queue: one slow
// clone could keep the task inside dm_poll_bio() for a long
// time.  Poll target bios once and let the caller decide
// whether to keep polling, reap completions or reschedule.
unsafe { bio_poll(core::ptr::addr_of_mut!((*io).tio.clone), iob, flags | 1u32) };
}
// bio_poll holds the last reference
atomic_read((unsafe { core::ptr::addr_of_mut!((*io).io_count) }) as *const __SlateRecord_types_h_190) == 1
}

#[unsafe(link_section = ".text.slate_distinct.dm.dm_poll_bio")]
extern "C" fn dm_poll_bio(bio: *mut bio, iob: *mut io_comp_batch, flags: u32) -> i32 {
let head: *mut *mut dm_io = dm_poll_list_head(bio);
let list: *mut dm_io = unsafe { *head };
let mut tmp: *mut dm_io = core::ptr::null_mut::<dm_io>();
let mut curr: *mut dm_io = unsafe { core::mem::zeroed() };
let mut next: *mut dm_io = unsafe { core::mem::zeroed() };
// Only poll normal bio which was marked as REQ_DM_POLL_LIST
if (unsafe { (*bio).bi_opf }) & 33554432u32 == 0 {
return 0;
}
let mut __v0: i64;
let __ret_warn_on: i32 = list.is_null() as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("860: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 860b - ., 3; .popsection\n# {0}", const 860 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_dm_poll_bio_1, sym __slate_asm_string_dm_poll_bio_2, const 2138 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("861: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 861b - ., 4; .popsection\n# {0}", const 861 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
// Restore .bi_private before possibly completing dm_io.
//
// bio_poll() is only possible once @bio has been completely
// submitted via submit_bio_noacct()'s depth-first submission.
// So there is no dm_queue_poll_io() race associated with
// clearing REQ_DM_POLL_LIST here.
let __v7: u32 = (unsafe { (*bio).bi_opf }) & 4261412863u32;
unsafe {
(*bio).bi_opf = __v7;
}
unsafe {
(*bio).bi_private = (*list).data;
}
curr = list;
next = unsafe { (*curr).next };
while !curr.is_null() {
if dm_poll_dm_io(curr, iob, flags) {
// clone_endio() has already occurred, so no
// error handling is needed here.
__dm_io_dec_pending(curr);
} else {
unsafe {
(*curr).next = tmp;
}
tmp = curr;
}
curr = next;
next = if !curr.is_null() { unsafe { (*curr).next } } else { core::ptr::null_mut::<dm_io>() };
}
// Not done?
if !tmp.is_null() {
unsafe {
(*bio).bi_opf = (*bio).bi_opf | 33554432u32;
}
// Reset bio->bi_private to dm_io list head
unsafe {
*head = tmp;
}
return 0;
}
1
}

/// An IDR is used to keep track of allocated minor numbers.
fn free_minor(minor: i32) {
spin_lock(unsafe { core::ptr::addr_of_mut!(_minor_lock) });
unsafe { idr_remove(core::ptr::addr_of_mut!(_minor_idr), minor as u64) };
spin_unlock(unsafe { core::ptr::addr_of_mut!(_minor_lock) });
}

/// See if the device with a specific minor # is free.
fn specific_minor(minor: i32) -> i32 {
let mut r: i32 = 0;
if minor >= 1048576i32 {
return 22i32.wrapping_neg();
}
unsafe { idr_preload(((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)) };
spin_lock(unsafe { core::ptr::addr_of_mut!(_minor_lock) });
r = unsafe { idr_alloc(core::ptr::addr_of_mut!(_minor_idr), 1i32.wrapping_neg() as *mut (), minor, minor.wrapping_add(1), (BIT!(11i32) as u32) | (BIT!(13i32) as u32)) };
spin_unlock(unsafe { core::ptr::addr_of_mut!(_minor_lock) });
idr_preload_end();
if r < 0 {
return if r == 28i32.wrapping_neg() { 16i32.wrapping_neg() } else { r };
}
0
}

fn next_free_minor(minor: &mut i32) -> i32 {
let mut r: i32 = 0;
unsafe { idr_preload(((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)) };
spin_lock(unsafe { core::ptr::addr_of_mut!(_minor_lock) });
r = unsafe { idr_alloc(core::ptr::addr_of_mut!(_minor_idr), 1i32.wrapping_neg() as *mut (), 0i32, 1048576i32, (BIT!(11i32) as u32) | (BIT!(13i32) as u32)) };
spin_unlock(unsafe { core::ptr::addr_of_mut!(_minor_lock) });
idr_preload_end();
if r < 0 {
return r;
}
unsafe {
*(minor as *mut i32) = r;
}
0
}

fn dm_queue_destroy_crypto_profile(q: *mut request_queue) {
}

fn cleanup_mapped_device(md: *mut mapped_device) {
if !unsafe { (*md).wq }.is_null() {
unsafe { destroy_workqueue((*md).wq) };
}
dm_free_md_mempools(unsafe { (*md).mempools });
if !unsafe { (*md).dax_dev }.is_null() {
dax_remove_host(unsafe { (*md).disk });
kill_dax(unsafe { (*md).dax_dev });
put_dax(unsafe { (*md).dax_dev });
unsafe {
(*md).dax_dev = core::ptr::null_mut::<dax_device>();
}
}
if !unsafe { (*md).disk }.is_null() {
spin_lock(unsafe { core::ptr::addr_of_mut!(_minor_lock) });
unsafe {
(*(*md).disk).private_data = core::ptr::null_mut::<()>();
}
spin_unlock(unsafe { core::ptr::addr_of_mut!(_minor_lock) });
if (dm_get_md_type(md) as u32) != 0 {
let mut td: *mut table_device = unsafe { core::mem::zeroed() };
unsafe { dm_sysfs_exit(md) };
td = list_first_entry__21!(unsafe { core::ptr::addr_of_mut!((*md).table_devices) });
while list_entry_is_head!(td, unsafe { core::ptr::addr_of_mut!((*md).table_devices) }, list) == 0 {
unsafe { bd_unlink_disk_holder((*td).dm_dev.bdev, (*md).disk) };
td = list_next_entry__28!(td, list);
}
// Hold lock to make sure del_gendisk() won't concurrent
// with open/close_table_device().
unsafe { mutex_lock(core::ptr::addr_of_mut!((*md).table_devices_lock)) };
unsafe { del_gendisk((*md).disk) };
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*md).table_devices_lock)) };
}
dm_queue_destroy_crypto_profile(unsafe { (*md).queue });
unsafe { put_disk((*md).disk) };
}
if !unsafe { (*md).pending_io }.is_null() {
unsafe { free_percpu((*md).pending_io as *mut ()) };
unsafe {
(*md).pending_io = core::ptr::null_mut::<u64>();
}
}
unsafe { cleanup_srcu_struct(core::ptr::addr_of_mut!((*md).io_barrier)) };
mutex_destroy(unsafe { core::ptr::addr_of_mut!((*md).suspend_lock) });
mutex_destroy(unsafe { core::ptr::addr_of_mut!((*md).type_lock) });
mutex_destroy(unsafe { core::ptr::addr_of_mut!((*md).table_devices_lock) });
mutex_destroy(unsafe { core::ptr::addr_of_mut!((*md).swap_bios_lock) });
unsafe { dm_mq_cleanup_mapped_device(md) };
}

/// Allocate and initialise a blank device with a given minor.
fn alloc_dev(mut minor: i32) -> *mut mapped_device {
let mut __slate_compound_24: __SlateRecord_types_h_197 = unsafe { core::mem::zeroed() };
let mut __slate_compound_25: __SlateRecord_types_h_197 = unsafe { core::mem::zeroed() };
let mut __slate_compound_26: spinlock = unsafe { core::mem::zeroed() };
let mut __slate_compound_27: spinlock = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut workqueue_struct> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut workqueue_struct = core::ptr::addr_of_mut!(_res) as *mut *mut workqueue_struct;
let mut old_md: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut () = core::ptr::addr_of_mut!(old_md) as *mut *mut ();
let mut md: core::mem::MaybeUninit<*mut mapped_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut mapped_device = core::ptr::addr_of_mut!(md) as *mut *mut mapped_device;
let mut dax_dev: core::mem::MaybeUninit<*mut dax_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut dax_device = core::ptr::addr_of_mut!(dax_dev) as *mut *mut dax_device;
let mut numa_node_id: i32 = unsafe { core::mem::zeroed() };
let mut r: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut i32 = core::ptr::addr_of_mut!(r) as *mut i32;
unsafe {
let __slate_auto_bytes_2 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
numa_node_id = dm_get_numa_node() as i32;
let __slate_auto_bytes_3 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
*__slate_slot_11 = kvzalloc_node!(core::mem::size_of::<mapped_device>() as u64, 3264u32, numa_node_id) as *mut mapped_device;
if (*__slate_slot_11).is_null() {
while false {
}
_printk(c"\x013device-mapper: core: unable to allocate device, out of memory.\n".as_ptr() as *const u8);
return core::ptr::null_mut::<mapped_device>();
} else {
if try_module_get(core::ptr::null_mut::<module>()) {
// get a minor number for the dev
if minor == 1i32.wrapping_neg() {
*__slate_slot_14 = next_free_minor(&mut (*core::ptr::addr_of_mut!(minor)));
} else {
*__slate_slot_14 = specific_minor(minor);
}
if *__slate_slot_14 >= 0 {
let __v8: i32 = __init_srcu_struct(core::ptr::addr_of_mut!((*(*__slate_slot_11)).io_barrier), c"&md->io_barrier".as_ptr() as *const u8, core::ptr::addr_of_mut!(__srcu_key));
*__slate_slot_14 = __v8;
if *__slate_slot_14 >= 0 {
(*(*__slate_slot_11)).numa_node_id = numa_node_id;
let __t4 = false;
(*(*__slate_slot_11)).__slate_bits_0.__set_init_tio_pdu(__t4);
(*(*__slate_slot_11)).r#type = 0;
loop {
__mutex_init(core::ptr::addr_of_mut!((*(*__slate_slot_11)).suspend_lock), c"&md->suspend_lock".as_ptr() as *const u8, core::ptr::addr_of_mut!(__key__2332));
if true {
break;
}
}
loop {
__mutex_init(core::ptr::addr_of_mut!((*(*__slate_slot_11)).type_lock), c"&md->type_lock".as_ptr() as *const u8, core::ptr::addr_of_mut!(__key__2333));
if true {
break;
}
}
loop {
__mutex_init(core::ptr::addr_of_mut!((*(*__slate_slot_11)).table_devices_lock), c"&md->table_devices_lock".as_ptr() as *const u8, core::ptr::addr_of_mut!(__key__2334));
if true {
break;
}
}
loop {
spinlock_check(core::ptr::addr_of_mut!((*(*__slate_slot_11)).deferred_lock));
*core::ptr::addr_of_mut!((*(*__slate_slot_11)).deferred_lock) = *{
__slate_compound_27 = spinlock { __slate_anon_0: {
let mut __t2: __SlateRecord_spinlock_types_h_18 = core::mem::zeroed();
__t2.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t3: __SlateRecord_qspinlock_types_h_44 = core::mem::zeroed();
__t3.val = __SlateRecord_types_h_190 { counter: 0 };
__t3
} } };
__t2
} };
core::ptr::addr_of_mut!(__slate_compound_27)
};
if true {
break;
}
}
atomic_set(core::ptr::addr_of_mut!((*(*__slate_slot_11)).holders), 1i32);
atomic_set(core::ptr::addr_of_mut!((*(*__slate_slot_11)).open_count), 0i32);
atomic_set(core::ptr::addr_of_mut!((*(*__slate_slot_11)).event_nr), 0i32);
atomic_set(core::ptr::addr_of_mut!((*(*__slate_slot_11)).uevent_seq), 0i32);
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*(*__slate_slot_11)).uevent_list));
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*(*__slate_slot_11)).table_devices));
loop {
spinlock_check(core::ptr::addr_of_mut!((*(*__slate_slot_11)).uevent_lock));
*core::ptr::addr_of_mut!((*(*__slate_slot_11)).uevent_lock) = *{
__slate_compound_26 = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = core::mem::zeroed();
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = core::mem::zeroed();
__t1.val = __SlateRecord_types_h_190 { counter: 0 };
__t1
} } };
__t0
} };
core::ptr::addr_of_mut!(__slate_compound_26)
};
if true {
break;
}
}
'__join_3: {
// default to bio-based until DM table is loaded and md->type
// established. If request-based table is loaded: blk-mq will
// override accordingly.
let __v7: *mut gendisk = __blk_alloc_disk(core::ptr::null_mut::<queue_limits>(), (*(*__slate_slot_11)).numa_node_id, core::ptr::addr_of_mut!(__key__2349));
(*(*__slate_slot_11)).disk = __v7;
if IS_ERR((*(*__slate_slot_11)).disk as *const ()) {
(*(*__slate_slot_11)).disk = core::ptr::null_mut::<gendisk>();
} else {
(*(*__slate_slot_11)).queue = (*(*(*__slate_slot_11)).disk).queue;
loop {
__init_waitqueue_head(core::ptr::addr_of_mut!((*(*__slate_slot_11)).wait), c"&md->wait".as_ptr() as *const u8, core::ptr::addr_of_mut!(__key__2356));
if true {
break;
}
}
loop {
__init_work(core::ptr::addr_of_mut!((*(*__slate_slot_11)).work), 0i32);
(*core::ptr::addr_of_mut!((*(*__slate_slot_11)).work)).data = *{
__slate_compound_25 = __SlateRecord_types_h_197 { counter: 4503599625273344i64 };
core::ptr::addr_of_mut!(__slate_compound_25)
};
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*(*__slate_slot_11)).work)).entry));
(*core::ptr::addr_of_mut!((*(*__slate_slot_11)).work)).func = Some(dm_wq_work);
if true {
if true {
break;
}
}
}
loop {
__init_work(core::ptr::addr_of_mut!((*(*__slate_slot_11)).requeue_work), 0i32);
(*core::ptr::addr_of_mut!((*(*__slate_slot_11)).requeue_work)).data = *{
__slate_compound_24 = __SlateRecord_types_h_197 { counter: 4503599625273344i64 };
core::ptr::addr_of_mut!(__slate_compound_24)
};
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*(*__slate_slot_11)).requeue_work)).entry));
(*core::ptr::addr_of_mut!((*(*__slate_slot_11)).requeue_work)).func = Some(dm_wq_requeue_work);
if true {
if true {
break;
}
}
}
loop {
__init_waitqueue_head(core::ptr::addr_of_mut!((*(*__slate_slot_11)).eventq), c"&md->eventq".as_ptr() as *const u8, core::ptr::addr_of_mut!(__key__2359));
if true {
break;
}
}
init_completion(core::ptr::addr_of_mut!((*(*__slate_slot_11)).kobj_holder.completion));
(*(*__slate_slot_11)).requeue_list = core::ptr::null_mut::<dm_io>();
(*(*__slate_slot_11)).swap_bios = get_swap_bios();
sema_init(core::ptr::addr_of_mut!((*(*__slate_slot_11)).swap_bios_semaphore), (*(*__slate_slot_11)).swap_bios);
loop {
__mutex_init(core::ptr::addr_of_mut!((*(*__slate_slot_11)).swap_bios_lock), c"&md->swap_bios_lock".as_ptr() as *const u8, core::ptr::addr_of_mut!(__key__2365));
if true {
break;
}
}
(*(*(*__slate_slot_11)).disk).major = _major as i32;
(*(*(*__slate_slot_11)).disk).first_minor = minor;
(*(*(*__slate_slot_11)).disk).minors = 1;
let __v6: *mut gendisk = (*(*__slate_slot_11)).disk;
(*__v6).flags = (*__v6).flags | 4;
(*(*(*__slate_slot_11)).disk).fops = core::ptr::addr_of!(dm_blk_dops.0);
(*(*(*__slate_slot_11)).disk).private_data = *__slate_slot_11 as *mut ();
sprintf((*(*(*__slate_slot_11)).disk).disk_name.as_mut_ptr() as *mut u8, c"dm-%d".as_ptr() as *const u8, minor);
*__slate_slot_12 = alloc_dax(*__slate_slot_11 as *mut (), core::ptr::addr_of!(dm_dax_ops.0));
if IS_ERR(*__slate_slot_12 as *const ()) {
if PTR_ERR(*__slate_slot_12 as *const ()) != (95i32.wrapping_neg() as i64) {
break '__join_3;
}
} else {
set_dax_nocache(*__slate_slot_12);
set_dax_nomc(*__slate_slot_12);
(*(*__slate_slot_11)).dax_dev = *__slate_slot_12;
if dax_add_host(*__slate_slot_12, (*(*__slate_slot_11)).disk) != 0 {
break '__join_3;
}
}
format_dev_t!((*(*__slate_slot_11)).name.as_mut_ptr() as *mut u8, _major << 20i32 | (minor as u32));
let __slate_auto_bytes_1 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_0 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_1 = alloc_workqueue_noprof(c"kdmflush/%s".as_ptr() as *const u8, 264u32, 0i32, (*(*__slate_slot_11)).name.as_mut_ptr() as *mut u8);
alloc_tag_restore!();
} else {
*__slate_slot_1 = alloc_workqueue_noprof(c"kdmflush/%s".as_ptr() as *const u8, 264u32, 0i32, (*(*__slate_slot_11)).name.as_mut_ptr() as *mut u8);
}
(*(*__slate_slot_11)).wq = *__slate_slot_1;
if !unsafe { (*(*__slate_slot_11)).wq }.is_null() {
(*(*__slate_slot_11)).pending_io = alloc_percpu!(u64);
if !unsafe { (*(*__slate_slot_11)).pending_io }.is_null() {
*__slate_slot_14 = dm_stats_init(core::ptr::addr_of_mut!((*(*__slate_slot_11)).stats));
if *__slate_slot_14 >= 0 {
// Populate the mapping, nobody knows we exist yet
spin_lock(core::ptr::addr_of_mut!(_minor_lock));
*__slate_slot_10 = idr_replace(core::ptr::addr_of_mut!(_minor_idr), *__slate_slot_11 as *mut (), minor as u64);
spin_unlock(core::ptr::addr_of_mut!(_minor_lock));
'__join_7: {
loop {
if unlikely!(*__slate_slot_10 != (1i32.wrapping_neg() as *mut ())) != 0 {
break '__join_7;
} else {
if true {
break;
}
}
}
return *__slate_slot_11;
}
core::arch::asm!("862: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 862b - ., 3; .popsection\n# {0}", const 862 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_alloc_dev_15, sym __slate_asm_string_alloc_dev_16, const 2407 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
}
}
}
}
cleanup_mapped_device(*__slate_slot_11);
}
free_minor(minor);
}
module_put(core::ptr::null_mut::<module>());
}
kvfree(*__slate_slot_11 as *const ());
return core::ptr::null_mut::<mapped_device>();
}
}
}

static mut __srcu_key: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__2332: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__2333: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__2334: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__2349: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__2356: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__2357: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__2358: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__2359: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__2365: lock_class_key = unsafe { core::mem::zeroed() };

fn free_dev(md: *mut mapped_device) {
let minor: i32 = MINOR!(disk_devt(unsafe { (*md).disk })) as i32;
unlock_fs(md);
cleanup_mapped_device(md);
let mut __v0: i64;
let __ret_warn_on: i32 = (list_empty((unsafe { core::ptr::addr_of_mut!((*md).table_devices) }) as *const list_head) == 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("863: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 863b - ., 3; .popsection\n# {0}", const 863 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_free_dev_1, sym __slate_asm_string_free_dev_2, const 2432 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("864: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 864b - ., 4; .popsection\n# {0}", const 864 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
unsafe { dm_stats_cleanup(core::ptr::addr_of_mut!((*md).stats)) };
free_minor(minor);
unsafe { module_put(core::ptr::null_mut::<module>()) };
unsafe { kvfree(md as *const ()) };
}

/// Bind a table to the device.
#[unsafe(link_section = ".text.slate_distinct.dm.event_callback")]
extern "C" fn event_callback(context: *mut ()) {
let mut flags: u64 = 0;
let mut uevents: list_head = unsafe { core::mem::zeroed() };
uevents = list_head { next: core::ptr::addr_of_mut!(uevents), prev: core::ptr::addr_of_mut!(uevents) };
let md: *mut mapped_device = context as *mut mapped_device;
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*md).uevent_lock) }, flags);
list_splice_init(unsafe { core::ptr::addr_of_mut!((*md).uevent_list) }, core::ptr::addr_of_mut!(uevents));
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*md).uevent_lock) }, flags);
dm_send_uevents(unsafe { &(*core::ptr::addr_of_mut!(uevents)) }, unsafe { core::ptr::addr_of_mut!((*disk_to_dev!(unsafe { (*md).disk })).kobj) });
atomic_inc(unsafe { core::ptr::addr_of_mut!((*md).event_nr) });
wake_up!(unsafe { core::ptr::addr_of_mut!((*md).eventq) });
dm_issue_global_event();
}

// Returns old map, which caller must destroy.
fn __bind(md: *mut mapped_device, t: *mut dm_table, limits: *mut queue_limits) -> *mut dm_table {
let mut _r_a_p__v: u64 = unsafe { core::mem::zeroed() };
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut old_size: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut u64 = core::ptr::addr_of_mut!(old_size) as *mut u64;
let mut size: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut u64 = core::ptr::addr_of_mut!(size) as *mut u64;
let mut old_map: core::mem::MaybeUninit<*mut dm_table> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut dm_table = core::ptr::addr_of_mut!(old_map) as *mut *mut dm_table;
unsafe {
'__join_0: {
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
lockdep_assert_held!(core::ptr::addr_of_mut!((*md).suspend_lock));
*__slate_slot_3 = dm_table_get_size(t);
*__slate_slot_2 = dm_get_size(md);
if !dm_table_supports_size_change(t, *__slate_slot_2, *__slate_slot_3) {
*__slate_slot_4 = ERR_PTR(22i32.wrapping_neg() as i64) as *mut dm_table;
} else {
set_capacity((*md).disk, *__slate_slot_3);
if !limits.is_null() {
ret = dm_table_set_restrictions(t, (*md).queue, limits);
if ret != 0 {
set_capacity((*md).disk, *__slate_slot_2);
*__slate_slot_4 = ERR_PTR(ret as i64) as *mut dm_table;
break '__join_0;
}
}
// Wipe any geometry if the size of the table changed.
if *__slate_slot_3 != *__slate_slot_2 {
memset(core::ptr::addr_of_mut!((*md).geometry) as *mut (), 0i32, core::mem::size_of::<hd_geometry>() as u64);
}
dm_table_event_callback(t, Some(event_callback), md as *mut ());
if dm_table_request_based(t) {
// Leverage the fact that request-based DM targets are
// immutable singletons - used to optimize dm_mq_queue_rq.
(*md).immutable_target = dm_table_get_immutable_target(t);
// There is no need to reload with request-based dm because the
// size of front_pad doesn't change.
//
// Note for future: If you are to reload bioset, prep-ed
// requests in the queue may refer to bio from the old bioset,
// so you must walk through the queue to unprep.
if unsafe { (*md).mempools }.is_null() {
(*md).mempools = (*t).mempools;
} else {
dm_free_md_mempools((*t).mempools);
}
} else {
// The md may already have mempools that need changing.
// If so, reload bioset because front_pad may have changed
// because a different table was loaded.
dm_free_md_mempools((*md).mempools);
(*md).mempools = (*t).mempools;
}
(*t).mempools = core::ptr::null_mut::<dm_md_mempools>();
*__slate_slot_4 = rcu_dereference_protected!(unsafe { (*md).map }, lockdep_is_held(core::ptr::addr_of_mut!((*md).suspend_lock) as *const ())) as *mut dm_table;
_r_a_p__v = (t as *mut ()) as u64;
loop {
kcsan_release!();
loop {
if true {
core::arch::asm!("", options(nostack, att_syntax));
loop {
if true {
__WRITE_ONCE__18!(*core::ptr::addr_of_mut!((*md).map), RCU_INITIALIZER!(_r_a_p__v as *mut ()));
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
if true {
break;
}
}
(*md).immutable_target_type = dm_table_get_immutable_target_type(t);
if !(*__slate_slot_4).is_null() {
dm_sync_table(md);
}
}
}
return *__slate_slot_4;
}
}

// Returns unbound table for the caller to free.
fn __unbind(md: *mut mapped_device) -> *mut dm_table {
let map: *mut dm_table = rcu_dereference_protected__4!(unsafe { (*md).map }, 1i32) as *mut dm_table;
if map.is_null() {
return core::ptr::null_mut::<dm_table>();
}
unsafe { dm_table_event_callback(map, None, core::ptr::null_mut::<()>()) };
loop {
loop {
if !(__native_word__7!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_868() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__18!((*md).map, RCU_INITIALIZER!(core::ptr::null_mut::<()>()));
if true {
break;
}
}
dm_sync_table(md);
map
}

/// Constructor for a new device.
#[unsafe(no_mangle)]
extern "C" fn dm_create(minor: i32, result: *mut *mut mapped_device) -> i32 {
let md: *mut mapped_device = alloc_dev(minor);
if md.is_null() {
return 6i32.wrapping_neg();
}
dm_ima_init(md);
unsafe {
*result = md;
}
0
}

/// Functions to manage md->type.
/// All are required to hold md->type_lock.
#[unsafe(no_mangle)]
extern "C" fn dm_lock_md_type(md: *mut mapped_device) {
unsafe { mutex_lock(core::ptr::addr_of_mut!((*md).type_lock)) };
}

#[unsafe(no_mangle)]
extern "C" fn dm_unlock_md_type(md: *mut mapped_device) {
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*md).type_lock)) };
}

#[unsafe(no_mangle)]
extern "C" fn dm_get_md_type(md: *mut mapped_device) -> u32 {
unsafe { (*md).r#type }
}

#[unsafe(no_mangle)]
extern "C" fn dm_get_immutable_target_type(md: *mut mapped_device) -> *mut target_type {
unsafe { (*md).immutable_target_type }
}

/// Setup the DM device's queue based on md's type
#[unsafe(no_mangle)]
extern "C" fn dm_setup_md_queue(md: *mut mapped_device, t: *mut dm_table) -> i32 {
let mut r: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i32 = core::ptr::addr_of_mut!(r) as *mut i32;
let mut td: core::mem::MaybeUninit<*mut table_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut table_device = core::ptr::addr_of_mut!(td) as *mut *mut table_device;
let mut limits: core::mem::MaybeUninit<queue_limits> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut queue_limits = core::ptr::addr_of_mut!(limits) as *mut queue_limits;
let mut r#type: u32 = unsafe { core::mem::zeroed() };
unsafe {
r#type = dm_table_get_type(t);
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 192usize);
let __slate_auto_bytes_1 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __ret_warn_on: i32 = ((r#type as u32) == 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("869: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 869b - ., 3; .popsection\n# {0}", const 869 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_dm_setup_md_queue_13, sym __slate_asm_string_dm_setup_md_queue_14, const 2607 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("870: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 870b - ., 4; .popsection\n# {0}", const 870 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
if (r#type as u32) == 2 {
(*(*md).disk).fops = core::ptr::addr_of!(dm_rq_blk_dops.0);
*__slate_slot_5 = dm_mq_init_request_queue(md, t);
if *__slate_slot_5 != 0 {
while false {
}
_printk(c"\x013device-mapper: core: Cannot initialize queue for request-based dm mapped device\n".as_ptr() as *const u8);
return *__slate_slot_5;
}
}
*__slate_slot_5 = dm_calculate_queue_limits(t, core::ptr::addr_of_mut!(*__slate_slot_7));
if *__slate_slot_5 != 0 {
while false {
}
_printk(c"\x013device-mapper: core: Cannot calculate initial queue limits\n".as_ptr() as *const u8);
return *__slate_slot_5;
} else {
*__slate_slot_5 = dm_table_set_restrictions(t, (*md).queue, core::ptr::addr_of_mut!(*__slate_slot_7));
if *__slate_slot_5 != 0 {
return *__slate_slot_5;
} else {
// Hold lock to make sure add_disk() and del_gendisk() won't concurrent
// with open_table_device() and close_table_device().
mutex_lock(core::ptr::addr_of_mut!((*md).table_devices_lock));
*__slate_slot_5 = add_disk((*md).disk);
if *__slate_slot_5 != 0 {
mutex_unlock(core::ptr::addr_of_mut!((*md).table_devices_lock));
return *__slate_slot_5;
} else {
// Register the holder relationship for devices added before the disk
// was live.
*__slate_slot_6 = list_first_entry__21!(core::ptr::addr_of_mut!((*md).table_devices));
'__join_3: {
while list_entry_is_head!(*__slate_slot_6, core::ptr::addr_of_mut!((*md).table_devices), list) == 0 {
*__slate_slot_5 = bd_link_disk_holder((*(*__slate_slot_6)).dm_dev.bdev, (*md).disk);
if *__slate_slot_5 != 0 {
break '__join_3;
} else {
*__slate_slot_6 = list_next_entry__28!(*__slate_slot_6, list);
}
}
mutex_unlock(core::ptr::addr_of_mut!((*md).table_devices_lock));
*__slate_slot_5 = dm_sysfs_init(md);
if *__slate_slot_5 != 0 {
mutex_lock(core::ptr::addr_of_mut!((*md).table_devices_lock));
} else {
(*md).r#type = r#type;
return 0;
}
}
*__slate_slot_6 = list_prev_entry!(*__slate_slot_6, list);
while list_entry_is_head!(*__slate_slot_6, core::ptr::addr_of_mut!((*md).table_devices), list) == 0 {
bd_unlink_disk_holder((*(*__slate_slot_6)).dm_dev.bdev, (*md).disk);
*__slate_slot_6 = list_prev_entry!(*__slate_slot_6, list);
}
del_gendisk((*md).disk);
mutex_unlock(core::ptr::addr_of_mut!((*md).table_devices_lock));
return *__slate_slot_5;
}
}
}
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.dm.dm_get_md")]
extern "C" fn dm_get_md(dev: u32) -> *mut mapped_device {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut __v2: bool = unsafe { core::mem::zeroed() };
let mut minor: u32 = unsafe { core::mem::zeroed() };
let mut md: core::mem::MaybeUninit<*mut mapped_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut mapped_device = core::ptr::addr_of_mut!(md) as *mut *mut mapped_device;
unsafe {
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
minor = MINOR!(dev);
if MAJOR!(dev) != _major || minor >= 1048576u32 {
return core::ptr::null_mut::<mapped_device>();
} else {
spin_lock(core::ptr::addr_of_mut!(_minor_lock));
*__slate_slot_4 = idr_find(core::ptr::addr_of_mut!(_minor_idr) as *const idr, minor as u64) as *mut mapped_device;
if (*__slate_slot_4).is_null() || *__slate_slot_4 == ((1i32.wrapping_neg() as *mut ()) as *mut mapped_device) {
__v2 = true;
} else {
__v2 = MINOR!(disk_devt(dm_disk(*__slate_slot_4))) != minor;
}
if __v2 {
__v1 = true;
} else {
__v1 = test_bit!(3i32, core::ptr::addr_of_mut!((*(*__slate_slot_4)).flags)) != 0;
}
if __v1 {
__v0 = true;
} else {
__v0 = dm_deleting_md(*__slate_slot_4) != 0;
}
if __v0 {
*__slate_slot_4 = core::ptr::null_mut::<mapped_device>();
} else {
dm_get(*__slate_slot_4);
}
spin_unlock(core::ptr::addr_of_mut!(_minor_lock));
return *__slate_slot_4;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_dm_get_md_871: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u32) -> *mut mapped_device>, *mut ()>(Some(dm_get_md)) };

#[unsafe(no_mangle)]
extern "C" fn dm_get_mdptr(md: *mut mapped_device) -> *mut () {
unsafe { (*md).interface_ptr }
}

#[unsafe(no_mangle)]
extern "C" fn dm_set_mdptr(md: *mut mapped_device, ptr: *mut ()) {
unsafe {
(*md).interface_ptr = ptr;
}
}

#[unsafe(no_mangle)]
extern "C" fn dm_get(md: *mut mapped_device) {
atomic_inc(unsafe { core::ptr::addr_of_mut!((*md).holders) });
loop {
if unlikely__2!(test_bit!(3i32, unsafe { core::ptr::addr_of_mut!((*md).flags) })) != 0 {
loop {
unsafe {
core::arch::asm!("872: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 872b - ., 3; .popsection\n# {0}", const 872 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_dm_get_1, sym __slate_asm_string_dm_get_2, const 2704 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.dm.dm_hold")]
extern "C" fn dm_hold(md: *mut mapped_device) -> i32 {
spin_lock(unsafe { core::ptr::addr_of_mut!(_minor_lock) });
if test_bit!(3i32, unsafe { core::ptr::addr_of_mut!((*md).flags) }) != 0 {
spin_unlock(unsafe { core::ptr::addr_of_mut!(_minor_lock) });
return 16i32.wrapping_neg();
}
dm_get(md);
spin_unlock(unsafe { core::ptr::addr_of_mut!(_minor_lock) });
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_dm_hold_873: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut mapped_device) -> i32>, *mut ()>(Some(dm_hold)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.dm.dm_device_name")]
extern "C" fn dm_device_name(md: *mut mapped_device) -> *const u8 {
(unsafe { (*md).name.as_mut_ptr() as *mut u8 }) as *const u8
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_dm_device_name_874: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut mapped_device) -> *const u8>, *mut ()>(Some(dm_device_name)) };

fn __dm_destroy(md: *mut mapped_device, wait: bool) {
let mut srcu_idx: i32 = 0;
might_sleep!();
spin_lock(unsafe { core::ptr::addr_of_mut!(_minor_lock) });
unsafe { idr_replace(core::ptr::addr_of_mut!(_minor_idr), 1i32.wrapping_neg() as *mut (), MINOR!(disk_devt(dm_disk(md))) as u64) };
set_bit(3i64, (unsafe { core::ptr::addr_of_mut!((*md).flags) }) as *mut u64);
spin_unlock(unsafe { core::ptr::addr_of_mut!(_minor_lock) });
unsafe { blk_mark_disk_dead((*md).disk) };
// Take suspend_lock so that presuspend and postsuspend methods
// do not race with internal suspend.
unsafe { mutex_lock(core::ptr::addr_of_mut!((*md).suspend_lock)) };
let map: *mut dm_table = dm_get_live_table(md, core::ptr::addr_of_mut!(srcu_idx));
if dm_suspended_md(md) == 0 {
unsafe { dm_table_presuspend_targets(map) };
set_bit(1i64, (unsafe { core::ptr::addr_of_mut!((*md).flags) }) as *mut u64);
set_bit(8i64, (unsafe { core::ptr::addr_of_mut!((*md).flags) }) as *mut u64);
unsafe { dm_table_postsuspend_targets(map) };
}
// dm_put_live_table must be before fsleep, otherwise deadlock is possible
dm_put_live_table(md, srcu_idx);
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*md).suspend_lock)) };
// Rare, but there may be I/O requests still going to complete,
// for example.  Wait for all references to disappear.
// No one should increment the reference count of the mapped_device,
// after the mapped_device state becomes DMF_FREEING.
if wait {
while atomic_read((unsafe { core::ptr::addr_of_mut!((*md).holders) }) as *const __SlateRecord_types_h_190) != 0 {
fsleep(1000u64);
}
} else {
if atomic_read((unsafe { core::ptr::addr_of_mut!((*md).holders) }) as *const __SlateRecord_types_h_190) != 0 {
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x014device-mapper: core: %s: Forcibly removing mapped_device still in use! (%d users)\n".as_ptr() as *const u8, dm_device_name(md), atomic_read(core::ptr::addr_of_mut!((*md).holders) as *const __SlateRecord_types_h_190)) };
}
}
unsafe { dm_table_destroy(__unbind(md)) };
free_dev(md);
}

#[unsafe(no_mangle)]
extern "C" fn dm_destroy(md: *mut mapped_device) {
__dm_destroy(md, true);
}

#[unsafe(no_mangle)]
extern "C" fn dm_destroy_immediate(md: *mut mapped_device) {
__dm_destroy(md, false);
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.dm.dm_put")]
extern "C" fn dm_put(md: *mut mapped_device) {
atomic_dec(unsafe { core::ptr::addr_of_mut!((*md).holders) });
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_dm_put_875: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut mapped_device)>, *mut ()>(Some(dm_put)) };

fn dm_in_flight_bios(md: *mut mapped_device) -> bool {
let mut cpu: i32 = 0;
let mut sum: u64 = 0;
loop {
cpu = find_next_bit(cpumask_bits!((unsafe { core::ptr::addr_of_mut!(__cpu_possible_mask) }) as *const cpumask), 64u64, cpu as u64) as i32;
if (cpu as u32) >= 64 {
break;
}
let __v0: u64 = unsafe { *per_cpu_ptr__7!(unsafe { (*md).pending_io }, cpu) };
sum = sum.wrapping_add(__v0);
cpu = cpu.wrapping_add(1);
}
sum != 0
}

fn dm_wait_for_bios_completion(md: *mut mapped_device, task_state: u32) -> i32 {
let mut r: i32 = 0;
let mut wait: wait_queue_entry = unsafe { core::mem::zeroed() };
wait = {
let mut __t0: wait_queue_entry = unsafe { core::mem::zeroed() };
__t0.private = get_current() as *mut ();
__t0.func = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut wait_queue_entry, u32, i32, *mut ()) -> i32>>(autoremove_wake_function as *const ()) };
__t0.entry = list_head { next: core::ptr::addr_of_mut!(wait.entry), prev: core::ptr::addr_of_mut!(wait.entry) };
__t0
};
while true {
unsafe { prepare_to_wait(core::ptr::addr_of_mut!((*md).wait), core::ptr::addr_of_mut!(wait), task_state as i32) };
if !dm_in_flight_bios(md) {
break;
}
if signal_pending_state(task_state, get_current()) != 0 {
r = 512i32.wrapping_neg();
break;
}
unsafe { io_schedule() };
}
unsafe { finish_wait(core::ptr::addr_of_mut!((*md).wait), core::ptr::addr_of_mut!(wait)) };
loop {
kcsan_rmb!();
loop {
kcsan_rmb!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
if true {
break;
}
}
r
}

fn dm_wait_for_completion(md: *mut mapped_device, task_state: u32) -> i32 {
let mut r: i32 = 0;
if !queue_is_mq(unsafe { (*md).queue }) {
return dm_wait_for_bios_completion(md, task_state);
}
while true {
if !unsafe { blk_mq_queue_inflight((*md).queue) } {
break;
}
if signal_pending_state(task_state, get_current()) != 0 {
r = 512i32.wrapping_neg();
break;
}
fsleep(5000u64);
}
r
}

/// Process the deferred bios
#[unsafe(link_section = ".text.slate_distinct.dm.dm_wq_work")]
extern "C" fn dm_wq_work(work: *mut work_struct) {
let md: *mut mapped_device = container_of__11!(work, mapped_device);
while test_bit!(0i32, unsafe { core::ptr::addr_of_mut!((*md).flags) }) == 0 {
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*md).deferred_lock) });
let bio: *mut bio = bio_list_pop(unsafe { core::ptr::addr_of_mut!((*md).deferred) });
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*md).deferred_lock) });
if bio.is_null() {
break;
}
unsafe { submit_bio_noacct(bio) };
cond_resched__15!();
}
}

fn dm_queue_flush(md: *mut mapped_device) {
clear_bit(5i64, (unsafe { core::ptr::addr_of_mut!((*md).flags) }) as *mut u64);
clear_bit(0i64, (unsafe { core::ptr::addr_of_mut!((*md).flags) }) as *mut u64);
smp_mb__after_atomic!();
queue_work(unsafe { (*md).wq }, unsafe { core::ptr::addr_of_mut!((*md).work) });
}

/// Swap in a new table, returning the old one for the caller to destroy.
#[unsafe(no_mangle)]
extern "C" fn dm_swap_table(md: *mut mapped_device, table: *mut dm_table) -> *mut dm_table {
// To avoid a potential deadlock locking the queue limits, disallow
// updating the queue limits during a table swap, when updating an
// immutable request-based dm device (dm-multipath) during a noflush
// suspend. It is userspace's responsibility to make sure that the new
// table uses the same limits as the existing table, if it asks for a
// noflush suspend.
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut r: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut i32 = core::ptr::addr_of_mut!(r) as *mut i32;
let mut update_limits: bool = unsafe { core::mem::zeroed() };
let mut limits: core::mem::MaybeUninit<queue_limits> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut queue_limits = core::ptr::addr_of_mut!(limits) as *mut queue_limits;
let mut map: *mut dm_table = unsafe { core::mem::zeroed() };
let mut live_map: *mut dm_table = unsafe { core::mem::zeroed() };
unsafe {
'__join_0: {
live_map = core::ptr::null_mut::<dm_table>();
map = ERR_PTR(22i32.wrapping_neg() as i64) as *mut dm_table;
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 192usize);
update_limits = true;
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
mutex_lock(core::ptr::addr_of_mut!((*md).suspend_lock));
// device must be suspended
if dm_suspended_md(md) != 0 {
if dm_request_based(md) != 0 && !unsafe { (*md).immutable_target }.is_null() {
__v0 = __noflush_suspending(md) != 0;
} else {
__v0 = false;
}
if __v0 {
update_limits = false;
} else {
if dm_table_has_no_data_devices(table) {
live_map = dm_get_live_table_fast(md);
if !live_map.is_null() {
*__slate_slot_3 = (*(*md).queue).limits;
}
dm_put_live_table_fast(md);
}
}
// If the new table has no data devices, retain the existing limits.
// This helps multipath with queue_if_no_path if all paths disappear,
// then new I/O is queued based on these limits, and then some paths
// reappear.
if update_limits && live_map.is_null() {
*__slate_slot_1 = dm_calculate_queue_limits(table, core::ptr::addr_of_mut!(*__slate_slot_3));
if *__slate_slot_1 != 0 {
map = ERR_PTR(*__slate_slot_1 as i64) as *mut dm_table;
break '__join_0;
}
}
map = __bind(md, table, if update_limits { core::ptr::addr_of_mut!(*__slate_slot_3) } else { core::ptr::null_mut::<queue_limits>() });
dm_issue_global_event();
}
}
mutex_unlock(core::ptr::addr_of_mut!((*md).suspend_lock));
return map;
}
}

/// Functions to lock and unlock any filesystem running on the
/// device.
fn lock_fs(md: *mut mapped_device) -> i32 {
let mut r: i32 = 0;
let mut __v0: i64;
let __ret_warn_on: i32 = (test_bit!(2i32, unsafe { core::ptr::addr_of_mut!((*md).flags) }) != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("876: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 876b - ., 3; .popsection\n# {0}", const 876 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_lock_fs_2, sym __slate_asm_string_lock_fs_3, const 2940 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("877: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 877b - ., 4; .popsection\n# {0}", const 877 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
r = unsafe { bdev_freeze((*(*md).disk).part0) };
if r == 0 {
set_bit(2i64, (unsafe { core::ptr::addr_of_mut!((*md).flags) }) as *mut u64);
}
r
}

fn unlock_fs(md: *mut mapped_device) {
if test_bit!(2i32, unsafe { core::ptr::addr_of_mut!((*md).flags) }) == 0 {
return;
}
unsafe { bdev_thaw((*(*md).disk).part0) };
clear_bit(2i64, (unsafe { core::ptr::addr_of_mut!((*md).flags) }) as *mut u64);
}

/// @suspend_flags: DM_SUSPEND_LOCKFS_FLAG and/or DM_SUSPEND_NOFLUSH_FLAG
/// @task_state: e.g. TASK_INTERRUPTIBLE or TASK_UNINTERRUPTIBLE
/// @dmf_suspended_flag: DMF_SUSPENDED or DMF_SUSPENDED_INTERNALLY
///
/// If __dm_suspend returns 0, the device is completely quiescent
/// now. There is no request-processing activity. All new requests
/// are being added to md->deferred list.
fn __dm_suspend(md: *mut mapped_device, map: *mut dm_table, suspend_flags: u32, task_state: u32, dmf_suspended_flag: i32) -> i32 {
let do_lockfs: bool = suspend_flags & 1u32 != 0;
let noflush: bool = suspend_flags & 2u32 != 0;
let mut r: i32 = 0;
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!((*md).suspend_lock) });
// DMF_NOFLUSH_SUSPENDING must be set before presuspend.
if noflush {
set_bit(5i64, (unsafe { core::ptr::addr_of_mut!((*md).flags) }) as *mut u64);
} else {
let mut __v1: i32;
if false {
unsafe { _printk(c"\x017device-mapper: core: %s: suspending with flush\n".as_ptr() as *const u8, dm_device_name(md)) };
}
__v1 = 0;
}
// This gets reverted if there's an error later and the targets
// provide the .presuspend_undo hook.
unsafe { dm_table_presuspend_targets(map) };
// Flush I/O to the device.
// Any I/O submitted after lock_fs() may not be flushed.
// noflush takes precedence over do_lockfs.
// (lock_fs() flushes I/Os and waits for them to complete.)
if !noflush && do_lockfs {
r = lock_fs(md);
if r != 0 {
unsafe { dm_table_presuspend_undo_targets(map) };
return r;
}
}
// Here we must make sure that no processes are submitting requests
// to target drivers i.e. no one may be executing
// dm_split_and_process_bio from dm_submit_bio.
//
// To get all processes out of dm_split_and_process_bio in dm_submit_bio,
// we take the write lock. To prevent any process from reentering
// dm_split_and_process_bio from dm_submit_bio and quiesce the thread
// (dm_wq_work), we set DMF_BLOCK_IO_FOR_SUSPEND and call
// flush_workqueue(md->wq).
set_bit(0i64, (unsafe { core::ptr::addr_of_mut!((*md).flags) }) as *mut u64);
if !map.is_null() {
unsafe { synchronize_srcu(core::ptr::addr_of_mut!((*md).io_barrier)) };
}
// Stop md->queue before flushing md->wq in case request-based
// dm defers requests to md->wq from md->queue.
let __v2: bool = !map.is_null() && (unsafe { dm_request_based(md) }) != 0;
if __v2 {
unsafe { dm_stop_queue((*md).queue) };
set_bit(10i64, (unsafe { core::ptr::addr_of_mut!((*md).flags) }) as *mut u64);
}
flush_workqueue!(unsafe { (*md).wq });
// At this point no more requests are entering target request routines.
// We call dm_wait_for_completion to wait for all existing requests
// to finish.
if !map.is_null() {
r = dm_wait_for_completion(md, task_state);
}
if r == 0 {
set_bit(dmf_suspended_flag as i64, (unsafe { core::ptr::addr_of_mut!((*md).flags) }) as *mut u64);
}
if !map.is_null() {
unsafe { synchronize_srcu(core::ptr::addr_of_mut!((*md).io_barrier)) };
}
// were we interrupted ?
if r < 0 {
dm_queue_flush(md);
if test_and_clear_bit(10i64, (unsafe { core::ptr::addr_of_mut!((*md).flags) }) as *mut u64) {
unsafe { dm_start_queue((*md).queue) };
}
unlock_fs(md);
unsafe { dm_table_presuspend_undo_targets(map) };
// pushback list is already flushed, so skip flush
}
r
}

/// We need to be able to change a mapping table under a mounted
/// filesystem.  For example we might want to move some data in
/// the background.  Before the table can be swapped with
/// dm_bind_table, dm_suspend must be called to flush any in
/// flight bios and ensure that any further io gets deferred.
///
/// Suspend mechanism in request-based dm.
///
/// 1. Flush all I/Os by lock_fs() if needed.
/// 2. Stop dispatching any I/O by stopping the request_queue.
/// 3. Wait for all in-flight I/Os to be completed or requeued.
///
/// To abort suspend, start the request_queue.
#[unsafe(no_mangle)]
extern "C" fn dm_suspend(md: *mut mapped_device, mut suspend_flags: u32) -> i32 {
let mut r: i32 = unsafe { core::mem::zeroed() };
let mut map: *mut dm_table = unsafe { core::mem::zeroed() };
unsafe {
map = core::ptr::null_mut::<dm_table>();
r = 0;
'__join_0: {
'__join_8: {
loop {
mutex_lock_nested!(core::ptr::addr_of_mut!((*md).suspend_lock));
if dm_suspended_md(md) != 0 {
break '__join_8;
} else {
if dm_suspended_internally_md(md) != 0 {
// already internally suspended, wait for internal resume
mutex_unlock(core::ptr::addr_of_mut!((*md).suspend_lock));
r = wait_on_bit(core::ptr::addr_of_mut!((*md).flags), 7i32, 1u32);
if r != 0 {
return r;
}
} else {
break;
}
}
}
map = rcu_dereference_protected!(unsafe { (*md).map }, lockdep_is_held(core::ptr::addr_of_mut!((*md).suspend_lock) as *const ())) as *mut dm_table;
if map.is_null() {
// avoid deadlock with fs/namespace.c:do_mount()
let __v0: u32 = suspend_flags & 4294967294u32;
suspend_flags = __v0;
}
r = __dm_suspend(md, map, suspend_flags, 1u32, 1i32);
if r != 0 {
break '__join_0;
} else {
set_bit(8i64, core::ptr::addr_of_mut!((*md).flags) as *mut u64);
dm_table_postsuspend_targets(map);
clear_bit(8i64, core::ptr::addr_of_mut!((*md).flags) as *mut u64);
break '__join_0;
}
}
r = 22i32.wrapping_neg();
}
mutex_unlock(core::ptr::addr_of_mut!((*md).suspend_lock));
return r;
}
}

fn __dm_resume(md: *mut mapped_device, map: *mut dm_table) -> i32 {
if !map.is_null() {
let r: i32 = unsafe { dm_table_resume_targets(map) };
if r != 0 {
return r;
}
}
dm_queue_flush(md);
// Flushing deferred I/Os must be done after targets are resumed
// so that mapping of targets can work correctly.
// Request-based dm is queueing the deferred I/Os in its request_queue.
if test_and_clear_bit(10i64, (unsafe { core::ptr::addr_of_mut!((*md).flags) }) as *mut u64) {
unsafe { dm_start_queue((*md).queue) };
}
unlock_fs(md);
0
}

#[unsafe(no_mangle)]
extern "C" fn dm_resume(md: *mut mapped_device) -> i32 {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut map: *mut dm_table = unsafe { core::mem::zeroed() };
let mut r: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut i32 = core::ptr::addr_of_mut!(r) as *mut i32;
unsafe {
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
map = core::ptr::null_mut::<dm_table>();
'__join_0: {
loop {
*__slate_slot_3 = 22i32.wrapping_neg();
mutex_lock_nested!(core::ptr::addr_of_mut!((*md).suspend_lock));
if dm_suspended_md(md) == 0 {
__v1 = true;
} else {
__v1 = test_bit!(3i32, core::ptr::addr_of_mut!((*md).flags)) != 0;
}
if __v1 {
break '__join_0;
} else {
if dm_suspended_internally_md(md) != 0 {
// already internally suspended, wait for internal resume
mutex_unlock(core::ptr::addr_of_mut!((*md).suspend_lock));
*__slate_slot_3 = wait_on_bit(core::ptr::addr_of_mut!((*md).flags), 7i32, 1u32);
if *__slate_slot_3 != 0 {
return *__slate_slot_3;
}
} else {
break;
}
}
}
map = rcu_dereference_protected!(unsafe { (*md).map }, lockdep_is_held(core::ptr::addr_of_mut!((*md).suspend_lock) as *const ())) as *mut dm_table;
if map.is_null() {
__v0 = true;
} else {
__v0 = dm_table_get_size(map) == 0;
}
if !__v0 {
*__slate_slot_3 = __dm_resume(md, map);
if *__slate_slot_3 == 0 {
clear_bit(1i64, core::ptr::addr_of_mut!((*md).flags) as *mut u64);
}
}
}
mutex_unlock(core::ptr::addr_of_mut!((*md).suspend_lock));
return *__slate_slot_3;
}
}

// Internal suspend/resume works like userspace-driven suspend. It waits
// until all bios finish and prevents issuing new bios to the target drivers.
// It may be used only from the kernel.
fn __dm_internal_suspend(md: *mut mapped_device, suspend_flags: u32) {
let mut map: *mut dm_table = core::ptr::null_mut::<dm_table>();
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!((*md).suspend_lock) });
let __v2: u32 = unsafe { (*md).internal_suspend_count };
unsafe {
(*md).internal_suspend_count = __v2.wrapping_add(1);
}
if __v2 != 0 {
return;
}
// nested internal suspend
if dm_suspended_md(md) != 0 {
set_bit(7i64, (unsafe { core::ptr::addr_of_mut!((*md).flags) }) as *mut u64);
return; // nest suspend
}
map = rcu_dereference_protected!(unsafe { (*md).map }, unsafe { lockdep_is_held(core::ptr::addr_of_mut!((*md).suspend_lock) as *const ()) }) as *mut dm_table;
// Using TASK_UNINTERRUPTIBLE because only NOFLUSH internal suspend is
// supported.  Properly supporting a TASK_INTERRUPTIBLE internal suspend
// would require changing .presuspend to return an error -- avoid this
// until there is a need for more elaborate variants of internal suspend.
__dm_suspend(md, map, suspend_flags, 2u32, 7i32);
set_bit(8i64, (unsafe { core::ptr::addr_of_mut!((*md).flags) }) as *mut u64);
unsafe { dm_table_postsuspend_targets(map) };
clear_bit(8i64, (unsafe { core::ptr::addr_of_mut!((*md).flags) }) as *mut u64);
}

fn __dm_internal_resume(md: *mut mapped_device) {
let mut map: core::mem::MaybeUninit<*mut dm_table> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut dm_table = core::ptr::addr_of_mut!(map) as *mut *mut dm_table;
let mut r: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i32 = core::ptr::addr_of_mut!(r) as *mut i32;
unsafe {
'__slate_dispatch: {
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
'__join_9: {
loop {
if unlikely!((*md).internal_suspend_count == 0) != 0 {
break '__join_9;
} else {
if true {
break;
}
}
}
let __v2: u32 = (*md).internal_suspend_count;
let __v1: u32 = __v2.wrapping_sub(1);
(*md).internal_suspend_count = __v1;
if __v1 != 0 {
return;
} else {
// resume from nested internal suspend
if dm_suspended_md(md) == 0 {
// resume from nested suspend
*__slate_slot_4 = rcu_dereference_protected!(unsafe { (*md).map }, lockdep_is_held(core::ptr::addr_of_mut!((*md).suspend_lock) as *const ())) as *mut dm_table;
*__slate_slot_5 = __dm_resume(md, *__slate_slot_4);
if *__slate_slot_5 != 0 {
while false {
}
// If a preresume method of some target failed, we are in a
// tricky situation. We can't return an error to the caller. We
// can't fake success because then the "resume" and
// "postsuspend" methods would not be paired correctly, and it
// would break various targets, for example it would cause list
// corruption in the "origin" target.
//
// So, we fake normal suspend here, to make sure that the
// "resume" and "postsuspend" methods will be paired correctly.
_printk(c"\x013device-mapper: core: Preresume method failed: %d\n".as_ptr() as *const u8, *__slate_slot_5);
set_bit(1i64, core::ptr::addr_of_mut!((*md).flags) as *mut u64);
}
}
clear_bit(7i64, core::ptr::addr_of_mut!((*md).flags) as *mut u64);
smp_mb__after_atomic!();
wake_up_bit(core::ptr::addr_of_mut!((*md).flags), 7i32);
break '__slate_dispatch;
}
}
core::arch::asm!("878: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 878b - ., 3; .popsection\n# {0}", const 878 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string___dm_internal_resume_12, sym __slate_asm_string___dm_internal_resume_13, const 3215 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.dm.dm_internal_suspend_noflush")]
extern "C" fn dm_internal_suspend_noflush(md: *mut mapped_device) {
unsafe { mutex_lock(core::ptr::addr_of_mut!((*md).suspend_lock)) };
__dm_internal_suspend(md, 2u32);
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*md).suspend_lock)) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_dm_internal_suspend_noflush_879: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut mapped_device)>, *mut ()>(Some(dm_internal_suspend_noflush)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.dm.dm_internal_resume")]
extern "C" fn dm_internal_resume(md: *mut mapped_device) {
unsafe { mutex_lock(core::ptr::addr_of_mut!((*md).suspend_lock)) };
__dm_internal_resume(md);
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*md).suspend_lock)) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_dm_internal_resume_880: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut mapped_device)>, *mut ()>(Some(dm_internal_resume)) };

// Fast variants of internal suspend/resume hold md->suspend_lock,
// which prevents interaction with userspace-driven suspend.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.dm.dm_internal_suspend_fast")]
extern "C" fn dm_internal_suspend_fast(md: *mut mapped_device) {
unsafe { mutex_lock(core::ptr::addr_of_mut!((*md).suspend_lock)) };
let __v0: bool = dm_suspended_md(md) != 0 || dm_suspended_internally_md(md) != 0;
if __v0 {
return;
}
set_bit(0i64, (unsafe { core::ptr::addr_of_mut!((*md).flags) }) as *mut u64);
unsafe { synchronize_srcu(core::ptr::addr_of_mut!((*md).io_barrier)) };
flush_workqueue!(unsafe { (*md).wq });
dm_wait_for_completion(md, 2u32);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_dm_internal_suspend_fast_881: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut mapped_device)>, *mut ()>(Some(dm_internal_suspend_fast)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.dm.dm_internal_resume_fast")]
extern "C" fn dm_internal_resume_fast(md: *mut mapped_device) {
let mut __v0: bool = unsafe { core::mem::zeroed() };
unsafe {
if dm_suspended_md(md) != 0 {
__v0 = true;
} else {
__v0 = dm_suspended_internally_md(md) != 0;
}
if !__v0 {
dm_queue_flush(md);
}
mutex_unlock(core::ptr::addr_of_mut!((*md).suspend_lock));
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_dm_internal_resume_fast_882: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut mapped_device)>, *mut ()>(Some(dm_internal_resume_fast)) };

/// Event notification.
#[unsafe(no_mangle)]
extern "C" fn dm_kobject_uevent(md: *mut mapped_device, action: u32, cookie: u32, need_resize_uevent: bool) -> i32 {
let mut r: i32 = 0;
let mut noio_flag: u32 = 0;
let mut udev_cookie: __SlateAlign16<[u8; 24]> = __SlateAlign16([0u8; 24]);
let mut envp: __SlateAlign16<[*mut u8; 3]> = __SlateAlign16([core::ptr::null_mut::<u8>(), core::ptr::null_mut::<u8>(), core::ptr::null_mut::<u8>()]);
let mut envpp: *mut *mut u8 = envp.0.as_mut_ptr() as *mut *mut u8;
if cookie != 0 {
unsafe { snprintf(udev_cookie.0.as_mut_ptr() as *mut u8, 24u64, c"%s=%u".as_ptr() as *const u8, c"DM_COOKIE".as_ptr() as *mut u8, cookie) };
let __v0: *mut *mut u8 = envpp;
envpp = __v0.wrapping_offset(1 as isize);
unsafe {
*__v0 = udev_cookie.0.as_mut_ptr() as *mut u8;
}
}
if need_resize_uevent {
let __v2: *mut *mut u8 = envpp;
envpp = __v2.wrapping_offset(1 as isize);
unsafe {
*__v2 = c"RESIZE=1".as_ptr() as *mut u8;
}
}
noio_flag = memalloc_noio_save();
r = unsafe { kobject_uevent_env(core::ptr::addr_of_mut!((*disk_to_dev!(unsafe { (*md).disk })).kobj), action, envp.0.as_mut_ptr() as *mut *mut u8) };
memalloc_noio_restore(noio_flag);
r
}

#[unsafe(no_mangle)]
extern "C" fn dm_next_uevent_seq(md: *mut mapped_device) -> u32 {
atomic_add_return(1i32, unsafe { core::ptr::addr_of_mut!((*md).uevent_seq) }) as u32
}

#[unsafe(no_mangle)]
extern "C" fn dm_get_event_nr(md: *mut mapped_device) -> u32 {
atomic_read((unsafe { core::ptr::addr_of_mut!((*md).event_nr) }) as *const __SlateRecord_types_h_190) as u32
}

#[unsafe(no_mangle)]
extern "C" fn dm_wait_event(md: *mut mapped_device, event_nr: i32) -> i32 {
let mut __int: i64 = unsafe { core::mem::zeroed() };
let mut __ret: i64 = unsafe { core::mem::zeroed() };
let mut __wq_entry: core::mem::MaybeUninit<wait_queue_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut wait_queue_entry = core::ptr::addr_of_mut!(__wq_entry) as *mut wait_queue_entry;
let mut __ret_1: i32 = unsafe { core::mem::zeroed() };
unsafe {
__ret_1 = 0;
might_sleep!();
if event_nr == atomic_read(core::ptr::addr_of_mut!((*md).event_nr) as *const __SlateRecord_types_h_190) {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 40usize);
__ret = 0;
init_wait_entry(core::ptr::addr_of_mut!(*__slate_slot_2), if false { 1i32 } else { 0i32 });
'__join_1: {
'__join_2: {
loop {
__int = prepare_to_wait_event(core::ptr::addr_of_mut!((*md).eventq), core::ptr::addr_of_mut!(*__slate_slot_2), 1i32);
if event_nr != atomic_read(core::ptr::addr_of_mut!((*md).event_nr) as *const __SlateRecord_types_h_190) {
break '__join_2;
} else {
if ___wait_is_interruptible!(1i32) && __int != 0 {
break;
} else {
schedule();
if event_nr != atomic_read(core::ptr::addr_of_mut!((*md).event_nr) as *const __SlateRecord_types_h_190) {
break '__join_2;
}
}
}
}
__ret = __int;
break '__join_1;
}
finish_wait(core::ptr::addr_of_mut!((*md).eventq), core::ptr::addr_of_mut!(*__slate_slot_2));
}
__ret_1 = __ret as i32;
}
return __ret_1;
}
}

#[unsafe(no_mangle)]
extern "C" fn dm_uevent_add(md: *mut mapped_device, elist: *mut list_head) {
let mut flags: u64 = 0;
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*md).uevent_lock) }, flags);
list_add(elist, unsafe { core::ptr::addr_of_mut!((*md).uevent_list) });
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*md).uevent_lock) }, flags);
}

/// The gendisk is only valid as long as you have a reference
/// count on 'md'.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.dm.dm_disk")]
extern "C" fn dm_disk(md: *mut mapped_device) -> *mut gendisk {
unsafe { (*md).disk }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_dm_disk_883: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut mapped_device) -> *mut gendisk>, *mut ()>(Some(dm_disk)) };

#[unsafe(no_mangle)]
extern "C" fn dm_kobject(md: *mut mapped_device) -> *mut kobject {
unsafe { core::ptr::addr_of_mut!((*md).kobj_holder.kobj) }
}

#[unsafe(no_mangle)]
extern "C" fn dm_get_from_kobject(kobj: *mut kobject) -> *mut mapped_device {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut md: core::mem::MaybeUninit<*mut mapped_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut mapped_device = core::ptr::addr_of_mut!(md) as *mut *mut mapped_device;
unsafe {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __v1: *mut mapped_device = ((kobj as *const u8).wrapping_offset(-(offsetof__105!(mapped_device) as isize)) as *mut ()) as *mut mapped_device;
*__slate_slot_2 = __v1;
spin_lock(core::ptr::addr_of_mut!(_minor_lock));
if test_bit!(3i32, core::ptr::addr_of_mut!((*(*__slate_slot_2)).flags)) != 0 {
__v0 = true;
} else {
__v0 = dm_deleting_md(*__slate_slot_2) != 0;
}
if __v0 {
*__slate_slot_2 = core::ptr::null_mut::<mapped_device>();
} else {
dm_get(*__slate_slot_2);
}
spin_unlock(core::ptr::addr_of_mut!(_minor_lock));
return *__slate_slot_2;
}
}

#[unsafe(no_mangle)]
extern "C" fn dm_suspended_md(md: *mut mapped_device) -> i32 {
test_bit!(1i32, unsafe { core::ptr::addr_of_mut!((*md).flags) })
}

fn dm_post_suspending_md(md: *mut mapped_device) -> i32 {
test_bit!(8i32, unsafe { core::ptr::addr_of_mut!((*md).flags) })
}

#[unsafe(no_mangle)]
extern "C" fn dm_suspended_internally_md(md: *mut mapped_device) -> i32 {
test_bit!(7i32, unsafe { core::ptr::addr_of_mut!((*md).flags) })
}

#[unsafe(no_mangle)]
extern "C" fn dm_test_deferred_remove_flag(md: *mut mapped_device) -> i32 {
test_bit!(6i32, unsafe { core::ptr::addr_of_mut!((*md).flags) })
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.dm.dm_suspended")]
extern "C" fn dm_suspended(ti: *mut dm_target) -> i32 {
dm_suspended_md(unsafe { (*(*ti).table).md })
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_dm_suspended_884: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut dm_target) -> i32>, *mut ()>(Some(dm_suspended)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.dm.dm_post_suspending")]
extern "C" fn dm_post_suspending(ti: *mut dm_target) -> i32 {
dm_post_suspending_md(unsafe { (*(*ti).table).md })
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_dm_post_suspending_885: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut dm_target) -> i32>, *mut ()>(Some(dm_post_suspending)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.dm.dm_noflush_suspending")]
extern "C" fn dm_noflush_suspending(ti: *mut dm_target) -> i32 {
__noflush_suspending(unsafe { (*(*ti).table).md })
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_dm_noflush_suspending_886: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut dm_target) -> i32>, *mut ()>(Some(dm_noflush_suspending)) };

#[unsafe(no_mangle)]
extern "C" fn dm_free_md_mempools(pools: *mut dm_md_mempools) {
if pools.is_null() {
return;
}
unsafe { bioset_exit(core::ptr::addr_of_mut!((*pools).bs)) };
unsafe { bioset_exit(core::ptr::addr_of_mut!((*pools).io_bs)) };
unsafe { kfree(pools as *const ()) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct dm_blkdev_id {
    id: *mut u8,
    r#type: u32,
}


#[unsafe(link_section = ".text.slate_distinct.dm.__dm_get_unique_id")]
extern "C" fn __dm_get_unique_id(ti: *mut dm_target, dev: *mut dm_dev, start: u64, len: u64, data: *mut ()) -> i32 {
let fops: *const block_device_operations = unsafe { (*(*(*dev).bdev).bd_disk).fops };
if (unsafe { (*fops).get_unique_id }) == None {
return 0;
}
unsafe { unsafe { (*fops).get_unique_id }.unwrap()((*(*dev).bdev).bd_disk, (*(data as *mut dm_blkdev_id)).id, (*(data as *mut dm_blkdev_id)).r#type) }
}

/// Allow access to get_unique_id() for the first device returning a
/// non-zero result.  Reasonable use expects all devices to have the
/// same unique id.
#[unsafe(link_section = ".text.slate_distinct.dm.dm_blk_get_unique_id")]
extern "C" fn dm_blk_get_unique_id(disk: *mut gendisk, id: *mut u8, r#type: u32) -> i32 {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut dm_id: core::mem::MaybeUninit<dm_blkdev_id> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut dm_blkdev_id = core::ptr::addr_of_mut!(dm_id) as *mut dm_blkdev_id;
let mut srcu_idx: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut i32 = core::ptr::addr_of_mut!(srcu_idx) as *mut i32;
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut ti: core::mem::MaybeUninit<*mut dm_target> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut dm_target = core::ptr::addr_of_mut!(ti) as *mut *mut dm_target;
let mut table: core::mem::MaybeUninit<*mut dm_table> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut dm_table = core::ptr::addr_of_mut!(table) as *mut *mut dm_table;
let mut md: *mut mapped_device = unsafe { core::mem::zeroed() };
unsafe {
md = (*disk).private_data as *mut mapped_device;
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
ret = 0;
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
core::ptr::write(__slate_slot_1, {
let mut __t0: dm_blkdev_id = core::mem::zeroed();
__t0.id = id;
__t0.r#type = r#type;
__t0
});
*__slate_slot_5 = dm_get_live_table(md, core::ptr::addr_of_mut!(*__slate_slot_2));
if (*__slate_slot_5).is_null() {
__v0 = true;
} else {
__v0 = dm_table_get_size(*__slate_slot_5) == 0;
}
if !__v0 {
// We only support devices that have a single target
if (*(*__slate_slot_5)).num_targets == 1 {
*__slate_slot_4 = dm_table_get_target(*__slate_slot_5, 0u32);
if (*(*(*__slate_slot_4)).r#type).iterate_devices != None {
ret = unsafe { (*(*(*__slate_slot_4)).r#type).iterate_devices }.unwrap()(*__slate_slot_4, Some(__dm_get_unique_id), core::ptr::addr_of_mut!(*__slate_slot_1) as *mut ());
}
}
}
dm_put_live_table(md, *__slate_slot_2);
return ret;
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct dm_pr {
    old_key: u64,
    new_key: u64,
    flags: u32,
    abort: bool,
    fail_early: bool,
    ret: i32,
    r#type: u32,
    read_keys: *mut pr_keys,
    rsv: *mut pr_held_reservation,
}


fn dm_call_pr(bdev: *mut block_device, r#fn: Option<unsafe extern "C" fn(*mut dm_target, *mut dm_dev, u64, u64, *mut ()) -> i32>, pr: *mut dm_pr) -> i32 {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut srcu_idx: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut i32 = core::ptr::addr_of_mut!(srcu_idx) as *mut i32;
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut ti: core::mem::MaybeUninit<*mut dm_target> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut dm_target = core::ptr::addr_of_mut!(ti) as *mut *mut dm_target;
let mut table: core::mem::MaybeUninit<*mut dm_table> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut dm_table = core::ptr::addr_of_mut!(table) as *mut *mut dm_table;
let mut md: *mut mapped_device = unsafe { core::mem::zeroed() };
unsafe {
md = (*(*bdev).bd_disk).private_data as *mut mapped_device;
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
ret = 25i32.wrapping_neg();
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_4 = dm_get_live_table(md, core::ptr::addr_of_mut!(*__slate_slot_1));
if (*__slate_slot_4).is_null() {
__v0 = true;
} else {
__v0 = dm_table_get_size(*__slate_slot_4) == 0;
}
if !__v0 {
// We only support devices that have a single target
if (*(*__slate_slot_4)).num_targets == 1 {
*__slate_slot_3 = dm_table_get_target(*__slate_slot_4, 0u32);
if dm_suspended_md(md) != 0 {
ret = 11i32.wrapping_neg();
} else {
ret = 22i32.wrapping_neg();
if (*(*(*__slate_slot_3)).r#type).iterate_devices != None {
unsafe { (*(*(*__slate_slot_3)).r#type).iterate_devices }.unwrap()(*__slate_slot_3, r#fn, pr as *mut ());
ret = 0;
}
}
}
}
dm_put_live_table(md, *__slate_slot_1);
return ret;
}
}

/// For register / unregister we need to manually call out to every path.
#[unsafe(link_section = ".text.slate_distinct.dm.__dm_pr_register")]
extern "C" fn __dm_pr_register(ti: *mut dm_target, dev: *mut dm_dev, start: u64, len: u64, data: *mut ()) -> i32 {
let ops: *const pr_ops = unsafe { (*(*(*(*dev).bdev).bd_disk).fops).pr_ops };
let mut ret: i32 = 0;
if ops.is_null() || (unsafe { (*ops).pr_register }) == None {
unsafe {
(*(data as *mut dm_pr)).ret = 95i32.wrapping_neg();
}
return 1i32.wrapping_neg();
}
ret = unsafe { unsafe { (*ops).pr_register }.unwrap()((*dev).bdev, (*(data as *mut dm_pr)).old_key, (*(data as *mut dm_pr)).new_key, (*(data as *mut dm_pr)).flags) };
if ret == 0 {
return 0;
}
if (unsafe { (*(data as *mut dm_pr)).ret }) == 0 {
unsafe {
(*(data as *mut dm_pr)).ret = ret;
}
}
if unsafe { (*(data as *mut dm_pr)).fail_early } {
return 1i32.wrapping_neg();
}
0
}

#[unsafe(link_section = ".text.slate_distinct.dm.dm_pr_register")]
extern "C" fn dm_pr_register(bdev: *mut block_device, old_key: u64, new_key: u64, flags: u32) -> i32 {
let mut pr: dm_pr = {
let mut __t0: dm_pr = unsafe { core::mem::zeroed() };
__t0.old_key = old_key;
__t0.new_key = new_key;
__t0.flags = flags;
__t0.fail_early = true;
__t0.ret = 0i32;
__t0
};
let mut ret: i32 = dm_call_pr(bdev, Some(__dm_pr_register), core::ptr::addr_of_mut!(pr));
if ret != 0 {
// Didn't even get to register a path
return ret;
}
if pr.ret == 0 {
return 0;
}
ret = pr.ret;
if new_key == 0 {
return ret;
}
// unregister all paths if we failed to register any path
pr.old_key = new_key;
pr.new_key = 0;
pr.flags = 0;
pr.fail_early = false;
dm_call_pr(bdev, Some(__dm_pr_register), core::ptr::addr_of_mut!(pr));
ret
}

#[unsafe(link_section = ".text.slate_distinct.dm.__dm_pr_reserve")]
extern "C" fn __dm_pr_reserve(ti: *mut dm_target, dev: *mut dm_dev, start: u64, len: u64, data: *mut ()) -> i32 {
let ops: *const pr_ops = unsafe { (*(*(*(*dev).bdev).bd_disk).fops).pr_ops };
if ops.is_null() || (unsafe { (*ops).pr_reserve }) == None {
unsafe {
(*(data as *mut dm_pr)).ret = 95i32.wrapping_neg();
}
return 1i32.wrapping_neg();
}
unsafe {
(*(data as *mut dm_pr)).ret = unsafe { (*ops).pr_reserve }.unwrap()((*dev).bdev, (*(data as *mut dm_pr)).old_key, (*(data as *mut dm_pr)).r#type, (*(data as *mut dm_pr)).flags);
}
if (unsafe { (*(data as *mut dm_pr)).ret }) == 0 {
return 1i32.wrapping_neg();
}
0
}

#[unsafe(link_section = ".text.slate_distinct.dm.dm_pr_reserve")]
extern "C" fn dm_pr_reserve(bdev: *mut block_device, key: u64, r#type: u32, flags: u32) -> i32 {
let mut pr: dm_pr = {
let mut __t0: dm_pr = unsafe { core::mem::zeroed() };
__t0.old_key = key;
__t0.flags = flags;
__t0.fail_early = false;
__t0.ret = 0i32;
__t0.r#type = r#type;
__t0
};
let ret: i32 = dm_call_pr(bdev, Some(__dm_pr_reserve), core::ptr::addr_of_mut!(pr));
if ret != 0 {
return ret;
}
pr.ret
}

/// If there is a non-All Registrants type of reservation, the release must be
/// sent down the holding path. For the cases where there is no reservation or
/// the path is not the holder the device will also return success, so we must
/// try each path to make sure we got the correct path.
#[unsafe(link_section = ".text.slate_distinct.dm.__dm_pr_release")]
extern "C" fn __dm_pr_release(ti: *mut dm_target, dev: *mut dm_dev, start: u64, len: u64, data: *mut ()) -> i32 {
let ops: *const pr_ops = unsafe { (*(*(*(*dev).bdev).bd_disk).fops).pr_ops };
if ops.is_null() || (unsafe { (*ops).pr_release }) == None {
unsafe {
(*(data as *mut dm_pr)).ret = 95i32.wrapping_neg();
}
return 1i32.wrapping_neg();
}
unsafe {
(*(data as *mut dm_pr)).ret = unsafe { (*ops).pr_release }.unwrap()((*dev).bdev, (*(data as *mut dm_pr)).old_key, (*(data as *mut dm_pr)).r#type);
}
if (unsafe { (*(data as *mut dm_pr)).ret }) != 0 {
return 1i32.wrapping_neg();
}
0
}

#[unsafe(link_section = ".text.slate_distinct.dm.dm_pr_release")]
extern "C" fn dm_pr_release(bdev: *mut block_device, key: u64, r#type: u32) -> i32 {
let mut pr: dm_pr = {
let mut __t0: dm_pr = unsafe { core::mem::zeroed() };
__t0.old_key = key;
__t0.fail_early = false;
__t0.r#type = r#type;
__t0
};
let ret: i32 = dm_call_pr(bdev, Some(__dm_pr_release), core::ptr::addr_of_mut!(pr));
if ret != 0 {
return ret;
}
pr.ret
}

#[unsafe(link_section = ".text.slate_distinct.dm.__dm_pr_preempt")]
extern "C" fn __dm_pr_preempt(ti: *mut dm_target, dev: *mut dm_dev, start: u64, len: u64, data: *mut ()) -> i32 {
let ops: *const pr_ops = unsafe { (*(*(*(*dev).bdev).bd_disk).fops).pr_ops };
if ops.is_null() || (unsafe { (*ops).pr_preempt }) == None {
unsafe {
(*(data as *mut dm_pr)).ret = 95i32.wrapping_neg();
}
return 1i32.wrapping_neg();
}
unsafe {
(*(data as *mut dm_pr)).ret = unsafe { (*ops).pr_preempt }.unwrap()((*dev).bdev, (*(data as *mut dm_pr)).old_key, (*(data as *mut dm_pr)).new_key, (*(data as *mut dm_pr)).r#type, (*(data as *mut dm_pr)).abort);
}
if (unsafe { (*(data as *mut dm_pr)).ret }) == 0 {
return 1i32.wrapping_neg();
}
0
}

#[unsafe(link_section = ".text.slate_distinct.dm.dm_pr_preempt")]
extern "C" fn dm_pr_preempt(bdev: *mut block_device, old_key: u64, new_key: u64, r#type: u32, abort: bool) -> i32 {
let mut pr: dm_pr = {
let mut __t0: dm_pr = unsafe { core::mem::zeroed() };
__t0.old_key = old_key;
__t0.new_key = new_key;
__t0.fail_early = false;
__t0.r#type = r#type;
__t0
};
let ret: i32 = dm_call_pr(bdev, Some(__dm_pr_preempt), core::ptr::addr_of_mut!(pr));
if ret != 0 {
return ret;
}
pr.ret
}

#[unsafe(link_section = ".text.slate_distinct.dm.dm_pr_clear")]
extern "C" fn dm_pr_clear(mut bdev: *mut block_device, key: u64) -> i32 {
let mut forward: bool = unsafe { core::mem::zeroed() };
let mut srcu_idx: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(srcu_idx) as *mut i32;
let mut r: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i32 = core::ptr::addr_of_mut!(r) as *mut i32;
let mut ops: core::mem::MaybeUninit<*const pr_ops> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *const pr_ops = core::ptr::addr_of_mut!(ops) as *mut *const pr_ops;
let mut md: *mut mapped_device = unsafe { core::mem::zeroed() };
unsafe {
md = (*(*bdev).bd_disk).private_data as *mut mapped_device;
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
forward = true;
// Not a real ioctl, but targets must not interpret non-DM ioctls
*__slate_slot_5 = dm_prepare_ioctl(md, core::ptr::addr_of_mut!(*__slate_slot_4), core::ptr::addr_of_mut!(bdev), 0u32, 0u64, core::ptr::addr_of_mut!(forward));
if *__slate_slot_5 >= 0 {
let __ret_warn_on: i32 = !forward as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("887: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 887b - ., 3; .popsection\n# {0}", const 887 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_dm_pr_clear_8, sym __slate_asm_string_dm_pr_clear_9, const 3716 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("888: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 888b - ., 4; .popsection\n# {0}", const 888 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
*__slate_slot_6 = (*(*(*bdev).bd_disk).fops).pr_ops;
if !(*__slate_slot_6).is_null() && (*(*__slate_slot_6)).pr_clear != None {
*__slate_slot_5 = unsafe { (*(*__slate_slot_6)).pr_clear }.unwrap()(bdev, key);
} else {
*__slate_slot_5 = 95i32.wrapping_neg();
}
}
dm_unprepare_ioctl(md, *__slate_slot_4);
return *__slate_slot_5;
}
}

#[unsafe(link_section = ".text.slate_distinct.dm.__dm_pr_read_keys")]
extern "C" fn __dm_pr_read_keys(ti: *mut dm_target, dev: *mut dm_dev, start: u64, len: u64, data: *mut ()) -> i32 {
let ops: *const pr_ops = unsafe { (*(*(*(*dev).bdev).bd_disk).fops).pr_ops };
if ops.is_null() || (unsafe { (*ops).pr_read_keys }) == None {
unsafe {
(*(data as *mut dm_pr)).ret = 95i32.wrapping_neg();
}
return 1i32.wrapping_neg();
}
unsafe {
(*(data as *mut dm_pr)).ret = unsafe { (*ops).pr_read_keys }.unwrap()((*dev).bdev, (*(data as *mut dm_pr)).read_keys);
}
if (unsafe { (*(data as *mut dm_pr)).ret }) == 0 {
return 1i32.wrapping_neg();
}
0
}

#[unsafe(link_section = ".text.slate_distinct.dm.dm_pr_read_keys")]
extern "C" fn dm_pr_read_keys(bdev: *mut block_device, keys: *mut pr_keys) -> i32 {
let mut pr: dm_pr = {
let mut __t0: dm_pr = unsafe { core::mem::zeroed() };
__t0.read_keys = keys;
__t0
};
let ret: i32 = dm_call_pr(bdev, Some(__dm_pr_read_keys), core::ptr::addr_of_mut!(pr));
if ret != 0 {
return ret;
}
pr.ret
}

#[unsafe(link_section = ".text.slate_distinct.dm.__dm_pr_read_reservation")]
extern "C" fn __dm_pr_read_reservation(ti: *mut dm_target, dev: *mut dm_dev, start: u64, len: u64, data: *mut ()) -> i32 {
let ops: *const pr_ops = unsafe { (*(*(*(*dev).bdev).bd_disk).fops).pr_ops };
if ops.is_null() || (unsafe { (*ops).pr_read_reservation }) == None {
unsafe {
(*(data as *mut dm_pr)).ret = 95i32.wrapping_neg();
}
return 1i32.wrapping_neg();
}
unsafe {
(*(data as *mut dm_pr)).ret = unsafe { (*ops).pr_read_reservation }.unwrap()((*dev).bdev, (*(data as *mut dm_pr)).rsv);
}
if (unsafe { (*(data as *mut dm_pr)).ret }) == 0 {
return 1i32.wrapping_neg();
}
0
}

#[unsafe(link_section = ".text.slate_distinct.dm.dm_pr_read_reservation")]
extern "C" fn dm_pr_read_reservation(bdev: *mut block_device, rsv: *mut pr_held_reservation) -> i32 {
let mut pr: dm_pr = {
let mut __t0: dm_pr = unsafe { core::mem::zeroed() };
__t0.rsv = rsv;
__t0
};
let ret: i32 = dm_call_pr(bdev, Some(__dm_pr_read_reservation), core::ptr::addr_of_mut!(pr));
if ret != 0 {
return ret;
}
pr.ret
}

static dm_pr_ops: __SlateConst<pr_ops> = __SlateConst(pr_ops { pr_register: Some(dm_pr_register), pr_reserve: Some(dm_pr_reserve), pr_release: Some(dm_pr_release), pr_preempt: Some(dm_pr_preempt), pr_clear: Some(dm_pr_clear), pr_read_keys: Some(dm_pr_read_keys), pr_read_reservation: Some(dm_pr_read_reservation) });

static dm_blk_dops: __SlateConst<block_device_operations> = __SlateConst({
let mut __t0: block_device_operations = unsafe { core::mem::zeroed() };
__t0.submit_bio = Some(dm_submit_bio);
__t0.poll_bio = Some(dm_poll_bio);
__t0.open = Some(dm_blk_open);
__t0.release = Some(dm_blk_close);
__t0.ioctl = Some(dm_blk_ioctl);
__t0.getgeo = Some(dm_blk_getgeo);
__t0.report_zones = None;
__t0.get_unique_id = Some(dm_blk_get_unique_id);
__t0.owner = core::ptr::null_mut::<module>();
__t0.pr_ops = unsafe { core::ptr::addr_of!(dm_pr_ops.0) };
__t0
});

static dm_rq_blk_dops: __SlateConst<block_device_operations> = __SlateConst({
let mut __t0: block_device_operations = unsafe { core::mem::zeroed() };
__t0.open = Some(dm_blk_open);
__t0.release = Some(dm_blk_close);
__t0.ioctl = Some(dm_blk_ioctl);
__t0.getgeo = Some(dm_blk_getgeo);
__t0.get_unique_id = Some(dm_blk_get_unique_id);
__t0.owner = core::ptr::null_mut::<module>();
__t0.pr_ops = unsafe { core::ptr::addr_of!(dm_pr_ops.0) };
__t0
});

static dm_dax_ops: __SlateConst<dax_operations> = __SlateConst(dax_operations { direct_access: Some(dm_dax_direct_access), zero_page_range: Some(dm_dax_zero_page_range), recovery_write: Some(dm_dax_recovery_write) });

/// module hooks
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_dm_init_890: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i32>, *mut ()>(Some(dm_init)) };

#[unsafe(link_section = ".exitcall.exit")]
static mut __exitcall_dm_exit: Option<unsafe extern "C" fn()> = Some(dm_exit);

fn __check_major() -> *mut u32 {
unsafe { core::ptr::addr_of_mut!(major) }
}

static __param_str_major: [u8; 13] = [100, 109, 95, 109, 111, 100, 46, 109, 97, 106, 111, 114, 0];

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_major: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_major) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_uint) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_dm_c_3838>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_dm_c_3838_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_dm_c_3838_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_dm_c_3838_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_dm_c_3838_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_dm_c_3838_6>() as u64) as i32).wrapping_add(0i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(major) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_891: [u8; 27] = [100, 109, 95, 109, 111, 100, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 109, 97, 106, 111, 114, 58, 117, 105, 110, 116, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_892: [u8; 56] = [100, 109, 95, 109, 111, 100, 46, 112, 97, 114, 109, 61, 109, 97, 106, 111, 114, 58, 84, 104, 101, 32, 109, 97, 106, 111, 114, 32, 110, 117, 109, 98, 101, 114, 32, 111, 102, 32, 116, 104, 101, 32, 100, 101, 118, 105, 99, 101, 32, 109, 97, 112, 112, 101, 114, 0];

fn __check_reserved_bio_based_ios() -> *mut u32 {
unsafe { core::ptr::addr_of_mut!(reserved_bio_based_ios) }
}

static __param_str_reserved_bio_based_ios: __SlateAlign16<[u8; 30]> = __SlateAlign16([100u8, 109u8, 95u8, 109u8, 111u8, 100u8, 46u8, 114u8, 101u8, 115u8, 101u8, 114u8, 118u8, 101u8, 100u8, 95u8, 98u8, 105u8, 111u8, 95u8, 98u8, 97u8, 115u8, 101u8, 100u8, 95u8, 105u8, 111u8, 115u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_reserved_bio_based_ios: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_reserved_bio_based_ios.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_uint) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_dm_c_3841>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_dm_c_3841_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_dm_c_3841_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_dm_c_3841_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_dm_c_3841_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_dm_c_3841_6>() as u64) as i32).wrapping_add(420i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(reserved_bio_based_ios) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_893: [u8; 44] = [100, 109, 95, 109, 111, 100, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 114, 101, 115, 101, 114, 118, 101, 100, 95, 98, 105, 111, 95, 98, 97, 115, 101, 100, 95, 105, 111, 115, 58, 117, 105, 110, 116, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_894: [u8; 70] = [100, 109, 95, 109, 111, 100, 46, 112, 97, 114, 109, 61, 114, 101, 115, 101, 114, 118, 101, 100, 95, 98, 105, 111, 95, 98, 97, 115, 101, 100, 95, 105, 111, 115, 58, 82, 101, 115, 101, 114, 118, 101, 100, 32, 73, 79, 115, 32, 105, 110, 32, 98, 105, 111, 45, 98, 97, 115, 101, 100, 32, 109, 101, 109, 112, 111, 111, 108, 115, 0];

fn __check_dm_numa_node() -> *mut i32 {
unsafe { core::ptr::addr_of_mut!(dm_numa_node) }
}

static __param_str_dm_numa_node: __SlateAlign16<[u8; 20]> = __SlateAlign16([100u8, 109u8, 95u8, 109u8, 111u8, 100u8, 46u8, 100u8, 109u8, 95u8, 110u8, 117u8, 109u8, 97u8, 95u8, 110u8, 111u8, 100u8, 101u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_dm_numa_node: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_dm_numa_node.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_int) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_dm_c_3844>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_dm_c_3844_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_dm_c_3844_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_dm_c_3844_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_dm_c_3844_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_dm_c_3844_6>() as u64) as i32).wrapping_add(420i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(dm_numa_node) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_895: [u8; 33] = [100, 109, 95, 109, 111, 100, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 100, 109, 95, 110, 117, 109, 97, 95, 110, 111, 100, 101, 58, 105, 110, 116, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_896: [u8; 68] = [100, 109, 95, 109, 111, 100, 46, 112, 97, 114, 109, 61, 100, 109, 95, 110, 117, 109, 97, 95, 110, 111, 100, 101, 58, 78, 85, 77, 65, 32, 110, 111, 100, 101, 32, 102, 111, 114, 32, 68, 77, 32, 100, 101, 118, 105, 99, 101, 32, 109, 101, 109, 111, 114, 121, 32, 97, 108, 108, 111, 99, 97, 116, 105, 111, 110, 115, 0];

fn __check_swap_bios() -> *mut i32 {
unsafe { core::ptr::addr_of_mut!(swap_bios) }
}

static __param_str_swap_bios: __SlateAlign16<[u8; 17]> = __SlateAlign16([100u8, 109u8, 95u8, 109u8, 111u8, 100u8, 46u8, 115u8, 119u8, 97u8, 112u8, 95u8, 98u8, 105u8, 111u8, 115u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_swap_bios: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_swap_bios.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_int) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_dm_c_3847>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_dm_c_3847_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_dm_c_3847_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_dm_c_3847_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_dm_c_3847_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_dm_c_3847_6>() as u64) as i32).wrapping_add(420i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(swap_bios) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_897: [u8; 30] = [100, 109, 95, 109, 111, 100, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 115, 119, 97, 112, 95, 98, 105, 111, 115, 58, 105, 110, 116, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_898: [u8; 56] = [100, 109, 95, 109, 111, 100, 46, 112, 97, 114, 109, 61, 115, 119, 97, 112, 95, 98, 105, 111, 115, 58, 77, 97, 120, 105, 109, 117, 109, 32, 97, 108, 108, 111, 119, 101, 100, 32, 105, 110, 102, 108, 105, 103, 104, 116, 32, 115, 119, 97, 112, 32, 73, 79, 115, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_899: [u8; 40] = [100, 109, 95, 109, 111, 100, 46, 100, 101, 115, 99, 114, 105, 112, 116, 105, 111, 110, 61, 100, 101, 118, 105, 99, 101, 45, 109, 97, 112, 112, 101, 114, 32, 100, 114, 105, 118, 101, 114, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_900: [u8; 54] = [100, 109, 95, 109, 111, 100, 46, 97, 117, 116, 104, 111, 114, 61, 74, 111, 101, 32, 84, 104, 111, 114, 110, 98, 101, 114, 32, 60, 100, 109, 45, 100, 101, 118, 101, 108, 64, 108, 105, 115, 116, 115, 46, 108, 105, 110, 117, 120, 46, 100, 101, 118, 62, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_901: [u8; 30] = [100, 109, 95, 109, 111, 100, 46, 102, 105, 108, 101, 61, 100, 114, 105, 118, 101, 114, 115, 47, 109, 100, 47, 100, 109, 45, 109, 111, 100, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_902: [u8; 19] = [100, 109, 95, 109, 111, 100, 46, 108, 105, 99, 101, 110, 115, 101, 61, 71, 80, 76, 0];

static __slate_asm_string___dm_internal_resume_12: [u8; 1] = *b"\0";

static __slate_asm_string___dm_internal_resume_13: [u8; 45] = *b"/home/takashi/c-corpus/linux/drivers/md/dm.c\0";

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string___map_bio_10: [u8; 45] = *b"/home/takashi/c-corpus/linux/drivers/md/dm.c\0";

static __slate_asm_string___map_bio_9: [u8; 1] = *b"\0";

static __slate_asm_string___send_duplicate_bios_1: [u8; 1] = *b"\0";

static __slate_asm_string___send_duplicate_bios_2: [u8; 45] = *b"/home/takashi/c-corpus/linux/drivers/md/dm.c\0";

static __slate_asm_string_alloc_dev_15: [u8; 1] = *b"\0";

static __slate_asm_string_alloc_dev_16: [u8; 45] = *b"/home/takashi/c-corpus/linux/drivers/md/dm.c\0";

static __slate_asm_string_clone_endio_10: [u8; 45] = *b"/home/takashi/c-corpus/linux/drivers/md/dm.c\0";

static __slate_asm_string_clone_endio_15: [u8; 1] = *b"\0";

static __slate_asm_string_clone_endio_16: [u8; 45] = *b"/home/takashi/c-corpus/linux/drivers/md/dm.c\0";

static __slate_asm_string_clone_endio_9: [u8; 1] = *b"\0";

static __slate_asm_string_dm_accept_partial_bio_0: [u8; 1] = *b"\0";

static __slate_asm_string_dm_accept_partial_bio_1: [u8; 45] = *b"/home/takashi/c-corpus/linux/drivers/md/dm.c\0";

static __slate_asm_string_dm_accept_partial_bio_10: [u8; 1] = *b"\0";

static __slate_asm_string_dm_accept_partial_bio_11: [u8; 45] = *b"/home/takashi/c-corpus/linux/drivers/md/dm.c\0";

static __slate_asm_string_dm_accept_partial_bio_15: [u8; 1] = *b"\0";

static __slate_asm_string_dm_accept_partial_bio_16: [u8; 45] = *b"/home/takashi/c-corpus/linux/drivers/md/dm.c\0";

static __slate_asm_string_dm_accept_partial_bio_23: [u8; 1] = *b"\0";

static __slate_asm_string_dm_accept_partial_bio_24: [u8; 45] = *b"/home/takashi/c-corpus/linux/drivers/md/dm.c\0";

static __slate_asm_string_dm_accept_partial_bio_28: [u8; 1] = *b"\0";

static __slate_asm_string_dm_accept_partial_bio_29: [u8; 45] = *b"/home/takashi/c-corpus/linux/drivers/md/dm.c\0";

static __slate_asm_string_dm_accept_partial_bio_33: [u8; 1] = *b"\0";

static __slate_asm_string_dm_accept_partial_bio_34: [u8; 45] = *b"/home/takashi/c-corpus/linux/drivers/md/dm.c\0";

static __slate_asm_string_dm_accept_partial_bio_38: [u8; 1] = *b"\0";

static __slate_asm_string_dm_accept_partial_bio_39: [u8; 45] = *b"/home/takashi/c-corpus/linux/drivers/md/dm.c\0";

static __slate_asm_string_dm_accept_partial_bio_5: [u8; 1] = *b"\0";

static __slate_asm_string_dm_accept_partial_bio_6: [u8; 45] = *b"/home/takashi/c-corpus/linux/drivers/md/dm.c\0";

static __slate_asm_string_dm_bio_from_per_bio_data_0: [u8; 1] = *b"\0";

static __slate_asm_string_dm_bio_from_per_bio_data_1: [u8; 45] = *b"/home/takashi/c-corpus/linux/drivers/md/dm.c\0";

static __slate_asm_string_dm_blk_close_6: [u8; 1] = *b"\0";

static __slate_asm_string_dm_blk_close_7: [u8; 45] = *b"/home/takashi/c-corpus/linux/drivers/md/dm.c\0";

static __slate_asm_string_dm_dax_zero_page_range_8: [u8; 1] = *b"\0";

static __slate_asm_string_dm_dax_zero_page_range_9: [u8; 45] = *b"/home/takashi/c-corpus/linux/drivers/md/dm.c\0";

static __slate_asm_string_dm_get_1: [u8; 1] = *b"\0";

static __slate_asm_string_dm_get_2: [u8; 45] = *b"/home/takashi/c-corpus/linux/drivers/md/dm.c\0";

static __slate_asm_string_dm_handle_requeue_7: [u8; 1] = *b"\0";

static __slate_asm_string_dm_handle_requeue_8: [u8; 45] = *b"/home/takashi/c-corpus/linux/drivers/md/dm.c\0";

static __slate_asm_string_dm_poll_bio_1: [u8; 1] = *b"\0";

static __slate_asm_string_dm_poll_bio_2: [u8; 45] = *b"/home/takashi/c-corpus/linux/drivers/md/dm.c\0";

static __slate_asm_string_dm_poll_dm_io_1: [u8; 1] = *b"\0";

static __slate_asm_string_dm_poll_dm_io_2: [u8; 45] = *b"/home/takashi/c-corpus/linux/drivers/md/dm.c\0";

static __slate_asm_string_dm_pr_clear_8: [u8; 1] = *b"\0";

static __slate_asm_string_dm_pr_clear_9: [u8; 45] = *b"/home/takashi/c-corpus/linux/drivers/md/dm.c\0";

static __slate_asm_string_dm_setup_md_queue_13: [u8; 1] = *b"\0";

static __slate_asm_string_dm_setup_md_queue_14: [u8; 45] = *b"/home/takashi/c-corpus/linux/drivers/md/dm.c\0";

static __slate_asm_string_dm_table_get_target_0: [u8; 1] = *b"\0";

static __slate_asm_string_dm_table_get_target_1: [u8; 50] = *b"/home/takashi/c-corpus/linux/drivers/md/dm-core.h\0";

static __slate_asm_string_free_dev_1: [u8; 1] = *b"\0";

static __slate_asm_string_free_dev_2: [u8; 45] = *b"/home/takashi/c-corpus/linux/drivers/md/dm.c\0";

static __slate_asm_string_init_clone_info_4: [u8; 1] = *b"\0";

static __slate_asm_string_init_clone_info_5: [u8; 45] = *b"/home/takashi/c-corpus/linux/drivers/md/dm.c\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

static __slate_asm_string_lock_fs_2: [u8; 1] = *b"\0";

static __slate_asm_string_lock_fs_3: [u8; 45] = *b"/home/takashi/c-corpus/linux/drivers/md/dm.c\0";

static __slate_asm_string_srcu_read_unlock_1: [u8; 1] = *b"\0";

static __slate_asm_string_srcu_read_unlock_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/srcu.h\0";

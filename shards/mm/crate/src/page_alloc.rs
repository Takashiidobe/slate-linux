
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
//! Manages the free list, the system allocates free pages here.
//! Note that kmalloc() lives in slab.c
//!
//! Copyright (C) 1991, 1992, 1993, 1994  Linus Torvalds
//! Swap reorganised 29.12.95, Stephen Tweedie
//! Support of BIGMEM added by Gerhard Wichert, Siemens AG, July 1999
//! Reshaped it to be a zoned allocator, Ingo Molnar, Red Hat, 1999
//! Discontiguous memory support, Kanoj Sarcar, SGI, Nov 1999
//! Zone balancing, Kanoj Sarcar, SGI, Jan 2000
//! Per cpu hot/cold page lists, bulk allocation, Martin J. Bligh, Sept 2002
//!         (lots of bits borrowed from Ingo Molnar & Andrew Morton)
//! __DECLARE_FLEX_ARRAY() - Declare a flexible array usable in a union
//!
//! @TYPE: The type of each flexible array element
//! @NAME: The name of the flexible array member
//!
//! In order to have a flexible array member in a union or alone in a
//! struct, it needs to be wrapped in an anonymous struct with at least 1
//! named member, but that member can be empty.
//! This counts to 15. Any more, it will return 16th argument.
//! Operation not permitted
//! Value too large for defined data type
//! restart by calling sys_restart_syscall
//! aligned_u64 should be used in defining kernel<->userspace ABIs to avoid
//! common 32/64-bit compat problems.
//! 64-bit values align to 4-byte boundaries on x86_32 (and possibly other
//! architectures) and to 8-byte boundaries on 64-bit architectures.  The new
//! aligned_64 type enforces 8-byte alignment so that structs containing
//! aligned_64 values have the same alignment on 32-bit and 64-bit architectures.
//! No conversions are necessary between 32-bit user-space and a 64-bit kernel.
//! These are the special 64-bit data types that are 8-byte aligned
//! See linux/instrumentation.h
//! Begin/end of an instrumentation safe region
//! 64 bit
//! unused, was: #define EX_TYPE_COPY	 4
//! Optimization barrier
//!
//! The "volatile" is due to gcc bugs
//! Yes, this permits 64-bit accesses on 32-bit architectures. These will
//! actually be atomic in some cases (namely Armv7 + LPAE), but for others we
//! rely on the access being split into 2x32-bit accesses for a 32-bit quantity
//! (e.g. a virtual address) and a strong prevailing wind.
//! Call a function once. Similar to DO_ONCE(), but does not use jump label
//! patching via static keys.
//! Force a compilation error if condition is true, but also produce a
//! result (of value 0 and type int), so the expression can be used
//! e.g. in a structure initializer (or where-ever else comma expressions
//! aren't permitted).
//!
//! Take an error message as an optional second argument. If omitted,
//! default to the stringification of the tested expression.
//! These are for everybody (although not all archs will actually
//! discard it in modules)
//! warning conditions
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
//! # chars in a path name including nul
//! Internal kernel timer frequency
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! issue num suppressed message on exit
//! strlen("ratelimit") + 1
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
//! label
//! Defines x86 CPU feature bits
//! 64bit FS base
//! Needs two entries
//! PAGE_SHIFT determines the page size.
//!
//! Note: This definition is required because PAGE_SHIFT is used
//! in several places throughout the codebase.
//! writeable
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
//! used to decode ioctl numbers..
//! nodemask_pr_args - printf args to output a nodemask
//! @maskp: nodemask to be printed
//!
//! Can be used to provide arguments for '%*pb[l]' when printing a nodemask.
//! Wrap the arch provided IRQ routines to provide appropriate checks.
//! The size of each LDT entry.
//! Ensure that leaf 0x2 cache and TLB type values do not intersect,
//! since they share the same type field at struct cpuid_0x2_table.
//! Instruction encoding provided for binutils backwards compatibility.
//! Defines VMX CPU feature bits
//! Parameters used to convert the timespec values:
//! AUX clock support. AUXiliary clocks are dynamically configured by
//! enabling a clock ID. These clock can be steered independently of the
//! core timekeeper. The kernel can support up to 8 auxiliary clocks, but
//! the actual limit depends on eventual architecture constraints vs. VDSO.
//! signal pending
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
//! New user namespace
//! These constants are defined in the MSan LLVM instrumentation pass.
//! ignore signal
//! bit-flags
//! Convenience macros for the sake of set_current_state:
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
//! Instruction opcode for WRMSRNS supported in binutils >= 2.40
//! Values for SRCU Tree srcu_data ->srcu_reader_flavor, but also used by rcutorture.
//! Suits me
//! percpu_counter batch for local add or sub
//! Uprobes on this MM assume 32-bit code
//! local_lock_init - Runtime initialize a lock instance
//! @lock:	The lock variable
//! PAGE_ALLOC_COSTLY_ORDER is the order at which allocations are deemed
//! costly to service.  That is between allocation orders which should
//! coalesce naturally under reasonable reclaim pressure and those which
//! will not.
//! If the distance between nodes in a system is larger than RECLAIM_DISTANCE
//! (in whatever arch specific measurement units returned by node_distance())
//! and node_reclaim_mode is enabled then the VM will only call node_reclaim()
//! on nodes within this distance.
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
//! Extract the high bits for type
//! Must be a compile-time constant, so implement it as a macro
//! native kernel pointers tag
//! Resource type
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
//! kmem_cache_alloc and friends return pointers aligned to ARCH_SLAB_MINALIGN.
//! kmalloc and friends return pointers aligned to both ARCH_KMALLOC_MINALIGN
//! and ARCH_SLAB_MINALIGN, but here we only assume the former alignment.
//! return the ->next pointer of a list_head in an rcu safe
//! way, we must not access it directly
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
//! Keep the same order as in fs/proc/proc_sysctl.c
//! return the first ptr or next element in an RCU protected list
//! Keep unconverted code working
//! used to be O_SYNC, see below
//! Valid hint values for F_{GET,SET}_RW_HINT. 0 is "not set", or can be
//! used to clear any hints previously set.
//! Gives us 8 prio classes with 13-bits of data for each class
//! per-IO O_SYNC
//! File supports atomic writes
//! number of env pointers
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
//! should be >= TASK_COMM_LEN
//! in linux/sched.h
//! hlist_nulls_first_rcu - returns the first element of the hash list.
//! @head: the head of the list.
//! define the enumeration of all cgroup subsystems
//! To avoid confusing the compiler (and generating warnings) with code
//! that attempts to access what would be a 0-element array (i.e. sized
//! to a potentially empty array when CGROUP_SUBSYS_COUNT == 0), this
//! constant expression can be added.
//! Generate SD flag indexes
//! 31 pointers + header align the folio_batch structure to a power of two
//! Internal flags that share the struct mempolicy flags word with
//! "mode flags".  These flags are allocated from bit 0 up, as they
//! are never OR'ed into the mode in mempolicy API arguments.
//! If more than 16 keys are ever supported, a thorough audit
//! will be necessary to ensure that the types that store key
//! numbers and masks have sufficient capacity.
//! Currently, the max observed size in the kernel code is
//! JUMP_LABEL_NOP_SIZE/RELATIVEJUMP_SIZE, which are 5.
//! Raise it if needed.
//! Set global bits for all 4 regs
//! Define bits that are always set to 1 in DR7, only bit 10 is
//! architecturally reserved to '1'.
//!
//! This is also the init/reset value for DR7.
//! Replace with "lkgs %di" once binutils support LKGS instruction
//! Macro to enforce the same ordering and stack sizes
//! swapcache pages are stored in the swapper_space radix tree.  We want to
//! get good packing density in that tree, so the index should be dense in
//! the low-order bits.
//!
//! We arrange the `type' and `offset' fields so that `type' is at the six
//! high-order bits of the swp_entry_t and `offset' is right-aligned in the
//! remaining bits.  Although `type' itself needs only five bits, we allow for
//! shmem/tmpfs to shift it all up a further one bit: see swp_to_radix_entry().
//!
//! swp_entry_t's are *never* stored anywhere in their arch-dependent format.
//! Temporary until swp_entry_t eliminated.
//! Watch out for PAE architecture, which has an unsigned long, and might not
//! have enough bits to store all physical address and flags. So far we have
//! enough room for all our flags.
//! modes for vm.memfd_noexec sysctl
//! These values are stored in task->ptrace_message
//! by ptrace_stop to describe the current syscall-stop.
//! SEIZE used, enable new behavior
//! These are specified by iBCS2
//! Epoll event masks
//! The ALLOC_WMARK bits are used as an index to zone->watermark
//! See the comment in asm-generic/io.h about ioremap_np().
//! These macros should be used after a dma_map_sg call has been done
//! to get bus addresses of each of the SG entries and their lengths.
//! You should only work with the number of sg entries dma_map_sg
//! returns, or alternatively stop on the first sg_dma_len(sg) which
//! is 0.
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
// DECLARE_FLEX_ARRAY() - Declare a flexible array usable in a union
//
// @TYPE: The type of each flexible array element
// @NAME: The name of the flexible array member
//
// In order to have a flexible array member in a union or alone in a
// struct, it needs to be wrapped in an anonymous struct with at least 1
// named member, but that member can be empty.
// Concatenate two parameters, but allow them to be expanded beforehand.
// No such file or directory
// No such device or address
// Argument list too long
// Try again
// Out of memory
// Bad address
// Device or resource busy
// No such device
// Invalid argument
// Text file busy
// No space left on device
// Operation not supported on transport endpoint
// Memory page has hardware error
// Operation is not supported
// objtool annotation to ignore the alternatives and only consider the original
// instruction(s).
// This should be used to refer to an instruction that is considered
// terminating, like a noreturn CALL or UD2 when we know they are not -- eg
// WARN using UD2.
// Annotate a special section entry.  This emables livepatch module generation
// to find and extract individual special section entries as needed.
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
// reg := -EIO
// reg := -EIO
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
// modpost check for section mismatches during the kernel build.
// A section mismatch happens when there are references from a
// code or data section to an init section (both code or data).
// The init sections are (for most archs) discarded by the kernel
// when early init has completed so all such references are potential bugs.
// For exit sections the same issue exists.
//
// The following markers are used for the cases where the reference to
// the *init / *exit section (code or data) is valid and will teach
// modpost not to issue a warning.  Intended semantics is that a code or
// data tagged __ref* can reference code or data from init section without
// producing a warning (of course, no warning does not mean code is
// correct, so optimally document why the __ref is needed and why it's OK).
//
// The markers follow same syntax rules as __init / __initdata.
// Format: <modname>__<counter>_<line>_<fn>
// Format: __<prefix>__<iid><id>
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
// __swahw32 - return a word-swapped 32-bit value
// @x: value to wordswap
//
// __swahw32(0x12340000) is 0x00001234
// __swahb32 - return a high and low byte-swapped 32-bit value
// @x: value to byteswap
//
// __swahb32(0x12345678) is 0x34127856
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
// DOC: Watermark modifiers
//
// Watermark modifiers -- controls access to emergency reserves
//
// %__GFP_HIGH indicates that the caller is high-priority and that granting
// the request is necessary before the system can make forward progress.
// For example creating an IO context to clean pages and requests
// from atomic context.
//
// %__GFP_MEMALLOC allows access to all memory. This should only be used when
// the caller guarantees the allocation will allow more memory to be freed
// very shortly e.g. process exiting or swapping. Users either should
// be the MM or co-ordinating closely with the VM (e.g. swap over NFS).
// Users of this flag have to be extremely careful to not deplete the reserve
// completely and implement a throttling mechanism which controls the
// consumption of the reserve based on the amount of freed memory.
// Usage of a pre-allocated pool (e.g. mempool) should be always considered
// before using this flag.
//
// %__GFP_NOMEMALLOC is used to explicitly forbid access to emergency reserves.
// This takes precedence over the %__GFP_MEMALLOC flag if both are set.
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
// kswapd can wake
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
// IS_ERR_VALUE - Detect an error pointer.
// @x: The pointer to check.
//
// Like IS_ERR(), but does not generate a compiler warning if result is unused.
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
// Overflow Flag
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
// N 32-bit words worth of info
// N 32-bit bug flags
// "tsc" Time Stamp Counter
// "sse"
// "rdtscp" RDTSCP
// "rep_good" REP microcode works well
// "popcnt" POPCNT instruction
// "hypervisor" Running on a hypervisor
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
// Like MOV_GS except MSR_KERNEL_GS_BASE = GS.base
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
// Needs two entries
// Number of entries in the GDT table:
// Bit size and mask of CPU number stored in the per CPU data (and TSC_AUX)
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
// Atomic operations are already serializing on x86
// min - return minimum of two values of the same or compatible types
// @x: first value
// @y: second value
// max - return maximum of two values of the same or compatible types
// @x: first value
// @y: second value
// max3 - return maximum of three values
// @x: first value
// @y: second value
// @z: third value
// clamp - return a value clamped to a given range with typechecking
// @val: current value
// @lo: lowest allowable value
// @hi: highest allowable value
//
// This macro checks @val/@lo/@hi to make sure they have compatible
// signedness.
// clamp_t - return a value clamped to a given range using a given type
// @type: the type of variable to use
// @val: current value
// @lo: minimum allowable value
// @hi: maximum allowable value
//
// This macro does no typechecking and uses temporary variables of type
// @type to make all the comparisons.
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
// is terminating due to OOM killer
// rescheduling necessary
// idle is polling for TIF_NEED_RESCHED
// re-arm the timer
// This function is needed by all SMP systems. It must _always_ be valid
// from the initial startup.
// For trivial one-depth nesting of a lock-class, the following
// global define can be used. (Subsystems with multiple levels
// of nesting should define their own lock-nesting subclasses.)
// Writer mask
// Remapping rwlock architecture specific functions to the corresponding
// queued rwlock functions.
// Huge pages are a constant size, but don't exceed the maximum allocation
// granularity.
// SCHED_ISO: reserved but not implemented yet
// disable sas during sighandling
// Same as rb_first(), but O(1)
// Don't call further
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
// Shift up (to get rid of type), then down to get value
// Shift the offset up "too far" by TYPE bits, then down again
// The offset is inverted by a binary not operation to make the high
// physical bits set.
// Currently stuck as a macro due to indirect forward reference to
// linux/mmzone.h's __section_mem_map_addr() definition:
// All top-level MITIGATION_PAGE_TABLE_ISOLATION page tables are order-1 pages
// (8k-aligned and 8k in size).  The kernel one is at the beginning 4k and
// the user one is in the last 4k.  To switch between them, you
// just need to flip the 12th bit in their addresses.
// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
// Resource extended types
// No address assigned yet
// SLUB directly allocates requests fitting in to an order-1 page
// (PAGE_SIZE*2).  Larger requests are passed to the page allocator.
// Maximum size for which we actually use a slab cache
// Define gfp bits that should not be set for KMALLOC_NORMAL.
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
// Implementation specific max size
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
// Generate SD flag bits
// identify shared policies
// Enabling zone reclaim means the page allocator will attempt to fulfill
// the allocation request on the current node by triggering reclaim and
// trying to shrink the current node.
// Fallback allocations on the next candidates in the zonelist are considered
// when reclaim fails to free up enough memory in the current node/zone.
//
// These bit locations are exposed in the vm.zone_reclaim_mode sysctl.
// New bits are OK, but existing bits should not be changed.
// Enable zone reclaim
// Writeout pages during reclaim
// Unmap pages during reclaim
// These special macros can be used to get or set a debugging register
// "Poisoned" here is meant in the very general sense of "future accesses are
// invalid", instead of referring very specifically to hardware memory errors.
// This marker is meant to represent any of various different causes of this.
//
// Note that, when encountered by the faulting logic, PTEs with this marker will
// result in VM_FAULT_HWPOISON and thus regardless trigger hardware memory error
// logic.
// Indicates that, on fault, this PTE will case a SIGSEGV signal to be
// sent. This means guard markers behave in effect as if the region were mapped
// PROT_NONE, rather than if they were a memory hole or equivalent.
// PT_TRACE_* event enable flags
// We overload the LSB of the page pointer to indicate whether it's
// a valid sg entry, or whether it points to the start of a new scatterlist.
// Those low bits are there for everyone! (thanks mason :-)
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
macro_rules! ALIGN_DOWN {
($x:expr, $a:expr) => {
__ALIGN_KERNEL!($x.wrapping_sub($a.wrapping_sub(1)), $a)
};
}

/// @a is a power of 2 value
macro_rules! ALIGN {
($x:expr, $a:expr) => {
__ALIGN_KERNEL!($x, $a)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__15 {
() => {
((core::mem::size_of::<[ctl_table; 9]>() as u64) / (core::mem::size_of::<ctl_table>() as u64)).wrapping_add(__must_be_array__76!() as u64)
};
}

/// ASSERT_EXCLUSIVE_BITS - assert no concurrent writes to subset of bits in @var
///
/// Bit-granular variant of ASSERT_EXCLUSIVE_WRITER().
///
/// Assert that there are no concurrent writes to a subset of bits in @var;
/// concurrent readers are permitted. This assertion captures more detailed
/// bit-level properties, compared to the other (word granularity) assertions.
/// Only the bits set in @mask are checked for concurrent modifications, while
/// ignoring the remaining bits, i.e. concurrent writes (or reads) to ~mask bits
/// are ignored.
///
/// Use this for variables, where some bits must not be modified concurrently,
/// yet other bits are expected to be modified concurrently.
///
/// For example, variables where, after initialization, some bits are read-only,
/// but other bits may still be modified concurrently. A reader may wish to
/// assert that this is true as follows:
///
/// .. code-block:: c
///
/// 	ASSERT_EXCLUSIVE_BITS(flags, READ_ONLY_MASK);
/// 	foo = (READ_ONCE(flags) & READ_ONLY_MASK) >> READ_ONLY_SHIFT;
///
/// Note: The access that immediately follows ASSERT_EXCLUSIVE_BITS() is assumed
/// to access the masked bits only, and KCSAN optimistically assumes it is
/// therefore safe, even in the presence of data races, and marking it with
/// READ_ONCE() is optional from KCSAN's point-of-view. We caution, however, that
/// it may still be advisable to do so, since we cannot reason about all compiler
/// optimizations when it comes to bit manipulations (on the reader and writer
/// side). If you are sure nothing can go wrong, we can write the above simply
/// as:
///
/// .. code-block:: c
///
/// 	ASSERT_EXCLUSIVE_BITS(flags, READ_ONLY_MASK);
/// 	foo = (flags & READ_ONLY_MASK) >> READ_ONLY_SHIFT;
///
/// Another example, where this may be used, is when certain bits of @var may
/// only be modified when holding the appropriate lock, but other bits may still
/// be modified concurrently. Writers, where other bits may change concurrently,
/// could use the assertion as follows:
///
/// .. code-block:: c
///
/// 	spin_lock(&foo_lock);
/// 	ASSERT_EXCLUSIVE_BITS(flags, FOO_MASK);
/// 	old_flags = flags;
/// 	new_flags = (old_flags & ~FOO_MASK) | (new_foo << FOO_SHIFT);
/// 	if (cmpxchg(&flags, old_flags, new_flags) != old_flags) { ... }
/// 	spin_unlock(&foo_lock);
///
/// @var: variable to assert on
/// @mask: only check for modifications to bits set in @mask
macro_rules! ASSERT_EXCLUSIVE_BITS {
($var:expr, $mask:expr) => {
loop {
kcsan_set_access_mask($mask);
__kcsan_check_access((unsafe { core::ptr::addr_of!($var) }) as *const (), core::mem::size_of::<u64>() as u64, 8i32);
kcsan_set_access_mask(0u64);
kcsan_atomic_next(1i32);
if true {
break;
}
}
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

/// BUILD_BUG_ON_INVALID() permits the compiler to check the validity of the
/// expression but avoids the generation of any code, even if that expression
/// has side-effects.
macro_rules! BUILD_BUG_ON_INVALID {
() => {
};
}

macro_rules! GENMASK_INPUT_CHECK {
() => {
core::mem::size_of::<__SlateRecord_find_h_67>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__4 {
() => {
core::mem::size_of::<__SlateRecord_page_flags_h_276>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__2 {
() => {
core::mem::size_of::<__SlateRecord_find_h_207>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__11 {
() => {
core::mem::size_of::<__SlateRecord_mm_h_5491>() as i32
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
macro_rules! GENMASK_TYPE__4 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__4!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
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
macro_rules! GENMASK_TYPE__15 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__11!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub($h))
};
}

macro_rules! GENMASK__10 {
($h:expr, $l:expr) => {
GENMASK_TYPE__15!(u64, $h, $l)
};
}

macro_rules! GENMASK {
($h:expr, $l:expr) => {
GENMASK_TYPE!(u64, $h, $l)
};
}

macro_rules! GENMASK__3 {
($h:expr, $l:expr) => {
GENMASK_TYPE__4!(u64, $h, $l)
};
}

macro_rules! GENMASK__2 {
($h:expr, $l:expr) => {
GENMASK_TYPE__2!(u64, $h, $l)
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

macro_rules! MIN_T__2 {
($type:ty, $a:expr, $b:expr) => {
if (($a) as $type) < (($b) as $type) { ($a) as $type } else { ($b) as $type }
};
}

macro_rules! MIN_T {
($type:ty, $a:expr) => {
($a) as $type
};
}

macro_rules! NODE_DATA {
($nid:expr) => {
unsafe { *(core::ptr::addr_of_mut!(node_data) as *mut *mut pglist_data).wrapping_offset(($nid) as isize) }
};
}

/// to align the pointer to the (next) page boundary
macro_rules! PAGE_ALIGN {
($addr:expr) => {
ALIGN!($addr, 4096u64)
};
}

macro_rules! PFN_PHYS {
($x:expr) => {
(($x) as u64) << 12i32
};
}

/// Page flags policies wrt compound pages
///
/// PF_POISONED_CHECK
///     check if this struct page poisoned/uninitialized
///
/// PF_ANY:
///     the page flag is relevant for small, head and tail pages.
///
/// PF_HEAD:
///     for compound page all operations related to the page flag applied to
///     head page.
///
/// PF_NO_TAIL:
///     modifications of the page flag must be done on small or head pages,
///     checks can be done on tail pages too.
///
/// PF_NO_COMPOUND:
///     the page flag is not relevant for compound pages.
///
/// PF_SECOND:
///     the page flag is stored in the first tail page.
macro_rules! PF_POISONED_CHECK {
($page:expr) => {
{
let mut __v: *const page;
VM_BUG_ON_PGFLAGS!();
__v = $page;
__v
}
};
}

/// Page flags policies wrt compound pages
///
/// PF_POISONED_CHECK
///     check if this struct page poisoned/uninitialized
///
/// PF_ANY:
///     the page flag is relevant for small, head and tail pages.
///
/// PF_HEAD:
///     for compound page all operations related to the page flag applied to
///     head page.
///
/// PF_NO_TAIL:
///     modifications of the page flag must be done on small or head pages,
///     checks can be done on tail pages too.
///
/// PF_NO_COMPOUND:
///     the page flag is not relevant for compound pages.
///
/// PF_SECOND:
///     the page flag is stored in the first tail page.
macro_rules! PF_POISONED_CHECK__2 {
($page:expr) => {
let mut __v: *const page;
VM_BUG_ON_PGFLAGS!();
__v = $page;
};
}

macro_rules! PHYS_PFN {
($x:expr) => {
($x) >> 12i32
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

macro_rules! SECTION_NR_TO_ROOT {
($sec:expr) => {
($sec) / ((_AC!() << 12i32) / (core::mem::size_of::<mem_section>() as u64))
};
}

macro_rules! UL {
() => {
_UL!()
};
}

macro_rules! VM_BUG_ON_PAGE {
() => {
VM_BUG_ON!();
};
}

macro_rules! VM_BUG_ON_PGFLAGS {
() => {
BUILD_BUG_ON_INVALID!();
};
}

macro_rules! VM_BUG_ON {
() => {
BUILD_BUG_ON_INVALID!();
};
}

macro_rules! VM_WARN_ON_FOLIO {
() => {
BUILD_BUG_ON_INVALID!();
};
}

macro_rules! VM_WARN_ON_ONCE {
() => {
BUILD_BUG_ON_INVALID!();
};
}

macro_rules! VM_WARN_ON_PAGE {
() => {
BUILD_BUG_ON_INVALID!();
};
}

macro_rules! VM_WARN_ON {
() => {
BUILD_BUG_ON_INVALID!();
};
}

macro_rules! _AC {
() => {
1u64
};
}

macro_rules! _AC__2 {
() => {
18446744071562067968u64
};
}

macro_rules! _AC__3 {
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

macro_rules! __ALIGN_KERNEL {
($x:expr, $a:expr) => {
__ALIGN_KERNEL_MASK!($x, (($a) as u64).wrapping_sub(1))
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

macro_rules! __WRITE_ONCE__8 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut u64, ($val) as u64) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE__3 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut i32, $val) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE__32 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut *mut capture_control, $val) };
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

macro_rules! __WRITE_ONCE__33 {
($x:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut *mut capture_control, core::ptr::null_mut::<capture_control>()) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE__4 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut u64, $val) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE__22 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut i32, ($val) as i32) };
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

macro_rules! __clear_bit {
($nr:expr, $addr:expr) => {
let __v: bool = true && false && (($addr) as u64) != 0;
if __v && false {
generic___clear_bit(($nr) as u64, ($addr) as *mut u64);
} else {
___clear_bit(($nr) as u64, ($addr) as *mut u64);
}
};
}

macro_rules! __cmp_once__6 {
($type:ty, $x:expr, $y:expr) => {
{
let mut __v: $type;
let __v2: $type = ($x) as $type;
let __v3: $type = $y;
__v = if __v2 > __v3 { __v2 } else { __v3 };
__v
}
};
}

macro_rules! __cmp_once {
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

macro_rules! __cmp_once__5 {
($type:ty, $x:expr, $y:expr) => {
{
let mut __v: $type;
let __v2: $type = $x;
let __v3: $type = $y;
__v = if __v2 > __v3 { __v2 } else { __v3 };
__v
}
};
}

macro_rules! __cmp_once__2 {
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

macro_rules! __must_be_array__76 {
() => {
core::mem::size_of::<__SlateRecord_page_alloc_c_7018>() as i32
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__4 {
() => {
((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__22 {
() => {
((const { core::mem::size_of::<*mut llist_node>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut llist_node>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut llist_node>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut llist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__19 {
() => {
((const { core::mem::size_of::<*mut mem_section_usage>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut mem_section_usage>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut mem_section_usage>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut mem_section_usage>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__6 {
() => {
((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__3 {
() => {
((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__10 {
() => {
((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__2 {
() => {
((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__9 {
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

macro_rules! __pa {
($x:expr) => {
__phys_addr!(($x) as u64)
};
}

macro_rules! __phys_addr {
($x:expr) => {
__phys_addr_nodebug($x)
};
}

macro_rules! __ratelimit__3 {
($state:expr) => {
unsafe { ___ratelimit($state, c"calculate_min_free_kbytes".as_ptr() as *const u8) }
};
}

macro_rules! __ratelimit__8 {
($state:expr) => {
unsafe { ___ratelimit($state, c"warn_alloc".as_ptr() as *const u8) }
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

/// Are two types/vars the same type (ignoring qualifiers)?
macro_rules! __same_type {
() => {
0i32
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
macro_rules! __sign_use__2 {
($ux:expr) => {
if is_signed_type__2!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul(((core::mem::size_of::<i32>() as u64) < 4) as i32)) }
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
if is_signed_type__3!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul((((const { core::mem::size_of::<u32>() }) as u64) < 4) as i32)) }
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
macro_rules! __sign_use__4 {
($ux:expr) => {
if is_signed_type!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul(((core::mem::size_of::<u64>() as u64) < 4) as i32)) }
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
macro_rules! __sign_use__6 {
($ux:expr) => {
if is_signed_type__4!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul(((core::mem::size_of::<i64>() as u64) < 4) as i32)) }
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
2i32.wrapping_add(__is_nonneg!($ux) as i32)
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

macro_rules! __types_ok3__3 {
($ux:expr, $uy:expr, $uz:expr) => {
__sign_use__3!($ux) & __sign_use__3!($uy) & __sign_use__3!($uz)
};
}

macro_rules! __types_ok3__2 {
($ux:expr, $uy:expr, $uz:expr) => {
__sign_use__2!($ux) & __sign_use__2!($uy) & __sign_use__2!($uz)
};
}

macro_rules! __types_ok__5 {
($ux:expr, $uy:expr) => {
__sign_use__5!($ux) & __sign_use__5!($uy)
};
}

macro_rules! __types_ok {
($ux:expr, $uy:expr) => {
__sign_use__2!($ux) & __sign_use__2!($uy)
};
}

macro_rules! __types_ok__4 {
($ux:expr, $uy:expr) => {
__sign_use__3!($ux) & __sign_use__3!($uy)
};
}

macro_rules! __types_ok__2 {
($ux:expr, $uy:expr) => {
__sign_use!($ux) & __sign_use!($uy)
};
}

macro_rules! __types_ok__3 {
($ux:expr, $uy:expr) => {
__sign_use__2!($ux) & __sign_use__4!($uy)
};
}

macro_rules! __types_ok__6 {
($ux:expr, $uy:expr) => {
__sign_use__6!($ux) & __sign_use__6!($uy)
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
macro_rules! __verify_pcpu_ptr__3 {
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
macro_rules! __verify_pcpu_ptr__13 {
() => {
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<per_cpu_pages>() as *const ();
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
macro_rules! __verify_pcpu_ptr__4 {
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
macro_rules! __verify_pcpu_ptr__14 {
() => {
loop {
let __v: *const () = core::ptr::null_mut::<per_cpu_zonestat>() as *const ();
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
macro_rules! __verify_pcpu_ptr__17 {
() => {
loop {
let __v: *const () = core::ptr::null_mut::<per_cpu_pages>() as *const ();
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

macro_rules! _static_cpu_has {
($bit:expr) => {
__static_cpu_has__const1(($bit) as u16) as i32
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

macro_rules! compound_head__2 {
($page:expr) => {
_compound_head(($page) as *const page) as *mut page
};
}

macro_rules! cond_resched__3 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/mm/page_alloc.c".as_ptr() as *const u8, 4457i32, 0u32);
__v = _cond_resched();
};
}

macro_rules! cond_resched__13 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/mm/page_alloc.c".as_ptr() as *const u8, 5081i32, 0u32);
__v = _cond_resched();
};
}

macro_rules! cond_resched__21 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/mm/page_alloc.c".as_ptr() as *const u8, 4707i32, 0u32);
__v = _cond_resched();
};
}

macro_rules! cond_resched__74 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/mm/page_alloc.c".as_ptr() as *const u8, 4242i32, 0u32);
__v = _cond_resched();
};
}

macro_rules! cond_resched__82 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/mm/page_alloc.c".as_ptr() as *const u8, 5326i32, 0u32);
__v = _cond_resched();
};
}

macro_rules! cond_resched__88 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/mm/page_alloc.c".as_ptr() as *const u8, 4444i32, 0u32);
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
macro_rules! container_of__8 {
($ptr:expr) => {
{
let mut __v: *mut page;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof!() as isize)) }) as *mut ()) as *mut page;
__v
}
};
}

/// This is the default CPU features testing macro to use in code.
///
/// It is for detection of features which need kernel infrastructure to be
/// used.  It may *not* directly test the CPU itself.  Use the cpu_has() family
/// if you want true runtime testing of CPU features, like in hypervisor code
/// where you are supporting a possible guest feature where host support for it
/// is not relevant.
macro_rules! cpu_feature_enabled {
($bit:expr) => {
_static_cpu_has!($bit)
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

/// data_race - mark an expression as containing intentional data races
///
/// This data_race() macro is useful for situations in which data races
/// should be forgiven.  One example is diagnostic code that accesses
/// shared variables but is not a part of the core synchronization design.
/// For example, if accesses to a given variable are protected by a lock,
/// except for diagnostic code, then the accesses under the lock should
/// be plain C-language accesses and those in the diagnostic code should
/// use data_race().  This way, KCSAN will complain if buggy lockless
/// accesses to that variable are introduced, even if the buggy accesses
/// are protected by READ_ONCE() or WRITE_ONCE().
///
/// This macro *does not* affect normal code generation, but is a hint
/// to tooling that data races here are to be ignored.  If the access must
/// be atomic *and* KCSAN should ignore the access, use both data_race()
/// and READ_ONCE(), for example, data_race(READ_ONCE(x)).
macro_rules! data_race {
($expr:expr) => {
{
let mut __v: u32;
__kcsan_disable_current();
let __v2: u32 = $expr;
__kcsan_enable_current();
__v = __v2;
__v
}
};
}

macro_rules! div64_ul {
($x:expr, $y:expr) => {
div64_u64($x, $y)
};
}

macro_rules! first_node {
($src:expr) => {
__first_node((unsafe { core::ptr::addr_of_mut!($src) }) as *const __SlateRecord_nodemask_types_h_17)
};
}

macro_rules! free_page__2 {
($addr:expr) => {
free_pages($addr, 0u32);
};
}

macro_rules! get_pageblock_migratetype__2 {
($page:expr) => {
get_pfnblock_migratetype(($page) as *const page, ((unsafe { $page.offset_from(vmemmap_base as *mut page) }) as i64) as u64)
};
}

macro_rules! hardirq_count {
() => {
(preempt_count() as u64) & __IRQ_MASK!(4i32) << 24i32
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

/// ilog2 - log base 2 of 32-bit or a 64-bit unsigned value
/// @n: parameter
///
/// constant-capable log of base 2 calculation
/// - this can be used to initialise global variables from constant data, hence
/// the massive ternary operator construction
///
/// selects the appropriately-sized optimised version depending on sizeof(n)
macro_rules! ilog2 {
($n:expr) => {
if unsafe { __slate_intrinsic_f217c4720f04f6db($n) } { if ($n) < 2 { 0i32 } else { 63i32.wrapping_sub($n.leading_zeros() as i32) } } else { if (core::mem::size_of::<u64>() as u64) <= 4 { __ilog2_u32(($n) as u32) } else { __ilog2_u64($n) } }
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
macro_rules! ilog2__7 {
($n:expr) => {
63i32.wrapping_sub((($n) as u64).leading_zeros() as i32)
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
if true { if ($n) < 2 { 0i32 } else { 63i32.wrapping_sub($n.leading_zeros() as i32) } } else { if (core::mem::size_of::<u64>() as u64) <= 4 { __ilog2_u32(($n) as u32) } else { __ilog2_u64($n) } }
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
macro_rules! ilog2__8 {
($n:expr) => {
if unsafe { __slate_intrinsic_a39e427d3b6c2df4($n) } { if ($n) < 2 { 0i32 } else { 63i32.wrapping_sub((($n) as u64).leading_zeros() as i32) } } else { if (core::mem::size_of::<i32>() as u64) <= 4 { __ilog2_u32(($n) as u32) } else { __ilog2_u64(($n) as u64) } }
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
macro_rules! ilog2__3 {
($n:expr) => {
if true { if ($n) < 2 { 0i32 } else { 63i32.wrapping_sub((($n) as u64).leading_zeros() as i32) } } else { if (core::mem::size_of::<i32>() as u64) <= 4 { __ilog2_u32(($n) as u32) } else { __ilog2_u64(($n) as u64) } }
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
macro_rules! ilog2__5 {
($n:expr) => {
if unsafe { __slate_intrinsic_f217c4720f04f6db($n) } { if ($n) < 2 { 0i32 } else { 63i32.wrapping_sub($n.leading_zeros() as i32) } } else { __ilog2_u64($n) }
};
}

macro_rules! in_hardirq {
() => {
hardirq_count!()
};
}

macro_rules! in_interrupt {
() => {
irq_count!()
};
}

/// Macros to retrieve the current execution context:
///
/// in_nmi()		- We're in NMI context
/// in_hardirq()		- We're in hard IRQ context
/// in_serving_softirq()	- We're in softirq context
/// in_task()		- We're in task context
macro_rules! in_nmi {
() => {
nmi_count!()
};
}

macro_rules! in_serving_softirq {
() => {
softirq_count!() & 256u64
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

macro_rules! is_migrate_cma {
() => {
0i32
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type__3 {
() => {
(1i32.wrapping_neg() as u32) < 1
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type__4 {
() => {
(1i32.wrapping_neg() as i64) < 1
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type__2 {
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

/// kcsan_check_write - check regular write access for races
///
/// @ptr: address of access
/// @size: size of access
macro_rules! kcsan_check_write {
($ptr:expr, $size:expr) => {
kcsan_check_access(absolute_pointer!($ptr) as *const (), $size, 1i32);
};
}

macro_rules! kcsan_mb {
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

/// Subtract two ktime_t variables. rem = lhs -rhs:
macro_rules! ktime_sub {
($lhs:expr, $rhs:expr) => {
$lhs.wrapping_sub($rhs)
};
}

macro_rules! likely {
($x:expr) => {
!(!$x) as i64
};
}

macro_rules! likely__6 {
($x:expr) => {
!$x.is_null() as i64
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

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__9 {
($ptr:expr, $type:ty) => {
list_entry__2!(unsafe { (*$ptr).next }, $type)
};
}

/// list_last_entry - get the last element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_last_entry__6 {
($ptr:expr, $type:ty) => {
list_entry__2!(unsafe { (*$ptr).prev }, $type)
};
}

/// llist_entry - get the struct of this entry
/// @ptr:	the &struct llist_node pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the llist_node within the struct.
macro_rules! llist_entry__4 {
($ptr:expr) => {
container_of__8!($ptr)
};
}

macro_rules! local_irq_restore {
($flags:expr) => {
loop {
raw_local_irq_restore!($flags);
if true {
break;
}
}
};
}

macro_rules! local_irq_save {
($flags:expr) => {
loop {
raw_local_irq_save!($flags);
if true {
break;
}
}
};
}

macro_rules! lock_acquire {
() => {
while false {
}
};
}

macro_rules! lock_acquire_exclusive {
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

macro_rules! lockdep_assert_held {
($l:expr) => {
__assume_ctx_lock!($l);
};
}

/// max_t - return maximum of two values, using the specified type
/// @type: data type to use
/// @x: first value
/// @y: second value
macro_rules! max_t {
($type:ty, $x:expr, $y:expr) => {
__cmp_once__6!($type, $x, $y)
};
}

/// max_t - return maximum of two values, using the specified type
/// @type: data type to use
/// @x: first value
/// @y: second value
macro_rules! max_t__2 {
($type:ty, $x:expr, $y:expr) => {
__cmp_once__5!($type, $x, $y)
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
macro_rules! member_address_is_nonnull__3 {
($ptr:expr) => {
(($ptr) as u64).wrapping_add(offsetof!()) != 0
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
macro_rules! min_t__2 {
($type:ty, $x:expr, $y:expr) => {
__cmp_once__2!($type, $x, $y)
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

/// Calculate "x * n / d" without unnecessary overflow or loss of precision.
macro_rules! mult_frac {
($x:expr, $n:expr, $d:expr) => {
{
let mut __v: u64;
let __v2: u64 = $x;
let __v3: i32 = $n;
let __v4: i32 = $d;
let __v5: u64 = __v2 / (__v4 as u64);
let __v6: u64 = __v2 % (__v4 as u64);
__v = __v5.wrapping_mul(__v3 as u64).wrapping_add(__v6.wrapping_mul(__v3 as u64) / (__v4 as u64));
__v
}
};
}

macro_rules! next_node {
($n:expr, $src:expr) => {
__next_node($n, (unsafe { core::ptr::addr_of_mut!($src) }) as *const __SlateRecord_nodemask_types_h_17)
};
}

macro_rules! nmi_count {
() => {
(preempt_count() as u64) & __IRQ_MASK!(1i32.wrapping_add(3i32.wrapping_mul(IS_ENABLED__2!()))) << 28i32
};
}

macro_rules! node_distance {
($a:expr, $b:expr) => {
unsafe { __node_distance($a, $b) }
};
}

/// No static inline type checking - see Subtlety (1) above.
macro_rules! node_isset__2 {
($node:expr, $nodemask:expr) => {
test_bit__2!($node, unsafe { $nodemask.bits.as_mut_ptr() as *mut u64 })
};
}

/// No static inline type checking - see Subtlety (1) above.
macro_rules! node_isset {
($node:expr, $nodemask:expr) => {
test_bit__2!($node, unsafe { core::ptr::addr_of_mut!($nodemask.bits) as *mut u64 })
};
}

macro_rules! node_online {
($node:expr) => {
node_state($node, 1u32)
};
}

/// The inline keyword gives the compiler room to decide to inline, or
/// not inline a function as it sees best.  However, as these functions
/// are called in both __init and non-__init functions, if they are not
/// inlined we will end up with a section mismatch error (of the type of
/// freeable items not being freed).  So we must use __always_inline here
/// to fix the problem.  If other functions in the future also end up in
/// this situation they will also need to be annotated as __always_inline
macro_rules! node_set {
($node:expr, $dst:expr) => {
__node_set($node, (unsafe { core::ptr::addr_of_mut!($dst) }) as *mut __SlateRecord_nodemask_types_h_17);
};
}

macro_rules! offsetof {
() => {
8u64
};
}

/// order_base_2 - calculate the (rounded up) base 2 order of the argument
/// @n: parameter
///
/// The first few values calculated by this routine:
///  ob2(0) = 0
///  ob2(1) = 0
///  ob2(2) = 1
///  ob2(3) = 2
///  ob2(4) = 2
///  ob2(5) = 3
///  ... and so on.
macro_rules! order_base_2 {
($n:expr) => {
if true { if ($n) == 0 || ($n) == 1 { 0i32 } else { ilog2__2!($n.wrapping_sub(1)).wrapping_add(1) } } else { __order_base_2($n) }
};
}

macro_rules! page_address__2 {
($page:expr) => {
lowmem_page_address($page)
};
}

macro_rules! page_address {
($page:expr) => {
lowmem_page_address(($page) as *const page)
};
}

/// page_folio - Converts from page to folio.
/// @p: The page.
///
/// Every page is part of a folio.  This function cannot be called on a
/// NULL pointer.
///
/// Context: No reference, nor lock is required on @page.  If the caller
/// does not hold a reference, this call may race with a folio split, so
/// it should re-check the folio still contains this page after gaining
/// a reference on the folio.
/// Return: The folio which contains this page.
macro_rules! page_folio {
($p:expr) => {
_compound_head(($p) as *const page) as *mut folio
};
}

/// page_folio - Converts from page to folio.
/// @p: The page.
///
/// Every page is part of a folio.  This function cannot be called on a
/// NULL pointer.
///
/// Context: No reference, nor lock is required on @page.  If the caller
/// does not hold a reference, this call may race with a folio split, so
/// it should re-check the folio still contains this page after gaining
/// a reference on the folio.
/// Return: The folio which contains this page.
macro_rules! page_folio__2 {
($p:expr) => {
_compound_head($p) as *const folio
};
}

/// page_private can be used on tail pages.  However, PagePrivate is only
/// checked by the VM on the head page.  So page_private on the tail pages
/// should be used for data that's ancillary to the head page (eg attaching
/// buffer heads to tail pages after attaching buffer heads to the head page)
macro_rules! page_private {
($page:expr) => {
unsafe { (*$page).__slate_anon_1.__slate_anon_0.private }
};
}

macro_rules! page_ref_tracepoint_active {
() => {
0i32
};
}

macro_rules! page_reported {
() => {
0i32
};
}

macro_rules! page_to_phys {
($page:expr) => {
PFN_PHYS!(((unsafe { $page.offset_from(vmemmap_base as *mut page) }) as i64) as u64)
};
}

macro_rules! page_to_virt {
($x:expr) => {
__va!(PFN_PHYS!(((unsafe { $x.offset_from((vmemmap_base as *mut page) as *const page) }) as i64) as u64))
};
}

macro_rules! pageblock_end_pfn__2 {
($pfn:expr) => {
ALIGN!($pfn.wrapping_add(1), 1u64 << if 9u32 < 10 { 9u32 } else { 10u32 })
};
}

macro_rules! pageblock_start_pfn__2 {
($pfn:expr) => {
ALIGN_DOWN!($pfn, 1u64 << if 9u32 < 10 { 9u32 } else { 10u32 })
};
}

/// In some cases we do not need to pin the task to the CPU because we are
/// already given a specific cpu's pcp pointer.
macro_rules! pcp_spin_lock_nopin {
($ptr:expr) => {
spin_lock(unsafe { core::ptr::addr_of_mut!((*$ptr).lock) });
};
}

macro_rules! pcp_spin_unlock_nopin {
($ptr:expr) => {
spin_unlock(unsafe { core::ptr::addr_of_mut!((*$ptr).lock) });
};
}

macro_rules! per_cpu_offset {
($x:expr) => {
unsafe { __per_cpu_offset[($x) as usize] }
};
}

macro_rules! per_cpu_ptr__2 {
($ptr:expr, $cpu:expr) => {
{
let mut __v: *mut per_cpu_pages;
__verify_pcpu_ptr__17!();
__v = (((($ptr) as u64) as *mut per_cpu_pages) as u64).wrapping_add(per_cpu_offset!($cpu)) as *mut per_cpu_pages;
__v
}
};
}

macro_rules! per_cpu_ptr {
($ptr:expr, $cpu:expr) => {
{
let mut __v: *mut per_cpu_zonestat;
__verify_pcpu_ptr__14!();
__v = (((($ptr) as u64) as *mut per_cpu_zonestat) as u64).wrapping_add(per_cpu_offset!($cpu)) as *mut per_cpu_zonestat;
__v
}
};
}

macro_rules! pfn_to_online_page {
($pfn:expr) => {
{
let mut __v: *mut page;
let mut __v2: *mut page = core::ptr::null_mut::<page>();
if pfn_valid($pfn) != 0 {
__v2 = ((unsafe { vmemmap_base }) as *mut page).wrapping_offset(($pfn) as isize);
}
__v = __v2;
__v
}
};
}

macro_rules! pgtable_l5_enabled {
() => {
cpu_feature_enabled!(528i32)
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

/// The printk_deferred_enter/exit macros are available only as a hack for
/// some code paths that need to defer all printk console printing. Interrupts
/// must be disabled for the deferred duration.
macro_rules! printk_deferred_enter {
() => {
unsafe { __printk_deferred_enter() };
};
}

macro_rules! printk_deferred_exit {
() => {
unsafe { __printk_deferred_exit() };
};
}

macro_rules! raw_check_bogus_irq_restore {
() => {
while false {
}
};
}

macro_rules! raw_local_irq_restore {
($flags:expr) => {
loop {
typecheck!(u64);
raw_check_bogus_irq_restore!();
arch_local_irq_restore($flags);
if true {
break;
}
}
};
}

macro_rules! raw_local_irq_save {
($flags:expr) => {
loop {
typecheck!(u64);
unsafe {
$flags = arch_local_irq_save();
}
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

macro_rules! raw_spin_lock_irqsave__2 {
($lock:expr, $flags:expr) => {
loop {
typecheck!(u64);
unsafe {
$flags = _raw_spin_lock_irqsave($lock);
}
if true {
break;
}
}
};
}

/// Define the various spin_lock methods.  Note we define these
/// regardless of whether CONFIG_SMP or CONFIG_PREEMPTION are set. The
/// various methods are defined as nops in the case they are not
/// required.
macro_rules! raw_spin_trylock {
($lock:expr) => {
unsafe { _raw_spin_trylock($lock) }
};
}

macro_rules! raw_spin_trylock_irqsave {
($lock:expr, $flags:expr) => {
_raw_spin_trylock_irqsave($lock, unsafe { core::ptr::addr_of_mut!($flags) })
};
}

macro_rules! raw_spin_unlock {
($lock:expr) => {
unsafe { _raw_spin_unlock($lock) };
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

/// read_seqcount_begin() - begin a seqcount_t read critical section
/// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
///
/// Return: count to be passed to read_seqcount_retry()
macro_rules! read_seqcount_begin__2 {
($s:expr) => {
{
let mut __v: u32;
let mut __v2: u32;
let mut __v3: u32 = 0;
loop {
let __v4: u32 = seqprop_sequence__3!($s);
__v3 = __v4;
if __v4 & 1 == 0 {
break;
}
cpu_relax();
}
kcsan_atomic_next(1000i32);
__v2 = __v3;
__v = __v2;
__v
}
};
}

/// read_seqcount_begin() - begin a seqcount_t read critical section
/// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
///
/// Return: count to be passed to read_seqcount_retry()
macro_rules! read_seqcount_begin {
($s:expr) => {
{
let mut __v: u32;
let mut __v2: u32;
let mut __v3: u32 = 0;
loop {
let __v4: u32 = seqprop_sequence!($s);
__v3 = __v4;
if __v4 & 1 == 0 {
break;
}
cpu_relax();
}
kcsan_atomic_next(1000i32);
__v2 = __v3;
__v = __v2;
__v
}
};
}

/// read_seqcount_retry() - end a seqcount_t read critical section
/// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
/// @start: count, from read_seqcount_begin()
///
/// read_seqcount_retry closes the read critical section of given
/// seqcount_t.  If the critical section was invalid, it must be ignored
/// (and typically retried).
///
/// Return: true if a read section retry is required, else false
macro_rules! read_seqcount_retry__3 {
($s:expr, $start:expr) => {
do_read_seqcount_retry(seqprop_const_ptr__3!($s), $start)
};
}

/// read_seqcount_retry() - end a seqcount_t read critical section
/// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
/// @start: count, from read_seqcount_begin()
///
/// read_seqcount_retry closes the read critical section of given
/// seqcount_t.  If the critical section was invalid, it must be ignored
/// (and typically retried).
///
/// Return: true if a read section retry is required, else false
macro_rules! read_seqcount_retry {
($s:expr, $start:expr) => {
do_read_seqcount_retry(seqprop_const_ptr!($s), $start)
};
}

/// rounddown_pow_of_two - round the given value down to nearest power of two
/// @n: parameter
///
/// round the given value down to the nearest power of two
/// - the result is undefined when n == 0
/// - this can be used to initialise global variables from constant data
macro_rules! rounddown_pow_of_two {
($n:expr) => {
if unsafe { __slate_intrinsic_a39e427d3b6c2df4($n) } { 1u64 << ilog2__8!($n) } else { __rounddown_pow_of_two(($n) as u64) }
};
}

/// roundup_pow_of_two - round the given value up to nearest power of two
/// @n: parameter
///
/// round the given value up to the nearest power of two
/// - the result is undefined when n == 0
/// - this can be used to initialise global variables from constant data
macro_rules! roundup_pow_of_two__4 {
($n:expr) => {
1u64 << ilog2__7!($n.wrapping_sub(1)).wrapping_add(1)
};
}

/// roundup_pow_of_two - round the given value up to nearest power of two
/// @n: parameter
///
/// round the given value up to the nearest power of two
/// - the result is undefined when n == 0
/// - this can be used to initialise global variables from constant data
macro_rules! roundup_pow_of_two {
($n:expr) => {
if true { if ($n) == 1 { 1u64 } else { 1u64 << ilog2__3!($n.wrapping_sub(1)).wrapping_add(1) } } else { __roundup_pow_of_two(($n) as u64) }
};
}

macro_rules! seqcount_acquire {
() => {
lock_acquire_exclusive!();
};
}

macro_rules! seqcount_release {
() => {
lock_release!();
};
}

macro_rules! seqprop_const_ptr {
($s:expr) => {
__seqprop_spinlock_const_ptr(($s) as *const seqcount_spinlock)
};
}

macro_rules! seqprop_const_ptr__3 {
($s:expr) => {
__seqprop_spinlock_const_ptr($s)
};
}

macro_rules! seqprop_sequence__3 {
($s:expr) => {
__seqprop_spinlock_sequence($s)
};
}

macro_rules! seqprop_sequence {
($s:expr) => {
__seqprop_spinlock_sequence(($s) as *const seqcount_spinlock)
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
macro_rules! small_const_nbits__2 {
($nbits:expr) => {
(unsafe { __slate_intrinsic_f217c4720f04f6db($nbits) }) && ($nbits) <= 64 && ($nbits) > 0
};
}

macro_rules! softirq_count {
() => {
(preempt_count() as u64) & __IRQ_MASK!(8i32) << 8i32
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

macro_rules! spin_lock_irqsave__2 {
($lock:expr, $flags:expr) => {
loop {
raw_spin_lock_irqsave__2!(spinlock_check($lock), $flags);
__release!();
__acquire!();
if true {
break;
}
}
};
}

macro_rules! spin_trylock_irqsave {
($lock:expr, $flags:expr) => {
_spin_trylock_irqsave($lock, core::ptr::addr_of_mut!($flags))
};
}

macro_rules! static_branch_likely {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = !arch_static_branch__const23(unsafe { core::ptr::addr_of_mut!((*$x).key) }, true);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_maybe {
($x:expr) => {
static_branch_unlikely__2!($x)
};
}

macro_rules! static_branch_maybe__3 {
($x:expr) => {
static_branch_unlikely!($x)
};
}

macro_rules! static_branch_unlikely__18 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const18(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
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
__v2 = arch_static_branch__const9(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
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
__v2 = arch_static_branch__const14(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
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
__v2 = arch_static_branch__const10(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
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
__v2 = arch_static_branch__const15(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__19 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const19(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__21 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const22(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
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
__v2 = arch_static_branch__const12(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__15 {
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

macro_rules! static_branch_unlikely__16 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const17(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__20 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const20(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
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
__v2 = arch_static_branch__const3(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
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

macro_rules! static_branch_unlikely__7 {
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

macro_rules! static_branch_unlikely__17 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const16(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
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

macro_rules! str_off_on {
($v:expr) => {
str_on_off(($v) == 0)
};
}

macro_rules! task_thread_info {
($task:expr) => {
unsafe { core::ptr::addr_of_mut!((*$task).thread_info) }
};
}

macro_rules! test_bit__2 {
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

macro_rules! test_bit__6 {
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

macro_rules! test_bit__4 {
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

macro_rules! test_bit__10 {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db($nr) }) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
const_test_bit($nr, ($addr) as *const u64) as i32
} else {
_test_bit($nr, ($addr) as *const u64) as i32
};
__v
}
};
}

macro_rules! test_bit {
($nr:expr, $addr:expr) => {
{
let __v: bool = true && false && (($addr) as u64) != 0;
let __v2: i32 = if __v && false {
const_test_bit(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit(($nr) as u64, ($addr) as *const u64) as i32
};
__v2
}
};
}

macro_rules! test_bit__3 {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if true && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*$addr) } {
const_test_bit(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit(($nr) as u64, ($addr) as *const u64) as i32
};
__v
}
};
}

macro_rules! test_thread_flag {
($flag:expr) => {
test_ti_thread_flag(current_thread_info!(), $flag)
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
let __v: bool = typecheck__2!(u64) != 0 && typecheck__3!(u64) != 0;
__v && ($b.wrapping_sub($a) as i64) < 0
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
time_after!($b, $a)
};
}

/// time_is_after_jiffies - return true if a is after jiffies
/// @a: time (unsigned long) to compare to jiffies
///
/// Return: %true is time a is after jiffies, otherwise %false.
macro_rules! time_is_after_jiffies {
($a:expr) => {
time_before!(unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) }, $a)
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

/// Check at compile time that something is of a particular type.
/// Always evaluates to 1 so you may use it easily in comparisons.
macro_rules! typecheck__2 {
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

macro_rules! unlikely__13 {
($x:expr) => {
!$x.is_null() as i64
};
}

macro_rules! va_end {
() => {
};
}

/// virt_to_page(kaddr) returns a valid pointer if and only if
/// virt_addr_valid(kaddr) returns true.
macro_rules! virt_to_page {
($kaddr:expr) => {
((unsafe { vmemmap_base }) as *mut page).wrapping_offset((__pa!($kaddr) >> 12i32) as isize)
};
}

/// write_seqlock_irqsave() - start a non-interruptible seqlock_t write
///                           section
/// @lock:  Pointer to seqlock_t
/// @flags: Stack-allocated storage for saving caller's local interrupt
///         state, to be passed to write_sequnlock_irqrestore().
///
/// _irqsave variant of write_seqlock(). Use it only if the read side
/// section, or other write sections, can be invoked from hardirq context.
macro_rules! write_seqlock_irqsave {
($lock:expr, $flags:expr) => {
loop {
$flags = __write_seqlock_irqsave($lock);
if true {
break;
}
}
};
}

/// zone_idx() returns 0 for the ZONE_DMA zone, 1 for the ZONE_NORMAL zone, etc.
macro_rules! zone_idx__2 {
($zone:expr) => {
(unsafe { $zone.offset_from(((*(*$zone).zone_pgdat).node_zones.as_mut_ptr() as *mut zone) as *const zone) }) as i64
};
}

/// zone_idx() returns 0 for the ZONE_DMA zone, 1 for the ZONE_NORMAL zone, etc.
macro_rules! zone_idx {
($zone:expr) => {
(unsafe { $zone.offset_from(((*(*$zone).zone_pgdat).node_zones.as_mut_ptr() as *mut zone) as *mut zone) }) as i64
};
}

unsafe extern "C" {
    static mut __SCK__WARN_trap: static_call_key;
    static mut kmsg_fops: file_operations;
    static mut __per_cpu_offset: [u64; 64];
    static mut this_cpu_off: u64;
    static mut __preempt_count: u64;
    static mut __SCK__preempt_schedule: static_call_key;
    static mut __SCK__might_resched: static_call_key;
    static mut system_state: u32;
    static mut node_data: [*mut pglist_data; 0];
    static mut __cpu_possible_mask: cpumask;
    static mut __cpu_online_mask: cpumask;
    static mut __num_online_cpus: __SlateRecord_types_h_190;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut phys_base: u64;
    static mut page_offset_base: u64;
    static mut vmemmap_base: u64;
    static mut boot_cpu_data: cpuinfo_x86;
    static mut cpuinfo_op: seq_operations;
    static mut cpu_number: i32;
    static mut __SCK__cond_resched: static_call_key;
    static mut jiffies: u64;
    static mut mem_section: *mut *mut mem_section;
    static mut node_to_cpumask_map: [[cpumask; 1]; 64];
    static mut sysctl_vals: [i32; 0];
    static mut _totalram_pages: __SlateRecord_types_h_197;
    static mut vm_event_states: vm_event_state;
    static mut init_on_alloc: static_key_false;
    static mut init_on_free: static_key_false;
    static mut kstat: kernel_stat;
    static mut totalreserve_pages: u64;
    static mut oom_lock: mutex;
    static mut tracepoint_srcu: srcu_struct;
    static mut cpusets_pre_enable_key: static_key_false;
    static mut cpusets_enabled_key: static_key_false;
    static mut cpusets_insane_config_key: static_key_false;
    static mut __SCK__tp_func_mm_page_free: static_call_key;
    static mut __tracepoint_mm_page_free: tracepoint;
    static mut __SCK__tp_func_mm_page_free_batched: static_call_key;
    static mut __tracepoint_mm_page_free_batched: tracepoint;
    static mut __SCK__tp_func_mm_page_alloc: static_call_key;
    static mut __tracepoint_mm_page_alloc: tracepoint;
    static mut __SCK__tp_func_mm_page_alloc_zone_locked: static_call_key;
    static mut __tracepoint_mm_page_alloc_zone_locked: tracepoint;
    static mut __SCK__tp_func_mm_page_pcpu_drain: static_call_key;
    static mut __tracepoint_mm_page_pcpu_drain: tracepoint;
    static mut __SCK__tp_func_mm_page_alloc_extfrag: static_call_key;
    static mut __tracepoint_mm_page_alloc_extfrag: tracepoint;
    static mut __SCK__tp_func_mm_setup_per_zone_wmarks: static_call_key;
    static mut __tracepoint_mm_setup_per_zone_wmarks: tracepoint;
    static mut __SCK__tp_func_mm_setup_per_zone_lowmem_reserve: static_call_key;
    static mut __tracepoint_mm_setup_per_zone_lowmem_reserve: tracepoint;
    static mut __SCK__tp_func_mm_calculate_totalreserve_pages: static_call_key;
    static mut __tracepoint_mm_calculate_totalreserve_pages: tracepoint;
    static mut __SCK__tp_func_reclaim_retry_zone: static_call_key;
    static mut __tracepoint_reclaim_retry_zone: tracepoint;
    static mut __SCK__tp_func_compact_retry: static_call_key;
    static mut __tracepoint_compact_retry: tracepoint;
    static mut policy_zone: u32;
    static mut delayacct_key: static_key_false;
    static mut sysctl_min_unmapped_ratio: i32;
    static mut sysctl_min_slab_ratio: i32;
    static mut node_reclaim_mode: i32;
    static mut check_pages_enabled: static_key_false;
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn add_taint(arg0: u32, arg1: u32);
    fn ___ratelimit(arg0: *mut ratelimit_state, arg1: *const u8) -> i32;
    fn _printk(arg0: *const u8, ...) -> i32;
    fn __printk_deferred_enter();
    fn __printk_deferred_exit();
    fn dump_stack();
    fn dump_page(arg0: *const page, arg1: *const u8);
    fn __bad_size_call_parameter();
    fn ____wrong_branch_error() -> bool;
    fn __xchg_wrong_size();
    fn __cmpxchg_wrong_size();
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn _find_next_bit(arg0: *const u64, arg1: u64, arg2: u64) -> u64;
    fn _find_first_bit(arg0: *const u64, arg1: u64) -> u64;
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn __bitmap_weight(arg0: *const u64, arg1: u32) -> u32;
    fn int_sqrt(arg0: u64) -> u64;
    fn __cond_resched() -> i32;
    fn __SCT__might_resched() -> i32;
    fn __compiletime_assert_16();
    fn __compiletime_assert_17();
    fn __compiletime_assert_18();
    fn __compiletime_assert_21();
    fn __compiletime_assert_22();
    fn __compiletime_assert_50();
    fn __compiletime_assert_53();
    fn __compiletime_assert_57();
    fn __virt_addr_valid(arg0: u64) -> bool;
    fn __compiletime_assert_61();
    fn __compiletime_assert_63();
    fn _raw_spin_lock(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irqsave(arg0: *mut raw_spinlock) -> u64;
    fn _raw_spin_trylock(arg0: *mut raw_spinlock) -> i32;
    fn _raw_spin_unlock(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irqrestore(arg0: *mut raw_spinlock, arg1: u64);
    fn mutex_lock(arg0: *mut mutex);
    fn mutex_trylock(arg0: *mut mutex) -> i32;
    fn mutex_unlock(arg0: *mut mutex);
    fn __compiletime_assert_92();
    fn __compiletime_assert_93();
    fn __compiletime_assert_94();
    fn __compiletime_assert_95();
    fn __compiletime_assert_96();
    fn __compiletime_assert_97();
    fn __compiletime_assert_98();
    fn __compiletime_assert_99();
    fn __compiletime_assert_100();
    fn __compiletime_assert_101();
    fn __compiletime_assert_102();
    fn __compiletime_assert_103();
    fn __compiletime_assert_104();
    fn __compiletime_assert_105();
    fn __compiletime_assert_106();
    fn __compiletime_assert_107();
    fn __compiletime_assert_108();
    fn __compiletime_assert_109();
    fn __compiletime_assert_110();
    fn schedule_timeout_uninterruptible(arg0: i64) -> i64;
    fn __SCT__cond_resched() -> i32;
    fn rcu_is_watching() -> bool;
    fn pcpu_alloc_noprof(arg0: u64, arg1: u64, arg2: bool, arg3: u32) -> *mut ();
    fn free_percpu(arg0: *mut ());
    fn __msecs_to_jiffies(arg0: u32) -> u64;
    fn __compiletime_assert_167();
    fn __srcu_check_read_flavor(arg0: *mut srcu_struct, arg1: i32);
    fn __compiletime_assert_187();
    fn __compiletime_assert_188();
    fn __compiletime_assert_189();
    fn __compiletime_assert_190();
    fn __compiletime_assert_191();
    fn __compiletime_assert_192();
    fn __compiletime_assert_193();
    fn wakeup_kswapd(arg0: *mut zone, arg1: u32, arg2: i32, arg3: u32);
    fn kswapd_clear_hopeless(arg0: *mut pglist_data, arg1: u32);
    fn kswapd_test_hopeless(arg0: *mut pglist_data) -> bool;
    fn first_online_pgdat() -> *mut pglist_data;
    fn next_online_pgdat(arg0: *mut pglist_data) -> *mut pglist_data;
    fn next_zone(arg0: *mut zone) -> *mut zone;
    fn __next_zones_zonelist(arg0: *mut zoneref, arg1: u32, arg2: *const __SlateRecord_nodemask_types_h_17) -> *mut zoneref;
    fn __compiletime_assert_201();
    fn __node_distance(arg0: i32, arg1: i32) -> i32;
    fn alloc_pages_noprof(arg0: u32, arg1: u32) -> *mut page;
    fn __compiletime_assert_205();
    fn get_cpu_cacheinfo(arg0: u32) -> *mut cpu_cacheinfo;
    fn cache_get_priv_group(arg0: *mut cacheinfo) -> *const attribute_group;
    fn proc_dostring(arg0: *const ctl_table, arg1: i32, arg2: *mut (), arg3: *mut u64, arg4: *mut i64) -> i32;
    fn proc_dointvec_minmax(arg0: *const ctl_table, arg1: i32, arg2: *mut (), arg3: *mut u64, arg4: *mut i64) -> i32;
    fn __register_sysctl_init(arg0: *const u8, arg1: *const ctl_table, arg2: *const u8, arg3: u64);
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_376();
    fn __compiletime_assert_419();
    fn vm_events_fold_cpu(arg0: i32);
    fn __mod_zone_page_state(arg0: *mut zone, arg1: u32, arg2: i64);
    fn mod_zone_page_state(arg0: *mut zone, arg1: u32, arg2: i64);
    fn cpu_vm_stats_fold(arg0: i32);
    fn refresh_zone_stat_thresholds();
    fn drain_zonestat(arg0: *mut zone, arg1: *mut per_cpu_zonestat);
    fn __show_mem(arg0: u32, arg1: *const __SlateRecord_nodemask_types_h_17, arg2: i32);
    fn __compiletime_assert_488();
    fn print_modules();
    fn node_dirty_ok(arg0: *mut pglist_data) -> bool;
    fn out_of_memory(arg0: *mut oom_control) -> bool;
    fn __cpuhp_setup_state(arg0: i32, arg1: *const u8, arg2: bool, arg3: Option<unsafe extern "C" fn(u32) -> i32>, arg4: Option<unsafe extern "C" fn(u32) -> i32>, arg5: bool) -> i32;
    fn cpuset_init_current_mems_allowed();
    fn cpuset_nodemask_valid_mems_allowed(arg0: *const __SlateRecord_nodemask_types_h_17) -> i32;
    fn cpuset_current_node_allowed(arg0: i32, arg1: u32) -> bool;
    fn cpuset_print_current_mems_allowed();
    fn try_to_compact_pages(arg0: u32, arg1: u32, arg2: u32, arg3: *const alloc_context, arg4: u32, arg5: *mut capture_control) -> u32;
    fn compaction_defer_reset(arg0: *mut zone, arg1: i32, arg2: bool);
    fn compaction_zonelist_suitable(arg0: *mut alloc_context, arg1: i32, arg2: i32, arg3: u32) -> bool;
    fn __traceiter_mm_page_free(arg0: *mut (), arg1: *mut page, arg2: u32) -> i32;
    fn __SCT__tp_func_mm_page_free(arg0: *mut (), arg1: *mut page, arg2: u32) -> i32;
    fn __compiletime_assert_679();
    fn __traceiter_mm_page_free_batched(arg0: *mut (), arg1: *mut page) -> i32;
    fn __SCT__tp_func_mm_page_free_batched(arg0: *mut (), arg1: *mut page) -> i32;
    fn __compiletime_assert_686();
    fn __traceiter_mm_page_alloc(arg0: *mut (), arg1: *mut page, arg2: u32, arg3: u32, arg4: i32) -> i32;
    fn __SCT__tp_func_mm_page_alloc(arg0: *mut (), arg1: *mut page, arg2: u32, arg3: u32, arg4: i32) -> i32;
    fn __compiletime_assert_693();
    fn __traceiter_mm_page_alloc_zone_locked(arg0: *mut (), arg1: *mut page, arg2: u32, arg3: i32, arg4: i32) -> i32;
    fn __SCT__tp_func_mm_page_alloc_zone_locked(arg0: *mut (), arg1: *mut page, arg2: u32, arg3: i32, arg4: i32) -> i32;
    fn __compiletime_assert_700();
    fn __traceiter_mm_page_pcpu_drain(arg0: *mut (), arg1: *mut page, arg2: u32, arg3: i32) -> i32;
    fn __SCT__tp_func_mm_page_pcpu_drain(arg0: *mut (), arg1: *mut page, arg2: u32, arg3: i32) -> i32;
    fn __compiletime_assert_707();
    fn __traceiter_mm_page_alloc_extfrag(arg0: *mut (), arg1: *mut page, arg2: i32, arg3: i32, arg4: i32, arg5: i32) -> i32;
    fn __SCT__tp_func_mm_page_alloc_extfrag(arg0: *mut (), arg1: *mut page, arg2: i32, arg3: i32, arg4: i32, arg5: i32) -> i32;
    fn __compiletime_assert_714();
    fn __traceiter_mm_setup_per_zone_wmarks(arg0: *mut (), arg1: *mut zone) -> i32;
    fn __SCT__tp_func_mm_setup_per_zone_wmarks(arg0: *mut (), arg1: *mut zone) -> i32;
    fn __compiletime_assert_721();
    fn __traceiter_mm_setup_per_zone_lowmem_reserve(arg0: *mut (), arg1: *mut zone, arg2: *mut zone, arg3: i64) -> i32;
    fn __SCT__tp_func_mm_setup_per_zone_lowmem_reserve(arg0: *mut (), arg1: *mut zone, arg2: *mut zone, arg3: i64) -> i32;
    fn __compiletime_assert_728();
    fn __traceiter_mm_calculate_totalreserve_pages(arg0: *mut (), arg1: u64) -> i32;
    fn __SCT__tp_func_mm_calculate_totalreserve_pages(arg0: *mut (), arg1: u64) -> i32;
    fn __compiletime_assert_735();
    fn __traceiter_reclaim_retry_zone(arg0: *mut (), arg1: *mut zoneref, arg2: i32, arg3: u64, arg4: u64, arg5: u64, arg6: i32, arg7: bool) -> i32;
    fn __SCT__tp_func_reclaim_retry_zone(arg0: *mut (), arg1: *mut zoneref, arg2: i32, arg3: u64, arg4: u64, arg5: u64, arg6: i32, arg7: bool) -> i32;
    fn __compiletime_assert_756();
    fn __traceiter_compact_retry(arg0: *mut (), arg1: i32, arg2: u32, arg3: u32, arg4: i32, arg5: i32, arg6: bool) -> i32;
    fn __SCT__tp_func_compact_retry(arg0: *mut (), arg1: i32, arg2: u32, arg3: u32, arg4: i32, arg5: i32, arg6: bool) -> i32;
    fn __compiletime_assert_798();
    fn __delayacct_compact_start();
    fn __delayacct_compact_end();
    fn lru_add_drain_cpu(arg0: i32);
    fn zone_reclaimable_pages(arg0: *mut zone) -> u64;
    fn try_to_free_pages(arg0: *mut zonelist, arg1: i32, arg2: u32, arg3: *const __SlateRecord_nodemask_types_h_17) -> u64;
    fn mlock_drain_remote(arg0: i32);
    fn node_reclaim(arg0: *mut pglist_data, arg1: u32, arg2: u32) -> u64;
    fn mminit_verify_zonelist();
    fn __compiletime_assert_845();
    fn __compiletime_assert_846();
    fn __compiletime_assert_847();
    fn __compiletime_assert_848();
    fn __compiletime_assert_859();
    fn __compiletime_assert_860();
    fn __compiletime_assert_861();
    fn __compiletime_assert_864();
    fn __compiletime_assert_869();
    fn __compiletime_assert_870();
    fn __compiletime_assert_879();
    fn __compiletime_assert_882();
    fn __compiletime_assert_883();
    fn __compiletime_assert_886();
    fn __compiletime_assert_889();
    fn __compiletime_assert_892();
    fn __compiletime_assert_895();
    fn __compiletime_assert_898();
    fn __compiletime_assert_903();
    fn __compiletime_assert_906();
    fn __compiletime_assert_909();
    fn __compiletime_assert_912();
    fn __compiletime_assert_915();
    fn __compiletime_assert_918();
    fn __compiletime_assert_919();
    fn __compiletime_assert_920();
    fn __compiletime_assert_924();
    fn __compiletime_assert_927();
    fn __compiletime_assert_928();
    fn __compiletime_assert_931();
    fn __compiletime_assert_934();
    fn __compiletime_assert_937();
    fn __compiletime_assert_941();
    fn __compiletime_assert_942();
    fn __compiletime_assert_943();
    fn __compiletime_assert_944();
    fn __compiletime_assert_948();
    fn __compiletime_assert_949();
    fn __compiletime_assert_952();
    fn __compiletime_assert_957();
    fn __compiletime_assert_960();
    fn __compiletime_assert_965();
    fn __compiletime_assert_968();
    fn __compiletime_assert_972();
    fn __compiletime_assert_973();
    fn __compiletime_assert_974();
    fn __compiletime_assert_977();
    fn __compiletime_assert_987();
    fn __compiletime_assert_988();
    fn __compiletime_assert_989();
    fn __compiletime_assert_993();
    fn __compiletime_assert_994();
    fn __compiletime_assert_997();
    fn __compiletime_assert_1000();
    fn __compiletime_assert_1003();
    fn __compiletime_assert_1006();
    fn __compiletime_assert_1015();
    fn __compiletime_assert_1020();
    fn __compiletime_assert_1023();
    fn __compiletime_assert_1024();
    fn __compiletime_assert_1025();
    fn __compiletime_assert_1030();
    fn __compiletime_assert_1031();
    fn __compiletime_assert_1032();
    fn __compiletime_assert_1033();
    fn __compiletime_assert_1036();
    fn __compiletime_assert_1039();
    fn __compiletime_assert_1040();
    fn __compiletime_assert_1041();
    fn __compiletime_assert_1070();
    fn __compiletime_assert_1073();
    fn __compiletime_assert_1074();
    fn __compiletime_assert_1075();
    fn __compiletime_assert_1076();
    fn __compiletime_assert_1086();
    fn __compiletime_assert_1087();
    fn __compiletime_assert_1093();
    fn __compiletime_assert_1094();
    fn __compiletime_assert_1101();
    fn __compiletime_assert_1103();
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
struct rcuwait {
    task: *mut task_struct,
}


fn kasan_check_read(p: *const (), size: u32) -> bool {
true
}

fn kasan_check_write(p: *const (), size: u32) -> bool {
true
}

fn __kcsan_check_access(ptr: *const (), size: u64, r#type: i32) {
}

fn kcsan_nestable_atomic_begin() {
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_static_call_types_h_65 {
    r#type: u64,
    mods: *mut static_call_mod,
    sites: *mut static_call_site,
}


fn kcsan_nestable_atomic_end() {
}

fn kcsan_atomic_next(n: i32) {
}

fn kcsan_set_access_mask(mask: u64) {
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


fn kcsan_check_access(ptr: *const (), size: u64, r#type: i32) {
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


fn __kcsan_enable_current() {
}

fn __kcsan_disable_current() {
}

// kcsan_check_read - check regular read access for races
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
union __SlateRecord_jump_label_h_95 {
    r#type: u64,
    entries: *mut jump_entry,
    next: *mut static_key_mod,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct va_format {
    fmt: *const u8,
    va: *mut core::ffi::VaList<'static>,
}


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
// pr_alert - Print an alert-level message
// @fmt: format string
// @...: arguments for the format string
//
// This macro expands to a printk with KERN_ALERT loglevel. It uses pr_fmt() to
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
// pr_cont - Continues a previous log message in the same line.
// @fmt: format string
// @...: arguments for the format string
//
// This macro expands to a printk with KERN_CONT loglevel. It should only be
// used when continuing a log message with no newline ('\n') enclosed. Otherwise
// it defaults back to KERN_DEFAULT loglevel.
#[repr(C)]
#[derive(Clone, Copy)]
struct static_key_mod {
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
struct static_key {
    enabled: __SlateRecord_types_h_190,
    __slate_anon_1: __SlateRecord_jump_label_h_95,
}


#[inline(always)]
fn arch_static_branch__const2(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0} + {1} + 2 - . \n\t.popsection \n\t", sym init_on_alloc, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0} + {1} + 2 - . \n\t.popsection \n\t", sym init_on_free, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0} + {1} + 2 - . \n\t.popsection \n\t", sym cpusets_enabled_key, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0} + {1} + 2 - . \n\t.popsection \n\t", sym cpusets_insane_config_key, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0} + {1} + 2 - . \n\t.popsection \n\t", sym cpusets_pre_enable_key, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0} + {1} + 2 - . \n\t.popsection \n\t", sym cpusets_enabled_key, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
}

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_find_h_67 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_find_h_207 {
}


#[inline(always)]
fn arch_static_branch__const8(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_mm_page_free, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_mm_page_free_batched, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_mm_page_alloc, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_mm_page_alloc_zone_locked, const 0 as u8, label {
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
struct io_bitmap {
}


#[inline(always)]
fn arch_static_branch__const12(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_mm_page_pcpu_drain, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_mm_page_alloc_extfrag, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_mm_setup_per_zone_wmarks, const 0 as u8, label {
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
fn arch_static_branch__const15(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_mm_setup_per_zone_lowmem_reserve, const 0 as u8, label {
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
fn arch_static_branch__const16(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_mm_calculate_totalreserve_pages, const 0 as u8, label {
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
fn arch_static_branch__const17(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_reclaim_retry_zone, const 0 as u8, label {
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
fn arch_static_branch__const18(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_compact_retry, const 0 as u8, label {
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
fn arch_static_branch__const19(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0} + {1} + 2 - . \n\t.popsection \n\t", sym delayacct_key, const 0 as u8, label {
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
fn arch_static_branch__const20(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0} + {1} + 2 - . \n\t.popsection \n\t", sym delayacct_key, const 0 as u8, label {
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
union __SlateRecord_processor_h_111 {
    hw_cpu_type: u32,
    __slate_anon_1: __SlateRecord_processor_h_113,
    __slate_anon_2: __SlateRecord_processor_h_119,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_processor_h_113 {
    __slate_bits_0: __slate_bits::__SlateBits150U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_processor_h_119 {
    __slate_bits_0: __slate_bits::__SlateBits151U0,
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


#[inline(always)]
fn arch_static_branch__const22(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0} + {1} + 2 - . \n\t.popsection \n\t", sym check_pages_enabled, const 0 as u8, label {
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
struct perf_event {
}


#[inline(always)]
fn arch_static_branch__const23(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0} + {1} + 2 - . \n\t.popsection \n\t", sym vm_numa_stat_key, const 1 as u8, label {
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
struct static_key_true {
    key: static_key,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
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
    __slate_bits_0: __slate_bits::__SlateBits104U0,
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
fn native_irq_disable() {
unsafe {
core::arch::asm!("cli", options(nostack, att_syntax));
}
}

#[inline(always)]
fn native_irq_enable() {
unsafe {
core::arch::asm!("sti", options(nostack, att_syntax));
}
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


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits105U0,
}


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
// PGDIR_SHIFT determines what a top-level page table entry can map
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


// No need to mask any bits for !PAE
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_types_h_368 {
    pud: u64,
}


#[inline(always)]
fn arch_atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
// Note for KASAN: we deliberately don't use READ_ONCE_NOCHECK() here,
// it's non-inlined function that increases binary size and stack usage.
__READ_ONCE!((*v).counter)
}

#[inline(always)]
fn arch_atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
__WRITE_ONCE__3!((*v).counter, i);
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
fn arch_atomic64_inc(mut v: *mut __SlateRecord_types_h_197) {
unsafe {
core::arch::asm!("lock incq ({0:r})\n# {1}", in(reg) core::ptr::addr_of_mut!((*v).counter), in(reg) core::ptr::addr_of!((*v).counter), options(nostack, att_syntax));
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
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
fn raw_atomic_dec_and_test(v: *mut __SlateRecord_types_h_190) -> bool {
arch_atomic_dec_and_test(v)
}

#[inline(always)]
fn raw_atomic64_read(v: *const __SlateRecord_types_h_197) -> i64 {
arch_atomic64_read(v)
}

#[inline(always)]
fn raw_atomic64_add(i: i64, v: *mut __SlateRecord_types_h_197) {
arch_atomic64_add(i, v);
}

#[inline(always)]
fn raw_atomic64_inc(v: *mut __SlateRecord_types_h_197) {
arch_atomic64_inc(v);
}

#[inline(always)]
fn raw_atomic_long_read(v: *const __SlateRecord_types_h_197) -> i64 {
raw_atomic64_read(v)
}

#[inline(always)]
fn raw_atomic_long_add(i: i64, v: *mut __SlateRecord_types_h_197) {
raw_atomic64_add(i, v);
}

#[inline(always)]
fn raw_atomic_long_inc(v: *mut __SlateRecord_types_h_197) {
raw_atomic64_inc(v);
}

fn kmsan_unpoison_memory(address: *const (), size: u64) {
}

#[inline(always)]
fn instrument_write(v: *const (), size: u64) {
kasan_check_write(v, size as u32);
kcsan_check_write!(v, size);
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
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("11: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 11b - ., 3; .popsection\n# {0}", const 11 as i32, options(nostack, att_syntax));
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
core::arch::asm!("12: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 12b - ., 4; .popsection\n# {0}", const 12 as i32, options(nostack, att_syntax));
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
#[repr(C)]
#[derive(Clone, Copy)]
struct futex_pi_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct robust_list_head {
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
struct futex_private_hash {
}


#[inline(always)]
fn atomic_dec_and_test(v: *mut __SlateRecord_types_h_190) -> bool {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_dec_and_test(v)
}

#[inline(always)]
fn atomic_long_read(v: *const __SlateRecord_types_h_197) -> i64 {
instrument_atomic_read(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_read(v)
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
struct sem_undo_list {
}


#[inline(always)]
fn generic___set_bit(nr: u64, addr: *mut u64) {
let mask: u64 = BIT_MASK!(nr);
let p: *mut u64 = (addr as *mut u64).wrapping_offset(BIT_WORD!(nr) as isize);
unsafe {
*p = *p | mask;
}
}

#[inline(always)]
fn generic___clear_bit(nr: u64, addr: *mut u64) {
let mask: u64 = BIT_MASK!(nr);
let p: *mut u64 = (addr as *mut u64).wrapping_offset(BIT_WORD!(nr) as isize);
let __v2: u64 = (unsafe { *p }) & !mask;
unsafe {
*p = __v2;
}
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
fn arch___set_bit(mut nr: u64, mut addr: *mut u64) {
unsafe {
core::arch::asm!(" btsq  {1:r},({0:r})", in(reg) addr as *mut i64, in(reg) nr as u64, options(nostack, att_syntax));
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer_clock_base {
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
fn arch___clear_bit(mut nr: u64, mut addr: *mut u64) {
unsafe {
core::arch::asm!(" btrq  {1:r},({0:r})", in(reg) addr as *mut i64, in(reg) nr as u64, options(nostack, att_syntax));
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct seccomp_filter {
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
fn __fls(mut word: u64) -> u64 {
if unsafe { __slate_intrinsic_f217c4720f04f6db(word) } {
return 63i32.wrapping_sub(word.leading_zeros() as i32) as u64;
}
unsafe {
let __t0: u64;
core::arch::asm!("bsr {1:r},{0:r}", lateout(reg) __t0, in(reg) word as u64, options(pure, nomem, nostack, att_syntax));
word = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
word
}

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

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_signal_types_h_14 {
    si_signo: i32,
    si_errno: i32,
    si_code: i32,
    _sifields: __sifields,
}


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
fn ___set_bit(nr: u64, addr: *mut u64) {
instrument_write(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch___set_bit(nr, addr);
}

#[inline(always)]
fn ___clear_bit(nr: u64, addr: *mut u64) {
instrument_write(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch___clear_bit(nr, addr);
}

#[inline(always)]
fn _test_bit(nr: u64, addr: *const u64) -> bool {
instrument_atomic_read(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_bit(nr, addr)
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

fn fls_long(l: u64) -> u32 {
if ((const { core::mem::size_of::<u64>() }) as u64) == 4 {
return fls(l as u32) as u32;
}
fls64(l) as u32
}

// assign_bit - Assign value to a bit in memory
// @nr: the bit to set
// @addr: the address to start counting from
// @value: the value to assign
#[inline(always)]
fn find_next_bit(addr: *const u64, size: u64, offset: u64) -> u64 {
if small_const_nbits__2!(size) {
let mut val: u64 = 0;
if unlikely!(offset >= size) != 0 {
return size;
}
val = (unsafe { *addr }) & GENMASK!(size.wrapping_sub(1), offset);
return if val != 0 { __ffs!(val) } else { size };
}
unsafe { _find_next_bit(addr, size, offset) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct workqueue_struct {
}


#[inline(always)]
fn find_first_bit(addr: *const u64, size: u64) -> u64 {
if small_const_nbits__2!(size) {
let val: u64 = (unsafe { *addr }) & GENMASK__2!(size.wrapping_sub(1), 0i32);
return if val != 0 { __ffs!(val) } else { size };
}
unsafe { _find_first_bit(addr, size) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
}


#[inline(always)]
fn bitmap_empty(src: *const u64, nbits: u32) -> bool {
if small_const_nbits!(nbits) {
return (unsafe { *src }) & BITMAP_LAST_WORD_MASK!(nbits) == 0;
}
find_first_bit(src, nbits as u64) == (nbits as u64)
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
fn bitmap_weight(src: *const u64, nbits: u32) -> u32 {
if small_const_nbits!(nbits) {
return hweight_long((unsafe { *src }) & BITMAP_LAST_WORD_MASK!(nbits)) as u32;
}
unsafe { __bitmap_weight(src, nbits) }
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
fn __ilog2_u32(n: u32) -> i32 {
fls(n).wrapping_sub(1)
}

#[inline(always)]
fn __ilog2_u64(n: u64) -> i32 {
fls64(n).wrapping_sub(1)
}

#[inline(always)]
fn is_power_of_2(n: u64) -> bool {
n.wrapping_sub(1) < n ^ n.wrapping_sub(1)
}

fn __roundup_pow_of_two(n: u64) -> u64 {
1u64 << fls_long(n.wrapping_sub(1))
}

fn __rounddown_pow_of_two(n: u64) -> u64 {
1u64 << fls_long(n).wrapping_sub(1)
}

fn __order_base_2(n: u64) -> i32 {
if n > 1 { ilog2!(n.wrapping_sub(1)).wrapping_add(1) } else { 0 }
}

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
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


#[inline(always)]
fn might_resched() {
let mut __v0: Option<unsafe extern "C" fn() -> i32>;
__v0 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn() -> i32>>(__SCT__might_resched as *const ()) };
unsafe { __v0.unwrap()() };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct audit_context {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bio_list {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_plug {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_net_context {
}


#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__might_resched_15: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__might_resched) }) as *mut ();

#[repr(C)]
#[derive(Clone, Copy)]
struct cfs_rq {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_struct {
}


fn __might_resched(file: *const u8, line: i32, offsets: u32) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_task {
}


fn INIT_LIST_HEAD(list: *mut list_head) {
loop {
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_16() };
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
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_17() };
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

#[repr(C)]
#[derive(Clone, Copy)]
struct nameidata {
}


fn __list_add_valid(new: *mut list_head, prev: *mut list_head, next: *mut list_head) -> bool {
true
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


fn __list_del_entry_valid(entry: *mut list_head) -> bool {
true
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rq {
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
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_18() };
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
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_21() };
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

#[repr(C)]
#[derive(Clone, Copy)]
struct task_exec_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct task_group {
}


fn list_del(entry: *mut list_head) {
__list_del_entry(entry);
unsafe {
(*entry).next = ((256i32 as *const u8).wrapping_offset(_AC__3!() as isize) as *mut ()) as *mut list_head;
}
unsafe {
(*entry).prev = ((290i32 as *const u8).wrapping_offset(_AC__3!() as isize) as *mut ()) as *mut list_head;
}
}

fn list_move_tail(list: *mut list_head, head: *mut list_head) {
__list_del_entry(list);
list_add_tail(list, head);
}

fn list_empty(head: *const list_head) -> i32 {
let mut __v0: *mut list_head;
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_22() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*head).next) as *const *mut list_head) };
(__v0 == (head as *mut list_head)) as i32
}

// list_first_entry_or_null - get the first element from a list
// @ptr:	the list head to take the element from.
// @type:	the type of the struct this is embedded in.
// @member:	the name of the list_head within the struct.
//
// Note that if the list is empty, it returns NULL.
// list_next_entry - get the next element in list
// @pos:	the type * to cursor
// @member:	the name of the list_head within the struct.
// list_for_each	-	iterate over a list
// @pos:	the &struct list_head to use as a loop cursor.
// @head:	the head for your list.
// list_entry_is_head - test if the entry points to the head of the list
// @pos:	the type * to cursor
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
// list_for_each_entry	-	iterate over list of given type
// @pos:	the type * to use as a loop cursor.
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
#[inline(always)]
fn __nodemask_pr_numnodes(m: *const __SlateRecord_nodemask_types_h_17) -> u32 {
if !m.is_null() { unsafe { nr_node_ids } } else { 0 }
}

#[inline(always)]
fn __nodemask_pr_bits(m: *const __SlateRecord_nodemask_types_h_17) -> *const u64 {
if !m.is_null() { unsafe { (*m).bits.as_ptr() as *const u64 } } else { core::ptr::null::<u64>() }
}

#[inline(always)]
fn __node_set(node: i32, dstp: *mut __SlateRecord_nodemask_types_h_17) {
set_bit(node as i64, unsafe { (*dstp).bits.as_mut_ptr() as *mut u64 });
}

#[inline(always)]
fn __first_node(srcp: *const __SlateRecord_nodemask_types_h_17) -> u32 {
let mut __v0: u64;
let __UNIQUE_ID_x__48: i32 = 64i32;
let __UNIQUE_ID_y__49: u64 = find_first_bit(unsafe { (*srcp).bits.as_ptr() as *const u64 }, 64u64);
loop {
if __types_ok__3!(__UNIQUE_ID_x__48, __UNIQUE_ID_y__49) == 0 {
unsafe { __compiletime_assert_50() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if (__UNIQUE_ID_x__48 as u64) < __UNIQUE_ID_y__49 { __UNIQUE_ID_x__48 as u64 } else { __UNIQUE_ID_y__49 };
__v0 as u32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rq_flags {
}


#[inline(always)]
fn __next_node(n: i32, srcp: *const __SlateRecord_nodemask_types_h_17) -> u32 {
let mut __v0: u64;
let __UNIQUE_ID_x__51: i32 = 64i32;
let __UNIQUE_ID_y__52: u64 = find_next_bit(unsafe { (*srcp).bits.as_ptr() as *const u64 }, 64u64, n.wrapping_add(1) as u64);
loop {
if __types_ok__3!(__UNIQUE_ID_x__51, __UNIQUE_ID_y__52) == 0 {
unsafe { __compiletime_assert_53() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if (__UNIQUE_ID_x__51 as u64) < __UNIQUE_ID_y__52 { __UNIQUE_ID_x__51 as u64 } else { __UNIQUE_ID_y__52 };
__v0 as u32
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
fn node_state(node: i32, state: u32) -> i32 {
node_isset!(node, node_states.0[(state as u32) as usize])
}

#[inline(always)]
fn next_memory_node(nid: i32) -> u32 {
next_node!(nid, node_states.0[3])
}

#[inline(always)]
fn cpu_max_bits_warn(cpu: u32, bits: u32) {
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
fn cpumask_set_cpu(cpu: u32, dstp: *mut cpumask) {
set_bit(cpumask_check(cpu) as i64, cpumask_bits__2!(dstp) as *mut u64);
}

#[inline(always)]
fn cpumask_clear_cpu(cpu: i32, dstp: *mut cpumask) {
clear_bit(cpumask_check(cpu as u32) as i64, cpumask_bits__2!(dstp) as *mut u64);
}

#[inline(always)]
fn cpumask_test_cpu(cpu: i32, cpumask: *const cpumask) -> bool {
test_bit__6!(cpumask_check(cpu as u32), unsafe { (*cpumask).bits.as_ptr() as *const u64 }) != 0
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
fn cpumask_empty(srcp: *const cpumask) -> bool {
bitmap_empty(cpumask_bits!(srcp), 64u32)
}

#[inline(always)]
fn cpumask_weight(srcp: *const cpumask) -> u32 {
bitmap_weight(cpumask_bits!(srcp), 64u32)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct gendisk {
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
fn num_online_cpus() -> u32 {
raw_atomic_read((unsafe { core::ptr::addr_of_mut!(__num_online_cpus) }) as *const __SlateRecord_types_h_190) as u32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct vm_struct {
}


#[inline(always)]
fn cpu_online(cpu: u32) -> bool {
cpumask_test_cpu(cpu as i32, (unsafe { core::ptr::addr_of_mut!(__cpu_online_mask) }) as *const cpumask)
}

#[inline(always)]
fn arch_local_save_flags() -> u64 {
native_save_fl()
}

#[inline(always)]
fn arch_local_irq_disable() {
native_irq_disable();
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[inline(always)]
fn arch_local_irq_enable() {
native_irq_enable();
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
}


#[inline(always)]
fn arch_local_irq_save() -> u64 {
let flags: u64 = arch_local_save_flags();
arch_local_irq_disable();
flags
}

#[inline(always)]
fn arch_irqs_disabled_flags(flags: u64) -> i32 {
(flags & _BITUL!(9i32) == 0) as i32
}

#[inline(always)]
fn arch_local_irq_restore(flags: u64) {
if arch_irqs_disabled_flags(flags) == 0 {
arch_local_irq_enable();
}
}

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct cacheline_padding {
    x: [u8; 0],
}


#[inline(always)]
fn get_current() -> *mut task_struct {
if IS_ENABLED!() != 0 {
let mut __v0: *mut task_struct;
loop {
if true {
unsafe { __compiletime_assert_57() };
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
__verify_pcpu_ptr__2!();
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
struct uprobe {
}


#[repr(C, align(4))]
#[derive(Clone, Copy)]
struct leaf_0x0_0 {
    __slate_bits_0: __slate_bits::__SlateBits141U0,
}


#[repr(C, align(4))]
#[derive(Clone, Copy)]
struct leaf_0x1_0 {
    __slate_bits_0: __slate_bits::__SlateBits142U0,
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


#[inline(always)]
fn __phys_addr_nodebug(mut x: u64) -> u64 {
let y: u64 = x.wrapping_sub(_AC__2!());
// use the carry flag to determine if x was < __START_KERNEL_map
x = y.wrapping_add(if x > y { unsafe { phys_base } } else { _AC__2!().wrapping_sub(unsafe { page_offset_base }) });
x
}

fn clear_pages(mut addr: *mut (), npages: u32) {
let mut len: u64 = (npages as u64).wrapping_mul(_AC!() << 12i32);
// Clean up KMSAN metadata for the pages being cleared. The assembly call
// below clobbers @addr, so perform unpoisoning before it.
kmsan_unpoison_memory(addr as *const (), len);
// The inline asm embeds a CALL instruction and usually that is a no-no
// due to the compiler not knowing that and thus being unable to track
// callee-clobbered registers.
//
// In this case that is fine because the registers clobbered by
// __clear_pages_unrolled() are part of the inline asm register
// specification.
unsafe {
let __t0: u64;
let __t1: u64;
core::arch::asm!("# ALT: oldinstr\n771:\n\t# ALT: oldinstr\n771:\n\tcall __clear_pages_unrolled\n772:\n# ALT: padding\n.skip -(((775f-774f)-(772b-771b)) > 0) * ((775f-774f)-(772b-771b)),0x90\n773:\n.pushsection .altinstructions, \"aM\", @progbits, 14\n .long 771b - .\n .long 774f - .\n .4byte ( 3*32+16)\n .byte 773b-771b\n .byte 775f-774f\n.popsection\n.pushsection .altinstr_replacement, \"ax\"\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n# ALT: replacement\n774:\n\tshrq $3, %rcx; rep stosq\n775:\n.popsection\n\n772:\n# ALT: padding\n.skip -(((775f-774f)-(772b-771b)) > 0) * ((775f-774f)-(772b-771b)),0x90\n773:\n.pushsection .altinstructions, \"aM\", @progbits, 14\n .long 771b - .\n .long 774f - .\n .4byte ( 9*32+ 9)\n .byte 773b-771b\n .byte 775f-774f\n.popsection\n.pushsection .altinstr_replacement, \"ax\"\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n# ALT: replacement\n774:\n\trep stosb\n775:\n.popsection\n", inlateout("cx") len as u64 => __t0, inlateout("di") addr as u64 => __t1, in("ax") (0 as i32) as u32, options(att_syntax));
len = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
addr = unsafe { core::mem::transmute::<u64, *mut ()>(__t1 as u64) };
}
}

fn clear_page(addr: *mut ()) {
clear_pages(addr, 1u32);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct range {
    start: u64,
    end: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe_xol_ops {
}


#[inline(always)]
fn get_order(mut size: u64) -> i32 {
if unsafe { __slate_intrinsic_f217c4720f04f6db(size) } {
if size == 0 {
return 52i32;
}
if size < 4096u64 {
return 0;
}
return ilog2!(size.wrapping_sub(1)).wrapping_sub(12).wrapping_add(1);
}
size = size.wrapping_sub(1);
size >>= 12i32;
fls64(size)
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

fn div64_u64(dividend: u64, divisor: u64) -> u64 {
dividend / divisor
}

// DIV_U64_ROUND_UP - unsigned 64bit divide with 32bit divisor rounded up
// @ll: unsigned 64bit dividend
// @d: unsigned 32bit divisor
//
// Divide unsigned 64bit dividend by unsigned 32bit divisor
// and round up.
//
// Return: dividend / divisor rounded up
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
struct xol_area {
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
    __slate_bits_0: __slate_bits::__SlateBits139U0,
}


// sizeof(unsigned long) coming from an extra "long" at the end of the
// iobitmap. The limit is inclusive, i.e. the last valid byte.
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
    __slate_bits_0: __slate_bits::__SlateBits123U0,
    pkru: u32,
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
fn llist_empty(head: *const llist_head) -> bool {
let mut __v0: *mut llist_node;
loop {
if !(__native_word__22!() || ((const { core::mem::size_of::<*mut llist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_61() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*head).first) as *const *mut llist_node) };
__v0.is_null()
}

fn llist_add_batch(new_first: *mut llist_node, new_last: *mut llist_node, head: *mut llist_head) -> bool {
let mut first: *mut llist_node = unsafe { core::mem::zeroed() };
let mut __v10: *mut llist_node;
loop {
if !(__native_word__22!() || ((const { core::mem::size_of::<*mut llist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
if unlikely!(!__v2) != 0 {
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
struct mem_cgroup {
}


fn llist_add(new: *mut llist_node, head: *mut llist_head) -> bool {
llist_add_batch(new, new, head)
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
struct __call_single_node {
    llist: llist_node,
    __slate_anon_1: __SlateRecord_smp_types_h_60,
    src: u16,
    dst: u16,
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
struct __kernel_timespec {
    tv_sec: i64,
    tv_nsec: i64,
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


#[inline(always)]
fn __static_cpu_has__const1(bit: u16) -> bool {
unsafe {
let mut __slate_asm_target: usize = 0usize;
core::arch::asm!("# ALT: oldinstr\n771:\n\t# ALT: oldinstr\n771:\n\tjmp 6f\n772:\n# ALT: padding\n.skip -(((775f-774f)-(772b-771b)) > 0) * ((775f-774f)-(772b-771b)),0x90\n773:\n.pushsection .altinstructions, \"aM\", @progbits, 14\n .long 771b - .\n .long 774f - .\n .4byte ( 3*32+21)\n .byte 773b-771b\n .byte 775f-774f\n.popsection\n.pushsection .altinstr_replacement, \"ax\"\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n# ALT: replacement\n774:\n\tjmp {4}\n775:\n.popsection\n\n772:\n# ALT: padding\n.skip -(((775f-774f)-(772b-771b)) > 0) * ((775f-774f)-(772b-771b)),0x90\n773:\n.pushsection .altinstructions, \"aM\", @progbits, 14\n .long 771b - .\n .long 774f - .\n .4byte {0}\n .byte 773b-771b\n .byte 775f-774f\n.popsection\n.pushsection .altinstr_replacement, \"ax\"\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n# ALT: replacement\n774:\n\t\n775:\n.popsection\n.pushsection .altinstr_aux,\"ax\"\n6:\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n testb ${1}, {2}+114(%rip)\n jnz {3}\n jmp {4}\n.popsection\n", const 528 as u16, const 1 as i32, sym boot_cpu_data, label {
__slate_asm_target = 1usize;
}, label {
__slate_asm_target = 2usize;
}, options(nostack, att_syntax));
if __slate_asm_target != 1usize {
if __slate_asm_target == 2usize {
return false;
}
}
return true;
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct thread_info {
    flags: u64,
    syscall_work: u64,
    status: u32,
    cpu: u32,
}


// uses I/O bitmap
// 32-bit address space on 64 bits
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


fn test_ti_thread_flag(ti: *mut thread_info, flag: i32) -> i32 {
test_bit__2!(flag, unsafe { core::ptr::addr_of_mut!((*ti).flags) })
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
struct tracepoint_func {
    func: *mut (),
    data: *mut (),
    prio: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_1062 {
    rb: rb_node,
    rb_subtree_last: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tracepoint_ext {
    regfunc: Option<unsafe extern "C" fn() -> i32>,
    unregfunc: Option<unsafe extern "C" fn()>,
    __slate_bits_0: __slate_bits::__SlateBits181U0,
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
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_page_flags_h_276 {
}


#[inline(always)]
fn _raw_spin_trylock_irqsave(lock__0: *mut raw_spinlock, flags: *mut u64) -> bool {
local_irq_save!(*flags);
if (unsafe { _raw_spin_trylock(lock__0) }) != 0 {
return true;
}
local_irq_restore!(unsafe { *flags });
false
}

// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
#[inline(always)]
fn spinlock_check(lock__0: *mut spinlock) -> *mut raw_spinlock {
unsafe { core::ptr::addr_of_mut!((*lock__0).__slate_anon_0.rlock) }
}

#[inline(always)]
fn spin_lock(lock__0: *mut spinlock) {
raw_spin_lock!(unsafe { core::ptr::addr_of_mut!((*lock__0).__slate_anon_0.rlock) });
}

#[inline(always)]
fn spin_trylock(lock__0: *mut spinlock) -> i32 {
raw_spin_trylock!(unsafe { core::ptr::addr_of_mut!((*lock__0).__slate_anon_0.rlock) })
}

#[inline(always)]
fn spin_unlock(lock__0: *mut spinlock) {
raw_spin_unlock!(unsafe { core::ptr::addr_of_mut!((*lock__0).__slate_anon_0.rlock) });
}

#[inline(always)]
fn spin_unlock_irqrestore(lock__0: *mut spinlock, flags: u64) {
raw_spin_unlock_irqrestore!(unsafe { core::ptr::addr_of_mut!((*lock__0).__slate_anon_0.rlock) }, flags);
}

#[inline(always)]
fn _spin_trylock_irqsave(lock__0: *mut spinlock, flags: *mut u64) -> bool {
raw_spin_trylock_irqsave!(spinlock_check(lock__0), *flags)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_spinlock_h_642 {
    lock: *mut spinlock,
    flags: u64,
}


#[inline(always)]
fn class_spinlock_irqsave_destructor(_T: *mut __SlateRecord_spinlock_h_642) {
spin_unlock_irqrestore(unsafe { (*_T).lock }, unsafe { (*_T).flags });
}

#[inline(always)]
fn class_spinlock_irqsave_constructor(l: *mut spinlock) -> __SlateRecord_spinlock_h_642 {
let mut _t: __SlateRecord_spinlock_h_642 = {
let mut __t0: __SlateRecord_spinlock_h_642 = unsafe { core::mem::zeroed() };
__t0.lock = l;
__t0
};
let _T: *mut __SlateRecord_spinlock_h_642 = core::ptr::addr_of_mut!(_t);
spin_lock_irqsave__2!(unsafe { (*_T).lock }, (*_T).flags);
return _t;
}

#[repr(C)]
#[derive(Clone, Copy)]
struct memory_tier {
}


#[inline(always)]
fn __class_spinlock_irqsave_cleanup_ctx(_T: *mut *mut __SlateRecord_spinlock_h_642) {
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
struct device_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wait_queue_head {
    lock: spinlock,
    head: list_head,
}


fn waitqueue_active(wq_head: *mut wait_queue_head) -> i32 {
(list_empty((unsafe { core::ptr::addr_of_mut!((*wq_head).head) }) as *const list_head) == 0) as i32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct optimistic_spin_queue {
    tail: __SlateRecord_types_h_190,
}


fn debug_check_no_locks_freed(from: *const (), len: u64) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache {
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
struct seqcount {
    sequence: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqcount_raw_spinlock {
    seqcount: seqcount,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqcount_spinlock {
    seqcount: seqcount,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqcount_rwlock {
    seqcount: seqcount,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqcount_mutex {
    seqcount: seqcount,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqlock {
    seqcount: seqcount_spinlock,
    lock: spinlock,
}


// SEQCNT_ZERO() - static initializer for seqcount_t
// @name: Name of the seqcount_t instance
// SEQCOUNT_LOCKNAME()	- Instantiate seqcount_LOCKNAME_t and helpers
// seqprop_LOCKNAME_*()	- Property accessors for seqcount_LOCKNAME_t
//
// @lockname:		"LOCKNAME" part of seqcount_LOCKNAME_t
// @locktype:		LOCKNAME canonical C data type
// @preemptible:	preemptibility of above locktype
// @lockbase:		prefix for associated lock/unlock
// 									 * Re-read the sequence counter since the (possibly			 * preempted) writer made progress.
// PREEMPT_RT relies on the above LOCK+UNLOCK
fn __seqprop_const_ptr(s: &seqcount) -> *const seqcount {
s as *const seqcount
}

fn __seqprop_sequence(s: &seqcount) -> u32 {
let mut __v0: u32;
let mut ___p1: u32 = 0;
let mut __v1: u32;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_92() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*(s as *const seqcount)).sequence) as *const u32) };
___p1 = __v1;
loop {
if !__native_word__3!() {
unsafe { __compiletime_assert_93() };
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
__v0
}

#[inline(always)]
fn __seqprop_raw_spinlock_const_ptr(s: &seqcount_raw_spinlock) -> *const seqcount {
unsafe { core::ptr::addr_of!((*(s as *const seqcount_raw_spinlock)).seqcount) }
}

#[inline(always)]
fn __seqprop_raw_spinlock_sequence(s: &seqcount_raw_spinlock) -> u32 {
let mut seq: u32 = 0;
let mut __v0: u32;
{
let mut ___p1: u32 = 0;
let mut __v1: u32;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_94() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*(s as *const seqcount_raw_spinlock)).seqcount.sequence) as *const u32) };
___p1 = __v1;
loop {
if !__native_word__3!() {
unsafe { __compiletime_assert_95() };
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
}
seq = __v0;
if IS_ENABLED!() == 0 {
return seq;
}
if false && unlikely__3!(seq & 1) != 0 {
let mut __v4: u32;
let mut ___p1: u32 = 0;
let mut __v5: u32;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_96() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v5 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*(s as *const seqcount_raw_spinlock)).seqcount.sequence) as *const u32) };
___p1 = __v5;
loop {
if !__native_word__3!() {
unsafe { __compiletime_assert_97() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
__v4 = ___p1;
seq = __v4;
}
seq
}

#[inline(always)]
fn __seqprop_spinlock_const_ptr(s: *const seqcount_spinlock) -> *const seqcount {
unsafe { core::ptr::addr_of!((*s).seqcount) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mnt_idmap {
}


#[inline(always)]
fn __seqprop_spinlock_sequence(s: *const seqcount_spinlock) -> u32 {
let mut seq: u32 = 0;
let mut __v0: u32;
{
let mut ___p1: u32 = 0;
let mut __v1: u32;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_98() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*s).seqcount.sequence) as *const u32) };
___p1 = __v1;
loop {
if !__native_word__3!() {
unsafe { __compiletime_assert_99() };
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
}
seq = __v0;
if IS_ENABLED!() == 0 {
return seq;
}
if false && unlikely__3!(seq & 1) != 0 {
let mut __v4: u32;
let mut ___p1: u32 = 0;
let mut __v5: u32;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_100() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v5 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*s).seqcount.sequence) as *const u32) };
___p1 = __v5;
loop {
if !__native_word__3!() {
unsafe { __compiletime_assert_101() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
__v4 = ___p1;
seq = __v4;
}
seq
}

#[inline(always)]
fn __seqprop_rwlock_const_ptr(s: &seqcount_rwlock) -> *const seqcount {
unsafe { core::ptr::addr_of!((*(s as *const seqcount_rwlock)).seqcount) }
}

#[inline(always)]
fn __seqprop_rwlock_sequence(s: &seqcount_rwlock) -> u32 {
let mut seq: u32 = 0;
let mut __v0: u32;
{
let mut ___p1: u32 = 0;
let mut __v1: u32;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_102() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*(s as *const seqcount_rwlock)).seqcount.sequence) as *const u32) };
___p1 = __v1;
loop {
if !__native_word__3!() {
unsafe { __compiletime_assert_103() };
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
}
seq = __v0;
if IS_ENABLED!() == 0 {
return seq;
}
if false && unlikely__3!(seq & 1) != 0 {
let mut __v4: u32;
let mut ___p1: u32 = 0;
let mut __v5: u32;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_104() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v5 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*(s as *const seqcount_rwlock)).seqcount.sequence) as *const u32) };
___p1 = __v5;
loop {
if !__native_word__3!() {
unsafe { __compiletime_assert_105() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
__v4 = ___p1;
seq = __v4;
}
seq
}

#[inline(always)]
fn __seqprop_mutex_const_ptr(s: &seqcount_mutex) -> *const seqcount {
unsafe { core::ptr::addr_of!((*(s as *const seqcount_mutex)).seqcount) }
}

#[inline(always)]
fn __seqprop_mutex_sequence(s: &seqcount_mutex) -> u32 {
let mut seq: u32 = 0;
let mut __v0: u32;
{
let mut ___p1: u32 = 0;
let mut __v1: u32;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_106() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*(s as *const seqcount_mutex)).seqcount.sequence) as *const u32) };
___p1 = __v1;
loop {
if !__native_word__3!() {
unsafe { __compiletime_assert_107() };
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
}
seq = __v0;
if IS_ENABLED!() == 0 {
return seq;
}
if true && unlikely__3!(seq & 1) != 0 {
let mut __v4: u32;
let mut ___p1: u32 = 0;
let mut __v5: u32;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_108() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v5 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*(s as *const seqcount_mutex)).seqcount.sequence) as *const u32) };
___p1 = __v5;
loop {
if !__native_word__3!() {
unsafe { __compiletime_assert_109() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
__v4 = ___p1;
seq = __v4;
}
seq
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


// __read_seqcount_begin() - begin a seqcount_t read section
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
//
// Return: count to be passed to read_seqcount_retry()
// raw_read_seqcount_begin() - begin a seqcount_t read section w/o lockdep
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
fn do___read_seqcount_retry(s: *const seqcount, start: u32) -> i32 {
kcsan_atomic_next(0i32);
let mut __v0: u32;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_110() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*s).sequence) as *const u32) };
((__v0 != start) as i64) as i32
}

fn do_read_seqcount_retry(s: *const seqcount, start: u32) -> i32 {
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
do___read_seqcount_retry(s, start)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array_ptr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct net {
}


fn do_raw_write_seqcount_begin(__v0: *mut seqcount) {
kcsan_nestable_atomic_begin();
let __v1: u32 = unsafe { (*__v0).sequence };
unsafe {
(*__v0).sequence = __v1.wrapping_add(1);
}
loop {
kcsan_wmb!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct key_type {
}


fn do_raw_write_seqcount_end(__v2: *mut seqcount) {
loop {
kcsan_wmb!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
let __v3: u32 = unsafe { (*__v2).sequence };
unsafe {
(*__v2).sequence = __v3.wrapping_add(1);
}
kcsan_nestable_atomic_end();
}

fn do_write_seqcount_begin_nested(s: *mut seqcount, subclass: i32) {
seqcount_acquire!();
do_raw_write_seqcount_begin(s);
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


// write_seqcount_begin() - start a seqcount_t write side critical section
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
//
// Context: sequence counter write side sections must be serialized and
// non-preemptible. Preemption will be automatically disabled if and
// only if the seqcount write serialization lock is associated, and
// preemptible.  If readers can be invoked from hardirq or softirq
// context, interrupts or bottom halves must be respectively disabled.
fn do_write_seqcount_begin(s: *mut seqcount) {
do_write_seqcount_begin_nested(s, 0i32);
}

// write_seqcount_end() - end a seqcount_t write side critical section
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
//
// Context: Preemption will be automatically re-enabled if and only if
// the seqcount write serialization lock is associated, and preemptible.
fn do_write_seqcount_end(s: *mut seqcount) {
seqcount_release!();
do_raw_write_seqcount_end(s);
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


// seqlock_init() - dynamic initializer for seqlock_t
// @sl: Pointer to the seqlock_t instance
// DEFINE_SEQLOCK(sl) - Define a statically allocated seqlock_t
// @sl: Name of the seqlock_t instance
fn read_seqbegin(sl: *const seqlock) -> u32 {
read_seqcount_begin__2!(unsafe { core::ptr::addr_of!((*sl).seqcount) })
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_cred_h_147 {
    non_rcu: i32,
    rcu: callback_head,
}


fn read_seqretry(sl: *const seqlock, start: u32) -> u32 {
read_seqcount_retry__3!(unsafe { core::ptr::addr_of!((*sl).seqcount) }, start) as u32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pidfs_attr {
}


fn __write_seqlock_irqsave(sl: *mut seqlock) -> u64 {
let mut flags: u64 = 0;
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*sl).lock) }, flags);
do_write_seqcount_begin(unsafe { core::ptr::addr_of_mut!((*sl).seqcount.seqcount) });
flags
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pid_h_62 {
    ino: u64,
    pidfs_hash: rhash_head,
    stashed: *mut dentry,
    attr: *mut pidfs_attr,
}


fn write_sequnlock_irqrestore(sl: *mut seqlock, flags: u64) {
do_write_seqcount_end(unsafe { core::ptr::addr_of_mut!((*sl).seqcount.seqcount) });
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*sl).lock) }, flags);
}

// read_seqlock_excl_irqsave() - start a non-interruptible seqlock_t
// 				 locking reader section
// @lock:  Pointer to seqlock_t
// @flags: Stack-allocated storage for saving caller's local interrupt
//         state, to be passed to read_sequnlock_excl_irqrestore().
//
// _irqsave variant of read_seqlock_excl(). Use this only if the seqlock_t
// write side section, *or other read sections*, can be invoked from a
// hardirq context.
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
struct refcount_struct {
    refs: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kref {
    refcount: refcount_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_struct {
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
struct tty_audit_buf {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct rb_root {
    rb_node: *mut rb_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rb_root_cached {
    rb_root: rb_root,
    rb_leftmost: *mut rb_node,
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
struct plist_node {
    prio: i32,
    prio_list: list_head,
    node_list: list_head,
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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
struct rlimit {
    rlim_cur: u64,
    rlim_max: u64,
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
struct syscall_user_dispatch {
    selector: *mut u8,
    offset: u64,
    len: u64,
    on_dispatch: bool,
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
struct posix_cputimers {
    bases: [posix_cputimer_base; 3],
    timers_active: u32,
    expiry_active: u32,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq_event {
    __slate_anon_0: __SlateRecord_rseq_types_h_38,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq_ids {
    __slate_anon_0: __SlateRecord_rseq_types_h_74,
    node_id: u32,
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
struct rseq_data {
    usrptr: *mut rseq,
    len: u32,
    sig: u32,
    event: rseq_event,
    ids: rseq_ids,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct sched_mm_cid {
    active: u32,
    cid: u32,
    node: hlist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct request_queue {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_context {
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct mm_cid_pcpu {
    cid: u32,
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
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
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
struct load_weight {
    weight: u64,
    inv_weight: u32,
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
struct posix_acl {
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
    __slate_bits_0: __slate_bits::__SlateBits324U0,
    dl_timer: hrtimer,
    inactive_timer: hrtimer,
    rq: *mut rq,
    server_pick_task: Option<unsafe extern "C" fn(*mut sched_dl_entity, *mut rq_flags) -> *mut task_struct>,
    pi_se: *mut sched_dl_entity,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct kmap_ctrl {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_cmd {
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
    __slate_bits_0: __slate_bits::__SlateBits67U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits67U1,
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
    __slate_bits_2: __slate_bits::__SlateBits67U2,
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
// Getting shut down
// I'm a workqueue worker
// Dumped core
// Allocating memory to free memory. See memalloc_noreclaim_save()
// Kernel thread cloned from userspace thread
// This thread should not be frozen
// I am kcompactd
// I am kswapd
// All allocations inherit GFP_NOFS. See memalloc_nfs_save()
// All allocations inherit GFP_NOIO. See memalloc_noio_save()
// I am a kernel thread
// Userland is not allowed to meddle with cpus_mask
// Allocations constrained to zones which allow long term pinning.
// See memalloc_pin_save()
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
struct kernfs_open_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_iattrs {
}


#[inline(always)]
fn _cond_resched() -> i32 {
let mut __v0: Option<unsafe extern "C" fn() -> i32>;
__v0 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn() -> i32>>(__SCT__cond_resched as *const ()) };
unsafe { __v0.unwrap()() }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
}


#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__cond_resched_116: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__cond_resched) }) as *mut ();

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
struct rcu_gp_seq {
    norm: u64,
    exp: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
}


// Dependency order vs. p above.
// Dependency order vs. p above.
// RCU_INITIALIZER() - statically initialize an RCU-protected global variable
// @v: The value to statically initialize with.
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
fn rcu_read_lock_sched() {
loop {
preempt_count_inc!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
__acquire_shared!();
__acquire_shared!();
rcu_lock_acquire!();
RCU_LOCKDEP_WARN!(!unsafe { rcu_is_watching() });
}

#[repr(C)]
#[derive(Clone, Copy)]
struct sock {
}


fn rcu_read_unlock_sched() {
RCU_LOCKDEP_WARN!(!unsafe { rcu_is_watching() });
rcu_lock_release!();
__release_shared!();
__release_shared!();
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
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_133: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

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


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_h_5491 {
}


fn mem_alloc_profiling_enabled() -> bool {
false
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
struct old_timespec32 {
    tv_sec: i32,
    tv_nsec: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_uio_h_78 {
    nr_segs: u64,
    folioq_slot: u8,
    xarray_start: i64,
}


// Have the 32-bit jiffies value wrap 5 minutes after boot
// so jiffies wrap bugs show up earlier.
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
fn jiffies_to_msecs(j: u64) -> u32 {
1u64.wrapping_mul(j) as u32
}

fn _msecs_to_jiffies(m: u32) -> u64 {
((m as i64).wrapping_add(1000 / 1000).wrapping_sub(1) / (1000 / 1000)) as u64
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
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
}


fn debug_check_no_obj_freed(address: *const (), size: u64) {
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
struct module_param_attrs {
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
if !(__native_word__10!() || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
__verify_pcpu_ptr__4!();
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
__verify_pcpu_ptr__3!();
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


fn __srcu_read_unlock_fast(ssp: *mut srcu_struct, mut scp: *mut srcu_ctr) {
__release_shared!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
// Avoid leaking the critical section.
if IS_ENABLED!() == 0 {
loop {
__verify_pcpu_ptr__4!();
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
__verify_pcpu_ptr__3!();
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

fn srcu_check_read_flavor(ssp: *mut srcu_struct, read_flavor: i32) {
if IS_ENABLED!() != 0 {
unsafe { __srcu_check_read_flavor(ssp, read_flavor) };
}
}

// srcu_read_lock().
// srcu_read_lock_nmisafe().
// 					0x4		// SRCU-lite is no longer with us.
// srcu_read_lock_fast(), also NMI-safe.
// srcu_read_lock_fast_updown().
fn srcu_read_lock_fast_notrace(ssp: *mut srcu_struct) -> *mut srcu_ctr {
srcu_check_read_flavor(ssp, 4i32);
let retval: *mut srcu_ctr = __srcu_read_lock_fast(ssp);
retval
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
fn class_srcu_fast_notrace_destructor(_T: *mut __SlateRecord_srcu_h_645) {
srcu_read_unlock_fast_notrace(unsafe { (*_T).lock }, unsafe { (*_T).scp });
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


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_taskset {
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


// Tracks mm_cpumask
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


fn set_page_private(page: *mut page, private: u64) {
unsafe {
(*page).__slate_anon_1.__slate_anon_0.private = private;
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_701 {
    v: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wake_irq {
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
struct dev_pm_qos {
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
// mm is unstable for copy_from_user
// MMF_HAS_PINNED: Whether this mm has pinned any pages.  This can be either
// replaced in the future by mm.pinned_vm when it becomes stable, or grow into
// a counter on its own. We're aggresive on this bit for now: even if the
// pinned pages were unpinned later on, we'll still keep this bit set for the
// lifecycle of this mm, just for simplicity.
// FOLL_PIN has run, never cleared
#[inline(always)]
fn compound_info_has_mask() -> bool {
// Limit mask usage to HugeTLB vmemmap optimization (HVO) where it
// makes a difference.
//
// The approach with mask would work in the wider set of conditions,
// but it requires validating that struct pages are naturally aligned
// for all orders up to the MAX_FOLIO_ORDER, which can be tricky.
if IS_ENABLED__2!() == 0 {
return false;
}
is_power_of_2(core::mem::size_of::<page>() as u64)
}

#[inline(always)]
fn _compound_head(page: *const page) -> u64 {
let mut info: u64 = 0;
let mut __v0: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_187() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*page).__slate_anon_1.__slate_anon_2.compound_info) as *const u64) };
info = __v0;
let mut mask: u64 = 0;
if !compound_info_has_mask() {
// Bit 0 encodes PageTail()
if info & 1 != 0 {
return info.wrapping_sub(1);
}
return page as u64;
}
// If compound_info_has_mask() is true the rest of the info encodes
// the mask that converts the address of the tail page to the head page.
//
// No need to clear bit 0 in the mask as 'page' always has it clear.
//
// Let's do it in a branchless manner.
// Non-tail: -1UL, Tail: 0
mask = (info & 1).wrapping_sub(1);
// Non-tail: -1UL, Tail: info
mask |= info;
(page as u64) & mask
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
fn set_compound_head(tail: *mut page, head: *const page, order: u32) {
let mut shift: u32 = 0;
let mut mask: u64 = 0;
if !compound_info_has_mask() {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_188() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__4!((*tail).__slate_anon_1.__slate_anon_2.compound_info, (head as u64) | 1);
if true {
break;
}
}
return;
}
// If the size of struct page is power-of-2, bits [shift:0] of the
// virtual address of compound head are zero.
//
// Calculate mask that can be applied to the virtual address of
// the tail page to get address of the head page.
shift = order.wrapping_add(order_base_2!(core::mem::size_of::<page>() as u64) as u32);
mask = GENMASK__3!(63i32, shift);
// Bit 0 encodes PageTail()
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_189() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__4!((*tail).__slate_anon_1.__slate_anon_2.compound_info, mask | 1);
if true {
break;
}
}
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
fn clear_compound_head(page: *mut page) {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_190() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__8!((*page).__slate_anon_1.__slate_anon_2.compound_info, 0i32);
if true {
break;
}
}
}

// folio_page - Return a page from a folio.
// @folio: The folio.
// @n: The page number to return.
//
// @n is relative to the start of the folio.  This function does not
// check that the page number lies within @folio; the caller is presumed
// to have a reference to the page.
#[inline(always)]
fn PageTail(page: *const page) -> i32 {
let mut __v0: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_191() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*page).__slate_anon_1.__slate_anon_2.compound_info) as *const u64) };
(__v0 & 1) as i32
}

#[inline(always)]
fn PageCompound(page: *const page) -> i32 {
let __v0: bool = if test_bit__3!(6i32, unsafe { core::ptr::addr_of!((*page).flags.f) }) != 0 {
true
} else {
let mut __v2: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_192() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*page).__slate_anon_1.__slate_anon_2.compound_info) as *const u64) };
__v2 & 1 != 0
};
__v0 as i32
}

fn PagePoisoned(page: &page) -> i32 {
let mut __v0: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_193() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*(page as *const page)).flags.f) as *const u64) };
(__v0 == (1i64.wrapping_neg() as u64)) as i32
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
struct disk_stats {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_holder_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct partition_meta_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_cluster_info {
}


fn const_folio_flags(folio: *const folio, n: u32) -> *const u64 {
let page: *const page = unsafe { core::ptr::addr_of!((*folio).__slate_anon_0.page) };
VM_BUG_ON_PGFLAGS!();
VM_BUG_ON_PGFLAGS!();
unsafe { core::ptr::addr_of!((*page.wrapping_offset(n as isize)).flags.f) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct swap_ops {
}


fn folio_flags(folio: *mut folio, n: u32) -> *mut u64 {
let page: *mut page = unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.page) };
VM_BUG_ON_PGFLAGS!();
VM_BUG_ON_PGFLAGS!();
unsafe { core::ptr::addr_of_mut!((*page.wrapping_offset(n as isize)).flags.f) }
}

// Which page is the flag stored in
// Macros to create function definitions for page flags
#[inline(always)]
fn PageLRU(page: *const page) -> i32 {
let __v0: bool = if true && false {
let mut __v1: *const page;
__v1 = _compound_head(page) as *const page;
((unsafe { core::ptr::addr_of!((*__v1).flags.f) }) as u64) != 0
} else {
false
};
let __v2: i32 = if __v0 && false {
let mut __v3: *const page;
__v3 = _compound_head(page) as *const page;
const_test_bit(5u64, (unsafe { core::ptr::addr_of!((*__v3).flags.f) }) as *const u64) as i32
} else {
let mut __v4: *const page;
__v4 = _compound_head(page) as *const page;
_test_bit(5u64, (unsafe { core::ptr::addr_of!((*__v4).flags.f) }) as *const u64) as i32
};
__v2
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mempolicy_h_54 {
    cpuset_mems_allowed: __SlateRecord_nodemask_types_h_17,
    user_nodemask: __SlateRecord_nodemask_types_h_17,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_pin {
}


#[inline(always)]
fn PageReserved(page: &page) -> i32 {
let __v0: bool = if true && false {
let mut __v1: *const page;
let mut __v2: *const page;
__v2 = page as *const page;
__v1 = __v2;
((unsafe { core::ptr::addr_of!((*__v1).flags.f) }) as u64) != 0
} else {
false
};
let __v3: i32 = if __v0 && false {
let mut __v4: *const page;
let mut __v5: *const page;
__v5 = page as *const page;
__v4 = __v5;
const_test_bit(13u64, (unsafe { core::ptr::addr_of!((*__v4).flags.f) }) as *const u64) as i32
} else {
let mut __v6: *const page;
let mut __v7: *const page;
__v7 = page as *const page;
__v6 = __v7;
_test_bit(13u64, (unsafe { core::ptr::addr_of!((*__v6).flags.f) }) as *const u64) as i32
};
__v3
}

#[inline(always)]
fn ClearPageReserved(page: *mut page) {
let mut __v0: *mut page;
VM_BUG_ON_PGFLAGS!();
let mut __v1: *mut page;
VM_BUG_ON_PGFLAGS!();
__v1 = page;
__v0 = __v1;
clear_bit(13i64, (unsafe { core::ptr::addr_of_mut!((*__v0).flags.f) }) as *mut u64);
}

#[inline(always)]
fn __ClearPageReserved(page: *mut page) {
let __v0: bool = if true && false {
let mut __v1: *mut page;
let mut __v2: *mut page;
__v2 = page;
__v1 = __v2;
((unsafe { core::ptr::addr_of_mut!((*__v1).flags.f) }) as u64) != 0
} else {
false
};
if __v0 && false {
let mut __v3: *mut page;
let mut __v4: *mut page;
__v4 = page;
__v3 = __v4;
generic___clear_bit(13u64, (unsafe { core::ptr::addr_of_mut!((*__v3).flags.f) }) as *mut u64);
} else {
let mut __v5: *mut page;
let mut __v6: *mut page;
__v6 = page;
__v5 = __v6;
___clear_bit(13u64, (unsafe { core::ptr::addr_of_mut!((*__v5).flags.f) }) as *mut u64);
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_page_alloc_c_7018 {
}


#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

mod __slate_bits {
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits624U0 {
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
    pub struct __SlateBits507U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits324U0 {
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
    pub struct __SlateBits67U0 {
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
    pub struct __SlateBits67U1 {
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
    pub struct __SlateBits447U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits710U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits788U0 {
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
    pub struct __SlateBits327U0 {
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
    pub struct __SlateBits67U2 {
        #[bits(1)]
        pub mce_ripv: u64,
        #[bits(1)]
        pub mce_whole_page: u64,
        #[bits(62)]
        pub __mce_reserved: u64,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits105U0 {
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
    pub struct __SlateBits123U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits817U0 {
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
    pub struct __SlateBits814U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits817U1 {
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
    pub struct __SlateBits817U2 {
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
    pub struct __SlateBits817U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits661U0 {
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
    pub struct __SlateBits104U0 {
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
    pub struct __SlateBits150U0 {
        #[bits(24)]
        pub intel_native_model_id: u32,
        #[bits(8)]
        pub intel_type: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits151U0 {
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
    pub struct __SlateBits141U0 {
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
    pub struct __SlateBits142U0 {
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
    pub struct __SlateBits139U0 {
        #[bits(1)]
        pub initialized: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits181U0 {
        #[bits(1)]
        pub faultable: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
}

fn PageHighMem(page: *const page) -> i32 {
0
}

#[inline(always)]
fn folio_test_mlocked(folio: *const folio) -> bool {
test_bit!(20i32, const_folio_flags(folio, 0u32)) != 0
}

#[inline(always)]
fn __folio_clear_mlocked(folio: *mut folio) {
__clear_bit!(20i32, folio_flags(folio, 0u32));
}

fn PageHWPoison(page: *const page) -> i32 {
0
}

#[inline(always)]
fn __ClearPageReported(page: *mut page) {
let __v0: bool = if true && false {
let mut __v1: *mut page;
let mut __v2: *mut page;
__v2 = page;
__v1 = __v2;
((unsafe { core::ptr::addr_of_mut!((*__v1).flags.f) }) as u64) != 0
} else {
false
};
if __v0 && false {
let mut __v3: *mut page;
let mut __v4: *mut page;
__v4 = page;
__v3 = __v4;
generic___clear_bit(3u64, (unsafe { core::ptr::addr_of_mut!((*__v3).flags.f) }) as *mut u64);
} else {
let mut __v5: *mut page;
let mut __v6: *mut page;
__v6 = page;
__v5 = __v6;
___clear_bit(3u64, (unsafe { core::ptr::addr_of_mut!((*__v5).flags.f) }) as *mut u64);
}
}

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
#[inline(always)]
fn folio_test_anon(folio: *const folio) -> bool {
((unsafe { (*folio).__slate_anon_0.__slate_anon_0.mapping }) as u64) & 1 != 0
}

#[inline(always)]
fn folio_test_head(folio: *const folio) -> bool {
test_bit!(6i32, const_folio_flags(folio, 0u32)) != 0
}

#[inline(always)]
fn PageHead(page: *const page) -> i32 {
PF_POISONED_CHECK__2!(page);
test_bit__3!(6i32, unsafe { core::ptr::addr_of!((*page).flags.f) })
}

#[inline(always)]
fn __SetPageHead(page: *mut page) {
let __v0: bool = if true && false {
let mut __v1: *mut page;
__v1 = page;
((unsafe { core::ptr::addr_of_mut!((*__v1).flags.f) }) as u64) != 0
} else {
false
};
if __v0 && false {
let mut __v2: *mut page;
__v2 = page;
generic___set_bit(6u64, (unsafe { core::ptr::addr_of_mut!((*__v2).flags.f) }) as *mut u64);
} else {
let mut __v3: *mut page;
__v3 = page;
___set_bit(6u64, (unsafe { core::ptr::addr_of_mut!((*__v3).flags.f) }) as *mut u64);
}
}

fn folio_test_large(folio: *const folio) -> bool {
folio_test_head(folio)
}

fn folio_set_large_rmappable(folio: *mut folio) {
}

fn page_type_has_type(page_type: i32) -> bool {
page_type < -16777216i32
}

fn page_has_type(page: *const page) -> bool {
page_type_has_type(data_race!(unsafe { (*page).__slate_anon_2.page_type }) as i32)
}

#[inline(always)]
fn PageBuddy(page: *const page) -> i32 {
(data_race!((unsafe { (*page).__slate_anon_2.page_type }) >> 24i32) == 240) as i32
}

#[inline(always)]
fn __SetPageBuddy(page: *mut page) {
if PageBuddy(page as *const page) != 0 {
return;
}
VM_BUG_ON_PAGE!();
unsafe {
(*page).__slate_anon_2.page_type = 4026531840u32;
}
}

#[inline(always)]
fn __ClearPageBuddy(page: *mut page) {
if (unsafe { (*page).__slate_anon_2.page_type }) == !0 {
return;
}
VM_BUG_ON_PAGE!();
unsafe {
(*page).__slate_anon_2.page_type = !0;
}
}

#[inline(always)]
fn PageOffline(page: *const page) -> i32 {
(data_race!((unsafe { (*page).__slate_anon_2.page_type }) >> 24i32) == 241) as i32
}

#[inline(always)]
fn __ClearPageOffline(page: &mut page) {
if (unsafe { (*(page as *mut page)).__slate_anon_2.page_type }) == !0 {
return;
}
VM_BUG_ON_PAGE!();
unsafe {
(*(page as *mut page)).__slate_anon_2.page_type = !0;
}
}

#[inline(always)]
fn PageZsmalloc(page: *const page) -> i32 {
(data_race!((unsafe { (*page).__slate_anon_2.page_type }) >> 24i32) == 246) as i32
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.page_alloc.is_free_buddy_page")]
extern "C" fn is_free_buddy_page(page: *const page) -> bool {
let pfn: u64 = ((unsafe { page.offset_from((vmemmap_base as *mut page) as *const page) }) as i64) as u64;
let mut order: u32 = 0;
while order < 11u32 {
let head: *const page = page.wrapping_offset(-((pfn & ((1i32 << order).wrapping_sub(1) as u64)) as isize));
let __v0: bool = if PageBuddy(head) != 0 {
let mut __v1: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1101() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*head).__slate_anon_1.__slate_anon_0.private) as *const u64) };
__v1 >= (order as u64)
} else {
false
};
if __v0 {
break;
}
order = order.wrapping_add(1);
}
order <= 10
}

#[inline(always)]
fn PageMovableOps(page: *const page) -> i32 {
let __v0: bool = if true && false {
let mut __v1: *const page;
let mut __v2: *const page;
__v2 = _compound_head(page) as *const page;
__v1 = __v2;
((unsafe { core::ptr::addr_of!((*__v1).flags.f) }) as u64) != 0
} else {
false
};
let __v3: i32 = if __v0 && false {
let mut __v4: *const page;
let mut __v5: *const page;
__v5 = _compound_head(page) as *const page;
__v4 = __v5;
const_test_bit(3u64, (unsafe { core::ptr::addr_of!((*__v4).flags.f) }) as *const u64) as i32
} else {
let mut __v6: *const page;
let mut __v7: *const page;
__v7 = _compound_head(page) as *const page;
__v6 = __v7;
_test_bit(3u64, (unsafe { core::ptr::addr_of!((*__v6).flags.f) }) as *const u64) as i32
};
__v3
}

fn page_has_movable_ops(page: *const page) -> bool {
PageMovableOps(page) != 0 && (PageOffline(page) != 0 || PageZsmalloc(page) != 0)
}

// Flags checked when a page is freed.  Pages being freed should not have
// these flags set.  If they are, there is a problem.
// Flags checked when a page is prepped for return by the page allocator.
// Pages being prepped should not have these flags set.  If they are set,
// there has been a kernel bug or struct page corruption.
//
// __PG_HWPOISON is exceptional because it needs to be kept beyond page's
// alloc-free cycle to prevent from reusing the page.
// Flags stored in the second page of a compound page.  They may overlap
// the CHECK_AT_FREE flags above, so need to be cleared.
// order
#[repr(C)]
#[derive(Clone, Copy)]
struct zswap_lruvec_state {
}


#[unsafe(no_mangle)]
static migratetype_names: __SlateAlign16<[*const u8; 4]> = __SlateAlign16([(b"Unmovable\0".as_ptr() as *mut u8) as *const u8, (b"Movable\0".as_ptr() as *mut u8) as *const u8, (b"Reclaimable\0".as_ptr() as *mut u8) as *const u8, (b"HighAtomic\0".as_ptr() as *mut u8) as *const u8]);

fn migratetype_is_mergeable(mt: i32) -> bool {
mt < 3
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".data..read_mostly")]
static mut page_group_by_mobility_disabled: i32 = unsafe { core::mem::zeroed() };

#[repr(C)]
#[derive(Clone, Copy)]
struct free_area {
    free_list: [list_head; 4],
    nr_free: u64,
}


// We do arithmetic on the LRU lists in various places in the code,
// so it is important to keep the active lists LRU_ACTIVE higher in
// the array than the corresponding inactive lists, and to keep
// the *_FILE lists LRU_FILE higher than the corresponding _ANON lists.
//
// This has to be kept in sync with the statistics in zone_stat_item
// above and the descriptions in vmstat_text in mm/vmstat.c
#[repr(C)]
#[derive(Clone, Copy)]
struct lru_cost {
    count: u64,
    last_rotated: u64,
    last_io: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct lruvec {
    lists: [list_head; 5],
    lru_lock: spinlock,
    cost: [lru_cost; 2],
    cost_lock: spinlock,
    nonresident_age: __SlateRecord_types_h_197,
    refaults: [u64; 2],
    flags: u64,
    zswap_lruvec_state: zswap_lruvec_state,
}


// Flags used in pcp->flags field.
//
// PCPF_PREV_FREE_HIGH_ORDER: a high-order page is freed in the
// previous page freeing.  To avoid to drain PCP for an accident
// high-order page freeing.
//
// PCPF_FREE_HIGH_BATCH: preserve "pcp->batch" pages in PCP before
// draining PCP for consecutive high-order pages freeing without
// allocation if data cache slice of CPU is large enough.  To reduce
// zone lock contention and keep cache-hot pages reusing.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct per_cpu_pages {
    lock: spinlock,
    count: i32,
    high: i32,
    high_min: i32,
    high_max: i32,
    batch: i32,
    flags: u8,
    alloc_factor: u8,
    expire: u8,
    free_count: i16,
    lists: [list_head; 12],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct per_cpu_zonestat {
    vm_stat_diff: [i8; 10],
    stat_threshold: i8,
    vm_numa_event: [u64; 6],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct per_cpu_nodestat {
    stat_threshold: i8,
    vm_node_stat_diff: [i8; 66],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct zone {
    _watermark: [u64; 4],
    watermark_boost: u64,
    nr_reserved_highatomic: u64,
    nr_free_highatomic: u64,
    lowmem_reserve: [i64; 4],
    node: i32,
    zone_pgdat: *mut pglist_data,
    per_cpu_pageset: *mut per_cpu_pages,
    per_cpu_zonestats: *mut per_cpu_zonestat,
    pageset_high_min: i32,
    pageset_high_max: i32,
    pageset_batch: i32,
    zone_start_pfn: u64,
    managed_pages: __SlateRecord_types_h_197,
    spanned_pages: u64,
    present_pages: u64,
    name: *const u8,
    initialized: i32,
    _pad1_: cacheline_padding,
    free_area: [free_area; 11],
    flags: u64,
    lock: spinlock,
    trylock_free_pages: llist_head,
    _pad2_: cacheline_padding,
    percpu_drift_mark: u64,
    compact_cached_free_pfn: u64,
    compact_cached_migrate_pfn: [u64; 2],
    compact_init_migrate_pfn: u64,
    compact_init_free_pfn: u64,
    compact_considered: u32,
    compact_defer_shift: u32,
    compact_order_failed: i32,
    compact_blockskip_flush: bool,
    contiguous: bool,
    _pad3_: cacheline_padding,
    vm_stat: [__SlateRecord_types_h_197; 10],
    vm_numa_event: [__SlateRecord_types_h_197; 6],
    vmemmap_tails: [*mut page; 16],
}


fn wmark_pages(z: *const zone, w: u32) -> u64 {
unsafe { *((*z)._watermark.as_ptr() as *const u64).wrapping_offset((w as u32) as isize) }.wrapping_add(unsafe { (*z).watermark_boost })
}

fn min_wmark_pages(z: *const zone) -> u64 {
wmark_pages(z, 0u32)
}

fn low_wmark_pages(z: *const zone) -> u64 {
wmark_pages(z, 1u32)
}

fn high_wmark_pages(z: *const zone) -> u64 {
wmark_pages(z, 2u32)
}

fn zone_managed_pages(zone: *const zone) -> u64 {
atomic_long_read(unsafe { core::ptr::addr_of!((*zone).managed_pages) }) as u64
}

fn zone_end_pfn(zone: *const zone) -> u64 {
unsafe { (*zone).zone_start_pfn }.wrapping_add(unsafe { (*zone).spanned_pages })
}

fn zone_spans_pfn(zone: *const zone, pfn: u64) -> bool {
(unsafe { (*zone).zone_start_pfn }) <= pfn && pfn < zone_end_pfn(zone)
}

fn zone_is_initialized(zone: &zone) -> bool {
(unsafe { (*(zone as *const zone)).initialized }) != 0
}

fn memdesc_zonenum(flags: *const __SlateRecord_mm_types_h_41) -> u32 {
ASSERT_EXCLUSIVE_BITS!((*flags).f, 3u64 << (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_sub(2).wrapping_mul((true as i32) as u64));
((unsafe { (*flags).f }) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_sub(2).wrapping_mul((true as i32) as u64) & 3u64) as u32
}

fn page_zonenum(page: *const page) -> u32 {
memdesc_zonenum(unsafe { core::ptr::addr_of!((*page).flags) })
}

fn folio_zonenum(folio: *const folio) -> u32 {
memdesc_zonenum(unsafe { core::ptr::addr_of!((*folio).__slate_anon_0.__slate_anon_0.flags) })
}

// Maximum number of zones on a zonelist
#[repr(C)]
#[derive(Clone, Copy)]
struct zoneref {
    zone: *mut zone,
    zone_idx: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct zonelist {
    _zonerefs: [zoneref; 257],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pglist_data {
    node_zones: [zone; 4],
    node_zonelists: [zonelist; 2],
    nr_zones: i32,
    node_start_pfn: u64,
    node_present_pages: u64,
    node_spanned_pages: u64,
    node_id: i32,
    kswapd_wait: wait_queue_head,
    pfmemalloc_wait: wait_queue_head,
    reclaim_wait: [wait_queue_head; 4],
    nr_writeback_throttled: __SlateRecord_types_h_190,
    nr_reclaim_start: u64,
    kswapd: *mut task_struct,
    kswapd_order: i32,
    kswapd_highest_zoneidx: u32,
    kswapd_failures: __SlateRecord_types_h_190,
    kcompactd_max_order: i32,
    kcompactd_highest_zoneidx: u32,
    kcompactd_wait: wait_queue_head,
    kcompactd: *mut task_struct,
    proactive_compact_trigger: bool,
    totalreserve_pages: u64,
    min_unmapped_pages: u64,
    min_slab_pages: u64,
    _pad1_: cacheline_padding,
    __lruvec: lruvec,
    flags: u64,
    _pad2_: cacheline_padding,
    per_cpu_nodestats: *mut per_cpu_nodestat,
    vm_stat: [__SlateRecord_types_h_197; 66],
    memtier: *mut memory_tier,
}


fn managed_zone(zone: *const zone) -> bool {
zone_managed_pages(zone) != 0
}

fn populated_zone(zone: *const zone) -> bool {
(unsafe { (*zone).present_pages }) != 0
}

fn zone_to_nid(zone: *const zone) -> i32 {
unsafe { (*zone).node }
}

#[unsafe(no_mangle)]
static mut movable_zone: i32 = unsafe { core::mem::zeroed() };

fn is_highmem_idx(idx: u32) -> i32 {
0
}

fn is_highmem(zone: *const zone) -> i32 {
is_highmem_idx(zone_idx__2!(zone) as u32)
}

fn has_managed_dma() -> bool {
has_managed_zone(0u32)
}

// for_each_online_pgdat - helper macro to iterate over all online nodes
// @pgdat: pointer to a pg_data_t variable
// for_each_zone - helper macro to iterate over all memory zones
// @zone: pointer to struct zone variable
//
// The user only needs to declare the zone variable, for_each_zone
// fills it in.
// do nothing
fn zonelist_zone(zoneref: *mut zoneref) -> *mut zone {
unsafe { (*zoneref).zone }
}

fn zonelist_zone_idx(zoneref: *const zoneref) -> i32 {
unsafe { (*zoneref).zone_idx }
}

fn zonelist_node_idx(zoneref: *const zoneref) -> i32 {
zone_to_nid((unsafe { (*zoneref).zone }) as *const zone)
}

#[inline(always)]
fn next_zones_zonelist(z: *mut zoneref, highest_zoneidx: u32, nodes: *const __SlateRecord_nodemask_types_h_17) -> *mut zoneref {
let __v0: bool = nodes.is_null() && (zonelist_zone_idx(z as *const zoneref) as u32) <= (highest_zoneidx as u32);
if __v0 {
return z;
}
unsafe { __next_zones_zonelist(z, highest_zoneidx, nodes) }
}

fn first_zones_zonelist(zonelist: *mut zonelist, highest_zoneidx: u32, nodes: *const __SlateRecord_nodemask_types_h_17) -> *mut zoneref {
next_zones_zonelist(unsafe { (*zonelist)._zonerefs.as_mut_ptr() as *mut zoneref }, highest_zoneidx, nodes)
}

// for_each_zone_zonelist_nodemask - helper macro to iterate over valid zones in a zonelist at or below a given zone index and within a nodemask
// @zone: The current zone in the iterator
// @z: The current pointer within zonelist->_zonerefs being iterated
// @zlist: The zonelist being iterated
// @highidx: The zone index of the highest zone to return
// @nodemask: Nodemask allowed by the allocator
//
// This iterator iterates though all zones at or below a given zone index and
// within a given nodemask
// for_each_zone_zonelist - helper macro to iterate over valid zones in a zonelist at or below a given zone index
// @zone: The current zone in the iterator
// @z: The current pointer within zonelist->zones being iterated
// @zlist: The zonelist being iterated
// @highidx: The zone index of the highest zone to return
//
// This iterator iterates though all zones at or below a given zone index.
fn pfn_to_section_nr(pfn: u64) -> u64 {
pfn >> 15i32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mem_section_usage {
    rcu: callback_head,
    subsection_map: [u64; 1],
    pageblock_flags: [u64; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mem_section {
    section_mem_map: u64,
    usage: *mut mem_section_usage,
}


fn section_to_usemap(ms: *mut mem_section) -> *mut u64 {
unsafe { (*(*ms).usage).pageblock_flags.as_mut_ptr() as *mut u64 }
}

fn __nr_to_section(nr: u64) -> *mut mem_section {
let root: u64 = SECTION_NR_TO_ROOT!(nr);
if unlikely!(root >= (1u64 << if pgtable_l5_enabled!() != 0 { 52i32 } else { 46i32 }.wrapping_sub(27)).wrapping_add(4096u64 / (core::mem::size_of::<mem_section>() as u64)).wrapping_sub(1) / (4096u64 / (core::mem::size_of::<mem_section>() as u64))) != 0 {
return core::ptr::null_mut::<mem_section>();
}
if unsafe { mem_section }.is_null() || unsafe { *unsafe { mem_section }.wrapping_offset(root as isize) }.is_null() {
return core::ptr::null_mut::<mem_section>();
}
unsafe { *unsafe { mem_section }.wrapping_offset(root as isize) }.wrapping_offset((nr & ((_AC!() << 12i32) / (core::mem::size_of::<mem_section>() as u64)).wrapping_sub(1)) as isize)
}

fn valid_section(section: *const mem_section) -> i32 {
(!section.is_null() && (unsafe { (*section).section_mem_map }) & BIT!(1i32) != 0) as i32
}

fn early_section(section: *const mem_section) -> i32 {
(!section.is_null() && (unsafe { (*section).section_mem_map }) & BIT!(3i32) != 0) as i32
}

fn __pfn_to_section(pfn: u64) -> *mut mem_section {
__nr_to_section(pfn_to_section_nr(pfn))
}

fn subsection_map_index(pfn: u64) -> i32 {
((pfn & 32767u64) / 512u64) as i32
}

fn pfn_section_valid(ms: *mut mem_section, pfn: u64) -> i32 {
let idx: i32 = subsection_map_index(pfn);
let mut __v0: *mut mem_section_usage;
loop {
if !(__native_word__19!() || ((const { core::mem::size_of::<*mut mem_section_usage>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_201() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*ms).usage) as *const *mut mem_section_usage) };
let usage: *mut mem_section_usage = __v0;
let __v2: i32 = if !usage.is_null() {
test_bit__2!(idx, unsafe { (*usage).subsection_map.as_mut_ptr() as *mut u64 })
} else {
0
};
__v2
}

fn pfn_valid(pfn: u64) -> i32 {
let mut ret: i32 = 0;
// Ensure the upper PAGE_SHIFT bits are clear in the
// pfn. Else it might lead to false positives when
// some of the upper bits are set, but the lower bits
// match a valid pfn.
if PHYS_PFN!(PFN_PHYS!(pfn)) != pfn {
return 0;
}
if pfn_to_section_nr(pfn) >= 1u64 << if pgtable_l5_enabled!() != 0 { 52i32 } else { 46i32 }.wrapping_sub(27) {
return 0;
}
let ms: *mut mem_section = __pfn_to_section(pfn);
rcu_read_lock_sched();
if valid_section(ms as *const mem_section) == 0 {
rcu_read_unlock_sched();
return 0;
}
// Traditionally early sections always returned pfn_valid() for
// the entire section-sized span.
let __v0: bool = early_section(ms as *const mem_section) != 0 || pfn_section_valid(ms, pfn) != 0;
ret = __v0 as i32;
rcu_read_unlock_sched();
ret
}

fn cpumask_of_node(node: i32) -> *const cpumask {
(unsafe { core::ptr::addr_of_mut!(node_to_cpumask_map[node as usize]) as *mut cpumask }) as *const cpumask
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".data..percpu")]
static mut numa_node: i32 = unsafe { core::mem::zeroed() };

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

fn numa_mem_id() -> i32 {
numa_node_id()
}

fn gfpflags_allow_blocking(gfp_flags: u32) -> bool {
gfp_flags & (BIT!(10i32) as u32) != 0
}

// GFP_ZONE_BAD is a bitmap for all combinations of __GFP_DMA, __GFP_DMA32
// __GFP_HIGHMEM and __GFP_MOVABLE that are not permitted. One flag per
// entry starting with bit 0. Bit is set if the combination is not
// allowed.
fn gfp_zone(flags: u32) -> u32 {
let mut z: u32 = 0;
let bit: i32 = (flags & ((BIT!(0i32) as u32) | (BIT!(1i32) as u32) | (BIT!(2i32) as u32) | (BIT!(3i32) as u32))) as i32;
z = ((2i32 | 0i32 << BIT!(0i32).wrapping_mul(2) | 2i32 << BIT!(1i32).wrapping_mul(2) | 1i32 << BIT!(2i32).wrapping_mul(2) | 2i32 << BIT!(3i32).wrapping_mul(2) | 0i32 << (BIT!(3i32) | BIT!(0i32)).wrapping_mul(2) | 3i32 << (BIT!(3i32) | BIT!(1i32)).wrapping_mul(2) | 1i32 << (BIT!(3i32) | BIT!(2i32)).wrapping_mul(2)) >> bit.wrapping_mul(2) & 3i32) as u32;
VM_BUG_ON!();
z
}

fn gfp_zonelist(flags: u32) -> i32 {
if unlikely__3!(flags & (BIT!(21i32) as u32)) != 0 {
return 1;
}
0
}

fn node_zonelist(nid: i32, flags: u32) -> *mut zonelist {
unsafe { (*NODE_DATA!(nid)).node_zonelists.as_mut_ptr() as *mut zonelist }.wrapping_offset(gfp_zonelist(flags) as isize)
}

fn arch_free_page(page: *mut page, order: i32) {
}

fn arch_alloc_page(page: *mut page, order: i32) {
}

fn warn_if_node_offline(this_node: i32, mut gfp_mask: u32) {
let warn_gfp: u32 = gfp_mask & ((BIT!(21i32) as u32) | (BIT!(13i32) as u32));
if warn_gfp != (BIT!(21i32) as u32) | (BIT!(13i32) as u32) {
return;
}
if node_online!(this_node) != 0 {
return;
}
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x014%pGg allocation from offline node %d\n".as_ptr() as *const u8, core::ptr::addr_of_mut!(gfp_mask), this_node) };
unsafe { dump_stack() };
}

fn gfp_has_flags(gfp: u32, flags: u32) -> bool {
gfp & flags == flags
}

fn gfp_has_io_fs(gfp: u32) -> bool {
gfp_has_flags(gfp, (BIT!(6i32) as u32) | (BIT!(7i32) as u32))
}

fn gfp_compaction_allowed(gfp_mask: u32) -> bool {
IS_ENABLED__2!() != 0 && gfp_mask & (BIT!(6i32) as u32) != 0
}

fn clear_page_tag_ref(page: *mut page) {
}

fn pgalloc_tag_split(folio: *mut folio, old_order: i32, new_order: i32) {
}

fn pgalloc_tag_get(page: *mut page) -> *mut alloc_tag {
core::ptr::null_mut::<alloc_tag>()
}

fn current_gfp_context(mut flags: u32) -> u32 {
let mut pflags: u32 = 0;
let mut __v0: u32;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_205() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*get_current()).flags) as *const u32) };
pflags = __v0;
if unlikely__3!(pflags & 269221888u32) != 0 {
// NOIO implies both NOIO and NOFS and it is a weaker context
// so always make sure it makes precedence
if pflags & 524288 != 0 {
let __v2: u32 = flags;
flags = __v2 & !((BIT!(6i32) as u32) | (BIT!(7i32) as u32));
} else {
if pflags & 262144 != 0 {
let __v4: u32 = flags;
flags = __v4 & !(BIT!(7i32) as u32);
}
}
if pflags & 268435456 != 0 {
let __v6: u32 = flags;
flags = __v6 & !(BIT!(3i32) as u32);
}
}
flags
}

fn fs_reclaim_acquire(gfp_mask: u32) {
}

fn fs_reclaim_release(gfp_mask: u32) {
}

fn might_alloc(gfp_mask: u32) {
fs_reclaim_acquire(gfp_mask);
fs_reclaim_release(gfp_mask);
if (unsafe { (*get_current()).flags }) & 2048 != 0 {
return;
}
might_sleep_if!(gfpflags_allow_blocking(gfp_mask));
}

fn memalloc_flags_save(flags: u32) -> u32 {
let oldflags: u32 = !unsafe { (*get_current()).flags } & flags;
let __v0: *mut task_struct = get_current();
unsafe {
(*__v0).flags = (*__v0).flags | flags;
}
oldflags
}

fn memalloc_flags_restore(flags: u32) {
let __v0: *mut task_struct = get_current();
let __v2: u32 = (unsafe { (*__v0).flags }) & !flags;
unsafe {
(*__v0).flags = __v2;
}
}

fn memalloc_noreclaim_save() -> u32 {
memalloc_flags_save(2048u32)
}

fn memalloc_noreclaim_restore(flags: u32) {
memalloc_flags_restore(flags);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_ref_data {
    count: __SlateRecord_types_h_197,
    release: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    confirm_switch: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    __slate_bits_0: __slate_bits::__SlateBits507U0,
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


fn __page_ref_set(page: *mut page, v: i32) {
}

fn __page_ref_mod_and_test(page: *mut page, v: i32, ret: i32) {
}

fn page_ref_count(page: *const page) -> i32 {
atomic_read(unsafe { core::ptr::addr_of!((*page)._refcount) })
}

fn folio_ref_count(folio: *const folio) -> i32 {
page_ref_count(unsafe { core::ptr::addr_of!((*folio).__slate_anon_0.page) })
}

fn page_count(page: *const page) -> i32 {
folio_ref_count(page_folio__2!(page))
}

fn set_page_count(page: *mut page, v: i32) {
atomic_set(unsafe { core::ptr::addr_of_mut!((*page)._refcount) }, v);
if page_ref_tracepoint_active!() != 0 {
__page_ref_set(page, v);
}
}

fn page_ref_dec_and_test(page: *mut page) -> i32 {
let ret: i32 = atomic_dec_and_test(unsafe { core::ptr::addr_of_mut!((*page)._refcount) }) as i32;
if page_ref_tracepoint_active!() != 0 {
__page_ref_mod_and_test(page, 1i32.wrapping_neg(), ret);
}
ret
}

fn page_table_check_alloc(page: *mut page, order: u32) {
}

fn page_table_check_free(page: *mut page, order: u32) {
}

fn kasan_hw_tags_enabled() -> bool {
false
}

fn kasan_enable_current() {
}

fn kasan_disable_current() {
}

core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_numa_node: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_node_states: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_movable_zone: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_nr_node_ids: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_nr_online_nodes: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_split_page: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_alloc_pages_bulk_noprof: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol___alloc_frozen_pages_noprof: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {7} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_alloc_pages_node_noprof: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {8} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol___folio_alloc_noprof: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {9} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_get_free_pages_noprof: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {10} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_get_zeroed_page_noprof: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {11} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol___free_pages: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {12} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_free_pages: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {13} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_alloc_pages_exact_noprof: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {14} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_free_pages_exact: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {15} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_nr_free_buffer_pages: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {16} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_adjust_managed_page_count: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {17} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_free_reserved_pages: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {18} ; .previous\n.section\t\".initcall2.init\", \"a\"\t\t\n__initcall__kmod_page_alloc__1095_6757_init_per_zone_wmark_min2:\t\t\t\n.long\t{19} - .\t\n.previous\t\t\t\t\t\n\n.section \".export_symbol\",\"a\" ; __export_symbol_is_free_buddy_page: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {20} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_alloc_pages_nolock_noprof: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {21} ; .previous", sym numa_node, sym node_states, sym movable_zone, sym nr_node_ids, sym nr_online_nodes, sym split_page, sym alloc_pages_bulk_noprof, sym __alloc_frozen_pages_noprof, sym alloc_pages_node_noprof, sym __folio_alloc_noprof, sym get_free_pages_noprof, sym get_zeroed_page_noprof, sym __free_pages, sym free_pages, sym alloc_pages_exact_noprof, sym free_pages_exact, sym nr_free_buffer_pages, sym adjust_managed_page_count, sym free_reserved_pages, sym init_per_zone_wmark_min, sym is_free_buddy_page, sym alloc_pages_nolock_noprof, options(att_syntax));

fn kasan_has_integrated_init() -> bool {
kasan_hw_tags_enabled()
}

fn kasan_poison_pages(page: *mut page, order: u32, init: bool) {
}

fn kasan_unpoison_pages(page: *mut page, order: u32, init: bool) -> bool {
false
}

fn kasan_reset_tag(addr: *const ()) -> *mut () {
addr as *mut ()
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


#[repr(C)]
#[derive(Clone, Copy)]
struct cpu_cacheinfo {
    info_list: *mut cacheinfo,
    per_cpu_data_slice_size: u32,
    num_levels: u32,
    num_leaves: u32,
    cpu_map_populated: bool,
    early_ci_levels: bool,
}


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
struct __SlateRecord_capability_h_24 {
    val: u64,
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
struct ctl_dir {
    header: ctl_table_header,
    root: rb_root,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ctl_table_set {
    is_seen: Option<unsafe extern "C" fn(*mut ctl_table_set) -> i32>,
    dir: ctl_dir,
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
struct rhash_head {
    next: *mut rhash_head,
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
    __slate_bits_0: __slate_bits::__SlateBits327U0,
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

fn __fatal_signal_pending(p: *mut task_struct) -> i32 {
unlikely__2!(sigismember(unsafe { core::ptr::addr_of_mut!((*p).pending.signal) }, 9i32)) as i32
}

fn fatal_signal_pending(p: *mut task_struct) -> i32 {
let __v0: bool = task_sigpending(p) != 0 && __fatal_signal_pending(p) != 0;
__v0 as i32
}

fn task_rlimit(task: *const task_struct, limit: u32) -> u64 {
let mut __v0: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_376() };
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

fn totalram_pages_add(count: i64) {
atomic_long_add(count, unsafe { core::ptr::addr_of_mut!(_totalram_pages) });
}

// Despite the naming, these are FLAGS not bits.
// Bits set in the VMA until the stack is in its final location
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
// Test whether any specified VMA flag is set, e.g.:
//
// if (vma_flags_test_any(flags, VMA_READ_BIT, VMA_MAYREAD_BIT)) { ... }
// Test whether ALL specified VMA flags are set, e.g.:
//
// if (vma_flags_test_all(flags, VMA_READ_BIT, VMA_MAYREAD_BIT)) { ... }
fn folio_large_order(folio: *const folio) -> u32 {
((unsafe { (*folio).__slate_anon_1.__slate_anon_0._flags_1 }) & 255) as u32
}

fn folio_large_nr_pages(folio: *const folio) -> u64 {
(1i64 << folio_large_order(folio)) as u64
}

fn compound_order(page: *const page) -> u32 {
let folio: *const folio = page as *const folio;
if test_bit__3!(6i32, unsafe { core::ptr::addr_of!((*folio).__slate_anon_0.__slate_anon_0.flags.f) }) == 0 {
return 0;
}
folio_large_order(folio)
}

fn folio_order(folio: *const folio) -> u32 {
if !folio_test_large(folio) {
return 0;
}
folio_large_order(folio)
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
struct xarray {
    xa_lock: spinlock,
    xa_flags: u32,
    xa_head: *mut (),
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


fn rt_or_dl_prio(prio: i32) -> bool {
unlikely!(prio < 100) != 0
}

fn rt_or_dl_task(p: *mut task_struct) -> bool {
rt_or_dl_prio(unsafe { (*p).prio })
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
if !(__native_word__6!() || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_419() };
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


// Contains huge pages
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
    __slate_bits_0: __slate_bits::__SlateBits710U0,
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
    __slate_bits_0: __slate_bits::__SlateBits661U0,
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


fn mod_mthp_stat(order: i32, item: u32, delta: i32) {
}

fn put_page_testzero(page: *mut page) -> i32 {
VM_BUG_ON_PAGE!();
page_ref_dec_and_test(page)
}

fn folio_large_mapcount(folio: *const folio) -> i32 {
VM_WARN_ON_FOLIO!();
atomic_read(unsafe { core::ptr::addr_of!((*folio).__slate_anon_1.__slate_anon_0.__slate_anon_2.__slate_anon_0._large_mapcount) }).wrapping_add(1)
}

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
fn page_zone_id(page: *mut page) -> i32 {
((unsafe { (*page).flags.f }) >> if (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6) < (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_sub(2) { (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6) } else { (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_sub(2) }.wrapping_mul(((8i32 != 0) as i32) as u64) & 255u64) as i32
}

fn memdesc_nid(mdf: *const __SlateRecord_mm_types_h_41) -> i32 {
ASSERT_EXCLUSIVE_BITS!((*mdf).f, 63u64 << (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_mul((true as i32) as u64));
((unsafe { (*mdf).f }) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_mul((true as i32) as u64) & 63u64) as i32
}

fn page_to_nid(page: *const page) -> i32 {
memdesc_nid(unsafe { core::ptr::addr_of!((*PF_POISONED_CHECK!(page)).flags) })
}

fn folio_nid(folio: *const folio) -> i32 {
memdesc_nid(unsafe { core::ptr::addr_of!((*folio).__slate_anon_0.__slate_anon_0.flags) })
}

fn page_cpupid_reset_last(page: *mut page) {
}

fn page_kasan_tag(page: *const page) -> u8 {
255
}

fn page_kasan_tag_reset(page: *mut page) {
}

fn page_zone(page: *const page) -> *mut zone {
unsafe { (*NODE_DATA!(page_to_nid(page))).node_zones.as_mut_ptr() as *mut zone }.wrapping_offset((page_zonenum(page) as u32) as isize)
}

fn folio_pgdat(folio: *const folio) -> *mut pglist_data {
NODE_DATA!(folio_nid(folio))
}

fn folio_zone(folio: *const folio) -> *mut zone {
unsafe { (*folio_pgdat(folio)).node_zones.as_mut_ptr() as *mut zone }.wrapping_offset((folio_zonenum(folio) as u32) as isize)
}

fn memdesc_section(mdf: *const __SlateRecord_mm_types_h_41) -> u64 {
0
}

fn folio_pfn(folio: *const folio) -> u64 {
((unsafe { core::ptr::addr_of!((*folio).__slate_anon_0.page).offset_from((vmemmap_base as *mut page) as *const page) }) as i64) as u64
}

fn folio_nr_pages(folio: *const folio) -> u64 {
if !folio_test_large(folio) {
return 1;
}
folio_large_nr_pages(folio)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct vm_event_state {
    event: [u64; 75],
}


fn count_vm_event(mut item: u32) {
loop {
__verify_pcpu_ptr__6!();
match (const { core::mem::size_of::<u64>() }) as u64 {
1 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: u64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incb %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decb %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u8 = 1u8;
if false {
let pto_tmp__: u64 = 1;
}
unsafe {
core::arch::asm!("addb {1:l}, %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
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
core::arch::asm!("incw %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decw %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u16 = 1u16;
if false {
let pto_tmp__: u64 = 1;
}
unsafe {
core::arch::asm!("addw {1:x}, %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, in(reg) pto_val__ as u16, options(nostack, att_syntax));
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
core::arch::asm!("incl %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decl %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u32 = 1u32;
if false {
let pto_tmp__: u64 = 1;
}
unsafe {
core::arch::asm!("addl {1:e}, %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, in(reg) pto_val__ as u32, options(nostack, att_syntax));
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
core::arch::asm!("incq %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decq %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u64 = 1;
if false {
let pto_tmp__: u64 = 1;
}
unsafe {
core::arch::asm!("addq {1:r}, %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, in(reg) pto_val__ as u64, options(nostack, att_syntax));
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

fn __count_vm_events(mut item: u32, delta: i64) {
loop {
__verify_pcpu_ptr__6!();
match (const { core::mem::size_of::<u64>() }) as u64 {
1 => {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_e1c548fa55baa574(delta) }) && (delta == 1 || delta == (1i32.wrapping_neg() as i64)) { delta as i32 } else { 0 };
if false {
let pao_tmp__: u64 = delta as u64;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incb %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decb %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u8 = ((delta as u64) & 255) as u8;
if false {
let pto_tmp__: u64 = delta as u64;
}
unsafe {
core::arch::asm!("addb {1:l}, %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
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
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_e1c548fa55baa574(delta) }) && (delta == 1 || delta == (1i32.wrapping_neg() as i64)) { delta as i32 } else { 0 };
if false {
let pao_tmp__: u64 = delta as u64;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incw %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decw %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u16 = ((delta as u64) & 65535) as u16;
if false {
let pto_tmp__: u64 = delta as u64;
}
unsafe {
core::arch::asm!("addw {1:x}, %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, in(reg) pto_val__ as u16, options(nostack, att_syntax));
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
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_e1c548fa55baa574(delta) }) && (delta == 1 || delta == (1i32.wrapping_neg() as i64)) { delta as i32 } else { 0 };
if false {
let pao_tmp__: u64 = delta as u64;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incl %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decl %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u32 = ((delta as u64) & 4294967295) as u32;
if false {
let pto_tmp__: u64 = delta as u64;
}
unsafe {
core::arch::asm!("addl {1:e}, %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, in(reg) pto_val__ as u32, options(nostack, att_syntax));
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
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_e1c548fa55baa574(delta) }) && (delta == 1 || delta == (1i32.wrapping_neg() as i64)) { delta as i32 } else { 0 };
if false {
let pao_tmp__: u64 = delta as u64;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incq %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decq %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u64 = delta as u64;
if false {
let pto_tmp__: u64 = delta as u64;
}
unsafe {
core::arch::asm!("addq {1:r}, %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, in(reg) pto_val__ as u64, options(nostack, att_syntax));
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

fn count_vm_events(mut item: u32, delta: i64) {
loop {
__verify_pcpu_ptr__6!();
match (const { core::mem::size_of::<u64>() }) as u64 {
1 => {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_e1c548fa55baa574(delta) }) && (delta == 1 || delta == (1i32.wrapping_neg() as i64)) { delta as i32 } else { 0 };
if false {
let pao_tmp__: u64 = delta as u64;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incb %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decb %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u8 = ((delta as u64) & 255) as u8;
if false {
let pto_tmp__: u64 = delta as u64;
}
unsafe {
core::arch::asm!("addb {1:l}, %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
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
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_e1c548fa55baa574(delta) }) && (delta == 1 || delta == (1i32.wrapping_neg() as i64)) { delta as i32 } else { 0 };
if false {
let pao_tmp__: u64 = delta as u64;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incw %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decw %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u16 = ((delta as u64) & 65535) as u16;
if false {
let pto_tmp__: u64 = delta as u64;
}
unsafe {
core::arch::asm!("addw {1:x}, %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, in(reg) pto_val__ as u16, options(nostack, att_syntax));
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
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_e1c548fa55baa574(delta) }) && (delta == 1 || delta == (1i32.wrapping_neg() as i64)) { delta as i32 } else { 0 };
if false {
let pao_tmp__: u64 = delta as u64;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incl %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decl %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u32 = ((delta as u64) & 4294967295) as u32;
if false {
let pto_tmp__: u64 = delta as u64;
}
unsafe {
core::arch::asm!("addl {1:e}, %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, in(reg) pto_val__ as u32, options(nostack, att_syntax));
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
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_e1c548fa55baa574(delta) }) && (delta == 1 || delta == (1i32.wrapping_neg() as i64)) { delta as i32 } else { 0 };
if false {
let pao_tmp__: u64 = delta as u64;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incq %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decq %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u64 = delta as u64;
if false {
let pto_tmp__: u64 = delta as u64;
}
unsafe {
core::arch::asm!("addq {1:r}, %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(vm_event_states.event) as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, in(reg) pto_val__ as u64, options(nostack, att_syntax));
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

fn zone_page_state(zone: *mut zone, item: u32) -> u64 {
let mut x: i64 = atomic_long_read(unsafe { (*zone).vm_stat.as_mut_ptr() as *mut __SlateRecord_types_h_197 }.wrapping_offset((item as u32) as isize) as *const __SlateRecord_types_h_197);
if x < 0 {
x = 0;
}
x as u64
}

fn zone_page_state_snapshot(zone: *mut zone, item: u32) -> u64 {
let mut x: i64 = atomic_long_read(unsafe { (*zone).vm_stat.as_mut_ptr() as *mut __SlateRecord_types_h_197 }.wrapping_offset((item as u32) as isize) as *const __SlateRecord_types_h_197);
let mut cpu: i32 = 0;
loop {
cpu = find_next_bit(cpumask_bits!((unsafe { core::ptr::addr_of_mut!(__cpu_online_mask) }) as *const cpumask), 64u64, cpu as u64) as i32;
if (cpu as u32) >= 64 {
break;
}
let __v0: i8 = unsafe { *((*per_cpu_ptr!(unsafe { (*zone).per_cpu_zonestats }, cpu)).vm_stat_diff.as_mut_ptr() as *mut i8).wrapping_offset((item as u32) as isize) };
x = x.wrapping_add((__v0 as i32) as i64);
cpu = cpu.wrapping_add(1);
}
if x < 0 {
x = 0;
}
x as u64
}

fn __count_numa_events(zone: *mut zone, mut item: u32, delta: i64) {
let mut pzstats: *mut per_cpu_zonestat = unsafe { (*zone).per_cpu_zonestats };
loop {
__verify_pcpu_ptr__6!();
match (const { core::mem::size_of::<u64>() }) as u64 {
1 => {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_e1c548fa55baa574(delta) }) && (delta == 1 || delta == (1i32.wrapping_neg() as i64)) { delta as i32 } else { 0 };
if false {
let pao_tmp__: u64 = delta as u64;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incb %gs:({0:r})", in(reg) (((*pzstats).vm_numa_event.as_mut_ptr() as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decb %gs:({0:r})", in(reg) (((*pzstats).vm_numa_event.as_mut_ptr() as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u8 = ((delta as u64) & 255) as u8;
if false {
let pto_tmp__: u64 = delta as u64;
}
unsafe {
core::arch::asm!("addb {1:l}, %gs:({0:r})", in(reg) (((*pzstats).vm_numa_event.as_mut_ptr() as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
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
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_e1c548fa55baa574(delta) }) && (delta == 1 || delta == (1i32.wrapping_neg() as i64)) { delta as i32 } else { 0 };
if false {
let pao_tmp__: u64 = delta as u64;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incw %gs:({0:r})", in(reg) (((*pzstats).vm_numa_event.as_mut_ptr() as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decw %gs:({0:r})", in(reg) (((*pzstats).vm_numa_event.as_mut_ptr() as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u16 = ((delta as u64) & 65535) as u16;
if false {
let pto_tmp__: u64 = delta as u64;
}
unsafe {
core::arch::asm!("addw {1:x}, %gs:({0:r})", in(reg) (((*pzstats).vm_numa_event.as_mut_ptr() as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, in(reg) pto_val__ as u16, options(nostack, att_syntax));
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
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_e1c548fa55baa574(delta) }) && (delta == 1 || delta == (1i32.wrapping_neg() as i64)) { delta as i32 } else { 0 };
if false {
let pao_tmp__: u64 = delta as u64;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incl %gs:({0:r})", in(reg) (((*pzstats).vm_numa_event.as_mut_ptr() as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decl %gs:({0:r})", in(reg) (((*pzstats).vm_numa_event.as_mut_ptr() as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u32 = ((delta as u64) & 4294967295) as u32;
if false {
let pto_tmp__: u64 = delta as u64;
}
unsafe {
core::arch::asm!("addl {1:e}, %gs:({0:r})", in(reg) (((*pzstats).vm_numa_event.as_mut_ptr() as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, in(reg) pto_val__ as u32, options(nostack, att_syntax));
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
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_e1c548fa55baa574(delta) }) && (delta == 1 || delta == (1i32.wrapping_neg() as i64)) { delta as i32 } else { 0 };
if false {
let pao_tmp__: u64 = delta as u64;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incq %gs:({0:r})", in(reg) (((*pzstats).vm_numa_event.as_mut_ptr() as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decq %gs:({0:r})", in(reg) (((*pzstats).vm_numa_event.as_mut_ptr() as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u64 = delta as u64;
if false {
let pto_tmp__: u64 = delta as u64;
}
unsafe {
core::arch::asm!("addq {1:r}, %gs:({0:r})", in(reg) (((*pzstats).vm_numa_event.as_mut_ptr() as *mut u64).wrapping_offset((item as u32) as isize) as u64) as *mut u64, in(reg) pto_val__ as u64, options(nostack, att_syntax));
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

fn zone_stat_mod_folio(folio: *mut folio, item: u32, nr: i64) {
unsafe { mod_zone_page_state(folio_zone(folio as *const folio), item, nr) };
}

#[inline(always)]
fn lowmem_page_address(page: *const page) -> *mut () {
page_to_virt!(page)
}

fn set_page_pfmemalloc(page: *mut page) {
unsafe {
(*page).__slate_anon_1.__slate_anon_0.__slate_anon_0.lru.next = (BIT!(1i32) as *mut ()) as *mut list_head;
}
}

fn clear_page_pfmemalloc(page: *mut page) {
unsafe {
(*page).__slate_anon_1.__slate_anon_0.__slate_anon_0.lru.next = core::ptr::null_mut::<list_head>();
}
}

fn kernel_poison_pages(page: *mut page, numpages: i32) {
}

fn kernel_unpoison_pages(page: *mut page, numpages: i32) {
}

fn want_init_on_alloc(flags: u32) -> bool {
if static_branch_maybe!(unsafe { core::ptr::addr_of_mut!(init_on_alloc) }) != 0 {
return true;
}
flags & (BIT!(8i32) as u32) != 0
}

fn want_init_on_free() -> bool {
static_branch_maybe__3!(unsafe { core::ptr::addr_of_mut!(init_on_free) }) != 0
}

fn debug_pagealloc_map_pages(page: *mut page, numpages: i32) {
}

fn debug_pagealloc_unmap_pages(page: *mut page, numpages: i32) {
}

fn page_is_guard(page: *const page) -> bool {
false
}

fn set_page_guard(zone: *mut zone, page: *mut page, order: u32) -> bool {
false
}

fn clear_page_guard(zone: *mut zone, page: *mut page, order: u32) {
}

fn accept_memory(start: u64, size: u64) {
}

// Mask used for checking in page_pool_page_is_pp() below. page->pp_magic is
// OR'ed with PP_SIGNATURE after the allocation in order to preserve bit 0 for
// the head page of compound page and bit 1 for pfmemalloc page, as well as the
// bits used for the DMA index. page_is_pfmemalloc() is checked in
// __page_pool_put_page() to avoid recycling the pfmemalloc page.
fn page_pool_page_is_pp(page: *const page) -> bool {
(unsafe { (*page).__slate_anon_1.__slate_anon_1.pp_magic }) & !(GENMASK__10!(if 32 < if true { (16045481047390945280u64.trailing_zeros() as i32) as u64 } else { variable__ffs(16045481047390945280u64) }.wrapping_sub(1u64.wrapping_add(__fls(64u64))) { 32u64 } else { if true { (16045481047390945280u64.trailing_zeros() as i32) as u64 } else { variable__ffs(16045481047390945280u64) }.wrapping_sub(1u64.wrapping_add(__fls(64u64))) }.wrapping_add(1u64.wrapping_add(__fls(64u64))).wrapping_sub(1), 1u64.wrapping_add(__fls(64u64))) | 3) == 64u64.wrapping_add(_AC__3!())
}

fn kmsan_alloc_page(page: *mut page, order: u32, flags: u32) {
}

fn kmsan_free_page(page: *mut page, order: u32) {
}

fn kmap_local_page(page: *const page) -> *mut () {
page_address__2!(page)
}

fn __kunmap_local(addr: *const ()) {
}

// kunmap_local - Unmap a page mapped via kmap_local_page().
// @__addr: An address within the page mapped
//
// @__addr can be any address within the mapped page.  Commonly it is the
// address return from kmap_local_page(), but it can also include offsets.
//
// Unmapping should be done in the reverse order of the mapping.  See
// kmap_local_page() for details.
fn clear_highpage_kasan_tagged(page: *mut page) {
let kaddr: *mut () = kmap_local_page(page as *const page);
clear_page(kasan_reset_tag(kaddr as *const ()));
loop {
loop {
if __same_type!() != 0 {
unsafe { __compiletime_assert_488() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__kunmap_local(kaddr as *const ());
if true {
break;
}
}
}

fn tag_clear_highpages(page: *mut page, numpages: i32, clear_pages: bool) -> bool {
clear_pages
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


fn str_on_off(v: bool) -> *const u8 {
(if v { c"on".as_ptr() as *mut u8 } else { c"off".as_ptr() as *mut u8 }) as *const u8
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
    __slate_bits_0: __slate_bits::__SlateBits788U0,
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
struct bio_vec {
    bv_page: *mut page,
    bv_len: u32,
    bv_offset: u32,
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
    __slate_bits_0: __slate_bits::__SlateBits817U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits817U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits817U2,
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
    __slate_bits_3: __slate_bits::__SlateBits817U3,
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
    __slate_bits_0: __slate_bits::__SlateBits814U0,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct folio_batch {
    nr: u8,
    i: u8,
    percpu_pvec_drained: bool,
    folios: [*mut folio; 31],
}


fn folio_batch_reinit(fbatch: *mut folio_batch) {
unsafe {
(*fbatch).nr = 0;
}
unsafe {
(*fbatch).i = 0;
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct writeback_control {
    nr_to_write: i64,
    pages_skipped: i64,
    range_start: i64,
    range_end: i64,
    sync_mode: u32,
    __slate_bits_0: __slate_bits::__SlateBits624U0,
    fbatch: folio_batch,
    index: u64,
    saved_err: i32,
}


fn PageMemcgKmem(page: *mut page) -> bool {
false
}

fn split_page_memcg(first: *mut page, order: u32) {
}

fn __memcg_kmem_charge_page(page: *mut page, gfp: u32, order: i32) -> i32 {
0
}

fn __memcg_kmem_uncharge_page(page: *mut page, order: i32) {
}

fn memcg_kmem_online() -> bool {
false
}

fn mem_cgroup_show_protected_memory(memcg: *mut mem_cgroup) {
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


fn pm_suspended_storage() -> bool {
!gfp_has_io_fs(unsafe { gfp_allowed_mask })
}

#[repr(C)]
#[derive(Clone, Copy)]
struct oom_control {
    zonelist: *mut zonelist,
    nodemask: *const __SlateRecord_nodemask_types_h_17,
    memcg: *mut mem_cgroup,
    gfp_mask: u32,
    order: i32,
    totalpages: u64,
    chosen: *mut task_struct,
    chosen_points: i64,
    constraint: u32,
}


fn tsk_is_oom_victim(tsk: *mut task_struct) -> bool {
!unsafe { (*(*tsk).signal).oom_mm }.is_null()
}

fn cpuhp_setup_state_nocalls(state: i32, name: *const u8, startup: Option<unsafe extern "C" fn(u32) -> i32>, teardown: Option<unsafe extern "C" fn(u32) -> i32>) -> i32 {
unsafe { __cpuhp_setup_state(state, name, false, startup, teardown, false) }
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
#[repr(C)]
#[derive(Clone, Copy)]
struct ldt_struct {
    entries: *mut desc_struct,
    nr_entries: u32,
    slot: i32,
}


// Init a new mm.  Used on mm copies, like at fork()
// and on mm's that are brand-new, like at execve().
fn cpusets_enabled() -> bool {
static_branch_unlikely__4!(unsafe { core::ptr::addr_of_mut!(cpusets_enabled_key) }) != 0
}

fn cpusets_insane_config() -> bool {
static_branch_unlikely__5!(unsafe { core::ptr::addr_of_mut!(cpusets_insane_config_key) }) != 0
}

fn __cpuset_zone_allowed(z: *mut zone, gfp_mask: u32) -> bool {
unsafe { cpuset_current_node_allowed(zone_to_nid(z as *const zone), gfp_mask) }
}

fn cpuset_memory_pressure_bump() {
}

fn read_mems_allowed_begin() -> u32 {
if static_branch_unlikely__6!(unsafe { core::ptr::addr_of_mut!(cpusets_pre_enable_key) }) == 0 {
return 0;
}
read_seqcount_begin!(unsafe { core::ptr::addr_of_mut!((*get_current()).mems_allowed_seq) })
}

fn read_mems_allowed_retry(seq: u32) -> bool {
if static_branch_unlikely__7!(unsafe { core::ptr::addr_of_mut!(cpusets_enabled_key) }) == 0 {
return false;
}
read_seqcount_retry!(unsafe { core::ptr::addr_of_mut!((*get_current()).mems_allowed_seq) }, seq) != 0
}

fn should_fail_alloc_page(gfp_mask: u32, order: u32) -> bool {
false
}

fn __do_trace_mm_page_free(page: *mut page, order: u32) {
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
let mut __UNIQUE_ID_guard_676: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_676) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_677: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_677) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_679() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_mm_page_free)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_678: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_678;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut page, u32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut page, u32) -> i32>>(__SCT__tp_func_mm_page_free as *const ()) };
unsafe { __v12.unwrap()(__data, page, order) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_mm_page_free: [u8; 13] = [109, 109, 95, 112, 97, 103, 101, 95, 102, 114, 101, 101, 0];

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_mm_page_free_680: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_mm_page_free) }) as *mut ();

fn trace_mm_page_free(page: *mut page, order: u32) {
if static_branch_unlikely__8!(unsafe { core::ptr::addr_of_mut!(__tracepoint_mm_page_free.key) }) != 0 {
__do_trace_mm_page_free(page, order);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_681: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_mm_page_free_batched(page: *mut page) {
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
let mut __UNIQUE_ID_guard_683: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_683) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_684: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_684) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_686() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_mm_page_free_batched)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_685: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_685;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut page) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut page) -> i32>>(__SCT__tp_func_mm_page_free_batched as *const ()) };
unsafe { __v12.unwrap()(__data, page) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_mm_page_free_batched: __SlateAlign16<[u8; 21]> = __SlateAlign16([109u8, 109u8, 95u8, 112u8, 97u8, 103u8, 101u8, 95u8, 102u8, 114u8, 101u8, 101u8, 95u8, 98u8, 97u8, 116u8, 99u8, 104u8, 101u8, 100u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_mm_page_free_batched_687: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_mm_page_free_batched) }) as *mut ();

fn trace_mm_page_free_batched(page: *mut page) {
if static_branch_unlikely__9!(unsafe { core::ptr::addr_of_mut!(__tracepoint_mm_page_free_batched.key) }) != 0 {
__do_trace_mm_page_free_batched(page);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_688: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_mm_page_alloc(page: *mut page, order: u32, gfp_flags: u32, migratetype: i32) {
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
let mut __UNIQUE_ID_guard_690: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_690) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_691: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_691) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_693() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_mm_page_alloc)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_692: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_692;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut page, u32, u32, i32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut page, u32, u32, i32) -> i32>>(__SCT__tp_func_mm_page_alloc as *const ()) };
unsafe { __v12.unwrap()(__data, page, order, gfp_flags, migratetype) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_mm_page_alloc: [u8; 14] = [109, 109, 95, 112, 97, 103, 101, 95, 97, 108, 108, 111, 99, 0];

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_mm_page_alloc_694: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_mm_page_alloc) }) as *mut ();

fn trace_mm_page_alloc(page: *mut page, order: u32, gfp_flags: u32, migratetype: i32) {
if static_branch_unlikely__10!(unsafe { core::ptr::addr_of_mut!(__tracepoint_mm_page_alloc.key) }) != 0 {
__do_trace_mm_page_alloc(page, order, gfp_flags, migratetype);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_695: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_mm_page_alloc_zone_locked(page: *mut page, order: u32, migratetype: i32, percpu_refill: i32) {
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
let mut __UNIQUE_ID_guard_697: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_697) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_698: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_698) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_700() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_mm_page_alloc_zone_locked)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_699: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_699;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut page, u32, i32, i32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut page, u32, i32, i32) -> i32>>(__SCT__tp_func_mm_page_alloc_zone_locked as *const ()) };
unsafe { __v12.unwrap()(__data, page, order, migratetype, percpu_refill) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_mm_page_alloc_zone_locked: __SlateAlign16<[u8; 26]> = __SlateAlign16([109u8, 109u8, 95u8, 112u8, 97u8, 103u8, 101u8, 95u8, 97u8, 108u8, 108u8, 111u8, 99u8, 95u8, 122u8, 111u8, 110u8, 101u8, 95u8, 108u8, 111u8, 99u8, 107u8, 101u8, 100u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_mm_page_alloc_zone_locked_701: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_mm_page_alloc_zone_locked) }) as *mut ();

fn trace_mm_page_alloc_zone_locked(page: *mut page, order: u32, migratetype: i32, percpu_refill: i32) {
if static_branch_unlikely__11!(unsafe { core::ptr::addr_of_mut!(__tracepoint_mm_page_alloc_zone_locked.key) }) != 0 {
__do_trace_mm_page_alloc_zone_locked(page, order, migratetype, percpu_refill);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_702: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_mm_page_pcpu_drain(page: *mut page, order: u32, migratetype: i32) {
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
let mut __UNIQUE_ID_guard_704: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_704) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_705: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_705) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_707() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_mm_page_pcpu_drain)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_706: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_706;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut page, u32, i32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut page, u32, i32) -> i32>>(__SCT__tp_func_mm_page_pcpu_drain as *const ()) };
unsafe { __v12.unwrap()(__data, page, order, migratetype) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_mm_page_pcpu_drain: __SlateAlign16<[u8; 19]> = __SlateAlign16([109u8, 109u8, 95u8, 112u8, 97u8, 103u8, 101u8, 95u8, 112u8, 99u8, 112u8, 117u8, 95u8, 100u8, 114u8, 97u8, 105u8, 110u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_mm_page_pcpu_drain_708: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_mm_page_pcpu_drain) }) as *mut ();

fn trace_mm_page_pcpu_drain(page: *mut page, order: u32, migratetype: i32) {
if static_branch_unlikely__14!(unsafe { core::ptr::addr_of_mut!(__tracepoint_mm_page_pcpu_drain.key) }) != 0 {
__do_trace_mm_page_pcpu_drain(page, order, migratetype);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_709: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_mm_page_alloc_extfrag(page: *mut page, alloc_order: i32, fallback_order: i32, alloc_migratetype: i32, fallback_migratetype: i32) {
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
let mut __UNIQUE_ID_guard_711: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_711) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_712: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_712) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_714() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_mm_page_alloc_extfrag)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_713: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_713;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut page, i32, i32, i32, i32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut page, i32, i32, i32, i32) -> i32>>(__SCT__tp_func_mm_page_alloc_extfrag as *const ()) };
unsafe { __v12.unwrap()(__data, page, alloc_order, fallback_order, alloc_migratetype, fallback_migratetype) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_mm_page_alloc_extfrag: __SlateAlign16<[u8; 22]> = __SlateAlign16([109u8, 109u8, 95u8, 112u8, 97u8, 103u8, 101u8, 95u8, 97u8, 108u8, 108u8, 111u8, 99u8, 95u8, 101u8, 120u8, 116u8, 102u8, 114u8, 97u8, 103u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_mm_page_alloc_extfrag_715: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_mm_page_alloc_extfrag) }) as *mut ();

fn trace_mm_page_alloc_extfrag(page: *mut page, alloc_order: i32, fallback_order: i32, alloc_migratetype: i32, fallback_migratetype: i32) {
if static_branch_unlikely__15!(unsafe { core::ptr::addr_of_mut!(__tracepoint_mm_page_alloc_extfrag.key) }) != 0 {
__do_trace_mm_page_alloc_extfrag(page, alloc_order, fallback_order, alloc_migratetype, fallback_migratetype);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_716: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_mm_setup_per_zone_wmarks(zone: *mut zone) {
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
let mut __UNIQUE_ID_guard_718: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_718) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_719: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_719) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_721() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_mm_setup_per_zone_wmarks)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_720: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_720;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut zone) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut zone) -> i32>>(__SCT__tp_func_mm_setup_per_zone_wmarks as *const ()) };
unsafe { __v12.unwrap()(__data, zone) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_mm_setup_per_zone_wmarks: __SlateAlign16<[u8; 25]> = __SlateAlign16([109u8, 109u8, 95u8, 115u8, 101u8, 116u8, 117u8, 112u8, 95u8, 112u8, 101u8, 114u8, 95u8, 122u8, 111u8, 110u8, 101u8, 95u8, 119u8, 109u8, 97u8, 114u8, 107u8, 115u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_mm_setup_per_zone_wmarks_722: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_mm_setup_per_zone_wmarks) }) as *mut ();

fn trace_mm_setup_per_zone_wmarks(zone: *mut zone) {
if static_branch_unlikely__12!(unsafe { core::ptr::addr_of_mut!(__tracepoint_mm_setup_per_zone_wmarks.key) }) != 0 {
__do_trace_mm_setup_per_zone_wmarks(zone);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_723: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_mm_setup_per_zone_lowmem_reserve(zone: *mut zone, upper_zone: *mut zone, lowmem_reserve: i64) {
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
let mut __UNIQUE_ID_guard_725: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_725) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_726: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_726) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_728() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_mm_setup_per_zone_lowmem_reserve)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_727: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_727;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut zone, *mut zone, i64) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut zone, *mut zone, i64) -> i32>>(__SCT__tp_func_mm_setup_per_zone_lowmem_reserve as *const ()) };
unsafe { __v12.unwrap()(__data, zone, upper_zone, lowmem_reserve) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_mm_setup_per_zone_lowmem_reserve: __SlateAlign16<[u8; 33]> = __SlateAlign16([109u8, 109u8, 95u8, 115u8, 101u8, 116u8, 117u8, 112u8, 95u8, 112u8, 101u8, 114u8, 95u8, 122u8, 111u8, 110u8, 101u8, 95u8, 108u8, 111u8, 119u8, 109u8, 101u8, 109u8, 95u8, 114u8, 101u8, 115u8, 101u8, 114u8, 118u8, 101u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_mm_setup_per_zone_lowmem_reserve_729: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_mm_setup_per_zone_lowmem_reserve) }) as *mut ();

fn trace_mm_setup_per_zone_lowmem_reserve(zone: *mut zone, upper_zone: *mut zone, lowmem_reserve: i64) {
if static_branch_unlikely__13!(unsafe { core::ptr::addr_of_mut!(__tracepoint_mm_setup_per_zone_lowmem_reserve.key) }) != 0 {
__do_trace_mm_setup_per_zone_lowmem_reserve(zone, upper_zone, lowmem_reserve);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_730: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_mm_calculate_totalreserve_pages(totalreserve_pages__0: u64) {
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
let mut __UNIQUE_ID_guard_732: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_732) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_733: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_733) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_735() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_mm_calculate_totalreserve_pages)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_734: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_734;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), u64) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), u64) -> i32>>(__SCT__tp_func_mm_calculate_totalreserve_pages as *const ()) };
unsafe { __v12.unwrap()(__data, totalreserve_pages__0) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_mm_calculate_totalreserve_pages: __SlateAlign16<[u8; 32]> = __SlateAlign16([109u8, 109u8, 95u8, 99u8, 97u8, 108u8, 99u8, 117u8, 108u8, 97u8, 116u8, 101u8, 95u8, 116u8, 111u8, 116u8, 97u8, 108u8, 114u8, 101u8, 115u8, 101u8, 114u8, 118u8, 101u8, 95u8, 112u8, 97u8, 103u8, 101u8, 115u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_mm_calculate_totalreserve_pages_736: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_mm_calculate_totalreserve_pages) }) as *mut ();

fn trace_mm_calculate_totalreserve_pages(totalreserve_pages__0: u64) {
if static_branch_unlikely__17!(unsafe { core::ptr::addr_of_mut!(__tracepoint_mm_calculate_totalreserve_pages.key) }) != 0 {
__do_trace_mm_calculate_totalreserve_pages(totalreserve_pages__0);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_737: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_reclaim_retry_zone(zoneref: *mut zoneref, order: i32, reclaimable: u64, available: u64, min_wmark: u64, no_progress_loops: i32, wmark_check: bool) {
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
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_756() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_reclaim_retry_zone)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_755: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_755;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut zoneref, i32, u64, u64, u64, i32, bool) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut zoneref, i32, u64, u64, u64, i32, bool) -> i32>>(__SCT__tp_func_reclaim_retry_zone as *const ()) };
unsafe { __v12.unwrap()(__data, zoneref, order, reclaimable, available, min_wmark, no_progress_loops, wmark_check) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_reclaim_retry_zone: __SlateAlign16<[u8; 19]> = __SlateAlign16([114u8, 101u8, 99u8, 108u8, 97u8, 105u8, 109u8, 95u8, 114u8, 101u8, 116u8, 114u8, 121u8, 95u8, 122u8, 111u8, 110u8, 101u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_reclaim_retry_zone_757: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_reclaim_retry_zone) }) as *mut ();

fn trace_reclaim_retry_zone(zoneref: *mut zoneref, order: i32, reclaimable: u64, available: u64, min_wmark: u64, no_progress_loops: i32, wmark_check: bool) {
if static_branch_unlikely__16!(unsafe { core::ptr::addr_of_mut!(__tracepoint_reclaim_retry_zone.key) }) != 0 {
__do_trace_reclaim_retry_zone(zoneref, order, reclaimable, available, min_wmark, no_progress_loops, wmark_check);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_758: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_compact_retry(order: i32, priority: u32, result: u32, retries: i32, max_retries: i32, ret: bool) {
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
let mut __UNIQUE_ID_guard_795: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_795) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_796: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_796) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_798() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_compact_retry)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_797: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_797;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), i32, u32, u32, i32, i32, bool) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), i32, u32, u32, i32, i32, bool) -> i32>>(__SCT__tp_func_compact_retry as *const ()) };
unsafe { __v12.unwrap()(__data, order, priority, result, retries, max_retries, ret) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_compact_retry: [u8; 14] = [99, 111, 109, 112, 97, 99, 116, 95, 114, 101, 116, 114, 121, 0];

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_compact_retry_799: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_compact_retry) }) as *mut ();

fn trace_compact_retry(order: i32, priority: u32, result: u32, retries: i32, max_retries: i32, ret: bool) {
if static_branch_unlikely__18!(unsafe { core::ptr::addr_of_mut!(__tracepoint_compact_retry.key) }) != 0 {
__do_trace_compact_retry(order, priority, result, retries, max_retries, ret);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_800: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn num_pages_contiguous(pages: *mut *mut page, nr_pages: u64) -> u64 {
let mut cur_page: *mut page = unsafe { *pages.wrapping_offset(0 as isize) };
let section: u64 = memdesc_section((unsafe { core::ptr::addr_of_mut!((*cur_page).flags) }) as *const __SlateRecord_mm_types_h_41);
let mut i: u64 = 1;
while i < nr_pages {
let __v0: *mut page = cur_page;
let __v1: *mut page = __v0.wrapping_offset(1 as isize);
cur_page = __v1;
if __v1 != unsafe { *pages.wrapping_offset(i as isize) } {
break;
}
// In unproblematic kernel configs, page_to_section() == 0 and
// the whole check will get optimized out.
if memdesc_section((unsafe { core::ptr::addr_of_mut!((*cur_page).flags) }) as *const __SlateRecord_mm_types_h_41) != section {
break;
}
i = i.wrapping_add(1);
}
i
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mempolicy {
    refcnt: __SlateRecord_types_h_190,
    mode: u16,
    flags: u16,
    nodes: __SlateRecord_nodemask_types_h_17,
    home_node: i32,
    w: __SlateRecord_mempolicy_h_54,
    rcu: callback_head,
}


fn check_highest_zone(k: u32) {
if (k as u32) > ((unsafe { policy_zone }) as u32) && (k as u32) != 3 {
unsafe {
policy_zone = k;
}
}
}

fn reset_page_owner(page: *mut page, order: u16) {
}

fn set_page_owner(page: *mut page, order: u16, gfp_mask: u32) {
}

fn split_page_owner(page: *mut page, old_order: i32, new_order: i32) {
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


fn psi_memstall_enter(flags: &u64) {
}

fn psi_memstall_leave(flags: &u64) {
}

fn khugepaged_min_free_kbytes_update() {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct task_delay_info {
    lock: raw_spinlock,
    blkio_start: u64,
    blkio_delay_max: u64,
    blkio_delay_min: u64,
    blkio_delay: u64,
    swapin_start: u64,
    swapin_delay_max: u64,
    swapin_delay_min: u64,
    swapin_delay: u64,
    blkio_count: u32,
    swapin_count: u32,
    freepages_start: u64,
    freepages_delay_max: u64,
    freepages_delay_min: u64,
    freepages_delay: u64,
    thrashing_start: u64,
    thrashing_delay_max: u64,
    thrashing_delay_min: u64,
    thrashing_delay: u64,
    compact_start: u64,
    compact_delay_max: u64,
    compact_delay_min: u64,
    compact_delay: u64,
    wpcopy_start: u64,
    wpcopy_delay_max: u64,
    wpcopy_delay_min: u64,
    wpcopy_delay: u64,
    irq_delay_max: u64,
    irq_delay_min: u64,
    irq_delay: u64,
    freepages_count: u32,
    thrashing_count: u32,
    compact_count: u32,
    wpcopy_count: u32,
    irq_count: u32,
    blkio_delay_max_ts: timespec64,
    swapin_delay_max_ts: timespec64,
    freepages_delay_max_ts: timespec64,
    thrashing_delay_max_ts: timespec64,
    compact_delay_max_ts: timespec64,
    wpcopy_delay_max_ts: timespec64,
    irq_delay_max_ts: timespec64,
}


fn delayacct_compact_start() {
if static_branch_unlikely__19!(unsafe { core::ptr::addr_of_mut!(delayacct_key) }) == 0 {
return;
}
if !unsafe { (*get_current()).delays }.is_null() {
unsafe { __delayacct_compact_start() };
}
}

fn delayacct_compact_end() {
if static_branch_unlikely__20!(unsafe { core::ptr::addr_of_mut!(delayacct_key) }) == 0 {
return;
}
if !unsafe { (*get_current()).delays }.is_null() {
unsafe { __delayacct_compact_end() };
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct anon_vma {
    root: *mut anon_vma,
    rwsem: rw_semaphore,
    refcount: __SlateRecord_types_h_190,
    num_children: u64,
    num_active_vmas: u64,
    parent: *mut anon_vma,
    rb_root: rb_root_cached,
}


// The GFP flags allowed during early boot
// Different from WARN_ON_ONCE(), no warning will be issued
// when we specify __GFP_NOWARN.
// Flags passed to __show_mem() and show_free_areas() to suppress output in
// various contexts.
// disallowed nodes
// Compare PTEs respecting the dirty bit.
// Compare PTEs respecting the soft-dirty bit.
// Compare PTEs respecting the writable bit.
// Merge PTE write bits: if any PTE in the batch is writable, modify the
// PTE at @ptentp to be writable.
// Merge PTE young and dirty bits: if any PTE in the batch is young or dirty,
// modify the PTE at @ptentp to be young or dirty, respectively.
fn set_page_refcounted(page: *mut page) {
VM_BUG_ON_PAGE!();
VM_BUG_ON_PAGE!();
set_page_count(page, 1i32);
}

// Maximum number of reclaim retries without progress before the OOM
// killer is consider the only way forward.
#[unsafe(no_mangle)]
static zone_names: __SlateAlign16<[*mut u8; 4]> = __SlateAlign16([b"DMA\0".as_ptr() as *mut u8, b"DMA32\0".as_ptr() as *mut u8, b"Normal\0".as_ptr() as *mut u8, b"Movable\0".as_ptr() as *mut u8]);

#[unsafe(no_mangle)]
static mut defrag_mode: i32 = unsafe { core::mem::zeroed() };

#[unsafe(no_mangle)]
extern "C" fn calculate_min_free_kbytes() {
let lowmem_kbytes: u64 = nr_free_buffer_pages().wrapping_mul(_AC!() << 12i32 >> 10i32);
let new_min_free_kbytes: i32 = (unsafe { int_sqrt(lowmem_kbytes.wrapping_mul(16)) }) as i32;
if new_min_free_kbytes > unsafe { user_min_free_kbytes } {
let mut __v0: i32;
let __UNIQUE_ID_v__1090: i32 = new_min_free_kbytes;
let __UNIQUE_ID_l__1091: i32 = 128;
let __UNIQUE_ID_h__1092: i32 = 262144;
loop {
if !(!statically_true!(__UNIQUE_ID_l__1091 > __UNIQUE_ID_h__1092)) {
unsafe { __compiletime_assert_1093() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if __types_ok3__2!(__UNIQUE_ID_v__1090, __UNIQUE_ID_l__1091, __UNIQUE_ID_h__1092) == 0 {
unsafe { __compiletime_assert_1094() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_v__1090 >= __UNIQUE_ID_h__1092 { __UNIQUE_ID_h__1092 } else { if __UNIQUE_ID_v__1090 <= __UNIQUE_ID_l__1091 { __UNIQUE_ID_l__1091 } else { __UNIQUE_ID_v__1090 } };
unsafe {
min_free_kbytes = __v0;
}
} else {
if __ratelimit__3!(unsafe { core::ptr::addr_of_mut!(_rs) }) != 0 {
let mut __v3: i32;
while false {
}
__v3 = unsafe { _printk(c"\x014min_free_kbytes is not updated to %d because user defined value %d is preferred\n".as_ptr() as *const u8, new_min_free_kbytes, user_min_free_kbytes) };
}
}
}

fn folio_set_order(folio: *mut folio, order: u32) {
let mut __v0: i64;
let mut __ret_warn_on: i32 = 0;
let __v1: bool = order == 0 || !folio_test_large(folio as *const folio);
__ret_warn_on = __v1 as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("842: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 842b - ., 3; .popsection\n# {0}", const 842 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_folio_set_order_2, sym __slate_asm_string_folio_set_order_3, const 737 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("843: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 843b - ., 4; .popsection\n# {0}", const 843 as i32, options(nostack, att_syntax));
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
VM_WARN_ON_ONCE!();
unsafe {
(*folio).__slate_anon_1.__slate_anon_0._flags_1 = (*folio).__slate_anon_1.__slate_anon_0._flags_1 & !255 | (order as u64);
}
}

fn page_rmappable_folio(page: *mut page) -> *mut folio {
let folio: *mut folio = page as *mut folio;
let __v0: bool = !folio.is_null() && folio_test_large(folio as *const folio);
if __v0 {
folio_set_large_rmappable(folio);
}
folio
}

fn prep_compound_head(page: *mut page, order: u32) {
let folio: *mut folio = page as *mut folio;
folio_set_order(folio, order);
atomic_set(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_1.__slate_anon_0.__slate_anon_2.__slate_anon_0._large_mapcount) }, 1i32.wrapping_neg());
if IS_ENABLED__2!() != 0 {
atomic_set(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_1.__slate_anon_0.__slate_anon_2.__slate_anon_0._nr_pages_mapped) }, 0i32);
}
if IS_ENABLED!() != 0 {
unsafe {
(*folio).__slate_anon_1.__slate_anon_0.__slate_anon_2.__slate_anon_0.__slate_anon_5._mm_ids = 0;
}
unsafe {
*((*folio).__slate_anon_1.__slate_anon_0.__slate_anon_2.__slate_anon_0._mm_id_mapcount.as_mut_ptr() as *mut i32).wrapping_offset(0 as isize) = 1i32.wrapping_neg();
}
unsafe {
*((*folio).__slate_anon_1.__slate_anon_0.__slate_anon_2.__slate_anon_0._mm_id_mapcount.as_mut_ptr() as *mut i32).wrapping_offset(1 as isize) = 1i32.wrapping_neg();
}
}
if IS_ENABLED__2!() != 0 || order > 1 {
atomic_set(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_1.__slate_anon_0.__slate_anon_2.__slate_anon_0._pincount) }, 0i32);
atomic_set(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_1.__slate_anon_0.__slate_anon_2.__slate_anon_0._entire_mapcount) }, 1i32.wrapping_neg());
}
if order > 1 {
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_2.__slate_anon_0._deferred_list) });
}
}

fn prep_compound_tail(tail: *mut page, head: *const page, order: u32) {
unsafe {
(*tail).__slate_anon_1.__slate_anon_0.mapping = ((1024i32 as *const u8).wrapping_offset(_AC__3!() as isize) as *mut ()) as *mut address_space;
}
set_compound_head(tail, head, order);
VM_WARN_ON_ONCE!();
}

#[repr(C)]
#[derive(Clone, Copy)]
struct capture_control {
    zone: *mut zone,
    migratetype: i32,
    order: i32,
    page: *mut page,
}


fn node_reclaim_enabled() -> bool {
// Is any node_reclaim_mode bit set?
(unsafe { node_reclaim_mode }) & 7i32 != 0
}

fn can_spin_trylock() -> bool {
// In PREEMPT_RT spin_trylock() will call raw_spin_lock() which is
// unsafe in NMI. If spin_trylock() is called from hard IRQ the current
// task may be waiting for one rt_spin_lock, but rt_spin_trylock() will
// mark the task as the owner of another rt_spin_lock which will
// confuse PI logic, so return immediately if called from hard IRQ or
// NMI.
//
// Note, irqs_disabled() case is ok. spin_trylock() can be called
// from raw_spin_lock_irqsave region.
let __v0: bool = IS_ENABLED!() != 0 && (in_nmi!() != 0 || in_hardirq!() != 0);
if __v0 {
return false;
}
// On UP, spin_trylock() always succeeds even when it is locked
let __v2: bool = IS_ENABLED__2!() == 0 && in_nmi!() != 0;
if __v2 {
return false;
}
true
}

fn deferred_pages_enabled() -> bool {
false
}

// don't check watermarks at all
// Mask to get the watermark bits
// Caller cannot block. Allow access
// to 25% of the min watermark or
// 62.5% if __GFP_HIGH is set.
// __GFP_HIGH set. Allow access to 50%
// of the min watermark.
// check for correct cpuset
// allow allocations from CMA areas
// avoid mixing pageblock types
// Allows access to MIGRATE_HIGHATOMIC
// Only use spin_trylock in allocation path
// allow waking of kswapd, __GFP_KSWAPD_RECLAIM set
// Avoid alloc_tag recursion for internal allocations.
//
// Callers must clear_page_tag_ref() before freeing to avoid warnings from
// alloc_tag_sub_check().
// Flags that allow allocations below the min watermark.
#[repr(C)]
#[derive(Clone, Copy)]
struct alloc_context {
    zonelist: *mut zonelist,
    nodemask: *const __SlateRecord_nodemask_types_h_17,
    preferred_zoneref: *mut zoneref,
    migratetype: i32,
    highest_zoneidx: u32,
    spread_dirty_pages: bool,
    alloc_flags: u32,
}


fn buddy_order(page: *mut page) -> u32 {
// PageBuddy() must be checked by the caller
page_private!(page) as u32
}

// Like buddy_order(), but for callers who cannot afford to hold the zone lock.
// PageBuddy() should be checked first by the caller to minimize race window,
// and invalid values must be handled gracefully.
//
// READ_ONCE is used so that if the caller assigns the result into a local
// variable and e.g. tests it for valid range before using, the compiler cannot
// decide to remove the variable and inline the page_private(page) multiple
// times, potentially observing different values in the tests and the actual
// use of the result.
fn page_is_buddy(page: *mut page, buddy: *mut page, order: u32) -> bool {
let __v0: bool = !page_is_guard(buddy as *const page) && PageBuddy(buddy as *const page) == 0;
if __v0 {
return false;
}
if buddy_order(buddy) != order {
return false;
}
// zone check is done late to avoid uselessly calculating
// zone/node ids for pages that could never merge.
if page_zone_id(page) != page_zone_id(buddy) {
return false;
}
VM_BUG_ON_PAGE!();
true
}

fn __find_buddy_pfn(page_pfn: u64, order: u32) -> u64 {
page_pfn ^ ((1i32 << order) as u64)
}

fn find_buddy_page_pfn(page: *mut page, pfn: u64, order: u32, buddy_pfn: *mut u64) -> *mut page {
let __buddy_pfn: u64 = __find_buddy_pfn(pfn, order);
let buddy: *mut page = page.wrapping_offset(__buddy_pfn.wrapping_sub(pfn) as isize);
if !buddy_pfn.is_null() {
unsafe {
*buddy_pfn = __buddy_pfn;
}
}
if page_is_buddy(page, buddy, order) {
return buddy;
}
core::ptr::null_mut::<page>()
}

fn free_area_empty(area: *mut free_area, migratetype: i32) -> bool {
list_empty(unsafe { (*area).free_list.as_mut_ptr() as *mut list_head }.wrapping_offset(migratetype as isize) as *const list_head) != 0
}

// Convert GFP flags to their corresponding migrate type
fn gfp_migratetype(gfp_flags: u32) -> i32 {
VM_WARN_ON!();
loop {
if 8u64 != BIT!(3i32) {
unsafe { __compiletime_assert_845() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if BIT!(3i32) >> 3i32 != 1 {
unsafe { __compiletime_assert_846() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if BIT!(4i32) >> 3i32 != 2 {
unsafe { __compiletime_assert_847() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (BIT!(3i32) | BIT!(4i32)) >> 3i32 != 3 {
unsafe { __compiletime_assert_848() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
if unlikely__2!(unsafe { page_group_by_mobility_disabled }) != 0 {
return 0;
}
// Group based on mobility
(((gfp_flags & ((BIT!(4i32) as u32) | (BIT!(3i32) as u32))) as u64) >> 3i32) as i32
}

#[unsafe(no_mangle)]
extern "C" fn decay_pcp_high(zone: *mut zone, pcp: *mut per_cpu_pages) -> bool {
let mut high_min: i32 = 0;
let mut to_drain: i32 = 0;
let mut to_drain_batched: i32 = 0;
let mut batch: i32 = 0;
let mut todo: bool = false;
let mut __v0: i32;
while false {
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*pcp).high_min) as *const i32) };
high_min = __v0;
let mut __v2: i32;
while false {
}
__v2 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*pcp).batch) as *const i32) };
batch = __v2;
// Decrease pcp->high periodically to try to free possible
// idle PCP pages.  And, avoid to free too many pages to
// control latency.  This caps pcp->high decrement too.
if (unsafe { (*pcp).high }) > high_min {
let mut __v4: i32;
let __UNIQUE_ID_x__921: i32 = unsafe { (*pcp).count }.wrapping_sub(batch << 5i32);
let __UNIQUE_ID_y__922: i32 = unsafe { (*pcp).high }.wrapping_sub((unsafe { (*pcp).high }) >> 3i32);
let __UNIQUE_ID_z__923: i32 = high_min;
loop {
if __types_ok3__3!(__UNIQUE_ID_x__921, __UNIQUE_ID_y__922, __UNIQUE_ID_z__923) == 0 {
unsafe { __compiletime_assert_924() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v4 = if __UNIQUE_ID_x__921 > if __UNIQUE_ID_y__922 > __UNIQUE_ID_z__923 { __UNIQUE_ID_y__922 } else { __UNIQUE_ID_z__923 } { __UNIQUE_ID_x__921 } else { if __UNIQUE_ID_y__922 > __UNIQUE_ID_z__923 { __UNIQUE_ID_y__922 } else { __UNIQUE_ID_z__923 } };
unsafe {
(*pcp).high = __v4;
}
if (unsafe { (*pcp).high }) > high_min {
todo = true;
}
}
to_drain = unsafe { (*pcp).count }.wrapping_sub(unsafe { (*pcp).high });
while to_drain > 0 {
let mut __v6: i32;
let __UNIQUE_ID_x__925: i32 = to_drain;
let __UNIQUE_ID_y__926: i32 = batch;
loop {
if __types_ok__4!(__UNIQUE_ID_x__925, __UNIQUE_ID_y__926) == 0 {
unsafe { __compiletime_assert_927() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v6 = if __UNIQUE_ID_x__925 < __UNIQUE_ID_y__926 { __UNIQUE_ID_x__925 } else { __UNIQUE_ID_y__926 };
to_drain_batched = __v6;
pcp_spin_lock_nopin!(pcp);
free_pcppages_bulk(zone, to_drain_batched, pcp, 0i32);
pcp_spin_unlock_nopin!(pcp);
todo = true;
to_drain = to_drain.wrapping_sub(to_drain_batched);
}
todo
}

#[unsafe(no_mangle)]
extern "C" fn drain_zone_pages(zone: *mut zone, pcp: *mut per_cpu_pages) {
let mut to_drain: i32 = 0;
let mut batch: i32 = 0;
let mut __v0: i32;
while false {
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*pcp).batch) as *const i32) };
batch = __v0;
let mut __v2: i32;
let __UNIQUE_ID_x__929: i32 = unsafe { (*pcp).count };
let __UNIQUE_ID_y__930: i32 = batch;
loop {
if __types_ok__4!(__UNIQUE_ID_x__929, __UNIQUE_ID_y__930) == 0 {
unsafe { __compiletime_assert_931() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = if __UNIQUE_ID_x__929 < __UNIQUE_ID_y__930 { __UNIQUE_ID_x__929 } else { __UNIQUE_ID_y__930 };
to_drain = __v2;
if to_drain > 0 {
pcp_spin_lock_nopin!(pcp);
free_pcppages_bulk(zone, to_drain, pcp, 0i32);
pcp_spin_unlock_nopin!(pcp);
}
}

fn shuffle_pick_tail() -> bool {
false
}

fn is_shuffle_order(order: i32) -> bool {
false
}

fn is_migrate_isolate(migratetype: i32) -> bool {
false
}

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


fn page_reporting_notify_free(order: u32) {
}

// No special request
// Skip free page reporting notification for the (possibly merged) page.
// This does not hinder free page reporting from grabbing the page,
// reporting it and marking it "reported" -  it only skips notifying
// the free page reporting infrastructure about a newly freed page. For
// example, used when temporarily pulling a page from a freelist and
// putting it back unmodified.
// Place the (possibly merged) page to the tail of the freelist. Will ignore
// page shuffling (relevant code - e.g., memory onlining - is expected to
// shuffle the whole zone).
//
// Note: No code should rely on this flag for correctness - it's purely
//       to allow for optimizations when handing back either fresh pages
//       (memory onlining) or untouched pages (page isolation, free page
//       reporting).
// Free the page without taking locks. Rely on trylock only.
// free_pages_prepare() has already been called for page(s) being freed.
/// Free Page Internal flags: for internal, non-pcp variants of free_pages().
/// prevent >1 _updater_ of zone percpu pageset ->high and ->batch fields
static mut pcp_batch_high_lock: mutex = {
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

// Locking a pcp requires a PCP lookup followed by a spinlock. To avoid
// a migration causing the wrong PCP to be locked and remote memory being
// potentially allocated, pin the task to the CPU for the lookup+lock.
// preempt_disable is used on !RT because it is faster than migrate_disable.
// migrate_disable is used on RT because otherwise RT spinlock usage is
// interfered with and a high priority task cannot preempt the allocator.
// A helper to lookup and trylock pcp with embedded spinlock.
// The return value should be used with the unlock helper.
// NULL return value means the trylock failed.
// On CONFIG_SMP=n the UP implementation of spin_trylock() never fails and thus
// is not compatible with our locking scheme. However we do not need pcp for
// scalability in the first place, so just make all the trylocks fail and take
// the slow path unconditionally.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_numa_node_854: *mut () = (unsafe { core::ptr::addr_of_mut!(numa_node) }) as *mut ();

#[unsafe(no_mangle)]
static mut vm_numa_stat_key: static_key_true = static_key_true { key: {
let mut __t0: static_key = unsafe { core::mem::zeroed() };
__t0.enabled = __SlateRecord_types_h_190 { counter: 1i32 };
__t0.__slate_anon_1 = __SlateRecord_jump_label_h_95 { r#type: 1u64 };
__t0
} };

static mut pcpu_drain_mutex: mutex = {
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

/// Array of node states.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".data..read_mostly")]
static mut node_states: __SlateAlign16<[__SlateRecord_nodemask_types_h_17; 6]> = __SlateAlign16({
let mut __t0: [__SlateRecord_nodemask_types_h_17; 6] = unsafe { core::mem::zeroed() };
__t0[0] = __SlateRecord_nodemask_types_h_17 { bits: [18446744073709551615u64 >> (64i32.wrapping_neg() & 63i32)] };
__t0[1] = __SlateRecord_nodemask_types_h_17 { bits: [1u64] };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_node_states_855: *mut () = (unsafe { core::ptr::addr_of_mut!(node_states.0) }) as *mut ();

#[unsafe(no_mangle)]
#[unsafe(link_section = ".data..read_mostly")]
static mut gfp_allowed_mask: u32 = 16773951;

/// results with 256, 32 in the lowmem_reserve sysctl:
/// 	1G machine -> (16M dma, 800M-16M normal, 1G-800M high)
/// 	1G machine -> (16M dma, 784M normal, 224M high)
/// 	NORMAL allocation will leave 784M/256 of ram reserved in the ZONE_DMA
/// 	HIGHMEM allocation will leave 224M/32 of ram reserved in ZONE_NORMAL
/// 	HIGHMEM allocation will leave (224M+784M)/256 of ram reserved in ZONE_DMA
///
/// TBD: should special case ZONE_DMA32 machines here - in those we normally
/// don't need any ZONE_NORMAL reservation
static mut sysctl_lowmem_reserve_ratio: __SlateAlign16<[i32; 4]> = __SlateAlign16([256i32, 256i32, 32i32, 0i32]);

#[unsafe(no_mangle)]
static mut min_free_kbytes: i32 = 1024;

#[unsafe(no_mangle)]
static mut user_min_free_kbytes: i32 = 1i32.wrapping_neg();

#[unsafe(link_section = ".data..read_mostly")]
static mut watermark_boost_factor: i32 = 15000;

static mut watermark_scale_factor: i32 = 10;

/// movable_zone is the "real" zone pages in ZONE_MOVABLE are taken from
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_movable_zone_856: *mut () = (unsafe { core::ptr::addr_of_mut!(movable_zone) }) as *mut ();

#[unsafe(no_mangle)]
#[unsafe(link_section = ".data..read_mostly")]
static mut nr_node_ids: u32 = 64;

#[unsafe(no_mangle)]
#[unsafe(link_section = ".data..read_mostly")]
static mut nr_online_nodes: u32 = 1;

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_nr_node_ids_857: *mut () = (unsafe { core::ptr::addr_of_mut!(nr_node_ids) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_nr_online_nodes_858: *mut () = (unsafe { core::ptr::addr_of_mut!(nr_online_nodes) }) as *mut ();

// When page allocations stall for longer than a threshold,
// ALLOC_STALL_WARN_MSECS, leave a warning in the kernel log.  Only one warning
// will be printed during this duration for the entire system.
static mut alloc_stall_warn_jiffies: u64 = (300i32.wrapping_neg().wrapping_mul(1000i32) as u32) as u64;

fn __free_unaccepted(page: *mut page) -> bool {
loop {
if true {
unsafe { __compiletime_assert_1103() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
false
}

fn _deferred_grow_zone(zone: *mut zone, order: u32) -> bool {
false
}

/// Return a pointer to the bitmap storing bits affecting a block of pages
fn get_pageblock_bitmap(page: *const page, pfn: u64) -> *mut u64 {
section_to_usemap(__pfn_to_section(pfn))
}

fn pfn_to_bitidx(page: *const page, mut pfn: u64) -> i32 {
pfn &= 32767u64;
(pfn >> MIN_T__2!(u32, 9i32, 10i32)).wrapping_mul(roundup_pow_of_two!(4i32)) as i32
}

#[inline(always)]
fn is_standalone_pb_bit(pb_bit: u32) -> bool {
(pb_bit as u32) >= 3 && (pb_bit as u32) < 4
}

#[inline(always)]
fn get_pfnblock_bitmap_bitidx(page: *const page, pfn: u64, bitmap_word: &mut *mut u64, bitidx: &mut u64) {
let mut word_bitidx: u64 = 0;
loop {
if roundup_pow_of_two__4!(4i32) != 4 {
unsafe { __compiletime_assert_859() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
while false {
}
VM_BUG_ON_PAGE!();
let bitmap: *mut u64 = get_pageblock_bitmap(page, pfn);
unsafe {
*(bitidx as *mut u64) = pfn_to_bitidx(page, pfn) as u64;
}
word_bitidx = (unsafe { *(bitidx as *mut u64) }) / 64;
unsafe {
*bitidx = *bitidx & 63u64;
}
unsafe {
*(bitmap_word as *mut *mut u64) = bitmap.wrapping_offset(word_bitidx as isize);
}
}

// __get_pfnblock_flags_mask - Return the requested group of flags for
// a pageblock_nr_pages block of pages
// @page: The page within the block of interest
// @pfn: The target page frame number
// @mask: mask of bits that the caller is interested in
//
// Return: pageblock_bits flags
fn __get_pfnblock_flags_mask(page: *const page, pfn: u64, mask: u64) -> u64 {
let mut bitmap_word: *mut u64 = unsafe { core::mem::zeroed() };
let mut bitidx: u64 = 0;
let mut word: u64 = 0;
get_pfnblock_bitmap_bitidx(page, pfn, unsafe { &mut (*core::ptr::addr_of_mut!(bitmap_word)) }, unsafe { &mut (*core::ptr::addr_of_mut!(bitidx)) });
// This races, without locks, with set_pfnblock_migratetype(). Ensure
// a consistent read of the memory array, so that results, even though
// racy, are not corrupted.
let mut __v0: u64;
while false {
}
__v0 = unsafe { core::ptr::read_volatile(bitmap_word as *const u64) };
word = __v0;
word >> bitidx & mask
}

/// get_pfnblock_bit - Check if a standalone bit of a pageblock is set
/// @page: The page within the block of interest
/// @pfn: The target page frame number
/// @pb_bit: pageblock bit to check
///
/// Return: true if the bit is set, otherwise false
#[unsafe(no_mangle)]
extern "C" fn get_pfnblock_bit(page: *const page, pfn: u64, pb_bit: u32) -> bool {
let mut bitmap_word: *mut u64 = unsafe { core::mem::zeroed() };
let mut bitidx: u64 = 0;
let mut __v0: i64;
let __ret_warn_on: i32 = !(!(!is_standalone_pb_bit(pb_bit))) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("862: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 862b - ., 3; .popsection\n# {0}", const 862 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_get_pfnblock_bit_1, sym __slate_asm_string_get_pfnblock_bit_2, const 423 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("863: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 863b - ., 4; .popsection\n# {0}", const 863 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
return false;
}
get_pfnblock_bitmap_bitidx(page, pfn, unsafe { &mut (*core::ptr::addr_of_mut!(bitmap_word)) }, unsafe { &mut (*core::ptr::addr_of_mut!(bitidx)) });
test_bit__10!(bitidx.wrapping_add((pb_bit as u32) as u64), bitmap_word) != 0
}

/// get_pfnblock_migratetype - Return the migratetype of a pageblock
/// @page: The page within the block of interest
/// @pfn: The target page frame number
///
/// Return: The migratetype of the pageblock
///
/// Use get_pfnblock_migratetype() if caller already has both @page and @pfn
/// to save a call to page_to_pfn().
#[unsafe(no_mangle)]
extern "C" fn get_pfnblock_migratetype(page: *const page, pfn: u64) -> u32 {
let mask: u64 = BIT!(0i32) | BIT!(1i32) | BIT!(2i32) | 0;
let flags: u64 = __get_pfnblock_flags_mask(page, pfn, mask);
(flags & (BIT!(0i32) | BIT!(1i32) | BIT!(2i32))) as u32
}

// __set_pfnblock_flags_mask - Set the requested group of flags for
// a pageblock_nr_pages block of pages
// @page: The page within the block of interest
// @pfn: The target page frame number
// @flags: The flags to set
// @mask: mask of bits that the caller is interested in
fn __set_pfnblock_flags_mask(page: *mut page, pfn: u64, mut flags: u64, mut mask: u64) {
let mut bitmap_word: *mut u64 = unsafe { core::mem::zeroed() };
let mut bitidx: u64 = 0;
let mut word: u64 = 0;
get_pfnblock_bitmap_bitidx(page as *const page, pfn, unsafe { &mut (*core::ptr::addr_of_mut!(bitmap_word)) }, unsafe { &mut (*core::ptr::addr_of_mut!(bitidx)) });
mask <<= bitidx;
flags <<= bitidx;
let mut __v14: u64;
while false {
}
__v14 = unsafe { core::ptr::read_volatile(bitmap_word as *const u64) };
word = __v14;
loop {
let mut __v17: i64;
let __v0: *mut u64 = bitmap_word;
let __v1: *mut u64 = core::ptr::addr_of_mut!(word);
kcsan_mb!();
instrument_atomic_read_write(__v0 as *const (), core::mem::size_of::<u64>() as u64);
instrument_read_write(__v1 as *const (), core::mem::size_of::<u64>() as u64);
let mut __v19: i64;
let mut __v2: bool = false;
let mut __v4: u64 = unsafe { *__v1 };
let mut __v5: u64 = word & !mask | flags;
match (const { core::mem::size_of::<u64>() }) as u64 {
1 => {
let mut __v6: *mut u8 = __v0 as *mut u8;
unsafe {
let __t0: u8;
let __t1: u64;
core::arch::asm!("lock cmpxchgb {2:r}, ({1:r})\nsetz {0}", lateout(reg_byte) __t0, in(reg) __v6, in(reg) __v5 as u64, inlateout("ax") __v4 as u64 => __t1, options(nostack, att_syntax));
__v2 = (__t0 as u8) != 0;
__v4 = unsafe { core::mem::transmute::<u64, u64>(__t1 as u64) };
}
}
2 => {
let mut __v7: *mut u16 = __v0 as *mut u16;
unsafe {
let __t2: u8;
let __t3: u64;
core::arch::asm!("lock cmpxchgw {2:r}, ({1:r})\nsetz {0}", lateout(reg_byte) __t2, in(reg) __v7, in(reg) __v5 as u64, inlateout("ax") __v4 as u64 => __t3, options(nostack, att_syntax));
__v2 = (__t2 as u8) != 0;
__v4 = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
}
4 => {
let mut __v8: *mut u32 = __v0 as *mut u32;
unsafe {
let __t4: u8;
let __t5: u64;
core::arch::asm!("lock cmpxchgl {2:r}, ({1:r})\nsetz {0}", lateout(reg_byte) __t4, in(reg) __v8, in(reg) __v5 as u64, inlateout("ax") __v4 as u64 => __t5, options(nostack, att_syntax));
__v2 = (__t4 as u8) != 0;
__v4 = unsafe { core::mem::transmute::<u64, u64>(__t5 as u64) };
}
}
8 => {
let mut __v9: *mut u64 = __v0 as *mut u64;
unsafe {
let __t6: u8;
let __t7: u64;
core::arch::asm!("lock cmpxchgq {2:r}, ({1:r})\nsetz {0}", lateout(reg_byte) __t6, in(reg) __v9, in(reg) __v5 as u64, inlateout("ax") __v4 as u64 => __t7, options(nostack, att_syntax));
__v2 = (__t6 as u8) != 0;
__v4 = unsafe { core::mem::transmute::<u64, u64>(__t7 as u64) };
}
}
_ => {
unsafe { __cmpxchg_wrong_size() };
}
}
if unlikely!(!__v2) != 0 {
unsafe {
*__v1 = __v4;
}
}
__v19 = __v2 as i64;
__v17 = __v19;
if __v17 != 0 {
break;
}
}
}

/// set_pfnblock_bit - Set a standalone bit of a pageblock
/// @page: The page within the block of interest
/// @pfn: The target page frame number
/// @pb_bit: pageblock bit to set
#[unsafe(no_mangle)]
extern "C" fn set_pfnblock_bit(page: *const page, pfn: u64, pb_bit: u32) {
let mut bitmap_word: *mut u64 = unsafe { core::mem::zeroed() };
let mut bitidx: u64 = 0;
let mut __v0: i64;
let __ret_warn_on: i32 = !(!(!is_standalone_pb_bit(pb_bit))) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("865: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 865b - ., 3; .popsection\n# {0}", const 865 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_set_pfnblock_bit_1, sym __slate_asm_string_set_pfnblock_bit_2, const 493 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("866: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 866b - ., 4; .popsection\n# {0}", const 866 as i32, options(nostack, att_syntax));
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
get_pfnblock_bitmap_bitidx(page, pfn, unsafe { &mut (*core::ptr::addr_of_mut!(bitmap_word)) }, unsafe { &mut (*core::ptr::addr_of_mut!(bitidx)) });
set_bit(bitidx.wrapping_add((pb_bit as u32) as u64) as i64, bitmap_word as *mut u64);
}

/// clear_pfnblock_bit - Clear a standalone bit of a pageblock
/// @page: The page within the block of interest
/// @pfn: The target page frame number
/// @pb_bit: pageblock bit to clear
#[unsafe(no_mangle)]
extern "C" fn clear_pfnblock_bit(page: *const page, pfn: u64, pb_bit: u32) {
let mut bitmap_word: *mut u64 = unsafe { core::mem::zeroed() };
let mut bitidx: u64 = 0;
let mut __v0: i64;
let __ret_warn_on: i32 = !(!(!is_standalone_pb_bit(pb_bit))) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("867: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 867b - ., 3; .popsection\n# {0}", const 867 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_clear_pfnblock_bit_1, sym __slate_asm_string_clear_pfnblock_bit_2, const 513 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("868: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 868b - ., 4; .popsection\n# {0}", const 868 as i32, options(nostack, att_syntax));
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
get_pfnblock_bitmap_bitidx(page, pfn, unsafe { &mut (*core::ptr::addr_of_mut!(bitmap_word)) }, unsafe { &mut (*core::ptr::addr_of_mut!(bitidx)) });
clear_bit(bitidx.wrapping_add((pb_bit as u32) as u64) as i64, bitmap_word as *mut u64);
}

/// set_pageblock_migratetype - Set the migratetype of a pageblock
/// @page: The page within the block of interest
/// @migratetype: migratetype to set
fn set_pageblock_migratetype(page: *mut page, mut migratetype: u32) {
if unlikely!((unsafe { page_group_by_mobility_disabled }) != 0 && (migratetype as u32) < 3) != 0 {
migratetype = 0;
}
__set_pfnblock_flags_mask(page, ((unsafe { page.offset_from(vmemmap_base as *mut page) }) as i64) as u64, (migratetype as u32) as u64, BIT!(0i32) | BIT!(1i32) | BIT!(2i32) | 0);
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn init_pageblock_migratetype(page: *mut page, mut migratetype: u32, isolate: bool) {
let mut flags: u64 = 0;
if unlikely!((unsafe { page_group_by_mobility_disabled }) != 0 && (migratetype as u32) < 3) != 0 {
migratetype = 0;
}
flags = (migratetype as u32) as u64;
__set_pfnblock_flags_mask(page, ((unsafe { page.offset_from(vmemmap_base as *mut page) }) as i64) as u64, flags, BIT!(0i32) | BIT!(1i32) | BIT!(2i32) | 0);
}

fn bad_range(zone: &zone, page: &page) -> bool {
false
}

fn bad_page(page: *mut page, reason: *const u8) {
unsafe {
'__join_2: {
// Allow a burst of 60 reports, then keep quiet for that minute;
// or allow a steady drip of one report per second.
if nr_shown == 60 {
if time_before!(core::ptr::read_volatile(core::ptr::addr_of!(jiffies)), unsafe { resume }) {
let __v5: u64 = nr_unshown;
nr_unshown = __v5.wrapping_add(1);
break '__join_2;
} else {
if nr_unshown != 0 {
while false {
}
_printk(c"\x011BUG: Bad page state: %lu messages suppressed\n".as_ptr() as *const u8, nr_unshown);
nr_unshown = 0;
}
nr_shown = 0;
}
}
let __v2: u64 = nr_shown;
nr_shown = __v2.wrapping_add(1);
if __v2 == 0 {
resume = core::ptr::read_volatile(core::ptr::addr_of!(jiffies)).wrapping_add(60000u64);
}
while false {
}
_printk(c"\x011BUG: Bad page state in process %s  pfn:%05lx\n".as_ptr() as *const u8, (*get_current()).comm.as_mut_ptr() as *mut u8, (page.offset_from(vmemmap_base as *mut page) as i64) as u64);
dump_page(page as *const page, reason);
print_modules();
dump_stack();
}
if PageBuddy(page as *const page) != 0 {
__ClearPageBuddy(page);
}
// Leave bad fields for debug, except PageBuddy could make trouble
add_taint(5u32, 1u32);
}
}

static mut resume: u64 = unsafe { core::mem::zeroed() };

static mut nr_shown: u64 = unsafe { core::mem::zeroed() };

static mut nr_unshown: u64 = unsafe { core::mem::zeroed() };

fn order_to_pindex(migratetype: i32, order: i32) -> u32 {
if IS_ENABLED!() != 0 {
let movable: bool = migratetype == 1;
if order > 3 {
return 12i32.wrapping_add(movable as i32) as u32;
}
}
3i32.wrapping_mul(order).wrapping_add(migratetype) as u32
}

fn pindex_to_order(pindex: u32) -> i32 {
let mut order: i32 = (pindex / 3) as i32;
if IS_ENABLED!() != 0 {
if pindex >= 12u32 {
order = 9i32;
}
}
order
}

fn pcp_allowed_order(order: u32) -> bool {
if order <= 3 {
return true;
}
false
}

// Higher-order pages are called "compound pages".  They are structured thusly:
//
// The first PAGE_SIZE page is called the "head page" and have PG_head set.
//
// The remaining PAGE_SIZE pages are called "tail pages". PageTail() is encoded
// in bit 0 of page->compound_info. The rest of bits is pointer to head page.
//
// The first tail page's ->compound_order holds the order of allocation.
// This usage means that zero-order pages may not be compound.
#[unsafe(no_mangle)]
extern "C" fn prep_compound_page(page: *mut page, order: u32) {
let nr_pages: i32 = 1i32 << order;
__SetPageHead(page);
for i in 1..nr_pages {
prep_compound_tail(page.wrapping_offset(i as isize), page as *const page, order);
}
prep_compound_head(page, order);
}

fn set_buddy_order(page: *mut page, order: u32) {
set_page_private(page, order as u64);
__SetPageBuddy(page);
}

fn task_capc(zone: *mut zone) -> *mut capture_control {
let capc: *mut capture_control = unsafe { (*get_current()).capture_control };
let __v0: bool = unlikely__13!(capc) != 0 && (unsafe { (*get_current()).flags }) & 2097152 == 0;
if __v0 && unsafe { (*capc).page }.is_null() && (unsafe { (*capc).zone }) == zone { capc } else { core::ptr::null_mut::<capture_control>() }
}

fn compaction_capture(capc: *mut capture_control, page: *mut page, order: i32, migratetype: i32) -> bool {
if capc.is_null() || order != unsafe { (*capc).order } {
return false;
}
// Do not accidentally pollute CMA or isolated regions
let __v0: bool = is_migrate_cma!() != 0 || is_migrate_isolate(migratetype);
if __v0 {
return false;
}
// Do not let lower order allocations pollute a movable pageblock
// unless compaction is also requesting movable pages.
// This might let an unmovable request use a reclaimable pageblock
// and vice-versa but no more than normal fallback logic which can
// have trouble finding a high-order free page.
if (order as u32) < MIN_T!(u32, 9i32) && migratetype == 1 && (unsafe { (*capc).migratetype }) != 1 {
return false;
}
if migratetype != unsafe { (*capc).migratetype } {
trace_mm_page_alloc_extfrag(page, unsafe { (*capc).order }, order, unsafe { (*capc).migratetype }, migratetype);
}
unsafe {
(*capc).page = page;
}
true
}

fn account_freepages(zone: *mut zone, nr_pages: i32, migratetype: i32) {
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!((*zone).lock) });
if is_migrate_isolate(migratetype) {
return;
}
unsafe { __mod_zone_page_state(zone, 0u32, nr_pages as i64) };
if is_migrate_cma!() != 0 {
unsafe { __mod_zone_page_state(zone, 9u32, nr_pages as i64) };
} else {
if migratetype == 3 {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_869() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__4!((*zone).nr_free_highatomic, unsafe { (*zone).nr_free_highatomic }.wrapping_add(nr_pages as u64));
if true {
break;
}
}
}
}
}

/// Used for pages not on another list
fn __add_to_free_list(page: *mut page, zone: *mut zone, order: u32, migratetype: i32, tail: bool) {
let area: *mut free_area = unsafe { (*zone).free_area.as_mut_ptr() as *mut free_area }.wrapping_offset(order as isize);
let nr_pages: i32 = 1i32 << order;
BUILD_BUG_ON_INVALID!();
if tail {
list_add_tail(unsafe { core::ptr::addr_of_mut!((*page).__slate_anon_1.__slate_anon_0.__slate_anon_0.buddy_list) }, unsafe { (*area).free_list.as_mut_ptr() as *mut list_head }.wrapping_offset(migratetype as isize));
} else {
list_add(unsafe { core::ptr::addr_of_mut!((*page).__slate_anon_1.__slate_anon_0.__slate_anon_0.buddy_list) }, unsafe { (*area).free_list.as_mut_ptr() as *mut list_head }.wrapping_offset(migratetype as isize));
}
let __v1: u64 = unsafe { (*area).nr_free };
unsafe {
(*area).nr_free = __v1.wrapping_add(1);
}
let __v3: bool = order >= MIN_T__2!(u32, 9i32, 10i32) && !is_migrate_isolate(migratetype);
if __v3 {
unsafe { __mod_zone_page_state(zone, 1u32, nr_pages as i64) };
}
}

/// Used for pages which are on another list. Move the pages to the tail
/// of the list - so the moved pages won't immediately be considered for
/// allocation again (e.g., optimization for memory onlining).
fn move_to_free_list(page: *mut page, zone: *mut zone, order: u32, old_mt: i32, new_mt: i32) {
let area: *mut free_area = unsafe { (*zone).free_area.as_mut_ptr() as *mut free_area }.wrapping_offset(order as isize);
let mut nr_pages: i32 = 1i32 << order;
// Free page moving can fail, so it happens before the type update
BUILD_BUG_ON_INVALID!();
list_move_tail(unsafe { core::ptr::addr_of_mut!((*page).__slate_anon_1.__slate_anon_0.__slate_anon_0.buddy_list) }, unsafe { (*area).free_list.as_mut_ptr() as *mut list_head }.wrapping_offset(new_mt as isize));
account_freepages(zone, nr_pages.wrapping_neg(), old_mt);
account_freepages(zone, nr_pages, new_mt);
let __v0: bool = order >= MIN_T__2!(u32, 9i32, 10i32) && (is_migrate_isolate(old_mt) as i32) != (is_migrate_isolate(new_mt) as i32);
if __v0 {
if !is_migrate_isolate(old_mt) {
nr_pages = nr_pages.wrapping_neg();
}
unsafe { __mod_zone_page_state(zone, 1u32, nr_pages as i64) };
}
}

fn __del_page_from_free_list(page: *mut page, zone: *mut zone, order: u32, migratetype: i32) {
let nr_pages: i32 = 1i32 << order;
BUILD_BUG_ON_INVALID!();
// clear reported state and update reported page count
if page_reported!() != 0 {
__ClearPageReported(page);
}
list_del(unsafe { core::ptr::addr_of_mut!((*page).__slate_anon_1.__slate_anon_0.__slate_anon_0.buddy_list) });
__ClearPageBuddy(page);
set_page_private(page, 0u64);
let __v0: *mut free_area = unsafe { (*zone).free_area.as_mut_ptr() as *mut free_area }.wrapping_offset(order as isize);
let __v1: u64 = unsafe { (*__v0).nr_free };
unsafe {
(*__v0).nr_free = __v1.wrapping_sub(1);
}
let __v3: bool = order >= MIN_T__2!(u32, 9i32, 10i32) && !is_migrate_isolate(migratetype);
if __v3 {
unsafe { __mod_zone_page_state(zone, 1u32, nr_pages.wrapping_neg() as i64) };
}
}

fn del_page_from_free_list(page: *mut page, zone: *mut zone, order: u32, migratetype: i32) {
__del_page_from_free_list(page, zone, order, migratetype);
account_freepages(zone, (1i32 << order).wrapping_neg(), migratetype);
}

fn get_page_from_free_area(area: *mut free_area, migratetype: i32) -> *mut page {
let mut __v0: *mut page;
let head__: *mut list_head = unsafe { (*area).free_list.as_mut_ptr() as *mut list_head }.wrapping_offset(migratetype as isize);
let mut __v1: *mut list_head;
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_870() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*head__).next) as *const *mut list_head) };
let pos__: *mut list_head = __v1;
let __v3: *mut page = if pos__ != head__ {
list_entry__2!(pos__, page)
} else {
core::ptr::null_mut::<page>()
};
__v0 = __v3;
__v0
}

/// If this is less than the 2nd largest possible page, check if the buddy
/// of the next-higher order is free. If it is, it's possible
/// that pages are being freed that will coalesce soon. In case,
/// that is happening, add the free page to the tail of the list
/// so it's less likely to be used soon and more likely to be merged
/// as a 2-level higher order page
fn buddy_merge_likely(pfn: u64, buddy_pfn: u64, page: *mut page, order: u32) -> bool {
let mut higher_page_pfn: u64 = 0;
if order >= 9u32 {
return false;
}
higher_page_pfn = buddy_pfn & pfn;
let higher_page: *mut page = page.wrapping_offset(higher_page_pfn.wrapping_sub(pfn) as isize);
!find_buddy_page_pfn(higher_page, higher_page_pfn, order.wrapping_add(1), core::ptr::null_mut::<u64>()).is_null()
}

fn change_pageblock_range(mut pageblock_page: *mut page, start_order: i32, migratetype: i32) {
let mut nr_pageblocks: i32 = 1i32 << (start_order as u32).wrapping_sub(MIN_T!(u32, 9i32));
loop {
let __v2: i32 = nr_pageblocks;
nr_pageblocks = __v2.wrapping_sub(1);
if __v2 == 0 {
break;
}
set_pageblock_migratetype(pageblock_page, migratetype as u32);
let __v0: *mut page = pageblock_page;
pageblock_page = __v0.wrapping_offset((1u64 << MIN_T!(u32, 9i32)) as isize);
}
}

// Freeing function for a buddy system allocator.
//
// The concept of a buddy system is to maintain direct-mapped table
// (containing bit values) for memory blocks of various "orders".
// The bottom level table contains the map for the smallest allocatable
// units of memory (here, pages), and each level above it describes
// pairs of units from the levels below, hence, "buddies".
// At a high level, all that happens here is marking the table entry
// at the bottom level available, and propagating the changes upward
// as necessary, plus some accounting needed to play nicely with other
// parts of the VM system.
// At each level, we keep a list of pages, which are heads of continuous
// free pages of length of (1 << order) and marked with PageBuddy.
// Page's order is recorded in page_private(page) field.
// So when we are allocating or freeing one, we can derive the state of the
// other.  That is, if we allocate a small block, and both were
// free, the remainder of the region must be split into blocks.
// If a block is freed, and its buddy is also free, then this
// triggers coalescing into a block of larger size.
//
// -- nyc
fn __free_one_page(mut page: *mut page, mut pfn: u64, zone: *mut zone, mut order: u32, migratetype: i32, fpi_flags: i32) {
let mut __v2: bool = unsafe { core::mem::zeroed() };
let mut __v3: bool = unsafe { core::mem::zeroed() };
let mut buddy_mt: i32 = unsafe { core::mem::zeroed() };
let mut to_tail: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut bool = core::ptr::addr_of_mut!(to_tail) as *mut bool;
let mut buddy: core::mem::MaybeUninit<*mut page> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut page = core::ptr::addr_of_mut!(buddy) as *mut *mut page;
let mut combined_pfn: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u64 = core::ptr::addr_of_mut!(combined_pfn) as *mut u64;
let mut buddy_pfn: u64 = unsafe { core::mem::zeroed() };
let mut capc: *mut capture_control = unsafe { core::mem::zeroed() };
unsafe {
capc = task_capc(zone);
buddy_pfn = 0;
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
VM_BUG_ON!();
VM_BUG_ON!();
VM_BUG_ON_PAGE!();
VM_BUG_ON_PAGE!();
account_freepages(zone, 1i32 << order, migratetype);
while order < 10 {
buddy_mt = migratetype;
if compaction_capture(capc, page, order as i32, migratetype) {
account_freepages(zone, (1i32 << order).wrapping_neg(), migratetype);
return;
} else {
*__slate_slot_6 = find_buddy_page_pfn(page, pfn, order, core::ptr::addr_of_mut!(buddy_pfn));
if (*__slate_slot_6).is_null() {
break;
} else {
if unlikely!(order >= MIN_T!(u32, 9i32)) != 0 {
// We want to prevent merge between freepages on pageblock
// without fallbacks and normal pageblock. Without this,
// pageblock isolation could cause incorrect freepage or CMA
// accounting or HIGHATOMIC accounting.
buddy_mt = (get_pfnblock_migratetype(*__slate_slot_6 as *const page, buddy_pfn) as u32) as i32;
if migratetype != buddy_mt {
if !migratetype_is_mergeable(migratetype) {
__v2 = true;
} else {
__v2 = !migratetype_is_mergeable(buddy_mt);
}
__v3 = __v2;
} else {
__v3 = false;
}
if __v3 {
break;
}
}
// Our buddy is free or it is CONFIG_DEBUG_PAGEALLOC guard page,
// merge with it and move up one order.
if page_is_guard(*__slate_slot_6 as *const page) {
clear_page_guard(zone, *__slate_slot_6, order);
} else {
__del_page_from_free_list(*__slate_slot_6, zone, order, buddy_mt);
}
if unlikely!(buddy_mt != migratetype) != 0 {
// Match buddy type. This ensures that an
// expand() down the line puts the sub-blocks
// on the right freelists.
change_pageblock_range(*__slate_slot_6, order as i32, migratetype);
}
*__slate_slot_7 = buddy_pfn & pfn;
page = page.wrapping_offset((*__slate_slot_7).wrapping_sub(pfn) as isize);
pfn = *__slate_slot_7;
order = order.wrapping_add(1);
}
}
}
set_buddy_order(page, order);
if fpi_flags & (BIT!(1i32) as i32) != 0 {
*__slate_slot_5 = true;
} else {
if is_shuffle_order(order as i32) {
*__slate_slot_5 = shuffle_pick_tail();
} else {
*__slate_slot_5 = buddy_merge_likely(pfn, buddy_pfn, page, order);
}
}
__add_to_free_list(page, zone, order, migratetype, *__slate_slot_5);
// Notify page reporting subsystem of freed page
if fpi_flags & (BIT!(0i32) as i32) == 0 {
page_reporting_notify_free(order);
}
}
}

/// A bad page could be due to a number of fields. Instead of multiple branches,
/// try and check multiple fields with one check. The caller must do a detailed
/// check if necessary.
fn page_expected_state(page: *mut page, check_flags: u64) -> bool {
if unlikely!(atomic_read((unsafe { core::ptr::addr_of_mut!((*page).__slate_anon_2._mapcount) }) as *const __SlateRecord_types_h_190) != 1i32.wrapping_neg()) != 0 {
return false;
}
if unlikely__4!(((unsafe { (*page).__slate_anon_1.__slate_anon_0.mapping }) as u64) | (page_ref_count(page as *const page) as u64) | ((page_pool_page_is_pp(page as *const page) as i32) as u64) | (unsafe { (*page).flags.f }) & check_flags) != 0 {
return false;
}
true
}

fn page_bad_reason(page: *mut page, flags: u64) -> *const u8 {
let mut bad_reason: *const u8 = core::ptr::null::<u8>();
if unlikely!(atomic_read((unsafe { core::ptr::addr_of_mut!((*page).__slate_anon_2._mapcount) }) as *const __SlateRecord_types_h_190) != 1i32.wrapping_neg()) != 0 {
bad_reason = c"nonzero mapcount".as_ptr() as *const u8;
}
if unlikely!(!unsafe { (*page).__slate_anon_1.__slate_anon_0.mapping }.is_null()) != 0 {
bad_reason = c"non-NULL mapping".as_ptr() as *const u8;
}
if unlikely!(page_ref_count(page as *const page) != 0) != 0 {
bad_reason = c"nonzero _refcount".as_ptr() as *const u8;
}
if unlikely__4!((unsafe { (*page).flags.f }) & flags) != 0 {
let mut __v0: i32;
let __UNIQUE_ID_x__877: i32 = 0;
let __UNIQUE_ID_y__878: i32 = 34i32;
loop {
if __types_ok!(__UNIQUE_ID_x__877, __UNIQUE_ID_y__878) == 0 {
unsafe { __compiletime_assert_879() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__877 < __UNIQUE_ID_y__878 { __UNIQUE_ID_x__877 } else { __UNIQUE_ID_y__878 };
let mut __v2: i32;
let __UNIQUE_ID_x__880: i32 = 0;
let __UNIQUE_ID_y__881: i32 = 34i32;
loop {
if __types_ok!(__UNIQUE_ID_x__880, __UNIQUE_ID_y__881) == 0 {
unsafe { __compiletime_assert_882() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = if __UNIQUE_ID_x__880 < __UNIQUE_ID_y__881 { __UNIQUE_ID_x__880 } else { __UNIQUE_ID_y__881 };
if flags == 4194303u64 | BIT!(0i32).wrapping_sub(1) << (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_sub(2).wrapping_sub(0).wrapping_sub(0).wrapping_sub(0) | (UL!() << __v0).wrapping_sub(1) << (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_sub(2).wrapping_sub(0).wrapping_sub(0).wrapping_sub(0).wrapping_sub(__v2 as u64) {
bad_reason = c"PAGE_FLAGS_CHECK_AT_PREP flag(s) set".as_ptr() as *const u8;
} else {
bad_reason = c"PAGE_FLAGS_CHECK_AT_FREE flag(s) set".as_ptr() as *const u8;
}
}
if unlikely!(page_pool_page_is_pp(page as *const page)) != 0 {
bad_reason = c"page_pool leak".as_ptr() as *const u8;
}
bad_reason
}

fn free_page_is_bad(page: *mut page) -> bool {
if likely!(page_expected_state(page, 1368355u64 | BIT!(0i32).wrapping_sub(1) << (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_sub(2).wrapping_sub(0).wrapping_sub(0).wrapping_sub(0))) != 0 {
return false;
}
// Something has gone sideways, find it
bad_page(page, page_bad_reason(page, 1368355u64 | BIT!(0i32).wrapping_sub(1) << (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_sub(2).wrapping_sub(0).wrapping_sub(0).wrapping_sub(0)));
true
}

fn is_check_pages_enabled() -> bool {
static_branch_unlikely__21!(unsafe { core::ptr::addr_of_mut!(check_pages_enabled) }) != 0
}

fn free_tail_page_prepare(head_page: *mut page, page: *mut page) -> i32 {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut folio: *mut folio = unsafe { core::mem::zeroed() };
unsafe {
folio = head_page as *mut folio;
ret = 1;
// We rely page->lru.next never has bit 0 set, unless the page
// is PageTail(). Let's make sure that's true even for poisoned ->lru.
loop {
if (((256i32 as *const u8).wrapping_offset(_AC__3!() as isize) as *mut ()) as u64) & 1 != 0 {
__compiletime_assert_883();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
'__join_0: {
if !is_check_pages_enabled() {
ret = 0;
} else {
let __t0: i64 = page.offset_from(head_page as *mut page) as i64;
if __t0 == 1 {
if unlikely__2!(folio_large_mapcount(folio as *const folio)) != 0 {
bad_page(page, c"nonzero large_mapcount".as_ptr() as *const u8);
break '__join_0;
} else {
// the first tail page: these may be in place of ->mapping
if IS_ENABLED__2!() != 0 {
__v0 = unlikely__2!(atomic_read(core::ptr::addr_of_mut!((*folio).__slate_anon_1.__slate_anon_0.__slate_anon_2.__slate_anon_0._nr_pages_mapped) as *const __SlateRecord_types_h_190)) != 0;
} else {
__v0 = false;
}
if __v0 {
bad_page(page, c"nonzero nr_pages_mapped".as_ptr() as *const u8);
break '__join_0;
} else {
if unlikely__2!(atomic_read(core::ptr::addr_of_mut!((*folio).__slate_anon_1.__slate_anon_0.__slate_anon_2.__slate_anon_0._entire_mapcount) as *const __SlateRecord_types_h_190).wrapping_add(1)) != 0 {
bad_page(page, c"nonzero entire_mapcount".as_ptr() as *const u8);
break '__join_0;
} else {
if unlikely__2!(atomic_read(core::ptr::addr_of_mut!((*folio).__slate_anon_1.__slate_anon_0.__slate_anon_2.__slate_anon_0._pincount) as *const __SlateRecord_types_h_190)) != 0 {
bad_page(page, c"nonzero pincount".as_ptr() as *const u8);
break '__join_0;
}
}
}
}
} else {
if __t0 == 2 {
if unlikely!(list_empty(core::ptr::addr_of_mut!((*folio).__slate_anon_2.__slate_anon_0._deferred_list) as *const list_head) == 0) != 0 {
bad_page(page, c"on deferred list".as_ptr() as *const u8);
break '__join_0;
}
// the second tail page: deferred_list overlaps ->mapping
} else {
if __t0 != 3 {
if (*page).__slate_anon_1.__slate_anon_0.mapping != (((1024i32 as *const u8).wrapping_offset(_AC__3!() as isize) as *mut ()) as *mut address_space) {
bad_page(page, c"corrupted mapping in tail page".as_ptr() as *const u8);
break '__join_0;
}
}
}
}
if unlikely!(PageTail(page as *const page) == 0) != 0 {
bad_page(page, c"PageTail not set".as_ptr() as *const u8);
} else {
if unlikely!(compound_head__2!(page) != head_page) != 0 {
bad_page(page, c"compound_head not consistent".as_ptr() as *const u8);
} else {
ret = 0;
}
}
}
}
(*page).__slate_anon_1.__slate_anon_0.mapping = core::ptr::null_mut::<address_space>();
clear_compound_head(page);
return ret;
}
// the third tail page: hugetlb specifics overlap ->mappings
}

/// Skip KASAN memory poisoning when either:
///
/// 1. For generic KASAN: deferred memory initialization has not yet completed.
///    Tag-based KASAN modes skip pages freed via deferred memory initialization
///    using page tags instead (see below).
/// 2. For tag-based KASAN modes: the page has a match-all KASAN tag, indicating
///    that error detection is disabled for accesses via the page address.
///
/// Pages will have match-all tags in the following circumstances:
///
/// 1. Pages are being initialized for the first time, including during deferred
///    memory init; see the call to page_kasan_tag_reset in __init_single_page.
/// 2. The allocation was not unpoisoned due to __GFP_SKIP_KASAN, with the
///    exception of pages unpoisoned by kasan_unpoison_vmalloc.
/// 3. The allocation was excluded from being checked due to sampling,
///    see the call to kasan_unpoison_pages.
///
/// Poisoning pages during deferred memory init will greatly lengthen the
/// process and cause problem in large memory systems as the deferred pages
/// initialization is done with interrupt disabled.
///
/// Assuming that there will be no reference to those newly initialized
/// pages before they are ever allocated, this should have no effect on
/// KASAN memory tracking as the poison will be properly inserted at page
/// allocation time. The only corner case is when pages are allocated by
/// on-demand allocation and then freed again before the deferred pages
/// initialization is done, but this is not likely to happen.
fn should_skip_kasan_poison(page: *mut page) -> bool {
if IS_ENABLED!() != 0 {
return deferred_pages_enabled();
}
(page_kasan_tag(page as *const page) as i32) == 255
}

fn clear_highpages_kasan_tagged(page: *mut page, numpages: i32) {
// s390's use of memset() could override KASAN redzones.
kasan_disable_current();
if IS_ENABLED!() == 0 {
clear_pages(kasan_reset_tag(page_address!(page) as *const ()), numpages as u32);
} else {
for i in 0..numpages {
clear_highpage_kasan_tagged(page.wrapping_offset(i as isize));
}
}
kasan_enable_current();
}

fn pgalloc_tag_add(page: *mut page, task: *mut task_struct, nr: u32, alloc_flags: u32) {
}

fn pgalloc_tag_sub(page: *mut page, nr: u32) {
}

fn pgalloc_tag_sub_pages(tag: *mut alloc_tag, nr: u32) {
}

#[inline(always)]
fn __free_pages_prepare(page: *mut page, order: u32, fpi_flags: i32) -> bool {
let mut bad: i32 = 0;
let skip_kasan_poison: bool = should_skip_kasan_poison(page);
let mut init: bool = want_init_on_free();
let compound: bool = PageCompound(page as *const page) != 0;
let folio: *mut folio = page_folio!(page);
if fpi_flags & (BIT!(3i32) as i32) != 0 {
return true;
}
VM_BUG_ON_PAGE!();
trace_mm_page_free(page, order);
kmsan_free_page(page, order);
let __v0: bool = memcg_kmem_online() && PageMemcgKmem(page);
if __v0 {
__memcg_kmem_uncharge_page(page, order as i32);
}
// In rare cases, when truncation or holepunching raced with
// munlock after VM_LOCKED was cleared, Mlocked may still be
// found set here.  This does not indicate a problem, unless
// "unevictable_pgs_cleared" appears worryingly large.
if unlikely!(folio_test_mlocked(folio as *const folio)) != 0 {
let nr_pages: i64 = folio_nr_pages(folio as *const folio) as i64;
__folio_clear_mlocked(folio);
zone_stat_mod_folio(folio, 8u32, nr_pages.wrapping_neg());
count_vm_events(58u32, nr_pages);
}
if unlikely__2!(PageHWPoison(page as *const page)) != 0 && order == 0 {
// Do not let hwpoison pages hit pcplists/buddy
reset_page_owner(page, order as u16);
page_table_check_free(page, order);
pgalloc_tag_sub(page, (1i32 << order) as u32);
// The page is isolated and accounted for.
// Mark the codetag as empty to avoid accounting error
// when the page is freed by unpoison_memory().
clear_page_tag_ref(page);
return false;
}
VM_BUG_ON_PAGE!();
// Check tail pages before head page information is cleared to
// avoid checking PageCompound for order-0 pages.
if unlikely__3!(order) != 0 {
if compound {
let __v1: *mut page = page.wrapping_offset(1 as isize);
let __v3: u64 = (unsafe { (*__v1).flags.f }) & 18446744073709485056u64;
unsafe {
(*__v1).flags.f = __v3;
}
}
for i in 1..(1i32 << order) {
'__slate_continue_9: {
let tail_page: *mut page = page.wrapping_offset(i as isize);
if compound {
let __v4: i32 = free_tail_page_prepare(page, tail_page);
bad = bad.wrapping_add(__v4);
}
if is_check_pages_enabled() {
if free_page_is_bad(tail_page) {
bad = bad.wrapping_add(1);
break '__slate_continue_9;
}
if (unsafe { (*tail_page).__slate_anon_1.__slate_anon_0.private }) != 0 {
bad_page(tail_page, c"nonzero private".as_ptr() as *const u8);
bad = bad.wrapping_add(1);
break '__slate_continue_9;
}
}
let mut __v12: i32;
let __UNIQUE_ID_x__884: i32 = 0;
let __UNIQUE_ID_y__885: i32 = 34i32;
loop {
if __types_ok__4!(__UNIQUE_ID_x__884, __UNIQUE_ID_y__885) == 0 {
unsafe { __compiletime_assert_886() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v12 = if __UNIQUE_ID_x__884 < __UNIQUE_ID_y__885 { __UNIQUE_ID_x__884 } else { __UNIQUE_ID_y__885 };
let mut __v14: i32;
let __UNIQUE_ID_x__887: i32 = 0;
let __UNIQUE_ID_y__888: i32 = 34i32;
loop {
if __types_ok__4!(__UNIQUE_ID_x__887, __UNIQUE_ID_y__888) == 0 {
unsafe { __compiletime_assert_889() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v14 = if __UNIQUE_ID_x__887 < __UNIQUE_ID_y__888 { __UNIQUE_ID_x__887 } else { __UNIQUE_ID_y__888 };
let __v16: u64 = !(4194303u64 | BIT!(0i32).wrapping_sub(1) << (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_sub(2).wrapping_sub(0).wrapping_sub(0).wrapping_sub(0) | (UL!() << __v12).wrapping_sub(1) << (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_sub(2).wrapping_sub(0).wrapping_sub(0).wrapping_sub(0).wrapping_sub(__v14 as u64));
unsafe {
(*tail_page).flags.f = (*tail_page).flags.f & __v16;
}
}
}
}
if folio_test_anon(folio as *const folio) {
mod_mthp_stat(order as i32, 17u32, 1i32.wrapping_neg());
unsafe {
(*folio).__slate_anon_0.__slate_anon_0.mapping = core::ptr::null_mut::<address_space>();
}
}
if unlikely!(page_has_type(page as *const page)) != 0 {
unsafe {
(*page).__slate_anon_2.page_type = !0;
}
}
// Reset the page_type (which overlays _mapcount)
if is_check_pages_enabled() {
if free_page_is_bad(page) {
bad = bad.wrapping_add(1);
}
if bad != 0 {
return false;
}
}
page_cpupid_reset_last(page);
let mut __v24: i32;
let __UNIQUE_ID_x__890: i32 = 0;
let __UNIQUE_ID_y__891: i32 = 34i32;
loop {
if __types_ok__4!(__UNIQUE_ID_x__890, __UNIQUE_ID_y__891) == 0 {
unsafe { __compiletime_assert_892() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v24 = if __UNIQUE_ID_x__890 < __UNIQUE_ID_y__891 { __UNIQUE_ID_x__890 } else { __UNIQUE_ID_y__891 };
let mut __v26: i32;
let __UNIQUE_ID_x__893: i32 = 0;
let __UNIQUE_ID_y__894: i32 = 34i32;
loop {
if __types_ok__4!(__UNIQUE_ID_x__893, __UNIQUE_ID_y__894) == 0 {
unsafe { __compiletime_assert_895() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v26 = if __UNIQUE_ID_x__893 < __UNIQUE_ID_y__894 { __UNIQUE_ID_x__893 } else { __UNIQUE_ID_y__894 };
let __v28: u64 = !(4194303u64 | BIT!(0i32).wrapping_sub(1) << (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_sub(2).wrapping_sub(0).wrapping_sub(0).wrapping_sub(0) | (UL!() << __v24).wrapping_sub(1) << (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_sub(2).wrapping_sub(0).wrapping_sub(0).wrapping_sub(0).wrapping_sub(__v26 as u64));
unsafe {
(*page).flags.f = (*page).flags.f & __v28;
}
unsafe {
(*page).__slate_anon_1.__slate_anon_0.private = 0;
}
reset_page_owner(page, order as u16);
page_table_check_free(page, order);
pgalloc_tag_sub(page, (1i32 << order) as u32);
if PageHighMem(page as *const page) == 0 && fpi_flags & (BIT!(2i32) as i32) == 0 {
debug_check_no_locks_freed(page_address!(page) as *const (), _AC!() << 12i32 << order);
debug_check_no_obj_freed(page_address!(page) as *const (), _AC!() << 12i32 << order);
}
kernel_poison_pages(page, 1i32 << order);
// As memory initialization might be integrated into KASAN,
// KASAN poisoning and memory initialization code must be
// kept together to avoid discrepancies in behavior.
//
// With hardware tag-based KASAN, memory tags must be set before the
// page becomes unavailable via debug_pagealloc or arch_free_page.
if !skip_kasan_poison {
kasan_poison_pages(page, order, init);
// Memory is already initialized if KASAN did it internally.
if kasan_has_integrated_init() {
init = false;
}
}
if init {
clear_highpages_kasan_tagged(page, 1i32 << order);
}
// arch_free_page() can make the page's contents inaccessible.  s390
// does this.  So nothing which can access the page's contents should
// happen after this.
arch_free_page(page, order as i32);
debug_pagealloc_unmap_pages(page, 1i32 << order);
true
}

#[inline(always)]
fn __free_pages_prepare__const21(page: *mut page, order: u32, fpi_flags: i32) -> bool {
let mut bad: i32 = 0;
let skip_kasan_poison: bool = should_skip_kasan_poison(page);
let mut init: bool = want_init_on_free();
let compound: bool = PageCompound(page as *const page) != 0;
let folio: *mut folio = page_folio!(page);
if fpi_flags & (BIT!(3i32) as i32) != 0 {
return true;
}
VM_BUG_ON_PAGE!();
trace_mm_page_free(page, order);
kmsan_free_page(page, order);
let __v0: bool = memcg_kmem_online() && PageMemcgKmem(page);
if __v0 {
__memcg_kmem_uncharge_page(page, order as i32);
}
// In rare cases, when truncation or holepunching raced with
// munlock after VM_LOCKED was cleared, Mlocked may still be
// found set here.  This does not indicate a problem, unless
// "unevictable_pgs_cleared" appears worryingly large.
if unlikely!(folio_test_mlocked(folio as *const folio)) != 0 {
let nr_pages: i64 = folio_nr_pages(folio as *const folio) as i64;
__folio_clear_mlocked(folio);
zone_stat_mod_folio(folio, 8u32, nr_pages.wrapping_neg());
count_vm_events(58u32, nr_pages);
}
if unlikely__2!(PageHWPoison(page as *const page)) != 0 && order == 0 {
// Do not let hwpoison pages hit pcplists/buddy
reset_page_owner(page, order as u16);
page_table_check_free(page, order);
pgalloc_tag_sub(page, (1i32 << order) as u32);
// The page is isolated and accounted for.
// Mark the codetag as empty to avoid accounting error
// when the page is freed by unpoison_memory().
clear_page_tag_ref(page);
return false;
}
VM_BUG_ON_PAGE!();
// Check tail pages before head page information is cleared to
// avoid checking PageCompound for order-0 pages.
if unlikely__3!(order) != 0 {
if compound {
let __v1: *mut page = page.wrapping_offset(1 as isize);
let __v3: u64 = (unsafe { (*__v1).flags.f }) & 18446744073709485056u64;
unsafe {
(*__v1).flags.f = __v3;
}
}
for i in 1..(1i32 << order) {
'__slate_continue_9: {
let tail_page: *mut page = page.wrapping_offset(i as isize);
if compound {
let __v4: i32 = free_tail_page_prepare(page, tail_page);
bad = bad.wrapping_add(__v4);
}
if is_check_pages_enabled() {
if free_page_is_bad(tail_page) {
bad = bad.wrapping_add(1);
break '__slate_continue_9;
}
if (unsafe { (*tail_page).__slate_anon_1.__slate_anon_0.private }) != 0 {
bad_page(tail_page, c"nonzero private".as_ptr() as *const u8);
bad = bad.wrapping_add(1);
break '__slate_continue_9;
}
}
let mut __v12: i32;
let __UNIQUE_ID_x__884: i32 = 0;
let __UNIQUE_ID_y__885: i32 = 34i32;
loop {
if __types_ok__4!(__UNIQUE_ID_x__884, __UNIQUE_ID_y__885) == 0 {
unsafe { __compiletime_assert_886() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v12 = if __UNIQUE_ID_x__884 < __UNIQUE_ID_y__885 { __UNIQUE_ID_x__884 } else { __UNIQUE_ID_y__885 };
let mut __v14: i32;
let __UNIQUE_ID_x__887: i32 = 0;
let __UNIQUE_ID_y__888: i32 = 34i32;
loop {
if __types_ok__4!(__UNIQUE_ID_x__887, __UNIQUE_ID_y__888) == 0 {
unsafe { __compiletime_assert_889() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v14 = if __UNIQUE_ID_x__887 < __UNIQUE_ID_y__888 { __UNIQUE_ID_x__887 } else { __UNIQUE_ID_y__888 };
let __v16: u64 = !(4194303u64 | BIT!(0i32).wrapping_sub(1) << (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_sub(2).wrapping_sub(0).wrapping_sub(0).wrapping_sub(0) | (UL!() << __v12).wrapping_sub(1) << (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_sub(2).wrapping_sub(0).wrapping_sub(0).wrapping_sub(0).wrapping_sub(__v14 as u64));
unsafe {
(*tail_page).flags.f = (*tail_page).flags.f & __v16;
}
}
}
}
if folio_test_anon(folio as *const folio) {
mod_mthp_stat(order as i32, 17u32, 1i32.wrapping_neg());
unsafe {
(*folio).__slate_anon_0.__slate_anon_0.mapping = core::ptr::null_mut::<address_space>();
}
}
if unlikely!(page_has_type(page as *const page)) != 0 {
unsafe {
(*page).__slate_anon_2.page_type = !0;
}
}
// Reset the page_type (which overlays _mapcount)
if is_check_pages_enabled() {
if free_page_is_bad(page) {
bad = bad.wrapping_add(1);
}
if bad != 0 {
return false;
}
}
page_cpupid_reset_last(page);
let mut __v24: i32;
let __UNIQUE_ID_x__890: i32 = 0;
let __UNIQUE_ID_y__891: i32 = 34i32;
loop {
if __types_ok__4!(__UNIQUE_ID_x__890, __UNIQUE_ID_y__891) == 0 {
unsafe { __compiletime_assert_892() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v24 = if __UNIQUE_ID_x__890 < __UNIQUE_ID_y__891 { __UNIQUE_ID_x__890 } else { __UNIQUE_ID_y__891 };
let mut __v26: i32;
let __UNIQUE_ID_x__893: i32 = 0;
let __UNIQUE_ID_y__894: i32 = 34i32;
loop {
if __types_ok__4!(__UNIQUE_ID_x__893, __UNIQUE_ID_y__894) == 0 {
unsafe { __compiletime_assert_895() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v26 = if __UNIQUE_ID_x__893 < __UNIQUE_ID_y__894 { __UNIQUE_ID_x__893 } else { __UNIQUE_ID_y__894 };
let __v28: u64 = !(4194303u64 | BIT!(0i32).wrapping_sub(1) << (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_sub(2).wrapping_sub(0).wrapping_sub(0).wrapping_sub(0) | (UL!() << __v24).wrapping_sub(1) << (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_sub(2).wrapping_sub(0).wrapping_sub(0).wrapping_sub(0).wrapping_sub(__v26 as u64));
unsafe {
(*page).flags.f = (*page).flags.f & __v28;
}
unsafe {
(*page).__slate_anon_1.__slate_anon_0.private = 0;
}
reset_page_owner(page, order as u16);
page_table_check_free(page, order);
pgalloc_tag_sub(page, (1i32 << order) as u32);
if PageHighMem(page as *const page) == 0 && fpi_flags & (BIT!(2i32) as i32) == 0 {
debug_check_no_locks_freed(page_address!(page) as *const (), _AC!() << 12i32 << order);
debug_check_no_obj_freed(page_address!(page) as *const (), _AC!() << 12i32 << order);
}
kernel_poison_pages(page, 1i32 << order);
// As memory initialization might be integrated into KASAN,
// KASAN poisoning and memory initialization code must be
// kept together to avoid discrepancies in behavior.
//
// With hardware tag-based KASAN, memory tags must be set before the
// page becomes unavailable via debug_pagealloc or arch_free_page.
if !skip_kasan_poison {
kasan_poison_pages(page, order, init);
// Memory is already initialized if KASAN did it internally.
if kasan_has_integrated_init() {
init = false;
}
}
if init {
clear_highpages_kasan_tagged(page, 1i32 << order);
}
// arch_free_page() can make the page's contents inaccessible.  s390
// does this.  So nothing which can access the page's contents should
// happen after this.
arch_free_page(page, order as i32);
debug_pagealloc_unmap_pages(page, 1i32 << order);
true
}

#[unsafe(no_mangle)]
extern "C" fn free_pages_prepare(page: *mut page, order: u32) -> bool {
__free_pages_prepare__const21(page, order, 0i32)
}

// Frees a number of pages from the PCP lists
// Assumes all pages on list are in same zone.
// count is the number of pages to free.
fn free_pcppages_bulk(zone: *mut zone, mut count: i32, pcp: *mut per_cpu_pages, mut pindex: i32) {
let mut order: u32 = 0;
let mut page: *mut page = unsafe { core::mem::zeroed() };
// Ensure proper count is passed which otherwise would stuck in the
// below while (list_empty(list)) loop.
let mut __v0: i32;
let __UNIQUE_ID_x__896: i32 = unsafe { (*pcp).count };
let __UNIQUE_ID_y__897: i32 = count;
loop {
if __types_ok__4!(__UNIQUE_ID_x__896, __UNIQUE_ID_y__897) == 0 {
unsafe { __compiletime_assert_898() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__896 < __UNIQUE_ID_y__897 { __UNIQUE_ID_x__896 } else { __UNIQUE_ID_y__897 };
count = __v0;
// Ensure requested pindex is drained first.
pindex = pindex.wrapping_sub(1);
let mut __UNIQUE_ID_guard_899: __SlateRecord_spinlock_h_642 = class_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!((*zone).lock) });
let __slate_cleanup_ptr_2 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_899) as *mut _;
let __slate_cleanup_2 = ::slate_support::Guard(&mut || class_spinlock_irqsave_destructor(__slate_cleanup_ptr_2 as *mut _));
let mut __UNIQUE_ID_unlock_900: *mut __SlateRecord_spinlock_h_642 = ((unsafe { core::ptr::addr_of_mut!((*zone).lock) }) as u64) as *mut __SlateRecord_spinlock_h_642;
let __slate_cleanup_ptr_3 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_900) as *mut _;
let __slate_cleanup_3 = ::slate_support::Guard(&mut || __class_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_3 as *mut _));
while count > 0 {
let mut list: *mut list_head = unsafe { core::mem::zeroed() };
let mut nr_pages: i32 = 0;
// Remove pages from lists in a round-robin fashion.
loop {
let __v5: i32 = pindex.wrapping_add(1);
pindex = __v5;
if __v5 > 11i32 {
pindex = 0;
}
list = unsafe { (*pcp).lists.as_mut_ptr() as *mut list_head }.wrapping_offset(pindex as isize);
if list_empty(list as *const list_head) == 0 {
break;
}
}
order = pindex_to_order(pindex as u32) as u32;
nr_pages = 1i32 << order;
loop {
let mut pfn: u64 = 0;
let mut mt: i32 = 0;
page = list_last_entry__6!(list, page);
pfn = ((unsafe { page.offset_from(vmemmap_base as *mut page) }) as i64) as u64;
mt = (get_pfnblock_migratetype(page as *const page, pfn) as u32) as i32;
// must delete to avoid corrupting pcp list
list_del(unsafe { core::ptr::addr_of_mut!((*page).__slate_anon_1.__slate_anon_0.__slate_anon_0.pcp_list) });
count = count.wrapping_sub(nr_pages);
let __v11: i32 = unsafe { (*pcp).count };
unsafe {
(*pcp).count = __v11.wrapping_sub(nr_pages);
}
__free_one_page(page, pfn, zone, order, mt, 0i32);
trace_mm_page_pcpu_drain(page, order, mt);
let __v14: bool = count > 0 && list_empty(list as *const list_head) == 0;
if !__v14 {
break;
}
}
}
}

/// Split a multi-block free page into its individual pageblocks.
fn split_large_buddy(zone: *mut zone, mut page: *mut page, mut pfn: u64, mut order: i32, fpi: i32) {
let end: u64 = pfn.wrapping_add((1i32 << order) as u64);
VM_WARN_ON_ONCE!();
// Caller removed page from freelist, buddy info cleared!
VM_WARN_ON_ONCE!();
if (order as u32) > MIN_T!(u32, 9i32) {
order = MIN_T!(u32, 9i32) as i32;
}
loop {
let mt: i32 = (get_pfnblock_migratetype(page as *const page, pfn) as u32) as i32;
__free_one_page(page, pfn, zone, order as u32, mt, fpi);
pfn = pfn.wrapping_add((1i32 << order) as u64);
if pfn == end {
break;
}
page = ((unsafe { vmemmap_base }) as *mut page).wrapping_offset(pfn as isize);
if false {
break;
}
}
}

fn add_page_to_zone_llist(zone: *mut zone, page: *mut page, order: u32) {
// Remember the order
unsafe {
(*page).__slate_anon_1.__slate_anon_0.private = order as u64;
}
// Add the page to the free list
llist_add(unsafe { core::ptr::addr_of_mut!((*page).__slate_anon_1.__slate_anon_0.__slate_anon_0.pcp_llist) }, unsafe { core::ptr::addr_of_mut!((*zone).trylock_free_pages) });
}

fn free_one_page(zone: *mut zone, page: *mut page, pfn: u64, order: u32, fpi_flags: i32) {
let mut flags: u64 = 0;
if unlikely__2!(fpi_flags & (BIT!(2i32) as i32)) != 0 {
let __v0: bool = !can_spin_trylock() || !spin_trylock_irqsave!(unsafe { core::ptr::addr_of_mut!((*zone).lock) }, flags);
if __v0 {
add_page_to_zone_llist(zone, page, order);
return;
}
} else {
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*zone).lock) }, flags);
}
// The lock succeeded. Process deferred pages.
let llhead: *mut llist_head = unsafe { core::ptr::addr_of_mut!((*zone).trylock_free_pages) };
if unlikely!(!llist_empty(llhead as *const llist_head) && fpi_flags & (BIT!(2i32) as i32) == 0) != 0 {
let mut p: *mut page = unsafe { core::mem::zeroed() };
let mut tmp: *mut page = unsafe { core::mem::zeroed() };
let llnode: *mut llist_node = llist_del_all(llhead);
p = llist_entry__4!(llnode);
loop {
let __v7: bool = if member_address_is_nonnull__3!(p) {
tmp = llist_entry__4!(unsafe { (*p).__slate_anon_1.__slate_anon_0.__slate_anon_0.pcp_llist.next });
true
} else {
false
};
if !__v7 {
break;
}
let p_order: u32 = (unsafe { (*p).__slate_anon_1.__slate_anon_0.private }) as u32;
split_large_buddy(zone, p, ((unsafe { p.offset_from(vmemmap_base as *mut page) }) as i64) as u64, p_order as i32, fpi_flags);
__count_vm_events(16u32, (1i32 << p_order) as i64);
p = tmp;
}
}
split_large_buddy(zone, page, pfn, order as i32, fpi_flags);
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*zone).lock) }, flags);
__count_vm_events(16u32, (1i32 << order) as i64);
}

fn __free_pages_ok(page: *mut page, order: u32, fpi_flags: i32) {
let pfn: u64 = ((unsafe { page.offset_from(vmemmap_base as *mut page) }) as i64) as u64;
let zone: *mut zone = page_zone(page as *const page);
if __free_pages_prepare(page, order, fpi_flags) {
free_one_page(zone, page, pfn, order, fpi_flags);
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn __free_pages_core(page: *mut page, order: u32, context: u32) {
let nr_pages: u32 = (1i32 << order) as u32;
let mut p: *mut page = page;
// When initializing the memmap, __init_single_page() sets the refcount
// of all pages to 1 ("allocated"/"not free"). We have to set the
// refcount of all involved pages to 0.
//
// Note that hotplugged memory pages are initialized to PageOffline().
// Pages freed from memblock might be marked as reserved.
let mut r#loop: u32 = 0;
while r#loop < nr_pages {
__ClearPageReserved(p);
set_page_count(p, 0i32);
r#loop = r#loop.wrapping_add(1);
let __v3: *mut page = p;
p = __v3.wrapping_offset(1 as isize);
}
// memblock adjusts totalram_pages() manually.
atomic_long_add(nr_pages as i64, unsafe { core::ptr::addr_of_mut!((*page_zone(page as *const page)).managed_pages) });
if page_contains_unaccepted(page, order) {
let __v5: bool = order == 10 && __free_unaccepted(page);
if __v5 {
return;
}
accept_memory(page_to_phys!(page), _AC!() << 12i32 << order);
}
// Bypass PCP and place fresh pages right to the tail, primarily
// relevant for memory onlining.
__free_pages_ok(page, order, BIT!(1i32) as i32);
}

/// Check that the whole (or subset of) a pageblock given by the interval of
/// [start_pfn, end_pfn) is valid and within the same zone, before scanning it
/// with the migration of free compaction scanner.
///
/// Return struct page pointer of start_pfn, or NULL if checks were not passed.
///
/// It's possible on some configurations to have a setup like node0 node1 node0
/// i.e. it's possible that all pages within a zones range of pages do not
/// belong to a single zone. We assume that a border between node0 and node1
/// can occur within a single pageblock, but not a node0 node1 node0
/// interleaving within a single pageblock. It is therefore sufficient to check
/// the first and last page of a pageblock and avoid checking each individual
/// page in a pageblock.
///
/// Note: the function may return non-NULL struct page even for a page block
/// which contains a memory hole (i.e. there is no physical memory for a subset
/// of the pfn range). For example, if the pageblock order is MAX_PAGE_ORDER, which
/// will fall into 2 sub-sections, and the end pfn of the pageblock may be hole
/// even though the start pfn is online and valid. This should be safe most of
/// the time because struct pages are still initialized via init_unavailable_range()
/// and pfn walkers shouldn't touch any physical memory range for which they do
/// not recognize any specific metadata in struct pages.
#[unsafe(no_mangle)]
extern "C" fn __pageblock_pfn_to_page(start_pfn: u64, mut end_pfn: u64, zone: *mut zone) -> *mut page {
// end_pfn is one past the range we are checking
end_pfn = end_pfn.wrapping_sub(1);
if pfn_valid(end_pfn) == 0 {
return core::ptr::null_mut::<page>();
}
let start_page: *mut page = pfn_to_online_page!(start_pfn);
if start_page.is_null() {
return core::ptr::null_mut::<page>();
}
if page_zone(start_page as *const page) != zone {
return core::ptr::null_mut::<page>();
}
let end_page: *mut page = ((unsafe { vmemmap_base }) as *mut page).wrapping_offset(end_pfn as isize);
// This gives a shorter code than deriving page_zone(end_page)
if page_zone_id(start_page) != page_zone_id(end_page) {
return core::ptr::null_mut::<page>();
}
start_page
}

/// The order of subdivision here is critical for the IO subsystem.
/// Please do not alter this order without good reasons and regression
/// testing. Specifically, as large blocks of memory are subdivided,
/// the order in which smaller blocks are delivered depends on the order
/// they're subdivided in this function. This is the primary factor
/// influencing the order in which pages are delivered to the IO
/// subsystem according to empirical testing, and this is also justified
/// by considering the behavior of a buddy system containing a single
/// large block of memory acted on by a series of small allocations.
/// This behavior is a critical factor in sglist merging's success.
///
/// -- nyc
fn expand(zone: *mut zone, page: *mut page, low: i32, mut high: i32, migratetype: i32) -> u32 {
let mut size: u32 = (1i32 << high) as u32;
let mut nr_added: u32 = 0;
while high > low {
high = high.wrapping_sub(1);
size >>= 1i32;
VM_BUG_ON_PAGE!();
// Mark as guard pages (or page), that will allow to
// merge back to allocator when buddy will be freed.
// Corresponding page table entries will not be touched,
// pages will stay not present in virtual address space
if !set_page_guard(zone, page.wrapping_offset(size as isize), high as u32) {
__add_to_free_list(page.wrapping_offset(size as isize), zone, high as u32, migratetype, false);
set_buddy_order(page.wrapping_offset(size as isize), high as u32);
nr_added = nr_added.wrapping_add(size);
}
}
nr_added
}

#[inline(always)]
fn page_del_and_expand(zone: *mut zone, page: *mut page, low: i32, high: i32, migratetype: i32) {
let mut nr_pages: i32 = 1i32 << high;
__del_page_from_free_list(page, zone, high as u32, migratetype);
let __v0: u32 = expand(zone, page, low, high, migratetype);
nr_pages = (nr_pages as u32).wrapping_sub(__v0) as i32;
account_freepages(zone, nr_pages.wrapping_neg(), migratetype);
}

fn check_new_page_bad(page: *mut page) {
if unlikely__2!(PageHWPoison(page as *const page)) != 0 {
// Don't complain about hwpoisoned pages
if PageBuddy(page as *const page) != 0 {
__ClearPageBuddy(page);
}
return;
}
let mut __v0: i32;
let __UNIQUE_ID_x__901: i32 = 0;
let __UNIQUE_ID_y__902: i32 = 34i32;
loop {
if __types_ok!(__UNIQUE_ID_x__901, __UNIQUE_ID_y__902) == 0 {
unsafe { __compiletime_assert_903() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__901 < __UNIQUE_ID_y__902 { __UNIQUE_ID_x__901 } else { __UNIQUE_ID_y__902 };
let mut __v2: i32;
let __UNIQUE_ID_x__904: i32 = 0;
let __UNIQUE_ID_y__905: i32 = 34i32;
loop {
if __types_ok!(__UNIQUE_ID_x__904, __UNIQUE_ID_y__905) == 0 {
unsafe { __compiletime_assert_906() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = if __UNIQUE_ID_x__904 < __UNIQUE_ID_y__905 { __UNIQUE_ID_x__904 } else { __UNIQUE_ID_y__905 };
bad_page(page, page_bad_reason(page, 4194303u64 | BIT!(0i32).wrapping_sub(1) << (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_sub(2).wrapping_sub(0).wrapping_sub(0).wrapping_sub(0) | (UL!() << __v0).wrapping_sub(1) << (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_sub(2).wrapping_sub(0).wrapping_sub(0).wrapping_sub(0).wrapping_sub(__v2 as u64)));
}

// This page is about to be returned from the page allocator
fn check_new_page(page: *mut page) -> bool {
let mut __v0: i32;
let __UNIQUE_ID_x__907: i32 = 0;
let __UNIQUE_ID_y__908: i32 = 34i32;
loop {
if __types_ok!(__UNIQUE_ID_x__907, __UNIQUE_ID_y__908) == 0 {
unsafe { __compiletime_assert_909() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__907 < __UNIQUE_ID_y__908 { __UNIQUE_ID_x__907 } else { __UNIQUE_ID_y__908 };
let mut __v2: i32;
let __UNIQUE_ID_x__910: i32 = 0;
let __UNIQUE_ID_y__911: i32 = 34i32;
loop {
if __types_ok!(__UNIQUE_ID_x__910, __UNIQUE_ID_y__911) == 0 {
unsafe { __compiletime_assert_912() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = if __UNIQUE_ID_x__910 < __UNIQUE_ID_y__911 { __UNIQUE_ID_x__910 } else { __UNIQUE_ID_y__911 };
if (!(!page_expected_state(page, 4194303u64 | BIT!(0i32).wrapping_sub(1) << (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_sub(2).wrapping_sub(0).wrapping_sub(0).wrapping_sub(0) | (UL!() << __v0).wrapping_sub(1) << (core::mem::size_of::<u64>() as u64).wrapping_mul(8).wrapping_sub(0).wrapping_sub(6).wrapping_sub(2).wrapping_sub(0).wrapping_sub(0).wrapping_sub(0).wrapping_sub(__v2 as u64) | 0)) as i64) != 0 {
return false;
}
check_new_page_bad(page);
true
}

fn check_new_pages(page: *mut page, order: u32) -> bool {
if is_check_pages_enabled() {
for i in 0..(1i32 << order) {
let p: *mut page = page.wrapping_offset(i as isize);
if check_new_page(p) {
return true;
}
}
}
false
}

fn should_skip_kasan_unpoison(flags: u32) -> bool {
// Don't skip if a software KASAN mode is enabled.
if IS_ENABLED!() != 0 || IS_ENABLED!() != 0 {
return false;
}
// Skip, if hardware tag-based KASAN is not enabled.
if !kasan_hw_tags_enabled() {
return true;
}
// With hardware tag-based KASAN enabled, skip if this has been
// requested via __GFP_SKIP_KASAN.
flags & 0 != 0
}

fn should_skip_init(flags: u32) -> bool {
// Don't skip, if hardware tag-based KASAN is not enabled.
if !kasan_hw_tags_enabled() {
return false;
}
// For hardware tag-based KASAN, skip if requested.
flags & 0 != 0
}

#[unsafe(no_mangle)]
extern "C" fn post_alloc_hook(page: *mut page, order: u32, gfp_flags: u32, alloc_flags: u32) {
let zero_tags: bool = gfp_flags & (BIT!(23i32) as u32) != 0;
let mut init: bool = false;
let __v0: bool = !want_init_on_free() && want_init_on_alloc(gfp_flags);
init = __v0 && !should_skip_init(gfp_flags);
let mut i: i32 = 0;
set_page_private(page, 0u64);
arch_alloc_page(page, order as i32);
debug_pagealloc_map_pages(page, 1i32 << order);
// Page unpoisoning must happen before memory initialization.
// Otherwise, the poison pattern will be overwritten for __GFP_ZERO
// allocations and the page unpoisoning code will complain.
kernel_unpoison_pages(page, 1i32 << order);
// As memory initialization might be integrated into KASAN,
// KASAN unpoisoning and memory initialization code must be
// kept together to avoid discrepancies in behavior.
// Clearing tags can efficiently clear the memory for us as well, if
// required.
if zero_tags {
init = tag_clear_highpages(page, 1i32 << order, init);
}
// clear_pages=
let __v2: bool = !should_skip_kasan_unpoison(gfp_flags) && kasan_unpoison_pages(page, order, init);
if __v2 {
// Take note that memory was initialized by KASAN.
if kasan_has_integrated_init() {
init = false;
}
} else {
// If memory tags have not been set by KASAN, reset the page
// tags to ensure page_address() dereferencing does not fault.
i = 0;
while i != 1i32 << order {
page_kasan_tag_reset(page.wrapping_offset(i as isize));
i = i.wrapping_add(1);
}
}
// If memory is still not initialized, initialize it now.
if init {
clear_highpages_kasan_tagged(page, 1i32 << order);
}
set_page_owner(page, order as u16, gfp_flags);
page_table_check_alloc(page, order);
pgalloc_tag_add(page, get_current(), (1i32 << order) as u32, alloc_flags);
}

fn prep_new_page(page: *mut page, order: u32, gfp_flags: u32, alloc_flags: u32) {
post_alloc_hook(page, order, gfp_flags, alloc_flags);
if order != 0 && gfp_flags & (BIT!(18i32) as u32) != 0 {
prep_compound_page(page, order);
}
// page is set pfmemalloc when ALLOC_NO_WATERMARKS was necessary to
// allocate the page. The expectation is that the caller is taking
// steps that will free more memory. The caller should avoid the page
// being used for !PFMEMALLOC purposes.
if alloc_flags & 4 != 0 {
set_page_pfmemalloc(page);
} else {
clear_page_pfmemalloc(page);
}
}

/// Go through the free lists for the given migratetype and remove
/// the smallest available page from the freelists
#[inline(always)]
fn __rmqueue_smallest(zone: *mut zone, order: u32, migratetype: i32) -> *mut page {
let mut current_order: u32 = 0;
// Find a page of the appropriate size in the preferred list
current_order = order;
while current_order < 11u32 {
let area: *mut free_area = unsafe { (*zone).free_area.as_mut_ptr() as *mut free_area }.wrapping_offset(current_order as isize);
let page: *mut page = get_page_from_free_area(area, migratetype);
if !page.is_null() {
page_del_and_expand(zone, page, order as i32, current_order as i32, migratetype);
trace_mm_page_alloc_zone_locked(page, order, migratetype, (pcp_allowed_order(order) && migratetype < 3) as i32);
return page;
}
current_order = current_order.wrapping_add(1);
}
core::ptr::null_mut::<page>()
}

/// Go through the free lists for the given migratetype and remove
/// the smallest available page from the freelists
#[inline(always)]
fn __rmqueue_smallest__const24(zone: *mut zone, order: u32, migratetype: i32) -> *mut page {
let mut current_order: u32 = 0;
// Find a page of the appropriate size in the preferred list
current_order = order;
while current_order < 11u32 {
let area: *mut free_area = unsafe { (*zone).free_area.as_mut_ptr() as *mut free_area }.wrapping_offset(current_order as isize);
let page: *mut page = get_page_from_free_area(area, migratetype);
if !page.is_null() {
page_del_and_expand(zone, page, order as i32, current_order as i32, migratetype);
trace_mm_page_alloc_zone_locked(page, order, migratetype, (pcp_allowed_order(order) && migratetype < 3) as i32);
return page;
}
current_order = current_order.wrapping_add(1);
}
core::ptr::null_mut::<page>()
}

/// This array describes the order lists are fallen back to when
/// the free lists for the desirable migrate type are depleted
///
/// The other migratetypes do not have fallbacks.
static mut fallbacks: __SlateAlign16<[[i32; 2]; 3]> = __SlateAlign16([[2i32, 1i32], [2i32, 0i32], [0i32, 1i32]]);

fn __rmqueue_cma_fallback(zone: *mut zone, order: u32) -> *mut page {
core::ptr::null_mut::<page>()
}

/// Move all free pages of a block to new type's freelist. Caller needs to
/// change the block type.
fn __move_freepages_block(zone: *mut zone, start_pfn: u64, old_mt: i32, new_mt: i32) -> i32 {
let mut pfn: u64 = 0;
let mut end_pfn: u64 = 0;
let mut order: u32 = 0;
let mut pages_moved: i32 = 0;
VM_WARN_ON!();
end_pfn = pageblock_end_pfn__2!(start_pfn);
pfn = start_pfn;
while pfn < end_pfn {
let page: *mut page = ((unsafe { vmemmap_base }) as *mut page).wrapping_offset(pfn as isize);
if PageBuddy(page as *const page) == 0 {
pfn = pfn.wrapping_add(1);
} else {
// Make sure we are not inadvertently changing nodes
VM_BUG_ON_PAGE!();
VM_BUG_ON_PAGE!();
order = buddy_order(page);
move_to_free_list(page, zone, order, old_mt, new_mt);
pfn = pfn.wrapping_add((1i32 << order) as u64);
pages_moved = pages_moved.wrapping_add(1i32 << order);
}
}
pages_moved
}

fn prep_move_freepages_block(zone: *mut zone, mut page: *mut page, start_pfn: &mut u64, num_free: *mut i32, num_movable: *mut i32) -> bool {
let mut pfn: u64 = ((unsafe { page.offset_from(vmemmap_base as *mut page) }) as i64) as u64;
let start: u64 = pageblock_start_pfn__2!(pfn);
let end: u64 = pageblock_end_pfn__2!(pfn);
// The caller only has the lock for @zone, don't touch ranges
// that straddle into other zones. While we could move part of
// the range that's inside the zone, this call is usually
// accompanied by other operations such as migratetype updates
// which also should be locked.
if !zone_spans_pfn(zone as *const zone, start) {
return false;
}
if !zone_spans_pfn(zone as *const zone, end.wrapping_sub(1)) {
return false;
}
unsafe {
*(start_pfn as *mut u64) = start;
}
if !num_free.is_null() {
unsafe {
*num_free = 0;
}
unsafe {
*num_movable = 0;
}
pfn = start;
while pfn < end {
page = ((unsafe { vmemmap_base }) as *mut page).wrapping_offset(pfn as isize);
if PageBuddy(page as *const page) != 0 {
let nr: i32 = 1i32 << buddy_order(page);
let __v1: i32 = unsafe { *num_free };
unsafe {
*num_free = __v1.wrapping_add(nr);
}
pfn = pfn.wrapping_add(nr as u64);
} else {
// We assume that pages that could be isolated for
// migration are movable. But we don't actually try
// isolating, as that would be expensive.
let __v6: bool = PageLRU(page as *const page) != 0 || page_has_movable_ops(page as *const page);
if __v6 {
let __v8: i32 = unsafe { *num_movable };
unsafe {
*num_movable = __v8.wrapping_add(1);
}
}
pfn = pfn.wrapping_add(1);
}
}
}
true
}

fn move_freepages_block(zone: *mut zone, page: *mut page, old_mt: i32, new_mt: i32) -> i32 {
let mut start_pfn: u64 = 0;
let mut res: i32 = 0;
if !prep_move_freepages_block(zone, page, unsafe { &mut (*core::ptr::addr_of_mut!(start_pfn)) }, core::ptr::null_mut::<i32>(), core::ptr::null_mut::<i32>()) {
return 1i32.wrapping_neg();
}
res = __move_freepages_block(zone, start_pfn, old_mt, new_mt);
set_pageblock_migratetype(((unsafe { vmemmap_base }) as *mut page).wrapping_offset(start_pfn as isize), new_mt as u32);
res
}

fn boost_watermark(zone: *mut zone) -> bool {
let mut max_boost: u64 = 0;
if (unsafe { watermark_boost_factor }) == 0 {
return false;
}
// Don't bother in zones that are unlikely to produce results.
// On small machines, including kdump capture kernels running
// in a small area, boosting the watermark can cause an out of
// memory situation immediately.
if (1u64 << MIN_T__2!(u32, 9i32, 10i32)).wrapping_mul(4) > zone_managed_pages(zone as *const zone) {
return false;
}
max_boost = mult_frac!(unsafe { *((*zone)._watermark.as_mut_ptr() as *mut u64).wrapping_offset(2 as isize) }, unsafe { watermark_boost_factor }, 10000i32);
// high watermark may be uninitialised if fragmentation occurs
// very early in boot so do not boost. We do not fall
// through and boost by pageblock_nr_pages as failing
// allocations that early means that reclaim is not going
// to help and it may even be impossible to reclaim the
// boosted watermark resulting in a hang.
if max_boost == 0 {
return false;
}
let mut __v6: u64;
let __UNIQUE_ID_x__913: u64 = 1u64 << MIN_T__2!(u32, 9i32, 10i32);
let __UNIQUE_ID_y__914: u64 = max_boost;
loop {
if __types_ok__2!(__UNIQUE_ID_x__913, __UNIQUE_ID_y__914) == 0 {
unsafe { __compiletime_assert_915() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v6 = if __UNIQUE_ID_x__913 > __UNIQUE_ID_y__914 { __UNIQUE_ID_x__913 } else { __UNIQUE_ID_y__914 };
max_boost = __v6;
let mut __v8: u64;
let __UNIQUE_ID_x__916: u64 = unsafe { (*zone).watermark_boost }.wrapping_add(1u64 << MIN_T__2!(u32, 9i32, 10i32));
let __UNIQUE_ID_y__917: u64 = max_boost;
loop {
if __types_ok__2!(__UNIQUE_ID_x__916, __UNIQUE_ID_y__917) == 0 {
unsafe { __compiletime_assert_918() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v8 = if __UNIQUE_ID_x__916 < __UNIQUE_ID_y__917 { __UNIQUE_ID_x__916 } else { __UNIQUE_ID_y__917 };
unsafe {
(*zone).watermark_boost = __v8;
}
true
}

/// When we are falling back to another migratetype during allocation, should we
/// try to claim an entire block to satisfy further allocations, instead of
/// polluting multiple pageblocks?
fn should_try_claim_block(order: u32, start_mt: i32) -> bool {
// Leaving this order check is intended, although there is
// relaxed order check in next check. The reason is that
// we can actually claim the whole pageblock if this condition met,
// but, below check doesn't guarantee it and that is just heuristic
// so could be changed anytime.
if order >= MIN_T__2!(u32, 9i32, 10i32) {
return true;
}
// Above a certain threshold, always try to claim, as it's likely there
// will be more free pages in the pageblock.
if order >= MIN_T__2!(u32, 9i32, 10i32) / 2 {
return true;
}
// Unmovable/reclaimable allocations would cause permanent
// fragmentations if they fell back to allocating from a movable block
// (polluting it), so we try to claim the whole block regardless of the
// allocation size. Later movable allocations can always steal from this
// block, which is less problematic.
if start_mt == 2 || start_mt == 0 {
return true;
}
if (unsafe { page_group_by_mobility_disabled }) != 0 {
return true;
}
// Movable pages won't cause permanent fragmentation, so when you alloc
// small pages, we just need to temporarily steal unmovable or
// reclaimable pages that are closest to the request size. After a
// while, memory compaction may occur to form large contiguous pages,
// and the next movable allocation may not need to steal.
false
}

/// Check whether there is a suitable fallback freepage with requested order.
/// If claimable is true, this function returns fallback_mt only if
/// we would do this whole-block claiming. This would help to reduce
/// fragmentation due to mixed migratetype pages in one pageblock.
#[unsafe(no_mangle)]
extern "C" fn find_suitable_fallback(area: *mut free_area, order: u32, migratetype: i32, claimable: bool, mt_out: *mut i32) -> u32 {
let mut i: i32 = 0;
let __v0: bool = claimable && !should_try_claim_block(order, migratetype);
if __v0 {
return 2;
}
if (unsafe { (*area).nr_free }) == 0 {
return 1;
}
while i < 2i32 {
let fallback_mt: i32 = unsafe { fallbacks.0[migratetype as usize][i as usize] };
if !free_area_empty(area, fallback_mt) {
if !mt_out.is_null() {
unsafe {
*mt_out = fallback_mt;
}
}
return 0;
}
i = i.wrapping_add(1);
}
1
}

/// This function implements actual block claiming behaviour. If order is large
/// enough, we can claim the whole pageblock for the requested migratetype. If
/// not, we check the pageblock for constituent pages; if at least half of the
/// pages are free or compatible, we can still claim the whole block, so pages
/// freed in the future will be put on the correct free list.
fn try_to_claim_block(zone: *mut zone, page: *mut page, current_order: i32, order: i32, start_type: i32, block_type: i32, alloc_flags: u32) -> *mut page {
let mut free_pages: i32 = 0;
let mut movable_pages: i32 = 0;
let mut alike_pages: i32 = 0;
let mut start_pfn: u64 = 0;
// Take ownership for orders >= pageblock_order
if (current_order as u32) >= MIN_T!(u32, 9i32) {
let mut nr_added: u32 = 0;
del_page_from_free_list(page, zone, current_order as u32, block_type);
change_pageblock_range(page, current_order, start_type);
nr_added = expand(zone, page, order, current_order, start_type);
account_freepages(zone, nr_added as i32, start_type);
return page;
}
// Boost watermarks to increase reclaim pressure to reduce the
// likelihood of future fallbacks. Wake kswapd now as the node
// may be balanced overall and kswapd will not wake naturally.
if boost_watermark(zone) && alloc_flags & 2048 != 0 {
set_bit(0i64, (unsafe { core::ptr::addr_of_mut!((*zone).flags) }) as *mut u64);
}
// moving whole block can fail due to zone boundary conditions
if !prep_move_freepages_block(zone, page, unsafe { &mut (*core::ptr::addr_of_mut!(start_pfn)) }, core::ptr::addr_of_mut!(free_pages), core::ptr::addr_of_mut!(movable_pages)) {
return core::ptr::null_mut::<page>();
}
// Determine how many pages are compatible with our allocation.
// For movable allocation, it's the number of movable pages which
// we just obtained. For other types it's a bit more tricky.
if start_type == 1 {
alike_pages = movable_pages;
} else {
// If we are falling back a RECLAIMABLE or UNMOVABLE allocation
// to MOVABLE pageblock, consider all non-movable pages as
// compatible. If it's UNMOVABLE falling back to RECLAIMABLE or
// vice versa, be conservative since we can't distinguish the
// exact migratetype of non-movable pages.
if block_type == 1 {
alike_pages = (1u64 << MIN_T!(u32, 9i32)).wrapping_sub(free_pages.wrapping_add(movable_pages) as u64) as i32;
} else {
alike_pages = 0;
}
}
// If a sufficient number of pages in the block are either free or of
// compatible migratability as our allocation, claim the whole block.
if free_pages.wrapping_add(alike_pages) >= 1i32 << MIN_T!(u32, 9i32).wrapping_sub(1) || (unsafe { page_group_by_mobility_disabled }) != 0 {
__move_freepages_block(zone, start_pfn, block_type, start_type);
set_pageblock_migratetype(((unsafe { vmemmap_base }) as *mut page).wrapping_offset(start_pfn as isize), start_type as u32);
return __rmqueue_smallest(zone, order as u32, start_type);
}
core::ptr::null_mut::<page>()
}

/// Try to allocate from some fallback migratetype by claiming the entire block,
/// i.e. converting it to the allocation's start migratetype.
///
/// The use of signed ints for order and current_order is a deliberate
/// deviation from the rest of this file, to make the for loop
/// condition simpler.
#[inline(always)]
fn __rmqueue_claim(zone: *mut zone, order: i32, start_migratetype: i32, alloc_flags: u32) -> *mut page {
// Find the largest available free page in the other list. This roughly
// approximates finding the pageblock with the most free pages, which
// would be too costly to do exactly.
let mut current_order: i32 = 10;
let mut min_order: i32 = order;
let mut page: *mut page = unsafe { core::mem::zeroed() };
let mut fallback_mt: i32 = 0;
// Do not steal pages from freelists belonging to other pageblocks
// i.e. orders < pageblock_order. If there are no local zones free,
// the zonelists will be reiterated without ALLOC_NOFRAGMENT.
if (order as u32) < MIN_T!(u32, 9i32) && alloc_flags & 256 != 0 {
min_order = MIN_T!(u32, 9i32) as i32;
}
while current_order >= min_order {
let mut result: u32 = 0;
let area: *mut free_area = unsafe { (*zone).free_area.as_mut_ptr() as *mut free_area }.wrapping_offset(current_order as isize);
result = find_suitable_fallback(area, current_order as u32, start_migratetype, true, core::ptr::addr_of_mut!(fallback_mt));
if (result as u32) != 1 {
if (result as u32) == 2 {
break;
}
page = get_page_from_free_area(area, fallback_mt);
page = try_to_claim_block(zone, page, current_order, order, start_migratetype, fallback_mt, alloc_flags);
if !page.is_null() {
trace_mm_page_alloc_extfrag(page, order, current_order, start_migratetype, fallback_mt);
return page;
}
}
current_order = current_order.wrapping_sub(1);
}
core::ptr::null_mut::<page>()
}

/// Try to steal a single page from some fallback migratetype. Leave the rest of
/// the block as its current migratetype, potentially causing fragmentation.
#[inline(always)]
fn __rmqueue_steal(zone: *mut zone, order: i32, start_migratetype: i32) -> *mut page {
let mut current_order: i32 = order;
let mut fallback_mt: i32 = 0;
while current_order < 11i32 {
let mut result: u32 = 0;
let area: *mut free_area = unsafe { (*zone).free_area.as_mut_ptr() as *mut free_area }.wrapping_offset(current_order as isize);
result = find_suitable_fallback(area, current_order as u32, start_migratetype, false, core::ptr::addr_of_mut!(fallback_mt));
if (result as u32) != 1 {
let page: *mut page = get_page_from_free_area(area, fallback_mt);
page_del_and_expand(zone, page, order, current_order, fallback_mt);
trace_mm_page_alloc_extfrag(page, order, current_order, start_migratetype, fallback_mt);
return page;
}
current_order = current_order.wrapping_add(1);
}
core::ptr::null_mut::<page>()
}

/// Do the hard work of removing an element from the buddy allocator.
/// Call me with the zone->lock already held.
#[inline(always)]
fn __rmqueue(zone: *mut zone, order: u32, migratetype: i32, alloc_flags: u32, mode: &mut u32) -> *mut page {
let mut page: *mut page = unsafe { core::mem::zeroed() };
// First try the freelists of the requested migratetype, then try
// fallbacks modes with increasing levels of fragmentation risk.
//
// The fallback logic is expensive and rmqueue_bulk() calls in
// a loop with the zone->lock held, meaning the freelists are
// not subject to any outside changes. Remember in *mode where
// we found pay dirt, to save us the search on the next call.
// Replenished preferred freelist, back to normal mode.
match (unsafe { *(mode as *mut u32) }) as u32 {
0 => {
page = __rmqueue_smallest(zone, order, migratetype);
if !page.is_null() {
return page;
}
if alloc_flags & 128 != 0 {
page = __rmqueue_cma_fallback(zone, order);
if !page.is_null() {
unsafe {
*(mode as *mut u32) = 1;
}
return page;
}
}
page = __rmqueue_claim(zone, order as i32, migratetype, alloc_flags);
if !page.is_null() {
// Replenished preferred freelist, back to normal mode.
unsafe {
*(mode as *mut u32) = 0;
}
return page;
}
if alloc_flags & 256 == 0 {
page = __rmqueue_steal(zone, order as i32, migratetype);
if !page.is_null() {
unsafe {
*(mode as *mut u32) = 3;
}
return page;
}
}
}
1 => {
if alloc_flags & 128 != 0 {
page = __rmqueue_cma_fallback(zone, order);
if !page.is_null() {
unsafe {
*(mode as *mut u32) = 1;
}
return page;
}
}
page = __rmqueue_claim(zone, order as i32, migratetype, alloc_flags);
if !page.is_null() {
// Replenished preferred freelist, back to normal mode.
unsafe {
*(mode as *mut u32) = 0;
}
return page;
}
if alloc_flags & 256 == 0 {
page = __rmqueue_steal(zone, order as i32, migratetype);
if !page.is_null() {
unsafe {
*(mode as *mut u32) = 3;
}
return page;
}
}
}
2 => {
page = __rmqueue_claim(zone, order as i32, migratetype, alloc_flags);
if !page.is_null() {
// Replenished preferred freelist, back to normal mode.
unsafe {
*(mode as *mut u32) = 0;
}
return page;
}
if alloc_flags & 256 == 0 {
page = __rmqueue_steal(zone, order as i32, migratetype);
if !page.is_null() {
unsafe {
*(mode as *mut u32) = 3;
}
return page;
}
}
}
3 => {
if alloc_flags & 256 == 0 {
page = __rmqueue_steal(zone, order as i32, migratetype);
if !page.is_null() {
unsafe {
*(mode as *mut u32) = 3;
}
return page;
}
}
}
_ => {
}
}
core::ptr::null_mut::<page>()
}

/// Obtain a specified number of elements from the buddy allocator, all under
/// a single hold of the lock, for efficiency.  Add them to the supplied list.
/// Returns the number of new pages which were placed at *list.
fn rmqueue_bulk(zone: *mut zone, order: u32, count: u64, list: *mut list_head, migratetype: i32, alloc_flags: u32) -> i32 {
let mut rmqm: u32 = 0;
let mut flags: u64 = 0;
let mut i: i32 = 0;
if unlikely__3!(alloc_flags & 1024) != 0 {
if !spin_trylock_irqsave!(unsafe { core::ptr::addr_of_mut!((*zone).lock) }, flags) {
return 0;
}
} else {
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*zone).lock) }, flags);
}
while (i as u64) < count {
let page: *mut page = __rmqueue(zone, order, migratetype, alloc_flags, unsafe { &mut (*core::ptr::addr_of_mut!(rmqm)) });
if unlikely!(page.is_null()) != 0 {
break;
}
// Split buddy pages returned by expand() are received here in
// physical page order. The page is added to the tail of
// caller's list. From the callers perspective, the linked list
// is ordered by page number under some conditions. This is
// useful for IO devices that can forward direction from the
// head, thus also in the physical page order. This is useful
// for IO devices that can merge IO requests if the physical
// pages are ordered properly.
list_add_tail(unsafe { core::ptr::addr_of_mut!((*page).__slate_anon_1.__slate_anon_0.__slate_anon_0.pcp_list) }, list);
i = i.wrapping_add(1);
}
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*zone).lock) }, flags);
i
}

// Called from the vmstat counter updater to decay the PCP high.
// Return whether there are addition works to do.
// Called from the vmstat counter updater to drain pagesets of this
// currently executing processor on remote nodes after they have
// expired.
// Drain pcplists of the indicated processor and zone.
fn drain_pages_zone(cpu: u32, zone: *mut zone) {
let pcp: *mut per_cpu_pages = per_cpu_ptr__2!(unsafe { (*zone).per_cpu_pageset }, cpu);
let mut count: i32 = 0;
loop {
pcp_spin_lock_nopin!(pcp);
count = unsafe { (*pcp).count };
if count != 0 {
let mut to_drain: i32 = 0;
let mut __v3: i32;
let __UNIQUE_ID_x__932: i32 = count;
let __UNIQUE_ID_y__933: i32 = (unsafe { (*pcp).batch }) << 5i32;
loop {
if __types_ok__4!(__UNIQUE_ID_x__932, __UNIQUE_ID_y__933) == 0 {
unsafe { __compiletime_assert_934() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = if __UNIQUE_ID_x__932 < __UNIQUE_ID_y__933 { __UNIQUE_ID_x__932 } else { __UNIQUE_ID_y__933 };
to_drain = __v3;
free_pcppages_bulk(zone, to_drain, pcp, 0i32);
count = count.wrapping_sub(to_drain);
}
pcp_spin_unlock_nopin!(pcp);
if count == 0 {
break;
}
}
}

/// Drain pcplists of all zones on the indicated processor.
fn drain_pages(cpu: u32) {
let mut zone: *mut zone = unsafe { core::mem::zeroed() };
zone = unsafe { (*first_online_pgdat()).node_zones.as_mut_ptr() as *mut zone };
while !zone.is_null() {
if populated_zone(zone as *const zone) {
drain_pages_zone(cpu, zone);
}
zone = unsafe { next_zone(zone) };
}
}

/// Spill all of this CPU's per-cpu pages back into the buddy allocator.
#[unsafe(no_mangle)]
extern "C" fn drain_local_pages(zone: *mut zone) {
let mut cpu: i32 = 0;
let mut __v0: i32;
__this_cpu_preempt_check(c"read".as_ptr() as *const u8);
let mut __v1: i32;
let mut pscr_ret__: i32 = 0;
__verify_pcpu_ptr__5!();
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v3: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v3 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v3;
}
2 => {
let mut __v5: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
4 => {
let mut __v6: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v6 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v6;
}
8 => {
let mut __v7: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v7 = pfo_val__ as i32;
pscr_ret__ = __v7;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v1 = pscr_ret__;
__v0 = __v1;
cpu = __v0;
if !zone.is_null() {
drain_pages_zone(cpu as u32, zone);
} else {
drain_pages(cpu as u32);
}
}

/// The implementation of drain_all_pages(), exposing an extra parameter to
/// drain on all cpus.
///
/// drain_all_pages() is optimized to only execute on cpus where pcplists are
/// not empty. The check for non-emptiness can however race with a free to
/// pcplist that has not yet increased the pcp->count from 0 to 1. Callers
/// that need the guarantee that every CPU has drained can disable the
/// optimizing racy check.
fn __drain_all_pages(zone: *mut zone, force_all_cpus: bool) {
// We don't care about racing with CPU hotplug event
// as offline notification will cause the notified
// cpu to drain that CPU pcps and on_each_cpu_mask
// disables preemption as part of its processing
let mut cpu: i32 = 0;
// Allocate in the BSS so we won't require allocation in
// direct reclaim path for CONFIG_CPUMASK_OFFSTACK=y
// Do not drain if one is already in progress unless it's specific to
// a zone. Such callers are primarily CMA and memory hotplug and need
// the drain to be complete when the call returns.
if unlikely!((unsafe { mutex_trylock(core::ptr::addr_of_mut!(pcpu_drain_mutex)) }) == 0) != 0 {
if zone.is_null() {
return;
}
unsafe { mutex_lock(core::ptr::addr_of_mut!(pcpu_drain_mutex)) };
}
loop {
cpu = find_next_bit(cpumask_bits!((unsafe { core::ptr::addr_of_mut!(__cpu_online_mask) }) as *const cpumask), 64u64, cpu as u64) as i32;
if (cpu as u32) >= 64 {
break;
}
let mut pcp: *mut per_cpu_pages = unsafe { core::mem::zeroed() };
let mut z: *mut zone = unsafe { core::mem::zeroed() };
let mut has_pcps: bool = false;
if force_all_cpus {
// The pcp.count check is racy, some callers need a
// guarantee that no cpu is missed.
has_pcps = true;
} else {
if !zone.is_null() {
pcp = per_cpu_ptr__2!(unsafe { (*zone).per_cpu_pageset }, cpu);
if (unsafe { (*pcp).count }) != 0 {
has_pcps = true;
}
} else {
z = unsafe { (*first_online_pgdat()).node_zones.as_mut_ptr() as *mut zone };
while !z.is_null() {
if populated_zone(z as *const zone) {
pcp = per_cpu_ptr__2!(unsafe { (*z).per_cpu_pageset }, cpu);
if (unsafe { (*pcp).count }) != 0 {
has_pcps = true;
break;
}
}
z = unsafe { next_zone(z) };
}
}
}
if has_pcps {
cpumask_set_cpu(cpu as u32, unsafe { core::ptr::addr_of_mut!(cpus_with_pcps) });
} else {
cpumask_clear_cpu(cpu, unsafe { core::ptr::addr_of_mut!(cpus_with_pcps) });
}
cpu = cpu.wrapping_add(1);
}
cpu = 0;
loop {
cpu = find_next_bit(cpumask_bits__2!(unsafe { core::ptr::addr_of_mut!(cpus_with_pcps) }) as *const u64, 64u64, cpu as u64) as i32;
if (cpu as u32) >= 64 {
break;
}
if !zone.is_null() {
drain_pages_zone(cpu as u32, zone);
} else {
drain_pages(cpu as u32);
}
cpu = cpu.wrapping_add(1);
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!(pcpu_drain_mutex)) };
}

static mut cpus_with_pcps: cpumask = unsafe { core::mem::zeroed() };

/// Spill all the per-cpu pages from all CPUs back into the buddy allocator.
///
/// When zone parameter is non-NULL, spill just the single zone's pages.
#[unsafe(no_mangle)]
extern "C" fn drain_all_pages(zone: *mut zone) {
__drain_all_pages(zone, false);
}

fn nr_pcp_free(pcp: *mut per_cpu_pages, mut batch: i32, high: i32, free_high: bool) -> i32 {
let mut min_nr_free: i32 = 0;
let mut max_nr_free: i32 = 0;
// Free as much as possible if batch freeing high-order pages.
if unlikely!(free_high) != 0 {
let mut __v0: i32;
let __UNIQUE_ID_x__935: i32 = unsafe { (*pcp).count };
let __UNIQUE_ID_y__936: i32 = batch << 5i32;
loop {
if __types_ok!(__UNIQUE_ID_x__935, __UNIQUE_ID_y__936) == 0 {
unsafe { __compiletime_assert_937() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__935 < __UNIQUE_ID_y__936 { __UNIQUE_ID_x__935 } else { __UNIQUE_ID_y__936 };
return __v0;
}
// Check for PCP disabled or boot pageset
if unlikely!(high < batch) != 0 {
return 1;
}
// Leave at least pcp->batch pages on the list
min_nr_free = batch;
max_nr_free = high.wrapping_sub(batch);
// Increase the batch number to the number of the consecutive
// freed pages to reduce zone lock contention.
let mut __v2: i32;
let __UNIQUE_ID_v__938: i32 = (unsafe { (*pcp).free_count }) as i32;
let __UNIQUE_ID_l__939: i32 = min_nr_free;
let __UNIQUE_ID_h__940: i32 = max_nr_free;
loop {
if !(!statically_true!(__UNIQUE_ID_l__939 > __UNIQUE_ID_h__940)) {
unsafe { __compiletime_assert_941() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if __types_ok3__2!(__UNIQUE_ID_v__938, __UNIQUE_ID_l__939, __UNIQUE_ID_h__940) == 0 {
unsafe { __compiletime_assert_942() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = if __UNIQUE_ID_v__938 >= __UNIQUE_ID_h__940 { __UNIQUE_ID_h__940 } else { if __UNIQUE_ID_v__938 <= __UNIQUE_ID_l__939 { __UNIQUE_ID_l__939 } else { __UNIQUE_ID_v__938 } };
batch = __v2;
batch
}

fn nr_pcp_high(pcp: *mut per_cpu_pages, zone: *mut zone, batch: i32, free_high: bool) -> i32 {
let mut high: i32 = 0;
let mut high_min: i32 = 0;
let mut high_max: i32 = 0;
let mut __v0: i32;
loop {
if !(__native_word__9!() || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_943() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*pcp).high_min) as *const i32) };
high_min = __v0;
let mut __v2: i32;
loop {
if !(__native_word__9!() || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_944() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*pcp).high_max) as *const i32) };
high_max = __v2;
let mut __v4: i32;
let __UNIQUE_ID_v__945: i32 = unsafe { (*pcp).high };
let __UNIQUE_ID_l__946: i32 = high_min;
let __UNIQUE_ID_h__947: i32 = high_max;
loop {
if !(!statically_true!(__UNIQUE_ID_l__946 > __UNIQUE_ID_h__947)) {
unsafe { __compiletime_assert_948() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if __types_ok3__2!(__UNIQUE_ID_v__945, __UNIQUE_ID_l__946, __UNIQUE_ID_h__947) == 0 {
unsafe { __compiletime_assert_949() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v4 = if __UNIQUE_ID_v__945 >= __UNIQUE_ID_h__947 { __UNIQUE_ID_h__947 } else { if __UNIQUE_ID_v__945 <= __UNIQUE_ID_l__946 { __UNIQUE_ID_l__946 } else { __UNIQUE_ID_v__945 } };
let __v7: i32 = __v4;
unsafe {
(*pcp).high = __v7;
}
high = __v7;
if unlikely!(high == 0) != 0 {
return 0;
}
if unlikely!(free_high) != 0 {
let mut __v8: i32;
let __UNIQUE_ID_x__950: i32 = high.wrapping_sub(batch << 5i32);
let __UNIQUE_ID_y__951: i32 = high_min;
loop {
if __types_ok!(__UNIQUE_ID_x__950, __UNIQUE_ID_y__951) == 0 {
unsafe { __compiletime_assert_952() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v8 = if __UNIQUE_ID_x__950 > __UNIQUE_ID_y__951 { __UNIQUE_ID_x__950 } else { __UNIQUE_ID_y__951 };
unsafe {
(*pcp).high = __v8;
}
return 0;
}
// If reclaim is active, limit the number of pages that can be
// stored on pcp lists
if test_bit__4!(1i32, unsafe { core::ptr::addr_of_mut!((*zone).flags) }) != 0 {
let free_count: i32 = max_t!(i32, unsafe { (*pcp).free_count }, batch);
let mut __v14: i32;
let __UNIQUE_ID_x__955: i32 = high.wrapping_sub(free_count);
let __UNIQUE_ID_y__956: i32 = high_min;
loop {
if __types_ok!(__UNIQUE_ID_x__955, __UNIQUE_ID_y__956) == 0 {
unsafe { __compiletime_assert_957() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v14 = if __UNIQUE_ID_x__955 > __UNIQUE_ID_y__956 { __UNIQUE_ID_x__955 } else { __UNIQUE_ID_y__956 };
unsafe {
(*pcp).high = __v14;
}
let mut __v16: i32;
let __UNIQUE_ID_x__958: i32 = batch << 2i32;
let __UNIQUE_ID_y__959: i32 = unsafe { (*pcp).high };
loop {
if __types_ok!(__UNIQUE_ID_x__958, __UNIQUE_ID_y__959) == 0 {
unsafe { __compiletime_assert_960() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v16 = if __UNIQUE_ID_x__958 < __UNIQUE_ID_y__959 { __UNIQUE_ID_x__958 } else { __UNIQUE_ID_y__959 };
return __v16;
}
if high_min == high_max {
return high;
}
if test_bit__4!(2i32, unsafe { core::ptr::addr_of_mut!((*zone).flags) }) != 0 {
let free_count: i32 = max_t!(i32, unsafe { (*pcp).free_count }, batch);
let mut __v22: i32;
let __UNIQUE_ID_x__963: i32 = high.wrapping_sub(free_count);
let __UNIQUE_ID_y__964: i32 = high_min;
loop {
if __types_ok!(__UNIQUE_ID_x__963, __UNIQUE_ID_y__964) == 0 {
unsafe { __compiletime_assert_965() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v22 = if __UNIQUE_ID_x__963 > __UNIQUE_ID_y__964 { __UNIQUE_ID_x__963 } else { __UNIQUE_ID_y__964 };
unsafe {
(*pcp).high = __v22;
}
let mut __v24: i32;
let __UNIQUE_ID_x__966: i32 = unsafe { (*pcp).count };
let __UNIQUE_ID_y__967: i32 = high_min;
loop {
if __types_ok!(__UNIQUE_ID_x__966, __UNIQUE_ID_y__967) == 0 {
unsafe { __compiletime_assert_968() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v24 = if __UNIQUE_ID_x__966 > __UNIQUE_ID_y__967 { __UNIQUE_ID_x__966 } else { __UNIQUE_ID_y__967 };
high = __v24;
} else {
if (unsafe { (*pcp).count }) >= high {
let need_high: i32 = ((unsafe { (*pcp).free_count }) as i32).wrapping_add(batch);
// pcp->high should be large enough to hold batch freed pages
if (unsafe { (*pcp).high }) < need_high {
let mut __v26: i32;
let __UNIQUE_ID_v__969: i32 = need_high;
let __UNIQUE_ID_l__970: i32 = high_min;
let __UNIQUE_ID_h__971: i32 = high_max;
loop {
if !(!statically_true!(__UNIQUE_ID_l__970 > __UNIQUE_ID_h__971)) {
unsafe { __compiletime_assert_972() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if __types_ok3__2!(__UNIQUE_ID_v__969, __UNIQUE_ID_l__970, __UNIQUE_ID_h__971) == 0 {
unsafe { __compiletime_assert_973() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v26 = if __UNIQUE_ID_v__969 >= __UNIQUE_ID_h__971 { __UNIQUE_ID_h__971 } else { if __UNIQUE_ID_v__969 <= __UNIQUE_ID_l__970 { __UNIQUE_ID_l__970 } else { __UNIQUE_ID_v__969 } };
unsafe {
(*pcp).high = __v26;
}
}
}
}
high
}

// Tune pcp alloc factor and adjust count & free_count. Free pages to bring the
// pcp's watermarks below high.
//
// May return a freed pcp, if during page freeing the pcp spinlock cannot be
// reacquired. Return true if pcp is locked, false otherwise.
fn free_frozen_page_commit(zone: *mut zone, mut pcp: *mut per_cpu_pages, page: *mut page, migratetype: i32, order: u32, fpi_flags: i32) -> bool {
let mut high: i32 = 0;
let mut batch: i32 = 0;
let mut to_free: i32 = 0;
let mut to_free_batched: i32 = 0;
let mut pindex: i32 = 0;
let mut cpu: i32 = 0;
let mut __v0: i32;
__this_cpu_preempt_check(c"read".as_ptr() as *const u8);
let mut __v1: i32;
{
let mut pscr_ret__: i32 = 0;
__verify_pcpu_ptr__5!();
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v3: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v3 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v3;
}
2 => {
let mut __v5: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
4 => {
let mut __v6: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v6 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v6;
}
8 => {
let mut __v7: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v7 = pfo_val__ as i32;
pscr_ret__ = __v7;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v1 = pscr_ret__;
}
__v0 = __v1;
cpu = __v0;
let mut ret: i32 = 1;
let mut free_high: bool = false;
// On freeing, reduce the number of pages that are batch allocated.
// See nr_pcp_alloc() where alloc_factor is increased for subsequent
// allocations.
unsafe {
(*pcp).alloc_factor = (((*pcp).alloc_factor as i32) >> 1i32) as u8;
}
__count_vm_events(16u32, (1i32 << order) as i64);
pindex = order_to_pindex(migratetype, order as i32) as i32;
list_add(unsafe { core::ptr::addr_of_mut!((*page).__slate_anon_1.__slate_anon_0.__slate_anon_0.pcp_list) }, unsafe { (*pcp).lists.as_mut_ptr() as *mut list_head }.wrapping_offset(pindex as isize));
let __v12: i32 = unsafe { (*pcp).count };
unsafe {
(*pcp).count = __v12.wrapping_add(1i32 << order);
}
let mut __v14: i32;
while false {
}
__v14 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*pcp).batch) as *const i32) };
batch = __v14;
// As high-order pages other than THP's stored on PCP can contribute
// to fragmentation, limit the number stored when PCP is heavily
// freeing without allocation. The remainder after bulk freeing
// stops will be drained from vmstat refresh context.
if order != 0 && order <= 3 {
free_high = ((unsafe { (*pcp).free_count }) as i32) >= batch.wrapping_add((unsafe { (*pcp).high_min }) / 2) && (((unsafe { (*pcp).flags }) as i32) as u64) & BIT!(0i32) != 0 && ((((unsafe { (*pcp).flags }) as i32) as u64) & BIT!(1i32) == 0 || (unsafe { (*pcp).count }) >= batch);
let __v17: u8 = unsafe { (*pcp).flags };
let __v18: u8 = (((__v17 as i32) as u64) | BIT!(0i32)) as u8;
unsafe {
(*pcp).flags = __v18;
}
} else {
if (((unsafe { (*pcp).flags }) as i32) as u64) & BIT!(0i32) != 0 {
let __v20: u8 = unsafe { (*pcp).flags };
let __v21: u8 = (((__v20 as i32) as u64) & !BIT!(0i32)) as u8;
unsafe {
(*pcp).flags = __v21;
}
}
}
if ((unsafe { (*pcp).free_count }) as i32) < batch << 5i32 {
unsafe {
(*pcp).free_count = ((*pcp).free_count as i32).wrapping_add(1i32 << order) as i16;
}
}
if unlikely__2!(fpi_flags & (BIT!(2i32) as i32)) != 0 {
// Do not attempt to take a zone lock. Let pcp->count get
// over high mark temporarily.
return true;
}
high = nr_pcp_high(pcp, zone, batch, free_high);
if (unsafe { (*pcp).count }) < high {
return true;
}
to_free = nr_pcp_free(pcp, batch, high, free_high);
while to_free > 0 && (unsafe { (*pcp).count }) > 0 {
let mut __v25: i32;
let __UNIQUE_ID_x__975: i32 = to_free;
let __UNIQUE_ID_y__976: i32 = batch;
loop {
if __types_ok__4!(__UNIQUE_ID_x__975, __UNIQUE_ID_y__976) == 0 {
unsafe { __compiletime_assert_977() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v25 = if __UNIQUE_ID_x__975 < __UNIQUE_ID_y__976 { __UNIQUE_ID_x__975 } else { __UNIQUE_ID_y__976 };
to_free_batched = __v25;
free_pcppages_bulk(zone, to_free_batched, pcp, pindex);
to_free = to_free.wrapping_sub(to_free_batched);
if to_free == 0 || (unsafe { (*pcp).count }) == 0 {
break;
}
spin_unlock(unsafe { core::ptr::addr_of_mut!((*pcp).lock) });
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
let mut __v32: *mut per_cpu_pages;
let mut _ret: *mut per_cpu_pages = unsafe { core::mem::zeroed() };
loop {
preempt_count_inc!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
let mut __v34: *mut per_cpu_pages;
__verify_pcpu_ptr__13!();
let mut __v36: *mut per_cpu_pages;
let mut tcp_ptr__: u64 = 0;
let mut __v37: u64;
{
let mut pfo_val__: u64 = 0;
unsafe {
let __t4: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t4, sym this_cpu_off, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t4 as u64) };
}
__v37 = pfo_val__;
}
tcp_ptr__ = __v37;
tcp_ptr__ = tcp_ptr__.wrapping_add((unsafe { (*zone).per_cpu_pageset }) as u64);
__v36 = tcp_ptr__ as *mut per_cpu_pages;
__v34 = __v36;
_ret = __v34;
if spin_trylock(unsafe { core::ptr::addr_of_mut!((*_ret).lock) }) == 0 {
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
_ret = core::ptr::null_mut::<per_cpu_pages>();
}
__v32 = _ret;
pcp = __v32;
if pcp.is_null() {
ret = 0;
break;
}
// Check if this thread has been migrated to a different CPU.
// If that is the case, give up and indicate that the pcp is
// returned in an unlocked state.
let mut __v42: i32;
__this_cpu_preempt_check(c"read".as_ptr() as *const u8);
let mut __v43: i32;
let mut pscr_ret__: i32 = 0;
__verify_pcpu_ptr__5!();
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v45: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t5: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t5, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t5 as u8) };
}
__v45 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v45;
}
2 => {
let mut __v47: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t6: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t6, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t6 as u16) };
}
__v47 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v47;
}
4 => {
let mut __v48: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t7: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t7, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t7 as u32) };
}
__v48 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v48;
}
8 => {
let mut __v49: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t8: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t8, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t8 as u64) };
}
__v49 = pfo_val__ as i32;
pscr_ret__ = __v49;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v43 = pscr_ret__;
__v42 = __v43;
if __v42 != cpu {
spin_unlock(unsafe { core::ptr::addr_of_mut!((*pcp).lock) });
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
ret = 0;
break;
}
}
let __v52: bool = test_bit__4!(2i32, unsafe { core::ptr::addr_of_mut!((*zone).flags) }) != 0 && zone_watermark_ok(zone, 0u32, high_wmark_pages(zone as *const zone), 3i32, 0u32);
if __v52 {
let pgdat: *mut pglist_data = unsafe { (*zone).zone_pgdat };
clear_bit(2i64, (unsafe { core::ptr::addr_of_mut!((*zone).flags) }) as *mut u64);
// Assume that memory pressure on this node is gone and may be
// in a reclaimable state. If a memory fallback node exists,
// direct reclaim may not have been triggered, causing a
// 'hopeless node' to stay in that state for a while.  Let
// kswapd work again by resetting kswapd_failures.
let __v54: bool = (unsafe { kswapd_test_hopeless(pgdat) }) && next_memory_node(unsafe { (*pgdat).node_id }) < 64u32;
if __v54 {
unsafe { kswapd_clear_hopeless(pgdat, 3u32) };
}
}
ret != 0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_978: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_979: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_980: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

/// Free a pcp page
fn __free_frozen_pages(page: *mut page, order: u32, fpi_flags: i32) {
let mut pcp: *mut per_cpu_pages = unsafe { core::mem::zeroed() };
let pfn: u64 = ((unsafe { page.offset_from(vmemmap_base as *mut page) }) as i64) as u64;
let mut migratetype: i32 = 0;
if !pcp_allowed_order(order) {
__free_pages_ok(page, order, fpi_flags);
return;
}
if !__free_pages_prepare(page, order, fpi_flags) {
return;
}
// We only track unmovable, reclaimable and movable on pcp lists.
// Place ISOLATE pages on the isolated list because they are being
// offlined but treat HIGHATOMIC and CMA as movable pages so we can
// get those areas back if necessary. Otherwise, we may have to free
// excessively into the page allocator
let zone: *mut zone = page_zone(page as *const page);
migratetype = (get_pfnblock_migratetype(page as *const page, pfn) as u32) as i32;
if unlikely!(migratetype >= 3) != 0 {
if unlikely!(is_migrate_isolate(migratetype)) != 0 {
free_one_page(zone, page, pfn, order, fpi_flags);
return;
}
migratetype = 1;
}
let __v0: bool = fpi_flags & (BIT!(2i32) as i32) != 0 && !can_spin_trylock();
if __v0 {
add_page_to_zone_llist(zone, page, order);
return;
}
let mut __v1: *mut per_cpu_pages;
let mut _ret: *mut per_cpu_pages = unsafe { core::mem::zeroed() };
loop {
preempt_count_inc!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
let mut __v3: *mut per_cpu_pages;
__verify_pcpu_ptr__13!();
let mut __v5: *mut per_cpu_pages;
let mut tcp_ptr__: u64 = 0;
let mut __v6: u64;
let mut pfo_val__: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t0, sym this_cpu_off, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
__v6 = pfo_val__;
tcp_ptr__ = __v6;
tcp_ptr__ = tcp_ptr__.wrapping_add((unsafe { (*zone).per_cpu_pageset }) as u64);
__v5 = tcp_ptr__ as *mut per_cpu_pages;
__v3 = __v5;
_ret = __v3;
if spin_trylock(unsafe { core::ptr::addr_of_mut!((*_ret).lock) }) == 0 {
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
_ret = core::ptr::null_mut::<per_cpu_pages>();
}
__v1 = _ret;
pcp = __v1;
if !pcp.is_null() {
if !free_frozen_page_commit(zone, pcp, page, migratetype, order, fpi_flags) {
return;
}
spin_unlock(unsafe { core::ptr::addr_of_mut!((*pcp).lock) });
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
} else {
free_one_page(zone, page, pfn, order, fpi_flags);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_981: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_982: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

#[unsafe(no_mangle)]
extern "C" fn free_frozen_pages(page: *mut page, order: u32) {
__free_frozen_pages(page, order, 0i32);
}

#[unsafe(no_mangle)]
extern "C" fn free_frozen_pages_nolock(page: *mut page, order: u32) {
__free_frozen_pages(page, order, BIT!(2i32) as i32);
}

/// Free a batch of folios
#[unsafe(no_mangle)]
extern "C" fn free_unref_folios(folios: *mut folio_batch) {
let mut pcp: *mut per_cpu_pages = core::ptr::null_mut::<per_cpu_pages>();
let mut locked_zone: *mut zone = core::ptr::null_mut::<zone>();
// Prepare folios for freeing
let mut i: i32 = 0;
let mut j: i32 = 0;
while i < ((unsafe { (*folios).nr }) as i32) {
let folio: *mut folio = unsafe { *((*folios).folios.as_mut_ptr() as *mut *mut folio).wrapping_offset(i as isize) };
let pfn: u64 = folio_pfn(folio as *const folio);
let order: u32 = folio_order(folio as *const folio);
if __free_pages_prepare__const21(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.page) }, order, 0i32) {
// Free orders not handled on the PCP directly to the
// allocator.
if !pcp_allowed_order(order) {
free_one_page(folio_zone(folio as *const folio), unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.page) }, pfn, order, 0i32);
} else {
unsafe {
(*folio).__slate_anon_0.__slate_anon_0.__slate_anon_4.private = (order as u64) as *mut ();
}
if j != i {
unsafe {
*((*folios).folios.as_mut_ptr() as *mut *mut folio).wrapping_offset(j as isize) = folio;
}
}
j = j.wrapping_add(1);
}
}
i = i.wrapping_add(1);
}
unsafe {
(*folios).nr = j as u8;
}
i = 0;
while i < ((unsafe { (*folios).nr }) as i32) {
'__slate_continue_8: {
let folio: *mut folio = unsafe { *((*folios).folios.as_mut_ptr() as *mut *mut folio).wrapping_offset(i as isize) };
let zone: *mut zone = folio_zone(folio as *const folio);
let pfn: u64 = folio_pfn(folio as *const folio);
let order: u32 = ((unsafe { (*folio).__slate_anon_0.__slate_anon_0.__slate_anon_4.private }) as u64) as u32;
let mut migratetype: i32 = 0;
unsafe {
(*folio).__slate_anon_0.__slate_anon_0.__slate_anon_4.private = core::ptr::null_mut::<()>();
}
migratetype = (get_pfnblock_migratetype((unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.page) }) as *const page, pfn) as u32) as i32;
// Different zone requires a different pcp lock
let __v5: bool = zone != locked_zone || is_migrate_isolate(migratetype);
if __v5 {
if !pcp.is_null() {
spin_unlock(unsafe { core::ptr::addr_of_mut!((*pcp).lock) });
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
locked_zone = core::ptr::null_mut::<zone>();
pcp = core::ptr::null_mut::<per_cpu_pages>();
}
// Free isolated pages directly to the
// allocator, see comment in free_frozen_pages.
if is_migrate_isolate(migratetype) {
free_one_page(zone, unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.page) }, pfn, order, 0i32);
break '__slate_continue_8;
}
// trylock is necessary as folios may be getting freed
// from IRQ or SoftIRQ context after an IO completion.
let mut __v9: *mut per_cpu_pages;
let mut _ret: *mut per_cpu_pages = unsafe { core::mem::zeroed() };
loop {
preempt_count_inc!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
let mut __v11: *mut per_cpu_pages;
__verify_pcpu_ptr__13!();
let mut __v13: *mut per_cpu_pages;
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
tcp_ptr__ = tcp_ptr__.wrapping_add((unsafe { (*zone).per_cpu_pageset }) as u64);
__v13 = tcp_ptr__ as *mut per_cpu_pages;
__v11 = __v13;
_ret = __v11;
if spin_trylock(unsafe { core::ptr::addr_of_mut!((*_ret).lock) }) == 0 {
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
_ret = core::ptr::null_mut::<per_cpu_pages>();
}
__v9 = _ret;
pcp = __v9;
if unlikely!(pcp.is_null()) != 0 {
free_one_page(zone, unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.page) }, pfn, order, 0i32);
break '__slate_continue_8;
}
locked_zone = zone;
}
// Non-isolated types over MIGRATE_PCPTYPES get added
// to the MIGRATE_MOVABLE pcp list.
if unlikely!(migratetype >= 3) != 0 {
migratetype = 1;
}
trace_mm_page_free_batched(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.page) });
if !free_frozen_page_commit(zone, pcp, unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.page) }, migratetype, order, 0i32) {
pcp = core::ptr::null_mut::<per_cpu_pages>();
locked_zone = core::ptr::null_mut::<zone>();
}
}
i = i.wrapping_add(1);
}
if !pcp.is_null() {
spin_unlock(unsafe { core::ptr::addr_of_mut!((*pcp).lock) });
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
}
folio_batch_reinit(folios);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_983: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_984: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_985: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

fn __split_page(page: *mut page, order: u32) {
VM_WARN_ON_PAGE!();
split_page_owner(page, order as i32, 0i32);
pgalloc_tag_split(page_folio!(page), order as i32, 0i32);
split_page_memcg(page, order);
}

/// split_page takes a non-compound higher-order page, and splits it into
/// n (1<<order) sub-pages: page[0..n]
/// Each sub-page must be freed individually.
///
/// Note: this is probably too low level an operation for use in drivers.
/// Please consult with lkml before using this in your driver.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.page_alloc.split_page")]
extern "C" fn split_page(page: *mut page, order: u32) {
VM_WARN_ON_PAGE!();
for i in 1..(1i32 << order) {
set_page_refcounted(page.wrapping_offset(i as isize));
}
__split_page(page, order);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_split_page_986: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut page, u32)>, *mut ()>(Some(split_page)) };

#[unsafe(no_mangle)]
extern "C" fn __isolate_free_page(mut page: *mut page, order: u32) -> i32 {
let zone: *mut zone = page_zone(page as *const page);
let mt: i32 = (get_pageblock_migratetype__2!(page) as u32) as i32;
if !is_migrate_isolate(mt) {
let mut watermark: u64 = 0;
// Obey watermarks as if the page was being allocated. We can
// emulate a high-order watermark check with a raised order-0
// watermark, because we already know our high-order page
// exists.
watermark = unsafe { *((*zone)._watermark.as_mut_ptr() as *mut u64).wrapping_offset(0 as isize) }.wrapping_add(1u64 << order);
if !zone_watermark_ok(zone, 0u32, watermark, 0i32, 128u32) {
return 0;
}
}
del_page_from_free_list(page, zone, order, mt);
// Set the pageblock if the isolated page is at least half of a
// pageblock
if order >= MIN_T!(u32, 9i32).wrapping_sub(1) {
let endpage: *mut page = page.wrapping_offset((1i32 << order) as isize).wrapping_offset(-(1 as isize));
while page < endpage {
let mt: i32 = (get_pageblock_migratetype__2!(page) as u32) as i32;
// Only change normal pageblocks (i.e., they can merge
// with others)
if migratetype_is_mergeable(mt) {
move_freepages_block(zone, page, mt, 1i32);
}
let __v1: *mut page = page;
page = __v1.wrapping_offset((1u64 << MIN_T!(u32, 9i32)) as isize);
}
}
(1u64 << order) as i32
}

/// __putback_isolated_page - Return a now-isolated page back where we got it
/// @page: Page that was isolated
/// @order: Order of the isolated page
/// @mt: The page's pageblock's migratetype
///
/// This function is meant to return a page pulled from the free lists via
/// __isolate_free_page back to the free lists they were pulled from.
#[unsafe(no_mangle)]
extern "C" fn __putback_isolated_page(page: *mut page, order: u32, mt: i32) {
let zone: *mut zone = page_zone(page as *const page);
// zone lock should be held when this function is called
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!((*zone).lock) });
// Return isolated page to tail of freelist.
__free_one_page(page, ((unsafe { page.offset_from(vmemmap_base as *mut page) }) as i64) as u64, zone, order, mt, (BIT!(0i32) as i32) | (BIT!(1i32) as i32));
}

/// Update NUMA hit/miss statistics
fn zone_statistics(preferred_zone: *mut zone, z: *mut zone, nr_account: i64) {
let mut local_stat: u32 = 4;
// skip numa counters update if numa stats is disabled
if static_branch_likely!(unsafe { core::ptr::addr_of_mut!(vm_numa_stat_key) }) == 0 {
return;
}
if zone_to_nid(z as *const zone) != numa_node_id() {
local_stat = 5;
}
if zone_to_nid(z as *const zone) == zone_to_nid(preferred_zone as *const zone) {
__count_numa_events(z, 0u32, nr_account);
} else {
__count_numa_events(z, 1u32, nr_account);
__count_numa_events(preferred_zone, 2u32, nr_account);
}
__count_numa_events(z, local_stat, nr_account);
}

#[inline(always)]
fn rmqueue_buddy(preferred_zone: *mut zone, zone: *mut zone, order: u32, alloc_flags: u32, migratetype: i32) -> *mut page {
let mut page: *mut page = unsafe { core::mem::zeroed() };
let mut flags: u64 = 0;
loop {
page = core::ptr::null_mut::<page>();
if unlikely__3!(alloc_flags & 1024) != 0 {
if !spin_trylock_irqsave!(unsafe { core::ptr::addr_of_mut!((*zone).lock) }, flags) {
return core::ptr::null_mut::<page>();
}
} else {
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*zone).lock) }, flags);
}
if alloc_flags & 512 != 0 {
page = __rmqueue_smallest__const24(zone, order, 3i32);
}
if page.is_null() {
let mut rmqm: u32 = 0;
page = __rmqueue(zone, order, migratetype, alloc_flags, unsafe { &mut (*core::ptr::addr_of_mut!(rmqm)) });
// If the allocation fails, allow OOM handling and
// order-0 (atomic) allocs access to HIGHATOMIC
// reserves as failing now is worse than failing a
// high-order atomic allocation in the future.
if page.is_null() && alloc_flags & 24u32 != 0 {
page = __rmqueue_smallest__const24(zone, order, 3i32);
}
if page.is_null() {
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*zone).lock) }, flags);
return core::ptr::null_mut::<page>();
}
}
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*zone).lock) }, flags);
if !check_new_pages(page, order) {
break;
}
}
// Slowpath (precarious) high-atomic allocations may reserve
// a pageblock for future use.
if unlikely!(alloc_flags & 512 != 0 && alloc_flags & 3u32 == 0) != 0 {
reserve_highatomic_pageblock(page, order as i32, zone);
}
__count_vm_events(4u32.wrapping_add(page_zonenum(page as *const page) as u32) as u32, (1i32 << order) as i64);
zone_statistics(preferred_zone, zone, 1i64);
page
}

fn nr_pcp_alloc(pcp: *mut per_cpu_pages, zone: *mut zone, order: i32) -> i32 {
let mut high: i32 = 0;
let mut base_batch: i32 = 0;
let mut batch: i32 = 0;
let mut max_nr_alloc: i32 = 0;
let mut high_max: i32 = 0;
let mut high_min: i32 = 0;
let mut __v0: i32;
loop {
if !(__native_word__9!() || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_987() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*pcp).batch) as *const i32) };
base_batch = __v0;
let mut __v2: i32;
loop {
if !(__native_word__9!() || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_988() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*pcp).high_min) as *const i32) };
high_min = __v2;
let mut __v4: i32;
loop {
if !(__native_word__9!() || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_989() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v4 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*pcp).high_max) as *const i32) };
high_max = __v4;
let mut __v6: i32;
let __UNIQUE_ID_v__990: i32 = unsafe { (*pcp).high };
let __UNIQUE_ID_l__991: i32 = high_min;
let __UNIQUE_ID_h__992: i32 = high_max;
loop {
if !(!statically_true!(__UNIQUE_ID_l__991 > __UNIQUE_ID_h__992)) {
unsafe { __compiletime_assert_993() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if __types_ok3__2!(__UNIQUE_ID_v__990, __UNIQUE_ID_l__991, __UNIQUE_ID_h__992) == 0 {
unsafe { __compiletime_assert_994() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v6 = if __UNIQUE_ID_v__990 >= __UNIQUE_ID_h__992 { __UNIQUE_ID_h__992 } else { if __UNIQUE_ID_v__990 <= __UNIQUE_ID_l__991 { __UNIQUE_ID_l__991 } else { __UNIQUE_ID_v__990 } };
let __v9: i32 = __v6;
unsafe {
(*pcp).high = __v9;
}
high = __v9;
// Check for PCP disabled or boot pageset
if unlikely!(high < base_batch) != 0 {
return 1;
}
if order != 0 {
batch = base_batch;
} else {
batch = base_batch << ((unsafe { (*pcp).alloc_factor }) as i32);
}
// If we had larger pcp->high, we could avoid to allocate from
// zone.
let __v10: bool = high_min != high_max && test_bit__4!(2i32, unsafe { core::ptr::addr_of_mut!((*zone).flags) }) == 0;
if __v10 {
let mut __v12: i32;
let __UNIQUE_ID_x__995: i32 = high.wrapping_add(batch);
let __UNIQUE_ID_y__996: i32 = high_max;
loop {
if __types_ok!(__UNIQUE_ID_x__995, __UNIQUE_ID_y__996) == 0 {
unsafe { __compiletime_assert_997() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v12 = if __UNIQUE_ID_x__995 < __UNIQUE_ID_y__996 { __UNIQUE_ID_x__995 } else { __UNIQUE_ID_y__996 };
let __v14: i32 = __v12;
unsafe {
(*pcp).high = __v14;
}
high = __v14;
}
if order == 0 {
let mut __v15: i32;
let __UNIQUE_ID_x__998: i32 = high.wrapping_sub(unsafe { (*pcp).count }).wrapping_sub(base_batch);
let __UNIQUE_ID_y__999: i32 = base_batch;
loop {
if __types_ok!(__UNIQUE_ID_x__998, __UNIQUE_ID_y__999) == 0 {
unsafe { __compiletime_assert_1000() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v15 = if __UNIQUE_ID_x__998 > __UNIQUE_ID_y__999 { __UNIQUE_ID_x__998 } else { __UNIQUE_ID_y__999 };
max_nr_alloc = __v15;
// Double the number of pages allocated each time there is
// subsequent allocation of order-0 pages without any freeing.
if batch <= max_nr_alloc && ((unsafe { (*pcp).alloc_factor }) as i32) < 5 {
unsafe {
(*pcp).alloc_factor = ((*pcp).alloc_factor as i32).wrapping_add(1) as u8;
}
}
let mut __v20: i32;
let __UNIQUE_ID_x__1001: i32 = batch;
let __UNIQUE_ID_y__1002: i32 = max_nr_alloc;
loop {
if __types_ok!(__UNIQUE_ID_x__1001, __UNIQUE_ID_y__1002) == 0 {
unsafe { __compiletime_assert_1003() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v20 = if __UNIQUE_ID_x__1001 < __UNIQUE_ID_y__1002 { __UNIQUE_ID_x__1001 } else { __UNIQUE_ID_y__1002 };
batch = __v20;
}
// Scale batch relative to order if batch implies free pages
// can be stored on the PCP. Batch can be 1 for small zones or
// for boot pagesets which should never store free pages as
// the pages may belong to arbitrary zones.
if batch > 1 {
let mut __v22: i32;
let __UNIQUE_ID_x__1004: i32 = batch >> order;
let __UNIQUE_ID_y__1005: i32 = 2;
loop {
if __types_ok!(__UNIQUE_ID_x__1004, __UNIQUE_ID_y__1005) == 0 {
unsafe { __compiletime_assert_1006() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v22 = if __UNIQUE_ID_x__1004 > __UNIQUE_ID_y__1005 { __UNIQUE_ID_x__1004 } else { __UNIQUE_ID_y__1005 };
batch = __v22;
}
batch
}

/// Remove page from the per-cpu list, caller must protect the list
fn __rmqueue_pcplist(zone: *mut zone, order: u32, migratetype: i32, alloc_flags: u32, pcp: *mut per_cpu_pages, list: *mut list_head) -> *mut page {
let mut page: *mut page = unsafe { core::mem::zeroed() };
loop {
if list_empty(list as *const list_head) != 0 {
let batch: i32 = nr_pcp_alloc(pcp, zone, order as i32);
let mut alloced: i32 = 0;
// Don't refill the list for a higher order atomic
// allocation under memory pressure, as this would
// not build up any HIGHATOMIC reserves, which
// might be needed soon.
//
// Instead, direct it towards the reserves by
// returning NULL, which will make the caller fall
// back to rmqueue_buddy. This will try to use the
// reserves first and grow them if needed.
if alloc_flags & 512 != 0 {
return core::ptr::null_mut::<page>();
}
alloced = rmqueue_bulk(zone, order, batch as u64, list, migratetype, alloc_flags);
let __v1: i32 = unsafe { (*pcp).count };
unsafe {
(*pcp).count = __v1.wrapping_add(alloced << order);
}
if unlikely__2!(list_empty(list as *const list_head)) != 0 {
return core::ptr::null_mut::<page>();
}
}
page = list_first_entry__9!(list, page);
list_del(unsafe { core::ptr::addr_of_mut!((*page).__slate_anon_1.__slate_anon_0.__slate_anon_0.pcp_list) });
let __v5: i32 = unsafe { (*pcp).count };
unsafe {
(*pcp).count = __v5.wrapping_sub(1i32 << order);
}
if !check_new_pages(page, order) {
break;
}
}
page
}

/// Lock and remove page from the per-cpu list
fn rmqueue_pcplist(preferred_zone: *mut zone, zone: *mut zone, order: u32, migratetype: i32, alloc_flags: u32) -> *mut page {
let mut pcp: *mut per_cpu_pages = unsafe { core::mem::zeroed() };
// spin_trylock may fail due to a parallel drain or IRQ reentrancy.
let mut __v0: *mut per_cpu_pages;
let mut _ret: *mut per_cpu_pages = unsafe { core::mem::zeroed() };
loop {
preempt_count_inc!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
let mut __v2: *mut per_cpu_pages;
__verify_pcpu_ptr__13!();
let mut __v4: *mut per_cpu_pages;
let mut tcp_ptr__: u64 = 0;
let mut __v5: u64;
let mut pfo_val__: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t0, sym this_cpu_off, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
__v5 = pfo_val__;
tcp_ptr__ = __v5;
tcp_ptr__ = tcp_ptr__.wrapping_add((unsafe { (*zone).per_cpu_pageset }) as u64);
__v4 = tcp_ptr__ as *mut per_cpu_pages;
__v2 = __v4;
_ret = __v2;
if spin_trylock(unsafe { core::ptr::addr_of_mut!((*_ret).lock) }) == 0 {
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
_ret = core::ptr::null_mut::<per_cpu_pages>();
}
__v0 = _ret;
pcp = __v0;
if pcp.is_null() {
return core::ptr::null_mut::<page>();
}
// On allocation, reduce the number of pages that are batch freed.
// See nr_pcp_free() where free_factor is increased for subsequent
// frees.
unsafe {
(*pcp).free_count = (((*pcp).free_count as i32) >> 1i32) as i16;
}
let list: *mut list_head = unsafe { (*pcp).lists.as_mut_ptr() as *mut list_head }.wrapping_offset(order_to_pindex(migratetype, order as i32) as isize);
let page: *mut page = __rmqueue_pcplist(zone, order, migratetype, alloc_flags, pcp, list);
spin_unlock(unsafe { core::ptr::addr_of_mut!((*pcp).lock) });
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
if !page.is_null() {
__count_vm_events(4u32.wrapping_add(page_zonenum(page as *const page) as u32) as u32, (1i32 << order) as i64);
zone_statistics(preferred_zone, zone, 1i64);
}
page
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_1007: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_1008: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

// Allocate a page from the given zone.
// Use pcplists for THP or "cheap" high-order allocations.
/// Do not instrument rmqueue() with KMSAN. This function may call
/// __msan_poison_alloca() through a call to set_pfnblock_migratetype().
/// If __msan_poison_alloca() attempts to allocate pages for the stack depot, it
/// may call rmqueue() again, which will result in a deadlock.
fn rmqueue(preferred_zone: *mut zone, zone: *mut zone, order: u32, gfp_flags: u32, alloc_flags: u32, migratetype: i32) -> *mut page {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut page: core::mem::MaybeUninit<*mut page> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut page = core::ptr::addr_of_mut!(page) as *mut *mut page;
unsafe {
'__join_5: {
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
if likely!(pcp_allowed_order(order)) != 0 {
*__slate_slot_1 = rmqueue_pcplist(preferred_zone, zone, order, migratetype, alloc_flags);
if likely__6!(*__slate_slot_1) != 0 {
break '__join_5;
}
}
*__slate_slot_1 = rmqueue_buddy(preferred_zone, zone, order, alloc_flags, migratetype);
}
if alloc_flags & 2048 != 0 {
__v0 = unlikely__2!(test_bit__4!(0i32, core::ptr::addr_of_mut!((*zone).flags))) != 0;
} else {
__v0 = false;
}
if __v0 {
clear_bit(0i64, core::ptr::addr_of_mut!((*zone).flags) as *mut u64);
wakeup_kswapd(zone, 0u32, 0i32, zone_idx!(zone) as u32);
}
// Separate test+clear to avoid unnecessary atomics
VM_BUG_ON_PAGE!();
return *__slate_slot_1;
}
}

/// Reserve the pageblock(s) surrounding an allocation request for
/// exclusive use of high-order atomic allocations if there are no
/// empty page blocks that contain a page with a suitable order
fn reserve_highatomic_pageblock(page: *mut page, order: i32, zone: *mut zone) {
let mut mt: i32 = 0;
let mut max_managed: u64 = 0;
// The number reserved as: minimum is 1 pageblock, maximum is
// roughly 1% of a zone. But if 1% of a zone falls below a
// pageblock size, then don't reserve any pageblocks.
// Check is race-prone but harmless.
if zone_managed_pages(zone as *const zone) / 100 < 1u64 << MIN_T!(u32, 9i32) {
return;
}
max_managed = ALIGN!(zone_managed_pages(zone as *const zone) / 100, 512u64);
if (unsafe { (*zone).nr_reserved_highatomic }) >= max_managed {
return;
}
let mut __UNIQUE_ID_guard_1009: __SlateRecord_spinlock_h_642 = class_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!((*zone).lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_1009) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_spinlock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_1010: *mut __SlateRecord_spinlock_h_642 = ((unsafe { core::ptr::addr_of_mut!((*zone).lock) }) as u64) as *mut __SlateRecord_spinlock_h_642;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_1010) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
// Recheck the nr_reserved_highatomic limit under the lock
if (unsafe { (*zone).nr_reserved_highatomic }) >= max_managed {
return;
}
// Yoink!
mt = (get_pageblock_migratetype__2!(page) as u32) as i32;
// Only reserve normal pageblocks (i.e., they can merge with others)
if !migratetype_is_mergeable(mt) {
return;
}
if (order as u32) < MIN_T!(u32, 9i32) {
if move_freepages_block(zone, page, mt, 3i32) == 1i32.wrapping_neg() {
return;
}
let __v3: u64 = unsafe { (*zone).nr_reserved_highatomic };
let __v4: u64 = __v3.wrapping_add(1u64 << MIN_T!(u32, 9i32));
unsafe {
(*zone).nr_reserved_highatomic = __v4;
}
} else {
change_pageblock_range(page, order, 3i32);
let __v6: u64 = unsafe { (*zone).nr_reserved_highatomic };
unsafe {
(*zone).nr_reserved_highatomic = __v6.wrapping_add((1i32 << order) as u64);
}
}
}

// Used when an allocation is about to fail under memory pressure. This
// potentially hurts the reliability of high-order allocations when under
// intense memory pressure but failed atomic allocations should be easier
// to recover from than an OOM.
//
// If @force is true, try to unreserve pageblocks even though highatomic
// pageblock is exhausted.
fn unreserve_highatomic_pageblock(ac: *const alloc_context, force: bool) -> bool {
let zonelist: *mut zonelist = unsafe { (*ac).zonelist };
let mut z: *mut zoneref = unsafe { core::mem::zeroed() };
let mut zone: *mut zone = unsafe { core::mem::zeroed() };
let mut order: i32 = 0;
let mut ret: i32 = 0;
z = first_zones_zonelist(zonelist, unsafe { (*ac).highest_zoneidx }, unsafe { (*ac).nodemask });
zone = zonelist_zone(z);
while !zone.is_null() {
// Preserve at least one pageblock unless memory pressure
// is really high.
if !(!force && (unsafe { (*zone).nr_reserved_highatomic }) <= 1u64 << MIN_T!(u32, 9i32)) {
let mut __UNIQUE_ID_guard_1011: __SlateRecord_spinlock_h_642 = class_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!((*zone).lock) });
let __slate_cleanup_ptr_2 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_1011) as *mut _;
let __slate_cleanup_2 = ::slate_support::Guard(&mut || class_spinlock_irqsave_destructor(__slate_cleanup_ptr_2 as *mut _));
let mut __UNIQUE_ID_unlock_1012: *mut __SlateRecord_spinlock_h_642 = ((unsafe { core::ptr::addr_of_mut!((*zone).lock) }) as u64) as *mut __SlateRecord_spinlock_h_642;
let __slate_cleanup_ptr_3 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_1012) as *mut _;
let __slate_cleanup_3 = ::slate_support::Guard(&mut || __class_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_3 as *mut _));
order = 0;
while order < 11i32 {
let area: *mut free_area = unsafe { (*zone).free_area.as_mut_ptr() as *mut free_area }.wrapping_offset(order as isize);
let mut size: u64 = 0;
let page: *mut page = get_page_from_free_area(area, 3i32);
if !page.is_null() {
let mut __v5: u64;
let __UNIQUE_ID_x__1013: u64 = 1u64 << MIN_T!(u32, 9i32);
let __UNIQUE_ID_y__1014: u64 = 1u64 << order;
while false {
}
__v5 = if __UNIQUE_ID_x__1013 > __UNIQUE_ID_y__1014 { __UNIQUE_ID_x__1013 } else { __UNIQUE_ID_y__1014 };
size = __v5;
// It should never happen but changes to
// locking could inadvertently allow a per-cpu
// drain to add pages to MIGRATE_HIGHATOMIC
// while unreserving so be safe and watch for
// underflows.
let mut __v7: i64;
{
let __ret_warn_on: i32 = (size > unsafe { (*zone).nr_reserved_highatomic }) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("1016: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1016b - ., 3; .popsection\n# {0}", const 1016 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_unreserve_highatomic_pageblock_8, sym __slate_asm_string_unreserve_highatomic_pageblock_9, const 3533 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("1017: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1017b - ., 4; .popsection\n# {0}", const 1017 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v7 = (__ret_warn_on != 0) as i64;
}
if __v7 != 0 {
size = unsafe { (*zone).nr_reserved_highatomic };
}
let __v13: u64 = unsafe { (*zone).nr_reserved_highatomic };
unsafe {
(*zone).nr_reserved_highatomic = __v13.wrapping_sub(size);
}
// Convert to ac->migratetype and avoid the normal
// pageblock stealing heuristics. Minimally, the caller
// is doing the work and needs the pages. More
// importantly, if the block was always converted to
// MIGRATE_UNMOVABLE or another type then the number
// of pageblocks that cannot be completely freed
// may increase.
if (order as u32) < MIN_T!(u32, 9i32) {
ret = move_freepages_block(zone, page, 3i32, unsafe { (*ac).migratetype });
} else {
move_to_free_list(page, zone, order as u32, 3i32, unsafe { (*ac).migratetype });
change_pageblock_range(page, order, unsafe { (*ac).migratetype });
ret = 1;
}
// Reserving the block(s) already succeeded,
// so this should not fail on zone boundaries.
let mut __v15: i64;
let __ret_warn_on: i32 = (ret == 1i32.wrapping_neg()) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("1018: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1018b - ., 3; .popsection\n# {0}", const 1018 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_unreserve_highatomic_pageblock_16, sym __slate_asm_string_unreserve_highatomic_pageblock_17, const 3562 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("1019: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1019b - ., 4; .popsection\n# {0}", const 1019 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v15 = (__ret_warn_on != 0) as i64;
if ret > 0 {
return ret != 0;
}
}
order = order.wrapping_add(1);
}
}
let __v22: *mut zoneref = z;
let __v23: *mut zoneref = __v22.wrapping_offset(1 as isize);
z = __v23;
z = next_zones_zonelist(__v23, unsafe { (*ac).highest_zoneidx }, unsafe { (*ac).nodemask });
zone = zonelist_zone(z);
}
false
}

fn __zone_watermark_unusable_free(z: *mut zone, order: u32, alloc_flags: u32) -> i64 {
let mut unusable_free: i64 = (1i32 << order).wrapping_sub(1) as i64;
// If the caller does not have rights to reserves below the min
// watermark then subtract the free pages reserved for highatomic.
if likely!(alloc_flags & 568u32 == 0) != 0 {
let mut __v0: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1020() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*z).nr_free_highatomic) as *const u64) };
let __v2: u64 = __v0;
unusable_free = (unusable_free as u64).wrapping_add(__v2) as i64;
}
unusable_free
}

/// Return true if free base pages are above 'mark'. For high-order checks it
/// will return true of the order-0 watermark is reached and there is at least
/// one free page of a suitable size. Checking now avoids taking the zone lock
/// to check in the allocation paths if no pages are free.
#[unsafe(no_mangle)]
extern "C" fn __zone_watermark_ok(z: *mut zone, order: u32, mark: u64, highest_zoneidx: i32, alloc_flags: u32, mut free_pages: i64) -> bool {
let mut min: i64 = mark as i64;
let mut o: i32 = 0;
// free_pages may go negative - that's OK
let __v0: i64 = __zone_watermark_unusable_free(z, order, alloc_flags);
free_pages = free_pages.wrapping_sub(__v0);
if unlikely__3!(alloc_flags & 568u32) != 0 {
// __GFP_HIGH allows access to 50% of the min reserve as well
// as OOM.
if alloc_flags & 32 != 0 {
min = min.wrapping_sub(min / 2);
// Non-blocking allocations (e.g. GFP_ATOMIC) can
// access more reserves than just __GFP_HIGH. Other
// non-blocking allocations requests such as GFP_NOWAIT
// or (GFP_KERNEL & ~__GFP_DIRECT_RECLAIM) do not get
// access to the min reserve.
if alloc_flags & 16 != 0 {
min = min.wrapping_sub(min / 4);
}
}
// OOM victims can try even harder than the normal reserve
// users on the grounds that it's definitely going to be in
// the exit path shortly and free memory. Any allocation it
// makes during the free path will be small and short-lived.
if alloc_flags & 8 != 0 {
min = min.wrapping_sub(min / 2);
}
}
// Check watermarks for an order-0 allocation request. If these
// are not met, then a high-order request also cannot go ahead
// even if a suitable page happened to be free.
if free_pages <= min.wrapping_add(unsafe { *((*z).lowmem_reserve.as_mut_ptr() as *mut i64).wrapping_offset(highest_zoneidx as isize) }) {
return false;
}
// If this is an order-0 request then the watermark is fine
if order == 0 {
return true;
}
// For a high-order request, check at least one suitable page is free
o = order as i32;
while o < 11i32 {
let area: *mut free_area = unsafe { (*z).free_area.as_mut_ptr() as *mut free_area }.wrapping_offset(o as isize);
if (unsafe { (*area).nr_free }) != 0 {
for mt in 0i32..3i32 {
if !free_area_empty(area, mt) {
return true;
}
}
let __v13: bool = alloc_flags & 520u32 != 0 && !free_area_empty(area, 3i32);
if __v13 {
return true;
}
}
o = o.wrapping_add(1);
}
false
}

#[unsafe(no_mangle)]
extern "C" fn zone_watermark_ok(z: *mut zone, order: u32, mark: u64, highest_zoneidx: i32, alloc_flags: u32) -> bool {
__zone_watermark_ok(z, order, mark, highest_zoneidx, alloc_flags, zone_page_state(z, 0u32) as i64)
}

fn zone_watermark_fast(z: *mut zone, order: u32, mut mark: u64, highest_zoneidx: i32, alloc_flags: u32, gfp_mask: u32) -> bool {
let free_pages: i64 = zone_page_state(z, 0u32) as i64;
// Fast check for order-0 only. If this fails then the reserves
// need to be calculated.
if order == 0 {
let mut usable_free: i64 = free_pages;
let reserved: i64 = __zone_watermark_unusable_free(z, 0u32, alloc_flags);
// reserved may over estimate high-atomic reserves.
let mut __v0: i64;
let __UNIQUE_ID_x__1021: i64 = usable_free;
let __UNIQUE_ID_y__1022: i64 = reserved;
loop {
if __types_ok__6!(__UNIQUE_ID_x__1021, __UNIQUE_ID_y__1022) == 0 {
unsafe { __compiletime_assert_1023() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__1021 < __UNIQUE_ID_y__1022 { __UNIQUE_ID_x__1021 } else { __UNIQUE_ID_y__1022 };
let __v2: i64 = __v0;
usable_free = usable_free.wrapping_sub(__v2);
if (usable_free as u64) > mark.wrapping_add((unsafe { *((*z).lowmem_reserve.as_mut_ptr() as *mut i64).wrapping_offset(highest_zoneidx as isize) }) as u64) {
return true;
}
}
if __zone_watermark_ok(z, order, mark, highest_zoneidx, alloc_flags, free_pages) {
return true;
}
// Ignore watermark boosting for __GFP_HIGH order-0 allocations
// when checking the min watermark. The min watermark is the
// point where boosting is ignored so that kswapd is woken up
// when below the low watermark.
if unlikely!(order == 0 && alloc_flags & 32 != 0 && (unsafe { (*z).watermark_boost }) != 0 && alloc_flags & 3u32 == 0) != 0 {
mark = unsafe { *((*z)._watermark.as_mut_ptr() as *mut u64).wrapping_offset(0 as isize) };
return __zone_watermark_ok(z, order, mark, highest_zoneidx, alloc_flags, free_pages);
}
false
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".data..read_mostly")]
static mut node_reclaim_distance: i32 = 30;

fn zone_allows_reclaim(local_zone: *mut zone, zone: *mut zone) -> bool {
node_distance!(zone_to_nid(local_zone as *const zone), zone_to_nid(zone as *const zone)) <= unsafe { node_reclaim_distance }
}

// The restriction on ZONE_DMA32 as being a suitable zone to use to avoid
// fragmentation is subtle. If the preferred zone was HIGHMEM then
// premature use of a lower zone may cause lowmem pressure problems that
// are worse than fragmentation. If the next zone is ZONE_DMA then it is
// probably too small. It only makes sense to spread allocations to avoid
// fragmentation between the Normal and DMA32 zones.
fn alloc_flags_nofragment(mut zone: *mut zone, gfp_mask: u32) -> u32 {
let mut alloc_flags: u32 = 0;
if gfp_mask & (BIT!(11i32) as u32) != 0 {
alloc_flags |= 2048;
}
if (unsafe { defrag_mode }) != 0 {
alloc_flags |= 256;
return alloc_flags;
}
if zone.is_null() {
return alloc_flags;
}
if zone_idx!(zone) != 2 {
return alloc_flags;
}
// If ZONE_DMA32 exists, assume it is the one after ZONE_NORMAL and
// the pointer is within zone->zone_pgdat->node_zones[]. Also assume
// on UMA that if Normal is populated then so is DMA32.
loop {
if 1i32 != 1 {
unsafe { __compiletime_assert_1024() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let __v5: bool = if (unsafe { nr_online_nodes }) > 1 {
let __v6: *mut zone = zone;
let __v7: *mut zone = __v6.wrapping_offset(-(1 as isize));
zone = __v7;
!populated_zone(__v7 as *const zone)
} else {
false
};
if __v5 {
return alloc_flags;
}
alloc_flags |= 256;
alloc_flags
}

/// Must be called after current_gfp_context() which can change gfp_mask
fn alloc_flags_cma(gfp_mask: u32) -> u32 {
0
}

// get_page_from_freelist goes through the zonelist trying to allocate
// a page.
fn get_page_from_freelist(gfp_mask: u32, order: u32, mut alloc_flags: i32, ac: *const alloc_context) -> *mut page {
let mut __v5: bool = unsafe { core::mem::zeroed() };
// If kswapd is already active on a node, keep looking
// for other nodes that might be idle. This can happen
// if another process has NUMA bindings and is causing
// kswapd wakeups on only some nodes. Avoid accidental
// "node_reclaim_mode"-like behavior in this case.
let mut __v6: bool = unsafe { core::mem::zeroed() };
let mut local_nid: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut i32 = core::ptr::addr_of_mut!(local_nid) as *mut i32;
let mut __v10: bool = unsafe { core::mem::zeroed() };
let mut __v11: bool = unsafe { core::mem::zeroed() };
let mut mark: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut u64 = core::ptr::addr_of_mut!(mark) as *mut u64;
let mut page: core::mem::MaybeUninit<*mut page> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut *mut page = core::ptr::addr_of_mut!(page) as *mut *mut page;
let mut skipped_kswapd_nodes: bool = unsafe { core::mem::zeroed() };
let mut skip_kswapd_nodes: bool = unsafe { core::mem::zeroed() };
let mut no_fallback: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut bool = core::ptr::addr_of_mut!(no_fallback) as *mut bool;
let mut last_pgdat_dirty_ok: bool = unsafe { core::mem::zeroed() };
let mut last_pgdat: *mut pglist_data = unsafe { core::mem::zeroed() };
let mut zone: core::mem::MaybeUninit<*mut zone> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut *mut zone = core::ptr::addr_of_mut!(zone) as *mut *mut zone;
let mut z: core::mem::MaybeUninit<*mut zoneref> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut *mut zoneref = core::ptr::addr_of_mut!(z) as *mut *mut zoneref;
unsafe {
let __slate_auto_bytes_2 = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
last_pgdat = core::ptr::null_mut::<pglist_data>();
last_pgdat_dirty_ok = false;
let __slate_auto_bytes_4 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 1usize);
skip_kswapd_nodes = nr_online_nodes > 1;
skipped_kswapd_nodes = false;
'__loop_47: loop {
*__slate_slot_16 = alloc_flags & 256 != 0;
// Scan zonelist, looking for a zone with enough free.
// See also cpuset_current_node_allowed() comment in kernel/cgroup/cpuset.c.
*__slate_slot_20 = (*ac).preferred_zoneref;
*__slate_slot_19 = zonelist_zone(*__slate_slot_20);
'__join_33: {
while !(*__slate_slot_19).is_null() {
{
let __slate_auto_bytes_1 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
if cpusets_enabled() && alloc_flags & 64 != 0 {
__v11 = !__cpuset_zone_allowed(*__slate_slot_19, gfp_mask);
} else {
__v11 = false;
}
}
'__join_6: {
if !__v11 {
// When allocating a page cache page for writing, we
// want to get it from a node that is within its dirty
// limit, such that no single node holds more than its
// proportional share of globally allowed dirty pages.
// The dirty limits take into account the node's
// lowmem reserves and high watermark so that kswapd
// should be able to balance it without having to
// write pages from its LRU list.
//
// XXX: For now, allow allocations to potentially
// exceed the per-node dirty limit in the slowpath
// (spread_dirty_pages unset) before going into reclaim,
// which is important when on a NUMA setup the allowed
// nodes are together not big enough to reach the
// global limit.  The proper fix for these situations
// will require awareness of nodes in the
// dirty-throttling and the flusher threads.
if (*ac).spread_dirty_pages {
if last_pgdat != (*(*__slate_slot_19)).zone_pgdat {
last_pgdat = (*(*__slate_slot_19)).zone_pgdat;
last_pgdat_dirty_ok = node_dirty_ok((*(*__slate_slot_19)).zone_pgdat);
}
if !last_pgdat_dirty_ok {
break '__join_6;
}
}
if *__slate_slot_16 && defrag_mode == 0 && nr_online_nodes > 1 {
__v10 = *__slate_slot_19 != zonelist_zone((*ac).preferred_zoneref);
} else {
__v10 = false;
}
if __v10 {
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
// If moving to a remote node, retry but allow
// fragmenting fallbacks. Locality is more important
// than fragmentation avoidance.
*__slate_slot_9 = zonelist_node_idx((*ac).preferred_zoneref as *const zoneref);
if zone_to_nid(*__slate_slot_19 as *const zone) != *__slate_slot_9 {
break '__join_33;
}
}
if skip_kswapd_nodes {
__v6 = waitqueue_active(core::ptr::addr_of_mut!((*(*(*__slate_slot_19)).zone_pgdat).kswapd_wait)) == 0;
} else {
__v6 = false;
}
if __v6 {
skipped_kswapd_nodes = true;
} else {
'__join_11: {
cond_accept_memory(*__slate_slot_19, order, alloc_flags);
// Detect whether the number of free pages is below high
// watermark.  If so, we will decrease pcp->high and free
// PCP pages in free path to reduce the possibility of
// premature page reclaiming.  Detection is done here to
// avoid to do that in hotter free path.
if test_bit__4!(2i32, core::ptr::addr_of_mut!((*(*__slate_slot_19)).flags)) == 0 {
*__slate_slot_12 = high_wmark_pages(*__slate_slot_19 as *const zone);
if zone_watermark_fast(*__slate_slot_19, order, *__slate_slot_12, ((*ac).highest_zoneidx as u32) as i32, alloc_flags as u32, gfp_mask) {
break '__join_11;
} else {
set_bit(2i64, core::ptr::addr_of_mut!((*(*__slate_slot_19)).flags) as *mut u64);
}
}
*__slate_slot_12 = wmark_pages(*__slate_slot_19 as *const zone, (alloc_flags & 3i32) as u32);
if !zone_watermark_fast(*__slate_slot_19, order, *__slate_slot_12, ((*ac).highest_zoneidx as u32) as i32, alloc_flags as u32, gfp_mask) {
if !cond_accept_memory(*__slate_slot_19, order, alloc_flags) {
// Watermark failed for this zone, but see if we can
// grow this zone if it contains deferred pages.
if deferred_pages_enabled() {
if _deferred_grow_zone(*__slate_slot_19, order) {
break '__join_11;
}
}
// Checked here to keep the fast path fast
while false {
}
if alloc_flags & 4 == 0 {
if !node_reclaim_enabled() {
__v5 = true;
} else {
__v5 = !zone_allows_reclaim(zonelist_zone((*ac).preferred_zoneref), *__slate_slot_19);
}
if __v5 {
break '__join_6;
} else {
if node_reclaim((*(*__slate_slot_19)).zone_pgdat, gfp_mask, order) == 0 {
break '__join_6;
} else {
// did we reclaim enough
if !zone_watermark_ok(*__slate_slot_19, order, *__slate_slot_12, ((*ac).highest_zoneidx as u32) as i32, alloc_flags as u32) {
break '__join_6;
}
}
}
}
}
}
}
loop {
*__slate_slot_13 = rmqueue(zonelist_zone((*ac).preferred_zoneref), *__slate_slot_19, order, gfp_mask, alloc_flags as u32, (*ac).migratetype);
if !(*__slate_slot_13).is_null() {
prep_new_page(*__slate_slot_13, order, gfp_mask, alloc_flags as u32);
return *__slate_slot_13;
} else {
if !cond_accept_memory(*__slate_slot_19, order, alloc_flags) {
// Try again if zone has deferred pages
if deferred_pages_enabled() {
if !_deferred_grow_zone(*__slate_slot_19, order) {
break;
}
} else {
break;
}
}
}
}
}
}
}
let __v4: *mut zoneref = *__slate_slot_20;
let __v3: *mut zoneref = __v4.wrapping_offset(1 as isize);
*__slate_slot_20 = __v3;
*__slate_slot_20 = next_zones_zonelist(__v3, (*ac).highest_zoneidx, (*ac).nodemask);
let __v2: *mut zone = zonelist_zone(*__slate_slot_20);
*__slate_slot_19 = __v2;
}
// If we skipped over nodes with active kswapds and found no
// idle nodes, retry and place anywhere the watermarks permit.
if skip_kswapd_nodes && skipped_kswapd_nodes {
skip_kswapd_nodes = false;
continue '__loop_47;
} else {
// It's possible on a UMA machine to get through all zones that are
// fragmented. If avoiding fragmentation, reset and try again.
if *__slate_slot_16 && defrag_mode == 0 {
let __v0: i32 = alloc_flags & !256;
alloc_flags = __v0;
continue '__loop_47;
} else {
break '__loop_47;
}
}
}
let __v7: i32 = alloc_flags & !256;
alloc_flags = __v7;
}
return core::ptr::null_mut::<page>();
}
}

fn warn_alloc_show_mem(gfp_mask: u32, nodemask: *const __SlateRecord_nodemask_types_h_17) {
let mut filter: u32 = 1;
// This documents exceptions given to allocations in certain
// contexts that are allowed to allocate outside current's set
// of allowed nodes.
if gfp_mask & (BIT!(19i32) as u32) == 0 {
let __v0: bool = tsk_is_oom_victim(get_current()) || (unsafe { (*get_current()).flags }) & 2052u32 != 0;
if __v0 {
let __v2: u32 = filter & !1;
filter = __v2;
}
}
if !in_task!() || gfp_mask & (BIT!(10i32) as u32) == 0 {
let __v4: u32 = filter & !1;
filter = __v4;
}
unsafe { __show_mem(filter, nodemask, (gfp_zone(gfp_mask) as u32) as i32) };
mem_cgroup_show_protected_memory(core::ptr::null_mut::<mem_cgroup>());
}

#[unsafe(no_mangle)]
unsafe extern "C" fn warn_alloc(mut gfp_mask: u32, nodemask: *const __SlateRecord_nodemask_types_h_17, fmt: *const u8, mut __va_args: ...) {
let mut vaf: va_format = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(vaf) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let mut args: core::ffi::VaList<'static> = unsafe { core::mem::zeroed() };
let __v0: bool = gfp_mask & (BIT!(13i32) as u32) != 0 || __ratelimit__8!(unsafe { core::ptr::addr_of_mut!(nopage_rs) }) == 0;
let __v1: bool = __v0 || gfp_mask & (BIT!(0i32) as u32) != 0 && !has_managed_dma();
if __v1 {
return;
}
args = unsafe { core::mem::transmute::<core::ffi::VaList<'_>, core::ffi::VaList<'static>>(__va_args.clone()) };
vaf.fmt = fmt;
vaf.va = core::ptr::addr_of_mut!(args);
let mut __v3: i32;
while false {
}
__v3 = unsafe { _printk(c"\x014%s: %pV, mode:%#x(%pGg), nodemask=%*pbl".as_ptr() as *const u8, (*get_current()).comm.as_mut_ptr() as *mut u8, core::ptr::addr_of_mut!(vaf), gfp_mask, core::ptr::addr_of_mut!(gfp_mask), __nodemask_pr_numnodes(nodemask), __nodemask_pr_bits(nodemask)) };
va_end!();
unsafe { cpuset_print_current_mems_allowed() };
let mut __v5: i32;
while false {
}
__v5 = unsafe { _printk(c"\x01c\n".as_ptr() as *const u8) };
unsafe { dump_stack() };
warn_alloc_show_mem(gfp_mask, nodemask);
}

static mut nopage_rs: ratelimit_state = {
let mut __t0: ratelimit_state = unsafe { core::mem::zeroed() };
__t0.lock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0.interval = 10000i32;
__t0.burst = 1i32;
__t0.flags = 0u32;
__t0
};

fn __alloc_pages_cpuset_fallback(gfp_mask: u32, order: u32, alloc_flags: u32, ac: *const alloc_context) -> *mut page {
let mut page: *mut page = unsafe { core::mem::zeroed() };
page = get_page_from_freelist(gfp_mask, order, (alloc_flags | 64) as i32, ac);
// fallback to ignore cpuset restriction if our nodes
// are depleted
if page.is_null() {
page = get_page_from_freelist(gfp_mask, order, alloc_flags as i32, ac);
}
page
}

fn __alloc_pages_may_oom(gfp_mask: u32, order: u32, ac: *const alloc_context, did_some_progress: &mut u64) -> *mut page {
let mut __ret_warn_once: i32 = unsafe { core::mem::zeroed() };
// Exhausted what can be done so it's blame time
let mut __v5: bool = unsafe { core::mem::zeroed() };
let mut page: core::mem::MaybeUninit<*mut page> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut page = core::ptr::addr_of_mut!(page) as *mut *mut page;
let mut oc: core::mem::MaybeUninit<oom_control> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut oom_control = core::ptr::addr_of_mut!(oc) as *mut oom_control;
unsafe {
core::ptr::write(__slate_slot_7, {
let mut __t0: oom_control = core::mem::zeroed();
__t0.zonelist = (*ac).zonelist;
__t0.nodemask = (*ac).nodemask;
__t0.memcg = core::ptr::null_mut::<mem_cgroup>();
__t0.gfp_mask = gfp_mask;
__t0.order = order as i32;
__t0
});
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*(did_some_progress as *mut u64) = 0;
// Acquire the oom lock.  If that fails, somebody else is
// making progress for us.
if mutex_trylock(core::ptr::addr_of_mut!(oom_lock)) == 0 {
*(did_some_progress as *mut u64) = 1;
schedule_timeout_uninterruptible(1i64);
return core::ptr::null_mut::<page>();
} else {
// Go through the zonelist yet one more time, keep very high watermark
// here, this is only to catch a parallel oom killing, we must fail if
// we're still under heavy pressure. But make sure that this reclaim
// attempt shall not depend on __GFP_DIRECT_RECLAIM && !__GFP_NORETRY
// allocation which will never fail due to oom_lock already held.
*__slate_slot_6 = get_page_from_freelist((gfp_mask | (BIT!(20i32) as u32)) & !(BIT!(10i32) as u32), order, ((*ac).alloc_flags | 2 | 64) as i32, ac);
if (*__slate_slot_6).is_null() {
// Coredumps can quickly deplete all memory reserves
if (*get_current()).flags & 512 == 0 {
// The OOM killer will not help higher order allocs
if order <= 3 {
// We have already exhausted all our reclaim opportunities without any
// success so it is time to admit defeat. We will skip the OOM killer
// because it is very likely that the caller has a more reasonable
// fallback than shooting a random task.
//
// The OOM killer may not free memory on a specific node.
if gfp_mask & ((BIT!(14i32) as u32) | (BIT!(21i32) as u32)) == 0 {
// The OOM killer does not needlessly kill tasks for lowmem
if ((*ac).highest_zoneidx as u32) >= 2 {
if !pm_suspended_storage() {
// XXX: GFP_NOFS allocations should rather fail than rely on
// other request to make a forward progress.
// We are in an unfortunate situation where out_of_memory cannot
// do much for this context but let's try it to at least get
// access to memory reserved if the current task is killed (see
// out_of_memory). Once filesystems are ready to handle allocation
// failures more gracefully we should just bail out here.
if out_of_memory(core::ptr::addr_of_mut!(*__slate_slot_7)) {
__v5 = true;
} else {
__ret_warn_once = (gfp_mask & (BIT!(15i32) as u32) != 0) as i32;
if unlikely!(gfp_mask & (BIT!(13i32) as u32) == 0 && __ret_warn_once != 0 && !__warned__4114) != 0 {
__warned__4114 = true;
let __ret_warn_on: i32 = true as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("1026: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1026b - ., 3; .popsection\n# {0}", const 1026 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___alloc_pages_may_oom_8, sym __slate_asm_string___alloc_pages_may_oom_9, const 4114 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("1027: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1027b - ., 4; .popsection\n# {0}", const 1027 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
}
__v5 = __ret_warn_once != 0;
}
if __v5 {
*(did_some_progress as *mut u64) = 1;
// Help non-failing allocations by giving them access to memory
// reserves
if gfp_mask & (BIT!(15i32) as u32) != 0 {
*__slate_slot_6 = __alloc_pages_cpuset_fallback(gfp_mask, order, (*ac).alloc_flags | 4, ac);
}
}
}
}
}
}
}
}
mutex_unlock(core::ptr::addr_of_mut!(oom_lock));
return *__slate_slot_6;
}
}
}

#[unsafe(link_section = ".data..once")]
static mut __warned__4114: bool = unsafe { core::mem::zeroed() };

// Maximum number of compaction retries with a progress before OOM
// killer is consider as the only way to move forward.
// Try memory compaction for high-order allocations before reclaim
fn __alloc_pages_direct_compact(gfp_mask: u32, order: u32, alloc_flags: u32, ac: *const alloc_context, prio: u32, compact_result: &mut u32) -> *mut page {
let mut page: *mut page = core::ptr::null_mut::<page>();
let mut pflags: u64 = 0;
let mut noreclaim_flag: u32 = 0;
let mut capc: capture_control = capture_control { zone: core::ptr::null_mut::<zone>(), migratetype: unsafe { (*ac).migratetype }, order: order as i32, page: core::ptr::null_mut::<page>() };
let mut compact_order: i32 = order as i32;
// If fallbacks are not permitted (defrag_mode), we either
// need to reclaim space in a block of matching type, or clear
// out an entire block to allow __rmqueue_claim() to convert.
//
// Reclaim by itself is primarily freeing space in movable
// blocks, since that's where the LRU pages live. So this
// works for movable requests, but not for others.
//
// For those, promote the order to help make blocks, instead
// of spinning in reclaim alone unproductively.
if alloc_flags & 256 != 0 && (unsafe { (*ac).migratetype }) != 1 {
let mut __v0: u32;
let __UNIQUE_ID_x__1028: u32 = order;
let __UNIQUE_ID_y__1029: u32 = MIN_T!(u32, 9i32);
while false {
}
__v0 = if __UNIQUE_ID_x__1028 > __UNIQUE_ID_y__1029 { __UNIQUE_ID_x__1028 } else { __UNIQUE_ID_y__1029 };
compact_order = __v0 as i32;
}
if compact_order == 0 {
return core::ptr::null_mut::<page>();
}
psi_memstall_enter(unsafe { &(*core::ptr::addr_of_mut!(pflags)) });
delayacct_compact_start();
fs_reclaim_acquire(gfp_mask);
noreclaim_flag = memalloc_noreclaim_save();
// Make sure the structs are really initialized before we expose the
// capture control, in case we are interrupted and the interrupt handler
// frees a page.
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
loop {
while false {
}
__WRITE_ONCE__32!((*get_current()).capture_control, core::ptr::addr_of_mut!(capc));
if true {
break;
}
}
unsafe {
*(compact_result as *mut u32) = try_to_compact_pages(gfp_mask, compact_order as u32, alloc_flags, ac, prio, core::ptr::addr_of_mut!(capc));
}
// Make sure we hide capture control first before we read the captured
// page pointer, otherwise an interrupt could free and capture a page
// and we would leak it.
loop {
while false {
}
__WRITE_ONCE__33!((*get_current()).capture_control);
if true {
break;
}
}
let mut __v8: *mut page;
while false {
}
__v8 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!(capc.page) as *const *mut page) };
page = __v8;
// Technically, it is also possible that compaction is skipped but
// the page is still captured out of luck(IRQ came and freed the page).
// Returning COMPACT_SUCCESS in such cases helps in properly accounting
// the COMPACT[STALL|FAIL] when compaction is skipped.
if !page.is_null() {
unsafe {
*(compact_result as *mut u32) = 8;
}
}
memalloc_noreclaim_restore(noreclaim_flag);
fs_reclaim_release(gfp_mask);
psi_memstall_leave(unsafe { &(*core::ptr::addr_of_mut!(pflags)) });
delayacct_compact_end();
if ((unsafe { *(compact_result as *mut u32) }) as u32) == 1 || ((unsafe { *(compact_result as *mut u32) }) as u32) == 2 {
return core::ptr::null_mut::<page>();
}
// At least in one zone compaction wasn't deferred or skipped, so let's
// count a compaction stall
count_vm_event(45u32);
// Prep a captured page if available
if !page.is_null() {
prep_new_page(page, order, gfp_mask, alloc_flags);
}
// Try get a page from the freelist if available
if page.is_null() {
page = get_page_from_freelist(gfp_mask, order, alloc_flags as i32, ac);
}
if !page.is_null() {
let zone: *mut zone = page_zone(page as *const page);
unsafe {
(*zone).compact_blockskip_flush = false;
}
unsafe { compaction_defer_reset(zone, compact_order, true) };
count_vm_event(47u32);
return page;
}
// It's bad if compaction run occurs and fails. The most likely reason
// is that pages exist, but not enough to satisfy watermarks.
count_vm_event(46u32);
cond_resched__74!();
core::ptr::null_mut::<page>()
}

fn should_compact_retry(gfp_mask: u32, ac: *mut alloc_context, order: i32, alloc_flags: i32, compact_result: u32, compact_priority: *mut u32, compaction_retries: *mut i32) -> bool {
let mut priority: u32 = unsafe { core::mem::zeroed() };
let mut retries: i32 = unsafe { core::mem::zeroed() };
let mut ret: bool = unsafe { core::mem::zeroed() };
let mut min_priority: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut i32 = core::ptr::addr_of_mut!(min_priority) as *mut i32;
let mut max_retries: i32 = unsafe { core::mem::zeroed() };
unsafe {
max_retries = 16;
let __slate_auto_bytes = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
ret = false;
retries = *compaction_retries;
priority = *compact_priority;
if order == 0 {
return false;
} else {
if fatal_signal_pending(get_current()) != 0 {
return false;
} else {
'__join_0: {
// Compaction was skipped due to a lack of free order-0
// migration targets. Continue if reclaim can help.
if (compact_result as u32) == 1 {
ret = compaction_zonelist_suitable(ac, order, alloc_flags, gfp_mask);
} else {
// Compaction managed to coalesce some page blocks, but the
// allocation failed presumably due to a race. Retry some.
if (compact_result as u32) == 8 {
// !costly requests are much more important than
// __GFP_RETRY_MAYFAIL costly ones because they are de
// facto nofail and invoke OOM killer to move on while
// costly can fail and users are ready to cope with
// that. 1/4 retries is rather arbitrary but we would
// need much more detailed feedback from compaction to
// make a better decision.
if order > 3 {
max_retries /= 4;
}
let __v4: i32 = *compaction_retries;
let __v3: i32 = __v4.wrapping_add(1);
*compaction_retries = __v3;
if __v3 <= max_retries {
ret = true;
break '__join_0;
}
}
// Compaction failed. Retry with increasing priority.
*__slate_slot_11 = if order > 3 { 1 } else { 0 };
if (*compact_priority as u32) > (*__slate_slot_11 as u32) {
*compact_priority = (*compact_priority as u32).wrapping_sub(1) as u32;
*compaction_retries = 0;
ret = true;
}
}
}
trace_compact_retry(order, priority, compact_result, retries, max_retries, ret);
return ret;
}
}
}
}

/// Zonelists may change due to hotplug during allocation. Detect when zonelists
/// have been rebuilt so allocation retries. Reader side does not lock and
/// retries the allocation if zonelist changes. Writer side is protected by the
/// embedded spin_lock.
static mut zonelist_update_seq: seqlock = seqlock { seqcount: seqcount_spinlock { seqcount: seqcount { sequence: 0u32 } }, lock: spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = unsafe { core::mem::zeroed() };
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0
} } };

fn zonelist_iter_begin() -> u32 {
if IS_ENABLED!() != 0 {
return read_seqbegin((unsafe { core::ptr::addr_of_mut!(zonelist_update_seq) }) as *const seqlock);
}
0
}

fn check_retry_zonelist(seq: u32) -> u32 {
if IS_ENABLED!() != 0 {
return read_seqretry((unsafe { core::ptr::addr_of_mut!(zonelist_update_seq) }) as *const seqlock, seq);
}
seq
}

/// Perform direct synchronous page reclaim
fn __perform_reclaim(gfp_mask: u32, order: u32, ac: *const alloc_context) -> u64 {
let mut noreclaim_flag: u32 = 0;
let mut progress: u64 = 0;
cond_resched__88!();
// We now go into synchronous reclaim
cpuset_memory_pressure_bump();
fs_reclaim_acquire(gfp_mask);
noreclaim_flag = memalloc_noreclaim_save();
progress = unsafe { try_to_free_pages((*ac).zonelist, order as i32, gfp_mask, (*ac).nodemask) };
memalloc_noreclaim_restore(noreclaim_flag);
fs_reclaim_release(gfp_mask);
cond_resched__3!();
progress
}

// The really slow allocator path where we enter direct reclaim
fn __alloc_pages_direct_reclaim(gfp_mask: u32, order: u32, alloc_flags: u32, ac: *const alloc_context, did_some_progress: *mut u64) -> *mut page {
let mut __UNIQUE_ID_y__1035: u32 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__1034: u32 = unsafe { core::mem::zeroed() };
let mut reclaim_order: i32 = unsafe { core::mem::zeroed() };
let mut drained: bool = unsafe { core::mem::zeroed() };
let mut pflags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut u64 = core::ptr::addr_of_mut!(pflags) as *mut u64;
let mut page: *mut page = unsafe { core::mem::zeroed() };
unsafe {
page = core::ptr::null_mut::<page>();
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
drained = false;
reclaim_order = order as i32;
// Match the slowpath compaction promotion in __alloc_pages_direct_compact
if alloc_flags & 256 != 0 && (*ac).migratetype != 1 {
__UNIQUE_ID_x__1034 = order;
__UNIQUE_ID_y__1035 = MIN_T!(u32, 9i32);
while false {
}
reclaim_order = (if __UNIQUE_ID_x__1034 > __UNIQUE_ID_y__1035 { __UNIQUE_ID_x__1034 } else { __UNIQUE_ID_y__1035 }) as i32;
}
psi_memstall_enter(&(*core::ptr::addr_of_mut!(*__slate_slot_5)));
*did_some_progress = __perform_reclaim(gfp_mask, reclaim_order as u32, ac);
if unlikely!(*did_some_progress == 0) == 0 {
loop {
page = get_page_from_freelist(gfp_mask, order, alloc_flags as i32, ac);
// If an allocation failed after direct reclaim, it could be because
// pages are pinned on the per-cpu lists or in high alloc reserves.
// Shrink them and try again
if page.is_null() && !drained {
unreserve_highatomic_pageblock(ac, false);
drain_all_pages(core::ptr::null_mut::<zone>());
drained = true;
} else {
break;
}
}
}
psi_memstall_leave(&(*core::ptr::addr_of_mut!(*__slate_slot_5)));
return page;
}
}

fn wake_all_kswapds(order: u32, gfp_mask: u32, ac: *const alloc_context) {
let mut z: *mut zoneref = unsafe { core::mem::zeroed() };
let mut zone: *mut zone = unsafe { core::mem::zeroed() };
let mut last_pgdat: *mut pglist_data = core::ptr::null_mut::<pglist_data>();
let highest_zoneidx: u32 = unsafe { (*ac).highest_zoneidx };
let reclaim_order: u32 = if (unsafe { defrag_mode }) != 0 {
let mut __v0: u32;
let __UNIQUE_ID_x__1037: u32 = order;
let __UNIQUE_ID_y__1038: u32 = MIN_T__2!(u32, 9i32, 10i32);
loop {
if __types_ok__5!(__UNIQUE_ID_x__1037, __UNIQUE_ID_y__1038) == 0 {
unsafe { __compiletime_assert_1039() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__1037 > __UNIQUE_ID_y__1038 { __UNIQUE_ID_x__1037 } else { __UNIQUE_ID_y__1038 };
__v0
} else {
order
};
z = first_zones_zonelist(unsafe { (*ac).zonelist }, highest_zoneidx, unsafe { (*ac).nodemask });
zone = zonelist_zone(z);
while !zone.is_null() {
if managed_zone(zone as *const zone) {
if last_pgdat != unsafe { (*zone).zone_pgdat } {
unsafe { wakeup_kswapd(zone, gfp_mask, reclaim_order as i32, highest_zoneidx) };
last_pgdat = unsafe { (*zone).zone_pgdat };
}
}
let __v4: *mut zoneref = z;
let __v5: *mut zoneref = __v4.wrapping_offset(1 as isize);
z = __v5;
z = next_zones_zonelist(__v5, highest_zoneidx, unsafe { (*ac).nodemask });
zone = zonelist_zone(z);
}
}

fn alloc_flags_nonblocking(gfp_mask: u32, order: u32) -> u32 {
let mut alloc_flags: u32 = 0;
if gfp_mask & (BIT!(10i32) as u32) != 0 {
return 0;
}
// Not worth trying to allocate harder for __GFP_NOMEMALLOC even
// if it can't schedule.
if gfp_mask & (BIT!(19i32) as u32) != 0 {
return 0;
}
alloc_flags |= 16;
if order > 0 && gfp_mask & (BIT!(5i32) as u32) != 0 {
alloc_flags |= 512;
}
alloc_flags
}

fn alloc_flags_slowpath(gfp_mask: u32, order: u32) -> u32 {
let mut alloc_flags: u32 = 64u32;
// The caller may dip into page reserves a bit more if the caller
// cannot run direct reclaim, or if the caller has realtime scheduling
// policy or is asking for __GFP_HIGH memory.  GFP_ATOMIC requests will
// set both ALLOC_NON_BLOCK and ALLOC_MIN_RESERVE(__GFP_HIGH).
if gfp_mask & (BIT!(5i32) as u32) != 0 {
alloc_flags |= 32;
}
if gfp_mask & (BIT!(11i32) as u32) != 0 {
alloc_flags |= 2048;
}
let __v4: u32 = alloc_flags_nonblocking(gfp_mask, order);
alloc_flags |= __v4;
if gfp_mask & (BIT!(10i32) as u32) == 0 {
// Ignore cpuset mems for non-blocking __GFP_HIGH (probably
// GFP_ATOMIC) rather than fail, see the comment for
// cpuset_current_node_allowed().
if alloc_flags & 32 != 0 {
let __v8: u32 = alloc_flags & 4294967231u32;
alloc_flags = __v8;
}
} else {
let __v9: bool = unlikely!(rt_or_dl_task(get_current())) != 0 && in_task!();
if __v9 {
alloc_flags |= 32;
}
}
let __v12: u32 = alloc_flags_cma(gfp_mask);
alloc_flags |= __v12;
if (unsafe { defrag_mode }) != 0 {
alloc_flags |= 256;
}
alloc_flags
}

fn oom_reserves_allowed(tsk: *mut task_struct) -> bool {
if !tsk_is_oom_victim(tsk) {
return false;
}
// !MMU doesn't have oom reaper so give access to memory reserves
// only to the thread with TIF_MEMDIE set
let __v0: bool = IS_ENABLED__2!() == 0 && test_thread_flag!(3i32) == 0;
if __v0 {
return false;
}
true
}

/// Distinguish requests which really need access to full memory
/// reserves from oom victims which can live with a portion of it
fn __gfp_pfmemalloc_flags(gfp_mask: u32) -> i32 {
if unlikely__3!(gfp_mask & (BIT!(19i32) as u32)) != 0 {
return 0;
}
if gfp_mask & (BIT!(17i32) as u32) != 0 {
return 4;
}
let __v0: bool = in_serving_softirq!() != 0 && (unsafe { (*get_current()).flags }) & 2048 != 0;
if __v0 {
return 4;
}
if in_interrupt!() == 0 {
if (unsafe { (*get_current()).flags }) & 2048 != 0 {
return 4;
} else {
if oom_reserves_allowed(get_current()) {
return 8;
}
}
}
0
}

#[unsafe(no_mangle)]
extern "C" fn gfp_pfmemalloc_allowed(gfp_mask: u32) -> bool {
__gfp_pfmemalloc_flags(gfp_mask) != 0
}

/// Checks whether it makes sense to retry the reclaim to make a forward progress
/// for the given allocation request.
///
/// We give up when we either have tried MAX_RECLAIM_RETRIES in a row
/// without success, or when we couldn't even meet the watermark if we
/// reclaimed all remaining pages on the LRU lists.
///
/// Returns true if a retry is viable or false to enter the oom path.
fn should_reclaim_retry(gfp_mask: u32, order: u32, ac: *mut alloc_context, alloc_flags: i32, did_some_progress: bool, no_progress_loops: *mut i32) -> bool {
let mut __v7: bool = unsafe { core::mem::zeroed() };
let mut wmark: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut bool = core::ptr::addr_of_mut!(wmark) as *mut bool;
let mut min_wmark: u64 = unsafe { core::mem::zeroed() };
let mut reclaimable: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut u64 = core::ptr::addr_of_mut!(reclaimable) as *mut u64;
let mut available: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut u64 = core::ptr::addr_of_mut!(available) as *mut u64;
let mut ret: bool = unsafe { core::mem::zeroed() };
let mut z: core::mem::MaybeUninit<*mut zoneref> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut *mut zoneref = core::ptr::addr_of_mut!(z) as *mut *mut zoneref;
let mut zone: core::mem::MaybeUninit<*mut zone> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut *mut zone = core::ptr::addr_of_mut!(zone) as *mut *mut zone;
unsafe {
{
let __slate_auto_bytes_2 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
ret = false;
// Costly allocations might have made a progress but this doesn't mean
// their order will become available due to high fragmentation so
// always increment the no progress counter for them
if did_some_progress && order <= 3 {
*no_progress_loops = 0;
} else {
let __v14: i32 = *no_progress_loops;
*no_progress_loops = __v14.wrapping_add(1);
}
}
if *no_progress_loops <= 16 {
// Keep reclaiming pages while there is a chance this will lead
// somewhere.  If none of the target zones can satisfy our allocation
// request even if all reclaimable pages are considered then we are
// screwed and have to go OOM.
*__slate_slot_17 = first_zones_zonelist((*ac).zonelist, (*ac).highest_zoneidx, (*ac).nodemask);
let __v12: *mut zone = zonelist_zone(*__slate_slot_17);
*__slate_slot_18 = __v12;
'__join_5: {
loop {
if !(*__slate_slot_18).is_null() {
let __slate_auto_bytes_3 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
min_wmark = min_wmark_pages(*__slate_slot_18 as *const zone);
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
if cpusets_enabled() && alloc_flags & 64 != 0 {
__v7 = !__cpuset_zone_allowed(*__slate_slot_18, gfp_mask);
} else {
__v7 = false;
}
if !__v7 {
let __v6: u64 = zone_reclaimable_pages(*__slate_slot_18);
*__slate_slot_10 = __v6;
*__slate_slot_11 = __v6;
let __v5: u64 = zone_page_state_snapshot(*__slate_slot_18, 0u32);
let __v4: u64 = *__slate_slot_11;
*__slate_slot_11 = __v4.wrapping_add(__v5);
// Would the allocation succeed if we reclaimed all
// reclaimable pages?
*__slate_slot_8 = __zone_watermark_ok(*__slate_slot_18, order, min_wmark, ((*ac).highest_zoneidx as u32) as i32, alloc_flags as u32, *__slate_slot_11 as i64);
trace_reclaim_retry_zone(*__slate_slot_17, order as i32, *__slate_slot_10, *__slate_slot_11, min_wmark, *no_progress_loops, *__slate_slot_8);
if *__slate_slot_8 {
break;
}
}
let __v2: *mut zoneref = *__slate_slot_17;
let __v1: *mut zoneref = __v2.wrapping_offset(1 as isize);
*__slate_slot_17 = __v1;
*__slate_slot_17 = next_zones_zonelist(__v1, (*ac).highest_zoneidx, (*ac).nodemask);
let __v0: *mut zone = zonelist_zone(*__slate_slot_17);
*__slate_slot_18 = __v0;
} else {
break '__join_5;
}
}
ret = true;
}
// Memory allocation/reclaim might be called from a WQ context and the
// current implementation of the WQ concurrency control doesn't
// recognize that a particular WQ is congested if the worker thread is
// looping without ever sleeping. Therefore we have to do a short sleep
// here rather than calling cond_resched().
if (*get_current()).flags & 32 != 0 {
schedule_timeout_uninterruptible(1i64);
} else {
cond_resched__21!();
}
}
if !ret {
return unreserve_highatomic_pageblock(ac as *const alloc_context, true);
} else {
// Before OOM, exhaust highatomic_reserve
return ret;
}
}
}

fn check_retry_cpuset(cpuset_mems_cookie: i32, ac: *mut alloc_context) -> bool {
// It's possible that cpuset's mems_allowed and the nodemask from
// mempolicy don't intersect. This should be normally dealt with by
// policy_nodemask(), but it's possible to race with cpuset update in
// such a way the check therein was true, and then it became false
// before we got our cpuset_mems_cookie here.
// This assumes that for all allocations, ac->nodemask can come only
// from MPOL_BIND mempolicy (whose documented semantics is to be ignored
// when it does not intersect with the cpuset restrictions) or the
// caller can deal with a violated nodemask.
let __v0: bool = cpusets_enabled() && !unsafe { (*ac).nodemask }.is_null() && (unsafe { cpuset_nodemask_valid_mems_allowed((*ac).nodemask) }) == 0;
if __v0 {
unsafe {
(*ac).nodemask = core::ptr::null::<__SlateRecord_nodemask_types_h_17>();
}
return true;
}
// When updating a task's mems_allowed or mempolicy nodemask, it is
// possible to race with parallel threads in such a way that our
// allocation can fail while the mask is being updated. If we are about
// to fail, check if the cpuset changed during allocation and if so,
// retry.
if read_mems_allowed_retry(cpuset_mems_cookie as u32) {
return true;
}
false
}

fn check_alloc_stall_warn(mut gfp_mask: u32, nodemask: *const __SlateRecord_nodemask_types_h_17, order: u32, alloc_start_time: u64) {
let stall_msecs: u64 = jiffies_to_msecs(unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) }.wrapping_sub(alloc_start_time)) as u64;
if likely!(stall_msecs < 10000u64) != 0 {
return;
}
let mut __v0: i32;
let mut __dummy: u64 = 0;
let mut __dummy2: u64 = 0;
core::ptr::addr_of_mut!(__dummy) == core::ptr::addr_of_mut!(__dummy2);
__v0 = 1;
let __v1: bool = __v0 != 0 && typecheck__3!(u64) != 0;
let __v5: bool = if __v1 {
let mut __v6: u64;
loop {
if !(((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1040() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v6 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!(alloc_stall_warn_jiffies) as *const u64) };
(unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) }.wrapping_sub(__v6) as i64) < 0
} else {
false
};
if __v5 {
return;
}
if gfp_mask & (BIT!(13i32) as u32) != 0 {
return;
}
if spin_trylock(unsafe { core::ptr::addr_of_mut!(alloc_stall_lock) }) == 0 {
return;
}
// Check again, this time under the lock
if time_is_after_jiffies!(unsafe { alloc_stall_warn_jiffies }) {
spin_unlock(unsafe { core::ptr::addr_of_mut!(alloc_stall_lock) });
return;
}
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1041() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__4!(alloc_stall_warn_jiffies, unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) }.wrapping_add(msecs_to_jiffies(10000u32)));
if true {
break;
}
}
spin_unlock(unsafe { core::ptr::addr_of_mut!(alloc_stall_lock) });
let mut __v18: i32;
while false {
}
__v18 = unsafe { _printk(c"\x014%s: page allocation stall for %lu secs: order:%d, mode:%#x(%pGg) nodemask=%*pbl".as_ptr() as *const u8, (*get_current()).comm.as_mut_ptr() as *mut u8, stall_msecs / 1000, order, gfp_mask, core::ptr::addr_of_mut!(gfp_mask), __nodemask_pr_numnodes(nodemask), __nodemask_pr_bits(nodemask)) };
unsafe { cpuset_print_current_mems_allowed() };
let mut __v20: i32;
while false {
}
__v20 = unsafe { _printk(c"\x01c\n".as_ptr() as *const u8) };
unsafe { dump_stack() };
warn_alloc_show_mem(gfp_mask, nodemask);
}

static mut alloc_stall_lock: spinlock = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = unsafe { core::mem::zeroed() };
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0
} };

fn __alloc_pages_slowpath(gfp_mask: u32, order: u32, ac: *mut alloc_context) -> *mut page {
let mut alloc_flags: u32 = unsafe { core::mem::zeroed() };
let mut __v1: bool = unsafe { core::mem::zeroed() };
// Deal with possible cpuset update races or zonelist updates to avoid
// a unnecessary OOM kill.
let mut __v2: bool = unsafe { core::mem::zeroed() };
// It doesn't make any sense to retry for the compaction if the order-0
// reclaim is not able to make any progress because the current
// implementation of the compaction depends on the sufficient amount
// of free memory (see __compaction_suitable)
let mut __v5: bool = unsafe { core::mem::zeroed() };
// Deal with possible cpuset update races or zonelist updates to avoid
// infinite retries. No "goto retry;" can be placed above this check
// unless it can execute just once.
let mut __v6: bool = unsafe { core::mem::zeroed() };
let mut alloc_start_time: u64 = unsafe { core::mem::zeroed() };
let mut can_retry_reserves: bool = unsafe { core::mem::zeroed() };
let mut compact_first: bool = unsafe { core::mem::zeroed() };
let mut reserve_flags: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut i32 = core::ptr::addr_of_mut!(reserve_flags) as *mut i32;
let mut zonelist_iter_cookie: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut u32 = core::ptr::addr_of_mut!(zonelist_iter_cookie) as *mut u32;
let mut cpuset_mems_cookie: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut u32 = core::ptr::addr_of_mut!(cpuset_mems_cookie) as *mut u32;
let mut no_progress_loops: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut i32 = core::ptr::addr_of_mut!(no_progress_loops) as *mut i32;
let mut compaction_retries: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut i32 = core::ptr::addr_of_mut!(compaction_retries) as *mut i32;
let mut compact_result: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut u32 = core::ptr::addr_of_mut!(compact_result) as *mut u32;
let mut compact_priority: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_25: *mut u32 = core::ptr::addr_of_mut!(compact_priority) as *mut u32;
let mut did_some_progress: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_26: *mut u64 = core::ptr::addr_of_mut!(did_some_progress) as *mut u64;
let mut alloc_flags_1: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_27: *mut u32 = core::ptr::addr_of_mut!(alloc_flags_1) as *mut u32;
let mut page: *mut page = unsafe { core::mem::zeroed() };
let mut costly_order: bool = unsafe { core::mem::zeroed() };
let mut nofail: bool = unsafe { core::mem::zeroed() };
let mut __v31: bool = unsafe { core::mem::zeroed() };
let mut can_compact: bool = unsafe { core::mem::zeroed() };
let mut can_direct_reclaim: bool = unsafe { core::mem::zeroed() };
let mut __slate_state: usize = 68usize;
unsafe {
loop {
match __slate_state {
0 => {
return page;
}
1 => {
warn_alloc(gfp_mask, (*ac).nodemask, c"page allocation failure: order:%u".as_ptr() as *const u8, order);
__slate_state = 0usize;
continue;
}
2 => {
cond_resched__13!();
__slate_state = 50usize;
continue;
}
3 => {
// Help non-failing allocations by giving some access to memory
// reserves normally used for high priority non-blocking
// allocations but do not use ALLOC_NO_WATERMARKS because this
// could deplete whole memory reserves which would just make
// the situation worse.
page = __alloc_pages_cpuset_fallback(gfp_mask, order, alloc_flags, ac as *const alloc_context);
if !page.is_null() {
__slate_state = 0usize;
continue;
} else {
__slate_state = 2usize;
continue;
}
}
4 => {
alloc_flags = (*ac).alloc_flags | 32;
// Lacking direct_reclaim we can't do anything to reclaim memory,
// we disregard these unreasonable nofail requests and still
// return NULL
if !can_direct_reclaim {
__slate_state = 1usize;
continue;
} else {
__slate_state = 3usize;
continue;
}
}
5 => {
// Deal with possible cpuset update races or zonelist updates to avoid
// a unnecessary OOM kill.
// Make sure that __GFP_NOFAIL request doesn't leak out and make sure
// we always retry
if unlikely!(nofail) != 0 {
__slate_state = 4usize;
continue;
} else {
__slate_state = 1usize;
continue;
}
}
6 => {
if __v1 {
__slate_state = 55usize;
continue;
} else {
__slate_state = 5usize;
continue;
}
}
7 => {
__v1 = true;
__slate_state = 6usize;
continue;
}
8 => {
__v1 = check_retry_zonelist(*__slate_slot_20) != 0;
__slate_state = 6usize;
continue;
}
9 => {
if check_retry_cpuset(*__slate_slot_21 as i32, ac) {
__slate_state = 7usize;
continue;
} else {
__slate_state = 8usize;
continue;
}
}
10 => {
*__slate_slot_22 = 0;
__slate_state = 50usize;
continue;
}
11 => {
// Retry as long as the OOM killer is making progress
if *__slate_slot_26 != 0 {
__slate_state = 10usize;
continue;
} else {
__slate_state = 9usize;
continue;
}
}
12 => {
// Avoid allocations with no watermarks from looping endlessly
if tsk_is_oom_victim(get_current()) && (*__slate_slot_27 & 8 != 0 || gfp_mask & (BIT!(19i32) as u32) != 0) {
__slate_state = 9usize;
continue;
} else {
__slate_state = 11usize;
continue;
}
}
13 => {
// Reclaim has failed us, start killing things
page = __alloc_pages_may_oom(gfp_mask, order, ac as *const alloc_context, &mut (*core::ptr::addr_of_mut!(*__slate_slot_26)));
if !page.is_null() {
__slate_state = 0usize;
continue;
} else {
__slate_state = 12usize;
continue;
}
}
14 => {
if __v2 {
__slate_state = 55usize;
continue;
} else {
__slate_state = 13usize;
continue;
}
}
15 => {
__v2 = true;
__slate_state = 14usize;
continue;
}
16 => {
__v2 = check_retry_zonelist(*__slate_slot_20) != 0;
__slate_state = 14usize;
continue;
}
17 => {
if check_retry_cpuset(*__slate_slot_21 as i32, ac) {
__slate_state = 15usize;
continue;
} else {
__slate_state = 16usize;
continue;
}
}
18 => {
let __v3: u32 = *__slate_slot_27 & 4294967039u32;
*__slate_slot_27 = __v3;
__slate_state = 50usize;
continue;
}
19 => {
// Reclaim/compaction failed to prevent the fallback
if defrag_mode != 0 && *__slate_slot_27 & 256 != 0 {
__slate_state = 18usize;
continue;
} else {
__slate_state = 17usize;
continue;
}
}
20 => {
if __v5 {
__slate_state = 50usize;
continue;
} else {
__slate_state = 19usize;
continue;
}
}
21 => {
__v5 = should_compact_retry(gfp_mask, ac, order as i32, *__slate_slot_27 as i32, *__slate_slot_24, core::ptr::addr_of_mut!(*__slate_slot_25), core::ptr::addr_of_mut!(*__slate_slot_23));
__slate_state = 20usize;
continue;
}
22 => {
__v5 = false;
__slate_state = 20usize;
continue;
}
23 => {
if *__slate_slot_26 > 0 && can_compact {
__slate_state = 21usize;
continue;
} else {
__slate_state = 22usize;
continue;
}
}
24 => {
if should_reclaim_retry(gfp_mask, order, ac, *__slate_slot_27 as i32, *__slate_slot_26 > 0, core::ptr::addr_of_mut!(*__slate_slot_22)) {
__slate_state = 50usize;
continue;
} else {
__slate_state = 23usize;
continue;
}
}
25 => {
if __v6 {
__slate_state = 55usize;
continue;
} else {
__slate_state = 24usize;
continue;
}
}
26 => {
__v6 = true;
__slate_state = 25usize;
continue;
}
27 => {
__v6 = check_retry_zonelist(*__slate_slot_20) != 0;
__slate_state = 25usize;
continue;
}
28 => {
if check_retry_cpuset(*__slate_slot_21 as i32, ac) {
__slate_state = 26usize;
continue;
} else {
__slate_state = 27usize;
continue;
}
}
29 => {
// Do not retry costly high order allocations unless they are
// __GFP_RETRY_MAYFAIL and we can compact
if costly_order && (!can_compact || gfp_mask & (BIT!(14i32) as u32) == 0) {
__slate_state = 9usize;
continue;
} else {
__slate_state = 28usize;
continue;
}
}
30 => {
// Do not loop if specifically requested
if gfp_mask & (BIT!(16i32) as u32) != 0 {
__slate_state = 9usize;
continue;
} else {
__slate_state = 29usize;
continue;
}
}
31 => {
compact_first = false;
__slate_state = 50usize;
continue;
}
32 => {
*__slate_slot_25 = 1;
__slate_state = 31usize;
continue;
}
33 => {
// For the initial compaction attempt we have lowered its
// priority. Restore it for further retries, if those are
// allowed. With __GFP_NORETRY there will be a single round of
// reclaim and compaction with the lowered priority.
if gfp_mask & (BIT!(16i32) as u32) == 0 {
__slate_state = 32usize;
continue;
} else {
__slate_state = 31usize;
continue;
}
}
34 => {
// THP page faults may attempt local node only first, but are
// then allowed to only compact, not reclaim, see
// alloc_pages_mpol().
//
// Compaction has failed above and we don't want such THP
// allocations to put reclaim pressure on a single node in a
// situation where other nodes might have plenty of available
// memory.
if gfp_has_flags(gfp_mask, (BIT!(16i32) as u32) | (BIT!(21i32) as u32)) {
__slate_state = 9usize;
continue;
} else {
__slate_state = 33usize;
continue;
}
}
35 => {
if compact_first {
__slate_state = 34usize;
continue;
} else {
__slate_state = 30usize;
continue;
}
}
36 => {
// Try direct compaction and then allocating
page = __alloc_pages_direct_compact(gfp_mask, order, *__slate_slot_27, ac as *const alloc_context, *__slate_slot_25, &mut (*core::ptr::addr_of_mut!(*__slate_slot_24)));
if !page.is_null() {
__slate_state = 0usize;
continue;
} else {
__slate_state = 35usize;
continue;
}
}
37 => {
page = __alloc_pages_direct_reclaim(gfp_mask, order, *__slate_slot_27, ac as *const alloc_context, core::ptr::addr_of_mut!(*__slate_slot_26));
if !page.is_null() {
__slate_state = 0usize;
continue;
} else {
__slate_state = 36usize;
continue;
}
}
38 => {
// If allocation has taken excessively long, warn about it
check_alloc_stall_warn(gfp_mask, (*ac).nodemask, order, alloc_start_time);
// Try direct reclaim and then allocating
if !compact_first {
__slate_state = 37usize;
continue;
} else {
__slate_state = 36usize;
continue;
}
}
39 => {
// Avoid recursion of direct reclaim
if (*get_current()).flags & 2048 != 0 {
__slate_state = 9usize;
continue;
} else {
__slate_state = 38usize;
continue;
}
}
40 => {
let __v7: u32 = *__slate_slot_27 & 4294967039u32;
*__slate_slot_27 = __v7;
__slate_state = 50usize;
continue;
}
41 => {
// Reclaim/compaction cannot run, so defrag_mode's strategy
// of enforcing ALLOC_NOFRAGMENT cannot be fulfilled. Allow
// fallbacks rather than failing the allocation outright.
if defrag_mode != 0 && *__slate_slot_27 & 256 != 0 && gfp_mask & (BIT!(11i32) as u32) != 0 {
__slate_state = 40usize;
continue;
} else {
__slate_state = 9usize;
continue;
}
}
42 => {
// Caller is not willing to reclaim, we can't balance anything
if !can_direct_reclaim {
__slate_state = 41usize;
continue;
} else {
__slate_state = 39usize;
continue;
}
}
43 => {
can_retry_reserves = false;
__slate_state = 50usize;
continue;
}
44 => {
(*ac).nodemask = core::ptr::null::<__SlateRecord_nodemask_types_h_17>();
(*ac).preferred_zoneref = first_zones_zonelist((*ac).zonelist, (*ac).highest_zoneidx, (*ac).nodemask);
// The first time we adjust anything due to being allowed to
// ignore memory policies or watermarks, retry immediately. This
// allows us to keep the first allocation attempt optimistic so
// it can succeed in a zone that is still above watermarks.
if can_retry_reserves {
__slate_state = 43usize;
continue;
} else {
__slate_state = 42usize;
continue;
}
}
45 => {
// Reset the nodemask and zonelist iterators if memory policies can be
// ignored. These allocations are high priority and system rather than
// user oriented.
if *__slate_slot_27 & 64 == 0 || *__slate_slot_19 != 0 {
__slate_state = 44usize;
continue;
} else {
__slate_state = 42usize;
continue;
}
}
46 => {
*__slate_slot_27 = alloc_flags_cma(gfp_mask) | (*__slate_slot_19 as u32) | (*ac).alloc_flags | *__slate_slot_27 & 2048;
__slate_state = 45usize;
continue;
}
47 => {
*__slate_slot_19 = __gfp_pfmemalloc_flags(gfp_mask);
if *__slate_slot_19 != 0 {
__slate_state = 46usize;
continue;
} else {
__slate_state = 45usize;
continue;
}
}
48 => {
// Ensure kswapd doesn't accidentally go to sleep as long as we loop
// The adjusted alloc_flags might result in immediate success, so try
// that first
page = get_page_from_freelist(gfp_mask, order, *__slate_slot_27 as i32, ac as *const alloc_context);
if !page.is_null() {
__slate_state = 0usize;
continue;
} else {
__slate_state = 47usize;
continue;
}
}
49 => {
wake_all_kswapds(order, gfp_mask, ac as *const alloc_context);
__slate_state = 48usize;
continue;
}
50 => {
if *__slate_slot_27 & 2048 != 0 {
__slate_state = 49usize;
continue;
} else {
__slate_state = 48usize;
continue;
}
}
51 => {
let z: *mut zoneref = first_zones_zonelist((*ac).zonelist, (*ac).highest_zoneidx, core::ptr::addr_of_mut!((*get_current()).mems_allowed) as *const __SlateRecord_nodemask_types_h_17);
if zonelist_zone(z).is_null() {
__slate_state = 9usize;
continue;
} else {
__slate_state = 50usize;
continue;
}
}
52 => {
// Check for insane configurations where the cpuset doesn't contain
// any suitable zone to satisfy the request - e.g. non-movable
// GFP_HIGHUSER allocations from MOVABLE nodes only.
if cpusets_insane_config() && gfp_mask & (BIT!(20i32) as u32) != 0 {
__slate_state = 51usize;
continue;
} else {
__slate_state = 50usize;
continue;
}
}
53 => {
// The fast path uses conservative alloc_flags to succeed only until
// kswapd needs to be woken up, and to avoid the cost of setting up
// alloc_flags precisely. So we do that now.
//
// Can't just or alloc_flags if it contains WMARK bits, but those flags
// shouldn't be set in ac->alloc_flags.
VM_WARN_ON!();
*__slate_slot_27 = (*ac).alloc_flags | alloc_flags_slowpath(gfp_mask, order);
// We need to recalculate the starting point for the zonelist iterator
// because we might have used different nodemask in the fast path, or
// there was a cpuset modification and we are retrying - otherwise we
// could end up iterating over non-eligible zones endlessly.
(*ac).preferred_zoneref = first_zones_zonelist((*ac).zonelist, (*ac).highest_zoneidx, (*ac).nodemask);
if zonelist_zone((*ac).preferred_zoneref).is_null() {
__slate_state = 9usize;
continue;
} else {
__slate_state = 52usize;
continue;
}
}
54 => {
compact_first = true;
*__slate_slot_25 = 2;
__slate_state = 53usize;
continue;
}
55 => {
*__slate_slot_23 = 0;
*__slate_slot_22 = 0;
*__slate_slot_24 = 1;
*__slate_slot_25 = 1;
*__slate_slot_21 = read_mems_allowed_begin();
*__slate_slot_20 = zonelist_iter_begin();
// For costly allocations, try direct compaction first, as it's likely
// that we have enough base pages and don't need to reclaim. For non-
// movable high-order allocations, do that as well, as compaction will
// try prevent permanent fragmentation by migrating from blocks of the
// same migratetype.
if can_compact && (costly_order || order > 0 && (*ac).migratetype != 1) {
__slate_state = 54usize;
continue;
} else {
__slate_state = 53usize;
continue;
}
}
56 => {
// PF_MEMALLOC request from this context is rather bizarre
// because we cannot reclaim anything and only can loop waiting
// for somebody to do a work for us.
__slate_state = 55usize;
continue;
}
57 => {
core::arch::asm!("1045: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1045b - ., 4; .popsection\n# {0}", const 1045 as i32, options(nostack, att_syntax));
if false {
__slate_state = 59usize;
continue;
} else {
__slate_state = 56usize;
continue;
}
}
58 => {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___alloc_pages_slowpath_35, sym __slate_asm_string___alloc_pages_slowpath_36, const 4816 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if false {
__slate_state = 58usize;
continue;
} else {
__slate_state = 57usize;
continue;
}
}
59 => {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("1044: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1044b - ., 3; .popsection\n# {0}", const 1044 as i32, options(nostack, att_syntax));
__slate_state = 58usize;
continue;
}
60 => {
// Also we don't support __GFP_NOFAIL without __GFP_DIRECT_RECLAIM,
// otherwise, we may result in lockup.
let __ret_warn_on: i32 = ((*get_current()).flags & 2048 != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
__slate_state = 59usize;
continue;
} else {
__slate_state = 56usize;
continue;
}
}
61 => {
core::arch::asm!("1043: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1043b - ., 4; .popsection\n# {0}", const 1043 as i32, options(nostack, att_syntax));
if false {
__slate_state = 63usize;
continue;
} else {
__slate_state = 60usize;
continue;
}
}
62 => {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___alloc_pages_slowpath_37, sym __slate_asm_string___alloc_pages_slowpath_38, const 4810 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if false {
__slate_state = 62usize;
continue;
} else {
__slate_state = 61usize;
continue;
}
}
63 => {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("1042: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1042b - ., 3; .popsection\n# {0}", const 1042 as i32, options(nostack, att_syntax));
__slate_state = 62usize;
continue;
}
64 => {
let __ret_warn_on: i32 = !can_direct_reclaim as i32;
if unlikely__2!(__ret_warn_on) != 0 {
__slate_state = 63usize;
continue;
} else {
__slate_state = 60usize;
continue;
}
}
65 => {
can_compact = __v31;
nofail = gfp_mask & (BIT!(15i32) as u32) != 0;
costly_order = order > 3;
page = core::ptr::null_mut::<page>();
let __slate_auto_bytes = __slate_slot_27 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_26 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_25 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_23 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_6 = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_7 = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
compact_first = false;
can_retry_reserves = true;
alloc_start_time = core::ptr::read_volatile(core::ptr::addr_of!(jiffies));
if unlikely!(nofail) != 0 {
__slate_state = 64usize;
continue;
} else {
__slate_state = 55usize;
continue;
}
}
66 => {
__v31 = gfp_compaction_allowed(gfp_mask);
__slate_state = 65usize;
continue;
}
67 => {
__v31 = false;
__slate_state = 65usize;
continue;
}
68 => {
can_direct_reclaim = gfp_mask & (BIT!(10i32) as u32) != 0;
if can_direct_reclaim {
__slate_state = 66usize;
continue;
} else {
__slate_state = 67usize;
continue;
}
}
_ => {
core::hint::unreachable_unchecked();
}
}
}
}
unsafe { core::mem::zeroed() }
}

fn prepare_alloc_pages(gfp_mask: u32, order: u32, preferred_nid: i32, nodemask: *mut __SlateRecord_nodemask_types_h_17, ac: &mut alloc_context, alloc_gfp: &mut u32, alloc_flags: &mut u32) -> bool {
unsafe {
(*(ac as *mut alloc_context)).highest_zoneidx = gfp_zone(gfp_mask);
}
unsafe {
(*(ac as *mut alloc_context)).zonelist = node_zonelist(preferred_nid, gfp_mask);
}
unsafe {
(*(ac as *mut alloc_context)).nodemask = nodemask as *const __SlateRecord_nodemask_types_h_17;
}
unsafe {
(*(ac as *mut alloc_context)).migratetype = gfp_migratetype(gfp_mask);
}
if cpusets_enabled() {
let __v1: u32 = unsafe { *alloc_gfp };
let __v2: u32 = __v1 | (BIT!(20i32) as u32);
unsafe {
*alloc_gfp = __v2;
}
// When we are in the interrupt context, it is irrelevant
// to the current task context. It means that any node ok.
if in_task!() && unsafe { (*(ac as *mut alloc_context)).nodemask }.is_null() {
unsafe {
(*(ac as *mut alloc_context)).nodemask = core::ptr::addr_of_mut!((*get_current()).mems_allowed) as *const __SlateRecord_nodemask_types_h_17;
}
} else {
unsafe {
*alloc_flags = *alloc_flags | 64;
}
}
}
might_alloc(gfp_mask);
// Don't invoke should_fail logic, since it may call
// get_random_u32() and printk() which need to spin_lock.
let __v6: bool = (unsafe { *(alloc_flags as *mut u32) }) & 1024 == 0 && should_fail_alloc_page(gfp_mask, order);
if __v6 {
return false;
}
let __v7: u32 = alloc_flags_cma(gfp_mask);
unsafe {
*alloc_flags = *alloc_flags | __v7;
}
// Dirty zone balancing only done in the fast path
unsafe {
(*(ac as *mut alloc_context)).spread_dirty_pages = gfp_mask & (BIT!(12i32) as u32) != 0;
}
// The preferred zone is used for statistics but crucially it is
// also used as the starting point for the zonelist iterator. It
// may get reset for allocations that ignore memory policies.
unsafe {
(*(ac as *mut alloc_context)).preferred_zoneref = first_zones_zonelist((*(ac as *mut alloc_context)).zonelist, (*(ac as *mut alloc_context)).highest_zoneidx, (*(ac as *mut alloc_context)).nodemask);
}
true
}

/// __alloc_pages_bulk - Allocate a number of order-0 pages to an array
/// @gfp: GFP flags for the allocation
/// @preferred_nid: The preferred NUMA node ID to allocate from
/// @nodemask: Set of nodes to allocate from, may be NULL
/// @nr_pages: The number of pages desired in the array
/// @page_array: Array to store the pages
///
/// This is a batched version of the page allocator that attempts to allocate
/// @nr_pages quickly.  Pages are added to @page_array.
///
/// Note that only the elements in @page_array that were cleared to %NULL on
/// entry are populated with newly allocated pages. @nr_pages is the maximum
/// number of pages that will be stored in the array.
///
/// Returns the number of pages in @page_array, including ones already
/// allocated on entry.  This can be less than the number requested in @nr_pages,
/// but all empty slots are filled from the beginning.  I.e., if all slots in
/// @page_array were set to %NULL on entry, the slots from 0 to the return value
/// - 1 will be filled.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.page_alloc.alloc_pages_bulk_noprof")]
extern "C" fn alloc_pages_bulk_noprof(mut gfp: u32, preferred_nid: i32, nodemask: *mut __SlateRecord_nodemask_types_h_17, nr_pages: i32, page_array: *mut *mut page) -> u64 {
let mut pfo_val__: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut u64 = core::ptr::addr_of_mut!(pfo_val__) as *mut u64;
let mut tcp_ptr__: u64 = unsafe { core::mem::zeroed() };
let mut _ret: core::mem::MaybeUninit<*mut per_cpu_pages> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut *mut per_cpu_pages = core::ptr::addr_of_mut!(_ret) as *mut *mut per_cpu_pages;
let mut __v20: bool = unsafe { core::mem::zeroed() };
let mut __v21: bool = unsafe { core::mem::zeroed() };
let mut __v22: bool = unsafe { core::mem::zeroed() };
let mut mark: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut u64 = core::ptr::addr_of_mut!(mark) as *mut u64;
let mut nr_account: i32 = unsafe { core::mem::zeroed() };
let mut nr_populated: i32 = unsafe { core::mem::zeroed() };
let mut alloc_flags: u32 = unsafe { core::mem::zeroed() };
let mut ac: core::mem::MaybeUninit<alloc_context> = core::mem::MaybeUninit::uninit();
let __slate_slot_31: *mut alloc_context = core::ptr::addr_of_mut!(ac) as *mut alloc_context;
let mut pcp_list: core::mem::MaybeUninit<*mut list_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_32: *mut *mut list_head = core::ptr::addr_of_mut!(pcp_list) as *mut *mut list_head;
let mut pcp: core::mem::MaybeUninit<*mut per_cpu_pages> = core::mem::MaybeUninit::uninit();
let __slate_slot_33: *mut *mut per_cpu_pages = core::ptr::addr_of_mut!(pcp) as *mut *mut per_cpu_pages;
let mut z: core::mem::MaybeUninit<*mut zoneref> = core::mem::MaybeUninit::uninit();
let __slate_slot_34: *mut *mut zoneref = core::ptr::addr_of_mut!(z) as *mut *mut zoneref;
let mut zone: core::mem::MaybeUninit<*mut zone> = core::mem::MaybeUninit::uninit();
let __slate_slot_35: *mut *mut zone = core::ptr::addr_of_mut!(zone) as *mut *mut zone;
let mut page: core::mem::MaybeUninit<*mut page> = core::mem::MaybeUninit::uninit();
let __slate_slot_36: *mut *mut page = core::ptr::addr_of_mut!(page) as *mut *mut page;
unsafe {
let __slate_auto_bytes_2 = __slate_slot_36 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_35 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_34 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_33 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_32 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_7 = __slate_slot_31 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 40usize);
alloc_flags = 1;
nr_populated = 0;
nr_account = 0;
// Skip populated array elements to determine if any pages need
// to be allocated before disabling IRQs.
while nr_populated < nr_pages && !unsafe { *page_array.wrapping_offset(nr_populated as isize) }.is_null() {
nr_populated = nr_populated.wrapping_add(1);
}
'__join_2: {
// No pages requested?
if unlikely!(nr_pages <= 0) == 0 {
// Already populated array?
if unlikely!(nr_pages.wrapping_sub(nr_populated) == 0) == 0 {
'__join_1: {
// Bulk allocator does not support memcg accounting.
if !(memcg_kmem_online() && gfp & (BIT!(22i32) as u32) != 0) {
// Use the single page allocator for one page.
if nr_pages.wrapping_sub(nr_populated) != 1 {
// May set ALLOC_NOFRAGMENT, fragmentation will return 1 page.
gfp &= gfp_allowed_mask;
if !prepare_alloc_pages(gfp, 0u32, preferred_nid, nodemask, &mut (*core::ptr::addr_of_mut!(*__slate_slot_31)), &mut (*core::ptr::addr_of_mut!(gfp)), &mut (*core::ptr::addr_of_mut!(alloc_flags))) {
break '__join_2;
} else {
// Find an allowed local zone that meets the low watermark.
*__slate_slot_34 = (*__slate_slot_31).preferred_zoneref;
*__slate_slot_35 = zonelist_zone(*__slate_slot_34);
'__loop_28: while !(*__slate_slot_35).is_null() {
let __slate_auto_bytes = __slate_slot_23 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
if cpusets_enabled() && alloc_flags & 64 != 0 {
__v22 = !__cpuset_zone_allowed(*__slate_slot_35, gfp);
} else {
__v22 = false;
}
if !__v22 {
if nr_online_nodes > 1 {
__v21 = *__slate_slot_35 != zonelist_zone((*__slate_slot_31).preferred_zoneref);
} else {
__v21 = false;
}
if __v21 {
__v20 = zone_to_nid(*__slate_slot_35 as *const zone) != zonelist_node_idx((*__slate_slot_31).preferred_zoneref as *const zoneref);
} else {
__v20 = false;
}
if __v20 {
break '__join_1;
} else {
cond_accept_memory(*__slate_slot_35, 0u32, alloc_flags as i32);
loop {
*__slate_slot_23 = wmark_pages(*__slate_slot_35 as *const zone, (alloc_flags & 3u32) as u32).wrapping_add(nr_pages as u64).wrapping_sub(nr_populated as u64);
if zone_watermark_fast(*__slate_slot_35, 0u32, *__slate_slot_23, zonelist_zone_idx((*__slate_slot_31).preferred_zoneref as *const zoneref), alloc_flags, gfp) {
break '__loop_28;
} else {
if !cond_accept_memory(*__slate_slot_35, 0u32, alloc_flags as i32) {
// Try again if zone has deferred pages
if deferred_pages_enabled() {
if !_deferred_grow_zone(*__slate_slot_35, 0u32) {
break;
}
} else {
break;
}
}
}
}
}
}
let __v19: *mut zoneref = *__slate_slot_34;
let __v18: *mut zoneref = __v19.wrapping_offset(1 as isize);
*__slate_slot_34 = __v18;
*__slate_slot_34 = next_zones_zonelist(__v18, (*__slate_slot_31).highest_zoneidx, (*__slate_slot_31).nodemask);
let __v17: *mut zone = zonelist_zone(*__slate_slot_34);
*__slate_slot_35 = __v17;
}
// If there are no allowed local zones that meets the watermarks then
// try to allocate a single page and reclaim if necessary.
if unlikely!((*__slate_slot_35).is_null()) == 0 {
let __slate_auto_bytes_1 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
loop {
preempt_count_inc!();
core::arch::asm!("", options(nostack, att_syntax));
if true {
break;
}
}
__verify_pcpu_ptr__13!();
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __t0: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t0, sym this_cpu_off, options(pure, readonly, nostack, att_syntax));
*__slate_slot_10 = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
tcp_ptr__ = *__slate_slot_10;
tcp_ptr__ = tcp_ptr__.wrapping_add((*(*__slate_slot_35)).per_cpu_pageset as u64);
*__slate_slot_15 = tcp_ptr__ as *mut per_cpu_pages;
if spin_trylock(core::ptr::addr_of_mut!((*(*__slate_slot_15)).lock)) == 0 {
loop {
core::arch::asm!("", options(nostack, att_syntax));
if unlikely!(preempt_count_dec_and_test!()) != 0 {
loop {
core::arch::asm!("call __SCT__preempt_schedule", options(att_syntax));
if true {
break;
}
}
}
if true {
break;
}
}
*__slate_slot_15 = core::ptr::null_mut::<per_cpu_pages>();
}
// spin_trylock may fail due to a parallel drain or IRQ reentrancy.
*__slate_slot_33 = *__slate_slot_15;
if !(*__slate_slot_33).is_null() {
// Attempt the batch allocation
*__slate_slot_32 = ((*(*__slate_slot_33)).lists.as_mut_ptr() as *mut list_head).wrapping_offset(order_to_pindex((*__slate_slot_31).migratetype, 0i32) as isize);
'__join_7: {
loop {
if nr_populated < nr_pages {
// Skip existing pages
if !unsafe { *page_array.wrapping_offset(nr_populated as isize) }.is_null() {
nr_populated = nr_populated.wrapping_add(1);
} else {
*__slate_slot_36 = __rmqueue_pcplist(*__slate_slot_35, 0u32, (*__slate_slot_31).migratetype, alloc_flags, *__slate_slot_33, *__slate_slot_32);
if unlikely!((*__slate_slot_36).is_null()) != 0 {
break;
} else {
nr_account = nr_account.wrapping_add(1);
prep_new_page(*__slate_slot_36, 0u32, gfp, 0u32);
set_page_refcounted(*__slate_slot_36);
let __v3: i32 = nr_populated;
nr_populated = __v3.wrapping_add(1);
*page_array.wrapping_offset(__v3 as isize) = *__slate_slot_36;
}
}
} else {
break '__join_7;
}
}
// Try and allocate at least one page
if nr_account == 0 {
spin_unlock(core::ptr::addr_of_mut!((*(*__slate_slot_33)).lock));
loop {
core::arch::asm!("", options(nostack, att_syntax));
if unlikely!(preempt_count_dec_and_test!()) != 0 {
loop {
core::arch::asm!("call __SCT__preempt_schedule", options(att_syntax));
if true {
break;
}
}
}
if true {
break '__join_1;
}
}
}
}
spin_unlock(core::ptr::addr_of_mut!((*(*__slate_slot_33)).lock));
loop {
core::arch::asm!("", options(nostack, att_syntax));
if unlikely!(preempt_count_dec_and_test!()) != 0 {
loop {
core::arch::asm!("call __SCT__preempt_schedule", options(att_syntax));
if true {
break;
}
}
}
if true {
break;
}
}
__count_vm_events(4i64.wrapping_add(zone_idx!(*__slate_slot_35)) as u32, nr_account as i64);
zone_statistics(zonelist_zone((*__slate_slot_31).preferred_zoneref), *__slate_slot_35, nr_account as i64);
break '__join_2;
}
}
}
}
}
}
*__slate_slot_36 = __alloc_pages_noprof(gfp, 0u32, preferred_nid, nodemask, 0u32);
if !(*__slate_slot_36).is_null() {
let __v1: i32 = nr_populated;
nr_populated = __v1.wrapping_add(1);
*page_array.wrapping_offset(__v1 as isize) = *__slate_slot_36;
}
}
}
}
return nr_populated as u64;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_1046: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_1047: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_1048: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_alloc_pages_bulk_noprof_1049: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u32, i32, *mut __SlateRecord_nodemask_types_h_17, i32, *mut *mut page) -> u64>, *mut ()>(Some(alloc_pages_bulk_noprof)) };

/// free_pages_bulk - Free an array of order-0 pages
/// @page_array: Array of pages to free
/// @nr_pages: The number of pages in the array
///
/// Free the order-0 pages. Adjacent entries whose PFNs form a contiguous
/// run are released with a single __free_contig_range() call.
///
/// This assumes page_array is sorted in ascending PFN order. Without that,
/// the function still frees all pages, but contiguous runs may not be
/// detected and the freeing pattern can degrade to freeing one page at a
/// time.
///
/// Context: Sleepable process context only; calls cond_resched()
#[unsafe(no_mangle)]
extern "C" fn free_pages_bulk(mut page_array: *mut *mut page, mut nr_pages: u64) {
while nr_pages != 0 {
let nr_contig: u64 = num_pages_contiguous(page_array, nr_pages);
__free_contig_range(((unsafe { unsafe { *page_array }.offset_from(vmemmap_base as *mut page) }) as i64) as u64, nr_contig);
nr_pages = nr_pages.wrapping_sub(nr_contig);
let __v2: *mut *mut page = page_array;
page_array = __v2.wrapping_offset(nr_contig as isize);
cond_resched__82!();
}
}

fn alloc_order_allowed(gfp: u32, order: u32, alloc_flags: u32) -> bool {
if alloc_flags & 1024 != 0 {
return pcp_allowed_order(order);
}
// There are several places where we assume that the order value is sane
// so bail out early if the request is out of bound.
let mut __v0: i64;
let __ret_warn_once: i32 = (order > 10) as i32;
if unlikely!(gfp & (BIT!(13i32) as u32) == 0 && __ret_warn_once != 0 && !unsafe { __warned__5340 }) != 0 {
unsafe {
__warned__5340 = true;
}
let mut __v1: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("1050: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1050b - ., 3; .popsection\n# {0}", const 1050 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_alloc_order_allowed_2, sym __slate_asm_string_alloc_order_allowed_3, const 5340 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("1051: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1051b - ., 4; .popsection\n# {0}", const 1051 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
}
__v0 = (__ret_warn_once != 0) as i64;
__v0 == 0
}

#[unsafe(link_section = ".data..once")]
static mut __warned__5340: bool = unsafe { core::mem::zeroed() };

fn alloc_nolock_allowed() -> bool {
if !can_spin_trylock() {
return false;
}
// Bailout, since _deferred_grow_zone() needs to take a lock
if deferred_pages_enabled() {
return false;
}
true
}

/// GFP flags to set for ALLOC_NOLOCK i.e. alloc_pages_nolock().
///
/// Do not specify __GFP_DIRECT_RECLAIM, since direct claim is not allowed.
/// Do not specify __GFP_KSWAPD_RECLAIM either, since wake up of kswapd
/// is not safe in arbitrary context.
///
/// These two are the conditions for gfpflags_allow_spinning() being true.
///
/// Specify __GFP_NOWARN since failing alloc_pages_nolock() is not a reason
/// to warn. Also warn would trigger printk() which is unsafe from
/// various contexts. We cannot use printk_deferred_enter() to mitigate,
/// since the running context is unknown.
///
/// Specify __GFP_ZERO to make sure that call to kmsan_alloc_page() below
/// is safe in any context. Also zeroing the page is mandatory for
/// BPF use cases.
///
/// Though __GFP_NOMEMALLOC is not checked in the code path below,
/// specify it here to highlight that alloc_pages_nolock()
/// doesn't want to deplete reserves.
static gfp_nolock: u32 = 794880;

/// This is the 'heart' of the zoned buddy allocator.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.page_alloc.__alloc_frozen_pages_noprof")]
extern "C" fn __alloc_frozen_pages_noprof(mut gfp: u32, order: u32, preferred_nid: i32, nodemask: *mut __SlateRecord_nodemask_types_h_17, alloc_flags: u32) -> *mut page {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut __ret_warn_on: i32 = unsafe { core::mem::zeroed() };
let mut fastpath_alloc_flags: u32 = unsafe { core::mem::zeroed() };
let mut ac: core::mem::MaybeUninit<alloc_context> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut alloc_context = core::ptr::addr_of_mut!(ac) as *mut alloc_context; // The gfp_t that was actually used for allocation
let mut alloc_gfp: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut u32 = core::ptr::addr_of_mut!(alloc_gfp) as *mut u32;
let mut page: core::mem::MaybeUninit<*mut page> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut *mut page = core::ptr::addr_of_mut!(page) as *mut *mut page;
unsafe {
let __slate_auto_bytes = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
core::ptr::write(__slate_slot_19, {
let mut __t0: alloc_context = core::mem::zeroed();
__t0.alloc_flags = alloc_flags;
__t0
});
fastpath_alloc_flags = alloc_flags;
__ret_warn_on = (alloc_flags & 4294962175u32 != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("1052: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1052b - ., 3; .popsection\n# {0}", const 1052 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___alloc_frozen_pages_noprof_22, sym __slate_asm_string___alloc_frozen_pages_noprof_23, const 5394 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("1053: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1053b - ., 4; .popsection\n# {0}", const 1053 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
// Other flags could be supported later if needed.
let __v17: i64 = (__ret_warn_on != 0) as i64;
if __v17 != 0 {
return core::ptr::null_mut::<page>();
} else {
if !alloc_order_allowed(gfp, order, alloc_flags) {
return core::ptr::null_mut::<page>();
} else {
if alloc_flags & 1024 != 0 {
// Certain other flags could be supported later if needed.
VM_WARN_ON_ONCE!();
if !alloc_nolock_allowed() {
return core::ptr::null_mut::<page>();
} else {
gfp |= gfp_nolock;
fastpath_alloc_flags |= 0;
}
} else {
fastpath_alloc_flags |= 1;
}
gfp &= gfp_allowed_mask;
// Apply scoped allocation constraints. This is mainly about GFP_NOFS
// resp. GFP_NOIO which has to be inherited for all allocation requests
// from a particular context which has been marked by
// memalloc_no{fs,io}_{save,restore}. And PF_MEMALLOC_PIN which ensures
// movable zones are not used during allocation.
gfp = current_gfp_context(gfp);
*__slate_slot_20 = gfp;
if !prepare_alloc_pages(gfp, order, preferred_nid, nodemask, &mut (*core::ptr::addr_of_mut!(*__slate_slot_19)), &mut (*core::ptr::addr_of_mut!(*__slate_slot_20)), &mut (*core::ptr::addr_of_mut!(fastpath_alloc_flags))) {
return core::ptr::null_mut::<page>();
} else {
if alloc_flags & 1024 == 0 {
// Forbid the first pass from falling back to types that
// fragment memory until all local zones are considered.
let __v6: u32 = alloc_flags_nofragment(zonelist_zone((*__slate_slot_19).preferred_zoneref), gfp);
fastpath_alloc_flags |= __v6;
}
fastpath_alloc_flags |= alloc_flags_nonblocking(gfp, order) & 512;
// First allocation attempt (or, for nolock, only attempt)
*__slate_slot_21 = get_page_from_freelist(*__slate_slot_20, order, fastpath_alloc_flags as i32, core::ptr::addr_of_mut!(*__slate_slot_19) as *const alloc_context);
if !(likely__6!(*__slate_slot_21) != 0 || alloc_flags & 1024 != 0) {
*__slate_slot_20 = gfp;
(*__slate_slot_19).spread_dirty_pages = false;
// Restore the original nodemask if it was potentially replaced with
// &cpuset_current_mems_allowed to optimize the fast-path attempt.
(*__slate_slot_19).nodemask = nodemask as *const __SlateRecord_nodemask_types_h_17;
*__slate_slot_21 = __alloc_pages_slowpath(*__slate_slot_20, order, core::ptr::addr_of_mut!(*__slate_slot_19));
}
if memcg_kmem_online() && gfp & (BIT!(22i32) as u32) != 0 && !(*__slate_slot_21).is_null() {
__v0 = unlikely!(__memcg_kmem_charge_page(*__slate_slot_21, gfp, order as i32) != 0) != 0;
} else {
__v0 = false;
}
if __v0 {
__free_frozen_pages(*__slate_slot_21, order, if alloc_flags & 1024 != 0 { BIT!(2i32) as i32 } else { 0i32 });
*__slate_slot_21 = core::ptr::null_mut::<page>();
}
trace_mm_page_alloc(*__slate_slot_21, order, *__slate_slot_20, (*__slate_slot_19).migratetype);
kmsan_alloc_page(*__slate_slot_21, order, *__slate_slot_20);
return *__slate_slot_21;
}
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___alloc_frozen_pages_noprof_1054: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u32, u32, i32, *mut __SlateRecord_nodemask_types_h_17, u32) -> *mut page>, *mut ()>(Some(__alloc_frozen_pages_noprof)) };

#[unsafe(no_mangle)]
extern "C" fn __alloc_pages_noprof(gfp: u32, order: u32, preferred_nid: i32, nodemask: *mut __SlateRecord_nodemask_types_h_17, alloc_flags: u32) -> *mut page {
let page: *mut page = __alloc_frozen_pages_noprof(gfp, order, preferred_nid, nodemask, alloc_flags);
if !page.is_null() {
set_page_refcounted(page);
}
page
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.page_alloc.alloc_pages_node_noprof")]
extern "C" fn alloc_pages_node_noprof(mut nid: i32, gfp_mask: u32, order: u32) -> *mut page {
if nid == 1i32.wrapping_neg() {
nid = numa_mem_id();
}
warn_if_node_offline(nid, gfp_mask);
__alloc_pages_noprof(gfp_mask, order, nid, core::ptr::null_mut::<__SlateRecord_nodemask_types_h_17>(), 0u32)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_alloc_pages_node_noprof_1055: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(i32, u32, u32) -> *mut page>, *mut ()>(Some(alloc_pages_node_noprof)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.page_alloc.__folio_alloc_noprof")]
extern "C" fn __folio_alloc_noprof(gfp: u32, order: u32, preferred_nid: i32, nodemask: *mut __SlateRecord_nodemask_types_h_17) -> *mut folio {
let page: *mut page = __alloc_pages_noprof(gfp | (BIT!(18i32) as u32), order, preferred_nid, nodemask, 0u32);
page_rmappable_folio(page)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___folio_alloc_noprof_1056: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u32, u32, i32, *mut __SlateRecord_nodemask_types_h_17) -> *mut folio>, *mut ()>(Some(__folio_alloc_noprof)) };

/// Common helper functions. Never use with __GFP_HIGHMEM because the returned
/// address cannot represent highmem pages. Use alloc_pages and then kmap if
/// you need to access high mem.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.page_alloc.get_free_pages_noprof")]
extern "C" fn get_free_pages_noprof(gfp_mask: u32, order: u32) -> u64 {
let page: *mut page = unsafe { alloc_pages_noprof(gfp_mask & !(BIT!(1i32) as u32), order) };
if page.is_null() {
return 0;
}
page_address!(page) as u64
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_get_free_pages_noprof_1057: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u32, u32) -> u64>, *mut ()>(Some(get_free_pages_noprof)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.page_alloc.get_zeroed_page_noprof")]
extern "C" fn get_zeroed_page_noprof(gfp_mask: u32) -> u64 {
get_free_pages_noprof(gfp_mask | (BIT!(8i32) as u32), 0u32)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_get_zeroed_page_noprof_1058: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u32) -> u64>, *mut ()>(Some(get_zeroed_page_noprof)) };

fn ___free_pages(page: *mut page, mut order: u32, fpi_flags: i32) {
// get PageHead before we drop reference
let head: i32 = PageHead(page as *const page);
// get alloc tag in case the page is released by others
let tag: *mut alloc_tag = pgalloc_tag_get(page);
if put_page_testzero(page) != 0 {
__free_frozen_pages(page, order, fpi_flags);
} else {
if head == 0 {
pgalloc_tag_sub_pages(tag, (1i32 << order).wrapping_sub(1) as u32);
loop {
let __v0: u32 = order;
order = __v0.wrapping_sub(1);
if __v0 <= 0 {
break;
}
// The "tail" pages of this non-compound high-order
// page will have no code tags, so to avoid warnings
// mark them as empty.
clear_page_tag_ref(page.wrapping_offset((1i32 << order) as isize));
__free_frozen_pages(page.wrapping_offset((1i32 << order) as isize), order, fpi_flags);
}
}
}
}

/// __free_pages - Free pages allocated with alloc_pages().
/// @page: The page pointer returned from alloc_pages().
/// @order: The order of the allocation.
///
/// This function can free multi-page allocations that are not compound
/// pages.  It does not check that the @order passed in matches that of
/// the allocation, so it is easy to leak memory.  Freeing more memory
/// than was allocated will probably emit a warning.
///
/// If the last reference to this page is speculative, it will be released
/// by put_page() which only frees the first page of a non-compound
/// allocation.  To prevent the remaining pages from being leaked, we free
/// the subsequent pages here.  If you want to use the page's reference
/// count to decide when to free the allocation, you should allocate a
/// compound page, and use put_page() instead of __free_pages().
///
/// Context: May be called in interrupt context or while holding a normal
/// spinlock, but not in NMI context or while holding a raw spinlock.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.page_alloc.__free_pages")]
extern "C" fn __free_pages(page: *mut page, order: u32) {
___free_pages(page, order, 0i32);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___free_pages_1059: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut page, u32)>, *mut ()>(Some(__free_pages)) };

/// Can be called while holding raw_spin_lock or from IRQ and NMI for any
/// page type (not only those that came from alloc_pages_nolock)
#[unsafe(no_mangle)]
extern "C" fn free_pages_nolock(page: *mut page, order: u32) {
___free_pages(page, order, BIT!(2i32) as i32);
}

/// free_pages - Free pages allocated with __get_free_pages().
/// @addr: The virtual address tied to a page returned from __get_free_pages().
/// @order: The order of the allocation.
///
/// This function behaves the same as __free_pages(). Use this function
/// to free pages when you only have a valid virtual address. If you have
/// the page, call __free_pages() instead.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.page_alloc.free_pages")]
extern "C" fn free_pages(addr: u64, order: u32) {
if addr != 0 {
VM_BUG_ON!();
__free_pages(virt_to_page!(addr as *mut ()), order);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_free_pages_1060: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u64, u32)>, *mut ()>(Some(free_pages)) };

fn make_alloc_exact(addr: u64, order: u32, size: u64) -> *mut () {
if addr != 0 {
let nr: u64 = size.wrapping_add(4096u64).wrapping_sub(1) / 4096u64;
let mut page: *mut page = virt_to_page!(addr as *mut ());
let mut last: *mut page = page.wrapping_offset(nr as isize);
__split_page(page, order);
loop {
let __v0: *mut page = last;
let __v1: *mut page = __v0.wrapping_offset(-(1 as isize));
last = __v1;
if !(page < __v1) {
break;
}
set_page_refcounted(last);
}
last = page.wrapping_offset((1u64 << order) as isize);
let __v3: *mut page = page;
page = __v3.wrapping_offset(nr as isize);
while page < last {
__free_pages_ok(page, 0u32, BIT!(1i32) as i32);
let __v6: *mut page = page;
page = __v6.wrapping_offset(1 as isize);
}
}
addr as *mut ()
}

/// alloc_pages_exact - allocate an exact number physically-contiguous pages.
/// @size: the number of bytes to allocate
/// @gfp_mask: GFP flags for the allocation, must not contain __GFP_COMP
///
/// This function is similar to alloc_pages(), except that it allocates the
/// minimum number of pages to satisfy the request.  alloc_pages() can only
/// allocate memory in power-of-two pages.
///
/// This function is also limited by MAX_PAGE_ORDER.
///
/// Memory allocated by this function must be released by free_pages_exact().
///
/// Return: pointer to the allocated area or %NULL in case of error.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.page_alloc.alloc_pages_exact_noprof")]
extern "C" fn alloc_pages_exact_noprof(size: u64, mut gfp_mask: u32) -> *mut () {
let order: u32 = get_order(size) as u32;
let mut addr: u64 = 0;
let mut __v0: i64;
let __ret_warn_on: i32 = (gfp_mask & ((BIT!(18i32) as u32) | (BIT!(1i32) as u32)) != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("1061: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1061b - ., 3; .popsection\n# {0}", const 1061 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_alloc_pages_exact_noprof_1, sym __slate_asm_string_alloc_pages_exact_noprof_2, const 5638 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("1062: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1062b - ., 4; .popsection\n# {0}", const 1062 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
let __v5: u32 = gfp_mask;
gfp_mask = __v5 & !((BIT!(18i32) as u32) | (BIT!(1i32) as u32));
}
addr = get_free_pages_noprof(gfp_mask, order);
make_alloc_exact(addr, order, size)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_alloc_pages_exact_noprof_1063: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u64, u32) -> *mut ()>, *mut ()>(Some(alloc_pages_exact_noprof)) };

/// alloc_pages_exact_nid - allocate an exact number of physically-contiguous
/// 			   pages on a node.
/// @nid: the preferred node ID where memory should be allocated
/// @size: the number of bytes to allocate
/// @gfp_mask: GFP flags for the allocation, must not contain __GFP_COMP
///
/// Like alloc_pages_exact(), but try to allocate on node nid first before falling
/// back.
///
/// Return: pointer to the allocated area or %NULL in case of error.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn alloc_pages_exact_nid_noprof(nid: i32, size: u64, mut gfp_mask: u32) -> *mut () {
let order: u32 = get_order(size) as u32;
let mut __v0: i64;
let __ret_warn_on: i32 = (gfp_mask & ((BIT!(18i32) as u32) | (BIT!(1i32) as u32)) != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("1064: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1064b - ., 3; .popsection\n# {0}", const 1064 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_alloc_pages_exact_nid_noprof_1, sym __slate_asm_string_alloc_pages_exact_nid_noprof_2, const 5663 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("1065: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1065b - ., 4; .popsection\n# {0}", const 1065 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
let __v5: u32 = gfp_mask;
gfp_mask = __v5 & !((BIT!(18i32) as u32) | (BIT!(1i32) as u32));
}
let p: *mut page = alloc_pages_node_noprof(nid, gfp_mask, order);
if p.is_null() {
return core::ptr::null_mut::<()>();
}
make_alloc_exact(page_address!(p) as u64, order, size)
}

/// free_pages_exact - release memory allocated via alloc_pages_exact()
/// @virt: the value returned by alloc_pages_exact.
/// @size: size of allocation, same value as passed to alloc_pages_exact().
///
/// Release the memory allocated by a previous call to alloc_pages_exact.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.page_alloc.free_pages_exact")]
extern "C" fn free_pages_exact(virt: *mut (), size: u64) {
let mut addr: u64 = virt as u64;
let end: u64 = addr.wrapping_add(PAGE_ALIGN!(size));
while addr < end {
free_page__2!(addr);
let __v0: u64 = addr;
addr = __v0.wrapping_add(_AC!() << 12i32);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_free_pages_exact_1066: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), u64)>, *mut ()>(Some(free_pages_exact)) };

/// nr_free_zone_pages - count number of pages beyond high watermark
/// @offset: The zone index of the highest zone
///
/// nr_free_zone_pages() counts the number of pages which are beyond the
/// high watermark within all zones at or below a given zone index.  For each
/// zone, the number of pages is calculated as:
///
///     nr_free_zone_pages = managed_pages - high_pages
///
/// Return: number of pages beyond high watermark.
fn nr_free_zone_pages(offset: i32) -> u64 {
let mut z: *mut zoneref = unsafe { core::mem::zeroed() };
let mut zone: *mut zone = unsafe { core::mem::zeroed() };
// Just pick one node, since fallback list is circular
let mut sum: u64 = 0;
let zonelist: *mut zonelist = node_zonelist(numa_node_id(), ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32));
z = first_zones_zonelist(zonelist, offset as u32, core::ptr::null::<__SlateRecord_nodemask_types_h_17>());
zone = zonelist_zone(z);
while !zone.is_null() {
let size: u64 = zone_managed_pages(zone as *const zone);
let high: u64 = high_wmark_pages(zone as *const zone);
if size > high {
sum = sum.wrapping_add(size.wrapping_sub(high));
}
let __v4: *mut zoneref = z;
let __v5: *mut zoneref = __v4.wrapping_offset(1 as isize);
z = __v5;
z = next_zones_zonelist(__v5, offset as u32, core::ptr::null::<__SlateRecord_nodemask_types_h_17>());
zone = zonelist_zone(z);
}
sum
}

/// nr_free_buffer_pages - count number of pages beyond high watermark
///
/// nr_free_buffer_pages() counts the number of pages which are beyond the high
/// watermark within ZONE_DMA and ZONE_NORMAL.
///
/// Return: number of pages beyond high watermark within ZONE_DMA and
/// ZONE_NORMAL.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.page_alloc.nr_free_buffer_pages")]
extern "C" fn nr_free_buffer_pages() -> u64 {
nr_free_zone_pages((gfp_zone(((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32) | (BIT!(20i32) as u32)) as u32) as i32)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_nr_free_buffer_pages_1067: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> u64>, *mut ()>(Some(nr_free_buffer_pages)) };

fn zoneref_set_zone(zone: *mut zone, zoneref: *mut zoneref) {
unsafe {
(*zoneref).zone = zone;
}
unsafe {
(*zoneref).zone_idx = zone_idx!(zone) as i32;
}
}

/// Builds allocation fallback zone lists.
///
/// Add all populated zones of a node to the zonelist.
fn build_zonerefs_node(pgdat: *mut pglist_data, zonerefs: *mut zoneref) -> i32 {
let mut zone_type: u32 = 4;
let mut nr_zones: i32 = 0;
loop {
zone_type = (zone_type as u32).wrapping_sub(1) as u32;
let zone: *mut zone = unsafe { (*pgdat).node_zones.as_mut_ptr() as *mut zone }.wrapping_offset((zone_type as u32) as isize);
if populated_zone(zone as *const zone) {
let __v2: i32 = nr_zones;
nr_zones = __v2.wrapping_add(1);
zoneref_set_zone(zone, zonerefs.wrapping_offset(__v2 as isize));
check_highest_zone(zone_type);
}
if (zone_type as u32) == 0 {
break;
}
}
nr_zones
}

fn __parse_numa_zonelist_order(s: *mut u8) -> i32 {
// We used to support different zonelists modes but they turned
// out to be just not useful. Let's keep the warning in place
// if somebody still use the cmd line parameter so that we do
// not fail it silently
if !(((unsafe { *s }) as i32) == 100 || ((unsafe { *s }) as i32) == 68 || ((unsafe { *s }) as i32) == 110 || ((unsafe { *s }) as i32) == 78) {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x014Ignoring unsupported numa_zonelist_order value:  %s\n".as_ptr() as *const u8, s) };
return 22i32.wrapping_neg();
}
0
}

static mut numa_zonelist_order: [u8; 5] = [78, 111, 100, 101, 0];

/// sysctl handler for numa_zonelist_order
#[unsafe(link_section = ".text.slate_distinct.page_alloc.numa_zonelist_order_handler")]
extern "C" fn numa_zonelist_order_handler(table: *const ctl_table, write: i32, buffer: *mut (), length: *mut u64, ppos: *mut i64) -> i32 {
if write != 0 {
return __parse_numa_zonelist_order(buffer as *mut u8);
}
unsafe { proc_dostring(table, write, buffer, length, ppos) }
}

static mut node_load: __SlateAlign16<[i32; 64]> = __SlateAlign16(unsafe { core::mem::zeroed() });

/// find_next_best_node - find the next node that should appear in a given node's fallback list
/// @node: node whose fallback list we're appending
/// @used_node_mask: nodemask_t of already used nodes
///
/// We use a number of factors to determine which is the next node that should
/// appear on a given node's fallback list.  The node should not have appeared
/// already in @node's fallback list, and it should be the next closest node
/// according to the distance array (which contains arbitrary distance values
/// from each node to each node in the system), and should also prefer nodes
/// with no CPUs, since presumably they'll have very little allocation pressure
/// on them otherwise.
///
/// Return: node id of the found node or %NUMA_NO_NODE if no node is found.
#[unsafe(no_mangle)]
extern "C" fn find_next_best_node(node: i32, used_node_mask: *mut __SlateRecord_nodemask_types_h_17) -> i32 {
let mut n: i32 = 0;
let mut val: i32 = 0;
let mut min_val: i32 = 2147483647i32;
let mut best_node: i32 = 1i32.wrapping_neg();
// Use the local node if we haven't already, but for memoryless local
// node, we should skip it and fall back to other nodes.
let __v0: bool = node_isset__2!(node, *used_node_mask) == 0 && node_state(node, 3u32) != 0;
if __v0 {
node_set!(node, *used_node_mask);
return node;
}
n = first_node!(node_states.0[3]) as i32;
while n < 64i32 {
// Don't want a node to appear more than once
if node_isset__2!(n, *used_node_mask) == 0 {
// Use the distance array to find the distance
val = node_distance!(node, n);
// Penalize nodes under us ("prefer the next node")
val = val.wrapping_add((n < node) as i32);
// Give preference to headless and unused nodes
if !cpumask_empty(cpumask_of_node(n)) {
val = val.wrapping_add(1);
}
// Slight preference for less loaded node
val = val.wrapping_mul(64i32);
val = val.wrapping_add(unsafe { node_load.0[n as usize] });
if val < min_val {
min_val = val;
best_node = n;
}
}
n = next_node!(n, node_states.0[3]) as i32;
}
if best_node >= 0 {
node_set!(best_node, *used_node_mask);
}
best_node
}

/// Build zonelists ordered by node and zones within node.
/// This results in maximum locality--normal zone overflows into local
/// DMA zone, if any--but risks exhausting DMA zone.
fn build_zonelists_in_node_order(pgdat: *mut pglist_data, node_order__0: *mut i32, nr_nodes: u32) {
let mut zonerefs: *mut zoneref = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
zonerefs = unsafe { (*((*pgdat).node_zonelists.as_mut_ptr() as *mut zonelist).wrapping_offset(0 as isize))._zonerefs.as_mut_ptr() as *mut zoneref };
while (i as u32) < nr_nodes {
let mut nr_zones: i32 = 0;
let node: *mut pglist_data = NODE_DATA!(unsafe { *node_order__0.wrapping_offset(i as isize) });
nr_zones = build_zonerefs_node(node, zonerefs);
let __v0: *mut zoneref = zonerefs;
zonerefs = __v0.wrapping_offset(nr_zones as isize);
i = i.wrapping_add(1);
}
unsafe {
(*zonerefs).zone = core::ptr::null_mut::<zone>();
}
unsafe {
(*zonerefs).zone_idx = 0;
}
}

/// Build __GFP_THISNODE zonelists
fn build_thisnode_zonelists(pgdat: *mut pglist_data) {
let mut zonerefs: *mut zoneref = unsafe { core::mem::zeroed() };
let mut nr_zones: i32 = 0;
zonerefs = unsafe { (*((*pgdat).node_zonelists.as_mut_ptr() as *mut zonelist).wrapping_offset(1 as isize))._zonerefs.as_mut_ptr() as *mut zoneref };
nr_zones = build_zonerefs_node(pgdat, zonerefs);
let __v0: *mut zoneref = zonerefs;
zonerefs = __v0.wrapping_offset(nr_zones as isize);
unsafe {
(*zonerefs).zone = core::ptr::null_mut::<zone>();
}
unsafe {
(*zonerefs).zone_idx = 0;
}
}

fn build_zonelists(pgdat: *mut pglist_data) {
let mut __slate_compound_0: __SlateRecord_nodemask_types_h_17 = unsafe { core::mem::zeroed() };
let mut node: i32 = 0;
let mut nr_nodes: i32 = 0;
let mut used_mask: __SlateRecord_nodemask_types_h_17 = unsafe { *{
__slate_compound_0 = __SlateRecord_nodemask_types_h_17 { bits: [0] };
core::ptr::addr_of_mut!(__slate_compound_0)
} };
let mut local_node: i32 = 0;
let mut prev_node: i32 = 0;
// NUMA-aware ordering of nodes
local_node = unsafe { (*pgdat).node_id };
prev_node = local_node;
unsafe { memset((core::ptr::addr_of_mut!(node_order.0) as *mut i32) as *mut (), 0i32, core::mem::size_of::<[i32; 64]>() as u64) };
loop {
let __v6: i32 = find_next_best_node(local_node, core::ptr::addr_of_mut!(used_mask));
node = __v6;
if __v6 < 0 {
break;
}
// We don't want to pressure a particular node.
// So adding penalty to the first node in same
// distance group to make it round-robin.
if node_distance!(local_node, node) != node_distance!(local_node, prev_node) {
let __v1: *mut i32 = unsafe { core::ptr::addr_of_mut!(node_load.0) as *mut i32 }.wrapping_offset(node as isize);
let __v2: i32 = unsafe { *__v1 };
unsafe {
*__v1 = __v2.wrapping_add(1);
}
}
let __v4: i32 = nr_nodes;
nr_nodes = __v4.wrapping_add(1);
unsafe {
node_order.0[__v4 as usize] = node;
}
prev_node = node;
}
build_zonelists_in_node_order(pgdat, unsafe { core::ptr::addr_of_mut!(node_order.0) as *mut i32 }, nr_nodes as u32);
build_thisnode_zonelists(pgdat);
let mut __v8: i32;
while false {
}
__v8 = unsafe { _printk(c"\x016Fallback order for Node %d: ".as_ptr() as *const u8, local_node) };
node = 0;
while node < nr_nodes {
let mut __v10: i32;
while false {
}
__v10 = unsafe { _printk(c"\x01c%d ".as_ptr() as *const u8, node_order.0[node as usize]) };
node = node.wrapping_add(1);
}
let mut __v15: i32;
while false {
}
__v15 = unsafe { _printk(c"\x01c\n".as_ptr() as *const u8) };
}

static mut node_order: __SlateAlign16<[i32; 64]> = __SlateAlign16(unsafe { core::mem::zeroed() });

// These effectively disable the pcplists in the boot pageset completely
#[unsafe(link_section = ".data..percpu")]
static mut boot_pageset: per_cpu_pages = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".data..percpu")]
static mut boot_zonestats: per_cpu_zonestat = unsafe { core::mem::zeroed() };

fn __build_all_zonelists(data: *mut ()) {
let mut nid: i32 = 0;
let self_: *mut pglist_data = data as *mut pglist_data;
let mut flags: u64 = 0;
// The zonelist_update_seq must be acquired with irqsave because the
// reader can be invoked from IRQ with GFP_ATOMIC.
write_seqlock_irqsave!(unsafe { core::ptr::addr_of_mut!(zonelist_update_seq) }, flags);
// Also disable synchronous printk() to prevent any printk() from
// trying to hold port->lock, for
// tty_insert_flip_string_and_push_buffer() on other CPU might be
// calling kmalloc(GFP_ATOMIC | __GFP_NOWARN) with port->lock held.
printk_deferred_enter!();
unsafe { memset((core::ptr::addr_of_mut!(node_load.0) as *mut i32) as *mut (), 0i32, core::mem::size_of::<[i32; 64]>() as u64) };
// This node is hotadded and no memory is yet present.   So just
// building zonelists is fine - no need to touch other nodes.
let __v1: bool = !self_.is_null() && node_online!(unsafe { (*self_).node_id }) == 0;
if __v1 {
build_zonelists(self_);
} else {
// All possible nodes have pgdat preallocated
// in free_area_init
nid = first_node!(node_states.0[0]) as i32;
while nid < 64i32 {
let pgdat: *mut pglist_data = NODE_DATA!(nid);
build_zonelists(pgdat);
nid = next_node!(nid, node_states.0[0]) as i32;
}
}
printk_deferred_exit!();
write_sequnlock_irqrestore(unsafe { core::ptr::addr_of_mut!(zonelist_update_seq) }, flags);
}

#[inline(never)]
#[unsafe(link_section = ".init.text")]
fn build_all_zonelists_init() {
// Initialize the boot_pagesets that are going to be used
// for bootstrapping processors. The real pagesets for
// each zone will be allocated later when the per cpu
// allocator is available.
//
// boot_pagesets are used also for bootstrapping offline
// cpus if the system is already booted because the pagesets
// are needed to initialize allocators on a specific cpu too.
// F.e. the percpu allocator needs the page allocator which
// needs the percpu allocator in order to allocate its pagesets
// (a chicken-egg dilemma).
let mut cpu: i32 = 0;
__build_all_zonelists(core::ptr::null_mut::<()>());
loop {
cpu = find_next_bit(cpumask_bits!((unsafe { core::ptr::addr_of_mut!(__cpu_possible_mask) }) as *const cpumask), 64u64, cpu as u64) as i32;
if (cpu as u32) >= 64 {
break;
}
per_cpu_pages_init(per_cpu_ptr__2!(unsafe { core::ptr::addr_of_mut!(boot_pageset) }, cpu), per_cpu_ptr!(unsafe { core::ptr::addr_of_mut!(boot_zonestats) }, cpu));
cpu = cpu.wrapping_add(1);
}
unsafe { mminit_verify_zonelist() };
unsafe { cpuset_init_current_mems_allowed() };
}

/// unless system_state == SYSTEM_BOOTING.
///
/// __ref due to call of __init annotated helper build_all_zonelists_init
/// [protected by SYSTEM_BOOTING].
#[inline(never)]
#[unsafe(no_mangle)]
#[unsafe(link_section = ".ref.text")]
extern "C" fn build_all_zonelists(pgdat: *mut pglist_data) {
let mut vm_total_pages: u64 = 0;
if ((unsafe { system_state }) as u32) == 0 {
build_all_zonelists_init();
} else {
__build_all_zonelists(pgdat as *mut ());
// cpuset refresh routine should be here
}
// Get the number of free pages beyond high watermark in all zones.
vm_total_pages = nr_free_zone_pages((gfp_zone(((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32) | (BIT!(20i32) as u32) | (BIT!(1i32) as u32) | (BIT!(3i32) as u32) | 0) as u32) as i32);
// Disable grouping by mobility if the number of pages in the
// system is too low to allow the mechanism to work. It would be
// more accurate, but expensive to check per-zone. This check is
// made on memory-hotadd so a system can start with mobility
// disabled and enable it later
if vm_total_pages < (1u64 << MIN_T__2!(u32, 9i32, 10i32)).wrapping_mul(4) {
unsafe {
page_group_by_mobility_disabled = 1;
}
} else {
unsafe {
page_group_by_mobility_disabled = 0;
}
}
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x016Built %u zonelists, mobility grouping %s.  Total pages: %ld\n".as_ptr() as *const u8, nr_online_nodes, str_off_on!(unsafe { page_group_by_mobility_disabled }), vm_total_pages) };
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x016Policy zone: %s\n".as_ptr() as *const u8, zone_names.0[(policy_zone as u32) as usize]) };
}

fn zone_batchsize(zone: *mut zone) -> i32 {
let mut batch: i32 = 0;
// The number of pages to batch allocate is either ~0.025%
// of the zone or 256KB, whichever is smaller. The batch
// size is striking a balance between allocation latency
// and zone lock contention.
let mut __v0: u64;
let __UNIQUE_ID_x__1068: u64 = zone_managed_pages(zone as *const zone) >> 12i32;
let __UNIQUE_ID_y__1069: u64 = 262144 / (_AC!() << 12i32);
loop {
if __types_ok__2!(__UNIQUE_ID_x__1068, __UNIQUE_ID_y__1069) == 0 {
unsafe { __compiletime_assert_1070() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__1068 < __UNIQUE_ID_y__1069 { __UNIQUE_ID_x__1068 } else { __UNIQUE_ID_y__1069 };
batch = __v0 as i32;
if batch <= 1 {
return 1;
}
// Clamp the batch to a 2^n - 1 value. Having a power
// of 2 value was found to be more likely to have
// suboptimal cache aliasing properties in some cases.
//
// For example if 2 tasks are alternately allocating
// batches of pages, one task can end up with a lot
// of pages of one half of the possible page colors
// and the other with pages of the other colors.
batch = rounddown_pow_of_two!(batch.wrapping_add(batch / 2)).wrapping_sub(1) as i32;
batch
}

static mut percpu_pagelist_high_fraction: i32 = unsafe { core::mem::zeroed() };

fn zone_highsize(zone: *mut zone, batch: i32, cpu_online: i32, high_fraction: i32) -> i32 {
let mut high: i32 = 0;
let mut nr_split_cpus: i32 = 0;
// By default, the high value of the pcp is based on the zone
// low watermark so that if they are full then background
// reclaim will not be started prematurely.
// If percpu_pagelist_high_fraction is configured, the high
// value is based on a fraction of the managed pages in the
// zone.
let total_pages: u64 = if high_fraction == 0 {
low_wmark_pages(zone as *const zone)
} else {
zone_managed_pages(zone as *const zone) / (high_fraction as u64)
};
// Split the high value across all online CPUs local to the zone. Note
// that early in boot that CPUs may not be online yet and that during
// CPU hotplug that the cpumask is not yet updated when a CPU is being
// onlined. For memory nodes that have no CPUs, split the high value
// across all online CPUs to mitigate the risk that reclaim is triggered
// prematurely due to pages stored on pcp lists.
nr_split_cpus = cpumask_weight(cpumask_of_node(zone_to_nid(zone as *const zone))).wrapping_add(cpu_online as u32) as i32;
if nr_split_cpus == 0 {
nr_split_cpus = num_online_cpus() as i32;
}
high = (total_pages / (nr_split_cpus as u64)) as i32;
// Ensure high is at least batch*4. The multiple is based on the
// historical relationship between high and batch.
let mut __v0: i32;
let __UNIQUE_ID_x__1071: i32 = high;
let __UNIQUE_ID_y__1072: i32 = batch << 2i32;
loop {
if __types_ok!(__UNIQUE_ID_x__1071, __UNIQUE_ID_y__1072) == 0 {
unsafe { __compiletime_assert_1073() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__1071 > __UNIQUE_ID_y__1072 { __UNIQUE_ID_x__1071 } else { __UNIQUE_ID_y__1072 };
high = __v0;
high
}

// pcp->high and pcp->batch values are related and generally batch is lower
// than high. They are also related to pcp->count such that count is lower
// than high, and as soon as it reaches high, the pcplist is flushed.
//
// However, guaranteeing these relations at all times would require e.g. write
// barriers here but also careful usage of read barriers at the read side, and
// thus be prone to error and bad for performance. Thus the update only prevents
// store tearing. Any new users of pcp->batch, pcp->high_min and pcp->high_max
// should ensure they can cope with those fields changing asynchronously, and
// fully trust only the pcp->count field on the local CPU with interrupts
// disabled.
//
// mutex_is_locked(&pcp_batch_high_lock) required when calling this function
// outside of boot time (or some other assurance that no concurrent updaters
// exist).
fn pageset_update(pcp: *mut per_cpu_pages, high_min: u64, high_max: u64, batch: u64) {
loop {
loop {
if !(__native_word__9!() || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1074() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__22!((*pcp).batch, batch);
if true {
break;
}
}
loop {
loop {
if !(__native_word__9!() || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1075() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__22!((*pcp).high_min, high_min);
if true {
break;
}
}
loop {
loop {
if !(__native_word__9!() || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1076() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__22!((*pcp).high_max, high_max);
if true {
break;
}
}
}

/// Boot pageset table. One per cpu which is going to be used for all
/// zones and all nodes. The parameters will be set in such a way
/// that an item put on a list will immediately be handed over to
/// the buddy list. This is safe since pageset manipulation is done
/// with interrupts disabled.
///
/// The boot_pagesets must be kept even after bootup is complete for
/// unused processors and/or zones. They do play a role for bootstrapping
/// hotplugged processors.
///
/// zoneinfo_show() and maybe other functions do
/// not check if the processor is online before following the pageset pointer.
/// Other parts of the kernel may not check if the zone is available.
fn per_cpu_pages_init(pcp: *mut per_cpu_pages, pzstats: *mut per_cpu_zonestat) {
let mut __slate_compound_0: spinlock = unsafe { core::mem::zeroed() };
let mut pindex: i32 = 0;
unsafe { memset(pcp as *mut (), 0i32, core::mem::size_of::<per_cpu_pages>() as u64) };
unsafe { memset(pzstats as *mut (), 0i32, core::mem::size_of::<per_cpu_zonestat>() as u64) };
'__slate_break_1: loop {
spinlock_check(unsafe { core::ptr::addr_of_mut!((*pcp).lock) });
unsafe {
*core::ptr::addr_of_mut!((*pcp).lock) = *{
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
while pindex < 12i32 {
INIT_LIST_HEAD(unsafe { (*pcp).lists.as_mut_ptr() as *mut list_head }.wrapping_offset(pindex as isize));
pindex = pindex.wrapping_add(1);
}
// Set batch and high values safe for a boot pageset. A true percpu
// pageset's initialization will update them subsequently. Here we don't
// need to be as careful as pageset_update() as nobody can access the
// pageset yet.
unsafe {
(*pcp).high_min = 0;
}
unsafe {
(*pcp).high_max = 0;
}
unsafe {
(*pcp).batch = 1;
}
}

fn __zone_set_pageset_high_and_batch(zone: *mut zone, high_min: u64, high_max: u64, batch: u64) {
let mut cpu: i32 = 0;
loop {
cpu = find_next_bit(cpumask_bits!((unsafe { core::ptr::addr_of_mut!(__cpu_possible_mask) }) as *const cpumask), 64u64, cpu as u64) as i32;
if (cpu as u32) >= 64 {
break;
}
let pcp: *mut per_cpu_pages = per_cpu_ptr__2!(unsafe { (*zone).per_cpu_pageset }, cpu);
pageset_update(pcp, high_min, high_max, batch);
cpu = cpu.wrapping_add(1);
}
}

/// Calculate and set new high and batch values for all per-cpu pagesets of a
/// zone based on the zone's size.
fn zone_set_pageset_high_and_batch(zone: *mut zone, cpu_online: i32) {
let mut new_high_min: i32 = 0;
let mut new_high_max: i32 = 0;
let new_batch: i32 = zone_batchsize(zone);
if (unsafe { percpu_pagelist_high_fraction }) != 0 {
new_high_min = zone_highsize(zone, new_batch, cpu_online, unsafe { percpu_pagelist_high_fraction });
// PCP high is tuned manually, disable auto-tuning via
// setting high_min and high_max to the manual value.
new_high_max = new_high_min;
} else {
new_high_min = zone_highsize(zone, new_batch, cpu_online, 0i32);
new_high_max = zone_highsize(zone, new_batch, cpu_online, 8i32);
}
if (unsafe { (*zone).pageset_high_min }) == new_high_min && (unsafe { (*zone).pageset_high_max }) == new_high_max && (unsafe { (*zone).pageset_batch }) == new_batch {
return;
}
unsafe {
(*zone).pageset_high_min = new_high_min;
}
unsafe {
(*zone).pageset_high_max = new_high_max;
}
unsafe {
(*zone).pageset_batch = new_batch;
}
__zone_set_pageset_high_and_batch(zone, new_high_min as u64, new_high_max as u64, new_batch as u64);
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn setup_zone_pageset(zone: *mut zone) {
let mut cpu: i32 = 0;
// Size may be 0 on !SMP && !NUMA
if ((const { core::mem::size_of::<per_cpu_zonestat>() }) as u64) > 0 {
unsafe {
(*zone).per_cpu_zonestats = alloc_percpu!(per_cpu_zonestat);
}
}
unsafe {
(*zone).per_cpu_pageset = alloc_percpu!(per_cpu_pages);
}
loop {
cpu = find_next_bit(cpumask_bits!((unsafe { core::ptr::addr_of_mut!(__cpu_possible_mask) }) as *const cpumask), 64u64, cpu as u64) as i32;
if (cpu as u32) >= 64 {
break;
}
let pcp: *mut per_cpu_pages = per_cpu_ptr__2!(unsafe { (*zone).per_cpu_pageset }, cpu);
let pzstats: *mut per_cpu_zonestat = per_cpu_ptr!(unsafe { (*zone).per_cpu_zonestats }, cpu);
per_cpu_pages_init(pcp, pzstats);
cpu = cpu.wrapping_add(1);
}
zone_set_pageset_high_and_batch(zone, 0i32);
}

/// The zone indicated has a new number of managed_pages; batch sizes and percpu
/// page high values need to be recalculated.
fn zone_pcp_update(zone: *mut zone, cpu_online: i32) {
unsafe { mutex_lock(core::ptr::addr_of_mut!(pcp_batch_high_lock)) };
zone_set_pageset_high_and_batch(zone, cpu_online);
unsafe { mutex_unlock(core::ptr::addr_of_mut!(pcp_batch_high_lock)) };
}

fn zone_pcp_update_cacheinfo(zone: *mut zone, cpu: u32) {
let pcp: *mut per_cpu_pages = per_cpu_ptr__2!(unsafe { (*zone).per_cpu_pageset }, cpu);
let cci: *mut cpu_cacheinfo = unsafe { get_cpu_cacheinfo(cpu) };
// If data cache slice of CPU is large enough, "pcp->batch"
// pages can be preserved in PCP before draining PCP for
// consecutive high-order pages freeing without allocation.
// This can reduce zone lock contention without hurting
// cache-hot pages sharing.
pcp_spin_lock_nopin!(pcp);
if (unsafe { (*cci).per_cpu_data_slice_size }) >> 12i32 > (3i32.wrapping_mul(unsafe { (*pcp).batch }) as u32) {
let __v4: u8 = unsafe { (*pcp).flags };
let __v5: u8 = (((__v4 as i32) as u64) | BIT!(1i32)) as u8;
unsafe {
(*pcp).flags = __v5;
}
} else {
let __v7: u8 = unsafe { (*pcp).flags };
let __v8: u8 = (((__v7 as i32) as u64) & !BIT!(1i32)) as u8;
unsafe {
(*pcp).flags = __v8;
}
}
pcp_spin_unlock_nopin!(pcp);
}

#[unsafe(no_mangle)]
extern "C" fn setup_pcp_cacheinfo(cpu: u32) {
let mut zone: *mut zone = unsafe { core::mem::zeroed() };
zone = unsafe { (*first_online_pgdat()).node_zones.as_mut_ptr() as *mut zone };
while !zone.is_null() {
if populated_zone(zone as *const zone) {
zone_pcp_update_cacheinfo(zone, cpu);
}
zone = unsafe { next_zone(zone) };
}
}

/// Allocate per cpu pagesets and initialize them.
/// Before this call only boot pagesets were available.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn setup_per_cpu_pageset() {
let mut pgdat: *mut pglist_data = unsafe { core::mem::zeroed() };
let mut zone: *mut zone = unsafe { core::mem::zeroed() };
// Unpopulated zones continue using the boot pagesets.
// The numa stats for these pagesets need to be reset.
// Otherwise, they will end up skewing the stats of
// the nodes these zones are associated with.
let mut cpu: i32 = 0;
zone = unsafe { (*first_online_pgdat()).node_zones.as_mut_ptr() as *mut zone };
while !zone.is_null() {
if populated_zone(zone as *const zone) {
setup_zone_pageset(zone);
}
zone = unsafe { next_zone(zone) };
}
loop {
cpu = find_next_bit(cpumask_bits!((unsafe { core::ptr::addr_of_mut!(__cpu_possible_mask) }) as *const cpumask), 64u64, cpu as u64) as i32;
if (cpu as u32) >= 64 {
break;
}
let pzstats: *mut per_cpu_zonestat = per_cpu_ptr!(unsafe { core::ptr::addr_of_mut!(boot_zonestats) }, cpu);
unsafe { memset(((*pzstats).vm_numa_event.as_mut_ptr() as *mut u64) as *mut (), 0i32, core::mem::size_of::<[u64; 6]>() as u64) };
cpu = cpu.wrapping_add(1);
}
pgdat = unsafe { first_online_pgdat() };
while !pgdat.is_null() {
unsafe {
(*pgdat).per_cpu_nodestats = alloc_percpu!(per_cpu_nodestat);
}
pgdat = unsafe { next_online_pgdat(pgdat) };
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn zone_pcp_init(zone: *mut zone) {
// per cpu subsystem is not up at this point. The following code
// relies on the ability of the linker to provide the
// offset of a (static) per cpu variable into the per cpu area.
unsafe {
(*zone).per_cpu_pageset = core::ptr::addr_of_mut!(boot_pageset);
}
unsafe {
(*zone).per_cpu_zonestats = core::ptr::addr_of_mut!(boot_zonestats);
}
unsafe {
(*zone).pageset_high_min = 0;
}
unsafe {
(*zone).pageset_high_max = 0;
}
unsafe {
(*zone).pageset_batch = 1;
}
if populated_zone(zone as *const zone) {
let mut __v0: i32;
if false {
unsafe { _printk(c"\x017  %s zone: %lu pages, LIFO batch:%u\n".as_ptr() as *const u8, (*zone).name, (*zone).present_pages, zone_batchsize(zone)) };
}
__v0 = 0;
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.page_alloc.adjust_managed_page_count")]
extern "C" fn adjust_managed_page_count(page: *mut page, count: i64) {
atomic_long_add(count, unsafe { core::ptr::addr_of_mut!((*page_zone(page as *const page)).managed_pages) });
totalram_pages_add(count);
setup_per_zone_lowmem_reserve();
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_adjust_managed_page_count_1077: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut page, i64)>, *mut ()>(Some(adjust_managed_page_count)) };

/// free_reserved_pages - free reserved pages
/// @page: First page to free.
/// @order: The page order to free.
///
/// Free pages allocated through memblock during boot, letting the buddy
/// manage them from now on.
///
/// @page must be naturally aligned to the order and the order must not
/// exceed MAX_PAGE_ORDER. All pages must be reserved.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.page_alloc.free_reserved_pages")]
extern "C" fn free_reserved_pages(page: *mut page, order: u32) {
let nr_pages: u64 = 1u64 << order;
let mut i: i32 = 0;
VM_WARN_ON_ONCE!();
VM_WARN_ON_ONCE!();
while (i as u64) < nr_pages {
clear_page_tag_ref(page.wrapping_offset(i as isize));
set_page_count(page.wrapping_offset(i as isize), 0i32);
ClearPageReserved(page.wrapping_offset(i as isize));
i = i.wrapping_add(1);
}
adjust_managed_page_count(page, nr_pages as i64);
__free_frozen_pages(page, order, 0i32);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_free_reserved_pages_1078: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut page, u32)>, *mut ()>(Some(free_reserved_pages)) };

#[unsafe(link_section = ".text.slate_distinct.page_alloc.page_alloc_cpu_dead")]
extern "C" fn page_alloc_cpu_dead(cpu: u32) -> i32 {
let mut zone: *mut zone = unsafe { core::mem::zeroed() };
unsafe { lru_add_drain_cpu(cpu as i32) };
unsafe { mlock_drain_remote(cpu as i32) };
drain_pages(cpu);
// Spill the event counters of the dead processor
// into the current processors event counters.
// This artificially elevates the count of the current
// processor.
unsafe { vm_events_fold_cpu(cpu as i32) };
// Zero the differential counters of the dead processor
// so that the vm statistics are consistent.
//
// This is only okay since the processor is dead and cannot
// race with what we are doing.
unsafe { cpu_vm_stats_fold(cpu as i32) };
zone = unsafe { (*first_online_pgdat()).node_zones.as_mut_ptr() as *mut zone };
while !zone.is_null() {
if populated_zone(zone as *const zone) {
zone_pcp_update(zone, 0i32);
}
zone = unsafe { next_zone(zone) };
}
0
}

#[unsafe(link_section = ".text.slate_distinct.page_alloc.page_alloc_cpu_online")]
extern "C" fn page_alloc_cpu_online(cpu: u32) -> i32 {
let mut zone: *mut zone = unsafe { core::mem::zeroed() };
zone = unsafe { (*first_online_pgdat()).node_zones.as_mut_ptr() as *mut zone };
while !zone.is_null() {
if populated_zone(zone as *const zone) {
zone_pcp_update(zone, 1i32);
}
zone = unsafe { next_zone(zone) };
}
0
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn page_alloc_init_cpuhp() {
let ret: i32 = cpuhp_setup_state_nocalls(30i32, c"mm/page_alloc:pcp".as_ptr() as *const u8, Some(page_alloc_cpu_online), Some(page_alloc_cpu_dead));
let mut __v0: i64;
let __ret_warn_on: i32 = (ret < 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("1079: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1079b - ., 3; .popsection\n# {0}", const 1079 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_page_alloc_init_cpuhp_1, sym __slate_asm_string_page_alloc_init_cpuhp_2, const 6511 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("1080: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1080b - ., 4; .popsection\n# {0}", const 1080 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
}

/// calculate_totalreserve_pages - called when sysctl_lowmem_reserve_ratio
/// 	or min_free_kbytes changes.
fn calculate_totalreserve_pages() {
let mut pgdat: *mut pglist_data = unsafe { core::mem::zeroed() };
let mut reserve_pages: u64 = 0;
let mut i: u32 = 0;
let mut j: u32 = 0;
pgdat = unsafe { first_online_pgdat() };
while !pgdat.is_null() {
unsafe {
(*pgdat).totalreserve_pages = 0;
}
i = 0;
while (i as u32) < 4 {
let zone: *mut zone = unsafe { (*pgdat).node_zones.as_mut_ptr() as *mut zone }.wrapping_offset((i as u32) as isize);
let mut max: i64 = 0;
let managed_pages: u64 = zone_managed_pages(zone as *const zone);
// lowmem_reserve[j] is monotonically non-decreasing
// in j for a given zone (see
// setup_per_zone_lowmem_reserve()). The maximum
// valid reserve lives at the highest index with a
// non-zero value, so scan backwards and stop at the
// first hit.
j = 3u32;
while (j as u32) > (i as u32) {
if (unsafe { *((*zone).lowmem_reserve.as_mut_ptr() as *mut i64).wrapping_offset((j as u32) as isize) }) != 0 {
max = unsafe { *((*zone).lowmem_reserve.as_mut_ptr() as *mut i64).wrapping_offset((j as u32) as isize) };
break;
}
j = (j as u32).wrapping_sub(1) as u32;
}
// we treat the high watermark as reserved pages.
let __v3: u64 = high_wmark_pages(zone as *const zone);
max = (max as u64).wrapping_add(__v3) as i64;
max = min_t__2!(u64, max, managed_pages) as i64;
let __v10: u64 = unsafe { (*pgdat).totalreserve_pages };
unsafe {
(*pgdat).totalreserve_pages = __v10.wrapping_add(max as u64);
}
reserve_pages = reserve_pages.wrapping_add(max as u64);
i = (i as u32).wrapping_add(1) as u32;
}
pgdat = unsafe { next_online_pgdat(pgdat) };
}
unsafe {
totalreserve_pages = reserve_pages;
}
trace_mm_calculate_totalreserve_pages(unsafe { totalreserve_pages });
}

/// setup_per_zone_lowmem_reserve - called whenever
/// 	sysctl_lowmem_reserve_ratio changes.  Ensures that each zone
/// 	has a correct pages reserved value, so an adequate number of
/// 	pages are left in the zone after a successful __alloc_pages().
fn setup_per_zone_lowmem_reserve() {
let mut pgdat: *mut pglist_data = unsafe { core::mem::zeroed() };
let mut i: u32 = 0;
let mut j: u32 = 0;
// For a given zone node_zones[i], lowmem_reserve[j] (j > i)
// represents how many pages in zone i must effectively be kept
// in reserve when deciding whether an allocation class that is
// allowed to allocate from zones up to j may fall back into
// zone i.
//
// As j increases, the allocation class can use a strictly larger
// set of fallback zones and therefore must not be allowed to
// deplete low zones more aggressively than a less flexible one.
// As a result, lowmem_reserve[j] is required to be monotonically
// non-decreasing in j for each zone i. Callers such as
// calculate_totalreserve_pages() rely on this monotonicity when
// selecting the maximum reserve entry.
pgdat = unsafe { first_online_pgdat() };
while !pgdat.is_null() {
i = 0;
while (i as u32) < 3u32 {
let zone: *mut zone = unsafe { (*pgdat).node_zones.as_mut_ptr() as *mut zone }.wrapping_offset((i as u32) as isize);
let ratio: i32 = unsafe { sysctl_lowmem_reserve_ratio.0[(i as u32) as usize] };
let mut clear: bool = false;
clear = ratio == 0 || zone_managed_pages(zone as *const zone) == 0;
let mut managed_pages: u64 = 0;
j = (i as u32).wrapping_add(1) as u32;
while (j as u32) < 4 {
let upper_zone: *mut zone = unsafe { (*pgdat).node_zones.as_mut_ptr() as *mut zone }.wrapping_offset((j as u32) as isize);
let __v1: u64 = zone_managed_pages(upper_zone as *const zone);
managed_pages = managed_pages.wrapping_add(__v1);
if clear {
unsafe {
*((*zone).lowmem_reserve.as_mut_ptr() as *mut i64).wrapping_offset((j as u32) as isize) = 0;
}
} else {
unsafe {
*((*zone).lowmem_reserve.as_mut_ptr() as *mut i64).wrapping_offset((j as u32) as isize) = (managed_pages / (ratio as u64)) as i64;
}
}
trace_mm_setup_per_zone_lowmem_reserve(zone, upper_zone, unsafe { *((*zone).lowmem_reserve.as_mut_ptr() as *mut i64).wrapping_offset((j as u32) as isize) });
j = (j as u32).wrapping_add(1) as u32;
}
i = (i as u32).wrapping_add(1) as u32;
}
pgdat = unsafe { next_online_pgdat(pgdat) };
}
// update totalreserve_pages
calculate_totalreserve_pages();
}

fn __setup_per_zone_wmarks() {
let pages_min: u64 = ((unsafe { min_free_kbytes }) >> 2i32) as u64;
let mut lowmem_pages: u64 = 0;
let mut zone: *mut zone = unsafe { core::mem::zeroed() };
let mut flags: u64 = 0;
// Calculate total number of !ZONE_HIGHMEM and !ZONE_MOVABLE pages
zone = unsafe { (*first_online_pgdat()).node_zones.as_mut_ptr() as *mut zone };
while !zone.is_null() {
if is_highmem(zone as *const zone) == 0 && zone_idx!(zone) != 3 {
let __v0: u64 = zone_managed_pages(zone as *const zone);
lowmem_pages = lowmem_pages.wrapping_add(__v0);
}
zone = unsafe { next_zone(zone) };
}
zone = unsafe { (*first_online_pgdat()).node_zones.as_mut_ptr() as *mut zone };
while !zone.is_null() {
let mut tmp: u64 = 0;
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*zone).lock) }, flags);
tmp = pages_min.wrapping_mul(zone_managed_pages(zone as *const zone));
tmp = div64_ul!(tmp, lowmem_pages);
if is_highmem(zone as *const zone) != 0 || zone_idx!(zone) == 3 {
// __GFP_HIGH and PF_MEMALLOC allocations usually don't
// need highmem and movable zones pages, so cap pages_min
// to a small  value here.
//
// The WMARK_HIGH-WMARK_LOW and (WMARK_LOW-WMARK_MIN)
// deltas control async page reclaim, and so should
// not be capped for highmem and movable zones.
let mut min_pages: u64 = zone_managed_pages(zone as *const zone) / 1024;
let mut __v9: u64;
let __UNIQUE_ID_v__1083: u64 = min_pages;
let __UNIQUE_ID_l__1084: u64 = 32;
let __UNIQUE_ID_h__1085: u64 = 128;
loop {
if !(!statically_true!(__UNIQUE_ID_l__1084 > __UNIQUE_ID_h__1085)) {
unsafe { __compiletime_assert_1086() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
while false {
}
__v9 = if __UNIQUE_ID_v__1083 >= __UNIQUE_ID_h__1085 { __UNIQUE_ID_h__1085 } else { if __UNIQUE_ID_v__1083 <= __UNIQUE_ID_l__1084 { __UNIQUE_ID_l__1084 } else { __UNIQUE_ID_v__1083 } };
min_pages = __v9;
unsafe {
*((*zone)._watermark.as_mut_ptr() as *mut u64).wrapping_offset(0 as isize) = min_pages;
}
} else {
// If it's a lowmem zone, reserve a number of pages
// proportionate to the zone's size.
unsafe {
*((*zone)._watermark.as_mut_ptr() as *mut u64).wrapping_offset(0 as isize) = tmp;
}
}
// Set the kswapd watermarks distance according to the
// scale factor in proportion to available memory, but
// ensure a minimum size on small systems.
tmp = max_t__2!(u64, tmp >> 2i32, mult_frac!(zone_managed_pages(zone as *const zone), unsafe { watermark_scale_factor }, 10000i32));
unsafe {
(*zone).watermark_boost = 0;
}
unsafe {
*((*zone)._watermark.as_mut_ptr() as *mut u64).wrapping_offset(1 as isize) = min_wmark_pages(zone as *const zone).wrapping_add(tmp);
}
unsafe {
*((*zone)._watermark.as_mut_ptr() as *mut u64).wrapping_offset(2 as isize) = low_wmark_pages(zone as *const zone).wrapping_add(tmp);
}
unsafe {
*((*zone)._watermark.as_mut_ptr() as *mut u64).wrapping_offset(3 as isize) = high_wmark_pages(zone as *const zone).wrapping_add(tmp);
}
trace_mm_setup_per_zone_wmarks(zone);
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*zone).lock) }, flags);
zone = unsafe { next_zone(zone) };
}
// update totalreserve_pages
calculate_totalreserve_pages();
}

/// setup_per_zone_wmarks - called when min_free_kbytes changes
/// or when memory is hot-{added|removed}
///
/// Ensures that the watermark[min,low,high] values for each zone are set
/// correctly with respect to min_free_kbytes.
#[unsafe(no_mangle)]
extern "C" fn setup_per_zone_wmarks() {
let mut zone: *mut zone = unsafe { core::mem::zeroed() };
spin_lock(unsafe { core::ptr::addr_of_mut!(lock) });
__setup_per_zone_wmarks();
spin_unlock(unsafe { core::ptr::addr_of_mut!(lock) });
// The watermark size have changed so update the pcpu batch
// and high limits or the limits may be inappropriate.
zone = unsafe { (*first_online_pgdat()).node_zones.as_mut_ptr() as *mut zone };
while !zone.is_null() {
zone_pcp_update(zone, 0i32);
zone = unsafe { next_zone(zone) };
}
}

static mut lock: spinlock = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = unsafe { core::mem::zeroed() };
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0
} };

// Initialise min_free_kbytes.
//
// For small machines we want it small (128k min).  For large machines
// we want it large (256MB max).  But it is not linear, because network
// bandwidth does not increase linearly with machine size.  We use
//
// 	min_free_kbytes = 4 * sqrt(lowmem_kbytes), for better accuracy:
// 	min_free_kbytes = sqrt(lowmem_kbytes * 16)
//
// which yields
//
// 16MB:	512k
// 32MB:	724k
// 64MB:	1024k
// 128MB:	1448k
// 256MB:	2048k
// 512MB:	2896k
// 1024MB:	4096k
// 2048MB:	5792k
// 4096MB:	8192k
// 8192MB:	11584k
// 16384MB:	16384k
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

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn init_per_zone_wmark_min() -> i32 {
calculate_min_free_kbytes();
setup_per_zone_wmarks();
unsafe { refresh_zone_stat_thresholds() };
setup_per_zone_lowmem_reserve();
setup_min_unmapped_ratio();
setup_min_slab_ratio();
khugepaged_min_free_kbytes_update();
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_init_per_zone_wmark_min_1096: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i32>, *mut ()>(Some(init_per_zone_wmark_min)) };

/// min_free_kbytes_sysctl_handler - just a wrapper around proc_dointvec() so
/// 	that we can call two helper functions whenever min_free_kbytes
/// 	changes.
#[unsafe(link_section = ".text.slate_distinct.page_alloc.min_free_kbytes_sysctl_handler")]
extern "C" fn min_free_kbytes_sysctl_handler(table: *const ctl_table, write: i32, buffer: *mut (), length: *mut u64, ppos: *mut i64) -> i32 {
let rc: i32 = unsafe { proc_dointvec_minmax(table, write, buffer, length, ppos) };
if rc != 0 {
return rc;
}
if write != 0 {
unsafe {
user_min_free_kbytes = min_free_kbytes;
}
setup_per_zone_wmarks();
}
0
}

#[unsafe(link_section = ".text.slate_distinct.page_alloc.watermark_scale_factor_sysctl_handler")]
extern "C" fn watermark_scale_factor_sysctl_handler(table: *const ctl_table, write: i32, buffer: *mut (), length: *mut u64, ppos: *mut i64) -> i32 {
let rc: i32 = unsafe { proc_dointvec_minmax(table, write, buffer, length, ppos) };
if rc != 0 {
return rc;
}
if write != 0 {
setup_per_zone_wmarks();
}
0
}

fn setup_min_unmapped_ratio() {
let mut pgdat: *mut pglist_data = unsafe { core::mem::zeroed() };
let mut zone: *mut zone = unsafe { core::mem::zeroed() };
pgdat = unsafe { first_online_pgdat() };
while !pgdat.is_null() {
unsafe {
(*pgdat).min_unmapped_pages = 0;
}
pgdat = unsafe { next_online_pgdat(pgdat) };
}
zone = unsafe { (*first_online_pgdat()).node_zones.as_mut_ptr() as *mut zone };
while !zone.is_null() {
let __v1: u64 = zone_managed_pages(zone as *const zone).wrapping_mul((unsafe { sysctl_min_unmapped_ratio }) as u64) / 100;
let __v2: *mut pglist_data = unsafe { (*zone).zone_pgdat };
let __v3: u64 = unsafe { (*__v2).min_unmapped_pages };
unsafe {
(*__v2).min_unmapped_pages = __v3.wrapping_add(__v1);
}
zone = unsafe { next_zone(zone) };
}
}

#[unsafe(link_section = ".text.slate_distinct.page_alloc.sysctl_min_unmapped_ratio_sysctl_handler")]
extern "C" fn sysctl_min_unmapped_ratio_sysctl_handler(table: *const ctl_table, write: i32, buffer: *mut (), length: *mut u64, ppos: *mut i64) -> i32 {
let rc: i32 = unsafe { proc_dointvec_minmax(table, write, buffer, length, ppos) };
if rc != 0 {
return rc;
}
if write != 0 {
setup_min_unmapped_ratio();
}
0
}

fn setup_min_slab_ratio() {
let mut pgdat: *mut pglist_data = unsafe { core::mem::zeroed() };
let mut zone: *mut zone = unsafe { core::mem::zeroed() };
pgdat = unsafe { first_online_pgdat() };
while !pgdat.is_null() {
unsafe {
(*pgdat).min_slab_pages = 0;
}
pgdat = unsafe { next_online_pgdat(pgdat) };
}
zone = unsafe { (*first_online_pgdat()).node_zones.as_mut_ptr() as *mut zone };
while !zone.is_null() {
let __v1: u64 = zone_managed_pages(zone as *const zone).wrapping_mul((unsafe { sysctl_min_slab_ratio }) as u64) / 100;
let __v2: *mut pglist_data = unsafe { (*zone).zone_pgdat };
let __v3: u64 = unsafe { (*__v2).min_slab_pages };
unsafe {
(*__v2).min_slab_pages = __v3.wrapping_add(__v1);
}
zone = unsafe { next_zone(zone) };
}
}

#[unsafe(link_section = ".text.slate_distinct.page_alloc.sysctl_min_slab_ratio_sysctl_handler")]
extern "C" fn sysctl_min_slab_ratio_sysctl_handler(table: *const ctl_table, write: i32, buffer: *mut (), length: *mut u64, ppos: *mut i64) -> i32 {
let rc: i32 = unsafe { proc_dointvec_minmax(table, write, buffer, length, ppos) };
if rc != 0 {
return rc;
}
if write != 0 {
setup_min_slab_ratio();
}
0
}

/// lowmem_reserve_ratio_sysctl_handler - just a wrapper around
/// 	proc_dointvec_minmax() so that we can call
/// 	setup_per_zone_lowmem_reserve() when the sysctl is written.
///
/// The reserve ratio obviously has absolutely no relation with the
/// minimum watermarks. The lowmem reserve ratio can only make sense
/// if in function of the boot time zone sizes.
#[unsafe(link_section = ".text.slate_distinct.page_alloc.lowmem_reserve_ratio_sysctl_handler")]
extern "C" fn lowmem_reserve_ratio_sysctl_handler(table: *const ctl_table, write: i32, buffer: *mut (), length: *mut u64, ppos: *mut i64) -> i32 {
let mut tmp: ctl_table = unsafe { *table };
let mut ratio: __SlateAlign16<[i32; 4]> = __SlateAlign16([0i32; 4]);
let mut rc: i32 = 0;
if write == 0 {
return unsafe { proc_dointvec_minmax(table, write, buffer, length, ppos) };
}
// proc_dointvec_max() works incrementally. Use a buffer and only set
// the values if all of them parse cleanly.
unsafe { memcpy(ratio.0.as_mut_ptr() as *mut (), (core::ptr::addr_of_mut!(sysctl_lowmem_reserve_ratio.0) as *mut i32) as *const (), core::mem::size_of::<[i32; 4]>() as u64) };
tmp.data = ratio.0.as_mut_ptr() as *mut ();
rc = unsafe { proc_dointvec_minmax(core::ptr::addr_of_mut!(tmp) as *const ctl_table, write, buffer, length, ppos) };
if rc != 0 {
return rc;
}
unsafe { memcpy((core::ptr::addr_of_mut!(sysctl_lowmem_reserve_ratio.0) as *mut i32) as *mut (), ratio.0.as_mut_ptr() as *const (), core::mem::size_of::<[i32; 4]>() as u64) };
setup_per_zone_lowmem_reserve();
0
}

/// percpu_pagelist_high_fraction - changes the pcp->high for each zone on each
/// cpu. It is the fraction of total pages in each zone that a hot per cpu
/// pagelist can have before it gets flushed back to buddy allocator.
#[unsafe(link_section = ".text.slate_distinct.page_alloc.percpu_pagelist_high_fraction_sysctl_handler")]
extern "C" fn percpu_pagelist_high_fraction_sysctl_handler(table: *const ctl_table, write: i32, buffer: *mut (), length: *mut u64, ppos: *mut i64) -> i32 {
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut old_percpu_pagelist_high_fraction: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut i32 = core::ptr::addr_of_mut!(old_percpu_pagelist_high_fraction) as *mut i32;
let mut zone: core::mem::MaybeUninit<*mut zone> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut zone = core::ptr::addr_of_mut!(zone) as *mut *mut zone;
unsafe {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
// Avoid using pcp_batch_high_lock for reads as the value is read
// atomically and a race with offlining is harmless.
if write == 0 {
return proc_dointvec_minmax(table, write, buffer, length, ppos);
} else {
mutex_lock(core::ptr::addr_of_mut!(pcp_batch_high_lock));
*__slate_slot_1 = percpu_pagelist_high_fraction;
*__slate_slot_0 = proc_dointvec_minmax(table, write, buffer, length, ppos);
if *__slate_slot_0 >= 0 {
// Sanity checking to avoid pcp imbalance
if percpu_pagelist_high_fraction != 0 && percpu_pagelist_high_fraction < 8 {
percpu_pagelist_high_fraction = *__slate_slot_1;
*__slate_slot_0 = 22i32.wrapping_neg();
} else {
// No change?
if percpu_pagelist_high_fraction != *__slate_slot_1 {
*__slate_slot_2 = (*first_online_pgdat()).node_zones.as_mut_ptr() as *mut zone;
while !(*__slate_slot_2).is_null() {
if populated_zone(*__slate_slot_2 as *const zone) {
zone_set_pageset_high_and_batch(*__slate_slot_2, 0i32);
}
*__slate_slot_2 = next_zone(*__slate_slot_2);
}
}
}
}
mutex_unlock(core::ptr::addr_of_mut!(pcp_batch_high_lock));
return *__slate_slot_0;
}
}
}

static page_alloc_sysctl_table: __SlateAlign16<[ctl_table; 9]> = __SlateAlign16([{
let mut __t0: ctl_table = unsafe { core::mem::zeroed() };
__t0.procname = (b"min_free_kbytes\0".as_ptr() as *mut u8) as *const u8;
__t0.data = (unsafe { core::ptr::addr_of_mut!(min_free_kbytes) }) as *mut ();
__t0.maxlen = (core::mem::size_of::<i32>() as u64) as i32;
__t0.mode = 420u16;
__t0.proc_handler = Some(min_free_kbytes_sysctl_handler);
__t0.extra1 = unsafe { core::ptr::addr_of!(sysctl_vals) as *const i32 }.wrapping_offset(0i32 as isize) as *mut ();
__t0
}, {
let mut __t1: ctl_table = unsafe { core::mem::zeroed() };
__t1.procname = (b"watermark_boost_factor\0".as_ptr() as *mut u8) as *const u8;
__t1.data = (unsafe { core::ptr::addr_of_mut!(watermark_boost_factor) }) as *mut ();
__t1.maxlen = (core::mem::size_of::<i32>() as u64) as i32;
__t1.mode = 420u16;
__t1.proc_handler = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*const ctl_table, i32, *mut (), *mut u64, *mut i64) -> i32>>(proc_dointvec_minmax as *const ()) };
__t1.extra1 = unsafe { core::ptr::addr_of!(sysctl_vals) as *const i32 }.wrapping_offset(0i32 as isize) as *mut ();
__t1
}, {
let mut __t2: ctl_table = unsafe { core::mem::zeroed() };
__t2.procname = (b"watermark_scale_factor\0".as_ptr() as *mut u8) as *const u8;
__t2.data = (unsafe { core::ptr::addr_of_mut!(watermark_scale_factor) }) as *mut ();
__t2.maxlen = (core::mem::size_of::<i32>() as u64) as i32;
__t2.mode = 420u16;
__t2.proc_handler = Some(watermark_scale_factor_sysctl_handler);
__t2.extra1 = unsafe { core::ptr::addr_of!(sysctl_vals) as *const i32 }.wrapping_offset(1i32 as isize) as *mut ();
__t2.extra2 = unsafe { core::ptr::addr_of!(sysctl_vals) as *const i32 }.wrapping_offset(8i32 as isize) as *mut ();
__t2
}, {
let mut __t3: ctl_table = unsafe { core::mem::zeroed() };
__t3.procname = (b"defrag_mode\0".as_ptr() as *mut u8) as *const u8;
__t3.data = (unsafe { core::ptr::addr_of_mut!(defrag_mode) }) as *mut ();
__t3.maxlen = (core::mem::size_of::<i32>() as u64) as i32;
__t3.mode = 420u16;
__t3.proc_handler = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*const ctl_table, i32, *mut (), *mut u64, *mut i64) -> i32>>(proc_dointvec_minmax as *const ()) };
__t3.extra1 = unsafe { core::ptr::addr_of!(sysctl_vals) as *const i32 }.wrapping_offset(0i32 as isize) as *mut ();
__t3.extra2 = unsafe { core::ptr::addr_of!(sysctl_vals) as *const i32 }.wrapping_offset(1i32 as isize) as *mut ();
__t3
}, {
let mut __t4: ctl_table = unsafe { core::mem::zeroed() };
__t4.procname = (b"percpu_pagelist_high_fraction\0".as_ptr() as *mut u8) as *const u8;
__t4.data = (unsafe { core::ptr::addr_of_mut!(percpu_pagelist_high_fraction) }) as *mut ();
__t4.maxlen = (core::mem::size_of::<i32>() as u64) as i32;
__t4.mode = 420u16;
__t4.proc_handler = Some(percpu_pagelist_high_fraction_sysctl_handler);
__t4.extra1 = unsafe { core::ptr::addr_of!(sysctl_vals) as *const i32 }.wrapping_offset(0i32 as isize) as *mut ();
__t4
}, {
let mut __t5: ctl_table = unsafe { core::mem::zeroed() };
__t5.procname = (b"lowmem_reserve_ratio\0".as_ptr() as *mut u8) as *const u8;
__t5.data = (unsafe { core::ptr::addr_of_mut!(sysctl_lowmem_reserve_ratio.0) }) as *mut ();
__t5.maxlen = (core::mem::size_of::<[i32; 4]>() as u64) as i32;
__t5.mode = 420u16;
__t5.proc_handler = Some(lowmem_reserve_ratio_sysctl_handler);
__t5.extra1 = unsafe { core::ptr::addr_of!(sysctl_vals) as *const i32 }.wrapping_offset(0i32 as isize) as *mut ();
__t5
}, {
let mut __t6: ctl_table = unsafe { core::mem::zeroed() };
__t6.procname = (b"numa_zonelist_order\0".as_ptr() as *mut u8) as *const u8;
__t6.data = (unsafe { core::ptr::addr_of_mut!(numa_zonelist_order) }) as *mut ();
__t6.maxlen = 16i32;
__t6.mode = 420u16;
__t6.proc_handler = Some(numa_zonelist_order_handler);
__t6
}, {
let mut __t7: ctl_table = unsafe { core::mem::zeroed() };
__t7.procname = (b"min_unmapped_ratio\0".as_ptr() as *mut u8) as *const u8;
__t7.data = (unsafe { core::ptr::addr_of_mut!(sysctl_min_unmapped_ratio) }) as *mut ();
__t7.maxlen = (core::mem::size_of::<i32>() as u64) as i32;
__t7.mode = 420u16;
__t7.proc_handler = Some(sysctl_min_unmapped_ratio_sysctl_handler);
__t7.extra1 = unsafe { core::ptr::addr_of!(sysctl_vals) as *const i32 }.wrapping_offset(0i32 as isize) as *mut ();
__t7.extra2 = unsafe { core::ptr::addr_of!(sysctl_vals) as *const i32 }.wrapping_offset(5i32 as isize) as *mut ();
__t7
}, {
let mut __t8: ctl_table = unsafe { core::mem::zeroed() };
__t8.procname = (b"min_slab_ratio\0".as_ptr() as *mut u8) as *const u8;
__t8.data = (unsafe { core::ptr::addr_of_mut!(sysctl_min_slab_ratio) }) as *mut ();
__t8.maxlen = (core::mem::size_of::<i32>() as u64) as i32;
__t8.mode = 420u16;
__t8.proc_handler = Some(sysctl_min_slab_ratio_sysctl_handler);
__t8.extra1 = unsafe { core::ptr::addr_of!(sysctl_vals) as *const i32 }.wrapping_offset(0i32 as isize) as *mut ();
__t8.extra2 = unsafe { core::ptr::addr_of!(sysctl_vals) as *const i32 }.wrapping_offset(5i32 as isize) as *mut ();
__t8
}]);

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn page_alloc_sysctl_init() {
unsafe { __register_sysctl_init(c"vm".as_ptr() as *const u8, core::ptr::addr_of!(page_alloc_sysctl_table.0) as *const ctl_table, c"page_alloc_sysctl_table".as_ptr() as *const u8, ARRAY_SIZE__15!()) };
}

fn free_prepared_contig_range(mut page: *mut page, mut nr_pages: u64) {
let mut pfn: u64 = ((unsafe { page.offset_from(vmemmap_base as *mut page) }) as i64) as u64;
while nr_pages != 0 {
let mut order: u32 = 0;
// We are limited by the largest buddy order.
order = (if pfn != 0 { __ffs!(pfn) } else { 10u64 }) as u32;
// Don't exceed the number of pages to free.
order = min_t!(u32, order, ilog2__5!(nr_pages));
order = min_t!(u32, order, 10i32);
// Free the chunk as a single block. Our caller has already
// called free_pages_prepare() for each order-0 page.
__free_frozen_pages(page, order, BIT!(3i32) as i32);
pfn = pfn.wrapping_add(1u64 << order);
let __v8: *mut page = page;
page = __v8.wrapping_offset((1u64 << order) as isize);
nr_pages = nr_pages.wrapping_sub(1u64 << order);
}
}

fn __free_contig_range_common(pfn: u64, nr_pages: u64, is_frozen: bool) {
let mut page: *mut page = unsafe { core::mem::zeroed() };
let mut start: *mut page = core::ptr::null_mut::<page>();
let mut nr_start: u64 = 0;
let mut start_sec: u64 = 0;
for i in 0..nr_pages {
let mut can_free: bool = true;
// Contiguous PFNs might not have contiguous "struct pages"
// in some kernel configs: page++ across a section boundary
// is undefined. Use pfn_to_page() for each PFN.
page = ((unsafe { vmemmap_base }) as *mut page).wrapping_offset(pfn.wrapping_add(i) as isize);
VM_WARN_ON_ONCE!();
VM_WARN_ON_ONCE!();
if !is_frozen {
can_free = put_page_testzero(page) != 0;
}
if can_free {
can_free = free_pages_prepare(page, 0u32);
}
if !can_free {
if !start.is_null() {
free_prepared_contig_range(start, i.wrapping_sub(nr_start));
start = core::ptr::null_mut::<page>();
}
} else {
let __v1: bool = !start.is_null() && memdesc_section((unsafe { core::ptr::addr_of_mut!((*page).flags) }) as *const __SlateRecord_mm_types_h_41) != start_sec;
if __v1 {
free_prepared_contig_range(start, i.wrapping_sub(nr_start));
start = page;
nr_start = i;
start_sec = memdesc_section((unsafe { core::ptr::addr_of_mut!((*page).flags) }) as *const __SlateRecord_mm_types_h_41);
} else {
if start.is_null() {
start = page;
nr_start = i;
start_sec = memdesc_section((unsafe { core::ptr::addr_of_mut!((*page).flags) }) as *const __SlateRecord_mm_types_h_41);
}
}
}
}
if !start.is_null() {
free_prepared_contig_range(start, nr_pages.wrapping_sub(nr_start));
}
}

/// __free_contig_range - Free contiguous range of order-0 pages.
/// @pfn: Page frame number of the first page in the range.
/// @nr_pages: Number of pages to free.
///
/// For each order-0 struct page in the physically contiguous range, put a
/// reference. Free any page who's reference count falls to zero. The
/// implementation is functionally equivalent to, but significantly faster than
/// calling __free_page() for each struct page in a loop.
///
/// Memory allocated with alloc_pages(order>=1) then subsequently split to
/// order-0 with split_page() is an example of appropriate contiguous pages that
/// can be freed with this API.
///
/// Context: May be called in interrupt context or while holding a normal
/// spinlock, but not in NMI context or while holding a raw spinlock.
#[unsafe(no_mangle)]
extern "C" fn __free_contig_range(pfn: u64, nr_pages: u64) {
__free_contig_range_common(pfn, nr_pages, false); // is_frozen=
}

/// Effectively disable pcplists for the zone by setting the high limit to 0
/// and draining all cpus. A concurrent page freeing on another CPU that's about
/// to put the page on pcplist will either finish before the drain and the page
/// will be drained, or observe the new high limit and skip the pcplist.
///
/// Must be paired with a call to zone_pcp_enable().
#[unsafe(no_mangle)]
extern "C" fn zone_pcp_disable(zone: *mut zone) {
unsafe { mutex_lock(core::ptr::addr_of_mut!(pcp_batch_high_lock)) };
__zone_set_pageset_high_and_batch(zone, 0u64, 0u64, 1u64);
__drain_all_pages(zone, true);
}

#[unsafe(no_mangle)]
extern "C" fn zone_pcp_enable(zone: *mut zone) {
__zone_set_pageset_high_and_batch(zone, (unsafe { (*zone).pageset_high_min }) as u64, (unsafe { (*zone).pageset_high_max }) as u64, (unsafe { (*zone).pageset_batch }) as u64);
unsafe { mutex_unlock(core::ptr::addr_of_mut!(pcp_batch_high_lock)) };
}

#[unsafe(no_mangle)]
extern "C" fn zone_pcp_reset(zone: *mut zone) {
let mut cpu: i32 = 0;
if (unsafe { (*zone).per_cpu_pageset }) != unsafe { core::ptr::addr_of_mut!(boot_pageset) } {
cpu = 0;
loop {
cpu = find_next_bit(cpumask_bits!((unsafe { core::ptr::addr_of_mut!(__cpu_online_mask) }) as *const cpumask), 64u64, cpu as u64) as i32;
if (cpu as u32) >= 64 {
break;
}
let pzstats: *mut per_cpu_zonestat = per_cpu_ptr!(unsafe { (*zone).per_cpu_zonestats }, cpu);
unsafe { drain_zonestat(zone, pzstats) };
cpu = cpu.wrapping_add(1);
}
unsafe { free_percpu((*zone).per_cpu_pageset as *mut ()) };
unsafe {
(*zone).per_cpu_pageset = core::ptr::addr_of_mut!(boot_pageset);
}
if (unsafe { (*zone).per_cpu_zonestats }) != unsafe { core::ptr::addr_of_mut!(boot_zonestats) } {
unsafe { free_percpu((*zone).per_cpu_zonestats as *mut ()) };
unsafe {
(*zone).per_cpu_zonestats = core::ptr::addr_of_mut!(boot_zonestats);
}
}
}
}

// This function returns a stable result only if called under zone lock.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_is_free_buddy_page_1102: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const page) -> bool>, *mut ()>(Some(is_free_buddy_page)) };

#[unsafe(no_mangle)]
extern "C" fn has_managed_zone(zone: u32) -> bool {
let mut pgdat: *mut pglist_data = unsafe { core::mem::zeroed() };
pgdat = unsafe { first_online_pgdat() };
while !pgdat.is_null() {
if managed_zone(unsafe { (*pgdat).node_zones.as_mut_ptr() as *mut zone }.wrapping_offset((zone as u32) as isize) as *const zone) {
return true;
}
pgdat = unsafe { next_online_pgdat(pgdat) };
}
false
}

fn page_contains_unaccepted(page: *mut page, order: u32) -> bool {
false
}

fn cond_accept_memory(zone: *mut zone, order: u32, alloc_flags: i32) -> bool {
false
}

#[unsafe(no_mangle)]
extern "C" fn alloc_frozen_pages_nolock_noprof(gfp_flags: u32, mut nid: i32, order: u32) -> *mut page {
if nid == 1i32.wrapping_neg() {
nid = numa_node_id();
}
__alloc_frozen_pages_noprof(gfp_flags, order, nid, core::ptr::null_mut::<__SlateRecord_nodemask_types_h_17>(), 1024u32)
}

/// alloc_pages_nolock - opportunistic reentrant allocation from any context
/// @gfp_flags: GFP flags. Only __GFP_ACCOUNT, plus some flags that get set
///             internally regardless (see %gfp_nolock) are allowed.
/// @nid: node to allocate from
/// @order: allocation order size
///
/// Allocates pages of a given order from the given node. This is safe to
/// call from any context where RCU is watching (from atomic, NMI, and also
/// reentrant allocator -> tracepoint -> alloc_pages_nolock_noprof).
/// Allocation is best effort and to be expected to fail easily so nobody should
/// rely on the success. Failures are not reported via warn_alloc().
/// See always fail conditions below.
///
/// Return: allocated page or NULL on failure. NULL does not mean EBUSY or EAGAIN.
/// It means ENOMEM. There is no reason to call it again and expect !NULL.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.page_alloc.alloc_pages_nolock_noprof")]
extern "C" fn alloc_pages_nolock_noprof(gfp_flags: u32, nid: i32, order: u32) -> *mut page {
let page: *mut page = alloc_frozen_pages_nolock_noprof(gfp_flags, nid, order);
if !page.is_null() {
set_page_refcounted(page);
}
page
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_alloc_pages_nolock_noprof_1104: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u32, i32, u32) -> *mut page>, *mut ()>(Some(alloc_pages_nolock_noprof)) };

static __slate_asm_string___alloc_frozen_pages_noprof_22: [u8; 1] = *b"\0";

static __slate_asm_string___alloc_frozen_pages_noprof_23: [u8; 45] = *b"/home/takashi/c-corpus/linux/mm/page_alloc.c\0";

static __slate_asm_string___alloc_pages_may_oom_8: [u8; 1] = *b"\0";

static __slate_asm_string___alloc_pages_may_oom_9: [u8; 45] = *b"/home/takashi/c-corpus/linux/mm/page_alloc.c\0";

static __slate_asm_string___alloc_pages_slowpath_35: [u8; 1] = *b"\0";

static __slate_asm_string___alloc_pages_slowpath_36: [u8; 45] = *b"/home/takashi/c-corpus/linux/mm/page_alloc.c\0";

static __slate_asm_string___alloc_pages_slowpath_37: [u8; 1] = *b"\0";

static __slate_asm_string___alloc_pages_slowpath_38: [u8; 45] = *b"/home/takashi/c-corpus/linux/mm/page_alloc.c\0";

static __slate_asm_string_alloc_order_allowed_2: [u8; 1] = *b"\0";

static __slate_asm_string_alloc_order_allowed_3: [u8; 45] = *b"/home/takashi/c-corpus/linux/mm/page_alloc.c\0";

static __slate_asm_string_alloc_pages_exact_nid_noprof_1: [u8; 1] = *b"\0";

static __slate_asm_string_alloc_pages_exact_nid_noprof_2: [u8; 45] = *b"/home/takashi/c-corpus/linux/mm/page_alloc.c\0";

static __slate_asm_string_alloc_pages_exact_noprof_1: [u8; 1] = *b"\0";

static __slate_asm_string_alloc_pages_exact_noprof_2: [u8; 45] = *b"/home/takashi/c-corpus/linux/mm/page_alloc.c\0";

static __slate_asm_string_clear_pfnblock_bit_1: [u8; 1] = *b"\0";

static __slate_asm_string_clear_pfnblock_bit_2: [u8; 45] = *b"/home/takashi/c-corpus/linux/mm/page_alloc.c\0";

static __slate_asm_string_folio_set_order_2: [u8; 1] = *b"\0";

static __slate_asm_string_folio_set_order_3: [u8; 43] = *b"/home/takashi/c-corpus/linux/mm/internal.h\0";

static __slate_asm_string_get_pfnblock_bit_1: [u8; 1] = *b"\0";

static __slate_asm_string_get_pfnblock_bit_2: [u8; 45] = *b"/home/takashi/c-corpus/linux/mm/page_alloc.c\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

static __slate_asm_string_page_alloc_init_cpuhp_1: [u8; 1] = *b"\0";

static __slate_asm_string_page_alloc_init_cpuhp_2: [u8; 45] = *b"/home/takashi/c-corpus/linux/mm/page_alloc.c\0";

static __slate_asm_string_set_pfnblock_bit_1: [u8; 1] = *b"\0";

static __slate_asm_string_set_pfnblock_bit_2: [u8; 45] = *b"/home/takashi/c-corpus/linux/mm/page_alloc.c\0";

static __slate_asm_string_unreserve_highatomic_pageblock_16: [u8; 1] = *b"\0";

static __slate_asm_string_unreserve_highatomic_pageblock_17: [u8; 45] = *b"/home/takashi/c-corpus/linux/mm/page_alloc.c\0";

static __slate_asm_string_unreserve_highatomic_pageblock_8: [u8; 1] = *b"\0";

static __slate_asm_string_unreserve_highatomic_pageblock_9: [u8; 45] = *b"/home/takashi/c-corpus/linux/mm/page_alloc.c\0";

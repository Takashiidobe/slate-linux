
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
//! linux/mm/swapfile.c
//!
//! Copyright (C) 1991, 1992, 1993, 1994  Linus Torvalds
//! Swap reorganised 29.12.95, Stephen Tweedie
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
//! Optimization barrier
//!
//! The "volatile" is due to gcc bugs
//! Yes, this permits 64-bit accesses on 32-bit architectures. These will
//! actually be atomic in some cases (namely Armv7 + LPAE), but for others we
//! rely on the access being split into 2x32-bit accesses for a 32-bit quantity
//! (e.g. a virtual address) and a strong prevailing wind.
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
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
//! Internal kernel timer frequency
//! Sign Flag
//! Force a compilation error if condition is true, but also produce a
//! result (of value 0 and type int), so the expression can be used
//! e.g. in a structure initializer (or where-ever else comma expressions
//! aren't permitted).
//!
//! Take an error message as an optional second argument. If omitted,
//! default to the stringification of the tested expression.
//! These are for everybody (although not all archs will actually
//! discard it in modules)
//! See linux/instrumentation.h
//! 64 bit
//! unused, was: #define EX_TYPE_COPY	 4
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
//! inhibit tail-call optimization
//! Call a function once. Similar to DO_ONCE(), but does not use jump label
//! patching via static keys.
//! warning conditions
//! # chars in a path name including nul
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
//! This counts to 15. Any more, it will return 16th argument.
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
//! Defines x86 CPU feature bits
//! Ensure that leaf 0x2 cache and TLB type values do not intersect,
//! since they share the same type field at struct cpuid_0x2_table.
//! Check at compile time that something is of a particular type.
//! Always evaluates to 1 so you may use it easily in comparisons.
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
//! We do the locked ops that don't return the old value as
//! a mask operation on a byte.
//! Compile time versions of __arch_hweightN()
//! writeable
//! Operation not permitted
//! Value too large for defined data type
//! restart by calling sys_restart_syscall
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
//! u64_to_user_ptr - cast a pointer passed as u64 from user space to void __user *
//! @x: The u64 value from user space, usually via IOCTL
//!
//! u64_to_user_ptr() simply casts a pointer passed as u64 from user space to void
//! __user * correctly. Using this lets us get rid of all the tiresome casts.
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
//! Change personality of the currently running process.
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
//! Shortcut for the queued_spin_lock_slowpath() function that allows
//! virt to hijack it.
//!
//! Returns:
//!   true - lock has been negotiated, all done;
//!   false - queued_spin_lock_slowpath() will do its thing.
//! A writer holds the lock
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
//! Change timeval to jiffies, trying to avoid the
//! most obvious overflows..
//!
//! And some not so obvious.
//!
//! Note that we don't want to return LONG_MAX, because
//! for various timeout reasons we often end up having
//! to wait "jiffies+1" in order to guarantee that we wait
//! at _least_ "jiffies" - so "jiffies+1" had better still
//! be positive.
//! Values for SRCU Tree srcu_data ->srcu_reader_flavor, but also used by rcutorture.
//! Suits me
//! percpu_counter batch for local add or sub
//! local_lock_init - Runtime initialize a lock instance
//! @lock:	The lock variable
//! Helper macro to avoid gfp flags if they are the default one
//! return the ->next pointer of a list_head in an rcu safe
//! way, we must not access it directly
//! MXCSR will be initialized
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
//! not fcntl
//! Valid hint values for F_{GET,SET}_RW_HINT. 0 is "not set", or can be
//! used to clear any hints previously set.
//! Gives us 8 prio classes with 13-bits of data for each class
//! per-IO O_SYNC
//! File supports atomic writes
//! Resource type
//! number of env pointers
//! Generate SD flag indexes
//! Desired design of maximum size and alignment (see RFC2553)
//! Mapping of registers to parameters for syscalls on x86-64 and x32
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
//! Maximum number of blkcg policies allowed to be registered concurrently.
//! Defined here to simplify include dependency.
//! Internal flags that share the struct mempolicy flags word with
//! "mode flags".  These flags are allocated from bit 0 up, as they
//! are never OR'ed into the mode in mempolicy API arguments.
//! should be >= TASK_COMM_LEN
//! in linux/sched.h
//! hlist_nulls_first_rcu - returns the first element of the hash list.
//! @head: the head of the list.
//! define the enumeration of all cgroup subsystems
//! To avoid confusing the compiler (and generating warnings) with code
//! that attempts to access what would be a 0-element array (i.e. sized
//! to a potentially empty array when CGROUP_SUBSYS_COUNT == 0), this
//! constant expression can be added.
//! 31 pointers + header align the folio_batch structure to a power of two
//! set if swap priority specified
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
//! page can be read
//! stack-like segment
//! ***DANGEROUS*** don't add guard page
//! tell ->d_revalidate() to trust no cache
//! This is a list of *what* is being read, not *how* nor *where*.
//! ld/ldx fields
//! Define a list of socket types which can be the argument for
//! skc_to_*_sock() helpers. All these sockets should have
//! sock_common as the first argument in its memory layout.
//! Currently, the max observed size in the kernel code is
//! JUMP_LABEL_NOP_SIZE/RELATIVEJUMP_SIZE, which are 5.
//! Raise it if needed.
//! modes for vm.memfd_noexec sysctl
//! These values are stored in task->ptrace_message
//! by ptrace_stop to describe the current syscall-stop.
//! SEIZE used, enable new behavior
//! Choose 31 as it makes rqspinlock_held cacheline-aligned.
//! Disallow growing, shrinking, or hole punching in the inode. Combined with
//! folio pinning, makes sure the inode's mapping stays fixed.
//!
//! In some ways similar to F_SEAL_GROW | F_SEAL_SHRINK, but can be removed and
//! isn't directly visible to userspace.
//! Flags in backing_dev_info::capability
//!
//! BDI_CAP_WRITEBACK:		Supports dirty page writeback, and dirty pages
//! 				should contribute to accounting
//! BDI_CAP_STRICTLIMIT:		Keep number of dirty pages below bdi threshold
//! These are specified by iBCS2
//! Epoll event masks
//! The PERF_METRICS MSR is modeled as several magic fixed-mode PMCs, one for
//! each TopDown metric event.
//!
//! Internally the TopDown metric events are mapped to the FxCtr 3 (SLOTS).
//! Macro to enforce the same ordering and stack sizes
//! PLIST_HEAD_INIT - static struct plist_head initializer
//! @head:	struct plist_head variable name
//! Temporary until swp_entry_t eliminated.
//! Swap table marker, 0x1 means shadow, 0x2 means PFN (SWP_TB_PFN_MARK)
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
// gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-weak-function-attribute
// gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-weak-variable-attribute
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
// roundup - round up to the next specified multiple
// @x: the value to up
// @y: multiple to round up to
//
// Rounds @x up to next multiple of @y. If @y will always be a power
// of 2, consider using the faster round_up().
// Overflow Flag
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
// Format: <modname>__<counter>_<line>_<fn>
// Format: __<prefix>__<iid><id>
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
// Concatenate two parameters, but allow them to be expanded beforehand.
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
// Xen paravirtual guest
// Intel-defined CPU features, CPUID level 0x00000007:0 (EBX), word 9
// "fsgsbase" RDFSBASE, WRFSBASE, RDGSBASE, WRGSBASE instructions
// "erms" Enhanced REP MOVSB/STOSB instructions
// "smap" Supervisor Mode Access Prevention
// "clflushopt" CLFLUSHOPT instruction
// "clwb" CLWB instruction
// IA32_TSC_DEADLINE and X2APIC MSRs need fencing
// "lass" Linear Address Space Separation
// Fast short REP STOSB
// "fred" Flexible Return and Event Delivery
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
// Atomic operations are already serializing on x86
// The following macros are non-atomic versions of their non-underscored
// counterparts.
// Check that the bitops prototypes are sane
// assign_bit - Assign value to a bit in memory
// @nr: the bit to set
// @addr: the address to start counting from
// @value: the value to assign
// No such file or directory
// Interrupted system call
// No such device or address
// Argument list too long
// Try again
// Out of memory
// Permission denied
// Bad address
// Device or resource busy
// No such device
// Invalid argument
// Text file busy
// No space left on device
// Math result not representable
// Operation not supported on transport endpoint
// Stale file handle
// Memory page has hardware error
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
// min_t - return minimum of two values, using the specified type
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
// swap - swap values of @a and @b
// @a: first value
// @b: second value
// Use these carefully: no type checking, and uses the arguments
// multiple times. Use for obvious constants only.
// net/core/page_pool.c
//
// page_pool uses additional free bits within this value to store data, see the
// definition of PP_DMA_INDEX_MASK in mm.h
// No static inline type checking - see Subtlety (1) above.
// for_each_cpu - iterate over every cpu in a mask
// @cpu: the (optionally unsigned) integer iterator
// @mask: the cpumask pointer
//
// After the loop, cpu is >= nr_cpu_ids.
// to_cpumask - convert a NR_CPUS bitmap to a struct cpumask *
// @bitmap: the bitmap
//
// There are a few places where cpumask_var_t isn't appropriate and
// static cpumasks must be used (eg. very early boot), yet we don't
// expose the definition of 'struct cpumask'.
//
// This does the conversion, and can be used as a constant initializer.
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
// SCHED_ISO: reserved but not implemented yet
// disable sas during sighandling
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
// srcu_read_lock().
// srcu_read_lock_nmisafe().
// 					0x4		// SRCU-lite is no longer with us.
// srcu_read_lock_fast(), also NMI-safe.
// srcu_read_lock_fast_updown().
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
// used to be O_SYNC, see below
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
// Resource extended types
// No address assigned yet
// Generate SD flag bits
// Implementation specific max size
// Mapping of registers to parameters for syscalls on i386
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
// page can be written
// page can be executed
// page may be used for atomic ops
// perform synchronous page faults for the mapping
// pages are locked
// PT_TRACE_* event enable flags
// We abuse bit 3 from flags to pass exact information, see
// perf_arch_misc_flags() and the comment with PERF_EFLAGS_EXACT.
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
macro_rules! ALIGN_DOWN__3 {
($x:expr, $a:expr) => {
__ALIGN_KERNEL!($x.wrapping_sub($a.wrapping_sub(1) as u64), $a)
};
}

macro_rules! ALIGN_DOWN {
($x:expr, $a:expr) => {
__ALIGN_KERNEL!($x.wrapping_sub($a.wrapping_sub(1)), $a)
};
}

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

macro_rules! BITS_PER_TYPE__2 {
($type:ty) => {
(core::mem::size_of::<$type>() as u64).wrapping_mul(8)
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

macro_rules! IS_ENCRYPTED {
($inode:expr) => {
(unsafe { (*$inode).i_flags }) & 16384u32
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

macro_rules! IS_ROOT {
($x:expr) => {
($x) == ((unsafe { (*$x).d_parent }) as *const dentry)
};
}

macro_rules! IS_SWAPFILE {
($inode:expr) => {
(unsafe { (*$inode).i_flags }) & 256u32
};
}

/// in mm/page_alloc.c
macro_rules! K {
($x:expr) => {
($x) << 2i32
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

macro_rules! S_ISBLK {
($m:expr) => {
(($m) as i32) & 61440 == 24576
};
}

macro_rules! S_ISREG {
($m:expr) => {
(($m) as i32) & 61440 == 32768
};
}

macro_rules! UL {
() => {
_UL!()
};
}

macro_rules! VM_BUG_ON_FOLIO {
() => {
VM_BUG_ON!();
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

macro_rules! VM_WARN_ON_ONCE_FOLIO {
() => {
BUILD_BUG_ON_INVALID!();
};
}

macro_rules! VM_WARN_ON_ONCE {
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

macro_rules! _AT {
($T:ty, $X:expr) => {
($X) as $T
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

macro_rules! __ALIGN_KERNEL__2 {
($x:expr, $a:expr) => {
__ALIGN_KERNEL_MASK!($x, (($a) as u32).wrapping_sub(1))
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

macro_rules! __SLAB_FLAG_BIT {
($nr:expr) => {
1u32 << $nr
};
}

macro_rules! __WRITE_ONCE__5 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut __SlateRecord_pgtable_64_types_h_21, $val) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE__26 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut *mut swap_info_struct, $val) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE__18 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut *mut __SlateRecord_types_h_197, $val) };
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

macro_rules! __WRITE_ONCE__6 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut i64, $val) };
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
macro_rules! __native_word__4 {
() => {
((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__15 {
() => {
((const { core::mem::size_of::<*mut __SlateRecord_types_h_197>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut __SlateRecord_types_h_197>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut __SlateRecord_types_h_197>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut __SlateRecord_types_h_197>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__8 {
() => {
((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__6 {
() => {
((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__7 {
() => {
((const { core::mem::size_of::<__SlateRecord_pgtable_64_types_h_21>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<__SlateRecord_pgtable_64_types_h_21>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<__SlateRecord_pgtable_64_types_h_21>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<__SlateRecord_pgtable_64_types_h_21>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__17 {
() => {
((const { core::mem::size_of::<*mut swap_info_struct>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut swap_info_struct>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut swap_info_struct>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut swap_info_struct>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__13 {
() => {
((const { core::mem::size_of::<*mut rb_node>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut rb_node>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut rb_node>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut rb_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
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

macro_rules! __pte {
($x:expr) => {
native_make_pte($x)
};
}

macro_rules! __ratelimit__5 {
($state:expr) => {
unsafe { ___ratelimit($state, c"swap_dup_entry_direct".as_ptr() as *const u8) }
};
}

macro_rules! __ratelimit {
($state:expr) => {
unsafe { ___ratelimit($state, c"get_swap_device".as_ptr() as *const u8) }
};
}

macro_rules! __rcu_dereference_protected__3 {
($p:expr, $c:expr) => {
{
let mut __v: *mut __SlateRecord_types_h_197;
RCU_LOCKDEP_WARN!(($c) == 0);
__v = $p;
__v
}
};
}

macro_rules! __rcu_dereference_protected {
($p:expr, $c:expr) => {
{
let mut __v: *mut __SlateRecord_types_h_197;
RCU_LOCKDEP_WARN__2!(($c) == 0);
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

/// This looks more complex than it should be. But we need to
/// get the type for the ~ right in round_down (it needs to be
/// as wide as the result!), and we want to evaluate the macro
/// arguments just once each.
macro_rules! __round_mask__5 {
($y:expr) => {
$y.wrapping_sub(1) as u64
};
}

/// This looks more complex than it should be. But we need to
/// get the type for the ~ right in round_down (it needs to be
/// as wide as the result!), and we want to evaluate the macro
/// arguments just once each.
macro_rules! __round_mask__2 {
($y:expr) => {
$y.wrapping_sub(1) as u64
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

macro_rules! __swab32 {
($x:expr) => {
unsafe { __slate_intrinsic_e94e9cf815168b51(($x) as u32) }
};
}

macro_rules! __swp_entry_to_pte {
($x:expr) => {
__pte!($x.val)
};
}

/// Shift up (to get rid of type), then down to get value
macro_rules! __swp_offset {
($x:expr) => {
!$x.val << 5i32 >> 14i32
};
}

/// Extract the high bits for type
macro_rules! __swp_type {
($x:expr) => {
$x.val >> 59i32
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

macro_rules! __types_ok__8 {
($ux:expr, $uy:expr) => {
__sign_use__5!($ux) & __sign_use!($uy)
};
}

macro_rules! __types_ok {
($ux:expr, $uy:expr) => {
__sign_use__2!($ux) & __sign_use__2!($uy)
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
macro_rules! __verify_pcpu_ptr__30 {
() => {
loop {
let __v: *const () = core::ptr::null_mut::<*mut swap_info_struct>() as *const ();
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
macro_rules! __verify_pcpu_ptr__35 {
() => {
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<percpu_swap_cluster>() as *const ();
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
macro_rules! __verify_pcpu_ptr__19 {
() => {
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<*mut swap_info_struct>() as *const ();
if true {
break;
}
}
};
}

macro_rules! _static_cpu_has__2 {
($bit:expr) => {
__static_cpu_has__const4(($bit) as u16) as i32
};
}

/// abs - return absolute value of an argument
/// @x: the value.
///
/// If it is unsigned type, @x is converted to signed type first.
/// char is treated as if it was signed (regardless of whether it really is)
/// but the macro's return type is preserved as char.
///
/// NOTE, for signed type if @x is the minimum, the returned result is undefined
/// as there is not enough bits to represent it as a positive number.
///
/// Return: an absolute value of @x.
macro_rules! abs {
($x:expr) => {
{
let mut __v: i32;
let __v2: i32 = ($x) as i32;
__v = if __v2 < 0 { __v2.wrapping_neg() } else { __v2 };
__v
}
};
}

/// abs - return absolute value of an argument
/// @x: the value.
///
/// If it is unsigned type, @x is converted to signed type first.
/// char is treated as if it was signed (regardless of whether it really is)
/// but the macro's return type is preserved as char.
///
/// NOTE, for signed type if @x is the minimum, the returned result is undefined
/// as there is not enough bits to represent it as a positive number.
///
/// Return: an absolute value of @x.
macro_rules! abs__3 {
($x:expr) => {
{
let mut __v: i32;
let __v2: i32 = $x;
__v = if __v2 < 0 { __v2.wrapping_neg() } else { __v2 };
__v
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

macro_rules! alloc_hooks__2 {
($_do_alloc:expr) => {
{
let mut __v: *mut folio;
let mut __v2: *mut folio;
let mut __v3: *mut folio = unsafe { core::mem::zeroed() };
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

macro_rules! alloc_tag_restore {
() => {
while false {
}
};
}

macro_rules! arch_atomic64_sub_return {
($i:expr, $v:expr) => {
arch_atomic64_add_return($i.wrapping_neg(), $v)
};
}

/// Remapping spinlock architecture specific functions to the corresponding
/// queued spinlock functions.
macro_rules! arch_spin_is_locked {
($l:expr) => {
queued_spin_is_locked($l)
};
}

macro_rules! blk_queue_rot {
($q:expr) => {
(unsafe { (*$q).limits.features }) & 4u32
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

macro_rules! cond_resched__30 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/mm/swapfile.c".as_ptr() as *const u8, 2611i32, 0u32);
__v = _cond_resched();
};
}

macro_rules! cond_resched__53 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/mm/swapfile.c".as_ptr() as *const u8, 2780i32, 0u32);
__v = _cond_resched();
};
}

macro_rules! cond_resched__66 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/mm/swapfile.c".as_ptr() as *const u8, 306i32, 0u32);
__v = _cond_resched();
};
}

macro_rules! cond_resched__69 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/mm/swapfile.c".as_ptr() as *const u8, 2697i32, 0u32);
__v = _cond_resched();
};
}

macro_rules! cond_resched__75 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/mm/swapfile.c".as_ptr() as *const u8, 1081i32, 0u32);
__v = _cond_resched();
};
}

macro_rules! cond_resched__77 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/mm/swapfile.c".as_ptr() as *const u8, 294i32, 0u32);
__v = _cond_resched();
};
}

macro_rules! cond_resched__87 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/mm/swapfile.c".as_ptr() as *const u8, 2727i32, 0u32);
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
macro_rules! container_of__26 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__50!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__35 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__38!() as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__7 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__14!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__42 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__49!($type) as isize)) }) as *mut ()) as *mut $type;
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
let mut __v: *mut swap_cluster_info;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__37!() as isize)) }) as *mut ()) as *mut swap_cluster_info;
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
macro_rules! container_of__53 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__41!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! cpu_feature_enabled__2 {
($bit:expr) => {
_static_cpu_has__2!($bit)
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

macro_rules! folio_add_anon_rmap_pte {
($folio:expr, $page:expr, $vma:expr, $address:expr, $flags:expr) => {
unsafe { folio_add_anon_rmap_ptes($folio, $page, 1i32, $vma, $address, $flags) };
};
}

/// folio_page - Return a page from a folio.
/// @folio: The folio.
/// @n: The page number to return.
///
/// @n is relative to the start of the folio.  This function does not
/// check that the page number lies within @folio; the caller is presumed
/// to have a reference to the page.
macro_rules! folio_page {
($folio:expr, $n:expr) => {
unsafe { core::ptr::addr_of_mut!((*$folio).__slate_anon_0.page) }.wrapping_offset(($n) as isize)
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

/// Shortcuts
macro_rules! kmem_cache_zalloc {
($_k:expr, $_flags:expr) => {
alloc_hooks!(unsafe { kmem_cache_alloc_noprof($_k, ($_flags) | 256u32) })
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

macro_rules! likely {
($x:expr) => {
!(!$x) as i64
};
}

macro_rules! likely__7 {
($x:expr) => {
!$x.is_null() as i64
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__6 {
($ptr:expr, $type:ty) => {
container_of__12!($ptr, $type)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__18 {
($ptr:expr) => {
container_of__45!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__26 {
($ptr:expr, $type:ty) => {
container_of__35!($ptr, $type)
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
macro_rules! list_first_entry__19 {
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
macro_rules! list_first_entry__14 {
($ptr:expr, $type:ty) => {
list_entry__6!(unsafe { (*$ptr).next }, $type)
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__20 {
($pos:expr, $member:ident) => {
list_entry__18!(unsafe { (*$pos).$member.next })
};
}

macro_rules! lockdep_assert_held {
($l:expr) => {
__assume_ctx_lock!($l);
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

macro_rules! offsetof__18 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, info.badpages) as u64
};
}

macro_rules! offsetof__37 {
() => {
core::mem::offset_of!(swap_cluster_info, list) as u64
};
}

macro_rules! offsetof__38 {
() => {
440u64
};
}

macro_rules! offsetof__41 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, discard_work) as u64
};
}

macro_rules! offsetof__7 {
() => {
core::mem::offset_of!(swap_info_struct, list.node_list) as u64
};
}

macro_rules! offsetof__11 {
() => {
core::mem::offset_of!(swap_info_struct, avail_list.node_list) as u64
};
}

macro_rules! offsetof__49 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, users) as u64
};
}

macro_rules! offsetof__50 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, reclaim_work) as u64
};
}

macro_rules! offsetof__13 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, list) as u64
};
}

macro_rules! offsetof__14 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, rb_node) as u64
};
}

macro_rules! offsetof__26 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, magic.magic) as u64
};
}

macro_rules! p4d_addr_end {
($addr:expr, $end:expr) => {
{
let mut __v: u64;
let __v2: u64 = $addr.wrapping_add(_AC!() << 39i32) & !(_AC!() << 39i32).wrapping_sub(1);
__v = if __v2.wrapping_sub(1) < $end.wrapping_sub(1) { __v2 } else { $end };
__v
}
};
}

macro_rules! p4d_val {
($x:expr) => {
native_p4d_val($x)
};
}

macro_rules! page_address__2 {
($page:expr) => {
lowmem_page_address($page)
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

macro_rules! page_ref_tracepoint_active {
() => {
0i32
};
}

macro_rules! page_to_virt {
($x:expr) => {
__va!(PFN_PHYS!(((unsafe { $x.offset_from((vmemmap_base as *mut page) as *const page) }) as i64) as u64))
};
}

macro_rules! per_cpu_offset {
($x:expr) => {
unsafe { __per_cpu_offset[($x) as usize] }
};
}

macro_rules! per_cpu_ptr__15 {
($ptr:expr, $cpu:expr) => {
{
let mut __v: *mut *mut swap_info_struct;
__verify_pcpu_ptr__30!();
__v = (((($ptr) as u64) as *mut *mut swap_info_struct) as u64).wrapping_add(per_cpu_offset!($cpu)) as *mut *mut swap_info_struct;
__v
}
};
}

macro_rules! pgd_addr_end {
($addr:expr, $end:expr) => {
{
let mut __v: u64;
let __v2: u64 = $addr.wrapping_add(_AC!() << unsafe { pgdir_shift }) & !(_AC!() << unsafe { pgdir_shift }).wrapping_sub(1);
__v = if __v2.wrapping_sub(1) < $end.wrapping_sub(1) { __v2 } else { $end };
__v
}
};
}

/// Must be a compile-time constant, so implement it as a macro
macro_rules! pgd_index {
($a:expr) => {
($a) >> unsafe { pgdir_shift } & 511u64
};
}

macro_rules! pgd_offset {
($mm:expr, $address:expr) => {
pgd_offset_pgd(unsafe { (*$mm).__slate_anon_0.pgd }, $address)
};
}

macro_rules! pgd_val {
($x:expr) => {
native_pgd_val($x)
};
}

macro_rules! pgprot_val {
($x:expr) => {
$x.pgprot
};
}

macro_rules! pgtable_l5_enabled__4 {
() => {
cpu_feature_enabled__2!(528i32)
};
}

macro_rules! pmd_addr_end {
($addr:expr, $end:expr) => {
{
let mut __v: u64;
let __v2: u64 = $addr.wrapping_add(_AC!() << 21i32) & !(_AC!() << 21i32).wrapping_sub(1);
__v = if __v2.wrapping_sub(1) < $end.wrapping_sub(1) { __v2 } else { $end };
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

macro_rules! pte_next_pfn {
($pte:expr) => {
pte_advance_pfn($pte, 1u64)
};
}

macro_rules! pte_unmap_unlock {
($pte:expr, $ptl:expr) => {
loop {
spin_unlock($ptl);
pte_unmap($pte);
if true {
break;
}
}
};
}

macro_rules! pte_val {
($x:expr) => {
native_pte_val($x)
};
}

macro_rules! pud_addr_end {
($addr:expr, $end:expr) => {
{
let mut __v: u64;
let __v2: u64 = $addr.wrapping_add(_AC!() << 30i32) & !(_AC!() << 30i32).wrapping_sub(1);
__v = if __v2.wrapping_sub(1) < $end.wrapping_sub(1) { __v2 } else { $end };
__v
}
};
}

macro_rules! pud_val {
($x:expr) => {
native_pud_val($x)
};
}

macro_rules! raw_spin_is_locked {
($lock:expr) => {
arch_spin_is_locked!(unsafe { core::ptr::addr_of_mut!((*$lock).raw_lock) })
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

/// Define the various spin_lock methods.  Note we define these
/// regardless of whether CONFIG_SMP or CONFIG_PREEMPTION are set. The
/// various methods are defined as nops in the case they are not
/// required.
macro_rules! raw_spin_trylock {
($lock:expr) => {
unsafe { _raw_spin_trylock($lock) }
};
}

macro_rules! raw_spin_unlock {
($lock:expr) => {
unsafe { _raw_spin_unlock($lock) };
};
}

macro_rules! rb_entry {
($ptr:expr, $type:ty) => {
container_of__7!($ptr, $type)
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
__rcu_dereference_protected!($p, $c)
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
__rcu_dereference_protected__3!($p, $c)
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

/// round_down - round down to next specified power of 2
/// @x: the value to round
/// @y: multiple to round down to (must be a power of 2)
///
/// Rounds @x down to next multiple of @y (which must be a power of 2).
/// To perform arbitrary rounding down, use rounddown() below.
macro_rules! round_down__3 {
($x:expr, $y:expr) => {
($x) & !__round_mask__5!($y)
};
}

/// round_up - round up to next specified power of 2
/// @x: the value to round
/// @y: multiple to round up to (must be a power of 2)
///
/// Rounds @x up to next multiple of @y (which must be a power of 2).
/// To perform arbitrary rounding up, use roundup() below.
macro_rules! round_up__2 {
($x:expr, $y:expr) => {
($x.wrapping_sub(1) | __round_mask__2!($y)).wrapping_add(1)
};
}

macro_rules! set_pte {
($ptep:expr, $pte:expr) => {
native_set_pte($ptep, $pte);
};
}

macro_rules! set_pte_at {
($mm:expr, $addr:expr, $ptep:expr, $pte:expr) => {
set_ptes($mm, $addr, $ptep, $pte, 1u32);
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

macro_rules! static_branch_unlikely__3 {
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

macro_rules! test_bit__5 {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if (unsafe { __slate_intrinsic_a39e427d3b6c2df4($nr) }) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*$addr) } {
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

macro_rules! type_max {
() => {
__type_max!()
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

/// virt_to_page(kaddr) returns a valid pointer if and only if
/// virt_addr_valid(kaddr) returns true.
macro_rules! virt_to_page {
($kaddr:expr) => {
((unsafe { vmemmap_base }) as *mut page).wrapping_offset((__pa!($kaddr) >> 12i32) as isize)
};
}

macro_rules! wake_up_interruptible {
($x:expr) => {
unsafe { __wake_up($x, 1u32, 1i32, core::ptr::null_mut::<()>()) };
};
}

unsafe extern "C" {
    static mut kmsg_fops: file_operations;
    static mut __per_cpu_offset: [u64; 64];
    static mut this_cpu_off: u64;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut phys_base: u64;
    static mut page_offset_base: u64;
    static mut vmemmap_base: u64;
    static mut pgdir_shift: u32;
    static mut ptrs_per_p4d: u32;
    static mut __supported_pte_mask: u64;
    static mut __SCK__might_resched: static_call_key;
    static mut node_data: [*mut pglist_data; 0];
    static mut __cpu_possible_mask: cpumask;
    static mut boot_cpu_data: cpuinfo_x86;
    static mut cpuinfo_op: seq_operations;
    static mut __preempt_count: u64;
    static mut __SCK__preempt_schedule: static_call_key;
    static mut __SCK__cond_resched: static_call_key;
    static mut system_percpu_wq: *mut workqueue_struct;
    static mut percpu_counter_batch: i32;
    static mut init_mm: mm_struct;
    static mut gfp_allowed_mask: u32;
    static mut sighand_cachep: *mut kmem_cache;
    static mut __tracepoint_mmap_lock_start_locking: tracepoint;
    static mut __tracepoint_mmap_lock_acquire_returned: tracepoint;
    static mut __tracepoint_mmap_lock_released: tracepoint;
    static mut mmlist_lock: spinlock;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut kstat: kernel_stat;
    static mut vm_committed_as: percpu_counter;
    static mut vm_committed_as_batch: i32;
    static mut perf_callchain_entry: perf_callchain_entry;
    static mut swap_bdev_ops: swap_ops;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn ___ratelimit(arg0: *mut ratelimit_state, arg1: *const u8) -> i32;
    fn _printk(arg0: *const u8, ...) -> i32;
    fn __bad_size_call_parameter();
    fn __compiletime_assert_0();
    fn ____wrong_branch_error() -> bool;
    fn __xchg_wrong_size();
    fn __cmpxchg_wrong_size();
    fn __xadd_wrong_size();
    fn _find_next_bit(arg0: *const u64, arg1: u64, arg2: u64) -> u64;
    fn memcmp(arg0: *const (), arg1: *const (), arg2: u64) -> i32;
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
    fn _raw_spin_trylock(arg0: *mut raw_spinlock) -> i32;
    fn _raw_spin_unlock(arg0: *mut raw_spinlock);
    fn __SCT__cond_resched() -> i32;
    fn __wake_up(arg0: *mut wait_queue_head, arg1: u32, arg2: i32, arg3: *mut ()) -> i32;
    fn __init_swait_queue_head(arg0: *mut swait_queue_head, arg1: *const u8, arg2: *mut lock_class_key);
    fn wait_for_completion(arg0: *mut completion);
    fn complete(arg0: *mut completion);
    fn mutex_lock(arg0: *mut mutex);
    fn mutex_unlock(arg0: *mut mutex);
    fn call_rcu(arg0: *mut callback_head, arg1: Option<unsafe extern "C" fn(*mut callback_head)>);
    fn synchronize_rcu();
    fn __rcu_read_lock();
    fn __rcu_read_unlock();
    fn rcu_is_watching() -> bool;
    fn rb_insert_color(arg0: *mut rb_node, arg1: *mut rb_root);
    fn rb_erase(arg0: *mut rb_node, arg1: *mut rb_root);
    fn rb_next(arg0: *const rb_node) -> *mut rb_node;
    fn mas_find(arg0: *mut ma_state, arg1: u64) -> *mut ();
    fn down_read(arg0: *mut rw_semaphore);
    fn down_write(arg0: *mut rw_semaphore);
    fn up_read(arg0: *mut rw_semaphore);
    fn up_write(arg0: *mut rw_semaphore);
    fn queue_work_on(arg0: i32, arg1: *mut workqueue_struct, arg2: *mut work_struct) -> bool;
    fn flush_work(arg0: *mut work_struct) -> bool;
    fn percpu_counter_add_batch(arg0: *mut percpu_counter, arg1: i64, arg2: i32);
    fn __compiletime_assert_187();
    fn folio_alloc_noprof(arg0: u32, arg1: u32) -> *mut folio;
    fn mmput(arg0: *mut mm_struct);
    fn __mmap_lock_do_trace_start_locking(arg0: *mut mm_struct, arg1: bool);
    fn __mmap_lock_do_trace_acquire_returned(arg0: *mut mm_struct, arg1: bool, arg2: bool);
    fn __mmap_lock_do_trace_released(arg0: *mut mm_struct, arg1: bool);
    fn __compiletime_assert_290();
    fn capable(arg0: i32) -> bool;
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_359();
    fn __compiletime_assert_384();
    fn percpu_ref_init(arg0: *mut percpu_ref, arg1: Option<unsafe extern "C" fn(*mut percpu_ref)>, arg2: u32, arg3: u32) -> i32;
    fn percpu_ref_exit(arg0: *mut percpu_ref);
    fn percpu_ref_kill_and_confirm(arg0: *mut percpu_ref, arg1: Option<unsafe extern "C" fn(*mut percpu_ref)>);
    fn percpu_ref_resurrect(arg0: *mut percpu_ref);
    fn percpu_ref_is_zero(arg0: *mut percpu_ref) -> bool;
    fn __compiletime_assert_385();
    fn __kmem_cache_create_args(arg0: *const u8, arg1: u32, arg2: *mut kmem_cache_args, arg3: u32) -> *mut kmem_cache;
    fn kfree(arg0: *const ());
    fn __compiletime_assert_390();
    fn __builtin_unreachable();
    fn kmem_cache_alloc_noprof(arg0: *mut kmem_cache, arg1: u32) -> *mut ();
    fn kmem_cache_free(arg0: *mut kmem_cache, arg1: *mut ());
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kvmalloc_node_noprof(arg0: u64, arg1: u64, arg2: u32, arg3: i32) -> *mut ();
    fn kvfree(arg0: *const ());
    fn __compiletime_assert_399();
    fn __compiletime_assert_400();
    fn file_open_name(arg0: *mut filename, arg1: i32, arg2: u16) -> *mut file;
    fn filp_close(arg0: *mut file, arg1: *mut ()) -> i32;
    fn getname_flags(arg0: *const u8, arg1: i32) -> *mut filename;
    fn putname(arg0: *mut filename);
    fn inode_dio_wait(arg0: *mut inode);
    fn __compiletime_assert_424();
    fn pgd_clear_bad(arg0: *mut __SlateRecord_pgtable_types_h_295);
    fn p4d_clear_bad(arg0: *mut __SlateRecord_pgtable_types_h_342);
    fn pud_clear_bad(arg0: *mut __SlateRecord_pgtable_types_h_368);
    fn cache_get_priv_group(arg0: *mut cacheinfo) -> *const attribute_group;
    fn __folio_put(arg0: *mut folio);
    fn mm_trace_rss_stat(arg0: *mut mm_struct, arg1: i32);
    fn __pte_offset_map(arg0: *mut __SlateRecord_pgtable_64_types_h_22, arg1: u64, arg2: *mut __SlateRecord_pgtable_64_types_h_22) -> *mut __SlateRecord_pgtable_64_types_h_21;
    fn pte_offset_map_lock(arg0: *mut mm_struct, arg1: *mut __SlateRecord_pgtable_64_types_h_22, arg2: u64, arg3: *mut *mut spinlock) -> *mut __SlateRecord_pgtable_64_types_h_21;
    fn __compiletime_assert_518();
    fn blkdev_issue_discard(arg0: *mut block_device, arg1: u64, arg2: u64, arg3: u32) -> i32;
    fn I_BDEV(arg0: *mut inode) -> *mut block_device;
    fn filemap_write_and_wait_range(arg0: *mut address_space, arg1: i64, arg2: i64) -> i32;
    fn read_cache_folio(arg0: *mut address_space, arg1: u64, arg2: Option<unsafe extern "C" fn(*mut file, *mut folio) -> i32>, arg3: *mut file) -> *mut folio;
    fn __folio_lock(arg0: *mut folio);
    fn folio_unlock(arg0: *mut folio);
    fn folio_wait_writeback(arg0: *mut folio);
    fn seq_open(arg0: *mut file, arg1: *const seq_operations) -> i32;
    fn seq_read(arg0: *mut file, arg1: *mut u8, arg2: u64, arg3: *mut i64) -> i64;
    fn seq_lseek(arg0: *mut file, arg1: i64, arg2: i32) -> i64;
    fn seq_release(arg0: *mut inode, arg1: *mut file) -> i32;
    fn seq_write(arg0: *mut seq_file, arg1: *const (), arg2: u64) -> i32;
    fn seq_printf(arg0: *mut seq_file, arg1: *const u8, ...);
    fn seq_putc(arg0: *mut seq_file, arg1: u8);
    fn __seq_puts(arg0: *mut seq_file, arg1: *const u8);
    #[link_name = "strlen"]
    fn __builtin_strlen(arg0: *const u8) -> u64;
    fn seq_file_path(arg0: *mut seq_file, arg1: *mut file, arg2: *const u8) -> i32;
    fn generic_swapfile_activate(arg0: *mut swap_info_struct, arg1: *mut file, arg2: *mut u64) -> i32;
    fn security_vm_enough_memory_mm(arg0: *mut mm_struct, arg1: i64) -> i32;
    fn shmem_unuse(arg0: u32) -> i32;
    fn proc_create(arg0: *const u8, arg1: u16, arg2: *mut proc_dir_entry, arg3: *const proc_ops) -> *mut proc_dir_entry;
    fn folio_add_anon_rmap_ptes(arg0: *mut folio, arg1: *mut page, arg2: i32, arg3: *mut vm_area_struct, arg4: u64, arg5: i32);
    fn folio_add_new_anon_rmap(arg0: *mut folio, arg1: *mut vm_area_struct, arg2: u64, arg3: i32);
    fn plist_add(arg0: *mut plist_node, arg1: *mut plist_head);
    fn plist_del(arg0: *mut plist_node, arg1: *mut plist_head);
    fn plist_requeue(arg0: *mut plist_node, arg1: *mut plist_head);
    fn __compiletime_assert_731();
    fn sio_pool_init() -> i32;
    fn swap_cache_get_folio(arg0: __SlateRecord_mm_types_h_288) -> *mut folio;
    fn swap_cache_del_folio(arg0: *mut folio);
    fn __swap_cache_add_folio(arg0: *mut swap_cluster_info, arg1: *mut folio, arg2: __SlateRecord_mm_types_h_288);
    fn swapin_readahead(arg0: __SlateRecord_mm_types_h_288, arg1: u32, arg2: *mut vm_fault) -> *mut folio;
    fn __compiletime_assert_732();
    fn __compiletime_assert_736();
    fn __compiletime_assert_735();
    fn __compiletime_assert_742();
    fn __compiletime_assert_746();
    fn __compiletime_assert_745();
    fn __compiletime_assert_752();
    fn __compiletime_assert_761();
    fn __compiletime_assert_762();
    fn __compiletime_assert_774();
    fn __compiletime_assert_777();
    fn __compiletime_assert_786();
    fn __compiletime_assert_788();
    fn __compiletime_assert_790();
    fn __compiletime_assert_806();
    fn folio_add_lru_vma(arg0: *mut folio, arg1: *mut vm_area_struct);
    fn __compiletime_assert_818();
    fn __compiletime_assert_821();
    fn __compiletime_assert_822();
    fn __compiletime_assert_823();
    fn __compiletime_assert_824();
    fn __compiletime_assert_826();
    fn __compiletime_assert_827();
    fn __compiletime_assert_828();
    fn __compiletime_assert_829();
    fn __compiletime_assert_841();
    fn __compiletime_assert_850();
    fn __compiletime_assert_851();
    fn __compiletime_assert_854();
    fn __compiletime_assert_860();
    fn __compiletime_assert_866();
    fn __compiletime_assert_869();
    fn __compiletime_assert_872();
    fn __compiletime_assert_879();
    fn __compiletime_assert_882();
    fn __compiletime_assert_885();
    fn __compiletime_assert_888();
    fn __compiletime_assert_891();
    fn __compiletime_assert_899();
    fn __compiletime_assert_906();
    fn __compiletime_assert_914();
    fn __compiletime_assert_922();
    fn __compiletime_assert_925();
    fn __compiletime_assert_926();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_1c5364fcd9da51fe(arg0: u32) -> bool;
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
    #[link_name = "llvm.is.constant.i8"]
    fn __slate_intrinsic_70b71f185ebb8c4a(arg0: u8) -> bool;
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_a39e427d3b6c2df4(arg0: i32) -> bool;
    #[link_name = "llvm.is.constant.i64"]
    fn __slate_intrinsic_e1c548fa55baa574(arg0: i64) -> bool;
    #[link_name = "llvm.bswap.i32"]
    fn __slate_intrinsic_e94e9cf815168b51(arg0: u32) -> u32;
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


#[repr(C)]
#[derive(Clone, Copy)]
struct sysinfo {
    uptime: i64,
    loads: [u64; 3],
    totalram: u64,
    freeram: u64,
    sharedram: u64,
    bufferram: u64,
    totalswap: u64,
    freeswap: u64,
    procs: u16,
    pad: u16,
    totalhigh: u64,
    freehigh: u64,
    mem_unit: u32,
    _f: [u8; 0],
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
struct timespec64 {
    tv_sec: i64,
    tv_nsec: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_bitmap {
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


#[inline(always)]
fn __must_check_overflow(overflow: bool) -> bool {
unlikely!(overflow) != 0
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

// flex_array_size() - Calculate size of a flexible array member
//                     within an enclosing structure.
// @p: Pointer to the structure.
// @member: Name of the flexible array member.
// @count: Number of elements in the array.
//
// Calculates size of a flexible array of @count number of @member
// elements, at the end of structure @p.
//
// Return: number of bytes needed or SIZE_MAX on overflow.
// struct_size() - Calculate size of structure with trailing flexible array.
// @p: Pointer to the structure.
// @member: Name of the array member.
// @count: Number of elements in the array.
//
// Calculates size of memory needed for structure of @p followed by an
// array of @count number of @member elements.
//
// Return: number of bytes needed or SIZE_MAX on overflow.
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
// __swahw32 - return a word-swapped 32-bit value
// @x: value to wordswap
//
// __swahw32(0x12340000) is 0x00001234
// __swahb32 - return a high and low byte-swapped 32-bit value
// @x: value to byteswap
//
// __swahb32(0x12345678) is 0x34127856
#[inline(always)]
fn __swab32p(p: *const u32) -> u32 {
__swab32!(unsafe { *p })
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
fn __swab32s(p: *mut u32) {
unsafe {
*p = __swab32p(p as *const u32);
}
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
struct fred_cs {
    __slate_bits_0: __slate_bits::__SlateBits102U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fred_ss {
    __slate_bits_0: __slate_bits::__SlateBits103U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_ptrace_h_144 {
    cs: u16,
    csx: u64,
    fred_cs: fred_cs,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_ptrace_h_156 {
    ss: u16,
    ssx: u64,
    fred_ss: fred_ss,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pt_regs {
    r15: u64,
    r14: u64,
    r13: u64,
    r12: u64,
    bp: u64,
    bx: u64,
    r11: u64,
    r10: u64,
    r9: u64,
    r8: u64,
    ax: u64,
    cx: u64,
    dx: u64,
    si: u64,
    di: u64,
    orig_ax: u64,
    ip: u64,
    __slate_anon_17: __SlateRecord_ptrace_h_144,
    flags: u64,
    sp: u64,
    __slate_anon_20: __SlateRecord_ptrace_h_156,
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

#[repr(C, align(4))]
#[derive(Clone, Copy)]
struct leaf_0x0_0 {
    __slate_bits_0: __slate_bits::__SlateBits107U0,
}


#[repr(C, align(4))]
#[derive(Clone, Copy)]
struct leaf_0x1_0 {
    __slate_bits_0: __slate_bits::__SlateBits108U0,
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

#[repr(C)]
#[derive(Clone, Copy)]
struct range {
    start: u64,
    end: u64,
}


#[inline(always)]
fn const_test_bit(nr: u64, addr: *const u64) -> bool {
let p: *const u64 = (addr as *const u64).wrapping_offset(BIT_WORD!(nr) as isize);
let mask: u64 = BIT_MASK!(nr);
let val: u64 = unsafe { *p };
val & mask != 0
}

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
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
fn arch_test_and_set_bit(mut nr: i64, mut addr: *mut u64) -> bool {
let mut __v0: bool;
let mut c: bool = false;
unsafe {
let __t0: u8;
core::arch::asm!("lock  btsq  {2:r}, ({0:r})\nsetc {1}", in(reg) addr, lateout(reg_byte) __t0, in(reg) nr as u64, options(nostack, att_syntax));
c = (__t0 as u8) != 0;
}
__v0 = c;
__v0
}

#[inline(always)]
fn arch_test_and_set_bit_lock(nr: i64, addr: *mut u64) -> bool {
arch_test_and_set_bit(nr, addr)
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

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_jump_label_h_95 {
    r#type: u64,
    entries: *mut jump_entry,
    next: *mut static_key_mod,
}


#[inline(always)]
fn instrument_read_write(v: *const (), size: u64) {
kasan_check_write(v, size as u32);
kcsan_check_read_write!(v, size);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct static_key_mod {
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
struct __SlateRecord_find_h_67 {
}


#[inline(always)]
fn set_bit(nr: i64, addr: *mut u64) {
instrument_atomic_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_set_bit(nr, addr);
}

#[inline(always)]
fn _test_bit(nr: u64, addr: *const u64) -> bool {
instrument_atomic_read(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_bit(nr, addr)
}

fn test_and_set_bit_lock(nr: i64, addr: *mut u64) -> bool {
instrument_atomic_read_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_and_set_bit_lock(nr, addr)
}

#[inline(always)]
fn __ilog2_u32(n: u32) -> i32 {
fls(n).wrapping_sub(1)
}

#[inline(always)]
fn __ilog2_u64(n: u64) -> i32 {
fls64(n).wrapping_sub(1)
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
    __slate_bits_0: __slate_bits::__SlateBits160U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_processor_h_119 {
    __slate_bits_0: __slate_bits::__SlateBits161U0,
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
fn native_pgd_val(pgd: __SlateRecord_pgtable_types_h_295) -> u64 {
pgd.pgd & !0
}

fn pgd_flags(pgd: __SlateRecord_pgtable_types_h_295) -> u64 {
native_pgd_val(pgd) & !(((!(_AC!() << 12i32).wrapping_sub(1) as i64) as u64) & 4503599627370495u64)
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_types_h_342 {
    p4d: u64,
}


fn native_p4d_val(p4d: __SlateRecord_pgtable_types_h_342) -> u64 {
p4d.p4d
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_types_h_368 {
    pud: u64,
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


fn native_pud_val(pud: __SlateRecord_pgtable_types_h_368) -> u64 {
pud.pud
}

fn p4d_pfn_mask(p4d: __SlateRecord_pgtable_types_h_342) -> u64 {
// No 512 GiB huge pages yet
((!(_AC!() << 12i32).wrapping_sub(1) as i64) as u64) & 4503599627370495u64
}

fn p4d_flags_mask(p4d: __SlateRecord_pgtable_types_h_342) -> u64 {
!p4d_pfn_mask(p4d)
}

fn p4d_flags(p4d: __SlateRecord_pgtable_types_h_342) -> u64 {
native_p4d_val(p4d) & p4d_flags_mask(p4d)
}

fn pud_pfn_mask(pud: __SlateRecord_pgtable_types_h_368) -> u64 {
if native_pud_val(pud) & _AT!(u64, 1i32) << 7i32 != 0 {
return ((!(_AC!() << 30i32).wrapping_sub(1) as i64) as u64) & 4503599627370495u64;
} else {
return ((!(_AC!() << 12i32).wrapping_sub(1) as i64) as u64) & 4503599627370495u64;
}
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
}


fn pud_flags_mask(pud: __SlateRecord_pgtable_types_h_368) -> u64 {
!pud_pfn_mask(pud)
}

fn pud_flags(pud: __SlateRecord_pgtable_types_h_368) -> u64 {
native_pud_val(pud) & pud_flags_mask(pud)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
}


fn native_make_pte(val: u64) -> __SlateRecord_pgtable_64_types_h_21 {
let mut __slate_compound_0: __SlateRecord_pgtable_64_types_h_21 = unsafe { core::mem::zeroed() };
return unsafe { *{
__slate_compound_0 = __SlateRecord_pgtable_64_types_h_21 { pte: val };
core::ptr::addr_of_mut!(__slate_compound_0)
} };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct futex_pi_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct robust_list_head {
}


fn native_pte_val(pte: __SlateRecord_pgtable_64_types_h_21) -> u64 {
pte.pte
}

fn pte_flags(pte: __SlateRecord_pgtable_64_types_h_21) -> u64 {
native_pte_val(pte) & !(((!(_AC!() << 12i32).wrapping_sub(1) as i64) as u64) & 4503599627370495u64)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct futex_private_hash {
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits137U0,
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
struct sem_undo_list {
}


#[inline(always)]
fn IS_ERR_OR_NULL(ptr: *const ()) -> bool {
unlikely!(ptr.is_null()) != 0 || IS_ERR_VALUE!(ptr as u64) != 0
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_mmap_lock_start_locking, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_mmap_lock_acquire_returned, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_mmap_lock_released, const 0 as u8, label {
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
    __slate_bits_0: __slate_bits::__SlateBits144U0,
}


#[inline(always)]
fn arch_atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
// Note for KASAN: we deliberately don't use READ_ONCE_NOCHECK() here,
// it's non-inlined function that increases binary size and stack usage.
__READ_ONCE!((*v).counter)
}

#[inline(always)]
fn arch_atomic_inc(mut v: *mut __SlateRecord_types_h_190) {
unsafe {
core::arch::asm!("lock incl ({0:r})", in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer_clock_base {
}


#[inline(always)]
fn arch_atomic_dec(mut v: *mut __SlateRecord_types_h_190) {
unsafe {
core::arch::asm!("lock decl ({0:r})", in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
}
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
struct seccomp_filter {
}


#[inline(always)]
fn arch_atomic_try_cmpxchg(v: *mut __SlateRecord_types_h_190, old: &mut i32, mut __new: i32) -> bool {
let mut __v0: i64;
let mut success: bool = false;
let mut __old: i32 = unsafe { *old };
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __ptr: *mut u8 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u8;
unsafe {
let __t0: u8;
let __t1: u32;
core::arch::asm!("lock cmpxchgb {2:e}, ({1:r})\nsetz {0}", lateout(reg_byte) __t0, in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t1, options(nostack, att_syntax));
success = (__t0 as u8) != 0;
__old = unsafe { core::mem::transmute::<u32, i32>(__t1 as u32) };
}
}
2 => {
let mut __ptr: *mut u16 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u16;
unsafe {
let __t2: u8;
let __t3: u32;
core::arch::asm!("lock cmpxchgw {2:e}, ({1:r})\nsetz {0}", lateout(reg_byte) __t2, in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t3, options(nostack, att_syntax));
success = (__t2 as u8) != 0;
__old = unsafe { core::mem::transmute::<u32, i32>(__t3 as u32) };
}
}
4 => {
let mut __ptr: *mut u32 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u32;
unsafe {
let __t4: u8;
let __t5: u32;
core::arch::asm!("lock cmpxchgl {2:e}, ({1:r})\nsetz {0}", lateout(reg_byte) __t4, in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t5, options(nostack, att_syntax));
success = (__t4 as u8) != 0;
__old = unsafe { core::mem::transmute::<u32, i32>(__t5 as u32) };
}
}
8 => {
let mut __ptr: *mut u64 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u64;
unsafe {
let __t6: u8;
let __t7: u32;
core::arch::asm!("lock cmpxchgq {2:e}, ({1:r})\nsetz {0}", lateout(reg_byte) __t6, in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t7, options(nostack, att_syntax));
success = (__t6 as u8) != 0;
__old = unsafe { core::mem::transmute::<u32, i32>(__t7 as u32) };
}
}
_ => {
unsafe { __cmpxchg_wrong_size() };
}
}
if unlikely!(!success) != 0 {
unsafe {
*old = __old;
}
}
__v0 = success as i64;
__v0 != 0
}

#[inline(always)]
fn arch_atomic64_read(v: *const __SlateRecord_types_h_197) -> i64 {
__READ_ONCE__2!((*v).counter)
}

#[inline(always)]
fn arch_atomic64_set(v: *mut __SlateRecord_types_h_197, i: i64) {
__WRITE_ONCE__6!((*v).counter, i);
}

#[inline(always)]
fn arch_atomic64_add(mut i: i64, mut v: *mut __SlateRecord_types_h_197) {
unsafe {
core::arch::asm!("lock addq {1:r}, ({0:r})\n# {2}", in(reg) core::ptr::addr_of_mut!((*v).counter), in(reg) i as u64, in(reg) core::ptr::addr_of!((*v).counter), options(nostack, att_syntax));
}
}

#[inline(always)]
fn arch_atomic64_sub(mut i: i64, mut v: *mut __SlateRecord_types_h_197) {
unsafe {
core::arch::asm!("lock subq {1:r}, ({0:r})\n# {2}", in(reg) core::ptr::addr_of_mut!((*v).counter), in(reg) i as u64, in(reg) core::ptr::addr_of!((*v).counter), options(nostack, att_syntax));
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
fn arch_atomic64_add_return(i: i64, mut v: *mut __SlateRecord_types_h_197) -> i64 {
let mut __v0: i64;
let mut __ret: i64 = i;
match (const { core::mem::size_of::<i64>() }) as u64 {
1 => {
unsafe {
let __t0: u64;
core::arch::asm!("lock xaddb {0:l}, ({1:r})", inlateout(reg) __ret as u64 => __t0, in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, i64>(__t0 as u64) };
}
}
2 => {
unsafe {
let __t1: u64;
core::arch::asm!("lock xaddw {0:x}, ({1:r})", inlateout(reg) __ret as u64 => __t1, in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, i64>(__t1 as u64) };
}
}
4 => {
unsafe {
let __t2: u64;
core::arch::asm!("lock xaddl {0:r}, ({1:r})", inlateout(reg) __ret as u64 => __t2, in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, i64>(__t2 as u64) };
}
}
8 => {
unsafe {
let __t3: u64;
core::arch::asm!("lock xaddq {0:r}, ({1:r})", inlateout(reg) __ret as u64 => __t3, in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, i64>(__t3 as u64) };
}
}
_ => {
unsafe { __xadd_wrong_size() };
}
}
__v0 = __ret;
i.wrapping_add(__v0)
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
fn arch_atomic64_try_cmpxchg(v: *mut __SlateRecord_types_h_197, _old: *mut i64, mut __new: i64) -> bool {
let mut __v0: i64;
let mut success: bool = false;
let mut __old: i64 = unsafe { *_old };
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
if unlikely!(!success) != 0 {
unsafe {
*_old = __old;
}
}
__v0 = success as i64;
__v0 != 0
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
fn arch_atomic64_xchg(mut v: *mut __SlateRecord_types_h_197, mut __ret: i64) -> i64 {
let mut __v0: i64;
match (const { core::mem::size_of::<i64>() }) as u64 {
1 => {
unsafe {
let __t0: u64;
core::arch::asm!("xchgb {0:l}, ({1:r})", inlateout(reg) __ret as u64 => __t0, in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, i64>(__t0 as u64) };
}
}
2 => {
unsafe {
let __t1: u64;
core::arch::asm!("xchgw {0:x}, ({1:r})", inlateout(reg) __ret as u64 => __t1, in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, i64>(__t1 as u64) };
}
}
4 => {
unsafe {
let __t2: u64;
core::arch::asm!("xchgl {0:r}, ({1:r})", inlateout(reg) __ret as u64 => __t2, in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, i64>(__t2 as u64) };
}
}
8 => {
unsafe {
let __t3: u64;
core::arch::asm!("xchgq {0:r}, ({1:r})", inlateout(reg) __ret as u64 => __t3, in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, i64>(__t3 as u64) };
}
}
_ => {
unsafe { __xchg_wrong_size() };
}
}
__v0 = __ret;
__v0
}

#[inline(always)]
fn arch_atomic64_fetch_and(i: i64, v: *mut __SlateRecord_types_h_197) -> i64 {
let mut val: i64 = arch_atomic64_read(v as *const __SlateRecord_types_h_197);
while !arch_atomic64_try_cmpxchg(v, core::ptr::addr_of_mut!(val), val & i) {
}
val
}

#[inline(always)]
fn arch_atomic64_or(mut i: i64, mut v: *mut __SlateRecord_types_h_197) {
unsafe {
core::arch::asm!("lock orq {1:r}, ({0:r})", in(reg) core::ptr::addr_of_mut!((*v).counter), in(reg) i as u64, options(nostack, att_syntax));
}
}

#[inline(always)]
fn raw_atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
arch_atomic_read(v)
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
fn raw_atomic_try_cmpxchg(v: *mut __SlateRecord_types_h_190, old: &mut i32, new: i32) -> bool {
arch_atomic_try_cmpxchg(v, old, new)
}

#[inline(always)]
fn raw_atomic_dec_and_test(v: *mut __SlateRecord_types_h_190) -> bool {
arch_atomic_dec_and_test(v)
}

#[inline(always)]
fn raw_atomic_fetch_add_unless(v: *mut __SlateRecord_types_h_190, a: i32, u: i32) -> i32 {
let mut c: i32 = raw_atomic_read(v as *const __SlateRecord_types_h_190);
while unlikely!(c == u) == 0 {
if !(!raw_atomic_try_cmpxchg(v, unsafe { &mut (*core::ptr::addr_of_mut!(c)) }, c.wrapping_add(a))) {
break;
}
}
c
}

#[inline(always)]
fn raw_atomic_add_unless(v: *mut __SlateRecord_types_h_190, a: i32, u: i32) -> bool {
raw_atomic_fetch_add_unless(v, a, u) != u
}

#[inline(always)]
fn raw_atomic_inc_not_zero(v: *mut __SlateRecord_types_h_190) -> bool {
raw_atomic_add_unless(v, 1i32, 0i32)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct net_device {
}


#[inline(always)]
fn raw_atomic64_read(v: *const __SlateRecord_types_h_197) -> i64 {
arch_atomic64_read(v)
}

#[inline(always)]
fn raw_atomic64_set(v: *mut __SlateRecord_types_h_197, i: i64) {
arch_atomic64_set(v, i);
}

#[inline(always)]
fn raw_atomic64_add(i: i64, v: *mut __SlateRecord_types_h_197) {
arch_atomic64_add(i, v);
}

#[inline(always)]
fn raw_atomic64_add_return_relaxed(i: i64, v: *mut __SlateRecord_types_h_197) -> i64 {
arch_atomic64_add_return(i, v)
}

#[inline(always)]
fn raw_atomic64_sub(i: i64, v: *mut __SlateRecord_types_h_197) {
arch_atomic64_sub(i, v);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct workqueue_struct {
}


#[inline(always)]
fn raw_atomic64_sub_return_relaxed(i: i64, v: *mut __SlateRecord_types_h_197) -> i64 {
arch_atomic64_sub_return!(i, v)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
}


#[inline(always)]
fn raw_atomic64_fetch_and_relaxed(i: i64, v: *mut __SlateRecord_types_h_197) -> i64 {
arch_atomic64_fetch_and(i, v)
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
fn raw_atomic64_or(i: i64, v: *mut __SlateRecord_types_h_197) {
arch_atomic64_or(i, v);
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
fn raw_atomic64_xchg_relaxed(v: *mut __SlateRecord_types_h_197, new: i64) -> i64 {
arch_atomic64_xchg(v, new)
}

#[inline(always)]
fn raw_atomic64_try_cmpxchg(v: *mut __SlateRecord_types_h_197, old: *mut i64, new: i64) -> bool {
arch_atomic64_try_cmpxchg(v, old, new)
}

#[inline(always)]
fn raw_atomic64_sub_and_test(i: i64, v: *mut __SlateRecord_types_h_197) -> bool {
arch_atomic64_sub_and_test(i, v)
}

#[inline(always)]
fn raw_atomic64_fetch_add_unless(v: *mut __SlateRecord_types_h_197, a: i64, u: i64) -> i64 {
let mut c: i64 = raw_atomic64_read(v as *const __SlateRecord_types_h_197);
while unlikely!(c == u) == 0 {
if !(!raw_atomic64_try_cmpxchg(v, core::ptr::addr_of_mut!(c), c.wrapping_add(a))) {
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

#[inline(always)]
fn raw_atomic_long_read(v: *const __SlateRecord_types_h_197) -> i64 {
raw_atomic64_read(v)
}

#[inline(always)]
fn raw_atomic_long_set(v: *mut __SlateRecord_types_h_197, i: i64) {
raw_atomic64_set(v, i);
}

#[inline(always)]
fn raw_atomic_long_add(i: i64, v: *mut __SlateRecord_types_h_197) {
raw_atomic64_add(i, v);
}

#[inline(always)]
fn raw_atomic_long_add_return_relaxed(i: i64, v: *mut __SlateRecord_types_h_197) -> i64 {
raw_atomic64_add_return_relaxed(i, v)
}

#[inline(always)]
fn raw_atomic_long_sub(i: i64, v: *mut __SlateRecord_types_h_197) {
raw_atomic64_sub(i, v);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


#[inline(always)]
fn raw_atomic_long_sub_return_relaxed(i: i64, v: *mut __SlateRecord_types_h_197) -> i64 {
raw_atomic64_sub_return_relaxed(i, v)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct audit_context {
}


#[inline(always)]
fn raw_atomic_long_fetch_and_relaxed(i: i64, v: *mut __SlateRecord_types_h_197) -> i64 {
raw_atomic64_fetch_and_relaxed(i, v)
}

#[inline(always)]
fn raw_atomic_long_or(i: i64, v: *mut __SlateRecord_types_h_197) {
raw_atomic64_or(i, v);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_local_storage {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_net_context {
}


#[inline(always)]
fn raw_atomic_long_xchg_relaxed(v: *mut __SlateRecord_types_h_197, new: i64) -> i64 {
raw_atomic64_xchg_relaxed(v, new)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct cfs_rq {
}


#[inline(always)]
fn raw_atomic_long_try_cmpxchg(v: *mut __SlateRecord_types_h_197, old: *mut i64, new: i64) -> bool {
raw_atomic64_try_cmpxchg(v, old as *mut i64, new)
}

#[inline(always)]
fn raw_atomic_long_sub_and_test(i: i64, v: *mut __SlateRecord_types_h_197) -> bool {
raw_atomic64_sub_and_test(i, v)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_task {
}


#[inline(always)]
fn raw_atomic_long_inc_not_zero(v: *mut __SlateRecord_types_h_197) -> bool {
raw_atomic64_inc_not_zero(v)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct nameidata {
}


#[inline(always)]
fn atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
instrument_atomic_read(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_read(v)
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
struct pipe_inode_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rcu_node {
}


#[inline(always)]
fn atomic_dec_and_test(v: *mut __SlateRecord_types_h_190) -> bool {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_dec_and_test(v)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rq {
}


#[inline(always)]
fn atomic_inc_not_zero(v: *mut __SlateRecord_types_h_190) -> bool {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_inc_not_zero(v)
}

#[inline(always)]
fn atomic_long_read(v: *const __SlateRecord_types_h_197) -> i64 {
instrument_atomic_read(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_read(v)
}

#[inline(always)]
fn atomic_long_set(v: *mut __SlateRecord_types_h_197, i: i64) {
instrument_atomic_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_set(v, i);
}

#[inline(always)]
fn atomic_long_add(i: i64, v: *mut __SlateRecord_types_h_197) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_add(i, v);
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
fn atomic_long_add_return_relaxed(i: i64, v: *mut __SlateRecord_types_h_197) -> i64 {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_add_return_relaxed(i, v)
}

#[inline(always)]
fn atomic_long_sub(i: i64, v: *mut __SlateRecord_types_h_197) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_sub(i, v);
}

#[inline(always)]
fn atomic_long_sub_return_relaxed(i: i64, v: *mut __SlateRecord_types_h_197) -> i64 {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_sub_return_relaxed(i, v)
}

#[inline(always)]
fn atomic_long_fetch_and_relaxed(i: i64, v: *mut __SlateRecord_types_h_197) -> i64 {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_fetch_and_relaxed(i, v)
}

#[inline(always)]
fn atomic_long_or(i: i64, v: *mut __SlateRecord_types_h_197) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_or(i, v);
}

#[inline(always)]
fn atomic_long_xchg_relaxed(v: *mut __SlateRecord_types_h_197, new: i64) -> i64 {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_xchg_relaxed(v, new)
}

#[inline(always)]
fn atomic_long_try_cmpxchg(v: *mut __SlateRecord_types_h_197, old: *mut i64, new: i64) -> bool {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
instrument_read_write(old as *const (), core::mem::size_of::<i64>() as u64);
raw_atomic_long_try_cmpxchg(v, old, new)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rq_flags {
}


#[inline(always)]
fn atomic_long_sub_and_test(i: i64, v: *mut __SlateRecord_types_h_197) -> bool {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_sub_and_test(i, v)
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
fn atomic_long_inc_not_zero(v: *mut __SlateRecord_types_h_197) -> bool {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_inc_not_zero(v)
}

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
struct cpumask {
    bits: [u64; 1],
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

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__might_resched_13: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__might_resched) }) as *mut ();

fn __might_resched(file: *const u8, line: i32, offsets: u32) {
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


fn INIT_LIST_HEAD(list: *mut list_head) {
loop {
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
(*entry).next = ((256i32 as *const u8).wrapping_offset(_AC__3!() as isize) as *mut ()) as *mut list_head;
}
unsafe {
(*entry).prev = ((290i32 as *const u8).wrapping_offset(_AC__3!() as isize) as *mut ()) as *mut list_head;
}
}

fn list_del_init(entry: *mut list_head) {
__list_del_entry(entry);
INIT_LIST_HEAD(entry);
}

fn list_move_tail(list: *mut list_head, head: *mut list_head) {
__list_del_entry(list);
list_add_tail(list, head);
}

fn list_is_head(list: *const list_head, head: *const list_head) -> i32 {
(list == head) as i32
}

fn list_empty(head: *const list_head) -> i32 {
let mut __v0: *mut list_head;
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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

// list_for_each	-	iterate over a list
// @pos:	the &struct list_head to use as a loop cursor.
// @head:	the head for your list.
// list_for_each_safe - iterate over a list safe against removal of list entry
// @pos:	the &struct list_head to use as a loop cursor.
// @n:		another &struct list_head to use as temporary storage
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
struct mem_cgroup {
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
    __slate_bits_0: __slate_bits::__SlateBits106U0,
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
    __slate_bits_0: __slate_bits::__SlateBits145U0,
    pkru: u32,
}


#[inline(always)]
fn __static_cpu_has__const4(bit: u16) -> bool {
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
struct codetag {
    flags: u32,
    lineno: u32,
    modname: *const u8,
    function: *const u8,
    filename: *const u8,
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

#[repr(C)]
#[derive(Clone, Copy)]
struct maple_pnode {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_103 {
    slot: [*mut (); 16],
    __slate_anon_1: __SlateRecord_maple_tree_h_105,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_maple_tree_h_105 {
    pad: [*mut (); 15],
    meta: maple_metadata,
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
struct maple_enode {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct llist_head {
    first: *mut llist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_maple_tree_h_162 {
    node: *mut maple_node,
    max: u64,
    mt: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct llist_node {
    next: *mut llist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_maple_tree_h_167 {
    node: *mut maple_node,
    max: u64,
    start: u8,
    end: u8,
    mt: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_178 {
    pivot: [u64; 3],
    __slate_anon_1: __SlateRecord_maple_tree_h_180,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_maple_tree_h_180 {
    _pad: [*mut (); 2],
    max: u64,
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
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_324 {
    __slate_anon_0: __SlateRecord_maple_tree_h_325,
    __slate_anon_1: __SlateRecord_maple_tree_h_329,
    mr64: maple_range_64,
    ma64: maple_arange_64,
    cp: maple_copy,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_maple_tree_h_325 {
    parent: *mut maple_pnode,
    slot: [*mut (); 31],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_maple_tree_h_329 {
    pad: *mut (),
    rcu: callback_head,
    piv_parent: *mut maple_enode,
    parent_slot: u8,
    r#type: u32,
    slot_len: u8,
    ma_flags: u32,
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
struct slab_sheaf {
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
#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
}


fn test_ti_thread_flag(ti: *mut thread_info, flag: i32) -> i32 {
test_bit__2!(flag, unsafe { core::ptr::addr_of_mut!((*ti).flags) })
}

#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe {
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
struct uprobe_xol_ops {
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
struct sysv_sem {
    undo_list: *mut sem_undo_list,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sysv_shm {
    shm_clist: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct plist_head {
    node_list: list_head,
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
struct ldt_struct {
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
struct timerqueue_linked_node {
    node: rb_node_linked,
    expires: i64,
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
struct refcount_struct {
    refs: __SlateRecord_types_h_190,
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
union sigval {
    sival_int: i32,
    sival_ptr: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_1062 {
    rb: rb_node,
    rb_subtree_last: u64,
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
    __slate_bits_0: __slate_bits::__SlateBits254U0,
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
#[inline(always)]
fn queued_spin_is_locked(lock: *mut qspinlock) -> i32 {
// Any !0 state indicates it is locked, even if _Q_LOCKED_VAL
// isn't immediately observable.
atomic_read((unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.val) }) as *const __SlateRecord_types_h_190)
}

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

#[inline(always)]
fn spin_trylock(lock: *mut spinlock) -> i32 {
raw_spin_trylock!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) })
}

#[inline(always)]
fn spin_lock_irq(lock: *mut spinlock) {
raw_spin_lock_irq!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[inline(always)]
fn spin_unlock(lock: *mut spinlock) {
raw_spin_unlock!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[repr(C)]
#[derive(Clone, Copy)]
struct memory_tier {
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
struct kmem_cache {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct seqcount {
    sequence: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqcount_spinlock {
    seqcount: seqcount,
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
struct seqlock {
    seqcount: seqcount_spinlock,
    lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key_type {
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
struct unwind_task_info {
    unwind_mask: __SlateRecord_types_h_197,
    cache: *mut unwind_cache,
    work: callback_head,
    id: unwind_task_id,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_cred_h_147 {
    non_rcu: i32,
    rcu: callback_head,
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
struct pidfs_attr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct load_weight {
    weight: u64,
    inv_weight: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pid_h_62 {
    ino: u64,
    pidfs_hash: rhash_head,
    stashed: *mut dentry,
    attr: *mut pidfs_attr,
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
    __slate_bits_0: __slate_bits::__SlateBits319U0,
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
struct tty_struct {
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
struct tty_audit_buf {
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
    __slate_bits_0: __slate_bits::__SlateBits70U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits70U1,
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
    __slate_bits_2: __slate_bits::__SlateBits70U2,
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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
struct old_timespec32 {
    tv_sec: i32,
    tv_nsec: i32,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct hlist_nulls_node {
    next: *mut hlist_nulls_node,
    pprev: *mut *mut hlist_nulls_node,
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


// Defined as a macro as it is a very low level header included from
// areas that don't even know about current.  This gives the rcu_read_lock()
// nesting depth, but makes sense only if CONFIG_PREEMPT_RCU -- in other
// types of kernel builds, the rcu_read_lock() nesting depth is unknowable.
fn rcu_read_lock_held() -> i32 {
1
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
struct fserror_event {
}


fn rcu_read_unlock() {
RCU_LOCKDEP_WARN!(!unsafe { rcu_is_watching() });
rcu_lock_release!(); // Keep acq info for rls diags.
__release_shared!();
unsafe { __rcu_read_unlock() };
}

fn rb_first(root: *const rb_root) -> *mut rb_node {
let mut n: *mut rb_node = unsafe { core::mem::zeroed() };
n = unsafe { (*root).rb_node };
if n.is_null() {
return core::ptr::null_mut::<rb_node>();
}
while !unsafe { (*n).rb_left }.is_null() {
n = unsafe { (*n).rb_left };
}
n
}

fn rb_link_node(node: *mut rb_node, parent: *mut rb_node, rb_link: *mut *mut rb_node) {
unsafe {
(*node).__rb_parent_color = parent as u64;
}
unsafe {
(*node).rb_right = core::ptr::null_mut::<rb_node>();
}
unsafe {
(*node).rb_left = core::ptr::null_mut::<rb_node>();
}
unsafe {
*rb_link = node;
}
}

// Same as rb_first(), but O(1)
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


// 256 bytes including ->parent
// 256 bytes
// 240 bytes
#[repr(C)]
#[derive(Clone, Copy)]
struct maple_metadata {
    end: u8,
    gap: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct maple_range_64 {
    parent: *mut maple_pnode,
    pivot: [u64; 15],
    __slate_anon_2: __SlateRecord_maple_tree_h_103,
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
struct maple_arange_64 {
    parent: *mut maple_pnode,
    pivot: [u64; 9],
    slot: [*mut (); 10],
    gap: [u64; 10],
    meta: maple_metadata,
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
struct maple_copy {
    dst: [__SlateRecord_maple_tree_h_162; 3],
    src: [__SlateRecord_maple_tree_h_167; 4],
    slot: [*mut (); 3],
    gap: [u64; 3],
    min: u64,
    __slate_anon_5: __SlateRecord_maple_tree_h_178,
    end: u8,
    s_count: u8,
    d_count: u8,
    split: u8,
    data: u8,
    height: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct maple_tree {
    __slate_anon_0: __SlateRecord_maple_tree_h_261,
    ma_flags: u32,
    ma_root: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct maple_node {
    __slate_anon_0: __SlateRecord_maple_tree_h_324,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ma_state {
    tree: *mut maple_tree,
    index: u64,
    last: u64,
    node: *mut maple_enode,
    min: u64,
    max: u64,
    sheaf: *mut slab_sheaf,
    alloc: *mut maple_node,
    node_request: u64,
    status: u32,
    depth: u8,
    offset: u8,
    mas_flags: u8,
    end: u8,
    store_type: u32,
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

fn queue_work(wq: *mut workqueue_struct, work: *mut work_struct) -> bool {
unsafe { queue_work_on(64i32, wq, work) }
}

fn schedule_work(work: *mut work_struct) -> bool {
queue_work(unsafe { system_percpu_wq }, work)
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


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
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
struct fsnotify_mark_connector {
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
struct file_kattr {
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
struct posix_acl {
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


fn percpu_counter_add(fbc: *mut percpu_counter, amount: i64) {
unsafe { percpu_counter_add_batch(fbc, amount, percpu_counter_batch) };
}

fn percpu_counter_inc(fbc: *mut percpu_counter) {
percpu_counter_add(fbc, 1i64);
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


fn percpu_counter_dec(fbc: *mut percpu_counter) {
percpu_counter_add(fbc, 1i32.wrapping_neg() as i64);
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
struct io_uring_cmd {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_41 {
    f: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct audit_names {
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
    __slate_bits_0: __slate_bits::__SlateBits475U0,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_open_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_iattrs {
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
struct kernfs_root {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct vma_iterator {
    mas: ma_state,
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

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
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
struct sock {
}


fn const_folio_flags(folio: *const folio, n: u32) -> *const u64 {
let page: *const page = unsafe { core::ptr::addr_of!((*folio).__slate_anon_0.page) };
VM_BUG_ON_PGFLAGS!();
VM_BUG_ON_PGFLAGS!();
unsafe { core::ptr::addr_of!((*page.wrapping_offset(n as isize)).flags.f) }
}

fn folio_flags(folio: *mut folio, n: u32) -> *mut u64 {
let page: *mut page = unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.page) };
VM_BUG_ON_PGFLAGS!();
VM_BUG_ON_PGFLAGS!();
unsafe { core::ptr::addr_of_mut!((*page.wrapping_offset(n as isize)).flags.f) }
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
#[inline(always)]
fn folio_test_locked(folio: *const folio) -> bool {
test_bit!(0i32, const_folio_flags(folio, 0u32)) != 0
}

#[inline(always)]
fn folio_set_dirty(folio: *mut folio) {
set_bit(4i64, folio_flags(folio, 0u32) as *mut u64);
}

#[inline(always)]
fn folio_test_swapbacked(folio: *const folio) -> bool {
test_bit!(17i32, const_folio_flags(folio, 0u32)) != 0
}

#[inline(always)]
fn folio_test_writeback(folio: *const folio) -> bool {
test_bit!(1i32, const_folio_flags(folio, 0u32)) != 0
}

#[inline(always)]
fn folio_test_swapcache(folio: *const folio) -> bool {
folio_test_swapbacked(folio) && test_bit!(10i32, const_folio_flags(folio, 0u32)) != 0
}

fn PageHWPoison(page: *const page) -> i32 {
0
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

fn folio_test_uptodate(folio: *const folio) -> bool {
let ret: bool = test_bit!(3i32, const_folio_flags(folio, 0u32)) != 0;
// Must ensure that the data we read out of the folio is loaded
// _after_ we've loaded folio->flags to check the uptodate bit.
// We can skip the barrier if the folio is not uptodate, because
// we wouldn't be reading anything from it.
//
// See folio_mark_uptodate() for the other side of the story.
if ret {
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
}
ret
}

#[inline(always)]
fn folio_test_head(folio: *const folio) -> bool {
test_bit!(6i32, const_folio_flags(folio, 0u32)) != 0
}

fn folio_test_large(folio: *const folio) -> bool {
folio_test_head(folio)
}

fn page_type_has_type(page_type: i32) -> bool {
page_type < -16777216i32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct wake_irq {
}


fn page_mapcount_is_type(mapcount: u32) -> bool {
page_type_has_type(mapcount.wrapping_sub(1) as i32)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct local_lock {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_qos {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct local_trylock {
    acquired: u8,
}


fn local_lock_acquire(l: *mut local_lock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct fwnode_handle {
}


fn local_lock_release(l: *mut local_lock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct zswap_lruvec_state {
}


fn zswap_invalidate(swp: __SlateRecord_mm_types_h_288) {
}

fn zswap_swapon(r#type: i32, nr_pages: u64) -> i32 {
0
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


fn zswap_swapoff(r#type: i32) {
}

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


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
}


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
struct module_param_attrs {
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


// zone_idx() returns 0 for the ZONE_DMA zone, 1 for the ZONE_NORMAL zone, etc.
// GFP_ZONE_BAD is a bitmap for all combinations of __GFP_DMA, __GFP_DMA32
// __GFP_HIGHMEM and __GFP_MOVABLE that are not permitted. One flag per
// entry starting with bit 0. Bit is set if the combination is not
// allowed.
fn gfp_has_flags(gfp: u32, flags: u32) -> bool {
gfp & flags == flags
}

fn gfp_has_io_fs(gfp: u32) -> bool {
gfp_has_flags(gfp, (BIT!(6i32) as u32) | (BIT!(7i32) as u32))
}

fn mmget(mm: *mut mm_struct) {
atomic_inc(unsafe { core::ptr::addr_of_mut!((*mm).__slate_anon_0.mm_users) });
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


fn mmget_not_zero(mm: *mut mm_struct) -> bool {
atomic_inc_not_zero(unsafe { core::ptr::addr_of_mut!((*mm).__slate_anon_0.mm_users) })
}

fn xa_mk_value(v: u64) -> *mut () {
let mut __v0: i64;
let __ret_warn_on: i32 = ((v as i64) < 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("207: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 207b - ., 3; .popsection\n# {0}", const 207 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_xa_mk_value_1, sym __slate_asm_string_xa_mk_value_2, const 60 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("208: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 208b - ., 4; .popsection\n# {0}", const 208 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
(v << 1i32 | 1) as *mut ()
}

fn xa_is_value(entry: *const ()) -> bool {
(entry as u64) & 1 != 0
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
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct list_lru_one {
    list: list_head,
    nr_items: i64,
    lock: spinlock,
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


fn hlist_bl_unhashed(h: *const hlist_bl_node) -> bool {
unsafe { (*h).pprev }.is_null()
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
struct __SlateRecord_uuid_h_21 {
    b: [u8; 16],
}


// The length of a UUID string ("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
// not including trailing NUL.
fn __mmap_lock_trace_start_locking(mm: *mut mm_struct, write: bool) {
if static_branch_unlikely__3!(unsafe { core::ptr::addr_of_mut!(__tracepoint_mmap_lock_start_locking.key) }) != 0 {
unsafe { __mmap_lock_do_trace_start_locking(mm, write) };
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct disk_stats {
}


fn __mmap_lock_trace_acquire_returned(mm: *mut mm_struct, write: bool, success: bool) {
if static_branch_unlikely__2!(unsafe { core::ptr::addr_of_mut!(__tracepoint_mmap_lock_acquire_returned.key) }) != 0 {
unsafe { __mmap_lock_do_trace_acquire_returned(mm, write, success) };
}
}

fn __mmap_lock_trace_released(mm: *mut mm_struct, write: bool) {
if static_branch_unlikely!(unsafe { core::ptr::addr_of_mut!(__tracepoint_mmap_lock_released.key) }) != 0 {
unsafe { __mmap_lock_do_trace_released(mm, write) };
}
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


// VMA locks do not behave like most ordinary locks found in the kernel, so we
// cannot quite have full lockdep tracking in the way we would ideally prefer.
//
// Read locks act as shared locks which exclude an exclusive lock being
// taken. We therefore mark these accordingly on read lock acquire/release.
//
// Write locks are acquired exclusively per-VMA, but released in a shared
// fashion, that is upon vma_end_write_all(), we update the mmap's seqcount such
// that write lock is released.
//
// We therefore cannot track write locks per-VMA, nor do we try. Mitigating this
// is the fact that, of course, we do lockdep-track the mmap lock rwsem which
// must be held when taking a VMA write lock.
//
// We do, however, want to indicate that during either acquisition of a VMA
// write lock or detachment of a VMA that we require the lock held be exclusive,
// so we utilise lockdep to do so.
fn mmap_read_lock(mm: *mut mm_struct) {
__mmap_lock_trace_start_locking(mm, false);
unsafe { down_read(core::ptr::addr_of_mut!((*mm).__slate_anon_0.mmap_lock)) };
__mmap_lock_trace_acquire_returned(mm, false, true);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct request {
}


fn mmap_read_unlock(mm: *mut mm_struct) {
__mmap_lock_trace_released(mm, false);
unsafe { up_read(core::ptr::addr_of_mut!((*mm).__slate_anon_0.mmap_lock)) };
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
struct bio_alloc_cache {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct elevator_queue {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_trace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_flush_queue {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pr_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rq_qos {
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


fn cc_get_mask() -> u64 {
0
}

#[inline(always)]
fn pte_mkuffd(pte: __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_pgtable_64_types_h_21 {
return pte;
}

#[repr(C)]
#[derive(Clone, Copy)]
struct timer_rand_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cdrom_device_info {
}


#[inline(always)]
fn pte_swp_uffd(pte: __SlateRecord_pgtable_64_types_h_21) -> i32 {
0
}

#[inline(always)]
fn pte_swp_clear_uffd(pte: __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_pgtable_64_types_h_21 {
return pte;
}

#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_ctx {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_hw_ctx {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_tags {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_tag_set {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mempolicy_h_54 {
    cpuset_mems_allowed: __SlateRecord_nodemask_types_h_17,
    user_nodemask: __SlateRecord_nodemask_types_h_17,
}


fn page_table_check_ptes_set(mm: *mut mm_struct, addr: u64, ptep: *mut __SlateRecord_pgtable_64_types_h_21, pte: __SlateRecord_pgtable_64_types_h_21, nr: u32) {
}

fn pte_set_flags(pte: __SlateRecord_pgtable_64_types_h_21, set: u64) -> __SlateRecord_pgtable_64_types_h_21 {
let v: u64 = native_pte_val(pte);
return native_make_pte(v | set);
}

fn pte_clear_flags(pte: __SlateRecord_pgtable_64_types_h_21, clear: u64) -> __SlateRecord_pgtable_64_types_h_21 {
let v: u64 = native_pte_val(pte);
return native_make_pte(v & !clear);
}

fn mksaveddirty_shift(mut v: u64) -> u64 {
let cond: u64 = !v >> 1i32 & 1;
v |= (v >> 6i32 & cond) << 58i32;
let __v3: u64 = v & !(cond << 6i32);
v = __v3;
v
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


fn clear_saveddirty_shift(mut v: u64) -> u64 {
let cond: u64 = v >> 1i32 & 1;
v |= (v >> 58i32 & cond) << 6i32;
let __v3: u64 = v & !(cond << 58i32);
v = __v3;
v
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
struct proc_dir_entry {
}


fn pte_mksaveddirty(pte: __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_pgtable_64_types_h_21 {
let mut v: u64 = native_pte_val(pte);
v = mksaveddirty_shift(v);
return native_make_pte(v);
}

fn pte_clear_saveddirty(pte: __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_pgtable_64_types_h_21 {
let mut v: u64 = native_pte_val(pte);
v = clear_saveddirty_shift(v);
return native_make_pte(v);
}

fn pte_mkold(pte: __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_pgtable_64_types_h_21 {
return pte_clear_flags(pte, _AT!(u64, 1i32) << 5i32);
}

fn pte_mksoft_dirty(pte: __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_pgtable_64_types_h_21 {
return pte_set_flags(pte, _AT!(u64, 0i32));
}

fn massage_pgprot(pgprot: pgprot) -> u64 {
let mut protval: u64 = pgprot_val!(pgprot);
if protval & _AT!(u64, 1i32) << 0i32 != 0 {
protval &= unsafe { __supported_pte_mask };
}
protval
}

fn check_pgprot(pgprot: pgprot) -> u64 {
let massaged_val: u64 = massage_pgprot(pgprot);
// mmdebug.h can not be included here because of dependencies
massaged_val
}

fn pfn_pte(page_nr: u64, pgprot: pgprot) -> __SlateRecord_pgtable_64_types_h_21 {
let mut pfn: u64 = page_nr << 12i32;
// This bit combination is used to mark shadow stacks
let mut __v0: i64;
let __ret_warn_on: i32 = (pgprot_val!(pgprot) & (_AT!(u64, 1i32) << 6i32 | _AT!(u64, 1i32) << 1i32) == _AT!(u64, 1i32) << 6i32) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("286: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 286b - ., 3; .popsection\n# {0}", const 286 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_pfn_pte_1, sym __slate_asm_string_pfn_pte_2, const 735 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("287: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 287b - ., 4; .popsection\n# {0}", const 287 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
let __v5: u64 = protnone_mask(pgprot_val!(pgprot));
pfn ^= __v5;
let __v8: u64 = pfn;
pfn = __v8 & (((!(_AC!() << 12i32).wrapping_sub(1) as i64) as u64) & 4503599627370495u64);
return __pte!(pfn | check_pgprot(pgprot));
}

fn pte_modify(pte: __SlateRecord_pgtable_64_types_h_21, newprot: pgprot) -> __SlateRecord_pgtable_64_types_h_21 {
let mut val: u64 = pte_val!(pte);
let oldval: u64 = val;
let mut pte_result: __SlateRecord_pgtable_64_types_h_21 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(pte_result) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
// Chop off the NX bit (if present), and add the NX portion of
// the newprot (if present):
let __v0: u64 = ((!(_AC!() << 12i32).wrapping_sub(1) as i64) as u64) & 4503599627370495u64 | _AT!(u64, 1i32) << 4i32 | _AT!(u64, 1i32) << 3i32 | _AT!(u64, 1i32) << 9i32 | _AT!(u64, 1i32) << 5i32 | (_AT!(u64, 1i32) << 6i32 | _AT!(u64, 1i32) << 58i32) | _AT!(u64, 0i32) | _AT!(u64, cc_get_mask()) | _AT!(u64, 0i32) | _AT!(u64, 1i32) << 7i32;
val &= __v0;
let __v3: u64 = check_pgprot(newprot) & !(((!(_AC!() << 12i32).wrapping_sub(1) as i64) as u64) & 4503599627370495u64 | _AT!(u64, 1i32) << 4i32 | _AT!(u64, 1i32) << 3i32 | _AT!(u64, 1i32) << 9i32 | _AT!(u64, 1i32) << 5i32 | (_AT!(u64, 1i32) << 6i32 | _AT!(u64, 1i32) << 58i32) | _AT!(u64, 0i32) | _AT!(u64, cc_get_mask()) | _AT!(u64, 0i32) | _AT!(u64, 1i32) << 7i32);
val |= __v3;
val = flip_protnone_guard(oldval, val, ((!(_AC!() << 12i32).wrapping_sub(1) as i64) as u64) & 4503599627370495u64);
pte_result = __pte!(val);
// To avoid creating Write=0,Dirty=1 PTEs, pte_modify() needs to avoid:
//  1. Marking Write=0 PTEs Dirty=1
//  2. Marking Dirty=1 PTEs Write=0
//
// The first case cannot happen because the _PAGE_CHG_MASK will filter
// out any Dirty bit passed in newprot. Handle the second case by
// going through the mksaveddirty exercise. Only do this if the old
// value was Write=1 to avoid doing this on Shadow Stack PTEs.
if oldval & _AT!(u64, 1i32) << 1i32 != 0 {
pte_result = pte_mksaveddirty(pte_result);
} else {
pte_result = pte_clear_saveddirty(pte_result);
}
return pte_result;
}

#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_taskset {
}


fn native_set_pte(ptep: *mut __SlateRecord_pgtable_64_types_h_21, pte: __SlateRecord_pgtable_64_types_h_21) {
loop {
loop {
if !(__native_word__7!() || ((const { core::mem::size_of::<__SlateRecord_pgtable_64_types_h_21>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_290() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__5!(*ptep, pte);
if true {
break;
}
}
}

// Shift the offset up "too far" by TYPE bits, then down again
// The offset is inverted by a binary not operation to make the high
// physical bits set.
fn __pte_needs_invert(val: u64) -> bool {
val != 0 && val & _AT!(u64, 1i32) << 0i32 == 0
}

fn protnone_mask(val: u64) -> u64 {
if __pte_needs_invert(val) { !0 } else { 0 }
}

fn flip_protnone_guard(oldval: u64, mut val: u64, mask: u64) -> u64 {
// When a PTE transitions from NONE to !NONE or vice-versa
// invert the PFN part to stop speculation.
// pte_pfn undoes this when needed.
if (__pte_needs_invert(oldval) as i32) != (__pte_needs_invert(val) as i32) {
val = val & !mask | !val & mask;
}
val
}

// mprotect needs to preserve PAT and encryption bits when updating
// vm_page_prot
fn pte_none(pte: __SlateRecord_pgtable_64_types_h_21) -> i32 {
(pte.pte & !(_AT!(u64, 1i32) << 6i32 | _AT!(u64, 1i32) << 5i32) == 0) as i32
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


fn pte_same(a: __SlateRecord_pgtable_64_types_h_21, b: __SlateRecord_pgtable_64_types_h_21) -> i32 {
(a.pte == b.pte) as i32
}

fn pte_advance_pfn(pte: __SlateRecord_pgtable_64_types_h_21, nr: u64) -> __SlateRecord_pgtable_64_types_h_21 {
if __pte_needs_invert(pte_val!(pte)) {
return __pte!(pte_val!(pte).wrapping_sub(nr << 12i32));
}
return __pte!(pte_val!(pte).wrapping_add(nr << 12i32));
}

fn pte_present(a: __SlateRecord_pgtable_64_types_h_21) -> i32 {
(pte_flags(a) & (_AT!(u64, 1i32) << 0i32 | _AT!(u64, 1i32) << 8i32)) as i32
}

// Currently stuck as a macro due to indirect forward reference to
// linux/mmzone.h's __section_mem_map_addr() definition:
fn pud_none(pud: __SlateRecord_pgtable_types_h_368) -> i32 {
(native_pud_val(pud) & !(_AT!(u64, 1i32) << 6i32 | _AT!(u64, 1i32) << 5i32) == 0) as i32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_swap_h_126 {
    reserved: [u8; 4086],
    magic: [u8; 10],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_swap_h_130 {
    bootbits: [u8; 1024],
    version: u32,
    last_page: u32,
    nr_badpages: u32,
    sws_uuid: [u8; 16],
    sws_volume: [u8; 16],
    padding: [u32; 117],
    badpages: [u32; 1],
}


fn pud_pgtable(pud: __SlateRecord_pgtable_types_h_368) -> *mut __SlateRecord_pgtable_64_types_h_22 {
__va!(pud_val!(pud) & pud_pfn_mask(pud)) as *mut __SlateRecord_pgtable_64_types_h_22
}

fn pud_bad(pud: __SlateRecord_pgtable_types_h_368) -> i32 {
(pud_flags(pud) & !(_AT!(u64, 1i32) << 0i32 | _AT!(u64, 1i32) << 1i32 | 0 | _AT!(u64, 1i32) << 5i32 | 0 | _AT!(u64, 1i32) << 6i32 | 0 | 0 | _AT!(u64, 0u64) | _AT!(u64, 1i32) << 2i32) != 0) as i32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct swap_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_vmalloc_h_53 {
    next: *mut vm_struct,
    llnode: llist_node,
}


fn p4d_none(p4d: __SlateRecord_pgtable_types_h_342) -> i32 {
(native_p4d_val(p4d) & !(_AT!(u64, 1i32) << 6i32 | _AT!(u64, 1i32) << 5i32) == 0) as i32
}

fn p4d_pgtable(p4d: __SlateRecord_pgtable_types_h_342) -> *mut __SlateRecord_pgtable_types_h_368 {
__va!(p4d_val!(p4d) & p4d_pfn_mask(p4d)) as *mut __SlateRecord_pgtable_types_h_368
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1528 {
    map_type: u32,
    key_size: u32,
    value_size: u32,
    max_entries: u32,
    map_flags: u32,
    inner_map_fd: u32,
    numa_node: u32,
    map_name: [u8; 16],
    map_ifindex: u32,
    btf_fd: u32,
    btf_key_type_id: u32,
    btf_value_type_id: u32,
    btf_vmlinux_value_type_id: u32,
    map_extra: u64,
    value_type_btf_obj_fd: i32,
    map_token_fd: i32,
    excl_prog_hash: u64,
    excl_prog_hash_size: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1581 {
    map_fd: u32,
    key: u64,
    __slate_anon_2: __SlateRecord_bpf_h_1584,
    flags: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1584 {
    value: u64,
    next_key: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1591 {
    in_batch: u64,
    out_batch: u64,
    keys: u64,
    values: u64,
    count: u32,
    map_fd: u32,
    elem_flags: u64,
    flags: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1608 {
    prog_type: u32,
    insn_cnt: u32,
    insns: u64,
    license: u64,
    log_level: u32,
    log_size: u32,
    log_buf: u64,
    kern_version: u32,
    prog_flags: u32,
    prog_name: [u8; 16],
    prog_ifindex: u32,
    expected_attach_type: u32,
    prog_btf_fd: u32,
    func_info_rec_size: u32,
    func_info: u64,
    func_info_cnt: u32,
    line_info_rec_size: u32,
    line_info: u64,
    line_info_cnt: u32,
    attach_btf_id: u32,
    __slate_anon_20: __SlateRecord_bpf_h_1633,
    core_relo_cnt: u32,
    fd_array: u64,
    core_relos: u64,
    core_relo_rec_size: u32,
    log_true_size: u32,
    prog_token_fd: i32,
    fd_array_cnt: u32,
    signature: u64,
    signature_size: u32,
    keyring_id: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1633 {
    attach_prog_fd: u32,
    attach_btf_obj_fd: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1674 {
    pathname: u64,
    bpf_fd: u32,
    file_flags: u32,
    path_fd: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1687 {
    __slate_anon_0: __SlateRecord_bpf_h_1688,
    attach_bpf_fd: u32,
    attach_type: u32,
    attach_flags: u32,
    replace_bpf_fd: u32,
    __slate_anon_5: __SlateRecord_bpf_h_1696,
    expected_revision: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1688 {
    target_fd: u32,
    target_ifindex: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1696 {
    relative_fd: u32,
    relative_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1703 {
    prog_fd: u32,
    retval: u32,
    data_size_in: u32,
    data_size_out: u32,
    data_in: u64,
    data_out: u64,
    repeat: u32,
    duration: u32,
    ctx_size_in: u32,
    ctx_size_out: u32,
    ctx_in: u64,
    ctx_out: u64,
    flags: u32,
    cpu: u32,
    batch_size: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1727 {
    __slate_anon_0: __SlateRecord_bpf_h_1728,
    next_id: u32,
    open_flags: u32,
    fd_by_id_token_fd: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1728 {
    start_id: u32,
    prog_id: u32,
    map_id: u32,
    btf_id: u32,
    link_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1740 {
    bpf_fd: u32,
    info_len: u32,
    info: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1746 {
    __slate_anon_0: __SlateRecord_bpf_h_1747,
    attach_type: u32,
    query_flags: u32,
    attach_flags: u32,
    prog_ids: u64,
    __slate_anon_5: __SlateRecord_bpf_h_1755,
    __slate_bits_0: __slate_bits::__SlateBits974U0,
    prog_attach_flags: u64,
    link_ids: u64,
    link_attach_flags: u64,
    revision: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1747 {
    target_fd: u32,
    target_ifindex: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1755 {
    prog_cnt: u32,
    count: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1769 {
    name: u64,
    prog_fd: u32,
    __slate_bits_0: __slate_bits::__SlateBits977U0,
    cookie: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1776 {
    btf: u64,
    btf_log_buf: u64,
    btf_size: u32,
    btf_log_size: u32,
    btf_log_level: u32,
    btf_log_true_size: u32,
    btf_flags: u32,
    btf_token_fd: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1794 {
    pid: u32,
    fd: u32,
    flags: u32,
    buf_len: u32,
    buf: u64,
    prog_id: u32,
    fd_type: u32,
    probe_offset: u64,
    probe_addr: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1810 {
    __slate_anon_0: __SlateRecord_bpf_h_1811,
    __slate_anon_1: __SlateRecord_bpf_h_1815,
    attach_type: u32,
    flags: u32,
    __slate_anon_4: __SlateRecord_bpf_h_1821,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1811 {
    prog_fd: u32,
    map_fd: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1815 {
    target_fd: u32,
    target_ifindex: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1821 {
    target_btf_id: u32,
    __slate_anon_1: __SlateRecord_bpf_h_1823,
    perf_event: __SlateRecord_bpf_h_1827,
    kprobe_multi: __SlateRecord_bpf_h_1834,
    tracing: __SlateRecord_bpf_h_1841,
    netfilter: __SlateRecord_bpf_h_1850,
    tcx: __SlateRecord_bpf_h_1856,
    uprobe_multi: __SlateRecord_bpf_h_1863,
    netkit: __SlateRecord_bpf_h_1873,
    cgroup: __SlateRecord_bpf_h_1880,
    tracing_multi: __SlateRecord_bpf_h_1887,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1823 {
    iter_info: u64,
    iter_info_len: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1827 {
    bpf_cookie: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1834 {
    flags: u32,
    cnt: u32,
    syms: u64,
    addrs: u64,
    cookies: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1841 {
    target_btf_id: u32,
    cookie: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1850 {
    pf: u32,
    hooknum: u32,
    priority: i32,
    flags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1856 {
    __slate_anon_0: __SlateRecord_bpf_h_1857,
    expected_revision: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1857 {
    relative_fd: u32,
    relative_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1863 {
    path: u64,
    offsets: u64,
    ref_ctr_offsets: u64,
    cookies: u64,
    cnt: u32,
    flags: u32,
    pid: u32,
    path_fd: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1873 {
    __slate_anon_0: __SlateRecord_bpf_h_1874,
    expected_revision: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1874 {
    relative_fd: u32,
    relative_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1880 {
    __slate_anon_0: __SlateRecord_bpf_h_1881,
    expected_revision: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1881 {
    relative_fd: u32,
    relative_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1887 {
    ids: u64,
    cookies: u64,
    cnt: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1895 {
    link_fd: u32,
    __slate_anon_1: __SlateRecord_bpf_h_1897,
    flags: u32,
    __slate_anon_3: __SlateRecord_bpf_h_1904,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1897 {
    new_prog_fd: u32,
    new_map_fd: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1904 {
    old_prog_fd: u32,
    old_map_fd: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1916 {
    link_fd: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1920 {
    r#type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1924 {
    link_fd: u32,
    flags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1929 {
    prog_fd: u32,
    map_fd: u32,
    flags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1935 {
    flags: u32,
    bpffs_fd: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1940 {
    stream_buf: u64,
    stream_buf_len: u32,
    stream_id: u32,
    prog_fd: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1947 {
    map_fd: u32,
    prog_fd: u32,
    flags: u32,
}


fn p4d_bad(p4d: __SlateRecord_pgtable_types_h_342) -> i32 {
let mut ignore_flags: u64 = _AT!(u64, 1i32) << 0i32 | _AT!(u64, 1i32) << 1i32 | 0 | _AT!(u64, 1i32) << 5i32 | 0 | _AT!(u64, 1i32) << 6i32 | 0 | 0 | _AT!(u64, 0u64) | _AT!(u64, 1i32) << 2i32;
if IS_ENABLED__2!() != 0 {
let __v0: u64 = ignore_flags;
ignore_flags = __v0 | _AT!(u64, 1i32) << 63i32;
}
(p4d_flags(p4d) & !ignore_flags != 0) as i32
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_6842 {
    raw_tracepoint: __SlateRecord_bpf_h_6843,
    tracing: __SlateRecord_bpf_h_6849,
    cgroup: __SlateRecord_bpf_h_6856,
    iter: __SlateRecord_bpf_h_6860,
    netns: __SlateRecord_bpf_h_6884,
    xdp: __SlateRecord_bpf_h_6888,
    struct_ops: __SlateRecord_bpf_h_6891,
    netfilter: __SlateRecord_bpf_h_6894,
    kprobe_multi: __SlateRecord_bpf_h_6900,
    uprobe_multi: __SlateRecord_bpf_h_6907,
    tracing_multi: __SlateRecord_bpf_h_6917,
    perf_event: __SlateRecord_bpf_h_6926,
    tcx: __SlateRecord_bpf_h_6959,
    netkit: __SlateRecord_bpf_h_6963,
    sockmap: __SlateRecord_bpf_h_6967,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6843 {
    tp_name: u64,
    tp_name_len: u32,
    __slate_bits_0: __slate_bits::__SlateBits1010U0,
    cookie: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6849 {
    attach_type: u32,
    target_obj_id: u32,
    target_btf_id: u32,
    __slate_bits_0: __slate_bits::__SlateBits1011U0,
    cookie: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6856 {
    cgroup_id: u64,
    attach_type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6860 {
    target_name: u64,
    target_name_len: u32,
    __slate_anon_2: __SlateRecord_bpf_h_6868,
    __slate_anon_3: __SlateRecord_bpf_h_6873,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_6868 {
    map: __SlateRecord_bpf_h_6869,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6869 {
    map_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_6873 {
    cgroup: __SlateRecord_bpf_h_6874,
    task: __SlateRecord_bpf_h_6878,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6874 {
    cgroup_id: u64,
    order: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6878 {
    tid: u32,
    pid: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6884 {
    netns_ino: u32,
    attach_type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6888 {
    ifindex: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6891 {
    map_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6894 {
    pf: u32,
    hooknum: u32,
    priority: i32,
    flags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6900 {
    addrs: u64,
    count: u32,
    flags: u32,
    missed: u64,
    cookies: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6907 {
    path: u64,
    offsets: u64,
    ref_ctr_offsets: u64,
    cookies: u64,
    path_size: u32,
    count: u32,
    flags: u32,
    pid: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6917 {
    attach_type: u32,
    count: u32,
    btf_obj_id: u32,
    __slate_bits_0: __slate_bits::__SlateBits1025U0,
    ids: u64,
    addrs: u64,
    cookies: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6926 {
    r#type: u32,
    __slate_bits_0: __slate_bits::__SlateBits1026U0,
    __slate_anon_2: __SlateRecord_bpf_h_6929,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_6929 {
    uprobe: __SlateRecord_bpf_h_6930,
    kprobe: __SlateRecord_bpf_h_6937,
    tracepoint: __SlateRecord_bpf_h_6945,
    event: __SlateRecord_bpf_h_6951,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6930 {
    file_name: u64,
    name_len: u32,
    offset: u32,
    cookie: u64,
    ref_ctr_offset: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6937 {
    func_name: u64,
    name_len: u32,
    offset: u32,
    addr: u64,
    missed: u64,
    cookie: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6945 {
    tp_name: u64,
    name_len: u32,
    __slate_bits_0: __slate_bits::__SlateBits1030U0,
    cookie: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6951 {
    config: u64,
    r#type: u32,
    __slate_bits_0: __slate_bits::__SlateBits1031U0,
    cookie: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6959 {
    ifindex: u32,
    attach_type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6963 {
    ifindex: u32,
    attach_type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6967 {
    map_id: u32,
    attach_type: u32,
}


fn p4d_index(address: u64) -> u64 {
address >> 39i32 & (unsafe { ptrs_per_p4d }.wrapping_sub(1) as u64)
}

fn pgd_page_vaddr(pgd: __SlateRecord_pgtable_types_h_295) -> u64 {
__va!(pgd_val!(pgd) & (((!(_AC!() << 12i32).wrapping_sub(1) as i64) as u64) & 4503599627370495u64)) as u64
}

fn p4d_offset(pgd: *mut __SlateRecord_pgtable_types_h_295, address: u64) -> *mut __SlateRecord_pgtable_types_h_342 {
if pgtable_l5_enabled__4!() == 0 {
return pgd as *mut __SlateRecord_pgtable_types_h_342;
}
(pgd_page_vaddr(unsafe { *pgd }) as *mut __SlateRecord_pgtable_types_h_342).wrapping_offset(p4d_index(address) as isize)
}

fn pgd_bad(pgd: __SlateRecord_pgtable_types_h_295) -> i32 {
let mut ignore_flags: u64 = _AT!(u64, 1i32) << 2i32;
if pgtable_l5_enabled__4!() == 0 {
return 0;
}
let __v0: u64 = ignore_flags;
ignore_flags = __v0 | _AT!(u64, 1i32) << 63i32;
(pgd_flags(pgd) & !ignore_flags != _AT!(u64, 1i32) << 0i32 | _AT!(u64, 1i32) << 1i32 | 0 | _AT!(u64, 1i32) << 5i32 | 0 | _AT!(u64, 1i32) << 6i32 | 0 | 0 | _AT!(u64, 0u64)) as i32
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_btf_h_64 {
    size: u32,
    r#type: u32,
}


fn pgd_none(pgd: __SlateRecord_pgtable_types_h_295) -> i32 {
if pgtable_l5_enabled__4!() == 0 {
return 0;
}
// There is no need to do a workaround for the KNL stray
// A/D bit erratum here.  PGDs only point to page tables
// except on 32-bit non-PAE which is not supported on
// KNL.
(native_pgd_val(pgd) == 0) as i32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct btf {
}


// All top-level MITIGATION_PAGE_TABLE_ISOLATION page tables are order-1 pages
// (8k-aligned and 8k in size).  The kernel one is at the beginning 4k and
// the user one is in the last 4k.  To switch between them, you
// just need to flip the 12th bit in their addresses.
fn pte_swp_exclusive(pte: __SlateRecord_pgtable_64_types_h_21) -> bool {
pte_flags(pte) & _AT!(u64, 1i32) << 3i32 != 0
}

fn pte_swp_clear_exclusive(pte: __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_pgtable_64_types_h_21 {
return pte_clear_flags(pte, _AT!(u64, 1i32) << 3i32);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_verifier_log {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_pin {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ftrace_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_verifier_env {
}


fn pte_swp_soft_dirty(pte: __SlateRecord_pgtable_64_types_h_21) -> i32 {
(pte_flags(pte) & _AT!(u64, 0i32)) as i32
}

fn pte_swp_clear_soft_dirty(pte: __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_pgtable_64_types_h_21 {
return pte_clear_flags(pte, _AT!(u64, 0i32));
}

#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_arena {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_iter_aux_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_local_storage_map {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_func_state {
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
union __SlateRecord_bpf_h_256 {
    kptr: btf_field_kptr,
    graph_root: btf_field_graph_root,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ctl_table_set {
    is_seen: Option<unsafe extern "C" fn(*mut ctl_table_set) -> i32>,
    dir: ctl_dir,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_329 {
    work: work_struct,
    rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_offload_dev {
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
union __SlateRecord_bpf_h_996 {
    __slate_anon_0: __SlateRecord_bpf_h_997,
    arg_type: [u32; 12],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_997 {
    arg1_type: u32,
    arg2_type: u32,
    arg3_type: u32,
    arg4_type: u32,
    arg5_type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1006 {
    __slate_anon_0: __SlateRecord_bpf_h_1007,
    arg_btf_id: [*mut u32; 12],
    __slate_anon_2: __SlateRecord_bpf_h_1015,
    arg_size: [u64; 12],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1007 {
    arg1_btf_id: *mut u32,
    arg2_btf_id: *mut u32,
    arg3_btf_id: *mut u32,
    arg4_btf_id: *mut u32,
    arg5_btf_id: *mut u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1015 {
    arg1_size: u64,
    arg2_size: u64,
    arg3_size: u64,
    arg4_size: u64,
    arg5_size: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array {
    root: *mut assoc_array_ptr,
    nr_leaves_on_tree: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1109 {
    ctx_field_size: i32,
    __slate_anon_1: __SlateRecord_bpf_h_1111,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1111 {
    btf: *mut btf,
    btf_id: u32,
    ref_id: u32,
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
struct bpf_reg_state {
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
union __SlateRecord_bpf_h_1393 {
    rcu: callback_head,
    work: work_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1423 {
    model: btf_func_model,
    addr: *mut (),
    ftrace_managed: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1442 {
    old_image: *mut bpf_tramp_image,
    old_flags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rhash_head {
    next: *mut rhash_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rhlist_head {
    rhead: rhash_head,
    next: *mut rhlist_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1705 {
    tail_call: __SlateRecord_bpf_h_1706,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1706 {
    map: *mut bpf_map,
    key: u32,
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
struct bpf_kfunc_desc_tab {
}


fn hrtimer_get_expires(timer: *const hrtimer) -> i64 {
unsafe { (*timer).node.expires }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1821_2 {
    keyring_serial: i32,
    keyring_type: u8,
    verdict: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_kfunc_btf_tab {
}


fn hrtimer_get_softexpires(timer: *const hrtimer) -> i64 {
unsafe { (*timer)._softexpires }
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1892 {
    work: work_struct,
    rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1926 {
    digest: [u8; 32],
    tag: [u8; 8],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_prog_stats {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sock_fprog_kern {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1937 {
    __slate_anon_0: __SlateRecord_bpf_h_1938,
    __slate_anon_1: __SlateRecord_bpf_h_1939,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1938 {
    __empty_insns: __SlateRecord_bpf_h_1938_2,
    insns: [sock_filter; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1938_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1939 {
    __empty_insnsi: __SlateRecord_bpf_h_1939_2,
    insnsi: [bpf_insn; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1939_2 {
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
union __SlateRecord_bpf_h_1964 {
    rcu: callback_head,
    work: work_struct,
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
union __SlateRecord_bpf_h_2532 {
    cgroup_storage: [*mut bpf_cgroup_storage; 2],
    bpf_cookie: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_cgroup_storage {
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
union __SlateRecord_fs_context_h_66 {
    string: *mut u8,
    blob: *mut (),
    name: *mut filename,
    file: *mut file,
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
    __slate_bits_0: __slate_bits::__SlateBits322U0,
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


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_fs_parser_h_56 {
    boolean: bool,
    int_32: i32,
    uint_32: u32,
    uint_64: u64,
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
}


// group exit in progress
fn task_sigpending(p: *mut task_struct) -> i32 {
unlikely__2!(test_tsk_thread_flag(p, 1i32)) as i32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ring_buffer_iter {
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

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_perf_event_h_410 {
    sample_period: u64,
    sample_freq: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_perf_event_h_471 {
    wakeup_events: u32,
    wakeup_watermark: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_perf_event_h_477 {
    bp_addr: u64,
    kprobe_func: u64,
    uprobe_path: u64,
    config1: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_perf_event_h_483 {
    bp_len: u64,
    kprobe_addr: u64,
    probe_offset: u64,
    config2: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_perf_event_h_530 {
    aux_action: u32,
    __slate_anon_1: __SlateRecord_perf_event_h_532,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_532 {
    __slate_bits_0: __slate_bits::__SlateBits1150U0,
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

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_1322 {
    __slate_bits_0: __slate_bits::__SlateBits1152U0,
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
struct __SlateRecord_perf_event_h_1512 {
    var1_dw: u32,
    var2_w: u16,
    var3_w: u16,
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
union __SlateRecord_perf_event_h_76 {
    next: *mut perf_raw_frag,
    pad: u64,
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
union __SlateRecord_perf_event_h_149 {
    __slate_anon_0: __SlateRecord_perf_event_h_150,
    __slate_anon_1: __SlateRecord_perf_event_h_165,
    __slate_anon_2: __SlateRecord_perf_event_h_174,
    __slate_anon_3: __SlateRecord_perf_event_h_177,
    __slate_anon_4: __SlateRecord_perf_event_h_181,
    __slate_anon_5: __SlateRecord_perf_event_h_186,
    __slate_anon_6: __SlateRecord_perf_event_h_196,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_150 {
    config: u64,
    config1: u64,
    last_tag: u64,
    dyn_constraint: u64,
    config_base: u64,
    event_base: u64,
    event_base_rdpmc: i32,
    idx: i32,
    last_cpu: i32,
    flags: i32,
    extra_reg: hw_perf_event_extra,
    branch_reg: hw_perf_event_extra,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_165 {
    aux_config: u64,
    aux_paused: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_174 {
    hrtimer: hrtimer,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_177 {
    tp_list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_181 {
    pwr_acc: u64,
    ptsc: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_186 {
    info: arch_hw_breakpoint,
    bp_list: rhlist_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_196 {
    iommu_bank: u8,
    iommu_cntr: u8,
    padding: u16,
    conf: u64,
    conf1: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_perf_event_h_244 {
    __slate_anon_0: __SlateRecord_perf_event_h_245,
    __slate_anon_1: __SlateRecord_perf_event_h_259,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_245 {
    last_period: u64,
    period_left: __SlateRecord_local64_h_26,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_259 {
    saved_metric: u64,
    saved_slots: u64,
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
struct perf_buffer {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct event_filter {
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
// VFS does not apply umask
#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_perf_event_h_1156 {
    flags: u64,
    aux_flags: u64,
    __slate_anon_2: __SlateRecord_perf_event_h_1159,
}


#[repr(C, align(8))]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_1159 {
    __slate_bits_0: __slate_bits::__SlateBits1192U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_perf_event_h_1163 {
    addr: *mut (),
    head: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct lockref {
    __slate_anon_0: __SlateRecord_lockref_h_26,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_1315 {
    pid: u32,
    tid: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_1321 {
    cpu: u32,
    reserved: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_array {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct array_buffer {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tracer {
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


fn d_unhashed(dentry: *const dentry) -> i32 {
hlist_bl_unhashed(unsafe { core::ptr::addr_of!((*dentry).d_hash) }) as i32
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_trace_events_h_275 {
    __slate_anon_0: __SlateRecord_trace_events_h_276,
    define_fields: Option<unsafe extern "C" fn(*mut trace_event_call) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_trace_events_h_276 {
    name: *const u8,
    size: i32,
    align: i32,
    __slate_bits_0: __slate_bits::__SlateBits1209U0,
    filter_type: i32,
    len: i32,
}


fn d_unlinked(dentry: *const dentry) -> i32 {
(d_unhashed(dentry) != 0 && !IS_ROOT!(dentry)) as i32
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_trace_events_h_379 {
    name: *const u8,
    tp: *mut tracepoint,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_trace_events_h_390 {
    module: *mut (),
    refcnt: __SlateRecord_types_h_190,
}


fn cant_mount(dentry: *const dentry) -> i32 {
((unsafe { (*dentry).d_flags }) & 256) as i32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct path {
    mnt: *mut vfsmount,
    dentry: *mut dentry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_swapfile_c_3082 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_swapfile_c_3613 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_swapfile_c_3613_2 {
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
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1126U0 {
        #[bits(8)]
        pub r#type: u32,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits319U0 {
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
    pub struct __SlateBits70U0 {
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
    pub struct __SlateBits70U1 {
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
    pub struct __SlateBits748U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits925U0 {
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
    pub struct __SlateBits665U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits322U0 {
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
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits774U0 {
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
    pub struct __SlateBits771U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits774U1 {
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
    pub struct __SlateBits774U2 {
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
    pub struct __SlateBits774U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits70U2 {
        #[bits(1)]
        pub mce_ripv: u64,
        #[bits(1)]
        pub mce_whole_page: u64,
        #[bits(62)]
        pub __mce_reserved: u64,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits137U0 {
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
    pub struct __SlateBits1178U0 {
        #[bits(1)]
        pub embedded: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1192U0 {
        #[bits(1)]
        pub skip_read: u64,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1128U0 {
        #[bits(1)]
        pub disabled: u64,
        #[bits(1)]
        pub inherit: u64,
        #[bits(1)]
        pub pinned: u64,
        #[bits(1)]
        pub exclusive: u64,
        #[bits(1)]
        pub exclude_user: u64,
        #[bits(1)]
        pub exclude_kernel: u64,
        #[bits(1)]
        pub exclude_hv: u64,
        #[bits(1)]
        pub exclude_idle: u64,
        #[bits(1)]
        pub mmap: u64,
        #[bits(1)]
        pub comm: u64,
        #[bits(1)]
        pub freq: u64,
        #[bits(1)]
        pub inherit_stat: u64,
        #[bits(1)]
        pub enable_on_exec: u64,
        #[bits(1)]
        pub task: u64,
        #[bits(1)]
        pub watermark: u64,
        #[bits(2)]
        pub precise_ip: u64,
        #[bits(1)]
        pub mmap_data: u64,
        #[bits(1)]
        pub sample_id_all: u64,
        #[bits(1)]
        pub exclude_host: u64,
        #[bits(1)]
        pub exclude_guest: u64,
        #[bits(1)]
        pub exclude_callchain_kernel: u64,
        #[bits(1)]
        pub exclude_callchain_user: u64,
        #[bits(1)]
        pub mmap2: u64,
        #[bits(1)]
        pub comm_exec: u64,
        #[bits(1)]
        pub use_clockid: u64,
        #[bits(1)]
        pub context_switch: u64,
        #[bits(1)]
        pub write_backward: u64,
        #[bits(1)]
        pub namespaces: u64,
        #[bits(1)]
        pub ksymbol: u64,
        #[bits(1)]
        pub bpf_event: u64,
        #[bits(1)]
        pub aux_output: u64,
        #[bits(1)]
        pub cgroup: u64,
        #[bits(1)]
        pub text_poke: u64,
        #[bits(1)]
        pub build_id: u64,
        #[bits(1)]
        pub inherit_thread: u64,
        #[bits(1)]
        pub remove_on_exec: u64,
        #[bits(1)]
        pub sigtrap: u64,
        #[bits(1)]
        pub defer_callchain: u64,
        #[bits(1)]
        pub defer_output: u64,
        #[bits(24)]
        pub __reserved_1: u64,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1150U0 {
        #[bits(1)]
        pub aux_start_paused: u32,
        #[bits(1)]
        pub aux_pause: u32,
        #[bits(1)]
        pub aux_resume: u32,
        #[bits(29)]
        pub __reserved_3: u32,
    }
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1153U0 {
        #[bits(1)]
        pub mispred: u64,
        #[bits(1)]
        pub predicted: u64,
        #[bits(1)]
        pub in_tx: u64,
        #[bits(1)]
        pub abort: u64,
        #[bits(16)]
        pub cycles: u64,
        #[bits(4)]
        pub r#type: u64,
        #[bits(2)]
        pub spec: u64,
        #[bits(4)]
        pub new_type: u64,
        #[bits(3)]
        pub r#priv: u64,
        #[bits(31)]
        pub reserved: u64,
    }
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1152U0 {
        #[bits(5)]
        pub mem_op: u64,
        #[bits(14)]
        pub mem_lvl: u64,
        #[bits(5)]
        pub mem_snoop: u64,
        #[bits(2)]
        pub mem_lock: u64,
        #[bits(7)]
        pub mem_dtlb: u64,
        #[bits(4)]
        pub mem_lvl_num: u64,
        #[bits(1)]
        pub mem_remote: u64,
        #[bits(2)]
        pub mem_snoopx: u64,
        #[bits(3)]
        pub mem_blk: u64,
        #[bits(3)]
        pub mem_hops: u64,
        #[bits(5)]
        pub mem_region: u64,
        #[bits(13)]
        pub mem_rsvd: u64,
    }
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits102U0 {
        #[bits(16)]
        pub cs: u64,
        #[bits(2)]
        pub sl: u64,
        #[bits(1)]
        pub wfe: u64,
        #[bits(45, access = na)]
        pub __slate_pad_3: u64,
    }
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits103U0 {
        #[bits(16)]
        pub ss: u64,
        #[bits(1)]
        pub sti: u64,
        #[bits(1)]
        pub swevent: u64,
        #[bits(1)]
        pub nmi: u64,
        #[bits(13, access = na)]
        pub __slate_pad_4: u16,
        #[bits(8)]
        pub vector: u64,
        #[bits(8, access = na)]
        pub __slate_pad_6: u8,
        #[bits(4)]
        pub r#type: u64,
        #[bits(4, access = na)]
        pub __slate_pad_8: u8,
        #[bits(1)]
        pub enclave: u64,
        #[bits(1)]
        pub l: u64,
        #[bits(1)]
        pub nested: u64,
        #[bits(1, access = na)]
        pub __slate_pad_12: u8,
        #[bits(4)]
        pub insnlen: u64,
    }
    #[bitfields::bitfield([u8; 3], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1042U0 {
        #[bits(1)]
        pub jited: u32,
        #[bits(1)]
        pub jit_requested: u32,
        #[bits(1)]
        pub jit_required: u32,
        #[bits(1)]
        pub gpl_compatible: u32,
        #[bits(1)]
        pub cb_access: u32,
        #[bits(1)]
        pub dst_needed: u32,
        #[bits(1)]
        pub blinding_requested: u32,
        #[bits(1)]
        pub blinded: u32,
        #[bits(1)]
        pub is_func: u32,
        #[bits(1)]
        pub kprobe_override: u32,
        #[bits(1)]
        pub has_callchain_buf: u32,
        #[bits(1)]
        pub enforce_expected_attach_type: u32,
        #[bits(1)]
        pub call_get_stack: u32,
        #[bits(1)]
        pub call_get_func_ip: u32,
        #[bits(1)]
        pub call_session_cookie: u32,
        #[bits(1)]
        pub tstamp_type_access: u32,
        #[bits(1)]
        pub sleepable: u32,
        #[bits(7, access = na)]
        pub __slate_pad_17: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits954U0 {
        #[bits(4)]
        pub dst_reg: u8,
        #[bits(4)]
        pub src_reg: u8,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits974U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits977U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1010U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1011U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1025U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1026U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1030U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1031U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1209U0 {
        #[bits(1)]
        pub is_signed: u32,
        #[bits(1)]
        pub needs_test: u32,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits254U0 {
        #[bits(1)]
        pub faultable: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits145U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield([u8; 3], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits656U0 {
        #[bits(8)]
        pub purpose: u32,
        #[bits(8)]
        pub phase: u32,
        #[bits(1)]
        pub need_free: bool,
        #[bits(1)]
        pub global: bool,
        #[bits(1)]
        pub oldapi: bool,
        #[bits(1)]
        pub exclusive: bool,
        #[bits(4, access = na)]
        pub __slate_pad_6: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits630U0 {
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
    pub struct __SlateBits680U0 {
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
    pub struct __SlateBits144U0 {
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
    pub struct __SlateBits475U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits160U0 {
        #[bits(24)]
        pub intel_native_model_id: u32,
        #[bits(8)]
        pub intel_type: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits161U0 {
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
    pub struct __SlateBits107U0 {
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
    pub struct __SlateBits108U0 {
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
    pub struct __SlateBits106U0 {
        #[bits(1)]
        pub initialized: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct delayed_call {
    r#fn: Option<unsafe extern "C" fn(*mut ())>,
    arg: *mut (),
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
    __slate_bits_0: __slate_bits::__SlateBits665U0,
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
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_385() };
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

fn percpu_ref_tryget_live_rcu(r#ref: *mut percpu_ref) -> bool {
let mut percpu_count: *mut u64 = unsafe { core::mem::zeroed() };
let mut ret: bool = false;
let mut __v0: i64;
let __ret_warn_on: i32 = (rcu_read_lock_held() == 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("386: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 386b - ., 3; .popsection\n# {0}", const 386 as i32, options(nostack, att_syntax));
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
core::arch::asm!("387: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 387b - ., 4; .popsection\n# {0}", const 387 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if likely!(__ref_is_percpu(r#ref, unsafe { &mut (*core::ptr::addr_of_mut!(percpu_count)) })) != 0 {
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

fn percpu_ref_tryget_live(r#ref: *mut percpu_ref) -> bool {
let mut ret: bool = false;
rcu_read_lock();
ret = percpu_ref_tryget_live_rcu(r#ref);
rcu_read_unlock();
ret
}

fn percpu_ref_put_many(r#ref: *mut percpu_ref, nr: u64) {
let mut percpu_count: *mut u64 = unsafe { core::mem::zeroed() };
rcu_read_lock();
if __ref_is_percpu(r#ref, unsafe { &mut (*core::ptr::addr_of_mut!(percpu_count)) }) {
loop {
__verify_pcpu_ptr__6!();
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
if unlikely!(atomic_long_sub_and_test(nr as i64, unsafe { core::ptr::addr_of_mut!((*(*r#ref).data).count) })) != 0 {
unsafe { unsafe { (*(*r#ref).data).release }.unwrap()(r#ref) };
}
}
rcu_read_unlock();
}

fn percpu_ref_put(r#ref: *mut percpu_ref) {
percpu_ref_put_many(r#ref, 1u64);
}

// Panic if kmem_cache_create() fails
// define SLAB_TYPESAFE_BY_RCU - **WARNING** READ THIS!
//
// This delays freeing the SLAB page by a grace period, it does _NOT_
// delay object freeing. This means that if you do kmem_cache_free()
// that memory location is free to be reused at any time. Thus it may
// be possible to see another object there in the same RCU grace period.
//
// This feature only ensures the memory location backing the object
// stays valid, the trick to using this is relying on an independent
// object validation pass. Something like:
//
// ::
//
//  begin:
//   rcu_read_lock();
//   obj = lockless_lookup(key);
//   if (obj) {
//     if (!try_get_ref(obj)) // might fail for free objects
//       rcu_read_unlock();
//       goto begin;
//
//     if (obj->key != key) { // not the object we expected
//       put_ref(obj);
//       rcu_read_unlock();
//       goto begin;
//     }
//   }
//  rcu_read_unlock();
//
// This is useful if we need to approach a kernel structure obliquely,
// from its address obtained without the usual locking. We can lock
// the structure to stabilize it and check it's still at the given address,
// only if we can be sure that the memory has not been meanwhile reused
// for some other kind of object (which our subsystem's lock might corrupt).
//
// rcu_read_lock before reading the address, then rcu_read_unlock after
// taking the spinlock within the structure expected at that address.
//
// Note that object identity check has to be done *after* acquiring a
// reference, therefore user has to ensure proper ordering for loads.
// Similarly, when initializing objects allocated with SLAB_TYPESAFE_BY_RCU,
// the newly allocated object has to be fully initialized *before* its
// refcount gets initialized and proper ordering for stores is required.
// refcount_{add|inc}_not_zero_acquire() and refcount_set_release() are
// designed with the proper fences required for reference counting objects
// allocated with SLAB_TYPESAFE_BY_RCU.
//
// Note that it is not possible to acquire a lock within a structure
// allocated with SLAB_TYPESAFE_BY_RCU without first acquiring a reference
// as described above.  The reason is that SLAB_TYPESAFE_BY_RCU pages
// are not zeroed before being given to the slab, which means that any
// locks must be initialized after each and every kmem_struct_alloc().
// Alternatively, make the ctor passed to kmem_cache_create() initialize
// the locks at page-allocation time, as is done in __i915_request_ctor(),
// sighand_ctor(), and anon_vma_ctor().  Such a ctor permits readers
// to safely acquire those ctor-initialized locks under rcu_read_lock()
// protection.
//
// Note that SLAB_TYPESAFE_BY_RCU was originally named SLAB_DESTROY_BY_RCU.
#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache_args {
    align: u32,
    useroffset: u32,
    usersize: u32,
    freeptr_offset: u32,
    use_freeptr_offset: bool,
    ctor: Option<unsafe extern "C" fn(*mut ())>,
    sheaf_capacity: u32,
}


fn __kmem_cache_create(name: *const u8, size: u32, align: u32, flags: u32, ctor: Option<unsafe extern "C" fn(*mut ())>) -> *mut kmem_cache {
let mut kmem_args: kmem_cache_args = {
let mut __t0: kmem_cache_args = unsafe { core::mem::zeroed() };
__t0.align = align;
__t0.ctor = ctor;
__t0
};
unsafe { __kmem_cache_create_args(name, size, core::ptr::addr_of_mut!(kmem_args), flags) }
}

fn __kmem_cache_default_args(name: *const u8, size: u32, args: *mut kmem_cache_args, flags: u32) -> *mut kmem_cache {
let mut kmem_default_args: kmem_cache_args = {
let mut __t0: kmem_cache_args = unsafe { core::mem::zeroed() };
__t0
};
// Make sure we don't get passed garbage.
let mut __v0: i64;
let __ret_warn_on: i32 = !args.is_null() as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("388: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 388b - ., 3; .popsection\n# {0}", const 388 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___kmem_cache_default_args_1, sym __slate_asm_string___kmem_cache_default_args_2, const 448 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
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
if __v0 != 0 {
return ERR_PTR(22i32.wrapping_neg() as i64) as *mut kmem_cache;
}
unsafe { __kmem_cache_create_args(name, size, core::ptr::addr_of_mut!(kmem_default_args), flags) }
}

// kmem_cache_create - Create a kmem cache.
// @__name: A string which is used in /proc/slabinfo to identify this cache.
// @__object_size: The size of objects to be created in this cache.
// @__args: Optional arguments, see &struct kmem_cache_args. Passing %NULL
// 	    means defaults will be used for all the arguments.
//
// This is currently implemented as a macro using ``_Generic()`` to call
// either the new variant of the function, or a legacy one.
//
// The new variant has 4 parameters:
// ``kmem_cache_create(name, object_size, args, flags)``
//
// See __kmem_cache_create_args() which implements this.
//
// The legacy variant has 5 parameters:
// ``kmem_cache_create(name, object_size, align, flags, ctor)``
//
// The align and ctor parameters map to the respective fields of
// &struct kmem_cache_args
//
// Context: Cannot be called within a interrupt, but can be interrupted.
//
// Return: a pointer to the cache on success, NULL on failure.
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
// All kvzalloc aliases for kmalloc_(obj|objs|flex).
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
fn inode_lock(inode: *mut inode) {
unsafe { down_write(core::ptr::addr_of_mut!((*inode).i_rwsem)) };
}

fn inode_unlock(inode: *mut inode) {
unsafe { up_write(core::ptr::addr_of_mut!((*inode).i_rwsem)) };
}

fn i_size_read(inode: *const inode) -> i64 {
// Pairs with smp_store_release() in i_size_write()
let mut __v0: i64;
let mut ___p1: i64 = 0;
let mut __v1: i64;
loop {
if !(__native_word__8!() || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_399() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*inode).i_size) as *const i64) };
___p1 = __v1;
loop {
if !__native_word__8!() {
unsafe { __compiletime_assert_400() };
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


fn file_inode(f: *const file) -> *mut inode {
unsafe { (*f).f_inode }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct fasync_struct {
    fa_lock: rwlock,
    magic: i32,
    fa_fd: i32,
    fa_next: *mut fasync_struct,
    fa_file: *mut file,
    fa_rcu: callback_head,
}


core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_nr_swap_pages: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_add_swap_extent: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section\t\".initcall6.init\", \"a\"\t\t\n__initcall__kmod_swapfile__920_3342_procswaps_init6:\t\t\t\n.long\t{2} - .\t\n.previous\t\t\t\t\t\n\n.section\t\".initcall7.init\", \"a\"\t\t\n__initcall__kmod_swapfile__923_3351_max_swapfiles_check7:\t\t\t\n.long\t{3} - .\t\n.previous\t\t\t\t\t\n\n.section\t\".initcall4.init\", \"a\"\t\t\n__initcall__kmod_swapfile__929_3958_swapfile_init4:\t\t\t\n.long\t{4} - .\t\n.previous\t\t\t\t\t\n", sym nr_swap_pages, sym add_swap_extent, sym procswaps_init, sym max_swapfiles_check, sym swapfile_init, options(att_syntax));

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


// Do not truncate: swapon got its bmaps
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
struct __filename_head {
    name: *const u8,
    refcnt: i32,
    aname: *mut audit_names,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct filename {
    __slate_anon_0: __filename_head,
    iname: [u8; 168],
}


fn getname(name: *const u8) -> *mut filename {
unsafe { getname_flags(name, 0i32) }
}

#[inline(always)]
fn class_filename_destructor(p: *mut *mut filename) {
let _T: *mut filename = unsafe { *p };
unsafe { putname(_T) };
}

#[inline(always)]
fn class_filename_constructor(p: *const u8) -> *mut filename {
let t: *mut filename = getname(p);
t
}

#[repr(C)]
#[derive(Clone, Copy)]
struct offset_ctx {
    mt: maple_tree,
    next_offset: u64,
}


fn __page_ref_mod(page: *mut page, v: i32) {
}

fn __page_ref_mod_and_test(page: *mut page, v: i32, ret: i32) {
}

fn page_ref_count(page: *const page) -> i32 {
atomic_read(unsafe { core::ptr::addr_of!((*page)._refcount) })
}

fn folio_ref_count(folio: &folio) -> i32 {
page_ref_count(unsafe { core::ptr::addr_of!((*(folio as *const folio)).__slate_anon_0.page) })
}

fn page_ref_inc(page: *mut page) {
atomic_inc(unsafe { core::ptr::addr_of_mut!((*page)._refcount) });
if page_ref_tracepoint_active!() != 0 {
__page_ref_mod(page, 1i32);
}
}

fn folio_ref_inc(folio: *mut folio) {
page_ref_inc(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.page) });
}

fn page_ref_dec_and_test(page: *mut page) -> i32 {
let ret: i32 = atomic_dec_and_test(unsafe { core::ptr::addr_of_mut!((*page)._refcount) }) as i32;
if page_ref_tracepoint_active!() != 0 {
__page_ref_mod_and_test(page, 1i32.wrapping_neg(), ret);
}
ret
}

fn pmd_index(address: u64) -> u64 {
address >> 21i32 & 511u64
}

fn pud_index(address: u64) -> u64 {
address >> 30i32 & 511u64
}

fn pte_unmap(pte: *mut __SlateRecord_pgtable_64_types_h_21) {
rcu_read_unlock();
}

fn pmd_offset(pud: *mut __SlateRecord_pgtable_types_h_368, address: u64) -> *mut __SlateRecord_pgtable_64_types_h_22 {
pud_pgtable(unsafe { *pud }).wrapping_offset(pmd_index(address) as isize)
}

fn pud_offset(p4d: *mut __SlateRecord_pgtable_types_h_342, address: u64) -> *mut __SlateRecord_pgtable_types_h_368 {
p4d_pgtable(unsafe { *p4d }).wrapping_offset(pud_index(address) as isize)
}

fn pgd_offset_pgd(pgd: *mut __SlateRecord_pgtable_types_h_295, address: u64) -> *mut __SlateRecord_pgtable_types_h_295 {
pgd.wrapping_offset(pgd_index!(address) as isize)
}

// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
fn set_ptes(mm: *mut mm_struct, addr: u64, mut ptep: *mut __SlateRecord_pgtable_64_types_h_21, mut pte: __SlateRecord_pgtable_64_types_h_21, mut nr: u32) {
page_table_check_ptes_set(mm, addr, ptep, pte, nr);
loop {
set_pte!(ptep, pte);
let __v1: u32 = nr.wrapping_sub(1);
nr = __v1;
if __v1 == 0 {
break;
}
let __v3: *mut __SlateRecord_pgtable_64_types_h_21 = ptep;
ptep = __v3.wrapping_offset(1 as isize);
pte = pte_next_pfn!(pte);
}
}

fn ptep_get(ptep: *mut __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_pgtable_64_types_h_21 {
let mut __v0: __SlateRecord_pgtable_64_types_h_21;
loop {
if !(__native_word__7!() || ((const { core::mem::size_of::<__SlateRecord_pgtable_64_types_h_21>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_424() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(ptep as *const __SlateRecord_pgtable_64_types_h_21) };
return __v0;
}

fn ptep_get_lockless(ptep: *mut __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_pgtable_64_types_h_21 {
return ptep_get(ptep);
}

fn arch_swap_invalidate_page(r#type: i32, offset: u64) {
}

fn arch_swap_invalidate_area(r#type: i32) {
}

fn arch_swap_restore(entry: __SlateRecord_mm_types_h_288, folio: *mut folio) {
}

fn pgd_none_or_clear_bad(pgd: *mut __SlateRecord_pgtable_types_h_295) -> i32 {
if pgd_none(unsafe { *pgd }) != 0 {
return 1;
}
if unlikely__2!(pgd_bad(unsafe { *pgd })) != 0 {
unsafe { pgd_clear_bad(pgd) };
return 1;
}
0
}

fn p4d_none_or_clear_bad(p4d: *mut __SlateRecord_pgtable_types_h_342) -> i32 {
if p4d_none(unsafe { *p4d }) != 0 {
return 1;
}
if unlikely__2!(p4d_bad(unsafe { *p4d })) != 0 {
unsafe { p4d_clear_bad(p4d) };
return 1;
}
0
}

fn pud_none_or_clear_bad(pud: *mut __SlateRecord_pgtable_types_h_368) -> i32 {
if pud_none(unsafe { *pud }) != 0 {
return 1;
}
if unlikely__2!(pud_bad(unsafe { *pud })) != 0 {
unsafe { pud_clear_bad(pud) };
return 1;
}
0
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


fn folio_page_idx(folio: *const folio, page: *const page) -> u64 {
((unsafe { page.offset_from(core::ptr::addr_of!((*folio).__slate_anon_0.page) as *const page) }) as i64) as u64
}

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


fn mm_flags_test(flag: i32, mm: *const mm_struct) -> bool {
test_bit__5!(flag, unsafe { (*core::ptr::addr_of!((*mm).__slate_anon_0.flags)).__mm_flags.as_ptr() as *const u64 }) != 0
}

// Helper macro which bitwise-or combines the specified input flags into a
// vma_flags_t bitmap value. E.g.:
//
// vma_flags_t flags = mk_vma_flags(VMA_IO_BIT, VMA_PFNMAP_BIT,
//              VMA_DONTEXPAND_BIT, VMA_DONTDUMP_BIT);
//
// The compiler cleverly optimises away all of the work and this ends up being
// equivalent to aggregating the values manually.
#[inline(always)]
fn vma_flags_test(flags: *const __SlateRecord_mm_types_h_871, bit: i32) -> bool {
let bitmap: *const u64 = unsafe { (*flags).__vma_flags.as_ptr() as *const u64 };
test_bit__5!(bit, bitmap) != 0
}

// Test whether any specified VMA flag is set, e.g.:
//
// if (vma_flags_test_any(flags, VMA_READ_BIT, VMA_MAYREAD_BIT)) { ... }
// Test whether ALL specified VMA flags are set, e.g.:
//
// if (vma_flags_test_all(flags, VMA_READ_BIT, VMA_MAYREAD_BIT)) { ... }
fn vma_next(vmi: &mut vma_iterator) -> *mut vm_area_struct {
// Uses mas_find() to get the first VMA when the iterator starts.
// Calling mas_next() could skip the first entry.
(unsafe { mas_find(core::ptr::addr_of_mut!((*(vmi as *mut vma_iterator)).mas), 18446744073709551615u64) }) as *mut vm_area_struct
}

fn folio_large_order(folio: *const folio) -> u32 {
((unsafe { (*folio).__slate_anon_1.__slate_anon_0._flags_1 }) & 255) as u32
}

fn folio_large_nr_pages(folio: *const folio) -> u64 {
(1i64 << folio_large_order(folio)) as u64
}

fn folio_order(folio: *const folio) -> u32 {
if !folio_test_large(folio) {
return 0;
}
folio_large_order(folio)
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
    __slate_bits_0: __slate_bits::__SlateBits748U0,
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
    __slate_bits_0: __slate_bits::__SlateBits680U0,
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


fn put_page_testzero(page: *mut page) -> i32 {
VM_BUG_ON_PAGE!();
page_ref_dec_and_test(page)
}

fn folio_put_testzero(folio: *mut folio) -> i32 {
put_page_testzero(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.page) })
}

fn folio_large_mapcount(folio: *const folio) -> i32 {
VM_WARN_ON_FOLIO!();
atomic_read(unsafe { core::ptr::addr_of!((*folio).__slate_anon_1.__slate_anon_0.__slate_anon_2.__slate_anon_0._large_mapcount) }).wrapping_add(1)
}

fn folio_mapcount(folio: *const folio) -> i32 {
let mut mapcount: i32 = 0;
if likely!(!folio_test_large(folio)) != 0 {
mapcount = atomic_read(unsafe { core::ptr::addr_of!((*folio).__slate_anon_0.__slate_anon_0._mapcount) }).wrapping_add(1);
if page_mapcount_is_type(mapcount as u32) {
mapcount = 0;
}
return mapcount;
}
folio_large_mapcount(folio)
}

fn folio_mapped(folio: *const folio) -> bool {
folio_mapcount(folio) >= 1
}

fn virt_to_folio(x: *const ()) -> *mut folio {
let page: *mut page = virt_to_page!(x);
page_folio!(page)
}

// 127: arbitrary random number, small enough to assemble well
fn folio_get(folio: *mut folio) {
VM_BUG_ON_FOLIO!();
folio_ref_inc(folio);
}

fn folio_put(folio: *mut folio) {
if folio_put_testzero(folio) != 0 {
unsafe { __folio_put(folio) };
}
}

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
fn mk_pte(page: *const page, pgprot: pgprot) -> __SlateRecord_pgtable_64_types_h_21 {
return pfn_pte(((unsafe { page.offset_from((vmemmap_base as *mut page) as *const page) }) as i64) as u64, pgprot);
}

fn folio_nr_pages(folio: *const folio) -> u64 {
if !folio_test_large(folio) {
return 1;
}
folio_large_nr_pages(folio)
}

#[inline(always)]
fn lowmem_page_address(page: *const page) -> *mut () {
page_to_virt!(page)
}

fn folio_address(folio: *const folio) -> *mut () {
page_address__2!(unsafe { core::ptr::addr_of!((*folio).__slate_anon_0.page) })
}

fn inc_mm_counter(mm: *mut mm_struct, member: i32) {
percpu_counter_inc(unsafe { (*mm).__slate_anon_0.rss_stat.as_mut_ptr() as *mut percpu_counter }.wrapping_offset(member as isize));
unsafe { mm_trace_rss_stat(mm, member) };
}

fn dec_mm_counter(mm: *mut mm_struct, member: i32) {
percpu_counter_dec(unsafe { (*mm).__slate_anon_0.rss_stat.as_mut_ptr() as *mut percpu_counter }.wrapping_offset(member as isize));
unsafe { mm_trace_rss_stat(mm, member) };
}

fn pte_offset_map(pmd: *mut __SlateRecord_pgtable_64_types_h_22, addr: u64) -> *mut __SlateRecord_pgtable_64_types_h_21 {
unsafe { __pte_offset_map(pmd, addr, core::ptr::null_mut::<__SlateRecord_pgtable_64_types_h_22>()) }
}

// Mask used for checking in page_pool_page_is_pp() below. page->pp_magic is
// OR'ed with PP_SIGNATURE after the allocation in order to preserve bit 0 for
// the head page of compound page and bit 1 for pfmemalloc page, as well as the
// bits used for the DMA index. page_is_pfmemalloc() is checked in
// __page_pool_put_page() to avoid recycling the pfmemalloc page.
fn kmap_local_folio(folio: *const folio, offset: u64) -> *mut () {
(unsafe { (folio_address(folio) as *const u8).wrapping_offset(offset as isize) }) as *mut ()
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
fn folio_release_kmap(folio: *mut folio, addr: *mut ()) {
loop {
loop {
if __same_type!() != 0 {
unsafe { __compiletime_assert_518() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__kunmap_local(addr as *const ());
if true {
break;
}
}
folio_put(folio);
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
    __slate_bits_0: __slate_bits::__SlateBits774U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits774U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits774U2,
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
    __slate_bits_3: __slate_bits::__SlateBits774U3,
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
    __slate_bits_0: __slate_bits::__SlateBits771U0,
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


fn blk_queue_is_zoned(q: *mut request_queue) -> bool {
IS_ENABLED!() != 0 && (unsafe { (*q).limits.features }) & 1024u32 != 0
}

fn bdev_get_queue(bdev: *mut block_device) -> *mut request_queue {
return unsafe { (*bdev).bd_queue }; // this is never NULL
}

fn bdev_limits(bdev: *mut block_device) -> *mut queue_limits {
unsafe { core::ptr::addr_of_mut!((*bdev_get_queue(bdev)).limits) }
}

fn bdev_max_discard_sectors(bdev: *mut block_device) -> u32 {
unsafe { (*bdev_limits(bdev)).max_discard_sectors }
}

fn bdev_rot(bdev: *mut block_device) -> bool {
blk_queue_rot!(bdev_get_queue(bdev)) != 0
}

fn bdev_synchronous(bdev: *mut block_device) -> bool {
(unsafe { (*(*(*bdev).bd_disk).queue).limits.features }) & 64u32 != 0
}

fn bdev_has_integrity_csum(bdev: *mut block_device) -> bool {
let lim: *mut queue_limits = bdev_limits(bdev);
IS_ENABLED!() != 0 && (((unsafe { (*lim).integrity.csum_type }) as u8) as i32) != 0
}

fn bdev_stable_writes(bdev: *mut block_device) -> bool {
bdev_has_integrity_csum(bdev) || (unsafe { (*bdev_limits(bdev)).features }) & 32u32 != 0
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
fn is_vma_hugetlb_flags(flags: *const __SlateRecord_mm_types_h_871) -> bool {
vma_flags_test(flags, 22i32)
}

fn is_vm_hugetlb_page(vma: *const vm_area_struct) -> bool {
is_vma_hugetlb_flags(unsafe { core::ptr::addr_of!((*vma).__slate_anon_3.flags) })
}

fn filemap_write_and_wait(mapping: *mut address_space) -> i32 {
unsafe { filemap_write_and_wait_range(mapping, 0i64, 9223372036854775807i64) }
}

fn inode_drain_writes(inode: *mut inode) -> i32 {
unsafe { inode_dio_wait(inode) };
filemap_write_and_wait(unsafe { (*inode).i_mapping })
}

fn mapping_min_folio_order(mapping: *const address_space) -> u32 {
if IS_ENABLED!() == 0 {
return 0;
}
(((unsafe { (*mapping).flags }) & 2031616u64) >> 16i32) as u32
}

fn folio_file_page(folio: *mut folio, index: u64) -> *mut page {
folio_page!(folio, index & folio_nr_pages(folio as *const folio).wrapping_sub(1))
}

fn read_mapping_folio(mapping: *mut address_space, index: u64, file: *mut file) -> *mut folio {
unsafe { read_cache_folio(mapping, index, None, file) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct wait_page_queue {
    folio: *mut folio,
    bit_nr: i32,
    wait: wait_queue_entry,
}


fn folio_trylock(folio: *mut folio) -> bool {
likely!(!test_and_set_bit_lock(0i64, folio_flags(folio, 0u32) as *mut u64)) != 0
}

fn folio_lock(folio: *mut folio) {
might_sleep!();
if !folio_trylock(folio) {
unsafe { __folio_lock(folio) };
}
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
struct mempolicy {
    refcnt: __SlateRecord_types_h_190,
    mode: u16,
    flags: u16,
    nodes: __SlateRecord_nodemask_types_h_17,
    home_node: i32,
    w: __SlateRecord_mempolicy_h_54,
    rcu: callback_head,
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


#[inline(always)]
fn seq_puts(m: *mut seq_file, s: *const u8) {
if !unsafe { __slate_intrinsic_70b71f185ebb8c4a(*s) } {
unsafe { __seq_puts(m, s) };
} else {
if (unsafe { *s.wrapping_offset(0 as isize) }) != 0 && (unsafe { *s.wrapping_offset(1 as isize) }) == 0 {
unsafe { seq_putc(m, *s.wrapping_offset(0 as isize)) };
} else {
unsafe { seq_write(m, s as *const (), __builtin_strlen(s)) };
}
}
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


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_local64_h_26 {
    a: __SlateRecord_local_h_12,
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
    __slate_bits_0: __slate_bits::__SlateBits925U0,
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


fn userfaultfd_rwp(vma: *mut vm_area_struct) -> bool {
false
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
    __slate_bits_0: __slate_bits::__SlateBits630U0,
    fbatch: folio_batch,
    index: u64,
    saved_err: i32,
}


// enable discard for swap
// discard swap area at swapon-time
// discard page-clusters after use
#[repr(C)]
#[derive(Clone, Copy)]
union swap_header {
    magic: __SlateRecord_swap_h_126,
    info: __SlateRecord_swap_h_130,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct reclaim_state {
    reclaimed: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_extent {
    rb_node: rb_node,
    start_page: u64,
    nr_pages: u64,
    start_block: u64,
}


// Max bad pages in the new format..
// The first page in the swap file is the swap header, which is always marked
// bad to prevent it from being allocated as an entry. This also prevents the
// cluster to which it belongs being marked free. Therefore 0 is safe to use as
// a sentinel to indicate an entry is not valid.
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


#[unsafe(no_mangle)]
static mut nr_swap_pages: __SlateRecord_types_h_197 = unsafe { core::mem::zeroed() };

#[unsafe(no_mangle)]
static mut total_swap_pages: i64 = unsafe { core::mem::zeroed() };

fn vm_swap_full() -> bool {
atomic_long_read((unsafe { core::ptr::addr_of_mut!(nr_swap_pages) }) as *const __SlateRecord_types_h_197).wrapping_mul(2) < unsafe { total_swap_pages }
}

#[unsafe(no_mangle)]
extern "C" fn swp_swapcount(entry: __SlateRecord_mm_types_h_288) -> i32 {
let mut swp_tb: u64 = 0;
let mut count: i32 = 0;
let si: *mut swap_info_struct = get_swap_device(entry);
if si.is_null() {
return 0;
}
let ci: *mut swap_cluster_info = swap_cluster_lock(si, swp_offset(entry));
swp_tb = __swap_table_get(ci, swp_cluster_offset(entry));
count = swp_tb_get_count(swp_tb);
let mut __v0: i32;
let __UNIQUE_ID_x__897: i32 = 5;
let __UNIQUE_ID_y__898: i32 = 22i32;
loop {
if __types_ok!(__UNIQUE_ID_x__897, __UNIQUE_ID_y__898) == 0 {
unsafe { __compiletime_assert_899() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__897 < __UNIQUE_ID_y__898 { __UNIQUE_ID_x__897 } else { __UNIQUE_ID_y__898 };
if count == (1i32 << __v0.wrapping_sub((22i32 > 2) as i32)).wrapping_sub(1) {
count = (unsafe { *unsafe { (*ci).extend_table }.wrapping_offset(swp_cluster_offset(entry) as isize) }) as i32;
}
swap_cluster_unlock(ci);
put_swap_device(si);
if count < 0 { 0 } else { count }
}

#[unsafe(no_mangle)]
extern "C" fn swap_put_entries_direct(entry: __SlateRecord_mm_types_h_288, nr: i32) {
let mut __UNIQUE_ID_y__905: u64 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__904: u64 = unsafe { core::mem::zeroed() };
let mut __ret_warn_on: i32 = unsafe { core::mem::zeroed() };
let mut __ret_warn_on_1: i32 = unsafe { core::mem::zeroed() };
let mut si: core::mem::MaybeUninit<*mut swap_info_struct> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut swap_info_struct = core::ptr::addr_of_mut!(si) as *mut *mut swap_info_struct;
let mut cluster_end: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut u64 = core::ptr::addr_of_mut!(cluster_end) as *mut u64;
let mut offset: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut u64 = core::ptr::addr_of_mut!(offset) as *mut u64;
let mut end_offset: u64 = unsafe { core::mem::zeroed() };
let mut start_offset: u64 = unsafe { core::mem::zeroed() };
unsafe {
start_offset = swp_offset(entry);
end_offset = start_offset.wrapping_add(nr as u64);
let __slate_auto_bytes = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_9 = get_swap_device(entry);
__ret_warn_on_1 = (*__slate_slot_9).is_null() as i32;
if unlikely__2!(__ret_warn_on_1) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("900: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 900b - ., 3; .popsection\n# {0}", const 900 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_swap_put_entries_direct_16, sym __slate_asm_string_swap_put_entries_direct_17, const 2130 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("901: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 901b - ., 4; .popsection\n# {0}", const 901 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
let __v8: i64 = (__ret_warn_on_1 != 0) as i64;
if __v8 != 0 {
return;
} else {
__ret_warn_on = (end_offset > ((*(*__slate_slot_9)).max as u64)) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("902: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 902b - ., 3; .popsection\n# {0}", const 902 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_swap_put_entries_direct_14, sym __slate_asm_string_swap_put_entries_direct_15, const 2132 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("903: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 903b - ., 4; .popsection\n# {0}", const 903 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
let __v5: i64 = (__ret_warn_on != 0) as i64;
if __v5 == 0 {
// Put entries and reclaim cache in each cluster
*__slate_slot_11 = start_offset;
loop {
__UNIQUE_ID_x__904 = round_up__2!((*__slate_slot_11).wrapping_add(1), 256i32);
__UNIQUE_ID_y__905 = end_offset;
while false {
}
*__slate_slot_10 = if __UNIQUE_ID_x__904 < __UNIQUE_ID_y__905 { __UNIQUE_ID_x__904 } else { __UNIQUE_ID_y__905 };
swap_put_entries_cluster(*__slate_slot_9, *__slate_slot_11, (*__slate_slot_10).wrapping_sub(*__slate_slot_11) as i32, true);
*__slate_slot_11 = *__slate_slot_10;
if *__slate_slot_11 >= end_offset {
break;
}
}
}
put_swap_device(*__slate_slot_9);
}
}
}

fn put_swap_device(si: *mut swap_info_struct) {
percpu_ref_put(unsafe { core::ptr::addr_of_mut!((*si).users) });
}

fn mem_cgroup_try_charge_swap(folio: *mut folio) -> i32 {
0
}

fn mem_cgroup_uncharge_swap(id: u16, nr_pages: u32) {
}

fn mem_cgroup_swap_full(folio: *mut folio) -> bool {
vm_swap_full()
}

#[unsafe(no_mangle)]
static mut swapfile_maximum_size: u64 = unsafe { core::mem::zeroed() };

#[unsafe(no_mangle)]
static mut swap_migration_ad_supported: bool = unsafe { core::mem::zeroed() };

fn pte_swp_clear_flags(mut pte: __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_pgtable_64_types_h_21 {
if pte_swp_exclusive(pte) {
pte = pte_swp_clear_exclusive(pte);
}
if pte_swp_soft_dirty(pte) != 0 {
pte = pte_swp_clear_soft_dirty(pte);
}
if pte_swp_uffd(pte) != 0 {
pte = pte_swp_clear_uffd(pte);
}
return pte;
}

fn swp_entry(r#type: u64, offset: u64) -> __SlateRecord_mm_types_h_288 {
let mut ret: __SlateRecord_mm_types_h_288 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(ret) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
ret.val = r#type << 58i32 | offset & 288230376151711743u64;
return ret;
}

fn swp_type(entry: __SlateRecord_mm_types_h_288) -> u32 {
(entry.val >> 58i32) as u32
}

fn swp_offset(entry: __SlateRecord_mm_types_h_288) -> u64 {
entry.val & 288230376151711743u64
}

fn swp_entry_to_pte(entry: __SlateRecord_mm_types_h_288) -> __SlateRecord_pgtable_64_types_h_21 {
let mut __slate_compound_0: __SlateRecord_mm_types_h_288 = unsafe { core::mem::zeroed() };
let mut arch_entry: __SlateRecord_mm_types_h_288 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(arch_entry) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
arch_entry = unsafe { *{
__slate_compound_0 = __SlateRecord_mm_types_h_288 { val: !(swp_offset(entry) as u64) << 14i32 >> 5i32 | (swp_type(entry) as u64) << 59i32 };
core::ptr::addr_of_mut!(__slate_compound_0)
} };
return __swp_entry_to_pte!(arch_entry);
}

fn make_hwpoison_entry(page: *mut page) -> __SlateRecord_mm_types_h_288 {
return swp_entry(0u64, 0u64);
}

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
fn make_pte_marker_entry(marker: u64) -> __SlateRecord_mm_types_h_288 {
return swp_entry(31u64, marker);
}

fn make_poisoned_swp_entry() -> __SlateRecord_mm_types_h_288 {
return make_pte_marker_entry(BIT!(1i32));
}

fn vm_acct_memory(pages: i64) {
unsafe { percpu_counter_add_batch(core::ptr::addr_of_mut!(vm_committed_as), pages, vm_committed_as_batch) };
}

fn vm_unacct_memory(pages: i64) {
vm_acct_memory(pages.wrapping_neg());
}

// Optimisation macro.  It is equivalent to:
//      (x & bit1) ? bit2 : 0
// but this version is faster.
// ("bit1" and "bit2" must be single bits)
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


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_struct {
    users: i32,
    seq: seqlock,
    umask: i32,
    in_exec: i32,
    root: path,
    pwd: path,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xattr_handler {
    name: *const u8,
    prefix: *const u8,
    flags: i32,
    list: Option<unsafe extern "C" fn(*mut dentry) -> bool>,
    get: Option<unsafe extern "C" fn(*const xattr_handler, *mut dentry, *mut inode, *const u8, *mut (), u64) -> i32>,
    set: Option<unsafe extern "C" fn(*const xattr_handler, *mut mnt_idmap, *mut dentry, *mut inode, *const u8, *const (), u64, i32) -> i32>,
}


// double word (64-bit)
#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_insn {
    code: u8,
    __slate_bits_0: __slate_bits::__SlateBits954U0,
    off: i16,
    imm: i32,
}


// 1 << 13
// insn[0].src_reg:  BPF_PSEUDO_FUNC
// insn[0].imm:      insn offset to the func
// insn[1].imm:      0
// insn[0].off:      0
// insn[1].off:      0
// ldimm64 rewrite:  address of the function
// verifier type:    PTR_TO_FUNC.
#[repr(C)]
#[derive(Clone, Copy)]
union bpf_attr {
    __slate_anon_0: __SlateRecord_bpf_h_1528,
    __slate_anon_1: __SlateRecord_bpf_h_1581,
    batch: __SlateRecord_bpf_h_1591,
    __slate_anon_3: __SlateRecord_bpf_h_1608,
    __slate_anon_4: __SlateRecord_bpf_h_1674,
    __slate_anon_5: __SlateRecord_bpf_h_1687,
    test: __SlateRecord_bpf_h_1703,
    __slate_anon_7: __SlateRecord_bpf_h_1727,
    info: __SlateRecord_bpf_h_1740,
    query: __SlateRecord_bpf_h_1746,
    raw_tracepoint: __SlateRecord_bpf_h_1769,
    __slate_anon_11: __SlateRecord_bpf_h_1776,
    task_fd_query: __SlateRecord_bpf_h_1794,
    link_create: __SlateRecord_bpf_h_1810,
    link_update: __SlateRecord_bpf_h_1895,
    link_detach: __SlateRecord_bpf_h_1916,
    enable_stats: __SlateRecord_bpf_h_1920,
    iter_create: __SlateRecord_bpf_h_1924,
    prog_bind_map: __SlateRecord_bpf_h_1929,
    token_create: __SlateRecord_bpf_h_1935,
    prog_stream_read: __SlateRecord_bpf_h_1940,
    prog_assoc_struct_ops: __SlateRecord_bpf_h_1947,
}


// The description below is an attempt at providing documentation to eBPF
// developers about the multiple available eBPF helper functions. It can be
// parsed and used to produce a manual page. The workflow is the following,
// and requires the rst2man utility:
//
//     $ ./scripts/bpf_doc.py  *             --filename include/uapi/linux/bpf.h > /tmp/bpf-helpers.rst
//     $ rst2man /tmp/bpf-helpers.rst > /tmp/bpf-helpers.7
//     $ man /tmp/bpf-helpers.7
//
// Note that in order to produce this external documentation, some RST
// formatting is used in the descriptions to get "bold" and "italics" in
// manual pages. Also note that the few trailing white spaces are
// intentional, removing them would break paragraphs for rst2man.
//
// Start of BPF helper function descriptions:
//
// void *bpf_map_lookup_elem(struct bpf_map *map, const void *key)
// 	Description
// 		Perform a lookup in *map* for an entry associated to *key*.
// 	Return
// 		Map value associated to *key*, or **NULL** if no entry was
// 		found.
//
// long bpf_map_update_elem(struct bpf_map *map, const void *key, const void *value, u64 flags)
// 	Description
// 		Add or update the value of the entry associated to *key* in
// 		*map* with *value*. *flags* is one of:
//
// 		**BPF_NOEXIST**
// 			The entry for *key* must not exist in the map.
// 		**BPF_EXIST**
// 			The entry for *key* must already exist in the map.
// 		**BPF_ANY**
// 			No condition on the existence of the entry for *key*.
//
// 		Flag value **BPF_NOEXIST** cannot be used for maps of types
// 		**BPF_MAP_TYPE_ARRAY** or **BPF_MAP_TYPE_PERCPU_ARRAY**  (all
// 		elements always exist), the helper would return an error.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_map_delete_elem(struct bpf_map *map, const void *key)
// 	Description
// 		Delete entry with *key* from *map*.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_probe_read(void *dst, u32 size, const void *unsafe_ptr)
// 	Description
// 		For tracing programs, safely attempt to read *size* bytes from
// 		kernel space address *unsafe_ptr* and store the data in *dst*.
//
// 		Generally, use **bpf_probe_read_user**\ () or
// 		**bpf_probe_read_kernel**\ () instead.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// u64 bpf_ktime_get_ns(void)
// 	Description
// 		Return the time elapsed since system boot, in nanoseconds.
// 		Does not include time the system was suspended.
// 		See: **clock_gettime**\ (**CLOCK_MONOTONIC**)
// 	Return
// 		Current *ktime*.
//
// long bpf_trace_printk(const char *fmt, u32 fmt_size, ...)
// 	Description
// 		This helper is a "printk()-like" facility for debugging. It
// 		prints a message defined by format *fmt* (of size *fmt_size*)
// 		to file *\/sys/kernel/tracing/trace* from TraceFS, if
// 		available. It can take up to three additional **u64**
// 		arguments (as an eBPF helpers, the total number of arguments is
// 		limited to five).
//
// 		Each time the helper is called, it appends a line to the trace.
// 		Lines are discarded while *\/sys/kernel/tracing/trace* is
// 		open, use *\/sys/kernel/tracing/trace_pipe* to avoid this.
// 		The format of the trace is customizable, and the exact output
// 		one will get depends on the options set in
// 		*\/sys/kernel/tracing/trace_options* (see also the
// 		*README* file under the same directory). However, it usually
// 		defaults to something like:
//
// 		::
//
// 			telnet-470   [001] .N.. 419421.045894: 0x00000001: <formatted msg>
//
// 		In the above:
//
// 			* ``telnet`` is the name of the current task.
// 			* ``470`` is the PID of the current task.
// 			* ``001`` is the CPU number on which the task is
// 			  running.
// 			* In ``.N..``, each character refers to a set of
// 			  options (whether irqs are enabled, scheduling
// 			  options, whether hard/softirqs are running, level of
// 			  preempt_disabled respectively). **N** means that
// 			  **TIF_NEED_RESCHED** and **PREEMPT_NEED_RESCHED**
// 			  are set.
// 			* ``419421.045894`` is a timestamp.
// 			* ``0x00000001`` is a fake value used by BPF for the
// 			  instruction pointer register.
// 			* ``<formatted msg>`` is the message formatted with
// 			  *fmt*.
//
// 		The conversion specifiers supported by *fmt* are similar, but
// 		more limited than for printk(). They are **%d**, **%i**,
// 		**%u**, **%x**, **%ld**, **%li**, **%lu**, **%lx**, **%lld**,
// 		**%lli**, **%llu**, **%llx**, **%p**, **%s**. No modifier (size
// 		of field, padding with zeroes, etc.) is available, and the
// 		helper will return **-EINVAL** (but print nothing) if it
// 		encounters an unknown specifier.
//
// 		Also, note that **bpf_trace_printk**\ () is slow, and should
// 		only be used for debugging purposes. For this reason, a notice
// 		block (spanning several lines) is printed to kernel logs and
// 		states that the helper should not be used "for production use"
// 		the first time this helper is used (or more precisely, when
// 		**trace_printk**\ () buffers are allocated). For passing values
// 		to user space, perf events should be preferred.
// 	Return
// 		The number of bytes written to the buffer, or a negative error
// 		in case of failure.
//
// u32 bpf_get_prandom_u32(void)
// 	Description
// 		Get a pseudo-random number.
//
// 		From a security point of view, this helper uses its own
// 		pseudo-random internal state, and cannot be used to infer the
// 		seed of other random functions in the kernel. However, it is
// 		essential to note that the generator used by the helper is not
// 		cryptographically secure.
// 	Return
// 		A random 32-bit unsigned value.
//
// u32 bpf_get_smp_processor_id(void)
// 	Description
// 		Get the SMP (symmetric multiprocessing) processor id. Note that
// 		all programs run with migration disabled, which means that the
// 		SMP processor id is stable during all the execution of the
// 		program.
// 	Return
// 		The SMP id of the processor running the program.
// 	Attributes
// 		__bpf_fastcall
//
// long bpf_skb_store_bytes(struct sk_buff *skb, u32 offset, const void *from, u32 len, u64 flags)
// 	Description
// 		Store *len* bytes from address *from* into the packet
// 		associated to *skb*, at *offset*. The *flags* are a combination
// 		of the following values:
//
// 		**BPF_F_RECOMPUTE_CSUM**
// 			Automatically update *skb*\ **->csum** after storing the
// 			bytes.
// 		**BPF_F_INVALIDATE_HASH**
// 			Set *skb*\ **->hash**, *skb*\ **->swhash** and *skb* * 			**->l4hash** to 0.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_l3_csum_replace(struct sk_buff *skb, u32 offset, u64 from, u64 to, u64 size)
// 	Description
// 		Recompute the layer 3 (e.g. IP) checksum for the packet
// 		associated to *skb*. Computation is incremental, so the helper
// 		must know the former value of the header field that was
// 		modified (*from*), the new value of this field (*to*), and the
// 		number of bytes (2 or 4) for this field, stored in *size*.
// 		Alternatively, it is possible to store the difference between
// 		the previous and the new values of the header field in *to*, by
// 		setting *from* and *size* to 0. For both methods, *offset*
// 		indicates the location of the IP checksum within the packet.
//
// 		This helper works in combination with **bpf_csum_diff**\ (),
// 		which does not update the checksum in-place, but offers more
// 		flexibility and can handle sizes larger than 2 or 4 for the
// 		checksum to update.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_l4_csum_replace(struct sk_buff *skb, u32 offset, u64 from, u64 to, u64 flags)
// 	Description
// 		Recompute the layer 4 (e.g. TCP, UDP or ICMP) checksum for the
// 		packet associated to *skb*. Computation is incremental, so the
// 		helper must know the former value of the header field that was
// 		modified (*from*), the new value of this field (*to*), and the
// 		number of bytes (2 or 4) for this field, stored on the lowest
// 		four bits of *flags*. Alternatively, it is possible to store
// 		the difference between the previous and the new values of the
// 		header field in *to*, by setting *from* and the four lowest
// 		bits of *flags* to 0. For both methods, *offset* indicates the
// 		location of the IP checksum within the packet. In addition to
// 		the size of the field, *flags* can be added (bitwise OR) actual
// 		flags. With **BPF_F_MARK_MANGLED_0**, a null checksum is left
// 		untouched (unless **BPF_F_MARK_ENFORCE** is added as well), and
// 		for updates resulting in a null checksum the value is set to
// 		**CSUM_MANGLED_0** instead. Flag **BPF_F_PSEUDO_HDR** indicates
// 		that the modified header field is part of the pseudo-header.
// 		Flag **BPF_F_IPV6** should be set for IPv6 packets.
//
// 		This helper works in combination with **bpf_csum_diff**\ (),
// 		which does not update the checksum in-place, but offers more
// 		flexibility and can handle sizes larger than 2 or 4 for the
// 		checksum to update.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_tail_call(void *ctx, struct bpf_map *prog_array_map, u32 index)
// 	Description
// 		This special helper is used to trigger a "tail call", or in
// 		other words, to jump into another eBPF program. The same stack
// 		frame is used (but values on stack and in registers for the
// 		caller are not accessible to the callee). This mechanism allows
// 		for program chaining, either for raising the maximum number of
// 		available eBPF instructions, or to execute given programs in
// 		conditional blocks. For security reasons, there is an upper
// 		limit to the number of successive tail calls that can be
// 		performed.
//
// 		Upon call of this helper, the program attempts to jump into a
// 		program referenced at index *index* in *prog_array_map*, a
// 		special map of type **BPF_MAP_TYPE_PROG_ARRAY**, and passes
// 		*ctx*, a pointer to the context.
//
// 		If the call succeeds, the kernel immediately runs the first
// 		instruction of the new program. This is not a function call,
// 		and it never returns to the previous program. If the call
// 		fails, then the helper has no effect, and the caller continues
// 		to run its subsequent instructions. A call can fail if the
// 		destination program for the jump does not exist (i.e. *index*
// 		is superior to the number of entries in *prog_array_map*), or
// 		if the maximum number of tail calls has been reached for this
// 		chain of programs. This limit is defined in the kernel by the
// 		macro **MAX_TAIL_CALL_CNT** (not accessible to user space),
// 		which is currently set to 33.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_clone_redirect(struct sk_buff *skb, u32 ifindex, u64 flags)
// 	Description
// 		Clone and redirect the packet associated to *skb* to another
// 		net device of index *ifindex*. Both ingress and egress
// 		interfaces can be used for redirection. The **BPF_F_INGRESS**
// 		value in *flags* is used to make the distinction (ingress path
// 		is selected if the flag is present, egress path otherwise).
// 		This is the only flag supported for now.
//
// 		In comparison with **bpf_redirect**\ () helper,
// 		**bpf_clone_redirect**\ () has the associated cost of
// 		duplicating the packet buffer, but this can be executed out of
// 		the eBPF program. Conversely, **bpf_redirect**\ () is more
// 		efficient, but it is handled through an action code where the
// 		redirection happens only after the eBPF program has returned.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure. Positive
// 		error indicates a potential drop or congestion in the target
// 		device. The particular positive error codes are not defined.
//
// u64 bpf_get_current_pid_tgid(void)
// 	Description
// 		Get the current pid and tgid.
// 	Return
// 		A 64-bit integer containing the current tgid and pid, and
// 		created as such:
// 		*current_task*\ **->tgid << 32 \|**
// 		*current_task*\ **->pid**.
//
// u64 bpf_get_current_uid_gid(void)
// 	Description
// 		Get the current uid and gid.
// 	Return
// 		A 64-bit integer containing the current GID and UID, and
// 		created as such: *current_gid* **<< 32 \|** *current_uid*.
//
// long bpf_get_current_comm(void *buf, u32 size_of_buf)
// 	Description
// 		Copy the **comm** attribute of the current task into *buf* of
// 		*size_of_buf*. The **comm** attribute contains the name of
// 		the executable (excluding the path) for the current task. The
// 		*size_of_buf* must be strictly positive. On success, the
// 		helper makes sure that the *buf* is NUL-terminated. On failure,
// 		it is filled with zeroes.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// u32 bpf_get_cgroup_classid(struct sk_buff *skb)
// 	Description
// 		Retrieve the classid for the current task, i.e. for the net_cls
// 		cgroup to which *skb* belongs.
//
// 		This helper can be used on TC egress path, but not on ingress.
//
// 		The net_cls cgroup provides an interface to tag network packets
// 		based on a user-provided identifier for all traffic coming from
// 		the tasks belonging to the related cgroup. See also the related
// 		kernel documentation, available from the Linux sources in file
// 		*Documentation/admin-guide/cgroup-v1/net_cls.rst*.
//
// 		The Linux kernel has two versions for cgroups: there are
// 		cgroups v1 and cgroups v2. Both are available to users, who can
// 		use a mixture of them, but note that the net_cls cgroup is for
// 		cgroup v1 only. This makes it incompatible with BPF programs
// 		run on cgroups, which is a cgroup-v2-only feature (a socket can
// 		only hold data for one version of cgroups at a time).
//
// 		This helper is only available is the kernel was compiled with
// 		the **CONFIG_CGROUP_NET_CLASSID** configuration option set to
// 		"**y**" or to "**m**".
// 	Return
// 		The classid, or 0 for the default unconfigured classid.
//
// long bpf_skb_vlan_push(struct sk_buff *skb, __be16 vlan_proto, u16 vlan_tci)
// 	Description
// 		Push a *vlan_tci* (VLAN tag control information) of protocol
// 		*vlan_proto* to the packet associated to *skb*, then update
// 		the checksum. Note that if *vlan_proto* is different from
// 		**ETH_P_8021Q** and **ETH_P_8021AD**, it is considered to
// 		be **ETH_P_8021Q**.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_vlan_pop(struct sk_buff *skb)
// 	Description
// 		Pop a VLAN header from the packet associated to *skb*.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_get_tunnel_key(struct sk_buff *skb, struct bpf_tunnel_key *key, u32 size, u64 flags)
// 	Description
// 		Get tunnel metadata. This helper takes a pointer *key* to an
// 		empty **struct bpf_tunnel_key** of **size**, that will be
// 		filled with tunnel metadata for the packet associated to *skb*.
// 		The *flags* can be set to **BPF_F_TUNINFO_IPV6**, which
// 		indicates that the tunnel is based on IPv6 protocol instead of
// 		IPv4.
//
// 		The **struct bpf_tunnel_key** is an object that generalizes the
// 		principal parameters used by various tunneling protocols into a
// 		single struct. This way, it can be used to easily make a
// 		decision based on the contents of the encapsulation header,
// 		"summarized" in this struct. In particular, it holds the IP
// 		address of the remote end (IPv4 or IPv6, depending on the case)
// 		in *key*\ **->remote_ipv4** or *key*\ **->remote_ipv6**. Also,
// 		this struct exposes the *key*\ **->tunnel_id**, which is
// 		generally mapped to a VNI (Virtual Network Identifier), making
// 		it programmable together with the **bpf_skb_set_tunnel_key** * 		() helper.
//
// 		Let's imagine that the following code is part of a program
// 		attached to the TC ingress interface, on one end of a GRE
// 		tunnel, and is supposed to filter out all messages coming from
// 		remote ends with IPv4 address other than 10.0.0.1:
//
// 		::
//
// 			int ret;
// 			struct bpf_tunnel_key key = {};
//
// 			ret = bpf_skb_get_tunnel_key(skb, &key, sizeof(key), 0);
// 			if (ret < 0)
// 				return TC_ACT_SHOT;	// drop packet
//
// 			if (key.remote_ipv4 != 0x0a000001)
// 				return TC_ACT_SHOT;	// drop packet
//
// 			return TC_ACT_OK;		// accept packet
//
// 		This interface can also be used with all encapsulation devices
// 		that can operate in "collect metadata" mode: instead of having
// 		one network device per specific configuration, the "collect
// 		metadata" mode only requires a single device where the
// 		configuration can be extracted from this helper.
//
// 		This can be used together with various tunnels such as VXLan,
// 		Geneve, GRE or IP in IP (IPIP).
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_set_tunnel_key(struct sk_buff *skb, struct bpf_tunnel_key *key, u32 size, u64 flags)
// 	Description
// 		Populate tunnel metadata for packet associated to *skb.* The
// 		tunnel metadata is set to the contents of *key*, of *size*. The
// 		*flags* can be set to a combination of the following values:
//
// 		**BPF_F_TUNINFO_IPV6**
// 			Indicate that the tunnel is based on IPv6 protocol
// 			instead of IPv4.
// 		**BPF_F_ZERO_CSUM_TX**
// 			For IPv4 packets, add a flag to tunnel metadata
// 			indicating that checksum computation should be skipped
// 			and checksum set to zeroes.
// 		**BPF_F_DONT_FRAGMENT**
// 			Add a flag to tunnel metadata indicating that the
// 			packet should not be fragmented.
// 		**BPF_F_SEQ_NUMBER**
// 			Add a flag to tunnel metadata indicating that a
// 			sequence number should be added to tunnel header before
// 			sending the packet. This flag was added for GRE
// 			encapsulation, but might be used with other protocols
// 			as well in the future.
// 		**BPF_F_NO_TUNNEL_KEY**
// 			Add a flag to tunnel metadata indicating that no tunnel
// 			key should be set in the resulting tunnel header.
//
// 		Here is a typical usage on the transmit path:
//
// 		::
//
// 			struct bpf_tunnel_key key;
// 			     populate key ...
// 			bpf_skb_set_tunnel_key(skb, &key, sizeof(key), 0);
// 			bpf_clone_redirect(skb, vxlan_dev_ifindex, 0);
//
// 		See also the description of the **bpf_skb_get_tunnel_key**\ ()
// 		helper for additional information.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// u64 bpf_perf_event_read(struct bpf_map *map, u64 flags)
// 	Description
// 		Read the value of a perf event counter. This helper relies on a
// 		*map* of type **BPF_MAP_TYPE_PERF_EVENT_ARRAY**. The nature of
// 		the perf event counter is selected when *map* is updated with
// 		perf event file descriptors. The *map* is an array whose size
// 		is the number of available CPUs, and each cell contains a value
// 		relative to one CPU. The value to retrieve is indicated by
// 		*flags*, that contains the index of the CPU to look up, masked
// 		with **BPF_F_INDEX_MASK**. Alternatively, *flags* can be set to
// 		**BPF_F_CURRENT_CPU** to indicate that the value for the
// 		current CPU should be retrieved.
//
// 		Note that before Linux 4.13, only hardware perf event can be
// 		retrieved.
//
// 		Also, be aware that the newer helper
// 		**bpf_perf_event_read_value**\ () is recommended over
// 		**bpf_perf_event_read**\ () in general. The latter has some ABI
// 		quirks where error and counter value are used as a return code
// 		(which is wrong to do since ranges may overlap). This issue is
// 		fixed with **bpf_perf_event_read_value**\ (), which at the same
// 		time provides more features over the **bpf_perf_event_read** * 		() interface. Please refer to the description of
// 		**bpf_perf_event_read_value**\ () for details.
// 	Return
// 		The value of the perf event counter read from the map, or a
// 		negative error code in case of failure.
//
// long bpf_redirect(u32 ifindex, u64 flags)
// 	Description
// 		Redirect the packet to another net device of index *ifindex*.
// 		This helper is somewhat similar to **bpf_clone_redirect** * 		(), except that the packet is not cloned, which provides
// 		increased performance.
//
// 		Except for XDP, both ingress and egress interfaces can be used
// 		for redirection. The **BPF_F_INGRESS** value in *flags* is used
// 		to make the distinction (ingress path is selected if the flag
// 		is present, egress path otherwise). Currently, XDP only
// 		supports redirection to the egress interface, and accepts no
// 		flag at all.
//
// 		The same effect can also be attained with the more generic
// 		**bpf_redirect_map**\ (), which uses a BPF map to store the
// 		redirect target instead of providing it directly to the helper.
// 	Return
// 		For XDP, the helper returns **XDP_REDIRECT** on success or
// 		**XDP_ABORTED** on error. For other program types, the values
// 		are **TC_ACT_REDIRECT** on success or **TC_ACT_SHOT** on
// 		error.
//
// u32 bpf_get_route_realm(struct sk_buff *skb)
// 	Description
// 		Retrieve the realm or the route, that is to say the
// 		**tclassid** field of the destination for the *skb*. The
// 		identifier retrieved is a user-provided tag, similar to the
// 		one used with the net_cls cgroup (see description for
// 		**bpf_get_cgroup_classid**\ () helper), but here this tag is
// 		held by a route (a destination entry), not by a task.
//
// 		Retrieving this identifier works with the clsact TC egress hook
// 		(see also **tc-bpf(8)**), or alternatively on conventional
// 		classful egress qdiscs, but not on TC ingress path. In case of
// 		clsact TC egress hook, this has the advantage that, internally,
// 		the destination entry has not been dropped yet in the transmit
// 		path. Therefore, the destination entry does not need to be
// 		artificially held via **netif_keep_dst**\ () for a classful
// 		qdisc until the *skb* is freed.
//
// 		This helper is available only if the kernel was compiled with
// 		**CONFIG_IP_ROUTE_CLASSID** configuration option.
// 	Return
// 		The realm of the route for the packet associated to *skb*, or 0
// 		if none was found.
//
// long bpf_perf_event_output(void *ctx, struct bpf_map *map, u64 flags, void *data, u64 size)
// 	Description
// 		Write raw *data* blob into a special BPF perf event held by
// 		*map* of type **BPF_MAP_TYPE_PERF_EVENT_ARRAY**. This perf
// 		event must have the following attributes: **PERF_SAMPLE_RAW**
// 		as **sample_type**, **PERF_TYPE_SOFTWARE** as **type**, and
// 		**PERF_COUNT_SW_BPF_OUTPUT** as **config**.
//
// 		The *flags* are used to indicate the index in *map* for which
// 		the value must be put, masked with **BPF_F_INDEX_MASK**.
// 		Alternatively, *flags* can be set to **BPF_F_CURRENT_CPU**
// 		to indicate that the index of the current CPU core should be
// 		used.
//
// 		The value to write, of *size*, is passed through eBPF stack and
// 		pointed by *data*.
//
// 		The context of the program *ctx* needs also be passed to the
// 		helper.
//
// 		On user space, a program willing to read the values needs to
// 		call **perf_event_open**\ () on the perf event (either for
// 		one or for all CPUs) and to store the file descriptor into the
// 		*map*. This must be done before the eBPF program can send data
// 		into it. An example is available in file
// 		*samples/bpf/trace_output_user.c* in the Linux kernel source
// 		tree (the eBPF program counterpart is in
// 		*samples/bpf/trace_output.bpf.c*).
//
// 		**bpf_perf_event_output**\ () achieves better performance
// 		than **bpf_trace_printk**\ () for sharing data with user
// 		space, and is much better suitable for streaming data from eBPF
// 		programs.
//
// 		Note that this helper is not restricted to tracing use cases
// 		and can be used with programs attached to TC or XDP as well,
// 		where it allows for passing data to user space listeners. Data
// 		can be:
//
// 		* Only custom structs,
// 		* Only the packet payload, or
// 		* A combination of both.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_load_bytes(const void *skb, u32 offset, void *to, u32 len)
// 	Description
// 		This helper was provided as an easy way to load data from a
// 		packet. It can be used to load *len* bytes from *offset* from
// 		the packet associated to *skb*, into the buffer pointed by
// 		*to*.
//
// 		Since Linux 4.7, usage of this helper has mostly been replaced
// 		by "direct packet access", enabling packet data to be
// 		manipulated with *skb*\ **->data** and *skb*\ **->data_end**
// 		pointing respectively to the first byte of packet data and to
// 		the byte after the last byte of packet data. However, it
// 		remains useful if one wishes to read large quantities of data
// 		at once from a packet into the eBPF stack.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_get_stackid(void *ctx, struct bpf_map *map, u64 flags)
// 	Description
// 		Walk a user or a kernel stack and return its id. To achieve
// 		this, the helper needs *ctx*, which is a pointer to the context
// 		on which the tracing program is executed, and a pointer to a
// 		*map* of type **BPF_MAP_TYPE_STACK_TRACE**.
//
// 		The last argument, *flags*, holds the number of stack frames to
// 		skip (from 0 to 255), masked with
// 		**BPF_F_SKIP_FIELD_MASK**. The next bits can be used to set
// 		a combination of the following flags:
//
// 		**BPF_F_USER_STACK**
// 			Collect a user space stack instead of a kernel stack.
// 		**BPF_F_FAST_STACK_CMP**
// 			Compare stacks by hash only.
// 		**BPF_F_REUSE_STACKID**
// 			If two different stacks hash into the same *stackid*,
// 			discard the old one.
//
// 		The stack id retrieved is a 32 bit long integer handle which
// 		can be further combined with other data (including other stack
// 		ids) and used as a key into maps. This can be useful for
// 		generating a variety of graphs (such as flame graphs or off-cpu
// 		graphs).
//
// 		For walking a stack, this helper is an improvement over
// 		**bpf_probe_read**\ (), which can be used with unrolled loops
// 		but is not efficient and consumes a lot of eBPF instructions.
// 		Instead, **bpf_get_stackid**\ () can collect up to
// 		**PERF_MAX_STACK_DEPTH** both kernel and user frames. Note that
// 		this limit can be controlled with the **sysctl** program, and
// 		that it should be manually increased in order to profile long
// 		user stacks (such as stacks for Java programs). To do so, use:
//
// 		::
//
// 			# sysctl kernel.perf_event_max_stack=<new value>
// 	Return
// 		The positive or null stack id on success, or a negative error
// 		in case of failure.
//
// s64 bpf_csum_diff(__be32 *from, u32 from_size, __be32 *to, u32 to_size, __wsum seed)
// 	Description
// 		Compute a checksum difference, from the raw buffer pointed by
// 		*from*, of length *from_size* (that must be a multiple of 4),
// 		towards the raw buffer pointed by *to*, of size *to_size*
// 		(same remark). An optional *seed* can be added to the value
// 		(this can be cascaded, the seed may come from a previous call
// 		to the helper).
//
// 		This is flexible enough to be used in several ways:
//
// 		* With *from_size* == 0, *to_size* > 0 and *seed* set to
// 		  checksum, it can be used when pushing new data.
// 		* With *from_size* > 0, *to_size* == 0 and *seed* set to
// 		  checksum, it can be used when removing data from a packet.
// 		* With *from_size* > 0, *to_size* > 0 and *seed* set to 0, it
// 		  can be used to compute a diff. Note that *from_size* and
// 		  *to_size* do not need to be equal.
//
// 		This helper can be used in combination with
// 		**bpf_l3_csum_replace**\ () and **bpf_l4_csum_replace**\ (), to
// 		which one can feed in the difference computed with
// 		**bpf_csum_diff**\ ().
// 	Return
// 		The checksum result, or a negative error code in case of
// 		failure.
//
// long bpf_skb_get_tunnel_opt(struct sk_buff *skb, void *opt, u32 size)
// 	Description
// 		Retrieve tunnel options metadata for the packet associated to
// 		*skb*, and store the raw tunnel option data to the buffer *opt*
// 		of *size*.
//
// 		This helper can be used with encapsulation devices that can
// 		operate in "collect metadata" mode (please refer to the related
// 		note in the description of **bpf_skb_get_tunnel_key**\ () for
// 		more details). A particular example where this can be used is
// 		in combination with the Geneve encapsulation protocol, where it
// 		allows for pushing (with **bpf_skb_get_tunnel_opt**\ () helper)
// 		and retrieving arbitrary TLVs (Type-Length-Value headers) from
// 		the eBPF program. This allows for full customization of these
// 		headers.
// 	Return
// 		The size of the option data retrieved.
//
// long bpf_skb_set_tunnel_opt(struct sk_buff *skb, void *opt, u32 size)
// 	Description
// 		Set tunnel options metadata for the packet associated to *skb*
// 		to the option data contained in the raw buffer *opt* of *size*.
//
// 		See also the description of the **bpf_skb_get_tunnel_opt**\ ()
// 		helper for additional information.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_change_proto(struct sk_buff *skb, __be16 proto, u64 flags)
// 	Description
// 		Change the protocol of the *skb* to *proto*. Currently
// 		supported are transition from IPv4 to IPv6, and from IPv6 to
// 		IPv4. The helper takes care of the groundwork for the
// 		transition, including resizing the socket buffer. The eBPF
// 		program is expected to fill the new headers, if any, via
// 		**skb_store_bytes**\ () and to recompute the checksums with
// 		**bpf_l3_csum_replace**\ () and **bpf_l4_csum_replace** * 		(). The main case for this helper is to perform NAT64
// 		operations out of an eBPF program.
//
// 		Internally, the GSO type is marked as dodgy so that headers are
// 		checked and segments are recalculated by the GSO/GRO engine.
// 		The size for GSO target is adapted as well.
//
// 		All values for *flags* are reserved for future usage, and must
// 		be left at zero.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_change_type(struct sk_buff *skb, u32 type)
// 	Description
// 		Change the packet type for the packet associated to *skb*. This
// 		comes down to setting *skb*\ **->pkt_type** to *type*, except
// 		the eBPF program does not have a write access to *skb* * 		**->pkt_type** beside this helper. Using a helper here allows
// 		for graceful handling of errors.
//
// 		The major use case is to change incoming *skb*s to
// 		**PACKET_HOST** in a programmatic way instead of having to
// 		recirculate via **redirect**\ (..., **BPF_F_INGRESS**), for
// 		example.
//
// 		Note that *type* only allows certain values. At this time, they
// 		are:
//
// 		**PACKET_HOST**
// 			Packet is for us.
// 		**PACKET_BROADCAST**
// 			Send packet to all.
// 		**PACKET_MULTICAST**
// 			Send packet to group.
// 		**PACKET_OTHERHOST**
// 			Send packet to someone else.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_under_cgroup(struct sk_buff *skb, struct bpf_map *map, u32 index)
// 	Description
// 		Check whether *skb* is a descendant of the cgroup2 held by
// 		*map* of type **BPF_MAP_TYPE_CGROUP_ARRAY**, at *index*.
// 	Return
// 		The return value depends on the result of the test, and can be:
//
// 		* 0, if the *skb* failed the cgroup2 descendant test.
// 		* 1, if the *skb* succeeded the cgroup2 descendant test.
// 		* A negative error code, if an error occurred.
//
// u32 bpf_get_hash_recalc(struct sk_buff *skb)
// 	Description
// 		Retrieve the hash of the packet, *skb*\ **->hash**. If it is
// 		not set, in particular if the hash was cleared due to mangling,
// 		recompute this hash. Later accesses to the hash can be done
// 		directly with *skb*\ **->hash**.
//
// 		Calling **bpf_set_hash_invalid**\ (), changing a packet
// 		prototype with **bpf_skb_change_proto**\ (), or calling
// 		**bpf_skb_store_bytes**\ () with the
// 		**BPF_F_INVALIDATE_HASH** are actions susceptible to clear
// 		the hash and to trigger a new computation for the next call to
// 		**bpf_get_hash_recalc**\ ().
// 	Return
// 		The 32-bit hash.
//
// u64 bpf_get_current_task(void)
// 	Description
// 		Get the current task.
// 	Return
// 		A pointer to the current task struct.
//
// long bpf_probe_write_user(void *dst, const void *src, u32 len)
// 	Description
// 		Attempt in a safe way to write *len* bytes from the buffer
// 		*src* to *dst* in memory. It only works for threads that are in
// 		user context, and *dst* must be a valid user space address.
//
// 		This helper should not be used to implement any kind of
// 		security mechanism because of TOC-TOU attacks, but rather to
// 		debug, divert, and manipulate execution of semi-cooperative
// 		processes.
//
// 		Keep in mind that this feature is meant for experiments, and it
// 		has a risk of crashing the system and running programs.
// 		Therefore, when an eBPF program using this helper is attached,
// 		a warning including PID and process name is printed to kernel
// 		logs.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_current_task_under_cgroup(struct bpf_map *map, u32 index)
// 	Description
// 		Check whether the probe is being run is the context of a given
// 		subset of the cgroup2 hierarchy. The cgroup2 to test is held by
// 		*map* of type **BPF_MAP_TYPE_CGROUP_ARRAY**, at *index*.
// 	Return
// 		The return value depends on the result of the test, and can be:
//
// 		* 1, if current task belongs to the cgroup2.
// 		* 0, if current task does not belong to the cgroup2.
// 		* A negative error code, if an error occurred.
//
// long bpf_skb_change_tail(struct sk_buff *skb, u32 len, u64 flags)
// 	Description
// 		Resize (trim or grow) the packet associated to *skb* to the
// 		new *len*. The *flags* are reserved for future usage, and must
// 		be left at zero.
//
// 		The basic idea is that the helper performs the needed work to
// 		change the size of the packet, then the eBPF program rewrites
// 		the rest via helpers like **bpf_skb_store_bytes**\ (),
// 		**bpf_l3_csum_replace**\ (), **bpf_l3_csum_replace**\ ()
// 		and others. This helper is a slow path utility intended for
// 		replies with control messages. And because it is targeted for
// 		slow path, the helper itself can afford to be slow: it
// 		implicitly linearizes, unclones and drops offloads from the
// 		*skb*.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_pull_data(struct sk_buff *skb, u32 len)
// 	Description
// 		Pull in non-linear data in case the *skb* is non-linear and not
// 		all of *len* are part of the linear section. Make *len* bytes
// 		from *skb* readable and writable. If a zero value is passed for
// 		*len*, then all bytes in the linear part of *skb* will be made
// 		readable and writable.
//
// 		This helper is only needed for reading and writing with direct
// 		packet access.
//
// 		For direct packet access, testing that offsets to access
// 		are within packet boundaries (test on *skb*\ **->data_end**) is
// 		susceptible to fail if offsets are invalid, or if the requested
// 		data is in non-linear parts of the *skb*. On failure the
// 		program can just bail out, or in the case of a non-linear
// 		buffer, use a helper to make the data available. The
// 		**bpf_skb_load_bytes**\ () helper is a first solution to access
// 		the data. Another one consists in using **bpf_skb_pull_data**
// 		to pull in once the non-linear parts, then retesting and
// 		eventually access the data.
//
// 		At the same time, this also makes sure the *skb* is uncloned,
// 		which is a necessary condition for direct write. As this needs
// 		to be an invariant for the write part only, the verifier
// 		detects writes and adds a prologue that is calling
// 		**bpf_skb_pull_data()** to effectively unclone the *skb* from
// 		the very beginning in case it is indeed cloned.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// s64 bpf_csum_update(struct sk_buff *skb, __wsum csum)
// 	Description
// 		Add the checksum *csum* into *skb*\ **->csum** in case the
// 		driver has supplied a checksum for the entire packet into that
// 		field. Return an error otherwise. This helper is intended to be
// 		used in combination with **bpf_csum_diff**\ (), in particular
// 		when the checksum needs to be updated after data has been
// 		written into the packet through direct packet access.
// 	Return
// 		The checksum on success, or a negative error code in case of
// 		failure.
//
// void bpf_set_hash_invalid(struct sk_buff *skb)
// 	Description
// 		Invalidate the current *skb*\ **->hash**. It can be used after
// 		mangling on headers through direct packet access, in order to
// 		indicate that the hash is outdated and to trigger a
// 		recalculation the next time the kernel tries to access this
// 		hash or when the **bpf_get_hash_recalc**\ () helper is called.
// 	Return
// 		void.
//
// long bpf_get_numa_node_id(void)
// 	Description
// 		Return the id of the current NUMA node. The primary use case
// 		for this helper is the selection of sockets for the local NUMA
// 		node, when the program is attached to sockets using the
// 		**SO_ATTACH_REUSEPORT_EBPF** option (see also **socket(7)**),
// 		but the helper is also available to other eBPF program types,
// 		similarly to **bpf_get_smp_processor_id**\ ().
// 	Return
// 		The id of current NUMA node.
//
// long bpf_skb_change_head(struct sk_buff *skb, u32 len, u64 flags)
// 	Description
// 		Grows headroom of packet associated to *skb* and adjusts the
// 		offset of the MAC header accordingly, adding *len* bytes of
// 		space. It automatically extends and reallocates memory as
// 		required.
//
// 		This helper can be used on a layer 3 *skb* to push a MAC header
// 		for redirection into a layer 2 device.
//
// 		All values for *flags* are reserved for future usage, and must
// 		be left at zero.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_xdp_adjust_head(struct xdp_buff *xdp_md, int delta)
// 	Description
// 		Adjust (move) *xdp_md*\ **->data** by *delta* bytes. Note that
// 		it is possible to use a negative value for *delta*. This helper
// 		can be used to prepare the packet for pushing or popping
// 		headers.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_probe_read_str(void *dst, u32 size, const void *unsafe_ptr)
// 	Description
// 		Copy a NUL terminated string from an unsafe kernel address
// 		*unsafe_ptr* to *dst*. See **bpf_probe_read_kernel_str**\ () for
// 		more details.
//
// 		Generally, use **bpf_probe_read_user_str**\ () or
// 		**bpf_probe_read_kernel_str**\ () instead.
// 	Return
// 		On success, the strictly positive length of the string,
// 		including the trailing NUL character. On error, a negative
// 		value.
//
// u64 bpf_get_socket_cookie(struct sk_buff *skb)
// 	Description
// 		If the **struct sk_buff** pointed by *skb* has a known socket,
// 		retrieve the cookie (generated by the kernel) of this socket.
// 		If no cookie has been set yet, generate a new cookie. Once
// 		generated, the socket cookie remains stable for the life of the
// 		socket. This helper can be useful for monitoring per socket
// 		networking traffic statistics as it provides a global socket
// 		identifier that can be assumed unique.
// 	Return
// 		A 8-byte long unique number on success, or 0 if the socket
// 		field is missing inside *skb*.
//
// u64 bpf_get_socket_cookie(struct bpf_sock_addr *ctx)
// 	Description
// 		Equivalent to bpf_get_socket_cookie() helper that accepts
// 		*skb*, but gets socket from **struct bpf_sock_addr** context.
// 	Return
// 		A 8-byte long unique number.
//
// u64 bpf_get_socket_cookie(struct bpf_sock_ops *ctx)
// 	Description
// 		Equivalent to **bpf_get_socket_cookie**\ () helper that accepts
// 		*skb*, but gets socket from **struct bpf_sock_ops** context.
// 	Return
// 		A 8-byte long unique number.
//
// u64 bpf_get_socket_cookie(struct sock *sk)
// 	Description
// 		Equivalent to **bpf_get_socket_cookie**\ () helper that accepts
// 		*sk*, but gets socket from a BTF **struct sock**. This helper
// 		also works for sleepable programs.
// 	Return
// 		A 8-byte long unique number or 0 if *sk* is NULL.
//
// u32 bpf_get_socket_uid(struct sk_buff *skb)
// 	Description
// 		Get the owner UID of the socked associated to *skb*.
// 	Return
// 		The owner UID of the socket associated to *skb*. If the socket
// 		is **NULL**, or if it is not a full socket (i.e. if it is a
// 		time-wait or a request socket instead), **overflowuid** value
// 		is returned (note that **overflowuid** might also be the actual
// 		UID value for the socket).
//
// long bpf_set_hash(struct sk_buff *skb, u32 hash)
// 	Description
// 		Set the full hash for *skb* (set the field *skb*\ **->hash**)
// 		to value *hash*.
// 	Return
// 		0
//
// long bpf_setsockopt(void *bpf_socket, int level, int optname, void *optval, int optlen)
// 	Description
// 		Emulate a call to **setsockopt()** on the socket associated to
// 		*bpf_socket*, which must be a full socket. The *level* at
// 		which the option resides and the name *optname* of the option
// 		must be specified, see **setsockopt(2)** for more information.
// 		The option value of length *optlen* is pointed by *optval*.
//
// 		*bpf_socket* should be one of the following:
//
// 		* **struct bpf_sock_ops** for **BPF_PROG_TYPE_SOCK_OPS**.
// 		* **struct bpf_sock_addr** for **BPF_CGROUP_INET4_CONNECT**,
// 		  **BPF_CGROUP_INET6_CONNECT** and **BPF_CGROUP_UNIX_CONNECT**.
//
// 		This helper actually implements a subset of **setsockopt()**.
// 		It supports the following *level*\ s:
//
// 		* **SOL_SOCKET**, which supports the following *optname*\ s:
// 		  **SO_RCVBUF**, **SO_SNDBUF**, **SO_MAX_PACING_RATE**,
// 		  **SO_PRIORITY**, **SO_RCVLOWAT**, **SO_MARK**,
// 		  **SO_BINDTODEVICE**, **SO_KEEPALIVE**, **SO_REUSEADDR**,
// 		  **SO_REUSEPORT**, **SO_BINDTOIFINDEX**, **SO_TXREHASH**.
// 		* **IPPROTO_TCP**, which supports the following *optname*\ s:
// 		  **TCP_CONGESTION**, **TCP_BPF_IW**,
// 		  **TCP_BPF_SNDCWND_CLAMP**, **TCP_SAVE_SYN**,
// 		  **TCP_KEEPIDLE**, **TCP_KEEPINTVL**, **TCP_KEEPCNT**,
// 		  **TCP_SYNCNT**, **TCP_USER_TIMEOUT**, **TCP_NOTSENT_LOWAT**,
// 		  **TCP_NODELAY**, **TCP_MAXSEG**, **TCP_WINDOW_CLAMP**,
// 		  **TCP_THIN_LINEAR_TIMEOUTS**, **TCP_BPF_DELACK_MAX**,
// 		  **TCP_BPF_RTO_MIN**, **TCP_BPF_SOCK_OPS_CB_FLAGS**.
// 		* **IPPROTO_IP**, which supports *optname* **IP_TOS**.
// 		* **IPPROTO_IPV6**, which supports the following *optname*\ s:
// 		  **IPV6_TCLASS**, **IPV6_AUTOFLOWLABEL**.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_adjust_room(struct sk_buff *skb, s32 len_diff, u32 mode, u64 flags)
// 	Description
// 		Grow or shrink the room for data in the packet associated to
// 		*skb* by *len_diff*, and according to the selected *mode*.
//
// 		By default, the helper will reset any offloaded checksum
// 		indicator of the skb to CHECKSUM_NONE. This can be avoided
// 		by the following flag:
//
// 		* **BPF_F_ADJ_ROOM_NO_CSUM_RESET**: Do not reset offloaded
// 		  checksum data of the skb to CHECKSUM_NONE.
//
// 		There are two supported modes at this time:
//
// 		* **BPF_ADJ_ROOM_MAC**: Adjust room at the mac layer
// 		  (room space is added or removed between the layer 2 and
// 		  layer 3 headers).
//
// 		* **BPF_ADJ_ROOM_NET**: Adjust room at the network layer
// 		  (room space is added or removed between the layer 3 and
// 		  layer 4 headers).
//
// 		The following flags are supported at this time:
//
// 		* **BPF_F_ADJ_ROOM_FIXED_GSO**: Do not adjust gso_size.
// 		  Adjusting mss in this way is not allowed for datagrams.
//
// 		* **BPF_F_ADJ_ROOM_ENCAP_L3_IPV4**,
// 		  **BPF_F_ADJ_ROOM_ENCAP_L3_IPV6**:
// 		  Any new space is reserved to hold a tunnel header.
// 		  Configure skb offsets and other fields accordingly.
//
// 		* **BPF_F_ADJ_ROOM_ENCAP_L4_GRE**,
// 		  **BPF_F_ADJ_ROOM_ENCAP_L4_UDP**:
// 		  Use with ENCAP_L3 flags to further specify the tunnel type.
//
// 		* **BPF_F_ADJ_ROOM_ENCAP_L2**\ (*len*):
// 		  Use with ENCAP_L3/L4 flags to further specify the tunnel
// 		  type; *len* is the length of the inner MAC header.
//
// 		* **BPF_F_ADJ_ROOM_ENCAP_L2_ETH**:
// 		  Use with BPF_F_ADJ_ROOM_ENCAP_L2 flag to further specify the
// 		  L2 type as Ethernet.
//
// 		* **BPF_F_ADJ_ROOM_DECAP_L3_IPV4**,
// 		  **BPF_F_ADJ_ROOM_DECAP_L3_IPV6**:
// 		  Indicate the new IP header version after decapsulating the
// 		  outer IP header. Used when the inner and outer IP versions
// 		  are different. These flags only trigger a protocol change
// 		  without clearing any tunnel-specific GSO flags.
//
// 		* **BPF_F_ADJ_ROOM_DECAP_L4_GRE**:
// 		  Clear GRE tunnel GSO flags (SKB_GSO_GRE and SKB_GSO_GRE_CSUM)
// 		  when decapsulating a GRE tunnel.
//
// 		* **BPF_F_ADJ_ROOM_DECAP_L4_UDP**:
// 		  Clear UDP tunnel GSO flags (SKB_GSO_UDP_TUNNEL and
// 		  SKB_GSO_UDP_TUNNEL_CSUM) when decapsulating a UDP tunnel.
//
// 		* **BPF_F_ADJ_ROOM_DECAP_IPXIP4**:
// 		  Clear IPIP/SIT tunnel GSO flag (SKB_GSO_IPXIP4) when decapsulating
// 		  a tunnel with an outer IPv4 header (IPv4-in-IPv4 or IPv6-in-IPv4).
//
// 		* **BPF_F_ADJ_ROOM_DECAP_IPXIP6**:
// 		  Clear IPv6 encapsulation tunnel GSO flag (SKB_GSO_IPXIP6) when
// 		  decapsulating a tunnel with an outer IPv6 header (IPv6-in-IPv6
// 		  or IPv4-in-IPv6).
//
// 		When using the decapsulation flags above, the skb->encapsulation
// 		flag is automatically cleared if all tunnel-specific GSO flags
// 		(SKB_GSO_UDP_TUNNEL, SKB_GSO_UDP_TUNNEL_CSUM, SKB_GSO_GRE,
// 		SKB_GSO_GRE_CSUM, SKB_GSO_IPXIP4, SKB_GSO_IPXIP6) have been
// 		removed from the packet. This handles cases where all tunnel
// 		layers have been decapsulated.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_redirect_map(struct bpf_map *map, u64 key, u64 flags)
// 	Description
// 		Redirect the packet to the endpoint referenced by *map* at
// 		index *key*. Depending on its type, this *map* can contain
// 		references to net devices (for forwarding packets through other
// 		ports), or to CPUs (for redirecting XDP frames to another CPU;
// 		but this is only implemented for native XDP (with driver
// 		support) as of this writing).
//
// 		The lower two bits of *flags* are used as the return code if
// 		the map lookup fails. This is so that the return value can be
// 		one of the XDP program return codes up to **XDP_TX**, as chosen
// 		by the caller. The higher bits of *flags* can be set to
// 		BPF_F_BROADCAST or BPF_F_EXCLUDE_INGRESS as defined below.
//
// 		With BPF_F_BROADCAST the packet will be broadcasted to all the
// 		interfaces in the map, with BPF_F_EXCLUDE_INGRESS the ingress
// 		interface will be excluded when do broadcasting.
//
// 		See also **bpf_redirect**\ (), which only supports redirecting
// 		to an ifindex, but doesn't require a map to do so.
// 	Return
// 		**XDP_REDIRECT** on success, or the value of the two lower bits
// 		of the *flags* argument on error.
//
// long bpf_sk_redirect_map(struct sk_buff *skb, struct bpf_map *map, u32 key, u64 flags)
// 	Description
// 		Redirect the packet to the socket referenced by *map* (of type
// 		**BPF_MAP_TYPE_SOCKMAP**) at index *key*. Both ingress and
// 		egress interfaces can be used for redirection. The
// 		**BPF_F_INGRESS** value in *flags* is used to make the
// 		distinction (ingress path is selected if the flag is present,
// 		egress path otherwise). This is the only flag supported for now.
// 	Return
// 		**SK_PASS** on success, or **SK_DROP** on error.
//
// long bpf_sock_map_update(struct bpf_sock_ops *skops, struct bpf_map *map, void *key, u64 flags)
// 	Description
// 		Add an entry to, or update a *map* referencing sockets. The
// 		*skops* is used as a new value for the entry associated to
// 		*key*. *flags* is one of:
//
// 		**BPF_NOEXIST**
// 			The entry for *key* must not exist in the map.
// 		**BPF_EXIST**
// 			The entry for *key* must already exist in the map.
// 		**BPF_ANY**
// 			No condition on the existence of the entry for *key*.
//
// 		If the *map* has eBPF programs (parser and verdict), those will
// 		be inherited by the socket being added. If the socket is
// 		already attached to eBPF programs, this results in an error.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_xdp_adjust_meta(struct xdp_buff *xdp_md, int delta)
// 	Description
// 		Adjust the address pointed by *xdp_md*\ **->data_meta** by
// 		*delta* (which can be positive or negative). Note that this
// 		operation modifies the address stored in *xdp_md*\ **->data**,
// 		so the latter must be loaded only after the helper has been
// 		called.
//
// 		The use of *xdp_md*\ **->data_meta** is optional and programs
// 		are not required to use it. The rationale is that when the
// 		packet is processed with XDP (e.g. as DoS filter), it is
// 		possible to push further meta data along with it before passing
// 		to the stack, and to give the guarantee that an ingress eBPF
// 		program attached as a TC classifier on the same device can pick
// 		this up for further post-processing. Since TC works with socket
// 		buffers, it remains possible to set from XDP the **mark** or
// 		**priority** pointers, or other pointers for the socket buffer.
// 		Having this scratch space generic and programmable allows for
// 		more flexibility as the user is free to store whatever meta
// 		data they need.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_perf_event_read_value(struct bpf_map *map, u64 flags, struct bpf_perf_event_value *buf, u32 buf_size)
// 	Description
// 		Read the value of a perf event counter, and store it into *buf*
// 		of size *buf_size*. This helper relies on a *map* of type
// 		**BPF_MAP_TYPE_PERF_EVENT_ARRAY**. The nature of the perf event
// 		counter is selected when *map* is updated with perf event file
// 		descriptors. The *map* is an array whose size is the number of
// 		available CPUs, and each cell contains a value relative to one
// 		CPU. The value to retrieve is indicated by *flags*, that
// 		contains the index of the CPU to look up, masked with
// 		**BPF_F_INDEX_MASK**. Alternatively, *flags* can be set to
// 		**BPF_F_CURRENT_CPU** to indicate that the value for the
// 		current CPU should be retrieved.
//
// 		This helper behaves in a way close to
// 		**bpf_perf_event_read**\ () helper, save that instead of
// 		just returning the value observed, it fills the *buf*
// 		structure. This allows for additional data to be retrieved: in
// 		particular, the enabled and running times (in *buf* * 		**->enabled** and *buf*\ **->running**, respectively) are
// 		copied. In general, **bpf_perf_event_read_value**\ () is
// 		recommended over **bpf_perf_event_read**\ (), which has some
// 		ABI issues and provides fewer functionalities.
//
// 		These values are interesting, because hardware PMU (Performance
// 		Monitoring Unit) counters are limited resources. When there are
// 		more PMU based perf events opened than available counters,
// 		kernel will multiplex these events so each event gets certain
// 		percentage (but not all) of the PMU time. In case that
// 		multiplexing happens, the number of samples or counter value
// 		will not reflect the case compared to when no multiplexing
// 		occurs. This makes comparison between different runs difficult.
// 		Typically, the counter value should be normalized before
// 		comparing to other experiments. The usual normalization is done
// 		as follows.
//
// 		::
//
// 			normalized_counter = counter * t_enabled / t_running
//
// 		Where t_enabled is the time enabled for event and t_running is
// 		the time running for event since last normalization. The
// 		enabled and running times are accumulated since the perf event
// 		open. To achieve scaling factor between two invocations of an
// 		eBPF program, users can use CPU id as the key (which is
// 		typical for perf array usage model) to remember the previous
// 		value and do the calculation inside the eBPF program.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_perf_prog_read_value(struct bpf_perf_event_data *ctx, struct bpf_perf_event_value *buf, u32 buf_size)
// 	Description
// 		For an eBPF program attached to a perf event, retrieve the
// 		value of the event counter associated to *ctx* and store it in
// 		the structure pointed by *buf* and of size *buf_size*. Enabled
// 		and running times are also stored in the structure (see
// 		description of helper **bpf_perf_event_read_value**\ () for
// 		more details).
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_getsockopt(void *bpf_socket, int level, int optname, void *optval, int optlen)
// 	Description
// 		Emulate a call to **getsockopt()** on the socket associated to
// 		*bpf_socket*, which must be a full socket. The *level* at
// 		which the option resides and the name *optname* of the option
// 		must be specified, see **getsockopt(2)** for more information.
// 		The retrieved value is stored in the structure pointed by
// 		*opval* and of length *optlen*.
//
// 		*bpf_socket* should be one of the following:
//
// 		* **struct bpf_sock_ops** for **BPF_PROG_TYPE_SOCK_OPS**.
// 		* **struct bpf_sock_addr** for **BPF_CGROUP_INET4_CONNECT**,
// 		  **BPF_CGROUP_INET6_CONNECT** and **BPF_CGROUP_UNIX_CONNECT**.
//
// 		This helper actually implements a subset of **getsockopt()**.
// 		It supports the same set of *optname*\ s that is supported by
// 		the **bpf_setsockopt**\ () helper.  The exceptions are
// 		**TCP_BPF_*** is **bpf_setsockopt**\ () only and
// 		**TCP_SAVED_SYN** is **bpf_getsockopt**\ () only.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_override_return(struct pt_regs *regs, u64 rc)
// 	Description
// 		Used for error injection, this helper uses kprobes to override
// 		the return value of the probed function, and to set it to *rc*.
// 		The first argument is the context *regs* on which the kprobe
// 		works.
//
// 		This helper works by setting the PC (program counter)
// 		to an override function which is run in place of the original
// 		probed function. This means the probed function is not run at
// 		all. The replacement function just returns with the required
// 		value.
//
// 		This helper has security implications, and thus is subject to
// 		restrictions. It is only available if the kernel was compiled
// 		with the **CONFIG_BPF_KPROBE_OVERRIDE** configuration
// 		option, and in this case it only works on functions tagged with
// 		**ALLOW_ERROR_INJECTION** in the kernel code.
// 	Return
// 		0
//
// long bpf_sock_ops_cb_flags_set(struct bpf_sock_ops *bpf_sock, int argval)
// 	Description
// 		Attempt to set the value of the **bpf_sock_ops_cb_flags** field
// 		for the full TCP socket associated to *bpf_sock_ops* to
// 		*argval*.
//
// 		The primary use of this field is to determine if there should
// 		be calls to eBPF programs of type
// 		**BPF_PROG_TYPE_SOCK_OPS** at various points in the TCP
// 		code. A program of the same type can change its value, per
// 		connection and as necessary, when the connection is
// 		established. This field is directly accessible for reading, but
// 		this helper must be used for updates in order to return an
// 		error if an eBPF program tries to set a callback that is not
// 		supported in the current kernel.
//
// 		*argval* is a flag array which can combine these flags:
//
// 		* **BPF_SOCK_OPS_RTO_CB_FLAG** (retransmission time out)
// 		* **BPF_SOCK_OPS_RETRANS_CB_FLAG** (retransmission)
// 		* **BPF_SOCK_OPS_STATE_CB_FLAG** (TCP state change)
// 		* **BPF_SOCK_OPS_RTT_CB_FLAG** (every RTT)
//
// 		Therefore, this function can be used to clear a callback flag by
// 		setting the appropriate bit to zero. e.g. to disable the RTO
// 		callback:
//
// 		**bpf_sock_ops_cb_flags_set(bpf_sock,**
// 			**bpf_sock->bpf_sock_ops_cb_flags & ~BPF_SOCK_OPS_RTO_CB_FLAG)**
//
// 		Here are some examples of where one could call such eBPF
// 		program:
//
// 		* When RTO fires.
// 		* When a packet is retransmitted.
// 		* When the connection terminates.
// 		* When a packet is sent.
// 		* When a packet is received.
// 	Return
// 		Code **-EINVAL** if the socket is not a full TCP socket;
// 		otherwise, a positive number containing the bits that could not
// 		be set is returned (which comes down to 0 if all bits were set
// 		as required).
//
// long bpf_msg_redirect_map(struct sk_msg_buff *msg, struct bpf_map *map, u32 key, u64 flags)
// 	Description
// 		This helper is used in programs implementing policies at the
// 		socket level. If the message *msg* is allowed to pass (i.e. if
// 		the verdict eBPF program returns **SK_PASS**), redirect it to
// 		the socket referenced by *map* (of type
// 		**BPF_MAP_TYPE_SOCKMAP**) at index *key*. Both ingress and
// 		egress interfaces can be used for redirection. The
// 		**BPF_F_INGRESS** value in *flags* is used to make the
// 		distinction (ingress path is selected if the flag is present,
// 		egress path otherwise). This is the only flag supported for now.
// 	Return
// 		**SK_PASS** on success, or **SK_DROP** on error.
//
// long bpf_msg_apply_bytes(struct sk_msg_buff *msg, u32 bytes)
// 	Description
// 		For socket policies, apply the verdict of the eBPF program to
// 		the next *bytes* (number of bytes) of message *msg*.
//
// 		For example, this helper can be used in the following cases:
//
// 		* A single **sendmsg**\ () or **sendfile**\ () system call
// 		  contains multiple logical messages that the eBPF program is
// 		  supposed to read and for which it should apply a verdict.
// 		* An eBPF program only cares to read the first *bytes* of a
// 		  *msg*. If the message has a large payload, then setting up
// 		  and calling the eBPF program repeatedly for all bytes, even
// 		  though the verdict is already known, would create unnecessary
// 		  overhead.
//
// 		When called from within an eBPF program, the helper sets a
// 		counter internal to the BPF infrastructure, that is used to
// 		apply the last verdict to the next *bytes*. If *bytes* is
// 		smaller than the current data being processed from a
// 		**sendmsg**\ () or **sendfile**\ () system call, the first
// 		*bytes* will be sent and the eBPF program will be re-run with
// 		the pointer for start of data pointing to byte number *bytes*
// 		**+ 1**. If *bytes* is larger than the current data being
// 		processed, then the eBPF verdict will be applied to multiple
// 		**sendmsg**\ () or **sendfile**\ () calls until *bytes* are
// 		consumed.
//
// 		Note that if a socket closes with the internal counter holding
// 		a non-zero value, this is not a problem because data is not
// 		being buffered for *bytes* and is sent as it is received.
// 	Return
// 		0
//
// long bpf_msg_cork_bytes(struct sk_msg_buff *msg, u32 bytes)
// 	Description
// 		For socket policies, prevent the execution of the verdict eBPF
// 		program for message *msg* until *bytes* (byte number) have been
// 		accumulated.
//
// 		This can be used when one needs a specific number of bytes
// 		before a verdict can be assigned, even if the data spans
// 		multiple **sendmsg**\ () or **sendfile**\ () calls. The extreme
// 		case would be a user calling **sendmsg**\ () repeatedly with
// 		1-byte long message segments. Obviously, this is bad for
// 		performance, but it is still valid. If the eBPF program needs
// 		*bytes* bytes to validate a header, this helper can be used to
// 		prevent the eBPF program to be called again until *bytes* have
// 		been accumulated.
// 	Return
// 		0
//
// long bpf_msg_pull_data(struct sk_msg_buff *msg, u32 start, u32 end, u64 flags)
// 	Description
// 		For socket policies, pull in non-linear data from user space
// 		for *msg* and set pointers *msg*\ **->data** and *msg* * 		**->data_end** to *start* and *end* bytes offsets into *msg*,
// 		respectively.
//
// 		If a program of type **BPF_PROG_TYPE_SK_MSG** is run on a
// 		*msg* it can only parse data that the (**data**, **data_end**)
// 		pointers have already consumed. For **sendmsg**\ () hooks this
// 		is likely the first scatterlist element. But for calls relying
// 		on the **sendpage** handler (e.g. **sendfile**\ ()) this will
// 		be the range (**0**, **0**) because the data is shared with
// 		user space and by default the objective is to avoid allowing
// 		user space to modify data while (or after) eBPF verdict is
// 		being decided. This helper can be used to pull in data and to
// 		set the start and end pointer to given values. Data will be
// 		copied if necessary (i.e. if data was not linear and if start
// 		and end pointers do not point to the same chunk).
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
//
// 		All values for *flags* are reserved for future usage, and must
// 		be left at zero.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_bind(struct bpf_sock_addr *ctx, struct sockaddr *addr, int addr_len)
// 	Description
// 		Bind the socket associated to *ctx* to the address pointed by
// 		*addr*, of length *addr_len*. This allows for making outgoing
// 		connection from the desired IP address, which can be useful for
// 		example when all processes inside a cgroup should use one
// 		single IP address on a host that has multiple IP configured.
//
// 		This helper works for IPv4 and IPv6, TCP and UDP sockets. The
// 		domain (*addr*\ **->sa_family**) must be **AF_INET** (or
// 		**AF_INET6**). It's advised to pass zero port (**sin_port**
// 		or **sin6_port**) which triggers IP_BIND_ADDRESS_NO_PORT-like
// 		behavior and lets the kernel efficiently pick up an unused
// 		port as long as 4-tuple is unique. Passing non-zero port might
// 		lead to degraded performance.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_xdp_adjust_tail(struct xdp_buff *xdp_md, int delta)
// 	Description
// 		Adjust (move) *xdp_md*\ **->data_end** by *delta* bytes. It is
// 		possible to both shrink and grow the packet tail.
// 		Shrink done via *delta* being a negative integer.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_get_xfrm_state(struct sk_buff *skb, u32 index, struct bpf_xfrm_state *xfrm_state, u32 size, u64 flags)
// 	Description
// 		Retrieve the XFRM state (IP transform framework, see also
// 		**ip-xfrm(8)**) at *index* in XFRM "security path" for *skb*.
//
// 		The retrieved value is stored in the **struct bpf_xfrm_state**
// 		pointed by *xfrm_state* and of length *size*.
//
// 		All values for *flags* are reserved for future usage, and must
// 		be left at zero.
//
// 		This helper is available only if the kernel was compiled with
// 		**CONFIG_XFRM** configuration option.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_get_stack(void *ctx, void *buf, u32 size, u64 flags)
// 	Description
// 		Return a user or a kernel stack in bpf program provided buffer.
// 		To achieve this, the helper needs *ctx*, which is a pointer
// 		to the context on which the tracing program is executed.
// 		To store the stacktrace, the bpf program provides *buf* with
// 		a nonnegative *size*.
//
// 		The last argument, *flags*, holds the number of stack frames to
// 		skip (from 0 to 255), masked with
// 		**BPF_F_SKIP_FIELD_MASK**. The next bits can be used to set
// 		the following flags:
//
// 		**BPF_F_USER_STACK**
// 			Collect a user space stack instead of a kernel stack.
// 		**BPF_F_USER_BUILD_ID**
// 			Collect (build_id, file_offset) instead of ips for user
// 			stack, only valid if **BPF_F_USER_STACK** is also
// 			specified.
//
// 			*file_offset* is an offset relative to the beginning
// 			of the executable or shared object file backing the vma
// 			which the *ip* falls in. It is *not* an offset relative
// 			to that object's base address. Accordingly, it must be
// 			adjusted by adding (sh_addr - sh_offset), where
// 			sh_{addr,offset} correspond to the executable section
// 			containing *file_offset* in the object, for comparisons
// 			to symbols' st_value to be valid.
//
// 		**bpf_get_stack**\ () can collect up to
// 		**PERF_MAX_STACK_DEPTH** both kernel and user frames, subject
// 		to sufficient large buffer size. Note that
// 		this limit can be controlled with the **sysctl** program, and
// 		that it should be manually increased in order to profile long
// 		user stacks (such as stacks for Java programs). To do so, use:
//
// 		::
//
// 			# sysctl kernel.perf_event_max_stack=<new value>
// 	Return
// 		The non-negative copied *buf* length equal to or less than
// 		*size* on success, or a negative error in case of failure.
//
// long bpf_skb_load_bytes_relative(const void *skb, u32 offset, void *to, u32 len, u32 start_header)
// 	Description
// 		This helper is similar to **bpf_skb_load_bytes**\ () in that
// 		it provides an easy way to load *len* bytes from *offset*
// 		from the packet associated to *skb*, into the buffer pointed
// 		by *to*. The difference to **bpf_skb_load_bytes**\ () is that
// 		a fifth argument *start_header* exists in order to select a
// 		base offset to start from. *start_header* can be one of:
//
// 		**BPF_HDR_START_MAC**
// 			Base offset to load data from is *skb*'s mac header.
// 		**BPF_HDR_START_NET**
// 			Base offset to load data from is *skb*'s network header.
//
// 		In general, "direct packet access" is the preferred method to
// 		access packet data, however, this helper is in particular useful
// 		in socket filters where *skb*\ **->data** does not always point
// 		to the start of the mac header and where "direct packet access"
// 		is not available.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_fib_lookup(void *ctx, struct bpf_fib_lookup *params, int plen, u32 flags)
// 	Description
// 		Do FIB lookup in kernel tables using parameters in *params*.
// 		If lookup is successful and result shows packet is to be
// 		forwarded, the neighbor tables are searched for the nexthop.
// 		If successful (ie., FIB lookup shows forwarding and nexthop
// 		is resolved), the nexthop address is returned in ipv4_dst
// 		or ipv6_dst based on family, smac is set to mac address of
// 		egress device, dmac is set to nexthop mac address, rt_metric
// 		is set to metric from route (IPv4/IPv6 only), and ifindex
// 		is set to the device index of the nexthop from the FIB lookup.
//
// 		*plen* argument is the size of the passed in struct.
// 		*flags* argument can be a combination of one or more of the
// 		following values:
//
// 		**BPF_FIB_LOOKUP_DIRECT**
// 			Do a direct table lookup vs full lookup using FIB
// 			rules.
// 		**BPF_FIB_LOOKUP_TBID**
// 			Used with BPF_FIB_LOOKUP_DIRECT.
// 			Use the routing table ID present in *params*->tbid
// 			for the fib lookup.
// 		**BPF_FIB_LOOKUP_OUTPUT**
// 			Perform lookup from an egress perspective (default is
// 			ingress).
// 		**BPF_FIB_LOOKUP_SKIP_NEIGH**
// 			Skip the neighbour table lookup. *params*->dmac
// 			and *params*->smac will not be set as output. A common
// 			use case is to call **bpf_redirect_neigh**\ () after
// 			doing **bpf_fib_lookup**\ ().
// 		**BPF_FIB_LOOKUP_SRC**
// 			Derive and set source IP addr in *params*->ipv{4,6}_src
// 			for the nexthop. If the src addr cannot be derived,
// 			**BPF_FIB_LKUP_RET_NO_SRC_ADDR** is returned. In this
// 			case, *params*->dmac and *params*->smac are not set either.
// 		**BPF_FIB_LOOKUP_MARK**
// 			Use the mark present in *params*->mark for the fib lookup.
// 			This option should not be used with BPF_FIB_LOOKUP_DIRECT,
// 			as it only has meaning for full lookups.
// 		**BPF_FIB_LOOKUP_VLAN**
// 			If the fib lookup resolves to a VLAN device whose
// 			parent is a real (non-VLAN) device, set
// 			*params*->h_vlan_proto and *params*->h_vlan_TCI from
// 			the VLAN device and replace *params*->ifindex with the
// 			parent's ifindex. *params*->h_vlan_TCI carries the VID
// 			only, with PCP and DEI bits zero; a consumer wanting to
// 			set egress priority writes PCP itself. *params*->smac is
// 			the VLAN device's own address, which can differ from the
// 			parent's. Only the immediate parent is resolved; if it
// 			is itself a VLAN device (QinQ) or in another namespace,
// 			the egress cannot be reduced to a physical device plus
// 			one tag and the lookup returns
// 			**BPF_FIB_LKUP_RET_VLAN_FAILURE** with *params*->ifindex
// 			left at the input. To obtain the VLAN device's own
// 			ifindex, repeat the lookup without
// 			**BPF_FIB_LOOKUP_VLAN**, re-initializing *params*
// 			first: output fields overwrite the inputs they share
// 			storage with. The swap and the vlan fields
// 			are written only on success; other output fields keep
// 			the helper's existing behaviour, so a frag-needed result
// 			still reports the route mtu in *params*->mtu_result.
// 			This flag is only valid for XDP programs; tc programs
// 			receive -EINVAL since they can redirect to the VLAN
// 			device directly.
// 		**BPF_FIB_LOOKUP_VLAN_INPUT**
// 			Treat *params*->h_vlan_proto and *params*->h_vlan_TCI
// 			as an input VLAN tag and run the lookup as if ingress
// 			had happened on the VLAN subinterface carrying that tag
// 			on *params*->ifindex. The VID is the low 12 bits of
// 			*params*->h_vlan_TCI; *params*->h_vlan_proto must be
// 			ETH_P_8021Q or ETH_P_8021AD in network byte order, else
// 			**-EINVAL**. If *params*->ifindex is itself a VLAN
// 			device, its inner (QinQ) subinterface is matched; for a
// 			bond or team, pass the master's ifindex. An unmatched
// 			tag, a down device, or one in another namespace returns
// 			**BPF_FIB_LKUP_RET_NOT_FWDED**, mirroring real ingress.
// 			A VID of 0 is looked up literally, so do not set this
// 			flag for priority-tagged frames. Cannot be combined with
// 			**BPF_FIB_LOOKUP_TBID** or **BPF_FIB_LOOKUP_OUTPUT**
// 			(returns **-EINVAL**).
//
// 		*ctx* is either **struct xdp_md** for XDP programs or
// 		**struct sk_buff** tc cls_act programs.
// 	Return
// 		* < 0 if any input argument is invalid
// 		*   0 on success (packet is forwarded, nexthop neighbor exists)
// 		* > 0 one of **BPF_FIB_LKUP_RET_** codes explaining why the
// 		  packet is not forwarded or needs assist from full stack
//
// 		If lookup fails with BPF_FIB_LKUP_RET_FRAG_NEEDED, then the MTU
// 		was exceeded and output params->mtu_result contains the MTU.
//
// long bpf_sock_hash_update(struct bpf_sock_ops *skops, struct bpf_map *map, void *key, u64 flags)
// 	Description
// 		Add an entry to, or update a sockhash *map* referencing sockets.
// 		The *skops* is used as a new value for the entry associated to
// 		*key*. *flags* is one of:
//
// 		**BPF_NOEXIST**
// 			The entry for *key* must not exist in the map.
// 		**BPF_EXIST**
// 			The entry for *key* must already exist in the map.
// 		**BPF_ANY**
// 			No condition on the existence of the entry for *key*.
//
// 		If the *map* has eBPF programs (parser and verdict), those will
// 		be inherited by the socket being added. If the socket is
// 		already attached to eBPF programs, this results in an error.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_msg_redirect_hash(struct sk_msg_buff *msg, struct bpf_map *map, void *key, u64 flags)
// 	Description
// 		This helper is used in programs implementing policies at the
// 		socket level. If the message *msg* is allowed to pass (i.e. if
// 		the verdict eBPF program returns **SK_PASS**), redirect it to
// 		the socket referenced by *map* (of type
// 		**BPF_MAP_TYPE_SOCKHASH**) using hash *key*. Both ingress and
// 		egress interfaces can be used for redirection. The
// 		**BPF_F_INGRESS** value in *flags* is used to make the
// 		distinction (ingress path is selected if the flag is present,
// 		egress path otherwise). This is the only flag supported for now.
// 	Return
// 		**SK_PASS** on success, or **SK_DROP** on error.
//
// long bpf_sk_redirect_hash(struct sk_buff *skb, struct bpf_map *map, void *key, u64 flags)
// 	Description
// 		This helper is used in programs implementing policies at the
// 		skb socket level. If the sk_buff *skb* is allowed to pass (i.e.
// 		if the verdict eBPF program returns **SK_PASS**), redirect it
// 		to the socket referenced by *map* (of type
// 		**BPF_MAP_TYPE_SOCKHASH**) using hash *key*. Both ingress and
// 		egress interfaces can be used for redirection. The
// 		**BPF_F_INGRESS** value in *flags* is used to make the
// 		distinction (ingress path is selected if the flag is present,
// 		egress otherwise). This is the only flag supported for now.
// 	Return
// 		**SK_PASS** on success, or **SK_DROP** on error.
//
// long bpf_lwt_push_encap(struct sk_buff *skb, u32 type, void *hdr, u32 len)
// 	Description
// 		Encapsulate the packet associated to *skb* within a Layer 3
// 		protocol header. This header is provided in the buffer at
// 		address *hdr*, with *len* its size in bytes. *type* indicates
// 		the protocol of the header and can be one of:
//
// 		**BPF_LWT_ENCAP_SEG6**
// 			IPv6 encapsulation with Segment Routing Header
// 			(**struct ipv6_sr_hdr**). *hdr* only contains the SRH,
// 			the IPv6 header is computed by the kernel.
// 		**BPF_LWT_ENCAP_SEG6_INLINE**
// 			Only works if *skb* contains an IPv6 packet. Insert a
// 			Segment Routing Header (**struct ipv6_sr_hdr**) inside
// 			the IPv6 header.
// 		**BPF_LWT_ENCAP_IP**
// 			IP encapsulation (GRE/GUE/IPIP/etc). The outer header
// 			must be IPv4 or IPv6, followed by zero or more
// 			additional headers, up to **LWT_BPF_MAX_HEADROOM**
// 			total bytes in all prepended headers. Please note that
// 			if **skb_is_gso**\ (*skb*) is true, no more than two
// 			headers can be prepended, and the inner header, if
// 			present, should be either GRE or UDP/GUE.
//
// 		**BPF_LWT_ENCAP_SEG6**\ \* types can be called by BPF programs
// 		of type **BPF_PROG_TYPE_LWT_IN**; **BPF_LWT_ENCAP_IP** type can
// 		be called by bpf programs of types **BPF_PROG_TYPE_LWT_IN** and
// 		**BPF_PROG_TYPE_LWT_XMIT**.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_lwt_seg6_store_bytes(struct sk_buff *skb, u32 offset, const void *from, u32 len)
// 	Description
// 		Store *len* bytes from address *from* into the packet
// 		associated to *skb*, at *offset*. Only the flags, tag and TLVs
// 		inside the outermost IPv6 Segment Routing Header can be
// 		modified through this helper.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_lwt_seg6_adjust_srh(struct sk_buff *skb, u32 offset, s32 delta)
// 	Description
// 		Adjust the size allocated to TLVs in the outermost IPv6
// 		Segment Routing Header contained in the packet associated to
// 		*skb*, at position *offset* by *delta* bytes. Only offsets
// 		after the segments are accepted. *delta* can be as well
// 		positive (growing) as negative (shrinking).
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_lwt_seg6_action(struct sk_buff *skb, u32 action, void *param, u32 param_len)
// 	Description
// 		Apply an IPv6 Segment Routing action of type *action* to the
// 		packet associated to *skb*. Each action takes a parameter
// 		contained at address *param*, and of length *param_len* bytes.
// 		*action* can be one of:
//
// 		**SEG6_LOCAL_ACTION_END_X**
// 			End.X action: Endpoint with Layer-3 cross-connect.
// 			Type of *param*: **struct in6_addr**.
// 		**SEG6_LOCAL_ACTION_END_T**
// 			End.T action: Endpoint with specific IPv6 table lookup.
// 			Type of *param*: **int**.
// 		**SEG6_LOCAL_ACTION_END_B6**
// 			End.B6 action: Endpoint bound to an SRv6 policy.
// 			Type of *param*: **struct ipv6_sr_hdr**.
// 		**SEG6_LOCAL_ACTION_END_B6_ENCAP**
// 			End.B6.Encap action: Endpoint bound to an SRv6
// 			encapsulation policy.
// 			Type of *param*: **struct ipv6_sr_hdr**.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_rc_repeat(void *ctx)
// 	Description
// 		This helper is used in programs implementing IR decoding, to
// 		report a successfully decoded repeat key message. This delays
// 		the generation of a key up event for previously generated
// 		key down event.
//
// 		Some IR protocols like NEC have a special IR message for
// 		repeating last button, for when a button is held down.
//
// 		The *ctx* should point to the lirc sample as passed into
// 		the program.
//
// 		This helper is only available is the kernel was compiled with
// 		the **CONFIG_BPF_LIRC_MODE2** configuration option set to
// 		"**y**".
// 	Return
// 		0
//
// long bpf_rc_keydown(void *ctx, u32 protocol, u64 scancode, u32 toggle)
// 	Description
// 		This helper is used in programs implementing IR decoding, to
// 		report a successfully decoded key press with *scancode*,
// 		*toggle* value in the given *protocol*. The scancode will be
// 		translated to a keycode using the rc keymap, and reported as
// 		an input key down event. After a period a key up event is
// 		generated. This period can be extended by calling either
// 		**bpf_rc_keydown**\ () again with the same values, or calling
// 		**bpf_rc_repeat**\ ().
//
// 		Some protocols include a toggle bit, in case the button was
// 		released and pressed again between consecutive scancodes.
//
// 		The *ctx* should point to the lirc sample as passed into
// 		the program.
//
// 		The *protocol* is the decoded protocol number (see
// 		**enum rc_proto** for some predefined values).
//
// 		This helper is only available is the kernel was compiled with
// 		the **CONFIG_BPF_LIRC_MODE2** configuration option set to
// 		"**y**".
// 	Return
// 		0
//
// u64 bpf_skb_cgroup_id(struct sk_buff *skb)
// 	Description
// 		Return the cgroup v2 id of the socket associated with the *skb*.
// 		This is roughly similar to the **bpf_get_cgroup_classid**\ ()
// 		helper for cgroup v1 by providing a tag resp. identifier that
// 		can be matched on or used for map lookups e.g. to implement
// 		policy. The cgroup v2 id of a given path in the hierarchy is
// 		exposed in user space through the f_handle API in order to get
// 		to the same 64-bit id.
//
// 		This helper can be used on TC egress path, but not on ingress,
// 		and is available only if the kernel was compiled with the
// 		**CONFIG_SOCK_CGROUP_DATA** configuration option.
// 	Return
// 		The id is returned or 0 in case the id could not be retrieved.
//
// u64 bpf_get_current_cgroup_id(void)
// 	Description
// 		Get the current cgroup id based on the cgroup within which
// 		the current task is running.
// 	Return
// 		A 64-bit integer containing the current cgroup id based
// 		on the cgroup within which the current task is running.
//
// void *bpf_get_local_storage(void *map, u64 flags)
// 	Description
// 		Get the pointer to the local storage area.
// 		The type and the size of the local storage is defined
// 		by the *map* argument.
// 		The *flags* meaning is specific for each map type,
// 		and has to be 0 for cgroup local storage.
//
// 		Depending on the BPF program type, a local storage area
// 		can be shared between multiple instances of the BPF program,
// 		running simultaneously.
//
// 		A user should care about the synchronization by himself.
// 		For example, by using the **BPF_ATOMIC** instructions to alter
// 		the shared data.
// 	Return
// 		A pointer to the local storage area.
//
// long bpf_sk_select_reuseport(struct sk_reuseport_md *reuse, struct bpf_map *map, void *key, u64 flags)
// 	Description
// 		Select a **SO_REUSEPORT** socket from a
// 		**BPF_MAP_TYPE_REUSEPORT_SOCKARRAY** *map*.
// 		It checks the selected socket is matching the incoming
// 		request in the socket buffer.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// u64 bpf_skb_ancestor_cgroup_id(struct sk_buff *skb, int ancestor_level)
// 	Description
// 		Return id of cgroup v2 that is ancestor of cgroup associated
// 		with the *skb* at the *ancestor_level*.  The root cgroup is at
// 		*ancestor_level* zero and each step down the hierarchy
// 		increments the level. If *ancestor_level* == level of cgroup
// 		associated with *skb*, then return value will be same as that
// 		of **bpf_skb_cgroup_id**\ ().
//
// 		The helper is useful to implement policies based on cgroups
// 		that are upper in hierarchy than immediate cgroup associated
// 		with *skb*.
//
// 		The format of returned id and helper limitations are same as in
// 		**bpf_skb_cgroup_id**\ ().
// 	Return
// 		The id is returned or 0 in case the id could not be retrieved.
//
// struct bpf_sock *bpf_sk_lookup_tcp(void *ctx, struct bpf_sock_tuple *tuple, u32 tuple_size, u64 netns, u64 flags)
// 	Description
// 		Look for TCP socket matching *tuple*, optionally in a child
// 		network namespace *netns*. The return value must be checked,
// 		and if non-**NULL**, released via **bpf_sk_release**\ ().
//
// 		The *ctx* should point to the context of the program, such as
// 		the skb or socket (depending on the hook in use). This is used
// 		to determine the base network namespace for the lookup.
//
// 		*tuple_size* must be one of:
//
// 		**sizeof**\ (*tuple*\ **->ipv4**)
// 			Look for an IPv4 socket.
// 		**sizeof**\ (*tuple*\ **->ipv6**)
// 			Look for an IPv6 socket.
//
// 		If the *netns* is a negative signed 32-bit integer, then the
// 		socket lookup table in the netns associated with the *ctx*
// 		will be used. For the TC hooks, this is the netns of the device
// 		in the skb. For socket hooks, this is the netns of the socket.
// 		If *netns* is any other signed 32-bit value greater than or
// 		equal to zero then it specifies the ID of the netns relative to
// 		the netns associated with the *ctx*. *netns* values beyond the
// 		range of 32-bit integers are reserved for future use.
//
// 		All values for *flags* are reserved for future usage, and must
// 		be left at zero.
//
// 		This helper is available only if the kernel was compiled with
// 		**CONFIG_NET** configuration option.
// 	Return
// 		Pointer to **struct bpf_sock**, or **NULL** in case of failure.
// 		For sockets with reuseport option, the **struct bpf_sock**
// 		result is from *reuse*\ **->socks**\ [] using the hash of the
// 		tuple.
//
// struct bpf_sock *bpf_sk_lookup_udp(void *ctx, struct bpf_sock_tuple *tuple, u32 tuple_size, u64 netns, u64 flags)
// 	Description
// 		Look for UDP socket matching *tuple*, optionally in a child
// 		network namespace *netns*. The return value must be checked,
// 		and if non-**NULL**, released via **bpf_sk_release**\ ().
//
// 		The *ctx* should point to the context of the program, such as
// 		the skb or socket (depending on the hook in use). This is used
// 		to determine the base network namespace for the lookup.
//
// 		*tuple_size* must be one of:
//
// 		**sizeof**\ (*tuple*\ **->ipv4**)
// 			Look for an IPv4 socket.
// 		**sizeof**\ (*tuple*\ **->ipv6**)
// 			Look for an IPv6 socket.
//
// 		If the *netns* is a negative signed 32-bit integer, then the
// 		socket lookup table in the netns associated with the *ctx*
// 		will be used. For the TC hooks, this is the netns of the device
// 		in the skb. For socket hooks, this is the netns of the socket.
// 		If *netns* is any other signed 32-bit value greater than or
// 		equal to zero then it specifies the ID of the netns relative to
// 		the netns associated with the *ctx*. *netns* values beyond the
// 		range of 32-bit integers are reserved for future use.
//
// 		All values for *flags* are reserved for future usage, and must
// 		be left at zero.
//
// 		This helper is available only if the kernel was compiled with
// 		**CONFIG_NET** configuration option.
// 	Return
// 		Pointer to **struct bpf_sock**, or **NULL** in case of failure.
// 		For sockets with reuseport option, the **struct bpf_sock**
// 		result is from *reuse*\ **->socks**\ [] using the hash of the
// 		tuple.
//
// long bpf_sk_release(void *sock)
// 	Description
// 		Release the reference held by *sock*. *sock* must be a
// 		non-**NULL** pointer that was returned from
// 		**bpf_sk_lookup_xxx**\ ().
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_map_push_elem(struct bpf_map *map, const void *value, u64 flags)
// 	Description
// 		Push an element *value* in *map*. *flags* is one of:
//
// 		**BPF_EXIST**
// 			If the queue/stack is full, the oldest element is
// 			removed to make room for this.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_map_pop_elem(struct bpf_map *map, void *value)
// 	Description
// 		Pop an element from *map*.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_map_peek_elem(struct bpf_map *map, void *value)
// 	Description
// 		Get an element from *map* without removing it.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_msg_push_data(struct sk_msg_buff *msg, u32 start, u32 len, u64 flags)
// 	Description
// 		For socket policies, insert *len* bytes into *msg* at offset
// 		*start*.
//
// 		If a program of type **BPF_PROG_TYPE_SK_MSG** is run on a
// 		*msg* it may want to insert metadata or options into the *msg*.
// 		This can later be read and used by any of the lower layer BPF
// 		hooks.
//
// 		This helper may fail if under memory pressure (a malloc
// 		fails) in these cases BPF programs will get an appropriate
// 		error and BPF programs will need to handle them.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_msg_pop_data(struct sk_msg_buff *msg, u32 start, u32 len, u64 flags)
// 	Description
// 		Will remove *len* bytes from a *msg* starting at byte *start*.
// 		This may result in **ENOMEM** errors under certain situations if
// 		an allocation and copy are required due to a full ring buffer.
// 		However, the helper will try to avoid doing the allocation
// 		if possible. Other errors can occur if input parameters are
// 		invalid either due to *start* byte not being valid part of *msg*
// 		payload and/or *pop* value being to large.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_rc_pointer_rel(void *ctx, s32 rel_x, s32 rel_y)
// 	Description
// 		This helper is used in programs implementing IR decoding, to
// 		report a successfully decoded pointer movement.
//
// 		The *ctx* should point to the lirc sample as passed into
// 		the program.
//
// 		This helper is only available is the kernel was compiled with
// 		the **CONFIG_BPF_LIRC_MODE2** configuration option set to
// 		"**y**".
// 	Return
// 		0
//
// long bpf_spin_lock(struct bpf_spin_lock *lock)
// 	Description
// 		Acquire a spinlock represented by the pointer *lock*, which is
// 		stored as part of a value of a map. Taking the lock allows to
// 		safely update the rest of the fields in that value. The
// 		spinlock can (and must) later be released with a call to
// 		**bpf_spin_unlock**\ (\ *lock*\ ).
//
// 		Spinlocks in BPF programs come with a number of restrictions
// 		and constraints:
//
// 		* **bpf_spin_lock** objects are only allowed inside maps of
// 		  types **BPF_MAP_TYPE_HASH** and **BPF_MAP_TYPE_ARRAY** (this
// 		  list could be extended in the future).
// 		* BTF description of the map is mandatory.
// 		* The BPF program can take ONE lock at a time, since taking two
// 		  or more could cause dead locks.
// 		* Only one **struct bpf_spin_lock** is allowed per map element.
// 		* When the lock is taken, calls (either BPF to BPF or helpers)
// 		  are not allowed.
// 		* The **BPF_LD_ABS** and **BPF_LD_IND** instructions are not
// 		  allowed inside a spinlock-ed region.
// 		* The BPF program MUST call **bpf_spin_unlock**\ () to release
// 		  the lock, on all execution paths, before it returns.
// 		* The BPF program can access **struct bpf_spin_lock** only via
// 		  the **bpf_spin_lock**\ () and **bpf_spin_unlock**\ ()
// 		  helpers. Loading or storing data into the **struct
// 		  bpf_spin_lock** *lock*\ **;** field of a map is not allowed.
// 		* To use the **bpf_spin_lock**\ () helper, the BTF description
// 		  of the map value must be a struct and have **struct
// 		  bpf_spin_lock** *anyname*\ **;** field at the top level.
// 		  Nested lock inside another struct is not allowed.
// 		* The **struct bpf_spin_lock** *lock* field in a map value must
// 		  be aligned on a multiple of 4 bytes in that value.
// 		* Syscall with command **BPF_MAP_LOOKUP_ELEM** does not copy
// 		  the **bpf_spin_lock** field to user space.
// 		* Syscall with command **BPF_MAP_UPDATE_ELEM**, or update from
// 		  a BPF program, do not update the **bpf_spin_lock** field.
// 		* **bpf_spin_lock** cannot be on the stack or inside a
// 		  networking packet (it can only be inside of a map values).
// 		* **bpf_spin_lock** is available to root only.
// 		* Tracing programs and socket filter programs cannot use
// 		  **bpf_spin_lock**\ () due to insufficient preemption checks
// 		  (but this may change in the future).
// 		* **bpf_spin_lock** is not allowed in inner maps of map-in-map.
// 	Return
// 		0
//
// long bpf_spin_unlock(struct bpf_spin_lock *lock)
// 	Description
// 		Release the *lock* previously locked by a call to
// 		**bpf_spin_lock**\ (\ *lock*\ ).
// 	Return
// 		0
//
// struct bpf_sock *bpf_sk_fullsock(struct bpf_sock *sk)
// 	Description
// 		This helper gets a **struct bpf_sock** pointer such
// 		that all the fields in this **bpf_sock** can be accessed.
// 	Return
// 		A **struct bpf_sock** pointer on success, or **NULL** in
// 		case of failure.
//
// struct bpf_tcp_sock *bpf_tcp_sock(struct bpf_sock *sk)
// 	Description
// 		This helper gets a **struct bpf_tcp_sock** pointer from a
// 		**struct bpf_sock** pointer.
// 	Return
// 		A **struct bpf_tcp_sock** pointer on success, or **NULL** in
// 		case of failure.
//
// long bpf_skb_ecn_set_ce(struct sk_buff *skb)
// 	Description
// 		Set ECN (Explicit Congestion Notification) field of IP header
// 		to **CE** (Congestion Encountered) if current value is **ECT**
// 		(ECN Capable Transport). Otherwise, do nothing. Works with IPv6
// 		and IPv4.
// 	Return
// 		1 if the **CE** flag is set (either by the current helper call
// 		or because it was already present), 0 if it is not set.
//
// struct bpf_sock *bpf_get_listener_sock(struct bpf_sock *sk)
// 	Description
// 		Return a **struct bpf_sock** pointer in **TCP_LISTEN** state.
// 		**bpf_sk_release**\ () is unnecessary and not allowed.
// 	Return
// 		A **struct bpf_sock** pointer on success, or **NULL** in
// 		case of failure.
//
// struct bpf_sock *bpf_skc_lookup_tcp(void *ctx, struct bpf_sock_tuple *tuple, u32 tuple_size, u64 netns, u64 flags)
// 	Description
// 		Look for TCP socket matching *tuple*, optionally in a child
// 		network namespace *netns*. The return value must be checked,
// 		and if non-**NULL**, released via **bpf_sk_release**\ ().
//
// 		This function is identical to **bpf_sk_lookup_tcp**\ (), except
// 		that it also returns timewait or request sockets. Use
// 		**bpf_sk_fullsock**\ () or **bpf_tcp_sock**\ () to access the
// 		full structure.
//
// 		This helper is available only if the kernel was compiled with
// 		**CONFIG_NET** configuration option.
// 	Return
// 		Pointer to **struct bpf_sock**, or **NULL** in case of failure.
// 		For sockets with reuseport option, the **struct bpf_sock**
// 		result is from *reuse*\ **->socks**\ [] using the hash of the
// 		tuple.
//
// long bpf_tcp_check_syncookie(void *sk, void *iph, u32 iph_len, struct tcphdr *th, u32 th_len)
// 	Description
// 		Check whether *iph* and *th* contain a valid SYN cookie ACK for
// 		the listening socket in *sk*.
//
// 		*iph* points to the start of the IPv4 or IPv6 header, while
// 		*iph_len* contains **sizeof**\ (**struct iphdr**) or
// 		**sizeof**\ (**struct ipv6hdr**).
//
// 		*th* points to the start of the TCP header, while *th_len*
// 		contains the length of the TCP header (at least
// 		**sizeof**\ (**struct tcphdr**)).
// 	Return
// 		0 if *iph* and *th* are a valid SYN cookie ACK, or a negative
// 		error otherwise.
//
// long bpf_sysctl_get_name(struct bpf_sysctl *ctx, char *buf, size_t buf_len, u64 flags)
// 	Description
// 		Get name of sysctl in /proc/sys/ and copy it into provided by
// 		program buffer *buf* of size *buf_len*.
//
// 		The buffer is always NUL terminated, unless it's zero-sized.
//
// 		If *flags* is zero, full name (e.g. "net/ipv4/tcp_mem") is
// 		copied. Use **BPF_F_SYSCTL_BASE_NAME** flag to copy base name
// 		only (e.g. "tcp_mem").
// 	Return
// 		Number of character copied (not including the trailing NUL).
//
// 		**-E2BIG** if the buffer wasn't big enough (*buf* will contain
// 		truncated name in this case).
//
// long bpf_sysctl_get_current_value(struct bpf_sysctl *ctx, char *buf, size_t buf_len)
// 	Description
// 		Get current value of sysctl as it is presented in /proc/sys
// 		(incl. newline, etc), and copy it as a string into provided
// 		by program buffer *buf* of size *buf_len*.
//
// 		The whole value is copied, no matter what file position user
// 		space issued e.g. sys_read at.
//
// 		The buffer is always NUL terminated, unless it's zero-sized.
// 	Return
// 		Number of character copied (not including the trailing NUL).
//
// 		**-E2BIG** if the buffer wasn't big enough (*buf* will contain
// 		truncated name in this case).
//
// 		**-EINVAL** if current value was unavailable, e.g. because
// 		sysctl is uninitialized and read returns -EIO for it.
//
// long bpf_sysctl_get_new_value(struct bpf_sysctl *ctx, char *buf, size_t buf_len)
// 	Description
// 		Get new value being written by user space to sysctl (before
// 		the actual write happens) and copy it as a string into
// 		provided by program buffer *buf* of size *buf_len*.
//
// 		User space may write new value at file position > 0.
//
// 		The buffer is always NUL terminated, unless it's zero-sized.
// 	Return
// 		Number of character copied (not including the trailing NUL).
//
// 		**-E2BIG** if the buffer wasn't big enough (*buf* will contain
// 		truncated name in this case).
//
// 		**-EINVAL** if sysctl is being read.
//
// long bpf_sysctl_set_new_value(struct bpf_sysctl *ctx, const char *buf, size_t buf_len)
// 	Description
// 		Override new value being written by user space to sysctl with
// 		value provided by program in buffer *buf* of size *buf_len*.
//
// 		*buf* should contain a string in same form as provided by user
// 		space on sysctl write.
//
// 		User space may write new value at file position > 0. To override
// 		the whole sysctl value file position should be set to zero.
// 	Return
// 		0 on success.
//
// 		**-E2BIG** if the *buf_len* is too big.
//
// 		**-EINVAL** if sysctl is being read.
//
// long bpf_strtol(const char *buf, size_t buf_len, u64 flags, long *res)
// 	Description
// 		Convert the initial part of the string from buffer *buf* of
// 		size *buf_len* to a long integer according to the given base
// 		and save the result in *res*.
//
// 		The string may begin with an arbitrary amount of white space
// 		(as determined by **isspace**\ (3)) followed by a single
// 		optional '**-**' sign.
//
// 		Five least significant bits of *flags* encode base, other bits
// 		are currently unused.
//
// 		Base must be either 8, 10, 16 or 0 to detect it automatically
// 		similar to user space **strtol**\ (3).
// 	Return
// 		Number of characters consumed on success. Must be positive but
// 		no more than *buf_len*.
//
// 		**-EINVAL** if no valid digits were found or unsupported base
// 		was provided.
//
// 		**-ERANGE** if resulting value was out of range.
//
// long bpf_strtoul(const char *buf, size_t buf_len, u64 flags, unsigned long *res)
// 	Description
// 		Convert the initial part of the string from buffer *buf* of
// 		size *buf_len* to an unsigned long integer according to the
// 		given base and save the result in *res*.
//
// 		The string may begin with an arbitrary amount of white space
// 		(as determined by **isspace**\ (3)).
//
// 		Five least significant bits of *flags* encode base, other bits
// 		are currently unused.
//
// 		Base must be either 8, 10, 16 or 0 to detect it automatically
// 		similar to user space **strtoul**\ (3).
// 	Return
// 		Number of characters consumed on success. Must be positive but
// 		no more than *buf_len*.
//
// 		**-EINVAL** if no valid digits were found or unsupported base
// 		was provided.
//
// 		**-ERANGE** if resulting value was out of range.
//
// void *bpf_sk_storage_get(struct bpf_map *map, void *sk, void *value, u64 flags)
// 	Description
// 		Get a bpf-local-storage from a *sk*.
//
// 		Logically, it could be thought of getting the value from
// 		a *map* with *sk* as the **key**.  From this
// 		perspective,  the usage is not much different from
// 		**bpf_map_lookup_elem**\ (*map*, **&**\ *sk*) except this
// 		helper enforces the key must be a full socket and the map must
// 		be a **BPF_MAP_TYPE_SK_STORAGE** also.
//
// 		Underneath, the value is stored locally at *sk* instead of
// 		the *map*.  The *map* is used as the bpf-local-storage
// 		"type". The bpf-local-storage "type" (i.e. the *map*) is
// 		searched against all bpf-local-storages residing at *sk*.
//
// 		*sk* is a kernel **struct sock** pointer for LSM program.
// 		*sk* is a **struct bpf_sock** pointer for other program types.
//
// 		An optional *flags* (**BPF_SK_STORAGE_GET_F_CREATE**) can be
// 		used such that a new bpf-local-storage will be
// 		created if one does not exist.  *value* can be used
// 		together with **BPF_SK_STORAGE_GET_F_CREATE** to specify
// 		the initial value of a bpf-local-storage.  If *value* is
// 		**NULL**, the new bpf-local-storage will be zero initialized.
// 	Return
// 		A bpf-local-storage pointer is returned on success.
//
// 		**NULL** if not found or there was an error in adding
// 		a new bpf-local-storage.
//
// long bpf_sk_storage_delete(struct bpf_map *map, void *sk)
// 	Description
// 		Delete a bpf-local-storage from a *sk*.
// 	Return
// 		0 on success.
//
// 		**-ENOENT** if the bpf-local-storage cannot be found.
// 		**-EINVAL** if sk is not a fullsock (e.g. a request_sock).
//
// long bpf_send_signal(u32 sig)
// 	Description
// 		Send signal *sig* to the process of the current task.
// 		The signal may be delivered to any of this process's threads.
// 	Return
// 		0 on success or successfully queued.
//
// 		**-EBUSY** if work queue under nmi is full.
//
// 		**-EINVAL** if *sig* is invalid.
//
// 		**-EPERM** if no permission to send the *sig*.
//
// 		**-EAGAIN** if bpf program can try again.
//
// s64 bpf_tcp_gen_syncookie(void *sk, void *iph, u32 iph_len, struct tcphdr *th, u32 th_len)
// 	Description
// 		Try to issue a SYN cookie for the packet with corresponding
// 		IP/TCP headers, *iph* and *th*, on the listening socket in *sk*.
//
// 		*iph* points to the start of the IPv4 or IPv6 header, while
// 		*iph_len* contains **sizeof**\ (**struct iphdr**) or
// 		**sizeof**\ (**struct ipv6hdr**).
//
// 		*th* points to the start of the TCP header, while *th_len*
// 		contains the length of the TCP header with options (at least
// 		**sizeof**\ (**struct tcphdr**)).
// 	Return
// 		On success, lower 32 bits hold the generated SYN cookie in
// 		followed by 16 bits which hold the MSS value for that cookie,
// 		and the top 16 bits are unused.
//
// 		On failure, the returned value is one of the following:
//
// 		**-EINVAL** SYN cookie cannot be issued due to error
//
// 		**-ENOENT** SYN cookie should not be issued (no SYN flood)
//
// 		**-EOPNOTSUPP** kernel configuration does not enable SYN cookies
//
// 		**-EPROTONOSUPPORT** IP packet version is not 4 or 6
//
// long bpf_skb_output(void *ctx, struct bpf_map *map, u64 flags, void *data, u64 size)
// 	Description
// 		Write raw *data* blob into a special BPF perf event held by
// 		*map* of type **BPF_MAP_TYPE_PERF_EVENT_ARRAY**. This perf
// 		event must have the following attributes: **PERF_SAMPLE_RAW**
// 		as **sample_type**, **PERF_TYPE_SOFTWARE** as **type**, and
// 		**PERF_COUNT_SW_BPF_OUTPUT** as **config**.
//
// 		The *flags* are used to indicate the index in *map* for which
// 		the value must be put, masked with **BPF_F_INDEX_MASK**.
// 		Alternatively, *flags* can be set to **BPF_F_CURRENT_CPU**
// 		to indicate that the index of the current CPU core should be
// 		used.
//
// 		The value to write, of *size*, is passed through eBPF stack and
// 		pointed by *data*.
//
// 		*ctx* is a pointer to in-kernel struct sk_buff.
//
// 		This helper is similar to **bpf_perf_event_output**\ () but
// 		restricted to raw_tracepoint bpf programs.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_probe_read_user(void *dst, u32 size, const void *unsafe_ptr)
// 	Description
// 		Safely attempt to read *size* bytes from user space address
// 		*unsafe_ptr* and store the data in *dst*.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_probe_read_kernel(void *dst, u32 size, const void *unsafe_ptr)
// 	Description
// 		Safely attempt to read *size* bytes from kernel space address
// 		*unsafe_ptr* and store the data in *dst*.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_probe_read_user_str(void *dst, u32 size, const void *unsafe_ptr)
// 	Description
// 		Copy a NUL terminated string from an unsafe user address
// 		*unsafe_ptr* to *dst*. The *size* should include the
// 		terminating NUL byte. In case the string length is smaller than
// 		*size*, the target is not padded with further NUL bytes. If the
// 		string length is larger than *size*, just *size*-1 bytes are
// 		copied and the last byte is set to NUL.
//
// 		On success, returns the number of bytes that were written,
// 		including the terminal NUL. This makes this helper useful in
// 		tracing programs for reading strings, and more importantly to
// 		get its length at runtime. See the following snippet:
//
// 		::
//
// 			SEC("kprobe/sys_open")
// 			void bpf_sys_open(struct pt_regs *ctx)
// 			{
// 			        char buf[PATHLEN]; // PATHLEN is defined to 256
// 			        int res = bpf_probe_read_user_str(buf, sizeof(buf),
// 				                                  ctx->di);
//
// 				// Consume buf, for example push it to
// 				// userspace via bpf_perf_event_output(); we
// 				// can use res (the string length) as event
// 				// size, after checking its boundaries.
// 			}
//
// 		In comparison, using **bpf_probe_read_user**\ () helper here
// 		instead to read the string would require to estimate the length
// 		at compile time, and would often result in copying more memory
// 		than necessary.
//
// 		Another useful use case is when parsing individual process
// 		arguments or individual environment variables navigating
// 		*current*\ **->mm->arg_start** and *current* * 		**->mm->env_start**: using this helper and the return value,
// 		one can quickly iterate at the right offset of the memory area.
// 	Return
// 		On success, the strictly positive length of the output string,
// 		including the trailing NUL character. On error, a negative
// 		value.
//
// long bpf_probe_read_kernel_str(void *dst, u32 size, const void *unsafe_ptr)
// 	Description
// 		Copy a NUL terminated string from an unsafe kernel address *unsafe_ptr*
// 		to *dst*. Same semantics as with **bpf_probe_read_user_str**\ () apply.
// 	Return
// 		On success, the strictly positive length of the string, including
// 		the trailing NUL character. On error, a negative value.
//
// long bpf_tcp_send_ack(void *tp, u32 rcv_nxt)
// 	Description
// 		Send out a tcp-ack. *tp* is the in-kernel struct **tcp_sock**.
// 		*rcv_nxt* is the ack_seq to be sent out.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_send_signal_thread(u32 sig)
// 	Description
// 		Send signal *sig* to the thread corresponding to the current task.
// 	Return
// 		0 on success or successfully queued.
//
// 		**-EBUSY** if work queue under nmi is full.
//
// 		**-EINVAL** if *sig* is invalid.
//
// 		**-EPERM** if no permission to send the *sig*.
//
// 		**-EAGAIN** if bpf program can try again.
//
// u64 bpf_jiffies64(void)
// 	Description
// 		Obtain the 64bit jiffies
// 	Return
// 		The 64 bit jiffies
//
// long bpf_read_branch_records(struct bpf_perf_event_data *ctx, void *buf, u32 size, u64 flags)
// 	Description
// 		For an eBPF program attached to a perf event, retrieve the
// 		branch records (**struct perf_branch_entry**) associated to *ctx*
// 		and store it in the buffer pointed by *buf* up to size
// 		*size* bytes.
// 	Return
// 		On success, number of bytes written to *buf*. On error, a
// 		negative value.
//
// 		The *flags* can be set to **BPF_F_GET_BRANCH_RECORDS_SIZE** to
// 		instead return the number of bytes required to store all the
// 		branch entries. If this flag is set, *buf* may be NULL.
//
// 		**-EINVAL** if arguments invalid or **size** not a multiple
// 		of **sizeof**\ (**struct perf_branch_entry**\ ).
//
// 		**-ENOENT** if architecture does not support branch records.
//
// long bpf_get_ns_current_pid_tgid(u64 dev, u64 ino, struct bpf_pidns_info *nsdata, u32 size)
// 	Description
// 		Returns 0 on success, values for *pid* and *tgid* as seen from the current
// 		*namespace* will be returned in *nsdata*.
// 	Return
// 		0 on success, or one of the following in case of failure:
//
// 		**-EINVAL** if dev and inum supplied don't match dev_t and inode number
//              with nsfs of current task, or if dev conversion to dev_t lost high bits.
//
// 		**-ENOENT** if pidns does not exists for the current task.
//
// long bpf_xdp_output(void *ctx, struct bpf_map *map, u64 flags, void *data, u64 size)
// 	Description
// 		Write raw *data* blob into a special BPF perf event held by
// 		*map* of type **BPF_MAP_TYPE_PERF_EVENT_ARRAY**. This perf
// 		event must have the following attributes: **PERF_SAMPLE_RAW**
// 		as **sample_type**, **PERF_TYPE_SOFTWARE** as **type**, and
// 		**PERF_COUNT_SW_BPF_OUTPUT** as **config**.
//
// 		The *flags* are used to indicate the index in *map* for which
// 		the value must be put, masked with **BPF_F_INDEX_MASK**.
// 		Alternatively, *flags* can be set to **BPF_F_CURRENT_CPU**
// 		to indicate that the index of the current CPU core should be
// 		used.
//
// 		The value to write, of *size*, is passed through eBPF stack and
// 		pointed by *data*.
//
// 		*ctx* is a pointer to in-kernel struct xdp_buff.
//
// 		This helper is similar to **bpf_perf_eventoutput**\ () but
// 		restricted to raw_tracepoint bpf programs.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// u64 bpf_get_netns_cookie(void *ctx)
// 	Description
// 		Retrieve the cookie (generated by the kernel) of the network
// 		namespace the input *ctx* is associated with. The network
// 		namespace cookie remains stable for its lifetime and provides
// 		a global identifier that can be assumed unique. If *ctx* is
// 		NULL, then the helper returns the cookie for the initial
// 		network namespace. The cookie itself is very similar to that
// 		of **bpf_get_socket_cookie**\ () helper, but for network
// 		namespaces instead of sockets.
// 	Return
// 		A 8-byte long opaque number.
//
// u64 bpf_get_current_ancestor_cgroup_id(int ancestor_level)
// 	Description
// 		Return id of cgroup v2 that is ancestor of the cgroup associated
// 		with the current task at the *ancestor_level*. The root cgroup
// 		is at *ancestor_level* zero and each step down the hierarchy
// 		increments the level. If *ancestor_level* == level of cgroup
// 		associated with the current task, then return value will be the
// 		same as that of **bpf_get_current_cgroup_id**\ ().
//
// 		The helper is useful to implement policies based on cgroups
// 		that are upper in hierarchy than immediate cgroup associated
// 		with the current task.
//
// 		The format of returned id and helper limitations are same as in
// 		**bpf_get_current_cgroup_id**\ ().
// 	Return
// 		The id is returned or 0 in case the id could not be retrieved.
//
// long bpf_sk_assign(struct sk_buff *skb, void *sk, u64 flags)
// 	Description
// 		Helper is overloaded depending on BPF program type. This
// 		description applies to **BPF_PROG_TYPE_SCHED_CLS** and
// 		**BPF_PROG_TYPE_SCHED_ACT** programs.
//
// 		Assign the *sk* to the *skb*. When combined with appropriate
// 		routing configuration to receive the packet towards the socket,
// 		will cause *skb* to be delivered to the specified socket.
// 		Subsequent redirection of *skb* via  **bpf_redirect**\ (),
// 		**bpf_clone_redirect**\ () or other methods outside of BPF may
// 		interfere with successful delivery to the socket.
//
// 		This operation is only valid from TC ingress path.
//
// 		The *flags* argument must be zero.
// 	Return
// 		0 on success, or a negative error in case of failure:
//
// 		**-EINVAL** if specified *flags* are not supported.
//
// 		**-ENOENT** if the socket is unavailable for assignment.
//
// 		**-ENETUNREACH** if the socket is unreachable (wrong netns).
//
// 		**-EOPNOTSUPP** if the operation is not supported, for example
// 		a call from outside of TC ingress.
//
// long bpf_sk_assign(struct bpf_sk_lookup *ctx, struct bpf_sock *sk, u64 flags)
// 	Description
// 		Helper is overloaded depending on BPF program type. This
// 		description applies to **BPF_PROG_TYPE_SK_LOOKUP** programs.
//
// 		Select the *sk* as a result of a socket lookup.
//
// 		For the operation to succeed passed socket must be compatible
// 		with the packet description provided by the *ctx* object.
//
// 		L4 protocol (**IPPROTO_TCP** or **IPPROTO_UDP**) must
// 		be an exact match. While IP family (**AF_INET** or
// 		**AF_INET6**) must be compatible, that is IPv6 sockets
// 		that are not v6-only can be selected for IPv4 packets.
//
// 		Only TCP listeners and UDP unconnected sockets can be
// 		selected. *sk* can also be NULL to reset any previous
// 		selection.
//
// 		*flags* argument can combination of following values:
//
// 		* **BPF_SK_LOOKUP_F_REPLACE** to override the previous
// 		  socket selection, potentially done by a BPF program
// 		  that ran before us.
//
// 		* **BPF_SK_LOOKUP_F_NO_REUSEPORT** to skip
// 		  load-balancing within reuseport group for the socket
// 		  being selected.
//
// 		On success *ctx->sk* will point to the selected socket.
//
// 	Return
// 		0 on success, or a negative errno in case of failure.
//
// 		* **-EAFNOSUPPORT** if socket family (*sk->family*) is
// 		  not compatible with packet family (*ctx->family*).
//
// 		* **-EEXIST** if socket has been already selected,
// 		  potentially by another program, and
// 		  **BPF_SK_LOOKUP_F_REPLACE** flag was not specified.
//
// 		* **-EINVAL** if unsupported flags were specified.
//
// 		* **-EPROTOTYPE** if socket L4 protocol
// 		  (*sk->protocol*) doesn't match packet protocol
// 		  (*ctx->protocol*).
//
// 		* **-ESOCKTNOSUPPORT** if socket is not in allowed
// 		  state (TCP listening or UDP unconnected).
//
// u64 bpf_ktime_get_boot_ns(void)
// 	Description
// 		Return the time elapsed since system boot, in nanoseconds.
// 		Does include the time the system was suspended.
// 		See: **clock_gettime**\ (**CLOCK_BOOTTIME**)
// 	Return
// 		Current *ktime*.
//
// long bpf_seq_printf(struct seq_file *m, const char *fmt, u32 fmt_size, const void *data, u32 data_len)
// 	Description
// 		**bpf_seq_printf**\ () uses seq_file **seq_printf**\ () to print
// 		out the format string.
// 		The *m* represents the seq_file. The *fmt* and *fmt_size* are for
// 		the format string itself. The *data* and *data_len* are format string
// 		arguments. The *data* are a **u64** array and corresponding format string
// 		values are stored in the array. For strings and pointers where pointees
// 		are accessed, only the pointer values are stored in the *data* array.
// 		The *data_len* is the size of *data* in bytes - must be a multiple of 8.
//
// 		Formats **%s**, **%p{i,I}{4,6}** requires to read kernel memory.
// 		Reading kernel memory may fail due to either invalid address or
// 		valid address but requiring a major memory fault. If reading kernel memory
// 		fails, the string for **%s** will be an empty string, and the ip
// 		address for **%p{i,I}{4,6}** will be 0. Not returning error to
// 		bpf program is consistent with what **bpf_trace_printk**\ () does for now.
// 	Return
// 		0 on success, or a negative error in case of failure:
//
// 		**-EBUSY** if per-CPU memory copy buffer is busy, can try again
// 		by returning 1 from bpf program.
//
// 		**-EINVAL** if arguments are invalid, or if *fmt* is invalid/unsupported.
//
// 		**-E2BIG** if *fmt* contains too many format specifiers.
//
// 		**-EOVERFLOW** if an overflow happened: The same object will be tried again.
//
// long bpf_seq_write(struct seq_file *m, const void *data, u32 len)
// 	Description
// 		**bpf_seq_write**\ () uses seq_file **seq_write**\ () to write the data.
// 		The *m* represents the seq_file. The *data* and *len* represent the
// 		data to write in bytes.
// 	Return
// 		0 on success, or a negative error in case of failure:
//
// 		**-EOVERFLOW** if an overflow happened: The same object will be tried again.
//
// u64 bpf_sk_cgroup_id(void *sk)
// 	Description
// 		Return the cgroup v2 id of the socket *sk*.
//
// 		*sk* must be a non-**NULL** pointer to a socket, e.g. one
// 		returned from **bpf_sk_lookup_xxx**\ (),
// 		**bpf_sk_fullsock**\ (), etc. The format of returned id is
// 		same as in **bpf_skb_cgroup_id**\ ().
//
// 		This helper is available only if the kernel was compiled with
// 		the **CONFIG_SOCK_CGROUP_DATA** configuration option.
// 	Return
// 		The id is returned or 0 in case the id could not be retrieved.
//
// u64 bpf_sk_ancestor_cgroup_id(void *sk, int ancestor_level)
// 	Description
// 		Return id of cgroup v2 that is ancestor of cgroup associated
// 		with the *sk* at the *ancestor_level*.  The root cgroup is at
// 		*ancestor_level* zero and each step down the hierarchy
// 		increments the level. If *ancestor_level* == level of cgroup
// 		associated with *sk*, then return value will be same as that
// 		of **bpf_sk_cgroup_id**\ ().
//
// 		The helper is useful to implement policies based on cgroups
// 		that are upper in hierarchy than immediate cgroup associated
// 		with *sk*.
//
// 		The format of returned id and helper limitations are same as in
// 		**bpf_sk_cgroup_id**\ ().
// 	Return
// 		The id is returned or 0 in case the id could not be retrieved.
//
// long bpf_ringbuf_output(void *ringbuf, void *data, u64 size, u64 flags)
// 	Description
// 		Copy *size* bytes from *data* into a ring buffer *ringbuf*.
// 		If **BPF_RB_NO_WAKEUP** is specified in *flags*, no notification
// 		of new data availability is sent.
// 		If **BPF_RB_FORCE_WAKEUP** is specified in *flags*, notification
// 		of new data availability is sent unconditionally.
// 		If **0** is specified in *flags*, an adaptive notification
// 		of new data availability is sent.
//
// 		An adaptive notification is a notification sent whenever the user-space
// 		process has caught up and consumed all available payloads. In case the user-space
// 		process is still processing a previous payload, then no notification is needed
// 		as it will process the newly added payload automatically.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// void *bpf_ringbuf_reserve(void *ringbuf, u64 size, u64 flags)
// 	Description
// 		Reserve *size* bytes of payload in a ring buffer *ringbuf*.
// 		*flags* must be 0.
// 	Return
// 		Valid pointer with *size* bytes of memory available; NULL,
// 		otherwise.
//
// void bpf_ringbuf_submit(void *data, u64 flags)
// 	Description
// 		Submit reserved ring buffer sample, pointed to by *data*.
// 		If **BPF_RB_NO_WAKEUP** is specified in *flags*, no notification
// 		of new data availability is sent.
// 		If **BPF_RB_FORCE_WAKEUP** is specified in *flags*, notification
// 		of new data availability is sent unconditionally.
// 		If **0** is specified in *flags*, an adaptive notification
// 		of new data availability is sent.
//
// 		See 'bpf_ringbuf_output()' for the definition of adaptive notification.
// 	Return
// 		Nothing. Always succeeds.
//
// void bpf_ringbuf_discard(void *data, u64 flags)
// 	Description
// 		Discard reserved ring buffer sample, pointed to by *data*.
// 		If **BPF_RB_NO_WAKEUP** is specified in *flags*, no notification
// 		of new data availability is sent. Discarded records remain in
// 		the ring buffer until consumed by user space, so a later submit
// 		using adaptive wakeup might not wake up the consumer.
// 		If **BPF_RB_FORCE_WAKEUP** is specified in *flags*, notification
// 		of new data availability is sent unconditionally.
// 		If **0** is specified in *flags*, an adaptive notification
// 		of new data availability is sent.
//
// 		See 'bpf_ringbuf_output()' for the definition of adaptive notification.
// 	Return
// 		Nothing. Always succeeds.
//
// u64 bpf_ringbuf_query(void *ringbuf, u64 flags)
// 	Description
// 		Query various characteristics of provided ring buffer. What
// 		exactly is queries is determined by *flags*:
//
// 		* **BPF_RB_AVAIL_DATA**: Amount of data not yet consumed.
// 		* **BPF_RB_RING_SIZE**: The size of ring buffer.
// 		* **BPF_RB_CONS_POS**: Consumer position (can wrap around).
// 		* **BPF_RB_PROD_POS**: Producer(s) position (can wrap around).
// 		* **BPF_RB_OVERWRITE_POS**: Overwrite position (can wrap around).
//
// 		Data returned is just a momentary snapshot of actual values
// 		and could be inaccurate, so this facility should be used to
// 		power heuristics and for reporting, not to make 100% correct
// 		calculation.
// 	Return
// 		Requested value, or 0, if *flags* are not recognized.
//
// long bpf_csum_level(struct sk_buff *skb, u64 level)
// 	Description
// 		Change the skbs checksum level by one layer up or down, or
// 		reset it entirely to none in order to have the stack perform
// 		checksum validation. The level is applicable to the following
// 		protocols: TCP, UDP, GRE, SCTP, FCOE. For example, a decap of
// 		| ETH | IP | UDP | GUE | IP | TCP | into | ETH | IP | TCP |
// 		through **bpf_skb_adjust_room**\ () helper with passing in
// 		**BPF_F_ADJ_ROOM_NO_CSUM_RESET** flag would require one	call
// 		to **bpf_csum_level**\ () with **BPF_CSUM_LEVEL_DEC** since
// 		the UDP header is removed. Similarly, an encap of the latter
// 		into the former could be accompanied by a helper call to
// 		**bpf_csum_level**\ () with **BPF_CSUM_LEVEL_INC** if the
// 		skb is still intended to be processed in higher layers of the
// 		stack instead of just egressing at tc.
//
// 		There are three supported level settings at this time:
//
// 		* **BPF_CSUM_LEVEL_INC**: Increases skb->csum_level for skbs
// 		  with CHECKSUM_UNNECESSARY.
// 		* **BPF_CSUM_LEVEL_DEC**: Decreases skb->csum_level for skbs
// 		  with CHECKSUM_UNNECESSARY.
// 		* **BPF_CSUM_LEVEL_RESET**: Resets skb->csum_level to 0 and
// 		  sets CHECKSUM_NONE to force checksum validation by the stack.
// 		* **BPF_CSUM_LEVEL_QUERY**: No-op, returns the current
// 		  skb->csum_level.
// 	Return
// 		0 on success, or a negative error in case of failure. In the
// 		case of **BPF_CSUM_LEVEL_QUERY**, the current skb->csum_level
// 		is returned or the error code -EACCES in case the skb is not
// 		subject to CHECKSUM_UNNECESSARY.
//
// struct tcp6_sock *bpf_skc_to_tcp6_sock(void *sk)
// 	Description
// 		Dynamically cast a *sk* pointer to a *tcp6_sock* pointer.
// 	Return
// 		*sk* if casting is valid, or **NULL** otherwise.
//
// struct tcp_sock *bpf_skc_to_tcp_sock(void *sk)
// 	Description
// 		Dynamically cast a *sk* pointer to a *tcp_sock* pointer.
// 	Return
// 		*sk* if casting is valid, or **NULL** otherwise.
//
// struct tcp_timewait_sock *bpf_skc_to_tcp_timewait_sock(void *sk)
// 	Description
// 		Dynamically cast a *sk* pointer to a *tcp_timewait_sock* pointer.
// 	Return
// 		*sk* if casting is valid, or **NULL** otherwise.
//
// struct tcp_request_sock *bpf_skc_to_tcp_request_sock(void *sk)
// 	Description
// 		Dynamically cast a *sk* pointer to a *tcp_request_sock* pointer.
// 	Return
// 		*sk* if casting is valid, or **NULL** otherwise.
//
// struct udp6_sock *bpf_skc_to_udp6_sock(void *sk)
// 	Description
// 		Dynamically cast a *sk* pointer to a *udp6_sock* pointer.
// 	Return
// 		*sk* if casting is valid, or **NULL** otherwise.
//
// long bpf_get_task_stack(struct task_struct *task, void *buf, u32 size, u64 flags)
// 	Description
// 		Return a user or a kernel stack in bpf program provided buffer.
// 		Note: the user stack will only be populated if the *task* is
// 		the current task; all other tasks will return -EOPNOTSUPP.
// 		To achieve this, the helper needs *task*, which is a valid
// 		pointer to **struct task_struct**. To store the stacktrace, the
// 		bpf program provides *buf* with a nonnegative *size*.
//
// 		The last argument, *flags*, holds the number of stack frames to
// 		skip (from 0 to 255), masked with
// 		**BPF_F_SKIP_FIELD_MASK**. The next bits can be used to set
// 		the following flags:
//
// 		**BPF_F_USER_STACK**
// 			Collect a user space stack instead of a kernel stack.
// 			The *task* must be the current task.
// 		**BPF_F_USER_BUILD_ID**
// 			Collect buildid+offset instead of ips for user stack,
// 			only valid if **BPF_F_USER_STACK** is also specified.
//
// 		**bpf_get_task_stack**\ () can collect up to
// 		**PERF_MAX_STACK_DEPTH** both kernel and user frames, subject
// 		to sufficient large buffer size. Note that
// 		this limit can be controlled with the **sysctl** program, and
// 		that it should be manually increased in order to profile long
// 		user stacks (such as stacks for Java programs). To do so, use:
//
// 		::
//
// 			# sysctl kernel.perf_event_max_stack=<new value>
// 	Return
// 		The non-negative copied *buf* length equal to or less than
// 		*size* on success, or a negative error in case of failure.
//
// long bpf_load_hdr_opt(struct bpf_sock_ops *skops, void *searchby_res, u32 len, u64 flags)
// 	Description
// 		Load header option.  Support reading a particular TCP header
// 		option for bpf program (**BPF_PROG_TYPE_SOCK_OPS**).
//
// 		If *flags* is 0, it will search the option from the
// 		*skops*\ **->skb_data**.  The comment in **struct bpf_sock_ops**
// 		has details on what skb_data contains under different
// 		*skops*\ **->op**.
//
// 		The first byte of the *searchby_res* specifies the
// 		kind that it wants to search.
//
// 		If the searching kind is an experimental kind
// 		(i.e. 253 or 254 according to RFC6994).  It also
// 		needs to specify the "magic" which is either
// 		2 bytes or 4 bytes.  It then also needs to
// 		specify the size of the magic by using
// 		the 2nd byte which is "kind-length" of a TCP
// 		header option and the "kind-length" also
// 		includes the first 2 bytes "kind" and "kind-length"
// 		itself as a normal TCP header option also does.
//
// 		For example, to search experimental kind 254 with
// 		2 byte magic 0xeB9F, the searchby_res should be
// 		[ 254, 4, 0xeB, 0x9F, 0, 0, .... 0 ].
//
// 		To search for the standard window scale option (3),
// 		the *searchby_res* should be [ 3, 0, 0, .... 0 ].
// 		Note, kind-length must be 0 for regular option.
//
// 		Searching for No-Op (0) and End-of-Option-List (1) are
// 		not supported.
//
// 		*len* must be at least 2 bytes which is the minimal size
// 		of a header option.
//
// 		Supported flags:
//
// 		* **BPF_LOAD_HDR_OPT_TCP_SYN** to search from the
// 		  saved_syn packet or the just-received syn packet.
//
// 	Return
// 		> 0 when found, the header option is copied to *searchby_res*.
// 		The return value is the total length copied. On failure, a
// 		negative error code is returned:
//
// 		**-EINVAL** if a parameter is invalid.
//
// 		**-ENOMSG** if the option is not found.
//
// 		**-ENOENT** if no syn packet is available when
// 		**BPF_LOAD_HDR_OPT_TCP_SYN** is used.
//
// 		**-ENOSPC** if there is not enough space.  Only *len* number of
// 		bytes are copied.
//
// 		**-EFAULT** on failure to parse the header options in the
// 		packet.
//
// 		**-EPERM** if the helper cannot be used under the current
// 		*skops*\ **->op**.
//
// long bpf_store_hdr_opt(struct bpf_sock_ops *skops, const void *from, u32 len, u64 flags)
// 	Description
// 		Store header option.  The data will be copied
// 		from buffer *from* with length *len* to the TCP header.
//
// 		The buffer *from* should have the whole option that
// 		includes the kind, kind-length, and the actual
// 		option data.  The *len* must be at least kind-length
// 		long.  The kind-length does not have to be 4 byte
// 		aligned.  The kernel will take care of the padding
// 		and setting the 4 bytes aligned value to th->doff.
//
// 		This helper will check for duplicated option
// 		by searching the same option in the outgoing skb.
//
// 		This helper can only be called during
// 		**BPF_SOCK_OPS_WRITE_HDR_OPT_CB**.
//
// 	Return
// 		0 on success, or negative error in case of failure:
//
// 		**-EINVAL** If param is invalid.
//
// 		**-ENOSPC** if there is not enough space in the header.
// 		Nothing has been written
//
// 		**-EEXIST** if the option already exists.
//
// 		**-EFAULT** on failure to parse the existing header options.
//
// 		**-EPERM** if the helper cannot be used under the current
// 		*skops*\ **->op**.
//
// long bpf_reserve_hdr_opt(struct bpf_sock_ops *skops, u32 len, u64 flags)
// 	Description
// 		Reserve *len* bytes for the bpf header option.  The
// 		space will be used by **bpf_store_hdr_opt**\ () later in
// 		**BPF_SOCK_OPS_WRITE_HDR_OPT_CB**.
//
// 		If **bpf_reserve_hdr_opt**\ () is called multiple times,
// 		the total number of bytes will be reserved.
//
// 		This helper can only be called during
// 		**BPF_SOCK_OPS_HDR_OPT_LEN_CB**.
//
// 	Return
// 		0 on success, or negative error in case of failure:
//
// 		**-EINVAL** if a parameter is invalid.
//
// 		**-ENOSPC** if there is not enough space in the header.
//
// 		**-EPERM** if the helper cannot be used under the current
// 		*skops*\ **->op**.
//
// void *bpf_inode_storage_get(struct bpf_map *map, void *inode, void *value, u64 flags)
// 	Description
// 		Get a bpf_local_storage from an *inode*.
//
// 		Logically, it could be thought of as getting the value from
// 		a *map* with *inode* as the **key**.  From this
// 		perspective,  the usage is not much different from
// 		**bpf_map_lookup_elem**\ (*map*, **&**\ *inode*) except this
// 		helper enforces the key must be an inode and the map must also
// 		be a **BPF_MAP_TYPE_INODE_STORAGE**.
//
// 		Underneath, the value is stored locally at *inode* instead of
// 		the *map*.  The *map* is used as the bpf-local-storage
// 		"type". The bpf-local-storage "type" (i.e. the *map*) is
// 		searched against all bpf_local_storage residing at *inode*.
//
// 		An optional *flags* (**BPF_LOCAL_STORAGE_GET_F_CREATE**) can be
// 		used such that a new bpf_local_storage will be
// 		created if one does not exist.  *value* can be used
// 		together with **BPF_LOCAL_STORAGE_GET_F_CREATE** to specify
// 		the initial value of a bpf_local_storage.  If *value* is
// 		**NULL**, the new bpf_local_storage will be zero initialized.
// 	Return
// 		A bpf_local_storage pointer is returned on success.
//
// 		**NULL** if not found or there was an error in adding
// 		a new bpf_local_storage.
//
// int bpf_inode_storage_delete(struct bpf_map *map, void *inode)
// 	Description
// 		Delete a bpf_local_storage from an *inode*.
// 	Return
// 		0 on success.
//
// 		**-ENOENT** if the bpf_local_storage cannot be found.
//
// long bpf_d_path(const struct path *path, char *buf, u32 sz)
// 	Description
// 		Return full path for given **struct path** object, which
// 		needs to be the kernel BTF *path* object. The path is
// 		returned in the provided buffer *buf* of size *sz* and
// 		is zero terminated.
//
// 	Return
// 		On success, the strictly positive length of the string,
// 		including the trailing NUL character. On error, a negative
// 		value.
//
// long bpf_copy_from_user(void *dst, u32 size, const void *user_ptr)
// 	Description
// 		Read *size* bytes from user space address *user_ptr* and store
// 		the data in *dst*. This is a wrapper of **copy_from_user**\ ().
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_snprintf_btf(char *str, u32 str_size, struct btf_ptr *ptr, u32 btf_ptr_size, u64 flags)
// 	Description
// 		Use BTF to store a string representation of *ptr*->ptr in *str*,
// 		using *ptr*->type_id.  This value should specify the type
// 		that *ptr*->ptr points to. LLVM __builtin_btf_type_id(type, 1)
// 		can be used to look up vmlinux BTF type ids. Traversing the
// 		data structure using BTF, the type information and values are
// 		stored in the first *str_size* - 1 bytes of *str*.  Safe copy of
// 		the pointer data is carried out to avoid kernel crashes during
// 		operation.  Smaller types can use string space on the stack;
// 		larger programs can use map data to store the string
// 		representation.
//
// 		The string can be subsequently shared with userspace via
// 		bpf_perf_event_output() or ring buffer interfaces.
// 		bpf_trace_printk() is to be avoided as it places too small
// 		a limit on string size to be useful.
//
// 		*flags* is a combination of
//
// 		**BTF_F_COMPACT**
// 			no formatting around type information
// 		**BTF_F_NONAME**
// 			no struct/union member names/types
// 		**BTF_F_PTR_RAW**
// 			show raw (unobfuscated) pointer values;
// 			equivalent to printk specifier %px.
// 		**BTF_F_ZERO**
// 			show zero-valued struct/union members; they
// 			are not displayed by default
//
// 	Return
// 		The number of bytes that were written (or would have been
// 		written if output had to be truncated due to string size),
// 		or a negative error in cases of failure.
//
// long bpf_seq_printf_btf(struct seq_file *m, struct btf_ptr *ptr, u32 ptr_size, u64 flags)
// 	Description
// 		Use BTF to write to seq_write a string representation of
// 		*ptr*->ptr, using *ptr*->type_id as per bpf_snprintf_btf().
// 		*flags* are identical to those used for bpf_snprintf_btf.
// 	Return
// 		0 on success or a negative error in case of failure.
//
// u64 bpf_skb_cgroup_classid(struct sk_buff *skb)
// 	Description
// 		See **bpf_get_cgroup_classid**\ () for the main description.
// 		This helper differs from **bpf_get_cgroup_classid**\ () in that
// 		the cgroup v1 net_cls class is retrieved only from the *skb*'s
// 		associated socket instead of the current process.
// 	Return
// 		The id is returned or 0 in case the id could not be retrieved.
//
// long bpf_redirect_neigh(u32 ifindex, struct bpf_redir_neigh *params, int plen, u64 flags)
// 	Description
// 		Redirect the packet to another net device of index *ifindex*
// 		and fill in L2 addresses from neighboring subsystem. This helper
// 		is somewhat similar to **bpf_redirect**\ (), except that it
// 		populates L2 addresses as well, meaning, internally, the helper
// 		relies on the neighbor lookup for the L2 address of the nexthop.
//
// 		The helper will perform a FIB lookup based on the skb's
// 		networking header to get the address of the next hop, unless
// 		this is supplied by the caller in the *params* argument. The
// 		*plen* argument indicates the len of *params* and should be set
// 		to 0 if *params* is NULL.
//
// 		The *flags* argument is reserved and must be 0. The helper is
// 		currently only supported for tc BPF program types, and enabled
// 		for IPv4 and IPv6 protocols.
// 	Return
// 		The helper returns **TC_ACT_REDIRECT** on success or
// 		**TC_ACT_SHOT** on error.
//
// void *bpf_per_cpu_ptr(const void *percpu_ptr, u32 cpu)
//     Description
//             Take a pointer to a percpu ksym, *percpu_ptr*, and return a
//             pointer to the percpu kernel variable on *cpu*. A ksym is an
//             extern variable decorated with '__ksym'. For ksym, there is a
//             global var (either static or global) defined of the same name
//             in the kernel. The ksym is percpu if the global var is percpu.
//             The returned pointer points to the global percpu var on *cpu*.
//
//             bpf_per_cpu_ptr() has the same semantic as per_cpu_ptr() in the
//             kernel, except that bpf_per_cpu_ptr() may return NULL. This
//             happens if *cpu* is larger than nr_cpu_ids. The caller of
//             bpf_per_cpu_ptr() must check the returned value.
//     Return
//             A pointer pointing to the kernel percpu variable on *cpu*, or
//             NULL, if *cpu* is invalid.
//
// void *bpf_this_cpu_ptr(const void *percpu_ptr)
// 	Description
// 		Take a pointer to a percpu ksym, *percpu_ptr*, and return a
// 		pointer to the percpu kernel variable on this cpu. See the
// 		description of 'ksym' in **bpf_per_cpu_ptr**\ ().
//
// 		bpf_this_cpu_ptr() has the same semantic as this_cpu_ptr() in
// 		the kernel. Different from **bpf_per_cpu_ptr**\ (), it would
// 		never return NULL.
// 	Return
// 		A pointer pointing to the kernel percpu variable on this cpu.
//
// long bpf_redirect_peer(u32 ifindex, u64 flags)
// 	Description
// 		Redirect the packet to another net device of index *ifindex*.
// 		This helper is somewhat similar to **bpf_redirect**\ (), except
// 		that the redirection happens to the *ifindex*' peer device. If
// 		*flags* is 0, the netns switch takes place from ingress to
// 		ingress without going through the CPU's backlog queue. If the
// 		**BPF_F_EGRESS** flag is provided then redirection happens in
// 		the egress direction of the peer device.
//
// 		*skb*\ **->mark** and *skb*\ **->tstamp** are not cleared during
// 		the netns switch.
//
// 		If the *flags* argument is 0, the helper is currently only
// 		supported for tc BPF program types at the ingress hook and for
// 		veth and netkit target device types. The peer device must reside
// 		in a different network namespace.
// 	Return
// 		The helper returns **TC_ACT_REDIRECT** on success or
// 		**TC_ACT_SHOT** on error.
//
// void *bpf_task_storage_get(struct bpf_map *map, struct task_struct *task, void *value, u64 flags)
// 	Description
// 		Get a bpf_local_storage from the *task*.
//
// 		Logically, it could be thought of as getting the value from
// 		a *map* with *task* as the **key**.  From this
// 		perspective,  the usage is not much different from
// 		**bpf_map_lookup_elem**\ (*map*, **&**\ *task*) except this
// 		helper enforces the key must be a task_struct and the map must also
// 		be a **BPF_MAP_TYPE_TASK_STORAGE**.
//
// 		Underneath, the value is stored locally at *task* instead of
// 		the *map*.  The *map* is used as the bpf-local-storage
// 		"type". The bpf-local-storage "type" (i.e. the *map*) is
// 		searched against all bpf_local_storage residing at *task*.
//
// 		An optional *flags* (**BPF_LOCAL_STORAGE_GET_F_CREATE**) can be
// 		used such that a new bpf_local_storage will be
// 		created if one does not exist.  *value* can be used
// 		together with **BPF_LOCAL_STORAGE_GET_F_CREATE** to specify
// 		the initial value of a bpf_local_storage.  If *value* is
// 		**NULL**, the new bpf_local_storage will be zero initialized.
// 	Return
// 		A bpf_local_storage pointer is returned on success.
//
// 		**NULL** if not found or there was an error in adding
// 		a new bpf_local_storage.
//
// long bpf_task_storage_delete(struct bpf_map *map, struct task_struct *task)
// 	Description
// 		Delete a bpf_local_storage from a *task*.
// 	Return
// 		0 on success.
//
// 		**-ENOENT** if the bpf_local_storage cannot be found.
//
// struct task_struct *bpf_get_current_task_btf(void)
// 	Description
// 		Return a BTF pointer to the "current" task.
// 		This pointer can also be used in helpers that accept an
// 		*ARG_PTR_TO_BTF_ID* of type *task_struct*.
// 	Return
// 		Pointer to the current task.
//
// long bpf_bprm_opts_set(struct linux_binprm *bprm, u64 flags)
// 	Description
// 		Set or clear certain options on *bprm*:
//
// 		**BPF_F_BPRM_SECUREEXEC** Set the secureexec bit
// 		which sets the **AT_SECURE** auxv for glibc. The bit
// 		is cleared if the flag is not specified.
// 	Return
// 		**-EINVAL** if invalid *flags* are passed, zero otherwise.
//
// u64 bpf_ktime_get_coarse_ns(void)
// 	Description
// 		Return a coarse-grained version of the time elapsed since
// 		system boot, in nanoseconds. Does not include time the system
// 		was suspended.
//
// 		See: **clock_gettime**\ (**CLOCK_MONOTONIC_COARSE**)
// 	Return
// 		Current *ktime*.
//
// long bpf_ima_inode_hash(struct inode *inode, void *dst, u32 size)
// 	Description
// 		Returns the stored IMA hash of the *inode* (if it's available).
// 		If the hash is larger than *size*, then only *size*
// 		bytes will be copied to *dst*
// 	Return
// 		The **hash_algo** is returned on success,
// 		**-EOPNOTSUPP** if IMA is disabled or **-EINVAL** if
// 		invalid arguments are passed.
//
// struct socket *bpf_sock_from_file(struct file *file)
// 	Description
// 		If the given file represents a socket, returns the associated
// 		socket.
// 	Return
// 		A pointer to a struct socket on success or NULL if the file is
// 		not a socket.
//
// long bpf_check_mtu(void *ctx, u32 ifindex, u32 *mtu_len, s32 len_diff, u64 flags)
// 	Description
// 		Check packet size against exceeding MTU of net device (based
// 		on *ifindex*).  This helper will likely be used in combination
// 		with helpers that adjust/change the packet size.
//
// 		The argument *len_diff* can be used for querying with a planned
// 		size change. This allows to check MTU prior to changing packet
// 		ctx. Providing a *len_diff* adjustment that is larger than the
// 		actual packet size (resulting in negative packet size) will in
// 		principle not exceed the MTU, which is why it is not considered
// 		a failure.  Other BPF helpers are needed for performing the
// 		planned size change; therefore the responsibility for catching
// 		a negative packet size belongs in those helpers.
//
// 		Specifying *ifindex* zero means the MTU check is performed
// 		against the current net device.  This is practical if this isn't
// 		used prior to redirect.
//
// 		On input *mtu_len* must be a valid pointer, else verifier will
// 		reject BPF program.  If the value *mtu_len* is initialized to
// 		zero then the ctx packet size is use.  When value *mtu_len* is
// 		provided as input this specify the L3 length that the MTU check
// 		is done against. Remember XDP and TC length operate at L2, but
// 		this value is L3 as this correlate to MTU and IP-header tot_len
// 		values which are L3 (similar behavior as bpf_fib_lookup).
//
// 		The Linux kernel route table can configure MTUs on a more
// 		specific per route level, which is not provided by this helper.
// 		For route level MTU checks use the **bpf_fib_lookup**\ ()
// 		helper.
//
// 		*ctx* is either **struct xdp_md** for XDP programs or
// 		**struct sk_buff** for tc cls_act programs.
//
// 		The *flags* argument can be a combination of one or more of the
// 		following values:
//
// 		**BPF_MTU_CHK_SEGS**
// 			This flag will only works for *ctx* **struct sk_buff**.
// 			If packet context contains extra packet segment buffers
// 			(often knows as GSO skb), then MTU check is harder to
// 			check at this point, because in transmit path it is
// 			possible for the skb packet to get re-segmented
// 			(depending on net device features).  This could still be
// 			a MTU violation, so this flag enables performing MTU
// 			check against segments, with a different violation
// 			return code to tell it apart. Check cannot use len_diff.
//
// 		On return *mtu_len* pointer contains the MTU value of the net
// 		device.  Remember the net device configured MTU is the L3 size,
// 		which is returned here and XDP and TC length operate at L2.
// 		Helper take this into account for you, but remember when using
// 		MTU value in your BPF-code.
//
// 	Return
// 		* 0 on success, and populate MTU value in *mtu_len* pointer.
//
// 		* < 0 if any input argument is invalid (*mtu_len* not updated)
//
// 		MTU violations return positive values, but also populate MTU
// 		value in *mtu_len* pointer, as this can be needed for
// 		implementing PMTU handing:
//
// 		* **BPF_MTU_CHK_RET_FRAG_NEEDED**
// 		* **BPF_MTU_CHK_RET_SEGS_TOOBIG**
//
// long bpf_for_each_map_elem(struct bpf_map *map, void *callback_fn, void *callback_ctx, u64 flags)
// 	Description
// 		For each element in **map**, call **callback_fn** function with
// 		**map**, **callback_ctx** and other map-specific parameters.
// 		The **callback_fn** should be a static function and
// 		the **callback_ctx** should be a pointer to the stack.
// 		The **flags** is used to control certain aspects of the helper.
// 		Currently, the **flags** must be 0.
//
// 		The following are a list of supported map types and their
// 		respective expected callback signatures:
//
// 		BPF_MAP_TYPE_HASH, BPF_MAP_TYPE_PERCPU_HASH,
// 		BPF_MAP_TYPE_LRU_HASH, BPF_MAP_TYPE_LRU_PERCPU_HASH,
// 		BPF_MAP_TYPE_ARRAY, BPF_MAP_TYPE_PERCPU_ARRAY
//
// 		long (\*callback_fn)(struct bpf_map \*map, const void \*key, void \*value, void \*ctx);
//
// 		For per_cpu maps, the map_value is the value on the cpu where the
// 		bpf_prog is running.
//
// 		If **callback_fn** return 0, the helper will continue to the next
// 		element. If return value is 1, the helper will skip the rest of
// 		elements and return. Other return values are not used now.
//
// 	Return
// 		The number of traversed map elements for success, **-EINVAL** for
// 		invalid **flags**.
//
// long bpf_snprintf(char *str, u32 str_size, const char *fmt, u64 *data, u32 data_len)
// 	Description
// 		Outputs a string into the **str** buffer of size **str_size**
// 		based on a format string stored in a read-only map pointed by
// 		**fmt**.
//
// 		Each format specifier in **fmt** corresponds to one u64 element
// 		in the **data** array. For strings and pointers where pointees
// 		are accessed, only the pointer values are stored in the *data*
// 		array. The *data_len* is the size of *data* in bytes - must be
// 		a multiple of 8.
//
// 		Formats **%s** and **%p{i,I}{4,6}** require to read kernel
// 		memory. Reading kernel memory may fail due to either invalid
// 		address or valid address but requiring a major memory fault. If
// 		reading kernel memory fails, the string for **%s** will be an
// 		empty string, and the ip address for **%p{i,I}{4,6}** will be 0.
// 		Not returning error to bpf program is consistent with what
// 		**bpf_trace_printk**\ () does for now.
//
// 	Return
// 		The strictly positive length of the formatted string, including
// 		the trailing zero character. If the return value is greater than
// 		**str_size**, **str** contains a truncated string, guaranteed to
// 		be zero-terminated except when **str_size** is 0.
//
// 		Or **-EBUSY** if the per-CPU memory copy buffer is busy.
//
// long bpf_sys_bpf(u32 cmd, void *attr, u32 attr_size)
// 	Description
// 		Execute bpf syscall with given arguments.
// 	Return
// 		A syscall result.
//
// long bpf_btf_find_by_name_kind(char *name, int name_sz, u32 kind, int flags)
// 	Description
// 		Find BTF type with given name and kind in vmlinux BTF or in module's BTFs.
// 	Return
// 		Returns btf_id and btf_obj_fd in lower and upper 32 bits.
//
// long bpf_sys_close(u32 fd)
// 	Description
// 		Execute close syscall for given FD.
// 	Return
// 		A syscall result.
//
// long bpf_timer_init(struct bpf_timer *timer, struct bpf_map *map, u64 flags)
// 	Description
// 		Initialize the timer.
// 		First 4 bits of *flags* specify clockid.
// 		Only CLOCK_MONOTONIC, CLOCK_REALTIME, CLOCK_BOOTTIME are allowed.
// 		All other bits of *flags* are reserved.
// 		The verifier will reject the program if *timer* is not from
// 		the same *map*.
// 	Return
// 		0 on success.
// 		**-EBUSY** if *timer* is already initialized.
// 		**-EINVAL** if invalid *flags* are passed.
// 		**-EPERM** if *timer* is in a map that doesn't have any user references.
// 		The user space should either hold a file descriptor to a map with timers
// 		or pin such map in bpffs. When map is unpinned or file descriptor is
// 		closed all timers in the map will be cancelled and freed.
//
// long bpf_timer_set_callback(struct bpf_timer *timer, void *callback_fn)
// 	Description
// 		Configure the timer to call *callback_fn* static function.
// 	Return
// 		0 on success.
// 		**-EINVAL** if *timer* was not initialized with bpf_timer_init() earlier.
// 		**-EPERM** if *timer* is in a map that doesn't have any user references.
// 		The user space should either hold a file descriptor to a map with timers
// 		or pin such map in bpffs. When map is unpinned or file descriptor is
// 		closed all timers in the map will be cancelled and freed.
//
// long bpf_timer_start(struct bpf_timer *timer, u64 nsecs, u64 flags)
// 	Description
// 		Set timer expiration N nanoseconds from the current time. The
// 		configured callback will be invoked in soft irq context on some cpu
// 		and will not repeat unless another bpf_timer_start() is made.
// 		In such case the next invocation can migrate to a different cpu.
// 		Since struct bpf_timer is a field inside map element the map
// 		owns the timer. The bpf_timer_set_callback() will increment refcnt
// 		of BPF program to make sure that callback_fn code stays valid.
// 		When user space reference to a map reaches zero all timers
// 		in a map are cancelled and corresponding program's refcnts are
// 		decremented. This is done to make sure that Ctrl-C of a user
// 		process doesn't leave any timers running. If map is pinned in
// 		bpffs the callback_fn can re-arm itself indefinitely.
// 		bpf_map_update/delete_elem() helpers and user space sys_bpf commands
// 		cancel and free the timer in the given map element.
// 		The map can contain timers that invoke callback_fn-s from different
// 		programs. The same callback_fn can serve different timers from
// 		different maps if key/value layout matches across maps.
// 		Every bpf_timer_set_callback() can have different callback_fn.
//
// 		*flags* can be one of:
//
// 		**BPF_F_TIMER_ABS**
// 			Start the timer in absolute expire value instead of the
// 			default relative one.
// 		**BPF_F_TIMER_CPU_PIN**
// 			Timer will be pinned to the CPU of the caller.
//
// 	Return
// 		0 on success.
// 		**-EINVAL** if *timer* was not initialized with bpf_timer_init() earlier
// 		or invalid *flags* are passed.
//
// long bpf_timer_cancel(struct bpf_timer *timer)
// 	Description
// 		Cancel the timer and wait for callback_fn to finish if it was running.
// 	Return
// 		0 if the timer was not active.
// 		1 if the timer was active.
// 		**-EINVAL** if *timer* was not initialized with bpf_timer_init() earlier.
// 		**-EDEADLK** if callback_fn tried to call bpf_timer_cancel() on its
// 		own timer which would have led to a deadlock otherwise.
//
// u64 bpf_get_func_ip(void *ctx)
// 	Description
// 		Get address of the traced function (for tracing and kprobe programs).
//
// 		When called for kprobe program attached as uprobe it returns
// 		probe address for both entry and return uprobe.
//
// 	Return
// 		Address of the traced function for kprobe.
// 		0 for kprobes placed within the function (not at the entry).
// 		Address of the probe for uprobe and return uprobe.
//
// u64 bpf_get_attach_cookie(void *ctx)
// 	Description
// 		Get bpf_cookie value provided (optionally) during the program
// 		attachment. It might be different for each individual
// 		attachment, even if BPF program itself is the same.
// 		Expects BPF program context *ctx* as a first argument.
//
// 		Supported for the following program types:
// 			- kprobe/uprobe;
// 			- tracepoint;
// 			- perf_event.
// 	Return
// 		Value specified by user at BPF link creation/attachment time
// 		or 0, if it was not specified.
//
// long bpf_task_pt_regs(struct task_struct *task)
// 	Description
// 		Get the struct pt_regs associated with **task**.
// 	Return
// 		A pointer to struct pt_regs.
//
// long bpf_get_branch_snapshot(void *entries, u32 size, u64 flags)
// 	Description
// 		Get branch trace from hardware engines like Intel LBR. The
// 		hardware engine is stopped shortly after the helper is
// 		called. Therefore, the user need to filter branch entries
// 		based on the actual use case. To capture branch trace
// 		before the trigger point of the BPF program, the helper
// 		should be called at the beginning of the BPF program.
//
// 		The data is stored as struct perf_branch_entry into output
// 		buffer *entries*. *size* is the size of *entries* in bytes.
// 		*flags* is reserved for now and must be zero.
//
// 	Return
// 		On success, number of bytes written to *buf*. On error, a
// 		negative value.
//
// 		**-EINVAL** if *flags* is not zero.
//
// 		**-ENOENT** if architecture does not support branch records.
//
// long bpf_trace_vprintk(const char *fmt, u32 fmt_size, const void *data, u32 data_len)
// 	Description
// 		Behaves like **bpf_trace_printk**\ () helper, but takes an array of u64
// 		to format and can handle more format args as a result.
//
// 		Arguments are to be used as in **bpf_seq_printf**\ () helper.
// 	Return
// 		The number of bytes written to the buffer, or a negative error
// 		in case of failure.
//
// struct unix_sock *bpf_skc_to_unix_sock(void *sk)
// 	Description
// 		Dynamically cast a *sk* pointer to a *unix_sock* pointer.
// 	Return
// 		*sk* if casting is valid, or **NULL** otherwise.
//
// long bpf_kallsyms_lookup_name(const char *name, int name_sz, int flags, u64 *res)
// 	Description
// 		Get the address of a kernel symbol, returned in *res*. *res* is
// 		set to 0 if the symbol is not found.
// 	Return
// 		On success, zero. On error, a negative value.
//
// 		**-EINVAL** if *flags* is not zero.
//
// 		**-EINVAL** if string *name* is not the same size as *name_sz*.
//
// 		**-ENOENT** if symbol is not found.
//
// 		**-EPERM** if caller does not have permission to obtain kernel address.
//
// long bpf_find_vma(struct task_struct *task, u64 addr, void *callback_fn, void *callback_ctx, u64 flags)
// 	Description
// 		Find vma of *task* that contains *addr*, call *callback_fn*
// 		function with *task*, *vma*, and *callback_ctx*.
// 		The *callback_fn* should be a static function and
// 		the *callback_ctx* should be a pointer to the stack.
// 		The *flags* is used to control certain aspects of the helper.
// 		Currently, the *flags* must be 0.
//
// 		The expected callback signature is
//
// 		long (\*callback_fn)(struct task_struct \*task, struct vm_area_struct \*vma, void \*callback_ctx);
//
// 	Return
// 		0 on success.
// 		**-ENOENT** if *task->mm* is NULL, or no vma contains *addr*.
// 		**-EBUSY** if failed to try lock mmap_lock.
// 		**-EINVAL** for invalid **flags**.
//
// long bpf_loop(u32 nr_loops, void *callback_fn, void *callback_ctx, u64 flags)
// 	Description
// 		For **nr_loops**, call **callback_fn** function
// 		with **callback_ctx** as the context parameter.
// 		The **callback_fn** should be a static function and
// 		the **callback_ctx** should be a pointer to the stack.
// 		The **flags** is used to control certain aspects of the helper.
// 		Currently, the **flags** must be 0. Currently, nr_loops is
// 		limited to 1 << 23 (~8 million) loops.
//
// 		long (\*callback_fn)(u64 index, void \*ctx);
//
// 		where **index** is the current index in the loop. The index
// 		is zero-indexed.
//
// 		If **callback_fn** returns 0, the helper will continue to the next
// 		loop. If return value is 1, the helper will skip the rest of
// 		the loops and return. Other return values are not used now,
// 		and will be rejected by the verifier.
//
// 	Return
// 		The number of loops performed, **-EINVAL** for invalid **flags**,
// 		**-E2BIG** if **nr_loops** exceeds the maximum number of loops.
//
// long bpf_strncmp(const char *s1, u32 s1_sz, const char *s2)
// 	Description
// 		Do strncmp() between **s1** and **s2**. **s1** doesn't need
// 		to be null-terminated and **s1_sz** is the maximum storage
// 		size of **s1**. **s2** must be a read-only string.
// 	Return
// 		An integer less than, equal to, or greater than zero
// 		if the first **s1_sz** bytes of **s1** is found to be
// 		less than, to match, or be greater than **s2**.
//
// long bpf_get_func_arg(void *ctx, u32 n, u64 *value)
// 	Description
// 		Get **n**-th argument register (zero based) of the traced function (for tracing programs)
// 		returned in **value**.
//
// 	Return
// 		0 on success.
// 		**-EINVAL** if n >= argument register count of traced function.
//
// long bpf_get_func_ret(void *ctx, u64 *value)
// 	Description
// 		Get return value of the traced function (for tracing programs)
// 		in **value**.
//
// 	Return
// 		0 on success.
// 		**-EOPNOTSUPP** for tracing programs other than BPF_TRACE_FEXIT or BPF_MODIFY_RETURN.
//
// long bpf_get_func_arg_cnt(void *ctx)
// 	Description
// 		Get number of registers of the traced function (for tracing programs) where
// 		function arguments are stored in these registers.
//
// 	Return
// 		The number of argument registers of the traced function.
//
// int bpf_get_retval(void)
// 	Description
// 		Get the BPF program's return value that will be returned to the upper layers.
//
// 		This helper is currently supported by cgroup programs and only by the hooks
// 		where BPF program's return value is returned to the userspace via errno.
// 	Return
// 		The BPF program's return value.
//
// int bpf_set_retval(int retval)
// 	Description
// 		Set the BPF program's return value that will be returned to the upper layers.
//
// 		This helper is currently supported by cgroup programs and only by the hooks
// 		where BPF program's return value is returned to the userspace via errno.
//
// 		Note that there is the following corner case where the program exports an error
// 		via bpf_set_retval but signals success via 'return 1':
//
// 			bpf_set_retval(-EPERM);
// 			return 1;
//
// 		In this case, the BPF program's return value will use helper's -EPERM. This
// 		still holds true for cgroup/bind{4,6} which supports extra 'return 3' success case.
//
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// u64 bpf_xdp_get_buff_len(struct xdp_buff *xdp_md)
// 	Description
// 		Get the total size of a given xdp buff (linear and paged area)
// 	Return
// 		The total size of a given xdp buffer.
//
// long bpf_xdp_load_bytes(struct xdp_buff *xdp_md, u32 offset, void *buf, u32 len)
// 	Description
// 		This helper is provided as an easy way to load data from a
// 		xdp buffer. It can be used to load *len* bytes from *offset* from
// 		the frame associated to *xdp_md*, into the buffer pointed by
// 		*buf*.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_xdp_store_bytes(struct xdp_buff *xdp_md, u32 offset, void *buf, u32 len)
// 	Description
// 		Store *len* bytes from buffer *buf* into the frame
// 		associated to *xdp_md*, at *offset*.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_copy_from_user_task(void *dst, u32 size, const void *user_ptr, struct task_struct *tsk, u64 flags)
// 	Description
// 		Read *size* bytes from user space address *user_ptr* in *tsk*'s
// 		address space, and stores the data in *dst*. *flags* is not
// 		used yet and is provided for future extensibility. This helper
// 		can only be used by sleepable programs.
// 	Return
// 		0 on success, or a negative error in case of failure. On error
// 		*dst* buffer is zeroed out.
//
// long bpf_skb_set_tstamp(struct sk_buff *skb, u64 tstamp, u32 tstamp_type)
// 	Description
// 		Change the __sk_buff->tstamp_type to *tstamp_type*
// 		and set *tstamp* to the __sk_buff->tstamp together.
//
// 		If there is no need to change the __sk_buff->tstamp_type,
// 		the tstamp value can be directly written to __sk_buff->tstamp
// 		instead.
//
// 		BPF_SKB_TSTAMP_DELIVERY_MONO is the only tstamp that
// 		will be kept during bpf_redirect_*().  A non zero
// 		*tstamp* must be used with the BPF_SKB_TSTAMP_DELIVERY_MONO
// 		*tstamp_type*.
//
// 		A BPF_SKB_TSTAMP_UNSPEC *tstamp_type* can only be used
// 		with a zero *tstamp*.
//
// 		Only IPv4 and IPv6 skb->protocol are supported.
//
// 		This function is most useful when it needs to set a
// 		mono delivery time to __sk_buff->tstamp and then
// 		bpf_redirect_*() to the egress of an iface.  For example,
// 		changing the (rcv) timestamp in __sk_buff->tstamp at
// 		ingress to a mono delivery time and then bpf_redirect_*()
// 		to sch_fq@phy-dev.
// 	Return
// 		0 on success.
// 		**-EINVAL** for invalid input
// 		**-EOPNOTSUPP** for unsupported protocol
//
// long bpf_ima_file_hash(struct file *file, void *dst, u32 size)
// 	Description
// 		Returns a calculated IMA hash of the *file*.
// 		If the hash is larger than *size*, then only *size*
// 		bytes will be copied to *dst*
// 	Return
// 		The **hash_algo** is returned on success,
// 		**-EOPNOTSUPP** if the hash calculation failed or **-EINVAL** if
// 		invalid arguments are passed.
//
// void *bpf_kptr_xchg(void *dst, void *ptr)
// 	Description
// 		Exchange kptr at pointer *dst* with *ptr*, and return the old value.
// 		*dst* can be map value or local kptr. *ptr* can be NULL, otherwise
// 		it must be a referenced pointer which will be released when this helper
// 		is called.
// 	Return
// 		The old value of kptr (which can be NULL). The returned pointer
// 		if not NULL, is a reference which must be released using its
// 		corresponding release function, or moved into a BPF map before
// 		program exit.
//
// void *bpf_map_lookup_percpu_elem(struct bpf_map *map, const void *key, u32 cpu)
// 	Description
// 		Perform a lookup in *percpu map* for an entry associated to
// 		*key* on *cpu*.
// 	Return
// 		Map value associated to *key* on *cpu*, or **NULL** if no entry
// 		was found or *cpu* is invalid.
//
// struct mptcp_sock *bpf_skc_to_mptcp_sock(void *sk)
// 	Description
// 		Dynamically cast a *sk* pointer to a *mptcp_sock* pointer.
// 	Return
// 		*sk* if casting is valid, or **NULL** otherwise.
//
// long bpf_dynptr_from_mem(void *data, u64 size, u64 flags, struct bpf_dynptr *ptr)
// 	Description
// 		Get a dynptr to local memory *data*.
//
// 		*data* must be a ptr to a map value.
// 		The maximum *size* supported is DYNPTR_MAX_SIZE.
// 		*flags* is currently unused.
// 	Return
// 		0 on success, -E2BIG if the size exceeds DYNPTR_MAX_SIZE,
// 		-EINVAL if flags is not 0.
//
// long bpf_ringbuf_reserve_dynptr(void *ringbuf, u32 size, u64 flags, struct bpf_dynptr *ptr)
// 	Description
// 		Reserve *size* bytes of payload in a ring buffer *ringbuf*
// 		through the dynptr interface. *flags* must be 0.
//
// 		Please note that a corresponding bpf_ringbuf_submit_dynptr or
// 		bpf_ringbuf_discard_dynptr must be called on *ptr*, even if the
// 		reservation fails. This is enforced by the verifier.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// void bpf_ringbuf_submit_dynptr(struct bpf_dynptr *ptr, u64 flags)
// 	Description
// 		Submit reserved ring buffer sample, pointed to by *data*,
// 		through the dynptr interface. This is a no-op if the dynptr is
// 		invalid/null.
//
// 		For more information on *flags*, please see
// 		'bpf_ringbuf_submit'.
// 	Return
// 		Nothing. Always succeeds.
//
// void bpf_ringbuf_discard_dynptr(struct bpf_dynptr *ptr, u64 flags)
// 	Description
// 		Discard reserved ring buffer sample through the dynptr
// 		interface. This is a no-op if the dynptr is invalid/null.
//
// 		For more information on *flags*, please see
// 		'bpf_ringbuf_discard'.
// 	Return
// 		Nothing. Always succeeds.
//
// long bpf_dynptr_read(void *dst, u64 len, const struct bpf_dynptr *src, u64 offset, u64 flags)
// 	Description
// 		Read *len* bytes from *src* into *dst*, starting from *offset*
// 		into *src*.
// 		*flags* is currently unused.
// 	Return
// 		0 on success, -E2BIG if *offset* + *len* exceeds the length
// 		of *src*'s data, -EINVAL if *src* is an invalid dynptr or if
// 		*flags* is not 0.
//
// long bpf_dynptr_write(const struct bpf_dynptr *dst, u64 offset, void *src, u64 len, u64 flags)
// 	Description
// 		Write *len* bytes from *src* into *dst*, starting from *offset*
// 		into *dst*.
//
// 		*flags* must be 0 except for skb-type dynptrs.
//
// 		For skb-type dynptrs:
// 		    *  All data slices of the dynptr are automatically
// 		       invalidated after **bpf_dynptr_write**\ (). This is
// 		       because writing may pull the skb and change the
// 		       underlying packet buffer.
//
// 		    *  For *flags*, please see the flags accepted by
// 		       **bpf_skb_store_bytes**\ ().
// 	Return
// 		0 on success, -E2BIG if *offset* + *len* exceeds the length
// 		of *dst*'s data, -EINVAL if *dst* is an invalid dynptr or if *dst*
// 		is a read-only dynptr or if *flags* is not correct. For skb-type dynptrs,
// 		other errors correspond to errors returned by **bpf_skb_store_bytes**\ ().
//
// void *bpf_dynptr_data(const struct bpf_dynptr *ptr, u64 offset, u64 len)
// 	Description
// 		Get a pointer to the underlying dynptr data.
//
// 		*len* must be a statically known value. The returned data slice
// 		is invalidated whenever the dynptr is invalidated.
//
// 		skb and xdp type dynptrs may not use bpf_dynptr_data. They should
// 		instead use bpf_dynptr_slice and bpf_dynptr_slice_rdwr.
// 	Return
// 		Pointer to the underlying dynptr data, NULL if the dynptr is
// 		read-only, if the dynptr is invalid, or if the offset and length
// 		is out of bounds.
//
// s64 bpf_tcp_raw_gen_syncookie_ipv4(struct iphdr *iph, struct tcphdr *th, u32 th_len)
// 	Description
// 		Try to issue a SYN cookie for the packet with corresponding
// 		IPv4/TCP headers, *iph* and *th*, without depending on a
// 		listening socket.
//
// 		*iph* points to the IPv4 header.
//
// 		*th* points to the start of the TCP header, while *th_len*
// 		contains the length of the TCP header (at least
// 		**sizeof**\ (**struct tcphdr**)).
// 	Return
// 		On success, lower 32 bits hold the generated SYN cookie in
// 		followed by 16 bits which hold the MSS value for that cookie,
// 		and the top 16 bits are unused.
//
// 		On failure, the returned value is one of the following:
//
// 		**-EINVAL** if *th_len* is invalid.
//
// s64 bpf_tcp_raw_gen_syncookie_ipv6(struct ipv6hdr *iph, struct tcphdr *th, u32 th_len)
// 	Description
// 		Try to issue a SYN cookie for the packet with corresponding
// 		IPv6/TCP headers, *iph* and *th*, without depending on a
// 		listening socket.
//
// 		*iph* points to the IPv6 header.
//
// 		*th* points to the start of the TCP header, while *th_len*
// 		contains the length of the TCP header (at least
// 		**sizeof**\ (**struct tcphdr**)).
// 	Return
// 		On success, lower 32 bits hold the generated SYN cookie in
// 		followed by 16 bits which hold the MSS value for that cookie,
// 		and the top 16 bits are unused.
//
// 		On failure, the returned value is one of the following:
//
// 		**-EINVAL** if *th_len* is invalid.
//
// 		**-EPROTONOSUPPORT** if CONFIG_IPV6 is not builtin.
//
// long bpf_tcp_raw_check_syncookie_ipv4(struct iphdr *iph, struct tcphdr *th)
// 	Description
// 		Check whether *iph* and *th* contain a valid SYN cookie ACK
// 		without depending on a listening socket.
//
// 		*iph* points to the IPv4 header.
//
// 		*th* points to the TCP header.
// 	Return
// 		0 if *iph* and *th* are a valid SYN cookie ACK.
//
// 		On failure, the returned value is one of the following:
//
// 		**-EACCES** if the SYN cookie is not valid.
//
// long bpf_tcp_raw_check_syncookie_ipv6(struct ipv6hdr *iph, struct tcphdr *th)
// 	Description
// 		Check whether *iph* and *th* contain a valid SYN cookie ACK
// 		without depending on a listening socket.
//
// 		*iph* points to the IPv6 header.
//
// 		*th* points to the TCP header.
// 	Return
// 		0 if *iph* and *th* are a valid SYN cookie ACK.
//
// 		On failure, the returned value is one of the following:
//
// 		**-EACCES** if the SYN cookie is not valid.
//
// 		**-EPROTONOSUPPORT** if CONFIG_IPV6 is not builtin.
//
// u64 bpf_ktime_get_tai_ns(void)
// 	Description
// 		A nonsettable system-wide clock derived from wall-clock time but
// 		ignoring leap seconds.  This clock does not experience
// 		discontinuities and backwards jumps caused by NTP inserting leap
// 		seconds as CLOCK_REALTIME does.
//
// 		See: **clock_gettime**\ (**CLOCK_TAI**)
// 	Return
// 		Current *ktime*.
//
// long bpf_user_ringbuf_drain(struct bpf_map *map, void *callback_fn, void *ctx, u64 flags)
// 	Description
// 		Drain samples from the specified user ring buffer, and invoke
// 		the provided callback for each such sample:
//
// 		long (\*callback_fn)(const struct bpf_dynptr \*dynptr, void \*ctx);
//
// 		If **callback_fn** returns 0, the helper will continue to try
// 		and drain the next sample, up to a maximum of
// 		BPF_MAX_USER_RINGBUF_SAMPLES samples. If the return value is 1,
// 		the helper will skip the rest of the samples and return. Other
// 		return values are not used now, and will be rejected by the
// 		verifier.
// 	Return
// 		The number of drained samples if no error was encountered while
// 		draining samples, or 0 if no samples were present in the ring
// 		buffer. If a user-space producer was epoll-waiting on this map,
// 		and at least one sample was drained, they will receive an event
// 		notification notifying them of available space in the ring
// 		buffer. If the BPF_RB_NO_WAKEUP flag is passed to this
// 		function, no wakeup notification will be sent. If the
// 		BPF_RB_FORCE_WAKEUP flag is passed, a wakeup notification will
// 		be sent even if no sample was drained.
//
// 		On failure, the returned value is one of the following:
//
// 		**-EBUSY** if the ring buffer is contended, and another calling
// 		context was concurrently draining the ring buffer.
//
// 		**-EINVAL** if user-space is not properly tracking the ring
// 		buffer due to the producer position not being aligned to 8
// 		bytes, a sample not being aligned to 8 bytes, or the producer
// 		position not matching the advertised length of a sample.
//
// 		**-E2BIG** if user-space has tried to publish a sample which is
// 		larger than the size of the ring buffer, or which cannot fit
// 		within a struct bpf_dynptr.
//
// void *bpf_cgrp_storage_get(struct bpf_map *map, struct cgroup *cgroup, void *value, u64 flags)
// 	Description
// 		Get a bpf_local_storage from the *cgroup*.
//
// 		Logically, it could be thought of as getting the value from
// 		a *map* with *cgroup* as the **key**.  From this
// 		perspective,  the usage is not much different from
// 		**bpf_map_lookup_elem**\ (*map*, **&**\ *cgroup*) except this
// 		helper enforces the key must be a cgroup struct and the map must also
// 		be a **BPF_MAP_TYPE_CGRP_STORAGE**.
//
// 		In reality, the local-storage value is embedded directly inside of the
// 		*cgroup* object itself, rather than being located in the
// 		**BPF_MAP_TYPE_CGRP_STORAGE** map. When the local-storage value is
// 		queried for some *map* on a *cgroup* object, the kernel will perform an
// 		O(n) iteration over all of the live local-storage values for that
// 		*cgroup* object until the local-storage value for the *map* is found.
//
// 		An optional *flags* (**BPF_LOCAL_STORAGE_GET_F_CREATE**) can be
// 		used such that a new bpf_local_storage will be
// 		created if one does not exist.  *value* can be used
// 		together with **BPF_LOCAL_STORAGE_GET_F_CREATE** to specify
// 		the initial value of a bpf_local_storage.  If *value* is
// 		**NULL**, the new bpf_local_storage will be zero initialized.
// 	Return
// 		A bpf_local_storage pointer is returned on success.
//
// 		**NULL** if not found or there was an error in adding
// 		a new bpf_local_storage.
//
// long bpf_cgrp_storage_delete(struct bpf_map *map, struct cgroup *cgroup)
// 	Description
// 		Delete a bpf_local_storage from a *cgroup*.
// 	Return
// 		0 on success.
//
// 		**-ENOENT** if the bpf_local_storage cannot be found.
// This helper list is effectively frozen. If you are trying to		 * add a new helper, you should add a kfunc instead which has		 * less stability guarantees. See Documentation/bpf/kfuncs.rst
#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_link_info {
    r#type: u32,
    id: u32,
    prog_id: u32,
    __slate_anon_3: __SlateRecord_bpf_h_6842,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_func_info {
    insn_off: u32,
    type_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_line_info {
    insn_off: u32,
    file_name_off: u32,
    line_off: u32,
    line_col: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sock_filter {
    code: u16,
    jt: u8,
    jf: u8,
    k: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct btf_type {
    name_off: u32,
    info: u32,
    __slate_anon_2: __SlateRecord_btf_h_64,
}


// BTF_KIND_INT is followed by a u32 and the following
// is the 32 bits arrangement:
// Attributes stored in the BTF_INT_ENCODING
#[repr(C)]
#[derive(Clone, Copy)]
struct btf_member {
    name_off: u32,
    r#type: u32,
    offset: u32,
}


// If the struct/union type info kind_flag is set, the
// following two macros are used to access bitfield_size
// and bit_offset from btf_member.offset.
#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_mod {
    next: *mut static_call_mod,
    r#mod: *mut module,
    sites: *mut static_call_site,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_iter_seq_info {
    seq_ops: *const seq_operations,
    init_seq_private: Option<unsafe extern "C" fn(*mut (), *mut bpf_iter_aux_info) -> i32>,
    fini_seq_private: Option<unsafe extern "C" fn(*mut ())>,
    seq_priv_size: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_map_ops {
    map_alloc_check: Option<unsafe extern "C" fn(*mut bpf_attr) -> i32>,
    map_alloc: Option<unsafe extern "C" fn(*mut bpf_attr) -> *mut bpf_map>,
    map_release: Option<unsafe extern "C" fn(*mut bpf_map, *mut file)>,
    map_free: Option<unsafe extern "C" fn(*mut bpf_map)>,
    map_get_next_key: Option<unsafe extern "C" fn(*mut bpf_map, *mut (), *mut ()) -> i32>,
    map_release_uref: Option<unsafe extern "C" fn(*mut bpf_map)>,
    map_lookup_elem_sys_only: Option<unsafe extern "C" fn(*mut bpf_map, *mut ()) -> *mut ()>,
    map_lookup_batch: Option<unsafe extern "C" fn(*mut bpf_map, *const bpf_attr, *mut bpf_attr) -> i32>,
    map_lookup_and_delete_elem: Option<unsafe extern "C" fn(*mut bpf_map, *mut (), *mut (), u64) -> i32>,
    map_lookup_and_delete_batch: Option<unsafe extern "C" fn(*mut bpf_map, *const bpf_attr, *mut bpf_attr) -> i32>,
    map_update_batch: Option<unsafe extern "C" fn(*mut bpf_map, *mut file, *const bpf_attr, *mut bpf_attr) -> i32>,
    map_delete_batch: Option<unsafe extern "C" fn(*mut bpf_map, *const bpf_attr, *mut bpf_attr) -> i32>,
    map_lookup_elem: Option<unsafe extern "C" fn(*mut bpf_map, *mut ()) -> *mut ()>,
    map_update_elem: Option<unsafe extern "C" fn(*mut bpf_map, *mut (), *mut (), u64) -> i64>,
    map_delete_elem: Option<unsafe extern "C" fn(*mut bpf_map, *mut ()) -> i64>,
    map_push_elem: Option<unsafe extern "C" fn(*mut bpf_map, *mut (), u64) -> i64>,
    map_pop_elem: Option<unsafe extern "C" fn(*mut bpf_map, *mut ()) -> i64>,
    map_peek_elem: Option<unsafe extern "C" fn(*mut bpf_map, *mut ()) -> i64>,
    map_lookup_percpu_elem: Option<unsafe extern "C" fn(*mut bpf_map, *mut (), u32) -> *mut ()>,
    map_get_hash: Option<unsafe extern "C" fn(*mut bpf_map) -> i32>,
    map_fd_get_ptr: Option<unsafe extern "C" fn(*mut bpf_map, *mut file, i32) -> *mut ()>,
    map_fd_put_ptr: Option<unsafe extern "C" fn(*mut bpf_map, *mut (), bool)>,
    map_gen_lookup: Option<unsafe extern "C" fn(*mut bpf_map, *mut bpf_insn) -> i32>,
    map_fd_sys_lookup_elem: Option<unsafe extern "C" fn(*mut ()) -> u32>,
    map_seq_show_elem: Option<unsafe extern "C" fn(*mut bpf_map, *mut (), *mut seq_file)>,
    map_check_btf: Option<unsafe extern "C" fn(*mut bpf_map, *const btf, *const btf_type, *const btf_type) -> i32>,
    map_poke_track: Option<unsafe extern "C" fn(*mut bpf_map, *mut bpf_prog_aux) -> i32>,
    map_poke_untrack: Option<unsafe extern "C" fn(*mut bpf_map, *mut bpf_prog_aux)>,
    map_poke_run: Option<unsafe extern "C" fn(*mut bpf_map, u32, *mut bpf_prog, *mut bpf_prog)>,
    map_direct_value_addr: Option<unsafe extern "C" fn(*const bpf_map, *mut u64, u32) -> i32>,
    map_direct_value_meta: Option<unsafe extern "C" fn(*const bpf_map, u64, *mut u32) -> i32>,
    map_mmap: Option<unsafe extern "C" fn(*mut bpf_map, *mut vm_area_struct) -> i32>,
    map_mmap_fault: Option<unsafe extern "C" fn(*mut bpf_map, *mut vm_fault) -> u32>,
    map_poll: Option<unsafe extern "C" fn(*mut bpf_map, *mut file, *mut poll_table_struct) -> u32>,
    map_get_unmapped_area: Option<unsafe extern "C" fn(*mut file, u64, u64, u64, u64) -> u64>,
    map_local_storage_charge: Option<unsafe extern "C" fn(*mut bpf_local_storage_map, *mut (), u32) -> i32>,
    map_local_storage_uncharge: Option<unsafe extern "C" fn(*mut bpf_local_storage_map, *mut (), u32)>,
    map_owner_storage_ptr: Option<unsafe extern "C" fn(*mut ()) -> *mut *mut bpf_local_storage>,
    map_redirect: Option<unsafe extern "C" fn(*mut bpf_map, u64, u64) -> i64>,
    map_meta_equal: Option<unsafe extern "C" fn(*const bpf_map, *const bpf_map) -> bool>,
    map_set_for_each_callback_args: Option<unsafe extern "C" fn(*mut bpf_verifier_env, *mut bpf_func_state, *mut bpf_func_state) -> i32>,
    map_for_each_callback: Option<unsafe extern "C" fn(*mut bpf_map, Option<unsafe extern "C" fn(u64, u64, u64, u64, u64) -> u64>, *mut (), u64) -> i64>,
    map_mem_usage: Option<unsafe extern "C" fn(*const bpf_map) -> u64>,
    map_btf_id: *mut i32,
    iter_seq_info: *const bpf_iter_seq_info,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct btf_field_kptr {
    btf: *mut btf,
    module: *mut module,
    dtor: Option<unsafe extern "C" fn(*mut ())>,
    btf_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct btf_field_graph_root {
    btf: *mut btf,
    value_btf_id: u32,
    node_offset: u32,
    value_rec: *mut btf_record,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct btf_field {
    offset: u32,
    size: u32,
    r#type: u32,
    __slate_anon_3: __SlateRecord_bpf_h_256,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct btf_record {
    cnt: u32,
    field_mask: u32,
    spin_lock_off: i32,
    res_spin_lock_off: i32,
    timer_off: i32,
    wq_off: i32,
    refcount_off: i32,
    task_work_off: i32,
    fields: [btf_field; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_map_owner {
    r#type: u32,
    jited: bool,
    xdp_has_frags: bool,
    sleepable: bool,
    storage_cookie: [u64; 2],
    attach_func_proto: *const btf_type,
    expected_attach_type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_map {
    sha: [u8; 32],
    ops: *const bpf_map_ops,
    inner_map_meta: *mut bpf_map,
    security: *mut (),
    map_type: u32,
    key_size: u32,
    value_size: u32,
    max_entries: u32,
    map_extra: u64,
    map_flags: u32,
    id: u32,
    record: *mut btf_record,
    numa_node: i32,
    btf_key_type_id: u32,
    btf_value_type_id: u32,
    btf_vmlinux_value_type_id: u32,
    btf: *mut btf,
    name: [u8; 16],
    freeze_mutex: mutex,
    refcnt: __SlateRecord_types_h_197,
    usercnt: __SlateRecord_types_h_197,
    __slate_anon_21: __SlateRecord_bpf_h_329,
    writecnt: __SlateRecord_types_h_197,
    owner_lock: spinlock,
    owner: *mut bpf_map_owner,
    bypass_spec_v1: bool,
    frozen: bool,
    free_after_mult_rcu_gp: bool,
    free_after_rcu_gp: bool,
    sleepable_refcnt: __SlateRecord_types_h_197,
    elem_count: *mut i64,
    cookie: u64,
    excl_prog_sha: *mut u8,
}


// bpf_type_flag contains a set of flags that are applicable to the values of
// arg_type, ret_type and reg_type. For example, a pointer value may be null,
// or a memory is read-only. We classify types into two categories: base types
// and extended types. Extended types are base types combined with a type flag.
//
// Currently there are no more than 32 base types in arg_type, ret_type and
// reg_types.
// Max number of base types.
// Max number of all types.
// The longest tracepoint has 12 args.
// See include/trace/bpf_probe.h
//
// Also reuse this macro for maximum number of arguments a BPF function
// or a kfunc can have. Args 1-5 are passed in registers, args 6-12 via
// stack arg slots. The JIT may map some stack arg slots to registers based
// on the native calling convention (e.g., arg 6 to R9 on x86-64).
#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_func_proto {
    func: Option<unsafe extern "C" fn(u64, u64, u64, u64, u64) -> u64>,
    gpl_only: bool,
    pkt_access: bool,
    might_sleep: bool,
    allow_fastcall: bool,
    ret_type: u32,
    __slate_anon_6: __SlateRecord_bpf_h_996,
    __slate_anon_7: __SlateRecord_bpf_h_1006,
    ret_btf_id: *mut i32,
    allowed: Option<unsafe extern "C" fn(*const bpf_prog) -> bool>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_insn_access_aux {
    reg_type: u32,
    is_ldsx: bool,
    __slate_anon_2: __SlateRecord_bpf_h_1109,
    log: *mut bpf_verifier_log,
    is_retval: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_prog_ops {
    test_run: Option<unsafe extern "C" fn(*mut bpf_prog, *const bpf_attr, *mut bpf_attr) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_verifier_ops {
    get_func_proto: Option<unsafe extern "C" fn(u32, *const bpf_prog) -> *const bpf_func_proto>,
    is_valid_access: Option<unsafe extern "C" fn(i32, i32, u32, *const bpf_prog, *mut bpf_insn_access_aux) -> bool>,
    gen_prologue: Option<unsafe extern "C" fn(*mut bpf_insn, bool, *const bpf_prog) -> i32>,
    gen_epilogue: Option<unsafe extern "C" fn(*mut bpf_insn, *const bpf_prog, i16) -> i32>,
    gen_ld_abs: Option<unsafe extern "C" fn(*const bpf_insn, *mut bpf_insn) -> i32>,
    convert_ctx_access: Option<unsafe extern "C" fn(u32, *const bpf_insn, *mut bpf_insn, *mut bpf_prog, *mut u32) -> u32>,
    btf_struct_access: Option<unsafe extern "C" fn(*mut bpf_verifier_log, *const bpf_reg_state, i32, i32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_prog_offload {
    prog: *mut bpf_prog,
    netdev: *mut net_device,
    offdev: *mut bpf_offload_dev,
    dev_priv: *mut (),
    offloads: list_head,
    dev_state: bool,
    opt_failed: bool,
    jited_image: *mut (),
    jited_len: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct btf_func_model {
    ret_size: u8,
    ret_flags: u8,
    nr_args: u8,
    arg_size: [u8; 12],
    arg_flags: [u8; 12],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_ksym {
    start: u64,
    end: u64,
    name: [u8; 512],
    lnode: list_head,
    tnode: latch_tree_node,
    prog: bool,
    fp_start: u32,
    fp_end: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_tramp_skip {
    prog: *mut bpf_prog,
    nop: *mut (),
    target: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_tramp_image {
    image: *mut (),
    size: i32,
    ksym: bpf_ksym,
    pcref: percpu_ref,
    call_orig: bool,
    tr: *mut bpf_trampoline,
    list: list_head,
    nr_skips: i32,
    skips: *mut bpf_tramp_skip,
    __slate_anon_9: __SlateRecord_bpf_h_1393,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_trampoline {
    hlist_key: hlist_node,
    hlist_ip: hlist_node,
    fops: *mut ftrace_ops,
    refcnt: refcount_struct,
    flags: u32,
    key: u64,
    ip: u64,
    func: __SlateRecord_bpf_h_1423,
    extension_prog: *mut bpf_prog,
    progs_hlist: [hlist_head; 3],
    progs_cnt: [i32; 3],
    cur_image: *mut bpf_tramp_image,
    images: list_head,
    multi_attach: __SlateRecord_bpf_h_1442,
}


// Fits in 2048B
#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_func_info_aux {
    linkage: u16,
    unreliable: bool,
    called: [bool; 2],
    verified: [bool; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_jit_poke_descriptor {
    tailcall_target: *mut (),
    tailcall_bypass: *mut (),
    bypass_addr: *mut (),
    aux: *mut (),
    __slate_anon_4: __SlateRecord_bpf_h_1705,
    tailcall_target_stable: bool,
    adj_off: u8,
    reason: u16,
    insn_idx: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_ctx_arg_aux {
    offset: u32,
    reg_type: u32,
    btf: *mut btf,
    btf_id: u32,
    ref_id: u32,
    refcounted: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct btf_mod_pair {
    btf: *mut btf,
    module: *mut module,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_stream {
    capacity: __SlateRecord_types_h_190,
    log: llist_head,
    lock: mutex,
    backlog_head: *mut llist_node,
    backlog_tail: *mut llist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_prog_aux {
    refcnt: __SlateRecord_types_h_197,
    used_map_cnt: u32,
    used_btf_cnt: u32,
    max_ctx_offset: u32,
    max_pkt_offset: u32,
    max_tp_access: u32,
    stack_depth: u32,
    id: u32,
    func_cnt: u32,
    real_func_cnt: u32,
    func_idx: u32,
    attach_btf_id: u32,
    attach_st_ops_member_off: u32,
    ctx_arg_info_size: u32,
    max_rdonly_access: u32,
    max_rdwr_access: u32,
    subprog_start: u32,
    attach_btf: *mut btf,
    ctx_arg_info: *mut bpf_ctx_arg_aux,
    priv_stack_ptr: *mut (),
    dst_mutex: mutex,
    dst_prog: *mut bpf_prog,
    dst_trampoline: *mut bpf_trampoline,
    saved_dst_prog_type: u32,
    saved_dst_attach_type: u32,
    verifier_zext: bool,
    dev_bound: bool,
    offload_requested: bool,
    attach_btf_trace: bool,
    attach_tracing_prog: bool,
    tramp_linked: bool,
    func_proto_unreliable: bool,
    tail_call_reachable: bool,
    xdp_has_frags: bool,
    exception_cb: bool,
    exception_boundary: bool,
    is_extended: bool,
    jits_use_priv_stack: bool,
    priv_stack_requested: bool,
    changes_pkt_data: bool,
    might_sleep: bool,
    kprobe_write_ctx: bool,
    sig: __SlateRecord_bpf_h_1821_2,
    prog_array_member_cnt: u64,
    ext_mutex: mutex,
    arena: *mut bpf_arena,
    recursion_detected: Option<unsafe extern "C" fn(*mut bpf_prog)>,
    attach_func_proto: *const btf_type,
    attach_func_name: *const u8,
    func: *mut *mut bpf_prog,
    main_prog_aux: *mut bpf_prog_aux,
    jit_data: *mut (),
    poke_tab: *mut bpf_jit_poke_descriptor,
    kfunc_tab: *mut bpf_kfunc_desc_tab,
    kfunc_btf_tab: *mut bpf_kfunc_btf_tab,
    size_poke_tab: u32,
    ksym: bpf_ksym,
    ops: *const bpf_prog_ops,
    st_ops: *const bpf_struct_ops,
    used_maps: *mut *mut bpf_map,
    used_maps_mutex: mutex,
    used_btfs: *mut btf_mod_pair,
    prog: *mut bpf_prog,
    user: *mut user_struct,
    load_time: u64,
    verified_insns: u32,
    cgroup_atype: i32,
    cgroup_storage: [*mut bpf_map; 2],
    name: [u8; 16],
    bpf_exception_cb: Option<unsafe extern "C" fn(u64, u64, u64, u64, u64) -> u64>,
    stack_arg_sp_adjust: u16,
    security: *mut (),
    token: *mut bpf_token,
    offload: *mut bpf_prog_offload,
    btf: *mut btf,
    func_info: *mut bpf_func_info,
    func_info_aux: *mut bpf_func_info_aux,
    linfo: *mut bpf_line_info,
    jited_linfo: *mut *mut (),
    func_info_cnt: u32,
    nr_linfo: u32,
    linfo_idx: u32,
    r#mod: *mut module,
    num_exentries: u32,
    extable: *mut exception_table_entry,
    __slate_anon_85: __SlateRecord_bpf_h_1892,
    stream: [bpf_stream; 2],
    st_ops_assoc_mutex: mutex,
    st_ops_assoc: *mut bpf_map,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_prog {
    pages: u16,
    __slate_bits_0: __slate_bits::__SlateBits1042U0,
    r#type: u32,
    expected_attach_type: u32,
    len: u32,
    jited_len: u32,
    __slate_anon_22: __SlateRecord_bpf_h_1926,
    stats: *mut bpf_prog_stats,
    active: *mut u8,
    bpf_func: Option<unsafe extern "C" fn(*const (), *const bpf_insn) -> u32>,
    aux: *mut bpf_prog_aux,
    orig_prog: *mut sock_fprog_kern,
    __slate_anon_28: __SlateRecord_bpf_h_1937,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_link {
    refcnt: __SlateRecord_types_h_197,
    id: u32,
    r#type: u32,
    ops: *const bpf_link_ops,
    prog: *mut bpf_prog,
    flags: u32,
    attach_type: u32,
    __slate_anon_7: __SlateRecord_bpf_h_1964,
    sleepable: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_link_ops {
    release: Option<unsafe extern "C" fn(*mut bpf_link)>,
    dealloc: Option<unsafe extern "C" fn(*mut bpf_link)>,
    dealloc_deferred: Option<unsafe extern "C" fn(*mut bpf_link)>,
    detach: Option<unsafe extern "C" fn(*mut bpf_link) -> i32>,
    update_prog: Option<unsafe extern "C" fn(*mut bpf_link, *mut bpf_prog, *mut bpf_prog) -> i32>,
    show_fdinfo: Option<unsafe extern "C" fn(*const bpf_link, *mut seq_file)>,
    fill_link_info: Option<unsafe extern "C" fn(*const bpf_link, *mut bpf_link_info) -> i32>,
    update_map: Option<unsafe extern "C" fn(*mut bpf_link, *mut bpf_map, *mut bpf_map) -> i32>,
    poll: Option<unsafe extern "C" fn(*mut file, *mut poll_table_struct) -> u32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_token {
    work: work_struct,
    refcnt: __SlateRecord_types_h_197,
    userns: *mut user_namespace,
    allowed_cmds: u64,
    allowed_maps: u64,
    allowed_progs: u64,
    allowed_attachs: u64,
    security: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_struct_ops {
    verifier_ops: *const bpf_verifier_ops,
    init: Option<unsafe extern "C" fn(*mut btf) -> i32>,
    check_member: Option<unsafe extern "C" fn(*const btf_type, *const btf_member, *const bpf_prog) -> i32>,
    init_member: Option<unsafe extern "C" fn(*const btf_type, *const btf_member, *mut (), *const ()) -> i32>,
    reg: Option<unsafe extern "C" fn(*mut (), *mut bpf_link) -> i32>,
    unreg: Option<unsafe extern "C" fn(*mut (), *mut bpf_link)>,
    update: Option<unsafe extern "C" fn(*mut (), *mut (), *mut bpf_link) -> i32>,
    validate: Option<unsafe extern "C" fn(*mut ()) -> i32>,
    cfi_stubs: *mut (),
    owner: *mut module,
    name: *const u8,
    func_models: [btf_func_model; 64],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_prog_array_item {
    prog: *mut bpf_prog,
    __slate_anon_1: __SlateRecord_bpf_h_2532,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_prog_array {
    rcu: callback_head,
    items: [bpf_prog_array_item; 0],
}


// Per-cpu temp buffers used by printf-like helpers to store the bprintf binary
// arguments representation.
#[repr(C)]
#[derive(Clone, Copy)]
struct fs_parameter {
    key: *const u8,
    __slate_bits_0: __slate_bits::__SlateBits1126U0,
    __slate_anon_2: __SlateRecord_fs_context_h_66,
    size: u64,
    dirfd: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct p_log {
    prefix: *const u8,
    log: *mut fc_log,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_context {
    ops: *const fs_context_operations,
    uapi_mutex: mutex,
    fs_type: *mut file_system_type,
    fs_private: *mut (),
    sget_key: *mut (),
    root: *mut dentry,
    user_ns: *mut user_namespace,
    net_ns: *mut net,
    cred: *const cred,
    log: p_log,
    source: *const u8,
    security: *mut (),
    s_fs_info: *mut (),
    sb_flags: u32,
    sb_flags_mask: u32,
    s_iflags: u32,
    __slate_bits_0: __slate_bits::__SlateBits656U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_context_operations {
    free: Option<unsafe extern "C" fn(*mut fs_context)>,
    dup: Option<unsafe extern "C" fn(*mut fs_context, *mut fs_context) -> i32>,
    parse_param: Option<unsafe extern "C" fn(*mut fs_context, *mut fs_parameter) -> i32>,
    parse_monolithic: Option<unsafe extern "C" fn(*mut fs_context, *mut ()) -> i32>,
    get_tree: Option<unsafe extern "C" fn(*mut fs_context) -> i32>,
    reconfigure: Option<unsafe extern "C" fn(*mut fs_context) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fc_log {
    usage: refcount_struct,
    head: u8,
    tail: u8,
    need_free: u8,
    owner: *mut module,
    buffer: [*mut u8; 8],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_parameter_spec {
    name: *const u8,
    r#type: Option<unsafe extern "C" fn(*mut p_log, *const fs_parameter_spec, *mut fs_parameter, *mut fs_parse_result) -> i32>,
    opt: u8,
    flags: u16,
    data: *const (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_parse_result {
    negated: bool,
    __slate_anon_1: __SlateRecord_fs_parser_h_56,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct proc_ops {
    proc_flags: u32,
    proc_open: Option<unsafe extern "C" fn(*mut inode, *mut file) -> i32>,
    proc_read: Option<unsafe extern "C" fn(*mut file, *mut u8, u64, *mut i64) -> i64>,
    proc_read_iter: Option<unsafe extern "C" fn(*mut kiocb, *mut iov_iter) -> i64>,
    proc_write: Option<unsafe extern "C" fn(*mut file, *const u8, u64, *mut i64) -> i64>,
    proc_lseek: Option<unsafe extern "C" fn(*mut file, i64, i32) -> i64>,
    proc_release: Option<unsafe extern "C" fn(*mut inode, *mut file) -> i32>,
    proc_poll: Option<unsafe extern "C" fn(*mut file, *mut poll_table_struct) -> u32>,
    proc_ioctl: Option<unsafe extern "C" fn(*mut file, u32, u64) -> i64>,
    proc_compat_ioctl: Option<unsafe extern "C" fn(*mut file, u32, u64) -> i64>,
    proc_mmap: Option<unsafe extern "C" fn(*mut file, *mut vm_area_struct) -> i32>,
    proc_get_unmapped_area: Option<unsafe extern "C" fn(*mut file, u64, u64, u64, u64) -> u64>,
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


// No special request: A mapped anonymous (sub)page is possibly shared between
// processes.
// The anonymous (sub)page is exclusive to a single process.
fn ksm_might_need_to_copy(folio: *mut folio, vma: *mut vm_area_struct, addr: u64) -> *mut folio {
folio
}

#[repr(C)]
#[derive(Clone, Copy)]
struct trace_eval_map {
    system: *const u8,
    eval_string: *const u8,
    eval_value: u64,
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


fn poll_wait(filp: *mut file, wait_address: *mut wait_queue_head, p: *mut poll_table_struct) {
if !p.is_null() && (unsafe { (*p)._qproc }) != None {
unsafe { unsafe { (*p)._qproc }.unwrap()(filp, wait_address, p) };
// This memory barrier is paired in the wq_has_sleeper().
// See the comment above prepare_to_wait(), we need to
// ensure that subsequent tests in this thread can't be
// reordered with __add_wait_queue() in _qproc() paths.
loop {
kcsan_mb!();
unsafe {
core::arch::asm!("lock addl $0,-4(%rsp)", options(nostack, att_syntax));
}
if true {
break;
}
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct seq_buf {
    buffer: *mut u8,
    size: u64,
    len: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_seq {
    seq: seq_buf,
    readpos: u64,
    full: i32,
    buffer: [u8; 8156],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event_attr {
    r#type: u32,
    size: u32,
    config: u64,
    __slate_anon_3: __SlateRecord_perf_event_h_410,
    sample_type: u64,
    read_format: u64,
    __slate_bits_0: __slate_bits::__SlateBits1128U0,
    __slate_anon_46: __SlateRecord_perf_event_h_471,
    bp_type: u32,
    __slate_anon_48: __SlateRecord_perf_event_h_477,
    __slate_anon_49: __SlateRecord_perf_event_h_483,
    branch_sample_type: u64,
    sample_regs_user: u64,
    sample_stack_user: u32,
    clockid: i32,
    sample_regs_intr: u64,
    aux_watermark: u32,
    sample_max_stack: u16,
    __reserved_2: u16,
    aux_sample_size: u32,
    __slate_anon_59: __SlateRecord_perf_event_h_530,
    sig_data: u64,
    config3: u64,
    config4: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union perf_mem_data_src {
    val: u64,
    __slate_anon_1: __SlateRecord_perf_event_h_1322,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_branch_entry {
    from: u64,
    to: u64,
    __slate_bits_0: __slate_bits::__SlateBits1153U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
union perf_sample_weight {
    full: u64,
    __slate_anon_1: __SlateRecord_perf_event_h_1512,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct arch_hw_breakpoint {
    address: u64,
    mask: u64,
    len: u8,
    r#type: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_regs {
    abi: u64,
    regs: *mut pt_regs,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_callchain_entry {
    nr: u64,
    ip: [u64; 0],
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct perf_raw_frag {
    __slate_anon_0: __SlateRecord_perf_event_h_76,
    copy: Option<unsafe extern "C" fn(*mut (), *const (), u64, u64) -> u64>,
    data: *mut (),
    size: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_raw_record {
    frag: perf_raw_frag,
    size: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_branch_stack {
    nr: u64,
    hw_idx: u64,
    entries: [perf_branch_entry; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hw_perf_event_extra {
    config: u64,
    reg: u32,
    alloc: i32,
    idx: i32,
}


// hw_perf_event::flag values
//
// PERF_EVENT_FLAG_ARCH bits are reserved for architecture-specific
// usage.
#[repr(C)]
#[derive(Clone, Copy)]
struct hw_perf_event {
    __slate_anon_0: __SlateRecord_perf_event_h_149,
    target: *mut task_struct,
    addr_filters: *mut (),
    addr_filters_gen: u64,
    state: i32,
    prev_count: __SlateRecord_local64_h_26,
    sample_period: u64,
    __slate_anon_7: __SlateRecord_perf_event_h_244,
    interrupts_seq: u64,
    interrupts: u64,
    freq_time_stamp: u64,
    freq_count_stamp: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pmu {
    entry: list_head,
    events_lock: spinlock,
    events: list_head,
    module: *mut module,
    dev: *mut device,
    parent: *mut device,
    attr_groups: *mut *const attribute_group,
    attr_update: *mut *const attribute_group,
    name: *const u8,
    r#type: i32,
    capabilities: i32,
    scope: u32,
    cpu_pmu_context: *mut *mut perf_cpu_pmu_context,
    exclusive_cnt: __SlateRecord_types_h_190,
    task_ctx_nr: i32,
    hrtimer_interval_ms: i32,
    nr_addr_filters: u32,
    pmu_enable: Option<unsafe extern "C" fn(*mut pmu)>,
    pmu_disable: Option<unsafe extern "C" fn(*mut pmu)>,
    event_init: Option<unsafe extern "C" fn(*mut perf_event) -> i32>,
    event_mapped: Option<unsafe extern "C" fn(*mut perf_event, *mut mm_struct)>,
    event_unmapped: Option<unsafe extern "C" fn(*mut perf_event, *mut mm_struct)>,
    add: Option<unsafe extern "C" fn(*mut perf_event, i32) -> i32>,
    del: Option<unsafe extern "C" fn(*mut perf_event, i32)>,
    start: Option<unsafe extern "C" fn(*mut perf_event, i32)>,
    stop: Option<unsafe extern "C" fn(*mut perf_event, i32)>,
    read: Option<unsafe extern "C" fn(*mut perf_event)>,
    start_txn: Option<unsafe extern "C" fn(*mut pmu, u32)>,
    commit_txn: Option<unsafe extern "C" fn(*mut pmu) -> i32>,
    cancel_txn: Option<unsafe extern "C" fn(*mut pmu)>,
    event_idx: Option<unsafe extern "C" fn(*mut perf_event) -> i32>,
    sched_task: Option<unsafe extern "C" fn(*mut perf_event_pmu_context, *mut task_struct, bool)>,
    task_ctx_cache: *mut kmem_cache,
    setup_aux: Option<unsafe extern "C" fn(*mut perf_event, *mut *mut (), i32, bool) -> *mut ()>,
    free_aux: Option<unsafe extern "C" fn(*mut ())>,
    snapshot_aux: Option<unsafe extern "C" fn(*mut perf_event, *mut perf_output_handle, u64) -> i64>,
    addr_filters_validate: Option<unsafe extern "C" fn(*mut list_head) -> i32>,
    addr_filters_sync: Option<unsafe extern "C" fn(*mut perf_event)>,
    aux_output_match: Option<unsafe extern "C" fn(*mut perf_event) -> i32>,
    filter: Option<unsafe extern "C" fn(*mut pmu, i32) -> bool>,
    check_period: Option<unsafe extern "C" fn(*mut perf_event, u64) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_addr_filters_head {
    list: list_head,
    lock: raw_spinlock,
    nr_file_filters: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_addr_filter_range {
    start: u64,
    size: u64,
}


// Event capabilities. For event_caps and groups caps.
//
// PERF_EV_CAP_SOFTWARE: Is a software event.
// PERF_EV_CAP_READ_ACTIVE_PKG: A CPU event (or cgroup event) that can be read
// from any CPU in the package where it is active.
// PERF_EV_CAP_SIBLING: An event with this flag must be a group sibling and
// cannot be a group leader. If an event with this flag is detached from the
// group it is scheduled out and moved into an unrecoverable ERROR state.
// PERF_EV_CAP_READ_SCOPE: A CPU event that can be read from any CPU of the
// PMU scope where it is active.
#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
    event_entry: list_head,
    sibling_list: list_head,
    active_list: list_head,
    group_node: rb_node,
    group_index: u64,
    migrate_entry: list_head,
    hlist_entry: hlist_node,
    active_entry: list_head,
    nr_siblings: i32,
    event_caps: i32,
    group_caps: i32,
    group_generation: u32,
    group_leader: *mut perf_event,
    pmu: *mut pmu,
    pmu_private: *mut (),
    state: i32,
    attach_state: u32,
    count: __SlateRecord_local64_h_26,
    child_count: __SlateRecord_types_h_197,
    total_time_enabled: u64,
    total_time_running: u64,
    tstamp: u64,
    attr: perf_event_attr,
    header_size: u16,
    id_header_size: u16,
    read_size: u16,
    hw: hw_perf_event,
    ctx: *mut perf_event_context,
    pmu_ctx: *mut perf_event_pmu_context,
    refcount: __SlateRecord_types_h_197,
    child_total_time_enabled: __SlateRecord_types_h_197,
    child_total_time_running: __SlateRecord_types_h_197,
    child_mutex: mutex,
    child_list: list_head,
    parent: *mut perf_event,
    oncpu: i32,
    cpu: i32,
    owner_entry: list_head,
    owner: *mut task_struct,
    mmap_mutex: mutex,
    mmap_count: refcount_struct,
    rb: *mut perf_buffer,
    rb_entry: list_head,
    rcu_batches: u64,
    rcu_pending: i32,
    waitq: wait_queue_head,
    fasync: *mut fasync_struct,
    pending_wakeup: u32,
    pending_kill: u32,
    pending_disable: u32,
    pending_addr: u64,
    pending_irq: irq_work,
    pending_disable_irq: irq_work,
    pending_task: callback_head,
    pending_work: u32,
    event_limit: __SlateRecord_types_h_190,
    addr_filters: perf_addr_filters_head,
    addr_filter_ranges: *mut perf_addr_filter_range,
    addr_filters_gen: u64,
    aux_event: *mut perf_event,
    destroy: Option<unsafe extern "C" fn(*mut perf_event)>,
    callback_head: callback_head,
    ns: *mut pid_namespace,
    id: u64,
    lost_samples: __SlateRecord_types_h_197,
    clock: Option<unsafe extern "C" fn() -> u64>,
    overflow_handler: Option<unsafe extern "C" fn(*mut perf_event, *mut perf_sample_data, *mut pt_regs)>,
    overflow_handler_context: *mut (),
    prog: *mut bpf_prog,
    bpf_cookie: u64,
    tp_event: *mut trace_event_call,
    filter: *mut event_filter,
    cgrp: *mut perf_cgroup,
    security: *mut (),
    sb_list: list_head,
    pmu_list: list_head,
    orig_type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event_pmu_context {
    pmu: *mut pmu,
    ctx: *mut perf_event_context,
    pmu_ctx_entry: list_head,
    pinned_active: list_head,
    flexible_active: list_head,
    __slate_bits_0: __slate_bits::__SlateBits1178U0,
    nr_events: u32,
    nr_cgroups: u32,
    nr_freq: u32,
    refcount: __SlateRecord_types_h_190,
    callback_head: callback_head,
    rotate_necessary: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event_groups {
    tree: rb_root,
    index: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_time_ctx {
    time: u64,
    stamp: u64,
    offset: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event_context {
    lock: raw_spinlock,
    mutex: mutex,
    pmu_ctx_list: list_head,
    pinned_groups: perf_event_groups,
    flexible_groups: perf_event_groups,
    event_list: list_head,
    nr_events: i32,
    nr_user: i32,
    is_active: i32,
    nr_stat: i32,
    nr_freq: i32,
    rotate_disable: i32,
    refcount: refcount_struct,
    task: *mut task_struct,
    time: perf_time_ctx,
    timeguest: perf_time_ctx,
    parent_ctx: *mut perf_event_context,
    parent_gen: u64,
    generation: u64,
    pin_count: i32,
    nr_cgroups: i32,
    callback_head: callback_head,
    nr_no_switch_fast: __SlateRecord_local_h_12,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_ctx_data {
    callback_head: callback_head,
    refcount: refcount_struct,
    global: i32,
    ctx_cache: *mut kmem_cache,
    data: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_cpu_pmu_context {
    epc: perf_event_pmu_context,
    task_epc: *mut perf_event_pmu_context,
    sched_cb_entry: list_head,
    sched_cb_usage: i32,
    active_oncpu: i32,
    exclusive: i32,
    pmu_disable_count: i32,
    hrtimer_lock: raw_spinlock,
    hrtimer: hrtimer,
    hrtimer_interval: i64,
    hrtimer_active: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_output_handle {
    event: *mut perf_event,
    rb: *mut perf_buffer,
    wakeup: u64,
    size: u64,
    __slate_anon_4: __SlateRecord_perf_event_h_1156,
    __slate_anon_5: __SlateRecord_perf_event_h_1163,
    page: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_cgroup_info {
    time: perf_time_ctx,
    timeguest: perf_time_ctx,
    active: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_cgroup {
    css: cgroup_subsys_state,
    info: *mut perf_cgroup_info,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct perf_sample_data {
    sample_flags: u64,
    period: u64,
    dyn_size: u64,
    r#type: u64,
    tid_entry: __SlateRecord_perf_event_h_1315,
    time: u64,
    id: u64,
    cpu_entry: __SlateRecord_perf_event_h_1321,
    ip: u64,
    callchain: *mut perf_callchain_entry,
    raw: *mut perf_raw_record,
    br_stack: *mut perf_branch_stack,
    br_stack_cntr: *mut u64,
    weight: perf_sample_weight,
    data_src: perf_mem_data_src,
    txn: u64,
    regs_user: perf_regs,
    regs_intr: perf_regs,
    stack_user_size: u64,
    stream_id: u64,
    cgroup: u64,
    addr: u64,
    phys_addr: u64,
    data_page_size: u64,
    code_page_size: u64,
    aux_size: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_entry {
    r#type: u16,
    flags: u8,
    preempt_count: u8,
    pid: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_iterator {
    tr: *mut trace_array,
    trace: *mut tracer,
    array_buffer: *mut array_buffer,
    private: *mut (),
    cpu_file: i32,
    mutex: mutex,
    buffer_iter: *mut *mut ring_buffer_iter,
    iter_flags: u64,
    temp: *mut (),
    temp_size: u32,
    fmt: *mut u8,
    fmt_size: u32,
    wait_index: __SlateRecord_types_h_190,
    tmp_seq: trace_seq,
    started: [cpumask; 1],
    closed: bool,
    snapshot: bool,
    seq: trace_seq,
    ent: *mut trace_entry,
    lost_events: u64,
    leftover: i32,
    ent_size: i32,
    cpu: i32,
    ts: u64,
    pos: i64,
    idx: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_functions {
    trace: Option<unsafe extern "C" fn(*mut trace_iterator, i32, *mut trace_event) -> u32>,
    raw: Option<unsafe extern "C" fn(*mut trace_iterator, i32, *mut trace_event) -> u32>,
    hex: Option<unsafe extern "C" fn(*mut trace_iterator, i32, *mut trace_event) -> u32>,
    binary: Option<unsafe extern "C" fn(*mut trace_iterator, i32, *mut trace_event) -> u32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event {
    node: hlist_node,
    r#type: i32,
    funcs: *mut trace_event_functions,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_fields {
    r#type: *const u8,
    __slate_anon_1: __SlateRecord_trace_events_h_275,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_class {
    system: *const u8,
    probe: *mut (),
    perf_probe: *mut (),
    reg: Option<unsafe extern "C" fn(*mut trace_event_call, u32, *mut ()) -> i32>,
    fields_array: *mut trace_event_fields,
    get_fields: Option<unsafe extern "C" fn(*mut trace_event_call) -> *mut list_head>,
    fields: list_head,
    raw_init: Option<unsafe extern "C" fn(*mut trace_event_call) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_call {
    list: list_head,
    class: *mut trace_event_class,
    __slate_anon_2: __SlateRecord_trace_events_h_379,
    event: trace_event,
    print_fmt: *mut u8,
    __slate_anon_5: __SlateRecord_trace_events_h_390,
    data: *mut (),
    flags: i32,
    perf_refcount: i32,
    perf_events: *mut hlist_head,
    prog_array: *mut bpf_prog_array,
    perf_perm: Option<unsafe extern "C" fn(*mut trace_event_call, *mut perf_event) -> i32>,
}


fn set_current_oom_origin() {
unsafe {
(*(*get_current()).signal).oom_flag_origin = true;
}
}

fn clear_current_oom_origin() {
unsafe {
(*(*get_current()).signal).oom_flag_origin = false;
}
}

fn check_stable_address_space(mm: *mut mm_struct) -> u32 {
if unlikely!(mm_flags_test(22i32, mm as *const mm_struct)) != 0 {
return 2;
}
0
}

fn pm_suspended_storage() -> bool {
!gfp_has_io_fs(unsafe { gfp_allowed_mask })
}

// PLIST_HEAD - declare and init plist_head
// @head:	name for struct plist_head variable
fn plist_node_init(node: *mut plist_node, prio: i32) {
unsafe {
(*node).prio = prio;
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*node).prio_list) });
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*node).node_list) });
}

// plist_for_each_entry	- iterate over list of given type
// @pos:	the type * to use as a loop counter
// @head:	the head for your list
// @mem:	the name of the list_head within the struct
// plist_for_each_entry_safe - iterate safely over list of given type
// @pos:	the type * to use as a loop counter
// @n:		another type * to use as temporary storage
// @head:	the head for your list
// @m:		the name of the list_head within the struct
//
// Iterate over list of given type, safe against removal of list entry.
fn plist_node_empty(node: *const plist_node) -> i32 {
list_empty(unsafe { core::ptr::addr_of!((*node).node_list) })
}

fn softleaf_mk_none() -> __SlateRecord_mm_types_h_288 {
let mut __slate_compound_0: __SlateRecord_mm_types_h_288 = unsafe { core::mem::zeroed() };
return unsafe { *{
__slate_compound_0 = __SlateRecord_mm_types_h_288 { val: 0 };
core::ptr::addr_of_mut!(__slate_compound_0)
} };
}

fn softleaf_from_pte(mut pte: __SlateRecord_pgtable_64_types_h_21) -> __SlateRecord_mm_types_h_288 {
let mut __slate_compound_1: __SlateRecord_mm_types_h_288 = unsafe { core::mem::zeroed() };
let mut arch_entry: __SlateRecord_mm_types_h_288 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(arch_entry) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let __v0: bool = pte_present(pte) != 0 || pte_none(pte) != 0;
if __v0 {
return softleaf_mk_none();
}
pte = pte_swp_clear_flags(pte);
arch_entry = unsafe { *{
__slate_compound_1 = __SlateRecord_mm_types_h_288 { val: pte_val!(pte) };
core::ptr::addr_of_mut!(__slate_compound_1)
} };
// Temporary until swp_entry_t eliminated.
return swp_entry(__swp_type!(arch_entry), __swp_offset!(arch_entry));
}

fn softleaf_to_pte(entry: __SlateRecord_mm_types_h_288) -> __SlateRecord_pgtable_64_types_h_21 {
// Temporary until swp_entry_t eliminated.
return swp_entry_to_pte(entry);
}

fn softleaf_is_none(entry: __SlateRecord_mm_types_h_288) -> bool {
entry.val == 0
}

fn softleaf_type(entry: __SlateRecord_mm_types_h_288) -> u32 {
let mut type_num: u32 = 0;
if softleaf_is_none(entry) {
return 0;
}
type_num = (entry.val >> 58i32) as u32;
if type_num < 28u32 {
return 1;
}
match type_num {
28 => {
return 2;
}
29 => {
return 3;
}
30 => {
return 4;
}
31 => {
return 9;
}
_ => {
}
}
// Unknown entry type.
VM_WARN_ON_ONCE!();
0
}

fn softleaf_is_swap(entry: __SlateRecord_mm_types_h_288) -> bool {
(softleaf_type(entry) as u32) == 1
}

// At least 2 bits are needed to distinguish SWP_TB_COUNT_MAX, 1 and 0
// If there are enough bits besides PFN and marker, store zero flag inline
#[unsafe(no_mangle)]
static mut swap_info: __SlateAlign16<[*mut swap_info_struct; 28]> = __SlateAlign16(unsafe { core::mem::zeroed() });

#[repr(C)]
#[derive(Clone, Copy)]
struct swap_cluster_info {
    lock: spinlock,
    count: u16,
    flags: u8,
    order: u8,
    table: *mut __SlateRecord_types_h_197,
    extend_table: *mut u32,
    list: list_head,
}


fn swp_cluster_offset(entry: __SlateRecord_mm_types_h_288) -> u32 {
(swp_offset(entry) % 256) as u32
}

fn __swap_type_to_info(r#type: i32) -> *mut swap_info_struct {
let mut __v0: *mut swap_info_struct;
loop {
if !(__native_word__17!() || ((const { core::mem::size_of::<*mut swap_info_struct>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_731() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile((core::ptr::addr_of_mut!(swap_info.0) as *mut *mut swap_info_struct).wrapping_offset(r#type as isize) as *const *mut swap_info_struct) };
let si: *mut swap_info_struct = __v0; // rcu_dereference()
VM_WARN_ON_ONCE!(); // race with swapoff
si
}

fn __swap_entry_to_info(entry: __SlateRecord_mm_types_h_288) -> *mut swap_info_struct {
__swap_type_to_info(swp_type(entry) as i32)
}

fn __swap_offset_to_cluster(si: *mut swap_info_struct, offset: u64) -> *mut swap_cluster_info {
VM_WARN_ON_ONCE!(); // race with swapoff
VM_WARN_ON_ONCE!();
unsafe { (*si).cluster_info }.wrapping_offset((offset / 256) as isize)
}

fn __swap_entry_to_cluster(entry: __SlateRecord_mm_types_h_288) -> *mut swap_cluster_info {
__swap_offset_to_cluster(__swap_entry_to_info(entry), swp_offset(entry))
}

#[inline(always)]
fn __swap_cluster_lock(si: *mut swap_info_struct, offset: u64, irq: bool) -> *mut swap_cluster_info {
let ci: *mut swap_cluster_info = __swap_offset_to_cluster(si, offset);
// Nothing modifies swap cache in an IRQ context. All access to
// swap cache is wrapped by swap_cache_* helpers, and swap cache
// writeback is handled outside of IRQs. Swapin or swapout never
// occurs in IRQ, and neither does in-place split or replace.
//
// Besides, modifying swap cache requires synchronization with
// swap_map, which was never IRQ safe.
VM_WARN_ON_ONCE!();
VM_WARN_ON_ONCE!(); // race with swapoff
if irq {
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*ci).lock) });
} else {
spin_lock(unsafe { core::ptr::addr_of_mut!((*ci).lock) });
}
ci
}

fn swap_cluster_lock(si: *mut swap_info_struct, offset: u64) -> *mut swap_cluster_info {
__swap_cluster_lock(si, offset, false)
}

fn swap_cluster_unlock(ci: *mut swap_cluster_info) {
spin_unlock(unsafe { core::ptr::addr_of_mut!((*ci).lock) });
}

fn folio_matches_swap_entry(folio: *const folio, entry: __SlateRecord_mm_types_h_288) -> bool {
let folio_entry: __SlateRecord_mm_types_h_288 = unsafe { (*folio).__slate_anon_0.__slate_anon_0.__slate_anon_4.swap };
let nr_pages: i64 = folio_nr_pages(folio) as i64;
VM_WARN_ON_ONCE_FOLIO!();
if !folio_test_swapcache(folio) {
return false;
}
VM_WARN_ON_ONCE_FOLIO!();
folio_entry.val == round_down__3!(entry.val, nr_pages)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct swap_table {
    entries: [__SlateRecord_types_h_197; 256],
}


// NULL Entry, all 0
// Swapped out: shadow
// Flags: For PFN or shadow, contains SWAP_COUNT, width changes
// The first flag is zero bit (SWAP_TABLE_HAS_ZEROFLAG)
// Bad slot: ends with 0b1000 and rests of bits are all 1
fn null_to_swp_tb() -> u64 {
loop {
if ((const { core::mem::size_of::<u64>() }) as u64) != ((const { core::mem::size_of::<__SlateRecord_types_h_197>() }) as u64) {
unsafe { __compiletime_assert_732() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
0
}

fn __count_to_swp_tb(count: u8) -> u64 {
// At least three values are needed to distinguish free (0),
// used (count > 0 && count < SWP_TB_COUNT_MAX), and
// overflow (count == SWP_TB_COUNT_MAX).
loop {
let mut __v0: i32;
let __UNIQUE_ID_x__733: i32 = 5;
let __UNIQUE_ID_y__734: i32 = 22i32;
loop {
if __types_ok!(__UNIQUE_ID_x__733, __UNIQUE_ID_y__734) == 0 {
unsafe { __compiletime_assert_735() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__733 < __UNIQUE_ID_y__734 { __UNIQUE_ID_x__733 } else { __UNIQUE_ID_y__734 };
if __v0.wrapping_sub((22i32 > 2) as i32) < 2 {
unsafe { __compiletime_assert_736() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v3: i32;
let __UNIQUE_ID_x__740: i32 = 5;
let __UNIQUE_ID_y__741: i32 = 22i32;
loop {
if __types_ok!(__UNIQUE_ID_x__740, __UNIQUE_ID_y__741) == 0 {
unsafe { __compiletime_assert_742() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = if __UNIQUE_ID_x__740 < __UNIQUE_ID_y__741 { __UNIQUE_ID_x__740 } else { __UNIQUE_ID_y__741 };
(count as u64) << 64i32.wrapping_sub(__v3.wrapping_sub((22i32 > 2) as i32))
}

fn __flags_to_swp_tb(flags: u8) -> u64 {
loop {
let mut __v0: i32;
let __UNIQUE_ID_x__743: i32 = 5;
let __UNIQUE_ID_y__744: i32 = 22i32;
loop {
if __types_ok!(__UNIQUE_ID_x__743, __UNIQUE_ID_y__744) == 0 {
unsafe { __compiletime_assert_745() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__743 < __UNIQUE_ID_y__744 { __UNIQUE_ID_x__743 } else { __UNIQUE_ID_y__744 };
if __v0 > 8 {
unsafe { __compiletime_assert_746() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v3: i32;
let __UNIQUE_ID_x__750: i32 = 5;
let __UNIQUE_ID_y__751: i32 = 22i32;
loop {
if __types_ok!(__UNIQUE_ID_x__750, __UNIQUE_ID_y__751) == 0 {
unsafe { __compiletime_assert_752() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = if __UNIQUE_ID_x__750 < __UNIQUE_ID_y__751 { __UNIQUE_ID_x__750 } else { __UNIQUE_ID_y__751 };
(flags as u64) << 64i32.wrapping_sub(__v3)
}

fn shadow_to_swp_tb(shadow: *mut (), flags: u8) -> u64 {
loop {
if 64u64 != 8u64.wrapping_mul((const { core::mem::size_of::<u64>() }) as u64) {
unsafe { __compiletime_assert_761() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (xa_mk_value(0u64) as u64) != 1 {
unsafe { __compiletime_assert_762() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
VM_WARN_ON_ONCE!();
(shadow as u64) | 1 | __flags_to_swp_tb(flags)
}

fn swp_tb_is_null(swp_tb: u64) -> bool {
swp_tb == 0
}

fn swp_tb_is_folio(swp_tb: u64) -> bool {
swp_tb & BIT!(2i32).wrapping_sub(1) == 2
}

fn swp_tb_is_shadow(swp_tb: u64) -> bool {
xa_is_value(swp_tb as *const ())
}

fn swp_tb_is_bad(swp_tb: u64) -> bool {
swp_tb == 18446744073709551608u64
}

fn swp_tb_is_countable(swp_tb: u64) -> bool {
let __v0: bool = swp_tb_is_shadow(swp_tb) || swp_tb_is_folio(swp_tb);
__v0 || swp_tb_is_null(swp_tb)
}

fn __swp_tb_get_count(swp_tb: u64) -> u8 {
VM_WARN_ON!();
let mut __v0: i32;
let __UNIQUE_ID_x__772: i32 = 5;
let __UNIQUE_ID_y__773: i32 = 22i32;
loop {
if __types_ok!(__UNIQUE_ID_x__772, __UNIQUE_ID_y__773) == 0 {
unsafe { __compiletime_assert_774() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__772 < __UNIQUE_ID_y__773 { __UNIQUE_ID_x__772 } else { __UNIQUE_ID_y__773 };
let mut __v2: i32;
let __UNIQUE_ID_x__775: i32 = 5;
let __UNIQUE_ID_y__776: i32 = 22i32;
loop {
if __types_ok!(__UNIQUE_ID_x__775, __UNIQUE_ID_y__776) == 0 {
unsafe { __compiletime_assert_777() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = if __UNIQUE_ID_x__775 < __UNIQUE_ID_y__776 { __UNIQUE_ID_x__775 } else { __UNIQUE_ID_y__776 };
((swp_tb & !(18446744073709551615u64 >> __v0.wrapping_sub((22i32 > 2) as i32))) >> 64i32.wrapping_sub(__v2.wrapping_sub((22i32 > 2) as i32))) as u8
}

fn swp_tb_get_count(swp_tb: u64) -> i32 {
if swp_tb_is_countable(swp_tb) {
return __swp_tb_get_count(swp_tb) as i32;
}
22i32.wrapping_neg()
}

fn __swp_tb_mk_count(swp_tb: u64, count: i32) -> u64 {
let mut __v0: i32;
let __UNIQUE_ID_x__784: i32 = 5;
let __UNIQUE_ID_y__785: i32 = 22i32;
loop {
if __types_ok!(__UNIQUE_ID_x__784, __UNIQUE_ID_y__785) == 0 {
unsafe { __compiletime_assert_786() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__784 < __UNIQUE_ID_y__785 { __UNIQUE_ID_x__784 } else { __UNIQUE_ID_y__785 };
swp_tb & !(!(18446744073709551615u64 >> __v0.wrapping_sub((22i32 > 2) as i32))) | __count_to_swp_tb(count as u8)
}

fn __swap_table_set(ci: *mut swap_cluster_info, off: u32, swp_tb: u64) {
let table: *mut __SlateRecord_types_h_197 = rcu_dereference_protected!(unsafe { (*ci).table }, 1i32);
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!((*ci).lock) });
VM_WARN_ON_ONCE!();
atomic_long_set(table.wrapping_offset(off as isize), swp_tb as i64);
}

fn __swap_table_xchg(ci: *mut swap_cluster_info, off: u32, swp_tb: u64) -> u64 {
let table: *mut __SlateRecord_types_h_197 = rcu_dereference_protected!(unsafe { (*ci).table }, 1i32);
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!((*ci).lock) });
VM_WARN_ON_ONCE!();
// Ordering is guaranteed by cluster lock, relax
atomic_long_xchg_relaxed(table.wrapping_offset(off as isize), swp_tb as i64) as u64
}

fn __swap_table_get(ci: *mut swap_cluster_info, off: u32) -> u64 {
VM_WARN_ON_ONCE!();
let mut __v0: *mut __SlateRecord_types_h_197;
let mut __v1: *mut __SlateRecord_types_h_197;
loop {
if !(__native_word__15!() || ((const { core::mem::size_of::<*mut __SlateRecord_types_h_197>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_788() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*ci).table) as *const *mut __SlateRecord_types_h_197) };
let __UNIQUE_ID_rcu_787: *mut __SlateRecord_types_h_197 = __v1;
loop {
let __v4: bool = if false {
let __v5: bool = (unsafe { lockdep_is_held(core::ptr::addr_of_mut!((*ci).lock) as *const ()) }) != 0 || rcu_read_lock_held() != 0;
!__v5
} else {
false
};
if !__v4 {
break;
}
}
__v0 = __UNIQUE_ID_rcu_787;
let table: *mut __SlateRecord_types_h_197 = __v0;
atomic_long_read(table.wrapping_offset(off as isize) as *const __SlateRecord_types_h_197) as u64
}

fn swap_table_get(ci: *mut swap_cluster_info, off: u32) -> u64 {
let mut swp_tb: u64 = 0;
VM_WARN_ON_ONCE!();
rcu_read_lock();
let mut __v0: *mut __SlateRecord_types_h_197;
let mut __v1: *mut __SlateRecord_types_h_197;
loop {
if !(__native_word__15!() || ((const { core::mem::size_of::<*mut __SlateRecord_types_h_197>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_790() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*ci).table) as *const *mut __SlateRecord_types_h_197) };
let __UNIQUE_ID_rcu_789: *mut __SlateRecord_types_h_197 = __v1;
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
__v0 = __UNIQUE_ID_rcu_789;
let table: *mut __SlateRecord_types_h_197 = __v0;
let __v6: u64 = if !table.is_null() {
atomic_long_read(table.wrapping_offset(off as isize) as *const __SlateRecord_types_h_197) as u64
} else {
null_to_swp_tb()
};
swp_tb = __v6;
rcu_read_unlock();
swp_tb
}

fn __swap_table_clear_zero(ci: *mut swap_cluster_info, ci_off: u32) {
let mut swp_tb: u64 = __swap_table_get(ci, ci_off);
VM_WARN_ON!();
let mut __v0: i32;
let __UNIQUE_ID_x__804: i32 = 5;
let __UNIQUE_ID_y__805: i32 = 22i32;
loop {
if __types_ok!(__UNIQUE_ID_x__804, __UNIQUE_ID_y__805) == 0 {
unsafe { __compiletime_assert_806() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__804 < __UNIQUE_ID_y__805 { __UNIQUE_ID_x__804 } else { __UNIQUE_ID_y__805 };
let __v2: u64 = !(UL!() << 64i32.wrapping_sub(__v0));
swp_tb &= __v2;
__swap_table_set(ci, ci_off, swp_tb);
}

fn __swap_cgroup_get(ci: *mut swap_cluster_info, ci_off: u32) -> u16 {
0
}

fn __swap_cgroup_clear(ci: *mut swap_cluster_info, ci_off: u32, nr: u64) -> u16 {
0
}

fn folio_swap(entry: __SlateRecord_mm_types_h_288, folio: *const folio) -> __SlateRecord_mm_types_h_288 {
let swap: __SlateRecord_mm_types_h_288 = __SlateRecord_mm_types_h_288 { val: ALIGN_DOWN!(entry.val, folio_nr_pages(folio)) };
return swap;
}

// Compare PTEs respecting the dirty bit.
// Compare PTEs respecting the soft-dirty bit.
// Compare PTEs respecting the writable bit.
// Merge PTE write bits: if any PTE in the batch is writable, modify the
// PTE at @ptentp to be writable.
// Merge PTE young and dirty bits: if any PTE in the batch is young or dirty,
// modify the PTE at @ptentp to be young or dirty, respectively.
#[repr(C)]
#[derive(Clone, Copy)]
struct capture_control {
    zone: *mut zone,
    migratetype: i32,
    order: i32,
    page: *mut page,
}


fn move_cluster(si: *mut swap_info_struct, ci: *mut swap_cluster_info, list: *mut list_head, new_flags: u32) {
VM_WARN_ON!();
loop {
if 1i32 << ((const { core::mem::size_of::<u8>() }) as u64).wrapping_mul(8) < 6 {
unsafe { __compiletime_assert_841() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!((*ci).lock) });
spin_lock(unsafe { core::ptr::addr_of_mut!((*si).lock) });
if ((unsafe { (*ci).flags }) as i32) == 0 {
list_add_tail(unsafe { core::ptr::addr_of_mut!((*ci).list) }, list);
} else {
list_move_tail(unsafe { core::ptr::addr_of_mut!((*ci).list) }, list);
}
spin_unlock(unsafe { core::ptr::addr_of_mut!((*si).lock) });
unsafe {
(*ci).flags = (new_flags as u32) as u8;
}
}

/// Protects the swap_info array, and the SWP_USED flag. swap_info contains
/// lazily allocated & freed swap device info struts, and SWP_USED indicates
/// which device is used, ~SWP_USED devices and can be reused.
///
/// Also protects swap_active_head total_swap_pages, and the SWP_WRITEOK flag.
static mut swap_lock: spinlock = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = unsafe { core::mem::zeroed() };
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0
} };

static mut nr_swapfiles: u32 = unsafe { core::mem::zeroed() };

/// Some modules use swappable objects and may try to swap them out under
/// memory pressure (via the shrinker). Before doing so, they may wish to
/// check to see if any swap space is available.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_nr_swap_pages_817: *mut () = (unsafe { core::ptr::addr_of_mut!(nr_swap_pages) }) as *mut ();

/// protected with swap_lock. reading in vm_swap_full() doesn't need lock
static Bad_file: __SlateAlign16<[u8; 21]> = __SlateAlign16([66u8, 97u8, 100u8, 32u8, 115u8, 119u8, 97u8, 112u8, 32u8, 102u8, 105u8, 108u8, 101u8, 32u8, 101u8, 110u8, 116u8, 114u8, 121u8, 32u8, 0u8]);

static Bad_offset: __SlateAlign16<[u8; 23]> = __SlateAlign16([66u8, 97u8, 100u8, 32u8, 115u8, 119u8, 97u8, 112u8, 32u8, 111u8, 102u8, 102u8, 115u8, 101u8, 116u8, 32u8, 101u8, 110u8, 116u8, 114u8, 121u8, 32u8, 0u8]);

/// all active swap_info_structs
/// protected with swap_lock, and ordered by priority.
static mut swap_active_head: plist_head = plist_head { node_list: list_head { next: unsafe { core::ptr::addr_of_mut!(swap_active_head.node_list) }, prev: unsafe { core::ptr::addr_of_mut!(swap_active_head.node_list) } } };

/// all available (active, not full) swap_info_structs
/// protected with swap_avail_lock, ordered by priority.
/// This is used by folio_alloc_swap() instead of swap_active_head
/// because swap_active_head includes all swap_info_structs,
/// but folio_alloc_swap() doesn't need to look at full ones.
/// This uses its own lock instead of swap_lock because when a
/// swap_info_struct changes between not-full/full, it needs to
/// add/remove itself to/from this list, but the swap_info_struct->lock
/// is held and the locking order requires swap_lock to be taken
/// before any swap_info_struct->lock.
static mut swap_avail_head: plist_head = plist_head { node_list: list_head { next: unsafe { core::ptr::addr_of_mut!(swap_avail_head.node_list) }, prev: unsafe { core::ptr::addr_of_mut!(swap_avail_head.node_list) } } };

static mut swap_avail_lock: spinlock = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = unsafe { core::mem::zeroed() };
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0
} };

static mut swap_table_cachep: *mut kmem_cache = unsafe { core::mem::zeroed() };

/// Protects si->swap_file for /proc/swaps usage
static mut swapon_mutex: mutex = {
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

static mut proc_poll_wait: wait_queue_head = {
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
__t0.head = list_head { next: unsafe { core::ptr::addr_of_mut!(proc_poll_wait.head) }, prev: unsafe { core::ptr::addr_of_mut!(proc_poll_wait.head) } };
__t0
};

/// Activity counter to indicate that a swapon or swapoff has occurred
static mut proc_poll_event: __SlateRecord_types_h_190 = __SlateRecord_types_h_190 { counter: 0i32 };

#[unsafe(no_mangle)]
static mut nr_rotate_swap: __SlateRecord_types_h_190 = __SlateRecord_types_h_190 { counter: 0i32 };

#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_swap_cluster {
    si: [*mut swap_info_struct; 1],
    offset: [u64; 1],
    lock: local_lock,
}


#[unsafe(link_section = ".data..percpu")]
static mut percpu_swap_cluster: percpu_swap_cluster = percpu_swap_cluster { si: [core::ptr::null_mut::<swap_info_struct>()], offset: [0u64], lock: local_lock {  } };

// May return NULL on invalid type, caller must check for NULL return
fn swap_type_to_info(r#type: i32) -> *mut swap_info_struct {
if r#type < 0 || r#type >= 28i32 {
return core::ptr::null_mut::<swap_info_struct>();
}
let mut __v0: *mut swap_info_struct;
loop {
if !(__native_word__17!() || ((const { core::mem::size_of::<*mut swap_info_struct>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_818() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile((core::ptr::addr_of_mut!(swap_info.0) as *mut *mut swap_info_struct).wrapping_offset(r#type as isize) as *const *mut swap_info_struct) };
return __v0; // rcu_dereference()
}

/// May return NULL on invalid entry, caller must check for NULL return
fn swap_entry_to_info(entry: __SlateRecord_mm_types_h_288) -> *mut swap_info_struct {
swap_type_to_info(swp_type(entry) as i32)
}

// Use the second highest bit of inuse_pages counter as the indicator
// if one swap device is on the available plist, so the atomic can
// still be updated arithmetically while having special data embedded.
//
// inuse_pages counter is the only thing indicating if a device should
// be on avail_lists or not (except swapon / swapoff). By embedding the
// off-list bit in the atomic counter, updates no longer need any lock
// to check the list status.
//
// This bit will be set if the device is not on the plist and not
// usable, will be cleared if the device is on the plist.
fn swap_usage_in_pages(si: *mut swap_info_struct) -> i64 {
((atomic_long_read((unsafe { core::ptr::addr_of_mut!((*si).inuse_pages) }) as *const __SlateRecord_types_h_197) as u64) & !(1u64 << BITS_PER_TYPE__2!(__SlateRecord_types_h_197).wrapping_sub(2))) as i64
}

// Reclaim the swap entry anyway if possible
// Reclaim the swap entry if there are no more mappings of the
// corresponding page
// Reclaim the swap entry if swap is getting full
fn swap_only_has_cache(ci: *mut swap_cluster_info, offset: u64, nr_pages: i32) -> bool {
let mut ci_off: u32 = (offset % 256) as u32;
let ci_end: u32 = ci_off.wrapping_add(nr_pages as u32);
let mut swp_tb: u64 = 0;
loop {
swp_tb = __swap_table_get(ci, ci_off);
VM_WARN_ON_ONCE!();
if swp_tb_get_count(swp_tb) != 0 {
return false;
}
let __v2: u32 = ci_off.wrapping_add(1);
ci_off = __v2;
if __v2 >= ci_end {
break;
}
}
true
}

/// returns number of pages in the folio that backs the swap entry. If positive,
/// the folio was reclaimed. If negative, the folio was not reclaimed. If 0, no
/// folio was associated with the swap entry.
fn __try_to_reclaim_swap(si: *mut swap_info_struct, mut offset: u64, flags: u64) -> i32 {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut __v2: bool = unsafe { core::mem::zeroed() };
let mut __v3: bool = unsafe { core::mem::zeroed() };
let mut __v4: bool = unsafe { core::mem::zeroed() };
let mut need_reclaim: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut bool = core::ptr::addr_of_mut!(need_reclaim) as *mut bool;
let mut nr_pages: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut i32 = core::ptr::addr_of_mut!(nr_pages) as *mut i32;
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut folio: core::mem::MaybeUninit<*mut folio> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut folio = core::ptr::addr_of_mut!(folio) as *mut *mut folio;
let mut ci: core::mem::MaybeUninit<*mut swap_cluster_info> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut swap_cluster_info = core::ptr::addr_of_mut!(ci) as *mut *mut swap_cluster_info;
let mut entry: core::mem::MaybeUninit<__SlateRecord_mm_types_h_288> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut __SlateRecord_mm_types_h_288 = core::ptr::addr_of_mut!(entry) as *mut __SlateRecord_mm_types_h_288;
unsafe {
core::ptr::write(__slate_slot_10, swp_entry((*si).r#type as u64, offset));
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
'__join_0: {
loop {
*__slate_slot_8 = swap_cache_get_folio(*__slate_slot_10);
if (*__slate_slot_8).is_null() {
return 0;
} else {
*__slate_slot_6 = folio_nr_pages(*__slate_slot_8 as *const folio) as i32;
*__slate_slot_7 = (*__slate_slot_6).wrapping_neg();
// We hold a folio lock here. We have to use trylock for
// avoiding deadlock. This is a special case and you should
// use folio_free_swap() with explicit folio_lock() in usual
// operations.
if !folio_trylock(*__slate_slot_8) {
break '__join_0;
} else {
// Offset could point to the middle of a large folio, or folio
// may no longer point to the expected offset before it's locked.
if !folio_matches_swap_entry(*__slate_slot_8 as *const folio, *__slate_slot_10) {
folio_unlock(*__slate_slot_8);
folio_put(*__slate_slot_8);
} else {
break;
}
}
}
}
offset = swp_offset((*(*__slate_slot_8)).__slate_anon_0.__slate_anon_0.__slate_anon_4.swap);
if flags & 1 != 0 {
__v4 = true;
} else {
if flags & 2 != 0 {
__v3 = !folio_mapped(*__slate_slot_8 as *const folio);
} else {
__v3 = false;
}
__v4 = __v3;
}
if __v4 {
__v2 = true;
} else {
if flags & 4 != 0 {
__v1 = mem_cgroup_swap_full(*__slate_slot_8);
} else {
__v1 = false;
}
__v2 = __v1;
}
*__slate_slot_5 = __v2;
if !(*__slate_slot_5) {
__v0 = true;
} else {
__v0 = !folio_swapcache_freeable(*__slate_slot_8);
}
if !__v0 {
// It's safe to delete the folio from swap cache only if the folio
// is in swap cache with swap count == 0. The slots have no page table
// reference or pending writeback, and can't be allocated to others.
*__slate_slot_9 = swap_cluster_lock(si, offset);
*__slate_slot_5 = swap_only_has_cache(*__slate_slot_9, offset, *__slate_slot_6);
swap_cluster_unlock(*__slate_slot_9);
if *__slate_slot_5 {
swap_cache_del_folio(*__slate_slot_8);
folio_set_dirty(*__slate_slot_8);
*__slate_slot_7 = *__slate_slot_6;
}
}
folio_unlock(*__slate_slot_8);
}
folio_put(*__slate_slot_8);
return *__slate_slot_7;
}
}

fn first_se(sis: *mut swap_info_struct) -> *mut swap_extent {
let rb: *mut rb_node = rb_first((unsafe { core::ptr::addr_of_mut!((*sis).swap_extent_root) }) as *const rb_root);
rb_entry!(rb, swap_extent)
}

fn next_se(se: *mut swap_extent) -> *mut swap_extent {
let rb: *mut rb_node = unsafe { rb_next(core::ptr::addr_of_mut!((*se).rb_node) as *const rb_node) };
let __v0: *mut swap_extent = if !rb.is_null() {
rb_entry!(rb, swap_extent)
} else {
core::ptr::null_mut::<swap_extent>()
};
__v0
}

/// swapon tell device that all the old swap contents can be discarded,
/// to allow the swap device to optimize its wear-levelling.
fn discard_swap(si: *mut swap_info_struct) -> i32 {
let mut se: *mut swap_extent = unsafe { core::mem::zeroed() };
let mut start_block: u64 = 0;
let mut nr_blocks: u64 = 0;
let mut err: i32 = 0;
// Do not discard the swap header page!
se = first_se(si);
start_block = unsafe { (*se).start_block }.wrapping_add(1) << 3i32;
nr_blocks = unsafe { (*se).nr_pages }.wrapping_sub(1) << 3i32;
if nr_blocks != 0 {
err = unsafe { blkdev_issue_discard((*si).bdev, start_block, nr_blocks, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)) };
if err != 0 {
return err;
}
cond_resched__77!();
}
se = next_se(se);
while !se.is_null() {
start_block = (unsafe { (*se).start_block }) << 3i32;
nr_blocks = (unsafe { (*se).nr_pages }) << 3i32;
err = unsafe { blkdev_issue_discard((*si).bdev, start_block, nr_blocks, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)) };
if err != 0 {
break;
}
cond_resched__66!();
se = next_se(se);
}
return err; // That will often be -EOPNOTSUPP
}

fn offset_to_swap_extent(sis: *mut swap_info_struct, offset: u64) -> *mut swap_extent {
let mut rb: *mut rb_node = unsafe { core::mem::zeroed() };
rb = unsafe { (*sis).swap_extent_root.rb_node };
while !rb.is_null() {
let se: *mut swap_extent = rb_entry!(rb, swap_extent);
if offset < unsafe { (*se).start_page } {
rb = unsafe { (*rb).rb_left };
} else {
if offset >= unsafe { (*se).start_page }.wrapping_add(unsafe { (*se).nr_pages }) {
rb = unsafe { (*rb).rb_right };
} else {
return se;
}
}
}
// It *must* be present
loop {
unsafe {
core::arch::asm!("819: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 819b - ., 3; .popsection\n# {0}", const 819 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_offset_to_swap_extent_2, sym __slate_asm_string_offset_to_swap_extent_3, const 328 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
unsafe { core::mem::zeroed() }
}

#[unsafe(no_mangle)]
extern "C" fn swap_folio_sector(folio: *mut folio) -> u64 {
let sis: *mut swap_info_struct = __swap_entry_to_info(unsafe { (*folio).__slate_anon_0.__slate_anon_0.__slate_anon_4.swap });
let mut sector: u64 = 0;
let offset: u64 = swp_offset(unsafe { (*folio).__slate_anon_0.__slate_anon_0.__slate_anon_4.swap });
let se: *mut swap_extent = offset_to_swap_extent(sis, offset);
sector = unsafe { (*se).start_block }.wrapping_add(offset.wrapping_sub(unsafe { (*se).start_page }));
sector << 3i32
}

/// swap allocation tell device that a cluster of swap can now be discarded,
/// to allow the swap device to optimize its wear-levelling.
fn discard_swap_cluster(si: *mut swap_info_struct, mut start_page: u64, mut nr_pages: u64) {
let mut se: *mut swap_extent = offset_to_swap_extent(si, start_page);
while nr_pages != 0 {
let offset: u64 = start_page.wrapping_sub(unsafe { (*se).start_page });
let mut start_block: u64 = unsafe { (*se).start_block }.wrapping_add(offset);
let mut nr_blocks: u64 = unsafe { (*se).nr_pages }.wrapping_sub(offset);
if nr_blocks > nr_pages {
nr_blocks = nr_pages;
}
start_page = start_page.wrapping_add(nr_blocks);
nr_pages = nr_pages.wrapping_sub(nr_blocks);
start_block <<= 3i32;
nr_blocks <<= 3i32;
if (unsafe { blkdev_issue_discard((*si).bdev, start_block, nr_blocks, (BIT!(10i32) | BIT!(11i32)) as u32) }) != 0 {
break;
}
se = next_se(se);
}
}

fn cluster_is_empty(info: *mut swap_cluster_info) -> bool {
((unsafe { (*info).count }) as i32) == 0
}

fn cluster_is_discard(info: *mut swap_cluster_info) -> bool {
((unsafe { (*info).flags }) as i32) == 5
}

fn cluster_table_is_alloced(ci: *mut swap_cluster_info) -> bool {
!rcu_dereference_protected__4!(unsafe { (*ci).table }, unsafe { lockdep_is_held(core::ptr::addr_of_mut!((*ci).lock) as *const ()) }).is_null()
}

fn cluster_is_usable(ci: *mut swap_cluster_info, order: i32) -> bool {
if unlikely!(((unsafe { (*ci).flags }) as i32) > 3) != 0 {
return false;
}
if !cluster_table_is_alloced(ci) {
return false;
}
if order == 0 {
return true;
}
cluster_is_empty(ci) || order == ((unsafe { (*ci).order }) as i32)
}

fn cluster_index(si: *mut swap_info_struct, ci: *mut swap_cluster_info) -> u32 {
((unsafe { ci.offset_from((*si).cluster_info as *mut swap_cluster_info) }) as i64) as u32
}

fn cluster_offset(si: *mut swap_info_struct, ci: *mut swap_cluster_info) -> u32 {
cluster_index(si, ci).wrapping_mul(256)
}

#[unsafe(link_section = ".text.slate_distinct.swapfile.swap_cluster_free_table_folio_rcu_cb")]
extern "C" fn swap_cluster_free_table_folio_rcu_cb(head: *mut callback_head) {
let mut __v0: *mut page;
__v0 = ((unsafe { (head as *const u8).wrapping_offset(-(8 as isize)) }) as *mut ()) as *mut page;
let folio: *mut folio = _compound_head(__v0 as *const page) as *mut folio;
folio_put(folio);
}

fn swap_cluster_free_table(ci: *mut swap_cluster_info) {
let mut table: *mut swap_table = unsafe { core::mem::zeroed() };
let mut __v0: *mut __SlateRecord_types_h_197;
let mut __v1: *mut __SlateRecord_types_h_197;
let mut __v2: *mut __SlateRecord_types_h_197;
loop {
if !(__native_word__15!() || ((const { core::mem::size_of::<*mut __SlateRecord_types_h_197>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_821() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*ci).table) as *const *mut __SlateRecord_types_h_197) };
let __UNIQUE_ID_rcu_820: *mut __SlateRecord_types_h_197 = __v2;
__v1 = __UNIQUE_ID_rcu_820;
__v0 = __v1;
table = __v0 as *mut swap_table;
if table.is_null() {
return;
}
let _r_a_p__v: u64 = 0;
if true && _r_a_p__v == 0 {
loop {
loop {
if !(__native_word__15!() || ((const { core::mem::size_of::<*mut __SlateRecord_types_h_197>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_822() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__18!((*ci).table, _r_a_p__v as *mut __SlateRecord_types_h_197);
if true {
break;
}
}
} else {
loop {
kcsan_release!();
loop {
loop {
if !__native_word__15!() {
unsafe { __compiletime_assert_823() };
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
if !(__native_word__15!() || ((const { core::mem::size_of::<*mut __SlateRecord_types_h_197>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_824() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__18!(*unsafe { core::ptr::addr_of_mut!((*ci).table) }, RCU_INITIALIZER!(_r_a_p__v as *mut __SlateRecord_types_h_197));
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
if ((const { core::mem::size_of::<swap_table>() }) as u64) != _AC!() << 12i32 {
unsafe { kmem_cache_free(swap_table_cachep, table as *mut ()) };
return;
}
unsafe { call_rcu(core::ptr::addr_of_mut!((*folio_page!(virt_to_folio(table as *const ()), 0i32)).__slate_anon_1.callback_head), Some(swap_cluster_free_table_folio_rcu_cb)) };
}

fn swap_cluster_alloc_table(ci: *mut swap_cluster_info, gfp: u32) -> i32 {
let mut table: *mut swap_table = core::ptr::null_mut::<swap_table>();
// The cluster must be empty and not on any list during allocation.
VM_WARN_ON_ONCE!();
let mut __v0: *mut __SlateRecord_types_h_197;
let mut __v1: *mut __SlateRecord_types_h_197;
let mut __v2: *mut __SlateRecord_types_h_197;
loop {
if !(__native_word__15!() || ((const { core::mem::size_of::<*mut __SlateRecord_types_h_197>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_826() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*ci).table) as *const *mut __SlateRecord_types_h_197) };
let __UNIQUE_ID_rcu_825: *mut __SlateRecord_types_h_197 = __v2;
__v1 = __UNIQUE_ID_rcu_825;
__v0 = __v1;
if !__v0.is_null() {
return 0;
}
if ((const { core::mem::size_of::<swap_table>() }) as u64) == _AC!() << 12i32 {
let folio: *mut folio = alloc_hooks__2!(unsafe { folio_alloc_noprof(gfp | 256u32, 0u32) });
if !folio.is_null() {
table = folio_address(folio as *const folio) as *mut swap_table;
}
} else {
table = kmem_cache_zalloc!(unsafe { swap_table_cachep }, gfp) as *mut swap_table;
}
if table.is_null() {
return 12i32.wrapping_neg();
}
let _r_a_p__v: u64 = table as u64;
if false && _r_a_p__v == 0 {
loop {
loop {
if !(__native_word__15!() || ((const { core::mem::size_of::<*mut __SlateRecord_types_h_197>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_827() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__18!((*ci).table, _r_a_p__v as *mut __SlateRecord_types_h_197);
if true {
break;
}
}
} else {
loop {
kcsan_release!();
loop {
loop {
if !__native_word__15!() {
unsafe { __compiletime_assert_828() };
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
if !(__native_word__15!() || ((const { core::mem::size_of::<*mut __SlateRecord_types_h_197>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_829() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__18!(*unsafe { core::ptr::addr_of_mut!((*ci).table) }, RCU_INITIALIZER!(_r_a_p__v as *mut __SlateRecord_types_h_197));
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
0
}

/// Sanity check to ensure nothing leaked, and the specified range is empty.
/// One special case is that bad slots can't be freed, so check the number of
/// bad slots for swapoff, and non-swapoff path must never free bad slots.
fn swap_cluster_assert_empty(ci: *mut swap_cluster_info, mut ci_off: u32, nr: u32, swapoff: bool) {
let ci_end: u32 = ci_off.wrapping_add(nr);
let mut swp_tb: u64 = 0;
let mut bad_slots: i32 = 0;
if IS_ENABLED!() == 0 && !swapoff {
return;
}
loop {
swp_tb = __swap_table_get(ci, ci_off);
if swp_tb_is_bad(swp_tb) {
bad_slots = bad_slots.wrapping_add(1);
} else {
let mut __v2: i64;
let __ret_warn_on: i32 = !(!(!swp_tb_is_null(swp_tb))) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("830: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 830b - ., 3; .popsection\n# {0}", const 830 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_swap_cluster_assert_empty_3, sym __slate_asm_string_swap_cluster_assert_empty_4, const 514 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("831: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 831b - ., 4; .popsection\n# {0}", const 831 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v2 = (__ret_warn_on != 0) as i64;
}
let mut __v7: i64;
let __ret_warn_on: i32 = (__swap_cgroup_get(ci, ci_off) != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("832: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 832b - ., 3; .popsection\n# {0}", const 832 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_swap_cluster_assert_empty_8, sym __slate_asm_string_swap_cluster_assert_empty_9, const 515 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("833: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 833b - ., 4; .popsection\n# {0}", const 833 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v7 = (__ret_warn_on != 0) as i64;
let __v14: u32 = ci_off.wrapping_add(1);
ci_off = __v14;
if __v14 >= ci_end {
break;
}
}
let mut __v15: i64;
{
let __ret_warn_on: i32 = (bad_slots != if swapoff { (unsafe { (*ci).count }) as i32 } else { 0 }) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("834: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 834b - ., 3; .popsection\n# {0}", const 834 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_swap_cluster_assert_empty_16, sym __slate_asm_string_swap_cluster_assert_empty_17, const 518 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("835: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 835b - ., 4; .popsection\n# {0}", const 835 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v15 = (__ret_warn_on != 0) as i64;
}
let mut __v20: i64;
let __ret_warn_on: i32 = (nr == 256 && !unsafe { (*ci).extend_table }.is_null()) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("836: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 836b - ., 3; .popsection\n# {0}", const 836 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_swap_cluster_assert_empty_21, sym __slate_asm_string_swap_cluster_assert_empty_22, const 519 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("837: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 837b - ., 4; .popsection\n# {0}", const 837 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v20 = (__ret_warn_on != 0) as i64;
}

/// Allocate swap table for one cluster. Attempt an atomic allocation first,
/// then fallback to sleeping allocation.
fn swap_cluster_populate(si: *mut swap_info_struct, ci: *mut swap_cluster_info) -> *mut swap_cluster_info {
let mut ret: i32 = 0;
// Only cluster isolation from the allocator does table allocation.
// Swap allocator uses percpu clusters and holds the local lock.
loop {
let mut __v0: *mut percpu_swap_cluster;
__verify_pcpu_ptr__35!();
let mut __v2: *mut percpu_swap_cluster;
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
let __v5: u64 = tcp_ptr__.wrapping_add((unsafe { core::ptr::addr_of_mut!(percpu_swap_cluster) }) as u64);
tcp_ptr__ = __v5;
__v2 = tcp_ptr__ as *mut percpu_swap_cluster;
__v0 = __v2;
if true {
break;
}
}
if (unsafe { (*si).flags }) & 16 == 0 {
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!((*si).global_cluster_lock) });
}
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!((*ci).lock) });
if swap_cluster_alloc_table(ci, (BIT!(5i32) as u32) | (BIT!(19i32) as u32) | (BIT!(13i32) as u32)) == 0 {
return ci;
}
// Try a sleep allocation. Each isolated free cluster may cause
// a sleep allocation, but there is a limited number of them, so
// the potential recursive allocation is limited.
spin_unlock(unsafe { core::ptr::addr_of_mut!((*ci).lock) });
if (unsafe { (*si).flags }) & 16 == 0 {
spin_unlock(unsafe { core::ptr::addr_of_mut!((*si).global_cluster_lock) });
}
loop {
__release!();
loop {
let mut __v10: *mut local_lock;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<local_lock>() as *const ();
if true {
break;
}
}
let mut __v12: *mut local_lock;
let mut tcp_ptr__: u64 = 0;
let mut __v13: u64;
let mut pfo_val__: u64 = 0;
unsafe {
let __t1: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t1, sym this_cpu_off, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t1 as u64) };
}
__v13 = pfo_val__;
tcp_ptr__ = __v13;
let __v15: u64 = tcp_ptr__.wrapping_add((unsafe { core::ptr::addr_of_mut!(percpu_swap_cluster.lock) }) as u64);
tcp_ptr__ = __v15;
__v12 = tcp_ptr__ as *mut local_lock;
__v10 = __v12;
let __l: *mut local_lock = __v10;
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
ret = swap_cluster_alloc_table(ci, (BIT!(5i32) as u32) | (BIT!(19i32) as u32) | (((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)));
// Back to atomic context. We might have migrated to a new CPU with a
// usable percpu cluster. But just keep using the isolated cluster to
// make things easier. Migration indicates a slight change of workload
// so using a new free cluster might not be a bad idea, and the worst
// could happen with ignoring the percpu cluster is fragmentation,
// which is acceptable since this fallback and race is rare.
loop {
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
let mut __v21: *mut local_lock;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<local_lock>() as *const ();
if true {
break;
}
}
let mut __v23: *mut local_lock;
let mut tcp_ptr__: u64 = 0;
let mut __v24: u64;
let mut pfo_val__: u64 = 0;
unsafe {
let __t2: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t2, sym this_cpu_off, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t2 as u64) };
}
__v24 = pfo_val__;
tcp_ptr__ = __v24;
let __v26: u64 = tcp_ptr__.wrapping_add((unsafe { core::ptr::addr_of_mut!(percpu_swap_cluster.lock) }) as u64);
tcp_ptr__ = __v26;
__v23 = tcp_ptr__ as *mut local_lock;
__v21 = __v23;
let __l: *mut local_lock = __v21;
local_lock_acquire(__l);
if true {
break;
}
}
__acquire!();
if true {
break;
}
}
if (unsafe { (*si).flags }) & 16 == 0 {
spin_lock(unsafe { core::ptr::addr_of_mut!((*si).global_cluster_lock) });
}
spin_lock(unsafe { core::ptr::addr_of_mut!((*ci).lock) });
if ret != 0 {
move_cluster(si, ci, unsafe { core::ptr::addr_of_mut!((*si).free_clusters) }, 1u32);
spin_unlock(unsafe { core::ptr::addr_of_mut!((*ci).lock) });
return core::ptr::null_mut::<swap_cluster_info>();
}
ci
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_839: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

/// Add a cluster to discard list and schedule it to do discard
fn swap_cluster_schedule_discard(si: *mut swap_info_struct, ci: *mut swap_cluster_info) {
VM_BUG_ON!();
move_cluster(si, ci, unsafe { core::ptr::addr_of_mut!((*si).discard_clusters) }, 5u32);
schedule_work(unsafe { core::ptr::addr_of_mut!((*si).discard_work) });
}

fn __free_cluster(si: *mut swap_info_struct, ci: *mut swap_cluster_info) {
swap_cluster_assert_empty(ci, 0u32, 256u32, false);
swap_cluster_free_table(ci);
move_cluster(si, ci, unsafe { core::ptr::addr_of_mut!((*si).free_clusters) }, 1u32);
unsafe {
(*ci).order = 0;
}
}

/// Isolate and lock the first cluster that is not contented on a list,
/// clean its flag before taken off-list. Cluster flag must be in sync
/// with list status, so cluster updaters can always know the cluster
/// list status without touching si lock.
///
/// Note it's possible that all clusters on a list are contented so
/// this returns NULL for an non-empty list.
fn isolate_lock_cluster(si: *mut swap_info_struct, list: *mut list_head) -> *mut swap_cluster_info {
let mut ci: *mut swap_cluster_info = unsafe { core::mem::zeroed() };
let mut found: *mut swap_cluster_info = core::ptr::null_mut::<swap_cluster_info>();
let mut flags: u8 = 0;
spin_lock(unsafe { core::ptr::addr_of_mut!((*si).lock) });
ci = list_first_entry__19!(list);
while list_entry_is_head!(ci, list, list) == 0 {
if spin_trylock(unsafe { core::ptr::addr_of_mut!((*ci).lock) }) != 0 {
// We may only isolate and clear flags of following lists
VM_BUG_ON!();
VM_BUG_ON!();
list_del(unsafe { core::ptr::addr_of_mut!((*ci).list) });
flags = unsafe { (*ci).flags };
unsafe {
(*ci).flags = 0;
}
found = ci;
break;
}
ci = list_next_entry__20!(ci, list);
}
spin_unlock(unsafe { core::ptr::addr_of_mut!((*si).lock) });
// Cluster's table is freed when and only when it's on the free list.
if !found.is_null() && (flags as i32) == 1 {
VM_WARN_ON_ONCE!();
VM_WARN_ON_ONCE!();
return swap_cluster_populate(si, found);
}
found
}

/// Doing discard actually. After a cluster discard is finished, the cluster
/// will be added to free cluster list. Discard cluster is a bit special as
/// they don't participate in allocation or reclaim, so clusters marked as
/// CLUSTER_FLAG_DISCARD must remain off-list or on discard list.
fn swap_do_scheduled_discard(si: *mut swap_info_struct) -> bool {
let mut ci: *mut swap_cluster_info = unsafe { core::mem::zeroed() };
let mut ret: bool = false;
let mut idx: u32 = 0;
spin_lock(unsafe { core::ptr::addr_of_mut!((*si).lock) });
while list_empty((unsafe { core::ptr::addr_of_mut!((*si).discard_clusters) }) as *const list_head) == 0 {
ci = list_first_entry__14!(unsafe { core::ptr::addr_of_mut!((*si).discard_clusters) }, swap_cluster_info);
// Delete the cluster from list to prepare for discard, but keep
// the CLUSTER_FLAG_DISCARD flag, percpu_swap_cluster could be
// pointing to it, or ran into by relocate_cluster.
list_del(unsafe { core::ptr::addr_of_mut!((*ci).list) });
idx = cluster_index(si, ci);
spin_unlock(unsafe { core::ptr::addr_of_mut!((*si).lock) });
discard_swap_cluster(si, idx.wrapping_mul(256) as u64, 256u64);
spin_lock(unsafe { core::ptr::addr_of_mut!((*ci).lock) });
// Discard is done, clear its flags as it's off-list, then
// return the cluster to allocation list.
unsafe {
(*ci).flags = 0;
}
__free_cluster(si, ci);
spin_unlock(unsafe { core::ptr::addr_of_mut!((*ci).lock) });
ret = true;
spin_lock(unsafe { core::ptr::addr_of_mut!((*si).lock) });
}
spin_unlock(unsafe { core::ptr::addr_of_mut!((*si).lock) });
ret
}

#[unsafe(link_section = ".text.slate_distinct.swapfile.swap_discard_work")]
extern "C" fn swap_discard_work(work: *mut work_struct) {
let si: *mut swap_info_struct = container_of__53!(work, swap_info_struct);
swap_do_scheduled_discard(si);
}

#[unsafe(link_section = ".text.slate_distinct.swapfile.swap_users_ref_free")]
extern "C" fn swap_users_ref_free(r#ref: *mut percpu_ref) {
let mut si: *mut swap_info_struct = unsafe { core::mem::zeroed() };
si = container_of__42!(r#ref, swap_info_struct);
unsafe { complete(core::ptr::addr_of_mut!((*si).comp)) };
}

/// Must be called after freeing if ci->count == 0, moves the cluster to free
/// or discard list.
fn free_cluster(si: *mut swap_info_struct, ci: *mut swap_cluster_info) {
VM_BUG_ON!();
VM_BUG_ON!();
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!((*ci).lock) });
// If the swap is discardable, prepare discard the cluster
// instead of free it immediately. The cluster will be freed
// after discard.
if (unsafe { (*si).flags }) & 1026u64 == 1026u64 {
swap_cluster_schedule_discard(si, ci);
return;
}
__free_cluster(si, ci);
}

/// Must be called after freeing if ci->count != 0, moves the cluster to
/// nonfull list.
fn partial_free_cluster(si: *mut swap_info_struct, ci: *mut swap_cluster_info) {
VM_BUG_ON!();
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!((*ci).lock) });
if ((unsafe { (*ci).flags }) as i32) != 2 {
move_cluster(si, ci, unsafe { (*si).nonfull_clusters.as_mut_ptr() as *mut list_head }.wrapping_offset(((unsafe { (*ci).order }) as i32) as isize), 2u32);
}
}

/// Must be called after allocation, moves the cluster to full or frag list.
/// Note: allocation doesn't acquire si lock, and may drop the ci lock for
/// reclaim, so the cluster could be any where when called.
fn relocate_cluster(si: *mut swap_info_struct, ci: *mut swap_cluster_info) {
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!((*ci).lock) });
// Discard cluster must remain off-list or on discard list
if cluster_is_discard(ci) {
return;
}
if (unsafe { (*ci).count }) == 0 {
if ((unsafe { (*ci).flags }) as i32) != 1 {
free_cluster(si, ci);
}
} else {
if ((unsafe { (*ci).count }) as i32) != 256 {
if ((unsafe { (*ci).flags }) as i32) != 3 {
move_cluster(si, ci, unsafe { (*si).frag_clusters.as_mut_ptr() as *mut list_head }.wrapping_offset(((unsafe { (*ci).order }) as i32) as isize), 3u32);
}
} else {
if ((unsafe { (*ci).flags }) as i32) != 4 {
move_cluster(si, ci, unsafe { core::ptr::addr_of_mut!((*si).full_clusters) }, 4u32);
}
}
}
}

/// The cluster corresponding to @offset will be accounted as having one bad
/// slot. The cluster will not be added to the free cluster list, and its
/// usage counter will be increased by 1. Only used for initialization.
fn swap_cluster_setup_bad_slot(si: *mut swap_info_struct, cluster_info: *mut swap_cluster_info, offset: u32, mask: bool) -> i32 {
let ci_off: u32 = offset % 256;
let idx: u64 = (offset / 256) as u64;
let mut ci: *mut swap_cluster_info = unsafe { core::mem::zeroed() };
let mut ret: i32 = 0;
// si->max may got shrunk by swap swap_activate()
if offset >= unsafe { (*si).max } && !mask {
let mut __v0: i32;
if false {
unsafe { _printk(c"\x017Ignoring bad slot %u (max: %u)\n".as_ptr() as *const u8, offset, (*si).max) };
}
__v0 = 0;
return 0;
}
// Account it, skip header slot: si->pages is initiated as
// si->max - 1. Also skip the masking of last cluster,
// si->pages doesn't include that part.
if offset != 0 && !mask {
let __v2: u32 = unsafe { (*si).pages };
unsafe {
(*si).pages = __v2.wrapping_sub(1);
}
}
if (unsafe { (*si).pages }) == 0 {
let mut __v4: i32;
while false {
}
__v4 = unsafe { _printk(c"\x014Empty swap-file\n".as_ptr() as *const u8) };
return 22i32.wrapping_neg();
}
ci = cluster_info.wrapping_offset(idx as isize);
// Need to allocate swap table first for initial bad slot marking.
let __v6: bool = (unsafe { (*ci).count }) == 0 && swap_cluster_alloc_table(ci, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)) != 0;
if __v6 {
return 12i32.wrapping_neg();
}
spin_lock(unsafe { core::ptr::addr_of_mut!((*ci).lock) });
// Check for duplicated bad swap slots.
if __swap_table_xchg(ci, ci_off, 18446744073709551608u64) != 0 {
let mut __v7: i32;
while false {
}
__v7 = unsafe { _printk(c"\x014Duplicated bad slot offset %d\n".as_ptr() as *const u8, offset) };
ret = 22i32.wrapping_neg();
} else {
unsafe {
(*ci).count = ((*ci).count as i32).wrapping_add(1) as u16;
}
}
spin_unlock(unsafe { core::ptr::addr_of_mut!((*ci).lock) });
let mut __v12: i64;
{
let __ret_warn_on: i32 = (((unsafe { (*ci).count }) as i32) > 256) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("842: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 842b - ., 3; .popsection\n# {0}", const 842 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_swap_cluster_setup_bad_slot_13, sym __slate_asm_string_swap_cluster_setup_bad_slot_14, const 827 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
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
__v12 = (__ret_warn_on != 0) as i64;
}
let mut __v17: i64;
let __ret_warn_on: i32 = ((unsafe { (*ci).flags }) != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("844: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 844b - ., 3; .popsection\n# {0}", const 844 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_swap_cluster_setup_bad_slot_18, sym __slate_asm_string_swap_cluster_setup_bad_slot_19, const 828 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("845: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 845b - ., 4; .popsection\n# {0}", const 845 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v17 = (__ret_warn_on != 0) as i64;
ret
}

/// Reclaim drops the ci lock, so the cluster may become unusable (freed or
/// stolen by a lower order). @usable will be set to false if that happens.
fn cluster_reclaim_range(si: *mut swap_info_struct, ci: *mut swap_cluster_info, start: u64, order: u32, usable: &mut bool) -> bool {
let nr_pages: u32 = (1i32 << order) as u32;
let mut offset: u64 = start;
let end: u64 = start.wrapping_add(nr_pages as u64);
let mut swp_tb: u64 = 0;
spin_unlock(unsafe { core::ptr::addr_of_mut!((*ci).lock) });
loop {
swp_tb = swap_table_get(ci, (offset % 256) as u32);
if swp_tb_get_count(swp_tb) != 0 {
break;
}
if swp_tb_is_folio(swp_tb) {
if __try_to_reclaim_swap(si, offset, 1u64) < 0 {
break;
}
}
let __v2: u64 = offset.wrapping_add(1);
offset = __v2;
if __v2 >= end {
break;
}
}
spin_lock(unsafe { core::ptr::addr_of_mut!((*ci).lock) });
// We just dropped ci->lock so cluster could be used by another
// order or got freed, check if it's still usable or empty.
if !cluster_is_usable(ci, order as i32) {
unsafe {
*(usable as *mut bool) = false;
}
return false;
}
unsafe {
*(usable as *mut bool) = true;
}
// Fast path, no need to scan if the whole cluster is empty
if cluster_is_empty(ci) {
return true;
}
// Recheck the range no matter reclaim succeeded or not, the slot
// could have been be freed while we are not holding the lock.
offset = start;
while offset < end {
swp_tb = __swap_table_get(ci, (offset % 256) as u32);
if !swp_tb_is_null(swp_tb) {
return false;
}
offset = offset.wrapping_add(1);
}
true
}

fn cluster_scan_range(si: *mut swap_info_struct, ci: *mut swap_cluster_info, offset: u64, nr_pages: u32, need_reclaim: &mut bool) -> bool {
let mut ci_off: u32 = (offset % 256) as u32;
let ci_end: u32 = ci_off.wrapping_add(nr_pages);
let mut swp_tb: u64 = 0;
loop {
swp_tb = __swap_table_get(ci, ci_off);
if !swp_tb_is_null(swp_tb) {
let __v1: bool = swp_tb_is_folio(swp_tb) && __swp_tb_get_count(swp_tb) == 0;
if __v1 {
if !vm_swap_full() {
return false;
}
unsafe {
*(need_reclaim as *mut bool) = true;
}
} else {
// Slot with zero count can only be NULL or folio
VM_WARN_ON!();
return false;
}
}
let __v3: u32 = ci_off.wrapping_add(1);
ci_off = __v3;
if __v3 >= ci_end {
break;
}
}
true
}

fn __swap_cluster_alloc_entries(si: *mut swap_info_struct, ci: *mut swap_cluster_info, folio: *mut folio, ci_off: u32) -> bool {
let mut order: u32 = 0;
let mut nr_pages: u64 = 0;
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!((*ci).lock) });
if (unsafe { (*si).flags }) & 2 == 0 {
return false;
}
// All mm swap allocation starts with a folio (folio_alloc_swap),
// it's also the only allocation path for large orders allocation.
// Such swap slots starts with count == 0 and will be increased
// upon folio unmap.
//
// Else, it's a exclusive order 0 allocation for hibernation.
// The slot starts with count == 1 and never increases.
if likely__7!(folio) != 0 {
order = folio_order(folio as *const folio);
nr_pages = (1i32 << order) as u64;
swap_cluster_assert_empty(ci, ci_off, nr_pages as u32, false);
unsafe { __swap_cache_add_folio(ci, folio, swp_entry((*si).r#type as u64, ci_off.wrapping_add(cluster_offset(si, ci)) as u64)) };
} else {
if IS_ENABLED__2!() != 0 {
order = 0;
nr_pages = 1;
swap_cluster_assert_empty(ci, ci_off, 1u32, false);
// Fake shadow placeholder with no flag, hibernation does not use the zeromap
__swap_table_set(ci, ci_off, __swp_tb_mk_count(shadow_to_swp_tb(core::ptr::null_mut::<()>(), 0u8), 1i32));
} else {
// Allocation without folio is only possible with hibernation
let mut __v1: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("846: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 846b - ., 3; .popsection\n# {0}", const 846 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___swap_cluster_alloc_entries_2, sym __slate_asm_string___swap_cluster_alloc_entries_3, const 947 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("847: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 847b - ., 4; .popsection\n# {0}", const 847 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
return false;
}
}
// The first allocation in a cluster makes the
// cluster exclusive to this order
if cluster_is_empty(ci) {
unsafe {
(*ci).order = order as u8;
}
}
unsafe {
(*ci).count = (((*ci).count as i32) as u64).wrapping_add(nr_pages) as u16;
}
swap_range_alloc(si, nr_pages as u32);
true
}

/// Try use a new cluster for current CPU and allocate from it.
fn alloc_swap_scan_cluster(si: *mut swap_info_struct, ci: *mut swap_cluster_info, folio: *mut folio, mut offset: u64) -> u32 {
let mut pto_val__: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut u64 = core::ptr::addr_of_mut!(pto_val__) as *mut u64;
let mut pto_val__: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut u64 = core::ptr::addr_of_mut!(pto_val__) as *mut u64;
let mut usable: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut bool = core::ptr::addr_of_mut!(usable) as *mut bool;
let mut ret: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut bool = core::ptr::addr_of_mut!(ret) as *mut bool;
let mut need_reclaim: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut bool = core::ptr::addr_of_mut!(need_reclaim) as *mut bool;
let mut nr_pages: u32 = unsafe { core::mem::zeroed() };
let mut end: u64 = unsafe { core::mem::zeroed() };
let mut __v27: u32 = unsafe { core::mem::zeroed() };
let mut order: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_28: *mut u32 = core::ptr::addr_of_mut!(order) as *mut u32;
let mut start: u64 = unsafe { core::mem::zeroed() };
let mut found: u32 = unsafe { core::mem::zeroed() };
let mut next: u32 = unsafe { core::mem::zeroed() };
unsafe {
next = 0;
found = 0;
start = ALIGN_DOWN__3!(offset, 256i32);
if likely__7!(folio) != 0 {
__v27 = folio_order(folio as *const folio);
} else {
__v27 = 0;
}
'__join_8: {
*__slate_slot_28 = __v27;
end = start.wrapping_add(256);
nr_pages = (1i32 << *__slate_slot_28) as u32;
let __slate_auto_bytes = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_1 = __slate_slot_23 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 1usize);
let __slate_auto_bytes = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
lockdep_assert_held!(core::ptr::addr_of_mut!((*ci).lock));
VM_WARN_ON!();
if !(end < (nr_pages as u64) || (((*ci).count as i32) as u32).wrapping_add(nr_pages) > 256) {
end = end.wrapping_sub(nr_pages as u64);
loop {
if offset <= end {
*__slate_slot_24 = false;
if cluster_scan_range(si, ci, offset, nr_pages, &mut (*core::ptr::addr_of_mut!(*__slate_slot_24))) {
if *__slate_slot_24 {
*__slate_slot_23 = cluster_reclaim_range(si, ci, offset, *__slate_slot_28, &mut (*core::ptr::addr_of_mut!(*__slate_slot_22)));
if !(*__slate_slot_22) {
break '__join_8;
} else {
if cluster_is_empty(ci) {
offset = start;
}
// Reclaim failed but cluster is usable, try next
if *__slate_slot_23 {
break;
}
}
} else {
break;
}
}
offset = offset.wrapping_add(nr_pages as u64);
} else {
break '__join_8;
}
}
if __swap_cluster_alloc_entries(si, ci, folio, (offset % 256) as u32) {
found = offset as u32;
offset = offset.wrapping_add(nr_pages as u64);
if ((*ci).count as i32) < 256 && offset <= end {
next = offset as u32;
}
}
}
}
relocate_cluster(si, ci);
swap_cluster_unlock(ci);
if (*si).flags & 16 != 0 {
loop {
__verify_pcpu_ptr__6!();
loop {
core::ptr::write(__slate_slot_9, next as u64);
core::arch::asm!("movq {1:r}, %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(percpu_swap_cluster.offset) as *mut u64).wrapping_offset(*__slate_slot_28 as isize) as u64) as *mut u64, in(reg) *__slate_slot_9 as u64, options(nostack, att_syntax));
if true {
break;
}
}
if true {
break;
}
}
loop {
__verify_pcpu_ptr__19!();
loop {
core::ptr::write(__slate_slot_1, si as u64);
core::arch::asm!("movq {1:r}, %gs:({0:r})", in(reg) ((core::ptr::addr_of_mut!(percpu_swap_cluster.si) as *mut *mut swap_info_struct).wrapping_offset(*__slate_slot_28 as isize) as u64) as *mut *mut swap_info_struct, in(reg) *__slate_slot_1 as u64, options(nostack, att_syntax));
if true {
break;
}
}
if true {
break;
}
}
} else {
*((*(*si).global_cluster).next.as_mut_ptr() as *mut u32).wrapping_offset(*__slate_slot_28 as isize) = next;
}
return found;
}
}

fn alloc_swap_scan_list(si: *mut swap_info_struct, list: *mut list_head, folio: *mut folio, scan_all: bool) -> u32 {
let mut found: u32 = 0;
loop {
let ci: *mut swap_cluster_info = isolate_lock_cluster(si, list);
let mut offset: u64 = 0;
if ci.is_null() {
break;
}
offset = cluster_offset(si, ci) as u64;
found = alloc_swap_scan_cluster(si, ci, folio, offset);
if found != 0 {
break;
}
if !scan_all {
break;
}
}
found
}

fn swap_reclaim_full_clusters(si: *mut swap_info_struct, force: bool) {
let mut to_scan: i64 = 1;
let mut offset: u64 = 0;
let mut end: u64 = 0;
let mut ci: *mut swap_cluster_info = unsafe { core::mem::zeroed() };
let mut swp_tb: u64 = 0;
let mut nr_reclaim: i32 = 0;
if force {
to_scan = swap_usage_in_pages(si) / 256;
}
loop {
let __v15: *mut swap_cluster_info = isolate_lock_cluster(si, unsafe { core::ptr::addr_of_mut!((*si).full_clusters) });
ci = __v15;
if __v15.is_null() {
break;
}
offset = cluster_offset(si, ci) as u64;
let mut __v0: u64;
let __UNIQUE_ID_x__848: u32 = unsafe { (*si).max };
let __UNIQUE_ID_y__849: u64 = offset.wrapping_add(256);
loop {
if __types_ok__8!(__UNIQUE_ID_x__848, __UNIQUE_ID_y__849) == 0 {
unsafe { __compiletime_assert_850() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if (__UNIQUE_ID_x__848 as u64) < __UNIQUE_ID_y__849 { __UNIQUE_ID_x__848 as u64 } else { __UNIQUE_ID_y__849 };
end = __v0;
to_scan = to_scan.wrapping_sub(1);
while offset < end {
'__slate_continue_10: {
swp_tb = swap_table_get(ci, (offset % 256) as u32);
let __v4: bool = swp_tb_is_folio(swp_tb) && __swp_tb_get_count(swp_tb) == 0;
if __v4 {
spin_unlock(unsafe { core::ptr::addr_of_mut!((*ci).lock) });
nr_reclaim = __try_to_reclaim_swap(si, offset, 1u64);
spin_lock(unsafe { core::ptr::addr_of_mut!((*ci).lock) });
if nr_reclaim != 0 {
let __v5: i32 = abs__3!(nr_reclaim);
offset = offset.wrapping_add(__v5 as u64);
break '__slate_continue_10;
}
}
offset = offset.wrapping_add(1);
}
}
// in case no swap cache is reclaimed
if ((unsafe { (*ci).flags }) as i32) == 0 {
relocate_cluster(si, ci);
}
swap_cluster_unlock(ci);
if to_scan <= 0 {
break;
}
// When 'force' is false, 'to_scan' is initialized to 1.
// The loop breaks above, making this cond_resched() unreachable
// in atomic contexts.
cond_resched__75!();
}
}

#[unsafe(link_section = ".text.slate_distinct.swapfile.swap_reclaim_work")]
extern "C" fn swap_reclaim_work(work: *mut work_struct) {
let si: *mut swap_info_struct = container_of__26!(work, swap_info_struct);
swap_reclaim_full_clusters(si, true);
}

/// Try to allocate swap entries with specified order and try set a new
/// cluster for current CPU too.
fn cluster_alloc_swap_entry(si: *mut swap_info_struct, folio: *mut folio) -> u64 {
// Order 0 stealing from higher order
let mut o: i32 = unsafe { core::mem::zeroed() };
let mut found: u32 = unsafe { core::mem::zeroed() };
let mut offset: u32 = unsafe { core::mem::zeroed() };
let mut __v5: u32 = unsafe { core::mem::zeroed() };
let mut order: u32 = unsafe { core::mem::zeroed() };
let mut ci: core::mem::MaybeUninit<*mut swap_cluster_info> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut swap_cluster_info = core::ptr::addr_of_mut!(ci) as *mut *mut swap_cluster_info;
unsafe {
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
if likely__7!(folio) != 0 {
__v5 = folio_order(folio as *const folio);
} else {
__v5 = 0;
}
order = __v5;
offset = 0;
found = 0;
// Swapfile is not block device so unable
// to allocate large entries.
if order != 0 && (*si).flags & 64 == 0 {
return 0;
} else {
'__join_2: {
if (*si).flags & 16 == 0 {
// Serialize HDD SWAP allocation for each device.
spin_lock(core::ptr::addr_of_mut!((*si).global_cluster_lock));
offset = *((*(*si).global_cluster).next.as_mut_ptr() as *mut u32).wrapping_offset(order as isize);
if offset != 0 {
*__slate_slot_7 = swap_cluster_lock(si, offset as u64);
// Cluster could have been used by another order
if cluster_is_usable(*__slate_slot_7, order as i32) {
if cluster_is_empty(*__slate_slot_7) {
offset = cluster_offset(si, *__slate_slot_7);
}
found = alloc_swap_scan_cluster(si, *__slate_slot_7, folio, offset as u64);
} else {
swap_cluster_unlock(*__slate_slot_7);
}
if found != 0 {
break '__join_2;
}
}
}
if (*si).flags & 1024 != 0 {
found = alloc_swap_scan_list(si, core::ptr::addr_of_mut!((*si).free_clusters), folio, false);
if found != 0 {
break '__join_2;
}
}
// If the device need discard, prefer new cluster over nonfull
// to spread out the writes.
if order < 9u32 {
found = alloc_swap_scan_list(si, ((*si).nonfull_clusters.as_mut_ptr() as *mut list_head).wrapping_offset(order as isize), folio, true);
if found != 0 {
break '__join_2;
}
}
if (*si).flags & 1024 == 0 {
found = alloc_swap_scan_list(si, core::ptr::addr_of_mut!((*si).free_clusters), folio, false);
if found != 0 {
break '__join_2;
}
}
// Try reclaim full clusters if free and nonfull lists are drained
if vm_swap_full() {
swap_reclaim_full_clusters(si, false);
}
if order < 9u32 {
// Scan only one fragment cluster is good enough. Order 0
// allocation will surely success, and large allocation
// failure is not critical. Scanning one cluster still
// keeps the list rotated and reclaimed (for clean swap cache).
found = alloc_swap_scan_list(si, ((*si).frag_clusters.as_mut_ptr() as *mut list_head).wrapping_offset(order as isize), folio, false);
if found != 0 {
break '__join_2;
}
}
if order == 0 {
o = 1;
while o < 1 {
// Clusters here have at least one usable slots and can't fail order 0
// allocation, but reclaim may drop si->lock and race with another user.
found = alloc_swap_scan_list(si, ((*si).frag_clusters.as_mut_ptr() as *mut list_head).wrapping_offset(o as isize), folio, true);
if found != 0 {
break;
} else {
found = alloc_swap_scan_list(si, ((*si).nonfull_clusters.as_mut_ptr() as *mut list_head).wrapping_offset(o as isize), folio, true);
if found != 0 {
break;
} else {
o = o.wrapping_add(1);
}
}
}
}
}
if (*si).flags & 16 == 0 {
spin_unlock(core::ptr::addr_of_mut!((*si).global_cluster_lock));
}
return found as u64;
}
}
}

/// SWAP_USAGE_OFFLIST_BIT can only be set by this helper.
fn del_from_avail_list(si: *mut swap_info_struct, swapoff: bool) {
let mut pages: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut u64 = core::ptr::addr_of_mut!(pages) as *mut u64;
unsafe {
'__join_0: {
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
spin_lock(core::ptr::addr_of_mut!(swap_avail_lock));
if swapoff {
// Forcefully remove it. Clear the SWP_WRITEOK flags for
// swapoff here so it's synchronized by both si->lock and
// swap_avail_lock, to ensure the result can be seen by
// add_to_avail_list.
lockdep_assert_held!(core::ptr::addr_of_mut!((*si).lock));
let __v0: u64 = (*si).flags & 18446744073709551613u64;
(*si).flags = __v0;
atomic_long_or((1u64 << BITS_PER_TYPE__2!(__SlateRecord_types_h_197).wrapping_sub(2)) as i64, core::ptr::addr_of_mut!((*si).inuse_pages));
} else {
// If not called by swapoff, take it off-list only if it's
// full and SWAP_USAGE_OFFLIST_BIT is not set (strictly
// si->inuse_pages == pages), any concurrent slot freeing,
// or device already removed from plist by someone else
// will make this return false.
*__slate_slot_3 = (*si).pages as u64;
if !atomic_long_try_cmpxchg(core::ptr::addr_of_mut!((*si).inuse_pages), core::ptr::addr_of_mut!(*__slate_slot_3) as *mut i64, (*__slate_slot_3 | 1u64 << BITS_PER_TYPE__2!(__SlateRecord_types_h_197).wrapping_sub(2)) as i64) {
break '__join_0;
}
}
plist_del(core::ptr::addr_of_mut!((*si).avail_list), core::ptr::addr_of_mut!(swap_avail_head));
}
spin_unlock(core::ptr::addr_of_mut!(swap_avail_lock));
}
}

// SWAP_USAGE_OFFLIST_BIT can only be cleared by this helper.
fn add_to_avail_list(si: *mut swap_info_struct, swapon: bool) {
let mut pages: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u64 = core::ptr::addr_of_mut!(pages) as *mut u64;
let mut val: core::mem::MaybeUninit<i64> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i64 = core::ptr::addr_of_mut!(val) as *mut i64;
unsafe {
'__join_0: {
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
spin_lock(core::ptr::addr_of_mut!(swap_avail_lock));
// Corresponding to SWP_WRITEOK clearing in del_from_avail_list
if swapon {
lockdep_assert_held!(core::ptr::addr_of_mut!((*si).lock));
(*si).flags = (*si).flags | 2;
} else {
while false {
}
let __v3: u64 = core::ptr::read_volatile(core::ptr::addr_of_mut!((*si).flags) as *const u64);
if __v3 & 2 == 0 {
break '__join_0;
}
}
if (atomic_long_read(core::ptr::addr_of_mut!((*si).inuse_pages) as *const __SlateRecord_types_h_197) as u64) & 1u64 << BITS_PER_TYPE__2!(__SlateRecord_types_h_197).wrapping_sub(2) != 0 {
*__slate_slot_5 = atomic_long_fetch_and_relaxed(!(1u64 << BITS_PER_TYPE__2!(__SlateRecord_types_h_197).wrapping_sub(2)) as i64, core::ptr::addr_of_mut!((*si).inuse_pages));
// When device is full and device is on the plist, only one updater will
// see (inuse_pages == si->pages) and will call del_from_avail_list. If
// that updater happen to be here, just skip adding.
*__slate_slot_4 = (*si).pages as u64;
if (*__slate_slot_5 as u64) == *__slate_slot_4 {
// Just like the cmpxchg in del_from_avail_list
if atomic_long_try_cmpxchg(core::ptr::addr_of_mut!((*si).inuse_pages), core::ptr::addr_of_mut!(*__slate_slot_4) as *mut i64, (*__slate_slot_4 | 1u64 << BITS_PER_TYPE__2!(__SlateRecord_types_h_197).wrapping_sub(2)) as i64) {
break '__join_0;
}
}
plist_add(core::ptr::addr_of_mut!((*si).avail_list), core::ptr::addr_of_mut!(swap_avail_head));
}
}
spin_unlock(core::ptr::addr_of_mut!(swap_avail_lock));
}
}

/// swap_usage_add / swap_usage_sub of each slot are serialized by ci->lock
/// within each cluster, so the total contribution to the global counter should
/// always be positive and cannot exceed the total number of usable slots.
fn swap_usage_add(si: *mut swap_info_struct, nr_entries: u32) -> bool {
let val: i64 = atomic_long_add_return_relaxed(nr_entries as i64, unsafe { core::ptr::addr_of_mut!((*si).inuse_pages) });
// If device is full, and SWAP_USAGE_OFFLIST_BIT is not set,
// remove it from the plist.
if unlikely!(val == ((unsafe { (*si).pages }) as i64)) != 0 {
del_from_avail_list(si, false);
return true;
}
false
}

fn swap_usage_sub(si: *mut swap_info_struct, nr_entries: u32) {
let val: i64 = atomic_long_sub_return_relaxed(nr_entries as i64, unsafe { core::ptr::addr_of_mut!((*si).inuse_pages) });
// If device is not full, and SWAP_USAGE_OFFLIST_BIT is set,
// add it to the plist.
if unlikely__4!((val as u64) & 1u64 << BITS_PER_TYPE__2!(__SlateRecord_types_h_197).wrapping_sub(2)) != 0 {
add_to_avail_list(si, false);
}
}

fn swap_range_alloc(si: *mut swap_info_struct, nr_entries: u32) {
if swap_usage_add(si, nr_entries) {
if vm_swap_full() {
schedule_work(unsafe { core::ptr::addr_of_mut!((*si).reclaim_work) });
}
}
atomic_long_sub(nr_entries as i64, unsafe { core::ptr::addr_of_mut!(nr_swap_pages) });
}

fn swap_range_free(si: *mut swap_info_struct, mut offset: u64, nr_entries: u32) {
let end: u64 = offset.wrapping_add(nr_entries as u64).wrapping_sub(1);
let mut swap_slot_free_notify: Option<unsafe extern "C" fn(*mut block_device, u64)> = unsafe { core::mem::zeroed() };
for i in 0..nr_entries {
zswap_invalidate(swp_entry((unsafe { (*si).r#type }) as u64, offset.wrapping_add(i as u64)));
}
if (unsafe { (*si).flags }) & 64 != 0 {
swap_slot_free_notify = unsafe { (*(*(*(*si).bdev).bd_disk).fops).swap_slot_free_notify };
} else {
swap_slot_free_notify = None;
}
while offset <= end {
arch_swap_invalidate_page((unsafe { (*si).r#type }) as i32, offset);
if swap_slot_free_notify != None {
unsafe { swap_slot_free_notify.unwrap()((*si).bdev, offset) };
}
offset = offset.wrapping_add(1);
}
// Make sure that try_to_unuse() observes si->inuse_pages reaching 0
// only after the above cleanups are done.
loop {
kcsan_wmb!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
atomic_long_add(nr_entries as i64, unsafe { core::ptr::addr_of_mut!(nr_swap_pages) });
swap_usage_sub(si, nr_entries);
}

fn get_swap_device_info(si: *mut swap_info_struct) -> bool {
if !percpu_ref_tryget_live(unsafe { core::ptr::addr_of_mut!((*si).users) }) {
return false;
}
// Guarantee the si->users are checked before accessing other
// fields of swap_info_struct, and si->flags (SWP_WRITEOK) is
// up to dated.
//
// Paired with the spin_unlock() after setup_swap_info() in
// enable_swap_info(), and smp_wmb() in swapoff.
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
true
}

/// Fast path try to get swap entries with specified order from current
/// CPU's swap entry pool (a cluster).
fn swap_alloc_fast(folio: *mut folio) -> bool {
let mut order: u32 = folio_order(folio as *const folio);
let mut offset: u32 = 0;
// Once allocated, swap_info_struct will never be completely freed,
// so checking it's liveness by get_swap_device_info is enough.
let mut __v0: *mut swap_info_struct;
{
let mut pscr_ret__: *mut swap_info_struct = unsafe { core::mem::zeroed() };
__verify_pcpu_ptr__19!();
match (const { core::mem::size_of::<*mut swap_info_struct>() }) as u64 {
1 => {
let mut __v2: *mut swap_info_struct;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:({1:r}), {0:l}", lateout(reg) __t0, in(reg) ((core::ptr::addr_of_mut!(percpu_swap_cluster.si) as *mut *mut swap_info_struct).wrapping_offset(order as isize) as u64) as *mut *mut swap_info_struct, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v2 = (pfo_val__ as u64) as *mut swap_info_struct;
pscr_ret__ = __v2;
}
2 => {
let mut __v4: *mut swap_info_struct;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:({1:r}), {0:x}", lateout(reg) __t1, in(reg) ((core::ptr::addr_of_mut!(percpu_swap_cluster.si) as *mut *mut swap_info_struct).wrapping_offset(order as isize) as u64) as *mut *mut swap_info_struct, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v4 = (pfo_val__ as u64) as *mut swap_info_struct;
pscr_ret__ = __v4;
}
4 => {
let mut __v5: *mut swap_info_struct;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:({1:r}), {0:e}", lateout(reg) __t2, in(reg) ((core::ptr::addr_of_mut!(percpu_swap_cluster.si) as *mut *mut swap_info_struct).wrapping_offset(order as isize) as u64) as *mut *mut swap_info_struct, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v5 = (pfo_val__ as u64) as *mut swap_info_struct;
pscr_ret__ = __v5;
}
8 => {
let mut __v6: *mut swap_info_struct;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:({1:r}), {0:r}", lateout(reg) __t3, in(reg) ((core::ptr::addr_of_mut!(percpu_swap_cluster.si) as *mut *mut swap_info_struct).wrapping_offset(order as isize) as u64) as *mut *mut swap_info_struct, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v6 = pfo_val__ as *mut swap_info_struct;
pscr_ret__ = __v6;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v0 = pscr_ret__;
}
let si: *mut swap_info_struct = __v0;
let mut __v7: u64;
let mut pscr_ret__: u64 = 0;
__verify_pcpu_ptr__6!();
match (const { core::mem::size_of::<u64>() }) as u64 {
1 => {
let mut __v9: u64;
let mut pfo_val__: u8 = 0;
unsafe {
let __t4: u32;
core::arch::asm!("movb %gs:({1:r}), {0:l}", lateout(reg) __t4, in(reg) ((core::ptr::addr_of_mut!(percpu_swap_cluster.offset) as *mut u64).wrapping_offset(order as isize) as u64) as *mut u64, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t4 as u8) };
}
__v9 = pfo_val__ as u64;
pscr_ret__ = __v9;
}
2 => {
let mut __v11: u64;
let mut pfo_val__: u16 = 0;
unsafe {
let __t5: u16;
core::arch::asm!("movw %gs:({1:r}), {0:x}", lateout(reg) __t5, in(reg) ((core::ptr::addr_of_mut!(percpu_swap_cluster.offset) as *mut u64).wrapping_offset(order as isize) as u64) as *mut u64, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t5 as u16) };
}
__v11 = pfo_val__ as u64;
pscr_ret__ = __v11;
}
4 => {
let mut __v12: u64;
let mut pfo_val__: u32 = 0;
unsafe {
let __t6: u32;
core::arch::asm!("movl %gs:({1:r}), {0:e}", lateout(reg) __t6, in(reg) ((core::ptr::addr_of_mut!(percpu_swap_cluster.offset) as *mut u64).wrapping_offset(order as isize) as u64) as *mut u64, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t6 as u32) };
}
__v12 = pfo_val__ as u64;
pscr_ret__ = __v12;
}
8 => {
let mut __v13: u64;
let mut pfo_val__: u64 = 0;
unsafe {
let __t7: u64;
core::arch::asm!("movq %gs:({1:r}), {0:r}", lateout(reg) __t7, in(reg) ((core::ptr::addr_of_mut!(percpu_swap_cluster.offset) as *mut u64).wrapping_offset(order as isize) as u64) as *mut u64, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t7 as u64) };
}
__v13 = pfo_val__;
pscr_ret__ = __v13;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v7 = pscr_ret__;
offset = __v7 as u32;
let __v14: bool = si.is_null() || offset == 0 || !get_swap_device_info(si);
if __v14 {
return false;
}
let ci: *mut swap_cluster_info = swap_cluster_lock(si, offset as u64);
if cluster_is_usable(ci, order as i32) {
if cluster_is_empty(ci) {
offset = cluster_offset(si, ci);
}
alloc_swap_scan_cluster(si, ci, folio, offset as u64);
} else {
swap_cluster_unlock(ci);
}
put_swap_device(si);
folio_test_swapcache(folio as *const folio)
}

/// Rotate the device and switch to a new cluster
fn swap_alloc_slow(folio: *mut folio) {
let mut next: core::mem::MaybeUninit<*mut swap_info_struct> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut swap_info_struct = core::ptr::addr_of_mut!(next) as *mut *mut swap_info_struct;
let mut si: core::mem::MaybeUninit<*mut swap_info_struct> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut swap_info_struct = core::ptr::addr_of_mut!(si) as *mut *mut swap_info_struct;
unsafe {
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
spin_lock(core::ptr::addr_of_mut!(swap_avail_lock));
'__loop_9: loop {
let __v4: *mut swap_info_struct = ((((*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(swap_avail_head)).node_list)).next as *mut ()) as *const u8).wrapping_offset(-(offsetof__11!() as isize)) as *mut ()) as *mut swap_info_struct;
*__slate_slot_6 = __v4;
let __v3: *mut swap_info_struct = ((((*(*__slate_slot_6)).avail_list.node_list.next as *mut ()) as *const u8).wrapping_offset(-(offsetof__11!() as isize)) as *mut ()) as *mut swap_info_struct;
*__slate_slot_5 = __v3;
loop {
if list_is_head(core::ptr::addr_of_mut!((*(*__slate_slot_6)).avail_list.node_list) as *const list_head, core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(swap_avail_head)).node_list) as *const list_head) == 0 {
// Rotate the device and switch to a new cluster
plist_requeue(core::ptr::addr_of_mut!((*(*__slate_slot_6)).avail_list), core::ptr::addr_of_mut!(swap_avail_head));
spin_unlock(core::ptr::addr_of_mut!(swap_avail_lock));
if get_swap_device_info(*__slate_slot_6) {
cluster_alloc_swap_entry(*__slate_slot_6, folio);
put_swap_device(*__slate_slot_6);
if folio_test_swapcache(folio as *const folio) {
return;
} else {
if folio_test_large(folio as *const folio) {
return;
}
}
}
spin_lock(core::ptr::addr_of_mut!(swap_avail_lock));
// if we got here, it's likely that si was almost full before,
// multiple callers probably all tried to get a page from the
// same si and it filled up before we could get one; or, the si
// filled up between us dropping swap_avail_lock.
// Since we dropped the swap_avail_lock, the swap_avail_list
// may have been modified; so if next is still in the
// swap_avail_head list then try it, otherwise start over if we
// have not gotten any slots.
if plist_node_empty(core::ptr::addr_of_mut!((*(*__slate_slot_5)).avail_list) as *const plist_node) != 0 {
continue '__loop_9;
} else {
*__slate_slot_6 = *__slate_slot_5;
let __v1: *mut swap_info_struct = ((((*(*__slate_slot_5)).avail_list.node_list.next as *mut ()) as *const u8).wrapping_offset(-(offsetof__11!() as isize)) as *mut ()) as *mut swap_info_struct;
*__slate_slot_5 = __v1;
}
} else {
break '__loop_9;
}
}
}
spin_unlock(core::ptr::addr_of_mut!(swap_avail_lock));
}
}

/// Discard pending clusters in a synchronized way when under high pressure.
/// Return: true if any cluster is discarded.
fn swap_sync_discard() -> bool {
let mut next: core::mem::MaybeUninit<*mut swap_info_struct> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut swap_info_struct = core::ptr::addr_of_mut!(next) as *mut *mut swap_info_struct;
let mut si: core::mem::MaybeUninit<*mut swap_info_struct> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut swap_info_struct = core::ptr::addr_of_mut!(si) as *mut *mut swap_info_struct;
let mut ret: bool = unsafe { core::mem::zeroed() };
unsafe {
ret = false;
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
spin_lock(core::ptr::addr_of_mut!(swap_lock));
'__loop_10: loop {
let __v4: *mut swap_info_struct = ((((*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(swap_active_head)).node_list)).next as *mut ()) as *const u8).wrapping_offset(-(offsetof__7!() as isize)) as *mut ()) as *mut swap_info_struct;
*__slate_slot_6 = __v4;
let __v3: *mut swap_info_struct = ((((*(*__slate_slot_6)).list.node_list.next as *mut ()) as *const u8).wrapping_offset(-(offsetof__7!() as isize)) as *mut ()) as *mut swap_info_struct;
*__slate_slot_5 = __v3;
loop {
if list_is_head(core::ptr::addr_of_mut!((*(*__slate_slot_6)).list.node_list) as *const list_head, core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(swap_active_head)).node_list) as *const list_head) == 0 {
spin_unlock(core::ptr::addr_of_mut!(swap_lock));
if get_swap_device_info(*__slate_slot_6) {
if (*(*__slate_slot_6)).flags & 1024 != 0 {
ret = swap_do_scheduled_discard(*__slate_slot_6);
}
put_swap_device(*__slate_slot_6);
}
if ret {
return true;
} else {
spin_lock(core::ptr::addr_of_mut!(swap_lock));
if plist_node_empty(core::ptr::addr_of_mut!((*(*__slate_slot_5)).list) as *const plist_node) != 0 {
continue '__loop_10;
} else {
*__slate_slot_6 = *__slate_slot_5;
let __v1: *mut swap_info_struct = ((((*(*__slate_slot_5)).list.node_list.next as *mut ()) as *const u8).wrapping_offset(-(offsetof__7!() as isize)) as *mut ()) as *mut swap_info_struct;
*__slate_slot_5 = __v1;
}
}
} else {
break '__loop_10;
}
}
}
spin_unlock(core::ptr::addr_of_mut!(swap_lock));
return false;
}
}

fn swap_extend_table_alloc(si: *mut swap_info_struct, ci: *mut swap_cluster_info, ci_off: u32, gfp: u32) -> i32 {
let mut __slate_compound_9: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_11: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_y__853: i32 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__852: i32 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut table: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut () = core::ptr::addr_of_mut!(table) as *mut *mut ();
let mut count: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut i32 = core::ptr::addr_of_mut!(count) as *mut i32;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_3 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_4 = _kzalloc_noprof((core::mem::size_of::<u32>() as u64).wrapping_mul(256), gfp, *{
__slate_compound_9 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_9)
});
alloc_tag_restore!();
} else {
*__slate_slot_4 = _kzalloc_noprof((core::mem::size_of::<u32>() as u64).wrapping_mul(256), gfp, *{
__slate_compound_11 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_11)
});
}
*__slate_slot_7 = *__slate_slot_4;
if (*__slate_slot_7).is_null() {
return 12i32.wrapping_neg();
} else {
spin_lock(core::ptr::addr_of_mut!((*ci).lock));
// Extend table allocation requires releasing ci lock first so it's
// possible that the slot has been freed, no longer overflowed, or
// a concurrent extend table allocation has already succeeded, so
// the allocation is no longer needed.
if cluster_table_is_alloced(ci) {
*__slate_slot_8 = swp_tb_get_count(__swap_table_get(ci, ci_off));
__UNIQUE_ID_x__852 = 5;
__UNIQUE_ID_y__853 = 22i32;
loop {
if __types_ok!(__UNIQUE_ID_x__852, __UNIQUE_ID_y__853) == 0 {
__compiletime_assert_854();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
let __v2: i32 = if __UNIQUE_ID_x__852 < __UNIQUE_ID_y__853 { __UNIQUE_ID_x__852 } else { __UNIQUE_ID_y__853 };
if !(*__slate_slot_8 < (1i32 << __v2.wrapping_sub((22i32 > 2) as i32)).wrapping_sub(1).wrapping_sub(1)) {
if unsafe { (*ci).extend_table }.is_null() {
(*ci).extend_table = *__slate_slot_7 as *mut u32;
spin_unlock(core::ptr::addr_of_mut!((*ci).lock));
return 0;
}
}
}
spin_unlock(core::ptr::addr_of_mut!((*ci).lock));
kfree(*__slate_slot_7 as *const ());
return 0;
}
}
}

#[unsafe(no_mangle)]
extern "C" fn swap_retry_table_alloc(entry: __SlateRecord_mm_types_h_288, gfp: u32) -> i32 {
let mut ret: i32 = 0;
let offset: u64 = swp_offset(entry);
let si: *mut swap_info_struct = get_swap_device(entry);
if si.is_null() {
return 0;
}
let ci: *mut swap_cluster_info = __swap_offset_to_cluster(si, offset);
ret = swap_extend_table_alloc(si, ci, swp_cluster_offset(entry), gfp);
put_swap_device(si);
ret
}

fn swap_extend_table_try_free(ci: *mut swap_cluster_info) {
let mut can_free: bool = true;
if unsafe { (*ci).extend_table }.is_null() {
return;
}
for i in 0u64..256u64 {
if (unsafe { *unsafe { (*ci).extend_table }.wrapping_offset(i as isize) }) != 0 {
can_free = false;
}
}
if can_free {
unsafe { kfree((*ci).extend_table as *const ()) };
unsafe {
(*ci).extend_table = core::ptr::null_mut::<u32>();
}
}
}

// Decrease the swap count of one slot, without freeing it
fn __swap_cluster_put_entry(ci: *mut swap_cluster_info, ci_off: u32) {
let mut count: i32 = 0;
let mut swp_tb: u64 = 0;
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!((*ci).lock) });
swp_tb = __swap_table_get(ci, ci_off);
count = __swp_tb_get_count(swp_tb) as i32;
VM_WARN_ON_ONCE!();
let mut __v1: i32;
let __UNIQUE_ID_x__858: i32 = 5;
let __UNIQUE_ID_y__859: i32 = 22i32;
loop {
if __types_ok!(__UNIQUE_ID_x__858, __UNIQUE_ID_y__859) == 0 {
unsafe { __compiletime_assert_860() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = if __UNIQUE_ID_x__858 < __UNIQUE_ID_y__859 { __UNIQUE_ID_x__858 } else { __UNIQUE_ID_y__859 };
if count == (1i32 << __v1.wrapping_sub((22i32 > 2) as i32)).wrapping_sub(1) {
count = (unsafe { *unsafe { (*ci).extend_table }.wrapping_offset(ci_off as isize) }) as i32;
// Overflow starts with SWP_TB_COUNT_MAX
count = count.wrapping_sub(1);
let mut __v5: i32;
let __UNIQUE_ID_x__864: i32 = 5;
let __UNIQUE_ID_y__865: i32 = 22i32;
loop {
if __types_ok!(__UNIQUE_ID_x__864, __UNIQUE_ID_y__865) == 0 {
unsafe { __compiletime_assert_866() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v5 = if __UNIQUE_ID_x__864 < __UNIQUE_ID_y__865 { __UNIQUE_ID_x__864 } else { __UNIQUE_ID_y__865 };
if count == (1i32 << __v5.wrapping_sub((22i32 > 2) as i32)).wrapping_sub(1).wrapping_sub(1) {
unsafe {
*unsafe { (*ci).extend_table }.wrapping_offset(ci_off as isize) = 0;
}
__swap_table_set(ci, ci_off, __swp_tb_mk_count(swp_tb, count));
} else {
unsafe {
*unsafe { (*ci).extend_table }.wrapping_offset(ci_off as isize) = count as u32;
}
}
} else {
let __v8: i32 = count.wrapping_sub(1);
count = __v8;
__swap_table_set(ci, ci_off, __swp_tb_mk_count(swp_tb, __v8));
}
// `SWP_TB_COUNT_MAX - 1` triggers extend table allocation. If the
// count was above that, then the extend table is no longer needed,
// so free it. And if we just put the count value from MAX - 1, it's
// also possible that a pending dup just attached an extend table.
let mut __v9: i32;
let __UNIQUE_ID_x__867: i32 = 5;
let __UNIQUE_ID_y__868: i32 = 22i32;
loop {
if __types_ok!(__UNIQUE_ID_x__867, __UNIQUE_ID_y__868) == 0 {
unsafe { __compiletime_assert_869() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v9 = if __UNIQUE_ID_x__867 < __UNIQUE_ID_y__868 { __UNIQUE_ID_x__867 } else { __UNIQUE_ID_y__868 };
let __v11: bool = if count == (1i32 << __v9.wrapping_sub((22i32 > 2) as i32)).wrapping_sub(1).wrapping_sub(2) {
true
} else {
let mut __v12: i32;
let __UNIQUE_ID_x__870: i32 = 5;
let __UNIQUE_ID_y__871: i32 = 22i32;
loop {
if __types_ok!(__UNIQUE_ID_x__870, __UNIQUE_ID_y__871) == 0 {
unsafe { __compiletime_assert_872() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v12 = if __UNIQUE_ID_x__870 < __UNIQUE_ID_y__871 { __UNIQUE_ID_x__870 } else { __UNIQUE_ID_y__871 };
count == (1i32 << __v12.wrapping_sub((22i32 > 2) as i32)).wrapping_sub(1).wrapping_sub(1)
};
if __v11 {
swap_extend_table_try_free(ci);
}
}

/// swap_put_entries_cluster - Decrease the swap count of slots within one cluster
/// @si: The swap device.
/// @offset: start offset of slots.
/// @nr: number of slots.
/// @reclaim_cache: if true, also reclaim the swap cache if slots are freed.
///
/// This helper decreases the swap count of a set of slots and tries to
/// batch free them. Also reclaims the swap cache if @reclaim_cache is true.
///
/// Context: The specified slots must be pinned by existing swap count or swap
/// cache reference, so they won't be released until this helper returns.
fn swap_put_entries_cluster(si: *mut swap_info_struct, mut offset: u64, nr: i32, reclaim_cache: bool) {
let mut ci_off: u32 = 0;
let mut ci_end: u32 = 0;
let end: u64 = offset.wrapping_add(nr as u64);
let mut need_reclaim: bool = false;
let mut nr_reclaimed: u32 = 0;
let mut swp_tb: u64 = 0;
let mut ci_batch: i32 = 1i32.wrapping_neg();
let ci: *mut swap_cluster_info = swap_cluster_lock(si, offset);
ci_off = (offset % 256) as u32;
ci_end = ci_off.wrapping_add(nr as u32);
loop {
'__slate_continue_0: {
swp_tb = __swap_table_get(ci, ci_off);
if swp_tb_get_count(swp_tb) == 1 {
// count == 1 and non-cached slots will be batch freed.
if !swp_tb_is_folio(swp_tb) {
if ci_batch == 1i32.wrapping_neg() {
ci_batch = ci_off as i32;
}
break '__slate_continue_0;
}
// count will be 0 after put, slot can be reclaimed
need_reclaim = true;
}
// A count != 1 or cached slot can't be freed. Put its swap
// count and then free the interrupted pending batch. Cached
// slots will be freed when folio is removed from swap cache
// (__swap_cache_del_folio).
__swap_cluster_put_entry(ci, ci_off);
if ci_batch != 1i32.wrapping_neg() {
__swap_cluster_free_entries(si, ci, ci_batch as u32, ci_off.wrapping_sub(ci_batch as u32));
ci_batch = 1i32.wrapping_neg();
}
}
let __v2: u32 = ci_off.wrapping_add(1);
ci_off = __v2;
if __v2 >= ci_end {
break;
}
}
if ci_batch != 1i32.wrapping_neg() {
__swap_cluster_free_entries(si, ci, ci_batch as u32, ci_off.wrapping_sub(ci_batch as u32));
}
swap_cluster_unlock(ci);
if !need_reclaim || !reclaim_cache {
return;
}
loop {
nr_reclaimed = __try_to_reclaim_swap(si, offset, 6u64) as u32;
offset = offset.wrapping_add(1);
if nr_reclaimed != 0 {
offset = round_up__2!(offset, abs!(nr_reclaimed));
}
if offset >= end {
break;
}
}
}

// Increase the swap count of one slot.
fn __swap_cluster_dup_entry(ci: *mut swap_cluster_info, ci_off: u32) -> i32 {
let mut count: i32 = 0;
let mut swp_tb: u64 = 0;
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!((*ci).lock) });
swp_tb = __swap_table_get(ci, ci_off);
// Bad or special slots can't be handled
let mut __v1: i64;
{
let __ret_warn_on: i32 = !(!swp_tb_is_bad(swp_tb)) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("873: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 873b - ., 3; .popsection\n# {0}", const 873 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___swap_cluster_dup_entry_2, sym __slate_asm_string___swap_cluster_dup_entry_3, const 1654 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("874: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 874b - ., 4; .popsection\n# {0}", const 874 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
}
if __v1 != 0 {
return 22i32.wrapping_neg();
}
count = __swp_tb_get_count(swp_tb) as i32;
// Must be either cached or have a count already
let mut __v6: i64;
{
let mut __ret_warn_on: i32 = 0;
let __v7: bool = count == 0 && !swp_tb_is_folio(swp_tb);
__ret_warn_on = __v7 as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("875: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 875b - ., 3; .popsection\n# {0}", const 875 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___swap_cluster_dup_entry_8, sym __slate_asm_string___swap_cluster_dup_entry_9, const 1658 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("876: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 876b - ., 4; .popsection\n# {0}", const 876 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v6 = (__ret_warn_on != 0) as i64;
}
if __v6 != 0 {
return 2i32.wrapping_neg();
}
let mut __v12: i32;
let __UNIQUE_ID_x__877: i32 = 5;
let __UNIQUE_ID_y__878: i32 = 22i32;
loop {
if __types_ok!(__UNIQUE_ID_x__877, __UNIQUE_ID_y__878) == 0 {
unsafe { __compiletime_assert_879() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v12 = if __UNIQUE_ID_x__877 < __UNIQUE_ID_y__878 { __UNIQUE_ID_x__877 } else { __UNIQUE_ID_y__878 };
if count < (1i32 << __v12.wrapping_sub((22i32 > 2) as i32)).wrapping_sub(1).wrapping_sub(1) {
__swap_table_set(ci, ci_off, __swp_tb_mk_count(swp_tb, count.wrapping_add(1)));
VM_WARN_ON_ONCE!();
} else {
let mut __v14: i32;
let __UNIQUE_ID_x__880: i32 = 5;
let __UNIQUE_ID_y__881: i32 = 22i32;
loop {
if __types_ok!(__UNIQUE_ID_x__880, __UNIQUE_ID_y__881) == 0 {
unsafe { __compiletime_assert_882() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v14 = if __UNIQUE_ID_x__880 < __UNIQUE_ID_y__881 { __UNIQUE_ID_x__880 } else { __UNIQUE_ID_y__881 };
if count == (1i32 << __v14.wrapping_sub((22i32 > 2) as i32)).wrapping_sub(1).wrapping_sub(1) {
if !unsafe { (*ci).extend_table }.is_null() {
VM_WARN_ON_ONCE!();
let mut __v16: i32;
let __UNIQUE_ID_x__883: i32 = 5;
let __UNIQUE_ID_y__884: i32 = 22i32;
loop {
if __types_ok!(__UNIQUE_ID_x__883, __UNIQUE_ID_y__884) == 0 {
unsafe { __compiletime_assert_885() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v16 = if __UNIQUE_ID_x__883 < __UNIQUE_ID_y__884 { __UNIQUE_ID_x__883 } else { __UNIQUE_ID_y__884 };
unsafe {
*unsafe { (*ci).extend_table }.wrapping_offset(ci_off as isize) = (1i32 << __v16.wrapping_sub((22i32 > 2) as i32)).wrapping_sub(1) as u32;
}
let mut __v18: i32;
let __UNIQUE_ID_x__886: i32 = 5;
let __UNIQUE_ID_y__887: i32 = 22i32;
loop {
if __types_ok!(__UNIQUE_ID_x__886, __UNIQUE_ID_y__887) == 0 {
unsafe { __compiletime_assert_888() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v18 = if __UNIQUE_ID_x__886 < __UNIQUE_ID_y__887 { __UNIQUE_ID_x__886 } else { __UNIQUE_ID_y__887 };
__swap_table_set(ci, ci_off, __swp_tb_mk_count(swp_tb, (1i32 << __v18.wrapping_sub((22i32 > 2) as i32)).wrapping_sub(1)));
} else {
return 12i32.wrapping_neg();
}
} else {
let mut __v20: i32;
let __UNIQUE_ID_x__889: i32 = 5;
let __UNIQUE_ID_y__890: i32 = 22i32;
loop {
if __types_ok!(__UNIQUE_ID_x__889, __UNIQUE_ID_y__890) == 0 {
unsafe { __compiletime_assert_891() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v20 = if __UNIQUE_ID_x__889 < __UNIQUE_ID_y__890 { __UNIQUE_ID_x__889 } else { __UNIQUE_ID_y__890 };
if count == (1i32 << __v20.wrapping_sub((22i32 > 2) as i32)).wrapping_sub(1) {
VM_WARN_ON_ONCE!();
let __v22: *mut u32 = unsafe { (*ci).extend_table }.wrapping_offset(ci_off as isize);
let __v23: u32 = unsafe { *__v22 };
unsafe {
*__v22 = __v23.wrapping_add(1);
}
} else {
// Never happens unless counting went wrong
let mut __v25: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("892: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 892b - ., 3; .popsection\n# {0}", const 892 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___swap_cluster_dup_entry_26, sym __slate_asm_string___swap_cluster_dup_entry_27, const 1678 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("893: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 893b - ., 4; .popsection\n# {0}", const 893 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v25 = (__ret_warn_on != 0) as i64;
}
}
}
0
}

/// swap_dup_entries_cluster: Increase the swap count of slots within one cluster.
/// @si: The swap device.
/// @offset: start offset of slots.
/// @nr: number of slots.
///
/// Context: The specified slots must be pinned by existing swap count or swap
/// cache reference, so they won't be released until this helper returns.
/// Return: 0 on success. -ENOMEM if the swap count maxed out (SWP_TB_COUNT_MAX)
/// and failed to allocate an extended table, -EINVAL if any entry is bad entry.
fn swap_dup_entries_cluster(si: *mut swap_info_struct, offset: u64, nr: i32) -> i32 {
let mut ci_end: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u32 = core::ptr::addr_of_mut!(ci_end) as *mut u32;
let mut ci_off: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut u32 = core::ptr::addr_of_mut!(ci_off) as *mut u32;
let mut ci_start: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u32 = core::ptr::addr_of_mut!(ci_start) as *mut u32;
let mut ci: core::mem::MaybeUninit<*mut swap_cluster_info> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut swap_cluster_info = core::ptr::addr_of_mut!(ci) as *mut *mut swap_cluster_info;
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
unsafe {
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_6 = (offset % 256) as u32;
*__slate_slot_4 = (*__slate_slot_6).wrapping_add(nr as u32);
*__slate_slot_5 = *__slate_slot_6;
*__slate_slot_7 = swap_cluster_lock(si, offset);
'__join_1: {
loop {
*__slate_slot_8 = __swap_cluster_dup_entry(*__slate_slot_7, *__slate_slot_5);
if unlikely__2!(*__slate_slot_8) != 0 {
if *__slate_slot_8 == 12i32.wrapping_neg() {
spin_unlock(core::ptr::addr_of_mut!((*(*__slate_slot_7)).lock));
*__slate_slot_8 = swap_extend_table_alloc(si, *__slate_slot_7, *__slate_slot_5, (BIT!(5i32) as u32) | (BIT!(11i32) as u32));
spin_lock(core::ptr::addr_of_mut!((*(*__slate_slot_7)).lock));
if *__slate_slot_8 != 0 {
break '__join_1;
}
} else {
break '__join_1;
}
} else {
let __v3: u32 = *__slate_slot_5;
let __v2: u32 = __v3.wrapping_add(1);
*__slate_slot_5 = __v2;
if __v2 >= *__slate_slot_4 {
break;
}
}
}
swap_cluster_unlock(*__slate_slot_7);
return 0;
}
loop {
let __v1: u32 = *__slate_slot_5;
*__slate_slot_5 = __v1.wrapping_sub(1);
if __v1 > *__slate_slot_6 {
__swap_cluster_put_entry(*__slate_slot_7, *__slate_slot_5);
} else {
break;
}
}
swap_extend_table_try_free(*__slate_slot_7);
swap_cluster_unlock(*__slate_slot_7);
return *__slate_slot_8;
}
}

/// folio_alloc_swap - allocate swap space for a folio
/// @folio: folio we want to move to swap
///
/// Allocate swap space for the folio and add the folio to the
/// swap cache.
///
/// Context: Caller needs to hold the folio lock.
/// Return: Whether the folio was added to the swap cache.
#[unsafe(no_mangle)]
extern "C" fn folio_alloc_swap(folio: *mut folio) -> i32 {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut pfo_val__: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut u64 = core::ptr::addr_of_mut!(pfo_val__) as *mut u64;
let mut tcp_ptr__: u64 = unsafe { core::mem::zeroed() };
let mut __l: core::mem::MaybeUninit<*mut local_lock> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut local_lock = core::ptr::addr_of_mut!(__l) as *mut *mut local_lock;
let mut __tl: core::mem::MaybeUninit<*mut local_trylock> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut local_trylock = core::ptr::addr_of_mut!(__tl) as *mut *mut local_trylock;
let mut pfo_val__: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut u64 = core::ptr::addr_of_mut!(pfo_val__) as *mut u64;
let mut tcp_ptr___1: u64 = unsafe { core::mem::zeroed() };
let mut __l: core::mem::MaybeUninit<*mut local_lock> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut *mut local_lock = core::ptr::addr_of_mut!(__l) as *mut *mut local_lock;
let mut __tl: core::mem::MaybeUninit<*mut local_trylock> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut *mut local_trylock = core::ptr::addr_of_mut!(__tl) as *mut *mut local_trylock;
unsafe {
let order: u32 = folio_order(folio as *const folio);
VM_BUG_ON_FOLIO!();
VM_BUG_ON_FOLIO!();
if order != 0 {
// Reject large allocation when THP_SWAP is disabled,
// the caller should split the folio and try again.
return 11i32.wrapping_neg();
} else {
loop {
preempt_count_inc!();
core::arch::asm!("", options(nostack, att_syntax));
if true {
loop {
let __slate_auto_bytes_3 = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<local_lock>() as *const ();
if true {
break;
}
}
let __slate_auto_bytes = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __t1: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t1, sym this_cpu_off, options(pure, readonly, nostack, att_syntax));
*__slate_slot_13 = unsafe { core::mem::transmute::<u64, u64>(__t1 as u64) };
tcp_ptr___1 = *__slate_slot_13;
let __v11: u64 = tcp_ptr___1.wrapping_add(core::ptr::addr_of_mut!(percpu_swap_cluster.lock) as u64);
tcp_ptr___1 = __v11;
*__slate_slot_19 = tcp_ptr___1 as *mut local_lock;
*__slate_slot_20 = *__slate_slot_19 as *mut local_trylock;
local_lock_acquire(*__slate_slot_19);
if true {
break;
}
}
__acquire!();
if true {
if !swap_alloc_fast(folio) {
swap_alloc_slow(folio);
}
loop {
__release!();
loop {
let __slate_auto_bytes_1 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<local_lock>() as *const ();
if true {
break;
}
}
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __t0: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t0, sym this_cpu_off, options(pure, readonly, nostack, att_syntax));
*__slate_slot_3 = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
tcp_ptr__ = *__slate_slot_3;
let __v1: u64 = tcp_ptr__.wrapping_add(core::ptr::addr_of_mut!(percpu_swap_cluster.lock) as u64);
tcp_ptr__ = __v1;
*__slate_slot_9 = tcp_ptr__ as *mut local_lock;
*__slate_slot_10 = *__slate_slot_9 as *mut local_trylock;
local_lock_release(*__slate_slot_9);
if true {
break;
}
}
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
if true {
break;
}
}
if order == 0 {
__v0 = unlikely!(!folio_test_swapcache(folio as *const folio)) != 0;
} else {
__v0 = false;
}
if __v0 {
if !swap_sync_discard() {
break;
}
} else {
break;
}
}
}
}
// Need to call this even if allocation failed, for MEMCG_SWAP_FAIL.
if unlikely__2!(mem_cgroup_try_charge_swap(folio)) != 0 {
swap_cache_del_folio(folio);
}
if unlikely!(!folio_test_swapcache(folio as *const folio)) != 0 {
return 12i32.wrapping_neg();
} else {
return 0;
}
}
}
// Allocation size should never exceed cluster size
// (HPAGE_PMD_SIZE).
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_896: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

/// folio_dup_swap() - Increase swap count of swap entries of a folio.
/// @folio: folio with swap entries bounded.
/// @page: if not NULL, only increase the swap count of this page.
///
/// Typically called when the folio is unmapped and have its swap entry to
/// take its place: Swap entries allocated to a folio has count == 0 and pinned
/// by swap cache. The swap cache pin doesn't increase the swap count. This
/// helper sets the initial count == 1 and increases the count as the folio is
/// unmapped and swap entries referencing the slots are generated to replace
/// the folio.
///
/// Context: Caller must ensure the folio is locked and in the swap cache.
/// NOTE: The caller also has to ensure there is no raced call to
/// swap_put_entries_direct on its swap entry before this helper returns, or
/// the swap count may underflow.
#[unsafe(no_mangle)]
extern "C" fn folio_dup_swap(folio: *mut folio, page: *mut page) -> i32 {
let mut entry: __SlateRecord_mm_types_h_288 = unsafe { (*folio).__slate_anon_0.__slate_anon_0.__slate_anon_4.swap };
let mut nr_pages: u64 = folio_nr_pages(folio as *const folio);
VM_WARN_ON_FOLIO!();
VM_WARN_ON_FOLIO!();
if !page.is_null() {
let __v0: u64 = folio_page_idx(folio as *const folio, page as *const page);
let __v1: u64 = entry.val;
entry.val = __v1.wrapping_add(__v0);
nr_pages = 1;
}
swap_dup_entries_cluster(swap_entry_to_info(entry), swp_offset(entry), nr_pages as i32)
}

/// folio_put_swap() - Decrease swap count of swap entries of a folio.
/// @folio: folio with swap entries bounded, must be in swap cache and locked.
/// @page: if not NULL, only decrease the swap count of this page.
///
/// This won't free the swap slots even if swap count drops to zero, they are
/// still pinned by the swap cache. User may call folio_free_swap to free them.
/// Context: Caller must ensure the folio is locked and in the swap cache.
#[unsafe(no_mangle)]
extern "C" fn folio_put_swap(folio: *mut folio, page: *mut page) {
let mut entry: __SlateRecord_mm_types_h_288 = unsafe { (*folio).__slate_anon_0.__slate_anon_0.__slate_anon_4.swap };
let mut nr_pages: u64 = folio_nr_pages(folio as *const folio);
let si: *mut swap_info_struct = __swap_entry_to_info(entry);
VM_WARN_ON_FOLIO!();
VM_WARN_ON_FOLIO!();
if !page.is_null() {
let __v0: u64 = folio_page_idx(folio as *const folio, page as *const page);
let __v1: u64 = entry.val;
entry.val = __v1.wrapping_add(__v0);
nr_pages = 1;
}
swap_put_entries_cluster(si, swp_offset(entry), nr_pages as i32, false);
}

/// When we get a swap entry, if there aren't some other ways to
/// prevent swapoff, such as the folio in swap cache is locked, RCU
/// reader side is locked, etc., the swap entry may become invalid
/// because of swapoff.  Then, we need to enclose all swap related
/// functions with get_swap_device() and put_swap_device(), unless the
/// swap functions call get/put_swap_device() by themselves.
///
/// RCU reader side lock (including any spinlock) is sufficient to
/// prevent swapoff, because synchronize_rcu() is called in swapoff()
/// before freeing data structures.
///
/// Check whether swap entry is valid in the swap device.  If so,
/// return pointer to swap_info_struct, and keep the swap entry valid
/// via preventing the swap device from being swapoff, until
/// put_swap_device() is called.  Otherwise return NULL.
///
/// Notice that swapoff or swapoff+swapon can still happen before the
/// percpu_ref_tryget_live() in get_swap_device() or after the
/// percpu_ref_put() in put_swap_device() if there isn't any other way
/// to prevent swapoff.  The caller must be prepared for that.  For
/// example, the following situation is possible.
///
///   CPU1				CPU2
///   do_swap_page()
///     ...				swapoff+swapon
///     swap_cache_alloc_folio()
///       // check swap_map
///     // verify PTE not changed
///
/// In __swap_duplicate(), the swap_map need to be checked before
/// changing partly because the specified swap entry may be for another
/// swap device which has been swapoff.  And in do_swap_page(), after
/// the page is read from the swap device, the PTE is verified not
/// changed with the page table locked to check whether the swap device
/// has been swapoff or swapoff+swapon.
#[unsafe(no_mangle)]
extern "C" fn get_swap_device(entry: __SlateRecord_mm_types_h_288) -> *mut swap_info_struct {
let mut offset: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut u64 = core::ptr::addr_of_mut!(offset) as *mut u64;
let mut si: core::mem::MaybeUninit<*mut swap_info_struct> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut swap_info_struct = core::ptr::addr_of_mut!(si) as *mut *mut swap_info_struct;
unsafe {
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
if entry.val != 0 {
*__slate_slot_3 = swap_entry_to_info(entry);
if (*__slate_slot_3).is_null() {
if __ratelimit!(core::ptr::addr_of_mut!(_rs__1902)) != 0 {
while false {
}
_printk(c"\x013%s: %s%08lx\n".as_ptr() as *const u8, c"get_swap_device".as_ptr() as *mut u8, core::ptr::addr_of!(Bad_file.0) as *const u8, entry.val);
}
} else {
if get_swap_device_info(*__slate_slot_3) {
*__slate_slot_2 = swp_offset(entry);
if *__slate_slot_2 >= ((*(*__slate_slot_3)).max as u64) {
if __ratelimit!(core::ptr::addr_of_mut!(_rs__1906)) != 0 {
while false {
}
_printk(c"\x013%s: %s%08lx\n".as_ptr() as *const u8, c"get_swap_device".as_ptr() as *mut u8, core::ptr::addr_of!(Bad_offset.0) as *const u8, entry.val);
}
percpu_ref_put(core::ptr::addr_of_mut!((*(*__slate_slot_3)).users));
return core::ptr::null_mut::<swap_info_struct>();
} else {
return *__slate_slot_3;
}
}
}
}
return core::ptr::null_mut::<swap_info_struct>();
}
}

static mut _rs__1902: ratelimit_state = {
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

static mut _rs__1906: ratelimit_state = {
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

/// Free a set of swap slots after their swap count dropped to zero, or will be
/// zero after putting the last ref (saves one __swap_cluster_put_entry call).
#[unsafe(no_mangle)]
extern "C" fn __swap_cluster_free_entries(si: *mut swap_info_struct, ci: *mut swap_cluster_info, ci_start: u32, nr_pages: u32) {
let mut old_tb: u64 = 0;
let mut batch_id: u16 = 0;
let mut id_cur: u16 = 0;
let mut ci_off: u32 = ci_start;
let ci_end: u32 = ci_start.wrapping_add(nr_pages);
let ci_head: u64 = cluster_offset(si, ci) as u64;
let mut batch_off: u32 = ci_off;
VM_WARN_ON!();
unsafe {
(*ci).count = (((*ci).count as i32) as u32).wrapping_sub(nr_pages) as u16;
}
loop {
old_tb = __swap_table_get(ci, ci_off);
// Freeing is done after release of the last swap count
// ref, or after swap cache is dropped
VM_WARN_ON!();
// Resetting the slot to NULL also clears the inline flags.
__swap_table_set(ci, ci_off, null_to_swp_tb());
if 22i32 <= 2 {
__swap_table_clear_zero(ci, ci_off);
}
// Uncharge swap slots by memcg in batches. Consecutive
// slots with the same cgroup id are uncharged together.
id_cur = __swap_cgroup_clear(ci, ci_off, 1u64);
if (batch_id as i32) != (id_cur as i32) {
if batch_id != 0 {
mem_cgroup_uncharge_swap(batch_id, ci_off.wrapping_sub(batch_off));
}
batch_id = id_cur;
batch_off = ci_off;
}
let __v5: u32 = ci_off.wrapping_add(1);
ci_off = __v5;
if __v5 >= ci_end {
break;
}
}
if batch_id != 0 {
mem_cgroup_uncharge_swap(batch_id, ci_off.wrapping_sub(batch_off));
}
swap_range_free(si, ci_head.wrapping_add(ci_start as u64), nr_pages);
swap_cluster_assert_empty(ci, ci_start, nr_pages, false);
if (unsafe { (*ci).count }) == 0 {
free_cluster(si, ci);
} else {
partial_free_cluster(si, ci);
}
}

#[unsafe(no_mangle)]
extern "C" fn __swap_count(entry: __SlateRecord_mm_types_h_288) -> i32 {
let ci: *mut swap_cluster_info = __swap_entry_to_cluster(entry);
let ci_off: u32 = swp_cluster_offset(entry);
swp_tb_get_count(__swap_table_get(ci, ci_off))
}

/// swap_entry_swapped - Check if the swap entry is swapped.
/// @si: the swap device.
/// @entry: the swap entry.
#[unsafe(no_mangle)]
extern "C" fn swap_entry_swapped(si: *mut swap_info_struct, entry: __SlateRecord_mm_types_h_288) -> bool {
let offset: u64 = swp_offset(entry);
let mut swp_tb: u64 = 0;
let ci: *mut swap_cluster_info = swap_cluster_lock(si, offset);
swp_tb = swap_table_get(ci, (offset % 256) as u32);
swap_cluster_unlock(ci);
swp_tb_get_count(swp_tb) > 0
}

// How many references to @entry are currently swapped out?
// This returns exact answer.
/// folio_maybe_swapped - Test if a folio covers any swap slot with count > 0.
///
/// Check if a folio is swapped. Holding the folio lock ensures the folio won't
/// go from not-swapped to swapped because the initial swap count increment can
/// only be done by folio_dup_swap, which also locks the folio. But a concurrent
/// decrease of swap count is possible through swap_put_entries_direct, so this
/// may return a false positive.
///
/// Context: Caller must ensure the folio is locked and in the swap cache.
fn folio_maybe_swapped(folio: *mut folio) -> bool {
let entry: __SlateRecord_mm_types_h_288 = unsafe { (*folio).__slate_anon_0.__slate_anon_0.__slate_anon_4.swap };
let mut ci_off: u32 = 0;
let mut ci_end: u32 = 0;
let mut ret: bool = false;
VM_WARN_ON_ONCE_FOLIO!();
VM_WARN_ON_ONCE_FOLIO!();
let ci: *mut swap_cluster_info = __swap_entry_to_cluster(entry);
ci_off = swp_cluster_offset(entry);
ci_end = (ci_off as u64).wrapping_add(folio_nr_pages(folio as *const folio)) as u32;
// Extra locking not needed, folio lock ensures its swap entries
// won't be released, the backing data won't be gone either.
rcu_read_lock();
loop {
if __swp_tb_get_count(__swap_table_get(ci, ci_off)) != 0 {
ret = true;
break;
}
let __v2: u32 = ci_off.wrapping_add(1);
ci_off = __v2;
if __v2 >= ci_end {
break;
}
}
rcu_read_unlock();
ret
}

fn folio_swapcache_freeable(folio: *mut folio) -> bool {
VM_BUG_ON_FOLIO!();
if !folio_test_swapcache(folio as *const folio) {
return false;
}
if folio_test_writeback(folio as *const folio) {
return false;
}
// Once hibernation has begun to create its image of memory,
// there's a danger that one of the calls to folio_free_swap()
// - most probably a call from __try_to_reclaim_swap() while
// hibernation is allocating its own swap pages for the image,
// but conceivably even a call from memory reclaim - will free
// the swap from a folio which has already been recorded in the
// image as a clean swapcache folio, and then reuse its swap for
// another page of the image.  On waking from hibernation, the
// original folio might be freed under memory pressure, then
// later read back in from swap, now with the wrong data.
//
// Hibernation suspends storage while it is writing the image
// to disk so check that here.
if pm_suspended_storage() {
return false;
}
true
}

/// folio_free_swap() - Free the swap space used for this folio.
/// @folio: The folio to remove.
///
/// If swap is getting full, or if there are no more mappings of this folio,
/// then call folio_free_swap to free its swap space.
///
/// Return: true if we were able to release the swap space.
#[unsafe(no_mangle)]
extern "C" fn folio_free_swap(folio: *mut folio) -> bool {
if !folio_swapcache_freeable(folio) {
return false;
}
if folio_maybe_swapped(folio) {
return false;
}
unsafe { swap_cache_del_folio(folio) };
folio_set_dirty(folio);
true
}

// swap_put_entries_direct() - Release reference on range of swap entries and
//                             reclaim their cache if no more references remain.
// @entry: First entry of range.
// @nr: Number of entries in range.
//
// For each swap entry in the contiguous range, release a reference. If any swap
// entries become free, try to reclaim their underlying folios, if present. The
// offset range is defined by [entry.offset, entry.offset + nr).
//
// Context: Caller must ensure there is no race condition on the reference
// owner. e.g., locking the PTL of a PTE containing the entry being released.
/// swap_alloc_hibernation_slot() - Allocate a swap slot for hibernation.
/// @type: swap device type index to allocate from.
///
/// The caller must ensure the swap device is stable, either by pinning
/// it (SWP_HIBERNATION) or by freezing user-space.
///
/// Return: a valid swp_entry_t on success, or an empty entry (val == 0)
/// on failure.
#[unsafe(no_mangle)]
extern "C" fn swap_alloc_hibernation_slot(r#type: i32) -> __SlateRecord_mm_types_h_288 {
let mut pfo_val__: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut u64 = core::ptr::addr_of_mut!(pfo_val__) as *mut u64;
let mut tcp_ptr__: u64 = unsafe { core::mem::zeroed() };
let mut __l: core::mem::MaybeUninit<*mut local_lock> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut local_lock = core::ptr::addr_of_mut!(__l) as *mut *mut local_lock;
let mut __tl: core::mem::MaybeUninit<*mut local_trylock> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut local_trylock = core::ptr::addr_of_mut!(__tl) as *mut *mut local_trylock;
let mut pfo_val___1: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut u64 = core::ptr::addr_of_mut!(pfo_val___1) as *mut u64;
let mut pscr_ret__: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut u64 = core::ptr::addr_of_mut!(pscr_ret__) as *mut u64;
let mut pfo_val___5: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut u64 = core::ptr::addr_of_mut!(pfo_val___5) as *mut u64;
let mut pscr_ret__: core::mem::MaybeUninit<*mut swap_info_struct> = core::mem::MaybeUninit::uninit();
let __slate_slot_28: *mut *mut swap_info_struct = core::ptr::addr_of_mut!(pscr_ret__) as *mut *mut swap_info_struct;
let mut pfo_val__: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_32: *mut u64 = core::ptr::addr_of_mut!(pfo_val__) as *mut u64;
let mut tcp_ptr___1: u64 = unsafe { core::mem::zeroed() };
let mut __l: core::mem::MaybeUninit<*mut local_lock> = core::mem::MaybeUninit::uninit();
let __slate_slot_38: *mut *mut local_lock = core::ptr::addr_of_mut!(__l) as *mut *mut local_lock;
let mut __tl: core::mem::MaybeUninit<*mut local_trylock> = core::mem::MaybeUninit::uninit();
let __slate_slot_39: *mut *mut local_trylock = core::ptr::addr_of_mut!(__tl) as *mut *mut local_trylock;
let mut entry: core::mem::MaybeUninit<__SlateRecord_mm_types_h_288> = core::mem::MaybeUninit::uninit();
let __slate_slot_40: *mut __SlateRecord_mm_types_h_288 = core::ptr::addr_of_mut!(entry) as *mut __SlateRecord_mm_types_h_288;
let mut ci: core::mem::MaybeUninit<*mut swap_cluster_info> = core::mem::MaybeUninit::uninit();
let __slate_slot_41: *mut *mut swap_cluster_info = core::ptr::addr_of_mut!(ci) as *mut *mut swap_cluster_info;
let mut offset: u64 = unsafe { core::mem::zeroed() };
let mut pcp_offset: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_43: *mut u64 = core::ptr::addr_of_mut!(pcp_offset) as *mut u64;
let mut si: *mut swap_info_struct = unsafe { core::mem::zeroed() };
let mut pcp_si: core::mem::MaybeUninit<*mut swap_info_struct> = core::mem::MaybeUninit::uninit();
let __slate_slot_45: *mut *mut swap_info_struct = core::ptr::addr_of_mut!(pcp_si) as *mut *mut swap_info_struct;
unsafe {
let __slate_auto_bytes_8 = __slate_slot_45 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 8usize);
si = swap_type_to_info(r#type);
let __slate_auto_bytes_9 = __slate_slot_43 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 8usize);
offset = 0;
let __slate_auto_bytes_10 = __slate_slot_41 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_10.add(0usize), 0u8, 8usize);
core::ptr::write(__slate_slot_40, __SlateRecord_mm_types_h_288 { val: 0 });
if !si.is_null() {
// Try the local cluster first if it matches the device. If
// not, try grab a new cluster and override local cluster.
loop {
preempt_count_inc!();
core::arch::asm!("", options(nostack, att_syntax));
if true {
loop {
let __slate_auto_bytes_6 = __slate_slot_39 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_7 = __slate_slot_38 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 8usize);
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<local_lock>() as *const ();
if true {
break;
}
}
let __slate_auto_bytes = __slate_slot_32 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym this_cpu_off, options(pure, readonly, nostack, att_syntax));
*__slate_slot_32 = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
tcp_ptr___1 = *__slate_slot_32;
let __v30: u64 = tcp_ptr___1.wrapping_add(core::ptr::addr_of_mut!(percpu_swap_cluster.lock) as u64);
tcp_ptr___1 = __v30;
*__slate_slot_38 = tcp_ptr___1 as *mut local_lock;
*__slate_slot_39 = *__slate_slot_38 as *mut local_trylock;
local_lock_acquire(*__slate_slot_38);
if true {
break;
}
}
__acquire!();
if true {
break;
}
}
}
{
let __slate_auto_bytes_3 = __slate_slot_28 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
__verify_pcpu_ptr__19!();
let __slate_auto_bytes_4 = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __t1: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t1, sym percpu_swap_cluster, options(nostack, att_syntax));
*__slate_slot_20 = unsafe { core::mem::transmute::<u64, u64>(__t1 as u64) };
*__slate_slot_28 = *__slate_slot_20 as *mut swap_info_struct;
*__slate_slot_45 = *__slate_slot_28;
let __slate_auto_bytes_5 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
__verify_pcpu_ptr__6!();
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __t2: u64;
core::arch::asm!("movq %gs:{1}+8(%rip), {0:r}", lateout(reg) __t2, sym percpu_swap_cluster, options(nostack, att_syntax));
*__slate_slot_10 = unsafe { core::mem::transmute::<u64, u64>(__t2 as u64) };
*__slate_slot_18 = *__slate_slot_10;
*__slate_slot_43 = *__slate_slot_18;
if *__slate_slot_45 == si && *__slate_slot_43 != 0 {
*__slate_slot_41 = swap_cluster_lock(si, *__slate_slot_43);
if cluster_is_usable(*__slate_slot_41, 0i32) {
offset = alloc_swap_scan_cluster(si, *__slate_slot_41, core::ptr::null_mut::<folio>(), *__slate_slot_43) as u64;
} else {
swap_cluster_unlock(*__slate_slot_41);
}
}
}
if offset == 0 {
offset = cluster_alloc_swap_entry(si, core::ptr::null_mut::<folio>());
}
loop {
__release!();
loop {
let __slate_auto_bytes_1 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
loop {
core::ptr::null_mut::<local_lock>() as *const ();
if true {
break;
}
}
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __t0: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t0, sym this_cpu_off, options(pure, readonly, nostack, att_syntax));
*__slate_slot_2 = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
tcp_ptr__ = *__slate_slot_2;
let __v0: u64 = tcp_ptr__.wrapping_add(core::ptr::addr_of_mut!(percpu_swap_cluster.lock) as u64);
tcp_ptr__ = __v0;
*__slate_slot_8 = tcp_ptr__ as *mut local_lock;
*__slate_slot_9 = *__slate_slot_8 as *mut local_trylock;
local_lock_release(*__slate_slot_8);
if true {
break;
}
}
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
if true {
break;
}
}
if offset != 0 {
*__slate_slot_40 = swp_entry((*si).r#type as u64, offset);
}
}
return *__slate_slot_40;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_909: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

/// swap_free_hibernation_slot() - Free a swap slot allocated for hibernation.
/// @entry: swap entry to free.
///
/// The caller must ensure the swap device is stable.
#[unsafe(no_mangle)]
extern "C" fn swap_free_hibernation_slot(entry: __SlateRecord_mm_types_h_288) {
let si: *mut swap_info_struct = __swap_entry_to_info(entry);
let offset: u64 = swp_offset(entry);
let ci: *mut swap_cluster_info = swap_cluster_lock(si, offset);
__swap_cluster_put_entry(ci, (offset % 256) as u32);
// A slot with a folio in the swap cache is freed when the folio
// leaves the cache, the same rule swap_put_entries_cluster() follows.
// Readahead can put a folio here, and freeing the slot now would
// leave that folio with no entry behind it.
if !swp_tb_is_folio(__swap_table_get(ci, (offset % 256) as u32)) {
__swap_cluster_free_entries(si, ci, (offset % 256) as u32, 1u32);
}
swap_cluster_unlock(ci);
// In theory readahead might add it to the swap cache by accident
__try_to_reclaim_swap(si, offset, 1u64);
}

fn __find_hibernation_swap_type(device: u32, offset: u64) -> i32 {
let mut r#type: i32 = 0;
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!(swap_lock) });
if device == 0 {
return 22i32.wrapping_neg();
}
while (r#type as u32) < unsafe { nr_swapfiles } {
let sis: *mut swap_info_struct = unsafe { swap_info.0[r#type as usize] };
if (unsafe { (*sis).flags }) & 2 != 0 {
if device == unsafe { (*(*sis).bdev).bd_dev } {
let se: *mut swap_extent = first_se(sis);
if (unsafe { (*se).start_block }) == offset {
return r#type;
}
}
}
r#type = r#type.wrapping_add(1);
}
19i32.wrapping_neg()
}

/// pin_hibernation_swap_type - Pin the swap device for hibernation
/// @device: Block device containing the resume image
/// @offset: Offset identifying the swap area
///
/// Locate the swap device for @device/@offset and mark it as pinned
/// for hibernation. While pinned, swapoff() is prevented.
///
/// Only one uswsusp context may pin a swap device at a time.
/// If already pinned, this function returns -EBUSY.
///
/// Return:
/// >= 0 on success (swap type).
/// -EINVAL if @device is invalid.
/// -ENODEV if the swap device is not found.
/// -EBUSY if the device is already pinned for hibernation.
#[unsafe(no_mangle)]
extern "C" fn pin_hibernation_swap_type(device: u32, offset: u64) -> i32 {
let mut r#type: i32 = 0;
spin_lock(unsafe { core::ptr::addr_of_mut!(swap_lock) });
r#type = __find_hibernation_swap_type(device, offset);
if r#type < 0 {
spin_unlock(unsafe { core::ptr::addr_of_mut!(swap_lock) });
return r#type;
}
let si: *mut swap_info_struct = swap_type_to_info(r#type);
let mut __v0: i64;
{
let __ret_warn_on: i32 = si.is_null() as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("910: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 910b - ., 3; .popsection\n# {0}", const 910 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_pin_hibernation_swap_type_1, sym __slate_asm_string_pin_hibernation_swap_type_2, const 2275 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("911: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 911b - ., 4; .popsection\n# {0}", const 911 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
}
if __v0 != 0 {
spin_unlock(unsafe { core::ptr::addr_of_mut!(swap_lock) });
return 19i32.wrapping_neg();
}
// hibernate_acquire() prevents concurrent hibernation sessions.
// This check additionally guards against double-pinning within
// the same session.
let mut __v5: i64;
let __ret_warn_on: i32 = ((unsafe { (*si).flags }) & 8192 != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("912: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 912b - ., 3; .popsection\n# {0}", const 912 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_pin_hibernation_swap_type_6, sym __slate_asm_string_pin_hibernation_swap_type_7, const 2285 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("913: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 913b - ., 4; .popsection\n# {0}", const 913 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v5 = (__ret_warn_on != 0) as i64;
if __v5 != 0 {
spin_unlock(unsafe { core::ptr::addr_of_mut!(swap_lock) });
return 16i32.wrapping_neg();
}
unsafe {
(*si).flags = (*si).flags | 8192;
}
spin_unlock(unsafe { core::ptr::addr_of_mut!(swap_lock) });
r#type
}

/// unpin_hibernation_swap_type - Unpin the swap device for hibernation
/// @type: Swap type previously returned by pin_hibernation_swap_type()
///
/// Clear the hibernation pin on the given swap device, allowing
/// swapoff() to proceed normally.
///
/// If @type does not refer to a valid swap device, this function
/// does nothing.
#[unsafe(no_mangle)]
extern "C" fn unpin_hibernation_swap_type(r#type: i32) {
spin_lock(unsafe { core::ptr::addr_of_mut!(swap_lock) });
let si: *mut swap_info_struct = swap_type_to_info(r#type);
if si.is_null() {
spin_unlock(unsafe { core::ptr::addr_of_mut!(swap_lock) });
return;
}
let __v2: u64 = (unsafe { (*si).flags }) & 18446744073709543423u64;
unsafe {
(*si).flags = __v2;
}
spin_unlock(unsafe { core::ptr::addr_of_mut!(swap_lock) });
}

/// find_hibernation_swap_type - Find swap type for hibernation
/// @device: Block device containing the resume image
/// @offset: Offset within the device identifying the swap area
///
/// Locate the swap device corresponding to @device and @offset.
///
/// Unlike pin_hibernation_swap_type(), this function only performs a
/// lookup and does not mark the swap device as pinned for hibernation.
///
/// This is safe in the sysfs-based hibernation path where user space
/// is already frozen and swapoff() cannot run concurrently.
///
/// Return:
/// A non-negative swap type on success.
/// -EINVAL if @device is invalid.
/// -ENODEV if no matching swap device is found.
#[unsafe(no_mangle)]
extern "C" fn find_hibernation_swap_type(device: u32, offset: u64) -> i32 {
let mut r#type: i32 = 0;
spin_lock(unsafe { core::ptr::addr_of_mut!(swap_lock) });
r#type = __find_hibernation_swap_type(device, offset);
spin_unlock(unsafe { core::ptr::addr_of_mut!(swap_lock) });
r#type
}

#[unsafe(no_mangle)]
extern "C" fn find_first_swap(device: *mut u32) -> i32 {
let mut r#type: i32 = 0;
spin_lock(unsafe { core::ptr::addr_of_mut!(swap_lock) });
while (r#type as u32) < unsafe { nr_swapfiles } {
let sis: *mut swap_info_struct = unsafe { swap_info.0[r#type as usize] };
if (unsafe { (*sis).flags }) & 2 != 0 {
unsafe {
*device = (*(*sis).bdev).bd_dev;
}
spin_unlock(unsafe { core::ptr::addr_of_mut!(swap_lock) });
return r#type;
}
r#type = r#type.wrapping_add(1);
}
spin_unlock(unsafe { core::ptr::addr_of_mut!(swap_lock) });
19i32.wrapping_neg()
}

/// Get the (PAGE_SIZE) block corresponding to given offset on the swapdev
/// corresponding to given index in swap_info (swap type).
#[unsafe(no_mangle)]
extern "C" fn swapdev_block(r#type: i32, offset: u64) -> u64 {
let si: *mut swap_info_struct = swap_type_to_info(r#type);
if si.is_null() || (unsafe { (*si).flags }) & 2 == 0 {
return 0;
}
let se: *mut swap_extent = offset_to_swap_extent(si, offset);
unsafe { (*se).start_block }.wrapping_add(offset.wrapping_sub(unsafe { (*se).start_page }))
}

/// Return either the total number of swap pages of given type, or the number
/// of free pages of that type (depending on @free)
///
/// This is needed for software suspend
#[unsafe(no_mangle)]
extern "C" fn count_swap_pages(r#type: i32, free: i32) -> u32 {
let mut n: u32 = 0;
spin_lock(unsafe { core::ptr::addr_of_mut!(swap_lock) });
if (r#type as u32) < unsafe { nr_swapfiles } {
let sis: *mut swap_info_struct = unsafe { swap_info.0[r#type as usize] };
spin_lock(unsafe { core::ptr::addr_of_mut!((*sis).lock) });
if (unsafe { (*sis).flags }) & 2 != 0 {
n = unsafe { (*sis).pages };
if free != 0 {
let __v0: i64 = swap_usage_in_pages(sis);
n = (n as i64).wrapping_sub(__v0) as u32;
}
}
spin_unlock(unsafe { core::ptr::addr_of_mut!((*sis).lock) });
}
spin_unlock(unsafe { core::ptr::addr_of_mut!(swap_lock) });
n
}

fn pte_same_as_swp(pte: __SlateRecord_pgtable_64_types_h_21, swp_pte: __SlateRecord_pgtable_64_types_h_21) -> i32 {
pte_same(pte_swp_clear_flags(pte), swp_pte)
}

/// No need to decide whether this PTE shares the swap entry with others,
/// just let do_wp_page work it out if a write is requested later - to
/// force COW, vm_page_prot omits write permission from any private vma.
fn unuse_pte(vma: *mut vm_area_struct, pmd: *mut __SlateRecord_pgtable_64_types_h_22, addr: u64, entry: __SlateRecord_mm_types_h_288, mut folio: *mut folio) -> i32 {
let mut __slate_compound_16: pgprot = unsafe { core::mem::zeroed() };
// See do_swap_page(): restore PAGE_NONE for RWP
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut rmap_flags: i32 = unsafe { core::mem::zeroed() };
let mut swp_entry: core::mem::MaybeUninit<__SlateRecord_mm_types_h_288> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut __SlateRecord_mm_types_h_288 = core::ptr::addr_of_mut!(swp_entry) as *mut __SlateRecord_mm_types_h_288;
let mut __v5: bool = unsafe { core::mem::zeroed() };
let mut __v6: bool = unsafe { core::mem::zeroed() };
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut hwpoisoned: bool = unsafe { core::mem::zeroed() };
let mut old_pte: core::mem::MaybeUninit<__SlateRecord_pgtable_64_types_h_21> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut __SlateRecord_pgtable_64_types_h_21 = core::ptr::addr_of_mut!(old_pte) as *mut __SlateRecord_pgtable_64_types_h_21;
let mut new_pte: core::mem::MaybeUninit<__SlateRecord_pgtable_64_types_h_21> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut __SlateRecord_pgtable_64_types_h_21 = core::ptr::addr_of_mut!(new_pte) as *mut __SlateRecord_pgtable_64_types_h_21;
let mut pte: core::mem::MaybeUninit<*mut __SlateRecord_pgtable_64_types_h_21> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut __SlateRecord_pgtable_64_types_h_21 = core::ptr::addr_of_mut!(pte) as *mut *mut __SlateRecord_pgtable_64_types_h_21;
let mut ptl: core::mem::MaybeUninit<*mut spinlock> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut spinlock = core::ptr::addr_of_mut!(ptl) as *mut *mut spinlock;
let mut swapcache: core::mem::MaybeUninit<*mut folio> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut *mut folio = core::ptr::addr_of_mut!(swapcache) as *mut *mut folio;
let mut page: core::mem::MaybeUninit<*mut page> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut *mut page = core::ptr::addr_of_mut!(page) as *mut *mut page;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
hwpoisoned = false;
ret = 1;
// If the folio is removed from swap cache by others, continue to
// unuse other PTEs. try_to_unuse may try again if we missed this one.
if !folio_matches_swap_entry(folio as *const folio, entry) {
return 0;
} else {
*__slate_slot_13 = folio;
folio = ksm_might_need_to_copy(folio, vma, addr);
if unlikely!(folio.is_null()) != 0 {
return 12i32.wrapping_neg();
} else {
if unlikely!(folio == (ERR_PTR(133i32.wrapping_neg() as i64) as *mut folio)) != 0 {
hwpoisoned = true;
folio = *__slate_slot_13;
}
*__slate_slot_14 = folio_file_page(folio, swp_offset(entry));
if PageHWPoison(*__slate_slot_14 as *const page) != 0 {
hwpoisoned = true;
}
*__slate_slot_11 = pte_offset_map_lock((*vma).vm_mm, pmd, addr, core::ptr::addr_of_mut!(*__slate_slot_12));
if (*__slate_slot_11).is_null() {
__v6 = true;
} else {
__v6 = pte_same_as_swp(ptep_get(*__slate_slot_11), swp_entry_to_pte(entry)) == 0;
}
if __v6 {
ret = 0;
} else {
*__slate_slot_9 = ptep_get(*__slate_slot_11);
if hwpoisoned {
__v5 = true;
} else {
__v5 = !folio_test_uptodate(folio as *const folio);
}
if __v5 {
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
dec_mm_counter((*vma).vm_mm, 2i32);
if hwpoisoned {
*__slate_slot_4 = make_hwpoison_entry(*__slate_slot_14);
} else {
*__slate_slot_4 = make_poisoned_swp_entry();
}
*__slate_slot_10 = swp_entry_to_pte(*__slate_slot_4);
ret = 0;
} else {
// Some architectures may have to restore extra metadata to the page
// when reading from swap. This metadata may be indexed by swap entry
// so this must be called before folio_put_swap().
arch_swap_restore(folio_swap(entry, folio as *const folio), folio);
dec_mm_counter((*vma).vm_mm, 2i32);
inc_mm_counter((*vma).vm_mm, 1i32);
folio_get(folio);
if folio == *__slate_slot_13 {
rmap_flags = 0;
// See do_swap_page(): writeback would be problematic.
// However, we do a folio_wait_writeback() just before this
// call and have the folio locked.
VM_BUG_ON_FOLIO!();
if pte_swp_exclusive(*__slate_slot_9) {
let __v2: i32 = rmap_flags;
rmap_flags = __v2 | (BIT!(0i32) as i32);
}
// We currently only expect small !anon folios, which are either
// fully exclusive or fully shared. If we ever get large folios
// here, we have to be careful.
if !folio_test_anon(folio as *const folio) {
VM_WARN_ON_ONCE!();
VM_WARN_ON_FOLIO!();
folio_add_new_anon_rmap(folio, vma, addr, rmap_flags);
} else {
folio_add_anon_rmap_pte!(folio, *__slate_slot_14, vma, addr, rmap_flags);
}
} else {
// ksm created a completely new copy
folio_add_new_anon_rmap(folio, vma, addr, BIT!(0i32) as i32);
folio_add_lru_vma(folio, vma);
}
*__slate_slot_10 = pte_mkold(mk_pte(*__slate_slot_14 as *const page, (*vma).vm_page_prot));
if pte_swp_soft_dirty(*__slate_slot_9) != 0 {
*__slate_slot_10 = pte_mksoft_dirty(*__slate_slot_10);
}
if pte_swp_uffd(*__slate_slot_9) != 0 {
*__slate_slot_10 = pte_mkuffd(*__slate_slot_10);
}
if pte_swp_uffd(*__slate_slot_9) != 0 {
__v0 = userfaultfd_rwp(vma);
} else {
__v0 = false;
}
if __v0 {
*__slate_slot_10 = pte_modify(*__slate_slot_10, *{
__slate_compound_16 = pgprot { pgprot: 0u64 | _AT!(u64, 1i32) << 5i32 | 0 | 0 | 0 | _AT!(u64, 1i32) << 8i32 };
core::ptr::addr_of_mut!(__slate_compound_16)
});
}
}
set_pte_at!(unsafe { (*vma).vm_mm }, addr, *__slate_slot_11, *__slate_slot_10);
folio_put_swap(*__slate_slot_13, folio_file_page(*__slate_slot_13, swp_offset(entry)));
}
if !(*__slate_slot_11).is_null() {
pte_unmap_unlock!(*__slate_slot_11, *__slate_slot_12);
}
if folio != *__slate_slot_13 {
folio_unlock(folio);
folio_put(folio);
}
return ret;
}
}
}
}

fn unuse_pte_range(vma: *mut vm_area_struct, pmd: *mut __SlateRecord_pgtable_64_types_h_22, mut addr: u64, end: u64, r#type: u32) -> i32 {
let mut pte: *mut __SlateRecord_pgtable_64_types_h_21 = core::ptr::null_mut::<__SlateRecord_pgtable_64_types_h_21>();
'__slate_break_2: loop {
'__slate_continue_2: {
let mut folio: *mut folio = unsafe { core::mem::zeroed() };
let mut swp_tb: u64 = 0;
let mut entry: __SlateRecord_mm_types_h_288 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(entry) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut ret: i32 = 0;
let mut ptent: __SlateRecord_pgtable_64_types_h_21 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(ptent) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let __v0: *mut __SlateRecord_pgtable_64_types_h_21 = pte;
pte = __v0.wrapping_offset(1 as isize);
if __v0.is_null() {
pte = pte_offset_map(pmd, addr);
if pte.is_null() {
break '__slate_break_2;
}
}
ptent = ptep_get_lockless(pte);
entry = softleaf_from_pte(ptent);
if softleaf_is_swap(entry) {
if swp_type(entry) == r#type {
pte_unmap(pte);
pte = core::ptr::null_mut::<__SlateRecord_pgtable_64_types_h_21>();
folio = unsafe { swap_cache_get_folio(entry) };
if folio.is_null() {
let mut vmf: vm_fault = {
let mut __t0: vm_fault = unsafe { core::mem::zeroed() };
__t0.__slate_anon_0 = {
let mut __t1: __SlateRecord_mm_h_754 = unsafe { core::mem::zeroed() };
__t1.vma = vma;
__t1.address = addr;
__t1.real_address = addr;
__t1
};
__t0.pmd = pmd;
__t0
};
folio = unsafe { swapin_readahead(entry, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32) | (BIT!(20i32) as u32) | (BIT!(1i32) as u32) | (BIT!(3i32) as u32) | 0, core::ptr::addr_of_mut!(vmf)) };
}
if folio.is_null() {
swp_tb = swap_table_get(__swap_entry_to_cluster(entry), swp_cluster_offset(entry));
if swp_tb_get_count(swp_tb) <= 0 {
break '__slate_continue_2;
}
return 12i32.wrapping_neg();
}
folio_lock(folio);
unsafe { folio_wait_writeback(folio) };
ret = unuse_pte(vma, pmd, addr, entry, folio);
if ret < 0 {
unsafe { folio_unlock(folio) };
folio_put(folio);
return ret;
}
folio_free_swap(folio);
unsafe { folio_unlock(folio) };
folio_put(folio);
}
}
}
let __v3: u64 = addr;
addr = __v3.wrapping_add(_AC!() << 12i32);
if addr == end {
break;
}
}
if !pte.is_null() {
pte_unmap(pte);
}
0
}

fn unuse_pmd_range(vma: *mut vm_area_struct, pud: *mut __SlateRecord_pgtable_types_h_368, mut addr: u64, end: u64, r#type: u32) -> i32 {
let mut pmd: *mut __SlateRecord_pgtable_64_types_h_22 = unsafe { core::mem::zeroed() };
let mut next: u64 = 0;
let mut ret: i32 = 0;
pmd = pmd_offset(pud, addr);
loop {
cond_resched__30!();
next = pmd_addr_end!(addr, end);
ret = unuse_pte_range(vma, pmd, addr, next, r#type);
if ret != 0 {
return ret;
}
let __v4: *mut __SlateRecord_pgtable_64_types_h_22 = pmd;
pmd = __v4.wrapping_offset(1 as isize);
addr = next;
if addr == end {
break;
}
}
0
}

fn unuse_pud_range(vma: *mut vm_area_struct, p4d: *mut __SlateRecord_pgtable_types_h_342, mut addr: u64, end: u64, r#type: u32) -> i32 {
let mut pud: *mut __SlateRecord_pgtable_types_h_368 = unsafe { core::mem::zeroed() };
let mut next: u64 = 0;
let mut ret: i32 = 0;
pud = pud_offset(p4d, addr);
loop {
next = pud_addr_end!(addr, end);
if pud_none_or_clear_bad(pud) == 0 {
ret = unuse_pmd_range(vma, pud, addr, next, r#type);
if ret != 0 {
return ret;
}
}
let __v3: *mut __SlateRecord_pgtable_types_h_368 = pud;
pud = __v3.wrapping_offset(1 as isize);
addr = next;
if addr == end {
break;
}
}
0
}

fn unuse_p4d_range(vma: *mut vm_area_struct, pgd: *mut __SlateRecord_pgtable_types_h_295, mut addr: u64, end: u64, r#type: u32) -> i32 {
let mut p4d: *mut __SlateRecord_pgtable_types_h_342 = unsafe { core::mem::zeroed() };
let mut next: u64 = 0;
let mut ret: i32 = 0;
p4d = p4d_offset(pgd, addr);
loop {
next = p4d_addr_end!(addr, end);
if p4d_none_or_clear_bad(p4d) == 0 {
ret = unuse_pud_range(vma, p4d, addr, next, r#type);
if ret != 0 {
return ret;
}
}
let __v3: *mut __SlateRecord_pgtable_types_h_342 = p4d;
p4d = __v3.wrapping_offset(1 as isize);
addr = next;
if addr == end {
break;
}
}
0
}

fn unuse_vma(vma: *mut vm_area_struct, r#type: u32) -> i32 {
let mut pgd: *mut __SlateRecord_pgtable_types_h_295 = unsafe { core::mem::zeroed() };
let mut addr: u64 = unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_start };
let end: u64 = unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_end };
let mut next: u64 = 0;
let mut ret: i32 = 0;
pgd = pgd_offset!(unsafe { (*vma).vm_mm }, addr);
loop {
next = pgd_addr_end!(addr, end);
if pgd_none_or_clear_bad(pgd) == 0 {
ret = unuse_p4d_range(vma, pgd, addr, next, r#type);
if ret != 0 {
return ret;
}
}
let __v3: *mut __SlateRecord_pgtable_types_h_295 = pgd;
pgd = __v3.wrapping_offset(1 as isize);
addr = next;
if addr == end {
break;
}
}
0
}

fn unuse_mm(mm: *mut mm_struct, r#type: u32) -> i32 {
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut vmi: core::mem::MaybeUninit<vma_iterator> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut vma_iterator = core::ptr::addr_of_mut!(vmi) as *mut vma_iterator;
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut vma: core::mem::MaybeUninit<*mut vm_area_struct> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut vm_area_struct = core::ptr::addr_of_mut!(vma) as *mut *mut vm_area_struct;
unsafe {
'__join_0: {
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
ret = 0;
core::ptr::write(__slate_slot_2, vma_iterator { mas: {
let mut __t0: ma_state = core::mem::zeroed();
__t0.tree = core::ptr::addr_of_mut!((*mm).__slate_anon_0.mm_mt);
__t0.index = 0u64;
__t0.node = core::ptr::null_mut::<maple_enode>();
__t0.status = 1u32;
__t0
} });
mmap_read_lock(mm);
if check_stable_address_space(mm) == 0 {
loop {
let __v0: *mut vm_area_struct = vma_next(&mut (*core::ptr::addr_of_mut!(*__slate_slot_2)));
*__slate_slot_4 = __v0;
if !__v0.is_null() {
if !unsafe { (*(*__slate_slot_4)).anon_vma }.is_null() {
__v1 = !is_vm_hugetlb_page(*__slate_slot_4 as *const vm_area_struct);
} else {
__v1 = false;
}
if __v1 {
ret = unuse_vma(*__slate_slot_4, r#type);
if ret != 0 {
break '__join_0;
}
}
cond_resched__69!();
} else {
break;
}
}
}
}
mmap_read_unlock(mm);
return ret;
}
}

/// Scan swap table from current position to next entry still in use.
/// Return 0 if there are no inuse entries after prev till end of
/// the map.
fn find_next_to_unuse(si: *mut swap_info_struct, prev: u32) -> u32 {
let mut i: u32 = 0;
let mut swp_tb: u64 = 0;
// No need for swap_lock here: we're just looking
// for whether an entry is in use, not modifying it; false
// hits are okay, and sys_swapoff() has already prevented new
// allocations from this area (while holding swap_lock).
i = prev.wrapping_add(1);
while i < unsafe { (*si).max } {
swp_tb = swap_table_get(__swap_offset_to_cluster(si, i as u64), i % 256);
let __v0: bool = !swp_tb_is_null(swp_tb) && !swp_tb_is_bad(swp_tb);
if __v0 {
break;
}
if i % 256 == 0 {
cond_resched__87!();
}
i = i.wrapping_add(1);
}
if i == unsafe { (*si).max } {
i = 0;
}
i
}

fn try_to_unuse(r#type: u32) -> i32 {
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut __v2: bool = unsafe { core::mem::zeroed() };
let mut __v4: bool = unsafe { core::mem::zeroed() };
let mut __v5: bool = unsafe { core::mem::zeroed() };
let mut i: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u32 = core::ptr::addr_of_mut!(i) as *mut u32;
let mut entry: core::mem::MaybeUninit<__SlateRecord_mm_types_h_288> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut __SlateRecord_mm_types_h_288 = core::ptr::addr_of_mut!(entry) as *mut __SlateRecord_mm_types_h_288;
let mut folio: core::mem::MaybeUninit<*mut folio> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut folio = core::ptr::addr_of_mut!(folio) as *mut *mut folio;
let mut si: *mut swap_info_struct = unsafe { core::mem::zeroed() };
let mut retval: i32 = unsafe { core::mem::zeroed() };
let mut p: core::mem::MaybeUninit<*mut list_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut list_head = core::ptr::addr_of_mut!(p) as *mut *mut list_head;
let mut mm: core::mem::MaybeUninit<*mut mm_struct> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut mm_struct = core::ptr::addr_of_mut!(mm) as *mut *mut mm_struct;
let mut prev_mm: core::mem::MaybeUninit<*mut mm_struct> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut *mut mm_struct = core::ptr::addr_of_mut!(prev_mm) as *mut *mut mm_struct;
unsafe {
'__join_1: {
let __slate_auto_bytes = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
retval = 0;
si = swap_info.0[r#type as usize];
let __slate_auto_bytes_3 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if swap_usage_in_pages(si) != 0 {
loop {
retval = shmem_unuse(r#type);
if retval != 0 {
return retval;
} else {
*__slate_slot_13 = core::ptr::addr_of_mut!(init_mm);
mmget(*__slate_slot_13);
spin_lock(core::ptr::addr_of_mut!(mmlist_lock));
*__slate_slot_11 = core::ptr::addr_of_mut!(init_mm.__slate_anon_0.mmlist);
loop {
if swap_usage_in_pages(si) != 0 {
__v5 = signal_pending(get_current()) == 0;
} else {
__v5 = false;
}
if __v5 {
let __v3: *mut list_head = (*(*__slate_slot_11)).next;
*__slate_slot_11 = __v3;
__v4 = __v3 != core::ptr::addr_of_mut!(init_mm.__slate_anon_0.mmlist);
} else {
__v4 = false;
}
if __v4 {
*__slate_slot_12 = list_entry__26!(*__slate_slot_11, mm_struct);
if mmget_not_zero(*__slate_slot_12) {
spin_unlock(core::ptr::addr_of_mut!(mmlist_lock));
mmput(*__slate_slot_13);
*__slate_slot_13 = *__slate_slot_12;
retval = unuse_mm(*__slate_slot_12, r#type);
if retval != 0 {
mmput(*__slate_slot_13);
return retval;
} else {
// Make sure that we aren't completely killing
// interactive performance.
cond_resched__53!();
spin_lock(core::ptr::addr_of_mut!(mmlist_lock));
}
}
} else {
break;
}
}
spin_unlock(core::ptr::addr_of_mut!(mmlist_lock));
mmput(*__slate_slot_13);
*__slate_slot_6 = 0;
loop {
if swap_usage_in_pages(si) != 0 {
__v2 = signal_pending(get_current()) == 0;
} else {
__v2 = false;
}
if __v2 {
let __v0: u32 = find_next_to_unuse(si, *__slate_slot_6);
*__slate_slot_6 = __v0;
__v1 = __v0 != 0;
} else {
__v1 = false;
}
if __v1 {
*__slate_slot_7 = swp_entry(r#type as u64, *__slate_slot_6 as u64);
*__slate_slot_8 = swap_cache_get_folio(*__slate_slot_7);
if !(*__slate_slot_8).is_null() {
// It is conceivable that a racing task removed this folio from
// swap cache just before we acquired the page lock. The folio
// might even be back in swap cache on another swap area. But
// that is okay, folio_free_swap() only removes stale folios.
folio_lock(*__slate_slot_8);
folio_wait_writeback(*__slate_slot_8);
folio_free_swap(*__slate_slot_8);
folio_unlock(*__slate_slot_8);
folio_put(*__slate_slot_8);
}
} else {
break;
}
}
// Lets check again to see if there are still swap entries in the map.
// If yes, we would need to do retry the unuse logic again.
// Under global memory pressure, swap entries can be reinserted back
// into process space after the mmlist loop above passes over them.
//
// Limit the number of retries? No: when mmget_not_zero()
// above fails, that mm is likely to be freeing swap from
// exit_mmap(), which proceeds at its own independent pace;
// and even shmem_writeout() could have been preempted after
// folio_alloc_swap(), temporarily hiding that swap.  It's easy
// and robust (though cpu-intensive) just to keep retrying.
if swap_usage_in_pages(si) != 0 {
if signal_pending(get_current()) != 0 {
break;
}
} else {
break '__join_1;
}
}
}
return 4i32.wrapping_neg();
}
}
loop {
kcsan_mb!();
core::arch::asm!("lock addl $0,-4(%rsp)", options(nostack, att_syntax));
if true {
break;
}
}
// Make sure that further cleanups after try_to_unuse() returns happen
// after swap_range_free() reduces si->inuse_pages to 0.
return 0;
}
}

/// After a successful try_to_unuse, if no swap is now in use, we know
/// we can empty the mmlist.  swap_lock must be held on entry and exit.
/// Note that mmlist_lock nests inside swap_lock, and an mm must be
/// added to the mmlist just after page_duplicate - before would be racy.
fn drain_mmlist() {
let mut p: *mut list_head = unsafe { core::mem::zeroed() };
let mut next: *mut list_head = unsafe { core::mem::zeroed() };
let mut r#type: u32 = 0;
while r#type < unsafe { nr_swapfiles } {
if swap_usage_in_pages(unsafe { swap_info.0[r#type as usize] }) != 0 {
return;
}
r#type = r#type.wrapping_add(1);
}
spin_lock(unsafe { core::ptr::addr_of_mut!(mmlist_lock) });
p = unsafe { (*core::ptr::addr_of_mut!(init_mm.__slate_anon_0.mmlist)).next };
next = unsafe { (*p).next };
while list_is_head(p as *const list_head, (unsafe { core::ptr::addr_of_mut!(init_mm.__slate_anon_0.mmlist) }) as *const list_head) == 0 {
list_del_init(p);
p = next;
next = unsafe { (*p).next };
}
spin_unlock(unsafe { core::ptr::addr_of_mut!(mmlist_lock) });
}

// Free all of a swapdev's extent information
fn destroy_swap_extents(sis: *mut swap_info_struct, swap_file: *mut file) {
loop {
let mut __v1: *mut rb_node;
loop {
if !(__native_word__13!() || ((const { core::mem::size_of::<*mut rb_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_914() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*sis).swap_extent_root)).rb_node) as *const *mut rb_node) };
if __v1.is_null() {
break;
}
let rb: *mut rb_node = unsafe { (*sis).swap_extent_root.rb_node };
let se: *mut swap_extent = rb_entry!(rb, swap_extent);
unsafe { rb_erase(rb, core::ptr::addr_of_mut!((*sis).swap_extent_root)) };
unsafe { kfree(se as *const ()) };
}
if (unsafe { (*sis).flags }) & 128 != 0 {
let mapping: *mut address_space = unsafe { (*swap_file).f_mapping };
let __v6: u64 = (unsafe { (*sis).flags }) & 18446744073709551487u64;
unsafe {
(*sis).flags = __v6;
}
if (unsafe { (*(*mapping).a_ops).swap_deactivate }) != None {
unsafe { unsafe { (*(*mapping).a_ops).swap_deactivate }.unwrap()(swap_file) };
}
}
}

/// Add a block range (and the corresponding page range) into this swapdev's
/// extent tree.
///
/// This function rather assumes that it is called in ascending page order.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.swapfile.add_swap_extent")]
extern "C" fn add_swap_extent(sis: *mut swap_info_struct, start_page: u64, nr_pages: u64, start_block: u64) -> i32 {
let mut __slate_compound_13: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_15: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut link: *mut *mut rb_node = unsafe { core::ptr::addr_of_mut!((*sis).swap_extent_root.rb_node) };
let mut parent: *mut rb_node = core::ptr::null_mut::<rb_node>();
let mut new_se: *mut swap_extent = unsafe { core::mem::zeroed() };
// place the new node at the right most since the
// function is called in ascending page order.
while !unsafe { *link }.is_null() {
parent = unsafe { *link };
link = unsafe { core::ptr::addr_of_mut!((*parent).rb_right) };
}
if !parent.is_null() {
let se: *mut swap_extent = rb_entry!(parent, swap_extent);
loop {
if unlikely!(unsafe { (*se).start_page }.wrapping_add(unsafe { (*se).nr_pages }) != start_page) != 0 {
loop {
unsafe {
core::arch::asm!("915: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 915b - ., 3; .popsection\n# {0}", const 915 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_add_swap_extent_2, sym __slate_asm_string_add_swap_extent_3, const 2906 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
if unsafe { (*se).start_block }.wrapping_add(unsafe { (*se).nr_pages }) == start_block {
// Merge it
let __v8: u64 = unsafe { (*se).nr_pages };
unsafe {
(*se).nr_pages = __v8.wrapping_add(nr_pages);
}
return 0;
}
}
// No merge, insert a new extent.
let mut __v10: *mut swap_extent;
let __obj_size: u64 = size_mul(core::mem::size_of::<swap_extent>() as u64, 1u64);
let mut __v11: *mut ();
let mut __v12: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_13 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_13)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_15 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_15)
} });
}
__v12 = _res;
__v11 = __v12;
__v10 = __v11 as *mut swap_extent;
new_se = __v10;
if new_se.is_null() {
return 12i32.wrapping_neg();
}
unsafe {
(*new_se).start_page = start_page;
}
unsafe {
(*new_se).nr_pages = nr_pages;
}
unsafe {
(*new_se).start_block = start_block;
}
rb_link_node(unsafe { core::ptr::addr_of_mut!((*new_se).rb_node) }, parent, link);
unsafe { rb_insert_color(core::ptr::addr_of_mut!((*new_se).rb_node), core::ptr::addr_of_mut!((*sis).swap_extent_root)) };
1
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_add_swap_extent_916: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut swap_info_struct, u64, u64, u64) -> i32>, *mut ()>(Some(add_swap_extent)) };

/// A `swap extent' is a simple thing which maps a contiguous range of pages
/// onto a contiguous range of disk blocks.  A rbtree of swap extents is
/// built at swapon time and is then used at swap_writepage/swap_read_folio
/// time for locating where on disk a page belongs.
///
/// If the swapfile is an S_ISBLK block device, a single extent is installed.
/// This is done so that the main operating code can treat S_ISBLK and S_ISREG
/// swap files identically.
///
/// Whether the swapdev is an S_ISREG file or an S_ISBLK blockdev, the swap
/// extent rbtree operates in PAGE_SIZE disk blocks.  Both S_ISREG and S_ISBLK
/// swapfiles are handled *identically* after swapon time.
///
/// For S_ISREG swapfiles, setup_swap_extents() will walk all the file's blocks
/// and will parse them into a rbtree, in PAGE_SIZE chunks.  If some stray
/// blocks are found which do not fall within the PAGE_SIZE alignment
/// requirements, they are simply tossed out - we will never use those blocks
/// for swapping.
///
/// For all swap devices we set S_SWAPFILE across the life of the swapon.  This
/// prevents users from writing to the swap device, which will corrupt memory.
///
/// The amount of disk space which a single swap extent represents varies.
/// Typically it is in the 1-4 megabyte range.  So we can have hundreds of
/// extents in the rbtree. - akpm.
fn setup_swap_extents(sis: *mut swap_info_struct, swap_file: *mut file, span: *mut u64) -> i32 {
let mapping: *mut address_space = unsafe { (*swap_file).f_mapping };
let inode: *mut inode = unsafe { (*mapping).host };
let mut ret: i32 = unsafe { sio_pool_init() };
if ret != 0 {
return ret;
}
unsafe {
(*sis).ops = core::ptr::addr_of!(swap_bdev_ops);
}
if S_ISBLK!(unsafe { (*inode).i_mode }) {
ret = add_swap_extent(sis, 0u64, (unsafe { (*sis).max }) as u64, 0u64);
unsafe {
*span = (*sis).pages as u64;
}
return ret;
}
if (unsafe { (*(*mapping).a_ops).swap_activate }) != None {
ret = unsafe { unsafe { (*(*mapping).a_ops).swap_activate }.unwrap()(sis, swap_file, span) };
if ret < 0 {
return ret;
}
unsafe {
(*sis).flags = (*sis).flags | 128;
}
return ret;
}
unsafe { generic_swapfile_activate(sis, swap_file, span) }
}

fn _enable_swap_info(si: *mut swap_info_struct) {
atomic_long_add((unsafe { (*si).pages }) as i64, unsafe { core::ptr::addr_of_mut!(nr_swap_pages) });
let __v0: i64 = unsafe { total_swap_pages };
unsafe {
total_swap_pages = __v0.wrapping_add((*si).pages as i64);
}
loop {
if unlikely!(raw_spin_is_locked!(unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(swap_lock)).__slate_anon_0.rlock) }) == 0) != 0 {
loop {
unsafe {
core::arch::asm!("917: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 917b - ., 3; .popsection\n# {0}", const 917 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string__enable_swap_info_2, sym __slate_asm_string__enable_swap_info_3, const 2990 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
unsafe { plist_add(core::ptr::addr_of_mut!((*si).list), core::ptr::addr_of_mut!(swap_active_head)) };
// Add back to available list
add_to_avail_list(si, true);
}

/// Called after the swap device is ready, resurrect its percpu ref, it's now
/// safe to reference it. Add it to the list to expose it to the allocator.
fn enable_swap_info(si: *mut swap_info_struct) {
unsafe { percpu_ref_resurrect(core::ptr::addr_of_mut!((*si).users)) };
spin_lock(unsafe { core::ptr::addr_of_mut!(swap_lock) });
spin_lock(unsafe { core::ptr::addr_of_mut!((*si).lock) });
_enable_swap_info(si);
spin_unlock(unsafe { core::ptr::addr_of_mut!((*si).lock) });
spin_unlock(unsafe { core::ptr::addr_of_mut!(swap_lock) });
}

fn reinsert_swap_info(si: *mut swap_info_struct) {
spin_lock(unsafe { core::ptr::addr_of_mut!(swap_lock) });
spin_lock(unsafe { core::ptr::addr_of_mut!((*si).lock) });
_enable_swap_info(si);
spin_unlock(unsafe { core::ptr::addr_of_mut!((*si).lock) });
spin_unlock(unsafe { core::ptr::addr_of_mut!(swap_lock) });
}

/// Called after clearing SWP_WRITEOK, ensures cluster_alloc_range
/// see the updated flags, so there will be no more allocations.
fn wait_for_allocation(si: *mut swap_info_struct) {
let mut offset: u64 = 0;
let end: u64 = ALIGN__2!(unsafe { (*si).max }, 256i32) as u64;
loop {
if unlikely__4!((unsafe { (*si).flags }) & 2) != 0 {
loop {
unsafe {
core::arch::asm!("918: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 918b - ., 3; .popsection\n# {0}", const 918 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_wait_for_allocation_0, sym __slate_asm_string_wait_for_allocation_1, const 3031 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
while offset < end {
let ci: *mut swap_cluster_info = swap_cluster_lock(si, offset);
swap_cluster_unlock(ci);
offset = offset.wrapping_add(256);
}
}

fn free_swap_cluster_info(cluster_info: *mut swap_cluster_info, maxpages: u64) {
let mut ci: *mut swap_cluster_info = unsafe { core::mem::zeroed() };
let nr_clusters: i32 = (maxpages.wrapping_add(256).wrapping_sub(1) / 256) as i32;
if cluster_info.is_null() {
return;
}
for i in 0..nr_clusters {
ci = cluster_info.wrapping_offset(i as isize);
// Cluster with bad marks count will have a remaining table
spin_lock(unsafe { core::ptr::addr_of_mut!((*ci).lock) });
if cluster_table_is_alloced(ci) {
swap_cluster_assert_empty(ci, 0u32, 256u32, true);
swap_cluster_free_table(ci);
}
spin_unlock(unsafe { core::ptr::addr_of_mut!((*ci).lock) });
}
unsafe { kvfree(cluster_info as *const ()) };
}

/// Called after swap device's reference count is dead, so
/// neither scan nor allocation will use it.
fn flush_percpu_swap_cluster(si: *mut swap_info_struct) {
let mut cpu: i32 = 0;
loop {
cpu = find_next_bit(cpumask_bits!((unsafe { core::ptr::addr_of_mut!(__cpu_possible_mask) }) as *const cpumask), 64u64, cpu as u64) as i32;
if (cpu as u32) >= 64 {
break;
}
let pcp_si: *mut *mut swap_info_struct = per_cpu_ptr__15!(unsafe { core::ptr::addr_of_mut!(percpu_swap_cluster.si) as *mut *mut swap_info_struct }, cpu);
// Invalidate the percpu swap cluster cache, si->users
// is dead, so no new user will point to it, just flush
// any existing user.
for i in 0i32..1i32 {
let mut __v3: *mut swap_info_struct;
let __ai_ptr: *mut *mut swap_info_struct = pcp_si.wrapping_offset(i as isize);
kcsan_mb!();
instrument_atomic_read_write(__ai_ptr as *const (), core::mem::size_of::<*mut swap_info_struct>() as u64);
let mut __v5: *mut swap_info_struct;
let mut __ret: *mut swap_info_struct = unsafe { core::mem::zeroed() };
let mut __old: *mut swap_info_struct = si;
let mut __new: *mut swap_info_struct = core::ptr::null_mut::<swap_info_struct>();
match (const { core::mem::size_of::<*mut swap_info_struct>() }) as u64 {
1 => {
let mut __ptr: *mut u8 = __ai_ptr as *mut u8;
unsafe {
let __t0: u64;
core::arch::asm!("lock cmpxchgb {1:r}, ({0:r})", in(reg) __ptr, in(reg) __new as u64, inlateout("ax") __old as u64 => __t0, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, *mut swap_info_struct>(__t0 as u64) };
}
}
2 => {
let mut __ptr: *mut u16 = __ai_ptr as *mut u16;
unsafe {
let __t1: u64;
core::arch::asm!("lock cmpxchgw {1:r}, ({0:r})", in(reg) __ptr, in(reg) __new as u64, inlateout("ax") __old as u64 => __t1, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, *mut swap_info_struct>(__t1 as u64) };
}
}
4 => {
let mut __ptr: *mut u32 = __ai_ptr as *mut u32;
unsafe {
let __t2: u64;
core::arch::asm!("lock cmpxchgl {1:r}, ({0:r})", in(reg) __ptr, in(reg) __new as u64, inlateout("ax") __old as u64 => __t2, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, *mut swap_info_struct>(__t2 as u64) };
}
}
8 => {
let mut __ptr: *mut u64 = __ai_ptr as *mut u64;
unsafe {
let __t3: u64;
core::arch::asm!("lock cmpxchgq {1:r}, ({0:r})", in(reg) __ptr, in(reg) __new as u64, inlateout("ax") __old as u64 => __t3, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, *mut swap_info_struct>(__t3 as u64) };
}
}
_ => {
unsafe { __cmpxchg_wrong_size() };
}
}
__v5 = __ret;
__v3 = __v5;
}
cpu = cpu.wrapping_add(1);
}
}

#[unsafe(no_mangle)]
extern "C" fn __x64_sys_swapoff(regs: *const pt_regs) -> i64 {
__se_sys_swapoff((unsafe { (*regs).di }) as i64)
}

#[unsafe(no_mangle)]
extern "C" fn __ia32_sys_swapoff(regs: *const pt_regs) -> i64 {
__se_sys_swapoff(((unsafe { (*regs).bx }) as u32) as i64)
}

fn __se_sys_swapoff(specialfile: i64) -> i64 {
let ret: i64 = __do_sys_swapoff(specialfile as *const u8);
while false {
}
ret
}

fn __do_sys_swapoff(specialfile: *const u8) -> i64 {
let mut pathname: core::mem::MaybeUninit<*mut filename> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut filename = core::ptr::addr_of_mut!(pathname) as *mut *mut filename;
let mut found: i32 = unsafe { core::mem::zeroed() };
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut maxpages: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut u32 = core::ptr::addr_of_mut!(maxpages) as *mut u32;
let mut inode: core::mem::MaybeUninit<*mut inode> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut inode = core::ptr::addr_of_mut!(inode) as *mut *mut inode;
let mut mapping: core::mem::MaybeUninit<*mut address_space> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut address_space = core::ptr::addr_of_mut!(mapping) as *mut *mut address_space;
let mut victim: core::mem::MaybeUninit<*mut file> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut *mut file = core::ptr::addr_of_mut!(victim) as *mut *mut file;
let mut swap_file: core::mem::MaybeUninit<*mut file> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut *mut file = core::ptr::addr_of_mut!(swap_file) as *mut *mut file;
let mut cluster_info: core::mem::MaybeUninit<*mut swap_cluster_info> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut *mut swap_cluster_info = core::ptr::addr_of_mut!(cluster_info) as *mut *mut swap_cluster_info;
let mut p: *mut swap_info_struct = unsafe { core::mem::zeroed() };
let __slate_cleanup_ptr_7 = __slate_slot_7;
let __slate_cleanup_armed_7 = core::cell::Cell::new(false);
let __slate_cleanup_7 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_7.get() {
unsafe { class_filename_destructor(__slate_cleanup_ptr_7 as *mut _) };
}
});
unsafe {
p = core::ptr::null_mut::<swap_info_struct>();
let __slate_auto_bytes = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
found = 0;
if !capable(21i32) {
return 1i32.wrapping_neg() as i64;
} else {
'__join_22: {
loop {
if unlikely!(unsafe { (*get_current()).mm }.is_null()) != 0 {
break '__join_22;
} else {
if true {
break;
}
}
}
core::ptr::write(__slate_slot_7, class_filename_constructor(specialfile));
__slate_cleanup_armed_7.set(true);
*__slate_slot_13 = file_open_name(*__slate_slot_7, 32770i32, 0u16);
if IS_ERR(*__slate_slot_13 as *const ()) {
return PTR_ERR(*__slate_slot_13 as *const ());
} else {
*__slate_slot_12 = (*(*__slate_slot_13)).f_mapping;
spin_lock(core::ptr::addr_of_mut!(swap_lock));
p = ((((*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(swap_active_head)).node_list)).next as *mut ()) as *const u8).wrapping_offset(-(offsetof__7!() as isize)) as *mut ()) as *mut swap_info_struct;
'__join_11: {
loop {
if list_is_head(core::ptr::addr_of_mut!((*p).list.node_list) as *const list_head, core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(swap_active_head)).node_list) as *const list_head) == 0 {
if (*p).flags & 2 != 0 {
if (*(*p).swap_file).f_mapping == *__slate_slot_12 {
break;
}
}
p = ((((*p).list.node_list.next as *mut ()) as *const u8).wrapping_offset(-(offsetof__7!() as isize)) as *mut ()) as *mut swap_info_struct;
} else {
break '__join_11;
}
}
found = 1;
}
if found == 0 {
*__slate_slot_9 = 22i32.wrapping_neg();
spin_unlock(core::ptr::addr_of_mut!(swap_lock));
} else {
// Refuse swapoff while the device is pinned for hibernation
if (*p).flags & 8192 != 0 {
*__slate_slot_9 = 16i32.wrapping_neg();
spin_unlock(core::ptr::addr_of_mut!(swap_lock));
} else {
if security_vm_enough_memory_mm((*get_current()).mm, (*p).pages as i64) == 0 {
vm_unacct_memory((*p).pages as i64);
spin_lock(core::ptr::addr_of_mut!((*p).lock));
del_from_avail_list(p, true);
plist_del(core::ptr::addr_of_mut!((*p).list), core::ptr::addr_of_mut!(swap_active_head));
atomic_long_sub((*p).pages as i64, core::ptr::addr_of_mut!(nr_swap_pages));
let __v4: i64 = total_swap_pages;
total_swap_pages = __v4.wrapping_sub((*p).pages as i64);
spin_unlock(core::ptr::addr_of_mut!((*p).lock));
spin_unlock(core::ptr::addr_of_mut!(swap_lock));
wait_for_allocation(p);
set_current_oom_origin();
*__slate_slot_9 = try_to_unuse((*p).r#type as u32);
clear_current_oom_origin();
if *__slate_slot_9 != 0 {
// re-insert swap space back into swap_list
reinsert_swap_info(p);
} else {
// Wait for swap operations protected by get/put_swap_device()
// to complete.  Because of synchronize_rcu() here, all swap
// operations protected by RCU reader side lock (including any
// spinlock) will be waited too.  This makes it easy to
// prevent folio_test_swapcache() and the following swap cache
// operations from racing with swapoff.
percpu_ref_kill(core::ptr::addr_of_mut!((*p).users));
synchronize_rcu();
wait_for_completion(core::ptr::addr_of_mut!((*p).comp));
flush_work(core::ptr::addr_of_mut!((*p).discard_work));
flush_work(core::ptr::addr_of_mut!((*p).reclaim_work));
flush_percpu_swap_cluster(p);
destroy_swap_extents(p, (*p).swap_file);
if (*p).flags & 16 == 0 {
atomic_dec(core::ptr::addr_of_mut!(nr_rotate_swap));
}
mutex_lock(core::ptr::addr_of_mut!(swapon_mutex));
spin_lock(core::ptr::addr_of_mut!(swap_lock));
spin_lock(core::ptr::addr_of_mut!((*p).lock));
drain_mmlist();
*__slate_slot_14 = (*p).swap_file;
(*p).swap_file = core::ptr::null_mut::<file>();
*__slate_slot_10 = (*p).max;
*__slate_slot_15 = (*p).cluster_info;
(*p).max = 0;
(*p).cluster_info = core::ptr::null_mut::<swap_cluster_info>();
spin_unlock(core::ptr::addr_of_mut!((*p).lock));
spin_unlock(core::ptr::addr_of_mut!(swap_lock));
arch_swap_invalidate_area((*p).r#type as i32);
zswap_swapoff((*p).r#type as i32);
mutex_unlock(core::ptr::addr_of_mut!(swapon_mutex));
kfree((*p).global_cluster as *const ());
(*p).global_cluster = core::ptr::null_mut::<swap_sequential_cluster>();
free_swap_cluster_info(*__slate_slot_15, *__slate_slot_10 as u64);
*__slate_slot_11 = (*(*__slate_slot_12)).host;
inode_lock(*__slate_slot_11);
let __v2: *mut inode = *__slate_slot_11;
let __v0: u32 = (*__v2).i_flags & 4294967039u32;
(*__v2).i_flags = __v0;
inode_unlock(*__slate_slot_11);
filp_close(*__slate_slot_14, core::ptr::null_mut::<()>());
// Clear the SWP_USED flag after all resources are freed so that swapon
// can reuse this swap_info in alloc_swap_info() safely.  It is ok to
// not hold p->lock after we cleared its SWP_WRITEOK.
spin_lock(core::ptr::addr_of_mut!(swap_lock));
(*p).flags = 0;
spin_unlock(core::ptr::addr_of_mut!(swap_lock));
*__slate_slot_9 = 0;
atomic_inc(core::ptr::addr_of_mut!(proc_poll_event));
wake_up_interruptible!(core::ptr::addr_of_mut!(proc_poll_wait));
}
} else {
*__slate_slot_9 = 12i32.wrapping_neg();
spin_unlock(core::ptr::addr_of_mut!(swap_lock));
}
}
}
filp_close(*__slate_slot_13, core::ptr::null_mut::<()>());
return *__slate_slot_9 as i64;
}
}
core::arch::asm!("919: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 919b - ., 3; .popsection\n# {0}", const 919 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string___do_sys_swapoff_17, sym __slate_asm_string___do_sys_swapoff_18, const 3095 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
}
unsafe { core::mem::zeroed() }
}

#[unsafe(link_section = ".text.slate_distinct.swapfile.swaps_poll")]
extern "C" fn swaps_poll(file: *mut file, wait: *mut poll_table_struct) -> u32 {
let seq: *mut seq_file = (unsafe { (*file).private_data }) as *mut seq_file;
poll_wait(file, unsafe { core::ptr::addr_of_mut!(proc_poll_wait) }, wait);
if (unsafe { (*seq).poll_event }) != atomic_read((unsafe { core::ptr::addr_of_mut!(proc_poll_event) }) as *const __SlateRecord_types_h_190) {
unsafe {
(*seq).poll_event = atomic_read(core::ptr::addr_of_mut!(proc_poll_event) as *const __SlateRecord_types_h_190);
}
return 1 | 64 | 8 | 2;
}
1 | 64
}

/// iterator
#[unsafe(link_section = ".text.slate_distinct.swapfile.swap_start")]
extern "C" fn swap_start(swap: *mut seq_file, pos: *mut i64) -> *mut () {
let mut r#type: i32 = 0;
let mut l: i64 = unsafe { *pos };
unsafe { mutex_lock(core::ptr::addr_of_mut!(swapon_mutex)) };
if l == 0 {
return 1i32 as *mut ();
}
loop {
let __v3: *mut swap_info_struct = swap_type_to_info(r#type);
let si: *mut swap_info_struct = __v3;
if __v3.is_null() {
break;
}
if !unsafe { (*si).swap_file }.is_null() {
let __v2: i64 = l.wrapping_sub(1);
l = __v2;
if __v2 == 0 {
return si as *mut ();
}
}
r#type = r#type.wrapping_add(1);
}
core::ptr::null_mut::<()>()
}

#[unsafe(link_section = ".text.slate_distinct.swapfile.swap_next")]
extern "C" fn swap_next(swap: *mut seq_file, v: *mut (), __v0: *mut i64) -> *mut () {
let mut si: *mut swap_info_struct = v as *mut swap_info_struct;
let mut r#type: i32 = if v == (1i32 as *mut ()) {
0
} else {
((unsafe { (*si).r#type }) as i32).wrapping_add(1)
};
let __v1: i64 = unsafe { *__v0 };
unsafe {
*__v0 = __v1.wrapping_add(1);
}
loop {
let __v4: *mut swap_info_struct = swap_type_to_info(r#type);
si = __v4;
if __v4.is_null() {
break;
}
if !unsafe { (*si).swap_file }.is_null() {
return si as *mut ();
}
r#type = r#type.wrapping_add(1);
}
core::ptr::null_mut::<()>()
}

#[unsafe(link_section = ".text.slate_distinct.swapfile.swap_stop")]
extern "C" fn swap_stop(swap: *mut seq_file, v: *mut ()) {
unsafe { mutex_unlock(core::ptr::addr_of_mut!(swapon_mutex)) };
}

#[unsafe(link_section = ".text.slate_distinct.swapfile.swap_show")]
extern "C" fn swap_show(swap: *mut seq_file, v: *mut ()) -> i32 {
let si: *mut swap_info_struct = v as *mut swap_info_struct;
let mut len: i32 = 0;
let mut bytes: u64 = 0;
let mut inuse: u64 = 0;
if si == (1i32 as *mut swap_info_struct) {
seq_puts(swap, c"Filename\t\t\t\tType\t\tSize\t\tUsed\t\tPriority\n".as_ptr() as *const u8);
return 0;
}
bytes = K!(unsafe { (*si).pages }) as u64;
inuse = K!(swap_usage_in_pages(si)) as u64;
let file: *mut file = unsafe { (*si).swap_file };
len = unsafe { seq_file_path(swap, file, c" \t\n\\".as_ptr() as *const u8) };
unsafe { seq_printf(swap, c"%*s%s\t%lu\t%s%lu\t%s%d\n".as_ptr() as *const u8, if len < 40 { 40i32.wrapping_sub(len) } else { 1i32 }, c" ".as_ptr() as *mut u8, if S_ISBLK!(unsafe { (*file_inode(file as *const file)).i_mode }) { c"partition".as_ptr() as *mut u8 } else { c"file\t".as_ptr() as *mut u8 }, bytes, if bytes < 10000000 { c"\t".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, inuse, if inuse < 10000000 { c"\t".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, (*si).prio as i32) };
0
}

static swaps_op: __SlateConst<seq_operations> = __SlateConst(seq_operations { start: Some(swap_start), stop: Some(swap_stop), next: Some(swap_next), show: Some(swap_show) });

#[unsafe(link_section = ".text.slate_distinct.swapfile.swaps_open")]
extern "C" fn swaps_open(inode: *mut inode, file: *mut file) -> i32 {
let ret: i32 = unsafe { seq_open(file, core::ptr::addr_of!(swaps_op.0)) };
if ret != 0 {
return ret;
}
let seq: *mut seq_file = (unsafe { (*file).private_data }) as *mut seq_file;
unsafe {
(*seq).poll_event = atomic_read(core::ptr::addr_of_mut!(proc_poll_event) as *const __SlateRecord_types_h_190);
}
0
}

static swaps_proc_ops: __SlateConst<proc_ops> = __SlateConst({
let mut __t0: proc_ops = unsafe { core::mem::zeroed() };
__t0.proc_flags = 1u32;
__t0.proc_open = Some(swaps_open);
__t0.proc_read = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut file, *mut u8, u64, *mut i64) -> i64>>(seq_read as *const ()) };
__t0.proc_lseek = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut file, i64, i32) -> i64>>(seq_lseek as *const ()) };
__t0.proc_release = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut inode, *mut file) -> i32>>(seq_release as *const ()) };
__t0.proc_poll = Some(swaps_poll);
__t0
});

#[unsafe(link_section = ".init.text")]
extern "C" fn procswaps_init() -> i32 {
unsafe { proc_create(c"swaps".as_ptr() as *const u8, 0u16, core::ptr::null_mut::<proc_dir_entry>(), core::ptr::addr_of!(swaps_proc_ops.0)) };
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_procswaps_init_921: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i32>, *mut ()>(Some(procswaps_init)) };

#[unsafe(link_section = ".init.text")]
extern "C" fn max_swapfiles_check() -> i32 {
loop {
if false {
unsafe { __compiletime_assert_922() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_max_swapfiles_check_924: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i32>, *mut ()>(Some(max_swapfiles_check)) };

fn alloc_swap_info() -> *mut swap_info_struct {
let mut __slate_compound_18: rb_root = unsafe { core::mem::zeroed() };
let mut __slate_compound_19: spinlock = unsafe { core::mem::zeroed() };
let mut p: *mut swap_info_struct = unsafe { core::mem::zeroed() };
let mut defer: *mut swap_info_struct = core::ptr::null_mut::<swap_info_struct>();
let mut r#type: u32 = 0;
let mut __v0: *mut swap_info_struct;
let __obj_size: u64 = size_mul(core::mem::size_of::<swap_info_struct>() as u64, 1u64);
__v0 = kvmalloc_node!(__obj_size, 3520u32, 1i32.wrapping_neg()) as *mut swap_info_struct;
p = __v0;
if p.is_null() {
return ERR_PTR(12i32.wrapping_neg() as i64) as *mut swap_info_struct;
}
if (unsafe { percpu_ref_init(core::ptr::addr_of_mut!((*p).users), Some(swap_users_ref_free), 2u32, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)) }) != 0 {
unsafe { kvfree(p as *const ()) };
return ERR_PTR(12i32.wrapping_neg() as i64) as *mut swap_info_struct;
}
spin_lock(unsafe { core::ptr::addr_of_mut!(swap_lock) });
while r#type < unsafe { nr_swapfiles } {
if (unsafe { (*swap_info.0[r#type as usize]).flags }) & 1 == 0 {
break;
}
r#type = r#type.wrapping_add(1);
}
if r#type >= 28u32 {
spin_unlock(unsafe { core::ptr::addr_of_mut!(swap_lock) });
unsafe { percpu_ref_exit(core::ptr::addr_of_mut!((*p).users)) };
unsafe { kvfree(p as *const ()) };
return ERR_PTR(1i32.wrapping_neg() as i64) as *mut swap_info_struct;
}
if r#type >= unsafe { nr_swapfiles } {
unsafe {
(*p).r#type = r#type as i8;
}
// Publish the swap_info_struct after initializing it.
// Note that kvzalloc() above zeroes all its fields.
loop {
kcsan_release!();
loop {
loop {
if !__native_word__17!() {
unsafe { __compiletime_assert_925() };
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
if !(__native_word__17!() || ((const { core::mem::size_of::<*mut swap_info_struct>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_926() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__26!(*unsafe { core::ptr::addr_of_mut!(swap_info.0) as *mut *mut swap_info_struct }.wrapping_offset(r#type as isize), p);
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
// rcu_assign_pointer()
let __v16: u32 = unsafe { nr_swapfiles };
unsafe {
nr_swapfiles = __v16.wrapping_add(1);
}
} else {
defer = p;
p = unsafe { swap_info.0[r#type as usize] };
// Do not memset this entry: a racing procfs swap_next()
// would be relying on p->type to remain valid.
}
unsafe {
(*p).swap_extent_root = *{
__slate_compound_18 = rb_root { rb_node: core::ptr::null_mut::<rb_node>() };
core::ptr::addr_of_mut!(__slate_compound_18)
};
}
plist_node_init(unsafe { core::ptr::addr_of_mut!((*p).list) }, 0i32);
plist_node_init(unsafe { core::ptr::addr_of_mut!((*p).avail_list) }, 0i32);
unsafe {
(*p).flags = 1;
}
spin_unlock(unsafe { core::ptr::addr_of_mut!(swap_lock) });
if !defer.is_null() {
unsafe { percpu_ref_exit(core::ptr::addr_of_mut!((*defer).users)) };
unsafe { kvfree(defer as *const ()) };
}
'__slate_break_20: loop {
spinlock_check(unsafe { core::ptr::addr_of_mut!((*p).lock) });
unsafe {
*core::ptr::addr_of_mut!((*p).lock) = *{
__slate_compound_19 = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = core::mem::zeroed();
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = core::mem::zeroed();
__t1.val = __SlateRecord_types_h_190 { counter: 0 };
__t1
} } };
__t0
} };
core::ptr::addr_of_mut!(__slate_compound_19)
};
}
if true {
break;
}
}
atomic_long_set(unsafe { core::ptr::addr_of_mut!((*p).inuse_pages) }, (1u64 << BITS_PER_TYPE__2!(__SlateRecord_types_h_197).wrapping_sub(2)) as i64);
init_completion(unsafe { core::ptr::addr_of_mut!((*p).comp) });
p
}

fn claim_swapfile(si: *mut swap_info_struct, inode: *mut inode) -> i32 {
if S_ISBLK!(unsafe { (*inode).i_mode }) {
unsafe {
(*si).bdev = I_BDEV(inode);
}
// Zoned block devices contain zones that have a sequential
// write only restriction.  Hence zoned block devices are not
// suitable for swapping.  Disallow them here.
if bdev_is_zoned(unsafe { (*si).bdev }) {
return 22i32.wrapping_neg();
}
unsafe {
(*si).flags = (*si).flags | 64;
}
} else {
if S_ISREG!(unsafe { (*inode).i_mode }) {
unsafe {
(*si).bdev = (*(*inode).i_sb).s_bdev;
}
}
}
0
}

/// Find out how many pages are allowed for a single swap device. There
/// are two limiting factors:
/// 1) the number of bits for the swap offset in the swp_entry_t type, and
/// 2) the number of bits in the swap pte, as defined by the different
/// architectures.
///
/// In order to find the largest possible bit mask, a swap entry with
/// swap type 0 and swap offset ~0UL is created, encoded to a swap pte,
/// decoded to a swp_entry_t again, and finally the swap offset is
/// extracted.
///
/// This will mask all the bits from the initial ~0UL mask that can't
/// be encoded in either the swp_entry_t or the architecture definition
/// of a swap pte.
#[unsafe(no_mangle)]
extern "C" fn generic_max_swapfile_size() -> u64 {
let mut __slate_compound_0: __SlateRecord_mm_types_h_288 = unsafe { core::mem::zeroed() };
let mut entry: __SlateRecord_mm_types_h_288 = swp_entry(0u64, 18446744073709551615u64);
let pte: __SlateRecord_pgtable_64_types_h_21 = softleaf_to_pte(entry);
// Since the PTE can be an invalid softleaf entry (e.g. the none PTE),
// we need to do this manually.
entry = unsafe { *{
__slate_compound_0 = __SlateRecord_mm_types_h_288 { val: pte_val!(pte) };
core::ptr::addr_of_mut!(__slate_compound_0)
} };
entry = swp_entry(__swp_type!(entry), __swp_offset!(entry));
swp_offset(entry).wrapping_add(1)
}

/// Can be overridden by an architecture for additional checks.
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn arch_max_swapfile_size() -> u64 {
generic_max_swapfile_size()
}

fn read_swap_header(si: *mut swap_info_struct, swap_header: *mut swap_header, inode: *mut inode) -> u64 {
let mut i: i32 = 0;
let mut maxpages: u64 = 0;
let mut swapfilepages: u64 = 0;
let mut last_page: u64 = 0;
if (unsafe { memcmp(b"SWAPSPACE2\0".as_ptr() as *const (), ((*swap_header).magic.magic.as_mut_ptr() as *mut u8) as *const (), 10u64) }) != 0 {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x013Unable to find swap-space signature\n".as_ptr() as *const u8) };
return 0;
}
// swap partition endianness hack...
if (unsafe { __slate_intrinsic_e94e9cf815168b51((*swap_header).info.version) }) == 1 {
__swab32s(unsafe { core::ptr::addr_of_mut!((*swap_header).info.version) });
__swab32s(unsafe { core::ptr::addr_of_mut!((*swap_header).info.last_page) });
__swab32s(unsafe { core::ptr::addr_of_mut!((*swap_header).info.nr_badpages) });
if ((unsafe { (*swap_header).info.nr_badpages }) as u64) > offsetof__26!(swap_header).wrapping_sub(offsetof__18!(swap_header)) / (core::mem::size_of::<i32>() as u64) {
return 0;
}
i = 0;
while (i as u32) < unsafe { (*swap_header).info.nr_badpages } {
__swab32s(unsafe { (*swap_header).info.badpages.as_mut_ptr() as *mut u32 }.wrapping_offset(i as isize));
i = i.wrapping_add(1);
}
}
// Check the swap header's sub-version
if (unsafe { (*swap_header).info.version }) != 1 {
let mut __v5: i32;
while false {
}
__v5 = unsafe { _printk(c"\x014Unable to handle swap header version %d\n".as_ptr() as *const u8, (*swap_header).info.version) };
return 0;
}
maxpages = unsafe { swapfile_maximum_size };
last_page = (unsafe { (*swap_header).info.last_page }) as u64;
if last_page == 0 {
let mut __v7: i32;
while false {
}
__v7 = unsafe { _printk(c"\x014Empty swap-file\n".as_ptr() as *const u8) };
return 0;
}
if last_page > maxpages {
let mut __v9: i32;
while false {
}
__v9 = unsafe { _printk(c"\x014Truncating oversized swap area, only using %luk out of %luk\n".as_ptr() as *const u8, K!(maxpages), K!(last_page)) };
}
if maxpages > last_page {
maxpages = last_page.wrapping_add(1);
// p->max is an unsigned int: don't overflow it
if (maxpages as u32) == 0 {
maxpages = 4294967295u64;
}
}
if maxpages == 0 {
return 0;
}
swapfilepages = (i_size_read(inode as *const inode) >> 12i32) as u64;
if swapfilepages != 0 && maxpages > swapfilepages {
let mut __v11: i32;
while false {
}
__v11 = unsafe { _printk(c"\x014Swap area shorter than signature indicates\n".as_ptr() as *const u8) };
return 0;
}
if (unsafe { (*swap_header).info.nr_badpages }) != 0 && S_ISREG!(unsafe { (*inode).i_mode }) {
return 0;
}
if ((unsafe { (*swap_header).info.nr_badpages }) as u64) > offsetof__26!(swap_header).wrapping_sub(offsetof__18!(swap_header)) / (core::mem::size_of::<i32>() as u64) {
return 0;
}
maxpages
}

fn setup_swap_clusters_info(si: *mut swap_info_struct, swap_header: *mut swap_header, maxpages: u64) -> i32 {
let mut __slate_compound_27: spinlock = unsafe { core::mem::zeroed() };
let mut __slate_compound_28: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_30: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_31: spinlock = unsafe { core::mem::zeroed() };
let mut ci: *mut swap_cluster_info = unsafe { core::mem::zeroed() };
let mut page_nr: u32 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut i: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut u64 = core::ptr::addr_of_mut!(i) as *mut u64;
let mut err: i32 = unsafe { core::mem::zeroed() };
let mut cluster_info: core::mem::MaybeUninit<*mut swap_cluster_info> = core::mem::MaybeUninit::uninit();
let __slate_slot_25: *mut *mut swap_cluster_info = core::ptr::addr_of_mut!(cluster_info) as *mut *mut swap_cluster_info;
let mut nr_clusters: u64 = unsafe { core::mem::zeroed() };
unsafe {
'__join_0: {
nr_clusters = maxpages.wrapping_add(256).wrapping_sub(1) / 256;
let __slate_auto_bytes_2 = __slate_slot_25 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
err = 12i32.wrapping_neg();
let __slate_auto_bytes_3 = __slate_slot_23 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __obj_size_1: u64 = size_mul(core::mem::size_of::<swap_cluster_info>() as u64, nr_clusters);
let __v22: *mut swap_cluster_info = kvmalloc_node!(__obj_size_1, 3520u32, 1i32.wrapping_neg()) as *mut swap_cluster_info;
*__slate_slot_25 = __v22;
if !(*__slate_slot_25).is_null() {
*__slate_slot_23 = 0;
while *__slate_slot_23 < nr_clusters {
loop {
spinlock_check(core::ptr::addr_of_mut!((*(*__slate_slot_25).wrapping_offset(*__slate_slot_23 as isize)).lock));
*core::ptr::addr_of_mut!((*(*__slate_slot_25).wrapping_offset(*__slate_slot_23 as isize)).lock) = *{
__slate_compound_31 = spinlock { __slate_anon_0: {
let mut __t2: __SlateRecord_spinlock_types_h_18 = core::mem::zeroed();
__t2.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t3: __SlateRecord_qspinlock_types_h_44 = core::mem::zeroed();
__t3.val = __SlateRecord_types_h_190 { counter: 0 };
__t3
} } };
__t2
} };
core::ptr::addr_of_mut!(__slate_compound_31)
};
if true {
break;
}
}
let __v20: u64 = *__slate_slot_23;
*__slate_slot_23 = __v20.wrapping_add(1);
}
if (*si).flags & 16 == 0 {
__obj_size = size_mul(core::mem::size_of::<swap_sequential_cluster>() as u64, 1u64);
let __slate_auto_bytes_1 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_13 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_14 = _kmalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_28 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_28)
});
alloc_tag_restore!();
} else {
*__slate_slot_14 = _kmalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_30 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_30)
});
}
(*si).global_cluster = *__slate_slot_14 as *mut swap_sequential_cluster;
if unsafe { (*si).global_cluster }.is_null() {
break '__join_0;
} else {
*__slate_slot_23 = 0;
while *__slate_slot_23 < 1 {
*((*(*si).global_cluster).next.as_mut_ptr() as *mut u32).wrapping_offset(*__slate_slot_23 as isize) = 0;
let __v12: u64 = *__slate_slot_23;
*__slate_slot_23 = __v12.wrapping_add(1);
}
loop {
spinlock_check(core::ptr::addr_of_mut!((*si).global_cluster_lock));
*core::ptr::addr_of_mut!((*si).global_cluster_lock) = *{
__slate_compound_27 = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = core::mem::zeroed();
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = core::mem::zeroed();
__t1.val = __SlateRecord_types_h_190 { counter: 0 };
__t1
} } };
__t0
} };
core::ptr::addr_of_mut!(__slate_compound_27)
};
if true {
break;
}
}
}
}
// Mark unusable pages (header page, bad pages, and the EOF part of
// the last cluster) as unavailable. The clusters aren't marked free
// yet, so no list operations are involved yet.
err = swap_cluster_setup_bad_slot(si, *__slate_slot_25, 0u32, false);
if err == 0 {
*__slate_slot_23 = 0;
'__join_19: {
while *__slate_slot_23 < ((*swap_header).info.nr_badpages as u64) {
page_nr = *((*swap_header).info.badpages.as_mut_ptr() as *mut u32).wrapping_offset(*__slate_slot_23 as isize);
if page_nr == 0 || page_nr > (*swap_header).info.last_page {
break '__join_19;
} else {
err = swap_cluster_setup_bad_slot(si, *__slate_slot_25, page_nr, false);
if err != 0 {
break '__join_0;
} else {
let __v8: u64 = *__slate_slot_23;
*__slate_slot_23 = __v8.wrapping_add(1);
}
}
}
*__slate_slot_23 = maxpages;
while *__slate_slot_23 < round_up__2!(maxpages, 256i32) {
err = swap_cluster_setup_bad_slot(si, *__slate_slot_25, *__slate_slot_23 as u32, true);
if err != 0 {
break '__join_0;
} else {
let __v6: u64 = *__slate_slot_23;
*__slate_slot_23 = __v6.wrapping_add(1);
}
}
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*si).free_clusters));
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*si).full_clusters));
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*si).discard_clusters));
*__slate_slot_23 = 0;
while *__slate_slot_23 < 1 {
INIT_LIST_HEAD(((*si).nonfull_clusters.as_mut_ptr() as *mut list_head).wrapping_offset(*__slate_slot_23 as isize));
INIT_LIST_HEAD(((*si).frag_clusters.as_mut_ptr() as *mut list_head).wrapping_offset(*__slate_slot_23 as isize));
let __v4: u64 = *__slate_slot_23;
*__slate_slot_23 = __v4.wrapping_add(1);
}
*__slate_slot_23 = 0;
while *__slate_slot_23 < nr_clusters {
ci = (*__slate_slot_25).wrapping_offset(*__slate_slot_23 as isize);
if (*ci).count != 0 {
(*ci).flags = 2;
list_add_tail(core::ptr::addr_of_mut!((*ci).list), ((*si).nonfull_clusters.as_mut_ptr() as *mut list_head).wrapping_offset(0 as isize));
} else {
(*ci).flags = 1;
list_add_tail(core::ptr::addr_of_mut!((*ci).list), core::ptr::addr_of_mut!((*si).free_clusters));
}
let __v1: u64 = *__slate_slot_23;
*__slate_slot_23 = __v1.wrapping_add(1);
}
(*si).cluster_info = *__slate_slot_25;
return 0;
}
while false {
}
_printk(c"\x014Bad slot offset is out of border: %d (last_page: %d)\n".as_ptr() as *const u8, page_nr, (*swap_header).info.last_page);
err = 22i32.wrapping_neg();
}
}
}
free_swap_cluster_info(*__slate_slot_25, maxpages);
return err;
}
}

#[unsafe(no_mangle)]
extern "C" fn __x64_sys_swapon(regs: *const pt_regs) -> i64 {
__se_sys_swapon((unsafe { (*regs).di }) as i64, (unsafe { (*regs).si }) as i64)
}

#[unsafe(no_mangle)]
extern "C" fn __ia32_sys_swapon(regs: *const pt_regs) -> i64 {
__se_sys_swapon(((unsafe { (*regs).bx }) as u32) as i64, ((unsafe { (*regs).cx }) as u32) as i64)
}

fn __se_sys_swapon(specialfile: i64, swap_flags: i64) -> i64 {
let ret: i64 = __do_sys_swapon(specialfile as *const u8, swap_flags as i32);
while false {
}
ret
}

fn __do_sys_swapon(specialfile: *const u8, swap_flags: i32) -> i64 {
let mut __slate_compound_53: __SlateRecord_types_h_197 = unsafe { core::mem::zeroed() };
let mut __slate_compound_54: __SlateRecord_types_h_197 = unsafe { core::mem::zeroed() };
let mut __v18: bool = unsafe { core::mem::zeroed() };
let mut __v22: bool = unsafe { core::mem::zeroed() };
let mut __v26: bool = unsafe { core::mem::zeroed() };
let mut __v30: bool = unsafe { core::mem::zeroed() };
let mut __ret_once: bool = unsafe { core::mem::zeroed() };
let mut __v38: bool = unsafe { core::mem::zeroed() };
let mut name: core::mem::MaybeUninit<*mut filename> = core::mem::MaybeUninit::uninit();
let __slate_slot_39: *mut *mut filename = core::ptr::addr_of_mut!(name) as *mut *mut filename;
let mut inced_nr_rotate_swap: bool = unsafe { core::mem::zeroed() };
let mut inode: *mut inode = unsafe { core::mem::zeroed() };
let mut folio: *mut folio = unsafe { core::mem::zeroed() };
let mut maxpages: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_43: *mut u64 = core::ptr::addr_of_mut!(maxpages) as *mut u64;
let mut span: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_44: *mut u64 = core::ptr::addr_of_mut!(span) as *mut u64;
let mut nr_extents: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_45: *mut i32 = core::ptr::addr_of_mut!(nr_extents) as *mut i32;
let mut swap_header: core::mem::MaybeUninit<*mut swap_header> = core::mem::MaybeUninit::uninit();
let __slate_slot_46: *mut *mut swap_header = core::ptr::addr_of_mut!(swap_header) as *mut *mut swap_header;
let mut error: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_47: *mut i32 = core::ptr::addr_of_mut!(error) as *mut i32;
let mut prio: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_48: *mut i32 = core::ptr::addr_of_mut!(prio) as *mut i32;
let mut dentry: core::mem::MaybeUninit<*mut dentry> = core::mem::MaybeUninit::uninit();
let __slate_slot_49: *mut *mut dentry = core::ptr::addr_of_mut!(dentry) as *mut *mut dentry;
let mut mapping: core::mem::MaybeUninit<*mut address_space> = core::mem::MaybeUninit::uninit();
let __slate_slot_50: *mut *mut address_space = core::ptr::addr_of_mut!(mapping) as *mut *mut address_space;
let mut swap_file: *mut file = unsafe { core::mem::zeroed() };
let mut si: core::mem::MaybeUninit<*mut swap_info_struct> = core::mem::MaybeUninit::uninit();
let __slate_slot_52: *mut *mut swap_info_struct = core::ptr::addr_of_mut!(si) as *mut *mut swap_info_struct;
let __slate_cleanup_ptr_39 = __slate_slot_39;
let __slate_cleanup_armed_39 = core::cell::Cell::new(false);
let __slate_cleanup_39 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_39.get() {
unsafe { class_filename_destructor(__slate_cleanup_ptr_39 as *mut _) };
}
});
unsafe {
let __slate_auto_bytes = __slate_slot_52 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
swap_file = core::ptr::null_mut::<file>();
let __slate_auto_bytes_1 = __slate_slot_50 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_49 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_48 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_47 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_46 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_45 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_7 = __slate_slot_44 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_43 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
folio = core::ptr::null_mut::<folio>();
inode = core::ptr::null_mut::<inode>();
inced_nr_rotate_swap = false;
if swap_flags & !(32767 | 32768 | 65536 | 131072 | 262144) != 0 {
return 22i32.wrapping_neg() as i64;
} else {
if !capable(21i32) {
return 1i32.wrapping_neg() as i64;
} else {
// Allocate or reuse existing !SWP_USED swap_info. The returned
// si will stay in a dying status, so nothing will access its content
// until enable_swap_info resurrects its percpu ref and expose it.
*__slate_slot_52 = alloc_swap_info();
if IS_ERR(*__slate_slot_52 as *const ()) {
return PTR_ERR(*__slate_slot_52 as *const ());
} else {
loop {
__init_work(core::ptr::addr_of_mut!((*(*__slate_slot_52)).discard_work), 0i32);
(*core::ptr::addr_of_mut!((*(*__slate_slot_52)).discard_work)).data = *{
__slate_compound_54 = __SlateRecord_types_h_197 { counter: 4503599625273344i64 };
core::ptr::addr_of_mut!(__slate_compound_54)
};
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*(*__slate_slot_52)).discard_work)).entry));
(*core::ptr::addr_of_mut!((*(*__slate_slot_52)).discard_work)).func = Some(swap_discard_work);
if true {
if true {
break;
}
}
}
loop {
__init_work(core::ptr::addr_of_mut!((*(*__slate_slot_52)).reclaim_work), 0i32);
(*core::ptr::addr_of_mut!((*(*__slate_slot_52)).reclaim_work)).data = *{
__slate_compound_53 = __SlateRecord_types_h_197 { counter: 4503599625273344i64 };
core::ptr::addr_of_mut!(__slate_compound_53)
};
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*(*__slate_slot_52)).reclaim_work)).entry));
(*core::ptr::addr_of_mut!((*(*__slate_slot_52)).reclaim_work)).func = Some(swap_reclaim_work);
if true {
if true {
break;
}
}
}
'__join_4: {
core::ptr::write(__slate_slot_39, class_filename_constructor(specialfile));
__slate_cleanup_armed_39.set(true);
swap_file = file_open_name(*__slate_slot_39, 32898i32, 0u16);
if IS_ERR(swap_file as *const ()) {
*__slate_slot_47 = PTR_ERR(swap_file as *const ()) as i32;
swap_file = core::ptr::null_mut::<file>();
} else {
*__slate_slot_50 = (*swap_file).f_mapping;
*__slate_slot_49 = (*swap_file).__slate_anon_10.f_path.dentry;
inode = (*(*__slate_slot_50)).host;
*__slate_slot_47 = claim_swapfile(*__slate_slot_52, inode);
if unlikely__2!(*__slate_slot_47) == 0 {
inode_lock(inode);
if d_unlinked(*__slate_slot_49 as *const dentry) != 0 {
__v38 = true;
} else {
__v38 = cant_mount(*__slate_slot_49 as *const dentry) != 0;
}
if __v38 {
*__slate_slot_47 = 2i32.wrapping_neg();
} else {
if IS_SWAPFILE!(inode) != 0 {
*__slate_slot_47 = 16i32.wrapping_neg();
} else {
if IS_ENCRYPTED!(inode) != 0 {
let __ret_do_once: bool = true;
let __ret_cond: bool = __ret_do_once;
__ret_once = false;
if unlikely!(__ret_cond) != 0 && unlikely!(!__already_done) != 0 {
__already_done = true;
__ret_once = true;
}
let __v35: i64 = __ret_once as i64;
if __v35 != 0 {
while false {
}
_printk(c"\x014Filesystem-level encrypted swapfile '%s' is unsupported. Create a loop device over it, or use dm-crypt\n".as_ptr() as *const u8, (*(*__slate_slot_39)).__slate_anon_0.name);
}
*__slate_slot_47 = 22i32.wrapping_neg();
} else {
// The swap subsystem needs a major overhaul to support this.
// It doesn't work yet so just disable it for now.
if mapping_min_folio_order(*__slate_slot_50 as *const address_space) > 0 {
*__slate_slot_47 = 22i32.wrapping_neg();
} else {
// Read the swap header.
if (*(*(*__slate_slot_50)).a_ops).read_folio == None {
*__slate_slot_47 = 22i32.wrapping_neg();
} else {
folio = read_mapping_folio(*__slate_slot_50, 0u64, swap_file);
if IS_ERR(folio as *const ()) {
*__slate_slot_47 = PTR_ERR(folio as *const ()) as i32;
} else {
*__slate_slot_46 = kmap_local_folio(folio as *const folio, 0u64) as *mut swap_header;
*__slate_slot_43 = read_swap_header(*__slate_slot_52, *__slate_slot_46, inode);
if unlikely!(*__slate_slot_43 == 0) != 0 {
*__slate_slot_47 = 22i32.wrapping_neg();
} else {
(*(*__slate_slot_52)).max = *__slate_slot_43 as u32;
(*(*__slate_slot_52)).pages = (*__slate_slot_43).wrapping_sub(1) as u32;
*__slate_slot_45 = setup_swap_extents(*__slate_slot_52, swap_file, core::ptr::addr_of_mut!(*__slate_slot_44));
if *__slate_slot_45 < 0 {
*__slate_slot_47 = *__slate_slot_45;
} else {
if (*(*__slate_slot_52)).pages != unsafe { (*(*__slate_slot_52)).max }.wrapping_sub(1) {
while false {
}
_printk(c"\x013swap:%u != (max:%u - 1)\n".as_ptr() as *const u8, (*(*__slate_slot_52)).pages, (*(*__slate_slot_52)).max);
*__slate_slot_47 = 22i32.wrapping_neg();
} else {
*__slate_slot_43 = (*(*__slate_slot_52)).max as u64;
// Set up the swap cluster info
*__slate_slot_47 = setup_swap_clusters_info(*__slate_slot_52, *__slate_slot_46, *__slate_slot_43);
if *__slate_slot_47 == 0 {
if !unsafe { (*(*__slate_slot_52)).bdev }.is_null() {
__v30 = bdev_stable_writes((*(*__slate_slot_52)).bdev);
} else {
__v30 = false;
}
if __v30 {
let __v29: *mut swap_info_struct = *__slate_slot_52;
(*__v29).flags = (*__v29).flags | 2048;
}
if !unsafe { (*(*__slate_slot_52)).bdev }.is_null() {
__v26 = bdev_synchronous((*(*__slate_slot_52)).bdev);
} else {
__v26 = false;
}
if __v26 {
let __v25: *mut swap_info_struct = *__slate_slot_52;
(*__v25).flags = (*__v25).flags | 4096;
}
if !unsafe { (*(*__slate_slot_52)).bdev }.is_null() {
__v22 = !bdev_rot((*(*__slate_slot_52)).bdev);
} else {
__v22 = false;
}
if __v22 {
let __v21: *mut swap_info_struct = *__slate_slot_52;
(*__v21).flags = (*__v21).flags | 16;
} else {
atomic_inc(core::ptr::addr_of_mut!(nr_rotate_swap));
inced_nr_rotate_swap = true;
}
if swap_flags & 65536 != 0 && !unsafe { (*(*__slate_slot_52)).bdev }.is_null() {
__v18 = bdev_max_discard_sectors((*(*__slate_slot_52)).bdev) != 0;
} else {
__v18 = false;
}
if __v18 {
// When discard is enabled for swap with no particular
// policy flagged, we set all swap discard flags here in
// order to sustain backward compatibility with older
// swapon(8) releases.
let __v17: *mut swap_info_struct = *__slate_slot_52;
(*__v17).flags = (*__v17).flags | 1540u64;
// By flagging sys_swapon, a sysadmin can tell us to
// either do single-time area discards only, or to just
// perform discards for released swap page-clusters.
// Now it's time to adjust the p->flags accordingly.
if swap_flags & 131072 != 0 {
let __v11: *mut swap_info_struct = *__slate_slot_52;
let __v9: u64 = (*__v11).flags & 18446744073709550591u64;
(*__v11).flags = __v9;
} else {
if swap_flags & 262144 != 0 {
let __v14: *mut swap_info_struct = *__slate_slot_52;
let __v12: u64 = (*__v14).flags & 18446744073709551103u64;
(*__v14).flags = __v12;
}
}
// issue a swapon-time discard if it's still required
if (*(*__slate_slot_52)).flags & 512 != 0 {
let err: i32 = discard_swap(*__slate_slot_52);
if unlikely__2!(err) != 0 {
while false {
}
_printk(c"\x013swapon: discard_swap(%p): %d\n".as_ptr() as *const u8, *__slate_slot_52, err);
}
}
}
*__slate_slot_47 = zswap_swapon((*(*__slate_slot_52)).r#type as i32, *__slate_slot_43);
if *__slate_slot_47 == 0 {
// Flush any pending IO and dirty mappings before we start using this
// swap device.
(*inode).i_flags = (*inode).i_flags | 256u32;
*__slate_slot_47 = inode_drain_writes(inode);
if *__slate_slot_47 != 0 {
let __v1: u32 = (*inode).i_flags & 4294967039u32;
(*inode).i_flags = __v1;
zswap_swapoff((*(*__slate_slot_52)).r#type as i32);
} else {
mutex_lock(core::ptr::addr_of_mut!(swapon_mutex));
*__slate_slot_48 = 1i32.wrapping_neg();
if swap_flags & 32768 != 0 {
*__slate_slot_48 = swap_flags & 32767;
}
// The plist prio is negated because plist ordering is
// low-to-high, while swap ordering is high-to-low
(*(*__slate_slot_52)).prio = *__slate_slot_48 as i16;
(*(*__slate_slot_52)).list.prio = ((*(*__slate_slot_52)).prio as i32).wrapping_neg();
(*(*__slate_slot_52)).avail_list.prio = ((*(*__slate_slot_52)).prio as i32).wrapping_neg();
(*(*__slate_slot_52)).swap_file = swap_file;
// Sets SWP_WRITEOK, resurrect the percpu ref, expose the swap device
enable_swap_info(*__slate_slot_52);
while false {
}
_printk(c"\x016Adding %uk swap on %s.  Priority:%d extents:%d across:%lluk %s%s%s%s\n".as_ptr() as *const u8, K!(unsafe { (*(*__slate_slot_52)).pages }), (*(*__slate_slot_39)).__slate_anon_0.name, (*(*__slate_slot_52)).prio as i32, *__slate_slot_45, K!(*__slate_slot_44), if (*(*__slate_slot_52)).flags & 16 != 0 { c"SS".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if (*(*__slate_slot_52)).flags & 4 != 0 { c"D".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if (*(*__slate_slot_52)).flags & 512 != 0 { c"s".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }, if (*(*__slate_slot_52)).flags & 1024 != 0 { c"c".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 });
mutex_unlock(core::ptr::addr_of_mut!(swapon_mutex));
atomic_inc(core::ptr::addr_of_mut!(proc_poll_event));
wake_up_interruptible!(core::ptr::addr_of_mut!(proc_poll_wait));
*__slate_slot_47 = 0;
break '__join_4;
}
}
}
}
}
}
}
}
}
}
}
}
inode_unlock(inode);
}
}
kfree((*(*__slate_slot_52)).global_cluster as *const ());
(*(*__slate_slot_52)).global_cluster = core::ptr::null_mut::<swap_sequential_cluster>();
inode = core::ptr::null_mut::<inode>();
destroy_swap_extents(*__slate_slot_52, swap_file);
free_swap_cluster_info((*(*__slate_slot_52)).cluster_info, (*(*__slate_slot_52)).max as u64);
(*(*__slate_slot_52)).cluster_info = core::ptr::null_mut::<swap_cluster_info>();
// Clear the SWP_USED flag after all resources are freed so
// alloc_swap_info can reuse this si safely.
spin_lock(core::ptr::addr_of_mut!(swap_lock));
(*(*__slate_slot_52)).flags = 0;
spin_unlock(core::ptr::addr_of_mut!(swap_lock));
if inced_nr_rotate_swap {
atomic_dec(core::ptr::addr_of_mut!(nr_rotate_swap));
}
if !swap_file.is_null() {
filp_close(swap_file, core::ptr::null_mut::<()>());
}
}
if !IS_ERR_OR_NULL(folio as *const ()) {
folio_release_kmap(folio, *__slate_slot_46 as *mut ());
}
if !inode.is_null() {
inode_unlock(inode);
}
return *__slate_slot_47 as i64;
}
}
}
}
}

static mut __key__3644: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__3645: lock_class_key = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".data..once")]
static mut __already_done: bool = unsafe { core::mem::zeroed() };

#[unsafe(no_mangle)]
extern "C" fn si_swapinfo(val: *mut sysinfo) {
let mut r#type: u32 = 0;
let mut nr_to_be_unused: u64 = 0;
spin_lock(unsafe { core::ptr::addr_of_mut!(swap_lock) });
while r#type < unsafe { nr_swapfiles } {
let si: *mut swap_info_struct = unsafe { swap_info.0[r#type as usize] };
if (unsafe { (*si).flags }) & 1 != 0 && (unsafe { (*si).flags }) & 2 == 0 {
let __v0: i64 = swap_usage_in_pages(si);
nr_to_be_unused = nr_to_be_unused.wrapping_add(__v0 as u64);
}
r#type = r#type.wrapping_add(1);
}
unsafe {
(*val).freeswap = (atomic_long_read(core::ptr::addr_of_mut!(nr_swap_pages) as *const __SlateRecord_types_h_197) as u64).wrapping_add(nr_to_be_unused);
}
unsafe {
(*val).totalswap = (total_swap_pages as u64).wrapping_add(nr_to_be_unused);
}
spin_unlock(unsafe { core::ptr::addr_of_mut!(swap_lock) });
}

/// swap_dup_entry_direct() - Increase reference count of a swap entry by one.
/// @entry: first swap entry from which we want to increase the refcount.
///
/// Returns 0 for success, or -ENOMEM if the extend table is required
/// but could not be atomically allocated.  Returns -EINVAL if the swap
/// entry is invalid, which might occur if a page table entry has got
/// corrupted.
///
/// Context: Caller must ensure there is no race condition on the reference
/// owner. e.g., locking the PTL of a PTE containing the entry being increased.
/// Also the swap entry must have a count >= 1. Otherwise folio_dup_swap should
/// be used.
#[unsafe(no_mangle)]
extern "C" fn swap_dup_entry_direct(entry: __SlateRecord_mm_types_h_288) -> i32 {
let si: *mut swap_info_struct = swap_entry_to_info(entry);
let mut __v0: i64;
let __ret_warn_on: i32 = si.is_null() as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("927: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 927b - ., 3; .popsection\n# {0}", const 927 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_swap_dup_entry_direct_1, sym __slate_asm_string_swap_dup_entry_direct_2, const 3885 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("928: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 928b - ., 4; .popsection\n# {0}", const 928 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
if __ratelimit__5!(unsafe { core::ptr::addr_of_mut!(_rs__3886) }) != 0 {
let mut __v5: i32;
while false {
}
__v5 = unsafe { _printk(c"\x013%s%08lx\n".as_ptr() as *const u8, core::ptr::addr_of!(Bad_file.0) as *const u8, entry.val) };
}
return 22i32.wrapping_neg();
}
// The caller must be increasing the swap count from a direct
// reference of the swap slot (e.g. a swap entry in page table).
// So the swap count must be >= 1.
VM_WARN_ON_ONCE!();
swap_dup_entries_cluster(si, swp_offset(entry), 1i32)
}

static mut _rs__3886: ratelimit_state = {
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

#[unsafe(link_section = ".init.text")]
extern "C" fn swapfile_init() -> i32 {
unsafe {
swapfile_maximum_size = arch_max_swapfile_size();
}
// Once a cluster is freed, it's swap table content is read
// only, and all swap cache readers (swap_cache_*) verifies
// the content before use. So it's safe to use RCU slab here.
unsafe {
swap_table_cachep = __kmem_cache_create(c"swap_table".as_ptr() as *const u8, core::mem::size_of::<swap_table>() as u32, 0u32, __SLAB_FLAG_BIT!(8i32) | __SLAB_FLAG_BIT!(9i32), None);
}
if (unsafe { swapfile_maximum_size }) >= 1u64 << if pgtable_l5_enabled__4!() != 0 { 52i32 } else { 46i32 }.wrapping_sub(12).wrapping_add(2) {
unsafe {
swap_migration_ad_supported = true;
}
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_swapfile_init_930: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i32>, *mut ()>(Some(swapfile_init)) };

static __slate_asm_string___do_sys_swapoff_17: [u8; 1] = *b"\0";

static __slate_asm_string___do_sys_swapoff_18: [u8; 43] = *b"/home/takashi/c-corpus/linux/mm/swapfile.c\0";

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string___kmem_cache_default_args_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmem_cache_default_args_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string___swap_cluster_alloc_entries_2: [u8; 1] = *b"\0";

static __slate_asm_string___swap_cluster_alloc_entries_3: [u8; 43] = *b"/home/takashi/c-corpus/linux/mm/swapfile.c\0";

static __slate_asm_string___swap_cluster_dup_entry_2: [u8; 1] = *b"\0";

static __slate_asm_string___swap_cluster_dup_entry_26: [u8; 1] = *b"\0";

static __slate_asm_string___swap_cluster_dup_entry_27: [u8; 43] = *b"/home/takashi/c-corpus/linux/mm/swapfile.c\0";

static __slate_asm_string___swap_cluster_dup_entry_3: [u8; 43] = *b"/home/takashi/c-corpus/linux/mm/swapfile.c\0";

static __slate_asm_string___swap_cluster_dup_entry_8: [u8; 1] = *b"\0";

static __slate_asm_string___swap_cluster_dup_entry_9: [u8; 43] = *b"/home/takashi/c-corpus/linux/mm/swapfile.c\0";

static __slate_asm_string__enable_swap_info_2: [u8; 1] = *b"\0";

static __slate_asm_string__enable_swap_info_3: [u8; 43] = *b"/home/takashi/c-corpus/linux/mm/swapfile.c\0";

static __slate_asm_string_add_swap_extent_2: [u8; 1] = *b"\0";

static __slate_asm_string_add_swap_extent_3: [u8; 43] = *b"/home/takashi/c-corpus/linux/mm/swapfile.c\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

static __slate_asm_string_offset_to_swap_extent_2: [u8; 1] = *b"\0";

static __slate_asm_string_offset_to_swap_extent_3: [u8; 43] = *b"/home/takashi/c-corpus/linux/mm/swapfile.c\0";

static __slate_asm_string_percpu_ref_tryget_live_rcu_1: [u8; 1] = *b"\0";

static __slate_asm_string_percpu_ref_tryget_live_rcu_2: [u8; 61] = *b"/home/takashi/c-corpus/linux/include/linux/percpu-refcount.h\0";

static __slate_asm_string_pfn_pte_1: [u8; 1] = *b"\0";

static __slate_asm_string_pfn_pte_2: [u8; 60] = *b"/home/takashi/c-corpus/linux/arch/x86/include/asm/pgtable.h\0";

static __slate_asm_string_pin_hibernation_swap_type_1: [u8; 1] = *b"\0";

static __slate_asm_string_pin_hibernation_swap_type_2: [u8; 43] = *b"/home/takashi/c-corpus/linux/mm/swapfile.c\0";

static __slate_asm_string_pin_hibernation_swap_type_6: [u8; 1] = *b"\0";

static __slate_asm_string_pin_hibernation_swap_type_7: [u8; 43] = *b"/home/takashi/c-corpus/linux/mm/swapfile.c\0";

static __slate_asm_string_swap_cluster_assert_empty_16: [u8; 1] = *b"\0";

static __slate_asm_string_swap_cluster_assert_empty_17: [u8; 43] = *b"/home/takashi/c-corpus/linux/mm/swapfile.c\0";

static __slate_asm_string_swap_cluster_assert_empty_21: [u8; 1] = *b"\0";

static __slate_asm_string_swap_cluster_assert_empty_22: [u8; 43] = *b"/home/takashi/c-corpus/linux/mm/swapfile.c\0";

static __slate_asm_string_swap_cluster_assert_empty_3: [u8; 1] = *b"\0";

static __slate_asm_string_swap_cluster_assert_empty_4: [u8; 43] = *b"/home/takashi/c-corpus/linux/mm/swapfile.c\0";

static __slate_asm_string_swap_cluster_assert_empty_8: [u8; 1] = *b"\0";

static __slate_asm_string_swap_cluster_assert_empty_9: [u8; 43] = *b"/home/takashi/c-corpus/linux/mm/swapfile.c\0";

static __slate_asm_string_swap_cluster_setup_bad_slot_13: [u8; 1] = *b"\0";

static __slate_asm_string_swap_cluster_setup_bad_slot_14: [u8; 43] = *b"/home/takashi/c-corpus/linux/mm/swapfile.c\0";

static __slate_asm_string_swap_cluster_setup_bad_slot_18: [u8; 1] = *b"\0";

static __slate_asm_string_swap_cluster_setup_bad_slot_19: [u8; 43] = *b"/home/takashi/c-corpus/linux/mm/swapfile.c\0";

static __slate_asm_string_swap_dup_entry_direct_1: [u8; 1] = *b"\0";

static __slate_asm_string_swap_dup_entry_direct_2: [u8; 43] = *b"/home/takashi/c-corpus/linux/mm/swapfile.c\0";

static __slate_asm_string_swap_put_entries_direct_14: [u8; 1] = *b"\0";

static __slate_asm_string_swap_put_entries_direct_15: [u8; 43] = *b"/home/takashi/c-corpus/linux/mm/swapfile.c\0";

static __slate_asm_string_swap_put_entries_direct_16: [u8; 1] = *b"\0";

static __slate_asm_string_swap_put_entries_direct_17: [u8; 43] = *b"/home/takashi/c-corpus/linux/mm/swapfile.c\0";

static __slate_asm_string_wait_for_allocation_0: [u8; 1] = *b"\0";

static __slate_asm_string_wait_for_allocation_1: [u8; 43] = *b"/home/takashi/c-corpus/linux/mm/swapfile.c\0";

static __slate_asm_string_xa_mk_value_1: [u8; 1] = *b"\0";

static __slate_asm_string_xa_mk_value_2: [u8; 52] = *b"/home/takashi/c-corpus/linux/include/linux/xarray.h\0";

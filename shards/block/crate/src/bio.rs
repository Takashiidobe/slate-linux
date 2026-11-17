
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
//! Copyright (C) 2001 Jens Axboe <axboe@kernel.dk>
//! This counts to 15. Any more, it will return 16th argument.
//! Operation not permitted
//! Package not installed
//! restart by calling sys_restart_syscall
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
//! error conditions
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
//! # chars in a file name
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
//! Macros to retrieve the current execution context:
//!
//! in_nmi()		- We're in NMI context
//! in_hardirq()		- We're in hard IRQ context
//! in_serving_softirq()	- We're in softirq context
//! in_task()		- We're in task context
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
//! Bit size and mask of CPU number stored in the per CPU data (and TSC_AUX)
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
//! An exchange-type operation, which takes a value and a pointer, and
//! returns the old value.
//! Prevent speculative execution past this barrier.
//! @a is a power of 2 value
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
//! Ensure that leaf 0x2 cache and TLB type values do not intersect,
//! since they share the same type field at struct cpuid_0x2_table.
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
//! Used in tsk->__state:
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
//! Suits me
//! percpu_counter batch for local add or sub
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
//! Must be a compile-time constant, so implement it as a macro
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
//! return the first ptr or next element in an RCU protected list
//! Keep unconverted code working
//! used to be O_SYNC, see below
//! Valid hint values for F_{GET,SET}_RW_HINT. 0 is "not set", or can be
//! used to clear any hints previously set.
//! Gives us 8 prio classes with 13-bits of data for each class
//! per-IO O_SYNC
//! File supports atomic writes
//! number of env pointers
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
//! 31 pointers + header align the folio_batch structure to a power of two
//! Maximum number of blkcg policies allowed to be registered concurrently.
//! Defined here to simplify include dependency.
//! ***DANGEROUS*** don't add guard page
//! See the comment in asm-generic/io.h about ioremap_np().
//! These macros should be used after a dma_map_sg call has been done
//! to get bus addresses of each of the SG entries and their lengths.
//! You should only work with the number of sg entries dma_map_sg
//! returns, or alternatively stop on the first sg_dma_len(sg) which
//! is 0.
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
// disable_context_analysis() - disables context analysis
//
// Disables context analysis. Must be paired with a later
// enable_context_analysis().
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
// Concatenate two parameters, but allow them to be expanded beforehand.
// No such file or directory
// Interrupted system call
// I/O error
// No such device or address
// Argument list too long
// Try again
// Out of memory
// Bad address
// Device or resource busy
// Invalid argument
// Not a typewriter
// Text file busy
// No space left on device
// Value too large for defined data type
// Operation not supported on transport endpoint
// Memory page has hardware error
// Operation is not supported
// DECLARE_FLEX_ARRAY() - Declare a flexible array usable in a union
//
// @TYPE: The type of each flexible array element
// @NAME: The name of the flexible array member
//
// In order to have a flexible array member in a union or alone in a
// struct, it needs to be wrapped in an anonymous struct with at least 1
// named member, but that member can be empty.
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
// warning conditions
// # chars in a path name including nul
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
// pr_warn - Print a warning-level message
// @fmt: format string
// @...: arguments for the format string
//
// This macro expands to a printk with KERN_WARNING loglevel. It uses pr_fmt()
// to generate the format string.
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
// min - return minimum of two values of the same or compatible types
// @x: first value
// @y: second value
// max - return maximum of two values of the same or compatible types
// @x: first value
// @y: second value
// min3 - return minimum of three values
// @x: first value
// @y: second value
// @z: third value
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
// No static inline type checking - see Subtlety (1) above.
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
// N 32-bit words worth of info
// signal notifications exist
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
// GFP_ZONE_BAD is a bitmap for all combinations of __GFP_DMA, __GFP_DMA32
// __GFP_HIGHMEM and __GFP_MOVABLE that are not permitted. One flag per
// entry starting with bit 0. Bit is set if the combination is not
// allowed.
// Currently stuck as a macro due to indirect forward reference to
// linux/mmzone.h's __section_mem_map_addr() definition:
// All top-level MITIGATION_PAGE_TABLE_ISOLATION page tables are order-1 pages
// (8k-aligned and 8k in size).  The kernel one is at the beginning 4k and
// the user one is in the last 4k.  To switch between them, you
// just need to flip the 12th bit in their addresses.
// Resource extended types
// No address assigned yet
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
// Implementation specific max size
// We overload the LSB of the page pointer to indicate whether it's
// a valid sg entry, or whether it points to the start of a new scatterlist.
// Those low bits are there for everyone! (thanks mason :-)
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
/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__14 {
() => {
((core::mem::size_of::<[biovec_slab; 4]>() as u64) / (core::mem::size_of::<biovec_slab>() as u64)).wrapping_add(__must_be_array__37!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__33 {
() => {
((core::mem::size_of::<[biovec_slab; 4]>() as u64) / (core::mem::size_of::<biovec_slab>() as u64)).wrapping_add(__must_be_array__13!() as u64)
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

macro_rules! _AC {
() => {
1u64
};
}

macro_rules! _AC__3 {
() => {
18446744071562067968u64
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

macro_rules! __SLAB_FLAG_BIT {
($nr:expr) => {
1u32 << $nr
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

macro_rules! __cmp_once__3 {
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

/// __ffs - find first set bit in word
/// @word: The word to search
///
/// Undefined if no bit exists, so code should check against 0 first.
macro_rules! __ffs {
($word:expr) => {
if unsafe { __slate_intrinsic_f217c4720f04f6db($word) } { ($word.trailing_zeros() as i32) as u64 } else { variable__ffs($word) }
};
}

macro_rules! __free_page {
($page:expr) => {
unsafe { __free_pages($page, 0u32) };
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

macro_rules! __must_be_array__13 {
() => {
core::mem::size_of::<__SlateRecord_bio_c_2028>() as i32
};
}

macro_rules! __must_be_array__37 {
() => {
core::mem::size_of::<__SlateRecord_bio_c_2177>() as i32
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word {
() => {
((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__21 {
() => {
((const { core::mem::size_of::<*mut bio>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut bio>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut bio>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut bio>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
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
macro_rules! __native_word__15 {
() => {
((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__2 {
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
if is_signed_type!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul(((core::mem::size_of::<u64>() as u64) < 4) as i32)) }
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

macro_rules! __types_ok3__4 {
($ux:expr, $uy:expr, $uz:expr) => {
__sign_use__6!($ux) & __sign_use__6!($uy) & __sign_use__3!($uz)
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

macro_rules! __types_ok__9 {
($ux:expr, $uy:expr) => {
__sign_use__6!($ux) & __sign_use__3!($uy)
};
}

macro_rules! __types_ok__2 {
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
macro_rules! __verify_pcpu_ptr__18 {
() => {
loop {
let __v: *const () = core::ptr::null_mut::<bio_complete_batch>() as *const ();
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
let __vpp_verify: *const () = core::ptr::null_mut::<bio_alloc_cache>() as *const ();
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
macro_rules! __verify_pcpu_ptr__26 {
() => {
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<bio_complete_batch>() as *const ();
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
macro_rules! __verify_pcpu_ptr__29 {
() => {
loop {
let __v: *const () = core::ptr::null_mut::<bio_alloc_cache>() as *const ();
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

macro_rules! alloc_hooks__2 {
($_do_alloc:expr) => {
{
let mut __v: i32;
let mut __v2: i32;
let mut __v3: i32 = 0;
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

macro_rules! alloc_hooks__4 {
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

macro_rules! bio_iter_iovec {
($bio:expr, $iter:expr) => {
bvec_iter_bvec((unsafe { (*$bio).bi_io_vec }) as *const bio_vec, $iter)
};
}

macro_rules! bio_sectors {
($bio:expr) => {
bvec_iter_sectors!((*$bio).bi_iter)
};
}

macro_rules! blk_queue_pci_p2pdma {
($q:expr) => {
(unsafe { (*$q).limits.features }) & 4096u32
};
}

macro_rules! bvec_iter_sectors {
($iter:expr) => {
(unsafe { $iter.bi_size }) >> 9i32
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
macro_rules! container_of__13 {
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
macro_rules! container_of__31 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__43!($type) as isize)) }) as *mut ()) as *mut $type;
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

macro_rules! hardirq_count {
() => {
(preempt_count() as u64) & __IRQ_MASK!(4i32) << 24i32
};
}

macro_rules! hlist_entry__3 {
($ptr:expr, $type:ty) => {
container_of__11!($ptr, $type)
};
}

macro_rules! hlist_entry_safe__4 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
let __v2: *mut hlist_node = $ptr;
let __v3: *mut $type = if !__v2.is_null() {
hlist_entry__3!(__v2, $type)
} else {
core::ptr::null_mut::<bio_set>()
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
macro_rules! ilog2 {
($n:expr) => {
if unsafe { __slate_intrinsic_f217c4720f04f6db($n) } { if ($n) < 2 { 0i32 } else { 63i32.wrapping_sub($n.leading_zeros() as i32) } } else { if (core::mem::size_of::<u64>() as u64) <= 4 { __ilog2_u32(($n) as u32) } else { __ilog2_u64($n) } }
};
}

macro_rules! in_hardirq {
() => {
hardirq_count!()
};
}

macro_rules! in_task {
() => {
(preempt_count() as u64) & (__IRQ_MASK!(1i32.wrapping_add(3i32.wrapping_mul(IS_ENABLED__2!()))) << 28i32 | __IRQ_MASK!(4i32) << 24i32 | 256u64) == 0
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

macro_rules! kcsan_mb {
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

macro_rules! likely {
($x:expr) => {
!(!$x) as i64
};
}

macro_rules! local_irq_disable {
() => {
loop {
raw_local_irq_disable!();
if true {
break;
}
}
};
}

macro_rules! local_irq_enable {
() => {
loop {
raw_local_irq_enable!();
if true {
break;
}
}
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

macro_rules! lockdep_assert_irqs_disabled {
() => {
while false {
}
};
}

macro_rules! mempool_init_slab_pool {
($_pool:expr, $_min_nr:expr, $_kc:expr) => {
alloc_hooks__2!(unsafe { mempool_init_noprof($_pool, ($_min_nr) as i32, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(u32, *mut ()) -> *mut ()>>(mempool_alloc_slab as *const ()) }, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut ())>>(mempool_free_slab as *const ()) }, ($_kc) as *mut ()) })
};
}

macro_rules! mempool_init_slab_pool__2 {
($_pool:expr, $_min_nr:expr, $_kc:expr) => {
alloc_hooks__2!(unsafe { mempool_init_noprof($_pool, $_min_nr, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(u32, *mut ()) -> *mut ()>>(mempool_alloc_slab as *const ()) }, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut ())>>(mempool_free_slab as *const ()) }, ($_kc) as *mut ()) })
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
macro_rules! min_t__3 {
($type:ty, $x:expr, $y:expr) => {
__cmp_once__3!($type, $x, $y)
};
}

macro_rules! offset_in_page {
($p:expr) => {
(($p) as u64) & !(!(_AC!() << 12i32).wrapping_sub(1))
};
}

macro_rules! offsetof__13 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, cpuhp_dead) as u64
};
}

macro_rules! offsetof__43 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, rescue_work) as u64
};
}

macro_rules! offsetof__20 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, work) as u64
};
}

macro_rules! offsetof__70 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, bi_max_vecs) as u64
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

macro_rules! page_ref_tracepoint_active {
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

macro_rules! per_cpu_offset {
($x:expr) => {
unsafe { __per_cpu_offset[($x) as usize] }
};
}

macro_rules! per_cpu_ptr__4 {
($ptr:expr, $cpu:expr) => {
{
let mut __v: *mut bio_complete_batch;
__verify_pcpu_ptr__18!();
__v = (((($ptr) as u64) as *mut bio_complete_batch) as u64).wrapping_add(per_cpu_offset!($cpu)) as *mut bio_complete_batch;
__v
}
};
}

macro_rules! per_cpu_ptr__8 {
($ptr:expr, $cpu:expr) => {
{
let mut __v: *mut bio_alloc_cache;
__verify_pcpu_ptr__29!();
__v = (((($ptr) as u64) as *mut bio_alloc_cache) as u64).wrapping_add(per_cpu_offset!($cpu)) as *mut bio_alloc_cache;
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

macro_rules! preemptible {
() => {
{
let __v: bool = preempt_count() == 0 && irqs_disabled!() == 0;
__v
}
};
}

macro_rules! raw_check_bogus_irq_restore {
() => {
while false {
}
};
}

/// Wrap the arch provided IRQ routines to provide appropriate checks.
macro_rules! raw_local_irq_disable {
() => {
arch_local_irq_disable();
};
}

macro_rules! raw_local_irq_enable {
() => {
arch_local_irq_enable();
};
}

macro_rules! raw_local_irq_restore {
($flags:expr) => {
loop {
typecheck__2!(u64);
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
typecheck__2!(u64);
$flags = arch_local_irq_save();
if true {
break;
}
}
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

/// rounddown_pow_of_two - round the given value down to nearest power of two
/// @n: parameter
///
/// round the given value down to the nearest power of two
/// - the result is undefined when n == 0
/// - this can be used to initialise global variables from constant data
macro_rules! rounddown_pow_of_two {
($n:expr) => {
if unsafe { __slate_intrinsic_f217c4720f04f6db($n) } { 1u64 << ilog2!($n) } else { __rounddown_pow_of_two($n) }
};
}

/// sizeof_field() - Report the size of a struct field in bytes
///
/// @TYPE: The structure containing the field of interest
/// @MEMBER: The field to return the size of
macro_rules! sizeof_field__2 {
() => {
(const { core::mem::size_of::<u16>() }) as u64
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
__release!(spinlock_check($lock));
__acquire!($lock);
if true {
break;
}
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

macro_rules! test_bit__4 {
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
macro_rules! typecheck__3 {
($type:ty) => {
let mut __v: i32;
let mut __v2: $type = 0;
let mut __v3: $type = 0;
core::ptr::addr_of_mut!(__v2) == core::ptr::addr_of_mut!(__v3);
__v = 1;
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

/// virt_to_page(kaddr) returns a valid pointer if and only if
/// virt_addr_valid(kaddr) returns true.
macro_rules! virt_to_page {
($kaddr:expr) => {
((unsafe { vmemmap_base }) as *mut page).wrapping_offset((__pa!($kaddr) >> 12i32) as isize)
};
}

macro_rules! xen_domain {
() => {
false
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
    static mut __cpu_possible_mask: cpumask;
    static mut __cpu_online_mask: cpumask;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut phys_base: u64;
    static mut page_offset_base: u64;
    static mut vmemmap_base: u64;
    static mut cpuinfo_op: seq_operations;
    static mut cpu_number: i32;
    static mut RCU: *const __ctx_lock_RCU;
    static mut system_percpu_wq: *mut workqueue_struct;
    static mut zero_page_pfn: u64;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut kstat: kernel_stat;
    static mut tracepoint_srcu: srcu_struct;
    static mut __SCK__tp_func_block_bio_complete: static_call_key;
    static mut __tracepoint_block_bio_complete: tracepoint;
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn panic(arg0: *const u8, ...);
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn __bad_size_call_parameter();
    fn ____wrong_branch_error() -> bool;
    fn _find_next_bit(arg0: *const u64, arg1: u64, arg2: u64) -> u64;
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn snprintf(arg0: *mut u8, arg1: u64, arg2: *const u8, ...) -> i32;
    fn __cond_resched() -> i32;
    fn __SCT__might_resched() -> i32;
    fn __compiletime_assert_16();
    fn __compiletime_assert_17();
    fn __compiletime_assert_57();
    fn _raw_spin_lock(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irqsave(arg0: *mut raw_spinlock) -> u64;
    fn _raw_spin_unlock(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irqrestore(arg0: *mut raw_spinlock, arg1: u64);
    fn mutex_lock(arg0: *mut mutex);
    fn mutex_unlock(arg0: *mut mutex);
    fn __rcu_read_lock();
    fn __rcu_read_unlock();
    fn rcu_is_watching() -> bool;
    fn wait_for_completion_io(arg0: *mut completion);
    fn wait_for_completion_io_timeout(arg0: *mut completion, arg1: u64) -> u64;
    fn complete(arg0: *mut completion);
    fn pcpu_alloc_noprof(arg0: u64, arg1: u64, arg2: bool, arg3: u32) -> *mut ();
    fn free_percpu(arg0: *mut ());
    fn alloc_workqueue_noprof(arg0: *const u8, arg1: u32, arg2: i32, ...) -> *mut workqueue_struct;
    fn destroy_workqueue(arg0: *mut workqueue_struct);
    fn queue_work_on(arg0: i32, arg1: *mut workqueue_struct, arg2: *mut work_struct) -> bool;
    fn disable_work_sync(arg0: *mut work_struct) -> bool;
    fn enable_work(arg0: *mut work_struct) -> bool;
    fn __compiletime_assert_167();
    fn __srcu_check_read_flavor(arg0: *mut srcu_struct, arg1: i32);
    fn __compiletime_assert_187();
    fn folio_alloc_noprof(arg0: u32, arg1: u32) -> *mut folio;
    fn __free_pages(arg0: *mut page, arg1: u32);
    fn __compiletime_assert_211();
    fn __kmem_cache_create_args(arg0: *const u8, arg1: u32, arg2: *mut kmem_cache_args, arg3: u32) -> *mut kmem_cache;
    fn kmem_cache_destroy(arg0: *mut kmem_cache);
    fn kfree(arg0: *const ());
    fn __compiletime_assert_257();
    fn __builtin_unreachable();
    fn kmem_cache_alloc_noprof(arg0: *mut kmem_cache, arg1: u32) -> *mut ();
    fn kmem_cache_free(arg0: *mut kmem_cache, arg1: *mut ());
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn kmem_cache_size(arg0: *mut kmem_cache) -> u32;
    fn cache_get_priv_group(arg0: *mut cacheinfo) -> *const attribute_group;
    fn __bad_copy_from();
    fn __bad_copy_to();
    fn __copy_overflow(arg0: i32, arg1: u64);
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_376();
    fn xa_load(arg0: *mut xarray, arg1: u64) -> *mut ();
    fn xa_store(arg0: *mut xarray, arg1: u64, arg2: *mut (), arg3: u32) -> *mut ();
    fn xa_erase(arg0: *mut xarray, arg1: u64) -> *mut ();
    fn __compiletime_assert_419();
    fn vmalloc_to_page(arg0: *const ()) -> *mut page;
    fn is_vmalloc_addr(arg0: *const ()) -> bool;
    fn __folio_put(arg0: *mut folio);
    fn unpin_user_page(arg0: *mut page);
    fn unpin_user_folio(arg0: *mut folio, arg1: u64);
    fn folio_mark_dirty(arg0: *mut folio) -> bool;
    fn folio_mark_dirty_lock(arg0: *mut folio) -> bool;
    fn __compiletime_assert_553();
    fn __compiletime_assert_570();
    fn __compiletime_assert_573();
    fn iov_iter_advance(arg0: *mut iov_iter, arg1: u64);
    fn iov_iter_revert(arg0: *mut iov_iter, arg1: u64);
    fn copy_page_from_iter(arg0: *mut page, arg1: u64, arg2: u64, arg3: *mut iov_iter) -> u64;
    fn copy_folio_from_iter_atomic(arg0: *mut folio, arg1: u64, arg2: u64, arg3: *mut iov_iter) -> u64;
    fn _copy_to_iter(arg0: *const (), arg1: u64, arg2: *mut iov_iter) -> u64;
    fn iov_iter_bvec(arg0: *mut iov_iter, arg1: u32, arg2: *const bio_vec, arg3: u64, arg4: u64);
    fn iov_iter_extract_bvecs(arg0: *mut iov_iter, arg1: *mut bio_vec, arg2: u64, arg3: *mut u16, arg4: u16, arg5: u32, arg6: u32) -> i64;
    fn __folio_lock(arg0: *mut folio);
    fn folio_unlock(arg0: *mut folio);
    fn mempool_exit(arg0: *mut mempool);
    fn mempool_init_noprof(arg0: *mut mempool, arg1: i32, arg2: Option<unsafe extern "C" fn(u32, *mut ()) -> *mut ()>, arg3: Option<unsafe extern "C" fn(*mut (), *mut ())>, arg4: *mut ()) -> i32;
    fn mempool_alloc_noprof(arg0: *mut mempool, arg1: u32) -> *mut ();
    fn mempool_free(arg0: *mut (), arg1: *mut mempool);
    fn mempool_alloc_slab(arg0: u32, arg1: *mut ()) -> *mut ();
    fn mempool_free_slab(arg0: *mut (), arg1: *mut ());
    fn __compiletime_assert_628();
    fn __compiletime_assert_631();
    fn submit_bio(arg0: *mut bio);
    fn __compiletime_assert_632();
    fn bio_associate_blkg(arg0: *mut bio);
    fn bio_clone_blkg_association(arg0: *mut bio, arg1: *mut bio);
    fn submit_bio_noacct(arg0: *mut bio);
    fn blk_status_to_errno(arg0: u8) -> i32;
    fn __bio_crypt_clone(arg0: *mut bio, arg1: *mut bio, arg2: u32) -> i32;
    fn __cpuhp_setup_state(arg0: i32, arg1: *const u8, arg2: bool, arg3: Option<unsafe extern "C" fn(u32) -> i32>, arg4: Option<unsafe extern "C" fn(u32) -> i32>, arg5: bool) -> i32;
    fn __cpuhp_state_add_instance(arg0: i32, arg1: *mut hlist_node, arg2: bool) -> i32;
    fn __cpuhp_state_remove_instance(arg0: i32, arg1: *mut hlist_node, arg2: bool) -> i32;
    fn __traceiter_block_bio_complete(arg0: *mut (), arg1: *mut request_queue, arg2: *mut bio) -> i32;
    fn __SCT__tp_func_block_bio_complete(arg0: *mut (), arg1: *mut request_queue, arg2: *mut bio) -> i32;
    fn __compiletime_assert_749();
    fn xen_biovec_phys_mergeable(arg0: *const bio_vec, arg1: *const page) -> bool;
    fn __bio_crypt_advance(arg0: *mut bio, arg1: u32);
    fn __bio_crypt_free_ctx(arg0: *mut bio);
    fn __rq_qos_done_bio(arg0: *mut rq_qos, arg1: *mut bio);
    fn __compiletime_assert_893();
    fn __compiletime_assert_901();
    fn __compiletime_assert_902();
    fn __compiletime_assert_926();
    fn __compiletime_assert_936();
    fn __compiletime_assert_945();
    fn __compiletime_assert_949();
    fn __compiletime_assert_960();
    fn __compiletime_assert_961();
    fn __compiletime_assert_962();
    fn __compiletime_assert_982();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_1c5364fcd9da51fe(arg0: u32) -> bool;
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
    #[link_name = "llvm.objectsize.i64.p0"]
    fn __slate_intrinsic_2913c0303de876b3(arg0: *const (), arg1: bool, arg2: bool, arg3: bool) -> u64;
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_a39e427d3b6c2df4(arg0: i32) -> bool;
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
struct static_call_site {
    addr: i32,
    key: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_static_call_types_h_65 {
    r#type: u64,
    mods: *mut static_call_mod,
    sites: *mut static_call_site,
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
#[inline(always)]
fn __must_check_overflow(overflow: bool) -> bool {
unlikely__2!(overflow) != 0
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

#[inline(always)]
fn __release_ctx_lock__1(var: *const raw_spinlock) {
}

#[inline(always)]
fn __release_shared_ctx_lock__1(var: &raw_spinlock) {
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
struct bug_entry {
    bug_addr_disp: i32,
    format_disp: i32,
    file_disp: i32,
    line: u16,
    flags: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct static_key_mod {
}


// IS_ERR_VALUE - Detect an error pointer.
// @x: The pointer to check.
//
// Like IS_ERR(), but does not generate a compiler warning if result is unused.
#[inline(always)]
fn ERR_PTR(error: i64) -> *mut () {
error as *mut ()
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
fn arch_static_branch__const1(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_block_bio_complete, const 0 as u8, label {
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
    __slate_bits_0: __slate_bits::__SlateBits103U0,
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

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_find_h_67 {
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

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits104U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_bitmap {
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
struct perf_event {
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

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


#[inline(always)]
fn arch_atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
__WRITE_ONCE__2!((*v).counter, i);
}

#[inline(always)]
fn arch_atomic_inc(mut v: *mut __SlateRecord_types_h_190) {
unsafe {
core::arch::asm!("lock incl ({0:r})", in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
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


#[inline(always)]
fn raw_atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
arch_atomic_read(v)
}

#[inline(always)]
fn raw_atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
arch_atomic_set(v, i);
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
}


#[inline(always)]
fn raw_atomic_inc(v: *mut __SlateRecord_types_h_190) {
arch_atomic_inc(v);
}

#[inline(always)]
fn raw_atomic_dec_and_test(v: *mut __SlateRecord_types_h_190) -> bool {
arch_atomic_dec_and_test(v)
}

#[inline(always)]
fn raw_atomic64_inc(v: *mut __SlateRecord_types_h_197) {
arch_atomic64_inc(v);
}

#[inline(always)]
fn raw_atomic64_sub_and_test(i: i64, v: *mut __SlateRecord_types_h_197) -> bool {
arch_atomic64_sub_and_test(i, v)
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

#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
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
fn atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
instrument_atomic_read(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_read(v)
}

#[inline(always)]
fn atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
instrument_atomic_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_set(v, i);
}

#[inline(always)]
fn atomic_inc(v: *mut __SlateRecord_types_h_190) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_inc(v);
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
fn atomic_long_sub_and_test(i: i64, v: *mut __SlateRecord_types_h_197) -> bool {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_sub_and_test(i, v)
}

#[inline(always)]
fn const_test_bit(nr: u64, addr: *const u64) -> bool {
let p: *const u64 = (addr as *const u64).wrapping_offset(BIT_WORD!(nr) as isize);
let mask: u64 = BIT_MASK!(nr);
let val: u64 = unsafe { *p };
val & mask != 0
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

#[repr(C)]
#[derive(Clone, Copy)]
struct futex_pi_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct robust_list_head {
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

#[repr(C)]
#[derive(Clone, Copy)]
struct futex_private_hash {
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
fn _test_bit(nr: u64, addr: *const u64) -> bool {
instrument_atomic_read(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_bit(nr, addr)
}

fn test_and_set_bit_lock(nr: i64, addr: *mut u64) -> bool {
instrument_atomic_read_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_and_set_bit_lock(nr, addr)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pid_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sem_undo_list {
}


// The following macros are non-atomic versions of their non-underscored
// counterparts.
// Check that the bitops prototypes are sane
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

#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer_clock_base {
}


#[inline(always)]
fn __ilog2_u64(n: u64) -> i32 {
fls64(n).wrapping_sub(1)
}

#[inline(always)]
fn is_power_of_2(n: u64) -> bool {
n.wrapping_sub(1) < n ^ n.wrapping_sub(1)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct seccomp_filter {
}


fn __rounddown_pow_of_two(n: u64) -> u64 {
1u64 << fls_long(n).wrapping_sub(1)
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
static mut __UNIQUE_ID_addressable___SCK__might_resched_15: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__might_resched) }) as *mut ();

fn INIT_LIST_HEAD(list: *mut list_head) {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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

// list_entry - get the struct for this entry
// @ptr:	the &struct list_head pointer.
// @type:	the type of the struct this is embedded in.
// @member:	the name of the list_head within the struct.
// list_first_entry - get the first element from a list
// @ptr:	the list head to take the element from.
// @type:	the type of the struct this is embedded in.
// @member:	the name of the list_head within the struct.
//
// Note, that list is expected to be not empty.
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
fn cpu_max_bits_warn(cpu: u32, bits: u32) {
}

#[inline(always)]
fn cpumask_check(cpu: u32) -> u32 {
cpu_max_bits_warn(cpu, 64u32);
cpu
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


// for_each_cpu - iterate over every cpu in a mask
// @cpu: the (optionally unsigned) integer iterator
// @mask: the cpumask pointer
//
// After the loop, cpu is >= nr_cpu_ids.
#[inline(always)]
fn cpumask_test_cpu(cpu: i32, cpumask: *const cpumask) -> bool {
test_bit__2!(cpumask_check(cpu as u32), unsafe { (*cpumask).bits.as_ptr() as *const u64 }) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_signal_types_h_14 {
    si_signo: i32,
    si_errno: i32,
    si_code: i32,
    _sifields: __sifields,
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
fn arch_local_irq_disable() {
native_irq_disable();
}

#[inline(always)]
fn arch_local_irq_enable() {
native_irq_enable();
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

#[inline(always)]
fn __phys_addr_nodebug(mut x: u64) -> u64 {
let y: u64 = x.wrapping_sub(_AC__3!());
// use the carry flag to determine if x was < __START_KERNEL_map
x = y.wrapping_add(if x > y { unsafe { phys_base } } else { _AC__3!().wrapping_sub(unsafe { page_offset_base }) });
x
}

#[repr(C)]
#[derive(Clone, Copy)]
struct workqueue_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct range {
    start: u64,
    end: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
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
    __slate_bits_0: __slate_bits::__SlateBits122U0,
    pkru: u32,
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
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


#[inline(always)]
fn __acquire_ctx_lock__2(var: *const spinlock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct audit_context {
}


#[inline(always)]
fn __acquire_shared_ctx_lock__2(var: &spinlock) {
}

#[inline(always)]
fn __release_ctx_lock__2(var: &spinlock) {
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
fn __release_shared_ctx_lock__2(var: &spinlock) {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct spinlock {
    __slate_anon_0: __SlateRecord_spinlock_types_h_18,
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
fn __acquire_ctx_lock__3(var: &rwlock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rq {
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
struct rwlock {
    raw_lock: qrwlock,
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
    __slate_bits_0: __slate_bits::__SlateBits161U0,
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
struct rq_flags {
}


#[inline(always)]
fn spin_unlock_irq(lock: *mut spinlock) {
raw_spin_unlock_irq!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
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
fn spin_unlock_irqrestore(lock: *mut spinlock, flags: u64) {
raw_spin_unlock_irqrestore!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) }, flags);
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
struct wait_queue_head {
    lock: spinlock,
    head: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct optimistic_spin_queue {
    tail: __SlateRecord_types_h_190,
}


#[inline(always)]
fn __acquire_ctx_lock__4(var: &mutex) {
}

#[inline(always)]
fn __acquire_shared_ctx_lock__4(var: &mutex) {
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
fn __release_ctx_lock__4(var: &mutex) {
}

#[inline(always)]
fn __release_shared_ctx_lock__4(var: &mutex) {
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
struct seqcount {
    sequence: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqcount_spinlock {
    seqcount: seqcount,
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

#[inline(always)]
fn __release_shared_ctx_lock__5(var: &seqlock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqlock {
    seqcount: seqcount_spinlock,
    lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
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
struct uprobe {
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
struct uprobe_xol_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sysv_sem {
    undo_list: *mut sem_undo_list,
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
struct timerqueue_linked_node {
    node: rb_node_linked,
    expires: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
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
struct task_cputime {
    stime: u64,
    utime: u64,
    sum_exec_runtime: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mem_cgroup {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_signal_h_25 {
    sig: [u64; 1],
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
struct kernel_siginfo {
    __slate_anon_0: __SlateRecord_signal_types_h_14,
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
struct posix_cputimers_work {
    work: callback_head,
    mutex: mutex,
    scheduled: u32,
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
struct rseq_ids {
    __slate_anon_0: __SlateRecord_rseq_types_h_74,
    node_id: u32,
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
struct __SlateRecord_uidgid_types_h_9 {
    val: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uidgid_types_h_13 {
    val: u32,
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
    __slate_bits_0: __slate_bits::__SlateBits293U0,
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
struct mnt_idmap {
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


// Per process flags
// I'm a virtual CPU
// I am an IDLE thread
// Allocating memory to free memory. See memalloc_noreclaim_save()
// Kernel thread cloned from userspace thread
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
struct assoc_array_ptr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct net {
}


#[inline(always)]
fn __acquire_ctx_lock__6(var: &__ctx_lock_RCU) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct key_type {
}


#[inline(always)]
fn __acquire_shared_ctx_lock__6(var: *const __ctx_lock_RCU) {
}

#[inline(always)]
fn __release_ctx_lock__6(var: &__ctx_lock_RCU) {
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


#[inline(always)]
fn __release_shared_ctx_lock__6(var: *const __ctx_lock_RCU) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __ctx_lock_RCU {
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
struct rcu_gp_seq {
    norm: u64,
    exp: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_cred_h_147 {
    non_rcu: i32,
    rcu: callback_head,
}


// Defined as a macro as it is a very low level header included from
// areas that don't even know about current.  This gives the rcu_read_lock()
// nesting depth, but makes sense only if CONFIG_PREEMPT_RCU -- in other
// types of kernel builds, the rcu_read_lock() nesting depth is unknowable.
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
#[inline(always)]
fn rcu_read_lock() {
unsafe { __rcu_read_lock() };
__acquire_shared!(unsafe { RCU });
rcu_lock_acquire!();
RCU_LOCKDEP_WARN!(!unsafe { rcu_is_watching() });
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pidfs_attr {
}


fn rcu_read_unlock() {
RCU_LOCKDEP_WARN!(!unsafe { rcu_is_watching() });
rcu_lock_release!(); // Keep acq info for rls diags.
__release_shared!(unsafe { RCU });
unsafe { __rcu_read_unlock() };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pid_h_62 {
    ino: u64,
    pidfs_hash: rhash_head,
    stashed: *mut dentry,
    attr: *mut pidfs_attr,
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

#[inline(always)]
fn __release_shared_ctx_lock__7(var: &rw_semaphore) {
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
struct tty_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swait_queue_head {
    lock: raw_spinlock,
    task_list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_audit_buf {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct completion {
    done: u32,
    wait: swait_queue_head,
}


// DECLARE_COMPLETION - declare and initialize a completion structure
// @work:  identifier for the completion structure
//
// This macro declares and initializes a completion structure. Generally used
// for static declarations. You should use the _ONSTACK variant for automatic
// variables.
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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

#[inline(always)]
fn __release_shared_ctx_lock__8(var: *const srcu_struct) {
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


#[inline(always)]
fn class_srcu_fast_notrace_destructor(_T: *mut __SlateRecord_srcu_h_645) {
srcu_read_unlock_fast_notrace(unsafe { (*_T).lock }, unsafe { (*_T).scp });
}

#[repr(C)]
#[derive(Clone, Copy)]
struct xattr_handler {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fserror_event {
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
struct fiemap_extent_info {
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
    __slate_bits_0: __slate_bits::__SlateBits421U0,
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
// MMF_HAS_PINNED: Whether this mm has pinned any pages.  This can be either
// replaced in the future by mm.pinned_vm when it becomes stable, or grow into
// a counter on its own. We're aggresive on this bit for now: even if the
// pinned pages were unpinned later on, we'll still keep this bit set for the
// lifecycle of this mm, just for simplicity.
// FOLL_PIN has run, never cleared
#[repr(C)]
#[derive(Clone, Copy)]
struct posix_acl {
}


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
if !(__native_word__2!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
fn folio_test_dirty(folio: *const folio) -> bool {
test_bit__4!(4i32, const_folio_flags(folio, 0u32)) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_cmd {
}


fn PageHighMem(page: *const page) -> i32 {
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
fn folio_test_head(folio: *const folio) -> bool {
test_bit__4!(6i32, const_folio_flags(folio, 0u32)) != 0
}

fn folio_test_large(folio: *const folio) -> bool {
folio_test_head(folio)
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
fn __acquire_ctx_lock__9(var: &local_lock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
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
struct local_lock {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
}


#[inline(always)]
fn __acquire_ctx_lock__10(var: &local_trylock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct sock {
}


#[inline(always)]
fn __acquire_shared_ctx_lock__10(var: &local_trylock) {
}

#[inline(always)]
fn __release_ctx_lock__10(var: &local_trylock) {
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
fn __release_shared_ctx_lock__10(var: &local_trylock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct local_trylock {
    acquired: u8,
}


fn memdesc_is_zone_device(mdf: *const __SlateRecord_mm_types_h_41) -> bool {
false
}

fn zone_device_pages_have_same_pgmap(a: *const page, b: *const page) -> bool {
true
}

fn folio_is_zone_device(folio: *const folio) -> bool {
memdesc_is_zone_device(unsafe { core::ptr::addr_of!((*folio).__slate_anon_0.__slate_anon_0.flags) })
}

// Maximum number of zones on a zonelist
// zone_idx() returns 0 for the ZONE_DMA zone, 1 for the ZONE_NORMAL zone, etc.
#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_ref_data {
    count: __SlateRecord_types_h_197,
    release: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    confirm_switch: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    __slate_bits_0: __slate_bits::__SlateBits456U0,
    rcu: callback_head,
    r#ref: *mut percpu_ref,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_ref {
    percpu_count_ptr: u64,
    data: *mut percpu_ref_data,
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
unsafe { __compiletime_assert_211() };
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
fn __acquire_ctx_lock__11(var: &__context_bitlock) {
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
fn __acquire_shared_ctx_lock__11(var: &__context_bitlock) {
}

#[inline(always)]
fn __release_ctx_lock__11(var: &__context_bitlock) {
}

#[inline(always)]
fn __release_shared_ctx_lock__11(var: &__context_bitlock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __context_bitlock {
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


fn __page_ref_mod_and_test(page: *mut page, v: i32, ret: i32) {
}

fn page_ref_count(page: &page) -> i32 {
atomic_read(unsafe { core::ptr::addr_of!((*(page as *const page))._refcount) })
}

#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_taskset {
}


fn page_ref_dec_and_test(page: *mut page) -> i32 {
let ret: i32 = atomic_dec_and_test(unsafe { core::ptr::addr_of_mut!((*page)._refcount) }) as i32;
if page_ref_tracepoint_active!() != 0 {
__page_ref_mod_and_test(page, 1i32.wrapping_neg(), ret);
}
ret
}

// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
fn is_zero_pfn(pfn: u64) -> i32 {
(pfn == unsafe { zero_page_pfn }) as i32
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


fn folio_is_pci_p2pdma(folio: *const folio) -> bool {
let __v0: bool = IS_ENABLED!() != 0 && folio_is_zone_device(folio);
__v0 && ((unsafe { (*(*folio).__slate_anon_0.__slate_anon_0.__slate_anon_1.pgmap).r#type }) as u32) == 5
}

fn is_pci_p2pdma_page(page: *const page) -> bool {
IS_ENABLED!() != 0 && folio_is_pci_p2pdma(page_folio__2!(page))
}

// define SLAB_HWCACHE_ALIGN - Align objects on cache line boundaries.
//
// Sufficiently large objects are aligned on cache line boundary. For object
// size smaller than a half of cache line size, the alignment is on the half of
// cache line size. In general, if object size is smaller than 1/2^n of cache
// line size, the alignment is adjusted to 1/2^n.
//
// If explicit alignment is also requested by the respective
// &struct kmem_cache_args field, the greater of both is alignments is applied.
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
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("255: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 255b - ., 3; .popsection\n# {0}", const 255 as i32, options(nostack, att_syntax));
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
core::arch::asm!("256: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 256b - ., 4; .popsection\n# {0}", const 256 as i32, options(nostack, att_syntax));
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
unsafe { __compiletime_assert_257() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
} else {
loop {
unsafe {
core::arch::asm!("258: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 258b - ., 3; .popsection\n# {0}", const 258 as i32, options(nostack, att_syntax));
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

#[repr(C)]
#[derive(Clone, Copy)]
struct wake_irq {
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
// All kzalloc aliases for kmalloc_(obj|objs|flex).
fn _kzalloc_noprof(size: u64, flags: u32, token: __SlateRecord_slab_h_532) -> *mut () {
_kmalloc_noprof(size, flags | (BIT!(8i32) as u32), token)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_qos {
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


fn sigismember(set: *mut __SlateRecord_signal_h_25, _sig: i32) -> i32 {
let sig: u64 = _sig.wrapping_sub(1) as u64;
if 1i32 == 1 {
return (1 & (unsafe { *((*set).sig.as_mut_ptr() as *mut u64).wrapping_offset(0 as isize) }) >> sig) as i32;
} else {
return (1 & (unsafe { *((*set).sig.as_mut_ptr() as *mut u64).wrapping_offset((sig / 64) as isize) }) >> sig % 64) as i32;
}
}

fn check_object_size(ptr: *const (), n: u64, to_user: bool) {
}

fn copy_overflow(size: i32, count: u64) {
if IS_ENABLED__2!() != 0 {
unsafe { __copy_overflow(size, count) };
}
}

#[inline(always)]
fn check_copy_size(addr: *const (), bytes: u64, is_source: bool) -> bool {
let sz: i32 = (unsafe { __slate_intrinsic_2913c0303de876b3(addr, const { false as bool }, const { false as bool }, const { false as bool }) }) as i32;
if unlikely__2!(sz >= 0 && (sz as u64) < bytes) != 0 {
if !unsafe { __slate_intrinsic_f217c4720f04f6db(bytes) } {
copy_overflow(sz, bytes);
} else {
if is_source {
unsafe { __bad_copy_from() };
} else {
unsafe { __bad_copy_to() };
}
}
return false;
}
let mut __v0: i64;
let __ret_warn_on: i32 = (bytes > 2147483647u64) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("313: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 313b - ., 3; .popsection\n# {0}", const 313 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_check_copy_size_1, sym __slate_asm_string_check_copy_size_2, const 57 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("314: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 314b - ., 4; .popsection\n# {0}", const 314 as i32, options(nostack, att_syntax));
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
check_object_size(addr, bytes, is_source);
true
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
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
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
struct module_param_attrs {
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
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct upid {
    nr: i32,
    ns: *mut pid_namespace,
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
union __SlateRecord_blk_types_h_239 {
    bi_cookie: u32,
    __bi_nr_segments: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct thread_group_cputimer {
    cputime_atomic: task_cputime_atomic,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_cluster_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct core_thread {
    task: *mut task_struct,
    next: *mut core_thread,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_ops {
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
    __slate_bits_0: __slate_bits::__SlateBits296U0,
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

fn __fatal_signal_pending(p: *mut task_struct) -> i32 {
unlikely!(sigismember(unsafe { core::ptr::addr_of_mut!((*p).pending.signal) }, 9i32)) as i32
}

fn fatal_signal_pending(p: *mut task_struct) -> i32 {
let __v0: bool = task_sigpending(p) != 0 && __fatal_signal_pending(p) != 0;
__v0 as i32
}

fn task_rlimit(task: *const task_struct, limit: u32) -> u64 {
let mut __v0: u64;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct elevator_queue {
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
fn folio_large_order(folio: *const folio) -> u32 {
((unsafe { (*folio).__slate_anon_1.__slate_anon_0._flags_1 }) & 255) as u32
}

fn folio_large_nr_pages(folio: *const folio) -> u64 {
(1i64 << folio_large_order(folio)) as u64
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pr_ops {
}


fn folio_order(folio: *const folio) -> u32 {
if !folio_test_large(folio) {
return 0;
}
folio_large_order(folio)
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


fn xa_mk_internal(v: u64) -> *mut () {
(v << 2i32 | 2) as *mut ()
}

#[repr(C)]
#[derive(Clone, Copy)]
struct timer_rand_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cdrom_device_info {
}


fn xa_is_internal(entry: *const ()) -> bool {
(entry as u64) & 3 == 2
}

fn xa_is_err(entry: *const ()) -> bool {
let __v0: bool = xa_is_internal(entry) && entry >= (xa_mk_internal(4095i32.wrapping_neg() as u64) as *const ());
__v0
}

fn xa_err(entry: *mut ()) -> i32 {
// xa_to_internal() would not do sign extension.
if xa_is_err(entry as *const ()) {
return ((entry as i64) >> 2i32) as i32;
}
0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_ctx {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xarray {
    xa_lock: spinlock,
    xa_flags: u32,
    xa_head: *mut (),
}


// DEFINE_XARRAY_FLAGS() - Define an XArray with custom flags.
// @name: A string that names your XArray.
// @flags: XA_FLAG values.
//
// This is intended for file scope definitions of XArrays.  It declares
// and initialises an empty XArray with the chosen name and flags.  It is
// equivalent to calling xa_init_flags() on the array, but it does the
// initialisation at compiletime instead of runtime.
// DEFINE_XARRAY() - Define an XArray.
// @name: A string that names your XArray.
//
// This is intended for file scope definitions of XArrays.  It declares
// and initialises an empty XArray with the chosen name.  It is equivalent
// to calling xa_init() on the array, but it does the initialisation at
// compiletime instead of runtime.
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
union __SlateRecord_vmalloc_h_53 {
    next: *mut vm_struct,
    llnode: llist_node,
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
union __SlateRecord_blk_cgroup_h_80 {
    async_bio_work: work_struct,
    free_work: work_struct,
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
struct __SlateRecord_bio_c_2028 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bio_c_2177 {
}


#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

mod __slate_bits {
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits579U0 {
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
    pub struct __SlateBits456U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits293U0 {
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
    pub struct __SlateBits845U0 {
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
    pub struct __SlateBits421U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits666U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits705U0 {
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
    pub struct __SlateBits296U0 {
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
    pub struct __SlateBits104U0 {
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
    pub struct __SlateBits122U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits617U0 {
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
    pub struct __SlateBits103U0 {
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
    pub struct __SlateBits161U0 {
        #[bits(1)]
        pub faultable: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
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
    __slate_bits_0: __slate_bits::__SlateBits666U0,
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
    __slate_bits_0: __slate_bits::__SlateBits617U0,
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


fn is_huge_zero_folio(folio: *const folio) -> bool {
false
}

fn put_page_testzero(page: *mut page) -> i32 {
VM_BUG_ON_PAGE!();
page_ref_dec_and_test(page)
}

fn folio_put_testzero(folio: *mut folio) -> i32 {
put_page_testzero(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.page) })
}

// 127: arbitrary random number, small enough to assemble well
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
fn is_zero_page(page: *const page) -> bool {
is_zero_pfn(((unsafe { page.offset_from((vmemmap_base as *mut page) as *const page) }) as i64) as u64) != 0
}

fn is_zero_folio(folio: *const folio) -> bool {
is_zero_page(unsafe { core::ptr::addr_of!((*folio).__slate_anon_0.page) })
}

fn folio_nr_pages(folio: *const folio) -> u64 {
if !folio_test_large(folio) {
return 1;
}
folio_large_nr_pages(folio)
}

core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_fs_bio_set: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_bio_uninit: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_bio_init: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_bio_reset: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_bio_reuse: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_bio_chain: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_blk_next_bio: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_bio_alloc_bioset: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {7} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_bio_kmalloc: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {8} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_zero_fill_bio: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {9} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_bio_put: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {10} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_bio_alloc_clone: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {11} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_bio_init_clone: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {12} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol___bio_add_page: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {13} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_bio_add_virt_nofail: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {14} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_bio_add_page: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {15} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_bio_add_folio_nofail: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {16} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_bio_add_folio: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {17} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_bio_add_vmalloc_chunk: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {18} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_bio_add_vmalloc: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {19} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol___bio_release_pages: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {20} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_bio_await: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {21} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_submit_bio_wait: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {22} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_bdev_rw_virt: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {23} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol___bio_advance: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {24} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_bio_copy_data: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {25} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_bio_free_pages: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {26} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol___bio_complete_in_task: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {27} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_bio_endio: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {28} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_bio_split: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {29} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_bio_trim: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {30} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_bioset_exit: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {31} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_bioset_init: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {32} ; .previous\n.section\t\".initcall4.init\", \"a\"\t\t\n__initcall__kmod_bio__983_2218_init_bio4:\t\t\t\n.long\t{33} - .\t\n.previous\t\t\t\t\t\n", sym fs_bio_set, sym bio_uninit, sym bio_init, sym bio_reset, sym bio_reuse, sym bio_chain, sym blk_next_bio, sym bio_alloc_bioset, sym bio_kmalloc, sym zero_fill_bio, sym bio_put, sym bio_alloc_clone, sym bio_init_clone, sym __bio_add_page, sym bio_add_virt_nofail, sym bio_add_page, sym bio_add_folio_nofail, sym bio_add_folio, sym bio_add_vmalloc_chunk, sym bio_add_vmalloc, sym __bio_release_pages, sym bio_await, sym submit_bio_wait, sym bdev_rw_virt, sym __bio_advance, sym bio_copy_data, sym bio_free_pages, sym __bio_complete_in_task, sym bio_endio, sym bio_split, sym bio_trim, sym bioset_exit, sym bioset_init, sym init_bio, options(att_syntax));

fn folio_next(folio: *mut folio) -> *mut folio {
folio_page!(folio, folio_nr_pages(folio as *const folio)) as *mut folio
}

fn folio_size(folio: *const folio) -> u64 {
_AC!() << 12i32 << folio_order(folio)
}

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
    __slate_bits_0: __slate_bits::__SlateBits705U0,
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


fn flush_dcache_page(page: *mut page) {
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
fn flush_kernel_vmap_range(vaddr: *mut (), size: i32) {
}

fn memzero_page(page: *mut page, offset: u64, len: u64) {
let addr: *mut u8 = kmap_local_page(page as *const page) as *mut u8;
VM_BUG_ON!();
unsafe { memset(addr.wrapping_offset(offset as isize) as *mut (), 0i32, len) };
flush_dcache_page(page);
loop {
loop {
if __same_type!() != 0 {
unsafe { __compiletime_assert_553() };
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
}

#[repr(C)]
#[derive(Clone, Copy)]
struct bio_vec {
    bv_page: *mut page,
    bv_len: u32,
    bv_offset: u32,
}


fn bvec_set_page(bv: *mut bio_vec, page: *mut page, len: u32, offset: u32) {
unsafe {
(*bv).bv_page = page;
}
unsafe {
(*bv).bv_len = len;
}
unsafe {
(*bv).bv_offset = offset;
}
}

fn bvec_set_folio(bv: *mut bio_vec, folio: *mut folio, len: u64, offset: u64) {
let nr: u64 = offset / (_AC!() << 12i32);
let mut __v0: i64;
let __ret_warn_on: i32 = (len > 4294967295u64) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("566: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 566b - ., 3; .popsection\n# {0}", const 566 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_bvec_set_folio_1, sym __slate_asm_string_bvec_set_folio_2, const 61 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("567: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 567b - ., 4; .popsection\n# {0}", const 567 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
bvec_set_page(bv, folio_page!(folio, nr), len as u32, (offset % (_AC!() << 12i32)) as u32);
}

fn bvec_folio(bv: *const bio_vec) -> *mut folio {
page_folio!(unsafe { (*bv).bv_page })
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
struct bvec_iter_all {
    bv: bio_vec,
    idx: i32,
    done: u32,
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
let __UNIQUE_ID_x__568: u32 = unsafe { (*__bvec_iter_bvec(bvecs, iter)).bv_len }.wrapping_sub(iter.bi_offset);
let __UNIQUE_ID_y__569: u32 = iter.bi_size;
loop {
if __types_ok!(__UNIQUE_ID_x__568, __UNIQUE_ID_y__569) == 0 {
unsafe { __compiletime_assert_570() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__568 < __UNIQUE_ID_y__569 { __UNIQUE_ID_x__568 } else { __UNIQUE_ID_y__569 };
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
fn mp_bvec_iter_bvec(bvecs: *const bio_vec, iter: bvec_iter) -> bio_vec {
let mut __slate_compound_0: bio_vec = unsafe { core::mem::zeroed() };
return unsafe { *{
__slate_compound_0 = bio_vec { bv_page: mp_bvec_iter_page(bvecs, iter), bv_len: mp_bvec_iter_len(bvecs, iter), bv_offset: mp_bvec_iter_offset(bvecs, iter) };
core::ptr::addr_of_mut!(__slate_compound_0)
} };
}

#[inline(always)]
fn bvec_iter_offset(bvecs: *const bio_vec, iter: bvec_iter) -> u32 {
((mp_bvec_iter_offset(bvecs, iter) as u64) % (_AC!() << 12i32)) as u32
}

#[inline(always)]
fn bvec_iter_len(bvecs: *const bio_vec, iter: bvec_iter) -> u32 {
let mut __v0: u64;
let __UNIQUE_ID_x__571: u32 = mp_bvec_iter_len(bvecs, iter);
let __UNIQUE_ID_y__572: u64 = (_AC!() << 12i32).wrapping_sub(bvec_iter_offset(bvecs, iter) as u64);
loop {
if __types_ok__4!(__UNIQUE_ID_x__571, __UNIQUE_ID_y__572) == 0 {
unsafe { __compiletime_assert_573() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if (__UNIQUE_ID_x__571 as u64) < __UNIQUE_ID_y__572 { __UNIQUE_ID_x__571 as u64 } else { __UNIQUE_ID_y__572 };
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

fn bvec_iter_advance(bv: *const bio_vec, iter: *mut bvec_iter, mut bytes: u32) -> bool {
let mut idx: u32 = unsafe { (*iter).bi_idx };
let mut __v0: i32;
let __ret_warn_on: i32 = (bytes > unsafe { (*iter).bi_size }) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
2i32 | BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"Attempted to advance past end of bvec iter\n".as_ptr() as *const u8) };
let mut __v1: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v1 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v2: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_bvec_iter_advance_3, sym __slate_asm_string_bvec_iter_advance_4, const 201 as i32, const 2323 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v2 = bug;
unsafe { __v1.unwrap()(__v2) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v0 = __ret_warn_on;
if __v0 != 0 {
unsafe {
(*iter).bi_size = 0;
}
return false;
}
let __v7: u32 = unsafe { (*iter).bi_size };
unsafe {
(*iter).bi_size = __v7.wrapping_sub(bytes);
}
bytes = bytes.wrapping_add(unsafe { (*iter).bi_offset });
while bytes != 0 && bytes >= unsafe { (*bv.wrapping_offset(idx as isize)).bv_len } {
bytes = bytes.wrapping_sub(unsafe { (*bv.wrapping_offset(idx as isize)).bv_len });
idx = idx.wrapping_add(1);
}
unsafe {
(*iter).bi_idx = idx;
}
unsafe {
(*iter).bi_offset = bytes;
}
true
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_574: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn bvec_iter_advance_single(bv: *const bio_vec, iter: &mut bvec_iter, bytes: u32) {
let mut done: u32 = unsafe { (*(iter as *mut bvec_iter)).bi_offset }.wrapping_add(bytes);
if done == unsafe { (*bv.wrapping_offset((*(iter as *mut bvec_iter)).bi_idx as isize)).bv_len } {
done = 0;
let __v1: u32 = unsafe { (*iter).bi_idx };
unsafe {
(*iter).bi_idx = __v1.wrapping_add(1);
}
}
unsafe {
(*(iter as *mut bvec_iter)).bi_offset = done;
}
let __v4: u32 = unsafe { (*iter).bi_size };
unsafe {
(*iter).bi_size = __v4.wrapping_sub(bytes);
}
}

fn bvec_init_iter_all(iter_all: &mut bvec_iter_all) -> *mut bio_vec {
unsafe {
(*(iter_all as *mut bvec_iter_all)).done = 0;
}
unsafe {
(*(iter_all as *mut bvec_iter_all)).idx = 0;
}
unsafe { core::ptr::addr_of_mut!((*(iter_all as *mut bvec_iter_all)).bv) }
}

fn bvec_advance(bvec: *const bio_vec, iter_all: *mut bvec_iter_all) {
let bv: *mut bio_vec = unsafe { core::ptr::addr_of_mut!((*iter_all).bv) };
if (unsafe { (*iter_all).done }) != 0 {
let __v1: *mut page = unsafe { (*bv).bv_page };
unsafe {
(*bv).bv_page = __v1.wrapping_offset(1 as isize);
}
unsafe {
(*bv).bv_offset = 0;
}
} else {
unsafe {
(*bv).bv_page = unsafe { (*bvec).bv_page }.wrapping_offset(((*bvec).bv_offset >> 12i32) as isize);
}
unsafe {
(*bv).bv_offset = (((*bvec).bv_offset as u64) & !(!(_AC!() << 12i32).wrapping_sub(1))) as u32;
}
}
unsafe {
(*bv).bv_len = min_t__3!(u32, (_AC!() << 12i32).wrapping_sub((*bv).bv_offset as u64), unsafe { (*bvec).bv_len }.wrapping_sub((*iter_all).done));
}
let __v7: u32 = unsafe { (*iter_all).done };
unsafe {
(*iter_all).done = __v7.wrapping_add((*bv).bv_len);
}
if (unsafe { (*iter_all).done }) == unsafe { (*bvec).bv_len } {
let __v10: i32 = unsafe { (*iter_all).idx };
unsafe {
(*iter_all).idx = __v10.wrapping_add(1);
}
unsafe {
(*iter_all).done = 0;
}
}
}

fn bvec_kmap_local(bvec: &bio_vec) -> *mut () {
(unsafe { (kmap_local_page((*(bvec as *const bio_vec)).bv_page as *const page) as *const u8).wrapping_offset((*(bvec as *const bio_vec)).bv_offset as isize) }) as *mut ()
}

fn memzero_bvec(bvec: &bio_vec) {
memzero_page(unsafe { (*(bvec as *const bio_vec)).bv_page }, (unsafe { (*(bvec as *const bio_vec)).bv_offset }) as u64, (unsafe { (*(bvec as *const bio_vec)).bv_len }) as u64);
}

fn bvec_virt(bvec: *mut bio_vec) -> *mut () {
let mut __v0: i64;
let __ret_warn_on: i32 = (PageHighMem((unsafe { (*bvec).bv_page }) as *const page) != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("577: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 577b - ., 3; .popsection\n# {0}", const 577 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_bvec_virt_1, sym __slate_asm_string_bvec_virt_2, const 333 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("578: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 578b - ., 4; .popsection\n# {0}", const 578 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
(unsafe { (page_address!(unsafe { (*bvec).bv_page }) as *const u8).wrapping_offset((*bvec).bv_offset as isize) }) as *mut ()
}

fn bvec_phys(bvec: *const bio_vec) -> u64 {
page_to_phys!(unsafe { (*bvec).bv_page }).wrapping_add((unsafe { (*bvec).bv_offset }) as u64)
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


// == READ
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


fn iov_iter_type(i: *const iov_iter) -> u32 {
(unsafe { (*i).iter_type }) as u32
}

fn iter_is_ubuf(i: *const iov_iter) -> bool {
(iov_iter_type(i) as u32) == 0
}

fn iter_is_iovec(i: *const iov_iter) -> bool {
(iov_iter_type(i) as u32) == 1
}

fn iov_iter_is_bvec(i: *const iov_iter) -> bool {
(iov_iter_type(i) as u32) == 2
}

fn user_backed_iter(i: *const iov_iter) -> bool {
iter_is_ubuf(i) || iter_is_iovec(i)
}

fn copy_folio_from_iter(folio: *mut folio, offset: u64, bytes: u64, i: *mut iov_iter) -> u64 {
unsafe { copy_page_from_iter(core::ptr::addr_of_mut!((*folio).__slate_anon_0.page), offset, bytes, i) }
}

#[inline(always)]
fn copy_to_iter(addr: *const (), bytes: u64, i: *mut iov_iter) -> u64 {
if check_copy_size(addr, bytes, true) {
return unsafe { _copy_to_iter(addr, bytes, i) };
}
0
}

fn iov_iter_count(i: *const iov_iter) -> u64 {
unsafe { (*i).__slate_anon_4.__slate_anon_1.count }
}

// Flags for iov_iter_get/extract_pages*()
//
// Allow P2PDMA on the extracted pages
fn iov_iter_extract_will_pin(iter: *const iov_iter) -> bool {
user_backed_iter(iter)
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
    __slate_bits_0: __slate_bits::__SlateBits734U0,
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


// max value of bi_iter.bi_size
fn bio_inline_vecs(bio: *mut bio) -> *mut bio_vec {
bio.wrapping_offset(1 as isize) as *mut bio_vec
}

fn bio_op(bio: *const bio) -> u32 {
((unsafe { (*bio).bi_opf }) & 255u32) as u32
}

fn op_is_write(op: u32) -> bool {
op & 1 != 0
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
    __slate_bits_0: __slate_bits::__SlateBits579U0,
    fbatch: folio_batch,
    index: u64,
    saved_err: i32,
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


fn mempool_initialized(pool: *mut mempool) -> bool {
!unsafe { (*pool).elements }.is_null()
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

fn bio_no_advance_iter(bio: *const bio) -> bool {
let __v0: bool = (bio_op(bio) as u32) == 3 || (bio_op(bio) as u32) == 5;
__v0 || (bio_op(bio) as u32) == 9
}

fn bio_next_segment(bio: *const bio, iter: *mut bvec_iter_all) -> bool {
if (unsafe { (*iter).idx }) >= ((unsafe { (*bio).bi_vcnt }) as i32) {
return false;
}
bvec_advance(unsafe { (*bio).bi_io_vec }.wrapping_offset((unsafe { (*iter).idx }) as isize) as *const bio_vec, iter);
true
}

// drivers should _never_ use the all version - the bio may have been split
// before it got to the driver and the driver won't own all of it
fn bio_advance_iter(bio: *const bio, iter: *mut bvec_iter, bytes: u32) {
let __v1: u64 = unsafe { (*iter).bi_sector };
unsafe {
(*iter).bi_sector = __v1.wrapping_add((bytes >> 9i32) as u64);
}
if bio_no_advance_iter(bio) {
let __v4: u32 = unsafe { (*iter).bi_size };
unsafe {
(*iter).bi_size = __v4.wrapping_sub(bytes);
}
} else {
bvec_iter_advance((unsafe { (*bio).bi_io_vec }) as *const bio_vec, iter, bytes);
}
// TODO: It is reasonable to complete bio with error here.
}

fn bio_advance_iter_single(bio: *const bio, iter: &mut bvec_iter, bytes: u32) {
let __v1: u64 = unsafe { (*iter).bi_sector };
unsafe {
(*iter).bi_sector = __v1.wrapping_add((bytes >> 9i32) as u64);
}
if bio_no_advance_iter(bio) {
let __v4: u32 = unsafe { (*iter).bi_size };
unsafe {
(*iter).bi_size = __v4.wrapping_sub(bytes);
}
} else {
bvec_iter_advance_single((unsafe { (*bio).bi_io_vec }) as *const bio_vec, iter, bytes);
}
}

fn bio_advance(bio: *mut bio, nbytes: u32) {
if nbytes == unsafe { (*bio).bi_iter.bi_size } {
unsafe {
(*bio).bi_iter.bi_size = 0;
}
return;
}
__bio_advance(bio, nbytes);
}

// iterate over multi-page bvec
// Iterate over all multi-page bvecs. Drivers shouldn't use this version for the
// same reasons as bio_for_each_segment_all().
fn bio_first_bvec_all(bio: *mut bio) -> *mut bio_vec {
let mut __v0: i64;
let __ret_warn_on: i32 = !(!bio_flagged(bio as *const bio, 1u32)) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("624: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 624b - ., 3; .popsection\n# {0}", const 624 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_bio_first_bvec_all_1, sym __slate_asm_string_bio_first_bvec_all_2, const 245 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("625: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 625b - ., 4; .popsection\n# {0}", const 625 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
unsafe { (*bio).bi_io_vec }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct folio_iter {
    folio: *mut folio,
    offset: u64,
    length: u64,
    _next: *mut folio,
    _seg_count: u64,
    _i: i32,
}


fn bio_first_folio(fi: &mut folio_iter, bio: *mut bio, i: i32) {
let bvec: *mut bio_vec = bio_first_bvec_all(bio).wrapping_offset(i as isize);
if unlikely__2!(i >= ((unsafe { (*bio).bi_vcnt }) as i32)) != 0 {
unsafe {
(*(fi as *mut folio_iter)).folio = core::ptr::null_mut::<folio>();
}
return;
}
unsafe {
(*(fi as *mut folio_iter)).folio = bvec_folio(bvec as *const bio_vec);
}
unsafe {
(*(fi as *mut folio_iter)).offset = ((*bvec).bv_offset as u64).wrapping_add((_AC!() << 12i32).wrapping_mul(folio_page_idx((*(fi as *mut folio_iter)).folio as *const folio, (*bvec).bv_page as *const page)));
}
unsafe {
(*(fi as *mut folio_iter))._seg_count = (*bvec).bv_len as u64;
}
let mut __v0: u64;
let __UNIQUE_ID_x__626: u64 = folio_size((unsafe { (*(fi as *mut folio_iter)).folio }) as *const folio).wrapping_sub(unsafe { (*(fi as *mut folio_iter)).offset });
let __UNIQUE_ID_y__627: u64 = unsafe { (*(fi as *mut folio_iter))._seg_count };
loop {
if __types_ok__2!(__UNIQUE_ID_x__626, __UNIQUE_ID_y__627) == 0 {
unsafe { __compiletime_assert_628() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__626 < __UNIQUE_ID_y__627 { __UNIQUE_ID_x__626 } else { __UNIQUE_ID_y__627 };
unsafe {
(*(fi as *mut folio_iter)).length = __v0;
}
unsafe {
(*(fi as *mut folio_iter))._next = folio_next((*(fi as *mut folio_iter)).folio);
}
unsafe {
(*(fi as *mut folio_iter))._i = i;
}
}

fn bio_next_folio(fi: &mut folio_iter, bio: *mut bio) {
let __v1: u64 = unsafe { (*fi)._seg_count };
unsafe {
(*fi)._seg_count = __v1.wrapping_sub((*(fi as *mut folio_iter)).length);
}
if (unsafe { (*(fi as *mut folio_iter))._seg_count }) != 0 {
unsafe {
(*(fi as *mut folio_iter)).folio = (*(fi as *mut folio_iter))._next;
}
unsafe {
(*(fi as *mut folio_iter)).offset = 0;
}
let mut __v3: u64;
let __UNIQUE_ID_x__629: u64 = folio_size((unsafe { (*(fi as *mut folio_iter)).folio }) as *const folio);
let __UNIQUE_ID_y__630: u64 = unsafe { (*(fi as *mut folio_iter))._seg_count };
loop {
if __types_ok__2!(__UNIQUE_ID_x__629, __UNIQUE_ID_y__630) == 0 {
unsafe { __compiletime_assert_631() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = if __UNIQUE_ID_x__629 < __UNIQUE_ID_y__630 { __UNIQUE_ID_x__629 } else { __UNIQUE_ID_y__630 };
unsafe {
(*(fi as *mut folio_iter)).length = __v3;
}
unsafe {
(*(fi as *mut folio_iter))._next = folio_next((*(fi as *mut folio_iter)).folio);
}
} else {
{
let __arg2 = unsafe { (*(fi as *mut folio_iter))._i }.wrapping_add(1);
bio_first_folio(fi, bio, __arg2)
};
}
}

// bio_for_each_folio_all - Iterate over each folio in a bio.
// @fi: struct folio_iter which is updated for each folio.
// @bio: struct bio to iterate over.
#[unsafe(no_mangle)]
static mut fs_bio_set: bio_set = unsafe { core::mem::zeroed() };

fn bio_alloc(bdev: *mut block_device, nr_vecs: u16, opf: u32, gfp_mask: u32) -> *mut bio {
bio_alloc_bioset(bdev, nr_vecs, opf, gfp_mask, unsafe { core::ptr::addr_of_mut!(fs_bio_set) })
}

fn bio_in_atomic() -> bool {
let __v0: bool = if IS_ENABLED__2!() != 0 {
let mut __v1: i32;
loop {
if !(__native_word__15!() || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_632() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*get_current()).rcu_read_lock_nesting) as *const i32) };
__v1 != 0
} else {
false
};
if __v0 {
return true;
}
if IS_ENABLED__2!() == 0 {
return true;
}
!preemptible!()
}

fn bio_init_inline(bio: *mut bio, bdev: *mut block_device, max_vecs: u16, opf: u32) {
bio_init(bio, bdev, bio_inline_vecs(bio), max_vecs, opf);
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.bio.bio_add_vmalloc_chunk")]
extern "C" fn bio_add_vmalloc_chunk(bio: *mut bio, vaddr: *mut (), mut len: u32) -> u32 {
let offset: u32 = offset_in_page!(vaddr) as u32;
let mut __v0: u64;
let __UNIQUE_ID_x__924: u32 = len;
let __UNIQUE_ID_y__925: u64 = (_AC!() << 12i32).wrapping_sub(offset as u64);
loop {
if __types_ok__4!(__UNIQUE_ID_x__924, __UNIQUE_ID_y__925) == 0 {
unsafe { __compiletime_assert_926() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if (__UNIQUE_ID_x__924 as u64) < __UNIQUE_ID_y__925 { __UNIQUE_ID_x__924 as u64 } else { __UNIQUE_ID_y__925 };
len = __v0 as u32;
if (bio_add_page(bio, unsafe { vmalloc_to_page(vaddr as *const ()) }, len, offset) as u32) < len {
return 0;
}
if op_is_write(bio_op(bio as *const bio) as u32) {
flush_kernel_vmap_range(vaddr, len as i32);
}
len
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.bio.bio_copy_data")]
extern "C" fn bio_copy_data(dst: *mut bio, src: *mut bio) {
let mut src_iter: bvec_iter = unsafe { (*src).bi_iter };
let mut dst_iter: bvec_iter = unsafe { (*dst).bi_iter };
while src_iter.bi_size != 0 && dst_iter.bi_size != 0 {
let mut src_bv: bio_vec = bio_iter_iovec!(src, src_iter);
let mut dst_bv: bio_vec = bio_iter_iovec!(dst, dst_iter);
let mut bytes: u32 = 0;
let mut __v0: u32;
let __UNIQUE_ID_x__958: u32 = src_bv.bv_len;
let __UNIQUE_ID_y__959: u32 = dst_bv.bv_len;
loop {
if __types_ok!(__UNIQUE_ID_x__958, __UNIQUE_ID_y__959) == 0 {
unsafe { __compiletime_assert_960() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__958 < __UNIQUE_ID_y__959 { __UNIQUE_ID_x__958 } else { __UNIQUE_ID_y__959 };
bytes = __v0;
let src_buf: *mut () = bvec_kmap_local(unsafe { &(*core::ptr::addr_of_mut!(src_bv)) });
let dst_buf: *mut () = bvec_kmap_local(unsafe { &(*core::ptr::addr_of_mut!(dst_bv)) });
unsafe { memcpy(dst_buf, src_buf as *const (), bytes as u64) };
loop {
loop {
if __same_type!() != 0 {
unsafe { __compiletime_assert_961() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__kunmap_local(dst_buf as *const ());
if true {
break;
}
}
loop {
loop {
if __same_type!() != 0 {
unsafe { __compiletime_assert_962() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__kunmap_local(src_buf as *const ());
if true {
break;
}
}
bio_advance_iter_single(src as *const bio, unsafe { &mut (*core::ptr::addr_of_mut!(src_iter)) }, bytes);
bio_advance_iter_single(dst as *const bio, unsafe { &mut (*core::ptr::addr_of_mut!(dst_iter)) }, bytes);
}
}

fn bio_release_pages(bio: *mut bio, mark_dirty: bool) {
if bio_flagged(bio as *const bio, 0u32) {
__bio_release_pages(bio, mark_dirty);
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct bio_list {
    head: *mut bio,
    tail: *mut bio,
}


fn bio_list_empty(bl: *const bio_list) -> i32 {
unsafe { (*bl).head }.is_null() as i32
}

fn bio_list_init(bl: *mut bio_list) {
unsafe {
(*bl).tail = core::ptr::null_mut::<bio>();
}
unsafe {
(*bl).head = core::ptr::null_mut::<bio>();
}
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

fn bio_list_merge(bl: *mut bio_list, bl2: &bio_list) {
if unsafe { (*(bl2 as *const bio_list)).head }.is_null() {
return;
}
if !unsafe { (*bl).tail }.is_null() {
unsafe {
(*(*bl).tail).bi_next = (*(bl2 as *const bio_list)).head;
}
} else {
unsafe {
(*bl).head = (*(bl2 as *const bio_list)).head;
}
}
unsafe {
(*bl).tail = (*(bl2 as *const bio_list)).tail;
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

// bio_set is used to allow other portions of the IO system to
// allocate their own private memory pools for bio and iovec structures.
// These memory pools in turn all allocate from the bio_slab
// and the bvec_slabs[].
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

fn bio_integrity_clone(bio: *mut bio, bio_src: *mut bio, gfp_mask: u32) -> i32 {
0
}

fn bio_integrity_advance(bio: *mut bio, bytes_done: u32) {
}

fn bio_integrity_trim(bio: *mut bio) {
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


fn bdev_nr_sectors(bdev: *mut block_device) -> u64 {
unsafe { (*bdev).bd_nr_sectors }
}

fn bdev_get_queue(bdev: *mut block_device) -> *mut request_queue {
return unsafe { (*bdev).bd_queue }; // this is never NULL
}

// interruptible by fatal signals
fn queue_segment_boundary(q: *const request_queue) -> u64 {
unsafe { (*q).limits.seg_boundary_mask }
}

fn queue_max_segment_size(q: *const request_queue) -> u32 {
unsafe { (*q).limits.max_segment_size }
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
fn bio_has_crypt_ctx(bio: *mut bio) -> bool {
false
}

fn bio_crypt_clone(dst: *mut bio, src: *mut bio, gfp_mask: u32) -> i32 {
if bio_has_crypt_ctx(src) {
return unsafe { __bio_crypt_clone(dst, src, gfp_mask) };
}
0
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

fn kmemleak_free(ptr: *const ()) {
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
fn __do_trace_block_bio_complete(q: *mut request_queue, bio: *mut bio) {
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
let mut __UNIQUE_ID_guard_746: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_746) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_747: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_747) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_749() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_block_bio_complete)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_748: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_748;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut request_queue, *mut bio) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut request_queue, *mut bio) -> i32>>(__SCT__tp_func_block_bio_complete as *const ()) };
unsafe { __v12.unwrap()(__data, q, bio) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_block_bio_complete: __SlateAlign16<[u8; 19]> = __SlateAlign16([98u8, 108u8, 111u8, 99u8, 107u8, 95u8, 98u8, 105u8, 111u8, 95u8, 99u8, 111u8, 109u8, 112u8, 108u8, 101u8, 116u8, 101u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_block_bio_complete_750: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_block_bio_complete) }) as *mut ();

fn trace_block_bio_complete(q: *mut request_queue, bio: *mut bio) {
if static_branch_unlikely!(unsafe { core::ptr::addr_of_mut!(__tracepoint_block_bio_complete.key) }) != 0 {
__do_trace_block_bio_complete(q, bio);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_751: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

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


fn bio_crypt_advance(bio: *mut bio, bytes: u32) {
if bio_has_crypt_ctx(bio) {
unsafe { __bio_crypt_advance(bio, bytes) };
}
}

fn bio_crypt_free_ctx(bio: *mut bio) {
if bio_has_crypt_ctx(bio) {
unsafe { __bio_crypt_free_ctx(bio) };
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct blk_flush_queue {
    mq_flush_lock: spinlock,
    __slate_bits_0: __slate_bits::__SlateBits845U0,
    rq_status: u8,
    flush_pending_since: u64,
    flush_queue: [list_head; 2],
    flush_data_in_flight: u64,
    flush_rq: *mut request,
    callback_head: callback_head,
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

fn zone_device_pages_compatible(a: *const page, b: *const page) -> bool {
let __v0: bool = is_pci_p2pdma_page(a) || is_pci_p2pdma_page(b);
if __v0 {
return zone_device_pages_have_same_pgmap(a, b);
}
true
}

fn bio_integrity_endio(bio: *mut bio) -> bool {
true
}

fn bio_integrity_free(bio: *mut bio) {
}

fn blk_zone_bio_endio(bio: *mut bio) {
}

// Optimized request reference counting. Ideally we'd make timeouts be more
// clever, as that's the only reason we need references at all... But until
// this happens, this is faster than using refcount_t. Also see:
//
// abc54d634334 ("io_uring: switch to atomic_t for io_kiocb reference count")
#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_debugfs_attr {
    name: *const u8,
    mode: u16,
    show: Option<unsafe extern "C" fn(*mut (), *mut seq_file) -> i32>,
    write: Option<unsafe extern "C" fn(*mut (), *const u8, u64, *mut i64) -> i64>,
    seq_ops: *const seq_operations,
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

fn rq_qos_done_bio(bio: *mut bio) {
let mut q: *mut request_queue = unsafe { core::mem::zeroed() };
let __v0: bool = unsafe { (*bio).bi_bdev }.is_null() || !bio_flagged(bio as *const bio, 8u32) && !bio_flagged(bio as *const bio, 9u32);
if __v0 {
return;
}
q = bdev_get_queue(unsafe { (*bio).bi_bdev });
// A BIO may carry BIO_QOS_* flags even if the associated request_queue
// does not have rq_qos enabled. This can happen with stacked block
// devices — for example, NVMe multipath, where it's possible that the
// bottom device has QoS enabled but the top device does not. Therefore,
// always verify that q->rq_qos is present and QoS is enabled before
// calling __rq_qos_done_bio().
if test_bit!(15i32, unsafe { core::ptr::addr_of_mut!((*q).queue_flags) }) != 0 && !unsafe { (*q).rq_qos }.is_null() {
unsafe { __rq_qos_done_bio((*q).rq_qos, bio) };
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct blkg_iostat {
    bytes: [u64; 3],
    ios: [u64; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blkg_iostat_set {
    sync: u64_stats_sync,
    blkg: *mut blkcg_gq,
    lnode: llist_node,
    lqueued: i32,
    cur: blkg_iostat,
    last: blkg_iostat,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blkcg_gq {
    q: *mut request_queue,
    q_node: list_head,
    blkcg_node: hlist_node,
    blkcg: *mut blkcg,
    parent: *mut blkcg_gq,
    refcnt: percpu_ref,
    online: bool,
    iostat_cpu: *mut blkg_iostat_set,
    iostat: blkg_iostat_set,
    pd: [*mut blkg_policy_data; 6],
    __slate_anon_10: __SlateRecord_blk_cgroup_h_80,
    use_delay: __SlateRecord_types_h_190,
    delay_nsec: __SlateRecord_types_h_197,
    delay_start: __SlateRecord_types_h_197,
    last_delay: u64,
    last_use: i32,
    callback_head: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blkcg {
    css: cgroup_subsys_state,
    lock: spinlock,
    online_pin: refcount_struct,
    congestion_count: __SlateRecord_types_h_190,
    blkg_tree: xarray,
    blkg_hint: *mut blkcg_gq,
    blkg_list: hlist_head,
    cpd: [*mut blkcg_policy_data; 6],
    all_blkcgs_node: list_head,
    lhead: *mut llist_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blkg_policy_data {
    blkg: *mut blkcg_gq,
    plid: i32,
    online: bool,
    callback_head: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blkcg_policy_data {
    blkcg: *mut blkcg,
    plid: i32,
}


fn blkg_put(blkg: *mut blkcg_gq) {
percpu_ref_put(unsafe { core::ptr::addr_of_mut!((*blkg).refcnt) });
}

#[repr(C)]
#[derive(Clone, Copy)]
struct bio_alloc_cache {
    free_list: *mut bio,
    free_list_irq: *mut bio,
    nr: u32,
    nr_irq: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct biovec_slab {
    nr_vecs: i32,
    name: *mut u8,
    slab: *mut kmem_cache,
}


#[unsafe(link_section = ".data..read_mostly")]
static mut bvec_slabs: __SlateAlign16<[biovec_slab; 4]> = __SlateAlign16([{
let mut __t0: biovec_slab = unsafe { core::mem::zeroed() };
__t0.nr_vecs = 16i32;
__t0.name = b"biovec-16\0".as_ptr() as *mut u8;
__t0
}, {
let mut __t1: biovec_slab = unsafe { core::mem::zeroed() };
__t1.nr_vecs = 64i32;
__t1.name = b"biovec-64\0".as_ptr() as *mut u8;
__t1
}, {
let mut __t2: biovec_slab = unsafe { core::mem::zeroed() };
__t2.nr_vecs = 128i32;
__t2.name = b"biovec-128\0".as_ptr() as *mut u8;
__t2
}, {
let mut __t3: biovec_slab = unsafe { core::mem::zeroed() };
__t3.nr_vecs = 256i32;
__t3.name = b"biovec-max\0".as_ptr() as *mut u8;
__t3
}]);

fn biovec_slab(nr_vecs: u16) -> *mut biovec_slab {
match nr_vecs as i32 {
5..=16 => {
// smaller bios use inline vecs
return unsafe { core::ptr::addr_of_mut!(bvec_slabs.0) as *mut biovec_slab }.wrapping_offset(0 as isize);
}
17..=64 => {
return unsafe { core::ptr::addr_of_mut!(bvec_slabs.0) as *mut biovec_slab }.wrapping_offset(1 as isize);
}
65..=128 => {
return unsafe { core::ptr::addr_of_mut!(bvec_slabs.0) as *mut biovec_slab }.wrapping_offset(2 as isize);
}
129..=256 => {
return unsafe { core::ptr::addr_of_mut!(bvec_slabs.0) as *mut biovec_slab }.wrapping_offset(3 as isize);
}
_ => {
loop {
unsafe {
core::arch::asm!("866: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 866b - ., 3; .popsection\n# {0}", const 866 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_biovec_slab_0, sym __slate_asm_string_biovec_slab_1, const 64 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
return core::ptr::null_mut::<biovec_slab>();
}
}
unsafe { core::mem::zeroed() }
}

// fs_bio_set is the bio_set containing bio and iovec memory pools used by
// IO code that does not need private memory pools.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_fs_bio_set_867: *mut () = (unsafe { core::ptr::addr_of_mut!(fs_bio_set) }) as *mut ();

/// Our slab pool management
#[repr(C)]
#[derive(Clone, Copy)]
struct bio_slab {
    slab: *mut kmem_cache,
    slab_ref: u32,
    slab_size: u32,
    name: [u8; 12],
}


static mut bio_slab_lock: mutex = {
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

static mut bio_slabs: xarray = xarray { xa_lock: spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = unsafe { core::mem::zeroed() };
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0
} }, xa_flags: 0u32, xa_head: core::ptr::null_mut::<()>() };

fn create_bio_slab(size: u32) -> *mut bio_slab {
let mut __slate_compound_7: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_9: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut bslab: *mut bio_slab = unsafe { core::mem::zeroed() };
unsafe {
__obj_size = size_mul(core::mem::size_of::<bio_slab>() as u64, 1u64);
let __slate_auto_bytes_1 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_0 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_1 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_7 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_7)
});
alloc_tag_restore!();
} else {
*__slate_slot_1 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_9 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_9)
});
}
bslab = *__slate_slot_1 as *mut bio_slab;
if bslab.is_null() {
return core::ptr::null_mut::<bio_slab>();
} else {
snprintf((*bslab).name.as_mut_ptr() as *mut u8, core::mem::size_of::<[u8; 12]>() as u64, c"bio-%d".as_ptr() as *const u8, size);
(*bslab).slab = __kmem_cache_create(((*bslab).name.as_mut_ptr() as *mut u8) as *const u8, size, core::mem::align_of::<u64>() as u32, __SLAB_FLAG_BIT!(4i32) | __SLAB_FLAG_BIT!(9i32), None);
if !unsafe { (*bslab).slab }.is_null() {
(*bslab).slab_ref = 1;
(*bslab).slab_size = size;
if xa_err(xa_store(core::ptr::addr_of_mut!(bio_slabs), size as u64, bslab as *mut (), ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32))) == 0 {
return bslab;
} else {
kmem_cache_destroy((*bslab).slab);
}
}
kfree(bslab as *const ());
return core::ptr::null_mut::<bio_slab>();
}
}
}

fn bs_bio_slab_size(bs: *mut bio_set) -> u32 {
((unsafe { (*bs).front_pad }) as u64).wrapping_add(core::mem::size_of::<bio>() as u64).wrapping_add((unsafe { (*bs).back_pad }) as u64) as u32
}

fn bio_slab_addr(bio: *mut bio) -> *mut () {
(unsafe { (bio as *const u8).wrapping_offset(-((*(*bio).bi_pool).front_pad as isize)) }) as *mut ()
}

fn bio_find_or_create_slab(bs: *mut bio_set) -> *mut kmem_cache {
let size: u32 = bs_bio_slab_size(bs);
let mut bslab: *mut bio_slab = unsafe { core::mem::zeroed() };
unsafe { mutex_lock(core::ptr::addr_of_mut!(bio_slab_lock)) };
bslab = (unsafe { xa_load(core::ptr::addr_of_mut!(bio_slabs), size as u64) }) as *mut bio_slab;
if !bslab.is_null() {
let __v1: u32 = unsafe { (*bslab).slab_ref };
unsafe {
(*bslab).slab_ref = __v1.wrapping_add(1);
}
} else {
bslab = create_bio_slab(size);
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!(bio_slab_lock)) };
if !bslab.is_null() {
return unsafe { (*bslab).slab };
}
core::ptr::null_mut::<kmem_cache>()
}

fn bio_put_slab(bs: *mut bio_set) {
let mut bug: core::mem::MaybeUninit<*mut bug_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut bug_entry = core::ptr::addr_of_mut!(bug) as *mut *mut bug_entry;
let mut __v11: core::mem::MaybeUninit<Option<unsafe extern "C" fn(*mut bug_entry, ...)>> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut Option<unsafe extern "C" fn(*mut bug_entry, ...)> = core::ptr::addr_of_mut!(__v11) as *mut Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
let mut __ret_warn_on_1: i32 = unsafe { core::mem::zeroed() };
let mut slab_size: u32 = unsafe { core::mem::zeroed() };
let mut bslab: *mut bio_slab = unsafe { core::mem::zeroed() };
unsafe {
bslab = core::ptr::null_mut::<bio_slab>();
slab_size = bs_bio_slab_size(bs);
mutex_lock(core::ptr::addr_of_mut!(bio_slab_lock));
bslab = xa_load(core::ptr::addr_of_mut!(bio_slabs), slab_size as u64) as *mut bio_slab;
__ret_warn_on_1 = bslab.is_null() as i32;
if unlikely!(__ret_warn_on_1) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
__WARN_validate_printf(c"\x013bio: unable to find slab!\n".as_ptr() as *const u8);
*__slate_slot_11 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_bio_put_slab_21, sym __slate_asm_string_bio_put_slab_22, const 151 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
*__slate_slot_9 = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
(*__slate_slot_11).unwrap()(*__slate_slot_9);
core::arch::asm!("", options(att_syntax));
if true {
break;
}
}
}
let __v14: i64 = (__ret_warn_on_1 != 0) as i64;
if __v14 == 0 {
{
let __ret_warn_on: i32 = ((*bslab).slab != (*bs).bio_slab) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("869: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 869b - ., 3; .popsection\n# {0}", const 869 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_bio_put_slab_19, sym __slate_asm_string_bio_put_slab_20, const 154 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
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
}
let __ret_warn_on: i32 = ((*bslab).slab_ref == 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
core::arch::asm!("871: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 871b - ., 3; .popsection\n# {0}", const 871 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_bio_put_slab_17, sym __slate_asm_string_bio_put_slab_18, const 156 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("872: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 872b - ., 4; .popsection\n# {0}", const 872 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
let __v1: u32 = (*bslab).slab_ref;
let __v0: u32 = __v1.wrapping_sub(1);
(*bslab).slab_ref = __v0;
if __v0 == 0 {
xa_erase(core::ptr::addr_of_mut!(bio_slabs), slab_size as u64);
kmem_cache_destroy((*bslab).slab);
kfree(bslab as *const ());
}
}
mutex_unlock(core::ptr::addr_of_mut!(bio_slab_lock));
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_868: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

/// Make the first allocation restricted and don't dump info on allocation
/// failures, since we'll fall back to the mempool in case of failure.
fn try_alloc_gfp(gfp: u32) -> u32 {
gfp & !((BIT!(10i32) as u32) | (BIT!(6i32) as u32)) | (BIT!(19i32) as u32) | (BIT!(16i32) as u32) | (BIT!(13i32) as u32)
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.bio.bio_uninit")]
extern "C" fn bio_uninit(bio: *mut bio) {
if !unsafe { (*bio).bi_blkg }.is_null() {
blkg_put(unsafe { (*bio).bi_blkg });
unsafe {
(*bio).bi_blkg = core::ptr::null_mut::<blkcg_gq>();
}
}
if !bio_integrity(bio).is_null() {
bio_integrity_free(bio);
}
bio_crypt_free_ctx(bio);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_bio_uninit_873: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut bio)>, *mut ()>(Some(bio_uninit)) };

fn bio_free(bio: *mut bio) {
let bs: *mut bio_set = unsafe { (*bio).bi_pool };
let p: *mut () = bio as *mut ();
let mut __v0: i64;
{
let __ret_warn_on: i32 = bs.is_null() as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("874: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 874b - ., 3; .popsection\n# {0}", const 874 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_bio_free_1, sym __slate_asm_string_bio_free_2, const 200 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("875: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 875b - ., 4; .popsection\n# {0}", const 875 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
}
let mut __v5: i64;
let __ret_warn_on: i32 = ((((unsafe { (*bio).bi_max_vecs }) as i32) as u32) > 256) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("876: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 876b - ., 3; .popsection\n# {0}", const 876 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_bio_free_6, sym __slate_asm_string_bio_free_7, const 201 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
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
__v5 = (__ret_warn_on != 0) as i64;
bio_uninit(bio);
if (((unsafe { (*bio).bi_max_vecs }) as i32) as u32) == 256 {
unsafe { mempool_free((*bio).bi_io_vec as *mut (), core::ptr::addr_of_mut!((*bs).bvec_pool)) };
} else {
if ((unsafe { (*bio).bi_max_vecs }) as i32) > 4 {
unsafe { kmem_cache_free((*biovec_slab((*bio).bi_max_vecs)).slab, (*bio).bi_io_vec as *mut ()) };
}
}
unsafe { mempool_free((p as *const u8).wrapping_offset(-((*bs).front_pad as isize)) as *mut (), core::ptr::addr_of_mut!((*bs).bio_pool)) };
}

/// Users of this function have their own bio allocation. Subsequently,
/// they must remember to pair any call to bio_init() with bio_uninit()
/// when IO has completed, or when the bio is released.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.bio.bio_init")]
extern "C" fn bio_init(bio: *mut bio, bdev: *mut block_device, table: *mut bio_vec, max_vecs: u16, opf: u32) {
unsafe {
(*bio).bi_next = core::ptr::null_mut::<bio>();
}
unsafe {
(*bio).bi_bdev = bdev;
}
unsafe {
(*bio).bi_opf = opf;
}
unsafe {
(*bio).bi_flags = 0;
}
unsafe {
(*bio).bi_ioprio = 0;
}
unsafe {
(*bio).bi_write_hint = 0;
}
unsafe {
(*bio).bi_write_stream = 0;
}
unsafe {
(*bio).bi_status = 0;
}
unsafe {
(*bio).bi_bvec_gap_bit = 0;
}
unsafe {
(*bio).bi_iter.bi_sector = 0;
}
unsafe {
(*bio).bi_iter.bi_size = 0;
}
unsafe {
(*bio).bi_iter.bi_idx = 0;
}
unsafe {
(*bio).bi_iter.bi_offset = 0;
}
unsafe {
(*bio).bi_end_io = None;
}
unsafe {
(*bio).bi_private = core::ptr::null_mut::<()>();
}
unsafe {
(*bio).bi_blkg = core::ptr::null_mut::<blkcg_gq>();
}
unsafe {
(*bio).issue_time_ns = 0;
}
if !bdev.is_null() {
unsafe { bio_associate_blkg(bio) };
}
unsafe {
(*bio).bi_iocost_cost = 0;
}
unsafe {
(*bio).bi_vcnt = 0;
}
atomic_set(unsafe { core::ptr::addr_of_mut!((*bio).__bi_remaining) }, 1i32);
atomic_set(unsafe { core::ptr::addr_of_mut!((*bio).__bi_cnt) }, 1i32);
unsafe {
(*bio).__slate_anon_12.bi_cookie = 1u32.wrapping_neg();
}
unsafe {
(*bio).bi_max_vecs = max_vecs;
}
unsafe {
(*bio).bi_io_vec = table;
}
unsafe {
(*bio).bi_pool = core::ptr::null_mut::<bio_set>();
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_bio_init_878: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut bio, *mut block_device, *mut bio_vec, u16, u32)>, *mut ()>(Some(bio_init)) };

/// bio_reset - reinitialize a bio
/// @bio:	bio to reset
/// @bdev:	block device to use the bio for
/// @opf:	operation and flags for bio
///
/// Description:
///   After calling bio_reset(), @bio will be in the same state as a freshly
///   allocated bio returned bio bio_alloc_bioset() - the only fields that are
///   preserved are the ones that are initialized by bio_alloc_bioset(). See
///   comment in struct bio.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.bio.bio_reset")]
extern "C" fn bio_reset(bio: *mut bio, bdev: *mut block_device, opf: u32) {
let bv: *mut bio_vec = unsafe { (*bio).bi_io_vec };
bio_uninit(bio);
unsafe { memset(bio as *mut (), 0i32, offsetof__70!(bio)) };
atomic_set(unsafe { core::ptr::addr_of_mut!((*bio).__bi_remaining) }, 1i32);
unsafe {
(*bio).bi_io_vec = bv;
}
unsafe {
(*bio).bi_bdev = bdev;
}
if !unsafe { (*bio).bi_bdev }.is_null() {
unsafe { bio_associate_blkg(bio) };
}
unsafe {
(*bio).bi_opf = opf;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_bio_reset_879: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut bio, *mut block_device, u32)>, *mut ()>(Some(bio_reset)) };

/// bio_reuse - reuse a bio with the payload left intact
/// @bio:	bio to reuse
/// @opf:	operation and flags for the next I/O
///
/// Allow reusing an existing bio for another operation with all set up
/// fields including the payload, device and end_io handler left intact.
///
/// Typically used when @bio is first used to read data which is then written
/// to another location without modification.  @bio must not be in-flight and
/// owned by the caller.  Can't be used for cloned bios.
///
/// Note: Can't be used when @bio has integrity or blk-crypto contexts for now.
/// Feel free to add that support when you need it, though.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.bio.bio_reuse")]
extern "C" fn bio_reuse(bio: *mut bio, opf: u32) {
let vcnt: u16 = unsafe { (*bio).bi_vcnt };
let mut i: u16 = 0;
let end_io: Option<unsafe extern "C" fn(*mut bio)> = unsafe { (*bio).bi_end_io };
let private: *mut () = unsafe { (*bio).bi_private };
let mut __v0: i64;
{
let __ret_warn_on: i32 = !(!bio_flagged(bio as *const bio, 1u32)) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("880: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 880b - ., 3; .popsection\n# {0}", const 880 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_bio_reuse_1, sym __slate_asm_string_bio_reuse_2, const 310 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("881: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 881b - ., 4; .popsection\n# {0}", const 881 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
}
let mut __v5: i64;
{
let __ret_warn_on: i32 = !bio_integrity(bio).is_null() as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("882: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 882b - ., 3; .popsection\n# {0}", const 882 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_bio_reuse_6, sym __slate_asm_string_bio_reuse_7, const 311 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("883: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 883b - ., 4; .popsection\n# {0}", const 883 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v5 = (__ret_warn_on != 0) as i64;
}
let mut __v10: i64;
let __ret_warn_on: i32 = !(!bio_has_crypt_ctx(bio)) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("884: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 884b - ., 3; .popsection\n# {0}", const 884 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_bio_reuse_11, sym __slate_asm_string_bio_reuse_12, const 312 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("885: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 885b - ., 4; .popsection\n# {0}", const 885 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v10 = (__ret_warn_on != 0) as i64;
bio_reset(bio, unsafe { (*bio).bi_bdev }, opf);
while (i as i32) < (vcnt as i32) {
let __v16: u32 = unsafe { (*bio).bi_iter.bi_size };
unsafe {
(*bio).bi_iter.bi_size = __v16.wrapping_add((*unsafe { (*bio).bi_io_vec }.wrapping_offset((i as i32) as isize)).bv_len);
}
i = (i as i32).wrapping_add(1) as u16;
}
unsafe {
(*bio).bi_vcnt = vcnt;
}
unsafe {
(*bio).bi_private = private;
}
unsafe {
(*bio).bi_end_io = end_io;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_bio_reuse_886: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut bio, u32)>, *mut ()>(Some(bio_reuse)) };

fn __bio_chain_endio(bio: *mut bio) -> *mut bio {
let parent: *mut bio = (unsafe { (*bio).bi_private }) as *mut bio;
if (unsafe { (*bio).bi_status }) != 0 && (unsafe { (*parent).bi_status }) == 0 {
unsafe {
(*parent).bi_status = (*bio).bi_status;
}
}
bio_put(bio);
parent
}

/// This function should only be used as a flag and must never be called.
/// If execution reaches here, it indicates a serious programming error.
#[unsafe(link_section = ".text.slate_distinct.bio.bio_chain_endio")]
extern "C" fn bio_chain_endio(bio: *mut bio) {
loop {
unsafe {
core::arch::asm!("887: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 887b - ., 3; .popsection\n# {0}", const 887 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_bio_chain_endio_0, sym __slate_asm_string_bio_chain_endio_1, const 339 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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

/// bio_chain - chain bio completions
/// @bio: the target bio
/// @parent: the parent bio of @bio
///
/// The caller won't have a bi_end_io called when @bio completes - instead,
/// @parent's bi_end_io won't be called until both @parent and @bio have
/// completed; the chained bio will also be freed when it completes.
///
/// The caller must not set bi_private or bi_end_io in @bio.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.bio.bio_chain")]
extern "C" fn bio_chain(bio: *mut bio, parent: *mut bio) {
loop {
if unlikely__2!(!unsafe { (*bio).bi_private }.is_null() || (unsafe { (*bio).bi_end_io }) != None) != 0 {
loop {
unsafe {
core::arch::asm!("888: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 888b - ., 3; .popsection\n# {0}", const 888 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_bio_chain_0, sym __slate_asm_string_bio_chain_1, const 355 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
unsafe {
(*bio).bi_private = parent as *mut ();
}
unsafe {
(*bio).bi_end_io = Some(bio_chain_endio);
}
bio_inc_remaining(parent);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_bio_chain_889: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut bio, *mut bio)>, *mut ()>(Some(bio_chain)) };

/// bio_chain_and_submit - submit a bio after chaining it to another one
/// @prev: bio to chain and submit
/// @new: bio to chain to
///
/// If @prev is non-NULL, chain it to @new and submit it.
///
/// Return: @new.
#[unsafe(no_mangle)]
extern "C" fn bio_chain_and_submit(prev: *mut bio, new: *mut bio) -> *mut bio {
if !prev.is_null() {
bio_chain(prev, new);
unsafe { submit_bio(prev) };
}
new
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.bio.blk_next_bio")]
extern "C" fn blk_next_bio(bio: *mut bio, bdev: *mut block_device, nr_pages: u32, opf: u32, gfp: u32) -> *mut bio {
bio_chain_and_submit(bio, bio_alloc(bdev, nr_pages as u16, opf, gfp))
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_blk_next_bio_890: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut bio, *mut block_device, u32, u32, u32) -> *mut bio>, *mut ()>(Some(blk_next_bio)) };

#[unsafe(link_section = ".text.slate_distinct.bio.bio_alloc_rescue")]
extern "C" fn bio_alloc_rescue(work: *mut work_struct) {
let bs: *mut bio_set = container_of__31!(work, bio_set);
while true {
spin_lock(unsafe { core::ptr::addr_of_mut!((*bs).rescue_lock) });
let bio: *mut bio = bio_list_pop(unsafe { core::ptr::addr_of_mut!((*bs).rescue_list) });
spin_unlock(unsafe { core::ptr::addr_of_mut!((*bs).rescue_lock) });
if bio.is_null() {
break;
}
unsafe { submit_bio_noacct(bio) };
}
}

/// submit_bio_noacct() converts recursion to iteration; this means if we're
/// running beneath it, any bios we allocate and submit will not be submitted
/// (and thus freed) until after we return.
///
/// This exposes us to a potential deadlock if we allocate multiple bios from the
/// same bio_set while running underneath submit_bio_noacct().  If we were to
/// allocate multiple bios (say a stacking block driver that was splitting bios),
/// we would deadlock if we exhausted the mempool's reserve.
///
/// We solve this, and guarantee forward progress by punting the bios on
/// current->bio_list to a per bio_set rescuer workqueue before blocking to wait
/// for elements being returned to the mempool.
fn punt_bios_to_rescuer(bs: *mut bio_set) {
let mut punt: bio_list = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(punt) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let mut nopunt: bio_list = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(nopunt) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let mut bio: *mut bio = unsafe { core::mem::zeroed() };
if unsafe { (*get_current()).bio_list }.is_null() || unsafe { (*bs).rescue_workqueue }.is_null() {
return;
}
let __v0: bool = bio_list_empty(unsafe { (*get_current()).bio_list }.wrapping_offset(0 as isize) as *const bio_list) != 0 && bio_list_empty(unsafe { (*get_current()).bio_list }.wrapping_offset(1 as isize) as *const bio_list) != 0;
if __v0 {
return;
}
// In order to guarantee forward progress we must punt only bios that
// were allocated from this bio_set; otherwise, if there was a bio on
// there for a stacking driver higher up in the stack, processing it
// could require allocating bios from this bio_set, and doing that from
// our own rescuer would be bad.
//
// Since bio lists are singly linked, pop them all instead of trying to
// remove from the middle of the list:
bio_list_init(core::ptr::addr_of_mut!(punt));
bio_list_init(core::ptr::addr_of_mut!(nopunt));
loop {
let __v1: *mut bio = bio_list_pop(unsafe { (*get_current()).bio_list }.wrapping_offset(0 as isize));
bio = __v1;
if __v1.is_null() {
break;
}
bio_list_add(if (unsafe { (*bio).bi_pool }) == bs { core::ptr::addr_of_mut!(punt) } else { core::ptr::addr_of_mut!(nopunt) }, bio);
}
unsafe {
*unsafe { (*get_current()).bio_list }.wrapping_offset(0 as isize) = nopunt;
}
bio_list_init(core::ptr::addr_of_mut!(nopunt));
loop {
let __v3: *mut bio = bio_list_pop(unsafe { (*get_current()).bio_list }.wrapping_offset(1 as isize));
bio = __v3;
if __v3.is_null() {
break;
}
bio_list_add(if (unsafe { (*bio).bi_pool }) == bs { core::ptr::addr_of_mut!(punt) } else { core::ptr::addr_of_mut!(nopunt) }, bio);
}
unsafe {
*unsafe { (*get_current()).bio_list }.wrapping_offset(1 as isize) = nopunt;
}
spin_lock(unsafe { core::ptr::addr_of_mut!((*bs).rescue_lock) });
bio_list_merge(unsafe { core::ptr::addr_of_mut!((*bs).rescue_list) }, unsafe { &(*core::ptr::addr_of_mut!(punt)) });
spin_unlock(unsafe { core::ptr::addr_of_mut!((*bs).rescue_lock) });
queue_work(unsafe { (*bs).rescue_workqueue }, unsafe { core::ptr::addr_of_mut!((*bs).rescue_work) });
}

fn bio_alloc_irq_cache_splice(cache: *mut bio_alloc_cache) {
let mut flags: u64 = 0;
// cache->free_list must be empty
let mut __v0: i64;
let __ret_warn_on: i32 = !unsafe { (*cache).free_list }.is_null() as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("891: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 891b - ., 3; .popsection\n# {0}", const 891 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_bio_alloc_irq_cache_splice_1, sym __slate_asm_string_bio_alloc_irq_cache_splice_2, const 465 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("892: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 892b - ., 4; .popsection\n# {0}", const 892 as i32, options(nostack, att_syntax));
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
local_irq_save!(flags);
unsafe {
(*cache).free_list = (*cache).free_list_irq;
}
unsafe {
(*cache).free_list_irq = core::ptr::null_mut::<bio>();
}
let __v9: u32 = unsafe { (*cache).nr };
unsafe {
(*cache).nr = __v9.wrapping_add((*cache).nr_irq);
}
unsafe {
(*cache).nr_irq = 0;
}
local_irq_restore!(flags);
}

fn bio_alloc_percpu_cache(bs: *mut bio_set) -> *mut bio {
let mut cache: *mut bio_alloc_cache = unsafe { core::mem::zeroed() };
let mut __v0: *mut bio_alloc_cache;
__verify_pcpu_ptr__13!();
let mut __v2: i32;
loop {
preempt_count_inc!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
let mut __v4: i32;
__this_cpu_preempt_check(c"read".as_ptr() as *const u8);
let mut __v5: i32;
let mut pscr_ret__: i32 = 0;
__verify_pcpu_ptr__3!();
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v7: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v7 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v7;
}
2 => {
let mut __v9: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v9 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v9;
}
4 => {
let mut __v10: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v10 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v10;
}
8 => {
let mut __v11: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v11 = pfo_val__ as i32;
pscr_ret__ = __v11;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v5 = pscr_ret__;
__v4 = __v5;
__v2 = __v4;
__v0 = ((((unsafe { (*bs).cache }) as u64) as *mut bio_alloc_cache) as u64).wrapping_add(unsafe { __per_cpu_offset[__v2 as usize] }) as *mut bio_alloc_cache;
cache = __v0;
if unsafe { (*cache).free_list }.is_null() {
let mut __v12: u32;
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_893() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v12 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*cache).nr_irq) as *const u32) };
if __v12 >= 16 {
bio_alloc_irq_cache_splice(cache);
}
if unsafe { (*cache).free_list }.is_null() {
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
return core::ptr::null_mut::<bio>();
}
}
let bio: *mut bio = unsafe { (*cache).free_list };
unsafe {
(*cache).free_list = (*bio).bi_next;
}
let __v17: u32 = unsafe { (*cache).nr };
unsafe {
(*cache).nr = __v17.wrapping_sub(1);
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
unsafe {
(*bio).bi_pool = bs;
}
kmemleak_alloc(bio_slab_addr(bio) as *const (), (unsafe { kmem_cache_size((*bs).bio_slab) }) as u64, 1i32, (BIT!(10i32) | BIT!(11i32)) as u32);
bio
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_894: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_895: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

/// bio_alloc_bioset - allocate a bio for I/O
/// @bdev:	block device to allocate the bio for (can be %NULL)
/// @nr_vecs:	number of bvecs to pre-allocate
/// @opf:	operation and flags for bio
/// @gfp:	the GFP_* mask given to the slab allocator
/// @bs:		the bio_set to allocate from.
///
/// Allocate a bio from the mempools in @bs.
///
/// If %__GFP_DIRECT_RECLAIM is set then bio_alloc will always be able to
/// allocate a bio.  This is due to the mempool guarantees.  To make this work,
/// callers must never allocate more than 1 bio at a time from the general pool.
/// Callers that need to allocate more than 1 bio must always submit the
/// previously allocated bio for IO before attempting to allocate a new one.
/// Failure to do so can cause deadlocks under memory pressure.
///
/// Note that when running under submit_bio_noacct() (i.e. any block driver),
/// bios are not submitted until after you return - see the code in
/// submit_bio_noacct() that converts recursion into iteration, to prevent
/// stack overflows.
///
/// This would normally mean allocating multiple bios under submit_bio_noacct()
/// would be susceptible to deadlocks, but we have
/// deadlock avoidance code that resubmits any blocked bios from a rescuer
/// thread.
///
/// However, we do not guarantee forward progress for allocations from other
/// mempools. Doing multiple allocations from the same mempool under
/// submit_bio_noacct() should be avoided - instead, use bio_set's front_pad
/// for per bio allocations.
///
/// Returns: Pointer to new bio on success, NULL on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.bio.bio_alloc_bioset")]
extern "C" fn bio_alloc_bioset(bdev: *mut block_device, mut nr_vecs: u16, mut opf: u32, mut gfp: u32, bs: *mut bio_set) -> *mut bio {
let mut bvecs: *mut bio_vec = core::ptr::null_mut::<bio_vec>();
let mut bio: *mut bio = core::ptr::null_mut::<bio>();
let saved_gfp: u32 = gfp;
let mut p: *mut () = unsafe { core::mem::zeroed() };
// should not use nobvec bioset for nr_vecs > 0
let mut __v0: i64;
let __ret_warn_on: i32 = (!mempool_initialized(unsafe { core::ptr::addr_of_mut!((*bs).bvec_pool) }) && (nr_vecs as i32) > 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("896: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 896b - ., 3; .popsection\n# {0}", const 896 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_bio_alloc_bioset_1, sym __slate_asm_string_bio_alloc_bioset_2, const 544 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("897: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 897b - ., 4; .popsection\n# {0}", const 897 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
return core::ptr::null_mut::<bio>();
}
if saved_gfp & (BIT!(10i32) as u32) != 0 {
gfp = try_alloc_gfp(gfp);
}
if !unsafe { (*bs).cache }.is_null() && (nr_vecs as i32) <= 4 {
// Set REQ_ALLOC_CACHE even if no cached bio is available to
// return the allocated bio to the percpu cache when done.
opf |= 8388608u32;
bio = bio_alloc_percpu_cache(bs);
} else {
let __v8: u32 = opf & 4286578687u32;
opf = __v8;
}
// For a bioset without a percpu cache, or when the percpu cache was
// empty, try a slab allocation with optimistic GFP_ flags before
// falling back to the mempool.
if bio.is_null() {
p = alloc_hooks!(unsafe { kmem_cache_alloc_noprof((*bs).bio_slab, gfp) });
if !p.is_null() {
bio = ((unsafe { (p as *const u8).wrapping_offset((*bs).front_pad as isize) }) as *mut ()) as *mut bio;
}
}
if !bio.is_null() && (nr_vecs as i32) > 4 {
let bvs: *mut biovec_slab = biovec_slab(nr_vecs);
// Upgrade nr_vecs to take full advantage of the allocation.
// We also rely on this in bio_free().
nr_vecs = (unsafe { (*bvs).nr_vecs }) as u16;
bvecs = alloc_hooks!(unsafe { kmem_cache_alloc_noprof((*bvs).slab, gfp) }) as *mut bio_vec;
if unlikely__2!(bvecs.is_null()) != 0 {
unsafe { kmem_cache_free((*bs).bio_slab, p) };
bio = core::ptr::null_mut::<bio>();
}
}
if unlikely__2!(bio.is_null()) != 0 {
// Give up if we are not allow to sleep as non-blocking mempool
// allocations just go back to the slab allocation.
if saved_gfp & (BIT!(10i32) as u32) == 0 {
return core::ptr::null_mut::<bio>();
}
punt_bios_to_rescuer(bs);
// Don't rob the mempools by returning to the per-CPU cache if
// we're tight on memory.
let __v20: u32 = opf & 4286578687u32;
opf = __v20;
p = alloc_hooks!(unsafe { mempool_alloc_noprof(core::ptr::addr_of_mut!((*bs).bio_pool), saved_gfp) });
bio = ((unsafe { (p as *const u8).wrapping_offset((*bs).front_pad as isize) }) as *mut ()) as *mut bio;
if (nr_vecs as i32) > 4 {
nr_vecs = 256;
bvecs = alloc_hooks!(unsafe { mempool_alloc_noprof(core::ptr::addr_of_mut!((*bs).bvec_pool), saved_gfp) }) as *mut bio_vec;
}
}
if nr_vecs != 0 && (nr_vecs as i32) <= 4 {
bio_init_inline(bio, bdev, nr_vecs, opf);
} else {
bio_init(bio, bdev, bvecs, nr_vecs, opf);
}
unsafe {
(*bio).bi_pool = bs;
}
bio
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_bio_alloc_bioset_898: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut block_device, u16, u32, u32, *mut bio_set) -> *mut bio>, *mut ()>(Some(bio_alloc_bioset)) };

/// bio_kmalloc - kmalloc a bio
/// @nr_vecs:	number of bio_vecs to allocate
/// @gfp_mask:   the GFP_* mask given to the slab allocator
///
/// Use kmalloc to allocate a bio (including bvecs).  The bio must be initialized
/// using bio_init() before use.  To free a bio returned from this function use
/// kfree() after calling bio_uninit().  A bio returned from this function can
/// be reused by calling bio_uninit() before calling bio_init() again.
///
/// Note that unlike bio_alloc() or bio_alloc_bioset() allocations from this
/// function are not backed by a mempool can fail.  Do not use this function
/// for allocations in the file system I/O path.
///
/// Returns: Pointer to new bio on success, NULL on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.bio.bio_kmalloc")]
extern "C" fn bio_kmalloc(nr_vecs: u16, gfp_mask: u32) -> *mut bio {
let mut __slate_compound_2: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_4: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
if (nr_vecs as i32) > 1024 {
return core::ptr::null_mut::<bio>();
}
let mut __v0: *mut ();
let mut __v1: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_noprof((core::mem::size_of::<bio>() as u64).wrapping_add(((nr_vecs as i32) as u64).wrapping_mul(core::mem::size_of::<bio_vec>() as u64)), gfp_mask, unsafe { *{
__slate_compound_2 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_2)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_noprof((core::mem::size_of::<bio>() as u64).wrapping_add(((nr_vecs as i32) as u64).wrapping_mul(core::mem::size_of::<bio_vec>() as u64)), gfp_mask, unsafe { *{
__slate_compound_4 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_4)
} });
}
__v1 = _res;
__v0 = __v1;
__v0 as *mut bio
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_bio_kmalloc_899: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u16, u32) -> *mut bio>, *mut ()>(Some(bio_kmalloc)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.bio.zero_fill_bio")]
extern "C" fn zero_fill_bio(bio: *mut bio) {
let mut bv: bio_vec = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(bv) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let mut iter: bvec_iter = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(iter) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 20usize);
}
iter = unsafe { (*bio).bi_iter };
loop {
let __v0: bool = if iter.bi_size != 0 {
bv = bio_iter_iovec!(bio, iter);
true
} else {
false
};
if !__v0 {
break;
}
memzero_bvec(unsafe { &(*core::ptr::addr_of_mut!(bv)) });
bio_advance_iter_single(bio as *const bio, unsafe { &mut (*core::ptr::addr_of_mut!(iter)) }, bv.bv_len);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_zero_fill_bio_900: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut bio)>, *mut ()>(Some(zero_fill_bio)) };

/// bio_truncate - truncate the bio to small size of @new_size
/// @bio:	the bio to be truncated
/// @new_size:	new size for truncating the bio
///
/// Description:
///   Truncate the bio to new size of @new_size. If bio_op(bio) is
///   REQ_OP_READ, zero the truncated part. This function should only
///   be used for handling corner cases, such as bio eod.
fn bio_truncate(bio: *mut bio, new_size: u32) {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut offset: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut u64 = core::ptr::addr_of_mut!(offset) as *mut u64;
let mut truncated: bool = unsafe { core::mem::zeroed() };
let mut done: u32 = unsafe { core::mem::zeroed() };
let mut iter: core::mem::MaybeUninit<bvec_iter> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut bvec_iter = core::ptr::addr_of_mut!(iter) as *mut bvec_iter;
let mut bv: core::mem::MaybeUninit<bio_vec> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut bio_vec = core::ptr::addr_of_mut!(bv) as *mut bio_vec;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 16usize);
let __slate_auto_bytes_2 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 20usize);
done = 0;
truncated = false;
if new_size >= (*bio).bi_iter.bi_size {
return;
} else {
if (bio_op(bio as *const bio) as u32) == 0 {
*__slate_slot_6 = (*bio).bi_iter;
loop {
if (*__slate_slot_6).bi_size != 0 {
*__slate_slot_7 = bio_iter_iovec!(bio, *__slate_slot_6);
__v0 = true;
} else {
__v0 = false;
}
if __v0 {
if done.wrapping_add((*__slate_slot_7).bv_len) > new_size {
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
if !truncated {
*__slate_slot_3 = new_size.wrapping_sub(done) as u64;
} else {
*__slate_slot_3 = 0;
}
memzero_page((*__slate_slot_7).bv_page, ((*__slate_slot_7).bv_offset as u64).wrapping_add(*__slate_slot_3), ((*__slate_slot_7).bv_len as u64).wrapping_sub(*__slate_slot_3));
truncated = true;
}
done = done.wrapping_add((*__slate_slot_7).bv_len);
bio_advance_iter_single(bio as *const bio, &mut (*core::ptr::addr_of_mut!(*__slate_slot_6)), (*__slate_slot_7).bv_len);
} else {
break;
}
}
}
(*bio).bi_iter.bi_size = new_size;
// Don't touch bvec table here and make it really immutable, since
// fs bio user has to retrieve all pages via bio_for_each_segment_all
// in its .end_bio() callback.
//
// It is enough to truncate bio by updating .bi_size since we can make
// correct bvec with the updated .bi_size for drivers.
}
}
}

/// guard_bio_eod - truncate a BIO to fit the block device
/// @bio:	bio to truncate
///
/// This allows us to do IO even on the odd last sectors of a device, even if the
/// block size is some multiple of the physical sector size.
///
/// We'll just truncate the bio to the size of the device, and clear the end of
/// the buffer head manually.  Truly out-of-range accesses will turn into actual
/// I/O errors, this only handles the "we need to be able to do I/O at the final
/// sector" case.
#[unsafe(no_mangle)]
extern "C" fn guard_bio_eod(bio: *mut bio) {
let mut maxsector: u64 = bdev_nr_sectors(unsafe { (*bio).bi_bdev });
if maxsector == 0 {
return;
}
// If the *whole* IO is past the end of the device,
// let it through, and the IO layer will turn it into
// an EIO.
if unlikely__2!((unsafe { (*bio).bi_iter.bi_sector }) >= maxsector) != 0 {
return;
}
maxsector = maxsector.wrapping_sub(unsafe { (*bio).bi_iter.bi_sector });
if likely!((((unsafe { (*bio).bi_iter.bi_size }) >> 9i32) as u64) <= maxsector) != 0 {
return;
}
bio_truncate(bio, (maxsector << 9i32) as u32);
}

fn __bio_alloc_cache_prune(cache: *mut bio_alloc_cache, nr: u32) -> i32 {
let mut i: u32 = 0;
loop {
let __v6: *mut bio = unsafe { (*cache).free_list };
let bio: *mut bio = __v6;
if __v6.is_null() {
break;
}
unsafe {
(*cache).free_list = (*bio).bi_next;
}
let __v1: u32 = unsafe { (*cache).nr };
unsafe {
(*cache).nr = __v1.wrapping_sub(1);
}
kmemleak_alloc(bio_slab_addr(bio) as *const (), (unsafe { kmem_cache_size((*(*bio).bi_pool).bio_slab) }) as u64, 1i32, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32));
bio_free(bio);
let __v4: u32 = i.wrapping_add(1);
i = __v4;
if __v4 == nr {
break;
}
}
i as i32
}

fn bio_alloc_cache_prune(cache: *mut bio_alloc_cache, mut nr: u32) {
let __v0: i32 = __bio_alloc_cache_prune(cache, nr);
nr = nr.wrapping_sub(__v0 as u32);
let mut __v3: *mut bio;
loop {
if !(__native_word__21!() || ((const { core::mem::size_of::<*mut bio>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_901() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*cache).free_list) as *const *mut bio) };
if __v3.is_null() {
bio_alloc_irq_cache_splice(cache);
__bio_alloc_cache_prune(cache, nr);
}
}

#[unsafe(link_section = ".text.slate_distinct.bio.bio_cpu_dead")]
extern "C" fn bio_cpu_dead(cpu: u32, node: *mut hlist_node) -> i32 {
let bs: *mut bio_set = hlist_entry_safe__4!(node, bio_set);
if !unsafe { (*bs).cache }.is_null() {
let cache: *mut bio_alloc_cache = per_cpu_ptr__8!(unsafe { (*bs).cache }, cpu);
bio_alloc_cache_prune(cache, 1u32.wrapping_neg());
}
0
}

fn bio_alloc_cache_destroy(bs: *mut bio_set) {
let mut cpu: i32 = 0;
if unsafe { (*bs).cache }.is_null() {
return;
}
cpuhp_state_remove_instance_nocalls(21i32, unsafe { core::ptr::addr_of_mut!((*bs).cpuhp_dead) });
loop {
cpu = find_next_bit(cpumask_bits!((unsafe { core::ptr::addr_of_mut!(__cpu_possible_mask) }) as *const cpumask), 64u64, cpu as u64) as i32;
if (cpu as u32) >= 64 {
break;
}
let cache: *mut bio_alloc_cache = per_cpu_ptr__8!(unsafe { (*bs).cache }, cpu);
bio_alloc_cache_prune(cache, 1u32.wrapping_neg());
cpu = cpu.wrapping_add(1);
}
unsafe { free_percpu((*bs).cache as *mut ()) };
unsafe {
(*bs).cache = core::ptr::null_mut::<bio_alloc_cache>();
}
}

fn bio_put_percpu_cache(bio: *mut bio) {
let mut pfo_val__: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut u32 = core::ptr::addr_of_mut!(pfo_val__) as *mut u32;
let mut pscr_ret__: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut i32 = core::ptr::addr_of_mut!(pscr_ret__) as *mut i32;
let mut cache: core::mem::MaybeUninit<*mut bio_alloc_cache> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut *mut bio_alloc_cache = core::ptr::addr_of_mut!(cache) as *mut *mut bio_alloc_cache;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
__verify_pcpu_ptr__13!();
loop {
preempt_count_inc!();
core::arch::asm!("", options(nostack, att_syntax));
if true {
break;
}
}
__this_cpu_preempt_check(c"read".as_ptr() as *const u8);
let __slate_auto_bytes = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
__verify_pcpu_ptr__3!();
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __t0: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t0, sym cpu_number, options(pure, readonly, nostack, att_syntax));
*__slate_slot_9 = unsafe { core::mem::transmute::<u32, u32>(__t0 as u32) };
*__slate_slot_15 = (*__slate_slot_9 as u64) as i32;
*__slate_slot_20 = ((((*(*bio).bi_pool).cache as u64) as *mut bio_alloc_cache) as u64).wrapping_add(__per_cpu_offset[*__slate_slot_15 as usize]) as *mut bio_alloc_cache;
while false {
}
'__join_3: {
let __v6: u32 = core::ptr::read_volatile(core::ptr::addr_of_mut!((*(*__slate_slot_20)).nr_irq) as *const u32);
if __v6.wrapping_add((*(*__slate_slot_20)).nr) <= 256 {
if in_task!() {
bio_uninit(bio);
(*bio).bi_next = (*(*__slate_slot_20)).free_list;
// Not necessary but helps not to iopoll already freed bios
(*bio).bi_bdev = core::ptr::null_mut::<block_device>();
(*(*__slate_slot_20)).free_list = bio;
let __v2: *mut bio_alloc_cache = *__slate_slot_20;
let __v1: u32 = (*__v2).nr;
(*__v2).nr = __v1.wrapping_add(1);
kmemleak_free(bio_slab_addr(bio) as *const ());
} else {
if in_hardirq!() != 0 {
lockdep_assert_irqs_disabled!();
bio_uninit(bio);
(*bio).bi_next = (*(*__slate_slot_20)).free_list_irq;
(*(*__slate_slot_20)).free_list_irq = bio;
let __v5: *mut bio_alloc_cache = *__slate_slot_20;
let __v4: u32 = (*__v5).nr_irq;
(*__v5).nr_irq = __v4.wrapping_add(1);
kmemleak_free(bio_slab_addr(bio) as *const ());
} else {
break '__join_3;
}
}
loop {
core::arch::asm!("", options(nostack, att_syntax));
if unlikely__2!(preempt_count_dec_and_test!()) != 0 {
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
return;
}
}
loop {
core::arch::asm!("", options(nostack, att_syntax));
if unlikely__2!(preempt_count_dec_and_test!()) != 0 {
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
bio_free(bio);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_903: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_904: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

/// bio_put - release a reference to a bio
/// @bio:   bio to release reference to
///
/// Description:
///   Put a reference to a &struct bio, either one you have gotten with
///   bio_alloc, bio_get or bio_clone_*. The last put of a bio will free it.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.bio.bio_put")]
extern "C" fn bio_put(bio: *mut bio) {
if unlikely__2!(bio_flagged(bio as *const bio, 4u32)) != 0 {
loop {
if unlikely__2!(atomic_read((unsafe { core::ptr::addr_of_mut!((*bio).__bi_cnt) }) as *const __SlateRecord_types_h_190) == 0) != 0 {
loop {
unsafe {
core::arch::asm!("905: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 905b - ., 3; .popsection\n# {0}", const 905 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_bio_put_0, sym __slate_asm_string_bio_put_1, const 845 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
if !atomic_dec_and_test(unsafe { core::ptr::addr_of_mut!((*bio).__bi_cnt) }) {
return;
}
}
if (unsafe { (*bio).bi_opf }) & 8388608u32 != 0 {
bio_put_percpu_cache(bio);
} else {
bio_free(bio);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_bio_put_906: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut bio)>, *mut ()>(Some(bio_put)) };

fn __bio_clone(bio: *mut bio, bio_src: *mut bio, gfp: u32) -> i32 {
bio_set_flag(bio, 1u32);
unsafe {
(*bio).bi_ioprio = (*bio_src).bi_ioprio;
}
unsafe {
(*bio).bi_write_hint = (*bio_src).bi_write_hint;
}
unsafe {
(*bio).bi_write_stream = (*bio_src).bi_write_stream;
}
unsafe {
(*bio).bi_bvec_gap_bit = (*bio_src).bi_bvec_gap_bit;
}
unsafe {
(*bio).bi_iter = (*bio_src).bi_iter;
}
unsafe {
(*bio).bi_io_vec = (*bio_src).bi_io_vec;
}
if !unsafe { (*bio).bi_bdev }.is_null() {
let __v0: bool = (unsafe { (*bio).bi_bdev }) == unsafe { (*bio_src).bi_bdev } && bio_flagged(bio_src as *const bio, 10u32);
if __v0 {
bio_set_flag(bio, 10u32);
}
unsafe { bio_clone_blkg_association(bio, bio_src) };
}
if bio_crypt_clone(bio, bio_src, gfp) < 0 {
return 12i32.wrapping_neg();
}
let __v1: bool = !bio_integrity(bio_src).is_null() && bio_integrity_clone(bio, bio_src, gfp) < 0;
if __v1 {
return 12i32.wrapping_neg();
}
0
}

/// bio_alloc_clone - clone a bio that shares the original bio's biovec
/// @bdev: block_device to clone onto
/// @bio_src: bio to clone from
/// @gfp: allocation priority
/// @bs: bio_set to allocate from
///
/// Allocate a new bio that is a clone of @bio_src. This reuses the bio_vecs
/// pointed to by @bio_src->bi_io_vec, and clones the iterator pointing to
/// the current position in it.  The caller owns the returned bio, but not
/// the bio_vecs, and must ensure the bio is freed before the memory
/// pointed to by @bio_Src->bi_io_vecs.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.bio.bio_alloc_clone")]
extern "C" fn bio_alloc_clone(bdev: *mut block_device, bio_src: *mut bio, gfp: u32, bs: *mut bio_set) -> *mut bio {
let bio: *mut bio = bio_alloc_bioset(bdev, 0u16, unsafe { (*bio_src).bi_opf }, gfp, bs);
if bio.is_null() {
return core::ptr::null_mut::<bio>();
}
if __bio_clone(bio, bio_src, gfp) < 0 {
bio_put(bio);
return core::ptr::null_mut::<bio>();
}
bio
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_bio_alloc_clone_907: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut block_device, *mut bio, u32, *mut bio_set) -> *mut bio>, *mut ()>(Some(bio_alloc_clone)) };

/// bio_init_clone - clone a bio that shares the original bio's biovec
/// @bdev: block_device to clone onto
/// @bio: bio to clone into
/// @bio_src: bio to clone from
/// @gfp: allocation priority
///
/// Initialize a new bio in caller provided memory that is a clone of @bio_src.
/// The same bio_vecs reuse and bio lifetime rules as bio_alloc_clone() apply.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.bio.bio_init_clone")]
extern "C" fn bio_init_clone(bdev: *mut block_device, bio: *mut bio, bio_src: *mut bio, gfp: u32) -> i32 {
let mut ret: i32 = 0;
bio_init(bio, bdev, core::ptr::null_mut::<bio_vec>(), 0u16, unsafe { (*bio_src).bi_opf });
ret = __bio_clone(bio, bio_src, gfp);
if ret != 0 {
bio_uninit(bio);
}
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_bio_init_clone_908: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut block_device, *mut bio, *mut bio, u32) -> i32>, *mut ()>(Some(bio_init_clone)) };

/// bio_full - check if the bio is full
/// @bio:	bio to check
/// @len:	length of one segment to be added
///
/// Return true if @bio is full and one segment with @len bytes can't be
/// added to the bio, otherwise return false
fn bio_full(bio: *mut bio, len: u32) -> bool {
if ((unsafe { (*bio).bi_vcnt }) as i32) >= ((unsafe { (*bio).bi_max_vecs }) as i32) {
return true;
}
if (unsafe { (*bio).bi_iter.bi_size }) > 4294967295u32.wrapping_sub(len) {
return true;
}
false
}

fn bvec_try_merge_page(bv: *mut bio_vec, page: *mut page, len: u32, off: u32) -> bool {
let bv_end: u64 = unsafe { (*bv).bv_offset }.wrapping_add(unsafe { (*bv).bv_len }) as u64;
let vec_end_addr: u64 = page_to_phys!(unsafe { (*bv).bv_page }).wrapping_add(bv_end).wrapping_sub(1);
let page_addr: u64 = page_to_phys!(page);
if vec_end_addr.wrapping_add(1) != page_addr.wrapping_add(off as u64) {
return false;
}
let __v0: bool = xen_domain!() && !unsafe { xen_biovec_phys_mergeable(bv as *const bio_vec, page as *const page) };
if __v0 {
return false;
}
if vec_end_addr & !(_AC!() << 12i32).wrapping_sub(1) != page_addr.wrapping_add(off as u64) & !(_AC!() << 12i32).wrapping_sub(1) {
if IS_ENABLED!() != 0 {
return false;
}
if unsafe { (*bv).bv_page }.wrapping_offset((bv_end / (_AC!() << 12i32)) as isize) != page.wrapping_offset(((off as u64) / (_AC!() << 12i32)) as isize) {
return false;
}
}
let __v2: u32 = unsafe { (*bv).bv_len };
unsafe {
(*bv).bv_len = __v2.wrapping_add(len);
}
true
}

/// Try to merge a page into a segment, while obeying the hardware segment
/// size limit.
///
/// This is kept around for the integrity metadata, which is still tries
/// to build the initial bio to the hardware limit and doesn't have proper
/// helpers to split.  Hopefully this will go away soon.
#[unsafe(no_mangle)]
extern "C" fn bvec_try_merge_hw_page(q: *mut request_queue, bv: *mut bio_vec, page: *mut page, len: u32, offset: u32) -> bool {
let mask: u64 = queue_segment_boundary(q as *const request_queue);
let addr1: u64 = bvec_phys(bv as *const bio_vec);
let addr2: u64 = page_to_phys!(page).wrapping_add(offset as u64).wrapping_add(len as u64).wrapping_sub(1);
if addr1 | mask != addr2 | mask {
return false;
}
if len > queue_max_segment_size(q as *const request_queue).wrapping_sub(unsafe { (*bv).bv_len }) {
return false;
}
bvec_try_merge_page(bv, page, len, offset)
}

/// __bio_add_page - add page(s) to a bio in a new segment
/// @bio: destination bio
/// @page: start page to add
/// @len: length of the data to add, may cross pages
/// @off: offset of the data relative to @page, may cross pages
///
/// Add the data at @page + @off to @bio as a new bvec.  The caller must ensure
/// that @bio has space for another bvec.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.bio.__bio_add_page")]
extern "C" fn __bio_add_page(bio: *mut bio, page: *mut page, len: u32, off: u32) {
let mut __v0: i64;
{
let __ret_warn_on: i32 = !(!bio_flagged(bio as *const bio, 1u32)) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("909: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 909b - ., 3; .popsection\n# {0}", const 909 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___bio_add_page_1, sym __slate_asm_string___bio_add_page_2, const 1009 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("910: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 910b - ., 4; .popsection\n# {0}", const 910 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
}
let mut __v5: i64;
let __ret_warn_on: i32 = !(!bio_full(bio, len)) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("911: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 911b - ., 3; .popsection\n# {0}", const 911 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___bio_add_page_6, sym __slate_asm_string___bio_add_page_7, const 1010 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("912: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 912b - ., 4; .popsection\n# {0}", const 912 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v5 = (__ret_warn_on != 0) as i64;
if is_pci_p2pdma_page(page as *const page) {
unsafe {
(*bio).bi_opf = (*bio).bi_opf | 16384u32;
}
}
bvec_set_page(unsafe { (*bio).bi_io_vec }.wrapping_offset(((unsafe { (*bio).bi_vcnt }) as i32) as isize), page, len, off);
let __v14: u32 = unsafe { (*bio).bi_iter.bi_size };
unsafe {
(*bio).bi_iter.bi_size = __v14.wrapping_add(len);
}
unsafe {
(*bio).bi_vcnt = ((*bio).bi_vcnt as i32).wrapping_add(1) as u16;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___bio_add_page_913: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut bio, *mut page, u32, u32)>, *mut ()>(Some(__bio_add_page)) };

/// bio_add_virt_nofail - add data in the direct kernel mapping to a bio
/// @bio: destination bio
/// @vaddr: data to add
/// @len: length of the data to add, may cross pages
///
/// Add the data at @vaddr to @bio.  The caller must have ensure a segment
/// is available for the added data.  No merging into an existing segment
/// will be performed.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.bio.bio_add_virt_nofail")]
extern "C" fn bio_add_virt_nofail(bio: *mut bio, vaddr: *mut (), len: u32) {
__bio_add_page(bio, virt_to_page!(vaddr), len, offset_in_page!(vaddr) as u32);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_bio_add_virt_nofail_914: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut bio, *mut (), u32)>, *mut ()>(Some(bio_add_virt_nofail)) };

/// bio_add_page	-	attempt to add page(s) to bio
/// @bio: destination bio
/// @page: start page to add
/// @len: vec entry length, may cross pages
/// @offset: vec entry offset relative to @page, may cross pages
///
/// Attempt to add page(s) to the bio_vec maplist. This will only fail
/// if either bio->bi_vcnt == bio->bi_max_vecs or it's a cloned bio.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.bio.bio_add_page")]
extern "C" fn bio_add_page(bio: *mut bio, page: *mut page, len: u32, offset: u32) -> i32 {
let mut __v0: i64;
{
let __ret_warn_on: i32 = !(!bio_flagged(bio as *const bio, 1u32)) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("915: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 915b - ., 3; .popsection\n# {0}", const 915 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_bio_add_page_1, sym __slate_asm_string_bio_add_page_2, const 1050 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("916: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 916b - ., 4; .popsection\n# {0}", const 916 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
}
if __v0 != 0 {
return 0;
}
let mut __v5: i64;
let __ret_warn_on: i32 = (len == 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("917: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 917b - ., 3; .popsection\n# {0}", const 917 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_bio_add_page_6, sym __slate_asm_string_bio_add_page_7, const 1052 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("918: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 918b - ., 4; .popsection\n# {0}", const 918 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v5 = (__ret_warn_on != 0) as i64;
if __v5 != 0 {
return 0;
}
if (unsafe { (*bio).bi_iter.bi_size }) > 4294967295u32.wrapping_sub(len) {
return 0;
}
if ((unsafe { (*bio).bi_vcnt }) as i32) > 0 {
let bv: *mut bio_vec = unsafe { (*bio).bi_io_vec }.wrapping_offset(((unsafe { (*bio).bi_vcnt }) as i32).wrapping_sub(1) as isize);
if !zone_device_pages_compatible((unsafe { (*bv).bv_page }) as *const page, page as *const page) {
return 0;
}
let __v10: bool = zone_device_pages_have_same_pgmap((unsafe { (*bv).bv_page }) as *const page, page as *const page) && bvec_try_merge_page(bv, page, len, offset);
if __v10 {
let __v12: u32 = unsafe { (*bio).bi_iter.bi_size };
unsafe {
(*bio).bi_iter.bi_size = __v12.wrapping_add(len);
}
return len as i32;
}
}
if ((unsafe { (*bio).bi_vcnt }) as i32) >= ((unsafe { (*bio).bi_max_vecs }) as i32) {
return 0;
}
__bio_add_page(bio, page, len, offset);
len as i32
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_bio_add_page_919: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut bio, *mut page, u32, u32) -> i32>, *mut ()>(Some(bio_add_page)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.bio.bio_add_folio_nofail")]
extern "C" fn bio_add_folio_nofail(bio: *mut bio, folio: *mut folio, len: u64, off: u64) {
let nr: u64 = off / (_AC!() << 12i32);
let mut __v0: i64;
let __ret_warn_on: i32 = (len > 4294967295u64) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("920: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 920b - ., 3; .popsection\n# {0}", const 920 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_bio_add_folio_nofail_1, sym __slate_asm_string_bio_add_folio_nofail_2, const 1081 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("921: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 921b - ., 4; .popsection\n# {0}", const 921 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
__bio_add_page(bio, folio_page!(folio, nr), len as u32, (off % (_AC!() << 12i32)) as u32);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_bio_add_folio_nofail_922: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut bio, *mut folio, u64, u64)>, *mut ()>(Some(bio_add_folio_nofail)) };

/// bio_add_folio - Attempt to add part of a folio to a bio.
/// @bio: BIO to add to.
/// @folio: Folio to add.
/// @len: How many bytes from the folio to add.
/// @off: First byte in this folio to add.
///
/// Filesystems that use folios can call this function instead of calling
/// bio_add_page() for each page in the folio.  If @off is bigger than
/// PAGE_SIZE, this function can create a bio_vec that starts in a page
/// after the bv_page.  BIOs do not support folios that are 4GiB or larger.
///
/// Return: Whether the addition was successful.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.bio.bio_add_folio")]
extern "C" fn bio_add_folio(bio: *mut bio, folio: *mut folio, len: u64, off: u64) -> bool {
let nr: u64 = off / (_AC!() << 12i32);
if len > 4294967295u64 {
return false;
}
bio_add_page(bio, folio_page!(folio, nr), len as u32, (off % (_AC!() << 12i32)) as u32) > 0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_bio_add_folio_923: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut bio, *mut folio, u64, u64) -> bool>, *mut ()>(Some(bio_add_folio)) };

// bio_add_vmalloc_chunk - add a vmalloc chunk to a bio
// @bio: destination bio
// @vaddr: vmalloc address to add
// @len: total length in bytes of the data to add
//
// Add data starting at @vaddr to @bio and return how many bytes were added.
// This may be less than the amount originally asked.  Returns 0 if no data
// could be added to @bio.
//
// This helper calls flush_kernel_vmap_range() for the range added.  For reads
// the caller still needs to manually call invalidate_kernel_vmap_range() in
// the completion handler.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_bio_add_vmalloc_chunk_927: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut bio, *mut (), u32) -> u32>, *mut ()>(Some(bio_add_vmalloc_chunk)) };

/// bio_add_vmalloc - add a vmalloc region to a bio
/// @bio: destination bio
/// @vaddr: vmalloc address to add
/// @len: total length in bytes of the data to add
///
/// Add data starting at @vaddr to @bio.  Return %true on success or %false if
/// @bio does not have enough space for the payload.
///
/// This helper calls flush_kernel_vmap_range() for the range added.  For reads
/// the caller still needs to manually call invalidate_kernel_vmap_range() in
/// the completion handler.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.bio.bio_add_vmalloc")]
extern "C" fn bio_add_vmalloc(bio: *mut bio, mut vaddr: *mut (), mut len: u32) -> bool {
loop {
let added: u32 = bio_add_vmalloc_chunk(bio, vaddr, len);
if added == 0 {
return false;
}
vaddr = (unsafe { (vaddr as *const u8).wrapping_offset(added as isize) }) as *mut ();
len = len.wrapping_sub(added);
if len == 0 {
break;
}
}
true
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_bio_add_vmalloc_928: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut bio, *mut (), u32) -> bool>, *mut ()>(Some(bio_add_vmalloc)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.bio.__bio_release_pages")]
extern "C" fn __bio_release_pages(bio: *mut bio, mark_dirty: bool) {
let mut fi: folio_iter = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(fi) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 48usize);
}
bio_first_folio(unsafe { &mut (*core::ptr::addr_of_mut!(fi)) }, bio, 0i32);
while !fi.folio.is_null() {
let mut nr_pages: u64 = 0;
if mark_dirty {
folio_lock(fi.folio);
unsafe { folio_mark_dirty(fi.folio) };
unsafe { folio_unlock(fi.folio) };
}
nr_pages = (fi.offset.wrapping_add(fi.length).wrapping_sub(1) / (_AC!() << 12i32)).wrapping_sub(fi.offset / (_AC!() << 12i32)).wrapping_add(1);
unsafe { unpin_user_folio(fi.folio, nr_pages) };
bio_next_folio(unsafe { &mut (*core::ptr::addr_of_mut!(fi)) }, bio);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___bio_release_pages_929: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut bio, bool)>, *mut ()>(Some(__bio_release_pages)) };

#[unsafe(no_mangle)]
extern "C" fn bio_iov_iter_set(bio: *mut bio, iter: *const iov_iter) -> bool {
if !iov_iter_is_bvec(iter) {
return false;
}
let mut __v0: i64;
let __ret_warn_on: i32 = ((unsafe { (*bio).bi_max_vecs }) != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("930: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 930b - ., 3; .popsection\n# {0}", const 930 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_bio_iov_iter_set_1, sym __slate_asm_string_bio_iov_iter_set_2, const 1190 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("931: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 931b - ., 4; .popsection\n# {0}", const 931 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
unsafe {
(*bio).bi_io_vec = (*iter).__slate_anon_4.__slate_anon_1.__slate_anon_0.bvec as *mut bio_vec;
}
unsafe {
(*bio).bi_iter.bi_idx = 0;
}
unsafe {
(*bio).bi_iter.bi_offset = (*iter).iov_offset as u32;
}
unsafe {
(*bio).bi_iter.bi_size = iov_iter_count(iter) as u32;
}
bio_set_flag(bio, 1u32);
true
}

/// Aligns the bio size to the len_align_mask, releasing excessive bio vecs that
/// __bio_iov_iter_get_pages may have inserted, and reverts the trimmed length
/// for the next iteration.
fn bio_iov_iter_align_down(bio: *mut bio, iter: *mut iov_iter, mut bv: *mut bio_vec, len_align_mask: u32) -> i32 {
let mut nbytes: u64 = ((unsafe { (*bio).bi_iter.bi_size }) & len_align_mask) as u64;
if nbytes == 0 {
return 0;
}
unsafe { iov_iter_revert(iter, nbytes) };
unsafe {
(*bio).bi_iter.bi_size = ((*bio).bi_iter.bi_size as u64).wrapping_sub(nbytes) as u32;
}
while nbytes >= ((unsafe { (*bv).bv_len }) as u64) {
if bio_flagged(bio as *const bio, 0u32) {
unsafe { unpin_user_page((*bv).bv_page) };
}
let __v5: u16 = ((unsafe { (*bio).bi_vcnt }) as i32).wrapping_sub(1) as u16;
unsafe {
(*bio).bi_vcnt = __v5;
}
if __v5 == 0 {
return 14i32.wrapping_neg();
}
nbytes = nbytes.wrapping_sub((unsafe { (*bv).bv_len }) as u64);
let __v8: *mut bio_vec = bv;
bv = __v8.wrapping_offset(-(1 as isize));
}
unsafe {
(*bv).bv_len = ((*bv).bv_len as u64).wrapping_sub(nbytes) as u32;
}
0
}

fn bio_iov_bvec_aligned(bio: *const bio, mem_align_mask: u32) -> bool {
let mut iter: bvec_iter = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(iter) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 20usize);
}
let mut bv: bio_vec = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(bv) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
// Correct callers never break the alignment requirements, so this
// exhaustive check is only paid for in debug builds.
iter = unsafe { (*bio).bi_iter };
loop {
let __v0: bool = if iter.bi_size != 0 {
bv = mp_bvec_iter_bvec((unsafe { (*bio).bi_io_vec }) as *const bio_vec, iter);
true
} else {
false
};
if !__v0 {
break;
}
if (bv.bv_offset | bv.bv_len) & mem_align_mask != 0 {
return false;
}
bvec_iter_advance_single((unsafe { (*bio).bi_io_vec }) as *const bio_vec, unsafe { &mut (*core::ptr::addr_of_mut!(iter)) }, bv.bv_len);
}
true
}

/// bio_iov_iter_get_pages - add user or kernel pages to a bio
/// @bio: bio to add pages to
/// @iter: iov iterator describing the region to be added
/// @mem_align_mask: the mask the source address and length must be aligned to,
/// 	0 for no requirement
/// @len_align_mask: the mask to align the total size to, 0 for any length
///
/// This takes either an iterator pointing to user memory, or one pointing to
/// kernel pages (BVEC iterator). If we're adding user pages, we pin them and
/// map them into the kernel. On IO completion, the caller should put those
/// pages. For bvec based iterators bio_iov_iter_get_pages() uses the provided
/// bvecs rather than copying them. Hence anyone issuing kiocb based IO needs
/// to ensure the bvecs and pages stay referenced until the submitted I/O is
/// completed by a call to ->ki_complete() or returns with an error other than
/// -EIOCBQUEUED. The caller needs to check if the bio is flagged BIO_NO_PAGE_REF
/// on IO completion. If it isn't, then pages should be released.
///
/// The function tries, but does not guarantee, to pin as many pages as
/// fit into the bio, or are requested in @iter, whatever is smaller. If
/// MM encounters an error pinning the requested pages, it stops. Error
/// is returned only if 0 pages could be pinned.
#[unsafe(no_mangle)]
extern "C" fn bio_iov_iter_get_pages(bio: *mut bio, iter: *mut iov_iter, mem_align_mask: u32, len_align_mask: u32) -> i32 {
let mut flags: u32 = 0;
let mut __v0: i64;
let __ret_warn_on: i32 = !(!bio_flagged(bio as *const bio, 1u32)) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("932: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 932b - ., 3; .popsection\n# {0}", const 932 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_bio_iov_iter_get_pages_1, sym __slate_asm_string_bio_iov_iter_get_pages_2, const 1289 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("933: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 933b - ., 4; .popsection\n# {0}", const 933 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
return 5i32.wrapping_neg();
}
if bio_iov_iter_set(bio, iter as *const iov_iter) {
let __v5: bool = iov_iter_is_bvec(iter as *const iov_iter) && !bio_iov_bvec_aligned(bio as *const bio, mem_align_mask);
if __v5 {
return 22i32.wrapping_neg();
}
unsafe { iov_iter_advance(iter, (*bio).bi_iter.bi_size as u64) };
return 0;
}
if iov_iter_extract_will_pin(iter as *const iov_iter) {
bio_set_flag(bio, 0u32);
}
if !unsafe { (*bio).bi_bdev }.is_null() && blk_queue_pci_p2pdma!(unsafe { (*(*(*bio).bi_bdev).bd_disk).queue }) != 0 {
flags |= 1;
}
loop {
let ret: i64 = unsafe { iov_iter_extract_bvecs(iter, (*bio).bi_io_vec, 4294967295u32.wrapping_sub((*bio).bi_iter.bi_size) as u64, core::ptr::addr_of_mut!((*bio).bi_vcnt), (*bio).bi_max_vecs, mem_align_mask, flags) };
if ret <= 0 {
// A misaligned vector fails the whole I/O.  Release any
// pages pinned by earlier iterations before returning
// since this bio won't be submitted to release them.
if ret == (22i32.wrapping_neg() as i64) {
bio_release_pages(bio, false);
bio_clear_flag(bio, 0u32);
unsafe {
(*bio).bi_vcnt = 0;
}
}
if (unsafe { (*bio).bi_vcnt }) == 0 {
return ret as i32;
}
break;
}
unsafe {
(*bio).bi_iter.bi_size = ((*bio).bi_iter.bi_size as i64).wrapping_add(ret) as u32;
}
let __v12: bool = iov_iter_count(iter as *const iov_iter) != 0 && !bio_full(bio, 0u32);
if !__v12 {
break;
}
}
if is_pci_p2pdma_page((unsafe { (*(*bio).bi_io_vec).bv_page }) as *const page) {
unsafe {
(*bio).bi_opf = (*bio).bi_opf | 16384u32;
}
}
bio_iov_iter_align_down(bio, iter, unsafe { (*bio).bi_io_vec }.wrapping_offset(((unsafe { (*bio).bi_vcnt }) as i32).wrapping_sub(1) as isize), len_align_mask)
}

fn folio_alloc_greedy(gfp: u32, size: *mut u64, minsize: u64) -> *mut folio {
while (unsafe { *size }) > minsize {
let folio: *mut folio = alloc_hooks__4!(unsafe { folio_alloc_noprof(gfp | 65536u32 | 8192u32, get_order(*size) as u32) });
if !folio.is_null() {
return folio;
}
unsafe {
*size = rounddown_pow_of_two!(unsafe { *size }.wrapping_sub(1));
}
}
alloc_hooks__4!(unsafe { folio_alloc_noprof(gfp, get_order(*size) as u32) })
}

fn bio_free_folios(bio: *mut bio) {
let mut bv: *mut bio_vec = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
bv = bio_first_bvec_all(bio);
while i < ((unsafe { (*bio).bi_vcnt }) as i32) {
let folio: *mut folio = bvec_folio(bv as *const bio_vec);
let __v1: bool = !is_zero_folio(folio as *const folio) && !is_huge_zero_folio(folio as *const folio);
if __v1 {
folio_put(folio);
}
i = i.wrapping_add(1);
let __v5: *mut bio_vec = bv;
bv = __v5.wrapping_offset(1 as isize);
}
}

fn bio_iov_iter_bounce_write(bio: *mut bio, iter: *mut iov_iter, __UNIQUE_ID_x__934: u64, minsize: u64) -> i32 {
let mut total_len: u64 = 0;
let mut __v0: u64;
let __UNIQUE_ID_y__935: u64 = iov_iter_count(iter as *const iov_iter);
loop {
if __types_ok__2!(__UNIQUE_ID_x__934, __UNIQUE_ID_y__935) == 0 {
unsafe { __compiletime_assert_936() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__934 < __UNIQUE_ID_y__935 { __UNIQUE_ID_x__934 } else { __UNIQUE_ID_y__935 };
total_len = __v0;
let mut __v2: i64;
{
let __ret_warn_on: i32 = !(!bio_flagged(bio as *const bio, 1u32)) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("937: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 937b - ., 3; .popsection\n# {0}", const 937 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_bio_iov_iter_bounce_write_3, sym __slate_asm_string_bio_iov_iter_bounce_write_4, const 1371 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("938: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 938b - ., 4; .popsection\n# {0}", const 938 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v2 = (__ret_warn_on != 0) as i64;
}
if __v2 != 0 {
return 22i32.wrapping_neg();
}
let mut __v7: i64;
{
let __ret_warn_on: i32 = ((unsafe { (*bio).bi_iter.bi_size }) != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("939: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 939b - ., 3; .popsection\n# {0}", const 939 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_bio_iov_iter_bounce_write_8, sym __slate_asm_string_bio_iov_iter_bounce_write_9, const 1373 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("940: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 940b - ., 4; .popsection\n# {0}", const 940 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v7 = (__ret_warn_on != 0) as i64;
}
if __v7 != 0 {
return 22i32.wrapping_neg();
}
let mut __v12: i64;
let __ret_warn_on: i32 = (((unsafe { (*bio).bi_vcnt }) as i32) >= ((unsafe { (*bio).bi_max_vecs }) as i32)) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("941: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 941b - ., 3; .popsection\n# {0}", const 941 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_bio_iov_iter_bounce_write_13, sym __slate_asm_string_bio_iov_iter_bounce_write_14, const 1375 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("942: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 942b - ., 4; .popsection\n# {0}", const 942 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v12 = (__ret_warn_on != 0) as i64;
if __v12 != 0 {
return 22i32.wrapping_neg();
}
loop {
let mut this_len: u64 = 0;
let mut __v17: u64;
let __UNIQUE_ID_x__943: u64 = total_len;
let __UNIQUE_ID_y__944: i32 = 1048576;
loop {
if __types_ok__9!(__UNIQUE_ID_x__943, __UNIQUE_ID_y__944) == 0 {
unsafe { __compiletime_assert_945() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v17 = if __UNIQUE_ID_x__943 < (__UNIQUE_ID_y__944 as u64) { __UNIQUE_ID_x__943 } else { __UNIQUE_ID_y__944 as u64 };
this_len = __v17;
let mut copied: u64 = 0;
if this_len > minsize.wrapping_mul(2) {
this_len = rounddown_pow_of_two!(this_len);
}
if ((unsafe { (*bio).bi_iter.bi_size }) as u64) > 4294967295u64.wrapping_sub(this_len) {
break;
}
let folio: *mut folio = folio_alloc_greedy(((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32), core::ptr::addr_of_mut!(this_len), minsize);
if folio.is_null() {
break;
}
bio_add_folio_nofail(bio, folio, this_len, 0u64);
if unsafe { (*iter).nofault } {
copied = unsafe { copy_folio_from_iter_atomic(folio, 0u64, this_len, iter) };
} else {
copied = copy_folio_from_iter(folio, 0u64, this_len, iter);
}
if copied < this_len {
// Need to revert the iov iter for all bytes we have
// copied.
//
// However the bio size differs from the real copied
// bytes as @this_len is queued but only advanced
// less than that.
// Need to compensate that for the revert.
unsafe { iov_iter_revert(iter, ((*bio).bi_iter.bi_size as u64).wrapping_sub(this_len).wrapping_add(copied)) };
bio_free_folios(bio);
return 14i32.wrapping_neg();
}
total_len = total_len.wrapping_sub(this_len);
if !(total_len != 0 && ((unsafe { (*bio).bi_vcnt }) as i32) < ((unsafe { (*bio).bi_max_vecs }) as i32)) {
break;
}
}
if (unsafe { (*bio).bi_iter.bi_size }) == 0 {
return 12i32.wrapping_neg();
}
bio_iov_iter_align_down(bio, iter, unsafe { (*bio).bi_io_vec }.wrapping_offset(((unsafe { (*bio).bi_vcnt }) as i32).wrapping_sub(1) as isize), minsize.wrapping_sub(1) as u32)
}

fn bio_iov_iter_bounce_read(bio: *mut bio, iter: *mut iov_iter, maxlen: u64, minsize: u64) -> i32 {
let mut ret: core::mem::MaybeUninit<i64> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i64 = core::ptr::addr_of_mut!(ret) as *mut i64;
let mut folio: core::mem::MaybeUninit<*mut folio> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut folio = core::ptr::addr_of_mut!(folio) as *mut *mut folio;
let mut __UNIQUE_ID_z__948: i32 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_y__947: u64 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__946: u64 = unsafe { core::mem::zeroed() };
let mut len: u64 = unsafe { core::mem::zeroed() };
unsafe {
__UNIQUE_ID_x__946 = iov_iter_count(iter as *const iov_iter);
__UNIQUE_ID_y__947 = maxlen;
__UNIQUE_ID_z__948 = 1048576;
loop {
if __types_ok3__4!(__UNIQUE_ID_x__946, __UNIQUE_ID_y__947, __UNIQUE_ID_z__948) == 0 {
__compiletime_assert_949();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
len = if __UNIQUE_ID_x__946 < if __UNIQUE_ID_y__947 < (__UNIQUE_ID_z__948 as u64) { __UNIQUE_ID_y__947 } else { __UNIQUE_ID_z__948 as u64 } { __UNIQUE_ID_x__946 } else { if __UNIQUE_ID_y__947 < (__UNIQUE_ID_z__948 as u64) { __UNIQUE_ID_y__947 } else { __UNIQUE_ID_z__948 as u64 } };
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_6 = folio_alloc_greedy(((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32), core::ptr::addr_of_mut!(len), minsize);
if (*__slate_slot_6).is_null() {
return 12i32.wrapping_neg();
} else {
'__join_0: {
'__join_4: {
loop {
*__slate_slot_5 = iov_iter_extract_bvecs(iter, unsafe { (*bio).bi_io_vec }.wrapping_offset(1 as isize), len, core::ptr::addr_of_mut!((*bio).bi_vcnt), ((*bio).bi_max_vecs as i32).wrapping_sub(1) as u16, 0u32, 0u32);
if *__slate_slot_5 <= 0 {
break;
} else {
len = len.wrapping_sub(*__slate_slot_5 as u64);
(*bio).bi_iter.bi_size = ((*bio).bi_iter.bi_size as i64).wrapping_add(*__slate_slot_5) as u32;
if !(len != 0 && ((*bio).bi_vcnt as i32) < ((*bio).bi_max_vecs as i32).wrapping_sub(1)) {
break '__join_4;
}
}
}
if (*bio).bi_vcnt == 0 {
break '__join_0;
}
}
// Set the folio directly here.  The above loop has already calculated
// the correct bi_size, and we use bi_vcnt for the user buffers.  That
// is safe as bi_vcnt is only used by the submitter and not the actual
// I/O path.
bvec_set_folio(unsafe { (*bio).bi_io_vec }.wrapping_offset(0 as isize), *__slate_slot_6, (*bio).bi_iter.bi_size as u64, 0u64);
if iov_iter_extract_will_pin(iter as *const iov_iter) {
bio_set_flag(bio, 0u32);
}
// The first vec stores the bounce buffer, so do not subtract 1 here.
*__slate_slot_5 = bio_iov_iter_align_down(bio, iter, unsafe { (*bio).bi_io_vec }.wrapping_offset(((*bio).bi_vcnt as i32) as isize), minsize.wrapping_sub(1) as u32) as i64;
if *__slate_slot_5 == 0 {
// Update the bounc buffer bv_len to the aligned down size.
(*unsafe { (*bio).bi_io_vec }.wrapping_offset(0 as isize)).bv_len = (*bio).bi_iter.bi_size;
return 0;
}
}
folio_put(*__slate_slot_6);
return *__slate_slot_5 as i32;
}
}
}

/// bio_iov_iter_bounce - bounce buffer data from an iter into a bio
/// @bio:	bio to send
/// @iter:	iter to read from / write into
/// @maxlen:	maximum size to bounce
/// @minsize:	minimum folio allocation size
///
/// Helper for direct I/O implementations that need to bounce buffer because
/// we need to checksum the data or perform other operations that require
/// consistency.  Allocates folios to back the bounce buffer, and for writes
/// copies the data into it.  Needs to be paired with bio_iov_iter_unbounce()
/// called on completion.
#[unsafe(no_mangle)]
extern "C" fn bio_iov_iter_bounce(bio: *mut bio, iter: *mut iov_iter, maxlen: u64, minsize: u64) -> i32 {
if op_is_write(bio_op(bio as *const bio) as u32) {
return bio_iov_iter_bounce_write(bio, iter, maxlen, minsize);
}
bio_iov_iter_bounce_read(bio, iter, maxlen, minsize)
}

fn bvec_unpin(bv: *mut bio_vec, mark_dirty: bool) {
let folio: *mut folio = bvec_folio(bv as *const bio_vec);
let nr_pages: u64 = ((unsafe { (*bv).bv_offset }.wrapping_add(unsafe { (*bv).bv_len }).wrapping_sub(1) as u64) / (_AC!() << 12i32)).wrapping_sub(((unsafe { (*bv).bv_offset }) as u64) / (_AC!() << 12i32)).wrapping_add(1);
if mark_dirty {
unsafe { folio_mark_dirty_lock(folio) };
}
unsafe { unpin_user_folio(folio, nr_pages) };
}

fn bio_iov_iter_unbounce_read(bio: *mut bio, is_error: bool, mut mark_dirty: bool) {
let len: u32 = unsafe { (*unsafe { (*bio).bi_io_vec }.wrapping_offset(0 as isize)).bv_len };
if likely!(!is_error) != 0 {
let buf: *mut () = bvec_virt(unsafe { (*bio).bi_io_vec }.wrapping_offset(0 as isize));
let mut to: iov_iter = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(to) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 40usize);
}
unsafe { iov_iter_bvec(core::ptr::addr_of_mut!(to), 0u32, unsafe { (*bio).bi_io_vec }.wrapping_offset(1 as isize) as *const bio_vec, (*bio).bi_vcnt as u64, len as u64) };
// copying to pinned pages should always work
let mut __v0: i64;
let __ret_warn_on: i32 = (copy_to_iter(buf as *const (), len as u64, core::ptr::addr_of_mut!(to)) != (len as u64)) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("950: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 950b - ., 3; .popsection\n# {0}", const 950 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_bio_iov_iter_unbounce_read_1, sym __slate_asm_string_bio_iov_iter_unbounce_read_2, const 1515 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("951: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 951b - ., 4; .popsection\n# {0}", const 951 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
} else {
// No need to mark folios dirty if never copied to them
mark_dirty = false;
}
if bio_flagged(bio as *const bio, 0u32) {
let mut i: i32 = 0;
while i < ((unsafe { (*bio).bi_vcnt }) as i32) {
bvec_unpin(unsafe { (*bio).bi_io_vec }.wrapping_offset(1i32.wrapping_add(i) as isize), mark_dirty);
i = i.wrapping_add(1);
}
}
folio_put(bvec_folio(unsafe { (*bio).bi_io_vec }.wrapping_offset(0 as isize) as *const bio_vec));
}

/// bio_iov_iter_unbounce - finish a bounce buffer operation
/// @bio:	completed bio
/// @is_error:	%true if an I/O error occurred and data should not be copied
/// @mark_dirty:	If %true, folios will be marked dirty.
///
/// Helper for direct I/O implementations that need to bounce buffer because
/// we need to checksum the data or perform other operations that require
/// consistency.  Called to complete a bio set up by bio_iov_iter_bounce().
/// Copies data back for reads, and marks the original folios dirty if
/// requested and then frees the bounce buffer.
#[unsafe(no_mangle)]
extern "C" fn bio_iov_iter_unbounce(bio: *mut bio, is_error: bool, mark_dirty: bool) {
if op_is_write(bio_op(bio as *const bio) as u32) {
bio_free_folios(bio);
} else {
bio_iov_iter_unbounce_read(bio, is_error, mark_dirty);
}
}

#[unsafe(link_section = ".text.slate_distinct.bio.bio_wait_end_io")]
extern "C" fn bio_wait_end_io(bio: *mut bio) {
unsafe { complete((*bio).bi_private as *mut completion) };
}

/// bio_await - call a function on a bio, and wait until it completes
/// @bio:	the bio which describes the I/O
/// @submit:	function called to submit the bio
/// @priv:	private data passed to @submit
///
/// Wait for the bio as well as any bio chained off it after executing the
/// passed in callback @submit.  The wait for the bio is set up before calling
/// @submit to ensure that the completion is captured.  If @submit is %NULL,
/// submit_bio() is used instead to submit the bio.
///
/// Note: this overrides the bi_private and bi_end_io fields in the bio.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.bio.bio_await")]
extern "C" fn bio_await(bio: *mut bio, r#priv: *mut (), submit: Option<unsafe extern "C" fn(*mut bio, *mut ())>) {
let mut __slate_compound_0: raw_spinlock = unsafe { core::mem::zeroed() };
let mut done: completion = unsafe { core::mem::zeroed() };
done = {
let mut __t0: completion = unsafe { core::mem::zeroed() };
__t0.done = 0u32;
__t0.wait = {
let mut __t1: swait_queue_head = unsafe { core::mem::zeroed() };
__t1.lock = unsafe { *{
__slate_compound_0 = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t2: __SlateRecord_qspinlock_types_h_44 = core::mem::zeroed();
__t2.val = __SlateRecord_types_h_190 { counter: 0 };
__t2
} } };
core::ptr::addr_of_mut!(__slate_compound_0)
} };
__t1.task_list = list_head { next: core::ptr::addr_of_mut!(done.wait.task_list), prev: core::ptr::addr_of_mut!(done.wait.task_list) };
__t1
};
__t0
};
unsafe {
(*bio).bi_private = core::ptr::addr_of_mut!(done) as *mut ();
}
unsafe {
(*bio).bi_end_io = Some(bio_wait_end_io);
}
unsafe {
(*bio).bi_opf = (*bio).bi_opf | 2048u32;
}
if submit != None {
unsafe { submit.unwrap()(bio, r#priv) };
} else {
unsafe { submit_bio(bio) };
}
blk_wait_io(core::ptr::addr_of_mut!(done));
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_bio_await_952: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut bio, *mut (), Option<unsafe extern "C" fn(*mut bio, *mut ())>)>, *mut ()>(Some(bio_await)) };

/// submit_bio_wait - submit a bio, and wait until it completes
/// @bio: The &struct bio which describes the I/O
///
/// Simple wrapper around submit_bio(). Returns 0 on success, or the error from
/// bio_endio() on failure.
///
/// WARNING: Unlike to how submit_bio() is usually used, this function does not
/// result in bio reference to be consumed. The caller must drop the reference
/// on his own.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.bio.submit_bio_wait")]
extern "C" fn submit_bio_wait(bio: *mut bio) -> i32 {
bio_await(bio, core::ptr::null_mut::<()>(), None);
unsafe { blk_status_to_errno((*bio).bi_status) }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_submit_bio_wait_953: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut bio) -> i32>, *mut ()>(Some(submit_bio_wait)) };

#[unsafe(link_section = ".text.slate_distinct.bio.bio_endio_cb")]
extern "C" fn bio_endio_cb(bio: *mut bio, r#priv: *mut ()) {
bio_endio(bio);
}

/// Submit @bio synchronously, or call bio_endio on it if the current process
/// is being killed.
#[unsafe(no_mangle)]
extern "C" fn bio_submit_or_kill(bio: *mut bio, flags: u32) -> i32 {
let __v0: bool = flags & 4u32 != 0 && fatal_signal_pending(get_current()) != 0;
if __v0 {
bio_await(bio, core::ptr::null_mut::<()>(), Some(bio_endio_cb));
return 4i32.wrapping_neg();
}
submit_bio_wait(bio)
}

/// bdev_rw_virt - synchronously read into / write from kernel mapping
/// @bdev:	block device to access
/// @sector:	sector to access
/// @data:	data to read/write
/// @len:	length in byte to read/write
/// @op:		operation (e.g. REQ_OP_READ/REQ_OP_WRITE)
///
/// Performs synchronous I/O to @bdev for @data/@len.  @data must be in
/// the kernel direct mapping and not a vmalloc address.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.bio.bdev_rw_virt")]
extern "C" fn bdev_rw_virt(bdev: *mut block_device, sector: u64, data: *mut (), len: u64, op: u32) -> i32 {
let mut bv: bio_vec = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(bv) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let mut bio: bio = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(bio) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 120usize);
}
let mut error: i32 = 0;
let mut __v0: i64;
let __ret_warn_on: i32 = !(!unsafe { is_vmalloc_addr(data as *const ()) }) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("954: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 954b - ., 3; .popsection\n# {0}", const 954 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_bdev_rw_virt_1, sym __slate_asm_string_bdev_rw_virt_2, const 1641 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("955: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 955b - ., 4; .popsection\n# {0}", const 955 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
return 5i32.wrapping_neg();
}
bio_init(core::ptr::addr_of_mut!(bio), bdev, core::ptr::addr_of_mut!(bv), 1u16, op as u32);
bio.bi_iter.bi_sector = sector;
bio_add_virt_nofail(core::ptr::addr_of_mut!(bio), data, len as u32);
error = submit_bio_wait(core::ptr::addr_of_mut!(bio));
bio_uninit(core::ptr::addr_of_mut!(bio));
error
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_bdev_rw_virt_956: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut block_device, u64, *mut (), u64, u32) -> i32>, *mut ()>(Some(bdev_rw_virt)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.bio.__bio_advance")]
extern "C" fn __bio_advance(bio: *mut bio, bytes: u32) {
if !bio_integrity(bio).is_null() {
bio_integrity_advance(bio, bytes);
}
bio_crypt_advance(bio, bytes);
bio_advance_iter(bio as *const bio, unsafe { core::ptr::addr_of_mut!((*bio).bi_iter) }, bytes);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___bio_advance_957: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut bio, u32)>, *mut ()>(Some(__bio_advance)) };

// bio_copy_data - copy contents of data buffers from one bio to another
// @src: source bio
// @dst: destination bio
//
// Stops when it reaches the end of either @src or @dst - that is, copies
// min(src->bi_size, dst->bi_size) bytes (or the equivalent for lists of bios).
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_bio_copy_data_963: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut bio, *mut bio)>, *mut ()>(Some(bio_copy_data)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.bio.bio_free_pages")]
extern "C" fn bio_free_pages(bio: *mut bio) {
let mut iter_all: bvec_iter_all = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(iter_all) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 24usize);
}
let bvec: *mut bio_vec = bvec_init_iter_all(unsafe { &mut (*core::ptr::addr_of_mut!(iter_all)) });
while bio_next_segment(bio as *const bio, core::ptr::addr_of_mut!(iter_all)) {
__free_page!(unsafe { (*bvec).bv_page });
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_bio_free_pages_964: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut bio)>, *mut ()>(Some(bio_free_pages)) };

// bio_set_pages_dirty() and bio_check_pages_dirty() are support functions
// for performing direct-IO in BIOs.
//
// The problem is that we cannot run folio_mark_dirty() from interrupt context
// because the required locks are not interrupt-safe.  So what we can do is to
// mark the pages dirty _before_ performing IO.  And in interrupt context,
// check that the pages are still dirty.   If so, fine.  If not, redirty them
// in process context.
//
// Note that this code is very hard to test under normal circumstances because
// direct-io pins the pages with get_user_pages().  This makes
// is_page_cache_freeable return false, and the VM will not clean the pages.
// But other code (eg, flusher threads) could clean the pages if they are mapped
// pagecache.
//
// Simply disabling the call to bio_set_pages_dirty() is a good way to test the
// deferred bio dirtying paths.
/// bio_set_pages_dirty() will mark all the bio's pages as dirty.
#[unsafe(no_mangle)]
extern "C" fn bio_set_pages_dirty(bio: *mut bio) {
let mut fi: folio_iter = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(fi) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 48usize);
}
bio_first_folio(unsafe { &mut (*core::ptr::addr_of_mut!(fi)) }, bio, 0i32);
while !fi.folio.is_null() {
folio_lock(fi.folio);
unsafe { folio_mark_dirty(fi.folio) };
unsafe { folio_unlock(fi.folio) };
bio_next_folio(unsafe { &mut (*core::ptr::addr_of_mut!(fi)) }, bio);
}
}

// bio_check_pages_dirty() will check that all the BIO's pages are still dirty.
// If they are, then fine.  If, however, some pages are clean then they must
// have been written out during the direct-IO read.  So we take another ref on
// the BIO and re-dirty the pages in process context.
//
// It is expected that bio_check_pages_dirty() will wholly own the BIO from
// here on.  It will unpin each page and will run one bio_put() against the
// BIO.
static mut bio_dirty_work: work_struct = work_struct { data: __SlateRecord_types_h_197 { counter: 4503599625273344i64 }, entry: list_head { next: unsafe { core::ptr::addr_of_mut!(bio_dirty_work.entry) }, prev: unsafe { core::ptr::addr_of_mut!(bio_dirty_work.entry) } }, func: Some(bio_dirty_fn) };

static mut bio_dirty_lock: spinlock = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = unsafe { core::mem::zeroed() };
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0
} };

static mut bio_dirty_list: *mut bio = unsafe { core::mem::zeroed() };

/// This runs in process context
#[unsafe(link_section = ".text.slate_distinct.bio.bio_dirty_fn")]
extern "C" fn bio_dirty_fn(work: *mut work_struct) {
let mut next: *mut bio = unsafe { core::mem::zeroed() };
spin_lock_irq(unsafe { core::ptr::addr_of_mut!(bio_dirty_lock) });
next = unsafe { bio_dirty_list };
unsafe {
bio_dirty_list = core::ptr::null_mut::<bio>();
}
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!(bio_dirty_lock) });
loop {
let __v0: *mut bio = next;
let bio: *mut bio = __v0;
if __v0.is_null() {
break;
}
next = (unsafe { (*bio).bi_private }) as *mut bio;
bio_release_pages(bio, true);
bio_put(bio);
}
}

#[unsafe(no_mangle)]
extern "C" fn bio_check_pages_dirty(bio: *mut bio) {
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
let mut fi: core::mem::MaybeUninit<folio_iter> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut folio_iter = core::ptr::addr_of_mut!(fi) as *mut folio_iter;
unsafe {
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 48usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
bio_first_folio(&mut (*core::ptr::addr_of_mut!(*__slate_slot_1)), bio, 0i32);
'__join_0: {
while !(*__slate_slot_1).folio.is_null() {
if !folio_test_dirty((*__slate_slot_1).folio as *const folio) {
break '__join_0;
} else {
bio_next_folio(&mut (*core::ptr::addr_of_mut!(*__slate_slot_1)), bio);
}
}
bio_release_pages(bio, false);
bio_put(bio);
return;
}
spin_lock_irqsave!(core::ptr::addr_of_mut!(bio_dirty_lock), *__slate_slot_0);
(*bio).bi_private = bio_dirty_list as *mut ();
bio_dirty_list = bio;
spin_unlock_irqrestore(core::ptr::addr_of_mut!(bio_dirty_lock), *__slate_slot_0);
schedule_work(core::ptr::addr_of_mut!(bio_dirty_work));
}
}

// Infrastructure for deferring bio completions to task-context via a per-CPU
// workqueue. Triggered either by the BIO_COMPLETE_IN_TASK bio flag (static
// decision at submit time) or by calling bio_complete_in_task() from
// bi_end_io() (dynamic decision at completion time).
#[repr(C)]
#[derive(Clone, Copy)]
struct bio_complete_batch {
    list: bio_list,
    work: work_struct,
    cpu: i32,
}


#[unsafe(link_section = ".data..percpu")]
static mut bio_complete_batch: bio_complete_batch = unsafe { core::mem::zeroed() };

static mut bio_complete_wq: *mut workqueue_struct = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".text.slate_distinct.bio.bio_complete_work_fn")]
extern "C" fn bio_complete_work_fn(w: *mut work_struct) {
let batch: *mut bio_complete_batch = container_of__13!(w, bio_complete_batch);
while true {
let mut list: bio_list = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(list) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
local_irq_disable!();
list = unsafe { (*batch).list };
bio_list_init(unsafe { core::ptr::addr_of_mut!((*batch).list) });
local_irq_enable!();
if bio_list_empty(core::ptr::addr_of_mut!(list) as *const bio_list) != 0 {
break;
}
loop {
let __v4: *mut bio = bio_list_pop(core::ptr::addr_of_mut!(list));
let bio: *mut bio = __v4;
if __v4.is_null() {
break;
}
unsafe { unsafe { (*bio).bi_end_io }.unwrap()(bio) };
}
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.bio.__bio_complete_in_task")]
extern "C" fn __bio_complete_in_task(bio: *mut bio) {
let mut batch: *mut bio_complete_batch = unsafe { core::mem::zeroed() };
let mut flags: u64 = 0;
let mut was_empty: bool = false;
local_irq_save!(flags);
let mut __v3: *mut bio_complete_batch;
__verify_pcpu_ptr__26!();
let mut __v5: *mut bio_complete_batch;
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
let __v8: u64 = tcp_ptr__.wrapping_add((unsafe { core::ptr::addr_of_mut!(bio_complete_batch) }) as u64);
tcp_ptr__ = __v8;
__v5 = tcp_ptr__ as *mut bio_complete_batch;
__v3 = __v5;
batch = __v3;
was_empty = bio_list_empty((unsafe { core::ptr::addr_of_mut!((*batch).list) }) as *const bio_list) != 0;
bio_list_add(unsafe { core::ptr::addr_of_mut!((*batch).list) }, bio);
local_irq_restore!(flags);
if was_empty {
unsafe { queue_work_on((*batch).cpu, bio_complete_wq, core::ptr::addr_of_mut!((*batch).work)) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___bio_complete_in_task_965: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut bio)>, *mut ()>(Some(__bio_complete_in_task)) };

fn bio_remaining_done(bio: *mut bio) -> bool {
// If we're not chaining, then ->__bi_remaining is always 1 and
// we always end io on the first invocation.
if !bio_flagged(bio as *const bio, 3u32) {
return true;
}
loop {
if unlikely__2!(atomic_read((unsafe { core::ptr::addr_of_mut!((*bio).__bi_remaining) }) as *const __SlateRecord_types_h_190) <= 0) != 0 {
loop {
unsafe {
core::arch::asm!("966: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 966b - ., 3; .popsection\n# {0}", const 966 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_bio_remaining_done_0, sym __slate_asm_string_bio_remaining_done_1, const 1861 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
if atomic_dec_and_test(unsafe { core::ptr::addr_of_mut!((*bio).__bi_remaining) }) {
bio_clear_flag(bio, 3u32);
return true;
}
false
}

/// bio_endio - end I/O on a bio
/// @bio:	bio
///
/// Description:
///   bio_endio() will end I/O on the whole bio. bio_endio() is the preferred
///   way to end I/O on a bio. No one should call bi_end_io() directly on a
///   bio unless they own it and thus know that it has an end_io function.
///
///   bio_endio() can be called several times on a bio that has been chained
///   using bio_chain().  The ->bi_end_io() function will only be called the
///   last time.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.bio.bio_endio")]
extern "C" fn bio_endio(mut bio: *mut bio) {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut __v1: bool = unsafe { core::mem::zeroed() };
unsafe {
loop {
if !bio_remaining_done(bio) {
return;
} else {
if !bio_integrity_endio(bio) {
return;
} else {
blk_zone_bio_endio(bio);
rq_qos_done_bio(bio);
if !unsafe { (*bio).bi_bdev }.is_null() {
__v1 = bio_flagged(bio as *const bio, 6u32);
} else {
__v1 = false;
}
if __v1 {
trace_block_bio_complete(bdev_get_queue((*bio).bi_bdev), bio);
bio_clear_flag(bio, 6u32);
}
// Need to have a real endio function for chained bios, otherwise
// various corner cases will break (like stacking block devices that
// save/restore bi_end_io) - however, we want to avoid unbounded
// recursion and blowing the stack. Tail call optimization would
// handle this, but compiling with frame pointers also disables
// gcc's sibling call optimization.
if (*bio).bi_end_io == Some(bio_chain_endio) {
bio = __bio_chain_endio(bio);
} else {
break;
}
}
}
}
// Release cgroup info.  We shouldn't have to do this here, but quite
// a few callers of bio_init fail to call bio_uninit, so we cover up
// for that here at least for now.
if !unsafe { (*bio).bi_blkg }.is_null() {
blkg_put((*bio).bi_blkg);
(*bio).bi_blkg = core::ptr::null_mut::<blkcg_gq>();
}
if bio_flagged(bio as *const bio, 13u32) {
__v0 = bio_in_atomic();
} else {
__v0 = false;
}
if __v0 {
__bio_complete_in_task(bio);
} else {
if (*bio).bi_end_io != None {
unsafe { (*bio).bi_end_io }.unwrap()(bio);
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_bio_endio_967: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut bio)>, *mut ()>(Some(bio_endio)) };

/// bio_split - split a bio
/// @bio:	bio to split
/// @sectors:	number of sectors to split from the front of @bio
/// @gfp:	gfp mask
/// @bs:		bio set to allocate from
///
/// Allocates and returns a new bio which represents @sectors from the start of
/// @bio, and updates @bio to represent the remaining sectors.
///
/// Unless this is a discard request the newly allocated bio will point
/// to @bio's bi_io_vec. It is the caller's responsibility to ensure that
/// neither @bio nor @bs are freed before the split bio.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.bio.bio_split")]
extern "C" fn bio_split(bio: *mut bio, sectors: i32, gfp: u32, bs: *mut bio_set) -> *mut bio {
let mut __v0: i64;
{
let __ret_warn_on: i32 = (sectors <= 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("968: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 968b - ., 3; .popsection\n# {0}", const 968 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_bio_split_1, sym __slate_asm_string_bio_split_2, const 1952 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("969: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 969b - ., 4; .popsection\n# {0}", const 969 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
}
if __v0 != 0 {
return ERR_PTR(22i32.wrapping_neg() as i64) as *mut bio;
}
let mut __v5: i64;
{
let __ret_warn_on: i32 = ((sectors as u32) >= bio_sectors!(bio)) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("970: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 970b - ., 3; .popsection\n# {0}", const 970 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_bio_split_6, sym __slate_asm_string_bio_split_7, const 1954 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("971: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 971b - ., 4; .popsection\n# {0}", const 971 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v5 = (__ret_warn_on != 0) as i64;
}
if __v5 != 0 {
return ERR_PTR(22i32.wrapping_neg() as i64) as *mut bio;
}
// Zone append commands cannot be split
let mut __v10: i64;
let __ret_warn_on: i32 = ((bio_op(bio as *const bio) as u32) == 7) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("972: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 972b - ., 3; .popsection\n# {0}", const 972 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_bio_split_11, sym __slate_asm_string_bio_split_12, const 1958 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("973: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 973b - ., 4; .popsection\n# {0}", const 973 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v10 = (__ret_warn_on != 0) as i64;
if __v10 != 0 {
return ERR_PTR(22i32.wrapping_neg() as i64) as *mut bio;
}
// atomic writes cannot be split
if (unsafe { (*bio).bi_opf }) & 134217728u32 != 0 {
return ERR_PTR(22i32.wrapping_neg() as i64) as *mut bio;
}
let split: *mut bio = bio_alloc_clone(unsafe { (*bio).bi_bdev }, bio, gfp, bs);
if split.is_null() {
return ERR_PTR(12i32.wrapping_neg() as i64) as *mut bio;
}
unsafe {
(*split).bi_iter.bi_size = (sectors << 9i32) as u32;
}
if !bio_integrity(split).is_null() {
bio_integrity_trim(split);
}
bio_advance(bio, unsafe { (*split).bi_iter.bi_size });
// The gap bit is set when splitting to limits and only applies to the
// front bio that was split off. The remaining bio will calcualte its
// gap value when it is subsequently split to limits, so it is safe to
// re-initialize the value back to 0.
unsafe {
(*bio).bi_bvec_gap_bit = 0;
}
if bio_flagged(bio as *const bio, 6u32) {
bio_set_flag(split, 6u32);
}
split
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_bio_split_974: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut bio, i32, u32, *mut bio_set) -> *mut bio>, *mut ()>(Some(bio_split)) };

/// bio_trim - trim a bio
/// @bio:	bio to trim
/// @offset:	number of sectors to trim from the front of @bio
/// @size:	size we want to trim @bio to, in sectors
///
/// This function is typically used for bios that are cloned and submitted
/// to the underlying device in parts.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.bio.bio_trim")]
extern "C" fn bio_trim(bio: *mut bio, offset: u64, mut size: u64) {
// We should never trim an atomic write
let mut __v0: i64;
{
let __ret_warn_on: i32 = ((unsafe { (*bio).bi_opf }) & 134217728u32 != 0 && size != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("975: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 975b - ., 3; .popsection\n# {0}", const 975 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_bio_trim_1, sym __slate_asm_string_bio_trim_2, const 2003 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("976: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 976b - ., 4; .popsection\n# {0}", const 976 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
}
if __v0 != 0 {
return;
}
let mut __v5: i64;
let __ret_warn_on: i32 = (offset > 8388607u64 || size > 8388607u64 || offset.wrapping_add(size) > (bio_sectors!(bio) as u64)) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("977: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 977b - ., 3; .popsection\n# {0}", const 977 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_bio_trim_6, sym __slate_asm_string_bio_trim_7, const 2007 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("978: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 978b - ., 4; .popsection\n# {0}", const 978 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v5 = (__ret_warn_on != 0) as i64;
if __v5 != 0 {
return;
}
size <<= 9i32;
if offset == 0 && size == ((unsafe { (*bio).bi_iter.bi_size }) as u64) {
return;
}
bio_advance(bio, (offset << 9i32) as u32);
unsafe {
(*bio).bi_iter.bi_size = size as u32;
}
if !bio_integrity(bio).is_null() {
bio_integrity_trim(bio);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_bio_trim_979: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut bio, u64, u64)>, *mut ()>(Some(bio_trim)) };

/// create memory pools for biovec's in a bio_set.
/// use the global biovec slabs created for general use.
fn biovec_init_pool(pool: *mut mempool, pool_entries: i32) -> i32 {
let bp: *mut biovec_slab = unsafe { core::ptr::addr_of_mut!(bvec_slabs.0) as *mut biovec_slab }.wrapping_offset(ARRAY_SIZE__33!() as isize).wrapping_offset(-(1 as isize));
mempool_init_slab_pool__2!(pool, pool_entries, unsafe { (*bp).slab })
}

/// bioset_exit - exit a bioset initialized with bioset_init()
///
/// May be called on a zeroed but uninitialized bioset (i.e. allocated with
/// kzalloc()).
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.bio.bioset_exit")]
extern "C" fn bioset_exit(bs: *mut bio_set) {
bio_alloc_cache_destroy(bs);
if !unsafe { (*bs).rescue_workqueue }.is_null() {
unsafe { destroy_workqueue((*bs).rescue_workqueue) };
}
unsafe {
(*bs).rescue_workqueue = core::ptr::null_mut::<workqueue_struct>();
}
unsafe { mempool_exit(core::ptr::addr_of_mut!((*bs).bio_pool)) };
unsafe { mempool_exit(core::ptr::addr_of_mut!((*bs).bvec_pool)) };
if !unsafe { (*bs).bio_slab }.is_null() {
bio_put_slab(bs);
}
unsafe {
(*bs).bio_slab = core::ptr::null_mut::<kmem_cache>();
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_bioset_exit_980: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut bio_set)>, *mut ()>(Some(bioset_exit)) };

/// bioset_init - Initialize a bio_set
/// @bs:		pool to initialize
/// @pool_size:	Number of bio and bio_vecs to cache in the mempool
/// @front_pad:	Number of bytes to allocate in front of the returned bio
/// @flags:	Flags to modify behavior, currently %BIOSET_NEED_BVECS
///              and %BIOSET_NEED_RESCUER
///
/// Description:
///    Set up a bio_set to be used with @bio_alloc_bioset. Allows the caller
///    to ask for a number of bytes to be allocated in front of the bio.
///    Front pad allocation is useful for embedding the bio inside
///    another structure, to avoid allocating extra data to go with the bio.
///    Note that the bio must be embedded at the END of that structure always,
///    or things will break badly.
///    If %BIOSET_NEED_BVECS is set in @flags, a separate pool will be allocated
///    for allocating iovecs.  This pool is not needed e.g. for bio_init_clone().
///    If %BIOSET_NEED_RESCUER is set, a workqueue is created which can be used
///    to dispatch queued requests when the mempool runs out of space.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.bio.bioset_init")]
extern "C" fn bioset_init(bs: *mut bio_set, pool_size: u32, front_pad: u32, flags: i32) -> i32 {
let mut __slate_compound_16: __SlateRecord_types_h_197 = unsafe { core::mem::zeroed() };
let mut __slate_compound_17: spinlock = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut workqueue_struct> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut workqueue_struct = core::ptr::addr_of_mut!(_res) as *mut *mut workqueue_struct;
let mut __v4: bool = unsafe { core::mem::zeroed() };
unsafe {
(*bs).front_pad = front_pad;
if flags & 1 != 0 {
(*bs).back_pad = 4u64.wrapping_mul(core::mem::size_of::<bio_vec>() as u64) as u32;
} else {
(*bs).back_pad = 0;
}
loop {
spinlock_check(core::ptr::addr_of_mut!((*bs).rescue_lock));
*core::ptr::addr_of_mut!((*bs).rescue_lock) = *{
__slate_compound_17 = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = core::mem::zeroed();
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = core::mem::zeroed();
__t1.val = __SlateRecord_types_h_190 { counter: 0 };
__t1
} } };
__t0
} };
core::ptr::addr_of_mut!(__slate_compound_17)
};
if true {
break;
}
}
bio_list_init(core::ptr::addr_of_mut!((*bs).rescue_list));
loop {
__init_work(core::ptr::addr_of_mut!((*bs).rescue_work), 0i32);
(*core::ptr::addr_of_mut!((*bs).rescue_work)).data = *{
__slate_compound_16 = __SlateRecord_types_h_197 { counter: 4503599625273344i64 };
core::ptr::addr_of_mut!(__slate_compound_16)
};
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*bs).rescue_work)).entry));
(*core::ptr::addr_of_mut!((*bs).rescue_work)).func = Some(bio_alloc_rescue);
if true {
if true {
break;
}
}
}
(*bs).bio_slab = bio_find_or_create_slab(bs);
if unsafe { (*bs).bio_slab }.is_null() {
return 12i32.wrapping_neg();
} else {
'__join_0: {
if mempool_init_slab_pool!(core::ptr::addr_of_mut!((*bs).bio_pool), pool_size, unsafe { (*bs).bio_slab }) == 0 {
if flags & 1 != 0 {
__v4 = biovec_init_pool(core::ptr::addr_of_mut!((*bs).bvec_pool), pool_size as i32) != 0;
} else {
__v4 = false;
}
if !__v4 {
if flags & 2 != 0 {
let __slate_auto_bytes_1 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_0 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_1 = alloc_workqueue_noprof(c"bioset".as_ptr() as *const u8, 264u32, 0i32);
alloc_tag_restore!();
} else {
*__slate_slot_1 = alloc_workqueue_noprof(c"bioset".as_ptr() as *const u8, 264u32, 0i32);
}
(*bs).rescue_workqueue = *__slate_slot_1;
if unsafe { (*bs).rescue_workqueue }.is_null() {
break '__join_0;
}
}
if flags & 4 != 0 {
(*bs).cache = alloc_percpu!(bio_alloc_cache);
if unsafe { (*bs).cache }.is_null() {
break '__join_0;
} else {
cpuhp_state_add_instance_nocalls(21i32, core::ptr::addr_of_mut!((*bs).cpuhp_dead));
}
}
return 0;
}
}
}
bioset_exit(bs);
return 12i32.wrapping_neg();
}
}
}

static mut __key__2089: lock_class_key = unsafe { core::mem::zeroed() };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_bioset_init_981: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut bio_set, u32, u32, i32) -> i32>, *mut ()>(Some(bioset_init)) };

#[unsafe(link_section = ".text.slate_distinct.bio.bio_complete_batch_cpu_online")]
extern "C" fn bio_complete_batch_cpu_online(cpu: u32) -> i32 {
let batch: *mut bio_complete_batch = per_cpu_ptr__4!(unsafe { core::ptr::addr_of_mut!(bio_complete_batch) }, cpu);
unsafe { enable_work(core::ptr::addr_of_mut!((*batch).work)) };
if bio_list_empty((unsafe { core::ptr::addr_of_mut!((*batch).list) }) as *const bio_list) == 0 {
unsafe { queue_work_on(cpu as i32, bio_complete_wq, core::ptr::addr_of_mut!((*batch).work)) };
}
0
}

/// Disable this CPU's work item so that it cannot run on an unbound worker
/// after the CPU is offlined.
#[unsafe(link_section = ".text.slate_distinct.bio.bio_complete_batch_cpu_down_prep")]
extern "C" fn bio_complete_batch_cpu_down_prep(cpu: u32) -> i32 {
unsafe { disable_work_sync(core::ptr::addr_of_mut!((*per_cpu_ptr__4!(core::ptr::addr_of_mut!(bio_complete_batch), cpu)).work)) };
0
}

/// Drain a dead CPU's deferred bio completions. The CPU is dead and the worker
/// is canceled so no locking is needed.
#[unsafe(link_section = ".text.slate_distinct.bio.bio_complete_batch_cpu_dead")]
extern "C" fn bio_complete_batch_cpu_dead(cpu: u32) -> i32 {
let batch: *mut bio_complete_batch = per_cpu_ptr__4!(unsafe { core::ptr::addr_of_mut!(bio_complete_batch) }, cpu);
loop {
let __v3: *mut bio = bio_list_pop(unsafe { core::ptr::addr_of_mut!((*batch).list) });
let bio: *mut bio = __v3;
if __v3.is_null() {
break;
}
unsafe { unsafe { (*bio).bi_end_io }.unwrap()(bio) };
}
0
}

#[unsafe(link_section = ".init.text")]
fn bio_complete_batch_init(cpu: i32) {
let mut __slate_compound_3: __SlateRecord_types_h_197 = unsafe { core::mem::zeroed() };
let batch: *mut bio_complete_batch = per_cpu_ptr__4!(unsafe { core::ptr::addr_of_mut!(bio_complete_batch) }, cpu);
bio_list_init(unsafe { core::ptr::addr_of_mut!((*batch).list) });
'__slate_break_5: loop {
'__slate_break_4: loop {
__init_work(unsafe { core::ptr::addr_of_mut!((*batch).work) }, 0i32);
unsafe {
(*core::ptr::addr_of_mut!((*batch).work)).data = *{
__slate_compound_3 = __SlateRecord_types_h_197 { counter: 4503599625273344i64 };
core::ptr::addr_of_mut!(__slate_compound_3)
};
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*batch).work)).entry) });
unsafe {
(*core::ptr::addr_of_mut!((*batch).work)).func = Some(bio_complete_work_fn);
}
if true {
break;
}
}
if true {
break;
}
}
unsafe {
(*batch).cpu = cpu;
}
if !cpu_online(cpu as u32) {
unsafe { disable_work_sync(core::ptr::addr_of_mut!((*batch).work)) };
}
}

static mut __key__2164: lock_class_key = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".init.text")]
extern "C" fn init_bio() -> i32 {
let mut i: i32 = 0;
loop {
if 14 > 8u64.wrapping_mul(sizeof_field__2!()) {
unsafe { __compiletime_assert_982() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
while (i as u64) < ARRAY_SIZE__14!() {
let bvs: *mut biovec_slab = unsafe { core::ptr::addr_of_mut!(bvec_slabs.0) as *mut biovec_slab }.wrapping_offset(i as isize);
unsafe {
(*bvs).slab = __kmem_cache_create((*bvs).name as *const u8, ((*bvs).nr_vecs as u64).wrapping_mul(core::mem::size_of::<bio_vec>() as u64) as u32, 0u32, __SLAB_FLAG_BIT!(4i32) | __SLAB_FLAG_BIT!(8i32), None);
}
i = i.wrapping_add(1);
}
i = 0;
loop {
i = find_next_bit(cpumask_bits!((unsafe { core::ptr::addr_of_mut!(__cpu_possible_mask) }) as *const cpumask), 64u64, i as u64) as i32;
if (i as u32) >= 64 {
break;
}
bio_complete_batch_init(i);
i = i.wrapping_add(1);
}
let mut __v7: *mut workqueue_struct;
let mut __v8: *mut workqueue_struct;
let mut _res: *mut workqueue_struct = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = unsafe { alloc_workqueue_noprof(c"bio_complete".as_ptr() as *const u8, 264u32, 0i32) };
alloc_tag_restore!();
} else {
_res = unsafe { alloc_workqueue_noprof(c"bio_complete".as_ptr() as *const u8, 264u32, 0i32) };
}
__v8 = _res;
__v7 = __v8;
unsafe {
bio_complete_wq = __v7;
}
if unsafe { bio_complete_wq }.is_null() {
unsafe { panic(c"bio: can't allocate bio_complete workqueue\n".as_ptr() as *const u8) };
unsafe { core::hint::unreachable_unchecked() };
}
// bio task-context completion draining on hot-unplugged CPUs:
//
//   1. Stop the per-CPU work item while the CPU is still online, so
//      that it cannot run on an unbound worker later.
//   2. Drain leftover bios added between worker disabling and CPU
//      offlining.
cpuhp_setup_state_nocalls(191i32, c"block/bio:complete:online".as_ptr() as *const u8, Some(bio_complete_batch_cpu_online), Some(bio_complete_batch_cpu_down_prep));
cpuhp_setup_state_nocalls(61i32, c"block/bio:complete:dead".as_ptr() as *const u8, None, Some(bio_complete_batch_cpu_dead));
cpuhp_setup_state_multi(21i32, c"block/bio:dead".as_ptr() as *const u8, None, Some(bio_cpu_dead));
if bioset_init(unsafe { core::ptr::addr_of_mut!(fs_bio_set) }, 2u32, 0u32, 5i32) != 0 {
unsafe { panic(c"bio: can't allocate bios\n".as_ptr() as *const u8) };
unsafe { core::hint::unreachable_unchecked() };
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_init_bio_984: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i32>, *mut ()>(Some(init_bio)) };

static __slate_asm_string___bio_add_page_1: [u8; 1] = *b"\0";

static __slate_asm_string___bio_add_page_2: [u8; 41] = *b"/home/takashi/c-corpus/linux/block/bio.c\0";

static __slate_asm_string___bio_add_page_6: [u8; 1] = *b"\0";

static __slate_asm_string___bio_add_page_7: [u8; 41] = *b"/home/takashi/c-corpus/linux/block/bio.c\0";

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string___kmem_cache_default_args_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmem_cache_default_args_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_bdev_rw_virt_1: [u8; 1] = *b"\0";

static __slate_asm_string_bdev_rw_virt_2: [u8; 41] = *b"/home/takashi/c-corpus/linux/block/bio.c\0";

static __slate_asm_string_bio_add_folio_nofail_1: [u8; 1] = *b"\0";

static __slate_asm_string_bio_add_folio_nofail_2: [u8; 41] = *b"/home/takashi/c-corpus/linux/block/bio.c\0";

static __slate_asm_string_bio_add_page_1: [u8; 1] = *b"\0";

static __slate_asm_string_bio_add_page_2: [u8; 41] = *b"/home/takashi/c-corpus/linux/block/bio.c\0";

static __slate_asm_string_bio_add_page_6: [u8; 1] = *b"\0";

static __slate_asm_string_bio_add_page_7: [u8; 41] = *b"/home/takashi/c-corpus/linux/block/bio.c\0";

static __slate_asm_string_bio_alloc_bioset_1: [u8; 1] = *b"\0";

static __slate_asm_string_bio_alloc_bioset_2: [u8; 41] = *b"/home/takashi/c-corpus/linux/block/bio.c\0";

static __slate_asm_string_bio_alloc_irq_cache_splice_1: [u8; 1] = *b"\0";

static __slate_asm_string_bio_alloc_irq_cache_splice_2: [u8; 41] = *b"/home/takashi/c-corpus/linux/block/bio.c\0";

static __slate_asm_string_bio_chain_0: [u8; 1] = *b"\0";

static __slate_asm_string_bio_chain_1: [u8; 41] = *b"/home/takashi/c-corpus/linux/block/bio.c\0";

static __slate_asm_string_bio_chain_endio_0: [u8; 1] = *b"\0";

static __slate_asm_string_bio_chain_endio_1: [u8; 41] = *b"/home/takashi/c-corpus/linux/block/bio.c\0";

static __slate_asm_string_bio_first_bvec_all_1: [u8; 1] = *b"\0";

static __slate_asm_string_bio_first_bvec_all_2: [u8; 49] = *b"/home/takashi/c-corpus/linux/include/linux/bio.h\0";

static __slate_asm_string_bio_free_1: [u8; 1] = *b"\0";

static __slate_asm_string_bio_free_2: [u8; 41] = *b"/home/takashi/c-corpus/linux/block/bio.c\0";

static __slate_asm_string_bio_free_6: [u8; 1] = *b"\0";

static __slate_asm_string_bio_free_7: [u8; 41] = *b"/home/takashi/c-corpus/linux/block/bio.c\0";

static __slate_asm_string_bio_iov_iter_bounce_write_13: [u8; 1] = *b"\0";

static __slate_asm_string_bio_iov_iter_bounce_write_14: [u8; 41] = *b"/home/takashi/c-corpus/linux/block/bio.c\0";

static __slate_asm_string_bio_iov_iter_bounce_write_3: [u8; 1] = *b"\0";

static __slate_asm_string_bio_iov_iter_bounce_write_4: [u8; 41] = *b"/home/takashi/c-corpus/linux/block/bio.c\0";

static __slate_asm_string_bio_iov_iter_bounce_write_8: [u8; 1] = *b"\0";

static __slate_asm_string_bio_iov_iter_bounce_write_9: [u8; 41] = *b"/home/takashi/c-corpus/linux/block/bio.c\0";

static __slate_asm_string_bio_iov_iter_get_pages_1: [u8; 1] = *b"\0";

static __slate_asm_string_bio_iov_iter_get_pages_2: [u8; 41] = *b"/home/takashi/c-corpus/linux/block/bio.c\0";

static __slate_asm_string_bio_iov_iter_set_1: [u8; 1] = *b"\0";

static __slate_asm_string_bio_iov_iter_set_2: [u8; 41] = *b"/home/takashi/c-corpus/linux/block/bio.c\0";

static __slate_asm_string_bio_iov_iter_unbounce_read_1: [u8; 1] = *b"\0";

static __slate_asm_string_bio_iov_iter_unbounce_read_2: [u8; 41] = *b"/home/takashi/c-corpus/linux/block/bio.c\0";

static __slate_asm_string_bio_put_0: [u8; 1] = *b"\0";

static __slate_asm_string_bio_put_1: [u8; 41] = *b"/home/takashi/c-corpus/linux/block/bio.c\0";

static __slate_asm_string_bio_put_slab_17: [u8; 1] = *b"\0";

static __slate_asm_string_bio_put_slab_18: [u8; 41] = *b"/home/takashi/c-corpus/linux/block/bio.c\0";

static __slate_asm_string_bio_put_slab_19: [u8; 1] = *b"\0";

static __slate_asm_string_bio_put_slab_20: [u8; 41] = *b"/home/takashi/c-corpus/linux/block/bio.c\0";

static __slate_asm_string_bio_put_slab_21: [u8; 29] = *b"\x013bio: unable to find slab!\n\0";

static __slate_asm_string_bio_put_slab_22: [u8; 41] = *b"/home/takashi/c-corpus/linux/block/bio.c\0";

static __slate_asm_string_bio_remaining_done_0: [u8; 1] = *b"\0";

static __slate_asm_string_bio_remaining_done_1: [u8; 41] = *b"/home/takashi/c-corpus/linux/block/bio.c\0";

static __slate_asm_string_bio_reuse_1: [u8; 1] = *b"\0";

static __slate_asm_string_bio_reuse_11: [u8; 1] = *b"\0";

static __slate_asm_string_bio_reuse_12: [u8; 41] = *b"/home/takashi/c-corpus/linux/block/bio.c\0";

static __slate_asm_string_bio_reuse_2: [u8; 41] = *b"/home/takashi/c-corpus/linux/block/bio.c\0";

static __slate_asm_string_bio_reuse_6: [u8; 1] = *b"\0";

static __slate_asm_string_bio_reuse_7: [u8; 41] = *b"/home/takashi/c-corpus/linux/block/bio.c\0";

static __slate_asm_string_bio_split_1: [u8; 1] = *b"\0";

static __slate_asm_string_bio_split_11: [u8; 1] = *b"\0";

static __slate_asm_string_bio_split_12: [u8; 41] = *b"/home/takashi/c-corpus/linux/block/bio.c\0";

static __slate_asm_string_bio_split_2: [u8; 41] = *b"/home/takashi/c-corpus/linux/block/bio.c\0";

static __slate_asm_string_bio_split_6: [u8; 1] = *b"\0";

static __slate_asm_string_bio_split_7: [u8; 41] = *b"/home/takashi/c-corpus/linux/block/bio.c\0";

static __slate_asm_string_bio_trim_1: [u8; 1] = *b"\0";

static __slate_asm_string_bio_trim_2: [u8; 41] = *b"/home/takashi/c-corpus/linux/block/bio.c\0";

static __slate_asm_string_bio_trim_6: [u8; 1] = *b"\0";

static __slate_asm_string_bio_trim_7: [u8; 41] = *b"/home/takashi/c-corpus/linux/block/bio.c\0";

static __slate_asm_string_biovec_slab_0: [u8; 1] = *b"\0";

static __slate_asm_string_biovec_slab_1: [u8; 41] = *b"/home/takashi/c-corpus/linux/block/bio.c\0";

static __slate_asm_string_bvec_iter_advance_3: [u8; 44] = *b"Attempted to advance past end of bvec iter\n\0";

static __slate_asm_string_bvec_iter_advance_4: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/bvec.h\0";

static __slate_asm_string_bvec_set_folio_1: [u8; 1] = *b"\0";

static __slate_asm_string_bvec_set_folio_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/bvec.h\0";

static __slate_asm_string_bvec_virt_1: [u8; 1] = *b"\0";

static __slate_asm_string_bvec_virt_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/bvec.h\0";

static __slate_asm_string_check_copy_size_1: [u8; 1] = *b"\0";

static __slate_asm_string_check_copy_size_2: [u8; 55] = *b"/home/takashi/c-corpus/linux/include/linux/ucopysize.h\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

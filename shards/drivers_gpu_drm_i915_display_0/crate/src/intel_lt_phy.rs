
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
//! SPDX-License-Identifier: MIT
//!
//! Copyright © 2025 Intel Corporation
//! Optimization barrier
//!
//! The "volatile" is due to gcc bugs
//! small_const_nbits(n) is true precisely when it is known at compile-time
//! that BITMAP_SIZE(n) is 1, i.e. 1 <= n <= BITS_PER_LONG. This allows
//! various bit/bitmap APIs to provide a fast inline implementation. Bitmaps
//! of size 0 are very rare, and a compile-time-known-size 0 is most likely
//! a sign of error. They will be handled correctly by the bit/bitmap APIs,
//! but using the out-of-line functions, so that the inline implementations
//! can unconditionally dereference the pointer(s).
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
//! kcsan_check_read - check regular read access for races
//!
//! @ptr: address of access
//! @size: size of access
//! Yes, this permits 64-bit accesses on 32-bit architectures. These will
//! actually be atomic in some cases (namely Armv7 + LPAE), but for others we
//! rely on the access being split into 2x32-bit accesses for a 32-bit quantity
//! (e.g. a virtual address) and a strong prevailing wind.
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
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
//! # chars in a path name including nul
//! Internal kernel timer frequency
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! issue num suppressed message on exit
//! Call a function once. Similar to DO_ONCE(), but does not use jump label
//! patching via static keys.
//! strlen("ratelimit") + 1
//! New user namespace
//! This counts to 15. Any more, it will return 16th argument.
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
//! instrument_get_user() - add instrumentation to get_user()-like macros
//! @to: destination variable, may not be address-taken
//!
//! get_user() and friends are fragile, so it may depend on the implementation
//! whether the instrumentation happens before or after the data is copied from
//! the userspace.
//! ilog2 - log base 2 of 32-bit or a 64-bit unsigned value
//! @n: parameter
//!
//! constant-capable log of base 2 calculation
//! - this can be used to initialise global variables from constant data, hence
//! the massive ternary operator construction
//!
//! selects the appropriately-sized optimised version depending on sizeof(n)
//! writeable
//! Operation not permitted
//! This error code is special: arch syscall entry code will return
//! -ENOSYS if users try to call a syscall that doesn't exist.  To keep
//! failures of syscalls that really do exist distinguishable from
//! failures due to attempts to use a nonexistent syscall, syscall
//! implementations should refrain from returning -ENOSYS.
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
//! An exchange-type operation, which takes a value and a pointer, and
//! returns the old value.
//! @a is a power of 2 value
//! for_each_set_bit_wrap - iterate over all set bits starting from @start, and
//! wrapping around the end of bitmap.
//! @bit: offset for current iteration
//! @addr: bitmap address to base the search on
//! @size: bitmap size in number of bits
//! @start: Starting bit for bitmap traversing, wrapping around the bitmap end
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
//! Physical address zone modifiers (see linux/mmzone.h - low four bits)
//!
//! Do not put any conditional on these. If necessary modify the definitions
//! without the underscores and use them consistently. The definitions here may
//! be used in bit comparisons.
//! __sign_use for integer expressions:
//!   bit #0 set if ok for unsigned comparisons
//!   bit #1 set if ok for signed comparisons
//!
//! In particular, statically non-negative signed integer expressions
//! are ok for both.
//!
//! NOTE! Unsigned types smaller than 'int' are implicitly converted to 'int'
//! in expressions, and are accepted for signed conversions for now.
//! This is debatable.
//!
//! Note that 'x' is the original expression, and 'ux' is the unique variable
//! that contains the value.
//!
//! We use 'ux' for pure type checking, and 'x' for when we need to look at the
//! value (but without evaluating it for side effects!
//! Careful to only ever evaluate it with sizeof() or __builtin_constant_p() etc).
//!
//! Pointers end up being checked by the normal C type rules at the actual
//! comparison, and these expressions only need to be careful to not cause
//! warnings for pointer use.
//! This looks more complex than it should be. But we need to
//! get the type for the ~ right in round_down (it needs to be
//! as wide as the result!), and we want to evaluate the macro
//! arguments just once each.
//! for_each_if - helper for handling conditionals in various for_each macros
//! @condition: The condition to check
//!
//! Typical use::
//!
//! 	#define for_each_foo_bar(x, y) \'
//! 		list_for_each_entry(x, y->list, head) \'
//! 			for_each_if(x->something == SOMETHING)
//!
//! The for_each_if() macro makes the use of for_each_foo_bar() less error
//! prone.
//! lower_32_bits - return bits 0-31 of a number
//! @n: the number we're accessing
//! These are non-NULL pointers that will result in page faults
//! under normal circumstances, used to verify that nobody uses
//! non-initialized list entries.
//! list_entry - get the struct for this entry
//! @ptr:	the &struct list_head pointer.
//! @type:	the type of the struct this is embedded in.
//! @member:	the name of the list_head within the struct.
//! used to decode ioctl numbers..
//! The inline keyword gives the compiler room to decide to inline, or
//! not inline a function as it sees best.  However, as these functions
//! are called in both __init and non-__init functions, if they are not
//! inlined we will end up with a section mismatch error (of the type of
//! freeable items not being freed).  So we must use __always_inline here
//! to fix the problem.  If other functions in the future also end up in
//! this situation they will also need to be annotated as __always_inline
//! for_each_cpu - iterate over every cpu in a mask
//! @cpu: the (optionally unsigned) integer iterator
//! @mask: the cpumask pointer
//!
//! After the loop, cpu is >= nr_cpu_ids.
//! Wrap the arch provided IRQ routines to provide appropriate checks.
//! Instruction encoding provided for binutils backwards compatibility.
//! Defines VMX CPU feature bits
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
//! We use the MSB for PREEMPT_NEED_RESCHED mostly because it is available.
//! Macros to retrieve the current execution context:
//!
//! in_nmi()		- We're in NMI context
//! in_hardirq()		- We're in hard IRQ context
//! in_serving_softirq()	- We're in softirq context
//! in_task()		- We're in task context
//! These constants are defined in the MSan LLVM instrumentation pass.
//! ignore signal
//! bit-flags
//! This function is needed by all SMP systems. It must _always_ be valid
//! from the initial startup.
//! If !LOCKDEP, the value is meaningless
//! Shortcut for the queued_spin_lock_slowpath() function that allows
//! virt to hijack it.
//!
//! Returns:
//!   true - lock has been negotiated, all done;
//!   false - queued_spin_lock_slowpath() will do its thing.
//! If a tracepoint needs to be called from a header file, it is not
//! recommended to call it directly, as tracepoints in header files
//! may cause side-effects and bloat the kernel. Instead, use
//! tracepoint_enabled() to test if the tracepoint is enabled, then if
//! it is, call a wrapper function defined in a C file that will then
//! call the tracepoint.
//!
//! For "trace_foo_bar()", you would need to create a wrapper function
//! in a C file to call trace_foo_bar():
//!   void do_trace_foo_bar(args) { trace_foo_bar(args); }
//! Then in the header file, declare the tracepoint:
//!   DECLARE_TRACEPOINT(foo_bar);
//! And call your wrapper:
//!   static inline void some_inlined_function() {
//!            [..]
//!            if (tracepoint_enabled(foo_bar))
//!                    do_trace_foo_bar(args);
//!            [..]
//!   }
//!
//! Note: tracepoint_enabled(foo_bar) is equivalent to trace_foo_bar_enabled()
//!   but is safe to have in headers, where trace_foo_bar_enabled() is not.
//! Remapping spinlock architecture specific functions to the corresponding
//! queued spinlock functions.
//! A writer holds the lock
//! Define the various spin_lock methods.  Note we define these
//! regardless of whether CONFIG_SMP or CONFIG_PREEMPTION are set. The
//! various methods are defined as nops in the case they are not
//! required.
//! Convenience macros for the sake of set_current_state:
//! mutex_init - initialize the mutex
//! @mutex: the mutex to be initialized
//!
//! Initialize the mutex to unlocked state.
//!
//! It is not allowed to initialize an already locked mutex.
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
//! Values for SRCU Tree srcu_data ->srcu_reader_flavor, but also used by rcutorture.
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
//! Keep unconverted code working
//! number of env pointers
//! Generate SD flag indexes
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
//! used to be O_SYNC, see below
//! Valid hint values for F_{GET,SET}_RW_HINT. 0 is "not set", or can be
//! used to clear any hints previously set.
//! Gives us 8 prio classes with 13-bits of data for each class
//! no-op
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
//! Generic info of form tag = "info"
//! When adding a new stereo mode don't forget to adjust DRM_MODE_FLAGS_3D_MAX
//! (define not exposed to user space).
//! Number of bits in the handle that stack depot doesn't use. Users may store
//! information in them via stack_depot_set/get_extra_bits.
//! FIELD_GET() - extract a bitfield element
//! @_mask: shifted mask defining the field's length and position
//! @_reg:  value of entire bitfield
//!
//! FIELD_GET() extracts the field specified by @_mask from the
//! bitfield passed in as @_reg by masking and shifting it down.
//! Wrapper macro to convert from port number to the index used in some of the
//! registers. For Display version 20 and above it converts the port number to a
//! single range, starting with the TC offsets. When used together with
//! _PICK_EVEN_2RANGES(idx, PORT_TC1, ...), this single range will be the second
//! range. Example:
//!
//! PORT_TC1 -> PORT_TC1
//! PORT_TC2 -> PORT_TC2
//! PORT_TC3 -> PORT_TC3
//! PORT_TC4 -> PORT_TC4
//! PORT_A   -> PORT_TC4 + 1
//! PORT_B   -> PORT_TC4 + 2
//! ...
//! Must be a compile-time constant, so implement it as a macro
//! Bits set in the VMA until the stack is in its final location
//! modes for vm.memfd_noexec sysctl
//! These values are stored in task->ptrace_message
//! by ptrace_stop to describe the current syscall-stop.
//! SEIZE used, enable new behavior
//! Currently, the max observed size in the kernel code is
//! JUMP_LABEL_NOP_SIZE/RELATIVEJUMP_SIZE, which are 5.
//! Raise it if needed.
//! Legacy last prefixes
//! REX2 M0
//! Kprobe status flags
//! Display platforms and subplatforms. Keep platforms in display version based
//! order, chronological order within a version, and subplatforms next to the
//! platform.
//! Invoke param, a function-like macro, for each intel display param, with
//! arguments:
//!
//! param(type, name, value, mode)
//!
//! type: parameter type, one of {bool, int, unsigned int, unsigned long, char *}
//! name: name of the parameter
//! value: initial/default value of the parameter
//! mode: debugfs file permissions, one of {0400, 0600, 0}, use 0 to not create
//!       debugfs file
//! -ENOENT means we got the ref, but there's no tracking
//! including 0
//! Let people know that we have them
//! See the comment in asm-generic/io.h about ioremap_np().
//! Number of standard BARs
//! Number of reset methods used in pci_reset_fn_methods array in pci.c
//! These macros should be used after a dma_map_sg call has been done
//! to get bus addresses of each of the SG entries and their lengths.
//! You should only work with the number of sg entries dma_map_sg
//! returns, or alternatively stop on the first sg_dma_len(sg) which
//! is 0.
//! DMA_ATTR_WRITE_COMBINE: Specifies that writes to the mapping may be
//! buffered to improve performance.
//! Amount of SAGV/QGV points, BSpec precisely defines this
//! Helpful configuration macro to be used in tables etc.
//! This macro is used to tag functions as "printf-like" because
//! some compilers (like GCC) can catch printf format string problems.
//! Value returned by acpi_os_get_thread_id. There is no standard "thread_id"
//! across operating systems or even the various UNIX systems. Since ACPICA
//! only needs the thread ID as a unique thread identifier, we use a u64
//! as the only common data type - it will accommodate any type of pointer or
//! any type of integer. It is up to the host-dependent OSL to cast the
//! native thread ID type to a u64 (in acpi_os_get_thread_id).
//! For values, see Producer/Consumer above
//! The maximum expected number of CSI-2 data lanes.
//!
//! This number is not expected to ever have to be equal to or greater than the
//! number of bits in an unsigned long variable, but if it needs to be increased
//! above that limit, code will need to be adjusted accordingly.
//! DWORD 3
//! should be >= TASK_COMM_LEN
//! in linux/sched.h
//! hlist_nulls_first_rcu - returns the first element of the hash list.
//! @head: the head of the list.
//! define the enumeration of all cgroup subsystems
//! To avoid confusing the compiler (and generating warnings) with code
//! that attempts to access what would be a 0-element array (i.e. sized
//! to a potentially empty array when CGROUP_SUBSYS_COUNT == 0), this
//! constant expression can be added.
//! kunmap_local - Unmap a page mapped via kmap_local_page().
//! @__addr: An address within the page mapped
//!
//! @__addr can be any address within the mapped page.  Commonly it is the
//! address return from kmap_local_page(), but it can also include offsets.
//!
//! Unmapping should be done in the reverse order of the mapping.  See
//! kmap_local_page() for details.
//! 31 pointers + header align the folio_batch structure to a power of two
//! Receiver Capability
//!  + ((B + 0.000122)*4096))
//! VESA Display Stream Compression DSC 1.2 constants
//! DRM_FORMAT_MAX_PLANES - maximum number of planes a DRM format can have
//! Following Macros take a byte at a time for bit(s) masking
//!
//! TODO: HDCP_2_2_MAX_CONTENT_STREAMS_CNT is based upon actual
//! H/W MST streams capacity.
//! This required to be moved out to platform specific header.
//! Each region is a minimum of 16k, and there are at most 255 of them.
//! LT Phy MAC Register
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
// __acquire() - function to acquire context lock exclusively
// @x: context lock instance pointer
//
// No-op function that acquires the given context lock instance @x exclusively.
// __release() - function to release context lock exclusively
// @x: context lock instance pointer
//
// No-op function that releases the given context lock instance @x.
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
// __acquire_shared() - function to acquire context lock shared
// @x: context lock instance pointer
//
// No-op function that acquires the given context lock instance @x with shared
// access.
// __release_shared() - function to release context lock shared
// @x: context lock instance pointer
//
// No-op function that releases the given context lock instance @x with shared
// access.
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
// Useful shorthand for "is this condition known at compile-time?"
//
// Note that the condition may involve non-constant values,
// but the compiler may know enough about the details of the
// values to determine that the condition is statically true.
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
// Force a compilation error if a constant expression is not a power of 2
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
// SCHED_ISO: reserved but not implemented yet
// Concatenate two parameters, but allow them to be expanded beforehand.
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
// "acpi" ACPI via MSR
// "sse"
// "rdtscp" RDTSCP
// "rep_good" REP microcode works well
// "monitor" MONITOR/MWAIT support
// "est" Enhanced SpeedStep
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
// Non-serializing WRMSR
// INVLPGB and TLBSYNC instructions supported
// "wbnoinvd" WBNOINVD instruction
// "hwp" Intel Hardware P-states
// "ospke" OS Protection Keys Enable
// Shadow stack
// "la57" 5-level page tables
// "fsrm" Fast Short Rep Mov
// "serialize" SERIALIZE instruction
// LFENCE always serializing / synchronizes RDTSC
// BUG word(s)
// "apic_c1e" AMD Erratum 400
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
// Atomic operations are already serializing on x86
// The following macros are non-atomic versions of their non-underscored
// counterparts.
// __ffs - find first set bit in word
// @word: The word to search
//
// Undefined if no bit exists, so code should check against 0 first.
// ffz - find first zero bit in word
// @word: The word to search
//
// Undefined if no zero exists, so code should check against ~0UL first.
// Check that the bitops prototypes are sane
// assign_bit - Assign value to a bit in memory
// @nr: the bit to set
// @addr: the address to start counting from
// @value: the value to assign
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
// No such file or directory
// I/O error
// No such device or address
// Argument list too long
// Try again
// Out of memory
// Bad address
// Device or resource busy
// No such device
// Invalid argument
// Not a typewriter
// Text file busy
// No space left on device
// Math result not representable
// Invalid system call number
// Protocol error
// Value too large for defined data type
// Operation not supported on transport endpoint
// Connection timed out
// Memory page has hardware error
// Operation is not supported
// IS_ERR_VALUE - Detect an error pointer.
// @x: The pointer to check.
//
// Like IS_ERR(), but does not generate a compiler warning if result is unused.
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
// Use these carefully: no type checking, and uses the arguments
// multiple times. Use for obvious constants only.
// container_of_const - cast a member of a structure out to the containing
// 			structure and preserve the const-ness of the pointer
// @ptr:		the pointer to the member
// @type:		the type of the container struct this is embedded in.
// @member:		the name of the member within the struct.
//
// Always prefer container_of_const() instead of container_of() in new code.
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
// Same as above but for u64 dividends. divisor must be a 32-bit
// number.
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
// No static inline type checking - see Subtlety (1) above.
// for_each_cpu_wrap - iterate over every cpu in a mask, starting at a specified location
// @cpu: the (optionally unsigned) integer iterator
// @mask: the cpumask pointer
// @start: the start location
//
// The implementation does not assume any bit in @mask is set (including @start).
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
// Are we running in atomic context?  WARNING: this macro cannot
// always detect atomic context; in particular, it cannot know about
// held spinlocks in non-preemptible kernels.  Thus it should not be
// used in the general case to determine whether sleeping is possible.
// Do not use in_atomic() in driver code.
// disable sas during sighandling
// For trivial one-depth nesting of a lock-class, the following
// global define can be used. (Subsystems with multiple levels
// of nesting should define their own lock-nesting subclasses.)
// Writer mask
// Remapping rwlock architecture specific functions to the corresponding
// queued rwlock functions.
// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
// Need to take variadic arguments even though we don't use them, as dev_fmt()
// may only just have been expanded and may result in multiple arguments.
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
// srcu_read_lock().
// srcu_read_lock_nmisafe().
// 					0x4		// SRCU-lite is no longer with us.
// srcu_read_lock_fast(), also NMI-safe.
// srcu_read_lock_fast_updown().
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
// Generate SD flag bits
// Implementation specific max size
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
// kmem_cache_alloc and friends return pointers aligned to ARCH_SLAB_MINALIGN.
// kmalloc and friends return pointers aligned to both ARCH_KMALLOC_MINALIGN
// and ARCH_SLAB_MINALIGN, but here we only assume the former alignment.
// SLUB directly allocates requests fitting in to an order-1 page
// (PAGE_SIZE*2).  Larger requests are passed to the page allocator.
// Maximum size for which we actually use a slab cache
// Define gfp bits that should not be set for KMALLOC_NORMAL.
// __alloc_objs - Allocate objects of a given type using
// @KMALLOC: which size-based kmalloc wrapper to allocate with.
// @GFP: GFP flags for the allocation.
// @TYPE: type to allocate space for.
// @COUNT: how many @TYPE objects to allocate.
//
// Returns: Newly allocated pointer to (first) @TYPE of @COUNT-many
// allocated @TYPE objects, or NULL on failure.
// All kzalloc aliases for kmalloc_(obj|objs|flex).
// per-IO, return -EAGAIN if operation would block
// per-IO O_APPEND
// per-IO negation of O_APPEND
// Atomic Write
// buffered IO that drops the cache after reading or writing data
// mask of flags supported by the kernel
// Given the arbitrary numbers in varargs, pick the 0-based __index'th number.
//
// Always prefer _PICK_EVEN() over this if the numbers are evenly spaced.
// Local integer constant expression version of is_power_of_2().
// REG_FIELD_PREP8() - Prepare a u8 bitfield value
// @__mask: shifted mask defining the field's length and position
// @__val: value to put in the field
//
// Local copy of FIELD_PREP() to generate an integer constant expression, force
// u8 and for consistency with REG_FIELD_GET8(), REG_BIT8() and REG_GENMASK8().
//
// @return: @__val masked and shifted into the field defined by @__mask.
// REG_FIELD_PREP() - Prepare a u32 bitfield value
// @__mask: shifted mask defining the field's length and position
// @__val: value to put in the field
//
// Local copy of FIELD_PREP() to generate an integer constant expression, force
// u32 and for consistency with REG_FIELD_GET(), REG_BIT() and REG_GENMASK().
//
// @return: @__val masked and shifted into the field defined by @__mask.
// REG_FIELD_GET8() - Extract a u8 bitfield value
// @__mask: shifted mask defining the field's length and position
// @__val: value to extract the bitfield value from
//
// Local wrapper for FIELD_GET() to force u8 and for consistency with
// REG_FIELD_PREP(), REG_BIT() and REG_GENMASK().
//
// @return: Masked and shifted value of the field defined by @__mask in @__val.
// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
// PT_TRACE_* event enable flags
// 0x66
//
// LPFX1
// 0x67
// x86-64 REX prefix
// 0x4X
// AVX VEX prefixes
// 2-bytes VEX prefix
// 3-bytes VEX prefix
// EVEX prefix
// x86-64 REX2 prefix
// 0xD5
// AMD XOP prefix
// 0x8F
// Escape opcodes
// Group opcodes (1-16)
// Immediates
// VEX bit flags
// VEX3 Byte2
// VEX bit fields
// EVEX Byte1
// VEX3 Byte1
// VEX2.M always 1
// VEX3 Byte2, VEX2 Byte1
// XOP Byte2
// XOP Byte3
// breakpoint has already gone
// probe is temporarily disabled
// probe is really optimized.
// NOTE:
// this flag is only for optimized_kprobe.
// probe is using ftrace
// Device/Port type
// Root Port
// Link Control
// Root Control
// Link Control 2
// We overload the LSB of the page pointer to indicate whether it's
// a valid sg entry, or whether it points to the start of a new scatterlist.
// Those low bits are there for everyone! (thanks mason :-)
// DMA_ATTR_NO_WARN: This tells the DMA-mapping subsystem to suppress
// allocation failure reports (similarly to __GFP_NOWARN).
// A dma_addr_t can hold any valid DMA or bus address for the platform.  It can
// be given to a device to use as a DMA source or target.  It is specific to a
// given device and there may be a translation between the CPU physical address
// space and the bus address space.
//
// DMA_MAPPING_ERROR is the magic error code if a mapping failed.  It should not
// be used directly in drivers, but checked for using dma_mapping_error()
// instead.
// Use the high bit to mark if we used swiotlb for one or more ranges.
// Flexible array members are not allowed to be part of a union under
// C99, but this is not for any technical reason. Work around the
// limitation.
// Fixed by ACPI spec
// DWORD 5
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
// 1.3
// 1.2
// 2.0
// eDP
// XXX 1.2?
// 16-bit little-endian
// DP 1.4
// eDP 1.4
// eDP 1.4
// DP-HDMI2.1 PCON DSC ENCODER SUPPORT
// 0x92 through 0x9E
// DSC Capabilities 0x60 through 0x6F
// table size 2k - maximum due to use
// of chars for next/prev indices
// LT Phy Vendor Register
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
macro_rules! ADDR_ASSIGN {
($i:expr, $pll_reg:expr, $lt_state:expr) => {
loop {
unsafe {
*((*$lt_state).addr_msb.as_mut_ptr() as *mut u8).wrapping_offset(($i) as isize) = ($pll_reg.addr >> 8i32 & 255) as u8;
}
unsafe {
*((*$lt_state).addr_lsb.as_mut_ptr() as *mut u8).wrapping_offset(($i) as isize) = ($pll_reg.addr & 255) as u8;
}
if true {
break;
}
}
};
}

macro_rules! BITS_PER_TYPE {
($type:ty) => {
(core::mem::size_of::<$type>() as u64).wrapping_mul(8)
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__4 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2056>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__13 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1151>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__29 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1128>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__43 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1230_3>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__51 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1020>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__58 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1846>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__87 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1126_2>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__132 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1881>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__165 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2114_2>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__166 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1922_2>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__181 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2055_2>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__182 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1134>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__184 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1900>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__189 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1126>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__193 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2092>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__195 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1882>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__205 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1129>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__216 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2114>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__217 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2152_2>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__252 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2052_2>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__258 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1130_2>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__266 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2060>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__313 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1921_2>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__396 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_996>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__420 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1125_4>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__433 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1985>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__459 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1125_2>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__501 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2061>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__565 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2060_2>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__596 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1961>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__603 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2052>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__613 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1843>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__615 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1212>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__621 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1133>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__658 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2055>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__689 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2059_2>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__714 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1129_2>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__724 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2011>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__759 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1150>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__763 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1161>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__804 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1134_2>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__806 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1921>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__810 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2059>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__811 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1032>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__820 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2000>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__831 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1984>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__837 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1922>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__904 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2057_2>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__910 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1147>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__911 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1965>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__924 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2099>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__931 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2220>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__961 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1130>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__964 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1041>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1029 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1960>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1048 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1132_2>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1086 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1231>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1087 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1920_2>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1097 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1229_2>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1102 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1843_2>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1109 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1125_3>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1113 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2053_2>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1128 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1133_2>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1149 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1873>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1151 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1128_2>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1157 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1229>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1159 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1211>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1164 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1891>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1173 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1226>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1178 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1132>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1197 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1848>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1205 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2057>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1226 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2051_2>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1231 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1154>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1234 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2051>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1248 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1989>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1253 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1920>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1292 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1890>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1362 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1125>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1371 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2061_2>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1400 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2056_2>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1436 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2228>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1455 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1872>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1468 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1899>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1477 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1214>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1504 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1999>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1509 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1969>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1516 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1160>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1520 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2110_3>() as i32
};
}

/// Fixed-type variants of BIT(), with additional checks like GENMASK_TYPE(). The
/// following examples generate compiler warnings due to -Wshift-count-overflow:
///
/// - BIT_U8(8)
/// - BIT_U32(-1)
/// - BIT_U32(40)
macro_rules! BIT_INPUT_CHECK__1527 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2053>() as i32
};
}

macro_rules! BIT_TYPE__12 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1157!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__43 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__58!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__47 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1516!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__52 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__184!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__55 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__193!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__57 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__603!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__80 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__811!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__84 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__615!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__110 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__804!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__185 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__763!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__209 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__205!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__243 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__759!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__254 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1371!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__264 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1109!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__276 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1164!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__281 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__266!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__330 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1520!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__344 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1292!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__351 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1436!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__359 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__217!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__387 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__806!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__397 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__51!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__419 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__29!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__433 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1234!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__450 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__189!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__486 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__724!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__487 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1151!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__506 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__181!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__525 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1231!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__537 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__195!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__539 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__831!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__542 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__658!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__580 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1197!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__592 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1455!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__596 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1226!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__626 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__165!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__652 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1362!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__695 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__820!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__712 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__216!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__738 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__87!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__761 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1248!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__787 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1468!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__791 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__621!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__800 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1477!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__805 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__132!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__831 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__459!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__866 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__13!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__876 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1087!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__877 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1048!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__878 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1113!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__879 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1159!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__884 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__43!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__904 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__964!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__911 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1097!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__915 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__911!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__935 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1178!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__954 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__501!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__956 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1102!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__961 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__4!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__976 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__433!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__994 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__931!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1009 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__810!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1019 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__714!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1031 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__396!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1042 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__252!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1076 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__904!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1091 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__313!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1117 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__258!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1133 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1173!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1143 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__166!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1149 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__613!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1153 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1149!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1164 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1086!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1193 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__565!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1207 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1253!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1215 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1205!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1216 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1029!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1220 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1400!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1222 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__182!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1282 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__961!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1283 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__420!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1307 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1504!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1365 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__910!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1379 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__837!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1380 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1509!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1385 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__924!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1419 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__596!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1439 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__689!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1446 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1128!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_TYPE__1459 {
($type:ty, $nr:expr) => {
(BIT_INPUT_CHECK__1527!() as u64).wrapping_add(BIT_ULL!($nr)) as $type
};
}

macro_rules! BIT_U32 {
($nr:expr) => {
BIT_TYPE__961!(u32, $nr)
};
}

macro_rules! BIT_U32__6 {
($nr:expr) => {
BIT_TYPE__866!(u32, $nr)
};
}

macro_rules! BIT_U32__21 {
($nr:expr) => {
BIT_TYPE__1042!(u32, $nr)
};
}

macro_rules! BIT_U32__46 {
($nr:expr) => {
BIT_TYPE__1193!(u32, $nr)
};
}

macro_rules! BIT_U32__48 {
($nr:expr) => {
BIT_TYPE__1446!(u32, $nr)
};
}

macro_rules! BIT_U32__161 {
($nr:expr) => {
BIT_TYPE__1164!(u32, $nr)
};
}

macro_rules! BIT_U32__174 {
($nr:expr) => {
BIT_TYPE__800!(u32, $nr)
};
}

macro_rules! BIT_U32__185 {
($nr:expr) => {
BIT_TYPE__954!(u32, $nr)
};
}

macro_rules! BIT_U32__195 {
($nr:expr) => {
BIT_TYPE__1459!(u32, $nr)
};
}

macro_rules! BIT_U32__205 {
($nr:expr) => {
BIT_TYPE__994!(u32, $nr)
};
}

macro_rules! BIT_U32__299 {
($nr:expr) => {
BIT_TYPE__450!(u32, $nr)
};
}

macro_rules! BIT_U32__307 {
($nr:expr) => {
BIT_TYPE__877!(u32, $nr)
};
}

macro_rules! BIT_U32__320 {
($nr:expr) => {
BIT_TYPE__915!(u32, $nr)
};
}

macro_rules! BIT_U32__344 {
($nr:expr) => {
BIT_TYPE__1143!(u32, $nr)
};
}

macro_rules! BIT_U32__351 {
($nr:expr) => {
BIT_TYPE__831!(u32, $nr)
};
}

macro_rules! BIT_U32__354 {
($nr:expr) => {
BIT_TYPE__911!(u32, $nr)
};
}

macro_rules! BIT_U32__364 {
($nr:expr) => {
BIT_TYPE__419!(u32, $nr)
};
}

macro_rules! BIT_U32__394 {
($nr:expr) => {
BIT_TYPE__1379!(u32, $nr)
};
}

macro_rules! BIT_U32__408 {
($nr:expr) => {
BIT_TYPE__1133!(u32, $nr)
};
}

macro_rules! BIT_U32__426 {
($nr:expr) => {
BIT_TYPE__487!(u32, $nr)
};
}

macro_rules! BIT_U32__428 {
($nr:expr) => {
BIT_TYPE__243!(u32, $nr)
};
}

macro_rules! BIT_U32__430 {
($nr:expr) => {
BIT_TYPE__1282!(u32, $nr)
};
}

macro_rules! BIT_U32__440 {
($nr:expr) => {
BIT_TYPE__876!(u32, $nr)
};
}

macro_rules! BIT_U32__446 {
($nr:expr) => {
BIT_TYPE__525!(u32, $nr)
};
}

macro_rules! BIT_U32__447 {
($nr:expr) => {
BIT_TYPE__1380!(u32, $nr)
};
}

macro_rules! BIT_U32__455 {
($nr:expr) => {
BIT_TYPE__185!(u32, $nr)
};
}

macro_rules! BIT_U32__457 {
($nr:expr) => {
BIT_TYPE__254!(u32, $nr)
};
}

macro_rules! BIT_U32__507 {
($nr:expr) => {
BIT_TYPE__904!(u32, $nr)
};
}

macro_rules! BIT_U32__550 {
($nr:expr) => {
BIT_TYPE__1307!(u32, $nr)
};
}

macro_rules! BIT_U32__557 {
($nr:expr) => {
BIT_TYPE__80!(u32, $nr)
};
}

macro_rules! BIT_U32__566 {
($nr:expr) => {
BIT_TYPE__1031!(u32, $nr)
};
}

macro_rules! BIT_U32__569 {
($nr:expr) => {
BIT_TYPE__652!(u32, $nr)
};
}

macro_rules! BIT_U32__573 {
($nr:expr) => {
BIT_TYPE__1222!(u32, $nr)
};
}

macro_rules! BIT_U32__574 {
($nr:expr) => {
BIT_TYPE__209!(u32, $nr)
};
}

macro_rules! BIT_U32__586 {
($nr:expr) => {
BIT_TYPE__1009!(u32, $nr)
};
}

macro_rules! BIT_U32__603 {
($nr:expr) => {
BIT_TYPE__1091!(u32, $nr)
};
}

macro_rules! BIT_U32__622 {
($nr:expr) => {
BIT_TYPE__1019!(u32, $nr)
};
}

macro_rules! BIT_U32__627 {
($nr:expr) => {
BIT_TYPE__55!(u32, $nr)
};
}

macro_rules! BIT_U32__630 {
($nr:expr) => {
BIT_TYPE__738!(u32, $nr)
};
}

macro_rules! BIT_U32__643 {
($nr:expr) => {
BIT_TYPE__626!(u32, $nr)
};
}

macro_rules! BIT_U32__681 {
($nr:expr) => {
BIT_TYPE__1207!(u32, $nr)
};
}

macro_rules! BIT_U32__768 {
($nr:expr) => {
BIT_TYPE__1076!(u32, $nr)
};
}

macro_rules! BIT_U32__776 {
($nr:expr) => {
BIT_TYPE__506!(u32, $nr)
};
}

macro_rules! BIT_U32__824 {
($nr:expr) => {
BIT_TYPE__1365!(u32, $nr)
};
}

macro_rules! BIT_U32__837 {
($nr:expr) => {
BIT_TYPE__330!(u32, $nr)
};
}

macro_rules! BIT_U32__839 {
($nr:expr) => {
BIT_TYPE__1117!(u32, $nr)
};
}

macro_rules! BIT_U32__848 {
($nr:expr) => {
BIT_TYPE__695!(u32, $nr)
};
}

macro_rules! BIT_U32__855 {
($nr:expr) => {
BIT_TYPE__1283!(u32, $nr)
};
}

macro_rules! BIT_U32__888 {
($nr:expr) => {
BIT_TYPE__1215!(u32, $nr)
};
}

macro_rules! BIT_U32__906 {
($nr:expr) => {
BIT_TYPE__879!(u32, $nr)
};
}

macro_rules! BIT_U32__908 {
($nr:expr) => {
BIT_TYPE__791!(u32, $nr)
};
}

macro_rules! BIT_U32__928 {
($nr:expr) => {
BIT_TYPE__12!(u32, $nr)
};
}

macro_rules! BIT_U32__985 {
($nr:expr) => {
BIT_TYPE__542!(u32, $nr)
};
}

macro_rules! BIT_U32__995 {
($nr:expr) => {
BIT_TYPE__976!(u32, $nr)
};
}

macro_rules! BIT_U32__996 {
($nr:expr) => {
BIT_TYPE__884!(u32, $nr)
};
}

macro_rules! BIT_U32__1000 {
($nr:expr) => {
BIT_TYPE__433!(u32, $nr)
};
}

macro_rules! BIT_U32__1014 {
($nr:expr) => {
BIT_TYPE__539!(u32, $nr)
};
}

macro_rules! BIT_U32__1045 {
($nr:expr) => {
BIT_TYPE__47!(u32, $nr)
};
}

macro_rules! BIT_U32__1068 {
($nr:expr) => {
BIT_TYPE__1385!(u32, $nr)
};
}

macro_rules! BIT_U32__1076 {
($nr:expr) => {
BIT_TYPE__596!(u32, $nr)
};
}

macro_rules! BIT_U32__1089 {
($nr:expr) => {
BIT_TYPE__935!(u32, $nr)
};
}

macro_rules! BIT_U32__1113 {
($nr:expr) => {
BIT_TYPE__84!(u32, $nr)
};
}

macro_rules! BIT_U32__1149 {
($nr:expr) => {
BIT_TYPE__110!(u32, $nr)
};
}

macro_rules! BIT_U32__1178 {
($nr:expr) => {
BIT_TYPE__264!(u32, $nr)
};
}

macro_rules! BIT_U32__1189 {
($nr:expr) => {
BIT_TYPE__57!(u32, $nr)
};
}

macro_rules! BIT_U32__1211 {
($nr:expr) => {
BIT_TYPE__761!(u32, $nr)
};
}

macro_rules! BIT_U32__1221 {
($nr:expr) => {
BIT_TYPE__387!(u32, $nr)
};
}

macro_rules! BIT_U32__1333 {
($nr:expr) => {
BIT_TYPE__1220!(u32, $nr)
};
}

macro_rules! BIT_U32__1347 {
($nr:expr) => {
BIT_TYPE__1439!(u32, $nr)
};
}

macro_rules! BIT_U32__1352 {
($nr:expr) => {
BIT_TYPE__712!(u32, $nr)
};
}

macro_rules! BIT_U32__1424 {
($nr:expr) => {
BIT_TYPE__281!(u32, $nr)
};
}

macro_rules! BIT_U32__1431 {
($nr:expr) => {
BIT_TYPE__878!(u32, $nr)
};
}

macro_rules! BIT_U32__1447 {
($nr:expr) => {
BIT_TYPE__351!(u32, $nr)
};
}

macro_rules! BIT_U32__1474 {
($nr:expr) => {
BIT_TYPE__397!(u32, $nr)
};
}

macro_rules! BIT_U8 {
($nr:expr) => {
BIT_TYPE__1419!(u8, $nr)
};
}

macro_rules! BIT_U8__2 {
($nr:expr) => {
BIT_TYPE__486!(u8, $nr)
};
}

macro_rules! BIT_U8__5 {
($nr:expr) => {
BIT_TYPE__43!(u8, $nr)
};
}

macro_rules! BIT_U8__6 {
($nr:expr) => {
BIT_TYPE__359!(u8, $nr)
};
}

macro_rules! BIT_U8__9 {
($nr:expr) => {
BIT_TYPE__52!(u8, $nr)
};
}

macro_rules! BIT_U8__10 {
($nr:expr) => {
BIT_TYPE__276!(u8, $nr)
};
}

macro_rules! BIT_U8__12 {
($nr:expr) => {
BIT_TYPE__344!(u8, $nr)
};
}

macro_rules! BIT_U8__15 {
($nr:expr) => {
BIT_TYPE__580!(u8, $nr)
};
}

macro_rules! BIT_U8__18 {
($nr:expr) => {
BIT_TYPE__592!(u8, $nr)
};
}

macro_rules! BIT_U8__20 {
($nr:expr) => {
BIT_TYPE__787!(u8, $nr)
};
}

macro_rules! BIT_U8__25 {
($nr:expr) => {
BIT_TYPE__1216!(u8, $nr)
};
}

macro_rules! BIT_U8__28 {
($nr:expr) => {
BIT_TYPE__537!(u8, $nr)
};
}

macro_rules! BIT_U8__29 {
($nr:expr) => {
BIT_TYPE__1149!(u8, $nr)
};
}

macro_rules! BIT_U8__30 {
($nr:expr) => {
BIT_TYPE__956!(u8, $nr)
};
}

macro_rules! BIT_U8__32 {
($nr:expr) => {
BIT_TYPE__805!(u8, $nr)
};
}

macro_rules! BIT_U8__34 {
($nr:expr) => {
BIT_TYPE__1153!(u8, $nr)
};
}

macro_rules! BIT_ULL {
($nr:expr) => {
ULL!() << $nr
};
}

macro_rules! BIT {
($nr:expr) => {
UL!() << $nr
};
}

macro_rules! BIT__3 {
($nr:expr) => {
UL!() << (($nr) as u32)
};
}

macro_rules! BUGFLAG_TAINT {
($taint:expr) => {
($taint) << 8i32
};
}

macro_rules! DATA_ASSIGN {
($i:expr, $pll_reg:expr, $lt_state:expr) => {
loop {
unsafe {
*((*((*$lt_state).data.as_mut_ptr() as *mut [u8; 4]).wrapping_offset(($i) as isize)).as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) = (($pll_reg.val & 4278190080) >> 24i32) as u8;
}
unsafe {
*((*((*$lt_state).data.as_mut_ptr() as *mut [u8; 4]).wrapping_offset(($i) as isize)).as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) = (($pll_reg.val & 16711680) >> 16i32) as u8;
}
unsafe {
*((*((*$lt_state).data.as_mut_ptr() as *mut [u8; 4]).wrapping_offset(($i) as isize)).as_mut_ptr() as *mut u8).wrapping_offset(2 as isize) = (($pll_reg.val & 65280) >> 8i32) as u8;
}
unsafe {
*((*((*$lt_state).data.as_mut_ptr() as *mut [u8; 4]).wrapping_offset(($i) as isize)).as_mut_ptr() as *mut u8).wrapping_offset(3 as isize) = ($pll_reg.val & 255) as u8;
}
if true {
break;
}
}
};
}

macro_rules! DISPLAY_RUNTIME_INFO {
($__display:expr) => {
unsafe { core::ptr::addr_of_mut!((*$__display).info.__runtime_info) }
};
}

macro_rules! DISPLAY_VER {
($__display:expr) => {
unsafe { (*DISPLAY_RUNTIME_INFO!($__display)).ip.ver }
};
}

macro_rules! DIV_ROUND_DOWN_ULL {
($ll:expr, $d:expr) => {
{
let mut __v: u64;
let mut __v2: u64 = $ll;
do_div!(__v2, $d);
__v = __v2;
__v
}
};
}

macro_rules! DIV_ROUND_UP_ULL__3 {
($ll:expr, $d:expr) => {
DIV_ROUND_DOWN_ULL!((($ll) as u64).wrapping_add($d).wrapping_sub(1), $d)
};
}

macro_rules! GENMASK_INPUT_CHECK__50 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1140>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__70 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1852>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__74 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2166>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__107 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2110>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__116 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2152>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__124 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1705>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__152 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1230>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__192 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1857_2>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__216 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1852_2>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__224 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1866>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__256 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1699>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__264 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1854>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__279 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2162>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__285 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2158>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__289 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1859>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__290 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2110_2>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__295 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1137>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__309 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1230_2>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__314 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1857>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__345 {
() => {
core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1862>() as i32
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
macro_rules! GENMASK_TYPE__54 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__290!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1).wrapping_sub(($h) as u64))
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
macro_rules! GENMASK_TYPE__71 {
($t:ty, $h:expr, $l:expr) => {
GENMASK_INPUT_CHECK__285!().wrapping_add((type_max__2!() as i32) << $l & (type_max__2!() as i32) >> BITS_PER_TYPE!($t).wrapping_sub(1).wrapping_sub(($h) as u64)) as $t
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
macro_rules! GENMASK_TYPE__74 {
($t:ty, $h:expr, $l:expr) => {
GENMASK_INPUT_CHECK__216!().wrapping_add((type_max__2!() as i32) << $l & (type_max__2!() as i32) >> BITS_PER_TYPE!($t).wrapping_sub(1).wrapping_sub(($h) as u64)) as $t
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
macro_rules! GENMASK_TYPE__75 {
($t:ty, $h:expr, $l:expr) => {
GENMASK_INPUT_CHECK__279!().wrapping_add((type_max__2!() as i32) << $l & (type_max__2!() as i32) >> BITS_PER_TYPE!($t).wrapping_sub(1).wrapping_sub(($h) as u64)) as $t
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
macro_rules! GENMASK_TYPE__119 {
($t:ty, $h:expr, $l:expr) => {
GENMASK_INPUT_CHECK__314!().wrapping_add((type_max__2!() as i32) << $l & (type_max__2!() as i32) >> BITS_PER_TYPE!($t).wrapping_sub(1).wrapping_sub(($h) as u64)) as $t
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
macro_rules! GENMASK_TYPE__134 {
($t:ty, $h:expr, $l:expr) => {
GENMASK_INPUT_CHECK__74!().wrapping_add((type_max__2!() as i32) << $l & (type_max__2!() as i32) >> BITS_PER_TYPE!($t).wrapping_sub(1).wrapping_sub(($h) as u64)) as $t
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
macro_rules! GENMASK_TYPE__141 {
($t:ty, $h:expr, $l:expr) => {
GENMASK_INPUT_CHECK__70!().wrapping_add((type_max__2!() as i32) << $l & (type_max__2!() as i32) >> BITS_PER_TYPE!($t).wrapping_sub(1).wrapping_sub(($h) as u64)) as $t
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
macro_rules! GENMASK_TYPE__153 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__309!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1).wrapping_sub(($h) as u64))
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
macro_rules! GENMASK_TYPE__166 {
($t:ty, $h:expr, $l:expr) => {
GENMASK_INPUT_CHECK__289!().wrapping_add((type_max__2!() as i32) << $l & (type_max__2!() as i32) >> BITS_PER_TYPE!($t).wrapping_sub(1).wrapping_sub(($h) as u64)) as $t
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
macro_rules! GENMASK_TYPE__169 {
($t:ty, $h:expr, $l:expr) => {
GENMASK_INPUT_CHECK__192!().wrapping_add((type_max__2!() as i32) << $l & (type_max__2!() as i32) >> BITS_PER_TYPE!($t).wrapping_sub(1).wrapping_sub(($h) as u64)) as $t
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
macro_rules! GENMASK_TYPE__206 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__124!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1).wrapping_sub(($h) as u64))
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
macro_rules! GENMASK_TYPE__215 {
($t:ty, $h:expr, $l:expr) => {
GENMASK_INPUT_CHECK__224!().wrapping_add((type_max__2!() as i32) << $l & (type_max__2!() as i32) >> BITS_PER_TYPE!($t).wrapping_sub(1).wrapping_sub(($h) as u64)) as $t
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
macro_rules! GENMASK_TYPE__221 {
($t:ty, $h:expr, $l:expr) => {
GENMASK_INPUT_CHECK__116!().wrapping_add((type_max__2!() as i32) << $l & (type_max__2!() as i32) >> BITS_PER_TYPE!($t).wrapping_sub(1).wrapping_sub(($h) as u64)) as $t
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
macro_rules! GENMASK_TYPE__245 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__50!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1).wrapping_sub(($h) as u64))
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
macro_rules! GENMASK_TYPE__266 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__256!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1).wrapping_sub(($h) as u64))
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
macro_rules! GENMASK_TYPE__312 {
($t:ty, $h:expr, $l:expr) => {
GENMASK_INPUT_CHECK__345!().wrapping_add((type_max__2!() as i32) << $l & (type_max__2!() as i32) >> BITS_PER_TYPE!($t).wrapping_sub(1).wrapping_sub(($h) as u64)) as $t
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
macro_rules! GENMASK_TYPE__320 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__107!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1).wrapping_sub(($h) as u64))
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
macro_rules! GENMASK_TYPE__338 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__152!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1).wrapping_sub(($h) as u64))
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
macro_rules! GENMASK_TYPE__350 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__295!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1).wrapping_sub(($h) as u64))
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
macro_rules! GENMASK_TYPE__352 {
($t:ty, $h:expr, $l:expr) => {
GENMASK_INPUT_CHECK__264!().wrapping_add((type_max__2!() as i32) << $l & (type_max__2!() as i32) >> BITS_PER_TYPE!($t).wrapping_sub(1).wrapping_sub(($h) as u64)) as $t
};
}

macro_rules! GENMASK_U32__24 {
($h:expr, $l:expr) => {
GENMASK_TYPE__320!(u32, $h, $l)
};
}

macro_rules! GENMASK_U32__60 {
($h:expr, $l:expr) => {
GENMASK_TYPE__266!(u32, $h, $l)
};
}

macro_rules! GENMASK_U32__133 {
($h:expr, $l:expr) => {
GENMASK_TYPE__153!(u32, $h, $l)
};
}

macro_rules! GENMASK_U32__135 {
($h:expr, $l:expr) => {
GENMASK_TYPE__245!(u32, $h, $l)
};
}

macro_rules! GENMASK_U32__168 {
($h:expr, $l:expr) => {
GENMASK_TYPE__206!(u32, $h, $l)
};
}

macro_rules! GENMASK_U32__171 {
($h:expr, $l:expr) => {
GENMASK_TYPE__350!(u32, $h, $l)
};
}

macro_rules! GENMASK_U32__215 {
($h:expr, $l:expr) => {
GENMASK_TYPE__54!(u32, $h, $l)
};
}

macro_rules! GENMASK_U32__301 {
($h:expr, $l:expr) => {
GENMASK_TYPE__338!(u32, $h, $l)
};
}

macro_rules! GENMASK_U8__2 {
($h:expr, $l:expr) => {
GENMASK_TYPE__74!(u8, $h, $l)
};
}

macro_rules! GENMASK_U8__5 {
($h:expr, $l:expr) => {
GENMASK_TYPE__141!(u8, $h, $l)
};
}

macro_rules! GENMASK_U8__6 {
($h:expr, $l:expr) => {
GENMASK_TYPE__221!(u8, $h, $l)
};
}

macro_rules! GENMASK_U8__7 {
($h:expr, $l:expr) => {
GENMASK_TYPE__166!(u8, $h, $l)
};
}

macro_rules! GENMASK_U8__8 {
($h:expr, $l:expr) => {
GENMASK_TYPE__215!(u8, $h, $l)
};
}

macro_rules! GENMASK_U8__10 {
($h:expr, $l:expr) => {
GENMASK_TYPE__352!(u8, $h, $l)
};
}

macro_rules! GENMASK_U8__11 {
($h:expr, $l:expr) => {
GENMASK_TYPE__134!(u8, $h, $l)
};
}

macro_rules! GENMASK_U8__14 {
($h:expr, $l:expr) => {
GENMASK_TYPE__71!(u8, $h, $l)
};
}

macro_rules! GENMASK_U8__16 {
($h:expr, $l:expr) => {
GENMASK_TYPE__169!(u8, $h, $l)
};
}

macro_rules! GENMASK_U8__17 {
($h:expr, $l:expr) => {
GENMASK_TYPE__312!(u8, $h, $l)
};
}

macro_rules! GENMASK_U8__22 {
($h:expr, $l:expr) => {
GENMASK_TYPE__75!(u8, $h, $l)
};
}

macro_rules! GENMASK_U8__25 {
($h:expr, $l:expr) => {
GENMASK_TYPE__119!(u8, $h, $l)
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

macro_rules! LT_PHY_TXY_CTL10 {
($idx:expr) => {
1034i32.wrapping_add(512i32.wrapping_mul($idx))
};
}

macro_rules! LT_PHY_TXY_CTL2 {
($idx:expr) => {
1026i32.wrapping_add(512i32.wrapping_mul($idx))
};
}

macro_rules! LT_PHY_TXY_CTL3 {
($idx:expr) => {
1027i32.wrapping_add(512i32.wrapping_mul($idx))
};
}

macro_rules! LT_PHY_TXY_CTL4 {
($idx:expr) => {
1028i32.wrapping_add(512i32.wrapping_mul($idx))
};
}

/// LT Phy Pipe Spec Registers
macro_rules! LT_PHY_TXY_CTL8 {
($idx:expr) => {
1032i32.wrapping_add(512i32.wrapping_mul($idx))
};
}

macro_rules! LT_PHY_TX_CURSOR {
($val:expr) => {
(((($val) as u8) as i32) << ((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2159_2>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 0i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(5)) as u8) as u64).trailing_zeros() as i32) & (((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2159_3>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 0i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(5)) as u8) as i32)).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2159_4>() as i32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2159_6>() as i32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2159_9>() as i32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2159_16>() as i32) as u8
};
}

macro_rules! LT_PHY_TX_CURSOR__2 {
($val:expr) => {
(((($val) as u8) as i32) << ((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2163_2>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 0i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(5)) as u8) as u64).trailing_zeros() as i32) & (((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2163_3>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 0i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(5)) as u8) as i32)).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2163_4>() as i32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2163_6>() as i32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2163_9>() as i32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2163_16>() as i32) as u8
};
}

macro_rules! LT_PHY_TX_CURSOR__3 {
($val:expr) => {
(((($val) as u8) as i32) << ((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2167_2>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 0i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(5)) as u8) as u64).trailing_zeros() as i32) & (((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2167_3>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 0i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(5)) as u8) as i32)).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2167_4>() as i32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2167_6>() as i32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2167_9>() as i32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2167_16>() as i32) as u8
};
}

macro_rules! LT_PHY_TX_SWING_LEVEL {
($val:expr) => {
(((($val) as u8) as i32) << ((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2153_2>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 4i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)) as u8) as u64).trailing_zeros() as i32) & (((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2153_3>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 4i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(7)) as u8) as i32)).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2153_4>() as i32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2153_6>() as i32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2153_9>() as i32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2153_16>() as i32) as u8
};
}

macro_rules! LT_PHY_TX_SWING {
($val:expr) => {
(((($val) as u8) as i32) << ((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2154_2>() as u64).wrapping_add(8u64) as u8) as u64).trailing_zeros() as i32) & (((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2154_3>() as u64).wrapping_add(8u64) as u8) as i32)).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2154_4>() as i32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2154_6>() as i32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2154_9>() as i32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2154_16>() as i32) as u8
};
}

macro_rules! LT_PHY_VDR_X_ADDR_LSB {
($idx:expr) => {
3077i32.wrapping_add(6i32.wrapping_mul($idx))
};
}

macro_rules! LT_PHY_VDR_X_ADDR_MSB {
($idx:expr) => {
3076i32.wrapping_add(6i32.wrapping_mul($idx))
};
}

macro_rules! LT_PHY_VDR_X_DATAY {
($idx:expr, $y:expr) => {
3078i32.wrapping_add(3i32.wrapping_sub($y)).wrapping_add(6i32.wrapping_mul($idx))
};
}

macro_rules! PLL_REG_ADDR {
($base:expr, $pll_type:expr) => {
if $pll_type { $base.wrapping_add(512) } else { $base }
};
}

macro_rules! Q32_TO_FRAC {
($x:expr) => {
($x) & 4294967295
};
}

macro_rules! Q32_TO_INT {
($x:expr) => {
($x) >> 32i32
};
}

macro_rules! REGVAL {
($i:expr, $lt_state:expr) => {
((unsafe { *((*((*$lt_state).data.as_ptr() as *const [u8; 4]).wrapping_offset(($i) as isize)).as_ptr() as *const u8).wrapping_offset(3 as isize) }) as i32) | ((unsafe { *((*((*$lt_state).data.as_ptr() as *const [u8; 4]).wrapping_offset(($i) as isize)).as_ptr() as *const u8).wrapping_offset(2 as isize) }) as i32) << 8i32 | ((unsafe { *((*((*$lt_state).data.as_ptr() as *const [u8; 4]).wrapping_offset(($i) as isize)).as_ptr() as *const u8).wrapping_offset(1 as isize) }) as i32) << 16i32 | ((unsafe { *((*((*$lt_state).data.as_ptr() as *const [u8; 4]).wrapping_offset(($i) as isize)).as_ptr() as *const u8).wrapping_offset(0 as isize) }) as i32) << 24i32
};
}

macro_rules! REG_BIT8__2 {
($n:expr) => {
BIT_U8__12!($n)
};
}

macro_rules! REG_BIT8__3 {
($n:expr) => {
BIT_U8__25!($n)
};
}

macro_rules! REG_BIT8__6 {
($n:expr) => {
BIT_U8__6!($n)
};
}

macro_rules! REG_BIT8__12 {
($n:expr) => {
BIT_U8__28!($n)
};
}

macro_rules! REG_BIT8__13 {
($n:expr) => {
BIT_U8__18!($n)
};
}

macro_rules! REG_BIT8__15 {
($n:expr) => {
BIT_U8__30!($n)
};
}

macro_rules! REG_BIT8__17 {
($n:expr) => {
BIT_U8__2!($n)
};
}

macro_rules! REG_BIT8__18 {
($n:expr) => {
BIT_U8__32!($n)
};
}

macro_rules! REG_BIT8__20 {
($n:expr) => {
BIT_U8__34!($n)
};
}

macro_rules! REG_BIT8__22 {
($n:expr) => {
BIT_U8__20!($n)
};
}

macro_rules! REG_BIT8__26 {
($n:expr) => {
BIT_U8__15!($n)
};
}

macro_rules! REG_BIT8__28 {
($n:expr) => {
BIT_U8!($n)
};
}

macro_rules! REG_BIT8__31 {
($n:expr) => {
BIT_U8__10!($n)
};
}

macro_rules! REG_BIT8__32 {
($n:expr) => {
BIT_U8__5!($n)
};
}

macro_rules! REG_BIT8__35 {
($n:expr) => {
BIT_U8__9!($n)
};
}

macro_rules! REG_BIT8__37 {
($n:expr) => {
BIT_U8__29!($n)
};
}

macro_rules! REG_BIT__15 {
($n:expr) => {
BIT_U32__1189!($n)
};
}

macro_rules! REG_BIT__31 {
($n:expr) => {
BIT_U32__1000!($n)
};
}

macro_rules! REG_BIT__38 {
($n:expr) => {
BIT_U32__1211!($n)
};
}

macro_rules! REG_BIT__56 {
($n:expr) => {
BIT_U32__195!($n)
};
}

macro_rules! REG_BIT__84 {
($n:expr) => {
BIT_U32__46!($n)
};
}

macro_rules! REG_BIT__115 {
($n:expr) => {
BIT_U32__6!($n)
};
}

macro_rules! REG_BIT__119 {
($n:expr) => {
BIT_U32__394!($n)
};
}

macro_rules! REG_BIT__178 {
($n:expr) => {
BIT_U32__320!($n)
};
}

macro_rules! REG_BIT__226 {
($n:expr) => {
BIT_U32__161!($n)
};
}

macro_rules! REG_BIT__252 {
($n:expr) => {
BIT_U32__681!($n)
};
}

macro_rules! REG_BIT__266 {
($n:expr) => {
BIT_U32__996!($n)
};
}

macro_rules! REG_BIT__296 {
($n:expr) => {
BIT_U32__776!($n)
};
}

macro_rules! REG_BIT__312 {
($n:expr) => {
BIT_U32__1076!($n)
};
}

macro_rules! REG_BIT__317 {
($n:expr) => {
BIT_U32__906!($n)
};
}

macro_rules! REG_BIT__353 {
($n:expr) => {
BIT_U32__603!($n)
};
}

macro_rules! REG_BIT__362 {
($n:expr) => {
BIT_U32__1474!($n)
};
}

macro_rules! REG_BIT__366 {
($n:expr) => {
BIT_U32__440!($n)
};
}

macro_rules! REG_BIT__426 {
($n:expr) => {
BIT_U32__299!($n)
};
}

macro_rules! REG_BIT__466 {
($n:expr) => {
BIT_U32__1113!($n)
};
}

macro_rules! REG_BIT__489 {
($n:expr) => {
BIT_U32__888!($n)
};
}

macro_rules! REG_BIT__494 {
($n:expr) => {
BIT_U32__995!($n)
};
}

macro_rules! REG_BIT__534 {
($n:expr) => {
BIT_U32__457!($n)
};
}

macro_rules! REG_BIT__540 {
($n:expr) => {
BIT_U32__837!($n)
};
}

macro_rules! REG_BIT__548 {
($n:expr) => {
BIT_U32__344!($n)
};
}

macro_rules! REG_BIT__553 {
($n:expr) => {
BIT_U32__824!($n)
};
}

macro_rules! REG_BIT__582 {
($n:expr) => {
BIT_U32__21!($n)
};
}

macro_rules! REG_BIT__610 {
($n:expr) => {
BIT_U32__185!($n)
};
}

macro_rules! REG_BIT__650 {
($n:expr) => {
BIT_U32__928!($n)
};
}

macro_rules! REG_BIT__652 {
($n:expr) => {
BIT_U32__364!($n)
};
}

macro_rules! REG_BIT__673 {
($n:expr) => {
BIT_U32__48!($n)
};
}

macro_rules! REG_BIT__689 {
($n:expr) => {
BIT_U32__574!($n)
};
}

macro_rules! REG_BIT__697 {
($n:expr) => {
BIT_U32__1347!($n)
};
}

macro_rules! REG_BIT__712 {
($n:expr) => {
BIT_U32__1447!($n)
};
}

macro_rules! REG_BIT__740 {
($n:expr) => {
BIT_U32__446!($n)
};
}

macro_rules! REG_BIT__781 {
($n:expr) => {
BIT_U32__768!($n)
};
}

macro_rules! REG_BIT__782 {
($n:expr) => {
BIT_U32__1431!($n)
};
}

macro_rules! REG_BIT__794 {
($n:expr) => {
BIT_U32__1089!($n)
};
}

macro_rules! REG_BIT__803 {
($n:expr) => {
BIT_U32!($n)
};
}

macro_rules! REG_BIT__812 {
($n:expr) => {
BIT_U32__354!($n)
};
}

macro_rules! REG_BIT__824 {
($n:expr) => {
BIT_U32__630!($n)
};
}

macro_rules! REG_BIT__825 {
($n:expr) => {
BIT_U32__908!($n)
};
}

macro_rules! REG_BIT__831 {
($n:expr) => {
BIT_U32__428!($n)
};
}

macro_rules! REG_BIT__844 {
($n:expr) => {
BIT_U32__1068!($n)
};
}

macro_rules! REG_BIT__855 {
($n:expr) => {
BIT_U32__985!($n)
};
}

macro_rules! REG_BIT__863 {
($n:expr) => {
BIT_U32__307!($n)
};
}

macro_rules! REG_BIT__868 {
($n:expr) => {
BIT_U32__839!($n)
};
}

macro_rules! REG_BIT__872 {
($n:expr) => {
BIT_U32__1014!($n)
};
}

macro_rules! REG_BIT__882 {
($n:expr) => {
BIT_U32__1352!($n)
};
}

macro_rules! REG_BIT__891 {
($n:expr) => {
BIT_U32__622!($n)
};
}

macro_rules! REG_BIT__911 {
($n:expr) => {
BIT_U32__351!($n)
};
}

macro_rules! REG_BIT__994 {
($n:expr) => {
BIT_U32__848!($n)
};
}

macro_rules! REG_BIT__1037 {
($n:expr) => {
BIT_U32__1333!($n)
};
}

macro_rules! REG_BIT__1060 {
($n:expr) => {
BIT_U32__408!($n)
};
}

macro_rules! REG_BIT__1067 {
($n:expr) => {
BIT_U32__447!($n)
};
}

macro_rules! REG_BIT__1074 {
($n:expr) => {
BIT_U32__430!($n)
};
}

macro_rules! REG_BIT__1087 {
($n:expr) => {
BIT_U32__855!($n)
};
}

macro_rules! REG_BIT__1117 {
($n:expr) => {
BIT_U32__426!($n)
};
}

macro_rules! REG_BIT__1121 {
($n:expr) => {
BIT_U32__550!($n)
};
}

macro_rules! REG_BIT__1165 {
($n:expr) => {
BIT_U32__569!($n)
};
}

macro_rules! REG_BIT__1166 {
($n:expr) => {
BIT_U32__557!($n)
};
}

macro_rules! REG_BIT__1192 {
($n:expr) => {
BIT_U32__1178!($n)
};
}

macro_rules! REG_BIT__1253 {
($n:expr) => {
BIT_U32__174!($n)
};
}

macro_rules! REG_BIT__1257 {
($n:expr) => {
BIT_U32__1424!($n)
};
}

macro_rules! REG_BIT__1270 {
($n:expr) => {
BIT_U32__1045!($n)
};
}

macro_rules! REG_BIT__1293 {
($n:expr) => {
BIT_U32__1149!($n)
};
}

macro_rules! REG_BIT__1316 {
($n:expr) => {
BIT_U32__627!($n)
};
}

macro_rules! REG_BIT__1335 {
($n:expr) => {
BIT_U32__586!($n)
};
}

macro_rules! REG_BIT__1341 {
($n:expr) => {
BIT_U32__507!($n)
};
}

macro_rules! REG_BIT__1350 {
($n:expr) => {
BIT_U32__1221!($n)
};
}

macro_rules! REG_BIT__1402 {
($n:expr) => {
BIT_U32__205!($n)
};
}

macro_rules! REG_BIT__1417 {
($n:expr) => {
BIT_U32__573!($n)
};
}

macro_rules! REG_BIT__1421 {
($n:expr) => {
BIT_U32__643!($n)
};
}

macro_rules! REG_BIT__1435 {
($n:expr) => {
BIT_U32__566!($n)
};
}

macro_rules! REG_BIT__1439 {
($n:expr) => {
BIT_U32__455!($n)
};
}

macro_rules! REG_GENMASK8__2 {
($high:expr, $low:expr) => {
GENMASK_U8__10!($high, $low)
};
}

macro_rules! REG_GENMASK8__3 {
($high:expr, $low:expr) => {
GENMASK_U8__25!($high, $low)
};
}

macro_rules! REG_GENMASK8__4 {
($high:expr, $low:expr) => {
GENMASK_U8__17!($high, $low)
};
}

macro_rules! REG_GENMASK8__6 {
($high:expr, $low:expr) => {
GENMASK_U8__16!($high, $low)
};
}

macro_rules! REG_GENMASK8__10 {
($high:expr, $low:expr) => {
GENMASK_U8__8!($high, $low)
};
}

macro_rules! REG_GENMASK8__11 {
($high:expr, $low:expr) => {
GENMASK_U8__11!($high, $low)
};
}

macro_rules! REG_GENMASK8__12 {
($high:expr, $low:expr) => {
GENMASK_U8__7!($high, $low)
};
}

macro_rules! REG_GENMASK8__15 {
($high:expr, $low:expr) => {
GENMASK_U8__22!($high, $low)
};
}

macro_rules! REG_GENMASK8__20 {
($high:expr, $low:expr) => {
GENMASK_U8__5!($high, $low)
};
}

macro_rules! REG_GENMASK8__23 {
($high:expr, $low:expr) => {
GENMASK_U8__2!($high, $low)
};
}

macro_rules! REG_GENMASK8__24 {
($high:expr, $low:expr) => {
GENMASK_U8__14!($high, $low)
};
}

macro_rules! REG_GENMASK8__25 {
($high:expr, $low:expr) => {
GENMASK_U8__6!($high, $low)
};
}

/// Wrappers over the generic fixed width BIT_U*() and GENMASK_U*()
/// implementations, for compatibility reasons with previous implementation.
macro_rules! REG_GENMASK__11 {
($high:expr, $low:expr) => {
GENMASK_U32__301!($high, $low)
};
}

/// Wrappers over the generic fixed width BIT_U*() and GENMASK_U*()
/// implementations, for compatibility reasons with previous implementation.
macro_rules! REG_GENMASK__112 {
($high:expr, $low:expr) => {
GENMASK_U32__171!($high, $low)
};
}

/// Wrappers over the generic fixed width BIT_U*() and GENMASK_U*()
/// implementations, for compatibility reasons with previous implementation.
macro_rules! REG_GENMASK__116 {
($high:expr, $low:expr) => {
GENMASK_U32__168!($high, $low)
};
}

/// Wrappers over the generic fixed width BIT_U*() and GENMASK_U*()
/// implementations, for compatibility reasons with previous implementation.
macro_rules! REG_GENMASK__128 {
($high:expr, $low:expr) => {
GENMASK_U32__24!($high, $low)
};
}

/// Wrappers over the generic fixed width BIT_U*() and GENMASK_U*()
/// implementations, for compatibility reasons with previous implementation.
macro_rules! REG_GENMASK__184 {
($high:expr, $low:expr) => {
GENMASK_U32__60!($high, $low)
};
}

/// Wrappers over the generic fixed width BIT_U*() and GENMASK_U*()
/// implementations, for compatibility reasons with previous implementation.
macro_rules! REG_GENMASK__215 {
($high:expr, $low:expr) => {
GENMASK_U32__133!($high, $low)
};
}

/// Wrappers over the generic fixed width BIT_U*() and GENMASK_U*()
/// implementations, for compatibility reasons with previous implementation.
macro_rules! REG_GENMASK__294 {
($high:expr, $low:expr) => {
GENMASK_U32__215!($high, $low)
};
}

/// Wrappers over the generic fixed width BIT_U*() and GENMASK_U*()
/// implementations, for compatibility reasons with previous implementation.
macro_rules! REG_GENMASK__300 {
($high:expr, $low:expr) => {
GENMASK_U32__135!($high, $low)
};
}

macro_rules! ULL {
() => {
_ULL!()
};
}

macro_rules! UL {
() => {
_UL!()
};
}

macro_rules! XELPDP_DDI_CLOCK_SELECT_MASK {
($display:expr) => {
if (DISPLAY_VER!($display) as i32) >= 30 { REG_GENMASK__128!(16i32, 12i32) } else { REG_GENMASK__294!(15i32, 12i32) }
};
}

macro_rules! XELPDP_DDI_CLOCK_SELECT_MASK__5 {
($display:expr) => {
if (DISPLAY_VER!($display) as i32) >= 30 { REG_GENMASK__11!(16i32, 12i32) } else { REG_GENMASK__215!(15i32, 12i32) }
};
}

macro_rules! XELPDP_DDI_CLOCK_SELECT_PREP {
($display:expr, $val:expr) => {
if (DISPLAY_VER!($display) as i32) >= 30 { ((($val) as u32) << (((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1224_2>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 12i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(16)) as u64).trailing_zeros() as i32) & (core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1224_3>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 12i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(16))).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1224_4>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1224_6>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1224_9>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1224_16>() as u32) } else { ((($val) as u32) << (((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1224_18>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 12i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)) as u64).trailing_zeros() as i32) & (core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1224_19>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 12i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15))).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1224_20>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1224_22>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1224_25>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1224_32>() as u32) }
};
}

macro_rules! XELPDP_DDI_CLOCK_SELECT_PREP__5 {
($display:expr, $val:expr) => {
if (DISPLAY_VER!($display) as i32) >= 30 { ((($val) as u32) << (((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1222_2>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 12i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(16)) as u64).trailing_zeros() as i32) & (core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1222_3>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 12i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(16))).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1222_4>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1222_6>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1222_9>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1222_16>() as u32) } else { ((($val) as u32) << (((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1222_18>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 12i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)) as u64).trailing_zeros() as i32) & (core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1222_19>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 12i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15))).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1222_20>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1222_22>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1222_25>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1222_32>() as u32) }
};
}

macro_rules! XELPDP_LANE_PCLK_PLL_ACK {
($lane:expr) => {
REG_BIT__1402!(30i32.wrapping_sub($lane.wrapping_mul(4)))
};
}

macro_rules! XELPDP_LANE_PCLK_PLL_ACK__3 {
($lane:expr) => {
REG_BIT__38!(30i32.wrapping_sub($lane.wrapping_mul(4)))
};
}

macro_rules! XELPDP_LANE_PCLK_PLL_ACK__4 {
($lane:expr) => {
REG_BIT__740!(30i32.wrapping_sub($lane.wrapping_mul(4)))
};
}

macro_rules! XELPDP_LANE_PCLK_PLL_ACK__5 {
($lane:expr) => {
REG_BIT__844!(30i32.wrapping_sub($lane.wrapping_mul(4)))
};
}

macro_rules! XELPDP_LANE_PCLK_PLL_ACK__6 {
($lane:expr) => {
REG_BIT__1067!(30i32.wrapping_sub($lane.wrapping_mul(4)))
};
}

macro_rules! XELPDP_LANE_PCLK_PLL_REQUEST {
($lane:expr) => {
REG_BIT__494!(31i32.wrapping_sub($lane.wrapping_mul(4)))
};
}

macro_rules! XELPDP_LANE_PCLK_PLL_REQUEST__2 {
($lane:expr) => {
REG_BIT__1316!(31i32.wrapping_sub($lane.wrapping_mul(4)))
};
}

macro_rules! XELPDP_LANE_PCLK_PLL_REQUEST__3 {
($lane:expr) => {
REG_BIT__831!(31i32.wrapping_sub($lane.wrapping_mul(4)))
};
}

macro_rules! XELPDP_LANE_PCLK_PLL_REQUEST__5 {
($lane:expr) => {
REG_BIT__872!(31i32.wrapping_sub($lane.wrapping_mul(4)))
};
}

macro_rules! XELPDP_LANE_PCLK_PLL_REQUEST__6 {
($lane:expr) => {
REG_BIT__115!(31i32.wrapping_sub($lane.wrapping_mul(4)))
};
}

macro_rules! XELPDP_LANE_PCLK_PLL_REQUEST__7 {
($lane:expr) => {
REG_BIT__178!(31i32.wrapping_sub($lane.wrapping_mul(4)))
};
}

macro_rules! XELPDP_PORT_M2P_ADDRESS__2 {
($val:expr) => {
((($val) as u32) << (((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1035_2>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 0i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(11)) as u64).trailing_zeros() as i32) & (core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1035_3>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 0i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(11))).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1035_4>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1035_6>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1035_9>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1035_16>() as u32)
};
}

macro_rules! XELPDP_PORT_M2P_DATA {
($val:expr) => {
((($val) as u32) << (((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1034_2>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 16i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(23)) as u64).trailing_zeros() as i32) & (core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1034_3>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 16i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(23))).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1034_4>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1034_6>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1034_9>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1034_16>() as u32)
};
}

macro_rules! _AC {
() => {
1u64
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__6 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1159_2>() as i32).wrapping_add(if (($__index) as i32) < $__c_index { _PICK_EVEN__2!($__index, $__a, $__b) } else { _PICK_EVEN!((($__index) as i32).wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__7 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1153_2>() as i32).wrapping_add(if (($__index) as i32) < $__c_index { _PICK_EVEN__2!($__index, $__a, $__b) } else { _PICK_EVEN!((($__index) as i32).wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__8 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1019>() as i32).wrapping_add(if ($__index) < $__c_index { _PICK_EVEN!($__index, $__a, $__b) } else { _PICK_EVEN!($__index.wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__14 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1163_2>() as i32).wrapping_add(if (($__index) as i32) < $__c_index { _PICK_EVEN__2!($__index, $__a, $__b) } else { _PICK_EVEN!((($__index) as i32).wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__16 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2098>() as i32).wrapping_add(if ($__index) < $__c_index { _PICK_EVEN!($__index, $__a, $__b) } else { _PICK_EVEN!($__index.wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__21 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1166>() as i32).wrapping_add(if ($__index) < $__c_index { _PICK_EVEN!($__index, $__a, $__b) } else { _PICK_EVEN!($__index.wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__23 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1019_2>() as i32).wrapping_add(if (($__index) as i32) < $__c_index { _PICK_EVEN__2!($__index, $__a, $__b) } else { _PICK_EVEN!((($__index) as i32).wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__33 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2219>() as i32).wrapping_add(if ($__index) < $__c_index { _PICK_EVEN!($__index, $__a, $__b) } else { _PICK_EVEN!($__index.wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__37 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1998>() as i32).wrapping_add(if ($__index) < $__c_index { _PICK_EVEN!($__index, $__a, $__b) } else { _PICK_EVEN!($__index.wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__44 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2071>() as i32).wrapping_add(if ($__index) < $__c_index { _PICK_EVEN!($__index, $__a, $__b) } else { _PICK_EVEN!($__index.wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__49 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2075>() as i32).wrapping_add(if ($__index) < $__c_index { _PICK_EVEN!($__index, $__a, $__b) } else { _PICK_EVEN!($__index.wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__50 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2003>() as i32).wrapping_add(if ($__index) < $__c_index { _PICK_EVEN!($__index, $__a, $__b) } else { _PICK_EVEN!($__index.wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__51 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1968>() as i32).wrapping_add(if ($__index) < $__c_index { _PICK_EVEN!($__index, $__a, $__b) } else { _PICK_EVEN!($__index.wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__58 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1149_2>() as i32).wrapping_add(if (($__index) as i32) < $__c_index { _PICK_EVEN__2!($__index, $__a, $__b) } else { _PICK_EVEN!((($__index) as i32).wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__67 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1029_2>() as i32).wrapping_add(if (($__index) as i32) < $__c_index { _PICK_EVEN__2!($__index, $__a, $__b) } else { _PICK_EVEN!((($__index) as i32).wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__68 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1153>() as i32).wrapping_add(if ($__index) < $__c_index { _PICK_EVEN!($__index, $__a, $__b) } else { _PICK_EVEN!($__index.wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__77 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1029>() as i32).wrapping_add(if ($__index) < $__c_index { _PICK_EVEN!($__index, $__a, $__b) } else { _PICK_EVEN!($__index.wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__78 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1178_2>() as i32).wrapping_add(if (($__index) as i32) < $__c_index { _PICK_EVEN__2!($__index, $__a, $__b) } else { _PICK_EVEN!((($__index) as i32).wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__82 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1031>() as i32).wrapping_add(if ($__index) < $__c_index { _PICK_EVEN!($__index, $__a, $__b) } else { _PICK_EVEN!($__index.wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__89 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1139>() as i32).wrapping_add(if ($__index) < $__c_index { _PICK_EVEN!($__index, $__a, $__b) } else { _PICK_EVEN!($__index.wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__96 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1988>() as i32).wrapping_add(if ($__index) < $__c_index { _PICK_EVEN!($__index, $__a, $__b) } else { _PICK_EVEN!($__index.wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__98 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2227>() as i32).wrapping_add(if ($__index) < $__c_index { _PICK_EVEN!($__index, $__a, $__b) } else { _PICK_EVEN!($__index.wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__112 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2091_2>() as i32).wrapping_add(if (($__index) as i32) < $__c_index { _PICK_EVEN__2!($__index, $__a, $__b) } else { _PICK_EVEN!((($__index) as i32).wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__113 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2109_2>() as i32).wrapping_add(if (($__index) as i32) < $__c_index { _PICK_EVEN__2!($__index, $__a, $__b) } else { _PICK_EVEN!((($__index) as i32).wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__122 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2071_2>() as i32).wrapping_add(if (($__index) as i32) < $__c_index { _PICK_EVEN__2!($__index, $__a, $__b) } else { _PICK_EVEN!((($__index) as i32).wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__126 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1210_2>() as i32).wrapping_add(if (($__index) as i32) < $__c_index { _PICK_EVEN__2!($__index, $__a, $__b) } else { _PICK_EVEN!((($__index) as i32).wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__129 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2023_2>() as i32).wrapping_add(if (($__index) as i32) < $__c_index { _PICK_EVEN__2!($__index, $__a, $__b) } else { _PICK_EVEN!((($__index) as i32).wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__130 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1163>() as i32).wrapping_add(if ($__index) < $__c_index { _PICK_EVEN!($__index, $__a, $__b) } else { _PICK_EVEN!($__index.wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__136 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2016_2>() as i32).wrapping_add(if (($__index) as i32) < $__c_index { _PICK_EVEN__2!($__index, $__a, $__b) } else { _PICK_EVEN!((($__index) as i32).wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__144 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2003_2>() as i32).wrapping_add(if (($__index) as i32) < $__c_index { _PICK_EVEN__2!($__index, $__a, $__b) } else { _PICK_EVEN!((($__index) as i32).wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__153 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2066>() as i32).wrapping_add(if ($__index) < $__c_index { _PICK_EVEN!($__index, $__a, $__b) } else { _PICK_EVEN!($__index.wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__160 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2082_2>() as i32).wrapping_add(if (($__index) as i32) < $__c_index { _PICK_EVEN__2!($__index, $__a, $__b) } else { _PICK_EVEN!((($__index) as i32).wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__162 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1983>() as i32).wrapping_add(if ($__index) < $__c_index { _PICK_EVEN!($__index, $__a, $__b) } else { _PICK_EVEN!($__index.wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__163 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_995>() as i32).wrapping_add(if ($__index) < $__c_index { _PICK_EVEN!($__index, $__a, $__b) } else { _PICK_EVEN!($__index.wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__165 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1031_2>() as i32).wrapping_add(if (($__index) as i32) < $__c_index { _PICK_EVEN__2!($__index, $__a, $__b) } else { _PICK_EVEN!((($__index) as i32).wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__172 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2075_2>() as i32).wrapping_add(if (($__index) as i32) < $__c_index { _PICK_EVEN__2!($__index, $__a, $__b) } else { _PICK_EVEN!((($__index) as i32).wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__178 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1172_2>() as i32).wrapping_add(if (($__index) as i32) < $__c_index { _PICK_EVEN__2!($__index, $__a, $__b) } else { _PICK_EVEN!((($__index) as i32).wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__180 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1228_2>() as i32).wrapping_add(if (($__index) as i32) < $__c_index { _PICK_EVEN__2!($__index, $__a, $__b) } else { _PICK_EVEN!((($__index) as i32).wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__186 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1139_2>() as i32).wrapping_add(if (($__index) as i32) < $__c_index { _PICK_EVEN__2!($__index, $__a, $__b) } else { _PICK_EVEN!((($__index) as i32).wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__193 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1964>() as i32).wrapping_add(if ($__index) < $__c_index { _PICK_EVEN!($__index, $__a, $__b) } else { _PICK_EVEN!($__index.wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__194 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2109>() as i32).wrapping_add(if ($__index) < $__c_index { _PICK_EVEN!($__index, $__a, $__b) } else { _PICK_EVEN!($__index.wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__196 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2113>() as i32).wrapping_add(if ($__index) < $__c_index { _PICK_EVEN!($__index, $__a, $__b) } else { _PICK_EVEN!($__index.wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__207 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1228>() as i32).wrapping_add(if ($__index) < $__c_index { _PICK_EVEN!($__index, $__a, $__b) } else { _PICK_EVEN!($__index.wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__208 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1210>() as i32).wrapping_add(if ($__index) < $__c_index { _PICK_EVEN!($__index, $__a, $__b) } else { _PICK_EVEN!($__index.wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__209 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1149>() as i32).wrapping_add(if ($__index) < $__c_index { _PICK_EVEN!($__index, $__a, $__b) } else { _PICK_EVEN!($__index.wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__210 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1166_2>() as i32).wrapping_add(if (($__index) as i32) < $__c_index { _PICK_EVEN__2!($__index, $__a, $__b) } else { _PICK_EVEN!((($__index) as i32).wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__213 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1964_2>() as i32).wrapping_add(if (($__index) as i32) < $__c_index { _PICK_EVEN__2!($__index, $__a, $__b) } else { _PICK_EVEN!((($__index) as i32).wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__225 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1968_2>() as i32).wrapping_add(if (($__index) as i32) < $__c_index { _PICK_EVEN__2!($__index, $__a, $__b) } else { _PICK_EVEN!((($__index) as i32).wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__234 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1983_2>() as i32).wrapping_add(if (($__index) as i32) < $__c_index { _PICK_EVEN__2!($__index, $__a, $__b) } else { _PICK_EVEN!((($__index) as i32).wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__236 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2098_2>() as i32).wrapping_add(if (($__index) as i32) < $__c_index { _PICK_EVEN__2!($__index, $__a, $__b) } else { _PICK_EVEN!((($__index) as i32).wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__239 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2082>() as i32).wrapping_add(if ($__index) < $__c_index { _PICK_EVEN!($__index, $__a, $__b) } else { _PICK_EVEN!($__index.wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__240 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1136>() as i32).wrapping_add(if ($__index) < $__c_index { _PICK_EVEN!($__index, $__a, $__b) } else { _PICK_EVEN!($__index.wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__250 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2091>() as i32).wrapping_add(if ($__index) < $__c_index { _PICK_EVEN!($__index, $__a, $__b) } else { _PICK_EVEN!($__index.wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__265 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2023>() as i32).wrapping_add(if ($__index) < $__c_index { _PICK_EVEN!($__index, $__a, $__b) } else { _PICK_EVEN!($__index.wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__268 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2219_2>() as i32).wrapping_add(if (($__index) as i32) < $__c_index { _PICK_EVEN__2!($__index, $__a, $__b) } else { _PICK_EVEN!((($__index) as i32).wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__285 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2066_2>() as i32).wrapping_add(if (($__index) as i32) < $__c_index { _PICK_EVEN__2!($__index, $__a, $__b) } else { _PICK_EVEN!((($__index) as i32).wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__291 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1178>() as i32).wrapping_add(if ($__index) < $__c_index { _PICK_EVEN!($__index, $__a, $__b) } else { _PICK_EVEN!($__index.wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__296 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1988_2>() as i32).wrapping_add(if (($__index) as i32) < $__c_index { _PICK_EVEN__2!($__index, $__a, $__b) } else { _PICK_EVEN!((($__index) as i32).wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__307 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1172>() as i32).wrapping_add(if ($__index) < $__c_index { _PICK_EVEN!($__index, $__a, $__b) } else { _PICK_EVEN!($__index.wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__311 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1159>() as i32).wrapping_add(if ($__index) < $__c_index { _PICK_EVEN!($__index, $__a, $__b) } else { _PICK_EVEN!($__index.wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__325 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2016>() as i32).wrapping_add(if ($__index) < $__c_index { _PICK_EVEN!($__index, $__a, $__b) } else { _PICK_EVEN!($__index.wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__329 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1998_2>() as i32).wrapping_add(if (($__index) as i32) < $__c_index { _PICK_EVEN__2!($__index, $__a, $__b) } else { _PICK_EVEN!((($__index) as i32).wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__330 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_2227_2>() as i32).wrapping_add(if (($__index) as i32) < $__c_index { _PICK_EVEN__2!($__index, $__a, $__b) } else { _PICK_EVEN!((($__index) as i32).wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Like _PICK_EVEN(), but supports 2 ranges of evenly spaced address offsets.
/// @__c_index corresponds to the index in which the second range starts to be
/// used. Using math interval notation, the first range is used for indexes [ 0,
/// @__c_index), while the second range is used for [ @__c_index, ... ). Example:
///
/// #define _FOO_A			0xf000
/// #define _FOO_B			0xf004
/// #define _FOO_C			0xf008
/// #define _SUPER_FOO_A			0xa000
/// #define _SUPER_FOO_B			0xa100
/// #define FOO(x)			_MMIO(_PICK_EVEN_2RANGES(x, 3,		 *					      _FOO_A, _FOO_B,			 *					      _SUPER_FOO_A, _SUPER_FOO_B))
///
/// This expands to:
/// 	0: 0xf000,
/// 	1: 0xf004,
/// 	2: 0xf008,
/// 	3: 0xa000,
/// 	4: 0xa100,
/// 	5: 0xa200,
/// 	...
macro_rules! _PICK_EVEN_2RANGES__335 {
($__index:expr, $__c_index:expr, $__a:expr, $__b:expr, $__c:expr, $__d:expr) => {
(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1146>() as i32).wrapping_add(if ($__index) < $__c_index { _PICK_EVEN!($__index, $__a, $__b) } else { _PICK_EVEN!($__index.wrapping_sub($__c_index), $__c, $__d) })
};
}

/// Given the first two numbers __a and __b of arbitrarily many evenly spaced
/// numbers, pick the 0-based __index'th value.
///
/// Always prefer this over _PICK() if the numbers are evenly spaced.
macro_rules! _PICK_EVEN {
($__index:expr, $__a:expr, $__b:expr) => {
$__a.wrapping_add($__index.wrapping_mul($__b.wrapping_sub($__a)))
};
}

/// Given the first two numbers __a and __b of arbitrarily many evenly spaced
/// numbers, pick the 0-based __index'th value.
///
/// Always prefer this over _PICK() if the numbers are evenly spaced.
macro_rules! _PICK_EVEN__2 {
($__index:expr, $__a:expr, $__b:expr) => {
$__a.wrapping_add((($__index) as i32).wrapping_mul($__b.wrapping_sub($__a)))
};
}

macro_rules! _PORT {
($port:expr, $a:expr, $b:expr) => {
_PICK_EVEN__2!($port, $a, $b)
};
}

macro_rules! _ULL {
() => {
_AC!()
};
}

macro_rules! _UL {
() => {
_AC!()
};
}

macro_rules! ____is_defined {
() => {
0i32
};
}

macro_rules! __bf_cast_unsigned__2 {
($x:expr) => {
($x) as u8
};
}

/// Conversion functions/macros from various pointer types to struct
/// intel_display pointer.
macro_rules! __drm_device_to_intel_display {
($p:expr) => {
{
let __v: *mut intel_display = if !$p.is_null() {
unsafe { __drm_to_display($p) }
} else {
core::ptr::null_mut::<intel_display>()
};
__v
}
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

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word {
() => {
((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__2 {
() => {
((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
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
macro_rules! __type_half_max__3 {
() => {
1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub((is_signed_type__4!() as i32) as u64)
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
1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((is_signed_type!() as i32) as u64)
};
}

macro_rules! __type_max {
() => {
__type_half_max!().wrapping_sub(1).wrapping_add(__type_half_max!())
};
}

macro_rules! __type_max__3 {
() => {
__type_half_max__3!().wrapping_sub(1).wrapping_add(__type_half_max__3!()) as u8
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

/// container_of - cast a member of a structure out to the containing structure
/// @ptr:	the pointer to the member.
/// @type:	the type of the container struct this is embedded in.
/// @member:	the name of the member within the struct.
///
/// WARNING: any const qualifier of @ptr is lost.
/// Do not use container_of() in new code.
macro_rules! container_of {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__4!($type) as isize)) }) as *mut ()) as *mut $type;
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
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__5!($type) as isize)) }) as *mut ()) as *mut $type;
__v
}
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
let mut __v: u32;
let __v2: u32 = ($base) as u32;
let __v3: u32 = ((($n) as u64) % (__v2 as u64)) as u32;
$n = (($n) as u64) / (__v2 as u64);
__v = __v3;
};
}

macro_rules! drm_debug_enabled {
($category:expr) => {
drm_debug_enabled_raw(($category) as u32)
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type__4 {
() => {
((1i32.wrapping_neg() as u8) as i32) < 1
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type {
() => {
(1i32.wrapping_neg() as u32) < 1
};
}

/// Subtract two ktime_t variables. rem = lhs -rhs:
macro_rules! ktime_sub {
($lhs:expr, $rhs:expr) => {
$lhs.wrapping_sub($rhs)
};
}

macro_rules! offsetof__4 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, dp) as u64
};
}

macro_rules! offsetof__6 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, base.base) as u64
};
}

macro_rules! offsetof__5 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, hdmi) as u64
};
}

macro_rules! phy_name {
($a:expr) => {
(($a) as i32).wrapping_add(65)
};
}

/// Convert various pointer types to struct intel_display pointer.
macro_rules! to_intel_display {
($p:expr) => {
__drm_device_to_intel_display!(unsafe { (*$p).base.dev })
};
}

/// Convert various pointer types to struct intel_display pointer.
macro_rules! to_intel_display__5 {
($p:expr) => {
__drm_device_to_intel_display!(unsafe { (*(*$p).uapi.crtc).dev })
};
}

macro_rules! type_max {
() => {
__type_max!()
};
}

macro_rules! type_max__2 {
() => {
__type_max__3!()
};
}

macro_rules! unlikely {
($x:expr) => {
(($x) != 0) as i64
};
}

macro_rules! unlikely__3 {
($x:expr) => {
(($x) != 0) as i64
};
}

unsafe extern "C" {
    static mut kmsg_fops: file_operations;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut __SCK__WARN_trap: static_call_key;
    static mut cpuinfo_op: seq_operations;
    static mut sighand_cachep: *mut kmem_cache;
    static mut __drm_debug: u64;
    static mut kstat: kernel_stat;
    fn __bad_size_call_parameter();
    fn __compiletime_assert_0();
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn irq_to_desc(arg0: u32) -> *mut irq_desc;
    fn _dev_err(arg0: *const device, arg1: *const u8, ...);
    fn _dev_warn(arg0: *const device, arg1: *const u8, ...);
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_388();
    fn __compiletime_assert_409();
    fn dev_driver_string(arg0: *const device) -> *const u8;
    fn drm_modeset_lock(arg0: *mut drm_modeset_lock, arg1: *mut drm_modeset_acquire_ctx) -> i32;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn __drm_printfn_dbg(arg0: *mut drm_printer, arg1: *mut va_format);
    fn drm_printf(arg0: *mut drm_printer, arg1: *const u8, ...);
    fn __drm_dev_dbg(arg0: *mut _ddebug, arg1: *const device, arg2: u32, arg3: *const u8, ...);
    fn intel_cx0_clear_response_ready_flag(arg0: *mut intel_encoder, arg1: i32);
    fn intel_readout_lane_count(arg0: *mut intel_encoder, arg1: i32, arg2: i32) -> i32;
    fn intel_mtl_tbt_pll_disable_clock(arg0: *mut intel_encoder);
    fn intel_cx0_powerdown_change_sequence(arg0: *mut intel_encoder, arg1: u8, arg2: u8);
    fn intel_cx0_setup_powerdown(arg0: *mut intel_encoder);
    fn intel_cx0_read(arg0: *mut intel_encoder, arg1: u8, arg2: u16) -> u8;
    fn intel_cx0_rmw(arg0: *mut intel_encoder, arg1: u8, arg2: u16, arg3: u8, arg4: u8, arg5: bool);
    fn intel_cx0_write(arg0: *mut intel_encoder, arg1: u8, arg2: u16, arg3: u8, arg4: bool);
    fn intel_cx0_wait_for_ack(arg0: *mut intel_encoder, arg1: i32, arg2: i32, arg3: *mut u32) -> i32;
    fn intel_cx0_bus_reset(arg0: *mut intel_encoder, arg1: i32);
    fn intel_ddi_level(arg0: *mut intel_encoder, arg1: *const intel_crtc_state, arg2: i32) -> i32;
    fn intel_ddi_link_symbol_clock(arg0: *mut intel_encoder, arg1: i32) -> i32;
    fn intel_ddi_buf_trans_get(arg0: *mut intel_encoder, arg1: *const intel_crtc_state, arg2: *mut i32) -> *const intel_ddi_buf_trans;
    fn intel_display_power_is_enabled(arg0: *mut intel_display, arg1: u32) -> bool;
    fn intel_display_power_get(arg0: *mut intel_display, arg1: u32) -> *mut ref_tracker;
    fn intel_display_power_put_unchecked(arg0: *mut intel_display, arg1: u32);
    fn intel_dmc_wl_get(arg0: *mut intel_display, arg1: __SlateRecord_i915_reg_defs_h_14);
    fn intel_dmc_wl_put(arg0: *mut intel_display, arg1: __SlateRecord_i915_reg_defs_h_14);
    fn to_intel_uncore(arg0: *mut drm_device) -> *mut intel_uncore;
    fn intel_de_wait_for_set_us(arg0: *mut intel_display, arg1: __SlateRecord_i915_reg_defs_h_14, arg2: u32, arg3: u32) -> i32;
    fn intel_de_wait_for_clear_us(arg0: *mut intel_display, arg1: __SlateRecord_i915_reg_defs_h_14, arg2: u32, arg3: u32) -> i32;
    fn intel_de_wait_for_set_ms(arg0: *mut intel_display, arg1: __SlateRecord_i915_reg_defs_h_14, arg2: u32, arg3: u32) -> i32;
    fn intel_de_wait_for_clear_ms(arg0: *mut intel_display, arg1: __SlateRecord_i915_reg_defs_h_14, arg2: u32, arg3: u32) -> i32;
    fn intel_encoder_to_phy(arg0: *mut intel_encoder) -> i32;
    fn __bad_udelay();
    fn __udelay(arg0: u64);
    fn __const_udelay(arg0: u64);
    fn drm_atomic_commit(arg0: *mut drm_atomic_commit) -> i32;
    fn drm_format_info(arg0: u32) -> *const drm_format_info;
    fn __drm_to_display(arg0: *mut drm_device) -> *mut intel_display;
    fn intel_dpll_clock_matches(arg0: i32, arg1: i32) -> bool;
    fn intel_hdmi_is_frl(arg0: u32) -> bool;
    fn intel_panel_use_ssc(arg0: *mut intel_display) -> bool;
    fn intel_psr_pause(arg0: *mut intel_dp);
    fn intel_psr_resume(arg0: *mut intel_dp);
    fn intel_tc_port_in_tbt_alt_mode(arg0: *mut intel_digital_port) -> bool;
    fn intel_tc_port_in_dp_alt_mode(arg0: *mut intel_digital_port) -> bool;
    fn intel_tc_port_get_pin_assignment(arg0: *mut intel_digital_port) -> u32;
    fn intel_tc_port_max_lane_count(arg0: *mut intel_digital_port) -> i32;
    fn __compiletime_assert_701();
    fn __compiletime_assert_702();
    fn __compiletime_assert_703();
    fn __compiletime_assert_704();
    fn __compiletime_assert_705();
    fn __compiletime_assert_706();
    fn __compiletime_assert_707();
    fn __compiletime_assert_708();
    fn __compiletime_assert_709();
    fn __compiletime_assert_710();
    fn __compiletime_assert_712();
    fn __compiletime_assert_713();
    fn __compiletime_assert_714();
    fn __compiletime_assert_715();
    fn __compiletime_assert_716();
    fn __compiletime_assert_719();
    fn __compiletime_assert_720();
    fn __compiletime_assert_721();
    fn __compiletime_assert_722();
    fn __compiletime_assert_723();
    fn __compiletime_assert_724();
    fn __compiletime_assert_725();
    fn __compiletime_assert_726();
    fn __compiletime_assert_727();
    fn __compiletime_assert_728();
}

unsafe extern "llvm-intrinsic" {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct lockdep_map {
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
struct va_format {
    fmt: *const u8,
    va: *mut core::ffi::VaList<'static>,
}


// Some subsystems have their own custom printk that applies a va_format to a
// generic format, for example, to include a device number or other metadata
// alongside the format supplied by the caller.
//
// In order to store these in the way they would be emitted by the printk
// infrastructure, the subsystem provides us with the start, fixed string, and
// any subsequent text in the format string.
//
// We take a variable argument list as pr_fmt/dev_fmt/etc are sometimes passed
// as multiple arguments (eg: `"%s: ", "blah"`), and we must only take the
// first one.
//
// subsys_fmt_prefix must be known at compile time, or compilation will fail
// (since this is a mistake). If fmt or level is not known at compile time, no
// index entry will be made (since this can legitimately happen).
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
struct bug_entry {
    bug_addr_disp: i32,
    format_disp: i32,
    file_disp: i32,
    line: u16,
    flags: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_bitmap {
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


// 4th level page in 5-level paging case
// 3rd level page
// PMD_SHIFT determines the size of the area a middle-level
// page table can map
// entries per page directory level
// We borrow bit 3 to remember PG_anon_exclusive.
#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_static_call_types_h_65 {
    r#type: u64,
    mods: *mut static_call_mod,
    sites: *mut static_call_site,
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


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits120U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct static_key {
    enabled: __SlateRecord_types_h_190,
    __slate_anon_1: __SlateRecord_jump_label_h_95,
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
struct static_key_false {
    key: static_key,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct orc_entry {
    sp_offset: i16,
    bp_offset: i16,
    __slate_bits_0: __slate_bits::__SlateBits127U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cpumask {
    bits: [u64; 1],
}


// cpumask_bits - get the bits in a cpumask
// @maskp: the struct cpumask *
//
// You should only assume nr_cpu_ids bits of this mask are valid.  This is
// a macro so it's const-correct.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_nodemask_types_h_17 {
    bits: [u64; 1],
}


fn div_u64_rem(dividend: u64, divisor: u32, remainder: &mut u32) -> u64 {
unsafe {
*(remainder as *mut u32) = (dividend % (divisor as u64)) as u32;
}
dividend / (divisor as u64)
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_jump_label_h_95 {
    r#type: u64,
    entries: *mut jump_entry,
    next: *mut static_key_mod,
}


fn div64_u64(dividend: u64, divisor: u64) -> u64 {
dividend / divisor
}

#[repr(C)]
#[derive(Clone, Copy)]
struct static_key_mod {
}


fn div_u64(dividend: u64, divisor: u32) -> u64 {
let mut remainder: u32 = 0;
div_u64_rem(dividend, divisor, unsafe { &mut (*core::ptr::addr_of_mut!(remainder)) })
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
struct perf_event {
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
struct pollfd {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_mm_data {
    phash: futex_mm_phash,
    unlock: futex_unlock_cs_ranges,
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
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
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
struct mutex_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct plist_head {
    node_list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_pi_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct robust_list_head {
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
struct futex_private_hash {
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
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
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
struct sem_undo_list {
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
struct hrtimer_clock_base {
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
struct seccomp_filter {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_work {
    node: __call_single_node,
    func: Option<unsafe extern "C" fn(*mut irq_work)>,
    irqwait: rcuwait,
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
struct workqueue_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
union unwind_task_id {
    __slate_anon_0: __SlateRecord_unwind_deferred_types_h_28,
    id: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
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
    __slate_bits_0: __slate_bits::__SlateBits286U0,
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
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
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
// Per-process atomic flags.
// May not gain new privileges.
// Spread page cache over cpuset
// Speculative Store Bypass disabled
// Speculative Store Bypass force disabled
// Indirect branch speculation restricted
// Indirect branch speculation permanently restricted
// Speculative Store Bypass clear on execve()
#[repr(C)]
#[derive(Clone, Copy)]
struct sched_cache_time {
    runtime: u64,
    epoch: u64,
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
struct rcu_gp_seq {
    norm: u64,
    exp: u64,
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
#[repr(C)]
#[derive(Clone, Copy)]
struct rq {
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
struct kref {
    refcount: refcount_struct,
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
struct rq_flags {
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
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
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
struct notifier_block {
    notifier_call: Option<unsafe extern "C" fn(*mut notifier_block, u64, *mut ()) -> i32>,
    next: *mut notifier_block,
    priority: i32,
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
struct blocking_notifier_head {
    rwsem: rw_semaphore,
    head: *mut notifier_block,
}


// Suits me
// Don't call further
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
struct gendisk {
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
struct vm_struct {
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
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_41 {
    f: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
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
struct uprobe {
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
    __slate_bits_0: __slate_bits::__SlateBits413U0,
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
struct uprobe_xol_ops {
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
struct ida {
    xa: xarray,
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
struct kernfs_elem_dir {
    subdirs: u64,
    children: rb_root,
    root: *mut kernfs_root,
    rev: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
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
struct ldt_struct {
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
    __slate_bits_0: __slate_bits::__SlateBits465U0,
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
struct mem_cgroup {
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
struct attribute {
    name: *const u8,
    mode: u16,
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
    __slate_bits_0: __slate_bits::__SlateBits467U0,
}


fn kobject_name(kobj: *const kobject) -> *const u8 {
unsafe { (*kobj).name }
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
struct kset {
    list: list_head,
    list_lock: spinlock,
    kobj: kobject,
    uevent_ops: *const kset_uevent_ops,
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
struct resource {
    start: u64,
    end: u64,
    name: *const u8,
    flags: u64,
    desc: u64,
    parent: *mut resource,
    sibling: *mut resource,
    child: *mut resource,
}


// Resource type
// PCI/ISA I/O ports
// Resource extended types
// Userland may not map this resource
// No address assigned yet
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
    __slate_bits_0: __slate_bits::__SlateBits492U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits492U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits492U2,
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
    __slate_bits_3: __slate_bits::__SlateBits492U3,
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
struct rhash_head {
    next: *mut rhash_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct poll_table_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_context {
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
struct iovec {
    iov_base: *mut (),
    iov_len: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
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


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
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
struct sock {
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
struct __SlateRecord_uuid_h_17 {
    b: [u8; 16],
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
struct exception_table_entry {
    insn: i32,
    fixup: i32,
    data: i32,
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
struct wake_irq {
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
    __slate_bits_0: __slate_bits::__SlateBits289U0,
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
fn task_rlimit(task: *const task_struct, limit: u32) -> u64 {
let mut __v0: u64;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_388() };
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
struct folio_queue {
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
struct percpu_rw_semaphore {
    rss: rcu_sync,
    read_count: *mut u32,
    writer: rcuwait,
    waiters: wait_queue_head,
    block: __SlateRecord_types_h_190,
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
struct kmem_cache {
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
struct mnt_idmap {
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
struct assoc_array_ptr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct net {
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
struct key_type {
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
struct lockref {
    __slate_anon_0: __SlateRecord_lockref_h_26,
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
struct tty_struct {
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
struct tty_audit_buf {
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
if !(__native_word!() || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_409() };
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
    __slate_bits_0: __slate_bits::__SlateBits638U0,
    rcu: callback_head,
    r#ref: *mut percpu_ref,
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
struct percpu_ref {
    percpu_count_ptr: u64,
    data: *mut percpu_ref_data,
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
struct offset_ctx {
    mt: maple_tree,
    next_offset: u64,
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
struct kparam_string {
    maxlen: u32,
    string: *mut u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct request_queue {
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
struct _ddebug {
    modname: *const u8,
    function: *const u8,
    filename: *const u8,
    format: *const u8,
    __slate_bits_0: __slate_bits::__SlateBits695U0,
    key: __SlateRecord_dynamic_debug_h_55,
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
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
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
struct posix_acl {
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
struct device_attribute {
    attr: attribute,
    __slate_anon_1: __SlateRecord_device_h_108,
    __slate_anon_2: __SlateRecord_device_h_114,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_cmd {
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
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
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
    __slate_bits_0: __slate_bits::__SlateBits489U0,
}


fn dev_name(dev: *const device) -> *const u8 {
// Use the init name until the kobject becomes available
if !unsafe { (*dev).init_name }.is_null() {
return unsafe { (*dev).init_name };
}
kobject_name(unsafe { core::ptr::addr_of!((*dev).kobj) })
}

fn dev_get_drvdata(dev: *const device) -> *mut () {
unsafe { (*dev).driver_data }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct drm_clip_rect {
    x1: u16,
    y1: u16,
    x2: u16,
    y2: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_dynamic_debug_h_55 {
    dd_key_true: static_key_true,
    dd_key_false: static_key_false,
}


// Link Status options
// extended-types: rather than continue to consume a bit per type,
// grab a chunk of the bits to use as integer type id.
#[repr(C)]
#[derive(Clone, Copy)]
struct drm_mode_fb_cmd2 {
    fb_id: u32,
    width: u32,
    height: u32,
    pixel_format: u32,
    flags: u32,
    handles: [u32; 4],
    pitches: [u32; 4],
    offsets: [u32; 4],
    modifier: [u64; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rt_mutex_base {
    wait_lock: raw_spinlock,
    waiters: rb_root_cached,
    owner: *mut task_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rt_mutex {
    rtmutex: rt_mutex_base,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct module_param_attrs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ww_mutex {
    base: mutex,
    ctx: *mut ww_acquire_ctx,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ww_acquire_ctx {
    task: *mut task_struct,
    stamp: u64,
    acquired: u32,
    wounded: u16,
    is_wait_die: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_modeset_acquire_ctx {
    ww_ctx: ww_acquire_ctx,
    contended: *mut drm_modeset_lock,
    stack_depot: u32,
    locked: list_head,
    trylock_only: bool,
    interruptible: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_modeset_lock {
    mutex: ww_mutex,
    head: list_head,
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
struct drm_mode_config_funcs {
    fb_create: Option<unsafe extern "C" fn(*mut drm_device, *mut drm_file, *const drm_format_info, *const drm_mode_fb_cmd2) -> *mut drm_framebuffer>,
    get_format_info: Option<unsafe extern "C" fn(u32, u64) -> *const drm_format_info>,
    mode_valid: Option<unsafe extern "C" fn(*mut drm_device, *const drm_display_mode) -> i32>,
    atomic_check: Option<unsafe extern "C" fn(*mut drm_device, *mut drm_atomic_commit) -> i32>,
    atomic_commit: Option<unsafe extern "C" fn(*mut drm_device, *mut drm_atomic_commit, bool) -> i32>,
    atomic_state_alloc: Option<unsafe extern "C" fn(*mut drm_device) -> *mut drm_atomic_commit>,
    atomic_state_clear: Option<unsafe extern "C" fn(*mut drm_atomic_commit)>,
    atomic_state_free: Option<unsafe extern "C" fn(*mut drm_atomic_commit)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_mode_config {
    mutex: mutex,
    connection_mutex: drm_modeset_lock,
    acquire_ctx: *mut drm_modeset_acquire_ctx,
    idr_mutex: mutex,
    object_idr: idr,
    tile_idr: idr,
    fb_lock: mutex,
    num_fb: i32,
    fb_list: list_head,
    connector_list_lock: spinlock,
    num_connector: i32,
    connector_ida: ida,
    connector_list: list_head,
    connector_free_list: llist_head,
    connector_free_work: work_struct,
    num_encoder: i32,
    encoder_list: list_head,
    num_total_plane: i32,
    plane_list: list_head,
    panic_lock: raw_spinlock,
    num_colorop: i32,
    colorop_list: list_head,
    num_crtc: i32,
    crtc_list: list_head,
    property_list: list_head,
    privobj_list: list_head,
    min_width: u32,
    min_height: u32,
    max_width: u32,
    max_height: u32,
    funcs: *const drm_mode_config_funcs,
    poll_enabled: bool,
    poll_running: bool,
    delayed_event: bool,
    output_poll_work: delayed_work,
    blob_lock: mutex,
    property_blob_list: list_head,
    edid_property: *mut drm_property,
    dpms_property: *mut drm_property,
    path_property: *mut drm_property,
    tile_property: *mut drm_property,
    panel_type_property: *mut drm_property,
    link_status_property: *mut drm_property,
    plane_type_property: *mut drm_property,
    prop_src_x: *mut drm_property,
    prop_src_y: *mut drm_property,
    prop_src_w: *mut drm_property,
    prop_src_h: *mut drm_property,
    prop_crtc_x: *mut drm_property,
    prop_crtc_y: *mut drm_property,
    prop_crtc_w: *mut drm_property,
    prop_crtc_h: *mut drm_property,
    prop_fb_id: *mut drm_property,
    prop_in_fence_fd: *mut drm_property,
    prop_out_fence_ptr: *mut drm_property,
    prop_crtc_id: *mut drm_property,
    prop_fb_damage_clips: *mut drm_property,
    prop_active: *mut drm_property,
    prop_mode_id: *mut drm_property,
    prop_vrr_enabled: *mut drm_property,
    dvi_i_subconnector_property: *mut drm_property,
    dvi_i_select_subconnector_property: *mut drm_property,
    dp_subconnector_property: *mut drm_property,
    tv_subconnector_property: *mut drm_property,
    tv_select_subconnector_property: *mut drm_property,
    legacy_tv_mode_property: *mut drm_property,
    tv_mode_property: *mut drm_property,
    tv_left_margin_property: *mut drm_property,
    tv_right_margin_property: *mut drm_property,
    tv_top_margin_property: *mut drm_property,
    tv_bottom_margin_property: *mut drm_property,
    tv_brightness_property: *mut drm_property,
    tv_contrast_property: *mut drm_property,
    tv_flicker_reduction_property: *mut drm_property,
    tv_overscan_property: *mut drm_property,
    tv_saturation_property: *mut drm_property,
    tv_hue_property: *mut drm_property,
    scaling_mode_property: *mut drm_property,
    aspect_ratio_property: *mut drm_property,
    content_type_property: *mut drm_property,
    degamma_lut_property: *mut drm_property,
    degamma_lut_size_property: *mut drm_property,
    ctm_property: *mut drm_property,
    gamma_lut_property: *mut drm_property,
    gamma_lut_size_property: *mut drm_property,
    background_color_property: *mut drm_property,
    suggested_x_property: *mut drm_property,
    suggested_y_property: *mut drm_property,
    non_desktop_property: *mut drm_property,
    panel_orientation_property: *mut drm_property,
    writeback_fb_id_property: *mut drm_property,
    writeback_pixel_formats_property: *mut drm_property,
    writeback_out_fence_ptr_property: *mut drm_property,
    hdr_output_metadata_property: *mut drm_property,
    content_protection_property: *mut drm_property,
    hdcp_content_type_property: *mut drm_property,
    preferred_depth: u32,
    prefer_shadow: u32,
    quirk_addfb_prefer_xbgr_30bpp: bool,
    quirk_addfb_prefer_host_byte_order: bool,
    async_page_flip: bool,
    fb_modifiers_not_supported: bool,
    normalize_zpos: bool,
    modifiers_property: *mut drm_property,
    async_modifiers_property: *mut drm_property,
    size_hints_property: *mut drm_property,
    cursor_width: u32,
    cursor_height: u32,
    suspend_state: *mut drm_atomic_commit,
    helper_private: *const drm_mode_config_helper_funcs,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_device {
    if_version: i32,
    r#ref: kref,
    dev: *mut device,
    dma_dev: *mut device,
    managed: __SlateRecord_drm_device_h_113,
    driver: *const drm_driver,
    dev_private: *mut (),
    primary: *mut drm_minor,
    render: *mut drm_minor,
    accel: *mut drm_minor,
    registered: bool,
    master: *mut drm_master,
    driver_features: u32,
    unplugged: bool,
    anon_inode: *mut inode,
    unique: *mut u8,
    master_mutex: mutex,
    open_count: __SlateRecord_types_h_190,
    filelist_mutex: mutex,
    filelist: list_head,
    filelist_internal: list_head,
    clientlist_mutex: mutex,
    clientlist: list_head,
    client_sysrq_list: list_head,
    vblank_disable_immediate: bool,
    vblank: *mut drm_vblank_crtc,
    vblank_time_lock: spinlock,
    vbl_lock: spinlock,
    max_vblank_count: u32,
    vblank_event_list: list_head,
    event_lock: spinlock,
    num_crtcs: u32,
    mode_config: drm_mode_config,
    object_name_lock: mutex,
    object_name_idr: idr,
    vma_offset_manager: *mut drm_vma_offset_manager,
    vram_mm: *mut drm_vram_mm,
    switch_power_state: u32,
    fb_helper: *mut drm_fb_helper,
    debugfs_root: *mut dentry,
    gem_lru_mutex: mutex,
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


fn drm_debug_enabled_raw(category: u32) -> bool {
unlikely__3!((unsafe { __drm_debug }) & BIT__3!(category)) != 0
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
struct drm_printer {
    printfn: Option<unsafe extern "C" fn(*mut drm_printer, *mut va_format)>,
    puts: Option<unsafe extern "C" fn(*mut drm_printer, *const u8)>,
    arg: *mut (),
    origin: *const (),
    prefix: *const u8,
    line: __SlateRecord_drm_print_h_181,
    category: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_device_h_108 {
    show: Option<unsafe extern "C" fn(*mut device, *mut device_attribute, *mut u8) -> i64>,
    show_const: Option<unsafe extern "C" fn(*mut device, *const device_attribute, *mut u8) -> i64>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_device_h_114 {
    store: Option<unsafe extern "C" fn(*mut device, *mut device_attribute, *const u8, u64) -> i64>,
    store_const: Option<unsafe extern "C" fn(*mut device, *const device_attribute, *const u8, u64) -> i64>,
}


fn drm_dbg_printer(drm: *mut drm_device, category: u32, prefix: *const u8) -> drm_printer {
let mut __v0: u64;
let mut __here: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("lea 0(%rip), {0:r}", lateout(reg) __t0, options(pure, nomem, nostack, att_syntax));
__here = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
__v0 = __here;
let p: drm_printer = {
let mut __t1: drm_printer = unsafe { core::mem::zeroed() };
__t1.printfn = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut drm_printer, *mut va_format)>>(__drm_printfn_dbg as *const ()) };
__t1.arg = drm as *mut ();
__t1.origin = __v0 as *const ();
__t1.prefix = prefix;
__t1.category = category;
__t1
};
// it's fine as we will be inlined
return p;
}

fn __drm_to_dev(drm: *const drm_device) -> *mut device {
if !drm.is_null() { unsafe { (*drm).dev } } else { core::ptr::null_mut::<device>() }
}

// Helper for struct drm_device based logging.
// Helper for struct drm_device based WARNs
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_i915_reg_defs_h_14 {
    reg: u32,
}


// These macros can be used on either i915_reg_t or i915_mcr_reg_t since they're
// simply operations on the register's offset and don't care about the MCR vs
// non-MCR nature of the register.
#[repr(C)]
#[derive(Clone, Copy)]
struct hsw_ddi_buf_trans {
    trans1: u32,
    trans2: u32,
    i_boost: u8,
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
struct bxt_ddi_buf_trans {
    margin: u8,
    scale: u8,
    enable: u8,
    deemphasis: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct icl_ddi_buf_trans {
    dw2_swing_sel: u8,
    dw7_n_scalar: u8,
    dw4_cursor_coeff: u8,
    dw4_post_cursor_2: u8,
    dw4_post_cursor_1: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct icl_mg_phy_ddi_buf_trans {
    cri_txdeemph_override_11_6: u8,
    cri_txdeemph_override_5_0: u8,
    cri_txdeemph_override_17_12: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tgl_dkl_phy_ddi_buf_trans {
    vswing: u8,
    preshoot: u8,
    de_emphasis: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dg2_snps_phy_buf_trans {
    vswing: u8,
    pre_cursor: u8,
    post_cursor: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xe3plpd_lt_phy_buf_trans {
    txswing: u8,
    txswing_level: u8,
    pre_cursor: u8,
    main_cursor: u8,
    post_cursor: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union intel_ddi_buf_trans_entry {
    hsw: hsw_ddi_buf_trans,
    bxt: bxt_ddi_buf_trans,
    icl: icl_ddi_buf_trans,
    mg: icl_mg_phy_ddi_buf_trans,
    dkl: tgl_dkl_phy_ddi_buf_trans,
    snps: dg2_snps_phy_buf_trans,
    lt: xe3plpd_lt_phy_buf_trans,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_ddi_buf_trans {
    entries: *const intel_ddi_buf_trans_entry,
    num_entries: u8,
    hdmi_default_entry: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdmi_any_infoframe {
    r#type: u32,
    version: u8,
    length: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdmi_avi_infoframe {
    r#type: u32,
    version: u8,
    length: u8,
    itc: bool,
    pixel_repeat: u8,
    colorspace: u32,
    scan_mode: u32,
    colorimetry: u32,
    picture_aspect: u32,
    active_aspect: u32,
    extended_colorimetry: u32,
    quantization_range: u32,
    nups: u32,
    video_code: u8,
    ycc_quantization_range: u32,
    content_type: u32,
    top_bar: u16,
    bottom_bar: u16,
    left_bar: u16,
    right_bar: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdmi_drm_infoframe {
    r#type: u32,
    version: u8,
    length: u8,
    eotf: u32,
    metadata_type: u32,
    display_primaries: [__SlateRecord_hdmi_h_214; 3],
    white_point: __SlateRecord_hdmi_h_217,
    max_display_mastering_luminance: u16,
    min_display_mastering_luminance: u16,
    max_cll: u16,
    max_fall: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_file {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdmi_spd_infoframe {
    r#type: u32,
    version: u8,
    length: u8,
    vendor: [u8; 8],
    product: [u8; 16],
    sdi: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdmi_audio_infoframe {
    r#type: u32,
    version: u8,
    length: u8,
    channels: u8,
    coding_type: u32,
    sample_size: u32,
    sample_frequency: u32,
    coding_type_ext: u32,
    channel_allocation: u8,
    level_shift_value: u8,
    downmix_inhibit: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdmi_vendor_infoframe {
    r#type: u32,
    version: u8,
    length: u8,
    oui: u32,
    vic: u8,
    s3d_struct: i32,
    s3d_ext_data: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdr_static_metadata {
    eotf: u8,
    metadata_type: u8,
    max_cll: u16,
    max_fall: u16,
    min_cll: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdr_sink_metadata {
    metadata_type: u32,
    __slate_anon_1: __SlateRecord_hdmi_h_406,
}


#[repr(C)]
#[derive(Clone, Copy)]
union hdmi_vendor_any_infoframe {
    any: __SlateRecord_hdmi_h_419,
    hdmi: hdmi_vendor_infoframe,
}


#[repr(C)]
#[derive(Clone, Copy)]
union hdmi_infoframe {
    any: hdmi_any_infoframe,
    avi: hdmi_avi_infoframe,
    spd: hdmi_spd_infoframe,
    vendor: hdmi_vendor_any_infoframe,
    audio: hdmi_audio_infoframe,
    drm: hdmi_drm_infoframe,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_mode_config_helper_funcs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_driver {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_minor {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_master {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_vblank_crtc {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_vma_offset_manager {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_vram_mm {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_fb_helper {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_mode_object {
    id: u32,
    r#type: u32,
    properties: *mut drm_object_properties,
    refcount: kref,
    free_cb: Option<unsafe extern "C" fn(*mut kref)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_device_h_113 {
    resources: list_head,
    final_kfree: *mut (),
    lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_object_properties {
    count: i32,
    properties: [*mut drm_property; 64],
    values: [u64; 64],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_print_h_181 {
    series: u32,
    counter: u32,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct irqaction {
    handler: Option<unsafe extern "C" fn(i32, *mut ()) -> u32>,
    __slate_anon_1: __SlateRecord_interrupt_h_125,
    affinity: *const cpumask,
    next: *mut irqaction,
    thread_fn: Option<unsafe extern "C" fn(i32, *mut ()) -> u32>,
    thread: *mut task_struct,
    secondary: *mut irqaction,
    irq: u32,
    flags: u32,
    thread_flags: u64,
    thread_mask: u64,
    name: *const u8,
    dir: *mut proc_dir_entry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_affinity_notify {
    irq: u32,
    kref: kref,
    work: work_struct,
    notify: Option<unsafe extern "C" fn(*mut irq_affinity_notify, *const cpumask)>,
    release: Option<unsafe extern "C" fn(*mut kref)>,
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
// Mask used for checking in page_pool_page_is_pp() below. page->pp_magic is
// OR'ed with PP_SIGNATURE after the allocation in order to preserve bit 0 for
// the head page of compound page and bit 1 for pfmemalloc page, as well as the
// bits used for the DMA index. page_is_pfmemalloc() is checked in
// __page_pool_put_page() to avoid recycling the pfmemalloc page.
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
struct intel_bios_encoder_data {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_property {
    head: list_head,
    base: drm_mode_object,
    flags: u32,
    name: [u8; 32],
    num_values: u32,
    values: *mut u64,
    dev: *mut drm_device,
    enum_list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_property_blob {
    base: drm_mode_object,
    dev: *mut drm_device,
    head_global: list_head,
    head_file: list_head,
    length: u64,
    data: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_scrambling {
    supported: bool,
    low_rates: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_scdc {
    supported: bool,
    read_request: bool,
    scrambling: drm_scrambling,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_hdmi_dsc_cap {
    v_1p2: bool,
    native_420: bool,
    all_bpp: bool,
    bpc_supported: u8,
    max_slices: u8,
    clk_per_slice: i32,
    max_lanes: u8,
    max_frl_rate_per_lane: u8,
    total_chunk_kbytes: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_hdmi_info {
    scdc: drm_scdc,
    y420_vdb_modes: [u64; 4],
    y420_cmdb_modes: [u64; 4],
    y420_dc_modes: u8,
    max_frl_rate_per_lane: u8,
    max_lanes: u8,
    dsc_cap: drm_hdmi_dsc_cap,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_monitor_range_info {
    min_vfreq: u16,
    max_vfreq: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_luminance_range_info {
    min_luminance: u32,
    max_luminance: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_amd_vsdb_info {
    version: u8,
    replay_mode: bool,
    panel_type: u8,
    luminance_range1: drm_luminance_range_info,
    luminance_range2: drm_luminance_range_info,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_display_info {
    width_mm: u32,
    height_mm: u32,
    bpc: u32,
    subpixel_order: u32,
    panel_orientation: i32,
    color_formats: u32,
    bus_formats: *const u32,
    num_bus_formats: u32,
    bus_flags: u32,
    max_tmds_clock: i32,
    dvi_dual: bool,
    is_hdmi: bool,
    has_audio: bool,
    has_hdmi_infoframe: bool,
    rgb_quant_range_selectable: bool,
    edid_hdmi_rgb444_dc_modes: u8,
    edid_hdmi_ycbcr444_dc_modes: u8,
    cea_rev: u8,
    hdmi: drm_hdmi_info,
    hdr_sink_metadata: hdr_sink_metadata,
    non_desktop: bool,
    monitor_range: drm_monitor_range_info,
    luminance_range: drm_luminance_range_info,
    mso_stream_count: u8,
    mso_pixel_overlap: u8,
    max_dsc_bpp: u32,
    vics: *mut u8,
    vics_len: i32,
    quirks: u32,
    source_physical_address: u16,
    amd_vsdb: drm_amd_vsdb_info,
    panel_type: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_tv_margins {
    bottom: u32,
    left: u32,
    right: u32,
    top: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_tv_connector_state {
    select_subconnector: u32,
    subconnector: u32,
    margins: drm_connector_tv_margins,
    legacy_mode: u32,
    mode: u32,
    brightness: u32,
    contrast: u32,
    flicker_reduction: u32,
    overscan: u32,
    saturation: u32,
    hue: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_hdmi_infoframe {
    data: hdmi_infoframe,
    set: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_hdmi_state {
    broadcast_rgb: u32,
    infoframes: __SlateRecord_drm_connector_h_1088,
    is_limited_range: bool,
    output_bpc: u32,
    output_format: u32,
    tmds_char_rate: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_hdmi_h_214 {
    x: u16,
    y: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_hdmi_h_217 {
    x: u16,
    y: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_state {
    connector: *mut drm_connector,
    crtc: *mut drm_crtc,
    best_encoder: *mut drm_encoder,
    link_status: u32,
    state: *mut drm_atomic_commit,
    commit: *mut drm_crtc_commit,
    tv: drm_tv_connector_state,
    self_refresh_aware: bool,
    picture_aspect_ratio: u32,
    content_type: u32,
    hdcp_content_type: u32,
    scaling_mode: u32,
    content_protection: u32,
    colorspace: u32,
    color_format: u32,
    writeback_job: *mut drm_writeback_job,
    max_requested_bpc: u8,
    max_bpc: u8,
    privacy_screen_sw_state: u32,
    hdr_output_metadata: *mut drm_property_blob,
    hdmi: drm_connector_hdmi_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_hdmi_audio_funcs {
    startup: Option<unsafe extern "C" fn(*mut drm_connector) -> i32>,
    prepare: Option<unsafe extern "C" fn(*mut drm_connector, *mut hdmi_codec_daifmt, *mut hdmi_codec_params) -> i32>,
    shutdown: Option<unsafe extern "C" fn(*mut drm_connector)>,
    mute_stream: Option<unsafe extern "C" fn(*mut drm_connector, bool, i32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_cec_funcs {
    phys_addr_invalidate: Option<unsafe extern "C" fn(*mut drm_connector)>,
    phys_addr_set: Option<unsafe extern "C" fn(*mut drm_connector, u16)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_infoframe_funcs {
    clear_infoframe: Option<unsafe extern "C" fn(*mut drm_connector) -> i32>,
    write_infoframe: Option<unsafe extern "C" fn(*mut drm_connector, *const u8, u64) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_hdmi_funcs {
    tmds_char_rate_valid: Option<unsafe extern "C" fn(*const drm_connector, *const drm_display_mode, u64) -> i32>,
    read_edid: Option<unsafe extern "C" fn(*mut drm_connector) -> *const drm_edid>,
    avi: drm_connector_infoframe_funcs,
    hdmi: drm_connector_infoframe_funcs,
    audio: drm_connector_infoframe_funcs,
    hdr_drm: drm_connector_infoframe_funcs,
    spd: drm_connector_infoframe_funcs,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_hdmi_h_406 {
    hdmi_type1: hdr_static_metadata,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_funcs {
    dpms: Option<unsafe extern "C" fn(*mut drm_connector, i32) -> i32>,
    reset: Option<unsafe extern "C" fn(*mut drm_connector)>,
    detect: Option<unsafe extern "C" fn(*mut drm_connector, bool) -> u32>,
    force: Option<unsafe extern "C" fn(*mut drm_connector)>,
    fill_modes: Option<unsafe extern "C" fn(*mut drm_connector, u32, u32) -> i32>,
    set_property: Option<unsafe extern "C" fn(*mut drm_connector, *mut drm_property, u64) -> i32>,
    late_register: Option<unsafe extern "C" fn(*mut drm_connector) -> i32>,
    early_unregister: Option<unsafe extern "C" fn(*mut drm_connector)>,
    destroy: Option<unsafe extern "C" fn(*mut drm_connector)>,
    atomic_create_state: Option<unsafe extern "C" fn(*mut drm_connector) -> *mut drm_connector_state>,
    atomic_duplicate_state: Option<unsafe extern "C" fn(*mut drm_connector) -> *mut drm_connector_state>,
    atomic_destroy_state: Option<unsafe extern "C" fn(*mut drm_connector, *mut drm_connector_state)>,
    atomic_set_property: Option<unsafe extern "C" fn(*mut drm_connector, *mut drm_connector_state, *mut drm_property, u64) -> i32>,
    atomic_get_property: Option<unsafe extern "C" fn(*mut drm_connector, *const drm_connector_state, *mut drm_property, *mut u64) -> i32>,
    atomic_print_state: Option<unsafe extern "C" fn(*mut drm_printer, *const drm_connector_state)>,
    oob_hotplug_event: Option<unsafe extern "C" fn(*mut drm_connector, u32)>,
    debugfs_init: Option<unsafe extern "C" fn(*mut drm_connector, *mut dentry)>,
    color_format: Option<unsafe extern "C" fn(*const drm_connector_state) -> u32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_hdmi_h_419 {
    r#type: u32,
    version: u8,
    length: u8,
    oui: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_cmdline_mode {
    name: [u8; 32],
    specified: bool,
    refresh_specified: bool,
    bpp_specified: bool,
    pixel_clock: u32,
    xres: i32,
    yres: i32,
    bpp: i32,
    refresh: i32,
    rb: bool,
    interlace: bool,
    cvt: bool,
    margins: bool,
    force: u32,
    rotation_reflection: u32,
    panel_orientation: i32,
    tv_margins: drm_connector_tv_margins,
    tv_mode: u32,
    tv_mode_specified: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_hdmi_audio {
    funcs: *const drm_connector_hdmi_audio_funcs,
    codec_pdev: *mut platform_device,
    lock: mutex,
    plugged_cb: Option<unsafe extern "C" fn(*mut device, bool)>,
    plugged_cb_dev: *mut device,
    last_state: bool,
    dai_port: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_hdmi {
    vendor: [u8; 8],
    product: [u8; 16],
    supported_formats: u64,
    funcs: *const drm_connector_hdmi_funcs,
    infoframes: __SlateRecord_drm_connector_h_2062,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_cec {
    mutex: mutex,
    funcs: *const drm_connector_cec_funcs,
    data: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_interrupt_h_125 {
    dev_id: *mut (),
    percpu_dev_id: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct proc_dir_entry {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector {
    dev: *mut drm_device,
    kdev: *mut device,
    attr: *mut device_attribute,
    fwnode: *mut fwnode_handle,
    head: list_head,
    global_connector_list_entry: list_head,
    base: drm_mode_object,
    name: *mut u8,
    mutex: mutex,
    index: u32,
    connector_type: i32,
    connector_type_id: i32,
    interlace_allowed: bool,
    doublescan_allowed: bool,
    stereo_allowed: bool,
    ycbcr_420_allowed: bool,
    registration_state: u32,
    modes: list_head,
    status: u32,
    probed_modes: list_head,
    display_info: drm_display_info,
    funcs: *const drm_connector_funcs,
    edid_blob_ptr: *mut drm_property_blob,
    properties: drm_object_properties,
    scaling_mode_property: *mut drm_property,
    vrr_capable_property: *mut drm_property,
    colorspace_property: *mut drm_property,
    color_format_property: *mut drm_property,
    path_blob_ptr: *mut drm_property_blob,
    max_bpc: u32,
    max_bpc_property: *mut drm_property,
    privacy_screen: *mut drm_privacy_screen,
    privacy_screen_notifier: notifier_block,
    privacy_screen_sw_state_property: *mut drm_property,
    privacy_screen_hw_state_property: *mut drm_property,
    broadcast_rgb_property: *mut drm_property,
    polled: u8,
    dpms: i32,
    helper_private: *const drm_connector_helper_funcs,
    cmdline_mode: drm_cmdline_mode,
    force: u32,
    edid_override: *const drm_edid,
    edid_override_mutex: mutex,
    epoch_counter: u64,
    possible_encoders: u32,
    encoder: *mut drm_encoder,
    eld: [u8; 128],
    eld_mutex: mutex,
    latency_present: [bool; 2],
    video_latency: [i32; 2],
    audio_latency: [i32; 2],
    ddc: *mut i2c_adapter,
    null_edid_counter: i32,
    bad_edid_counter: u32,
    edid_corrupt: bool,
    real_edid_checksum: u8,
    debugfs_entry: *mut dentry,
    state: *mut drm_connector_state,
    tile_blob_ptr: *mut drm_property_blob,
    has_tile: bool,
    tile_group: *mut drm_tile_group,
    tile_is_single_monitor: bool,
    num_h_tile: u8,
    num_v_tile: u8,
    tile_h_loc: u8,
    tile_v_loc: u8,
    tile_h_size: u16,
    tile_v_size: u16,
    free_node: llist_node,
    hdmi: drm_connector_hdmi,
    hdmi_audio: drm_connector_hdmi_audio,
    cec: drm_connector_cec,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_tile_group {
    refcount: kref,
    dev: *mut drm_device,
    id: i32,
    group_data: [u8; 9],
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
struct intel_cdclk_config {
    cdclk: u32,
    vco: u32,
    r#ref: u32,
    bypass: u32,
    voltage_level: u8,
    joined_mbus: bool,
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
struct mnt_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uts_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ipc_namespace {
}


// Platform group aliases
// g45 and gm45
// mobile platforms
// discrete graphics
// Display ver 2
// Display ver 3
// Display ver 4
// Display ver 5
// Display ver 6
// Display ver 7
// Display ver 8
// Display ver 9
// Display ver 11
// Display ver 12
// Display ver 13
// Display ver 14 (based on GMD ID)
// Display ver 20 (based on GMD ID)
// Display ver 14.1 (based on GMD ID)
// Display ver 30 (based on GMD ID)
// Display ver 35 (based on GMD ID)
#[repr(C)]
#[derive(Clone, Copy)]
struct intel_display_platforms {
    __slate_anon_0: __SlateRecord_intel_display_device_h_117,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct time_namespace {
}


// Keep in alphabetical order
#[repr(C)]
#[derive(Clone, Copy)]
struct intel_display_runtime_info {
    ip: intel_display_ip_ver,
    step: i32,
    step_name: [u8; 3],
    rawclk_freq: u32,
    pipe_mask: u8,
    cpu_transcoder_mask: u16,
    port_mask: u16,
    num_sprites: [u8; 4],
    num_scalers: [u8; 4],
    fbc_mask: u8,
    has_hdcp: bool,
    has_dmc: bool,
    has_dsc: bool,
    edp_typec_support: bool,
    has_dbuf_overlap_detection: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_display_device_info {
    __runtime_defaults: intel_display_runtime_info,
    abox_mask: u8,
    dbuf: __SlateRecord_intel_display_device_h_320,
    __slate_bits_0: __slate_bits::__SlateBits932U0,
    mmio_offset: u32,
    pipe_offsets: [u32; 9],
    trans_offsets: [u32; 9],
    cursor_offsets: [u32; 4],
    color: __SlateRecord_intel_display_device_h_337,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_display_params {
    dmc_firmware_path: *mut u8,
    vbt_firmware: *mut u8,
    lvds_channel_mode: i32,
    panel_use_ssc: i32,
    vbt_sdvo_panel_type: i32,
    enable_dc: i32,
    enable_dpt: bool,
    enable_dsb: bool,
    enable_flipq: bool,
    enable_sagv: bool,
    disable_power_well: i32,
    enable_ips: bool,
    invert_brightness: i32,
    edp_vswing: i32,
    enable_dpcd_backlight: i32,
    load_detect_test: bool,
    force_reset_modeset_test: bool,
    disable_display: bool,
    verbose_state_checks: bool,
    nuclear_pageflip: bool,
    enable_dp_mst: bool,
    enable_fbc: i32,
    enable_psr: i32,
    enable_dc_balance: bool,
    enable_panel_replay: i32,
    psr_safest_params: bool,
    enable_psr2_sel_fetch: bool,
    enable_dmc_wl: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct proc_ns_operations {
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct __SlateRecord_ns_common_types_h_112 {
    __ns_ref: refcount_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_pin {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_power_domain_mask {
    bits: [u64; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_connector_helper_funcs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dc3co_state {
    lock: mutex,
    allowed: bool,
    trigger: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i915_power_domains {
    initializing: bool,
    display_core_suspended: bool,
    dc3co_to_dc6: bool,
    power_well_count: i32,
    dc_state: u32,
    target_dc_state: u32,
    allowed_dc_mask: u32,
    init_wakeref: *mut ref_tracker,
    disable_wakeref: *mut ref_tracker,
    lock: mutex,
    domain_use_count: [i32; 76],
    async_put_work: delayed_work,
    async_put_wakeref: *mut ref_tracker,
    async_put_domains: [intel_power_domain_mask; 2],
    async_put_next_delay: i32,
    power_wells: *mut i915_power_well,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_privacy_screen {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_edid {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdmi_codec_daifmt {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdmi_codec_params {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_display_power_domain_set {
    mask: intel_power_domain_mask,
}


fn intel_display_power_put(display: *mut intel_display, domain: u32, wakeref: *mut ref_tracker) {
unsafe { intel_display_power_put_unchecked(display, domain) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dmc_wl {
    lock: spinlock,
    enabled: bool,
    taken: bool,
    refcount: refcount_struct,
    dc_state: u32,
    work: delayed_work,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i9xx_dpll_hw_state {
    dpll: u32,
    dpll_md: u32,
    fp0: u32,
    fp1: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hsw_dpll_hw_state {
    wrpll: u32,
    spll: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct skl_dpll_hw_state {
    ctrl1: u32,
    cfgcr1: u32,
    cfgcr2: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bxt_dpll_hw_state {
    ebb0: u32,
    ebb4: u32,
    pll0: u32,
    pll1: u32,
    pll2: u32,
    pll3: u32,
    pll6: u32,
    pll8: u32,
    pll9: u32,
    pll10: u32,
    pcsdw12: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct icl_dpll_hw_state {
    cfgcr0: u32,
    cfgcr1: u32,
    div0: u32,
    mg_refclkin_ctl: u32,
    mg_clktop2_coreclkctl1: u32,
    mg_clktop2_hsclkctl: u32,
    mg_pll_div0: u32,
    mg_pll_div1: u32,
    mg_pll_lf: u32,
    mg_pll_frac_lock: u32,
    mg_pll_ssc: u32,
    mg_pll_bias: u32,
    mg_pll_tdc_coldst_bias: u32,
    mg_pll_bias_mask: u32,
    mg_pll_tdc_coldst_bias_mask: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_mpllb_state {
    clock: u32,
    ref_control: u32,
    mpllb_cp: u32,
    mpllb_div: u32,
    mpllb_div2: u32,
    mpllb_fracn1: u32,
    mpllb_fracn2: u32,
    mpllb_sscen: u32,
    mpllb_sscstep: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_c10pll_state {
    tx: u8,
    cmn: u8,
    pll: [u8; 20],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_c20pll_state {
    tx: [u16; 3],
    cmn: [u16; 4],
    __slate_anon_2: __SlateRecord_intel_dpll_mgr_h_252,
    vdr: intel_c20pll_vdr_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_cx0pll_state {
    __slate_anon_0: __SlateRecord_intel_dpll_mgr_h_264,
    lane_count: i32,
    ssc_enabled: bool,
    use_c10: bool,
    tbt_mode: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_lt_phy_pll_state {
    addr_msb: [u8; 13],
    addr_lsb: [u8; 13],
    data: [[u8; 4]; 13],
    config: [u8; 3],
    ssc_enabled: bool,
    tbt_mode: bool,
    lane_count: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_connector_h_1088 {
    avi: drm_connector_hdmi_infoframe,
    hdr_drm: drm_connector_hdmi_infoframe,
    spd: drm_connector_hdmi_infoframe,
    hdmi: drm_connector_hdmi_infoframe,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dpll_hw_state {
    __slate_anon_0: __SlateRecord_intel_dpll_mgr_h_285,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dpll_state {
    pipe_mask: u8,
    hw_state: intel_dpll_hw_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_writeback_job {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dpll_info {
    name: *const u8,
    funcs: *const intel_dpll_funcs,
    id: i32,
    power_domain: u32,
    always_on: bool,
    is_alt_port_dpll: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dpll {
    state: intel_dpll_state,
    index: u8,
    active_mask: u8,
    on: bool,
    info: *const dpll_info,
    wakeref: *mut ref_tracker,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_global_state_funcs {
    atomic_duplicate_state: Option<unsafe extern "C" fn(*mut intel_global_obj) -> *mut intel_global_state>,
    atomic_destroy_state: Option<unsafe extern "C" fn(*mut intel_global_obj, *mut intel_global_state)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_global_obj {
    head: list_head,
    state: *mut intel_global_state,
    funcs: *const intel_global_state_funcs,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_global_state {
    obj: *mut intel_global_obj,
    state: *mut intel_atomic_state,
    commit: *mut intel_global_commit,
    r#ref: kref,
    changed: bool,
    serialized: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_device_id {
    vendor: u32,
    device: u32,
    subvendor: u32,
    subdevice: u32,
    class: u32,
    class_mask: u32,
    driver_data: u64,
    override_only: u32,
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
struct platform_device {
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
struct __SlateRecord_drm_connector_h_2062 {
    lock: mutex,
    audio: drm_connector_hdmi_infoframe,
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
#[repr(C)]
#[derive(Clone, Copy)]
struct pci_slot {
    bus: *mut pci_bus,
    list: list_head,
    hotplug: *mut hotplug_slot,
    number: u16,
    __slate_bits_0: __slate_bits::__SlateBits974U0,
    kobj: kobject,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_display_parent_interface {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_vpd {
    lock: mutex,
    len: u32,
    cap: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_intel_display_device_h_117 {
    __slate_anon_0: __SlateRecord_intel_display_device_h_118,
    bitmap: [u64; 2],
}


#[repr(C, align(8))]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_device_h_118 {
    __slate_bits_0: __slate_bits::__SlateBits929U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_dev {
    bus_list: list_head,
    bus: *mut pci_bus,
    subordinate: *mut pci_bus,
    sysdata: *mut (),
    procent: *mut proc_dir_entry,
    slot: *mut pci_slot,
    devfn: u32,
    vendor: u16,
    device: u16,
    subsystem_vendor: u16,
    subsystem_device: u16,
    class: u32,
    revision: u8,
    hdr_type: u8,
    rcec_ea: *mut rcec_ea,
    rcec: *mut pci_dev,
    devcap: u32,
    rebar_cap: u16,
    pcie_cap: u8,
    msi_cap: u8,
    msix_cap: u8,
    __slate_bits_0: __slate_bits::__SlateBits772U0,
    rom_base_reg: u8,
    pin: u8,
    pcie_flags_reg: u16,
    dma_alias_mask: *mut u64,
    driver: *mut pci_driver,
    dma_mask: u64,
    msi_addr_mask: u64,
    dma_parms: device_dma_parameters,
    current_state: i32,
    pm_cap: u8,
    __slate_bits_1: __slate_bits::__SlateBits772U1,
    d3hot_delay: u32,
    d3cold_delay: u32,
    l1ss: u16,
    link_state: *mut pcie_link_state,
    __slate_bits_2: __slate_bits::__SlateBits772U2,
    error_state: u32,
    dev: device,
    cfg_size: i32,
    irq: u32,
    resource: [resource; 11],
    driver_exclusive_resource: resource,
    __slate_bits_3: __slate_bits::__SlateBits772U3,
    dev_flags: u16,
    enable_cnt: __SlateRecord_types_h_190,
    pcie_cap_lock: spinlock,
    saved_config_space: [u32; 16],
    saved_cap_space: hlist_head,
    msix_base: *mut (),
    msi_lock: raw_spinlock,
    vpd: pci_vpd,
    link_bwctrl: *mut pcie_bwctrl_data,
    __slate_anon_116: __SlateRecord_pci_h_547,
    ats_cap: u16,
    ats_stu: u8,
    pri_cap: u16,
    pri_reqs_alloc: u32,
    __slate_bits_4: __slate_bits::__SlateBits772U4,
    pasid_cap: u16,
    pasid_features: u16,
    acs_cap: u16,
    acs_capabilities: u16,
    supported_speeds: u8,
    rom: u64,
    romlen: u64,
    priv_flags: u64,
    reset_methods: [u8; 8],
    wake: *mut gpio_desc,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_display_ip_ver {
    ver: u16,
    rel: u16,
    step: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_bus {
    node: list_head,
    parent: *mut pci_bus,
    children: list_head,
    devices: list_head,
    self_: *mut pci_dev,
    slots: list_head,
    resource: [*mut resource; 4],
    resources: list_head,
    busn_res: resource,
    ops: *mut pci_ops,
    sysdata: *mut (),
    procdir: *mut proc_dir_entry,
    number: u8,
    primary: u8,
    max_bus_speed: u8,
    cur_bus_speed: u8,
    name: [u8; 48],
    bridge_ctl: u16,
    bus_flags: u16,
    bridge: *mut device,
    dev: device,
    __slate_bits_0: __slate_bits::__SlateBits443U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_device_h_320 {
    size: u16,
    slice_mask: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_device_h_337 {
    degamma_lut_size: u32,
    gamma_lut_size: u32,
    degamma_lut_tests: u32,
    gamma_lut_tests: u32,
}


// Error values that may be returned by PCI functions
#[repr(C)]
#[derive(Clone, Copy)]
struct pci_ops {
    add_bus: Option<unsafe extern "C" fn(*mut pci_bus) -> i32>,
    remove_bus: Option<unsafe extern "C" fn(*mut pci_bus)>,
    map_bus: Option<unsafe extern "C" fn(*mut pci_bus, u32, i32) -> *mut ()>,
    read: Option<unsafe extern "C" fn(*mut pci_bus, u32, i32, i32, *mut u32) -> i32>,
    write: Option<unsafe extern "C" fn(*mut pci_bus, u32, i32, i32, u32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i915_power_well {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ref_tracker {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_dynids {
    lock: spinlock,
    list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_error_handlers {
    error_detected: Option<unsafe extern "C" fn(*mut pci_dev, u32) -> u32>,
    mmio_enabled: Option<unsafe extern "C" fn(*mut pci_dev) -> u32>,
    slot_reset: Option<unsafe extern "C" fn(*mut pci_dev) -> u32>,
    reset_prepare: Option<unsafe extern "C" fn(*mut pci_dev)>,
    reset_done: Option<unsafe extern "C" fn(*mut pci_dev)>,
    resume: Option<unsafe extern "C" fn(*mut pci_dev)>,
    cor_error_detected: Option<unsafe extern "C" fn(*mut pci_dev)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_driver {
    name: *const u8,
    id_table: *const pci_device_id,
    probe: Option<unsafe extern "C" fn(*mut pci_dev, *const pci_device_id) -> i32>,
    remove: Option<unsafe extern "C" fn(*mut pci_dev)>,
    suspend: Option<unsafe extern "C" fn(*mut pci_dev, pm_message) -> i32>,
    resume: Option<unsafe extern "C" fn(*mut pci_dev) -> i32>,
    shutdown: Option<unsafe extern "C" fn(*mut pci_dev)>,
    sriov_configure: Option<unsafe extern "C" fn(*mut pci_dev, i32) -> i32>,
    sriov_set_msix_vec_count: Option<unsafe extern "C" fn(*mut pci_dev, i32) -> i32>,
    sriov_get_vf_total_msix: Option<unsafe extern "C" fn(*mut pci_dev) -> u32>,
    err_handler: *const pci_error_handlers,
    groups: *mut *const attribute_group,
    dev_groups: *mut *const attribute_group,
    driver: device_driver,
    dynids: pci_dynids,
    driver_managed_dma: bool,
}


// Allow INTx interrupts
// Allow MSI interrupts
// Allow MSI-X interrupts
// These helpers provide future and backwards compatibility
// for accessing popular PCI BAR info
fn pci_get_drvdata(pdev: &mut pci_dev) -> *mut () {
dev_get_drvdata((unsafe { core::ptr::addr_of_mut!((*(pdev as *mut pci_dev)).dev) }) as *const device)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ilk_wm_values {
    wm_pipe: [u32; 3],
    wm_lp: [u32; 3],
    wm_lp_spr: [u32; 3],
    enable_fbc_wm: bool,
    partitioning: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dpll_funcs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct g4x_pipe_wm {
    plane: [u16; 8],
    fbc: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct g4x_sr_wm {
    plane: u16,
    cursor: u16,
    fbc: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vlv_wm_ddl_values {
    plane: [u8; 8],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vlv_wm_values {
    pipe: [g4x_pipe_wm; 3],
    sr: g4x_sr_wm,
    ddl: [vlv_wm_ddl_values; 3],
    level: u8,
    cxsr: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct g4x_wm_values {
    pipe: [g4x_pipe_wm; 2],
    sr: g4x_sr_wm,
    hpll: g4x_sr_wm,
    cxsr: bool,
    hpll_en: bool,
    fbc_en: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct skl_ddb_entry {
    start: u16,
    end: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_intel_dpll_mgr_h_252 {
    mplla: [u16; 10],
    mpllb: [u16; 11],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_c20pll_vdr_state {
    custom_width: u8,
    serdes_rate: u8,
    hdmi_rate: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_intel_dpll_mgr_h_264 {
    c10: intel_c10pll_state,
    c20: intel_c20pll_state,
}


// Amount of PSF GV points, BSpec precisely defines this
#[repr(C)]
#[derive(Clone, Copy)]
struct intel_modeset_funcs {
    get_pipe_config: Option<unsafe extern "C" fn(*mut intel_crtc, *mut intel_crtc_state) -> bool>,
    get_initial_plane_config: Option<unsafe extern "C" fn(*mut intel_crtc, *mut intel_initial_plane_config)>,
    fixup_initial_plane_config: Option<unsafe extern "C" fn(*mut intel_crtc, *const intel_initial_plane_config) -> bool>,
    crtc_enable: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_crtc)>,
    crtc_disable: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_crtc)>,
    commit_modeset_enables: Option<unsafe extern "C" fn(*mut intel_atomic_state)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_intel_dpll_mgr_h_285 {
    i9xx: i9xx_dpll_hw_state,
    hsw: hsw_dpll_hw_state,
    skl: skl_dpll_hw_state,
    bxt: bxt_dpll_hw_state,
    icl: icl_dpll_hw_state,
    mpllb: intel_mpllb_state,
    cx0pll: intel_cx0pll_state,
    ltpll: intel_lt_phy_pll_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_wm_funcs {
    update_wm: Option<unsafe extern "C" fn(*mut intel_display)>,
    compute_watermarks: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_crtc) -> i32>,
    initial_watermarks: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_crtc)>,
    atomic_update_watermarks: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_crtc)>,
    optimize_watermarks: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_crtc)>,
    compute_global_watermarks: Option<unsafe extern "C" fn(*mut intel_atomic_state) -> i32>,
    get_hw_state: Option<unsafe extern "C" fn(*mut intel_display)>,
    sanitize: Option<unsafe extern "C" fn(*mut intel_display)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_audio_state {
    encoder: *mut intel_encoder,
    eld: [u8; 128],
    needs_cpu_transcoder_id: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dsb {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_fbc {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_global_commit {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_audio {
    funcs: *const intel_audio_funcs,
    component: *mut i915_audio_component,
    component_registered: bool,
    mutex: mutex,
    power_refcount: i32,
    freq_cntrl: u32,
    state: [intel_audio_state; 9],
    lpe: __SlateRecord_intel_display_core_h_120,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dpll_global {
    funcs: *const intel_dpll_global_funcs,
    lock: mutex,
    num_dpll: i32,
    dplls: [intel_dpll; 9],
    mgr: *const intel_dpll_mgr,
    ref_clks: __SlateRecord_intel_display_core_h_141,
    pch_ssc_use: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_frontbuffer_tracking {
    lock: spinlock,
    busy_bits: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_hotplug {
    funcs: *const intel_hotplug_irq_funcs,
    hotplug_work: delayed_work,
    hpd: *const u32,
    pch_hpd: *const u32,
    stats: [__SlateRecord_intel_display_core_h_170; 15],
    event_bits: u32,
    retry_bits: u32,
    reenable_work: delayed_work,
    long_hpd_pin_mask: u32,
    short_hpd_pin_mask: u32,
    dig_port_work: work_struct,
    poll_init_work: work_struct,
    poll_enabled: bool,
    detection_work_enabled: bool,
    hpd_storm_threshold: u32,
    hpd_short_storm_enabled: u8,
    oob_hotplug_last_state: u64,
    dp_wq: *mut workqueue_struct,
    ignore_long_hpd: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_vbt_data {
    version: u16,
    __slate_bits_0: __slate_bits::__SlateBits1029U0,
    lvds_ssc_freq: i32,
    orientation: i32,
    override_afc_startup: bool,
    override_afc_startup_val: u8,
    crt_ddc_pin: i32,
    display_devices: list_head,
    bdb_blocks: list_head,
    sdvo_mappings: [sdvo_device_mapping; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_wm {
    funcs: *const intel_wm_funcs,
    pri_latency: [u16; 5],
    spr_latency: [u16; 5],
    cur_latency: [u16; 5],
    skl_latency: [u16; 8],
    __slate_anon_5: __SlateRecord_intel_display_core_h_280,
    num_levels: u8,
    wm_mutex: mutex,
    ipc_enabled: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_display {
    drm: *mut drm_device,
    platform: intel_display_platforms,
    pch_type: i32,
    parent: *const intel_display_parent_interface,
    pipe_list: list_head,
    access: __SlateRecord_intel_display_core_h_314,
    backlight: __SlateRecord_intel_display_core_h_319,
    bw: __SlateRecord_intel_display_core_h_324,
    cdclk: __SlateRecord_intel_display_core_h_340,
    color: __SlateRecord_intel_display_core_h_357,
    dbuf: __SlateRecord_intel_display_core_h_364,
    dbuf_bw: __SlateRecord_intel_display_core_h_371,
    dkl: __SlateRecord_intel_display_core_h_375,
    dmc: __SlateRecord_intel_display_core_h_383,
    dsi: __SlateRecord_intel_display_core_h_388,
    dram: __SlateRecord_intel_display_core_h_393,
    fbc: __SlateRecord_intel_display_core_h_397,
    fbdev: __SlateRecord_intel_display_core_h_408,
    fdi: __SlateRecord_intel_display_core_h_413,
    global: __SlateRecord_intel_display_core_h_421,
    gmbus: __SlateRecord_intel_display_core_h_425,
    hdcp: __SlateRecord_intel_display_core_h_443,
    hti: __SlateRecord_intel_display_core_h_457,
    info: __SlateRecord_intel_display_core_h_467,
    ips: __SlateRecord_intel_display_core_h_475,
    irq: __SlateRecord_intel_display_core_h_479,
    modeset: __SlateRecord_intel_display_core_h_521,
    pkgc: __SlateRecord_intel_display_core_h_526,
    pmdemand: __SlateRecord_intel_display_core_h_532,
    power: __SlateRecord_intel_display_core_h_541,
    pps: __SlateRecord_intel_display_core_h_553,
    properties: __SlateRecord_intel_display_core_h_560,
    quirks: __SlateRecord_intel_display_core_h_565,
    reset: __SlateRecord_intel_display_core_h_569,
    restore: __SlateRecord_intel_display_core_h_573,
    sagv: __SlateRecord_intel_display_core_h_585,
    sbi: __SlateRecord_intel_display_core_h_596,
    snps: __SlateRecord_intel_display_core_h_601,
    state: __SlateRecord_intel_display_core_h_609,
    vlv_clock: __SlateRecord_intel_display_core_h_619,
    wq: __SlateRecord_intel_display_core_h_624,
    dp_tunnel_mgr: *mut drm_dp_tunnel_mgr,
    audio: intel_audio,
    dpll: intel_dpll_global,
    fb_tracking: intel_frontbuffer_tracking,
    hotplug: intel_hotplug,
    opregion: *mut intel_opregion,
    overlay: *mut intel_overlay,
    params: intel_display_params,
    vbt: intel_vbt_data,
    wl: intel_dmc_wl,
    wm: intel_wm,
    psr_dc5_dc6_wa_work: work_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_uncore_mmio_debug {
    lock: spinlock,
    unclaimed_mmio_check: i32,
    saved_mmio_check: i32,
    suspend_count: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hotplug_slot {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_uncore_fw_get {
    force_wake_get: Option<unsafe extern "C" fn(*mut intel_uncore, u32)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pcie_bwctrl_data {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pcie_link_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_sriov {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rcec_ea {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_uncore_funcs {
    read_fw_domains: Option<unsafe extern "C" fn(*mut intel_uncore, __SlateRecord_i915_reg_defs_h_14) -> u32>,
    write_fw_domains: Option<unsafe extern "C" fn(*mut intel_uncore, __SlateRecord_i915_reg_defs_h_14) -> u32>,
    mmio_readb: Option<unsafe extern "C" fn(*mut intel_uncore, __SlateRecord_i915_reg_defs_h_14, bool) -> u8>,
    mmio_readw: Option<unsafe extern "C" fn(*mut intel_uncore, __SlateRecord_i915_reg_defs_h_14, bool) -> u16>,
    mmio_readl: Option<unsafe extern "C" fn(*mut intel_uncore, __SlateRecord_i915_reg_defs_h_14, bool) -> u32>,
    mmio_readq: Option<unsafe extern "C" fn(*mut intel_uncore, __SlateRecord_i915_reg_defs_h_14, bool) -> u64>,
    mmio_writeb: Option<unsafe extern "C" fn(*mut intel_uncore, __SlateRecord_i915_reg_defs_h_14, u8, bool)>,
    mmio_writew: Option<unsafe extern "C" fn(*mut intel_uncore, __SlateRecord_i915_reg_defs_h_14, u16, bool)>,
    mmio_writel: Option<unsafe extern "C" fn(*mut intel_uncore, __SlateRecord_i915_reg_defs_h_14, u32, bool)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_pci_h_547 {
    sriov: *mut pci_sriov,
    physfn: *mut pci_dev,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gpio_desc {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_forcewake_range {
    start: u32,
    end: u32,
    domains: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_uncore {
    regs: *mut (),
    i915: *mut drm_i915_private,
    gt: *mut intel_gt,
    rpm: *mut intel_runtime_pm,
    lock: spinlock,
    gsi_offset: u32,
    flags: u32,
    fw_domains_table: *const intel_forcewake_range,
    fw_domains_table_entries: u32,
    shadowed_reg_table: *const i915_mmio_range,
    shadowed_reg_table_entries: u32,
    pmic_bus_access_nb: notifier_block,
    fw_get_funcs: *const intel_uncore_fw_get,
    funcs: intel_uncore_funcs,
    fifo_count: u32,
    fw_domains: u32,
    fw_domains_active: u32,
    fw_domains_timer: u32,
    fw_domains_saved: u32,
    fw_domain: [*mut intel_uncore_forcewake_domain; 16],
    user_forcewake_count: u32,
    debug: *mut intel_uncore_mmio_debug,
}


// register access functions
fn intel_uncore_read(uncore: *mut intel_uncore, reg: __SlateRecord_i915_reg_defs_h_14) -> u32 {
unsafe { unsafe { (*uncore).funcs.mmio_readl }.unwrap()(uncore, reg, true) }
}

fn intel_uncore_write(uncore: *mut intel_uncore, reg: __SlateRecord_i915_reg_defs_h_14, val: u32) {
unsafe { unsafe { (*uncore).funcs.mmio_writel }.unwrap()(uncore, reg, val, true) };
}

// These are untraced mmio-accessors that are only valid to be used inside
// critical sections, such as inside IRQ handlers, where forcewake is explicitly
// controlled.
//
// Think twice, and think again, before using these.
//
// As an example, these accessors can possibly be used between:
//
// spin_lock_irq(&uncore->lock);
// intel_uncore_forcewake_get__locked();
//
// and
//
// intel_uncore_forcewake_put__locked();
// spin_unlock_irq(&uncore->lock);
//
//
// Note: some registers may not need forcewake held, so
// intel_uncore_forcewake_{get,put} can be omitted, see
// intel_uncore_forcewake_for_reg().
//
// Certain architectures will die if the same cacheline is concurrently accessed
// by different clients (e.g. on Ivybridge). Access to registers should
// therefore generally be serialised, by either the dev_priv->uncore.lock or
// a more localised lock guarding all access to that bank of registers.
fn intel_uncore_rmw(uncore: *mut intel_uncore, reg: __SlateRecord_i915_reg_defs_h_14, clear: u32, set: u32) -> u32 {
let old: u32 = intel_uncore_read(uncore, reg);
let val: u32 = old & !clear | set;
intel_uncore_write(uncore, reg, val);
old
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
fn __to_uncore(display: *mut intel_display) -> *mut intel_uncore {
unsafe { to_intel_uncore((*display).drm) }
}

fn intel_de_read(display: *mut intel_display, reg: __SlateRecord_i915_reg_defs_h_14) -> u32 {
let mut val: u32 = 0;
unsafe { intel_dmc_wl_get(display, reg) };
val = intel_uncore_read(__to_uncore(display), reg);
unsafe { intel_dmc_wl_put(display, reg) };
val
}

fn intel_de_write(display: *mut intel_display, reg: __SlateRecord_i915_reg_defs_h_14, val: u32) {
unsafe { intel_dmc_wl_get(display, reg) };
intel_uncore_write(__to_uncore(display), reg, val);
unsafe { intel_dmc_wl_put(display, reg) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct i915_audio_component {
}


fn intel_de_rmw(display: *mut intel_display, reg: __SlateRecord_i915_reg_defs_h_14, clear: u32, set: u32) -> u32 {
let mut val: u32 = 0;
unsafe { intel_dmc_wl_get(display, reg) };
val = intel_uncore_rmw(__to_uncore(display), reg, clear, set);
unsafe { intel_dmc_wl_put(display, reg) };
val
}

#[repr(C)]
#[derive(Clone, Copy)]
struct intel_audio_funcs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_cdclk_funcs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_cdclk_vals {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_color_funcs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dmc {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dpll_global_funcs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dpll_mgr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_fbdev {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_fdi_funcs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_hotplug_irq_funcs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pm_qos_constraints {
    list: plist_head,
    target_value: i32,
    default_value: i32,
    no_constraint_value: i32,
    r#type: u32,
    notifiers: *mut blocking_notifier_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_opregion {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_overlay {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pm_qos_request {
    node: plist_node,
    qos: *mut pm_qos_constraints,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pm_qos_flags_request {
    node: list_head,
    flags: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pm_qos_flags {
    list: list_head,
    effective_flags: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct freq_constraints {
    min_freq: pm_qos_constraints,
    min_freq_notifiers: blocking_notifier_head,
    max_freq: pm_qos_constraints,
    max_freq_notifiers: blocking_notifier_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_120 {
    platdev: *mut platform_device,
    irq: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct freq_qos_request {
    r#type: u32,
    pnode: plist_node,
    qos: *mut freq_constraints,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_141 {
    nssc: i32,
    ssc: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_qos_request {
    r#type: u32,
    data: __SlateRecord_pm_qos_h_109,
    dev: *mut device,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_qos {
    resume_latency: pm_qos_constraints,
    latency_tolerance: pm_qos_constraints,
    freq: freq_constraints,
    flags: pm_qos_flags,
    resume_latency_req: *mut dev_pm_qos_request,
    latency_tolerance_req: *mut dev_pm_qos_request,
    flags_req: *mut dev_pm_qos_request,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_170 {
    last_jiffies: u64,
    count: i32,
    blocked_count: i32,
    state: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cdev {
    kobj: kobject,
    owner: *mut module,
    ops: *const file_operations,
    list: list_head,
    dev: u32,
    count: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sdvo_device_mapping {
    initialized: u8,
    dvo_port: u8,
    target_addr: u8,
    dvo_wiring: u8,
    i2c_pin: u8,
    ddc_pin: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_common_data {
    state_use_accessors: u32,
    node: u32,
    handler_data: *mut (),
    msi_desc: *mut msi_desc,
    affinity: [cpumask; 1],
    effective_affinity: [cpumask; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_intel_display_core_h_280 {
    hw: ilk_wm_values,
    vlv: vlv_wm_values,
    g4x: g4x_wm_values,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_314 {
    any_task_allowed: bool,
    allowed_task: *mut task_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_319 {
    lock: mutex,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_324 {
    obj: intel_global_obj,
    max: [intel_bw_info; 6],
    psf_bw: [u32; 3],
    peakbw: [u32; 8],
    num_qgv_points: u8,
    num_psf_gv_points: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_bw_info {
    deratedbw: [u32; 8],
    num_planes: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_340 {
    funcs: *const intel_cdclk_funcs,
    hw: intel_cdclk_config,
    table: *const intel_cdclk_vals,
    obj: intel_global_obj,
    max_cdclk_freq: u32,
    max_dotclk_freq: u32,
    skl_preferred_vco_freq: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_357 {
    funcs: *const intel_color_funcs,
    glk_linear_degamma_lut: *mut drm_property_blob,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_364 {
    enabled_slices: u8,
    obj: intel_global_obj,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_371 {
    obj: intel_global_obj,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_375 {
    phy_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_383 {
    dmc: *mut intel_dmc,
    wakeref: *mut ref_tracker,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_388 {
    mmio_base: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_393 {
    info: *const dram_info,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dram_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_397 {
    instances: [*mut intel_fbc; 4],
    sys_cache: sys_cache_cfg,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sys_cache_cfg {
    lock: mutex,
    id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_408 {
    fbdev: *mut intel_fbdev,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_413 {
    funcs: *const intel_fdi_funcs,
    pll_freq: u32,
    rx_config: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_421 {
    obj_list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_425 {
    mmio_base: u32,
    mutex: mutex,
    bus: [*mut intel_gmbus; 15],
    wait_queue: wait_queue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_gmbus {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_443 {
    arbiter: *mut i915_hdcp_arbiter,
    comp_added: bool,
    gsc_context: *mut intel_hdcp_gsc_context,
    hdcp_mutex: mutex,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_hdcp_gsc_context {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_457 {
    state: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_467 {
    __device_info: *const intel_display_device_info,
    __runtime_info: intel_display_runtime_info,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_475 {
    false_color: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_479 {
    funcs: *const intel_display_irq_funcs,
    lock: spinlock,
    vlv_display_irqs_enabled: bool,
    vblank_enabled: u8,
    vblank_enable_count: i32,
    vblank_status_last_notified: bool,
    vblank_notify_work: work_struct,
    vlv_imr_mask: u32,
    ilk_de_imr_mask: u32,
    de_pipe_imr_mask: [u32; 4],
    pipestat_irq_mask: [u32; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_display_irq_funcs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_521 {
    funcs: *const intel_modeset_funcs,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_526 {
    linetime: [u16; 4],
    disable: [bool; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_532 {
    waitqueue: wait_queue_head,
    lock: mutex,
    obj: intel_global_obj,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_541 {
    domains: i915_power_domains,
    dc3co: intel_dc3co_state,
    chv_phy_control: u32,
    chv_phy_assert: [bool; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_553 {
    mmio_base: u32,
    mutex: mutex,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_560 {
    broadcast_rgb: *mut drm_property,
    force_audio: *mut drm_property,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_565 {
    mask: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_569 {
    count: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_573 {
    modeset_state: *mut drm_atomic_commit,
    reset_ctx: drm_modeset_acquire_ctx,
    saveDSPARB: u32,
    saveSWF0: [u32; 16],
    saveSWF1: [u32; 16],
    saveSWF3: [u32; 3],
    saveGCDGMBUS: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_585 {
    status: u32,
    block_time_us: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_596 {
    lock: mutex,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_601 {
    phy_failed_calibration: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_609 {
    chv_dpll_md: [u32; 4],
    bxt_phy_grc: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_619 {
    hpll_freq: u32,
    czclk_freq: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_core_h_624 {
    modeset: *mut workqueue_struct,
    flip: *mut workqueue_struct,
    cleanup: *mut workqueue_struct,
    unordered: *mut workqueue_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_tunnel_mgr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_i915_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_runtime_pm {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_data {
    mask: u32,
    irq: u32,
    hwirq: u64,
    common: *mut irq_common_data,
    chip: *mut irq_chip,
    domain: *mut irq_domain,
    parent_data: *mut irq_data,
    chip_data: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_gt {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_chip {
    name: *const u8,
    irq_startup: Option<unsafe extern "C" fn(*mut irq_data) -> u32>,
    irq_shutdown: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_enable: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_disable: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_ack: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_mask: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_mask_ack: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_unmask: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_eoi: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_set_affinity: Option<unsafe extern "C" fn(*mut irq_data, *const cpumask, bool) -> i32>,
    irq_pre_redirect: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_retrigger: Option<unsafe extern "C" fn(*mut irq_data) -> i32>,
    irq_set_type: Option<unsafe extern "C" fn(*mut irq_data, u32) -> i32>,
    irq_set_wake: Option<unsafe extern "C" fn(*mut irq_data, u32) -> i32>,
    irq_bus_lock: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_bus_sync_unlock: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_suspend: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_resume: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_pm_shutdown: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_calc_mask: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_print_chip: Option<unsafe extern "C" fn(*mut irq_data, *mut seq_file)>,
    irq_request_resources: Option<unsafe extern "C" fn(*mut irq_data) -> i32>,
    irq_release_resources: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_compose_msi_msg: Option<unsafe extern "C" fn(*mut irq_data, *mut msi_msg)>,
    irq_write_msi_msg: Option<unsafe extern "C" fn(*mut irq_data, *mut msi_msg)>,
    irq_get_irqchip_state: Option<unsafe extern "C" fn(*mut irq_data, u32, *mut bool) -> i32>,
    irq_set_irqchip_state: Option<unsafe extern "C" fn(*mut irq_data, u32, bool) -> i32>,
    irq_set_vcpu_affinity: Option<unsafe extern "C" fn(*mut irq_data, *mut ()) -> i32>,
    ipi_send_single: Option<unsafe extern "C" fn(*mut irq_data, u32)>,
    ipi_send_mask: Option<unsafe extern "C" fn(*mut irq_data, *const cpumask)>,
    irq_nmi_setup: Option<unsafe extern "C" fn(*mut irq_data) -> i32>,
    irq_nmi_teardown: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_force_complete_move: Option<unsafe extern "C" fn(*mut irq_data)>,
    flags: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irqstat {
    cnt: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_redirect {
    work: irq_work,
    target_cpu: u32,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct irq_desc {
    irq_common_data: irq_common_data,
    irq_data: irq_data,
    kstat_irqs: *mut irqstat,
    handle_irq: Option<unsafe extern "C" fn(*mut irq_desc)>,
    action: *mut irqaction,
    status_use_accessors: u32,
    core_internal_state__do_not_mess_with_it: u32,
    depth: u32,
    wake_depth: u32,
    tot_count: u64,
    last_unhandled: u64,
    irq_count: u32,
    irqs_unhandled: u32,
    threads_handled: __SlateRecord_types_h_190,
    threads_handled_last: i32,
    lock: raw_spinlock,
    percpu_enabled: *mut cpumask,
    redirect: irq_redirect,
    affinity_hint: *const cpumask,
    affinity_notify: *mut irq_affinity_notify,
    pending_mask: [cpumask; 1],
    threads_oneshot: u64,
    threads_active: __SlateRecord_types_h_190,
    wait_for_threads: wait_queue_head,
    nr_actions: u32,
    no_suspend_depth: u32,
    cond_suspend_depth: u32,
    force_resume_depth: u32,
    dir: *mut proc_dir_entry,
    refcnt: __SlateRecord_types_h_202,
    rcu: callback_head,
    kobj: kobject,
    request_mutex: mutex,
    parent_irq: i32,
    owner: *mut module,
    name: *const u8,
    resend_node: hlist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i915_mmio_range {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_uncore_forcewake_domain {
    uncore: *mut intel_uncore,
    id: u32,
    mask: u32,
    wake_count: u32,
    active: bool,
    timer: hrtimer,
    reg_set: *mut u32,
    reg_ack: *mut u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_chip_regs {
    enable: u64,
    disable: u64,
    mask: u64,
    ack: u64,
    eoi: u64,
    r#type: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_chip_type {
    chip: irq_chip,
    regs: irq_chip_regs,
    handler: Option<unsafe extern "C" fn(*mut irq_desc)>,
    r#type: u32,
    mask_cache_priv: u32,
    mask_cache: *mut u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_chip_generic {
    lock: raw_spinlock,
    reg_base: *mut (),
    reg_readl: Option<unsafe extern "C" fn(*mut ()) -> u32>,
    reg_writel: Option<unsafe extern "C" fn(u32, *mut ())>,
    suspend: Option<unsafe extern "C" fn(*mut irq_chip_generic)>,
    resume: Option<unsafe extern "C" fn(*mut irq_chip_generic)>,
    irq_base: u32,
    irq_cnt: u32,
    mask_cache: u32,
    wake_enabled: u32,
    wake_active: u32,
    num_ct: u32,
    private: *mut (),
    installed: u64,
    unused: u64,
    domain: *mut irq_domain,
    list: list_head,
    chip_types: [irq_chip_type; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain_chip_generic {
    irqs_per_chip: u32,
    num_chips: u32,
    irq_flags_to_clear: u32,
    irq_flags_to_set: u32,
    gc_flags: u32,
    exit: Option<unsafe extern "C" fn(*mut irq_chip_generic)>,
    gc: [*mut irq_chip_generic; 0],
}


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


#[repr(C)]
#[derive(Clone, Copy)]
struct of_phandle_args {
    np: *mut device_node,
    args_count: i32,
    args: [u32; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_fwspec {
    fwnode: *mut fwnode_handle,
    param_count: i32,
    param: [u32; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_fwspec_info {
    flags: u64,
    affinity: *const cpumask,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_pm_qos_h_109 {
    pnode: plist_node,
    flr: pm_qos_flags_request,
    freq: freq_qos_request,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_msg {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_desc {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain_ops {
    r#match: Option<unsafe extern "C" fn(*mut irq_domain, *mut device_node, u32) -> i32>,
    select: Option<unsafe extern "C" fn(*mut irq_domain, *mut irq_fwspec, u32) -> i32>,
    map: Option<unsafe extern "C" fn(*mut irq_domain, u32, u64) -> i32>,
    unmap: Option<unsafe extern "C" fn(*mut irq_domain, u32)>,
    xlate: Option<unsafe extern "C" fn(*mut irq_domain, *mut device_node, *const u32, u32, *mut u64, *mut u32) -> i32>,
    alloc: Option<unsafe extern "C" fn(*mut irq_domain, u32, u32, *mut ()) -> i32>,
    free: Option<unsafe extern "C" fn(*mut irq_domain, u32, u32)>,
    activate: Option<unsafe extern "C" fn(*mut irq_domain, *mut irq_data, bool) -> i32>,
    deactivate: Option<unsafe extern "C" fn(*mut irq_domain, *mut irq_data)>,
    translate: Option<unsafe extern "C" fn(*mut irq_domain, *mut irq_fwspec, *mut u64, *mut u32) -> i32>,
    get_fwspec_info: Option<unsafe extern "C" fn(*mut irq_fwspec, *mut irq_fwspec_info) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain {
    link: list_head,
    name: *const u8,
    ops: *const irq_domain_ops,
    host_data: *mut (),
    flags: u32,
    mapcount: u32,
    mutex: mutex,
    root: *mut irq_domain,
    fwnode: *mut fwnode_handle,
    bus_token: u32,
    gc: *mut irq_domain_chip_generic,
    dev: *mut device,
    pm_dev: *mut device,
    parent: *mut irq_domain,
    msi_parent_ops: *const msi_parent_ops,
    exit: Option<unsafe extern "C" fn(*mut irq_domain)>,
    hwirq_max: u64,
    revmap_size: u32,
    revmap_tree: xarray,
    revmap: [*mut irq_data; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gpio_chip {
    label: *const u8,
    gpiodev: *mut gpio_device,
    parent: *mut device,
    fwnode: *mut fwnode_handle,
    owner: *mut module,
    request: Option<unsafe extern "C" fn(*mut gpio_chip, u32) -> i32>,
    free: Option<unsafe extern "C" fn(*mut gpio_chip, u32)>,
    get_direction: Option<unsafe extern "C" fn(*mut gpio_chip, u32) -> i32>,
    direction_input: Option<unsafe extern "C" fn(*mut gpio_chip, u32) -> i32>,
    direction_output: Option<unsafe extern "C" fn(*mut gpio_chip, u32, i32) -> i32>,
    get: Option<unsafe extern "C" fn(*mut gpio_chip, u32) -> i32>,
    get_multiple: Option<unsafe extern "C" fn(*mut gpio_chip, *mut u64, *mut u64) -> i32>,
    set: Option<unsafe extern "C" fn(*mut gpio_chip, u32, i32) -> i32>,
    set_multiple: Option<unsafe extern "C" fn(*mut gpio_chip, *mut u64, *mut u64) -> i32>,
    set_config: Option<unsafe extern "C" fn(*mut gpio_chip, u32, u64) -> i32>,
    to_irq: Option<unsafe extern "C" fn(*mut gpio_chip, u32) -> i32>,
    dbg_show: Option<unsafe extern "C" fn(*mut seq_file, *mut gpio_chip)>,
    init_valid_mask: Option<unsafe extern "C" fn(*mut gpio_chip, *mut u64, u32) -> i32>,
    add_pin_ranges: Option<unsafe extern "C" fn(*mut gpio_chip) -> i32>,
    en_hw_timestamp: Option<unsafe extern "C" fn(*mut gpio_chip, u32, u64) -> i32>,
    dis_hw_timestamp: Option<unsafe extern "C" fn(*mut gpio_chip, u32, u64) -> i32>,
    base: i32,
    ngpio: u16,
    offset: u16,
    names: *const *const u8,
    can_sleep: bool,
}


#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_574: [u8; 19] = [105, 57, 49, 53, 46, 105, 109, 112, 111, 114, 116, 95, 110, 115, 61, 80, 87, 77, 0];

#[repr(C)]
#[derive(Clone, Copy)]
struct pwm_args {
    period: u64,
    polarity: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pwm_waveform {
    period_length_ns: u64,
    duty_length_ns: u64,
    duty_offset_ns: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pwm_state {
    period: u64,
    duty_cycle: u64,
    polarity: u32,
    enabled: bool,
    usage_power: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pwm_device {
    label: *const u8,
    flags: u64,
    hwpwm: u32,
    chip: *mut pwm_chip,
    args: pwm_args,
    state: pwm_state,
    last: pwm_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pwm_capture {
    period: u32,
    duty_cycle: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pwm_ops {
    request: Option<unsafe extern "C" fn(*mut pwm_chip, *mut pwm_device) -> i32>,
    free: Option<unsafe extern "C" fn(*mut pwm_chip, *mut pwm_device)>,
    capture: Option<unsafe extern "C" fn(*mut pwm_chip, *mut pwm_device, *mut pwm_capture, u64) -> i32>,
    sizeof_wfhw: u64,
    round_waveform_tohw: Option<unsafe extern "C" fn(*mut pwm_chip, *mut pwm_device, *const pwm_waveform, *mut ()) -> i32>,
    round_waveform_fromhw: Option<unsafe extern "C" fn(*mut pwm_chip, *mut pwm_device, *const (), *mut pwm_waveform) -> i32>,
    read_waveform: Option<unsafe extern "C" fn(*mut pwm_chip, *mut pwm_device, *mut ()) -> i32>,
    write_waveform: Option<unsafe extern "C" fn(*mut pwm_chip, *mut pwm_device, *const ()) -> i32>,
    apply: Option<unsafe extern "C" fn(*mut pwm_chip, *mut pwm_device, *const pwm_state) -> i32>,
    get_state: Option<unsafe extern "C" fn(*mut pwm_chip, *mut pwm_device, *mut pwm_state) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pwm_chip {
    dev: device,
    cdev: cdev,
    ops: *const pwm_ops,
    owner: *mut module,
    id: u32,
    npwm: u32,
    of_xlate: Option<unsafe extern "C" fn(*mut pwm_chip, *const of_phandle_args) -> *mut pwm_device>,
    atomic: bool,
    gpio: gpio_chip,
    uses_pwmchip_alloc: bool,
    operational: bool,
    __slate_anon_11: __SlateRecord_pwm_h_348,
    pwms: [pwm_device; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_parent_ops {
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


fn str_yes_no(v: bool) -> *const u8 {
(if v { c"yes".as_ptr() as *mut u8 } else { c"no".as_ptr() as *mut u8 }) as *const u8
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
struct gpio_device {
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
union __SlateRecord_pwm_h_348 {
    nonatomic_lock: mutex,
    atomic_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_bpf {
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
struct kthread_worker {
    flags: u32,
    lock: raw_spinlock,
    work_list: list_head,
    delayed_work_list: list_head,
    task: *mut task_struct,
    current_work: *mut kthread_work,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kthread_work {
    node: list_head,
    func: Option<unsafe extern "C" fn(*mut kthread_work)>,
    worker: *mut kthread_worker,
    canceling: i32,
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
struct cgroup_taskset {
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
    __slate_bits_0: __slate_bits::__SlateBits1154U0,
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
struct swap_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct regulator {
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
union __SlateRecord_i2c_h_551 {
    xfer: Option<unsafe extern "C" fn(*mut i2c_adapter, *mut i2c_msg, i32) -> i32>,
    master_xfer: Option<unsafe extern "C" fn(*mut i2c_adapter, *mut i2c_msg, i32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_i2c_h_557 {
    xfer_atomic: Option<unsafe extern "C" fn(*mut i2c_adapter, *mut i2c_msg, i32) -> i32>,
    master_xfer_atomic: Option<unsafe extern "C" fn(*mut i2c_adapter, *mut i2c_msg, i32) -> i32>,
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
    __slate_bits_0: __slate_bits::__SlateBits607U0,
    fbatch: folio_batch,
    index: u64,
    saved_err: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pinctrl {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pinctrl_state {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct cec_adapter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i2c_msg {
    addr: u16,
    flags: u16,
    len: u16,
    buf: *mut u8,
}


// Data for SMBus Messages
// As specified in SMBus standard
#[repr(C)]
#[derive(Clone, Copy)]
union i2c_smbus_data {
    byte: u8,
    word: u16,
    block: [u8; 34],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i2c_algorithm {
    __slate_anon_0: __SlateRecord_i2c_h_551,
    __slate_anon_1: __SlateRecord_i2c_h_557,
    smbus_xfer: Option<unsafe extern "C" fn(*mut i2c_adapter, u16, u16, u8, u8, i32, *mut i2c_smbus_data) -> i32>,
    smbus_xfer_atomic: Option<unsafe extern "C" fn(*mut i2c_adapter, u16, u16, u8, u8, i32, *mut i2c_smbus_data) -> i32>,
    functionality: Option<unsafe extern "C" fn(*mut i2c_adapter) -> u32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i2c_lock_operations {
    lock_bus: Option<unsafe extern "C" fn(*mut i2c_adapter, u32)>,
    trylock_bus: Option<unsafe extern "C" fn(*mut i2c_adapter, u32) -> i32>,
    unlock_bus: Option<unsafe extern "C" fn(*mut i2c_adapter, u32)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i2c_bus_recovery_info {
    recover_bus: Option<unsafe extern "C" fn(*mut i2c_adapter) -> i32>,
    get_scl: Option<unsafe extern "C" fn(*mut i2c_adapter) -> i32>,
    set_scl: Option<unsafe extern "C" fn(*mut i2c_adapter, i32)>,
    get_sda: Option<unsafe extern "C" fn(*mut i2c_adapter) -> i32>,
    set_sda: Option<unsafe extern "C" fn(*mut i2c_adapter, i32)>,
    get_bus_free: Option<unsafe extern "C" fn(*mut i2c_adapter) -> i32>,
    prepare_recovery: Option<unsafe extern "C" fn(*mut i2c_adapter)>,
    unprepare_recovery: Option<unsafe extern "C" fn(*mut i2c_adapter)>,
    scl_gpiod: *mut gpio_desc,
    sda_gpiod: *mut gpio_desc,
    pinctrl: *mut pinctrl,
    pins_default: *mut pinctrl_state,
    pins_gpio: *mut pinctrl_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i2c_adapter_quirks {
    flags: u64,
    max_num_msgs: i32,
    max_write_len: u16,
    max_read_len: u16,
    max_comb_1st_msg_len: u16,
    max_comb_2nd_msg_len: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i2c_adapter {
    owner: *mut module,
    class: u32,
    algo: *const i2c_algorithm,
    algo_data: *mut (),
    lock_ops: *const i2c_lock_operations,
    bus_lock: rt_mutex,
    mux_lock: rt_mutex,
    timeout: i32,
    retries: i32,
    dev: device,
    locked_flags: u64,
    nr: i32,
    name: [u8; 48],
    dev_released: completion,
    userspace_clients_lock: mutex,
    userspace_clients: list_head,
    bus_recovery_info: *mut i2c_bus_recovery_info,
    quirks: *const i2c_adapter_quirks,
    host_notify_domain: *mut irq_domain,
    bus_regulator: *mut regulator,
    debugfs: *mut dentry,
    addrs_in_instantiation: [u64; 2],
}


// Adapter locking functions, exported for shared pin cases
#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_vsc_sdp {
    sdp_type: u8,
    revision: u8,
    length: u8,
    pixelformat: u32,
    colorimetry: u32,
    bpc: i32,
    dynamic_range: u32,
    content_type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_as_sdp {
    sdp_type: u8,
    revision: u8,
    length: u8,
    vtotal: i32,
    target_rr: i32,
    duration_incr_ms: i32,
    duration_decr_ms: i32,
    target_rr_divider: bool,
    mode: u32,
    coasting_vtotal: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dma_fence {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_aux_msg {
    address: u32,
    request: u8,
    reply: u8,
    buffer: *mut (),
    size: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_aux_cec {
    lock: mutex,
    adap: *mut cec_adapter,
    connector: *mut drm_connector,
    unregister_work: delayed_work,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_plane_helper_funcs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_aux {
    name: *const u8,
    ddc: i2c_adapter,
    dev: *mut device,
    drm_dev: *mut drm_device,
    crtc: *mut drm_crtc,
    hw_mutex: mutex,
    crc_work: work_struct,
    crc_count: u8,
    transfer: Option<unsafe extern "C" fn(*mut drm_dp_aux, *mut drm_dp_aux_msg) -> i64>,
    wait_hpd_asserted: Option<unsafe extern "C" fn(*mut drm_dp_aux, u64) -> i32>,
    i2c_nack_count: u32,
    i2c_defer_count: u32,
    cec: drm_dp_aux_cec,
    is_remote: bool,
    powered_down: bool,
    no_zero_sized: bool,
    dpcd_probe_disabled: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_dpcd_ident {
    oui: [u8; 3],
    device_id: [u8; 6],
    hw_rev: u8,
    sw_major_rev: u8,
    sw_minor_rev: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_desc {
    ident: drm_dp_dpcd_ident,
    quirks: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_self_refresh_data {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_pending_vblank_event {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_crtc_helper_funcs {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_edp_backlight_info {
    pwmgen_bit_count: u8,
    pwm_freq_pre_divider: u8,
    max: u32,
    __slate_bits_0: __slate_bits::__SlateBits1204U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_phy_test_params {
    link_rate: i32,
    num_lanes: u8,
    phy_pattern: u8,
    hbr2_reset: [u8; 2],
    custom80: [u8; 10],
    enhanced_frame_cap: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_display_mode {
    clock: i32,
    hdisplay: u16,
    hsync_start: u16,
    hsync_end: u16,
    htotal: u16,
    hskew: u16,
    vdisplay: u16,
    vsync_start: u16,
    vsync_end: u16,
    vtotal: u16,
    vscan: u16,
    flags: u32,
    crtc_clock: i32,
    crtc_hdisplay: u16,
    crtc_hblank_start: u16,
    crtc_hblank_end: u16,
    crtc_hsync_start: u16,
    crtc_hsync_end: u16,
    crtc_htotal: u16,
    crtc_hskew: u16,
    crtc_vdisplay: u16,
    crtc_vblank_start: u16,
    crtc_vblank_end: u16,
    crtc_vsync_start: u16,
    crtc_vsync_end: u16,
    crtc_vtotal: u16,
    width_mm: u16,
    height_mm: u16,
    r#type: u8,
    expose_to_userspace: bool,
    head: list_head,
    name: [u8; 32],
    status: i32,
    picture_aspect_ratio: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kmsg_dump_detail {
    reason: u32,
    description: *const u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kmsg_dumper {
    list: list_head,
    dump: Option<unsafe extern "C" fn(*mut kmsg_dumper, *mut kmsg_dump_detail)>,
    max_reason: u32,
    registered: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_rect {
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_plane_state {
    plane: *mut drm_plane,
    crtc: *mut drm_crtc,
    fb: *mut drm_framebuffer,
    fence: *mut dma_fence,
    crtc_x: i32,
    crtc_y: i32,
    crtc_w: u32,
    crtc_h: u32,
    src_x: u32,
    src_y: u32,
    src_h: u32,
    src_w: u32,
    hotspot_x: i32,
    hotspot_y: i32,
    alpha: u16,
    pixel_blend_mode: u16,
    rotation: u32,
    zpos: u32,
    normalized_zpos: u32,
    color_encoding: u32,
    color_range: u32,
    fb_damage_clips: *mut drm_property_blob,
    ignore_damage_clips: bool,
    src: drm_rect,
    dst: drm_rect,
    visible: bool,
    scaling_filter: u32,
    color_pipeline: *mut drm_colorop,
    commit: *mut drm_crtc_commit,
    state: *mut drm_atomic_commit,
    __slate_bits_0: __slate_bits::__SlateBits1213U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_plane_funcs {
    update_plane: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_crtc, *mut drm_framebuffer, i32, i32, u32, u32, u32, u32, u32, u32, *mut drm_modeset_acquire_ctx) -> i32>,
    disable_plane: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_modeset_acquire_ctx) -> i32>,
    destroy: Option<unsafe extern "C" fn(*mut drm_plane)>,
    reset: Option<unsafe extern "C" fn(*mut drm_plane)>,
    set_property: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_property, u64) -> i32>,
    atomic_create_state: Option<unsafe extern "C" fn(*mut drm_plane) -> *mut drm_plane_state>,
    atomic_duplicate_state: Option<unsafe extern "C" fn(*mut drm_plane) -> *mut drm_plane_state>,
    atomic_destroy_state: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_plane_state)>,
    atomic_set_property: Option<unsafe extern "C" fn(*mut drm_plane, *mut drm_plane_state, *mut drm_property, u64) -> i32>,
    atomic_get_property: Option<unsafe extern "C" fn(*mut drm_plane, *const drm_plane_state, *mut drm_property, *mut u64) -> i32>,
    late_register: Option<unsafe extern "C" fn(*mut drm_plane) -> i32>,
    early_unregister: Option<unsafe extern "C" fn(*mut drm_plane)>,
    atomic_print_state: Option<unsafe extern "C" fn(*mut drm_printer, *const drm_plane_state)>,
    format_mod_supported: Option<unsafe extern "C" fn(*mut drm_plane, u32, u64) -> bool>,
    format_mod_supported_async: Option<unsafe extern "C" fn(*mut drm_plane, u32, u64) -> bool>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_plane {
    dev: *mut drm_device,
    head: list_head,
    name: *mut u8,
    mutex: drm_modeset_lock,
    base: drm_mode_object,
    possible_crtcs: u32,
    format_types: *mut u32,
    format_count: u32,
    format_default: bool,
    modifiers: *mut u64,
    modifier_count: u32,
    crtc: *mut drm_crtc,
    fb: *mut drm_framebuffer,
    old_fb: *mut drm_framebuffer,
    funcs: *const drm_plane_funcs,
    properties: drm_object_properties,
    r#type: u32,
    index: u32,
    helper_private: *const drm_plane_helper_funcs,
    state: *mut drm_plane_state,
    alpha_property: *mut drm_property,
    zpos_property: *mut drm_property,
    rotation_property: *mut drm_property,
    blend_mode_property: *mut drm_property,
    color_encoding_property: *mut drm_property,
    color_range_property: *mut drm_property,
    color_pipeline_property: *mut drm_property,
    scaling_filter_property: *mut drm_property,
    hotspot_x_property: *mut drm_property,
    hotspot_y_property: *mut drm_property,
    kmsg_panic: kmsg_dumper,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_crtc_crc_entry {
    has_frame_counter: bool,
    frame: u32,
    crcs: [u32; 10],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_crtc_crc {
    lock: spinlock,
    source: *const u8,
    opened: bool,
    overflow: bool,
    entries: *mut drm_crtc_crc_entry,
    head: i32,
    tail: i32,
    values_cnt: u64,
    wq: wait_queue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_crtc_state {
    crtc: *mut drm_crtc,
    enable: bool,
    active: bool,
    __slate_bits_0: __slate_bits::__SlateBits1225U0,
    no_vblank: bool,
    plane_mask: u32,
    connector_mask: u32,
    encoder_mask: u32,
    adjusted_mode: drm_display_mode,
    mode: drm_display_mode,
    mode_blob: *mut drm_property_blob,
    degamma_lut: *mut drm_property_blob,
    ctm: *mut drm_property_blob,
    gamma_lut: *mut drm_property_blob,
    background_color: u64,
    target_vblank: u32,
    async_flip: bool,
    vrr_enabled: bool,
    self_refresh_active: bool,
    scaling_filter: u32,
    sharpness_strength: u8,
    event: *mut drm_pending_vblank_event,
    commit: *mut drm_crtc_commit,
    state: *mut drm_atomic_commit,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_crtc_funcs {
    reset: Option<unsafe extern "C" fn(*mut drm_crtc)>,
    cursor_set: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_file, u32, u32, u32) -> i32>,
    cursor_set2: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_file, u32, u32, u32, i32, i32) -> i32>,
    cursor_move: Option<unsafe extern "C" fn(*mut drm_crtc, i32, i32) -> i32>,
    gamma_set: Option<unsafe extern "C" fn(*mut drm_crtc, *mut u16, *mut u16, *mut u16, u32, *mut drm_modeset_acquire_ctx) -> i32>,
    destroy: Option<unsafe extern "C" fn(*mut drm_crtc)>,
    set_config: Option<unsafe extern "C" fn(*mut drm_mode_set, *mut drm_modeset_acquire_ctx) -> i32>,
    page_flip: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_framebuffer, *mut drm_pending_vblank_event, u32, *mut drm_modeset_acquire_ctx) -> i32>,
    page_flip_target: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_framebuffer, *mut drm_pending_vblank_event, u32, u32, *mut drm_modeset_acquire_ctx) -> i32>,
    set_property: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_property, u64) -> i32>,
    atomic_create_state: Option<unsafe extern "C" fn(*mut drm_crtc) -> *mut drm_crtc_state>,
    atomic_duplicate_state: Option<unsafe extern "C" fn(*mut drm_crtc) -> *mut drm_crtc_state>,
    atomic_destroy_state: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_crtc_state)>,
    atomic_set_property: Option<unsafe extern "C" fn(*mut drm_crtc, *mut drm_crtc_state, *mut drm_property, u64) -> i32>,
    atomic_get_property: Option<unsafe extern "C" fn(*mut drm_crtc, *const drm_crtc_state, *mut drm_property, *mut u64) -> i32>,
    late_register: Option<unsafe extern "C" fn(*mut drm_crtc) -> i32>,
    early_unregister: Option<unsafe extern "C" fn(*mut drm_crtc)>,
    set_crc_source: Option<unsafe extern "C" fn(*mut drm_crtc, *const u8) -> i32>,
    verify_crc_source: Option<unsafe extern "C" fn(*mut drm_crtc, *const u8, *mut u64) -> i32>,
    get_crc_sources: Option<unsafe extern "C" fn(*mut drm_crtc, *mut u64) -> *const *const u8>,
    atomic_print_state: Option<unsafe extern "C" fn(*mut drm_printer, *const drm_crtc_state)>,
    get_vblank_counter: Option<unsafe extern "C" fn(*mut drm_crtc) -> u32>,
    enable_vblank: Option<unsafe extern "C" fn(*mut drm_crtc) -> i32>,
    disable_vblank: Option<unsafe extern "C" fn(*mut drm_crtc)>,
    get_vblank_timestamp: Option<unsafe extern "C" fn(*mut drm_crtc, *mut i32, *mut i64, bool) -> bool>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_crtc {
    dev: *mut drm_device,
    port: *mut device_node,
    head: list_head,
    name: *mut u8,
    mutex: drm_modeset_lock,
    base: drm_mode_object,
    primary: *mut drm_plane,
    cursor: *mut drm_plane,
    index: u32,
    cursor_x: i32,
    cursor_y: i32,
    enabled: bool,
    mode: drm_display_mode,
    hwmode: drm_display_mode,
    x: i32,
    y: i32,
    funcs: *const drm_crtc_funcs,
    gamma_size: u32,
    gamma_store: *mut u16,
    helper_private: *const drm_crtc_helper_funcs,
    properties: drm_object_properties,
    scaling_filter_property: *mut drm_property,
    sharpness_strength_property: *mut drm_property,
    state: *mut drm_crtc_state,
    commit_list: list_head,
    commit_lock: spinlock,
    debugfs_entry: *mut dentry,
    crc: drm_crtc_crc,
    fence_context: u32,
    fence_lock: spinlock,
    fence_seqno: u64,
    timeline_name: [u8; 32],
    self_refresh_data: *mut drm_self_refresh_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_mode_set {
    fb: *mut drm_framebuffer,
    crtc: *mut drm_crtc,
    mode: *mut drm_display_mode,
    x: u32,
    y: u32,
    connectors: *mut *mut drm_connector,
    num_connectors: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_crtc_commit {
    crtc: *mut drm_crtc,
    r#ref: kref,
    flip_done: completion,
    hw_done: completion,
    cleanup_done: completion,
    commit_entry: list_head,
    event: *mut drm_pending_vblank_event,
    abort_completion: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __drm_colorops_state {
    ptr: *mut drm_colorop,
    state: *mut drm_colorop_state,
    old_state: *mut drm_colorop_state,
    new_state: *mut drm_colorop_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __drm_planes_state {
    ptr: *mut drm_plane,
    state_to_destroy: *mut drm_plane_state,
    old_state: *mut drm_plane_state,
    new_state: *mut drm_plane_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_tunnel {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __drm_crtcs_state {
    ptr: *mut drm_crtc,
    state_to_destroy: *mut drm_crtc_state,
    old_state: *mut drm_crtc_state,
    new_state: *mut drm_crtc_state,
    commit: *mut drm_crtc_commit,
    out_fence_ptr: *mut i32,
    last_vblank_count: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __drm_connnectors_state {
    ptr: *mut drm_connector,
    state_to_destroy: *mut drm_connector_state,
    old_state: *mut drm_connector_state,
    new_state: *mut drm_connector_state,
    out_fence_ptr: *mut i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_private_state_funcs {
    atomic_create_state: Option<unsafe extern "C" fn(*mut drm_private_obj) -> *mut drm_private_state>,
    atomic_duplicate_state: Option<unsafe extern "C" fn(*mut drm_private_obj) -> *mut drm_private_state>,
    atomic_destroy_state: Option<unsafe extern "C" fn(*mut drm_private_obj, *mut drm_private_state)>,
    atomic_print_state: Option<unsafe extern "C" fn(*mut drm_printer, *const drm_private_state)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_private_obj {
    dev: *mut drm_device,
    head: list_head,
    lock: drm_modeset_lock,
    state: *mut drm_private_state,
    funcs: *const drm_private_state_funcs,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_encoder_helper_funcs {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_drm_fourcc_h_77 {
    cpp: [u8; 4],
    char_per_block: [u8; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_gem_object {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_private_state {
    state: *mut drm_atomic_commit,
    obj: *mut drm_private_obj,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct backlight_device {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct display_timing {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __drm_private_objs_state {
    ptr: *mut drm_private_obj,
    state_to_destroy: *mut drm_private_state,
    old_state: *mut drm_private_state,
    new_state: *mut drm_private_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_atomic_commit {
    r#ref: kref,
    dev: *mut drm_device,
    __slate_bits_0: __slate_bits::__SlateBits756U0,
    colorops: *mut __drm_colorops_state,
    planes: *mut __drm_planes_state,
    crtcs: *mut __drm_crtcs_state,
    num_connector: i32,
    connectors: *mut __drm_connnectors_state,
    num_private_objs: i32,
    private_objs: *mut __drm_private_objs_state,
    acquire_ctx: *mut drm_modeset_acquire_ctx,
    fake_commit: *mut drm_crtc_commit,
    commit_work: work_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_mst_port {
    topology_kref: kref,
    malloc_kref: kref,
    port_num: u8,
    input: bool,
    mcs: bool,
    ddps: bool,
    pdt: u8,
    ldps: bool,
    dpcd_rev: u8,
    num_sdp_streams: u8,
    num_sdp_stream_sinks: u8,
    full_pbn: u16,
    next: list_head,
    mstb: *mut drm_dp_mst_branch,
    aux: drm_dp_aux,
    passthrough_aux: *mut drm_dp_aux,
    parent: *mut drm_dp_mst_branch,
    connector: *mut drm_connector,
    mgr: *mut drm_dp_mst_topology_mgr,
    cached_edid: *const drm_edid,
    fec_capable: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_sideband_msg_hdr {
    lct: u8,
    lcr: u8,
    rad: [u8; 8],
    broadcast: bool,
    path_msg: bool,
    msg_len: u8,
    somt: bool,
    eomt: bool,
    seqno: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_sideband_msg_rx {
    chunk: [u8; 48],
    msg: [u8; 256],
    curchunk_len: u8,
    curchunk_idx: u8,
    curchunk_hdrlen: u8,
    curlen: u8,
    have_somt: bool,
    have_eomt: bool,
    initial_hdr: drm_dp_sideband_msg_hdr,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_mst_branch {
    topology_kref: kref,
    malloc_kref: kref,
    destroy_next: list_head,
    rad: [u8; 8],
    lct: u8,
    num_ports: i32,
    ports: list_head,
    port_parent: *mut drm_dp_mst_port,
    mgr: *mut drm_dp_mst_topology_mgr,
    link_address_sent: bool,
    guid: __SlateRecord_uuid_h_17,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_mst_topology_cbs {
    add_connector: Option<unsafe extern "C" fn(*mut drm_dp_mst_topology_mgr, *mut drm_dp_mst_port, *const u8) -> *mut drm_connector>,
    poll_hpd_irq: Option<unsafe extern "C" fn(*mut drm_dp_mst_topology_mgr)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_mst_topology_mgr {
    base: drm_private_obj,
    dev: *mut drm_device,
    cbs: *const drm_dp_mst_topology_cbs,
    max_dpcd_transaction_bytes: i32,
    aux: *mut drm_dp_aux,
    max_payloads: i32,
    conn_base_id: i32,
    up_req_recv: drm_dp_sideband_msg_rx,
    down_rep_recv: drm_dp_sideband_msg_rx,
    lock: mutex,
    probe_lock: mutex,
    __slate_bits_0: __slate_bits::__SlateBits1238U0,
    payload_count: u8,
    next_start_slot: u8,
    mst_primary: *mut drm_dp_mst_branch,
    dpcd: [u8; 15],
    sink_count: u8,
    funcs: *const drm_private_state_funcs,
    qlock: mutex,
    tx_msg_downq: list_head,
    tx_waitq: wait_queue_head,
    work: work_struct,
    tx_work: work_struct,
    destroy_port_list: list_head,
    destroy_branch_device_list: list_head,
    delayed_destroy_lock: mutex,
    delayed_destroy_wq: *mut workqueue_struct,
    delayed_destroy_work: work_struct,
    up_req_list: list_head,
    up_req_lock: mutex,
    up_req_work: work_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dp_tunnel_ref {
    tunnel: *mut drm_dp_tunnel,
    tracker: *mut ref_tracker,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dsc_rc_range_parameters {
    range_min_qp: u8,
    range_max_qp: u8,
    range_bpg_offset: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_dsc_config {
    line_buf_depth: u8,
    bits_per_component: u8,
    convert_rgb: bool,
    slice_count: u8,
    slice_width: u16,
    slice_height: u16,
    simple_422: bool,
    pic_width: u16,
    pic_height: u16,
    rc_tgt_offset_high: u8,
    rc_tgt_offset_low: u8,
    bits_per_pixel: u16,
    rc_edge_factor: u8,
    rc_quant_incr_limit1: u8,
    rc_quant_incr_limit0: u8,
    initial_xmit_delay: u16,
    initial_dec_delay: u16,
    block_pred_enable: bool,
    first_line_bpg_offset: u8,
    initial_offset: u16,
    rc_buf_thresh: [u16; 14],
    rc_range_params: [drm_dsc_rc_range_parameters; 15],
    rc_model_size: u16,
    flatness_min_qp: u8,
    flatness_max_qp: u8,
    initial_scale_value: u8,
    scale_decrement_interval: u16,
    scale_increment_interval: u16,
    nfl_bpg_offset: u16,
    slice_bpg_offset: u16,
    final_offset: u16,
    vbr_enable: bool,
    mux_word_size: u8,
    slice_chunk_size: u16,
    rc_bits: u16,
    dsc_version_minor: u8,
    dsc_version_major: u8,
    native_422: bool,
    native_420: bool,
    second_line_bpg_offset: u8,
    nsl_bpg_offset: u16,
    second_line_offset_adj: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_colorop_state {
    colorop: *mut drm_colorop,
    bypass: bool,
    curve_1d_type: u32,
    multiplier: u64,
    data: *mut drm_property_blob,
    lut1d_interpolation: u32,
    lut3d_interpolation: u32,
    state: *mut drm_atomic_commit,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_colorop_funcs {
    destroy: Option<unsafe extern "C" fn(*mut drm_colorop)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_colorop {
    dev: *mut drm_device,
    head: list_head,
    index: u32,
    base: drm_mode_object,
    plane: *mut drm_plane,
    state: *mut drm_colorop_state,
    properties: drm_object_properties,
    r#type: u32,
    next: *mut drm_colorop,
    type_property: *mut drm_property,
    bypass_property: *mut drm_property,
    size: u32,
    lut1d_interpolation_property: *mut drm_property,
    curve_1d_type_property: *mut drm_property,
    multiplier_property: *mut drm_property,
    size_property: *mut drm_property,
    lut3d_interpolation_property: *mut drm_property,
    data_property: *mut drm_property,
    next_property: *mut drm_property,
    funcs: *const drm_colorop_funcs,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_encoder_funcs {
    reset: Option<unsafe extern "C" fn(*mut drm_encoder)>,
    destroy: Option<unsafe extern "C" fn(*mut drm_encoder)>,
    late_register: Option<unsafe extern "C" fn(*mut drm_encoder) -> i32>,
    early_unregister: Option<unsafe extern "C" fn(*mut drm_encoder)>,
    debugfs_init: Option<unsafe extern "C" fn(*mut drm_encoder, *mut dentry)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_encoder {
    dev: *mut drm_device,
    head: list_head,
    base: drm_mode_object,
    name: *mut u8,
    encoder_type: i32,
    index: u32,
    possible_crtcs: u32,
    possible_clones: u32,
    crtc: *mut drm_crtc,
    bridge_chain: list_head,
    bridge_chain_mutex: mutex,
    funcs: *const drm_encoder_funcs,
    helper_private: *const drm_encoder_helper_funcs,
    debugfs_entry: *mut dentry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_format_info {
    format: u32,
    depth: u8,
    num_planes: u8,
    __slate_anon_3: __SlateRecord_drm_fourcc_h_77,
    block_w: [u8; 4],
    block_h: [u8; 4],
    hsub: u8,
    vsub: u8,
    has_alpha: bool,
    is_yuv: bool,
    is_color_indexed: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_framebuffer_funcs {
    destroy: Option<unsafe extern "C" fn(*mut drm_framebuffer)>,
    create_handle: Option<unsafe extern "C" fn(*mut drm_framebuffer, *mut drm_file, *mut u32) -> i32>,
    dirty: Option<unsafe extern "C" fn(*mut drm_framebuffer, *mut drm_file, u32, u32, *mut drm_clip_rect, u32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_framebuffer {
    dev: *mut drm_device,
    head: list_head,
    base: drm_mode_object,
    comm: [u8; 16],
    format: *const drm_format_info,
    funcs: *const drm_framebuffer_funcs,
    pitches: [u32; 4],
    offsets: [u32; 4],
    modifier: u64,
    width: u32,
    height: u32,
    flags: i32,
    internal_flags: u32,
    filp_head: list_head,
    obj: [*mut drm_gem_object; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_panel_funcs {
    prepare: Option<unsafe extern "C" fn(*mut drm_panel) -> i32>,
    enable: Option<unsafe extern "C" fn(*mut drm_panel) -> i32>,
    disable: Option<unsafe extern "C" fn(*mut drm_panel) -> i32>,
    unprepare: Option<unsafe extern "C" fn(*mut drm_panel) -> i32>,
    get_modes: Option<unsafe extern "C" fn(*mut drm_panel, *mut drm_connector) -> i32>,
    get_orientation: Option<unsafe extern "C" fn(*mut drm_panel) -> i32>,
    get_timings: Option<unsafe extern "C" fn(*mut drm_panel, u32, *mut display_timing) -> i32>,
    debugfs_init: Option<unsafe extern "C" fn(*mut drm_panel, *mut dentry)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_dsi_vbt_defs_h_50 {
    __slate_bits_0: __slate_bits::__SlateBits1290U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_dsi_vbt_defs_h_84 {
    __slate_bits_0: __slate_bits::__SlateBits1291U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_dsi_vbt_defs_h_102 {
    __slate_bits_0: __slate_bits::__SlateBits1292U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_dsi_vbt_defs_h_113 {
    __slate_bits_0: __slate_bits::__SlateBits1293U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_dsi_vbt_defs_h_122 {
    __slate_bits_0: __slate_bits::__SlateBits1294U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_dsi_vbt_defs_h_140 {
    __slate_bits_0: __slate_bits::__SlateBits1295U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_panel {
    dev: *mut device,
    backlight: *mut backlight_device,
    funcs: *const drm_panel_funcs,
    connector_type: i32,
    list: list_head,
    followers: list_head,
    follower_lock: mutex,
    prepare_prev_first: bool,
    prepared: bool,
    enabled: bool,
    container: *mut (),
    refcount: kref,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cec_notifier {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dp_link_caps {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dp_link_training {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_global_objs_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_hdcp_shim {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_tc_port {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_vblank_work {
    base: kthread_work,
    vblank: *mut drm_vblank_crtc,
    count: u64,
    cancelling: i32,
    node: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_cert_rx {
    receiver_id: [u8; 5],
    kpub_rx: [u8; 131],
    reserved: [u8; 2],
    dcp_signature: [u8; 384],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_streamid_type {
    stream_id: u8,
    stream_type: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_345 {
    max_link_rate: i32,
    rate: i32,
    lanes: i32,
    preemphasis: i32,
    vswing: i32,
    bpp: i32,
    pps: intel_pps_delays,
    drrs_msa_timing_delay: u8,
    low_vswing: bool,
    hobl: bool,
    dsc_disable: bool,
    pipe_joiner_enable: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_360 {
    enable: bool,
    full_link: bool,
    require_aux_wakeup: bool,
    idle_frames: i32,
    tp1_wakeup_time_us: i32,
    tp2_tp3_wakeup_time_us: i32,
    psr2_tp2_tp3_wakeup_time_us: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_370 {
    pwm_freq_hz: u16,
    brightness_precision_bits: u16,
    hdr_dpcd_refresh_timeout: u16,
    present: bool,
    active_low_pwm: bool,
    min_brightness: u8,
    controller: i8,
    r#type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_382 {
    panel_id: u16,
    config: *mut mipi_config,
    pps: *mut mipi_pps_data,
    bl_ports: u16,
    cabc_ports: u16,
    seq_version: u8,
    size: u32,
    data: *mut u8,
    sequence: [*const u8; 12],
    deassert_seq: *mut u8,
    orientation: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_407 {
    present: bool,
    level: u32,
    min: u32,
    max: u32,
    enabled: bool,
    combination_mode: bool,
    active_low_pwm: bool,
    alternate_pwm_increment: bool,
    pwm_level_min: u32,
    pwm_level_max: u32,
    pwm_enabled: bool,
    util_pin_active_low: bool,
    controller: u8,
    pwm: *mut pwm_device,
    pwm_state: pwm_state,
    edp: __SlateRecord_intel_display_types_h_427,
    device: *mut backlight_device,
    funcs: *const intel_panel_bl_funcs,
    pwm_funcs: *const intel_panel_bl_funcs,
    power: Option<unsafe extern "C" fn(*mut intel_connector, bool)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_intel_display_types_h_427 {
    vesa: __SlateRecord_intel_display_types_h_428,
    intel_cap: __SlateRecord_intel_display_types_h_432,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_428 {
    info: drm_edp_backlight_info,
    luminance_control_support: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_432 {
    sdr_uses_aux: bool,
    supports_2084_decode: bool,
    supports_2020_gamut: bool,
    supports_segmented_backlight: bool,
    supports_sdp_colorimetry: bool,
    supports_tone_mapping: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_tx_caps {
    version: u8,
    tx_cap_mask: [u8; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_559 {
    dsc_decompression_aux: *mut drm_dp_aux,
    dsc_dpcd: [u8; 16],
    fec_capability: u8,
    __slate_bits_0: __slate_bits::__SlateBits1321U0,
    dsc_branch_caps: __SlateRecord_intel_display_types_h_568,
    panel_replay_caps: __SlateRecord_intel_display_types_h_576,
    psr_caps: __SlateRecord_intel_display_types_h_588,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_568 {
    overall_throughput: __SlateRecord_intel_display_types_h_569,
    max_line_width: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_569 {
    rgb_yuv444: i32,
    yuv422_420: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_576 {
    dpcd: [u8; 7],
    support: bool,
    su_support: bool,
    dsc_support: u32,
    su_w_granularity: u16,
    su_y_granularity: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_588 {
    dpcd: [u8; 2],
    intel_wa_dpcd: u8,
    support: bool,
    su_support: bool,
    su_w_granularity: u16,
    su_y_granularity: u16,
    sync_latency: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_602 {
    port: *mut drm_dp_mst_port,
    dp: *mut intel_dp,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_607 {
    force_bpp_x16: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_ake_init {
    msg_id: u8,
    r_tx: [u8; 8],
    tx_caps: hdcp2_tx_caps,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dp_tunnel_inherited_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i915_vma {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_ake_send_cert {
    msg_id: u8,
    cert_rx: hdcp2_cert_rx,
    r_rx: [u8; 8],
    rx_caps: [u8; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_ake_no_stored_km {
    msg_id: u8,
    e_kpub_km: [u8; 128],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_ake_send_hprime {
    msg_id: u8,
    h_prime: [u8; 32],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_ake_send_pairing_info {
    msg_id: u8,
    e_kh_km: [u8; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_lc_init {
    msg_id: u8,
    r_n: [u8; 8],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_lc_send_lprime {
    msg_id: u8,
    l_prime: [u8; 32],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_853 {
    wm0: skl_wm_level,
    trans_wm: skl_wm_level,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_ske_send_eks {
    msg_id: u8,
    e_dkey_ks: [u8; 16],
    riv: [u8; 8],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_rep_send_receiverid_list {
    msg_id: u8,
    rx_info: [u8; 2],
    seq_num_v: [u8; 3],
    v_prime: [u8; 16],
    receiver_ids: [u8; 155],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_rep_send_ack {
    msg_id: u8,
    v: [u8; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp2_rep_stream_ready {
    msg_id: u8,
    m_prime: [u8; 32],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hdcp_port_data {
    hdcp_ddi: u32,
    hdcp_transcoder: u32,
    port_type: u8,
    protocol: u8,
    k: u16,
    seq_num_m: u32,
    streams: *mut hdcp2_streamid_type,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_intel_display_types_h_899 {
    ilk: __SlateRecord_intel_display_types_h_922,
    skl: __SlateRecord_intel_display_types_h_927,
    vlv: __SlateRecord_intel_display_types_h_948,
    g4x: __SlateRecord_intel_display_types_h_955,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_922 {
    intermediate: intel_pipe_wm,
    optimal: intel_pipe_wm,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_927 {
    raw: skl_pipe_wm,
    optimal: skl_pipe_wm,
    ddb: skl_ddb_entry,
    plane_ddb: [skl_ddb_entry; 8],
    plane_ddb_y: [skl_ddb_entry; 8],
    plane_min_ddb: [u16; 8],
    plane_interim_ddb: [u16; 8],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_948 {
    raw: [g4x_pipe_wm; 3],
    intermediate: vlv_wm_state,
    optimal: vlv_wm_state,
    fifo_state: vlv_fifo_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_955 {
    raw: [g4x_pipe_wm; 3],
    intermediate: g4x_wm_state,
    optimal: g4x_wm_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i915_hdcp_ops {
    owner: *mut module,
    initiate_hdcp2_session: Option<unsafe extern "C" fn(*mut device, *mut hdcp_port_data, *mut hdcp2_ake_init) -> i32>,
    verify_receiver_cert_prepare_km: Option<unsafe extern "C" fn(*mut device, *mut hdcp_port_data, *mut hdcp2_ake_send_cert, *mut bool, *mut hdcp2_ake_no_stored_km, *mut u64) -> i32>,
    verify_hprime: Option<unsafe extern "C" fn(*mut device, *mut hdcp_port_data, *mut hdcp2_ake_send_hprime) -> i32>,
    store_pairing_info: Option<unsafe extern "C" fn(*mut device, *mut hdcp_port_data, *mut hdcp2_ake_send_pairing_info) -> i32>,
    initiate_locality_check: Option<unsafe extern "C" fn(*mut device, *mut hdcp_port_data, *mut hdcp2_lc_init) -> i32>,
    verify_lprime: Option<unsafe extern "C" fn(*mut device, *mut hdcp_port_data, *mut hdcp2_lc_send_lprime) -> i32>,
    get_session_key: Option<unsafe extern "C" fn(*mut device, *mut hdcp_port_data, *mut hdcp2_ske_send_eks) -> i32>,
    repeater_check_flow_prepare_ack: Option<unsafe extern "C" fn(*mut device, *mut hdcp_port_data, *mut hdcp2_rep_send_receiverid_list, *mut hdcp2_rep_send_ack) -> i32>,
    verify_mprime: Option<unsafe extern "C" fn(*mut device, *mut hdcp_port_data, *mut hdcp2_rep_stream_ready) -> i32>,
    enable_hdcp_authentication: Option<unsafe extern "C" fn(*mut device, *mut hdcp_port_data) -> i32>,
    close_hdcp_session: Option<unsafe extern "C" fn(*mut device, *mut hdcp_port_data) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct i915_hdcp_arbiter {
    hdcp_dev: *mut device,
    ops: *const i915_hdcp_ops,
    mutex: mutex,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct mipi_config {
    panel_id: u16,
    __slate_anon_1: __SlateRecord_intel_dsi_vbt_defs_h_50,
    __slate_anon_2: __SlateRecord_intel_dsi_vbt_defs_h_84,
    __slate_anon_3: __SlateRecord_intel_dsi_vbt_defs_h_102,
    rsvd5: u8,
    target_burst_mode_freq: u32,
    dsi_ddr_clk: u32,
    bridge_ref_clk: u32,
    __slate_anon_8: __SlateRecord_intel_dsi_vbt_defs_h_113,
    __slate_anon_9: __SlateRecord_intel_dsi_vbt_defs_h_122,
    hs_tx_timeout: u32,
    lp_rx_timeout: u32,
    turn_around_timeout: u32,
    device_reset_timer: u32,
    master_init_timer: u32,
    dbi_bw_timer: u32,
    lp_byte_clk_val: u32,
    __slate_anon_17: __SlateRecord_intel_dsi_vbt_defs_h_140,
    clk_lane_switch_cnt: u32,
    hl_switch_cnt: u32,
    rsvd11: [u32; 6],
    tclk_miss: u8,
    tclk_post: u8,
    rsvd12: u8,
    tclk_pre: u8,
    tclk_prepare: u8,
    tclk_settle: u8,
    tclk_term_enable: u8,
    tclk_trail: u8,
    tclk_prepare_clkzero: u16,
    rsvd13: u8,
    td_term_enable: u8,
    teot: u8,
    ths_exit: u8,
    ths_prepare: u8,
    ths_prepare_hszero: u16,
    rsvd14: u8,
    ths_settle: u8,
    ths_skip: u8,
    ths_trail: u8,
    tinit: u8,
    tlpx: u8,
    rsvd15: [u8; 3],
    panel_enable: u8,
    bl_enable: u8,
    pwm_enable: u8,
    reset_r_n: u8,
    pwr_down_r: u8,
    stdby_r_n: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1036 {
    active: bool,
    enable: bool,
    degamma_lut: *mut drm_property_blob,
    gamma_lut: *mut drm_property_blob,
    ctm: *mut drm_property_blob,
    mode: drm_display_mode,
    pipe_mode: drm_display_mode,
    adjusted_mode: drm_display_mode,
    background_color: u32,
    scaling_filter: u32,
    sharpness_strength: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct icl_port_dpll {
    pll: *mut intel_dpll,
    hw_state: intel_dpll_hw_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1171 {
    ctrl: u32,
    div: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1230 {
    control: u32,
    pgm_ratios: u32,
    lvds_border_bits: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1237 {
    casf: intel_casf,
    dst: drm_rect,
    enabled: bool,
    force_thru: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_intel_display_types_h_1277 {
    csc_mode: u32,
    cgm_mode: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1303 {
    enable: u32,
    gcp: u32,
    avi: hdmi_infoframe,
    spd: hdmi_infoframe,
    hdmi: hdmi_infoframe,
    drm: hdmi_infoframe,
    vsc: drm_dp_vsc_sdp,
    as_sdp: drm_dp_as_sdp,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1347 {
    compression_enabled_on_link: bool,
    compression_enable: bool,
    slice_config: intel_dsc_slice_config,
    compressed_bpp_x16: u16,
    config: drm_dsc_config,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dsc_slice_config {
    pipes_per_line: i32,
    streams_per_pipe: i32,
    slices_per_stream: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1401 {
    enable: bool,
    in_range: bool,
    pipeline_full: u8,
    flipline: u16,
    vmin: u16,
    vmax: u16,
    guardband: u16,
    vsync_end: u32,
    vsync_start: u32,
    dc_balance: __SlateRecord_intel_display_types_h_1406,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1406 {
    enable: bool,
    vmin: u16,
    vmax: u16,
    guardband: u16,
    slope: u16,
    max_increase: u16,
    max_decrease: u16,
    vblank_target: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1416 {
    enable: bool,
    cmrr_n: u64,
    cmrr_m: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1422 {
    enable: bool,
    link_count: u8,
    pixel_overlap: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1437 {
    io_wake_lines: u8,
    fast_wake_lines: u8,
    check_entry_lines: u8,
    aux_less_wake_lines: u8,
    silence_period_sym_clocks: u8,
    lfps_half_cycle_num_of_syms: u8,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct mipi_pps_data {
    panel_on_delay: u16,
    bl_enable_delay: u16,
    bl_disable_delay: u16,
    panel_off_delay: u16,
    panel_power_cycle_delay: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_encoder {
    base: drm_encoder,
    r#type: u32,
    port: i32,
    cloneable: u16,
    pipe_mask: u8,
    link_check_work: delayed_work,
    link_check: Option<unsafe extern "C" fn(*mut intel_encoder)>,
    hotplug: Option<unsafe extern "C" fn(*mut intel_encoder, *mut intel_connector) -> u32>,
    compute_output_type: Option<unsafe extern "C" fn(*mut intel_encoder, *mut intel_crtc_state, *mut drm_connector_state) -> u32>,
    compute_config: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_encoder, *mut intel_crtc_state, *mut drm_connector_state) -> i32>,
    compute_config_late: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_encoder, *mut intel_crtc_state, *mut drm_connector_state) -> i32>,
    pre_pll_enable: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_encoder, *const intel_crtc_state, *const drm_connector_state)>,
    pre_enable: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_encoder, *const intel_crtc_state, *const drm_connector_state)>,
    enable: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_encoder, *const intel_crtc_state, *const drm_connector_state)>,
    disable: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_encoder, *const intel_crtc_state, *const drm_connector_state)>,
    post_disable: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_encoder, *const intel_crtc_state, *const drm_connector_state)>,
    post_pll_disable: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_encoder, *const intel_crtc_state, *const drm_connector_state)>,
    update_pipe: Option<unsafe extern "C" fn(*mut intel_atomic_state, *mut intel_encoder, *const intel_crtc_state, *const drm_connector_state)>,
    audio_enable: Option<unsafe extern "C" fn(*mut intel_encoder, *const intel_crtc_state, *const drm_connector_state)>,
    audio_disable: Option<unsafe extern "C" fn(*mut intel_encoder, *const intel_crtc_state, *const drm_connector_state)>,
    get_hw_state: Option<unsafe extern "C" fn(*mut intel_encoder, *mut i32) -> bool>,
    get_config: Option<unsafe extern "C" fn(*mut intel_encoder, *mut intel_crtc_state)>,
    sync_state: Option<unsafe extern "C" fn(*mut intel_encoder, *const intel_crtc_state)>,
    initial_fastset_check: Option<unsafe extern "C" fn(*mut intel_encoder, *mut intel_crtc_state) -> bool>,
    get_power_domains: Option<unsafe extern "C" fn(*mut intel_encoder, *mut intel_crtc_state)>,
    suspend: Option<unsafe extern "C" fn(*mut intel_encoder)>,
    suspend_complete: Option<unsafe extern "C" fn(*mut intel_encoder)>,
    shutdown: Option<unsafe extern "C" fn(*mut intel_encoder)>,
    shutdown_complete: Option<unsafe extern "C" fn(*mut intel_encoder)>,
    enable_clock: Option<unsafe extern "C" fn(*mut intel_encoder, *const intel_crtc_state)>,
    disable_clock: Option<unsafe extern "C" fn(*mut intel_encoder)>,
    is_clock_enabled: Option<unsafe extern "C" fn(*mut intel_encoder) -> bool>,
    port_pll_type: Option<unsafe extern "C" fn(*mut intel_encoder, *const intel_crtc_state) -> u32>,
    get_buf_trans: Option<unsafe extern "C" fn(*mut intel_encoder, *const intel_crtc_state, *mut i32) -> *const intel_ddi_buf_trans>,
    set_signal_levels: Option<unsafe extern "C" fn(*mut intel_encoder, *const intel_crtc_state)>,
    hpd_pin: u32,
    power_domain: u32,
    devdata: *const intel_bios_encoder_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1535 {
    active: __SlateRecord_intel_display_types_h_1537,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_intel_display_types_h_1537 {
    ilk: intel_pipe_wm,
    vlv: vlv_wm_state,
    g4x: g4x_wm_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1544 {
    mutex: mutex,
    work: delayed_work,
    refresh_rate: u32,
    frontbuffer_bits: u32,
    busy_frontbuffer_bits: u32,
    cpu_transcoder: i32,
    m_n: intel_link_m_n,
    m2_n2: intel_link_m_n,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1554 {
    flip_count: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1560 {
    start_vbl_count: u32,
    start_vbl_time: i64,
    min_vbl: i32,
    max_vbl: i32,
    scanline_start: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1588 {
    enabled: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1683 {
    r#type: u32,
    max_tmds_clock: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_panel_bl_funcs {
    setup: Option<unsafe extern "C" fn(*mut intel_connector, i32) -> i32>,
    get: Option<unsafe extern "C" fn(*mut intel_connector, i32) -> u32>,
    set: Option<unsafe extern "C" fn(*const drm_connector_state, u32)>,
    disable: Option<unsafe extern "C" fn(*const drm_connector_state, u32)>,
    enable: Option<unsafe extern "C" fn(*const intel_crtc_state, *const drm_connector_state, u32)>,
    hz_to_pwm: Option<unsafe extern "C" fn(*mut intel_connector, u32) -> u32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_pps_delays {
    power_up: u16,
    backlight_on: u16,
    backlight_off: u16,
    power_down: u16,
    power_cycle: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_vbt_panel_data {
    lfp_vbt_mode: *mut drm_display_mode,
    sdvo_lvds_vbt_mode: *mut drm_display_mode,
    panel_type: i32,
    __slate_bits_0: __slate_bits::__SlateBits1310U0,
    bios_lvds_val: u32,
    vrr: bool,
    seamless_drrs_min_refresh_rate: u8,
    drrs_type: u32,
    edp: __SlateRecord_intel_display_types_h_345,
    psr: __SlateRecord_intel_display_types_h_360,
    backlight: __SlateRecord_intel_display_types_h_370,
    dsi: __SlateRecord_intel_display_types_h_382,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_panel {
    base: *mut drm_panel,
    fixed_edid: *const drm_edid,
    fixed_modes: list_head,
    backlight: __SlateRecord_intel_display_types_h_407,
    vbt: intel_vbt_panel_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_hdcp {
    shim: *const intel_hdcp_shim,
    mutex: mutex,
    value: u64,
    check_work: delayed_work,
    prop_work: work_struct,
    hdcp_encrypted: bool,
    hdcp2_supported: bool,
    hdcp2_encrypted: bool,
    content_type: u8,
    is_paired: bool,
    is_repeater: bool,
    seq_num_v: u32,
    seq_num_m: u32,
    cp_irq_queue: wait_queue_head,
    cp_irq_count: __SlateRecord_types_h_190,
    cp_irq_count_cached: i32,
    cpu_transcoder: i32,
    stream_transcoder: i32,
    force_hdcp14: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_intel_display_types_h_1728 {
    vlv_pps_pipe: i32,
    pps_idx: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_connector {
    base: drm_connector,
    encoder: *mut intel_encoder,
    acpi_device_id: u32,
    get_hw_state: Option<unsafe extern "C" fn(*mut intel_connector) -> bool>,
    sync_state: Option<unsafe extern "C" fn(*mut intel_connector, *const intel_crtc_state)>,
    panel: intel_panel,
    detect_edid: *const drm_edid,
    hotplug_retries: i32,
    polled: u8,
    force_joined_pipes: i32,
    dp: __SlateRecord_intel_display_types_h_559,
    mst: __SlateRecord_intel_display_types_h_602,
    link: __SlateRecord_intel_display_types_h_607,
    modeset_retry_work: work_struct,
    hdcp: intel_hdcp,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1848 {
    active: bool,
    mst_probed_lane_count: i32,
    mst_probed_rate: i32,
    training: *mut intel_dp_link_training,
    caps: *mut intel_dp_link_caps,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1885 {
    stream_encoders: [*mut intel_dp_mst_encoder; 4],
    mgr: drm_dp_mst_topology_mgr,
    active_streams: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1919 {
    min_tmds_clock: i32,
    max_tmds_clock: i32,
    max_dotclock: i32,
    pcon_max_frl_bw: i32,
    max_bpc: u8,
    ycbcr_444_to_420: bool,
    ycbcr420_passthrough: bool,
    rgb_to_ycbcr: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1953 {
    transcoder: i32,
    lock: mutex,
    lobf_disable_debug: bool,
    sink_alpm_error: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_1963 {
    mask: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dpll {
    n: i32,
    m1: i32,
    m2: i32,
    p1: i32,
    p2: i32,
    dot: i32,
    vco: i32,
    m: i32,
    p: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_display_types_h_2000 {
    mutex: mutex,
    num_streams: u32,
    auth_status: bool,
    port_data: hdcp_port_data,
    mst_type1_capable: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_atomic_state {
    base: drm_atomic_commit,
    wakeref: *mut ref_tracker,
    global_objs: *mut intel_global_objs_state,
    num_global_objs: i32,
    internal: bool,
    dpll_set: bool,
    modeset: bool,
    dpll_state: [intel_dpll_state; 9],
    inherited_dp_tunnels: *mut intel_dp_tunnel_inherited_state,
    skip_intermediate_wm: bool,
    rps_interactive: bool,
    cleanup_work: work_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_initial_plane_config {
    fb: *mut drm_framebuffer,
    vma: *mut i915_vma,
    size: i32,
    base: u32,
    rotation: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_scaler {
    mode: u32,
    in_use: bool,
    hscale: i32,
    vscale: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_995 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_996 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1019 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1019_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1020 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1029 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1029_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1031 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1031_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1032 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1033_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1033_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1033_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1033_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1033_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1033_16 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1034_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1034_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1034_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1034_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1034_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1034_16 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1035_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1035_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1035_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1035_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1035_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1035_16 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1041 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1125 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1125_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1125_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1125_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1126 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1126_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1128 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1128_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1129 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1129_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1130 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1130_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1132 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1132_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1133 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1133_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1134 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1134_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1136 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1137 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1137_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1137_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1137_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1137_7 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1137_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1137_17 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1139 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1139_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1140 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1140_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1140_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1140_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1140_7 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1140_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1140_17 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1146 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1147 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1149 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1149_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1150 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1151 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1153 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1153_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1154 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1159 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1159_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1160 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1161 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1163 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1163_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1166 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1166_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1172 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1172_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1178 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1178_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1183_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1183_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1183_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1183_7 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1183_8 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1183_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1183_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1183_11 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1183_13 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1183_14 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1193_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1193_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1193_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1193_7 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1193_8 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1193_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1193_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1193_11 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1193_13 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1193_14 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1210 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1210_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1211 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1212 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1214 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1222_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1222_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1222_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1222_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1222_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1222_16 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1222_18 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1222_19 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1222_20 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1222_22 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1222_25 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1222_32 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1224_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1224_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1224_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1224_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1224_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1224_16 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1224_18 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1224_19 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1224_20 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1224_22 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1224_25 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1224_32 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1226 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1228 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1228_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1229 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1229_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1230 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1230_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1230_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1231 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1277_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1277_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1277_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1277_7 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1277_8 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1277_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1277_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1277_11 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1277_13 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1277_14 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1699 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1705 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1721_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1721_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1721_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1721_7 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1721_8 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1721_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1721_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1721_11 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1721_13 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1721_14 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1730_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1730_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1730_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1730_7 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1730_8 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1730_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1730_10 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1730_11 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1730_13 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1730_14 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1843 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1843_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1846 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1848 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1852 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1852_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1854 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1857 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1857_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1859 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1862 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1866 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1872 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1873 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1881 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1882 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1890 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1891 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1899 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1900 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1920 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1920_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1921 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1921_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1922 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1922_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1960 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1961 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1964 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1964_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1965 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1968 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1968_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1969 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1983 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1983_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1984 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1985 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1988 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1988_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1989 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1998 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1998_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_1999 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2000 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2003 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2003_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2011 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2016 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2016_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2023 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2023_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2051 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2051_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2052 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2052_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2053 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2053_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2055 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2055_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2056 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2056_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2057 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2057_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2059 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2059_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2060 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2060_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2061 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2061_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2066 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2066_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2071 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2071_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2075 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2075_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2082 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2082_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2091 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2091_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2092 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2098 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2098_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2099 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2109 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2109_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2110 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2110_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2110_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2113 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2114 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2114_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2152 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2152_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2153_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2153_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2153_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2153_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2153_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2153_16 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2154_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2154_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2154_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2154_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2154_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2154_16 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2158 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2159_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2159_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2159_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2159_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2159_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2159_16 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2162 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2163_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2163_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2163_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2163_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2163_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2163_16 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2166 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2167_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2167_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2167_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2167_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2167_9 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2167_16 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2219 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2219_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2220 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2227 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2227_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_lt_phy_c_2228 {
}


#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

mod __slate_bits {
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits607U0 {
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
    pub struct __SlateBits638U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits286U0 {
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
    pub struct __SlateBits465U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1154U0 {
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
    pub struct __SlateBits289U0 {
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
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits492U0 {
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
    pub struct __SlateBits489U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits492U1 {
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
    pub struct __SlateBits492U2 {
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
    pub struct __SlateBits492U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits467U0 {
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
    pub struct __SlateBits413U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits695U0 {
        #[bits(18)]
        pub lineno: u32,
        #[bits(6)]
        pub class_id: u32,
        #[bits(8)]
        pub flags: u32,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits756U0 {
        #[bits(1)]
        pub allow_modeset: bool,
        #[bits(1)]
        pub legacy_cursor_update: bool,
        #[bits(1)]
        pub async_update: bool,
        #[bits(1)]
        pub duplicated: bool,
        #[bits(1)]
        pub checked: bool,
        #[bits(1)]
        pub plane_color_pipeline: bool,
        #[bits(2, access = na)]
        pub __slate_pad_6: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1225U0 {
        #[bits(1)]
        pub planes_changed: bool,
        #[bits(1)]
        pub mode_changed: bool,
        #[bits(1)]
        pub active_changed: bool,
        #[bits(1)]
        pub connectors_changed: bool,
        #[bits(1)]
        pub zpos_changed: bool,
        #[bits(1)]
        pub color_mgmt_changed: bool,
        #[bits(2, access = na)]
        pub __slate_pad_6: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1213U0 {
        #[bits(1)]
        pub color_mgmt_changed: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1204U0 {
        #[bits(1)]
        pub lsb_reg_used: bool,
        #[bits(1)]
        pub aux_enable: bool,
        #[bits(1)]
        pub aux_set: bool,
        #[bits(1)]
        pub luminance_set: bool,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1310U0 {
        #[bits(1)]
        pub lvds_dither: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1290U0 {
        #[bits(1)]
        pub enable_dithering: u32,
        #[bits(1)]
        pub rsvd1: u32,
        #[bits(1)]
        pub is_bridge: u32,
        #[bits(2)]
        pub panel_arch_type: u32,
        #[bits(1)]
        pub is_cmd_mode: u32,
        #[bits(2)]
        pub video_transfer_mode: u32,
        #[bits(1)]
        pub cabc_supported: u32,
        #[bits(1)]
        pub pwm_blc: u32,
        #[bits(4)]
        pub videomode_color_format: u32,
        #[bits(2)]
        pub rotation: u32,
        #[bits(1)]
        pub bta_disable: u32,
        #[bits(15)]
        pub rsvd2: u32,
    }
    #[bitfields::bitfield([u8; 2], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1291U0 {
        #[bits(2)]
        pub dual_link: u16,
        #[bits(2)]
        pub lane_cnt: u16,
        #[bits(3)]
        pub pixel_overlap: u16,
        #[bits(1)]
        pub rgb_flip: u16,
        #[bits(2)]
        pub dl_dcs_cabc_ports: u16,
        #[bits(2)]
        pub dl_dcs_backlight_ports: u16,
        #[bits(1)]
        pub port_sync: u16,
        #[bits(3)]
        pub rsvd3: u16,
    }
    #[bitfields::bitfield([u8; 2], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1292U0 {
        #[bits(1)]
        pub dsi_usage: u16,
        #[bits(15)]
        pub rsvd4: u16,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1293U0 {
        #[bits(2)]
        pub byte_clk_sel: u8,
        #[bits(6)]
        pub rsvd6: u8,
    }
    #[bitfields::bitfield([u8; 2], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1294U0 {
        #[bits(1)]
        pub dphy_param_valid: u16,
        #[bits(1)]
        pub eot_pkt_disabled: u16,
        #[bits(1)]
        pub enable_clk_stop: u16,
        #[bits(1)]
        pub blanking_packets_during_bllp: u16,
        #[bits(1)]
        pub lp_clock_during_lpm: u16,
        #[bits(11)]
        pub rsvd7: u16,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1295U0 {
        #[bits(6)]
        pub prepare_cnt: u32,
        #[bits(2)]
        pub rsvd8: u32,
        #[bits(8)]
        pub clk_zero_cnt: u32,
        #[bits(5)]
        pub trail_cnt: u32,
        #[bits(3)]
        pub rsvd9: u32,
        #[bits(6)]
        pub exit_zero_cnt: u32,
        #[bits(2)]
        pub rsvd10: u32,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1321U0 {
        #[bits(1)]
        pub dsc_hblank_expansion_quirk: u8,
        #[bits(1)]
        pub dsc_throughput_quirk: u8,
        #[bits(1)]
        pub dsc_decompression_enabled: u8,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1238U0 {
        #[bits(1)]
        pub mst_state: bool,
        #[bits(1)]
        pub payload_id_table_cleared: bool,
        #[bits(1)]
        pub reset_rx_state: bool,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits803U0 {
        #[bits(1)]
        pub tunnel_suspended: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield([u8; 9], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits929U0 {
        #[bits(1)]
        pub g4x: u64,
        #[bits(1)]
        pub mobile: u64,
        #[bits(1)]
        pub dgfx: u64,
        #[bits(1)]
        pub i830: u64,
        #[bits(1)]
        pub i845g: u64,
        #[bits(1)]
        pub i85x: u64,
        #[bits(1)]
        pub i865g: u64,
        #[bits(1)]
        pub i915g: u64,
        #[bits(1)]
        pub i915gm: u64,
        #[bits(1)]
        pub i945g: u64,
        #[bits(1)]
        pub i945gm: u64,
        #[bits(1)]
        pub g33: u64,
        #[bits(1)]
        pub pineview: u64,
        #[bits(1)]
        pub i965g: u64,
        #[bits(1)]
        pub i965gm: u64,
        #[bits(1)]
        pub g45: u64,
        #[bits(1)]
        pub gm45: u64,
        #[bits(1)]
        pub ironlake: u64,
        #[bits(1)]
        pub sandybridge: u64,
        #[bits(1)]
        pub ivybridge: u64,
        #[bits(1)]
        pub valleyview: u64,
        #[bits(1)]
        pub haswell: u64,
        #[bits(1)]
        pub haswell_ult: u64,
        #[bits(1)]
        pub haswell_ulx: u64,
        #[bits(1)]
        pub broadwell: u64,
        #[bits(1)]
        pub broadwell_ult: u64,
        #[bits(1)]
        pub broadwell_ulx: u64,
        #[bits(1)]
        pub cherryview: u64,
        #[bits(1)]
        pub skylake: u64,
        #[bits(1)]
        pub skylake_ult: u64,
        #[bits(1)]
        pub skylake_ulx: u64,
        #[bits(1)]
        pub broxton: u64,
        #[bits(1)]
        pub kabylake: u64,
        #[bits(1)]
        pub kabylake_ult: u64,
        #[bits(1)]
        pub kabylake_ulx: u64,
        #[bits(1)]
        pub geminilake: u64,
        #[bits(1)]
        pub coffeelake: u64,
        #[bits(1)]
        pub coffeelake_ult: u64,
        #[bits(1)]
        pub coffeelake_ulx: u64,
        #[bits(1)]
        pub cometlake: u64,
        #[bits(1)]
        pub cometlake_ult: u64,
        #[bits(1)]
        pub cometlake_ulx: u64,
        #[bits(1)]
        pub icelake: u64,
        #[bits(1)]
        pub icelake_port_f: u64,
        #[bits(1)]
        pub jasperlake: u64,
        #[bits(1)]
        pub elkhartlake: u64,
        #[bits(1)]
        pub tigerlake: u64,
        #[bits(1)]
        pub tigerlake_uy: u64,
        #[bits(1)]
        pub rocketlake: u64,
        #[bits(1)]
        pub dg1: u64,
        #[bits(1)]
        pub alderlake_s: u64,
        #[bits(1)]
        pub alderlake_s_raptorlake_s: u64,
        #[bits(1)]
        pub alderlake_p: u64,
        #[bits(1)]
        pub alderlake_p_alderlake_n: u64,
        #[bits(1)]
        pub alderlake_p_raptorlake_p: u64,
        #[bits(1)]
        pub alderlake_p_raptorlake_u: u64,
        #[bits(1)]
        pub dg2: u64,
        #[bits(1)]
        pub dg2_g10: u64,
        #[bits(1)]
        pub dg2_g11: u64,
        #[bits(1)]
        pub dg2_g12: u64,
        #[bits(1)]
        pub meteorlake: u64,
        #[bits(1)]
        pub meteorlake_u: u64,
        #[bits(1)]
        pub lunarlake: u64,
        #[bits(1)]
        pub battlemage: u64,
        #[bits(1)]
        pub pantherlake: u64,
        #[bits(1)]
        pub pantherlake_wildcatlake: u64,
        #[bits(1)]
        pub novalake: u64,
        #[bits(5, access = na)]
        pub __slate_pad_67: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits932U0 {
        #[bits(1)]
        pub cursor_needs_physical: u8,
        #[bits(1)]
        pub has_cdclk_crawl: u8,
        #[bits(1)]
        pub has_cdclk_squash: u8,
        #[bits(1)]
        pub has_ddi: u8,
        #[bits(1)]
        pub has_dp_mst: u8,
        #[bits(1)]
        pub has_dsb: u8,
        #[bits(1)]
        pub has_fpga_dbg: u8,
        #[bits(1)]
        pub has_gmch: u8,
        #[bits(1)]
        pub has_hotplug: u8,
        #[bits(1)]
        pub has_hti: u8,
        #[bits(1)]
        pub has_ipc: u8,
        #[bits(1)]
        pub has_overlay: u8,
        #[bits(1)]
        pub has_psr: u8,
        #[bits(1)]
        pub has_psr_hw_tracking: u8,
        #[bits(1)]
        pub overlay_needs_physical: u8,
        #[bits(1)]
        pub supports_tv: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1029U0 {
        #[bits(1)]
        pub int_tv_support: u32,
        #[bits(1)]
        pub int_crt_support: u32,
        #[bits(1)]
        pub lvds_use_ssc: u32,
        #[bits(1)]
        pub int_lvds_support: u32,
        #[bits(1)]
        pub display_clock_mode: u32,
        #[bits(1)]
        pub fdi_rx_polarity_inverted: u32,
        #[bits(2, access = na)]
        pub __slate_pad_6: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits443U0 {
        #[bits(1)]
        pub is_added: u32,
        #[bits(1)]
        pub unsafe_warn: u32,
        #[bits(1)]
        pub flit_mode: u32,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits974U0 {
        #[bits(1)]
        pub per_func_slot: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits772U0 {
        #[bits(3)]
        pub pcie_mpss: u8,
        #[bits(5, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield([u8; 3], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits772U1 {
        #[bits(5)]
        pub pme_support: u32,
        #[bits(1)]
        pub pme_poll: u32,
        #[bits(1)]
        pub pinned: u32,
        #[bits(1)]
        pub config_rrs_sv: u32,
        #[bits(1)]
        pub imm_ready: u32,
        #[bits(1)]
        pub d1_support: u32,
        #[bits(1)]
        pub d2_support: u32,
        #[bits(1)]
        pub no_d1d2: u32,
        #[bits(1)]
        pub no_d3cold: u32,
        #[bits(1)]
        pub bridge_d3: u32,
        #[bits(1)]
        pub d3cold_allowed: u32,
        #[bits(1)]
        pub mmio_always_on: u32,
        #[bits(1)]
        pub wakeup_prepared: u32,
        #[bits(1)]
        pub skip_bus_pm: u32,
        #[bits(1)]
        pub ignore_hotplug: u32,
        #[bits(1)]
        pub hotplug_user_indicators: u32,
        #[bits(1)]
        pub clear_retrain_link: u32,
        #[bits(1)]
        pub no_bw_notif: u32,
        #[bits(2, access = na)]
        pub __slate_pad_18: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits772U2 {
        #[bits(1)]
        pub aspm_l0s_support: u32,
        #[bits(1)]
        pub aspm_l1_support: u32,
        #[bits(1)]
        pub ltr_path: u32,
        #[bits(1)]
        pub pasid_no_tlp: u32,
        #[bits(3)]
        pub eetlp_prefix_max: u32,
        #[bits(1, access = na)]
        pub __slate_pad_5: u8,
    }
    #[bitfields::bitfield([u8; 6], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits772U3 {
        #[bits(1)]
        pub transparent: u32,
        #[bits(1)]
        pub io_window: u32,
        #[bits(1)]
        pub pref_window: u32,
        #[bits(1)]
        pub pref_64_window: u32,
        #[bits(1)]
        pub multifunction: u32,
        #[bits(1)]
        pub is_busmaster: u32,
        #[bits(1)]
        pub no_msi: u32,
        #[bits(1)]
        pub block_cfg_access: u32,
        #[bits(1)]
        pub broken_parity_status: u32,
        #[bits(2)]
        pub irq_reroute_variant: u32,
        #[bits(1)]
        pub msi_enabled: u32,
        #[bits(1)]
        pub msix_enabled: u32,
        #[bits(1)]
        pub ari_enabled: u32,
        #[bits(1)]
        pub ats_enabled: u32,
        #[bits(1)]
        pub pasid_enabled: u32,
        #[bits(1)]
        pub pri_enabled: u32,
        #[bits(1)]
        pub tph_enabled: u32,
        #[bits(1)]
        pub fm_enabled: u32,
        #[bits(1)]
        pub is_managed: u32,
        #[bits(1)]
        pub is_msi_managed: u32,
        #[bits(1)]
        pub needs_freset: u32,
        #[bits(1)]
        pub state_saved: u32,
        #[bits(1)]
        pub is_physfn: u32,
        #[bits(1)]
        pub is_virtfn: u32,
        #[bits(1)]
        pub is_hotplug_bridge: u32,
        #[bits(1)]
        pub is_pciehp: u32,
        #[bits(1)]
        pub shpc_managed: u32,
        #[bits(1)]
        pub is_thunderbolt: u32,
        #[bits(1)]
        pub is_cxl: u32,
        #[bits(1)]
        pub untrusted: u32,
        #[bits(1)]
        pub external_facing: u32,
        #[bits(1)]
        pub broken_intx_masking: u32,
        #[bits(1)]
        pub io_window_1k: u32,
        #[bits(1)]
        pub irq_managed: u32,
        #[bits(1)]
        pub non_compliant_bars: u32,
        #[bits(1)]
        pub is_probed: u32,
        #[bits(1)]
        pub link_active_reporting: u32,
        #[bits(1)]
        pub no_vf_scan: u32,
        #[bits(1)]
        pub no_command_memory: u32,
        #[bits(1)]
        pub rom_bar_overlap: u32,
        #[bits(1)]
        pub rom_attr_enabled: u32,
        #[bits(1)]
        pub non_mappable_bars: u32,
        #[bits(5, access = na)]
        pub __slate_pad_42: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits772U4 {
        #[bits(1)]
        pub pasid_required: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct intel_crtc_scaler_state {
    scalers: [intel_scaler; 2],
    scaler_users: u32,
    scaler_id: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_wm_level {
    enable: bool,
    pri_val: u32,
    spr_val: u32,
    cur_val: u32,
    fbc_val: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_pipe_wm {
    wm: [intel_wm_level; 5],
    fbc_wm_enabled: bool,
    pipe_enabled: bool,
    sprites_enabled: bool,
    sprites_scaled: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct skl_wm_level {
    min_ddb_alloc: u16,
    min_ddb_alloc_uv: u16,
    blocks: u16,
    lines: u8,
    enable: bool,
    ignore_lines: bool,
    auto_min_alloc_wm_enable: bool,
    can_sagv: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct skl_plane_wm {
    wm: [skl_wm_level; 8],
    trans_wm: skl_wm_level,
    sagv: __SlateRecord_intel_display_types_h_853,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct skl_pipe_wm {
    planes: [skl_plane_wm; 8],
    use_sagv_wm: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vlv_wm_state {
    wm: [g4x_pipe_wm; 3],
    sr: [g4x_sr_wm; 3],
    num_levels: u8,
    cxsr: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vlv_fifo_state {
    plane: [u16; 8],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct g4x_wm_state {
    wm: g4x_pipe_wm,
    sr: g4x_sr_wm,
    hpll: g4x_sr_wm,
    cxsr: bool,
    hpll_en: bool,
    fbc_en: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_crtc_wm_state {
    __slate_anon_0: __SlateRecord_intel_display_types_h_899,
    need_postvbl_update: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_link_m_n {
    tu: u32,
    data_m: u32,
    data_n: u32,
    link_m: u32,
    link_n: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_csc_matrix {
    coeff: [u16; 9],
    preoff: [u16; 3],
    postoff: [u16; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct scaler_filter_coeff {
    sign: u16,
    exp: u16,
    mantissa: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_casf {
    coeff: [scaler_filter_coeff; 7],
    strength: u8,
    win_size: u8,
    enable: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_crtc_state {
    uapi: drm_crtc_state,
    hw: __SlateRecord_intel_display_types_h_1036,
    pre_csc_lut: *mut drm_property_blob,
    post_csc_lut: *mut drm_property_blob,
    csc: intel_csc_matrix,
    output_csc: intel_csc_matrix,
    quirks: u64,
    fb_bits: u32,
    update_pipe: bool,
    update_m_n: bool,
    update_lrr: bool,
    disable_cxsr: bool,
    update_wm_pre: bool,
    update_wm_post: bool,
    fifo_changed: bool,
    preload_luts: bool,
    inherited: bool,
    do_async_flip: bool,
    pipe_src: drm_rect,
    pixel_rate: u32,
    pixel_rate_cdclk: u32,
    has_pch_encoder: bool,
    has_infoframe: bool,
    cpu_transcoder: i32,
    limited_color_range: bool,
    output_types: u32,
    has_hdmi_sink: bool,
    has_audio: bool,
    dither: bool,
    dither_force_disable: bool,
    clock_set: bool,
    sdvo_tv_clock: bool,
    bw_constrained: bool,
    dpll: dpll,
    intel_dpll: *mut intel_dpll,
    dpll_hw_state: intel_dpll_hw_state,
    icl_port_dplls: [icl_port_dpll; 2],
    dsi_pll: __SlateRecord_intel_display_types_h_1171,
    max_link_bpp_x16: i32,
    max_pipe_bpp: i32,
    pipe_bpp: i32,
    min_hblank: i32,
    dp_m_n: intel_link_m_n,
    dp_m2_n2: intel_link_m_n,
    has_drrs: bool,
    has_psr: bool,
    has_sel_update: bool,
    enable_psr2_sel_fetch: bool,
    enable_psr2_su_region_et: bool,
    clear_psr2_sel_fetch: bool,
    req_psr2_sdp_prior_scanline: bool,
    has_panel_replay: bool,
    link_off_after_as_sdp_when_pr_active: bool,
    disable_as_sdp_when_pr_active: bool,
    wm_level_disabled: bool,
    pkg_c_latency_used: bool,
    panel_replay_dsc_support: u32,
    su_y_granularity: u16,
    active_non_psr_pipes: u8,
    entry_setup_frames: u8,
    no_psr_reason: *const u8,
    port_clock: i32,
    pixel_multiplier: u32,
    mode_flags: u8,
    lane_count: u8,
    lane_lat_optim_mask: u8,
    min_voltage_level: u8,
    gmch_pfit: __SlateRecord_intel_display_types_h_1230,
    pch_pfit: __SlateRecord_intel_display_types_h_1237,
    fdi_lanes: i32,
    fdi_m_n: intel_link_m_n,
    ips_enabled: bool,
    crc_enabled: bool,
    double_wide: bool,
    scaler_state: intel_crtc_scaler_state,
    hsw_workaround_pipe: i32,
    wm: intel_crtc_wm_state,
    min_cdclk: i32,
    plane_min_cdclk: [i32; 8],
    data_rate: [u32; 8],
    data_rate_y: [u32; 8],
    rel_data_rate: [u64; 8],
    rel_data_rate_y: [u64; 8],
    gamma_mode: u32,
    __slate_anon_84: __SlateRecord_intel_display_types_h_1277,
    enabled_planes: u8,
    active_planes: u8,
    scaled_planes: u8,
    nv12_planes: u8,
    c8_planes: u8,
    update_planes: u8,
    async_flip_planes: u8,
    framestart_delay: u8,
    msa_timing_delay: u8,
    infoframes: __SlateRecord_intel_display_types_h_1303,
    eld: [u8; 128],
    hdmi_scrambling: bool,
    hdmi_high_tmds_clock_ratio: bool,
    output_format: u32,
    sink_format: u32,
    gamma_enable: bool,
    csc_enable: bool,
    wgc_enable: bool,
    joiner_pipes: u8,
    dsc: __SlateRecord_intel_display_types_h_1347,
    dp_tunnel_ref: drm_dp_tunnel_ref,
    linetime: u16,
    ips_linetime: u16,
    enhanced_framing: bool,
    fec_enable: bool,
    sdp_split_enable: bool,
    master_transcoder: i32,
    sync_mode_slaves_mask: u16,
    mst_master_transcoder: i32,
    dsb_color: *mut intel_dsb,
    dsb_commit: *mut intel_dsb,
    use_dsb: bool,
    use_flipq: bool,
    psr2_man_track_ctl: u32,
    pipe_srcsz_early_tpt: u32,
    psr2_su_area: drm_rect,
    vrr: __SlateRecord_intel_display_types_h_1401,
    cmrr: __SlateRecord_intel_display_types_h_1416,
    splitter: __SlateRecord_intel_display_types_h_1422,
    vblank_work: drm_vblank_work,
    has_lobf: bool,
    set_context_latency: u16,
    alpm_state: __SlateRecord_intel_display_types_h_1437,
    plane_color_changed: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_pipe_crc {
    lock: spinlock,
    skipped: i32,
    source: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_flipq {
    start_mmioaddr: u32,
    flipq_id: u32,
    tail: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_crtc {
    base: drm_crtc,
    pipe_head: list_head,
    pipe: i32,
    active: bool,
    plane_ids_mask: u8,
    mode_flags: u8,
    vmax_vblank_start: u16,
    enabled_power_domains: intel_display_power_domain_set,
    hw_readout_power_domains: intel_display_power_domain_set,
    overlay: *mut intel_overlay,
    config: *mut intel_crtc_state,
    flip_done_event: *mut drm_pending_vblank_event,
    dsb_event: *mut drm_pending_vblank_event,
    flipq_event: *mut drm_pending_vblank_event,
    cpu_fifo_underrun_disabled: bool,
    pch_fifo_underrun_disabled: bool,
    flipq: [intel_flipq; 5],
    wm: __SlateRecord_intel_display_types_h_1535,
    drrs: __SlateRecord_intel_display_types_h_1544,
    dc_balance: __SlateRecord_intel_display_types_h_1554,
    scanline_offset: i32,
    debug: __SlateRecord_intel_display_types_h_1560,
    num_scalers: i32,
    vblank_pm_qos: pm_qos_request,
    pipe_crc: intel_pipe_crc,
    vblank_psr_notify: bool,
    cmtg: __SlateRecord_intel_display_types_h_1588,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_hdmi {
    hdmi_reg: __SlateRecord_i915_reg_defs_h_14,
    dp_dual_mode: __SlateRecord_intel_display_types_h_1683,
    attached_connector: *mut intel_connector,
    cec_notifier: *mut cec_notifier,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dp_compliance_data {
    edid: u64,
    video_pattern: u8,
    hdisplay: u16,
    vdisplay: u16,
    bpc: u8,
    phytest: drm_dp_phy_test_params,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dp_compliance {
    test_type: u64,
    test_data: intel_dp_compliance_data,
    test_active: bool,
    test_link_rate: i32,
    test_lane_count: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dp_pcon_frl {
    is_trained: bool,
    trained_rate_gbps: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_pps {
    panel_power_up_delay: i32,
    panel_power_down_delay: i32,
    panel_power_cycle_delay: i32,
    backlight_on_delay: i32,
    backlight_off_delay: i32,
    panel_vdd_work: delayed_work,
    want_panel_vdd: bool,
    initializing: bool,
    last_power_on: u64,
    last_backlight_off: u64,
    panel_power_off_time: i64,
    vdd_wakeref: *mut ref_tracker,
    __slate_anon_12: __SlateRecord_intel_display_types_h_1728,
    vlv_active_pipe: i32,
    bxt_pps_reset: bool,
    pps_delays: intel_pps_delays,
    bios_pps_delays: intel_pps_delays,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_psr {
    lock: mutex,
    debug: u32,
    sink_support: bool,
    source_support: bool,
    enabled: bool,
    pause_counter: i32,
    pipe: i32,
    transcoder: i32,
    active: bool,
    work: work_struct,
    busy_frontbuffer_bits: u32,
    link_standby: bool,
    sel_update_enabled: bool,
    psr2_sel_fetch_enabled: bool,
    psr2_sel_fetch_cff_enabled: bool,
    su_region_et_enabled: bool,
    req_psr2_sdp_prior_scanline: bool,
    last_entry_attempt: i64,
    last_exit: i64,
    sink_not_reliable: bool,
    irq_aux_error: bool,
    dc3co_allowed: bool,
    dc3co_work: delayed_work,
    su_w_granularity: u16,
    su_y_granularity: u16,
    source_panel_replay_support: bool,
    sink_panel_replay_support: bool,
    panel_replay_enabled: bool,
    dc3co_exit_delay: u32,
    entry_setup_frames: u8,
    io_wake_lines: u8,
    fast_wake_lines: u8,
    link_ok: bool,
    pkg_c_latency_used: bool,
    active_non_psr_pipes: u8,
    no_psr_reason: *const u8,
    vblank_wakeref: *mut ref_tracker,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dp {
    output_reg: __SlateRecord_i915_reg_defs_h_14,
    DP: u32,
    link_rate: i32,
    lane_count: u8,
    sink_count: u8,
    downstream_port_changed: bool,
    needs_modeset_retry: bool,
    use_max_params: bool,
    dpcd: [u8; 15],
    downstream_ports: [u8; 16],
    edp_dpcd: [u8; 5],
    lttpr_common_caps: [u8; 8],
    lttpr_phy_caps: [[u8; 3]; 8],
    pcon_dsc_dpcd: [u8; 13],
    num_source_rates: i32,
    source_rates: *const i32,
    num_sink_rates: i32,
    sink_rates: [i32; 8],
    use_rate_select: bool,
    max_sink_lane_count: i32,
    link: __SlateRecord_intel_display_types_h_1848,
    reset_link_params: bool,
    mso_link_count: i32,
    mso_pixel_overlap: i32,
    desc: drm_dp_desc,
    aux: drm_dp_aux,
    aux_busy_last_status: u32,
    train_set: [u8; 4],
    pps: intel_pps,
    is_mst: bool,
    mst_detect: u32,
    attached_connector: *mut intel_connector,
    as_sdp_supported: bool,
    as_sdp_v2_supported: bool,
    tunnel: *mut drm_dp_tunnel,
    __slate_bits_0: __slate_bits::__SlateBits803U0,
    disabled_uhbr_lane_mask: u8,
    mst: __SlateRecord_intel_display_types_h_1885,
    get_aux_clock_divider: Option<unsafe extern "C" fn(*mut intel_dp, i32) -> u32>,
    get_aux_send_ctl: Option<unsafe extern "C" fn(*mut intel_dp, i32, u32) -> u32>,
    aux_ch_ctl_reg: Option<unsafe extern "C" fn(*mut intel_dp) -> __SlateRecord_i915_reg_defs_h_14>,
    aux_ch_data_reg: Option<unsafe extern "C" fn(*mut intel_dp, i32) -> __SlateRecord_i915_reg_defs_h_14>,
    prepare_link_retrain: Option<unsafe extern "C" fn(*mut intel_dp, *const intel_crtc_state)>,
    set_link_train: Option<unsafe extern "C" fn(*mut intel_dp, *const intel_crtc_state, u8)>,
    set_idle_link_train: Option<unsafe extern "C" fn(*mut intel_dp, *const intel_crtc_state)>,
    preemph_max: Option<unsafe extern "C" fn(*mut intel_dp) -> u8>,
    voltage_max: Option<unsafe extern "C" fn(*mut intel_dp, *const intel_crtc_state) -> u8>,
    compliance: intel_dp_compliance,
    dfp: __SlateRecord_intel_display_types_h_1919,
    pm_qos: pm_qos_request,
    force_dsc_en: bool,
    force_dsc_output_format: i32,
    force_dsc_fractional_bpp_en: bool,
    force_dsc_bpc: i32,
    hobl_failed: bool,
    hobl_active: bool,
    frl: intel_dp_pcon_frl,
    psr: intel_psr,
    last_oui_write: u64,
    oui_valid: bool,
    colorimetry_support: bool,
    sst_split_sdp_support: bool,
    alpm: __SlateRecord_intel_display_types_h_1953,
    alpm_dpcd: u8,
    quirks: __SlateRecord_intel_display_types_h_1963,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_lspcon {
    active: bool,
    hdr_supported: bool,
    mode: u32,
    vendor: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_digital_port {
    base: intel_encoder,
    dp: intel_dp,
    hdmi: intel_hdmi,
    lspcon: intel_lspcon,
    hpd_pulse: Option<unsafe extern "C" fn(*mut intel_digital_port, bool) -> u32>,
    lane_reversal: bool,
    ddi_a_4_lanes: bool,
    release_cl2_override: bool,
    dedicated_external: bool,
    max_lanes: u8,
    aux_ch: i32,
    ddi_io_power_domain: u32,
    ddi_io_wakeref: *mut ref_tracker,
    aux_wakeref: *mut ref_tracker,
    tc: *mut intel_tc_port,
    hdcp: __SlateRecord_intel_display_types_h_2000,
    write_infoframe: Option<unsafe extern "C" fn(*mut intel_encoder, *const intel_crtc_state, u32, *const (), i64)>,
    read_infoframe: Option<unsafe extern "C" fn(*mut intel_encoder, *const intel_crtc_state, u32, *mut (), i64)>,
    set_infoframes: Option<unsafe extern "C" fn(*mut intel_encoder, bool, *const intel_crtc_state, *const drm_connector_state)>,
    infoframes_enabled: Option<unsafe extern "C" fn(*mut intel_encoder, *const intel_crtc_state) -> u32>,
    connected: Option<unsafe extern "C" fn(*mut intel_encoder) -> bool>,
    lock: Option<unsafe extern "C" fn(*mut intel_digital_port)>,
    unlock: Option<unsafe extern "C" fn(*mut intel_digital_port)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct intel_dp_mst_encoder {
    base: intel_encoder,
    pipe: i32,
    primary: *mut intel_digital_port,
    connector: *mut intel_connector,
}


fn intel_encoder_is_dig_port(encoder: *mut intel_encoder) -> bool {
match (unsafe { (*encoder).r#type }) as u32 {
6..=8 | 10 => {
return true;
}
_ => {
return false;
}
}
unsafe { core::mem::zeroed() }
}

fn intel_encoder_is_mst(encoder: *mut intel_encoder) -> bool {
((unsafe { (*encoder).r#type }) as u32) == 11
}

fn enc_to_mst(encoder: *mut intel_encoder) -> *mut intel_dp_mst_encoder {
let mut __v0: *mut intel_dp_mst_encoder;
__v0 = ((unsafe { ((core::ptr::addr_of_mut!((*encoder).base) as *mut ()) as *const u8).wrapping_offset(-(offsetof__6!(intel_dp_mst_encoder) as isize)) }) as *mut ()) as *mut intel_dp_mst_encoder;
__v0
}

fn enc_to_dig_port(encoder: *mut intel_encoder) -> *mut intel_digital_port {
let intel_encoder: *mut intel_encoder = encoder;
if intel_encoder_is_dig_port(intel_encoder) {
let mut __v0: *mut intel_digital_port;
__v0 = ((unsafe { ((core::ptr::addr_of_mut!((*encoder).base) as *mut ()) as *const u8).wrapping_offset(-(offsetof__6!(intel_digital_port) as isize)) }) as *mut ()) as *mut intel_digital_port;
return __v0;
} else {
if intel_encoder_is_mst(intel_encoder) {
return unsafe { (*enc_to_mst(encoder)).primary };
} else {
return core::ptr::null_mut::<intel_digital_port>();
}
}
}

fn enc_to_intel_dp(encoder: *mut intel_encoder) -> *mut intel_dp {
unsafe { core::ptr::addr_of_mut!((*enc_to_dig_port(encoder)).dp) }
}

fn dp_to_dig_port(intel_dp: *mut intel_dp) -> *mut intel_digital_port {
container_of!(intel_dp, intel_digital_port)
}

fn hdmi_to_dig_port(intel_hdmi: *mut intel_hdmi) -> *mut intel_digital_port {
container_of__2!(intel_hdmi, intel_digital_port)
}

fn intel_crtc_has_type(crtc_state: *const intel_crtc_state, r#type: u32) -> bool {
((unsafe { (*crtc_state).output_types }) as u64) & BIT__3!(r#type) != 0
}

fn intel_crtc_has_dp_encoder(crtc_state: *const intel_crtc_state) -> bool {
((unsafe { (*crtc_state).output_types }) as u64) & (BIT!(7i32) | BIT!(11i32) | BIT!(8i32)) != 0
}

// Helper for generic association. Map types to conversion functions/macros.
#[unsafe(no_mangle)]
extern "C" fn intel_lt_phy_calc_port_clock(display: *mut intel_display, lt_state: *const intel_lt_phy_pll_state) -> i32 {
let mut clk: i32 = 0;
let mut mode: u8 = 0;
let mut rate: u8 = 0;
let mut __v0: u8;
loop {
if (__bf_cast_unsigned__2!((((const { core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1721_3>() }) as u64) as i32).wrapping_add((((1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 0i32 & (((1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> ((const { core::mem::size_of::<u8>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2)) as u8) as i32) > (__bf_cast_unsigned__2!(18446744073709551615u64) as i32) {
unsafe { __compiletime_assert_719() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v2: u8;
loop {
if false {
unsafe { __compiletime_assert_720() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (((((const { core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1721_5>() }) as u64) as i32).wrapping_add((((1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 0i32 & (((1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> ((const { core::mem::size_of::<u8>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2)) as u8) as i32) == 0 {
unsafe { __compiletime_assert_721() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { (!((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1721_6>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 0i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2)) as u8) as i32) >> ((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1721_7>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 0i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2)) as u8) as u64).trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
unsafe { __compiletime_assert_722() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1721_8>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 0i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2)) as u8) as i32) as u64).wrapping_add(1u64 << ((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1721_9>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 0i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2)) as u8) as u64).trailing_zeros() as i32)) & ((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1721_10>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 0i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2)) as u8) as i32) as u64).wrapping_add(1u64 << ((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1721_11>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 0i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2)) as u8) as u64).trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_723() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = ((((unsafe { *((*lt_state).config.as_ptr() as *const u8).wrapping_offset(0 as isize) }) as i32) & (((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1721_13>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 0i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2)) as u8) as i32)) >> ((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1721_14>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 0i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2)) as u8) as u64).trailing_zeros() as i32)) as u8;
__v0 = __v2;
mode = __v0;
// For edp/dp read the clock value from the tables
// and return the clock as the algorithm used for
// calculating the port clock does not exactly matches
// with edp/dp clock.
if (mode as i32) == 3 {
let mut __v7: u8;
loop {
if (__bf_cast_unsigned__2!((((const { core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1730_3>() }) as u64) as i32).wrapping_add((((1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 3i32 & (((1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> ((const { core::mem::size_of::<u8>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6)) as u8) as i32) > (__bf_cast_unsigned__2!(18446744073709551615u64) as i32) {
unsafe { __compiletime_assert_724() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v9: u8;
loop {
if false {
unsafe { __compiletime_assert_725() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (((((const { core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1730_5>() }) as u64) as i32).wrapping_add((((1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 3i32 & (((1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> ((const { core::mem::size_of::<u8>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6)) as u8) as i32) == 0 {
unsafe { __compiletime_assert_726() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { (!((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1730_6>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 3i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6)) as u8) as i32) >> ((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1730_7>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 3i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6)) as u8) as u64).trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
unsafe { __compiletime_assert_727() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1730_8>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 3i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6)) as u8) as i32) as u64).wrapping_add(1u64 << ((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1730_9>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 3i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6)) as u8) as u64).trailing_zeros() as i32)) & ((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1730_10>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 3i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6)) as u8) as i32) as u64).wrapping_add(1u64 << ((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1730_11>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 3i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6)) as u8) as u64).trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_728() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v9 = ((((unsafe { *((*lt_state).config.as_ptr() as *const u8).wrapping_offset(0 as isize) }) as i32) & (((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1730_13>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 3i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6)) as u8) as i32)) >> ((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1730_14>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 3i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6)) as u8) as u64).trailing_zeros() as i32)) as u8;
__v7 = __v9;
rate = __v7;
clk = intel_lt_phy_get_dp_clock(rate) as i32;
} else {
if (mode as i32) == 4 {
clk = intel_lt_phy_calc_hdmi_port_clock(display, lt_state);
} else {
let mut __v14: i64;
let __ret_warn_on: i32 = !(c"Unsupported LT PHY Mode!\n".as_ptr() as *mut u8).is_null() as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(\"Unsupported LT PHY Mode!\\n\")".as_ptr() as *mut u8) };
let mut __v15: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v15 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v16: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_intel_lt_phy_calc_port_clock_17, sym __slate_asm_string_intel_lt_phy_calc_port_clock_18, const 1736 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v16 = bug;
unsafe { __v15.unwrap()(__v16, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(\"Unsupported LT PHY Mode!\\n\")".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v14 = (__ret_warn_on != 0) as i64;
clk = 25200;
}
}
clk
}

#[repr(C)]
#[derive(Clone, Copy)]
struct phy_param_t {
    val: u32,
    addr: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct lt_phy_params {
    pll_reg4: phy_param_t,
    pll_reg3: phy_param_t,
    pll_reg5: phy_param_t,
    pll_reg57: phy_param_t,
    lf: phy_param_t,
    tdc: phy_param_t,
    ssc: phy_param_t,
    bias2: phy_param_t,
    bias_trim: phy_param_t,
    dco_med: phy_param_t,
    dco_fine: phy_param_t,
    ssc_inj: phy_param_t,
    surv_bonus: phy_param_t,
}


static xe3plpd_lt_dp_rbr: intel_lt_phy_pll_state = {
let mut __t0: intel_lt_phy_pll_state = unsafe { core::mem::zeroed() };
__t0.addr_msb = [135u8, 135u8, 135u8, 135u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8];
__t0.addr_lsb = [16u8, 12u8, 20u8, 228u8, 12u8, 16u8, 20u8, 24u8, 72u8, 64u8, 76u8, 36u8, 68u8];
__t0.data = [[0u8, 76u8, 2u8, 0u8], [5u8, 10u8, 42u8, 32u8], [128u8, 0u8, 0u8, 0u8], [4u8, 4u8, 130u8, 40u8], [250u8, 22u8, 131u8, 17u8], [128u8, 15u8, 249u8, 83u8], [132u8, 38u8, 5u8, 4u8], [0u8, 224u8, 1u8, 0u8], [75u8, 72u8, 0u8, 0u8], [39u8, 8u8, 0u8, 0u8], [90u8, 19u8, 41u8, 19u8], [0u8, 91u8, 224u8, 10u8], [0u8, 0u8, 0u8, 0u8]];
__t0.config = [131u8, 45u8, 0u8];
__t0
};

static xe3plpd_lt_dp_hbr1: intel_lt_phy_pll_state = {
let mut __t0: intel_lt_phy_pll_state = unsafe { core::mem::zeroed() };
__t0.addr_msb = [135u8, 135u8, 135u8, 135u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8];
__t0.addr_lsb = [16u8, 12u8, 20u8, 228u8, 12u8, 16u8, 20u8, 24u8, 72u8, 64u8, 76u8, 36u8, 68u8];
__t0.data = [[0u8, 76u8, 2u8, 0u8], [3u8, 202u8, 52u8, 160u8], [224u8, 0u8, 0u8, 0u8], [5u8, 4u8, 129u8, 173u8], [250u8, 17u8, 131u8, 17u8], [128u8, 15u8, 249u8, 83u8], [132u8, 38u8, 7u8, 4u8], [0u8, 224u8, 1u8, 0u8], [67u8, 72u8, 0u8, 0u8], [39u8, 8u8, 0u8, 0u8], [90u8, 19u8, 41u8, 19u8], [0u8, 91u8, 224u8, 13u8], [0u8, 0u8, 0u8, 0u8]];
__t0.config = [139u8, 45u8, 0u8];
__t0
};

static xe3plpd_lt_dp_hbr2: intel_lt_phy_pll_state = {
let mut __t0: intel_lt_phy_pll_state = unsafe { core::mem::zeroed() };
__t0.addr_msb = [135u8, 135u8, 135u8, 135u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8];
__t0.addr_lsb = [16u8, 12u8, 20u8, 228u8, 12u8, 16u8, 20u8, 24u8, 72u8, 64u8, 76u8, 36u8, 68u8];
__t0.data = [[0u8, 76u8, 2u8, 0u8], [1u8, 77u8, 52u8, 160u8], [224u8, 0u8, 0u8, 0u8], [10u8, 4u8, 129u8, 218u8], [250u8, 17u8, 131u8, 17u8], [128u8, 15u8, 249u8, 83u8], [132u8, 38u8, 7u8, 4u8], [0u8, 224u8, 1u8, 0u8], [67u8, 72u8, 0u8, 0u8], [39u8, 8u8, 0u8, 0u8], [90u8, 19u8, 41u8, 19u8], [0u8, 91u8, 224u8, 13u8], [0u8, 0u8, 0u8, 0u8]];
__t0.config = [147u8, 45u8, 0u8];
__t0
};

static xe3plpd_lt_dp_hbr3: intel_lt_phy_pll_state = {
let mut __t0: intel_lt_phy_pll_state = unsafe { core::mem::zeroed() };
__t0.addr_msb = [135u8, 135u8, 135u8, 135u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8];
__t0.addr_lsb = [16u8, 12u8, 20u8, 228u8, 12u8, 16u8, 20u8, 24u8, 72u8, 64u8, 76u8, 36u8, 68u8];
__t0.data = [[0u8, 76u8, 2u8, 0u8], [1u8, 74u8, 52u8, 160u8], [224u8, 0u8, 0u8, 0u8], [5u8, 4u8, 128u8, 168u8], [250u8, 17u8, 131u8, 17u8], [128u8, 15u8, 249u8, 83u8], [132u8, 38u8, 7u8, 4u8], [0u8, 224u8, 1u8, 0u8], [67u8, 72u8, 0u8, 0u8], [39u8, 8u8, 0u8, 0u8], [90u8, 19u8, 41u8, 19u8], [0u8, 91u8, 224u8, 13u8], [0u8, 0u8, 0u8, 0u8]];
__t0.config = [155u8, 45u8, 0u8];
__t0
};

static xe3plpd_lt_dp_uhbr10: intel_lt_phy_pll_state = {
let mut __t0: intel_lt_phy_pll_state = unsafe { core::mem::zeroed() };
__t0.addr_msb = [133u8, 133u8, 133u8, 133u8, 134u8, 134u8, 134u8, 134u8, 134u8, 134u8, 134u8, 134u8, 134u8];
__t0.addr_lsb = [16u8, 12u8, 20u8, 228u8, 12u8, 16u8, 20u8, 24u8, 72u8, 64u8, 76u8, 36u8, 68u8];
__t0.data = [[0u8, 76u8, 2u8, 0u8], [1u8, 10u8, 32u8, 128u8], [106u8, 170u8, 170u8, 171u8], [0u8, 3u8, 4u8, 148u8], [250u8, 28u8, 131u8, 17u8], [128u8, 15u8, 249u8, 83u8], [132u8, 38u8, 4u8, 4u8], [0u8, 224u8, 1u8, 0u8], [69u8, 72u8, 0u8, 0u8], [39u8, 8u8, 0u8, 0u8], [90u8, 20u8, 42u8, 20u8], [0u8, 91u8, 224u8, 8u8], [0u8, 0u8, 0u8, 0u8]];
__t0.config = [67u8, 45u8, 0u8];
__t0
};

static xe3plpd_lt_dp_uhbr13_5: intel_lt_phy_pll_state = {
let mut __t0: intel_lt_phy_pll_state = unsafe { core::mem::zeroed() };
__t0.addr_msb = [135u8, 135u8, 135u8, 135u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8];
__t0.addr_lsb = [16u8, 12u8, 20u8, 228u8, 12u8, 16u8, 20u8, 24u8, 72u8, 64u8, 76u8, 36u8, 68u8];
__t0.data = [[0u8, 76u8, 2u8, 0u8], [2u8, 9u8, 43u8, 224u8], [144u8, 0u8, 0u8, 0u8], [8u8, 4u8, 128u8, 224u8], [250u8, 21u8, 131u8, 17u8], [128u8, 15u8, 249u8, 83u8], [132u8, 38u8, 6u8, 4u8], [0u8, 224u8, 1u8, 0u8], [73u8, 72u8, 0u8, 0u8], [39u8, 8u8, 0u8, 0u8], [90u8, 19u8, 41u8, 19u8], [0u8, 87u8, 224u8, 12u8], [0u8, 0u8, 0u8, 0u8]];
__t0.config = [203u8, 45u8, 0u8];
__t0
};

static xe3plpd_lt_dp_uhbr20: intel_lt_phy_pll_state = {
let mut __t0: intel_lt_phy_pll_state = unsafe { core::mem::zeroed() };
__t0.addr_msb = [133u8, 133u8, 133u8, 133u8, 134u8, 134u8, 134u8, 134u8, 134u8, 134u8, 134u8, 134u8, 134u8];
__t0.addr_lsb = [16u8, 12u8, 20u8, 228u8, 12u8, 16u8, 20u8, 24u8, 72u8, 64u8, 76u8, 36u8, 68u8];
__t0.data = [[0u8, 76u8, 2u8, 0u8], [1u8, 10u8, 32u8, 128u8], [106u8, 170u8, 170u8, 171u8], [0u8, 3u8, 4u8, 148u8], [250u8, 28u8, 131u8, 17u8], [128u8, 15u8, 249u8, 83u8], [132u8, 38u8, 4u8, 4u8], [0u8, 224u8, 1u8, 0u8], [69u8, 72u8, 0u8, 0u8], [39u8, 8u8, 0u8, 0u8], [90u8, 20u8, 42u8, 20u8], [0u8, 91u8, 224u8, 8u8], [0u8, 0u8, 0u8, 0u8]];
__t0.config = [83u8, 45u8, 0u8];
__t0
};

#[repr(C)]
#[derive(Clone, Copy)]
struct intel_lt_phy_pll_params {
    name: *const u8,
    is_hdmi: bool,
    clock_rate: i32,
    state: *const intel_lt_phy_pll_state,
}


static xe3plpd_lt_dp_tables: __SlateAlign16<[intel_lt_phy_pll_params; 8]> = __SlateAlign16([{
let mut __t0: intel_lt_phy_pll_params = unsafe { core::mem::zeroed() };
__t0.name = (b"xe3plpd_lt_dp_rbr\0".as_ptr() as *mut u8) as *const u8;
__t0.is_hdmi = 0i32 != 0i32;
__t0.clock_rate = 162000i32;
__t0.state = unsafe { core::ptr::addr_of!(xe3plpd_lt_dp_rbr) };
__t0
}, {
let mut __t1: intel_lt_phy_pll_params = unsafe { core::mem::zeroed() };
__t1.name = (b"xe3plpd_lt_dp_hbr1\0".as_ptr() as *mut u8) as *const u8;
__t1.is_hdmi = 0i32 != 0i32;
__t1.clock_rate = 270000i32;
__t1.state = unsafe { core::ptr::addr_of!(xe3plpd_lt_dp_hbr1) };
__t1
}, {
let mut __t2: intel_lt_phy_pll_params = unsafe { core::mem::zeroed() };
__t2.name = (b"xe3plpd_lt_dp_hbr2\0".as_ptr() as *mut u8) as *const u8;
__t2.is_hdmi = 0i32 != 0i32;
__t2.clock_rate = 540000i32;
__t2.state = unsafe { core::ptr::addr_of!(xe3plpd_lt_dp_hbr2) };
__t2
}, {
let mut __t3: intel_lt_phy_pll_params = unsafe { core::mem::zeroed() };
__t3.name = (b"xe3plpd_lt_dp_hbr3\0".as_ptr() as *mut u8) as *const u8;
__t3.is_hdmi = 0i32 != 0i32;
__t3.clock_rate = 810000i32;
__t3.state = unsafe { core::ptr::addr_of!(xe3plpd_lt_dp_hbr3) };
__t3
}, {
let mut __t4: intel_lt_phy_pll_params = unsafe { core::mem::zeroed() };
__t4.name = (b"xe3plpd_lt_dp_uhbr10\0".as_ptr() as *mut u8) as *const u8;
__t4.is_hdmi = 0i32 != 0i32;
__t4.clock_rate = 1000000i32;
__t4.state = unsafe { core::ptr::addr_of!(xe3plpd_lt_dp_uhbr10) };
__t4
}, {
let mut __t5: intel_lt_phy_pll_params = unsafe { core::mem::zeroed() };
__t5.name = (b"xe3plpd_lt_dp_uhbr13_5\0".as_ptr() as *mut u8) as *const u8;
__t5.is_hdmi = 0i32 != 0i32;
__t5.clock_rate = 1350000i32;
__t5.state = unsafe { core::ptr::addr_of!(xe3plpd_lt_dp_uhbr13_5) };
__t5
}, {
let mut __t6: intel_lt_phy_pll_params = unsafe { core::mem::zeroed() };
__t6.name = (b"xe3plpd_lt_dp_uhbr20\0".as_ptr() as *mut u8) as *const u8;
__t6.is_hdmi = 0i32 != 0i32;
__t6.clock_rate = 2000000i32;
__t6.state = unsafe { core::ptr::addr_of!(xe3plpd_lt_dp_uhbr20) };
__t6
}, {
let mut __t7: intel_lt_phy_pll_params = unsafe { core::mem::zeroed() };
__t7
}]);

static xe3plpd_lt_edp_2_16: intel_lt_phy_pll_state = {
let mut __t0: intel_lt_phy_pll_state = unsafe { core::mem::zeroed() };
__t0.addr_msb = [135u8, 135u8, 135u8, 135u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8];
__t0.addr_lsb = [16u8, 12u8, 20u8, 228u8, 12u8, 16u8, 20u8, 24u8, 72u8, 64u8, 76u8, 36u8, 68u8];
__t0.data = [[0u8, 76u8, 2u8, 0u8], [3u8, 202u8, 42u8, 32u8], [128u8, 0u8, 0u8, 0u8], [6u8, 4u8, 129u8, 188u8], [250u8, 22u8, 131u8, 17u8], [128u8, 15u8, 249u8, 83u8], [132u8, 38u8, 5u8, 4u8], [0u8, 224u8, 1u8, 0u8], [75u8, 72u8, 0u8, 0u8], [39u8, 8u8, 0u8, 0u8], [90u8, 19u8, 41u8, 19u8], [0u8, 91u8, 224u8, 10u8], [0u8, 0u8, 0u8, 0u8]];
__t0.config = [163u8, 45u8, 1u8];
__t0
};

static xe3plpd_lt_edp_2_43: intel_lt_phy_pll_state = {
let mut __t0: intel_lt_phy_pll_state = unsafe { core::mem::zeroed() };
__t0.addr_msb = [135u8, 135u8, 135u8, 135u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8];
__t0.addr_lsb = [16u8, 12u8, 20u8, 228u8, 12u8, 16u8, 20u8, 24u8, 72u8, 64u8, 76u8, 36u8, 68u8];
__t0.data = [[0u8, 76u8, 2u8, 0u8], [3u8, 202u8, 47u8, 96u8], [176u8, 0u8, 0u8, 0u8], [6u8, 4u8, 129u8, 188u8], [250u8, 19u8, 131u8, 17u8], [128u8, 15u8, 249u8, 83u8], [132u8, 38u8, 6u8, 4u8], [0u8, 224u8, 1u8, 0u8], [71u8, 72u8, 0u8, 0u8], [0u8, 0u8, 0u8, 0u8], [90u8, 19u8, 41u8, 19u8], [0u8, 91u8, 224u8, 12u8], [0u8, 0u8, 0u8, 0u8]];
__t0.config = [171u8, 45u8, 1u8];
__t0
};

static xe3plpd_lt_edp_3_24: intel_lt_phy_pll_state = {
let mut __t0: intel_lt_phy_pll_state = unsafe { core::mem::zeroed() };
__t0.addr_msb = [135u8, 135u8, 135u8, 135u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8];
__t0.addr_lsb = [16u8, 12u8, 20u8, 228u8, 12u8, 16u8, 20u8, 24u8, 72u8, 64u8, 76u8, 36u8, 68u8];
__t0.data = [[0u8, 76u8, 2u8, 0u8], [2u8, 138u8, 42u8, 32u8], [128u8, 0u8, 0u8, 0u8], [6u8, 4u8, 129u8, 40u8], [250u8, 22u8, 131u8, 17u8], [128u8, 15u8, 249u8, 83u8], [132u8, 38u8, 5u8, 4u8], [0u8, 224u8, 1u8, 0u8], [75u8, 72u8, 0u8, 0u8], [39u8, 8u8, 0u8, 0u8], [90u8, 19u8, 41u8, 19u8], [0u8, 91u8, 224u8, 10u8], [0u8, 0u8, 0u8, 0u8]];
__t0.config = [179u8, 45u8, 1u8];
__t0
};

static xe3plpd_lt_edp_4_32: intel_lt_phy_pll_state = {
let mut __t0: intel_lt_phy_pll_state = unsafe { core::mem::zeroed() };
__t0.addr_msb = [135u8, 135u8, 135u8, 135u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8];
__t0.addr_lsb = [16u8, 12u8, 20u8, 228u8, 12u8, 16u8, 20u8, 24u8, 72u8, 64u8, 76u8, 36u8, 68u8];
__t0.data = [[0u8, 76u8, 2u8, 0u8], [1u8, 77u8, 42u8, 32u8], [128u8, 0u8, 0u8, 0u8], [12u8, 4u8, 129u8, 188u8], [250u8, 22u8, 131u8, 17u8], [128u8, 15u8, 249u8, 83u8], [132u8, 38u8, 5u8, 4u8], [0u8, 224u8, 1u8, 0u8], [75u8, 72u8, 0u8, 0u8], [39u8, 8u8, 0u8, 0u8], [90u8, 19u8, 41u8, 19u8], [0u8, 91u8, 224u8, 10u8], [0u8, 0u8, 0u8, 0u8]];
__t0.config = [187u8, 45u8, 1u8];
__t0
};

static xe3plpd_lt_edp_6_75: intel_lt_phy_pll_state = {
let mut __t0: intel_lt_phy_pll_state = unsafe { core::mem::zeroed() };
__t0.addr_msb = [135u8, 135u8, 135u8, 135u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8];
__t0.addr_lsb = [16u8, 12u8, 20u8, 228u8, 12u8, 16u8, 20u8, 24u8, 72u8, 64u8, 76u8, 36u8, 68u8];
__t0.data = [[0u8, 76u8, 2u8, 0u8], [1u8, 74u8, 43u8, 224u8], [144u8, 0u8, 0u8, 0u8], [6u8, 4u8, 128u8, 168u8], [250u8, 21u8, 131u8, 17u8], [128u8, 15u8, 249u8, 83u8], [132u8, 38u8, 6u8, 4u8], [0u8, 224u8, 1u8, 0u8], [73u8, 72u8, 0u8, 0u8], [39u8, 8u8, 0u8, 0u8], [90u8, 19u8, 41u8, 19u8], [0u8, 87u8, 224u8, 12u8], [0u8, 0u8, 0u8, 0u8]];
__t0.config = [219u8, 45u8, 1u8];
__t0
};

static xe3plpd_lt_edp_tables: __SlateAlign16<[intel_lt_phy_pll_params; 10]> = __SlateAlign16([{
let mut __t0: intel_lt_phy_pll_params = unsafe { core::mem::zeroed() };
__t0.name = (b"xe3plpd_lt_dp_rbr\0".as_ptr() as *mut u8) as *const u8;
__t0.is_hdmi = 0i32 != 0i32;
__t0.clock_rate = 162000i32;
__t0.state = unsafe { core::ptr::addr_of!(xe3plpd_lt_dp_rbr) };
__t0
}, {
let mut __t1: intel_lt_phy_pll_params = unsafe { core::mem::zeroed() };
__t1.name = (b"xe3plpd_lt_edp_2_16\0".as_ptr() as *mut u8) as *const u8;
__t1.is_hdmi = 0i32 != 0i32;
__t1.clock_rate = 216000i32;
__t1.state = unsafe { core::ptr::addr_of!(xe3plpd_lt_edp_2_16) };
__t1
}, {
let mut __t2: intel_lt_phy_pll_params = unsafe { core::mem::zeroed() };
__t2.name = (b"xe3plpd_lt_edp_2_43\0".as_ptr() as *mut u8) as *const u8;
__t2.is_hdmi = 0i32 != 0i32;
__t2.clock_rate = 243000i32;
__t2.state = unsafe { core::ptr::addr_of!(xe3plpd_lt_edp_2_43) };
__t2
}, {
let mut __t3: intel_lt_phy_pll_params = unsafe { core::mem::zeroed() };
__t3.name = (b"xe3plpd_lt_dp_hbr1\0".as_ptr() as *mut u8) as *const u8;
__t3.is_hdmi = 0i32 != 0i32;
__t3.clock_rate = 270000i32;
__t3.state = unsafe { core::ptr::addr_of!(xe3plpd_lt_dp_hbr1) };
__t3
}, {
let mut __t4: intel_lt_phy_pll_params = unsafe { core::mem::zeroed() };
__t4.name = (b"xe3plpd_lt_edp_3_24\0".as_ptr() as *mut u8) as *const u8;
__t4.is_hdmi = 0i32 != 0i32;
__t4.clock_rate = 324000i32;
__t4.state = unsafe { core::ptr::addr_of!(xe3plpd_lt_edp_3_24) };
__t4
}, {
let mut __t5: intel_lt_phy_pll_params = unsafe { core::mem::zeroed() };
__t5.name = (b"xe3plpd_lt_edp_4_32\0".as_ptr() as *mut u8) as *const u8;
__t5.is_hdmi = 0i32 != 0i32;
__t5.clock_rate = 432000i32;
__t5.state = unsafe { core::ptr::addr_of!(xe3plpd_lt_edp_4_32) };
__t5
}, {
let mut __t6: intel_lt_phy_pll_params = unsafe { core::mem::zeroed() };
__t6.name = (b"xe3plpd_lt_dp_hbr2\0".as_ptr() as *mut u8) as *const u8;
__t6.is_hdmi = 0i32 != 0i32;
__t6.clock_rate = 540000i32;
__t6.state = unsafe { core::ptr::addr_of!(xe3plpd_lt_dp_hbr2) };
__t6
}, {
let mut __t7: intel_lt_phy_pll_params = unsafe { core::mem::zeroed() };
__t7.name = (b"xe3plpd_lt_edp_6_75\0".as_ptr() as *mut u8) as *const u8;
__t7.is_hdmi = 0i32 != 0i32;
__t7.clock_rate = 675000i32;
__t7.state = unsafe { core::ptr::addr_of!(xe3plpd_lt_edp_6_75) };
__t7
}, {
let mut __t8: intel_lt_phy_pll_params = unsafe { core::mem::zeroed() };
__t8.name = (b"xe3plpd_lt_dp_hbr3\0".as_ptr() as *mut u8) as *const u8;
__t8.is_hdmi = 0i32 != 0i32;
__t8.clock_rate = 810000i32;
__t8.state = unsafe { core::ptr::addr_of!(xe3plpd_lt_dp_hbr3) };
__t8
}, {
let mut __t9: intel_lt_phy_pll_params = unsafe { core::mem::zeroed() };
__t9
}]);

static xe3plpd_lt_hdmi_252: intel_lt_phy_pll_state = {
let mut __t0: intel_lt_phy_pll_state = unsafe { core::mem::zeroed() };
__t0.addr_msb = [135u8, 135u8, 135u8, 135u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8];
__t0.addr_lsb = [16u8, 12u8, 20u8, 228u8, 12u8, 16u8, 20u8, 24u8, 72u8, 64u8, 76u8, 36u8, 68u8];
__t0.data = [[0u8, 76u8, 2u8, 0u8], [12u8, 21u8, 39u8, 96u8], [0u8, 0u8, 0u8, 0u8], [8u8, 4u8, 152u8, 40u8], [66u8, 0u8, 132u8, 16u8], [128u8, 15u8, 217u8, 181u8], [134u8, 0u8, 0u8, 0u8], [1u8, 160u8, 1u8, 0u8], [75u8, 0u8, 0u8, 0u8], [40u8, 0u8, 0u8, 0u8], [0u8, 20u8, 42u8, 20u8], [0u8, 0u8, 0u8, 0u8], [0u8, 0u8, 0u8, 0u8]];
__t0.config = [132u8, 45u8, 0u8];
__t0
};

static xe3plpd_lt_hdmi_742p5: intel_lt_phy_pll_state = {
let mut __t0: intel_lt_phy_pll_state = unsafe { core::mem::zeroed() };
__t0.addr_msb = [135u8, 135u8, 135u8, 135u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8];
__t0.addr_lsb = [16u8, 12u8, 20u8, 228u8, 12u8, 16u8, 20u8, 24u8, 72u8, 64u8, 76u8, 36u8, 68u8];
__t0.data = [[0u8, 76u8, 2u8, 0u8], [4u8, 21u8, 38u8, 160u8], [96u8, 0u8, 0u8, 0u8], [8u8, 4u8, 136u8, 40u8], [250u8, 12u8, 132u8, 17u8], [128u8, 15u8, 217u8, 83u8], [134u8, 0u8, 0u8, 0u8], [1u8, 160u8, 1u8, 0u8], [75u8, 0u8, 0u8, 0u8], [40u8, 0u8, 0u8, 0u8], [0u8, 20u8, 42u8, 20u8], [0u8, 0u8, 0u8, 0u8], [0u8, 0u8, 0u8, 0u8]];
__t0.config = [132u8, 45u8, 0u8];
__t0
};

static xe3plpd_lt_hdmi_1p485: intel_lt_phy_pll_state = {
let mut __t0: intel_lt_phy_pll_state = unsafe { core::mem::zeroed() };
__t0.addr_msb = [135u8, 135u8, 135u8, 135u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8];
__t0.addr_lsb = [16u8, 12u8, 20u8, 228u8, 12u8, 16u8, 20u8, 24u8, 72u8, 64u8, 76u8, 36u8, 68u8];
__t0.data = [[0u8, 76u8, 2u8, 0u8], [2u8, 21u8, 38u8, 160u8], [96u8, 0u8, 0u8, 0u8], [8u8, 4u8, 132u8, 40u8], [250u8, 12u8, 132u8, 17u8], [128u8, 15u8, 217u8, 83u8], [134u8, 0u8, 0u8, 0u8], [1u8, 160u8, 1u8, 0u8], [75u8, 0u8, 0u8, 0u8], [40u8, 0u8, 0u8, 0u8], [0u8, 20u8, 42u8, 20u8], [0u8, 0u8, 0u8, 0u8], [0u8, 0u8, 0u8, 0u8]];
__t0.config = [132u8, 45u8, 0u8];
__t0
};

static xe3plpd_lt_hdmi_5p94: intel_lt_phy_pll_state = {
let mut __t0: intel_lt_phy_pll_state = unsafe { core::mem::zeroed() };
__t0.addr_msb = [135u8, 135u8, 135u8, 135u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8, 136u8];
__t0.addr_lsb = [16u8, 12u8, 20u8, 228u8, 12u8, 16u8, 20u8, 24u8, 72u8, 64u8, 76u8, 36u8, 68u8];
__t0.data = [[0u8, 76u8, 2u8, 0u8], [0u8, 149u8, 38u8, 160u8], [96u8, 0u8, 0u8, 0u8], [8u8, 4u8, 129u8, 40u8], [250u8, 12u8, 132u8, 17u8], [128u8, 15u8, 217u8, 83u8], [134u8, 0u8, 0u8, 0u8], [1u8, 160u8, 1u8, 0u8], [75u8, 0u8, 0u8, 0u8], [40u8, 0u8, 0u8, 0u8], [0u8, 20u8, 42u8, 20u8], [0u8, 0u8, 0u8, 0u8], [0u8, 0u8, 0u8, 0u8]];
__t0.config = [132u8, 45u8, 0u8];
__t0
};

static xe3plpd_lt_hdmi_tables: __SlateAlign16<[intel_lt_phy_pll_params; 5]> = __SlateAlign16([{
let mut __t0: intel_lt_phy_pll_params = unsafe { core::mem::zeroed() };
__t0.name = (b"xe3plpd_lt_hdmi_252\0".as_ptr() as *mut u8) as *const u8;
__t0.is_hdmi = 1i32 != 0i32;
__t0.clock_rate = 25200i32;
__t0.state = unsafe { core::ptr::addr_of!(xe3plpd_lt_hdmi_252) };
__t0
}, {
let mut __t1: intel_lt_phy_pll_params = unsafe { core::mem::zeroed() };
__t1.name = (b"xe3plpd_lt_hdmi_742p5\0".as_ptr() as *mut u8) as *const u8;
__t1.is_hdmi = 1i32 != 0i32;
__t1.clock_rate = 74250i32;
__t1.state = unsafe { core::ptr::addr_of!(xe3plpd_lt_hdmi_742p5) };
__t1
}, {
let mut __t2: intel_lt_phy_pll_params = unsafe { core::mem::zeroed() };
__t2.name = (b"xe3plpd_lt_hdmi_1p485\0".as_ptr() as *mut u8) as *const u8;
__t2.is_hdmi = 1i32 != 0i32;
__t2.clock_rate = 148500i32;
__t2.state = unsafe { core::ptr::addr_of!(xe3plpd_lt_hdmi_1p485) };
__t2
}, {
let mut __t3: intel_lt_phy_pll_params = unsafe { core::mem::zeroed() };
__t3.name = (b"xe3plpd_lt_hdmi_5p94\0".as_ptr() as *mut u8) as *const u8;
__t3.is_hdmi = 1i32 != 0i32;
__t3.clock_rate = 594000i32;
__t3.state = unsafe { core::ptr::addr_of!(xe3plpd_lt_hdmi_5p94) };
__t3
}, {
let mut __t4: intel_lt_phy_pll_params = unsafe { core::mem::zeroed() };
__t4
}]);

fn intel_lt_phy_get_owned_lane_mask(encoder: *mut intel_encoder) -> u8 {
let dig_port: *mut intel_digital_port = enc_to_dig_port(encoder);
if !unsafe { intel_tc_port_in_dp_alt_mode(dig_port) } {
return (BIT!(1i32) | BIT!(0i32)) as u8;
}
(if (unsafe { intel_tc_port_max_lane_count(dig_port) }) > 2 { BIT!(1i32) | BIT!(0i32) } else { BIT!(0i32) }) as u8
}

fn intel_lt_phy_read(encoder: *mut intel_encoder, lane_mask: u8, addr: u16) -> u8 {
unsafe { intel_cx0_read(encoder, lane_mask, addr) }
}

fn intel_lt_phy_write(encoder: *mut intel_encoder, lane_mask: u8, addr: u16, data: u8, committed: bool) {
unsafe { intel_cx0_write(encoder, lane_mask, addr, data, committed) };
}

fn intel_lt_phy_rmw(encoder: *mut intel_encoder, lane_mask: u8, addr: u16, clear: u8, set: u8, committed: bool) {
unsafe { intel_cx0_rmw(encoder, lane_mask, addr, clear, set, committed) };
}

fn intel_lt_phy_clear_status_p2p(encoder: *mut intel_encoder, lane: i32) {
let mut __slate_compound_1: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display!(encoder);
intel_de_rmw(display, unsafe { *{
__slate_compound_1 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__163!(if ((*encoder).port as i32) >= 3 { (*encoder).port as i32 } else { 7i32.wrapping_add((*encoder).port as i32).wrapping_sub(0) }, 3i32, 409664i32, 409920i32, 1503808i32, 1504320i32).wrapping_add(96).wrapping_add(lane.wrapping_mul(4)) as u32 };
core::ptr::addr_of_mut!(__slate_compound_1)
} }, REG_BIT__1435!(31i32), 0u32);
}

fn assert_dc_off(display: *mut intel_display) {
let mut enabled: bool = false;
enabled = unsafe { intel_display_power_is_enabled(display, 73u32) };
let mut __v0: i64;
let __ret_warn_on: i32 = !enabled as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(!enabled)".as_ptr() as *mut u8) };
let mut __v1: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v1 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v2: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_assert_dc_off_3, sym __slate_asm_string_assert_dc_off_4, const 1005 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v2 = bug;
unsafe { __v1.unwrap()(__v2, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(!enabled)".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_700: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __intel_lt_phy_p2p_write_once(encoder: *mut intel_encoder, lane: i32, addr: u16, data: u8, mac_reg_addr: __SlateRecord_i915_reg_defs_h_14, expected_mac_val: u8) -> i32 {
let mut __slate_compound_1: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_2: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_3: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_4: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_6: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display!(encoder);
let port: i32 = unsafe { (*encoder).port };
let phy: i32 = unsafe { intel_encoder_to_phy(encoder) };
let mut ack: i32 = 0;
let mut val: u32 = 0;
if (unsafe { intel_de_wait_for_clear_ms(display, if (DISPLAY_VER!(display) as i32) >= 20 { *{
__slate_compound_1 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__8!(if (port as i32) >= 3 { port as i32 } else { 7i32.wrapping_add(port as i32).wrapping_sub(0) }, 3i32, 409664i32, 409920i32, 1503808i32, 1504320i32).wrapping_add(lane.wrapping_mul(4)) as u32 };
core::ptr::addr_of_mut!(__slate_compound_1)
} } else { *{
__slate_compound_2 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__23!(port, 3i32, 409664i32, 409920i32, 1503808i32, 1504320i32).wrapping_add(lane.wrapping_mul(4)) as u32 };
core::ptr::addr_of_mut!(__slate_compound_2)
} }, REG_BIT__362!(24i32), 1u32) }) != 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"PHY %c Timeout waiting for previous transaction to complete. Resetting bus.\n".as_ptr() as *const u8, phy_name!(phy)) };
unsafe { intel_cx0_bus_reset(encoder, lane) };
return 110i32.wrapping_neg();
}
intel_de_rmw(display, if (DISPLAY_VER!(display) as i32) >= 20 { unsafe { *{
__slate_compound_3 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__77!(if (port as i32) >= 3 { port as i32 } else { 7i32.wrapping_add(port as i32).wrapping_sub(0) }, 3i32, 409664i32, 409920i32, 1503808i32, 1504320i32).wrapping_add(lane.wrapping_mul(4)).wrapping_add(8) as u32 };
core::ptr::addr_of_mut!(__slate_compound_3)
} } } else { unsafe { *{
__slate_compound_4 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__67!(port, 3i32, 409664i32, 409920i32, 1503808i32, 1504320i32).wrapping_add(lane.wrapping_mul(4)).wrapping_add(8) as u32 };
core::ptr::addr_of_mut!(__slate_compound_4)
} } }, 0u32, 0u32);
intel_de_write(display, if (DISPLAY_VER!(display) as i32) >= 20 { unsafe { *{
__slate_compound_5 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__82!(if (port as i32) >= 3 { port as i32 } else { 7i32.wrapping_add(port as i32).wrapping_sub(0) }, 3i32, 409664i32, 409920i32, 1503808i32, 1504320i32).wrapping_add(lane.wrapping_mul(4)) as u32 };
core::ptr::addr_of_mut!(__slate_compound_5)
} } } else { unsafe { *{
__slate_compound_6 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__165!(port, 3i32, 409664i32, 409920i32, 1503808i32, 1504320i32).wrapping_add(lane.wrapping_mul(4)) as u32 };
core::ptr::addr_of_mut!(__slate_compound_6)
} } }, REG_BIT__1166!(24i32) | (2u32 << (((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1033_2>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 27i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(30)) as u64).trailing_zeros() as i32) & (core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1033_3>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 27i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(30))).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1033_4>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1033_6>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1033_9>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1033_16>() as u32) | XELPDP_PORT_M2P_DATA!(data) | XELPDP_PORT_M2P_ADDRESS__2!(addr));
ack = unsafe { intel_cx0_wait_for_ack(encoder, 5i32, lane, core::ptr::addr_of_mut!(val)) };
if ack < 0 {
return ack;
}
if val & REG_BIT__1341!(15i32) != 0 {
unsafe { __drm_dev_dbg(core::ptr::null_mut::<_ddebug>(), __drm_to_dev((*display).drm as *const drm_device) as *const device, 2u32, c"PHY %c Error occurred during P2P write command. Status: 0x%x\n".as_ptr() as *const u8, phy_name!(phy), val) };
intel_lt_phy_clear_status_p2p(encoder, lane);
unsafe { intel_cx0_bus_reset(encoder, lane) };
return 22i32.wrapping_neg();
}
// RE-VISIT:
// This needs to be added to give PHY time to set everything up this was a requirement
// to get the display up and running
// This is the time PHY takes to settle down after programming the PHY.
udelay(150u64);
unsafe { intel_cx0_clear_response_ready_flag(encoder, lane) };
intel_lt_phy_clear_status_p2p(encoder, lane);
0
}

fn __intel_lt_phy_p2p_write(encoder: *mut intel_encoder, lane: i32, addr: u16, data: u8, mac_reg_addr: __SlateRecord_i915_reg_defs_h_14, expected_mac_val: u8) {
let display: *mut intel_display = to_intel_display!(encoder);
let phy: i32 = unsafe { intel_encoder_to_phy(encoder) };
// 3 tries is assumed to be enough to write successfully
let mut i: i32 = 0;
let mut status: i32 = 0;
assert_dc_off(display);
while i < 3 {
status = __intel_lt_phy_p2p_write_once(encoder, lane, addr, data, mac_reg_addr, expected_mac_val);
if status == 0 {
return;
}
i = i.wrapping_add(1);
}
loop {
if !unsafe { __print_once } {
unsafe {
__print_once = true;
}
while false {
}
unsafe { _dev_err(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] *ERROR* PHY %c P2P Write %04x failed after %d retries.\n".as_ptr() as *const u8, phy_name!(phy), addr as i32, i) };
}
if true {
break;
}
}
}

#[unsafe(link_section = ".data..read_mostly")]
static mut __print_once: bool = unsafe { core::mem::zeroed() };

fn intel_lt_phy_p2p_write(encoder: *mut intel_encoder, lane_mask: u8, addr: u16, data: u8, mac_reg_addr: __SlateRecord_i915_reg_defs_h_14, expected_mac_val: u8) {
for lane in 0i32..2i32 {
if ((lane_mask as i32) as u64) & BIT!(lane) != 0 {
__intel_lt_phy_p2p_write(encoder, lane, addr, data, mac_reg_addr, expected_mac_val);
}
}
}

fn intel_lt_phy_setup_powerdown(encoder: *mut intel_encoder, lane_count: u8) {
// The new PORT_BUF_CTL6 stuff for dc5 entry and exit needs to be handled
// by dmc firmware not explicitly mentioned in Bspec. This leaves this
// function as a wrapper only but keeping it expecting future changes.
unsafe { intel_cx0_setup_powerdown(encoder) };
}

fn intel_lt_phy_powerdown_change_sequence(encoder: *mut intel_encoder, lane_mask: u8, state: u8) {
unsafe { intel_cx0_powerdown_change_sequence(encoder, lane_mask, state) };
}

fn intel_lt_phy_lane_reset(encoder: *mut intel_encoder, lane_count: u8) {
let mut __slate_compound_1: [u32; 2] = unsafe { core::mem::zeroed() };
let mut __slate_compound_2: [u32; 2] = unsafe { core::mem::zeroed() };
let mut __slate_compound_3: [u32; 2] = unsafe { core::mem::zeroed() };
let mut __slate_compound_4: [u32; 2] = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: [u32; 2] = unsafe { core::mem::zeroed() };
let mut __slate_compound_6: [u32; 2] = unsafe { core::mem::zeroed() };
let mut __slate_compound_7: [u32; 2] = unsafe { core::mem::zeroed() };
let mut __slate_compound_8: [u32; 2] = unsafe { core::mem::zeroed() };
let mut __slate_compound_9: [u32; 2] = unsafe { core::mem::zeroed() };
let mut __slate_compound_10: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_11: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_12: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_13: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_14: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_15: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_16: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_17: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_19: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_20: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_21: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_22: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_23: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_24: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_26: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_27: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_29: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_30: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display!(encoder);
let port: i32 = unsafe { (*encoder).port };
let phy: i32 = unsafe { intel_encoder_to_phy(encoder) };
let owned_lane_mask: u8 = intel_lt_phy_get_owned_lane_mask(encoder);
let lane_pipe_reset: u32 = if ((owned_lane_mask as i32) as u64) == BIT!(1i32) | BIT!(0i32) { (unsafe { *((*{
__slate_compound_1 = [REG_BIT__1165!(31i32), REG_BIT__911!(30i32)];
core::ptr::addr_of_mut!(__slate_compound_1)
}).as_ptr() as *const u32).wrapping_offset(0 as isize) }) | unsafe { *((*{
__slate_compound_2 = [REG_BIT__1192!(31i32), REG_BIT__1087!(30i32)];
core::ptr::addr_of_mut!(__slate_compound_2)
}).as_ptr() as *const u32).wrapping_offset(1 as isize) } } else { unsafe { *((*{
__slate_compound_3 = [REG_BIT__426!(31i32), REG_BIT__824!(30i32)];
core::ptr::addr_of_mut!(__slate_compound_3)
}).as_ptr() as *const u32).wrapping_offset(0 as isize) } };
let lane_phy_current_status: u32 = if ((owned_lane_mask as i32) as u64) == BIT!(1i32) | BIT!(0i32) { (unsafe { *((*{
__slate_compound_4 = [REG_BIT__652!(29i32), REG_BIT__1117!(28i32)];
core::ptr::addr_of_mut!(__slate_compound_4)
}).as_ptr() as *const u32).wrapping_offset(0 as isize) }) | unsafe { *((*{
__slate_compound_5 = [REG_BIT__689!(29i32), REG_BIT__891!(28i32)];
core::ptr::addr_of_mut!(__slate_compound_5)
}).as_ptr() as *const u32).wrapping_offset(1 as isize) } } else { unsafe { *((*{
__slate_compound_6 = [REG_BIT__1074!(29i32), REG_BIT__868!(28i32)];
core::ptr::addr_of_mut!(__slate_compound_6)
}).as_ptr() as *const u32).wrapping_offset(0 as isize) } };
let lane_phy_pulse_status: u32 = if ((owned_lane_mask as i32) as u64) == BIT!(1i32) | BIT!(0i32) { (unsafe { *((*{
__slate_compound_7 = [REG_BIT__794!(27i32), REG_BIT__863!(26i32)];
core::ptr::addr_of_mut!(__slate_compound_7)
}).as_ptr() as *const u32).wrapping_offset(0 as isize) }) | unsafe { *((*{
__slate_compound_8 = [REG_BIT__825!(27i32), REG_BIT__673!(26i32)];
core::ptr::addr_of_mut!(__slate_compound_8)
}).as_ptr() as *const u32).wrapping_offset(1 as isize) } } else { unsafe { *((*{
__slate_compound_9 = [REG_BIT__1417!(27i32), REG_BIT__1293!(26i32)];
core::ptr::addr_of_mut!(__slate_compound_9)
}).as_ptr() as *const u32).wrapping_offset(0 as isize) } };
intel_de_rmw(display, unsafe { *{
__slate_compound_10 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__240!(if (port as i32) >= 3 { port as i32 } else { 7i32.wrapping_add(port as i32).wrapping_sub(0) }, 3i32, 409604i32, 409860i32, 1503744i32, 1504256i32).wrapping_add(52) as u32 };
core::ptr::addr_of_mut!(__slate_compound_10)
} }, REG_GENMASK__112!(4i32, 0i32), (31u32 << (((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1137_3>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 0i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(4)) as u64).trailing_zeros() as i32) & (core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1137_4>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 0i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(4))).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1137_5>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1137_7>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1137_10>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1137_17>() as u32));
intel_de_rmw(display, if (DISPLAY_VER!(display) as i32) >= 20 { unsafe { *{
__slate_compound_11 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__89!(if (port as i32) >= 3 { port as i32 } else { 7i32.wrapping_add(port as i32).wrapping_sub(0) }, 3i32, 409604i32, 409860i32, 1503744i32, 1504256i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_11)
} } } else { unsafe { *{
__slate_compound_12 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__186!(port, 3i32, 409604i32, 409860i32, 1503744i32, 1504256i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_12)
} } }, REG_GENMASK__300!(15i32, 12i32), (3u32 << (((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1140_3>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 12i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15)) as u64).trailing_zeros() as i32) & (core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1140_4>() as u32).wrapping_add((1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) << 12i32 & (1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1u32 << 8u64.wrapping_mul(core::mem::size_of::<u32>() as u64).wrapping_sub(1).wrapping_sub((((1i32.wrapping_neg() as u32) < 1) as i32) as u64)) >> (core::mem::size_of::<u32>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(15))).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1140_5>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1140_7>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1140_10>() as u32).wrapping_add(core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1140_17>() as u32));
intel_lt_phy_setup_powerdown(encoder, lane_count);
intel_lt_phy_powerdown_change_sequence(encoder, owned_lane_mask, 2u8);
intel_de_rmw(display, unsafe { *{
__slate_compound_13 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__335!(if (port as i32) >= 3 { port as i32 } else { 7i32.wrapping_add(port as i32).wrapping_sub(0) }, 3i32, 409604i32, 409860i32, 1503744i32, 1504256i32).wrapping_add(52) as u32 };
core::ptr::addr_of_mut!(__slate_compound_13)
} }, REG_BIT__553!(11i32), 0u32);
intel_de_rmw(display, if (DISPLAY_VER!(display) as i32) >= 20 { unsafe { *{
__slate_compound_14 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__209!(if (port as i32) >= 3 { port as i32 } else { 7i32.wrapping_add(port as i32).wrapping_sub(0) }, 3i32, 409824i32, 410080i32, 1503840i32, 1504352i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_14)
} } } else { unsafe { *{
__slate_compound_15 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__58!(port, 3i32, 409824i32, 410080i32, 1503840i32, 1504352i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_15)
} } }, XELPDP_LANE_PCLK_PLL_REQUEST__3!(0i32), XELPDP_LANE_PCLK_PLL_REQUEST__6!(0i32));
if (unsafe { intel_de_wait_for_set_ms(display, if (DISPLAY_VER!(display) as i32) >= 20 { *{
__slate_compound_16 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__68!(if (port as i32) >= 3 { port as i32 } else { 7i32.wrapping_add(port as i32).wrapping_sub(0) }, 3i32, 409824i32, 410080i32, 1503840i32, 1504352i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_16)
} } else { *{
__slate_compound_17 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__7!(port, 3i32, 409824i32, 410080i32, 1503840i32, 1504352i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_17)
} }, XELPDP_LANE_PCLK_PLL_ACK__4!(0i32), 2u32) }) != 0 {
while false {
}
unsafe { _dev_warn(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] PHY %c PLL MacCLK assertion ack not done\n".as_ptr() as *const u8, phy_name!(phy)) };
}
intel_de_rmw(display, if (DISPLAY_VER!(display) as i32) >= 20 { unsafe { *{
__slate_compound_19 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__311!(if (port as i32) >= 3 { port as i32 } else { 7i32.wrapping_add(port as i32).wrapping_sub(0) }, 3i32, 409824i32, 410080i32, 1503840i32, 1504352i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_19)
} } } else { unsafe { *{
__slate_compound_20 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__6!(port, 3i32, 409824i32, 410080i32, 1503840i32, 1504352i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_20)
} } }, REG_BIT__1270!(10i32), REG_BIT__1439!(10i32));
intel_de_rmw(display, if (DISPLAY_VER!(display) as i32) >= 20 { unsafe { *{
__slate_compound_21 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__130!(if (port as i32) >= 3 { port as i32 } else { 7i32.wrapping_add(port as i32).wrapping_sub(0) }, 3i32, 409604i32, 409860i32, 1503744i32, 1504256i32).wrapping_add(4) as u32 };
core::ptr::addr_of_mut!(__slate_compound_21)
} } } else { unsafe { *{
__slate_compound_22 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__14!(port, 3i32, 409604i32, 409860i32, 1503744i32, 1504256i32).wrapping_add(4) as u32 };
core::ptr::addr_of_mut!(__slate_compound_22)
} } }, lane_pipe_reset | lane_phy_pulse_status, 0u32);
if (unsafe { intel_de_wait_for_clear_ms(display, if (DISPLAY_VER!(display) as i32) >= 20 { *{
__slate_compound_23 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__21!(if (port as i32) >= 3 { port as i32 } else { 7i32.wrapping_add(port as i32).wrapping_sub(0) }, 3i32, 409604i32, 409860i32, 1503744i32, 1504256i32).wrapping_add(4) as u32 };
core::ptr::addr_of_mut!(__slate_compound_23)
} } else { *{
__slate_compound_24 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__210!(port, 3i32, 409604i32, 409860i32, 1503744i32, 1504256i32).wrapping_add(4) as u32 };
core::ptr::addr_of_mut!(__slate_compound_24)
} }, lane_phy_current_status, 2u32) }) != 0 {
while false {
}
unsafe { _dev_warn(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] PHY %c failed to bring out of lane reset\n".as_ptr() as *const u8, phy_name!(phy)) };
}
if (unsafe { intel_de_wait_for_set_ms(display, if (DISPLAY_VER!(display) as i32) >= 20 { *{
__slate_compound_26 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__307!(if (port as i32) >= 3 { port as i32 } else { 7i32.wrapping_add(port as i32).wrapping_sub(0) }, 3i32, 409604i32, 409860i32, 1503744i32, 1504256i32).wrapping_add(4) as u32 };
core::ptr::addr_of_mut!(__slate_compound_26)
} } else { *{
__slate_compound_27 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__178!(port, 3i32, 409604i32, 409860i32, 1503744i32, 1504256i32).wrapping_add(4) as u32 };
core::ptr::addr_of_mut!(__slate_compound_27)
} }, lane_phy_pulse_status, 1u32) }) != 0 {
while false {
}
unsafe { _dev_warn(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] PHY %c PLL rate not changed\n".as_ptr() as *const u8, phy_name!(phy)) };
}
intel_de_rmw(display, if (DISPLAY_VER!(display) as i32) >= 20 { unsafe { *{
__slate_compound_29 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__291!(if (port as i32) >= 3 { port as i32 } else { 7i32.wrapping_add(port as i32).wrapping_sub(0) }, 3i32, 409604i32, 409860i32, 1503744i32, 1504256i32).wrapping_add(4) as u32 };
core::ptr::addr_of_mut!(__slate_compound_29)
} } } else { unsafe { *{
__slate_compound_30 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__78!(port, 3i32, 409604i32, 409860i32, 1503744i32, 1504256i32).wrapping_add(4) as u32 };
core::ptr::addr_of_mut!(__slate_compound_30)
} } }, lane_phy_pulse_status, 0u32);
}

fn intel_lt_phy_is_hdmi(ltpll: *const intel_lt_phy_pll_state) -> bool {
let mut mode: u8 = 0;
let mut __v0: u8;
loop {
if (__bf_cast_unsigned__2!((((const { core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1183_3>() }) as u64) as i32).wrapping_add((((1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 0i32 & (((1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> ((const { core::mem::size_of::<u8>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2)) as u8) as i32) > (__bf_cast_unsigned__2!(18446744073709551615u64) as i32) {
unsafe { __compiletime_assert_701() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v2: u8;
loop {
if false {
unsafe { __compiletime_assert_702() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (((((const { core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1183_5>() }) as u64) as i32).wrapping_add((((1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 0i32 & (((1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> ((const { core::mem::size_of::<u8>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2)) as u8) as i32) == 0 {
unsafe { __compiletime_assert_703() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { (!((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1183_6>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 0i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2)) as u8) as i32) >> ((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1183_7>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 0i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2)) as u8) as u64).trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
unsafe { __compiletime_assert_704() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1183_8>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 0i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2)) as u8) as i32) as u64).wrapping_add(1u64 << ((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1183_9>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 0i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2)) as u8) as u64).trailing_zeros() as i32)) & ((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1183_10>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 0i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2)) as u8) as i32) as u64).wrapping_add(1u64 << ((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1183_11>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 0i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2)) as u8) as u64).trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_705() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = ((((unsafe { *((*ltpll).config.as_ptr() as *const u8).wrapping_offset(0 as isize) }) as i32) & (((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1183_13>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 0i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2)) as u8) as i32)) >> ((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1183_14>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 0i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2)) as u8) as u64).trailing_zeros() as i32)) as u8;
__v0 = __v2;
mode = __v0;
if (mode as i32) == 4 || (mode as i32) == 5 {
return true;
}
false
}

fn intel_lt_phy_is_dp(ltpll: *const intel_lt_phy_pll_state) -> bool {
let mut mode: u8 = 0;
let mut __v0: u8;
loop {
if (__bf_cast_unsigned__2!((((const { core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1193_3>() }) as u64) as i32).wrapping_add((((1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 0i32 & (((1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> ((const { core::mem::size_of::<u8>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2)) as u8) as i32) > (__bf_cast_unsigned__2!(18446744073709551615u64) as i32) {
unsafe { __compiletime_assert_706() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v2: u8;
loop {
if false {
unsafe { __compiletime_assert_707() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (((((const { core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1193_5>() }) as u64) as i32).wrapping_add((((1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 0i32 & (((1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> ((const { core::mem::size_of::<u8>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2)) as u8) as i32) == 0 {
unsafe { __compiletime_assert_708() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { (!((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1193_6>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 0i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2)) as u8) as i32) >> ((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1193_7>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 0i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2)) as u8) as u64).trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
unsafe { __compiletime_assert_709() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1193_8>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 0i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2)) as u8) as i32) as u64).wrapping_add(1u64 << ((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1193_9>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 0i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2)) as u8) as u64).trailing_zeros() as i32)) & ((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1193_10>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 0i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2)) as u8) as i32) as u64).wrapping_add(1u64 << ((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1193_11>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 0i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2)) as u8) as u64).trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_710() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = ((((unsafe { *((*ltpll).config.as_ptr() as *const u8).wrapping_offset(0 as isize) }) as i32) & (((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1193_13>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 0i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2)) as u8) as i32)) >> ((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1193_14>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 0i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(2)) as u8) as u64).trailing_zeros() as i32)) as u8;
__v0 = __v2;
mode = __v0;
if (mode as i32) == 3 {
return true;
}
false
}

fn intel_lt_phy_program_port_clock_ctl(encoder: *mut intel_encoder, ltpll: *const intel_lt_phy_pll_state, port_clock: i32, lane_reversal: bool) {
let mut __slate_compound_1: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_2: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_12: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_13: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display!(encoder);
let mut val: u32 = 0;
intel_de_rmw(display, if (DISPLAY_VER!(display) as i32) >= 20 { unsafe { *{
__slate_compound_1 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__208!(if ((*encoder).port as i32) >= 3 { (*encoder).port as i32 } else { 7i32.wrapping_add((*encoder).port as i32).wrapping_sub(0) }, 3i32, 409604i32, 409860i32, 1503744i32, 1504256i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_1)
} } } else { unsafe { *{
__slate_compound_2 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__126!(unsafe { (*encoder).port }, 3i32, 409604i32, 409860i32, 1503744i32, 1504256i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_2)
} } }, REG_BIT__317!(16i32), if lane_reversal { REG_BIT__466!(16i32) } else { 0u32 });
let __v3: u32 = val;
val = __v3 | REG_BIT__1253!(10i32);
// We actually mean MACCLK here and not MAXPCLK when using LT Phy
// but since the register bits still remain the same we use
// the same definition
let __v5: bool = intel_lt_phy_is_hdmi(ltpll) && unsafe { intel_hdmi_is_frl(port_clock as u32) };
if __v5 {
let __v6: u32 = val;
val = __v6 | XELPDP_DDI_CLOCK_SELECT_PREP__5!(display, 9i32);
} else {
let __v8: u32 = val;
val = __v8 | XELPDP_DDI_CLOCK_SELECT_PREP!(display, 8i32);
}
let __v10: u32 = val;
val = __v10 | if unsafe { (*ltpll).ssc_enabled } { REG_BIT__1060!(1i32) } else { 0 };
intel_de_rmw(display, if (DISPLAY_VER!(display) as i32) >= 20 { unsafe { *{
__slate_compound_12 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__207!(if ((*encoder).port as i32) >= 3 { (*encoder).port as i32 } else { 7i32.wrapping_add((*encoder).port as i32).wrapping_sub(0) }, 3i32, 409824i32, 410080i32, 1503840i32, 1504352i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_12)
} } } else { unsafe { *{
__slate_compound_13 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__180!(unsafe { (*encoder).port }, 3i32, 409824i32, 410080i32, 1503840i32, 1504352i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_13)
} } }, REG_BIT__650!(8i32) | REG_BIT__812!(10i32) | XELPDP_DDI_CLOCK_SELECT_MASK__5!(display) | REG_BIT__266!(1i32) | REG_BIT__226!(0i32), val);
}

fn intel_lt_phy_get_dp_clock(rate: u8) -> u32 {
match rate as i32 {
0 => {
return 162000;
}
1 => {
return 270000;
}
2 => {
return 540000;
}
3 => {
return 810000;
}
4 => {
return 216000;
}
5 => {
return 243000;
}
6 => {
return 324000;
}
7 => {
return 432000;
}
8 => {
return 1000000;
}
9 => {
return 1350000;
}
10 => {
return 2000000;
}
11 => {
return 675000;
}
_ => {
let mut __v0: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"Missing case (%s == %ld)\n".as_ptr() as *const u8, c"rate".as_ptr() as *mut u8, rate as i64) };
let mut __v1: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v1 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v2: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_intel_lt_phy_get_dp_clock_3, sym __slate_asm_string_intel_lt_phy_get_dp_clock_4, const 1262 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v2 = bug;
unsafe { __v1.unwrap()(__v2, c"rate".as_ptr() as *mut u8, rate as i64) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
return 0;
}
}
unsafe { core::mem::zeroed() }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_711: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn intel_lt_phy_config_changed(encoder: *mut intel_encoder, ltpll: *const intel_lt_phy_pll_state, port_clock: u32) -> bool {
let val: u8 = intel_lt_phy_read(encoder, BIT!(0i32) as u8, 3074u16);
let mut rate: u8 = 0;
let mut clock: u32 = 0;
let mut __v0: u8;
loop {
if (__bf_cast_unsigned__2!((((const { core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1277_3>() }) as u64) as i32).wrapping_add((((1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 3i32 & (((1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> ((const { core::mem::size_of::<u8>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6)) as u8) as i32) > (__bf_cast_unsigned__2!(18446744073709551615u64) as i32) {
unsafe { __compiletime_assert_712() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let mut __v2: u8;
loop {
if false {
unsafe { __compiletime_assert_713() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (((((const { core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1277_5>() }) as u64) as i32).wrapping_add((((1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 3i32 & (((1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul((const { core::mem::size_of::<u8>() }) as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> ((const { core::mem::size_of::<u8>() }) as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6)) as u8) as i32) == 0 {
unsafe { __compiletime_assert_714() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if (if true { (!((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1277_6>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 3i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6)) as u8) as i32) >> ((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1277_7>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 3i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6)) as u8) as u64).trailing_zeros() as i32)) as u32) & 0u32 } else { 0u32 }) != 0 {
unsafe { __compiletime_assert_715() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
if ((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1277_8>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 3i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6)) as u8) as i32) as u64).wrapping_add(1u64 << ((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1277_9>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 3i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6)) as u8) as u64).trailing_zeros() as i32)) & ((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1277_10>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 3i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6)) as u8) as i32) as u64).wrapping_add(1u64 << ((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1277_11>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 3i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6)) as u8) as u64).trailing_zeros() as i32)).wrapping_sub(1) != 0 {
unsafe { __compiletime_assert_716() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = (((val as i32) & (((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1277_13>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 3i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6)) as u8) as i32)) >> ((((core::mem::size_of::<__SlateRecord_intel_lt_phy_c_1277_14>() as i32).wrapping_add((((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) << 3i32 & (((1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)).wrapping_sub(1).wrapping_add(1i32 << 8u64.wrapping_mul(core::mem::size_of::<u8>() as u64).wrapping_sub(1).wrapping_sub(((((1i32.wrapping_neg() as u8) as i32) < 1) as i32) as u64)) as u8) as i32) >> (core::mem::size_of::<u8>() as u64).wrapping_mul(8).wrapping_sub(1).wrapping_sub(6)) as u8) as u64).trailing_zeros() as i32)) as u8;
__v0 = __v2;
rate = __v0;
// The only time we do not reconfigure the PLL is when we are
// using 1.62 Gbps clock since PHY PLL defaults to that
// otherwise we always need to reconfigure it.
if intel_lt_phy_is_dp(ltpll) {
clock = intel_lt_phy_get_dp_clock(rate);
if port_clock == 1620000 && port_clock == clock {
return false;
}
}
true
}

fn intel_lt_phy_transaction_begin(encoder: *mut intel_encoder) -> *mut ref_tracker {
let display: *mut intel_display = to_intel_display!(encoder);
let intel_dp: *mut intel_dp = enc_to_intel_dp(encoder);
unsafe { intel_psr_pause(intel_dp) };
let wakeref: *mut ref_tracker = unsafe { intel_display_power_get(display, 73u32) };
wakeref
}

fn intel_lt_phy_transaction_end(encoder: *mut intel_encoder, wakeref: *mut ref_tracker) {
let display: *mut intel_display = to_intel_display!(encoder);
let intel_dp: *mut intel_dp = enc_to_intel_dp(encoder);
unsafe { intel_psr_resume(intel_dp) };
intel_display_power_put(display, 73u32, wakeref);
}

fn intel_lt_phy_pll_tables_get(crtc_state: *mut intel_crtc_state, encoder: *mut intel_encoder) -> *const intel_lt_phy_pll_params {
if intel_crtc_has_dp_encoder(crtc_state as *const intel_crtc_state) {
if intel_crtc_has_type(crtc_state as *const intel_crtc_state, 8u32) {
return unsafe { core::ptr::addr_of!(xe3plpd_lt_edp_tables.0) as *const intel_lt_phy_pll_params };
}
return unsafe { core::ptr::addr_of!(xe3plpd_lt_dp_tables.0) as *const intel_lt_phy_pll_params };
} else {
if intel_crtc_has_type(crtc_state as *const intel_crtc_state, 6u32) {
return unsafe { core::ptr::addr_of!(xe3plpd_lt_hdmi_tables.0) as *const intel_lt_phy_pll_params };
}
}
let mut __v0: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"Missing case (%s == %ld)\n".as_ptr() as *const u8, c"encoder->type".as_ptr() as *mut u8, ((*encoder).r#type as u32) as i64) };
let mut __v1: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v1 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v2: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_intel_lt_phy_pll_tables_get_3, sym __slate_asm_string_intel_lt_phy_pll_tables_get_4, const 1327 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v2 = bug;
unsafe { __v1.unwrap()(__v2, c"encoder->type".as_ptr() as *mut u8, ((*encoder).r#type as u32) as i64) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
core::ptr::null::<intel_lt_phy_pll_params>()
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_717: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn intel_lt_phy_pll_is_ssc_enabled(crtc_state: *mut intel_crtc_state, encoder: *mut intel_encoder) -> bool {
let display: *mut intel_display = to_intel_display!(encoder);
if intel_crtc_has_dp_encoder(crtc_state as *const intel_crtc_state) {
if unsafe { intel_panel_use_ssc(display) } {
let intel_dp: *mut intel_dp = enc_to_intel_dp(encoder);
return ((unsafe { *((*intel_dp).dpcd.as_mut_ptr() as *mut u8).wrapping_offset(3 as isize) }) as i32) & 1i32 != 0;
}
}
false
}

fn mul_q32_u32(a_q32: u64, b: u32) -> u64 {
let mut p0: u64 = 0;
let mut p1: u64 = 0;
let mut carry: u64 = 0;
let mut result: u64 = 0;
let x_hi: u64 = a_q32 >> 32i32;
let x_lo: u64 = a_q32 & 4294967295;
p0 = x_lo.wrapping_mul(b as u64);
p1 = x_hi.wrapping_mul(b as u64);
carry = p0 >> 32i32;
result = (p1 << 32i32).wrapping_add(carry << 32i32).wrapping_add(p0 & 4294967295);
result
}

fn calculate_target_dco_and_loop_cnt(frequency_khz: u32, target_dco_mhz: &mut u64, loop_cnt: &mut u32) -> bool {
let mut ppm_value: u32 = 1;
let mut dco_min_freq: u32 = 11850;
let mut dco_max_freq: u32 = 16200;
let dco_min_freq_low: u32 = 10000;
let dco_max_freq_low: u32 = 12000;
let mut val: u64 = 0;
let refclk_khz: u64 = 38400;
let mut m2div: u64 = 0;
let mut val_with_frac: u64 = 0;
let mut ppm: u64 = 0;
let mut temp0: u64 = 0;
let mut temp1: u64 = 0;
let mut scale: u64 = 0;
let mut y: i32 = 0;
for ppm_cnt in 0i32..5i32 {
ppm_value = (if ppm_cnt == 2 { 2i32 } else { 1i32 }) as u32;
for dco_count in 0i32..2i32 {
if dco_count == 1 {
dco_min_freq = dco_min_freq_low;
dco_max_freq = dco_max_freq_low;
}
y = 2;
while y <= 255 {
val = div64_u64((y as u64).wrapping_mul(frequency_khz as u64), 200u64);
m2div = div64_u64(val << 32i32, refclk_khz);
m2div = mul_q32_u32(m2div, 500u32);
val_with_frac = mul_q32_u32(m2div, refclk_khz as u32);
val_with_frac = div64_u64(val_with_frac, 500u64);
temp1 = Q32_TO_INT!(val_with_frac);
temp0 = if temp1 > val { temp1.wrapping_sub(val) } else { val.wrapping_sub(temp1) };
ppm = div64_u64(temp0, val);
if temp1 >= (dco_min_freq as u64) && temp1 <= (dco_max_freq as u64) && ppm < (ppm_value as u64) {
// Round to two places
scale = 42949672u64;
temp0 = DIV_ROUND_UP_ULL__3!(val_with_frac, scale);
unsafe {
*(target_dco_mhz as *mut u64) = temp0.wrapping_mul(scale);
}
unsafe {
*(loop_cnt as *mut u32) = y as u32;
}
return true;
}
y = y.wrapping_add(2);
}
}
}
false
}

fn set_phy_vdr_addresses(p: &mut lt_phy_params, pll_type: i32) {
unsafe {
(*(p as *mut lt_phy_params)).pll_reg4.addr = PLL_REG_ADDR!(34064i32, pll_type != 0) as u32;
}
unsafe {
(*(p as *mut lt_phy_params)).pll_reg3.addr = PLL_REG_ADDR!(34060i32, pll_type != 0) as u32;
}
unsafe {
(*(p as *mut lt_phy_params)).pll_reg5.addr = PLL_REG_ADDR!(34068i32, pll_type != 0) as u32;
}
unsafe {
(*(p as *mut lt_phy_params)).pll_reg57.addr = PLL_REG_ADDR!(34276i32, pll_type != 0) as u32;
}
unsafe {
(*(p as *mut lt_phy_params)).lf.addr = PLL_REG_ADDR!(34316i32, pll_type != 0) as u32;
}
unsafe {
(*(p as *mut lt_phy_params)).tdc.addr = PLL_REG_ADDR!(34320i32, pll_type != 0) as u32;
}
unsafe {
(*(p as *mut lt_phy_params)).ssc.addr = PLL_REG_ADDR!(34324i32, pll_type != 0) as u32;
}
unsafe {
(*(p as *mut lt_phy_params)).bias2.addr = PLL_REG_ADDR!(34328i32, pll_type != 0) as u32;
}
unsafe {
(*(p as *mut lt_phy_params)).bias_trim.addr = PLL_REG_ADDR!(34376i32, pll_type != 0) as u32;
}
unsafe {
(*(p as *mut lt_phy_params)).dco_med.addr = PLL_REG_ADDR!(34368i32, pll_type != 0) as u32;
}
unsafe {
(*(p as *mut lt_phy_params)).dco_fine.addr = PLL_REG_ADDR!(34380i32, pll_type != 0) as u32;
}
unsafe {
(*(p as *mut lt_phy_params)).ssc_inj.addr = PLL_REG_ADDR!(34340i32, pll_type != 0) as u32;
}
unsafe {
(*(p as *mut lt_phy_params)).surv_bonus.addr = PLL_REG_ADDR!(34372i32, pll_type != 0) as u32;
}
}

fn compute_ssc(p: &mut lt_phy_params, ana_cfg: u32) {
let ssc_stepsize: i32 = 0;
let ssc_steplen: i32 = 0;
let ssc_steplog: i32 = 0;
unsafe {
(*(p as *mut lt_phy_params)).ssc.val = 2147483648u32 | ana_cfg << 24i32 | ((ssc_steplog << 16i32) as u32) | ((ssc_stepsize << 8i32) as u32) | (ssc_steplen as u32);
}
}

fn compute_bias2(p: &mut lt_phy_params) {
let ssc_en_local: u32 = 0;
let dynctrl_ovrd_en: u64 = 0;
unsafe {
(*(p as *mut lt_phy_params)).bias2.val = (dynctrl_ovrd_en << 31i32 | ((ssc_en_local << 30i32) as u64) | 8388608u64 | 16777216u64 | 2097152u64 | 256u64) as u32;
}
}

fn compute_tdc(p: &mut lt_phy_params, tdc_fine: u64) {
let settling_time: u32 = 15;
let bias_ovr_en: u32 = 1;
let coldstart: u32 = 1;
let true_lock: u32 = 2;
let early_lock: u32 = 1;
let lock_ovr_en: u32 = 1;
let lock_thr: u32 = (if tdc_fine != 0 { 3i32 } else { 5i32 }) as u32;
let unlock_thr: u32 = (if tdc_fine != 0 { 5i32 } else { 11i32 }) as u32;
unsafe {
(*(p as *mut lt_phy_params)).tdc.val = 2147483648u32.wrapping_add(settling_time << 16i32).wrapping_add(bias_ovr_en << 15i32).wrapping_add(lock_ovr_en << 14i32).wrapping_add(coldstart << 12i32).wrapping_add(true_lock << 10i32).wrapping_add(early_lock << 8i32).wrapping_add(unlock_thr << 4i32).wrapping_add(lock_thr);
}
}

fn compute_dco_med(p: &mut lt_phy_params) {
let cselmed_en: u32 = 0;
let cselmed_dyn_adj: u32 = 0;
let cselmed_ratio: u32 = 39;
let cselmed_thr: u32 = 8;
unsafe {
(*(p as *mut lt_phy_params)).dco_med.val = (cselmed_en << 31i32).wrapping_add(cselmed_dyn_adj << 30i32).wrapping_add(cselmed_ratio << 24i32).wrapping_add(cselmed_thr << 21i32);
}
}

fn compute_dco_fine(p: &mut lt_phy_params, dco_12g: u32) {
let dco_fine0_tune_2_0: u32 = (if dco_12g != 0 { 4i32 } else { 3i32 }) as u32;
let dco_fine1_tune_2_0: u32 = 2;
let dco_fine2_tune_2_0: u32 = (if dco_12g != 0 { 2i32 } else { 1i32 }) as u32;
let dco_fine3_tune_2_0: u32 = 5;
let dco_dith0_tune_2_0: u32 = (if dco_12g != 0 { 4i32 } else { 3i32 }) as u32;
let dco_dith1_tune_2_0: u32 = 2;
unsafe {
(*(p as *mut lt_phy_params)).dco_fine.val = (dco_dith1_tune_2_0 << 19i32).wrapping_add(dco_dith0_tune_2_0 << 16i32).wrapping_add(dco_fine3_tune_2_0 << 11i32).wrapping_add(dco_fine2_tune_2_0 << 8i32).wrapping_add(dco_fine1_tune_2_0 << 3i32).wrapping_add(dco_fine0_tune_2_0);
}
}

#[unsafe(no_mangle)]
extern "C" fn intel_lt_phy_calculate_hdmi_state(lt_state: *mut intel_lt_phy_pll_state, frequency_khz: u32) -> i32 {
let mut found: bool = false;
let mut p: lt_phy_params = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(p) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 104usize);
}
let dco_fmin: u32 = 11850;
let refclk_khz: u64 = 38400;
let refclk_mhz_int: u32 = 38u32;
let mut m2div: u64 = 0;
let mut target_dco_mhz: u64 = 0;
let mut tdc_fine: u64 = 0;
let mut tdc_targetcnt: u64 = 0;
let mut feedfwd_gain: u64 = 0;
let mut feedfwd_cal_en: u64 = 0;
let mut tdc_res: u64 = 30;
let mut prop_coeff: u32 = 0;
let mut int_coeff: u32 = 0;
let ndiv: u32 = 1;
let m1div: u32 = 1;
let mut m2div_int: u32 = 0;
let mut m2div_frac: u32 = 0;
let mut frac_en: u32 = 0;
let mut ana_cfg: u32 = 0;
let mut loop_cnt: u32 = 0;
let gain_ctrl: u32 = 2;
let mut postdiv: u32 = 0;
let mut dco_12g: u32 = 0;
let mut pll_type: u32 = 0;
let d1: u32 = 2;
let d3: u32 = 5;
let mut d4: u32 = 0;
let d5: u32 = 0;
let d6: u32 = 0;
let mut d6_new: u32 = 0;
let mut d7: u32 = 0;
let mut d8: u32 = 0;
let bonus_7_0: u32 = 0;
let csel2fo: u32 = 11;
let csel2fo_ovrd_en: u32 = 1;
let mut temp0: u64 = 0;
let mut temp1: u64 = 0;
let mut temp2: u64 = 0;
let mut temp3: u64 = 0;
p.surv_bonus.val = bonus_7_0 << 16i32;
p.pll_reg4.val = (refclk_mhz_int << 17i32).wrapping_add(ndiv << 9i32).wrapping_add(16u32);
p.bias_trim.val = (csel2fo_ovrd_en << 30i32).wrapping_add(csel2fo << 24i32);
p.ssc_inj.val = 0;
found = calculate_target_dco_and_loop_cnt(frequency_khz, unsafe { &mut (*core::ptr::addr_of_mut!(target_dco_mhz)) }, unsafe { &mut (*core::ptr::addr_of_mut!(loop_cnt)) });
if !found {
return 22i32.wrapping_neg();
}
m2div = div64_u64(target_dco_mhz, refclk_khz.wrapping_mul(ndiv as u64).wrapping_mul(m1div as u64));
m2div = mul_q32_u32(m2div, 1000u32);
if Q32_TO_INT!(m2div) > 511 {
return 22i32.wrapping_neg();
}
m2div_int = Q32_TO_INT!(m2div) as u32;
m2div_frac = Q32_TO_FRAC!(m2div) as u32;
frac_en = (if m2div_frac > 0 { 1i32 } else { 0i32 }) as u32;
if frac_en > 0 {
tdc_res = 70;
} else {
tdc_res = 36;
}
tdc_fine = (if tdc_res > 50 { 1i32 } else { 0i32 }) as u64;
temp0 = tdc_res.wrapping_mul(40).wrapping_mul(11);
temp1 = div64_u64(40000000u64.wrapping_add(temp0).wrapping_mul(500), temp0.wrapping_mul(refclk_khz));
temp2 = div64_u64(temp0.wrapping_mul(refclk_khz), 1000u64);
temp3 = div64_u64(80000000u64.wrapping_add(temp2), temp2);
tdc_targetcnt = (if tdc_res < 50 { temp1 as i32 } else { temp3 as i32 }) as u64;
tdc_targetcnt = ((tdc_targetcnt / 2) as i32) as u64;
temp0 = mul_q32_u32(target_dco_mhz, tdc_res as u32);
temp0 >>= 32i32;
let __v2: u64 = if m2div_frac > 0 {
div64_u64((m1div as u64).wrapping_mul(10000000), temp0)
} else {
0
};
feedfwd_gain = __v2;
feedfwd_cal_en = frac_en as u64;
temp0 = (Q32_TO_INT!(target_dco_mhz) as u32) as u64;
prop_coeff = (if temp0 >= (dco_fmin as u64) { 3i32 } else { 4i32 }) as u32;
int_coeff = (if temp0 >= (dco_fmin as u64) { 7i32 } else { 8i32 }) as u32;
ana_cfg = (if temp0 >= (dco_fmin as u64) { 8i32 } else { 6i32 }) as u32;
dco_12g = (if temp0 >= (dco_fmin as u64) { 0i32 } else { 1i32 }) as u32;
if temp0 > 12960 {
d7 = 10;
} else {
d7 = 8;
}
d8 = loop_cnt / 2;
d4 = d8.wrapping_mul(2);
// Compute pll_reg3,5,57 & lf
p.pll_reg3.val = (d4 << 21i32).wrapping_add(d3 << 18i32).wrapping_add(d1 << 15i32).wrapping_add(m2div_int << 5i32);
p.pll_reg5.val = m2div_frac;
postdiv = if d5 == 0 { 9 } else { d5 };
d6_new = if d6 == 0 { 40 } else { d6 };
p.pll_reg57.val = (d7 << 24i32).wrapping_add(postdiv << 15i32).wrapping_add(d8 << 7i32).wrapping_add(d6_new);
p.lf.val = ((frac_en << 31i32).wrapping_add(1073741824u32).wrapping_add(frac_en << 29i32) as u64).wrapping_add(feedfwd_cal_en << 28i32).wrapping_add(tdc_fine << 27i32).wrapping_add((gain_ctrl << 24i32) as u64).wrapping_add(feedfwd_gain << 16i32).wrapping_add((int_coeff << 12i32) as u64).wrapping_add((prop_coeff << 8i32) as u64).wrapping_add(tdc_targetcnt) as u32;
compute_ssc(unsafe { &mut (*core::ptr::addr_of_mut!(p)) }, ana_cfg);
compute_bias2(unsafe { &mut (*core::ptr::addr_of_mut!(p)) });
compute_tdc(unsafe { &mut (*core::ptr::addr_of_mut!(p)) }, tdc_fine);
compute_dco_med(unsafe { &mut (*core::ptr::addr_of_mut!(p)) });
compute_dco_fine(unsafe { &mut (*core::ptr::addr_of_mut!(p)) }, dco_12g);
pll_type = (if frequency_khz == 10000 || frequency_khz == 20000 || frequency_khz == 2500 || dco_12g == 1 { 0i32 } else { 1i32 }) as u32;
set_phy_vdr_addresses(unsafe { &mut (*core::ptr::addr_of_mut!(p)) }, pll_type as i32);
unsafe {
*((*lt_state).config.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) = 132;
}
unsafe {
*((*lt_state).config.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) = 45;
}
ADDR_ASSIGN!(0i32, p.pll_reg4, lt_state);
ADDR_ASSIGN!(1i32, p.pll_reg3, lt_state);
ADDR_ASSIGN!(2i32, p.pll_reg5, lt_state);
ADDR_ASSIGN!(3i32, p.pll_reg57, lt_state);
ADDR_ASSIGN!(4i32, p.lf, lt_state);
ADDR_ASSIGN!(5i32, p.tdc, lt_state);
ADDR_ASSIGN!(6i32, p.ssc, lt_state);
ADDR_ASSIGN!(7i32, p.bias2, lt_state);
ADDR_ASSIGN!(8i32, p.bias_trim, lt_state);
ADDR_ASSIGN!(9i32, p.dco_med, lt_state);
ADDR_ASSIGN!(10i32, p.dco_fine, lt_state);
ADDR_ASSIGN!(11i32, p.ssc_inj, lt_state);
ADDR_ASSIGN!(12i32, p.surv_bonus, lt_state);
DATA_ASSIGN!(0i32, p.pll_reg4, lt_state);
DATA_ASSIGN!(1i32, p.pll_reg3, lt_state);
DATA_ASSIGN!(2i32, p.pll_reg5, lt_state);
DATA_ASSIGN!(3i32, p.pll_reg57, lt_state);
DATA_ASSIGN!(4i32, p.lf, lt_state);
DATA_ASSIGN!(5i32, p.tdc, lt_state);
DATA_ASSIGN!(6i32, p.ssc, lt_state);
DATA_ASSIGN!(7i32, p.bias2, lt_state);
DATA_ASSIGN!(8i32, p.bias_trim, lt_state);
DATA_ASSIGN!(9i32, p.dco_med, lt_state);
DATA_ASSIGN!(10i32, p.dco_fine, lt_state);
DATA_ASSIGN!(11i32, p.ssc_inj, lt_state);
DATA_ASSIGN!(12i32, p.surv_bonus, lt_state);
0
}

fn intel_lt_phy_calc_hdmi_port_clock(display: *mut intel_display, lt_state: *const intel_lt_phy_pll_state) -> i32 {
let mut clk: i32 = 0;
let mut d8: u32 = 0;
let mut pll_reg_5: u32 = 0;
let mut pll_reg_3: u32 = 0;
let mut pll_reg_57: u32 = 0;
let mut m2div_frac: u32 = 0;
let mut m2div_int: u32 = 0;
let mut temp0: u64 = 0;
let mut temp1: u64 = 0;
// The algorithm uses '+' to combine bitfields when
// constructing PLL_reg3 and PLL_reg57:
// PLL_reg57 = (D7 << 24) + (postdiv << 15) + (D8 << 7) + D6_new;
// PLL_reg3 = (D4 << 21) + (D3 << 18) + (D1 << 15) + (m2div_int << 5);
//
// However, this is likely intended to be a bitwise OR operation,
// as each field occupies distinct, non-overlapping bits in the register.
//
// PLL_reg57 is composed of following fields packed into a 32-bit value:
// - D7: max value 10 -> fits in 4 bits -> placed at bits 24-27
// - postdiv: max value 9 -> fits in 4 bits -> placed at bits 15-18
// - D8: derived from loop_cnt / 2, max 127 -> fits in 7 bits
// 	(though 8 bits are given to it) -> placed at bits 7-14
// - D6_new: fits in lower 7 bits -> placed at bits 0-6
// PLL_reg57 = (D7 << 24) | (postdiv << 15) | (D8 << 7) | D6_new;
//
// Similarly, PLL_reg3 is packed as:
// - D4: max value 256 -> fits in 9 bits -> placed at bits 21-29
// - D3: max value 9 -> fits in 4 bits -> placed at bits 18-21
// - D1: max value 2 -> fits in 2 bits -> placed at bits 15-16
// - m2div_int: max value 511 -> fits in 9 bits (10 bits allocated)
//   -> placed at bits 5-14
// PLL_reg3 = (D4 << 21) | (D3 << 18) | (D1 << 15) | (m2div_int << 5);
pll_reg_5 = REGVAL!(2i32, lt_state) as u32;
pll_reg_3 = REGVAL!(1i32, lt_state) as u32;
pll_reg_57 = REGVAL!(3i32, lt_state) as u32;
m2div_frac = pll_reg_5;
// From forward algorithm we know
// m2div = 2 * m2
// val = y * frequency * 5
// So now,
// frequency = (m2 * 2 * refclk_khz / (d8 * 10))
// frequency = (m2div * refclk_khz / (d8 * 10))
d8 = (pll_reg_57 & REG_GENMASK__184!(14i32, 7i32)) >> 7i32;
if d8 == 0 {
let mut __v0: i64;
let __ret_warn_on: i32 = !(c"Invalid port clock using lowest HDMI portclock\n".as_ptr() as *mut u8).is_null() as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(\"Invalid port clock using lowest HDMI portclock\\n\")".as_ptr() as *mut u8) };
let mut __v1: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v1 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v2: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_intel_lt_phy_calc_hdmi_port_clock_3, sym __slate_asm_string_intel_lt_phy_calc_hdmi_port_clock_4, const 1702 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v2 = bug;
unsafe { __v1.unwrap()(__v2, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(\"Invalid port clock using lowest HDMI portclock\\n\")".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
return unsafe { xe3plpd_lt_hdmi_tables.0[0].clock_rate };
}
m2div_int = (pll_reg_3 & REG_GENMASK__116!(14i32, 5i32)) >> 5i32;
temp0 = (m2div_frac as u64).wrapping_mul(38400) >> 32i32;
temp1 = (m2div_int as u64).wrapping_mul(38400);
clk = div_u64(temp1.wrapping_add(temp0), d8.wrapping_mul(10)) as i32;
clk
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_718: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_729: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[unsafe(no_mangle)]
extern "C" fn intel_lt_phy_pll_calc_state(crtc_state: *mut intel_crtc_state, encoder: *mut intel_encoder, hw_state: *mut intel_dpll_hw_state) -> i32 {
let display: *mut intel_display = to_intel_display__5!(crtc_state);
let mut i: i32 = 0;
unsafe { memset(hw_state as *mut (), 0i32, core::mem::size_of::<intel_dpll_hw_state>() as u64) };
let tables: *const intel_lt_phy_pll_params = intel_lt_phy_pll_tables_get(crtc_state, encoder);
if tables.is_null() {
return 22i32.wrapping_neg();
}
while !unsafe { (*tables.wrapping_offset(i as isize)).name }.is_null() {
let clock: i32 = intel_lt_phy_calc_port_clock(display, unsafe { (*tables.wrapping_offset(i as isize)).state });
let mut __v1: i64;
let __ret_warn_on: i32 = !(!(!unsafe { intel_dpll_clock_matches(clock, (*tables.wrapping_offset(i as isize)).clock_rate) })) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(!intel_dpll_clock_matches(clock, tables[i].clock_rate))".as_ptr() as *mut u8) };
let mut __v2: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v2 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v3: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_intel_lt_phy_pll_calc_state_4, sym __slate_asm_string_intel_lt_phy_pll_calc_state_5, const 1761 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v3 = bug;
unsafe { __v2.unwrap()(__v3, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON(!intel_dpll_clock_matches(clock, tables[i].clock_rate))".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
if unsafe { intel_dpll_clock_matches((*crtc_state).port_clock, clock) } {
unsafe {
(*hw_state).__slate_anon_0.ltpll = *(*tables.wrapping_offset(i as isize)).state;
}
if intel_crtc_has_dp_encoder(crtc_state as *const intel_crtc_state) {
if intel_crtc_has_type(crtc_state as *const intel_crtc_state, 8u32) {
unsafe {
*((*hw_state).__slate_anon_0.ltpll.config.as_mut_ptr() as *mut u8).wrapping_offset(2 as isize) = 1;
}
}
}
unsafe {
(*hw_state).__slate_anon_0.ltpll.ssc_enabled = intel_lt_phy_pll_is_ssc_enabled(crtc_state, encoder);
}
unsafe {
(*hw_state).__slate_anon_0.ltpll.lane_count = (*crtc_state).lane_count as i32;
}
return 0;
}
i = i.wrapping_add(1);
}
if intel_crtc_has_type(crtc_state as *const intel_crtc_state, 6u32) {
unsafe {
(*hw_state).__slate_anon_0.ltpll.lane_count = (*crtc_state).lane_count as i32;
}
return intel_lt_phy_calculate_hdmi_state(unsafe { core::ptr::addr_of_mut!((*hw_state).__slate_anon_0.ltpll) }, (unsafe { (*crtc_state).port_clock }) as u32);
}
22i32.wrapping_neg()
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_730: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[unsafe(no_mangle)]
extern "C" fn intel_lt_phy_tbt_pll_calc_state(hw_state: *mut intel_dpll_hw_state) {
unsafe { memset(hw_state as *mut (), 0i32, core::mem::size_of::<intel_dpll_hw_state>() as u64) };
unsafe {
(*hw_state).__slate_anon_0.ltpll.tbt_mode = true;
}
}

fn intel_lt_phy_program_pll(encoder: *mut intel_encoder, ltpll: *const intel_lt_phy_pll_state) {
let owned_lane_mask: u8 = intel_lt_phy_get_owned_lane_mask(encoder);
let mut i: i32 = 0;
let mut j: i32 = 0;
let mut k: i32 = 0;
intel_lt_phy_write(encoder, owned_lane_mask, 3074u16, unsafe { *((*ltpll).config.as_ptr() as *const u8).wrapping_offset(0 as isize) }, true);
intel_lt_phy_write(encoder, BIT!(0i32) as u8, 3075u16, unsafe { *((*ltpll).config.as_ptr() as *const u8).wrapping_offset(1 as isize) }, true);
intel_lt_phy_write(encoder, owned_lane_mask, 3267u16, unsafe { *((*ltpll).config.as_ptr() as *const u8).wrapping_offset(2 as isize) }, true);
while i <= 12 {
intel_lt_phy_write(encoder, BIT!(0i32) as u8, LT_PHY_VDR_X_ADDR_MSB!(i) as u16, unsafe { *((*ltpll).addr_msb.as_ptr() as *const u8).wrapping_offset(i as isize) }, true);
intel_lt_phy_write(encoder, BIT!(0i32) as u8, LT_PHY_VDR_X_ADDR_LSB!(i) as u16, unsafe { *((*ltpll).addr_lsb.as_ptr() as *const u8).wrapping_offset(i as isize) }, true);
j = 3;
k = 0;
while j >= 0 {
intel_lt_phy_write(encoder, BIT!(0i32) as u8, LT_PHY_VDR_X_DATAY!(i, j) as u16, unsafe { *((*((*ltpll).data.as_ptr() as *const [u8; 4]).wrapping_offset(i as isize)).as_ptr() as *const u8).wrapping_offset(k as isize) }, true);
j = j.wrapping_sub(1);
k = k.wrapping_add(1);
}
i = i.wrapping_add(1);
}
}

fn intel_lt_phy_enable_disable_tx(encoder: *mut intel_encoder, ltpll: *const intel_lt_phy_pll_state) {
let mut __slate_compound_7: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_8: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_9: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_10: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_11: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_12: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_13: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_14: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let dig_port: *mut intel_digital_port = enc_to_dig_port(encoder);
let lane_reversal: bool = unsafe { (*dig_port).lane_reversal };
let lane_count: u8 = (unsafe { (*ltpll).lane_count }) as u8;
let is_dp_alt: bool = unsafe { intel_tc_port_in_dp_alt_mode(dig_port) };
let tc_pin: u32 = unsafe { intel_tc_port_get_pin_assignment(dig_port) };
let mut transmitter_mask: u8 = 0;
// We have a two transmitters per lane and total of 2 PHY lanes so a total
// of 4 transmitters. We prepare a mask of the lanes that need to be activated
// and the transmitter which need to be activated for each lane. TX 0,1 correspond
// to LANE0 and TX 2, 3 correspond to LANE1.
match lane_count as i32 {
1 => {
transmitter_mask = (if lane_reversal { REG_BIT8__37!(3i32) as i32 } else { REG_BIT8__15!(0i32) as i32 }) as u8;
if is_dp_alt {
if (tc_pin as u32) == 4 {
transmitter_mask = REG_BIT8__32!(0i32);
} else {
transmitter_mask = REG_BIT8__26!(1i32);
}
}
}
2 => {
transmitter_mask = (if lane_reversal { REG_GENMASK8__20!(3i32, 2i32) as i32 } else { REG_GENMASK8__23!(1i32, 0i32) as i32 }) as u8;
if is_dp_alt {
transmitter_mask = REG_GENMASK8__2!(1i32, 0i32);
}
}
3 => {
transmitter_mask = (if lane_reversal { REG_GENMASK8__3!(3i32, 1i32) as i32 } else { REG_GENMASK8__6!(2i32, 0i32) as i32 }) as u8;
if is_dp_alt {
transmitter_mask = REG_GENMASK8__12!(2i32, 0i32);
}
}
4 => {
transmitter_mask = REG_GENMASK8__4!(3i32, 0i32);
}
_ => {
let mut __v1: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"Missing case (%s == %ld)\n".as_ptr() as *const u8, c"lane_count".as_ptr() as *mut u8, lane_count as i64) };
let mut __v2: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v2 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v3: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_intel_lt_phy_enable_disable_tx_4, sym __slate_asm_string_intel_lt_phy_enable_disable_tx_5, const 1865 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v3 = bug;
unsafe { __v2.unwrap()(__v3, c"lane_count".as_ptr() as *mut u8, lane_count as i64) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
transmitter_mask = REG_GENMASK8__10!(3i32, 0i32);
}
}
if ((transmitter_mask as i32) as u64) & BIT!(0i32) != 0 {
intel_lt_phy_p2p_write(encoder, BIT!(0i32) as u8, LT_PHY_TXY_CTL10!(0i32) as u16, REG_BIT8__13!(0i32), unsafe { *{
__slate_compound_7 = __SlateRecord_i915_reg_defs_h_14 { reg: LT_PHY_TXY_CTL10!(0i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_7)
} }, REG_BIT8__20!(0i32));
} else {
intel_lt_phy_p2p_write(encoder, BIT!(0i32) as u8, LT_PHY_TXY_CTL10!(0i32) as u16, 0u8, unsafe { *{
__slate_compound_8 = __SlateRecord_i915_reg_defs_h_14 { reg: LT_PHY_TXY_CTL10!(0i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_8)
} }, 0u8);
}
if ((transmitter_mask as i32) as u64) & BIT!(1i32) != 0 {
intel_lt_phy_p2p_write(encoder, BIT!(0i32) as u8, LT_PHY_TXY_CTL10!(1i32) as u16, REG_BIT8__18!(0i32), unsafe { *{
__slate_compound_9 = __SlateRecord_i915_reg_defs_h_14 { reg: LT_PHY_TXY_CTL10!(1i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_9)
} }, REG_BIT8__12!(0i32));
} else {
intel_lt_phy_p2p_write(encoder, BIT!(0i32) as u8, LT_PHY_TXY_CTL10!(1i32) as u16, 0u8, unsafe { *{
__slate_compound_10 = __SlateRecord_i915_reg_defs_h_14 { reg: LT_PHY_TXY_CTL10!(1i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_10)
} }, 0u8);
}
if ((transmitter_mask as i32) as u64) & BIT!(2i32) != 0 {
intel_lt_phy_p2p_write(encoder, BIT!(1i32) as u8, LT_PHY_TXY_CTL10!(0i32) as u16, REG_BIT8__2!(0i32), unsafe { *{
__slate_compound_11 = __SlateRecord_i915_reg_defs_h_14 { reg: LT_PHY_TXY_CTL10!(0i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_11)
} }, REG_BIT8__31!(0i32));
} else {
intel_lt_phy_p2p_write(encoder, BIT!(1i32) as u8, LT_PHY_TXY_CTL10!(0i32) as u16, 0u8, unsafe { *{
__slate_compound_12 = __SlateRecord_i915_reg_defs_h_14 { reg: LT_PHY_TXY_CTL10!(0i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_12)
} }, 0u8);
}
if ((transmitter_mask as i32) as u64) & BIT!(3i32) != 0 {
intel_lt_phy_p2p_write(encoder, BIT!(1i32) as u8, LT_PHY_TXY_CTL10!(1i32) as u16, REG_BIT8__22!(0i32), unsafe { *{
__slate_compound_13 = __SlateRecord_i915_reg_defs_h_14 { reg: LT_PHY_TXY_CTL10!(1i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_13)
} }, REG_BIT8__35!(0i32));
} else {
intel_lt_phy_p2p_write(encoder, BIT!(1i32) as u8, LT_PHY_TXY_CTL10!(1i32) as u16, 0u8, unsafe { *{
__slate_compound_14 = __SlateRecord_i915_reg_defs_h_14 { reg: LT_PHY_TXY_CTL10!(1i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_14)
} }, 0u8);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_731: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[unsafe(no_mangle)]
extern "C" fn intel_lt_phy_pll_enable(encoder: *mut intel_encoder, pll: *mut intel_dpll, dpll_hw_state: *const intel_dpll_hw_state) {
let mut __slate_compound_1: [u32; 2] = unsafe { core::mem::zeroed() };
let mut __slate_compound_2: [u32; 2] = unsafe { core::mem::zeroed() };
let mut __slate_compound_3: [u32; 2] = unsafe { core::mem::zeroed() };
let mut __slate_compound_4: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_6: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_7: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_8: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_10: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_11: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_12: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_13: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_14: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_16: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_17: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_18: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_19: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_22: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_23: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_25: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_26: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_27: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display!(encoder);
let port_clock: i32 = intel_lt_phy_calc_port_clock(display, unsafe { core::ptr::addr_of!((*dpll_hw_state).__slate_anon_0.ltpll) });
let dig_port: *mut intel_digital_port = enc_to_dig_port(encoder);
let lane_reversal: bool = unsafe { (*dig_port).lane_reversal };
let owned_lane_mask: u8 = intel_lt_phy_get_owned_lane_mask(encoder);
let phy: i32 = unsafe { intel_encoder_to_phy(encoder) };
let port: i32 = unsafe { (*encoder).port };
let mut wakeref: *mut ref_tracker = core::ptr::null_mut::<ref_tracker>();
let lane_phy_pulse_status: u32 = if ((owned_lane_mask as i32) as u64) == BIT!(1i32) | BIT!(0i32) { (unsafe { *((*{
__slate_compound_1 = [REG_BIT__252!(27i32), REG_BIT__366!(26i32)];
core::ptr::addr_of_mut!(__slate_compound_1)
}).as_ptr() as *const u32).wrapping_offset(0 as isize) }) | unsafe { *((*{
__slate_compound_2 = [REG_BIT__1350!(27i32), REG_BIT__353!(26i32)];
core::ptr::addr_of_mut!(__slate_compound_2)
}).as_ptr() as *const u32).wrapping_offset(1 as isize) } } else { unsafe { *((*{
__slate_compound_3 = [REG_BIT__119!(27i32), REG_BIT__548!(26i32)];
core::ptr::addr_of_mut!(__slate_compound_3)
}).as_ptr() as *const u32).wrapping_offset(0 as isize) } };
let mut rate_update: u8 = 0;
wakeref = intel_lt_phy_transaction_begin(encoder);
// 1. Enable MacCLK at default 162 MHz frequency.
intel_lt_phy_lane_reset(encoder, (unsafe { (*dpll_hw_state).__slate_anon_0.ltpll.lane_count }) as u8);
// 2. Program PORT_CLOCK_CTL register to configure clock muxes, gating, and SSC.
intel_lt_phy_program_port_clock_ctl(encoder, unsafe { core::ptr::addr_of!((*dpll_hw_state).__slate_anon_0.ltpll) }, port_clock, lane_reversal);
// 3. Change owned PHY lanes power to Ready state.
intel_lt_phy_powerdown_change_sequence(encoder, owned_lane_mask, 2u8);
// 4. Read the PHY message bus VDR register PHY_VDR_0_Config check enabled PLL type,
// encoded rate and encoded mode.
if intel_lt_phy_config_changed(encoder, unsafe { core::ptr::addr_of!((*dpll_hw_state).__slate_anon_0.ltpll) }, port_clock as u32) {
// 5. Program the PHY internal PLL registers over PHY message bus for the desired
// frequency and protocol type
intel_lt_phy_program_pll(encoder, unsafe { core::ptr::addr_of!((*dpll_hw_state).__slate_anon_0.ltpll) });
// 6. Use the P2P transaction flow
//
// 6.1. Set the PHY VDR register 0xCC4[Rate Control VDR Update] = 1 over PHY message
// bus for Owned PHY Lanes.
//
// 6.2. Poll for P2P Transaction Ready = "1" and read the MAC message bus VDR
// register at offset 0xC00 for Owned PHY Lanes*.
//
// 6.3. Clear P2P transaction Ready bit.
intel_lt_phy_p2p_write(encoder, owned_lane_mask, 3268u16, REG_BIT8__3!(0i32), unsafe { *{
__slate_compound_4 = __SlateRecord_i915_reg_defs_h_14 { reg: 3072 };
core::ptr::addr_of_mut!(__slate_compound_4)
} }, REG_BIT8__28!(0i32));
// 7. Program PORT_CLOCK_CTL[PCLK PLL Request LN0] = 0.
intel_de_rmw(display, if (DISPLAY_VER!(display) as i32) >= 20 { unsafe { *{
__slate_compound_5 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__193!(if (port as i32) >= 3 { port as i32 } else { 7i32.wrapping_add(port as i32).wrapping_sub(0) }, 3i32, 409824i32, 410080i32, 1503840i32, 1504352i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_5)
} } } else { unsafe { *{
__slate_compound_6 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__213!(port, 3i32, 409824i32, 410080i32, 1503840i32, 1504352i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_6)
} } }, XELPDP_LANE_PCLK_PLL_REQUEST__7!(0i32), 0u32);
// 8. Poll for PORT_CLOCK_CTL[PCLK PLL Ack LN0]= 0.
if (unsafe { intel_de_wait_for_clear_us(display, if (DISPLAY_VER!(display) as i32) >= 20 { *{
__slate_compound_7 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__51!(if (port as i32) >= 3 { port as i32 } else { 7i32.wrapping_add(port as i32).wrapping_sub(0) }, 3i32, 409824i32, 410080i32, 1503840i32, 1504352i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_7)
} } else { *{
__slate_compound_8 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__225!(port, 3i32, 409824i32, 410080i32, 1503840i32, 1504352i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_8)
} }, XELPDP_LANE_PCLK_PLL_ACK__6!(0i32), 10u32) }) != 0 {
while false {
}
unsafe { _dev_warn(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] PHY %c PLL MacCLK ack deassertion timeout\n".as_ptr() as *const u8, phy_name!(phy)) };
}
// 9. Follow the Display Voltage Frequency Switching - Sequence Before Frequency
// Change. We handle this step in bxt_set_cdclk().
//
// 10. Program DDI_CLK_VALFREQ to match intended DDI clock frequency.
intel_de_write(display, unsafe { *{
__slate_compound_10 = __SlateRecord_i915_reg_defs_h_14 { reg: _PORT!(unsafe { (*encoder).port }, 409648i32, 409904i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_10)
} }, (unsafe { intel_ddi_link_symbol_clock(encoder, port_clock) }) as u32);
// 11. Program PORT_CLOCK_CTL[PCLK PLL Request LN0] = 1.
intel_de_rmw(display, if (DISPLAY_VER!(display) as i32) >= 20 { unsafe { *{
__slate_compound_11 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__162!(if (port as i32) >= 3 { port as i32 } else { 7i32.wrapping_add(port as i32).wrapping_sub(0) }, 3i32, 409824i32, 410080i32, 1503840i32, 1504352i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_11)
} } } else { unsafe { *{
__slate_compound_12 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__234!(port, 3i32, 409824i32, 410080i32, 1503840i32, 1504352i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_12)
} } }, XELPDP_LANE_PCLK_PLL_REQUEST__5!(0i32), XELPDP_LANE_PCLK_PLL_REQUEST!(0i32));
// 12. Poll for PORT_CLOCK_CTL[PCLK PLL Ack LN0]= 1.
if (unsafe { intel_de_wait_for_set_ms(display, if (DISPLAY_VER!(display) as i32) >= 20 { *{
__slate_compound_13 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__96!(if (port as i32) >= 3 { port as i32 } else { 7i32.wrapping_add(port as i32).wrapping_sub(0) }, 3i32, 409824i32, 410080i32, 1503840i32, 1504352i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_13)
} } else { *{
__slate_compound_14 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__296!(port, 3i32, 409824i32, 410080i32, 1503840i32, 1504352i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_14)
} }, XELPDP_LANE_PCLK_PLL_ACK__3!(0i32), 2u32) }) != 0 {
while false {
}
unsafe { _dev_warn(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] PHY %c PLL MacCLK ack assertion timeout\n".as_ptr() as *const u8, phy_name!(phy)) };
}
// 13. Ungate the forward clock by setting
// PORT_CLOCK_CTL[Forward Clock Ungate] = 1.
intel_de_rmw(display, if (DISPLAY_VER!(display) as i32) >= 20 { unsafe { *{
__slate_compound_16 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__37!(if (port as i32) >= 3 { port as i32 } else { 7i32.wrapping_add(port as i32).wrapping_sub(0) }, 3i32, 409824i32, 410080i32, 1503840i32, 1504352i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_16)
} } } else { unsafe { *{
__slate_compound_17 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__329!(port, 3i32, 409824i32, 410080i32, 1503840i32, 1504352i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_17)
} } }, REG_BIT__1121!(10i32), REG_BIT__994!(10i32));
// 14. SW clears PORT_BUF_CTL2 [PHY Pulse Status].
intel_de_rmw(display, if (DISPLAY_VER!(display) as i32) >= 20 { unsafe { *{
__slate_compound_18 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__50!(if (port as i32) >= 3 { port as i32 } else { 7i32.wrapping_add(port as i32).wrapping_sub(0) }, 3i32, 409604i32, 409860i32, 1503744i32, 1504256i32).wrapping_add(4) as u32 };
core::ptr::addr_of_mut!(__slate_compound_18)
} } } else { unsafe { *{
__slate_compound_19 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__144!(port, 3i32, 409604i32, 409860i32, 1503744i32, 1504256i32).wrapping_add(4) as u32 };
core::ptr::addr_of_mut!(__slate_compound_19)
} } }, lane_phy_pulse_status, lane_phy_pulse_status);
// 15. Clear the PHY VDR register 0xCC4[Rate Control VDR Update] over
// PHY message bus for Owned PHY Lanes.
rate_update = intel_lt_phy_read(encoder, BIT!(0i32) as u8, 3268u16);
let __v20: u8 = rate_update;
rate_update = ((__v20 as i32) & !(REG_BIT8__17!(0i32) as i32)) as u8;
intel_lt_phy_write(encoder, owned_lane_mask, 3268u16, rate_update, true);
// 16. Poll for PORT_BUF_CTL2 register PHY Pulse Status = 1 for Owned PHY Lanes.
if (unsafe { intel_de_wait_for_set_ms(display, if (DISPLAY_VER!(display) as i32) >= 20 { *{
__slate_compound_22 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__325!(if (port as i32) >= 3 { port as i32 } else { 7i32.wrapping_add(port as i32).wrapping_sub(0) }, 3i32, 409604i32, 409860i32, 1503744i32, 1504256i32).wrapping_add(4) as u32 };
core::ptr::addr_of_mut!(__slate_compound_22)
} } else { *{
__slate_compound_23 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__136!(port, 3i32, 409604i32, 409860i32, 1503744i32, 1504256i32).wrapping_add(4) as u32 };
core::ptr::addr_of_mut!(__slate_compound_23)
} }, lane_phy_pulse_status, 1u32) }) != 0 {
while false {
}
unsafe { _dev_warn(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] PHY %c PLL rate not changed\n".as_ptr() as *const u8, phy_name!(phy)) };
}
// 17. SW clears PORT_BUF_CTL2 [PHY Pulse Status].
intel_de_rmw(display, if (DISPLAY_VER!(display) as i32) >= 20 { unsafe { *{
__slate_compound_25 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__265!(if (port as i32) >= 3 { port as i32 } else { 7i32.wrapping_add(port as i32).wrapping_sub(0) }, 3i32, 409604i32, 409860i32, 1503744i32, 1504256i32).wrapping_add(4) as u32 };
core::ptr::addr_of_mut!(__slate_compound_25)
} } } else { unsafe { *{
__slate_compound_26 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__129!(port, 3i32, 409604i32, 409860i32, 1503744i32, 1504256i32).wrapping_add(4) as u32 };
core::ptr::addr_of_mut!(__slate_compound_26)
} } }, lane_phy_pulse_status, lane_phy_pulse_status);
} else {
intel_de_write(display, unsafe { *{
__slate_compound_27 = __SlateRecord_i915_reg_defs_h_14 { reg: _PORT!(unsafe { (*encoder).port }, 409648i32, 409904i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_27)
} }, (unsafe { intel_ddi_link_symbol_clock(encoder, port_clock) }) as u32);
}
// 18. Follow the Display Voltage Frequency Switching - Sequence After Frequency Change.
// We handle this step in bxt_set_cdclk()
//
// 19. Move the PHY powerdown state to Active and program to enable/disable transmitters
intel_lt_phy_powerdown_change_sequence(encoder, owned_lane_mask, 0u8);
intel_lt_phy_enable_disable_tx(encoder, unsafe { core::ptr::addr_of!((*dpll_hw_state).__slate_anon_0.ltpll) });
intel_lt_phy_transaction_end(encoder, wakeref);
}

#[unsafe(no_mangle)]
extern "C" fn intel_lt_phy_pll_disable(encoder: *mut intel_encoder) {
let mut __slate_compound_1: [u32; 2] = unsafe { core::mem::zeroed() };
let mut __slate_compound_2: [u32; 2] = unsafe { core::mem::zeroed() };
let mut __slate_compound_3: [u32; 2] = unsafe { core::mem::zeroed() };
let mut __slate_compound_4: [u32; 2] = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: [u32; 2] = unsafe { core::mem::zeroed() };
let mut __slate_compound_6: [u32; 2] = unsafe { core::mem::zeroed() };
let mut __slate_compound_7: [u32; 2] = unsafe { core::mem::zeroed() };
let mut __slate_compound_8: [u32; 2] = unsafe { core::mem::zeroed() };
let mut __slate_compound_9: [u32; 2] = unsafe { core::mem::zeroed() };
let mut __slate_compound_10: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_11: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_12: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_13: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_14: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_15: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_17: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_18: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_19: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_20: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_21: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_22: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_23: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_25: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_26: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_27: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display!(encoder);
let phy: i32 = unsafe { intel_encoder_to_phy(encoder) };
let port: i32 = unsafe { (*encoder).port };
let owned_lane_mask: u8 = intel_lt_phy_get_owned_lane_mask(encoder);
let lane_pipe_reset: u32 = if ((owned_lane_mask as i32) as u64) == BIT!(1i32) | BIT!(0i32) { (unsafe { *((*{
__slate_compound_1 = [REG_BIT__31!(31i32), REG_BIT__312!(30i32)];
core::ptr::addr_of_mut!(__slate_compound_1)
}).as_ptr() as *const u32).wrapping_offset(0 as isize) }) | unsafe { *((*{
__slate_compound_2 = [REG_BIT__15!(31i32), REG_BIT__582!(30i32)];
core::ptr::addr_of_mut!(__slate_compound_2)
}).as_ptr() as *const u32).wrapping_offset(1 as isize) } } else { unsafe { *((*{
__slate_compound_3 = [REG_BIT__56!(31i32), REG_BIT__782!(30i32)];
core::ptr::addr_of_mut!(__slate_compound_3)
}).as_ptr() as *const u32).wrapping_offset(0 as isize) } };
let lane_phy_current_status: u32 = if ((owned_lane_mask as i32) as u64) == BIT!(1i32) | BIT!(0i32) { (unsafe { *((*{
__slate_compound_4 = [REG_BIT__855!(29i32), REG_BIT__296!(28i32)];
core::ptr::addr_of_mut!(__slate_compound_4)
}).as_ptr() as *const u32).wrapping_offset(0 as isize) }) | unsafe { *((*{
__slate_compound_5 = [REG_BIT__803!(29i32), REG_BIT__1037!(28i32)];
core::ptr::addr_of_mut!(__slate_compound_5)
}).as_ptr() as *const u32).wrapping_offset(1 as isize) } } else { unsafe { *((*{
__slate_compound_6 = [REG_BIT__489!(29i32), REG_BIT__781!(28i32)];
core::ptr::addr_of_mut!(__slate_compound_6)
}).as_ptr() as *const u32).wrapping_offset(0 as isize) } };
let lane_phy_pulse_status: u32 = if ((owned_lane_mask as i32) as u64) == BIT!(1i32) | BIT!(0i32) { (unsafe { *((*{
__slate_compound_7 = [REG_BIT__1335!(27i32), REG_BIT__697!(26i32)];
core::ptr::addr_of_mut!(__slate_compound_7)
}).as_ptr() as *const u32).wrapping_offset(0 as isize) }) | unsafe { *((*{
__slate_compound_8 = [REG_BIT__1257!(27i32), REG_BIT__84!(26i32)];
core::ptr::addr_of_mut!(__slate_compound_8)
}).as_ptr() as *const u32).wrapping_offset(1 as isize) } } else { unsafe { *((*{
__slate_compound_9 = [REG_BIT__610!(27i32), REG_BIT__534!(26i32)];
core::ptr::addr_of_mut!(__slate_compound_9)
}).as_ptr() as *const u32).wrapping_offset(0 as isize) } };
let wakeref: *mut ref_tracker = intel_lt_phy_transaction_begin(encoder);
// 1. Clear PORT_BUF_CTL2 [PHY Pulse Status].
intel_de_rmw(display, if (DISPLAY_VER!(display) as i32) >= 20 { unsafe { *{
__slate_compound_10 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__153!(if (port as i32) >= 3 { port as i32 } else { 7i32.wrapping_add(port as i32).wrapping_sub(0) }, 3i32, 409604i32, 409860i32, 1503744i32, 1504256i32).wrapping_add(4) as u32 };
core::ptr::addr_of_mut!(__slate_compound_10)
} } } else { unsafe { *{
__slate_compound_11 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__285!(port, 3i32, 409604i32, 409860i32, 1503744i32, 1504256i32).wrapping_add(4) as u32 };
core::ptr::addr_of_mut!(__slate_compound_11)
} } }, lane_phy_pulse_status, lane_phy_pulse_status);
// 2. Set PORT_BUF_CTL2<port> Lane<PHY Lanes Owned> Pipe Reset to 1.
intel_de_rmw(display, if (DISPLAY_VER!(display) as i32) >= 20 { unsafe { *{
__slate_compound_12 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__44!(if (port as i32) >= 3 { port as i32 } else { 7i32.wrapping_add(port as i32).wrapping_sub(0) }, 3i32, 409604i32, 409860i32, 1503744i32, 1504256i32).wrapping_add(4) as u32 };
core::ptr::addr_of_mut!(__slate_compound_12)
} } } else { unsafe { *{
__slate_compound_13 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__122!(port, 3i32, 409604i32, 409860i32, 1503744i32, 1504256i32).wrapping_add(4) as u32 };
core::ptr::addr_of_mut!(__slate_compound_13)
} } }, lane_pipe_reset, lane_pipe_reset);
// 3. Poll for PORT_BUF_CTL2<port> Lane<PHY Lanes Owned> PHY Current Status == 1.
if (unsafe { intel_de_wait_for_set_us(display, if (DISPLAY_VER!(display) as i32) >= 20 { *{
__slate_compound_14 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__49!(if (port as i32) >= 3 { port as i32 } else { 7i32.wrapping_add(port as i32).wrapping_sub(0) }, 3i32, 409604i32, 409860i32, 1503744i32, 1504256i32).wrapping_add(4) as u32 };
core::ptr::addr_of_mut!(__slate_compound_14)
} } else { *{
__slate_compound_15 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__172!(port, 3i32, 409604i32, 409860i32, 1503744i32, 1504256i32).wrapping_add(4) as u32 };
core::ptr::addr_of_mut!(__slate_compound_15)
} }, lane_phy_current_status, 10u32) }) != 0 {
while false {
}
unsafe { _dev_warn(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] PHY %c failed to reset lane\n".as_ptr() as *const u8, phy_name!(phy)) };
}
// 4. Clear for PHY pulse status on owned PHY lanes.
intel_de_rmw(display, if (DISPLAY_VER!(display) as i32) >= 20 { unsafe { *{
__slate_compound_17 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__239!(if (port as i32) >= 3 { port as i32 } else { 7i32.wrapping_add(port as i32).wrapping_sub(0) }, 3i32, 409604i32, 409860i32, 1503744i32, 1504256i32).wrapping_add(4) as u32 };
core::ptr::addr_of_mut!(__slate_compound_17)
} } } else { unsafe { *{
__slate_compound_18 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__160!(port, 3i32, 409604i32, 409860i32, 1503744i32, 1504256i32).wrapping_add(4) as u32 };
core::ptr::addr_of_mut!(__slate_compound_18)
} } }, lane_phy_pulse_status, lane_phy_pulse_status);
// 5. Follow the Display Voltage Frequency Switching -
// Sequence Before Frequency Change. We handle this step in bxt_set_cdclk().
//
// 6. Program PORT_CLOCK_CTL[PCLK PLL Request LN0] = 0.
intel_de_rmw(display, if (DISPLAY_VER!(display) as i32) >= 20 { unsafe { *{
__slate_compound_19 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__250!(if (port as i32) >= 3 { port as i32 } else { 7i32.wrapping_add(port as i32).wrapping_sub(0) }, 3i32, 409824i32, 410080i32, 1503840i32, 1504352i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_19)
} } } else { unsafe { *{
__slate_compound_20 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__112!(port, 3i32, 409824i32, 410080i32, 1503840i32, 1504352i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_20)
} } }, XELPDP_LANE_PCLK_PLL_REQUEST__2!(0i32), 0u32);
// 7. Program DDI_CLK_VALFREQ to 0.
intel_de_write(display, unsafe { *{
__slate_compound_21 = __SlateRecord_i915_reg_defs_h_14 { reg: _PORT!(unsafe { (*encoder).port }, 409648i32, 409904i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_21)
} }, 0u32);
// 8. Poll for PORT_CLOCK_CTL[PCLK PLL Ack LN0]= 0.
if (unsafe { intel_de_wait_for_clear_us(display, if (DISPLAY_VER!(display) as i32) >= 20 { *{
__slate_compound_22 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__16!(if (port as i32) >= 3 { port as i32 } else { 7i32.wrapping_add(port as i32).wrapping_sub(0) }, 3i32, 409824i32, 410080i32, 1503840i32, 1504352i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_22)
} } else { *{
__slate_compound_23 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__236!(port, 3i32, 409824i32, 410080i32, 1503840i32, 1504352i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_23)
} }, XELPDP_LANE_PCLK_PLL_ACK__5!(0i32), 10u32) }) != 0 {
while false {
}
unsafe { _dev_warn(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] PHY %c PLL MacCLK ack deassertion timeout\n".as_ptr() as *const u8, phy_name!(phy)) };
}
// 9. Follow the Display Voltage Frequency Switching -
// Sequence After Frequency Change. We handle this step in bxt_set_cdclk().
//
// 10. Program PORT_CLOCK_CTL register to disable and gate clocks.
intel_de_rmw(display, if (DISPLAY_VER!(display) as i32) >= 20 { unsafe { *{
__slate_compound_25 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__194!(if (port as i32) >= 3 { port as i32 } else { 7i32.wrapping_add(port as i32).wrapping_sub(0) }, 3i32, 409824i32, 410080i32, 1503840i32, 1504352i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_25)
} } } else { unsafe { *{
__slate_compound_26 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__113!(port, 3i32, 409824i32, 410080i32, 1503840i32, 1504352i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_26)
} } }, XELPDP_DDI_CLOCK_SELECT_MASK!(display) | REG_BIT__540!(10i32), 0u32);
// 11. Program PORT_BUF_CTL5[MacCLK Reset_0] = 1 to assert MacCLK reset.
intel_de_rmw(display, unsafe { *{
__slate_compound_27 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__196!(if (port as i32) >= 3 { port as i32 } else { 7i32.wrapping_add(port as i32).wrapping_sub(0) }, 3i32, 409604i32, 409860i32, 1503744i32, 1504256i32).wrapping_add(52) as u32 };
core::ptr::addr_of_mut!(__slate_compound_27)
} }, REG_BIT__882!(11i32), REG_BIT__1421!(11i32));
intel_lt_phy_transaction_end(encoder, wakeref);
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.intel_lt_phy.intel_lt_phy_set_signal_levels")]
extern "C" fn intel_lt_phy_set_signal_levels(encoder: *mut intel_encoder, crtc_state: *const intel_crtc_state) {
let display: *mut intel_display = to_intel_display!(encoder);
let mut owned_lane_mask: u8 = 0;
let mut n_entries: i32 = 0;
let mut ln: i32 = 0;
let dig_port: *mut intel_digital_port = enc_to_dig_port(encoder);
if unsafe { intel_tc_port_in_tbt_alt_mode(dig_port) } {
return;
}
owned_lane_mask = intel_lt_phy_get_owned_lane_mask(encoder);
let wakeref: *mut ref_tracker = intel_lt_phy_transaction_begin(encoder);
let trans: *const intel_ddi_buf_trans = unsafe { intel_ddi_buf_trans_get(encoder, crtc_state, core::ptr::addr_of_mut!(n_entries)) };
let mut __v1: i32;
let __ret_warn_on: i32 = trans.is_null() as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
2i32 | BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: [drm] %s".as_ptr() as *const u8, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON_ONCE(!trans)".as_ptr() as *mut u8) };
let mut __v2: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v2 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v3: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_intel_lt_phy_set_signal_levels_4, sym __slate_asm_string_intel_lt_phy_set_signal_levels_5, const 2137 as i32, const 2323 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v3 = bug;
unsafe { __v2.unwrap()(__v3, dev_driver_string(__drm_to_dev((*display).drm as *const drm_device) as *const device), dev_name(__drm_to_dev((*display).drm as *const drm_device) as *const device), c"drm_WARN_ON_ONCE(!trans)".as_ptr() as *mut u8) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v1 = __ret_warn_on;
if __v1 != 0 {
intel_lt_phy_transaction_end(encoder, wakeref);
return;
}
while ln < ((unsafe { (*crtc_state).lane_count }) as i32) {
let level: i32 = unsafe { intel_ddi_level(encoder, crtc_state, ln) };
let lane: i32 = ln / 2;
let tx: i32 = ln % 2;
let lane_mask: u8 = (if lane == 0 { BIT!(0i32) } else { BIT!(1i32) }) as u8;
if (lane_mask as i32) & (owned_lane_mask as i32) != 0 {
intel_lt_phy_rmw(encoder, lane_mask, LT_PHY_TXY_CTL8!(tx) as u16, ((REG_GENMASK8__25!(7i32, 4i32) as i32) | (REG_BIT8__6!(3i32) as i32)) as u8, ((LT_PHY_TX_SWING_LEVEL!(unsafe { (*unsafe { (*trans).entries }.wrapping_offset(level as isize)).lt.txswing_level }) as i32) | (LT_PHY_TX_SWING!(unsafe { (*unsafe { (*trans).entries }.wrapping_offset(level as isize)).lt.txswing }) as i32)) as u8, true);
intel_lt_phy_rmw(encoder, lane_mask, LT_PHY_TXY_CTL2!(tx) as u16, REG_GENMASK8__24!(5i32, 0i32), LT_PHY_TX_CURSOR!(unsafe { (*unsafe { (*trans).entries }.wrapping_offset(level as isize)).lt.pre_cursor }), true);
intel_lt_phy_rmw(encoder, lane_mask, LT_PHY_TXY_CTL3!(tx) as u16, REG_GENMASK8__15!(5i32, 0i32), LT_PHY_TX_CURSOR__2!(unsafe { (*unsafe { (*trans).entries }.wrapping_offset(level as isize)).lt.main_cursor }), true);
intel_lt_phy_rmw(encoder, lane_mask, LT_PHY_TXY_CTL4!(tx) as u16, REG_GENMASK8__11!(5i32, 0i32), LT_PHY_TX_CURSOR__3!(unsafe { (*unsafe { (*trans).entries }.wrapping_offset(level as isize)).lt.post_cursor }), true);
}
ln = ln.wrapping_add(1);
}
intel_lt_phy_transaction_end(encoder, wakeref);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_732: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[unsafe(no_mangle)]
extern "C" fn intel_lt_phy_dump_hw_state(p: *mut drm_printer, hw_state: *const intel_lt_phy_pll_state) {
let mut i: i32 = 0;
let mut j: i32 = 0;
unsafe { drm_printf(p, c"lt_phy_pll_hw_state: lane count: %d, ssc enabled: %s, tbt mode: %s\n".as_ptr() as *const u8, (*hw_state).lane_count, str_yes_no((*hw_state).ssc_enabled), str_yes_no((*hw_state).tbt_mode)) };
while i < 3 {
unsafe { drm_printf(p, c"config[%d] = 0x%.4x,\n".as_ptr() as *const u8, i, *((*hw_state).config.as_ptr() as *const u8).wrapping_offset(i as isize) as i32) };
i = i.wrapping_add(1);
}
i = 0;
while i <= 12 {
j = 3;
while j >= 0 {
unsafe { drm_printf(p, c"vdr_data[%d][%d] = 0x%.4x,\n".as_ptr() as *const u8, i, j, *((*((*hw_state).data.as_ptr() as *const [u8; 4]).wrapping_offset(i as isize)).as_ptr() as *const u8).wrapping_offset(j as isize) as i32) };
j = j.wrapping_sub(1);
}
i = i.wrapping_add(1);
}
}

#[unsafe(no_mangle)]
extern "C" fn intel_lt_phy_pll_compare_hw_state(a: *const intel_lt_phy_pll_state, b: *const intel_lt_phy_pll_state) -> bool {
if (unsafe { (*a).tbt_mode }) || unsafe { (*b).tbt_mode } {
return true;
}
// With LT PHY values other than VDR0_CONFIG and VDR2_CONFIG are
// unreliable. They cannot always be read back since internally
// after power gating values are not restored back to the
// shadow VDR registers. Thus we do not compare the whole state
// just the two VDR registers.
if ((unsafe { *((*a).config.as_ptr() as *const u8).wrapping_offset(0 as isize) }) as i32) == ((unsafe { *((*b).config.as_ptr() as *const u8).wrapping_offset(0 as isize) }) as i32) && ((unsafe { *((*a).config.as_ptr() as *const u8).wrapping_offset(2 as isize) }) as i32) == ((unsafe { *((*b).config.as_ptr() as *const u8).wrapping_offset(2 as isize) }) as i32) {
return true;
}
false
}

fn intel_lt_phy_pll_is_enabled(encoder: *mut intel_encoder) -> bool {
let mut __slate_compound_1: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_2: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display!(encoder);
intel_de_read(display, if (DISPLAY_VER!(display) as i32) >= 20 { unsafe { *{
__slate_compound_1 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__33!(if ((*encoder).port as i32) >= 3 { (*encoder).port as i32 } else { 7i32.wrapping_add((*encoder).port as i32).wrapping_sub(0) }, 3i32, 409824i32, 410080i32, 1503840i32, 1504352i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_1)
} } } else { unsafe { *{
__slate_compound_2 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__268!(unsafe { (*encoder).port }, 3i32, 409824i32, 410080i32, 1503840i32, 1504352i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_2)
} } }) & XELPDP_LANE_PCLK_PLL_ACK!(0i32) != 0
}

fn readout_ssc_state(encoder: *mut intel_encoder) -> bool {
let mut __slate_compound_1: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let mut __slate_compound_2: __SlateRecord_i915_reg_defs_h_14 = unsafe { core::mem::zeroed() };
let display: *mut intel_display = to_intel_display!(encoder);
intel_de_read(display, if (DISPLAY_VER!(display) as i32) >= 20 { unsafe { *{
__slate_compound_1 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__98!(if ((*encoder).port as i32) >= 3 { (*encoder).port as i32 } else { 7i32.wrapping_add((*encoder).port as i32).wrapping_sub(0) }, 3i32, 409824i32, 410080i32, 1503840i32, 1504352i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_1)
} } } else { unsafe { *{
__slate_compound_2 = __SlateRecord_i915_reg_defs_h_14 { reg: _PICK_EVEN_2RANGES__330!(unsafe { (*encoder).port }, 3i32, 409824i32, 410080i32, 1503840i32, 1504352i32) as u32 };
core::ptr::addr_of_mut!(__slate_compound_2)
} } }) & REG_BIT__712!(1i32) != 0
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.intel_lt_phy.intel_lt_phy_tbt_pll_readout_hw_state")]
extern "C" fn intel_lt_phy_tbt_pll_readout_hw_state(display: *mut intel_display, pll: *mut intel_dpll, hw_state: *mut intel_dpll_hw_state) -> bool {
unsafe { memset(hw_state as *mut (), 0i32, core::mem::size_of::<intel_dpll_hw_state>() as u64) };
unsafe {
(*hw_state).__slate_anon_0.ltpll.tbt_mode = true;
}
true
}

#[unsafe(no_mangle)]
extern "C" fn intel_lt_phy_pll_readout_hw_state(encoder: *mut intel_encoder, pll_state: *mut intel_lt_phy_pll_state) -> bool {
let mut owned_lane_mask: u8 = 0;
let mut lane: u8 = 0;
let mut i: i32 = 0;
let mut j: i32 = 0;
let mut k: i32 = 0;
if !intel_lt_phy_pll_is_enabled(encoder) {
return false;
}
unsafe {
(*pll_state).tbt_mode = intel_tc_port_in_tbt_alt_mode(enc_to_dig_port(encoder));
}
if unsafe { (*pll_state).tbt_mode } {
return false;
}
owned_lane_mask = intel_lt_phy_get_owned_lane_mask(encoder);
let __v0: u64 = ((owned_lane_mask as i32) as u64) & BIT!(0i32);
lane = (if __v0 != 0 { __v0 } else { BIT!(1i32) }) as u8;
let wakeref: *mut ref_tracker = intel_lt_phy_transaction_begin(encoder);
unsafe {
(*pll_state).ssc_enabled = readout_ssc_state(encoder);
}
unsafe {
(*pll_state).lane_count = intel_readout_lane_count(encoder, BIT!(0i32) as i32, BIT!(1i32) as i32);
}
unsafe {
*((*pll_state).config.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) = intel_lt_phy_read(encoder, lane, 3074u16);
}
unsafe {
*((*pll_state).config.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) = intel_lt_phy_read(encoder, BIT!(0i32) as u8, 3075u16);
}
unsafe {
*((*pll_state).config.as_mut_ptr() as *mut u8).wrapping_offset(2 as isize) = intel_lt_phy_read(encoder, lane, 3267u16);
}
while i <= 12 {
j = 3;
k = 0;
while j >= 0 {
unsafe {
*((*((*pll_state).data.as_mut_ptr() as *mut [u8; 4]).wrapping_offset(i as isize)).as_mut_ptr() as *mut u8).wrapping_offset(k as isize) = intel_lt_phy_read(encoder, BIT!(0i32) as u8, LT_PHY_VDR_X_DATAY!(i, j) as u16);
}
j = j.wrapping_sub(1);
k = k.wrapping_add(1);
}
i = i.wrapping_add(1);
}
intel_lt_phy_transaction_end(encoder, wakeref);
true
}

#[unsafe(no_mangle)]
extern "C" fn intel_xe3plpd_pll_enable(encoder: *mut intel_encoder, pll: *mut intel_dpll, dpll_hw_state: *const intel_dpll_hw_state) {
intel_lt_phy_pll_enable(encoder, pll, dpll_hw_state);
}

#[unsafe(no_mangle)]
extern "C" fn intel_xe3plpd_pll_disable(encoder: *mut intel_encoder) {
let dig_port: *mut intel_digital_port = enc_to_dig_port(encoder);
if unsafe { intel_tc_port_in_tbt_alt_mode(dig_port) } {
unsafe { intel_mtl_tbt_pll_disable_clock(encoder) };
} else {
intel_lt_phy_pll_disable(encoder);
}
}

fn intel_lt_phy_pll_verify_clock(display: *mut intel_display, precomputed_clock: i32, pll_state_name: *const u8, pll_state: *const intel_lt_phy_pll_state, is_precomputed_state: bool) {
let mut p: drm_printer = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(p) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 56usize);
}
let clock: i32 = intel_lt_phy_calc_port_clock(display, pll_state);
if unsafe { intel_dpll_clock_matches(clock, precomputed_clock) } {
return;
}
while false {
}
unsafe { _dev_warn(__drm_to_dev((*display).drm as *const drm_device) as *const device, c"[drm] PLL state %s (%s): clock difference too high: computed %d, pre-computed %d\n".as_ptr() as *const u8, pll_state_name, if is_precomputed_state { c"precomputed".as_ptr() as *mut u8 } else { c"computed".as_ptr() as *mut u8 }, clock, precomputed_clock) };
if !drm_debug_enabled!(2i32) {
return;
}
p = drm_dbg_printer(unsafe { (*display).drm }, 2u32, core::ptr::null::<u8>());
unsafe { drm_printf(core::ptr::addr_of_mut!(p), c"PLL state %s (%s):\n".as_ptr() as *const u8, pll_state_name, if is_precomputed_state { c"precomputed".as_ptr() as *mut u8 } else { c"computed".as_ptr() as *mut u8 }) };
intel_lt_phy_dump_hw_state(core::ptr::addr_of_mut!(p), pll_state);
}

fn intel_lt_phy_pll_verify_params(display: *mut intel_display, pll_params: *const intel_lt_phy_pll_params) {
let mut pll_state: intel_lt_phy_pll_state = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(pll_state) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 88usize);
}
intel_lt_phy_pll_verify_clock(display, unsafe { (*pll_params).clock_rate }, unsafe { (*pll_params).name }, unsafe { (*pll_params).state }, true);
if !unsafe { (*pll_params).is_hdmi } {
return;
}
if intel_lt_phy_calculate_hdmi_state(core::ptr::addr_of_mut!(pll_state), (unsafe { (*pll_params).clock_rate }) as u32) != 0 {
return;
}
intel_lt_phy_pll_verify_clock(display, unsafe { (*pll_params).clock_rate }, unsafe { (*pll_params).name }, core::ptr::addr_of_mut!(pll_state) as *const intel_lt_phy_pll_state, false);
}

fn intel_lt_phy_pll_verify_tables(display: *mut intel_display, tables: *const intel_lt_phy_pll_params) {
let mut i: i32 = 0;
while !unsafe { (*tables.wrapping_offset(i as isize)).name }.is_null() {
intel_lt_phy_pll_verify_params(display, tables.wrapping_offset(i as isize));
i = i.wrapping_add(1);
}
}

#[unsafe(no_mangle)]
extern "C" fn intel_lt_phy_verify_plls(display: *mut intel_display) {
intel_lt_phy_pll_verify_tables(display, unsafe { core::ptr::addr_of!(xe3plpd_lt_dp_tables.0) as *const intel_lt_phy_pll_params });
intel_lt_phy_pll_verify_tables(display, unsafe { core::ptr::addr_of!(xe3plpd_lt_edp_tables.0) as *const intel_lt_phy_pll_params });
intel_lt_phy_pll_verify_tables(display, unsafe { core::ptr::addr_of!(xe3plpd_lt_hdmi_tables.0) as *const intel_lt_phy_pll_params });
}

static __slate_asm_string_assert_dc_off_3: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_assert_dc_off_4: [u8; 73] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_lt_phy.c\0";

static __slate_asm_string_intel_lt_phy_calc_hdmi_port_clock_3: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_intel_lt_phy_calc_hdmi_port_clock_4: [u8; 73] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_lt_phy.c\0";

static __slate_asm_string_intel_lt_phy_calc_port_clock_17: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_intel_lt_phy_calc_port_clock_18: [u8; 73] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_lt_phy.c\0";

static __slate_asm_string_intel_lt_phy_enable_disable_tx_4: [u8; 26] = *b"Missing case (%s == %ld)\n\0";

static __slate_asm_string_intel_lt_phy_enable_disable_tx_5: [u8; 73] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_lt_phy.c\0";

static __slate_asm_string_intel_lt_phy_get_dp_clock_3: [u8; 26] = *b"Missing case (%s == %ld)\n\0";

static __slate_asm_string_intel_lt_phy_get_dp_clock_4: [u8; 73] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_lt_phy.c\0";

static __slate_asm_string_intel_lt_phy_pll_calc_state_4: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_intel_lt_phy_pll_calc_state_5: [u8; 73] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_lt_phy.c\0";

static __slate_asm_string_intel_lt_phy_pll_tables_get_3: [u8; 26] = *b"Missing case (%s == %ld)\n\0";

static __slate_asm_string_intel_lt_phy_pll_tables_get_4: [u8; 73] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_lt_phy.c\0";

static __slate_asm_string_intel_lt_phy_set_signal_levels_4: [u8; 16] = *b"%s %s: [drm] %s\0";

static __slate_asm_string_intel_lt_phy_set_signal_levels_5: [u8; 73] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/i915/display/intel_lt_phy.c\0";

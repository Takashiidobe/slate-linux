
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
//! Copyright © 2021 Intel Corporation
//! small_const_nbits(n) is true precisely when it is known at compile-time
//! that BITMAP_SIZE(n) is 1, i.e. 1 <= n <= BITS_PER_LONG. This allows
//! various bit/bitmap APIs to provide a fast inline implementation. Bitmaps
//! of size 0 are very rare, and a compile-time-known-size 0 is most likely
//! a sign of error. They will be handled correctly by the bit/bitmap APIs,
//! but using the out-of-line functions, so that the inline implementations
//! can unconditionally dereference the pointer(s).
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
//! kcsan_check_read - check regular read access for races
//!
//! @ptr: address of access
//! @size: size of access
//! Yes, this permits 64-bit accesses on 32-bit architectures. These will
//! actually be atomic in some cases (namely Armv7 + LPAE), but for others we
//! rely on the access being split into 2x32-bit accesses for a 32-bit quantity
//! (e.g. a virtual address) and a strong prevailing wind.
//! inhibit tail-call optimization
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
//! Internal kernel timer frequency
//! Bitfields in the atomic value:
//!
//! When NR_CPUS < 16K
//!  0- 7: locked byte
//!  8-15: pending byte
//! 16-17: tail index
//! 18-31: tail cpu (+1)
//!
//! When NR_CPUS >= 16K
//!  0- 7: locked byte
//!     8: pending
//!  9-10: tail index
//! 11-31: tail cpu (+1)
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! issue num suppressed message on exit
//! strlen("ratelimit") + 1
//! Check at compile time that something is of a particular type.
//! Always evaluates to 1 so you may use it easily in comparisons.
//! Operation not permitted
//! Value too large for defined data type
//! Kernel pointers have redundant information, so we can use a
//! scheme where we can return either an error code or a normal
//! pointer with the same return value.
//!
//! This should be a per-architecture thing, to allow different
//! error and pointer decisions.
//! This counts to 15. Any more, it will return 16th argument.
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
//! We use the MSB for PREEMPT_NEED_RESCHED mostly because it is available.
//! Macros to retrieve the current execution context:
//!
//! in_nmi()		- We're in NMI context
//! in_hardirq()		- We're in hard IRQ context
//! in_serving_softirq()	- We're in softirq context
//! in_task()		- We're in task context
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
//! 4th level page in 5-level paging case
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
//! instrument_get_user() - add instrumentation to get_user()-like macros
//! @to: destination variable, may not be address-taken
//!
//! get_user() and friends are fragile, so it may depend on the implementation
//! whether the instrumentation happens before or after the data is copied from
//! the userspace.
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
//! restart by calling sys_restart_syscall
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
//! cpumask_bits - get the bits in a cpumask
//! @maskp: the struct cpumask *
//!
//! You should only assume nr_cpu_ids bits of this mask are valid.  This is
//! a macro so it's const-correct.
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
//! for_each_cpu - iterate over every cpu in a mask
//! @cpu: the (optionally unsigned) integer iterator
//! @mask: the cpumask pointer
//!
//! After the loop, cpu is >= nr_cpu_ids.
//! Wrap the arch provided IRQ routines to provide appropriate checks.
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
//! 32-bit address space on 64 bits
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
//! SCHED_ISO: reserved but not implemented yet
//! These constants are defined in the MSan LLVM instrumentation pass.
//! ignore signal
//! bit-flags
//! Convenience macros for the sake of set_current_state:
//! Dependency order vs. p above.
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
//! Subtract two ktime_t variables. rem = lhs -rhs:
//! Values for SRCU Tree srcu_data ->srcu_reader_flavor, but also used by rcutorture.
//! Suits me
//! percpu_counter batch for local add or sub
//! Opaque type representing current VMA (vm_area_struct) flag state. Must be
//! accessed via vma_flags_xxx() helper functions.
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
//! ***DANGEROUS*** don't add guard page
//! Desired design of maximum size and alignment (see RFC2553)
//! We encode errnos in the xas->xa_node.  If an error has happened, we need to
//! drop the lock to fix it, and once we've done so the xa_state is invalid.
//! The length of a UUID string ("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
//! not including trailing NUL.
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
//! current_cred - Access the current task's subjective credentials
//!
//! Access the subjective credentials of the current task.  RCU-safe,
//! since nobody else can modify it.
//! group exit in progress
//! sb->s_flags.  Note that these mirror the equivalent MS_* flags where
//! represented in both.
//! return the first ptr or next element in an RCU protected list
//! 192 bytes
//! Keep unconverted code working
//! used to be O_SYNC, see below
//! Valid hint values for F_{GET,SET}_RW_HINT. 0 is "not set", or can be
//! used to clear any hints previously set.
//! Gives us 8 prio classes with 13-bits of data for each class
//! vfs{g,u}ids are created from k{g,u}ids.
//! We don't allow them to be created from regular {u,g}id.
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
//! The IDR API does not expose the tagging functionality of the radix tree
//! to users.  Use tag 0 to track whether a node has free space below it.
//! number of env pointers
//! Generic info of form tag = "info"
//! module_init() - driver initialization entry point
//! @x: function to be run at kernel boot time or module insertion
//!
//! module_init() will either be called during do_initcalls() (if
//! builtin) or at module insertion time (if a module).  There can only
//! be one per module.
//! GPU_BUDDY_RANGE_ALLOCATION - Allocate within a specific address range
//!
//! When set, allocation is restricted to the range [start, end) specified
//! in gpu_buddy_alloc_blocks(). Without this flag, start/end are ignored
//! and allocation can use any free space.
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
// Format: <modname>__<counter>_<line>_<fn>
// Format: __<prefix>__<iid><id>
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
// pr_info - Print an info-level message
// @fmt: format string
// @...: arguments for the format string
//
// This macro expands to a printk with KERN_INFO loglevel. It uses pr_fmt() to
// generate the format string.
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
// No such file or directory
// No such device or address
// Argument list too long
// Try again
// Out of memory
// Bad address
// Device or resource busy
// Invalid argument
// Text file busy
// No space left on device
// Operation not supported on transport endpoint
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
// __verify_pcpu_ptr() verifies @ptr is a percpu pointer without evaluating
// @ptr and is invoked once before a percpu area is accessed by all
// accessors and operations.  This is performed in the generic part of
// percpu and arch overrides don't need to worry about it; however, if an
// arch wants to implement an arch-specific percpu accessor or operation,
// it may use __verify_pcpu_ptr() to verify the parameters.
//
// + 0 is required in order to convert the pointer type from a
// potential array type to a pointer to a single item of the array.
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
// entries per page directory level
// We borrow bit 3 to remember PG_anon_exclusive.
// Extracts the PFN from a (pte|pmd|pud|pgd)val_t of a 4KB page
// Extracts the flags from a (pte|pmd|pud|pgd)val_t
// This includes the protection key value.
// No need to mask any bits for !PAE
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
// Operation is not supported
// min - return minimum of two values of the same or compatible types
// @x: first value
// @y: second value
// max - return maximum of two values of the same or compatible types
// @x: first value
// @y: second value
// Use these carefully: no type checking, and uses the arguments
// multiple times. Use for obvious constants only.
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
// No static inline type checking - see Subtlety (1) above.
// to_cpumask - convert a NR_CPUS bitmap to a struct cpumask *
// @bitmap: the bitmap
//
// There are a few places where cpumask_var_t isn't appropriate and
// static cpumasks must be used (eg. very early boot), yet we don't
// expose the definition of 'struct cpumask'.
//
// This does the conversion, and can be used as a constant initializer.
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
// re-arm the timer
// For trivial one-depth nesting of a lock-class, the following
// global define can be used. (Subsystems with multiple levels
// of nesting should define their own lock-nesting subclasses.)
// Writer mask
// Remapping rwlock architecture specific functions to the corresponding
// queued rwlock functions.
// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
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
// disable sas during sighandling
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
// 256 bytes including ->parent
// 256 bytes
// 240 bytes
// Add a ktime_t variable and a scalar nanosecond value.
// res = kt + nsval:
// Subtract a scalar nanosecod from a ktime_t variable
// res = kt - nsval:
// Map the ktime_t to timespec conversion to ns_to_timespec function
// srcu_read_lock().
// srcu_read_lock_nmisafe().
// 					0x4		// SRCU-lite is no longer with us.
// srcu_read_lock_fast(), also NMI-safe.
// srcu_read_lock_fast_updown().
// Don't call further
// Opaque type representing current mm_struct flag state. Must be accessed via
// mm_flags_xxx() helper functions.
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
// reset direct map and flush TLB on unmap, can't be freed in atomic context
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
// Mount read-only
// Update the on-disk [acm]times lazily
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
// Allow a vfs{g,u}id to be used as a k{g,u}id where we want to compare
// whether the mapped value is identical to value of a k{g,u}id.
// per-IO, return -EAGAIN if operation would block
// per-IO O_APPEND
// per-IO negation of O_APPEND
// Atomic Write
// buffered IO that drops the cache after reading or writing data
// mask of flags supported by the kernel
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
// non-RWF related bits - start at 16
// Get bit address from inode->i_state to use with wait_var_event()
// infrastructre.
// Multigrain timestamps
//
// Conditionally use fine-grained ctime and mtime timestamps when there
// are users actively observing them via getattr. The primary use-case
// for this is NFS clients that use the ctime to distinguish between
// different states of the file, and that are often fooled by multiple
// operations that occur in the same coarse-grained timer tick.
// File system supports uncached read/write buffered IO
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
// Set the IDR flag and the IDR_FREE tag
// buffer for the variables
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
// What your module does.
// Only touch one cacheline for common rbtree-for-core-layout case.
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
macro_rules! BITS_PER_TYPE {
($type:ty) => {
(core::mem::size_of::<$type>() as u64).wrapping_mul(8i32 as $type)
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

macro_rules! GENMASK_INPUT_CHECK__10 {
() => {
core::mem::size_of::<__SlateRecord_buddy_c_41>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__11 {
() => {
core::mem::size_of::<__SlateRecord_buddy_c_193>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__14 {
() => {
core::mem::size_of::<__SlateRecord_gpu_buddy_h_255>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__17 {
() => {
core::mem::size_of::<__SlateRecord_buddy_c_182>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__20 {
() => {
core::mem::size_of::<__SlateRecord_buddy_c_210>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__21 {
() => {
core::mem::size_of::<__SlateRecord_gpu_buddy_h_261>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__26 {
() => {
core::mem::size_of::<__SlateRecord_gpu_buddy_h_249>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__29 {
() => {
core::mem::size_of::<__SlateRecord_buddy_c_222>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__31 {
() => {
core::mem::size_of::<__SlateRecord_gpu_buddy_h_243>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__35 {
() => {
core::mem::size_of::<__SlateRecord_buddy_c_187>() as i32
};
}

macro_rules! GENMASK_INPUT_CHECK__39 {
() => {
core::mem::size_of::<__SlateRecord_buddy_c_94>() as i32
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
macro_rules! GENMASK_TYPE__9 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__20!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
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
macro_rules! GENMASK_TYPE__13 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__31!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
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
macro_rules! GENMASK_TYPE__14 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__17!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
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
macro_rules! GENMASK_TYPE__19 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__10!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
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
macro_rules! GENMASK_TYPE__22 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__21!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
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
macro_rules! GENMASK_TYPE__25 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__11!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
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
macro_rules! GENMASK_TYPE__29 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__39!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
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
macro_rules! GENMASK_TYPE__30 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__14!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
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
macro_rules! GENMASK_TYPE__31 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__35!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
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
macro_rules! GENMASK_TYPE__37 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__29!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
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
macro_rules! GENMASK_TYPE__40 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__26!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
};
}

macro_rules! GENMASK_ULL__2 {
($h:expr, $l:expr) => {
GENMASK_TYPE__40!(u64, $h, $l)
};
}

macro_rules! GENMASK_ULL__4 {
($h:expr, $l:expr) => {
GENMASK_TYPE__31!(u64, $h, $l)
};
}

macro_rules! GENMASK_ULL__6 {
($h:expr, $l:expr) => {
GENMASK_TYPE__30!(u64, $h, $l)
};
}

macro_rules! GENMASK_ULL__8 {
($h:expr, $l:expr) => {
GENMASK_TYPE__19!(u64, $h, $l)
};
}

macro_rules! GENMASK_ULL__10 {
($h:expr, $l:expr) => {
GENMASK_TYPE__9!(u64, $h, $l)
};
}

macro_rules! GENMASK_ULL__12 {
($h:expr, $l:expr) => {
GENMASK_TYPE__14!(u64, $h, $l)
};
}

macro_rules! GENMASK_ULL__13 {
($h:expr, $l:expr) => {
GENMASK_TYPE__37!(u64, $h, $l)
};
}

macro_rules! GENMASK_ULL__15 {
($h:expr, $l:expr) => {
GENMASK_TYPE__22!(u64, $h, $l)
};
}

macro_rules! GENMASK_ULL__17 {
($h:expr, $l:expr) => {
GENMASK_TYPE__13!(u64, $h, $l)
};
}

macro_rules! GENMASK_ULL__20 {
($h:expr, $l:expr) => {
GENMASK_TYPE__25!(u64, $h, $l)
};
}

macro_rules! GENMASK_ULL__21 {
($h:expr, $l:expr) => {
GENMASK_TYPE__29!(u64, $h, $l)
};
}

macro_rules! IS_ALIGNED {
($x:expr, $a:expr) => {
($x) & (($a) as u64).wrapping_sub(1) == 0
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

macro_rules! PASS_TOKEN_PARAMS {
($_size:expr) => {
$_size
};
}

macro_rules! RB_CLEAR_NODE {
($node:expr) => {
unsafe {
(*$node).__rb_parent_color = ($node) as u64;
}
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
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut *mut rb_node, $val) };
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

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word {
() => {
((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__17 {
() => {
((const { core::mem::size_of::<*mut rb_node>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut rb_node>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut rb_node>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut rb_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

macro_rules! __or {
() => {
____or!()
};
}

macro_rules! __rb_color {
($pc:expr) => {
($pc) & 1
};
}

macro_rules! __rb_is_black {
($pc:expr) => {
__rb_color!($pc)
};
}

macro_rules! __rb_parent {
($pc:expr) => {
(($pc) & 18446744073709551612u64) as *mut rb_node
};
}

/// This looks more complex than it should be. But we need to
/// get the type for the ~ right in round_down (it needs to be
/// as wide as the result!), and we want to evaluate the macro
/// arguments just once each.
macro_rules! __round_mask {
($y:expr) => {
$y.wrapping_sub(1)
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

macro_rules! alloc_tag_restore {
() => {
while false {
}
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

macro_rules! cond_resched__7 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/drivers/gpu/buddy.c".as_ptr() as *const u8, 696i32, 0u32);
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
macro_rules! container_of__13 {
($ptr:expr) => {
{
let mut __v: *mut gpu_buddy_block;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__2!() as isize)) }) as *mut ()) as *mut gpu_buddy_block;
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
macro_rules! container_of__44 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__103!($type) as isize)) }) as *mut ()) as *mut $type;
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
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__2!() as isize)) }) as *mut ()) as *mut $type;
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

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type {
() => {
(1i32.wrapping_neg() as u64) < 1
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

macro_rules! likely {
($x:expr) => {
!(!$x) as i64
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__46 {
($ptr:expr, $type:ty) => {
container_of__17!($ptr, $type)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__8 {
($ptr:expr) => {
container_of__13!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__9 {
($ptr:expr, $type:ty) => {
container_of__44!($ptr, $type)
};
}

/// list_entry_is_head - test if the entry points to the head of the list
/// @pos:	the type * to cursor
/// @head:	the head for your list.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry_is_head__2 {
($pos:expr, $head:expr, $member:ident) => {
list_is_head((unsafe { core::ptr::addr_of_mut!((*$pos).__slate_anon_5.$member) }) as *const list_head, ($head) as *const list_head)
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__29 {
($ptr:expr) => {
list_entry__8!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__40 {
($ptr:expr, $type:ty) => {
list_entry__46!(unsafe { (*$ptr).next }, $type)
};
}

/// list_last_entry - get the last element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_last_entry {
($ptr:expr) => {
list_entry__8!(unsafe { (*$ptr).prev })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__22 {
($pos:expr, $member:ident) => {
list_entry__8!(unsafe { (*$pos).__slate_anon_5.$member.next })
};
}

macro_rules! offsetof__2 {
() => {
40u64
};
}

macro_rules! offsetof__103 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, tmp_link) as u64
};
}

/// range_overflows() - Check if a range is out of bounds
/// @start: Start of the range.
/// @size:  Size of the range.
/// @max:   Exclusive upper boundary.
///
/// A strict check to determine if the range [@start, @start + @size) is
/// invalid with respect to the allowable range [0, @max). Any range
/// starting at or beyond @max is considered an overflow, even if @size is 0.
///
/// Returns: true if the range is out of bounds.
macro_rules! range_overflows {
($start:expr, $size:expr, $max:expr) => {
{
let mut __v: bool;
let mut __v2: u64 = $start;
let mut __v3: u64 = $size;
let mut __v4: u64 = $max;
core::ptr::addr_of_mut!(__v2) == core::ptr::addr_of_mut!(__v3);
core::ptr::addr_of_mut!(__v2) == core::ptr::addr_of_mut!(__v4);
__v = __v2 >= __v4 || __v3 > __v4.wrapping_sub(__v2);
__v
}
};
}

macro_rules! rb_color {
($rb:expr) => {
__rb_color!(unsafe { (*$rb).__rb_parent_color })
};
}

macro_rules! rb_entry {
($ptr:expr, $type:ty) => {
container_of__17!($ptr, $type)
};
}

macro_rules! rb_entry__2 {
($ptr:expr) => {
container_of__13!($ptr)
};
}

macro_rules! rb_entry_safe {
($ptr:expr) => {
{
let mut __v: *mut gpu_buddy_block;
let __v2: *mut rb_node = $ptr;
let __v3: *mut gpu_buddy_block = if !__v2.is_null() {
rb_entry__2!(__v2)
} else {
core::ptr::null_mut::<gpu_buddy_block>()
};
__v = __v3;
__v
}
};
}

macro_rules! rb_is_black {
($rb:expr) => {
__rb_is_black!(unsafe { (*$rb).__rb_parent_color })
};
}

macro_rules! rb_parent {
($r:expr) => {
((unsafe { (*$r).__rb_parent_color }) & 18446744073709551612u64) as *mut rb_node
};
}

/// round_down - round down to next specified power of 2
/// @x: the value to round
/// @y: multiple to round down to (must be a power of 2)
///
/// Rounds @x down to next multiple of @y (which must be a power of 2).
/// To perform arbitrary rounding down, use rounddown() below.
macro_rules! round_down {
($x:expr, $y:expr) => {
($x) & !__round_mask!($y)
};
}

/// round_up - round up to next specified power of 2
/// @x: the value to round
/// @y: multiple to round up to (must be a power of 2)
///
/// Rounds @x up to next multiple of @y (which must be a power of 2).
/// To perform arbitrary rounding up, use roundup() below.
macro_rules! round_up {
($x:expr, $y:expr) => {
($x.wrapping_sub(1) | __round_mask!($y)).wrapping_add(1)
};
}

/// rounddown_pow_of_two - round the given value down to nearest power of two
/// @n: parameter
///
/// round the given value down to the nearest power of two
/// - the result is undefined when n == 0
/// - this can be used to initialise global variables from constant data
macro_rules! rounddown_pow_of_two__2 {
($n:expr) => {
if unsafe { __slate_intrinsic_f217c4720f04f6db($n) } { 1u64 << ilog2!($n) } else { __rounddown_pow_of_two($n) }
};
}

/// roundup_pow_of_two - round the given value up to nearest power of two
/// @n: parameter
///
/// round the given value up to the nearest power of two
/// - the result is undefined when n == 0
/// - this can be used to initialise global variables from constant data
macro_rules! roundup_pow_of_two__2 {
($n:expr) => {
if unsafe { __slate_intrinsic_f217c4720f04f6db($n) } { if ($n) == 1 { 1u64 } else { 1u64 << ilog2!($n.wrapping_sub(1)).wrapping_add(1) } } else { __roundup_pow_of_two($n) }
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

unsafe extern "C" {
    static mut __SCK__cond_resched: static_call_key;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn _printk(arg0: *const u8, ...) -> i32;
    fn __cond_resched() -> i32;
    fn __compiletime_assert_16();
    fn __compiletime_assert_17();
    fn __compiletime_assert_18();
    fn __compiletime_assert_21();
    fn __compiletime_assert_22();
    fn __SCT__cond_resched() -> i32;
    fn rb_prev(arg0: *const rb_node) -> *mut rb_node;
    fn rb_first_postorder(arg0: *const rb_root) -> *mut rb_node;
    fn rb_next_postorder(arg0: *const rb_node) -> *mut rb_node;
    fn __kmem_cache_create_args(arg0: *const u8, arg1: u32, arg2: *mut kmem_cache_args, arg3: u32) -> *mut kmem_cache;
    fn kmem_cache_destroy(arg0: *mut kmem_cache);
    fn kfree(arg0: *const ());
    fn __compiletime_assert_208();
    fn __builtin_unreachable();
    fn kmem_cache_alloc_noprof(arg0: *mut kmem_cache, arg1: u32) -> *mut ();
    fn kmem_cache_free(arg0: *mut kmem_cache, arg1: *mut ());
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __rb_insert_augmented(arg0: *mut rb_node, arg1: *mut rb_root, arg2: Option<unsafe extern "C" fn(*mut rb_node, *mut rb_node)>);
    fn __compiletime_assert_441();
    fn __compiletime_assert_442();
    fn __compiletime_assert_443();
    fn __rb_erase_color(arg0: *mut rb_node, arg1: *mut rb_root, arg2: Option<unsafe extern "C" fn(*mut rb_node, *mut rb_node)>);
    fn __compiletime_assert_453();
    fn __compiletime_assert_454();
    fn __compiletime_assert_455();
    fn __compiletime_assert_458();
    fn __compiletime_assert_477();
    fn __compiletime_assert_482();
    fn __compiletime_assert_485();
    fn __compiletime_assert_488();
    fn __compiletime_assert_490();
    fn __compiletime_assert_496();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_1c5364fcd9da51fe(arg0: u32) -> bool;
    #[link_name = "llvm.is.constant.i64"]
    fn __slate_intrinsic_f217c4720f04f6db(arg0: u64) -> bool;
}

// generic data direction definitions
// The type of an index into the pagecache.
#[repr(C)]
#[derive(Clone, Copy)]
struct list_head {
    next: *mut list_head,
    prev: *mut list_head,
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
union __SlateRecord_static_call_types_h_65 {
    r#type: u64,
    mods: *mut static_call_mod,
    sites: *mut static_call_site,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_mod {
}


#[inline(always)]
fn __must_check_overflow(overflow: bool) -> bool {
unlikely!(overflow) != 0
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

#[inline(always)]
fn ERR_PTR(error: i64) -> *mut () {
error as *mut ()
}

#[inline(always)]
fn IS_ERR(ptr: *const ()) -> bool {
IS_ERR_VALUE!(ptr as u64) != 0
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

#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache {
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
fn __arch_hweight64(mut w: u64) -> u64 {
let mut res: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("# ALT: oldinstr\n771:\n\tcall __sw_hweight64\n772:\n# ALT: padding\n.skip -(((775f-774f)-(772b-771b)) > 0) * ((775f-774f)-(772b-771b)),0x90\n773:\n.pushsection .altinstructions, \"aM\", @progbits, 14\n .long 771b - .\n .long 774f - .\n .4byte ( 4*32+23)\n .byte 773b-771b\n .byte 775f-774f\n.popsection\n.pushsection .altinstr_replacement, \"ax\"\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n# ALT: replacement\n774:\n\tpopcntq %rdi, %rax\n775:\n.popsection\n", lateout("ax") __t0, in("di") w as u64, options(pure, nomem, att_syntax));
res = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
res
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

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_gpu_buddy_h_124 {
    rb: rb_node,
    link: list_head,
}


fn __ffs64(word: u64) -> u32 {
__ffs!(word) as u32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_gpu_buddy_h_243 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_gpu_buddy_h_249 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_gpu_buddy_h_255 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_gpu_buddy_h_261 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_buddy_c_41 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_buddy_c_94 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_buddy_c_182 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_buddy_c_187 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_buddy_c_193 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_buddy_c_210 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_buddy_c_222 {
}


#[repr(transparent)]
struct __SlateConst<T>(T);

unsafe impl<T> Sync for __SlateConst<T> {
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
fn __might_resched(file: *const u8, line: i32, offsets: u32) {
}

// LIST_HEAD - definition of a &struct list_head with initialization values
// @name: name of the list_head
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
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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

fn list_del(entry: *mut list_head) {
__list_del_entry(entry);
unsafe {
(*entry).next = ((256i32 as *const u8).wrapping_offset(_AC__2!() as isize) as *mut ()) as *mut list_head;
}
unsafe {
(*entry).prev = ((290i32 as *const u8).wrapping_offset(_AC__2!() as isize) as *mut ()) as *mut list_head;
}
}

fn list_move(list: *mut list_head, head: *mut list_head) {
__list_del_entry(list);
list_add(list, head);
}

fn list_is_head(list: *const list_head, head: *const list_head) -> i32 {
(list == head) as i32
}

fn list_empty(head: *const list_head) -> i32 {
let mut __v0: *mut list_head;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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

core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_gpu_buddy_init: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_gpu_buddy_fini: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_gpu_buddy_reset_clear: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_gpu_buddy_free_block: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_gpu_buddy_allocated_addr_to_block: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_gpu_buddy_free_list: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_gpu_buddy_block_trim: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_gpu_buddy_alloc_blocks: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {7} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_gpu_buddy_block_print: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {8} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_gpu_buddy_print: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {9} ; .previous\n.section\t\".initcall6.init\", \"a\"\t\t\n__initcall__kmod_buddy__502_1630_gpu_buddy_module_init6:\t\t\t\n.long\t{10} - .\t\n.previous\t\t\t\t\t\n", sym gpu_buddy_init, sym gpu_buddy_fini, sym gpu_buddy_reset_clear, sym gpu_buddy_free_block, sym gpu_buddy_allocated_addr_to_block, sym gpu_buddy_free_list, sym gpu_buddy_block_trim, sym gpu_buddy_alloc_blocks, sym gpu_buddy_block_print, sym gpu_buddy_print, sym gpu_buddy_module_init, options(att_syntax));

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

fn list_splice_tail(list: *mut list_head, head: *mut list_head) {
if list_empty(list as *const list_head) == 0 {
__list_splice(list as *const list_head, unsafe { (*head).prev }, head);
}
}

// list_first_entry_or_null - get the first element from a list
// @ptr:	the list head to take the element from.
// @type:	the type of the struct this is embedded in.
// @member:	the name of the list_head within the struct.
//
// Note that if the list is empty, it returns NULL.
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
#[repr(C)]
#[derive(Clone, Copy)]
struct rb_node {
    __rb_parent_color: u64,
    rb_right: *mut rb_node,
    rb_left: *mut rb_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rb_root {
    rb_node: *mut rb_node,
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
// Number of contexts where an event can trigger:
//      task, softirq, hardirq, nmi.
// I am an IDLE thread
// Allocating memory to free memory. See memalloc_noreclaim_save()
// Kernel thread cloned from userspace thread
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
#[inline(always)]
fn _cond_resched() -> i32 {
let mut __v0: Option<unsafe extern "C" fn() -> i32>;
__v0 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn() -> i32>>(__SCT__cond_resched as *const ()) };
unsafe { __v0.unwrap()() }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__cond_resched_116: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__cond_resched) }) as *mut ();

fn rb_last(root: *const rb_root) -> *mut rb_node {
let mut n: *mut rb_node = unsafe { core::mem::zeroed() };
n = unsafe { (*root).rb_node };
if n.is_null() {
return core::ptr::null_mut::<rb_node>();
}
while !unsafe { (*n).rb_right }.is_null() {
n = unsafe { (*n).rb_right };
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

// rbtree_postorder_for_each_entry_safe - iterate in post-order over rb_root of
// given type allowing the backing memory of @pos to be invalidated
//
// @pos:	the 'type *' to use as a loop cursor.
// @n:		another 'type *' to use as temporary storage
// @root:	'rb_root *' of the rbtree.
// @field:	the name of the rb_node field within 'type'.
//
// rbtree_postorder_for_each_entry_safe() provides a similar guarantee as
// list_for_each_entry_safe() and allows the iteration to continue independent
// of changes to @pos by the body of the loop.
//
// Note, however, that it cannot handle other modifications that re-order the
// rbtree it is iterating over. This includes calling rb_erase() on @pos, as
// rb_erase() may rebalance the tree, causing us to miss some nodes.
// Same as rb_first(), but O(1)
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
struct kmem_cache_args {
    align: u32,
    useroffset: u32,
    usersize: u32,
    freeptr_offset: u32,
    use_freeptr_offset: bool,
    ctor: Option<unsafe extern "C" fn(*mut ())>,
    sheaf_capacity: u32,
}


// Please use this macro to create slab caches. Simply specify the
// name of the structure and maybe some flags that are listed above.
//
// The alignment of the struct determines object alignment. If you
// f.e. add ____cacheline_aligned_in_smp to the struct declaration
// then the objects will be properly aligned in SMP configurations.
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
unsafe { __compiletime_assert_208() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
} else {
loop {
unsafe {
core::arch::asm!("209: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 209b - ., 3; .popsection\n# {0}", const 209 as i32, options(nostack, att_syntax));
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
// kmalloc_objs - Allocate an array of the given type
// @VAR_OR_TYPE: Variable or type to allocate an array of.
// @COUNT: How many elements in the array.
// @...: optional GFP flags for the allocation (GFP_KERNEL when not specified).
//
// Returns: newly allocated pointer to array of @VAR_OR_TYPE on success,
// or NULL on failure.
fn _kmalloc_array_noprof(n: u64, size: u64, flags: u32, token: __SlateRecord_slab_h_532) -> *mut () {
let mut bytes: u64 = 0;
if unlikely!(check_mul_overflow!(n, size, core::ptr::addr_of_mut!(bytes))) != 0 {
return core::ptr::null_mut::<()>();
}
_kmalloc_noprof(bytes, flags, token)
}

// kcalloc - allocate memory for an array. The memory is set to zero.
// @n: number of elements.
// @size: element size.
// @flags: the type of memory to allocate (see kmalloc).
fn kmemleak_update_trace(ptr: *const ()) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rb_augment_callbacks {
    propagate: Option<unsafe extern "C" fn(*mut rb_node, *mut rb_node)>,
    copy: Option<unsafe extern "C" fn(*mut rb_node, *mut rb_node)>,
    rotate: Option<unsafe extern "C" fn(*mut rb_node, *mut rb_node)>,
}


fn rb_insert_augmented(node: *mut rb_node, root: *mut rb_root, augment: *const rb_augment_callbacks) {
unsafe { __rb_insert_augmented(node, root, (*augment).rotate) };
}

fn rb_set_parent(rb: *mut rb_node, p: *mut rb_node) {
unsafe {
(*rb).__rb_parent_color = rb_color!(rb).wrapping_add(p as u64);
}
}

fn rb_set_parent_color(rb: *mut rb_node, p: *mut rb_node, color: i32) {
unsafe {
(*rb).__rb_parent_color = (p as u64).wrapping_add(color as u64);
}
}

fn __rb_change_child(old: *mut rb_node, new: *mut rb_node, parent: *mut rb_node, root: *mut rb_root) {
if !parent.is_null() {
if (unsafe { (*parent).rb_left }) == old {
loop {
loop {
if !(__native_word__17!() || ((const { core::mem::size_of::<*mut rb_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_441() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__6!((*parent).rb_left, new);
if true {
break;
}
}
} else {
loop {
loop {
if !(__native_word__17!() || ((const { core::mem::size_of::<*mut rb_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_442() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__6!((*parent).rb_right, new);
if true {
break;
}
}
}
} else {
loop {
loop {
if !(__native_word__17!() || ((const { core::mem::size_of::<*mut rb_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_443() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__6!((*root).rb_node, new);
if true {
break;
}
}
}
}

#[inline(always)]
fn __rb_erase_augmented(node: *mut rb_node, root: *mut rb_root, augment: *const rb_augment_callbacks) -> *mut rb_node {
let child: *mut rb_node = unsafe { (*node).rb_right };
let mut tmp: *mut rb_node = unsafe { (*node).rb_left };
let mut parent: *mut rb_node = unsafe { core::mem::zeroed() };
let mut rebalance: *mut rb_node = unsafe { core::mem::zeroed() };
let mut pc: u64 = 0;
if tmp.is_null() {
// Case 1: node to erase has no more than 1 child (easy!)
//
// Note that if there is one child it must be red due to 5)
// and node must be black due to 4). We adjust colors locally
// so as to bypass __rb_erase_color() later on.
pc = unsafe { (*node).__rb_parent_color };
parent = __rb_parent!(pc);
__rb_change_child(node, child, parent, root);
if !child.is_null() {
unsafe {
(*child).__rb_parent_color = pc;
}
rebalance = core::ptr::null_mut::<rb_node>();
} else {
rebalance = if __rb_is_black!(pc) != 0 { parent } else { core::ptr::null_mut::<rb_node>() };
}
tmp = parent;
} else {
if child.is_null() {
// Still case 1, but this time the child is node->rb_left
let __v0: u64 = unsafe { (*node).__rb_parent_color };
pc = __v0;
unsafe {
(*tmp).__rb_parent_color = __v0;
}
parent = __rb_parent!(pc);
__rb_change_child(node, tmp, parent, root);
rebalance = core::ptr::null_mut::<rb_node>();
tmp = parent;
} else {
let mut successor: *mut rb_node = child;
let mut child2: *mut rb_node = unsafe { core::mem::zeroed() };
tmp = unsafe { (*child).rb_left };
if tmp.is_null() {
// Case 2: node's successor is its right child
//
//    (n)          (s)
//    / \          / 			 *  (x) (s)  ->  (x) (c)
//        			 *        (c)
parent = successor;
child2 = unsafe { (*successor).rb_right };
unsafe { unsafe { (*augment).copy }.unwrap()(node, successor) };
} else {
// Case 3: node's successor is leftmost under
// node's right child subtree
//
//    (n)          (s)
//    / \          / 			 *  (x) (y)  ->  (x) (y)
//      /            /
//    (p)          (p)
//    /            /
//  (s)          (c)
//    			 *    (c)
loop {
parent = successor;
successor = tmp;
tmp = unsafe { (*tmp).rb_left };
if tmp.is_null() {
break;
}
}
child2 = unsafe { (*successor).rb_right };
loop {
loop {
if !(__native_word__17!() || ((const { core::mem::size_of::<*mut rb_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_453() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__6!((*parent).rb_left, child2);
if true {
break;
}
}
loop {
loop {
if !(__native_word__17!() || ((const { core::mem::size_of::<*mut rb_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_454() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__6!((*successor).rb_right, child);
if true {
break;
}
}
rb_set_parent(child, successor);
unsafe { unsafe { (*augment).copy }.unwrap()(node, successor) };
unsafe { unsafe { (*augment).propagate }.unwrap()(parent, successor) };
}
tmp = unsafe { (*node).rb_left };
loop {
loop {
if !(__native_word__17!() || ((const { core::mem::size_of::<*mut rb_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_455() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__6!((*successor).rb_left, tmp);
if true {
break;
}
}
rb_set_parent(tmp, successor);
pc = unsafe { (*node).__rb_parent_color };
tmp = __rb_parent!(pc);
__rb_change_child(node, successor, tmp, root);
if !child2.is_null() {
rb_set_parent_color(child2, parent, 1i32);
rebalance = core::ptr::null_mut::<rb_node>();
} else {
rebalance = if rb_is_black!(successor) != 0 { parent } else { core::ptr::null_mut::<rb_node>() };
}
unsafe {
(*successor).__rb_parent_color = pc;
}
tmp = successor;
}
}
unsafe { unsafe { (*augment).propagate }.unwrap()(tmp, core::ptr::null_mut::<rb_node>()) };
rebalance
}

#[inline(always)]
fn rb_erase_augmented(node: *mut rb_node, root: *mut rb_root, augment: *const rb_augment_callbacks) {
let rebalance: *mut rb_node = __rb_erase_augmented(node, root, augment);
if !rebalance.is_null() {
unsafe { __rb_erase_color(rebalance, root, (*augment).rotate) };
}
}

// GPU_BUDDY_TOPDOWN_ALLOCATION - Allocate from top of address space
//
// Allocate starting from high addresses and working down. Useful for
// separating different allocation types (e.g., kernel vs userspace)
// to reduce fragmentation.
// GPU_BUDDY_CONTIGUOUS_ALLOCATION - Require physically contiguous blocks
//
// The allocation must be satisfied with a single contiguous block.
// If the requested size cannot be allocated contiguously, the
// allocation fails with -ENOSPC.
// GPU_BUDDY_CLEAR_ALLOCATION - Prefer pre-cleared (zeroed) memory
//
// Attempt to allocate from the clear tree first. If insufficient clear
// memory is available, falls back to dirty memory. Useful when the
// caller needs zeroed memory and wants to avoid GPU clear operations.
// GPU_BUDDY_CLEARED - Mark returned blocks as cleared
//
// Used with gpu_buddy_free_list() to indicate that the memory being
// freed has been cleared (zeroed). The blocks will be placed in the
// clear tree for future GPU_BUDDY_CLEAR_ALLOCATION requests.
// GPU_BUDDY_TRIM_DISABLE - Disable automatic block trimming
//
// By default, if an allocation is smaller than the allocated block,
// excess memory is trimmed and returned to the free pool. This flag
// disables trimming, keeping the full power-of-two block size.
#[repr(C)]
#[derive(Clone, Copy)]
struct gpu_buddy_block {
    header: u64,
    left: *mut gpu_buddy_block,
    right: *mut gpu_buddy_block,
    parent: *mut gpu_buddy_block,
    private: *mut (),
    __slate_anon_5: __SlateRecord_gpu_buddy_h_124,
    tmp_link: list_head,
    subtree_max_alignment: u32,
}


// Order-zero must be at least SZ_4K
#[repr(C)]
#[derive(Clone, Copy)]
struct gpu_buddy {
    free_trees: *mut *mut rb_root,
    roots: *mut *mut gpu_buddy_block,
    free_scoreboard: *mut u64,
    used_scoreboard: *mut u64,
    n_roots: u32,
    max_order: u32,
    chunk_size: u64,
    size: u64,
    avail: u64,
    clear_avail: u64,
}


fn gpu_buddy_driver_lock_held(mm: *mut gpu_buddy) {
}

fn gpu_buddy_block_offset(block: *const gpu_buddy_block) -> u64 {
(unsafe { (*block).header }) & GENMASK_ULL__17!(63i32, 12i32)
}

fn gpu_buddy_block_order(block: *mut gpu_buddy_block) -> u32 {
((unsafe { (*block).header }) & GENMASK_ULL__2!(5i32, 0i32)) as u32
}

fn gpu_buddy_block_is_free(block: *mut gpu_buddy_block) -> bool {
(unsafe { (*block).header }) & GENMASK_ULL__6!(11i32, 10i32) == 2048u64
}

fn gpu_buddy_block_is_clear(block: *mut gpu_buddy_block) -> bool {
(unsafe { (*block).header }) & GENMASK_ULL__15!(9i32, 9i32) != 0
}

fn gpu_buddy_block_size(mm: *mut gpu_buddy, block: *mut gpu_buddy_block) -> u64 {
(unsafe { (*mm).chunk_size }) << gpu_buddy_block_order(block)
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.buddy.gpu_buddy_alloc_blocks")]
extern "C" fn gpu_buddy_alloc_blocks(mm: *mut gpu_buddy, start: u64, end: u64, mut size: u64, mut min_block_size: u64, blocks: *mut list_head, flags: u64) -> i32 {
let mut trim_size: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut u64 = core::ptr::addr_of_mut!(trim_size) as *mut u64;
let mut temp: core::mem::MaybeUninit<list_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut list_head = core::ptr::addr_of_mut!(temp) as *mut list_head;
let mut trim_list: core::mem::MaybeUninit<*mut list_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut list_head = core::ptr::addr_of_mut!(trim_list) as *mut *mut list_head;
// Try allocation through force merge method
let mut __v13: bool = unsafe { core::mem::zeroed() };
let mut fallback_order: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut u32 = core::ptr::addr_of_mut!(fallback_order) as *mut u32;
let mut __UNIQUE_ID_y__495: u32 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__494: u32 = unsafe { core::mem::zeroed() };
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut pages: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut u64 = core::ptr::addr_of_mut!(pages) as *mut u64;
let mut allocated: core::mem::MaybeUninit<list_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut list_head = core::ptr::addr_of_mut!(allocated) as *mut list_head;
let mut order: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut u32 = core::ptr::addr_of_mut!(order) as *mut u32;
let mut min_order: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut u32 = core::ptr::addr_of_mut!(min_order) as *mut u32;
let mut original_min_size: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_25: *mut u64 = core::ptr::addr_of_mut!(original_min_size) as *mut u64;
let mut original_size: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_26: *mut u64 = core::ptr::addr_of_mut!(original_size) as *mut u64;
let mut block: *mut gpu_buddy_block = unsafe { core::mem::zeroed() };
unsafe {
block = core::ptr::null_mut::<gpu_buddy_block>();
let __slate_auto_bytes_1 = __slate_slot_26 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_25 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_23 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
core::ptr::write(__slate_slot_22, list_head { next: core::ptr::addr_of_mut!(*__slate_slot_22), prev: core::ptr::addr_of_mut!(*__slate_slot_22) });
let __slate_auto_bytes_5 = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
gpu_buddy_driver_lock_held(mm);
if size < (*mm).chunk_size {
return 22i32.wrapping_neg();
} else {
if min_block_size < (*mm).chunk_size {
return 22i32.wrapping_neg();
} else {
if !is_power_of_2(min_block_size) {
return 22i32.wrapping_neg();
} else {
if !IS_ALIGNED!(start | end | size, unsafe { (*mm).chunk_size }) {
return 22i32.wrapping_neg();
} else {
if end > (*mm).size {
return 22i32.wrapping_neg();
} else {
if range_overflows!(start, size, unsafe { (*mm).size }) {
return 22i32.wrapping_neg();
} else {
// Actual range allocation
if start.wrapping_add(size) == end {
if !IS_ALIGNED!(start | end, min_block_size) {
return 22i32.wrapping_neg();
} else {
return __gpu_buddy_alloc_range(mm, start, size, core::ptr::null_mut::<u64>(), blocks);
}
} else {
*__slate_slot_26 = size;
*__slate_slot_25 = min_block_size;
// Roundup the size to power of 2
if flags & BIT!(2i32) != 0 {
size = roundup_pow_of_two__2!(size);
min_block_size = size;
// Normalize the requested size to min_block_size for regular allocations.
// Offset-aligned allocations intentionally skip size rounding.
} else {
if !gpu_buddy_can_offset_align(size, min_block_size) {
size = round_up!(size, min_block_size);
}
}
*__slate_slot_21 = size >> ilog2!(unsafe { (*mm).chunk_size });
*__slate_slot_23 = fls(*__slate_slot_21 as u32).wrapping_sub(1) as u32;
*__slate_slot_24 = ilog2!(min_block_size).wrapping_sub(ilog2!(unsafe { (*mm).chunk_size })) as u32;
if *__slate_slot_23 > (*mm).max_order || size > (*mm).size {
if flags & BIT!(2i32) != 0 && flags & BIT!(0i32) == 0 {
return __alloc_contig_try_harder(mm, *__slate_slot_26, *__slate_slot_25, blocks);
} else {
return 22i32.wrapping_neg();
}
} else {
'__join_32: {
'__join_27: {
'__join_5: {
'__loop_36: loop {
__UNIQUE_ID_x__494 = *__slate_slot_23;
__UNIQUE_ID_y__495 = (fls(*__slate_slot_21 as u32) as u32).wrapping_sub(1);
while false {
}
*__slate_slot_23 = if __UNIQUE_ID_x__494 < __UNIQUE_ID_y__495 { __UNIQUE_ID_x__494 } else { __UNIQUE_ID_y__495 };
loop {
if unlikely!(*__slate_slot_23 > (*mm).max_order) != 0 {
break '__join_32;
} else {
if true {
break;
}
}
}
// Regular allocations must not allocate blocks smaller than min_block_size.
// Offset-aligned allocations deliberately bypass this constraint.
loop {
if unlikely!(size >= min_block_size && *__slate_slot_23 < *__slate_slot_24) != 0 {
break '__join_27;
} else {
if true {
break;
}
}
}
'__join_9: {
'__join_18: {
'__join_19: {
loop {
let __slate_auto_bytes = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
block = __gpu_buddy_alloc_blocks(mm, start, end, size, min_block_size, *__slate_slot_23, flags);
if !IS_ERR(block as *const ()) {
break '__join_9;
} else {
if size < min_block_size {
break '__join_19;
} else {
if *__slate_slot_23 == *__slate_slot_24 {
break;
} else {
let __v15: u32 = *__slate_slot_23;
*__slate_slot_23 = __v15.wrapping_sub(1);
if false {
break '__join_9;
}
}
}
}
}
*__slate_slot_16 = *__slate_slot_24;
break '__join_18;
}
*__slate_slot_16 = *__slate_slot_23;
}
if (*mm).clear_avail != 0 {
__v13 = __force_merge(mm, start, end, *__slate_slot_16) == 0;
} else {
__v13 = false;
}
if __v13 {
block = __gpu_buddy_alloc_blocks(mm, start, end, size, min_block_size, *__slate_slot_16, flags);
if !IS_ERR(block as *const ()) {
*__slate_slot_23 = *__slate_slot_16;
} else {
break '__loop_36;
}
} else {
break '__loop_36;
}
}
mark_allocated(mm, block);
let __v12: u64 = gpu_buddy_block_size(mm, block);
let __v10: u64 = (*mm).avail;
(*mm).avail = __v10.wrapping_sub(__v12);
if gpu_buddy_block_is_clear(block) {
let __v8: u64 = gpu_buddy_block_size(mm, block);
let __v6: u64 = (*mm).clear_avail;
(*mm).clear_avail = __v6.wrapping_sub(__v8);
}
kmemleak_update_trace(block as *const ());
list_add_tail(core::ptr::addr_of_mut!((*block).__slate_anon_5.link), core::ptr::addr_of_mut!(*__slate_slot_22));
let __v4: u64 = *__slate_slot_21;
let __v3: u64 = __v4.wrapping_sub(BIT!(*__slate_slot_23));
*__slate_slot_21 = __v3;
if *__slate_slot_21 == 0 {
break '__join_5;
} else {
if false {
break '__join_5;
}
}
}
// Try contiguous block allocation through
// try harder method.
if flags & BIT!(2i32) != 0 && flags & BIT!(0i32) == 0 {
return __alloc_contig_try_harder(mm, *__slate_slot_26, *__slate_slot_25, blocks);
} else {
*__slate_slot_20 = 28i32.wrapping_neg();
gpu_buddy_free_list_internal(mm, core::ptr::addr_of_mut!(*__slate_slot_22));
return *__slate_slot_20;
}
}
// Trim the allocated block to the required size
if flags & BIT!(5i32) == 0 && *__slate_slot_26 != size {
let __slate_auto_bytes_7 = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 8usize);
core::ptr::write(__slate_slot_1, list_head { next: core::ptr::addr_of_mut!(*__slate_slot_1), prev: core::ptr::addr_of_mut!(*__slate_slot_1) });
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_2 = core::ptr::addr_of_mut!(*__slate_slot_22);
*__slate_slot_0 = *__slate_slot_26;
if list_is_singular(core::ptr::addr_of_mut!(*__slate_slot_22) as *const list_head) == 0 {
block = list_last_entry!(core::ptr::addr_of_mut!(*__slate_slot_22));
list_move(core::ptr::addr_of_mut!((*block).__slate_anon_5.link), core::ptr::addr_of_mut!(*__slate_slot_1));
*__slate_slot_2 = core::ptr::addr_of_mut!(*__slate_slot_1);
*__slate_slot_0 = gpu_buddy_block_size(mm, block).wrapping_sub(size.wrapping_sub(*__slate_slot_26));
}
gpu_buddy_block_trim(mm, core::ptr::null_mut::<u64>(), *__slate_slot_0, *__slate_slot_2);
if list_empty(core::ptr::addr_of_mut!(*__slate_slot_1) as *const list_head) == 0 {
list_splice_tail(*__slate_slot_2, core::ptr::addr_of_mut!(*__slate_slot_22));
}
}
list_splice_tail(core::ptr::addr_of_mut!(*__slate_slot_22), blocks);
return 0;
}
core::arch::asm!("498: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 498b - ., 3; .popsection\n# {0}", const 498 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_gpu_buddy_alloc_blocks_29, sym __slate_asm_string_gpu_buddy_alloc_blocks_30, const 1468 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
core::arch::asm!("497: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 497b - ., 3; .popsection\n# {0}", const 497 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_gpu_buddy_alloc_blocks_31, sym __slate_asm_string_gpu_buddy_alloc_blocks_32, const 1463 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
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
unsafe { core::mem::zeroed() }
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.buddy.gpu_buddy_allocated_addr_to_block")]
extern "C" fn gpu_buddy_allocated_addr_to_block(mm: *mut gpu_buddy, addr: u64) -> *mut gpu_buddy_block {
let mut block: *mut gpu_buddy_block = unsafe { core::mem::zeroed() };
let mut dfs: list_head = unsafe { core::mem::zeroed() };
dfs = list_head { next: core::ptr::addr_of_mut!(dfs), prev: core::ptr::addr_of_mut!(dfs) };
let mut end: u64 = 0;
let mut i: i32 = 0;
gpu_buddy_driver_lock_held(mm);
end = addr.wrapping_add(unsafe { (*mm).chunk_size }).wrapping_sub(1);
while (i as u32) < unsafe { (*mm).n_roots } {
list_add_tail(unsafe { core::ptr::addr_of_mut!((*(*unsafe { (*mm).roots }.wrapping_offset(i as isize))).tmp_link) }, core::ptr::addr_of_mut!(dfs));
i = i.wrapping_add(1);
}
loop {
let mut block_start: u64 = 0;
let mut block_end: u64 = 0;
let mut __v3: *mut gpu_buddy_block;
let head__: *mut list_head = core::ptr::addr_of_mut!(dfs);
let mut __v4: *mut list_head;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_477() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v4 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*head__).next) as *const *mut list_head) };
let pos__: *mut list_head = __v4;
let __v6: *mut gpu_buddy_block = if pos__ != head__ {
list_entry__9!(pos__, gpu_buddy_block)
} else {
core::ptr::null_mut::<gpu_buddy_block>()
};
__v3 = __v6;
block = __v3;
if block.is_null() {
break;
}
list_del(unsafe { core::ptr::addr_of_mut!((*block).tmp_link) });
block_start = gpu_buddy_block_offset(block as *const gpu_buddy_block);
block_end = block_start.wrapping_add(gpu_buddy_block_size(mm, block)).wrapping_sub(1);
if overlaps(addr, end, block_start, block_end) {
if gpu_buddy_block_is_allocated(block) {
return block;
} else {
if gpu_buddy_block_is_free(block) {
return core::ptr::null_mut::<gpu_buddy_block>();
}
}
list_add(unsafe { core::ptr::addr_of_mut!((*(*block).right).tmp_link) }, core::ptr::addr_of_mut!(dfs));
list_add(unsafe { core::ptr::addr_of_mut!((*(*block).left).tmp_link) }, core::ptr::addr_of_mut!(dfs));
}
if false {
break;
}
}
ERR_PTR(6i32.wrapping_neg() as i64) as *mut gpu_buddy_block
}

// gpu_buddy_assert - assert a condition in the buddy allocator
// @condition: condition expected to be true
//
// When CONFIG_KUNIT is enabled, evaluates @condition and, if false, triggers
// a WARN_ON() and also calls kunit_fail_current_test() so that any running
// kunit test is properly marked as failed. The stringified condition is
// included in the failure message for easy identification.
//
// When CONFIG_KUNIT is not enabled, this reduces to WARN_ON() so production
// builds retain the same warning semantics as before.
static mut slab_blocks: *mut kmem_cache = unsafe { core::mem::zeroed() };

fn gpu_buddy_block_state(block: *mut gpu_buddy_block) -> u32 {
((unsafe { (*block).header }) & GENMASK_ULL__8!(11i32, 10i32)) as u32
}

fn gpu_buddy_block_is_allocated(block: *mut gpu_buddy_block) -> bool {
gpu_buddy_block_state(block) == 1024u32
}

fn gpu_buddy_block_is_split(block: *mut gpu_buddy_block) -> bool {
gpu_buddy_block_state(block) == 3072u32
}

fn gpu_buddy_block_offset_alignment(block: *mut gpu_buddy_block) -> u32 {
let offset: u64 = gpu_buddy_block_offset(block as *const gpu_buddy_block);
if offset == 0 {
return 65u32;
}
// __ffs64(0) is undefined; offset 0 is maximally aligned, so return
// a value greater than any possible alignment.
__ffs64(offset)
}

fn gpu_buddy_augment_cb_compute_max(node: *mut gpu_buddy_block, exit: bool) -> bool {
let mut child: *mut gpu_buddy_block = unsafe { core::mem::zeroed() };
let mut max: u32 = gpu_buddy_block_offset_alignment(node);
if !unsafe { (*node).__slate_anon_5.rb.rb_left }.is_null() {
child = rb_entry!(unsafe { (*node).__slate_anon_5.rb.rb_left }, gpu_buddy_block);
if (unsafe { (*child).subtree_max_alignment }) > max {
max = unsafe { (*child).subtree_max_alignment };
}
}
if !unsafe { (*node).__slate_anon_5.rb.rb_right }.is_null() {
child = rb_entry!(unsafe { (*node).__slate_anon_5.rb.rb_right }, gpu_buddy_block);
if (unsafe { (*child).subtree_max_alignment }) > max {
max = unsafe { (*child).subtree_max_alignment };
}
}
if exit && (unsafe { (*node).subtree_max_alignment }) == max {
return true;
}
unsafe {
(*node).subtree_max_alignment = max;
}
false
}

fn gpu_buddy_augment_cb_copy_single(new: *mut gpu_buddy_block, old: *mut gpu_buddy_block) {
unsafe {
(*new).subtree_max_alignment = (*old).subtree_max_alignment;
}
}

#[unsafe(link_section = ".text.slate_distinct.buddy.gpu_buddy_augment_cb_propagate")]
extern "C" fn gpu_buddy_augment_cb_propagate(mut rb: *mut rb_node, stop: *mut rb_node) {
while rb != stop {
let node: *mut gpu_buddy_block = rb_entry!(rb, gpu_buddy_block);
if gpu_buddy_augment_cb_compute_max(node, true) {
break;
}
rb = rb_parent!(unsafe { core::ptr::addr_of_mut!((*node).__slate_anon_5.rb) });
}
}

#[unsafe(link_section = ".text.slate_distinct.buddy.gpu_buddy_augment_cb_copy")]
extern "C" fn gpu_buddy_augment_cb_copy(rb_old: *mut rb_node, rb_new: *mut rb_node) {
let old: *mut gpu_buddy_block = rb_entry!(rb_old, gpu_buddy_block);
let new: *mut gpu_buddy_block = rb_entry!(rb_new, gpu_buddy_block);
gpu_buddy_augment_cb_copy_single(new, old);
}

#[unsafe(link_section = ".text.slate_distinct.buddy.gpu_buddy_augment_cb_rotate")]
extern "C" fn gpu_buddy_augment_cb_rotate(rb_old: *mut rb_node, rb_new: *mut rb_node) {
let old: *mut gpu_buddy_block = rb_entry!(rb_old, gpu_buddy_block);
let new: *mut gpu_buddy_block = rb_entry!(rb_new, gpu_buddy_block);
gpu_buddy_augment_cb_copy_single(new, old);
gpu_buddy_augment_cb_compute_max(old, false);
}

static gpu_buddy_augment_cb: __SlateConst<rb_augment_callbacks> = __SlateConst(rb_augment_callbacks { propagate: Some(gpu_buddy_augment_cb_propagate), copy: Some(gpu_buddy_augment_cb_copy), rotate: Some(gpu_buddy_augment_cb_rotate) });

fn gpu_block_alloc(mm: *mut gpu_buddy, parent: *mut gpu_buddy_block, order: u32, offset: u64) -> *mut gpu_buddy_block {
let mut block: *mut gpu_buddy_block = unsafe { core::mem::zeroed() };
loop {
if unlikely!(order > 51u32) != 0 {
loop {
unsafe {
core::arch::asm!("456: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 456b - ., 3; .popsection\n# {0}", const 456 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_gpu_block_alloc_0, sym __slate_asm_string_gpu_block_alloc_1, const 82 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
block = kmem_cache_zalloc!(unsafe { slab_blocks }, 3264u32) as *mut gpu_buddy_block;
if block.is_null() {
return core::ptr::null_mut::<gpu_buddy_block>();
}
unsafe {
(*block).header = offset;
}
unsafe {
(*block).header = (*block).header | (order as u64);
}
unsafe {
(*block).parent = parent;
}
RB_CLEAR_NODE!(unsafe { core::ptr::addr_of_mut!((*block).__slate_anon_5.rb) });
loop {
if unlikely__4!((unsafe { (*block).header }) & GENMASK_ULL__21!(8i32, 6i32)) != 0 {
loop {
unsafe {
core::arch::asm!("457: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 457b - ., 3; .popsection\n# {0}", const 457 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_gpu_block_alloc_13, sym __slate_asm_string_gpu_block_alloc_14, const 94 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
block
}

fn gpu_block_free(mm: *mut gpu_buddy, block: *mut gpu_buddy_block) {
unsafe { kmem_cache_free(slab_blocks, block as *mut ()) };
}

fn get_block_tree(block: *mut gpu_buddy_block) -> u32 {
(if gpu_buddy_block_is_clear(block) { 0i32 } else { 1i32 }) as u32
}

fn rbtree_get_free_block(node: *const rb_node) -> *mut gpu_buddy_block {
let __v0: *mut gpu_buddy_block = if !node.is_null() {
rb_entry!(node, gpu_buddy_block)
} else {
core::ptr::null_mut::<gpu_buddy_block>()
};
__v0
}

fn rbtree_last_free_block(root: *mut rb_root) -> *mut gpu_buddy_block {
rbtree_get_free_block(rb_last(root as *const rb_root) as *const rb_node)
}

fn rbtree_is_empty(root: *mut rb_root) -> bool {
let mut __v0: *mut rb_node;
loop {
if !(__native_word__17!() || ((const { core::mem::size_of::<*mut rb_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_458() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*root).rb_node) as *const *mut rb_node) };
__v0.is_null()
}

fn rbtree_insert(mm: *mut gpu_buddy, block: *mut gpu_buddy_block, tree: u32) {
let mut link: *mut *mut rb_node = unsafe { core::mem::zeroed() };
let mut parent: *mut rb_node = core::ptr::null_mut::<rb_node>();
let mut block_alignment: u32 = 0;
let mut order: u32 = 0;
let mut root: *mut rb_root = unsafe { core::mem::zeroed() };
order = gpu_buddy_block_order(block);
block_alignment = gpu_buddy_block_offset_alignment(block);
root = unsafe { *unsafe { (*mm).free_trees }.wrapping_offset((tree as u32) as isize) }.wrapping_offset(order as isize);
link = unsafe { core::ptr::addr_of_mut!((*root).rb_node) };
while !unsafe { *link }.is_null() {
parent = unsafe { *link };
let node: *mut gpu_buddy_block = rbtree_get_free_block(parent as *const rb_node);
// Manual augmentation update during insertion traversal. Required
// because rb_insert_augmented() only calls rotate callback during
// rotations. This ensures all ancestors on the insertion path have
// correct subtree_max_alignment values.
if (unsafe { (*node).subtree_max_alignment }) < block_alignment {
unsafe {
(*node).subtree_max_alignment = block_alignment;
}
}
if gpu_buddy_block_offset(block as *const gpu_buddy_block) < gpu_buddy_block_offset(node as *const gpu_buddy_block) {
link = unsafe { core::ptr::addr_of_mut!((*parent).rb_left) };
} else {
link = unsafe { core::ptr::addr_of_mut!((*parent).rb_right) };
}
}
unsafe {
(*block).subtree_max_alignment = block_alignment;
}
rb_link_node(unsafe { core::ptr::addr_of_mut!((*block).__slate_anon_5.rb) }, parent, link);
rb_insert_augmented(unsafe { core::ptr::addr_of_mut!((*block).__slate_anon_5.rb) }, root, unsafe { core::ptr::addr_of!(gpu_buddy_augment_cb.0) });
}

fn rbtree_remove(mm: *mut gpu_buddy, block: *mut gpu_buddy_block) {
let order: u32 = gpu_buddy_block_order(block);
let mut tree: u32 = 0;
tree = get_block_tree(block);
let root: *mut rb_root = unsafe { *unsafe { (*mm).free_trees }.wrapping_offset((tree as u32) as isize) }.wrapping_offset(order as isize);
rb_erase_augmented(unsafe { core::ptr::addr_of_mut!((*block).__slate_anon_5.rb) }, root, unsafe { core::ptr::addr_of!(gpu_buddy_augment_cb.0) });
RB_CLEAR_NODE!(unsafe { core::ptr::addr_of_mut!((*block).__slate_anon_5.rb) });
}

fn clear_reset(__v0: *mut gpu_buddy_block) {
let __v1: u64 = unsafe { (*__v0).header };
let __v2: u64 = __v1 & !GENMASK_ULL__12!(9i32, 9i32);
unsafe {
(*__v0).header = __v2;
}
}

fn mark_cleared(__v0: *mut gpu_buddy_block) {
let __v1: u64 = unsafe { (*__v0).header };
let __v2: u64 = __v1 | GENMASK_ULL__4!(9i32, 9i32);
unsafe {
(*__v0).header = __v2;
}
}

fn mark_allocated(mm: *mut gpu_buddy, block: *mut gpu_buddy_block) {
let __v1: u64 = unsafe { (*block).header };
let __v2: u64 = __v1 & !GENMASK_ULL__20!(11i32, 10i32);
unsafe {
(*block).header = __v2;
}
unsafe {
(*block).header = (*block).header | 1024u64;
}
let __v6: *mut u64 = unsafe { (*mm).free_scoreboard }.wrapping_offset(gpu_buddy_block_order(block) as isize);
let __v7: u64 = unsafe { *__v6 };
unsafe {
*__v6 = __v7.wrapping_sub(1);
}
let __v9: *mut u64 = unsafe { (*mm).used_scoreboard }.wrapping_offset(gpu_buddy_block_order(block) as isize);
let __v10: u64 = unsafe { *__v9 };
unsafe {
*__v9 = __v10.wrapping_add(1);
}
rbtree_remove(mm, block);
}

fn mark_free(mm: *mut gpu_buddy, block: *mut gpu_buddy_block) {
let mut tree: u32 = 0;
if gpu_buddy_block_is_allocated(block) {
let __v0: *mut u64 = unsafe { (*mm).used_scoreboard }.wrapping_offset(gpu_buddy_block_order(block) as isize);
let __v1: u64 = unsafe { *__v0 };
unsafe {
*__v0 = __v1.wrapping_sub(1);
}
}
let __v4: u64 = unsafe { (*block).header };
let __v5: u64 = __v4 & !GENMASK_ULL__10!(11i32, 10i32);
unsafe {
(*block).header = __v5;
}
unsafe {
(*block).header = (*block).header | 2048u64;
}
let __v9: *mut u64 = unsafe { (*mm).free_scoreboard }.wrapping_offset(gpu_buddy_block_order(block) as isize);
let __v10: u64 = unsafe { *__v9 };
unsafe {
*__v9 = __v10.wrapping_add(1);
}
tree = get_block_tree(block);
rbtree_insert(mm, block, tree);
}

fn mark_split(mm: *mut gpu_buddy, block: *mut gpu_buddy_block) {
let __v1: u64 = unsafe { (*block).header };
let __v2: u64 = __v1 & !GENMASK_ULL__13!(11i32, 10i32);
unsafe {
(*block).header = __v2;
}
unsafe {
(*block).header = (*block).header | 3072u64;
}
let __v6: *mut u64 = unsafe { (*mm).free_scoreboard }.wrapping_offset(gpu_buddy_block_order(block) as isize);
let __v7: u64 = unsafe { *__v6 };
unsafe {
*__v6 = __v7.wrapping_sub(1);
}
rbtree_remove(mm, block);
}

fn overlaps(s1: u64, e1: u64, s2: u64, e2: u64) -> bool {
s1 <= e2 && e1 >= s2
}

fn contains(s1: u64, e1: u64, s2: u64, e2: u64) -> bool {
s1 <= s2 && e1 >= e2
}

fn __get_buddy(block: *mut gpu_buddy_block) -> *mut gpu_buddy_block {
let parent: *mut gpu_buddy_block = unsafe { (*block).parent };
if parent.is_null() {
return core::ptr::null_mut::<gpu_buddy_block>();
}
if (unsafe { (*parent).left }) == block {
return unsafe { (*parent).right };
}
unsafe { (*parent).left }
}

fn __gpu_buddy_free(mm: *mut gpu_buddy, mut block: *mut gpu_buddy_block, force_merge: bool) -> u32 {
let mut order: u32 = 0;
loop {
let __v12: *mut gpu_buddy_block = unsafe { (*block).parent };
let parent: *mut gpu_buddy_block = __v12;
if __v12.is_null() {
break;
}
let buddy: *mut gpu_buddy_block = __get_buddy(block);
if !gpu_buddy_block_is_free(buddy) {
break;
}
if !force_merge {
// Check the block and its buddy clear state and exit
// the loop if they both have the dissimilar state.
if (gpu_buddy_block_is_clear(block) as i32) != (gpu_buddy_block_is_clear(buddy) as i32) {
break;
}
if gpu_buddy_block_is_clear(block) {
mark_cleared(parent);
}
}
rbtree_remove(mm, buddy);
let __v1: *mut u64 = unsafe { (*mm).free_scoreboard }.wrapping_offset(gpu_buddy_block_order(buddy) as isize);
let __v2: u64 = unsafe { *__v1 };
unsafe {
*__v1 = __v2.wrapping_sub(1);
}
let __v4: bool = force_merge && gpu_buddy_block_is_clear(buddy);
if __v4 {
let __v5: u64 = gpu_buddy_block_size(mm, buddy);
let __v7: u64 = unsafe { (*mm).clear_avail };
unsafe {
(*mm).clear_avail = __v7.wrapping_sub(__v5);
}
}
if gpu_buddy_block_is_allocated(block) {
let __v9: *mut u64 = unsafe { (*mm).used_scoreboard }.wrapping_offset(gpu_buddy_block_order(block) as isize);
let __v10: u64 = unsafe { *__v9 };
unsafe {
*__v9 = __v10.wrapping_sub(1);
}
}
gpu_block_free(mm, block);
gpu_block_free(mm, buddy);
block = parent;
}
order = gpu_buddy_block_order(block);
mark_free(mm, block);
order
}

fn __force_merge(mm: *mut gpu_buddy, start: u64, end: u64, min_order: u32) -> i32 {
let mut order: u32 = 0;
let mut i: i32 = 0;
if min_order == 0 {
return 12i32.wrapping_neg();
}
if min_order > unsafe { (*mm).max_order } {
return 22i32.wrapping_neg();
}
for tree in 0u32..2u32 {
i = min_order.wrapping_sub(1) as i32;
while i >= 0 {
let mut iter: *mut rb_node = rb_last(unsafe { *unsafe { (*mm).free_trees }.wrapping_offset(tree as isize) }.wrapping_offset(i as isize) as *const rb_root);
while !iter.is_null() {
let mut buddy: *mut gpu_buddy_block = unsafe { core::mem::zeroed() };
let mut block_start: u64 = 0;
let mut block_end: u64 = 0;
let block: *mut gpu_buddy_block = rbtree_get_free_block(iter as *const rb_node);
iter = unsafe { rb_prev(iter as *const rb_node) };
if !(block.is_null() || unsafe { (*block).parent }.is_null()) {
block_start = gpu_buddy_block_offset(block as *const gpu_buddy_block);
block_end = block_start.wrapping_add(gpu_buddy_block_size(mm, block)).wrapping_sub(1);
if contains(start, end, block_start, block_end) {
buddy = __get_buddy(block);
if gpu_buddy_block_is_free(buddy) {
let mut __v1: i64;
let __ret_warn_on: i32 = ((gpu_buddy_block_is_clear(block) as i32) == (gpu_buddy_block_is_clear(buddy) as i32)) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("459: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 459b - ., 3; .popsection\n# {0}", const 459 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___force_merge_2, sym __slate_asm_string___force_merge_3, const 342 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("460: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 460b - ., 4; .popsection\n# {0}", const 460 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
// Advance to the next node when the current node is the buddy,
// as freeing the block will also remove its buddy from the tree.
if iter == unsafe { core::ptr::addr_of_mut!((*buddy).__slate_anon_5.rb) } {
iter = unsafe { rb_prev(iter as *const rb_node) };
}
rbtree_remove(mm, block);
let __v6: *mut u64 = unsafe { (*mm).free_scoreboard }.wrapping_offset(gpu_buddy_block_order(block) as isize);
let __v7: u64 = unsafe { *__v6 };
unsafe {
*__v6 = __v7.wrapping_sub(1);
}
if gpu_buddy_block_is_clear(block) {
let __v9: u64 = gpu_buddy_block_size(mm, block);
let __v11: u64 = unsafe { (*mm).clear_avail };
unsafe {
(*mm).clear_avail = __v11.wrapping_sub(__v9);
}
}
order = __gpu_buddy_free(mm, block, true);
if order >= min_order {
return 0;
}
}
}
}
}
i = i.wrapping_sub(1);
}
}
12i32.wrapping_neg()
}

/// gpu_buddy_init - init memory manager
///
/// @mm: GPU buddy manager to initialize
/// @size: size in bytes to manage
/// @chunk_size: minimum page size in bytes for our allocations
///
/// Initializes the memory manager and its resources.
///
/// Returns:
/// 0 on success, error code on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.buddy.gpu_buddy_init")]
extern "C" fn gpu_buddy_init(mm: *mut gpu_buddy, mut size: u64, chunk_size: u64) -> i32 {
let mut __slate_compound_51: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_53: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_55: rb_root = unsafe { core::mem::zeroed() };
let mut __slate_compound_56: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_58: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_59: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_61: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_62: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_64: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_65: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_67: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut root_size: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut u64 = core::ptr::addr_of_mut!(root_size) as *mut u64;
let mut order: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut u32 = core::ptr::addr_of_mut!(order) as *mut u32;
let mut root: core::mem::MaybeUninit<*mut gpu_buddy_block> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut gpu_buddy_block = core::ptr::addr_of_mut!(root) as *mut *mut gpu_buddy_block;
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut _old_1: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old_1) as *mut *mut alloc_tag;
let mut _res_1: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut *mut () = core::ptr::addr_of_mut!(_res_1) as *mut *mut ();
let mut __obj_size_1: u64 = unsafe { core::mem::zeroed() };
let mut _old_2: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_29: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old_2) as *mut *mut alloc_tag;
let mut _res_2: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_30: *mut *mut () = core::ptr::addr_of_mut!(_res_2) as *mut *mut ();
let mut __obj_size_2: u64 = unsafe { core::mem::zeroed() };
let mut _old_3: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_35: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old_3) as *mut *mut alloc_tag;
let mut _res_3: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_36: *mut *mut () = core::ptr::addr_of_mut!(_res_3) as *mut *mut ();
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_39: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_40: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut offset: u64 = unsafe { core::mem::zeroed() };
let mut root_count: u32 = unsafe { core::mem::zeroed() };
let mut j: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_45: *mut u32 = core::ptr::addr_of_mut!(j) as *mut u32;
let mut i: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_46: *mut u32 = core::ptr::addr_of_mut!(i) as *mut u32;
unsafe {
let __slate_auto_bytes_7 = __slate_slot_46 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_8 = __slate_slot_45 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 4usize);
root_count = 0;
offset = 0;
if size < chunk_size {
return 22i32.wrapping_neg();
} else {
if chunk_size < 4096 {
return 22i32.wrapping_neg();
} else {
if !is_power_of_2(chunk_size) {
return 22i32.wrapping_neg();
} else {
size = round_down!(size, chunk_size);
(*mm).size = size;
(*mm).avail = size;
(*mm).clear_avail = 0;
(*mm).chunk_size = chunk_size;
(*mm).max_order = ilog2!(size).wrapping_sub(ilog2!(chunk_size)) as u32;
'__join_50: {
loop {
if unlikely!((*mm).max_order > 51u32) != 0 {
break '__join_50;
} else {
if true {
break;
}
}
}
let __slate_auto_bytes_6 = __slate_slot_40 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_39 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_39 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_40 = _kmalloc_array_noprof(unsafe { (*mm).max_order }.wrapping_add(1) as u64, core::mem::size_of::<u64>() as u64, 3520u32, *{
__slate_compound_65 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_65)
});
alloc_tag_restore!();
} else {
*__slate_slot_40 = _kmalloc_array_noprof(unsafe { (*mm).max_order }.wrapping_add(1) as u64, core::mem::size_of::<u64>() as u64, 3520u32, *{
__slate_compound_67 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_67)
});
}
(*mm).free_scoreboard = *__slate_slot_40 as *mut u64;
if unsafe { (*mm).free_scoreboard }.is_null() {
return 12i32.wrapping_neg();
} else {
let __slate_auto_bytes_5 = __slate_slot_36 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_35 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_35 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_36 = _kmalloc_array_noprof(unsafe { (*mm).max_order }.wrapping_add(1) as u64, core::mem::size_of::<u64>() as u64, 3520u32, *{
__slate_compound_62 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_62)
});
alloc_tag_restore!();
} else {
*__slate_slot_36 = _kmalloc_array_noprof(unsafe { (*mm).max_order }.wrapping_add(1) as u64, core::mem::size_of::<u64>() as u64, 3520u32, *{
__slate_compound_64 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_64)
});
}
(*mm).used_scoreboard = *__slate_slot_36 as *mut u64;
if !unsafe { (*mm).used_scoreboard }.is_null() {
__obj_size_2 = size_mul(core::mem::size_of::<*mut rb_root>() as u64, 2u64);
let __slate_auto_bytes_4 = __slate_slot_30 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_29 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_29 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_30 = _kmalloc_noprof(__obj_size_2, 3264u32, *{
__slate_compound_59 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_59)
});
alloc_tag_restore!();
} else {
*__slate_slot_30 = _kmalloc_noprof(__obj_size_2, 3264u32, *{
__slate_compound_61 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_61)
});
}
(*mm).free_trees = *__slate_slot_30 as *mut *mut rb_root;
if !unsafe { (*mm).free_trees }.is_null() {
*__slate_slot_46 = 0;
'__join_3: {
while *__slate_slot_46 < 2 {
__obj_size_1 = size_mul(core::mem::size_of::<rb_root>() as u64, unsafe { (*mm).max_order }.wrapping_add(1) as u64);
let __slate_auto_bytes_3 = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_23 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_23 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_24 = _kmalloc_noprof(__obj_size_1, 3264u32, *{
__slate_compound_56 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_56)
});
alloc_tag_restore!();
} else {
*__slate_slot_24 = _kmalloc_noprof(__obj_size_1, 3264u32, *{
__slate_compound_58 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_58)
});
}
*unsafe { (*mm).free_trees }.wrapping_offset(*__slate_slot_46 as isize) = *__slate_slot_24 as *mut rb_root;
if unsafe { *unsafe { (*mm).free_trees }.wrapping_offset(*__slate_slot_46 as isize) }.is_null() {
break '__join_3;
} else {
*__slate_slot_45 = 0;
while *__slate_slot_45 <= (*mm).max_order {
*unsafe { *unsafe { (*mm).free_trees }.wrapping_offset(*__slate_slot_46 as isize) }.wrapping_offset(*__slate_slot_45 as isize) = *{
__slate_compound_55 = rb_root { rb_node: core::ptr::null_mut::<rb_node>() };
core::ptr::addr_of_mut!(__slate_compound_55)
};
let __v22: u32 = *__slate_slot_45;
*__slate_slot_45 = __v22.wrapping_add(1);
}
let __v20: u32 = *__slate_slot_46;
*__slate_slot_46 = __v20.wrapping_add(1);
}
}
(*mm).n_roots = hweight64!(size) as u32;
__obj_size = size_mul(core::mem::size_of::<*mut gpu_buddy_block>() as u64, (*mm).n_roots as u64);
let __slate_auto_bytes_2 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_13 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_14 = _kmalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_51 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_51)
});
alloc_tag_restore!();
} else {
*__slate_slot_14 = _kmalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_53 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_53)
});
}
(*mm).roots = *__slate_slot_14 as *mut *mut gpu_buddy_block;
if !unsafe { (*mm).roots }.is_null() {
// Split into power-of-two blocks, in case we are given a size that is
// not itself a power-of-two.
'__join_6: {
'__join_16: {
'__join_12: {
loop {
let __slate_auto_bytes_9 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_11 = ilog2!(size).wrapping_sub(ilog2!(chunk_size)) as u32;
*__slate_slot_10 = chunk_size << *__slate_slot_11;
*__slate_slot_12 = gpu_block_alloc(mm, core::ptr::null_mut::<gpu_buddy_block>(), *__slate_slot_11, offset);
if (*__slate_slot_12).is_null() {
break '__join_6;
} else {
mark_free(mm, *__slate_slot_12);
loop {
if unlikely!(root_count > (*mm).max_order) != 0 {
break '__join_16;
} else {
if true {
break;
}
}
}
loop {
if unlikely!(gpu_buddy_block_size(mm, *__slate_slot_12) < chunk_size) != 0 {
break '__join_12;
} else {
if true {
break;
}
}
}
*unsafe { (*mm).roots }.wrapping_offset(root_count as isize) = *__slate_slot_12;
offset = offset.wrapping_add(*__slate_slot_10);
size = size.wrapping_sub(*__slate_slot_10);
root_count = root_count.wrapping_add(1);
if size == 0 {
break;
}
}
}
return 0;
}
core::arch::asm!("463: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 463b - ., 3; .popsection\n# {0}", const 463 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_gpu_buddy_init_47, sym __slate_asm_string_gpu_buddy_init_48, const 453 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
core::arch::asm!("462: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 462b - ., 3; .popsection\n# {0}", const 462 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_gpu_buddy_init_49, sym __slate_asm_string_gpu_buddy_init_50, const 452 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
loop {
let __v3: u32 = root_count;
root_count = __v3.wrapping_sub(1);
if __v3 != 0 {
gpu_block_free(mm, *unsafe { (*mm).roots }.wrapping_offset(root_count as isize));
} else {
break;
}
}
kfree((*mm).roots as *const ());
}
}
loop {
let __v1: u32 = *__slate_slot_46;
*__slate_slot_46 = __v1.wrapping_sub(1);
if __v1 != 0 {
kfree(*unsafe { (*mm).free_trees }.wrapping_offset(*__slate_slot_46 as isize) as *const ());
} else {
break;
}
}
kfree((*mm).free_trees as *const ());
}
kfree((*mm).used_scoreboard as *const ());
}
kfree((*mm).free_scoreboard as *const ());
return 12i32.wrapping_neg();
}
}
core::arch::asm!("461: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 461b - ., 3; .popsection\n# {0}", const 461 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_gpu_buddy_init_68, sym __slate_asm_string_gpu_buddy_init_69, const 400 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
}
}
}
unsafe { core::mem::zeroed() }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_gpu_buddy_init_464: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut gpu_buddy, u64, u64) -> i32>, *mut ()>(Some(gpu_buddy_init)) };

/// gpu_buddy_fini - tear down the memory manager
///
/// @mm: GPU buddy manager to free
///
/// Cleanup memory manager resources and the freetree
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.buddy.gpu_buddy_fini")]
extern "C" fn gpu_buddy_fini(mm: *mut gpu_buddy) {
let mut root_size: u64 = 0;
let mut size: u64 = unsafe { (*mm).size };
let mut start: u64 = 0;
let mut order: u32 = 0;
let mut i: i32 = 0;
while (i as u32) < unsafe { (*mm).n_roots } {
order = ilog2!(size).wrapping_sub(ilog2!(unsafe { (*mm).chunk_size })) as u32;
start = gpu_buddy_block_offset((unsafe { *unsafe { (*mm).roots }.wrapping_offset(i as isize) }) as *const gpu_buddy_block);
__force_merge(mm, start, start.wrapping_add(size), order);
let mut __v0: i64;
let __ret_warn_on: i32 = !(!(!gpu_buddy_block_is_free(unsafe { *unsafe { (*mm).roots }.wrapping_offset(i as isize) }))) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("465: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 465b - ., 3; .popsection\n# {0}", const 465 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_gpu_buddy_fini_1, sym __slate_asm_string_gpu_buddy_fini_2, const 503 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("466: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 466b - ., 4; .popsection\n# {0}", const 466 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
gpu_block_free(mm, unsafe { *unsafe { (*mm).roots }.wrapping_offset(i as isize) });
root_size = (unsafe { (*mm).chunk_size }) << order;
size = size.wrapping_sub(root_size);
i = i.wrapping_add(1);
}
let mut __v10: i64;
{
let __ret_warn_on: i32 = ((unsafe { (*mm).avail }) != unsafe { (*mm).size }) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("467: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 467b - ., 3; .popsection\n# {0}", const 467 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_gpu_buddy_fini_11, sym __slate_asm_string_gpu_buddy_fini_12, const 511 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("468: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 468b - ., 4; .popsection\n# {0}", const 468 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v10 = (__ret_warn_on != 0) as i64;
}
i = 0;
while (i as u32) <= unsafe { (*mm).max_order } {
let mut __v15: i64;
let __ret_warn_on: i32 = ((unsafe { *unsafe { (*mm).used_scoreboard }.wrapping_offset(i as isize) }) != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("469: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 469b - ., 3; .popsection\n# {0}", const 469 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_gpu_buddy_fini_16, sym __slate_asm_string_gpu_buddy_fini_17, const 514 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("470: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 470b - ., 4; .popsection\n# {0}", const 470 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v15 = (__ret_warn_on != 0) as i64;
i = i.wrapping_add(1);
}
i = 0;
while i < 2 {
unsafe { kfree(*unsafe { (*mm).free_trees }.wrapping_offset(i as isize) as *const ()) };
i = i.wrapping_add(1);
}
unsafe { kfree((*mm).free_trees as *const ()) };
unsafe { kfree((*mm).roots as *const ()) };
unsafe { kfree((*mm).free_scoreboard as *const ()) };
unsafe { kfree((*mm).used_scoreboard as *const ()) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_gpu_buddy_fini_471: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut gpu_buddy)>, *mut ()>(Some(gpu_buddy_fini)) };

fn split_block(mm: *mut gpu_buddy, block: *mut gpu_buddy_block) -> i32 {
let block_order: u32 = gpu_buddy_block_order(block).wrapping_sub(1);
let offset: u64 = gpu_buddy_block_offset(block as *const gpu_buddy_block);
loop {
if unlikely!(!gpu_buddy_block_is_free(block)) != 0 {
loop {
unsafe {
core::arch::asm!("472: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 472b - ., 3; .popsection\n# {0}", const 472 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_split_block_0, sym __slate_asm_string_split_block_1, const 531 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
if unlikely!(gpu_buddy_block_order(block) == 0) != 0 {
loop {
unsafe {
core::arch::asm!("473: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 473b - ., 3; .popsection\n# {0}", const 473 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_split_block_5, sym __slate_asm_string_split_block_6, const 532 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
(*block).left = gpu_block_alloc(mm, block, block_order, offset);
}
if unsafe { (*block).left }.is_null() {
return 12i32.wrapping_neg();
}
unsafe {
(*block).right = gpu_block_alloc(mm, block, block_order, offset.wrapping_add((*mm).chunk_size << block_order));
}
if unsafe { (*block).right }.is_null() {
gpu_block_free(mm, unsafe { (*block).left });
return 12i32.wrapping_neg();
}
mark_split(mm, block);
if gpu_buddy_block_is_clear(block) {
mark_cleared(unsafe { (*block).left });
mark_cleared(unsafe { (*block).right });
clear_reset(block);
}
mark_free(mm, unsafe { (*block).left });
mark_free(mm, unsafe { (*block).right });
0
}

/// gpu_buddy_reset_clear - reset blocks clear state
///
/// @mm: GPU buddy manager
/// @is_clear: blocks clear state
///
/// Reset the clear state based on @is_clear value for each block
/// in the freetree.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.buddy.gpu_buddy_reset_clear")]
extern "C" fn gpu_buddy_reset_clear(mm: *mut gpu_buddy, is_clear: bool) {
let mut src_tree: u32 = 0;
let mut dst_tree: u32 = 0;
let mut root_size: u64 = 0;
let mut size: u64 = 0;
let mut start: u64 = 0;
let mut order: u32 = 0;
let mut i: i32 = 0;
gpu_buddy_driver_lock_held(mm);
size = unsafe { (*mm).size };
while (i as u32) < unsafe { (*mm).n_roots } {
order = ilog2!(size).wrapping_sub(ilog2!(unsafe { (*mm).chunk_size })) as u32;
start = gpu_buddy_block_offset((unsafe { *unsafe { (*mm).roots }.wrapping_offset(i as isize) }) as *const gpu_buddy_block);
__force_merge(mm, start, start.wrapping_add(size), order);
root_size = (unsafe { (*mm).chunk_size }) << order;
size = size.wrapping_sub(root_size);
i = i.wrapping_add(1);
}
src_tree = (if is_clear { 1i32 } else { 0i32 }) as u32;
dst_tree = (if is_clear { 0i32 } else { 1i32 }) as u32;
i = 0;
while (i as u32) <= unsafe { (*mm).max_order } {
let root: *mut rb_root = unsafe { *unsafe { (*mm).free_trees }.wrapping_offset((src_tree as u32) as isize) }.wrapping_offset(i as isize);
let mut block: *mut gpu_buddy_block = unsafe { core::mem::zeroed() };
let mut tmp: *mut gpu_buddy_block = unsafe { core::mem::zeroed() };
block = rb_entry_safe!(unsafe { rb_first_postorder(root as *const rb_root) });
loop {
let __v17: bool = if !block.is_null() {
let mut __v18: i32;
tmp = rb_entry_safe!(unsafe { rb_next_postorder(core::ptr::addr_of_mut!((*block).__slate_anon_5.rb) as *const rb_node) });
__v18 = 1;
__v18 != 0
} else {
false
};
if !__v17 {
break;
}
rbtree_remove(mm, block);
if is_clear {
mark_cleared(block);
let __v9: u64 = gpu_buddy_block_size(mm, block);
let __v11: u64 = unsafe { (*mm).clear_avail };
unsafe {
(*mm).clear_avail = __v11.wrapping_add(__v9);
}
} else {
clear_reset(block);
let __v13: u64 = gpu_buddy_block_size(mm, block);
let __v15: u64 = unsafe { (*mm).clear_avail };
unsafe {
(*mm).clear_avail = __v15.wrapping_sub(__v13);
}
}
rbtree_insert(mm, block, dst_tree);
block = tmp;
}
i = i.wrapping_add(1);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_gpu_buddy_reset_clear_474: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut gpu_buddy, bool)>, *mut ()>(Some(gpu_buddy_reset_clear)) };

/// gpu_buddy_free_block - free a block
///
/// @mm: GPU buddy manager
/// @block: block to be freed
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.buddy.gpu_buddy_free_block")]
extern "C" fn gpu_buddy_free_block(mm: *mut gpu_buddy, block: *mut gpu_buddy_block) {
gpu_buddy_driver_lock_held(mm);
loop {
if unlikely!(!gpu_buddy_block_is_allocated(block)) != 0 {
loop {
unsafe {
core::arch::asm!("475: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 475b - ., 3; .popsection\n# {0}", const 475 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_gpu_buddy_free_block_0, sym __slate_asm_string_gpu_buddy_free_block_1, const 619 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
let __v5: u64 = gpu_buddy_block_size(mm, block);
let __v7: u64 = unsafe { (*mm).avail };
unsafe {
(*mm).avail = __v7.wrapping_add(__v5);
}
if gpu_buddy_block_is_clear(block) {
let __v9: u64 = gpu_buddy_block_size(mm, block);
let __v11: u64 = unsafe { (*mm).clear_avail };
unsafe {
(*mm).clear_avail = __v11.wrapping_add(__v9);
}
}
__gpu_buddy_free(mm, block, false);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_gpu_buddy_free_block_476: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut gpu_buddy, *mut gpu_buddy_block)>, *mut ()>(Some(gpu_buddy_free_block)) };

// gpu_buddy_allocated_addr_to_block - given relative address find the allocated block
//
// @mm: GPU buddy manager
// @addr: Relative address
//
// Returns:
// gpu_buddy_block on success, NULL or error code on failure
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_gpu_buddy_allocated_addr_to_block_478: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut gpu_buddy, u64) -> *mut gpu_buddy_block>, *mut ()>(Some(gpu_buddy_allocated_addr_to_block)) };

fn __gpu_buddy_free_list(mm: *mut gpu_buddy, objects: *mut list_head, mark_clear: bool, mark_dirty: bool) {
let mut block: *mut gpu_buddy_block = unsafe { core::mem::zeroed() };
let mut on: *mut gpu_buddy_block = unsafe { core::mem::zeroed() };
let mut __v0: i64;
let __ret_warn_on: i32 = (mark_dirty && mark_clear) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("479: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 479b - ., 3; .popsection\n# {0}", const 479 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___gpu_buddy_free_list_1, sym __slate_asm_string___gpu_buddy_free_list_2, const 688 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("480: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 480b - ., 4; .popsection\n# {0}", const 480 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
block = list_first_entry__29!(objects);
on = list_next_entry__22!(block, link);
while list_entry_is_head__2!(block, objects, link) == 0 {
if mark_clear {
mark_cleared(block);
} else {
if mark_dirty {
clear_reset(block);
}
}
gpu_buddy_free_block(mm, block);
cond_resched__7!();
block = on;
on = list_next_entry__22!(on, link);
}
INIT_LIST_HEAD(objects);
}

fn gpu_buddy_free_list_internal(mm: *mut gpu_buddy, objects: *mut list_head) {
// Don't touch the clear/dirty bit, since allocation is still internal
// at this point. For example we might have just failed part of the
// allocation.
__gpu_buddy_free_list(mm, objects, false, false);
}

/// gpu_buddy_free_list - free blocks
///
/// @mm: GPU buddy manager
/// @objects: input list head to free blocks
/// @flags: optional flags like GPU_BUDDY_CLEARED
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.buddy.gpu_buddy_free_list")]
extern "C" fn gpu_buddy_free_list(mm: *mut gpu_buddy, objects: *mut list_head, flags: u32) {
let mark_clear: bool = (flags as u64) & BIT!(4i32) != 0;
gpu_buddy_driver_lock_held(mm);
__gpu_buddy_free_list(mm, objects, mark_clear, !mark_clear);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_gpu_buddy_free_list_481: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut gpu_buddy, *mut list_head, u32)>, *mut ()>(Some(gpu_buddy_free_list)) };

fn block_incompatible(block: *mut gpu_buddy_block, flags: u32) -> bool {
let needs_clear: bool = (flags as u64) & BIT!(3i32) != 0;
(needs_clear as i32) != (gpu_buddy_block_is_clear(block) as i32)
}

fn __gpu_buddy_undo_splits(mm: *mut gpu_buddy, block: *mut gpu_buddy_block) {
let buddy: *mut gpu_buddy_block = __get_buddy(block);
let __v0: bool = !buddy.is_null() && (gpu_buddy_block_is_free(block) && gpu_buddy_block_is_free(buddy));
if __v0 {
rbtree_remove(mm, block);
let __v2: *mut u64 = unsafe { (*mm).free_scoreboard }.wrapping_offset(gpu_buddy_block_order(block) as isize);
let __v3: u64 = unsafe { *__v2 };
unsafe {
*__v2 = __v3.wrapping_sub(1);
}
__gpu_buddy_free(mm, block, false);
}
}

fn __alloc_range_bias(mm: *mut gpu_buddy, start: u64, mut end: u64, order: u32, flags: u64, fallback: bool) -> *mut gpu_buddy_block {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_y__487: u64 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__486: u64 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_y__484: u64 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__483: u64 = unsafe { core::mem::zeroed() };
let mut adjusted_start: u64 = unsafe { core::mem::zeroed() };
let mut __v10: *mut gpu_buddy_block = unsafe { core::mem::zeroed() };
let mut pos__: *mut list_head = unsafe { core::mem::zeroed() };
let mut head__: *mut list_head = unsafe { core::mem::zeroed() };
let mut block_end: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut u64 = core::ptr::addr_of_mut!(block_end) as *mut u64;
let mut block_start: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut u64 = core::ptr::addr_of_mut!(block_start) as *mut u64;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut dfs: core::mem::MaybeUninit<list_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut list_head = core::ptr::addr_of_mut!(dfs) as *mut list_head;
let mut block: core::mem::MaybeUninit<*mut gpu_buddy_block> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut *mut gpu_buddy_block = core::ptr::addr_of_mut!(block) as *mut *mut gpu_buddy_block;
let mut req_size: u64 = unsafe { core::mem::zeroed() };
unsafe {
req_size = (*mm).chunk_size << order;
let __slate_auto_bytes_1 = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
core::ptr::write(__slate_slot_21, list_head { next: core::ptr::addr_of_mut!(*__slate_slot_21), prev: core::ptr::addr_of_mut!(*__slate_slot_21) });
let __slate_auto_bytes_2 = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
end = end.wrapping_sub(1);
*__slate_slot_19 = 0;
while (*__slate_slot_19 as u32) < (*mm).n_roots {
list_add_tail(core::ptr::addr_of_mut!((*(*unsafe { (*mm).roots }.wrapping_offset(*__slate_slot_19 as isize))).tmp_link), core::ptr::addr_of_mut!(*__slate_slot_21));
let __v18: i32 = *__slate_slot_19;
*__slate_slot_19 = __v18.wrapping_add(1);
}
'__join_1: {
'__loop_30: loop {
let __slate_auto_bytes = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
head__ = core::ptr::addr_of_mut!(*__slate_slot_21);
while false {
}
let __v11: *mut list_head = core::ptr::read_volatile(core::ptr::addr_of_mut!((*head__).next) as *const *mut list_head);
pos__ = __v11;
if pos__ != head__ {
__v10 = list_entry__9!(pos__, gpu_buddy_block);
} else {
__v10 = core::ptr::null_mut::<gpu_buddy_block>();
}
*__slate_slot_22 = __v10;
if (*__slate_slot_22).is_null() {
break '__join_1;
} else {
'__join_2: {
list_del(core::ptr::addr_of_mut!((*(*__slate_slot_22)).tmp_link));
if gpu_buddy_block_order(*__slate_slot_22) >= order {
*__slate_slot_16 = gpu_buddy_block_offset(*__slate_slot_22 as *const gpu_buddy_block);
*__slate_slot_15 = (*__slate_slot_16).wrapping_add(gpu_buddy_block_size(mm, *__slate_slot_22)).wrapping_sub(1);
if overlaps(start, end, *__slate_slot_16, *__slate_slot_15) {
if !gpu_buddy_block_is_allocated(*__slate_slot_22) {
if *__slate_slot_16 < start || *__slate_slot_15 > end {
__UNIQUE_ID_x__483 = *__slate_slot_16;
__UNIQUE_ID_y__484 = start;
while false {
}
adjusted_start = if __UNIQUE_ID_x__483 > __UNIQUE_ID_y__484 { __UNIQUE_ID_x__483 } else { __UNIQUE_ID_y__484 };
__UNIQUE_ID_x__486 = *__slate_slot_15;
__UNIQUE_ID_y__487 = end;
while false {
}
let adjusted_end: u64 = if __UNIQUE_ID_x__486 < __UNIQUE_ID_y__487 { __UNIQUE_ID_x__486 } else { __UNIQUE_ID_y__487 };
if round_down!(adjusted_end.wrapping_add(1), req_size) <= round_up!(adjusted_start, req_size) {
break '__join_2;
}
}
if !fallback {
__v1 = block_incompatible(*__slate_slot_22, flags as u32);
} else {
__v1 = false;
}
if !__v1 {
if contains(start, end, *__slate_slot_16, *__slate_slot_15) {
__v0 = order == gpu_buddy_block_order(*__slate_slot_22);
} else {
__v0 = false;
}
if __v0 {
// Find the free block within the range.
if gpu_buddy_block_is_free(*__slate_slot_22) {
return *__slate_slot_22;
}
} else {
if !gpu_buddy_block_is_split(*__slate_slot_22) {
*__slate_slot_20 = split_block(mm, *__slate_slot_22);
if unlikely__2!(*__slate_slot_20) != 0 {
break '__loop_30;
}
}
list_add(core::ptr::addr_of_mut!((*(*(*__slate_slot_22)).right).tmp_link), core::ptr::addr_of_mut!(*__slate_slot_21));
list_add(core::ptr::addr_of_mut!((*(*(*__slate_slot_22)).left).tmp_link), core::ptr::addr_of_mut!(*__slate_slot_21));
}
}
}
}
}
}
if false {
break '__join_1;
}
}
}
__gpu_buddy_undo_splits(mm, *__slate_slot_22);
// We really don't want to leave around a bunch of split blocks, since
// bigger is better, so make sure we merge everything back before we
// free the allocated blocks.
return ERR_PTR(*__slate_slot_20 as i64) as *mut gpu_buddy_block;
}
return ERR_PTR(28i32.wrapping_neg() as i64) as *mut gpu_buddy_block;
}
}

fn __gpu_buddy_alloc_range_bias(mm: *mut gpu_buddy, start: u64, end: u64, order: u32, flags: u64) -> *mut gpu_buddy_block {
let fallback: bool = false;
let block: *mut gpu_buddy_block = __alloc_range_bias(mm, start, end, order, flags, fallback);
if IS_ERR(block as *const ()) {
return __alloc_range_bias(mm, start, end, order, flags, !fallback);
}
block
}

fn get_maxblock(mm: *mut gpu_buddy, mut i: u32, tree: u32) -> *mut gpu_buddy_block {
let mut max_block: *mut gpu_buddy_block = core::ptr::null_mut::<gpu_buddy_block>();
let mut block: *mut gpu_buddy_block = core::ptr::null_mut::<gpu_buddy_block>();
while i <= unsafe { (*mm).max_order } {
let root: *mut rb_root = unsafe { *unsafe { (*mm).free_trees }.wrapping_offset((tree as u32) as isize) }.wrapping_offset(i as isize);
block = rbtree_last_free_block(root);
if !block.is_null() {
if max_block.is_null() {
max_block = block;
} else {
if gpu_buddy_block_offset(block as *const gpu_buddy_block) > gpu_buddy_block_offset(max_block as *const gpu_buddy_block) {
max_block = block;
}
}
}
i = i.wrapping_add(1);
}
max_block
}

fn alloc_from_freetree(mm: *mut gpu_buddy, order: u32, flags: u64) -> *mut gpu_buddy_block {
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut tmp: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u32 = core::ptr::addr_of_mut!(tmp) as *mut u32;
let mut tree: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut u32 = core::ptr::addr_of_mut!(tree) as *mut u32;
let mut root: core::mem::MaybeUninit<*mut rb_root> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut rb_root = core::ptr::addr_of_mut!(root) as *mut *mut rb_root;
let mut block: *mut gpu_buddy_block = unsafe { core::mem::zeroed() };
unsafe {
block = core::ptr::null_mut::<gpu_buddy_block>();
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_8 = (if flags & BIT!(3i32) != 0 { 0i32 } else { 1i32 }) as u32;
if flags & BIT!(1i32) != 0 {
block = get_maxblock(mm, order, *__slate_slot_8);
if !block.is_null() {
*__slate_slot_7 = gpu_buddy_block_order(block);
}
// Store the obtained block order
} else {
*__slate_slot_7 = order;
while *__slate_slot_7 <= (*mm).max_order {
// Get RB tree root for this order and tree
*__slate_slot_9 = unsafe { *unsafe { (*mm).free_trees }.wrapping_offset((*__slate_slot_8 as u32) as isize) }.wrapping_offset(*__slate_slot_7 as isize);
block = rbtree_last_free_block(*__slate_slot_9);
if !block.is_null() {
break;
} else {
let __v5: u32 = *__slate_slot_7;
*__slate_slot_7 = __v5.wrapping_add(1);
}
}
}
if block.is_null() {
// Try allocating from the other tree
*__slate_slot_8 = (if (*__slate_slot_8 as u32) == 0 { 1i32 } else { 0i32 }) as u32;
*__slate_slot_7 = order;
while *__slate_slot_7 <= (*mm).max_order {
*__slate_slot_9 = unsafe { *unsafe { (*mm).free_trees }.wrapping_offset((*__slate_slot_8 as u32) as isize) }.wrapping_offset(*__slate_slot_7 as isize);
block = rbtree_last_free_block(*__slate_slot_9);
if !block.is_null() {
break;
} else {
let __v3: u32 = *__slate_slot_7;
*__slate_slot_7 = __v3.wrapping_add(1);
}
}
if block.is_null() {
return ERR_PTR(28i32.wrapping_neg() as i64) as *mut gpu_buddy_block;
}
}
'__join_7: {
loop {
if unlikely!(!gpu_buddy_block_is_free(block)) != 0 {
break '__join_7;
} else {
if true {
break;
}
}
}
while *__slate_slot_7 != order {
*__slate_slot_6 = split_block(mm, block);
if unlikely__2!(*__slate_slot_6) != 0 {
__gpu_buddy_undo_splits(mm, block);
return ERR_PTR(*__slate_slot_6 as i64) as *mut gpu_buddy_block;
} else {
block = (*block).right;
let __v1: u32 = *__slate_slot_7;
*__slate_slot_7 = __v1.wrapping_sub(1);
}
}
return block;
}
core::arch::asm!("489: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 489b - ., 3; .popsection\n# {0}", const 489 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_alloc_from_freetree_11, sym __slate_asm_string_alloc_from_freetree_12, const 930 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
unsafe { core::mem::zeroed() }
}

fn gpu_buddy_can_offset_align(size: u64, min_block_size: u64) -> bool {
size < min_block_size && is_power_of_2(size)
}

fn gpu_buddy_subtree_can_satisfy(node: *mut rb_node, alignment: u32) -> bool {
let block: *mut gpu_buddy_block = rbtree_get_free_block(node as *const rb_node);
(unsafe { (*block).subtree_max_alignment }) >= alignment
}

fn gpu_buddy_find_block_aligned(mm: *mut gpu_buddy, tree: u32, order: u32, alignment: u32, flags: u64) -> *mut gpu_buddy_block {
let root: *mut rb_root = unsafe { *unsafe { (*mm).free_trees }.wrapping_offset((tree as u32) as isize) }.wrapping_offset(order as isize);
let mut rb: *mut rb_node = unsafe { (*root).rb_node };
'__slate_break_0: while !rb.is_null() {
'__slate_continue_0: {
let block: *mut gpu_buddy_block = rbtree_get_free_block(rb as *const rb_node);
let left_node: *mut rb_node = unsafe { (*rb).rb_left };
let right_node: *mut rb_node = unsafe { (*rb).rb_right };
if !right_node.is_null() {
if gpu_buddy_subtree_can_satisfy(right_node, alignment) {
rb = right_node;
break '__slate_continue_0;
}
}
if gpu_buddy_block_offset_alignment(block) >= alignment {
return block;
}
if !left_node.is_null() {
if gpu_buddy_subtree_can_satisfy(left_node, alignment) {
rb = left_node;
break '__slate_continue_0;
}
}
break '__slate_break_0;
}
}
core::ptr::null_mut::<gpu_buddy_block>()
}

fn gpu_buddy_offset_aligned_allocation(mm: *mut gpu_buddy, size: u64, min_block_size: u64, flags: u64) -> *mut gpu_buddy_block {
let mut right: core::mem::MaybeUninit<*mut gpu_buddy_block> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut gpu_buddy_block = core::ptr::addr_of_mut!(right) as *mut *mut gpu_buddy_block;
let mut left: core::mem::MaybeUninit<*mut gpu_buddy_block> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut gpu_buddy_block = core::ptr::addr_of_mut!(left) as *mut *mut gpu_buddy_block;
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut pages: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut u64 = core::ptr::addr_of_mut!(pages) as *mut u64;
let mut tree: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u32 = core::ptr::addr_of_mut!(tree) as *mut u32;
let mut alignment: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u32 = core::ptr::addr_of_mut!(alignment) as *mut u32;
let mut tmp: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut u32 = core::ptr::addr_of_mut!(tmp) as *mut u32;
let mut order: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut u32 = core::ptr::addr_of_mut!(order) as *mut u32;
let mut block: *mut gpu_buddy_block = unsafe { core::mem::zeroed() };
unsafe {
block = core::ptr::null_mut::<gpu_buddy_block>();
let __slate_auto_bytes_1 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
*__slate_slot_7 = ilog2!(min_block_size) as u32;
*__slate_slot_5 = size >> ilog2!(unsafe { (*mm).chunk_size });
*__slate_slot_9 = fls(*__slate_slot_5 as u32).wrapping_sub(1) as u32;
*__slate_slot_6 = (if flags & BIT!(3i32) != 0 { 0i32 } else { 1i32 }) as u32;
*__slate_slot_8 = *__slate_slot_9;
while *__slate_slot_8 <= (*mm).max_order {
block = gpu_buddy_find_block_aligned(mm, *__slate_slot_6, *__slate_slot_8, *__slate_slot_7, flags);
if block.is_null() {
*__slate_slot_6 = (if (*__slate_slot_6 as u32) == 0 { 1i32 } else { 0i32 }) as u32;
block = gpu_buddy_find_block_aligned(mm, *__slate_slot_6, *__slate_slot_8, *__slate_slot_7, flags);
}
if !block.is_null() {
break;
} else {
let __v3: u32 = *__slate_slot_8;
*__slate_slot_8 = __v3.wrapping_add(1);
}
}
if block.is_null() {
return ERR_PTR(28i32.wrapping_neg() as i64) as *mut gpu_buddy_block;
} else {
while gpu_buddy_block_order(block) > *__slate_slot_9 {
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_4 = split_block(mm, block);
if unlikely__2!(*__slate_slot_4) != 0 {
__gpu_buddy_undo_splits(mm, block);
// We really don't want to leave around a bunch of split blocks, since
// bigger is better, so make sure we merge everything back before we
// free the allocated blocks.
return ERR_PTR(*__slate_slot_4 as i64) as *mut gpu_buddy_block;
} else {
*__slate_slot_1 = (*block).left;
*__slate_slot_0 = (*block).right;
if gpu_buddy_block_offset_alignment(*__slate_slot_0) >= *__slate_slot_7 {
block = *__slate_slot_0;
} else {
block = *__slate_slot_1;
}
}
}
return block;
}
}
}

fn __alloc_range(mm: *mut gpu_buddy, dfs: *mut list_head, start: u64, size: u64, blocks: *mut list_head, total_allocated_on_err: *mut u64) -> i32 {
let mut __v11: *mut gpu_buddy_block = unsafe { core::mem::zeroed() };
let mut pos__: *mut list_head = unsafe { core::mem::zeroed() };
let mut head__: *mut list_head = unsafe { core::mem::zeroed() };
let mut block_end: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut u64 = core::ptr::addr_of_mut!(block_end) as *mut u64;
let mut block_start: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut u64 = core::ptr::addr_of_mut!(block_start) as *mut u64;
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut end: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut u64 = core::ptr::addr_of_mut!(end) as *mut u64;
let mut allocated: core::mem::MaybeUninit<list_head> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut list_head = core::ptr::addr_of_mut!(allocated) as *mut list_head;
let mut total_allocated: u64 = unsafe { core::mem::zeroed() };
let mut block: core::mem::MaybeUninit<*mut gpu_buddy_block> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut *mut gpu_buddy_block = core::ptr::addr_of_mut!(block) as *mut *mut gpu_buddy_block;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
total_allocated = 0;
core::ptr::write(__slate_slot_20, list_head { next: core::ptr::addr_of_mut!(*__slate_slot_20), prev: core::ptr::addr_of_mut!(*__slate_slot_20) });
let __slate_auto_bytes_2 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
*__slate_slot_19 = start.wrapping_add(size).wrapping_sub(1);
'__join_3: {
'__join_7: {
'__join_19: {
'__join_15: {
'__loop_27: loop {
let __slate_auto_bytes = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
head__ = dfs;
while false {
}
let __v12: *mut list_head = core::ptr::read_volatile(core::ptr::addr_of_mut!((*head__).next) as *const *mut list_head);
pos__ = __v12;
if pos__ != head__ {
__v11 = list_entry__9!(pos__, gpu_buddy_block);
} else {
__v11 = core::ptr::null_mut::<gpu_buddy_block>();
}
*__slate_slot_22 = __v11;
if (*__slate_slot_22).is_null() {
break '__join_7;
} else {
'__join_8: {
list_del(core::ptr::addr_of_mut!((*(*__slate_slot_22)).tmp_link));
*__slate_slot_17 = gpu_buddy_block_offset(*__slate_slot_22 as *const gpu_buddy_block);
*__slate_slot_16 = (*__slate_slot_17).wrapping_add(gpu_buddy_block_size(mm, *__slate_slot_22)).wrapping_sub(1);
if overlaps(start, *__slate_slot_19, *__slate_slot_17, *__slate_slot_16) {
if gpu_buddy_block_is_allocated(*__slate_slot_22) {
break '__join_19;
} else {
if contains(start, *__slate_slot_19, *__slate_slot_17, *__slate_slot_16) {
if gpu_buddy_block_is_free(*__slate_slot_22) {
mark_allocated(mm, *__slate_slot_22);
let __v10: u64 = gpu_buddy_block_size(mm, *__slate_slot_22);
total_allocated = total_allocated.wrapping_add(__v10);
let __v7: u64 = gpu_buddy_block_size(mm, *__slate_slot_22);
let __v5: u64 = (*mm).avail;
(*mm).avail = __v5.wrapping_sub(__v7);
if gpu_buddy_block_is_clear(*__slate_slot_22) {
let __v3: u64 = gpu_buddy_block_size(mm, *__slate_slot_22);
let __v1: u64 = (*mm).clear_avail;
(*mm).clear_avail = __v1.wrapping_sub(__v3);
}
list_add_tail(core::ptr::addr_of_mut!((*(*__slate_slot_22)).__slate_anon_5.link), core::ptr::addr_of_mut!(*__slate_slot_20));
break '__join_8;
} else {
if (*mm).clear_avail == 0 {
break '__join_15;
}
}
}
if !gpu_buddy_block_is_split(*__slate_slot_22) {
*__slate_slot_18 = split_block(mm, *__slate_slot_22);
if unlikely__2!(*__slate_slot_18) != 0 {
break '__loop_27;
}
}
list_add(core::ptr::addr_of_mut!((*(*(*__slate_slot_22)).right).tmp_link), dfs);
list_add(core::ptr::addr_of_mut!((*(*(*__slate_slot_22)).left).tmp_link), dfs);
}
}
}
if false {
break '__join_7;
}
}
}
__gpu_buddy_undo_splits(mm, *__slate_slot_22);
// We really don't want to leave around a bunch of split blocks, since
// bigger is better, so make sure we merge everything back before we
// free the allocated blocks.
break '__join_3;
}
*__slate_slot_18 = 28i32.wrapping_neg();
break '__join_3;
}
*__slate_slot_18 = 28i32.wrapping_neg();
break '__join_3;
}
if total_allocated < size {
*__slate_slot_18 = 28i32.wrapping_neg();
} else {
list_splice_tail(core::ptr::addr_of_mut!(*__slate_slot_20), blocks);
return 0;
}
}
if *__slate_slot_18 == 28i32.wrapping_neg() && !total_allocated_on_err.is_null() {
list_splice_tail(core::ptr::addr_of_mut!(*__slate_slot_20), blocks);
*total_allocated_on_err = total_allocated;
} else {
gpu_buddy_free_list_internal(mm, core::ptr::addr_of_mut!(*__slate_slot_20));
}
return *__slate_slot_18;
}
}

fn __gpu_buddy_alloc_range(mm: *mut gpu_buddy, start: u64, size: u64, total_allocated_on_err: *mut u64, blocks: *mut list_head) -> i32 {
let mut dfs: list_head = unsafe { core::mem::zeroed() };
dfs = list_head { next: core::ptr::addr_of_mut!(dfs), prev: core::ptr::addr_of_mut!(dfs) };
let mut i: i32 = 0;
while (i as u32) < unsafe { (*mm).n_roots } {
list_add_tail(unsafe { core::ptr::addr_of_mut!((*(*unsafe { (*mm).roots }.wrapping_offset(i as isize))).tmp_link) }, core::ptr::addr_of_mut!(dfs));
i = i.wrapping_add(1);
}
__alloc_range(mm, core::ptr::addr_of_mut!(dfs), start, size, blocks, total_allocated_on_err)
}

fn __alloc_contig_aligned_retry(mm: *mut gpu_buddy, unaligned_offset: u64, size: u64, min_block_size: u64, blocks: *mut list_head) -> i32 {
let aligned_offset: u64 = round_down!(unaligned_offset, min_block_size);
__gpu_buddy_alloc_range(mm, aligned_offset, size, core::ptr::null_mut::<u64>(), blocks)
}

fn __alloc_contig_try_harder(mm: *mut gpu_buddy, size: u64, min_block_size: u64, blocks: *mut list_head) -> i32 {
let mut iter: core::mem::MaybeUninit<*mut rb_node> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut rb_node = core::ptr::addr_of_mut!(iter) as *mut *mut rb_node;
let mut root: core::mem::MaybeUninit<*mut rb_root> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut rb_root = core::ptr::addr_of_mut!(root) as *mut *mut rb_root;
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut modify_size: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut u64 = core::ptr::addr_of_mut!(modify_size) as *mut u64;
let mut order: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u32 = core::ptr::addr_of_mut!(order) as *mut u32;
let mut tree: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u32 = core::ptr::addr_of_mut!(tree) as *mut u32;
let mut block: core::mem::MaybeUninit<*mut gpu_buddy_block> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut gpu_buddy_block = core::ptr::addr_of_mut!(block) as *mut *mut gpu_buddy_block;
let mut filled: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut u64 = core::ptr::addr_of_mut!(filled) as *mut u64;
let mut lhs_offset: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut u64 = core::ptr::addr_of_mut!(lhs_offset) as *mut u64;
let mut rhs_offset: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut u64 = core::ptr::addr_of_mut!(rhs_offset) as *mut u64;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_6 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_7 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_8 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 4usize);
*__slate_slot_5 = rounddown_pow_of_two__2!(size);
*__slate_slot_6 = ilog2!(*__slate_slot_5).wrapping_sub(ilog2!(unsafe { (*mm).chunk_size })) as u32;
if *__slate_slot_6 == 0 {
return 28i32.wrapping_neg();
} else {
*__slate_slot_7 = 0;
while *__slate_slot_7 < 2 {
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_3 = unsafe { *unsafe { (*mm).free_trees }.wrapping_offset(*__slate_slot_7 as isize) }.wrapping_offset(*__slate_slot_6 as isize);
if !rbtree_is_empty(*__slate_slot_3) {
*__slate_slot_2 = rb_last(*__slate_slot_3 as *const rb_root);
while !(*__slate_slot_2).is_null() {
*__slate_slot_8 = rbtree_get_free_block(*__slate_slot_2 as *const rb_node);
*__slate_slot_11 = gpu_buddy_block_offset(*__slate_slot_8 as *const gpu_buddy_block);
// Allocate blocks traversing RHS
*__slate_slot_4 = __gpu_buddy_alloc_range(mm, *__slate_slot_11, size, core::ptr::addr_of_mut!(*__slate_slot_9), blocks);
if *__slate_slot_4 != 0 && *__slate_slot_4 != 28i32.wrapping_neg() {
return *__slate_slot_4;
} else {
if *__slate_slot_4 == 0 && IS_ALIGNED!(*__slate_slot_11, min_block_size) {
return 0;
} else {
if *__slate_slot_4 == 0 {
// Allocate the unaligned RHS offset using round_down
gpu_buddy_free_list_internal(mm, blocks);
*__slate_slot_4 = __alloc_contig_aligned_retry(mm, *__slate_slot_11, size, min_block_size, blocks);
if *__slate_slot_4 == 0 {
return 0;
} else {
if *__slate_slot_4 != 28i32.wrapping_neg() {
gpu_buddy_free_list_internal(mm, blocks);
return *__slate_slot_4;
}
}
} else {
if !(size.wrapping_sub(*__slate_slot_9) > *__slate_slot_11) {
*__slate_slot_10 = (*__slate_slot_11).wrapping_sub(size.wrapping_sub(*__slate_slot_9));
// Allocate the unaligned LHS offset using round_down
gpu_buddy_free_list_internal(mm, blocks);
*__slate_slot_4 = __alloc_contig_aligned_retry(mm, *__slate_slot_10, size, min_block_size, blocks);
if *__slate_slot_4 == 0 {
return 0;
} else {
if *__slate_slot_4 != 28i32.wrapping_neg() {
gpu_buddy_free_list_internal(mm, blocks);
return *__slate_slot_4;
}
}
}
}
gpu_buddy_free_list_internal(mm, blocks);
*__slate_slot_2 = rb_prev(*__slate_slot_2 as *const rb_node);
}
}
}
}
let __v1: u32 = *__slate_slot_7;
*__slate_slot_7 = __v1.wrapping_add(1);
}
return 28i32.wrapping_neg();
}
}
}

/// gpu_buddy_block_trim - free unused pages
///
/// @mm: GPU buddy manager
/// @start: start address to begin the trimming.
/// @new_size: original size requested
/// @blocks: Input and output list of allocated blocks.
/// MUST contain single block as input to be trimmed.
/// On success will contain the newly allocated blocks
/// making up the @new_size. Blocks always appear in
/// ascending order
///
/// For contiguous allocation, we round up the size to the nearest
/// power of two value, drivers consume *actual* size, so remaining
/// portions are unused and can be optionally freed with this function
///
/// Returns:
/// 0 on success, error code on failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.buddy.gpu_buddy_block_trim")]
extern "C" fn gpu_buddy_block_trim(mm: *mut gpu_buddy, start: *mut u64, new_size: u64, blocks: *mut list_head) -> i32 {
let mut block: *mut gpu_buddy_block = unsafe { core::mem::zeroed() };
let mut block_start: u64 = 0;
let mut block_end: u64 = 0;
let mut dfs: list_head = unsafe { core::mem::zeroed() };
dfs = list_head { next: core::ptr::addr_of_mut!(dfs), prev: core::ptr::addr_of_mut!(dfs) };
let mut new_start: u64 = 0;
let mut err: i32 = 0;
gpu_buddy_driver_lock_held(mm);
if list_is_singular(blocks as *const list_head) == 0 {
return 22i32.wrapping_neg();
}
block = list_first_entry__40!(blocks, gpu_buddy_block);
block_start = gpu_buddy_block_offset(block as *const gpu_buddy_block);
block_end = block_start.wrapping_add(gpu_buddy_block_size(mm, block));
let mut __v1: i64;
let __ret_warn_on: i32 = !(!(!gpu_buddy_block_is_allocated(block))) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("491: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 491b - ., 3; .popsection\n# {0}", const 491 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_gpu_buddy_block_trim_2, sym __slate_asm_string_gpu_buddy_block_trim_3, const 1300 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("492: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 492b - ., 4; .popsection\n# {0}", const 492 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
if __v1 != 0 {
return 22i32.wrapping_neg();
}
if new_size > gpu_buddy_block_size(mm, block) {
return 22i32.wrapping_neg();
}
if new_size == 0 || !IS_ALIGNED!(new_size, unsafe { (*mm).chunk_size }) {
return 22i32.wrapping_neg();
}
if new_size == gpu_buddy_block_size(mm, block) {
return 0;
}
new_start = block_start;
if !start.is_null() {
new_start = unsafe { *start };
if new_start < block_start {
return 22i32.wrapping_neg();
}
if !IS_ALIGNED!(new_start, unsafe { (*mm).chunk_size }) {
return 22i32.wrapping_neg();
}
if range_overflows!(new_start, new_size, block_end) {
return 22i32.wrapping_neg();
}
}
list_del(unsafe { core::ptr::addr_of_mut!((*block).__slate_anon_5.link) });
mark_free(mm, block);
let __v10: u64 = gpu_buddy_block_size(mm, block);
let __v12: u64 = unsafe { (*mm).avail };
unsafe {
(*mm).avail = __v12.wrapping_add(__v10);
}
if gpu_buddy_block_is_clear(block) {
let __v14: u64 = gpu_buddy_block_size(mm, block);
let __v16: u64 = unsafe { (*mm).clear_avail };
unsafe {
(*mm).clear_avail = __v16.wrapping_add(__v14);
}
}
// Prevent recursively freeing this node
let parent: *mut gpu_buddy_block = unsafe { (*block).parent };
unsafe {
(*block).parent = core::ptr::null_mut::<gpu_buddy_block>();
}
list_add(unsafe { core::ptr::addr_of_mut!((*block).tmp_link) }, core::ptr::addr_of_mut!(dfs));
err = __alloc_range(mm, core::ptr::addr_of_mut!(dfs), new_start, new_size, blocks, core::ptr::null_mut::<u64>());
if err != 0 {
mark_allocated(mm, block);
let __v18: u64 = gpu_buddy_block_size(mm, block);
let __v20: u64 = unsafe { (*mm).avail };
unsafe {
(*mm).avail = __v20.wrapping_sub(__v18);
}
if gpu_buddy_block_is_clear(block) {
let __v22: u64 = gpu_buddy_block_size(mm, block);
let __v24: u64 = unsafe { (*mm).clear_avail };
unsafe {
(*mm).clear_avail = __v24.wrapping_sub(__v22);
}
}
list_add(unsafe { core::ptr::addr_of_mut!((*block).__slate_anon_5.link) }, blocks);
}
unsafe {
(*block).parent = parent;
}
err
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_gpu_buddy_block_trim_493: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut gpu_buddy, *mut u64, u64, *mut list_head) -> i32>, *mut ()>(Some(gpu_buddy_block_trim)) };

fn __gpu_buddy_alloc_blocks(mm: *mut gpu_buddy, start: u64, end: u64, size: u64, min_block_size: u64, order: u32, flags: u64) -> *mut gpu_buddy_block {
if flags & BIT!(0i32) != 0 {
return __gpu_buddy_alloc_range_bias(mm, start, end, order, flags);
} else {
if size < min_block_size {
return gpu_buddy_offset_aligned_allocation(mm, size, min_block_size, flags);
} else {
return alloc_from_freetree(mm, order, flags);
}
}
// Allocate traversing within the range
// Allocate from an offset-aligned region without size rounding
// Allocate from freetree
}

// gpu_buddy_alloc_blocks - allocate power-of-two blocks
//
// @mm: GPU buddy manager to allocate from
// @start: start of the allowed range for this block
// @end: end of the allowed range for this block
// @size: size of the allocation in bytes
// @min_block_size: alignment of the allocation
// @blocks: output list head to add allocated blocks
// @flags: GPU_BUDDY_*_ALLOCATION flags
//
// alloc_range_bias() called on range limitations, which traverses
// the tree and returns the desired block.
//
// alloc_from_freetree() called when *no* range restrictions
// are enforced, which picks the block from the freetree.
//
// Returns:
// 0 on success, error code on failure.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_gpu_buddy_alloc_blocks_499: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut gpu_buddy, u64, u64, u64, u64, *mut list_head, u64) -> i32>, *mut ()>(Some(gpu_buddy_alloc_blocks)) };

/// gpu_buddy_block_print - print block information
///
/// @mm: GPU buddy manager
/// @block: GPU buddy block
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.buddy.gpu_buddy_block_print")]
extern "C" fn gpu_buddy_block_print(mm: *mut gpu_buddy, block: *mut gpu_buddy_block) {
let start: u64 = gpu_buddy_block_offset(block as *const gpu_buddy_block);
let size: u64 = gpu_buddy_block_size(mm, block);
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x016%#018llx-%#018llx: %llu\n".as_ptr() as *const u8, start, start.wrapping_add(size), size) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_gpu_buddy_block_print_500: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut gpu_buddy, *mut gpu_buddy_block)>, *mut ()>(Some(gpu_buddy_block_print)) };

/// gpu_buddy_print - print allocator state
///
/// @mm: GPU buddy manager
/// @p: GPU printer to use
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.buddy.gpu_buddy_print")]
extern "C" fn gpu_buddy_print(mm: *mut gpu_buddy) {
let mut order: i32 = 0;
gpu_buddy_driver_lock_held(mm);
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x016chunk_size: %lluKiB, total: %lluMiB, free: %lluMiB, clear_free: %lluMiB\n".as_ptr() as *const u8, (*mm).chunk_size >> 10i32, (*mm).size >> 20i32, (*mm).avail >> 20i32, (*mm).clear_avail >> 20i32) };
order = (unsafe { (*mm).max_order }) as i32;
while order >= 0 {
let free_count: u64 = unsafe { *unsafe { (*mm).free_scoreboard }.wrapping_offset(order as isize) };
let used_count: u64 = unsafe { *unsafe { (*mm).used_scoreboard }.wrapping_offset(order as isize) };
let block_size: u64 = (unsafe { (*mm).chunk_size }) << order;
let free: u64 = free_count.wrapping_mul(block_size);
let used: u64 = used_count.wrapping_mul(block_size);
if block_size < 1048576 {
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x016order-%2d free: %8llu KiB, used: %8llu KiB, free_blocks: %llu, used_blocks: %llu\n".as_ptr() as *const u8, order, free >> 10i32, used >> 10i32, free_count, used_count) };
} else {
let mut __v4: i32;
while false {
}
__v4 = unsafe { _printk(c"\x016order-%2d free: %8llu MiB, used: %8llu MiB, free_blocks: %llu, used_blocks: %llu\n".as_ptr() as *const u8, order, free >> 20i32, used >> 20i32, free_count, used_count) };
}
order = order.wrapping_sub(1);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_gpu_buddy_print_501: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut gpu_buddy)>, *mut ()>(Some(gpu_buddy_print)) };

#[unsafe(link_section = ".text.slate_distinct.buddy.gpu_buddy_module_exit")]
extern "C" fn gpu_buddy_module_exit() {
unsafe { kmem_cache_destroy(slab_blocks) };
}

#[unsafe(link_section = ".init.text")]
extern "C" fn gpu_buddy_module_init() -> i32 {
let mut __slate_compound_0: kmem_cache_args = unsafe { core::mem::zeroed() };
unsafe {
slab_blocks = __kmem_cache_create_args(c"gpu_buddy_block".as_ptr() as *const u8, core::mem::size_of::<gpu_buddy_block>() as u32, core::ptr::addr_of_mut!(*{
__slate_compound_0 = {
let mut __t0: kmem_cache_args = core::mem::zeroed();
__t0.align = core::mem::align_of::<gpu_buddy_block>() as u32;
__t0
};
core::ptr::addr_of_mut!(__slate_compound_0)
}), 0u32);
}
if unsafe { slab_blocks }.is_null() {
return 12i32.wrapping_neg();
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_gpu_buddy_module_init_503: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i32>, *mut ()>(Some(gpu_buddy_module_init)) };

#[unsafe(link_section = ".exitcall.exit")]
static mut __exitcall_gpu_buddy_module_exit: Option<unsafe extern "C" fn()> = Some(gpu_buddy_module_exit);

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_504: [u8; 38] = [98, 117, 100, 100, 121, 46, 100, 101, 115, 99, 114, 105, 112, 116, 105, 111, 110, 61, 71, 80, 85, 32, 66, 117, 100, 100, 121, 32, 65, 108, 108, 111, 99, 97, 116, 111, 114, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_505: [u8; 29] = [98, 117, 100, 100, 121, 46, 102, 105, 108, 101, 61, 100, 114, 105, 118, 101, 114, 115, 47, 103, 112, 117, 47, 98, 117, 100, 100, 121, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_506: [u8; 27] = [98, 117, 100, 100, 121, 46, 108, 105, 99, 101, 110, 115, 101, 61, 68, 117, 97, 108, 32, 77, 73, 84, 47, 71, 80, 76, 0];

static __slate_asm_string___force_merge_2: [u8; 1] = *b"\0";

static __slate_asm_string___force_merge_3: [u8; 49] = *b"/home/takashi/c-corpus/linux/drivers/gpu/buddy.c\0";

static __slate_asm_string___gpu_buddy_free_list_1: [u8; 1] = *b"\0";

static __slate_asm_string___gpu_buddy_free_list_2: [u8; 49] = *b"/home/takashi/c-corpus/linux/drivers/gpu/buddy.c\0";

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_alloc_from_freetree_11: [u8; 1] = *b"\0";

static __slate_asm_string_alloc_from_freetree_12: [u8; 49] = *b"/home/takashi/c-corpus/linux/drivers/gpu/buddy.c\0";

static __slate_asm_string_gpu_block_alloc_0: [u8; 1] = *b"\0";

static __slate_asm_string_gpu_block_alloc_1: [u8; 49] = *b"/home/takashi/c-corpus/linux/drivers/gpu/buddy.c\0";

static __slate_asm_string_gpu_block_alloc_13: [u8; 1] = *b"\0";

static __slate_asm_string_gpu_block_alloc_14: [u8; 49] = *b"/home/takashi/c-corpus/linux/drivers/gpu/buddy.c\0";

static __slate_asm_string_gpu_buddy_alloc_blocks_29: [u8; 1] = *b"\0";

static __slate_asm_string_gpu_buddy_alloc_blocks_30: [u8; 49] = *b"/home/takashi/c-corpus/linux/drivers/gpu/buddy.c\0";

static __slate_asm_string_gpu_buddy_alloc_blocks_31: [u8; 1] = *b"\0";

static __slate_asm_string_gpu_buddy_alloc_blocks_32: [u8; 49] = *b"/home/takashi/c-corpus/linux/drivers/gpu/buddy.c\0";

static __slate_asm_string_gpu_buddy_block_trim_2: [u8; 1] = *b"\0";

static __slate_asm_string_gpu_buddy_block_trim_3: [u8; 49] = *b"/home/takashi/c-corpus/linux/drivers/gpu/buddy.c\0";

static __slate_asm_string_gpu_buddy_fini_1: [u8; 1] = *b"\0";

static __slate_asm_string_gpu_buddy_fini_11: [u8; 1] = *b"\0";

static __slate_asm_string_gpu_buddy_fini_12: [u8; 49] = *b"/home/takashi/c-corpus/linux/drivers/gpu/buddy.c\0";

static __slate_asm_string_gpu_buddy_fini_16: [u8; 1] = *b"\0";

static __slate_asm_string_gpu_buddy_fini_17: [u8; 49] = *b"/home/takashi/c-corpus/linux/drivers/gpu/buddy.c\0";

static __slate_asm_string_gpu_buddy_fini_2: [u8; 49] = *b"/home/takashi/c-corpus/linux/drivers/gpu/buddy.c\0";

static __slate_asm_string_gpu_buddy_free_block_0: [u8; 1] = *b"\0";

static __slate_asm_string_gpu_buddy_free_block_1: [u8; 49] = *b"/home/takashi/c-corpus/linux/drivers/gpu/buddy.c\0";

static __slate_asm_string_gpu_buddy_init_47: [u8; 1] = *b"\0";

static __slate_asm_string_gpu_buddy_init_48: [u8; 49] = *b"/home/takashi/c-corpus/linux/drivers/gpu/buddy.c\0";

static __slate_asm_string_gpu_buddy_init_49: [u8; 1] = *b"\0";

static __slate_asm_string_gpu_buddy_init_50: [u8; 49] = *b"/home/takashi/c-corpus/linux/drivers/gpu/buddy.c\0";

static __slate_asm_string_gpu_buddy_init_68: [u8; 1] = *b"\0";

static __slate_asm_string_gpu_buddy_init_69: [u8; 49] = *b"/home/takashi/c-corpus/linux/drivers/gpu/buddy.c\0";

static __slate_asm_string_split_block_0: [u8; 1] = *b"\0";

static __slate_asm_string_split_block_1: [u8; 49] = *b"/home/takashi/c-corpus/linux/drivers/gpu/buddy.c\0";

static __slate_asm_string_split_block_5: [u8; 1] = *b"\0";

static __slate_asm_string_split_block_6: [u8; 49] = *b"/home/takashi/c-corpus/linux/drivers/gpu/buddy.c\0";


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
//! +++ deflate.c
//!
//! deflate.c -- compress data using the deflation algorithm
//! Copyright (C) 1995-1996 Jean-loup Gailly.
//! For conditions of distribution and use, see copyright notice in zlib.h
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
//! container_of - cast a member of a structure out to the containing structure
//! @ptr:	the pointer to the member.
//! @type:	the type of the container struct this is embedded in.
//! @member:	the name of the member within the struct.
//!
//! WARNING: any const qualifier of @ptr is lost.
//! Do not use container_of() in new code.
//! These are non-NULL pointers that will result in page faults
//! under normal circumstances, used to verify that nobody uses
//! non-initialized list entries.
//! See linux/instrumentation.h
//! 64 bit
//! unused, was: #define EX_TYPE_COPY	 4
//! Begin/end of an instrumentation safe region
//! __ADDRESSABLE() is used to ensure the key symbol doesn't get stripped from
//! the symbol table so that objtool can reference it when it generates the
//! .static_call_sites section.
//! inhibit tail-call optimization
//! Call a function once. Similar to DO_ONCE(), but does not use jump label
//! patching via static keys.
//! These are for everybody (although not all archs will actually
//! discard it in modules)
//! warning conditions
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
//! We need to compute the minimum and maximum values representable in a given
//! type. These macros may also be useful elsewhere. It would seem more obvious
//! to do something like:
//!
//! #define type_min(T) (T)(is_signed_type(T) ? (T)1 << (8*sizeof(T)-1) : 0)
//! #define type_max(T) (T)(is_signed_type(T) ? ((T)1 << (8*sizeof(T)-1)) - 1 : ~(T)0)
//!
//! Unfortunately, the middle expressions, strictly speaking, have
//! undefined behaviour, and at least some versions of gcc warn about
//! the type_max expression (but not if -fsanitize=undefined is in
//! effect; in that case, the warning is deferred to runtime...).
//!
//! The slightly excessive casting in type_min is to make sure the
//! macros also produce sensible values for the exotic type _Bool. [The
//! overflow checkers only almost work for _Bool, but that's
//! a-feature-not-a-bug, since people shouldn't be doing arithmetic on
//! _Bools. Besides, the gcc builtins don't allow _Bool* as third
//! argument.]
//!
//! Idea stolen from
//! https://mail-index.netbsd.org/tech-misc/2007/02/05/0000.html -
//! credit to Christian Biere.
//! Generate a mask for the specified type @t. Additional checks are made to
//! guarantee the value returned fits in that type, relying on
//! -Wshift-count-overflow compiler check to detect incompatible arguments.
//! For example, all these create build errors or warnings:
//!
//! - GENMASK(15, 20): wrong argument order
//! - GENMASK(72, 15): doesn't fit unsigned long
//! - GENMASK_U32(33, 15): doesn't fit in a u32
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
//! label
//! Prevent speculative execution past this barrier.
//! list_entry - get the struct for this entry
//! @ptr:	the &struct list_head pointer.
//! @type:	the type of the struct this is embedded in.
//! @member:	the name of the list_head within the struct.
//! This looks more complex than it should be. But we need to
//! get the type for the ~ right in round_down (it needs to be
//! as wide as the result!), and we want to evaluate the macro
//! arguments just once each.
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
//! 4th level page in 5-level paging case
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
//! An exchange-type operation, which takes a value and a pointer, and
//! returns the old value.
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
//! cpumask_bits - get the bits in a cpumask
//! @maskp: the struct cpumask *
//!
//! You should only assume nr_cpu_ids bits of this mask are valid.  This is
//! a macro so it's const-correct.
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
//! This is the default CPU features testing macro to use in code.
//!
//! It is for detection of features which need kernel infrastructure to be
//! used.  It may *not* directly test the CPU itself.  Use the cpu_has() family
//! if you want true runtime testing of CPU features, like in hypervisor code
//! where you are supporting a possible guest feature where host support for it
//! is not relevant.
//! We use the MSB for PREEMPT_NEED_RESCHED mostly because it is available.
//! Macros to retrieve the current execution context:
//!
//! in_nmi()		- We're in NMI context
//! in_hardirq()		- We're in hard IRQ context
//! in_serving_softirq()	- We're in softirq context
//! in_task()		- We're in task context
//! signal pending
//! 32-bit address space on 64 bits
//! SCHED_ISO: reserved but not implemented yet
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
//! GFP_ZONE_BAD is a bitmap for all combinations of __GFP_DMA, __GFP_DMA32
//! __GFP_HIGHMEM and __GFP_MOVABLE that are not permitted. One flag per
//! entry starting with bit 0. Bit is set if the combination is not
//! allowed.
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
//! 192 bytes
//! Keep unconverted code working
//! used to be O_SYNC, see below
//! Valid hint values for F_{GET,SET}_RW_HINT. 0 is "not set", or can be
//! used to clear any hints previously set.
//! Gives us 8 prio classes with 13-bits of data for each class
//! vfs{g,u}ids are created from k{g,u}ids.
//! We don't allow them to be created from regular {u,g}id.
//! kmem_cache_alloc and friends return pointers aligned to ARCH_SLAB_MINALIGN.
//! kmalloc and friends return pointers aligned to both ARCH_KMALLOC_MINALIGN
//! and ARCH_SLAB_MINALIGN, but here we only assume the former alignment.
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
//! Only touch one cacheline for common rbtree-for-core-layout case.
//! preset dictionary flag in zlib header
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
// Getting something that works in C and CPP for an arg that may or may
// not be defined is tricky.  Here, if we have "#define CONFIG_BOOGER 1"
// we match on the placeholder define, insert the "0," for arg1 and generate
// the triplet (0, 1, 0).  Then the last step cherry picks the 2nd arg (a one).
// When CONFIG_BOOGER is not defined, we generate a (... 1, 0) pair, and when
// the last step cherry picks the 2nd arg, we get a zero.
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
// IS_ENABLED(CONFIG_FOO) evaluates to 1 if CONFIG_FOO is set to 'y' or 'm',
// 0 otherwise.  Note that CONFIG_FOO=y results in "#define CONFIG_FOO 1" in
// autoconf.h, while CONFIG_FOO=m results in "#define CONFIG_FOO_MODULE 1".
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
// Is this type a native word size -- useful for atomic operations
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
// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
// bool and also pointer types.
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
// generic data direction definitions
// The type of an index into the pagecache.
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
// round_up - round up to next specified power of 2
// @x: the value to round
// @y: multiple to round up to (must be a power of 2)
//
// Rounds @x up to next multiple of @y (which must be a power of 2).
// To perform arbitrary rounding up, use roundup() below.
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
// N 32-bit words worth of info
// N 32-bit bug flags
// "tsc" Time Stamp Counter
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
// No such file or directory
// Argument list too long
// Try again
// Out of memory
// Bad address
// Device or resource busy
// Invalid argument
// Text file busy
// Operation not supported on transport endpoint
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
// Use these carefully: no type checking, and uses the arguments
// multiple times. Use for obvious constants only.
// No static inline type checking - see Subtlety (1) above.
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
// re-arm the timer
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
// Same as rb_first(), but O(1)
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
// SLUB directly allocates requests fitting in to an order-1 page
// (PAGE_SIZE*2).  Larger requests are passed to the page allocator.
// Maximum size for which we actually use a slab cache
// Define gfp bits that should not be set for KMALLOC_NORMAL.
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
/// Initialize the hash table (avoiding 64K overflow for 16 bit systems).
/// prev[] will be initialized on the fly.
macro_rules! CLEAR_HASH {
($s:expr) => {
unsafe {
*unsafe { (*$s).head }.wrapping_offset(unsafe { (*$s).hash_size }.wrapping_sub(1) as isize) = 0;
}
unsafe { memset(((*$s).head as *mut u8) as *mut (), 0i32, (unsafe { (*$s).hash_size }.wrapping_sub(1) as u64).wrapping_mul(core::mem::size_of::<u16>() as u64)) };
};
}

macro_rules! DEFLATE_DFLTCC_ENABLED {
() => {
0i32
};
}

macro_rules! DEFLATE_HOOK {
() => {
0i32
};
}

macro_rules! DEFLATE_NEED_CHECKSUM {
() => {
1i32
};
}

macro_rules! DEFLATE_RESET_HOOK {
() => {
while false {
}
};
}

macro_rules! DO16 {
($buf:expr, $s1:expr, $s2:expr) => {
DO8!($buf, 0i32, $s1, $s2);
DO8!($buf, 8i32, $s1, $s2);
};
}

macro_rules! DO1 {
($buf:expr, $i:expr, $s1:expr, $s2:expr) => {
let __v: u64 = $s1;
let __v2: u64 = __v.wrapping_add(((unsafe { *$buf.wrapping_offset(($i) as isize) }) as i32) as u64);
$s1 = __v2;
let __v3: u64 = $s2;
let __v4: u64 = __v3.wrapping_add($s1);
$s2 = __v4;
};
}

macro_rules! DO2 {
($buf:expr, $i:expr, $s1:expr, $s2:expr) => {
DO1!($buf, $i, $s1, $s2);
DO1!($buf, $i.wrapping_add(1), $s1, $s2);
};
}

macro_rules! DO4 {
($buf:expr, $i:expr, $s1:expr, $s2:expr) => {
DO2!($buf, $i, $s1, $s2);
DO2!($buf, $i.wrapping_add(2), $s1, $s2);
};
}

macro_rules! DO8 {
($buf:expr, $i:expr, $s1:expr, $s2:expr) => {
DO4!($buf, $i, $s1, $s2);
DO4!($buf, $i.wrapping_add(4), $s1, $s2);
};
}

/// Flush the current block, with given end-of-file flag.
/// IN assertion: strstart is set to the end of the current match.
macro_rules! FLUSH_BLOCK_ONLY {
($s:expr, $eof:expr) => {
unsafe { zlib_tr_flush_block($s, if (*$s).block_start >= 0 { unsafe { (*$s).window }.wrapping_offset(((*$s).block_start as u32) as isize) as *mut u8 } else { core::ptr::null_mut::<u8>() }, ((*$s).strstart as i64).wrapping_sub((*$s).block_start) as u64, $eof) };
unsafe {
(*$s).block_start = (*$s).strstart as i64;
}
flush_pending(unsafe { (*$s).strm });
};
}

/// Flush the current block, with given end-of-file flag.
/// IN assertion: strstart is set to the end of the current match.
macro_rules! FLUSH_BLOCK_ONLY__2 {
($s:expr, $eof:expr) => {
unsafe { zlib_tr_flush_block($s, if (*$s).block_start >= 0 { unsafe { (*$s).window }.wrapping_offset(((*$s).block_start as u32) as isize) as *mut u8 } else { core::ptr::null_mut::<u8>() }, ((*$s).strstart as i64).wrapping_sub((*$s).block_start) as u64, ($eof) as i32) };
unsafe {
(*$s).block_start = (*$s).strstart as i64;
}
flush_pending(unsafe { (*$s).strm });
};
}

/// Same but force premature exit if necessary.
macro_rules! FLUSH_BLOCK__2 {
($s:expr, $eof:expr) => {
{
FLUSH_BLOCK_ONLY__2!($s, $eof);
if (unsafe { (*(*$s).strm).avail_out }) == 0 {
return (if $eof { 2i32 } else { 0i32 }) as u32;
}
}
};
}

/// Same but force premature exit if necessary.
macro_rules! FLUSH_BLOCK {
($s:expr, $eof:expr) => {
{
FLUSH_BLOCK_ONLY!($s, $eof);
if (unsafe { (*(*$s).strm).avail_out }) == 0 {
return (if ($eof) != 0 { 2i32 } else { 0i32 }) as u32;
}
}
};
}

/// Insert string str in the dictionary and set match_head to the previous head
/// of the hash chain (the most recent string with same hash key). Return
/// the previous length of the hash chain.
/// IN  assertion: all calls to INSERT_STRING are made with consecutive
///    input characters and the first MIN_MATCH bytes of str are valid
///    (except for the last MIN_MATCH-1 bytes of the input file).
macro_rules! INSERT_STRING {
($s:expr, $str:expr, $match_head:expr) => {
UPDATE_HASH__2!($s, (*$s).ins_h, unsafe { *unsafe { (*$s).window }.wrapping_offset($str.wrapping_add(2u32) as isize) });
let __v: u32 = (unsafe { *unsafe { (*$s).head }.wrapping_offset((*$s).ins_h as isize) }) as u32;
$match_head = __v;
let __v2: u16 = __v as u16;
unsafe {
*unsafe { (*$s).prev }.wrapping_offset((($str) & (*$s).w_mask) as isize) = __v2;
}
let __v3: u16 = ($str) as u16;
unsafe {
*unsafe { (*$s).head }.wrapping_offset((*$s).ins_h as isize) = __v3;
}
};
}

macro_rules! MAX_DIST {
($s:expr) => {
unsafe { (*$s).w_size }.wrapping_sub(262u32)
};
}

/// Update a hash value with the given input byte
/// IN  assertion: all calls to UPDATE_HASH are made with consecutive
///    input characters, so that a running hash key can be computed from the
///    previous key instead of complete recalculation each time.
macro_rules! UPDATE_HASH {
($s:expr, $h:expr, $c:expr) => {
unsafe {
$h = (($h) << (*$s).hash_shift ^ ((($c) as i32) as u32)) & (*$s).hash_mask;
}
};
}

/// Update a hash value with the given input byte
/// IN  assertion: all calls to UPDATE_HASH are made with consecutive
///    input characters, so that a running hash key can be computed from the
///    previous key instead of complete recalculation each time.
macro_rules! UPDATE_HASH__2 {
($s:expr, $h:expr, $c:expr) => {
{
let __v: u32 = ((unsafe { $h }) << unsafe { (*$s).hash_shift } ^ ((($c) as i32) as u32)) & unsafe { (*$s).hash_mask };
unsafe {
$h = __v;
}
__v
}
};
}

/// Output a byte on the stream.
/// IN assertion: there is enough room in pending_buf.
macro_rules! put_byte {
($s:expr, $c:expr) => {
let __v: *mut deflate_state = $s;
let __v2: i32 = unsafe { (*__v).pending };
unsafe {
(*__v).pending = __v2.wrapping_add(1);
}
unsafe {
*unsafe { (*$s).pending_buf }.wrapping_offset(__v2 as isize) = $c;
}
};
}

/// Output a short LSB first on the stream.
/// IN assertion: there is enough room in pendingBuf.
macro_rules! put_short {
($s:expr, $w:expr) => {
put_byte!($s, ((($w) as i32) & 255) as u8);
put_byte!($s, (((($w) as u16) as i32) >> 8i32) as u8);
};
}

macro_rules! unlikely {
($x:expr) => {
!(!$x) as i64
};
}

macro_rules! zlib_deflate_head_memsize {
($memLevel:expr) => {
((1i32 << $memLevel.wrapping_add(7)) as u64).wrapping_mul(core::mem::size_of::<u16>() as u64)
};
}

macro_rules! zlib_deflate_overlay_memsize {
($memLevel:expr) => {
((1i32 << $memLevel.wrapping_add(6)) as u64).wrapping_mul((core::mem::size_of::<u16>() as u64).wrapping_add(2))
};
}

macro_rules! zlib_deflate_prev_memsize {
($windowBits:expr) => {
((1i32 << $windowBits) as u64).wrapping_mul(core::mem::size_of::<u16>() as u64)
};
}

macro_rules! zlib_deflate_window_memsize {
($windowBits:expr) => {
(2i32.wrapping_mul(1i32 << $windowBits) as u64).wrapping_mul(core::mem::size_of::<u8>() as u64)
};
}

unsafe extern "C" {
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn zlib_tr_init(arg0: *mut deflate_state);
    fn zlib_tr_tally(arg0: *mut deflate_state, arg1: u32, arg2: u32) -> i32;
    fn zlib_tr_flush_block(arg0: *mut deflate_state, arg1: *mut u8, arg2: u64, arg3: i32) -> u64;
    fn zlib_tr_align(arg0: *mut deflate_state);
    fn zlib_tr_stored_block(arg0: *mut deflate_state, arg1: *mut u8, arg2: u64, arg3: i32);
    fn zlib_tr_stored_type_only(arg0: *mut deflate_state);
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn __builtin_unreachable();
}

#[repr(C)]
#[derive(Clone, Copy)]
struct internal_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct z_stream_s {
    next_in: *const u8,
    avail_in: u64,
    total_in: u64,
    next_out: *mut u8,
    avail_out: u64,
    total_out: u64,
    msg: *mut u8,
    state: *mut internal_state,
    workspace: *mut (),
    data_type: i32,
    adler: u64,
    reserved: u64,
}


// will be removed, use Z_SYNC_FLUSH instead
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.deflate.zlib_deflate_workspacesize")]
extern "C" fn zlib_deflate_workspacesize(mut windowBits: i32, memLevel: i32) -> i32 {
if windowBits < 0 {
windowBits = windowBits.wrapping_neg();
}
// undocumented feature: suppress zlib header
// Since the return value is typically passed to vmalloc() unchecked...
loop {
if unlikely!(memLevel < 1 || memLevel > 8 || windowBits < 9 || windowBits > 15) != 0 {
loop {
unsafe {
core::arch::asm!("441: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 441b - ., 3; .popsection\n# {0}", const 441 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_zlib_deflate_workspacesize_0, sym __slate_asm_string_zlib_deflate_workspacesize_1, const 1137 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
(core::mem::size_of::<deflate_workspace>() as u64).wrapping_add(zlib_deflate_window_memsize!(windowBits)).wrapping_add(zlib_deflate_prev_memsize!(windowBits)).wrapping_add(zlib_deflate_head_memsize!(memLevel)).wrapping_add(zlib_deflate_overlay_memsize!(memLevel)) as i32
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_defutil_h_51 {
    freq: u16,
    code: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_defutil_h_51_2 {
    dad: u16,
    len: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct static_tree_desc_s {
}


// largest prime smaller than 65536
fn zlib_adler32(adler: u64, mut buf: *const u8, mut len: u32) -> u64 {
let mut s1: u64 = adler & 65535;
let mut s2: u64 = adler >> 16i32 & 65535;
let mut k: i32 = 0;
if buf.is_null() {
return 1;
}
while len > 0 {
k = (if len < 5552 { len } else { 5552u32 }) as i32;
len = len.wrapping_sub(k as u32);
while k >= 16 {
DO16!(buf, s1, s2);
let __v66: *const u8 = buf;
buf = __v66.wrapping_offset(16 as isize);
k = k.wrapping_sub(16);
}
if k != 0 {
loop {
let __v71: *const u8 = buf;
buf = __v71.wrapping_offset(1 as isize);
let __v73: u8 = unsafe { *__v71 };
s1 = s1.wrapping_add((__v73 as i32) as u64);
s2 = s2.wrapping_add(s1);
let __v80: i32 = k.wrapping_sub(1);
k = __v80;
if __v80 == 0 {
break;
}
}
}
s1 %= 65521;
s2 %= 65521;
}
s2 << 16i32 | s1
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ct_data_s {
    fc: __SlateRecord_defutil_h_51,
    dl: __SlateRecord_defutil_h_51_2,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tree_desc_s {
    dyn_tree: *mut ct_data_s,
    max_code: i32,
    stat_desc: *mut static_tree_desc_s,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct deflate_state {
    strm: *mut z_stream_s,
    status: i32,
    pending_buf: *mut u8,
    pending_buf_size: u64,
    pending_out: *mut u8,
    pending: i32,
    noheader: i32,
    data_type: u8,
    method: u8,
    last_flush: i32,
    w_size: u32,
    w_bits: u32,
    w_mask: u32,
    window: *mut u8,
    window_size: u64,
    prev: *mut u16,
    head: *mut u16,
    ins_h: u32,
    hash_size: u32,
    hash_bits: u32,
    hash_mask: u32,
    hash_shift: u32,
    block_start: i64,
    match_length: u32,
    prev_match: u32,
    match_available: i32,
    strstart: u32,
    match_start: u32,
    lookahead: u32,
    prev_length: u32,
    max_chain_length: u32,
    max_lazy_match: u32,
    level: i32,
    strategy: i32,
    good_match: u32,
    nice_match: i32,
    dyn_ltree: [ct_data_s; 573],
    dyn_dtree: [ct_data_s; 61],
    bl_tree: [ct_data_s; 39],
    l_desc: tree_desc_s,
    d_desc: tree_desc_s,
    bl_desc: tree_desc_s,
    bl_count: [u16; 16],
    heap: [i32; 573],
    heap_len: i32,
    heap_max: i32,
    depth: [u8; 573],
    l_buf: *mut u8,
    lit_bufsize: u32,
    last_lit: u32,
    d_buf: *mut u16,
    opt_len: u64,
    static_len: u64,
    compressed_len: u64,
    matches: u32,
    last_eob_len: i32,
    bi_buf: u16,
    bi_valid: i32,
}


#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

fn bi_flush(s: *mut deflate_state) {
if (unsafe { (*s).bi_valid }) == 16 {
put_short!(s, unsafe { (*s).bi_buf });
unsafe {
(*s).bi_buf = 0;
}
unsafe {
(*s).bi_valid = 0;
}
} else {
if (unsafe { (*s).bi_valid }) >= 8 {
put_byte!(s, (unsafe { (*s).bi_buf }) as u8);
unsafe {
(*s).bi_buf = (((*s).bi_buf as i32) >> 8i32) as u16;
}
let __v13: i32 = unsafe { (*s).bi_valid };
unsafe {
(*s).bi_valid = __v13.wrapping_sub(8);
}
}
}
}

fn flush_pending(strm: *mut z_stream_s) {
let mut len: u32 = 0;
let s: *mut deflate_state = (unsafe { (*strm).state }) as *mut deflate_state;
bi_flush(s);
len = (unsafe { (*s).pending }) as u32;
if (len as u64) > unsafe { (*strm).avail_out } {
len = (unsafe { (*strm).avail_out }) as u32;
}
if len == 0 {
return;
}
if !unsafe { (*strm).next_out }.is_null() {
unsafe { memcpy((*strm).next_out as *mut (), (*s).pending_out as *const (), len as u64) };
let __v1: *mut u8 = unsafe { (*strm).next_out };
unsafe {
(*strm).next_out = __v1.wrapping_offset(len as isize);
}
}
let __v4: *mut u8 = unsafe { (*s).pending_out };
unsafe {
(*s).pending_out = __v4.wrapping_offset(len as isize);
}
let __v7: u64 = unsafe { (*strm).total_out };
unsafe {
(*strm).total_out = __v7.wrapping_add(len as u64);
}
let __v10: u64 = unsafe { (*strm).avail_out };
unsafe {
(*strm).avail_out = __v10.wrapping_sub(len as u64);
}
unsafe {
(*s).pending = ((*s).pending as u32).wrapping_sub(len) as i32;
}
if (unsafe { (*s).pending }) == 0 {
unsafe {
(*s).pending_out = (*s).pending_buf;
}
}
}

// ALGORITHM
//
//     The "deflation" process depends on being able to identify portions
//     of the input text which are identical to earlier input (within a
//     sliding window trailing behind the input currently being processed).
//
//     The most straightforward technique turns out to be the fastest for
//     most input files: try all possible matches and select the longest.
//     The key feature of this algorithm is that insertions into the string
//     dictionary are very simple and thus fast, and deletions are avoided
//     completely. Insertions are performed at each input character, whereas
//     string matches are performed only when the previous match ends. So it
//     is preferable to spend more time in matches to allow very fast string
//     insertions and avoid deletions. The matching algorithm for small
//     strings is inspired from that of Rabin & Karp. A brute force approach
//     is used to find longer strings when a small match has been found.
//     A similar algorithm is used in comic (by Jan-Mark Wams) and freeze
//     (by Leonid Broukhis).
//        A previous version of this file used a more sophisticated algorithm
//     (by Fiala and Greene) which is guaranteed to run in linear amortized
//     time, but has a larger average cost, uses more memory and is patented.
//     However the F&G algorithm may be faster for some highly redundant
//     files if the parameter max_chain_length (described below) is too large.
//
// ACKNOWLEDGEMENTS
//
//     The idea of lazy evaluation of matches is due to Jan-Mark Wams, and
//     I found it in 'freeze' written by Leonid Broukhis.
//     Thanks to many people for bug reports and testing.
//
// REFERENCES
//
//     Deutsch, L.P.,"DEFLATE Compressed Data Format Specification".
//     Available in ftp://ds.internic.net/rfc/rfc1951.txt
//
//     A description of the Rabin and Karp algorithm is given in the book
//        "Algorithms" by R. Sedgewick, Addison-Wesley, p252.
//
//     Fiala,E.R., and Greene,D.H.
//        Data Compression with Finite Windows, Comm.ACM, 32,4 (1989) 490-595
// architecture-specific bits
// Function prototypes.
// Compression function. Returns the block state after the call.
// Local data
// Tail of hash chains
// Matches of length 3 are discarded if their distance exceeds TOO_FAR
// Minimum amount of lookahead, except at the end of the input file.
// See deflate.c for comments about the MIN_MATCH+1.
/// Workspace to be allocated for deflate processing
#[repr(C)]
#[derive(Clone, Copy)]
struct deflate_workspace {
    /// State memory for the deflator
    deflate_memory: deflate_state,
    window_memory: *mut u8,
    prev_memory: *mut u16,
    head_memory: *mut u16,
    overlay_memory: *mut u8,
}


/// Values for max_lazy_match, good_match and max_chain_length, depending on
/// the desired pack level (0..9). The values given below have been tuned to
/// exclude worst case performance for pathological files. Better values may be
/// found for specific files.
#[repr(C)]
#[derive(Clone, Copy)]
struct config_s {
    /// reduce lazy search above this match length
    good_length: u16,
    /// do not perform lazy search above this match length
    max_lazy: u16,
    /// quit search above this match length
    nice_length: u16,
    max_chain: u16,
    func: Option<unsafe extern "C" fn(*mut deflate_state, i32) -> u32>,
}


///      good lazy nice chain
///
/// 0
/// store only
/// 1
/// maximum speed, no lazy matches
/// 2
/// 3
/// 4
/// lazy matches
/// 5
/// 6
/// 7
/// 8
/// 9
/// maximum compression
static configuration_table: __SlateAlign16<[config_s; 10]> = __SlateAlign16([config_s { good_length: 0u16, max_lazy: 0u16, nice_length: 0u16, max_chain: 0u16, func: Some(deflate_stored) }, config_s { good_length: 4u16, max_lazy: 4u16, nice_length: 8u16, max_chain: 4u16, func: Some(deflate_fast) }, config_s { good_length: 4u16, max_lazy: 5u16, nice_length: 16u16, max_chain: 8u16, func: Some(deflate_fast) }, config_s { good_length: 4u16, max_lazy: 6u16, nice_length: 32u16, max_chain: 32u16, func: Some(deflate_fast) }, config_s { good_length: 4u16, max_lazy: 4u16, nice_length: 16u16, max_chain: 16u16, func: Some(deflate_slow) }, config_s { good_length: 8u16, max_lazy: 16u16, nice_length: 32u16, max_chain: 32u16, func: Some(deflate_slow) }, config_s { good_length: 8u16, max_lazy: 16u16, nice_length: 128u16, max_chain: 128u16, func: Some(deflate_slow) }, config_s { good_length: 8u16, max_lazy: 32u16, nice_length: 128u16, max_chain: 256u16, func: Some(deflate_slow) }, config_s { good_length: 32u16, max_lazy: 128u16, nice_length: 258u16, max_chain: 1024u16, func: Some(deflate_slow) }, config_s { good_length: 32u16, max_lazy: 258u16, nice_length: 258u16, max_chain: 4096u16, func: Some(deflate_slow) }]);

// Note: the deflate() code requires max_lazy >= MIN_MATCH and max_chain >= 4
// For deflate_fast() (levels <= 3) good is ignored and lazy has a different
// meaning.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.deflate.zlib_deflateInit2")]
extern "C" fn zlib_deflateInit2(strm: *mut z_stream_s, mut level: i32, method: i32, mut windowBits: i32, memLevel: i32, strategy: i32) -> i32 {
let mut noheader: i32 = 0;
let mut mem: *mut deflate_workspace = unsafe { core::mem::zeroed() };
let mut next: *mut u8 = unsafe { core::mem::zeroed() };
// We overlay pending_buf and d_buf+l_buf. This works since the average
// output size for (length,distance) codes is <= 24 bits.
if strm.is_null() {
return 2i32.wrapping_neg();
}
unsafe {
(*strm).msg = core::ptr::null_mut::<u8>();
}
if level == 1i32.wrapping_neg() {
level = 6;
}
mem = (unsafe { (*strm).workspace }) as *mut deflate_workspace;
if windowBits < 0 {
// undocumented feature: suppress zlib header
noheader = 1;
windowBits = windowBits.wrapping_neg();
}
if memLevel < 1 || memLevel > 8 || method != 8 || windowBits < 9 || windowBits > 15 || level < 0 || level > 9 || strategy < 0 || strategy > 2 {
return 2i32.wrapping_neg();
}
// Direct the workspace's pointers to the chunks that were allocated
// along with the deflate_workspace struct.
next = mem as *mut u8;
let __v0: *mut u8 = next;
next = __v0.wrapping_offset(core::mem::size_of::<deflate_workspace>() as isize);
unsafe {
(*mem).window_memory = next as *mut u8;
}
let __v2: *mut u8 = next;
next = __v2.wrapping_offset(zlib_deflate_window_memsize!(windowBits) as isize);
unsafe {
(*mem).prev_memory = next as *mut u16;
}
let __v4: *mut u8 = next;
next = __v4.wrapping_offset(zlib_deflate_prev_memsize!(windowBits) as isize);
unsafe {
(*mem).head_memory = next as *mut u16;
}
let __v6: *mut u8 = next;
next = __v6.wrapping_offset(zlib_deflate_head_memsize!(memLevel) as isize);
unsafe {
(*mem).overlay_memory = next;
}
let s: *mut deflate_state = unsafe { core::ptr::addr_of_mut!((*mem).deflate_memory) };
unsafe {
(*strm).state = s as *mut internal_state;
}
unsafe {
(*s).strm = strm;
}
unsafe {
(*s).noheader = noheader;
}
unsafe {
(*s).w_bits = windowBits as u32;
}
unsafe {
(*s).w_size = (1i32 << (*s).w_bits) as u32;
}
unsafe {
(*s).w_mask = unsafe { (*s).w_size }.wrapping_sub(1);
}
unsafe {
(*s).hash_bits = memLevel.wrapping_add(7) as u32;
}
unsafe {
(*s).hash_size = (1i32 << (*s).hash_bits) as u32;
}
unsafe {
(*s).hash_mask = unsafe { (*s).hash_size }.wrapping_sub(1);
}
unsafe {
(*s).hash_shift = unsafe { (*s).hash_bits }.wrapping_add(3).wrapping_sub(1) / 3;
}
unsafe {
(*s).window = (*mem).window_memory;
}
unsafe {
(*s).prev = (*mem).prev_memory;
}
unsafe {
(*s).head = (*mem).head_memory;
}
unsafe {
(*s).lit_bufsize = (1i32 << memLevel.wrapping_add(6)) as u32;
}
// 16K elements by default
let overlay: *mut u16 = (unsafe { (*mem).overlay_memory }) as *mut u16;
unsafe {
(*s).pending_buf = overlay as *mut u8;
}
unsafe {
(*s).pending_buf_size = ((*s).lit_bufsize as u64).wrapping_mul((core::mem::size_of::<u16>() as u64).wrapping_add(2));
}
unsafe {
(*s).d_buf = overlay.wrapping_offset((((*s).lit_bufsize as u64) / (core::mem::size_of::<u16>() as u64)) as isize);
}
unsafe {
(*s).l_buf = unsafe { (*s).pending_buf }.wrapping_offset(1u64.wrapping_add(core::mem::size_of::<u16>() as u64).wrapping_mul((*s).lit_bufsize as u64) as isize);
}
unsafe {
(*s).level = level;
}
unsafe {
(*s).strategy = strategy;
}
unsafe {
(*s).method = method as u8;
}
zlib_deflateReset(strm)
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.deflate.zlib_deflateReset")]
extern "C" fn zlib_deflateReset(strm: *mut z_stream_s) -> i32 {
if strm.is_null() || unsafe { (*strm).state }.is_null() {
return 2i32.wrapping_neg();
}
unsafe {
(*strm).total_out = 0;
}
unsafe {
(*strm).total_in = 0;
}
unsafe {
(*strm).msg = core::ptr::null_mut::<u8>();
}
unsafe {
(*strm).data_type = 2;
}
let s: *mut deflate_state = (unsafe { (*strm).state }) as *mut deflate_state;
unsafe {
(*s).pending = 0;
}
unsafe {
(*s).pending_out = (*s).pending_buf;
}
if (unsafe { (*s).noheader }) < 0 {
unsafe {
(*s).noheader = 0;
}
// was set to -1 by deflate(..., Z_FINISH);
}
unsafe {
(*s).status = if (*s).noheader != 0 { 113 } else { 42 };
}
unsafe {
(*strm).adler = 1;
}
unsafe {
(*s).last_flush = 0;
}
unsafe { zlib_tr_init(s) };
lm_init(s);
DEFLATE_RESET_HOOK!();
0
}

/// Put a short in the pending buffer. The 16-bit value is put in MSB order.
/// IN assertion: the stream state is correct and there is enough room in
/// pending_buf.
fn putShortMSB(s: *mut deflate_state, b: u32) {
put_byte!(s, (b >> 8i32) as u8);
put_byte!(s, (b & 255) as u8);
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.deflate.zlib_deflate")]
extern "C" fn zlib_deflate(strm: *mut z_stream_s, flush: i32) -> i32 {
let mut old_flush: i32 = 0; // value of flush param for previous deflate call
if strm.is_null() || unsafe { (*strm).state }.is_null() || flush > 5 || flush < 0 {
return 2i32.wrapping_neg();
}
let s: *mut deflate_state = (unsafe { (*strm).state }) as *mut deflate_state;
if unsafe { (*strm).next_in }.is_null() && (unsafe { (*strm).avail_in }) != 0 || (unsafe { (*s).status }) == 666 && flush != 5 {
return 2i32.wrapping_neg();
}
if (unsafe { (*strm).avail_out }) == 0 {
return 5i32.wrapping_neg();
}
unsafe {
(*s).strm = strm;
}
// just in case
old_flush = unsafe { (*s).last_flush };
unsafe {
(*s).last_flush = flush;
}
// Write the zlib header
if (unsafe { (*s).status }) == 42 {
let mut header: u32 = 8u32.wrapping_add(unsafe { (*s).w_bits }.wrapping_sub(8) << 4i32) << 8i32;
let mut level_flags: u32 = (unsafe { (*s).level }.wrapping_sub(1) >> 1i32) as u32;
if level_flags > 3 {
level_flags = 3;
}
header |= level_flags << 6i32;
if (unsafe { (*s).strstart }) != 0 {
header |= 32;
}
header = header.wrapping_add(31u32.wrapping_sub(header % 31));
unsafe {
(*s).status = 113;
}
putShortMSB(s, header);
// Save the adler32 of the preset dictionary:
if (unsafe { (*s).strstart }) != 0 {
putShortMSB(s, ((unsafe { (*strm).adler }) >> 16i32) as u32);
putShortMSB(s, ((unsafe { (*strm).adler }) & 65535) as u32);
}
unsafe {
(*strm).adler = 1;
}
}
// Flush as much pending output as possible
if (unsafe { (*s).pending }) != 0 {
flush_pending(strm);
if (unsafe { (*strm).avail_out }) == 0 {
// Since avail_out is 0, deflate will be called again with
// more output space, but possibly with both pending and
// avail_in equal to zero. There won't be anything to do,
// but this is not an error situation so make sure we
// return OK instead of BUF_ERROR at next call of deflate:
unsafe {
(*s).last_flush = 1i32.wrapping_neg();
}
return 0;
}
// Make sure there is something to do and avoid duplicate consecutive
// flushes. For repeated and useless calls with Z_FINISH, we keep
// returning Z_STREAM_END instead of Z_BUFF_ERROR.
} else {
if (unsafe { (*strm).avail_in }) == 0 && flush <= old_flush && flush != 5 {
return 5i32.wrapping_neg();
}
}
// User must not provide more input after the first FINISH:
if (unsafe { (*s).status }) == 666 && (unsafe { (*strm).avail_in }) != 0 {
return 5i32.wrapping_neg();
}
// Start a new block or continue the current one.
if (unsafe { (*strm).avail_in }) != 0 || (unsafe { (*s).lookahead }) != 0 || flush != 0 && (unsafe { (*s).status }) != 666 {
let mut bstate: u32 = 0;
let __v6: u32 = if DEFLATE_HOOK!() != 0 {
bstate as u32
} else {
(unsafe { unsafe { configuration_table.0[(*s).level as usize].func }.unwrap()(s, flush) }) as u32
};
bstate = __v6 as u32;
if (bstate as u32) == 2 || (bstate as u32) == 3 {
unsafe {
(*s).status = 666;
}
}
if (bstate as u32) == 0 || (bstate as u32) == 2 {
if (unsafe { (*strm).avail_out }) == 0 {
unsafe {
(*s).last_flush = 1i32.wrapping_neg();
}
// avoid BUF_ERROR next call, see above
}
return 0;
// If flush != Z_NO_FLUSH && avail_out == 0, the next call
// of deflate should use the same flush parameter to make sure
// that the flush is complete. So we don't have to output an
// empty block here, this will be done at next call. This also
// ensures that for a very small output buffer, we emit at most
// one empty block.
}
if (bstate as u32) == 1 {
if flush == 1 {
unsafe { zlib_tr_align(s) };
} else {
if flush == 2 {
// Output just the 3-bit `stored' block type value,
// but not a zero length.
unsafe { zlib_tr_stored_type_only(s) };
} else {
// FULL_FLUSH or SYNC_FLUSH
unsafe { zlib_tr_stored_block(s, core::ptr::null_mut::<u8>(), 0u64, 0i32) };
// For a full flush, this empty block will be recognized
// as a special marker by inflate_sync().
if flush == 4 {
CLEAR_HASH!(s);
// forget history
}
}
}
flush_pending(strm);
if (unsafe { (*strm).avail_out }) == 0 {
unsafe {
(*s).last_flush = 1i32.wrapping_neg();
}
// avoid BUF_ERROR at next call, see above
return 0;
}
}
}
if flush != 5 {
return 0;
}
if (unsafe { (*s).noheader }) == 0 {
// Write zlib trailer (adler32)
putShortMSB(s, ((unsafe { (*strm).adler }) >> 16i32) as u32);
putShortMSB(s, ((unsafe { (*strm).adler }) & 65535) as u32);
}
flush_pending(strm);
// If avail_out is zero, the application will call deflate again
// to flush the rest.
if (unsafe { (*s).noheader }) == 0 {
unsafe {
(*s).noheader = 1i32.wrapping_neg();
}
// write the trailer only once!
}
if (unsafe { (*s).pending }) == 0 {
return 1;
}
0
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.deflate.zlib_deflateEnd")]
extern "C" fn zlib_deflateEnd(strm: *mut z_stream_s) -> i32 {
let mut status: i32 = 0;
if strm.is_null() || unsafe { (*strm).state }.is_null() {
return 2i32.wrapping_neg();
}
let s: *mut deflate_state = (unsafe { (*strm).state }) as *mut deflate_state;
status = unsafe { (*s).status };
if status != 42 && status != 113 && status != 666 {
return 2i32.wrapping_neg();
}
unsafe {
(*strm).state = core::ptr::null_mut::<internal_state>();
}
if status == 113 { 3i32.wrapping_neg() } else { 0 }
}

/// Read a new buffer from the current input stream, update the adler32
/// and total number of bytes read.  All deflate() input goes through
/// this function so some applications may wish to modify it to avoid
/// allocating a large strm->next_in buffer and copying from it.
/// (See also flush_pending()).
fn read_buf(strm: *mut z_stream_s, buf: *mut u8, size: u32) -> i32 {
let mut len: u32 = (unsafe { (*strm).avail_in }) as u32;
if len > size {
len = size;
}
if len == 0 {
return 0;
}
let __v1: u64 = unsafe { (*strm).avail_in };
unsafe {
(*strm).avail_in = __v1.wrapping_sub(len as u64);
}
if DEFLATE_NEED_CHECKSUM!() != 0 {
if (unsafe { (*((*strm).state as *mut deflate_state)).noheader }) == 0 {
unsafe {
(*strm).adler = zlib_adler32((*strm).adler, (*strm).next_in, len);
}
}
}
unsafe { memcpy(buf as *mut (), (*strm).next_in as *const (), len as u64) };
let __v4: *const u8 = unsafe { (*strm).next_in };
unsafe {
(*strm).next_in = __v4.wrapping_offset(len as isize);
}
let __v7: u64 = unsafe { (*strm).total_in };
unsafe {
(*strm).total_in = __v7.wrapping_add(len as u64);
}
len as i32
}

/// Initialize the "longest match" routines for a new zlib stream
fn lm_init(s: *mut deflate_state) {
unsafe {
(*s).window_size = 2u64.wrapping_mul((*s).w_size as u64);
}
CLEAR_HASH!(s);
// Set the default configuration parameters:
unsafe {
(*s).max_lazy_match = configuration_table.0[(*s).level as usize].max_lazy as u32;
}
unsafe {
(*s).good_match = configuration_table.0[(*s).level as usize].good_length as u32;
}
unsafe {
(*s).nice_match = configuration_table.0[(*s).level as usize].nice_length as i32;
}
unsafe {
(*s).max_chain_length = configuration_table.0[(*s).level as usize].max_chain as u32;
}
unsafe {
(*s).strstart = 0;
}
unsafe {
(*s).block_start = 0;
}
unsafe {
(*s).lookahead = 0;
}
let __v0: u32 = 2u32;
unsafe {
(*s).prev_length = __v0;
}
unsafe {
(*s).match_length = __v0;
}
unsafe {
(*s).match_available = 0;
}
unsafe {
(*s).ins_h = 0;
}
}

/// Set match_start to the longest match starting at the given string and
/// return its length. Matches shorter or equal to prev_length are discarded,
/// in which case the result is equal to prev_length and match_start is
/// garbage.
/// IN assertions: cur_match is the head of the hash chain for the current
///   string (strstart) and its distance is <= MAX_DIST, and prev_length >= 1
/// OUT assertion: the match length is not greater than s->lookahead.
///
/// For 80x86 and 680x0, an optimized version will be provided in match.asm or
/// match.S. The code will be functionally equivalent.
///
/// # Arguments
///
/// * `cur_match` - current match
fn longest_match(s: *mut deflate_state, mut cur_match: u32) -> u32 {
let mut chain_length: u32 = unsafe { (*s).max_chain_length }; // max hash chain length
let mut scan: *mut u8 = unsafe { (*s).window }.wrapping_offset((unsafe { (*s).strstart }) as isize); // current string
let mut r#match: *mut u8 = unsafe { core::mem::zeroed() }; // matched string
let mut len: i32 = 0; // length of current match
let mut best_len: i32 = (unsafe { (*s).prev_length }) as i32; // best match length so far
let mut nice_match: i32 = unsafe { (*s).nice_match }; // stop if match long enough
let limit: u32 = if (unsafe { (*s).strstart }) > MAX_DIST!(s) { unsafe { (*s).strstart }.wrapping_sub(MAX_DIST!(s)) } else { 0 };
// Stop when cur_match becomes <= limit. To simplify the code,
// we prevent matches with the string of window index 0.
let prev: *mut u16 = unsafe { (*s).prev };
let wmask: u32 = unsafe { (*s).w_mask };
let strend: *mut u8 = unsafe { (*s).window }.wrapping_offset((unsafe { (*s).strstart }) as isize).wrapping_offset(258 as isize);
let mut scan_end1: u8 = unsafe { *scan.wrapping_offset(best_len.wrapping_sub(1) as isize) };
let mut scan_end: u8 = unsafe { *scan.wrapping_offset(best_len as isize) };
// The code is optimized for HASH_BITS >= 8 and MAX_MATCH-2 multiple of 16.
// It is easy to get rid of this optimization if necessary.
// Do not waste too much time if we already have a good match:
if (unsafe { (*s).prev_length }) >= unsafe { (*s).good_match } {
chain_length >>= 2i32;
}
// Do not look for matches beyond the end of the input. This is necessary
// to make deflate deterministic.
if (nice_match as u32) > unsafe { (*s).lookahead } {
nice_match = (unsafe { (*s).lookahead }) as i32;
}
loop {
r#match = unsafe { (*s).window }.wrapping_offset(cur_match as isize);
// Skip to next match if the match length cannot increase
// or if the match length is less than 2:
let __v2: bool = if ((unsafe { *r#match.wrapping_offset(best_len as isize) }) as i32) != (scan_end as i32) || ((unsafe { *r#match.wrapping_offset(best_len.wrapping_sub(1) as isize) }) as i32) != (scan_end1 as i32) || ((unsafe { *r#match }) as i32) != ((unsafe { *scan }) as i32) {
true
} else {
let __v3: *mut u8 = r#match;
let __v4: *mut u8 = __v3.wrapping_offset(1 as isize);
r#match = __v4;
((unsafe { *__v4 }) as i32) != ((unsafe { *scan.wrapping_offset(1 as isize) }) as i32)
};
if !__v2 {
// The check at best_len-1 can be removed because it will be made
// again later. (This heuristic is not always a win.)
// It is not necessary to compare scan[2] and match[2] since they
// are always equal when the other bytes match, given that
// the hash keys are equal and that HASH_BITS >= 8.
let __v6: *mut u8 = scan;
scan = __v6.wrapping_offset(2 as isize);
let __v8: *mut u8 = r#match;
r#match = __v8.wrapping_offset(1 as isize);
// We check for insufficient lookahead only every 8th comparison;
// the 256th check will be made at strstart+258.
loop {
let __v11: *mut u8 = scan;
let __v12: *mut u8 = __v11.wrapping_offset(1 as isize);
scan = __v12;
let __v13: *mut u8 = r#match;
let __v14: *mut u8 = __v13.wrapping_offset(1 as isize);
r#match = __v14;
let __v15: bool = if ((unsafe { *__v12 }) as i32) == ((unsafe { *__v14 }) as i32) {
let __v16: *mut u8 = scan;
let __v17: *mut u8 = __v16.wrapping_offset(1 as isize);
scan = __v17;
let __v18: *mut u8 = r#match;
let __v19: *mut u8 = __v18.wrapping_offset(1 as isize);
r#match = __v19;
((unsafe { *__v17 }) as i32) == ((unsafe { *__v19 }) as i32)
} else {
false
};
let __v20: bool = if __v15 {
let __v21: *mut u8 = scan;
let __v22: *mut u8 = __v21.wrapping_offset(1 as isize);
scan = __v22;
let __v23: *mut u8 = r#match;
let __v24: *mut u8 = __v23.wrapping_offset(1 as isize);
r#match = __v24;
((unsafe { *__v22 }) as i32) == ((unsafe { *__v24 }) as i32)
} else {
false
};
let __v25: bool = if __v20 {
let __v26: *mut u8 = scan;
let __v27: *mut u8 = __v26.wrapping_offset(1 as isize);
scan = __v27;
let __v28: *mut u8 = r#match;
let __v29: *mut u8 = __v28.wrapping_offset(1 as isize);
r#match = __v29;
((unsafe { *__v27 }) as i32) == ((unsafe { *__v29 }) as i32)
} else {
false
};
let __v30: bool = if __v25 {
let __v31: *mut u8 = scan;
let __v32: *mut u8 = __v31.wrapping_offset(1 as isize);
scan = __v32;
let __v33: *mut u8 = r#match;
let __v34: *mut u8 = __v33.wrapping_offset(1 as isize);
r#match = __v34;
((unsafe { *__v32 }) as i32) == ((unsafe { *__v34 }) as i32)
} else {
false
};
let __v35: bool = if __v30 {
let __v36: *mut u8 = scan;
let __v37: *mut u8 = __v36.wrapping_offset(1 as isize);
scan = __v37;
let __v38: *mut u8 = r#match;
let __v39: *mut u8 = __v38.wrapping_offset(1 as isize);
r#match = __v39;
((unsafe { *__v37 }) as i32) == ((unsafe { *__v39 }) as i32)
} else {
false
};
let __v40: bool = if __v35 {
let __v41: *mut u8 = scan;
let __v42: *mut u8 = __v41.wrapping_offset(1 as isize);
scan = __v42;
let __v43: *mut u8 = r#match;
let __v44: *mut u8 = __v43.wrapping_offset(1 as isize);
r#match = __v44;
((unsafe { *__v42 }) as i32) == ((unsafe { *__v44 }) as i32)
} else {
false
};
let __v45: bool = if __v40 {
let __v46: *mut u8 = scan;
let __v47: *mut u8 = __v46.wrapping_offset(1 as isize);
scan = __v47;
let __v48: *mut u8 = r#match;
let __v49: *mut u8 = __v48.wrapping_offset(1 as isize);
r#match = __v49;
((unsafe { *__v47 }) as i32) == ((unsafe { *__v49 }) as i32)
} else {
false
};
if !(__v45 && scan < strend) {
break;
}
}
len = 258i32.wrapping_sub(((unsafe { strend.offset_from(scan as *mut u8) }) as i64) as i32);
scan = strend.wrapping_offset(-(258 as isize));
if len > best_len {
unsafe {
(*s).match_start = cur_match;
}
best_len = len;
if len >= nice_match {
break;
}
scan_end1 = unsafe { *scan.wrapping_offset(best_len.wrapping_sub(1) as isize) };
scan_end = unsafe { *scan.wrapping_offset(best_len as isize) };
}
}
let __v50: u32 = (unsafe { *prev.wrapping_offset((cur_match & wmask) as isize) }) as u32;
cur_match = __v50;
let __v51: bool = if __v50 > limit {
let __v53: u32 = chain_length.wrapping_sub(1);
chain_length = __v53;
__v53 != 0
} else {
false
};
if !__v51 {
break;
}
}
if (best_len as u32) <= unsafe { (*s).lookahead } {
return best_len as u32;
}
unsafe { (*s).lookahead }
}

/// Fill the window when the lookahead becomes insufficient.
/// Updates strstart and lookahead.
///
/// IN assertion: lookahead < MIN_LOOKAHEAD
/// OUT assertions: strstart <= window_size-MIN_LOOKAHEAD
///    At least one byte has been read, or avail_in == 0; reads are
///    performed for at least two bytes (required for the zip translate_eol
///    option -- not supported here).
fn fill_window(s: *mut deflate_state) {
let mut n: u32 = 0;
let mut m: u32 = 0;
let mut p: *mut u16 = unsafe { core::mem::zeroed() };
let mut more: u32 = 0; // Amount of free space at the end of the window.
let wsize: u32 = unsafe { (*s).w_size };
loop {
more = unsafe { (*s).window_size }.wrapping_sub((unsafe { (*s).lookahead }) as u64).wrapping_sub((unsafe { (*s).strstart }) as u64) as u32;
// Deal with !@#$% 64K limit:
if more == 0 && (unsafe { (*s).strstart }) == 0 && (unsafe { (*s).lookahead }) == 0 {
more = wsize;
} else {
if more == (1i32.wrapping_neg() as u32) {
// Very unlikely, but possible on 16 bit machine if strstart == 0
// and lookahead == 1 (input done one byte at time)
more = more.wrapping_sub(1);
// If the window is almost full and there is insufficient lookahead,
// move the upper half to the lower one to make room in the upper half.
} else {
if (unsafe { (*s).strstart }) >= wsize.wrapping_add(MAX_DIST!(s)) {
unsafe { memcpy(((*s).window as *mut u8) as *mut (), ((*s).window as *mut u8).wrapping_offset(wsize as isize) as *const (), wsize as u64) };
let __v3: u32 = unsafe { (*s).match_start };
unsafe {
(*s).match_start = __v3.wrapping_sub(wsize);
}
let __v6: u32 = unsafe { (*s).strstart };
unsafe {
(*s).strstart = __v6.wrapping_sub(wsize);
}
// we now have strstart >= MAX_DIST
let __v9: i64 = unsafe { (*s).block_start };
unsafe {
(*s).block_start = __v9.wrapping_sub(wsize as i64);
}
// Slide the hash table (could be avoided with 32 bit values
// at the expense of memory usage). We slide even when level == 0
// to keep the hash table consistent if we switch back to level > 0
// later. (Using level 0 permanently is not an optimal usage of
// zlib, so we don't care about this pathological case.)
n = unsafe { (*s).hash_size };
p = unsafe { (*s).head }.wrapping_offset(n as isize);
loop {
let __v11: *mut u16 = p;
let __v12: *mut u16 = __v11.wrapping_offset(-(1 as isize));
p = __v12;
m = (unsafe { *__v12 }) as u32;
unsafe {
*p = (if m >= wsize { m.wrapping_sub(wsize) } else { 0u32 }) as u16;
}
let __v15: u32 = n.wrapping_sub(1);
n = __v15;
if __v15 == 0 {
break;
}
}
n = wsize;
p = unsafe { (*s).prev }.wrapping_offset(n as isize);
loop {
let __v16: *mut u16 = p;
let __v17: *mut u16 = __v16.wrapping_offset(-(1 as isize));
p = __v17;
m = (unsafe { *__v17 }) as u32;
unsafe {
*p = (if m >= wsize { m.wrapping_sub(wsize) } else { 0u32 }) as u16;
}
// If n is not on any hash chain, prev[n] is garbage but
// its value will never be used.
let __v20: u32 = n.wrapping_sub(1);
n = __v20;
if __v20 == 0 {
break;
}
}
more = more.wrapping_add(wsize);
}
}
}
if (unsafe { (*(*s).strm).avail_in }) == 0 {
return;
}
// If there was no sliding:
//    strstart <= WSIZE+MAX_DIST-1 && lookahead <= MIN_LOOKAHEAD - 1 &&
//    more == window_size - lookahead - strstart
// => more >= window_size - (MIN_LOOKAHEAD-1 + WSIZE + MAX_DIST-1)
// => more >= window_size - 2*WSIZE + 2
// In the BIG_MEM or MMAP case (not yet supported),
//   window_size == input_size + MIN_LOOKAHEAD  &&
//   strstart + s->lookahead <= input_size => more >= MIN_LOOKAHEAD.
// Otherwise, window_size == 2*WSIZE so more >= 2.
// If there was sliding, more >= WSIZE. So in all cases, more >= 2.
n = read_buf(unsafe { (*s).strm }, unsafe { (*s).window }.wrapping_offset((unsafe { (*s).strstart }) as isize).wrapping_offset((unsafe { (*s).lookahead }) as isize), more) as u32;
let __v24: u32 = unsafe { (*s).lookahead };
unsafe {
(*s).lookahead = __v24.wrapping_add(n);
}
// Initialize the hash value now that we have some input:
if (unsafe { (*s).lookahead }) >= 3 {
unsafe {
(*s).ins_h = *unsafe { (*s).window }.wrapping_offset((*s).strstart as isize) as u32;
}
UPDATE_HASH!(s, (*s).ins_h, unsafe { *unsafe { (*s).window }.wrapping_offset(unsafe { (*s).strstart }.wrapping_add(1) as isize) });
}
// If the whole input has less than MIN_MATCH bytes, ins_h is garbage,
// but this is not important since only literal bytes will be emitted.
if !((unsafe { (*s).lookahead }) < 262u32 && (unsafe { (*(*s).strm).avail_in }) != 0) {
break;
}
}
}

/// Copy without compression as much as possible from the input stream, return
/// the current block state.
/// This function does not insert new strings in the dictionary since
/// uncompressible data is probably not useful. This function is used
/// only for the level=0 compression option.
/// NOTE: this function should be optimized to avoid extra copying from
/// window to pending_buf.
#[unsafe(link_section = ".text.slate_distinct.deflate.deflate_stored")]
extern "C" fn deflate_stored(s: *mut deflate_state, flush: i32) -> u32 {
// Stored blocks are limited to 0xffff bytes, pending_buf is limited
// to pending_buf_size, and each stored block has a 5 byte header:
let mut max_block_size: u64 = 65535;
let mut max_start: u64 = 0;
if max_block_size > unsafe { (*s).pending_buf_size }.wrapping_sub(5) {
max_block_size = unsafe { (*s).pending_buf_size }.wrapping_sub(5);
}
// Copy as much as possible from input to output:
loop {
// Fill the window as much as possible:
if (unsafe { (*s).lookahead }) <= 1 {
fill_window(s);
if (unsafe { (*s).lookahead }) == 0 && flush == 0 {
return 0;
}
if (unsafe { (*s).lookahead }) == 0 {
break;
}
// flush the current block
}
let __v2: u32 = unsafe { (*s).strstart };
unsafe {
(*s).strstart = __v2.wrapping_add((*s).lookahead);
}
unsafe {
(*s).lookahead = 0;
}
// Emit a stored block if pending_buf will be full:
max_start = ((unsafe { (*s).block_start }) as u64).wrapping_add(max_block_size);
if (unsafe { (*s).strstart }) == 0 || ((unsafe { (*s).strstart }) as u64) >= max_start {
// strstart == 0 is possible when wraparound on 16-bit machine
unsafe {
(*s).lookahead = ((*s).strstart as u64).wrapping_sub(max_start) as u32;
}
unsafe {
(*s).strstart = max_start as u32;
}
FLUSH_BLOCK!(s, 0i32);
}
// Flush if we may have to slide, otherwise block_start may become
// negative and the data will be gone:
if unsafe { (*s).strstart }.wrapping_sub((unsafe { (*s).block_start }) as u32) >= MAX_DIST!(s) {
FLUSH_BLOCK!(s, 0i32);
}
}
FLUSH_BLOCK__2!(s, flush == 5);
(if flush == 5 { 3i32 } else { 1i32 }) as u32
}

/// Compress as much as possible from the input stream, return the current
/// block state.
/// This function does not perform lazy evaluation of matches and inserts
/// new strings in the dictionary only for unmatched strings or for short
/// matches. It is used only for the fast compression options.
#[unsafe(link_section = ".text.slate_distinct.deflate.deflate_fast")]
extern "C" fn deflate_fast(s: *mut deflate_state, flush: i32) -> u32 {
let mut hash_head: u32 = 0; // head of the hash chain
let mut bflush: i32 = 0; // set if current block must be flushed
loop {
// Make sure that we always have enough lookahead, except
// at the end of the input file. We need MAX_MATCH bytes
// for the next match, plus MIN_MATCH bytes to insert the
// string following the next match.
if (unsafe { (*s).lookahead }) < 262u32 {
fill_window(s);
if (unsafe { (*s).lookahead }) < 262u32 && flush == 0 {
return 0;
}
if (unsafe { (*s).lookahead }) == 0 {
break;
}
// flush the current block
}
// Insert the string window[strstart .. strstart+2] in the
// dictionary, and set hash_head to the head of the hash chain:
if (unsafe { (*s).lookahead }) >= 3 {
INSERT_STRING!(s, unsafe { (*s).strstart }, hash_head);
}
// Find the longest match, discarding those <= prev_length.
// At this point we have always match_length < MIN_MATCH
if hash_head != 0 && unsafe { (*s).strstart }.wrapping_sub(hash_head) <= MAX_DIST!(s) {
// To simplify the code, we prevent matches with the string
// of window index 0 (in particular we have to avoid a match
// of the string with itself at the start of the input file).
if (unsafe { (*s).strategy }) != 2 {
unsafe {
(*s).match_length = longest_match(s, hash_head);
}
}
// longest_match() sets match_start
}
if (unsafe { (*s).match_length }) >= 3 {
bflush = unsafe { zlib_tr_tally(s, unsafe { (*s).strstart }.wrapping_sub((*s).match_start), unsafe { (*s).match_length }.wrapping_sub(3)) };
let __v6: u32 = unsafe { (*s).lookahead };
unsafe {
(*s).lookahead = __v6.wrapping_sub((*s).match_length);
}
// Insert new strings in the hash table only if the match length
// is not too large. This saves time but degrades compression.
if (unsafe { (*s).match_length }) <= unsafe { (*s).max_lazy_match } && (unsafe { (*s).lookahead }) >= 3 {
let __v9: u32 = unsafe { (*s).match_length };
unsafe {
(*s).match_length = __v9.wrapping_sub(1);
}
// string at strstart already in hash table
loop {
let __v12: u32 = unsafe { (*s).strstart };
unsafe {
(*s).strstart = __v12.wrapping_add(1);
}
INSERT_STRING!(s, unsafe { (*s).strstart }, hash_head);
// strstart never exceeds WSIZE-MAX_MATCH, so there are
// always MIN_MATCH bytes ahead.
let __v20: u32 = unsafe { (*s).match_length };
let __v21: u32 = __v20.wrapping_sub(1);
unsafe {
(*s).match_length = __v21;
}
if __v21 == 0 {
break;
}
}
let __v23: u32 = unsafe { (*s).strstart };
unsafe {
(*s).strstart = __v23.wrapping_add(1);
}
} else {
let __v26: u32 = unsafe { (*s).strstart };
unsafe {
(*s).strstart = __v26.wrapping_add((*s).match_length);
}
unsafe {
(*s).match_length = 0;
}
unsafe {
(*s).ins_h = *unsafe { (*s).window }.wrapping_offset((*s).strstart as isize) as u32;
}
UPDATE_HASH!(s, (*s).ins_h, unsafe { *unsafe { (*s).window }.wrapping_offset(unsafe { (*s).strstart }.wrapping_add(1) as isize) });
// If lookahead < MIN_MATCH, ins_h is garbage, but it does not
// matter since it will be recomputed at next deflate call.
}
} else {
// No match, output a literal byte
bflush = unsafe { zlib_tr_tally(s, 0u32, *unsafe { (*s).window }.wrapping_offset((*s).strstart as isize) as u32) };
let __v29: u32 = unsafe { (*s).lookahead };
unsafe {
(*s).lookahead = __v29.wrapping_sub(1);
}
let __v32: u32 = unsafe { (*s).strstart };
unsafe {
(*s).strstart = __v32.wrapping_add(1);
}
}
if bflush != 0 {
FLUSH_BLOCK!(s, 0i32);
}
}
FLUSH_BLOCK__2!(s, flush == 5);
(if flush == 5 { 3i32 } else { 1i32 }) as u32
}

/// Same as above, but achieves better compression. We use a lazy
/// evaluation for matches: a match is finally adopted only if there is
/// no better match at the next window position.
#[unsafe(link_section = ".text.slate_distinct.deflate.deflate_slow")]
extern "C" fn deflate_slow(s: *mut deflate_state, flush: i32) -> u32 {
let mut hash_head: u32 = 0; // head of hash chain
let mut bflush: i32 = 0; // set if current block must be flushed
// Process the input block.
loop {
// Make sure that we always have enough lookahead, except
// at the end of the input file. We need MAX_MATCH bytes
// for the next match, plus MIN_MATCH bytes to insert the
// string following the next match.
if (unsafe { (*s).lookahead }) < 262u32 {
fill_window(s);
if (unsafe { (*s).lookahead }) < 262u32 && flush == 0 {
return 0;
}
if (unsafe { (*s).lookahead }) == 0 {
break;
}
// flush the current block
}
// Insert the string window[strstart .. strstart+2] in the
// dictionary, and set hash_head to the head of the hash chain:
if (unsafe { (*s).lookahead }) >= 3 {
INSERT_STRING!(s, unsafe { (*s).strstart }, hash_head);
}
// Find the longest match, discarding those <= prev_length.
unsafe {
(*s).prev_length = (*s).match_length;
}
unsafe {
(*s).prev_match = (*s).match_start;
}
unsafe {
(*s).match_length = 2u32;
}
if hash_head != 0 && (unsafe { (*s).prev_length }) < unsafe { (*s).max_lazy_match } && unsafe { (*s).strstart }.wrapping_sub(hash_head) <= MAX_DIST!(s) {
// To simplify the code, we prevent matches with the string
// of window index 0 (in particular we have to avoid a match
// of the string with itself at the start of the input file).
if (unsafe { (*s).strategy }) != 2 {
unsafe {
(*s).match_length = longest_match(s, hash_head);
}
}
// longest_match() sets match_start
if (unsafe { (*s).match_length }) <= 5 && ((unsafe { (*s).strategy }) == 1 || (unsafe { (*s).match_length }) == 3 && unsafe { (*s).strstart }.wrapping_sub(unsafe { (*s).match_start }) > 4096) {
// If prev_match is also MIN_MATCH, match_start is garbage
// but we will ignore the current match anyway.
unsafe {
(*s).match_length = 2u32;
}
}
}
// If there was a match at the previous step and the current
// match is not better, output the previous match:
if (unsafe { (*s).prev_length }) >= 3 && (unsafe { (*s).match_length }) <= unsafe { (*s).prev_length } {
let max_insert: u32 = unsafe { (*s).strstart }.wrapping_add(unsafe { (*s).lookahead }).wrapping_sub(3);
// Do not insert strings in hash table beyond this.
bflush = unsafe { zlib_tr_tally(s, unsafe { (*s).strstart }.wrapping_sub(1).wrapping_sub((*s).prev_match), unsafe { (*s).prev_length }.wrapping_sub(3)) };
// Insert in hash table all strings up to the end of the match.
// strstart-1 and strstart are already inserted. If there is not
// enough lookahead, the last two strings are not inserted in
// the hash table.
let __v7: u32 = unsafe { (*s).lookahead };
unsafe {
(*s).lookahead = __v7.wrapping_sub(unsafe { (*s).prev_length }.wrapping_sub(1));
}
let __v10: u32 = unsafe { (*s).prev_length };
unsafe {
(*s).prev_length = __v10.wrapping_sub(2);
}
loop {
let __v13: u32 = unsafe { (*s).strstart };
let __v14: u32 = __v13.wrapping_add(1);
unsafe {
(*s).strstart = __v14;
}
if __v14 <= max_insert {
INSERT_STRING!(s, unsafe { (*s).strstart }, hash_head);
}
let __v21: u32 = unsafe { (*s).prev_length };
let __v22: u32 = __v21.wrapping_sub(1);
unsafe {
(*s).prev_length = __v22;
}
if __v22 == 0 {
break;
}
}
unsafe {
(*s).match_available = 0;
}
unsafe {
(*s).match_length = 2u32;
}
let __v24: u32 = unsafe { (*s).strstart };
unsafe {
(*s).strstart = __v24.wrapping_add(1);
}
if bflush != 0 {
FLUSH_BLOCK!(s, 0i32);
}
} else {
if (unsafe { (*s).match_available }) != 0 {
// If there was no match at the previous position, output a
// single literal. If there was a match but the current match
// is longer, truncate the previous match to a single literal.
if (unsafe { zlib_tr_tally(s, 0u32, *unsafe { (*s).window }.wrapping_offset(unsafe { (*s).strstart }.wrapping_sub(1) as isize) as u32) }) != 0 {
FLUSH_BLOCK_ONLY!(s, 0i32);
}
let __v27: u32 = unsafe { (*s).strstart };
unsafe {
(*s).strstart = __v27.wrapping_add(1);
}
let __v30: u32 = unsafe { (*s).lookahead };
unsafe {
(*s).lookahead = __v30.wrapping_sub(1);
}
if (unsafe { (*(*s).strm).avail_out }) == 0 {
return 0;
}
} else {
// There is no previous match to compare with, wait for
// the next step to decide.
unsafe {
(*s).match_available = 1;
}
let __v33: u32 = unsafe { (*s).strstart };
unsafe {
(*s).strstart = __v33.wrapping_add(1);
}
let __v36: u32 = unsafe { (*s).lookahead };
unsafe {
(*s).lookahead = __v36.wrapping_sub(1);
}
}
}
}
if (unsafe { (*s).match_available }) != 0 {
unsafe { zlib_tr_tally(s, 0u32, *unsafe { (*s).window }.wrapping_offset(unsafe { (*s).strstart }.wrapping_sub(1) as isize) as u32) };
unsafe {
(*s).match_available = 0;
}
}
FLUSH_BLOCK__2!(s, flush == 5);
(if flush == 5 { 3i32 } else { 1i32 }) as u32
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.deflate.zlib_deflate_dfltcc_enabled")]
extern "C" fn zlib_deflate_dfltcc_enabled() -> i32 {
DEFLATE_DFLTCC_ENABLED!()
}

static __slate_asm_string_zlib_deflate_workspacesize_0: [u8; 1] = *b"\0";

static __slate_asm_string_zlib_deflate_workspacesize_1: [u8; 56] = *b"/home/takashi/c-corpus/linux/lib/zlib_deflate/deflate.c\0";

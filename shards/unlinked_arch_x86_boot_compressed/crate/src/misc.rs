
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
//! SPDX-License-Identifier: GPL-2.0
//!
//! When building position independent code with GCC using the -fPIC option,
//! (or even the -fPIE one on older versions), it will assume that we are
//! building a dynamic object (either a shared library or an executable) that
//! may have symbol references that can only be resolved at load time. For a
//! variety of reasons (ELF symbol preemption, the CoW footprint of the section
//! that is modified by the loader), this results in all references to symbols
//! with external linkage to go via entries in the Global Offset Table (GOT),
//! which carries absolute addresses which need to be fixed up when the
//! executable image is loaded at an offset which is different from its link
//! time offset.
//!
//! Fortunately, there is a way to inform the compiler that such symbol
//! references will be satisfied at link time rather than at load time, by
//! giving them 'hidden' visibility.
//! SPDX-License-Identifier: GPL-2.0
//!
//! misc.c
//!
//! This is a collection of several routines used to extract the kernel
//! which includes KASLR relocation, decompression, ELF parsing, and
//! relocation processing. Additionally included are the screen and serial
//! output functions and related debugging support functions.
//!
//! malloc by Hannu Savolainen 1993 and Matthias Urlichs 1994
//! puts by Nick Holloway 1993, better puts by Martin Mares 1995
//! High loaded stuff by Hans Lermen & Werner Almesberger, Feb. 1996
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
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
//! Force a compilation error if condition is true, but also produce a
//! result (of value 0 and type int), so the expression can be used
//! e.g. in a structure initializer (or where-ever else comma expressions
//! aren't permitted).
//!
//! Take an error message as an optional second argument. If omitted,
//! default to the stringification of the tested expression.
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
//! See linux/instrumentation.h
//! Begin/end of an instrumentation safe region
//! 64 bit
//! unused, was: #define EX_TYPE_COPY	 4
//! __ADDRESSABLE() is used to ensure the key symbol doesn't get stripped from
//! the symbol table so that objtool can reference it when it generates the
//! .static_call_sites section.
//! inhibit tail-call optimization
//! Call a function once. Similar to DO_ONCE(), but does not use jump label
//! patching via static keys.
//! These are for everybody (although not all archs will actually
//! discard it in modules)
//! warning conditions
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
//! Operation not permitted
//! Value too large for defined data type
//! restart by calling sys_restart_syscall
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
//! Check at compile time that something is of a particular type.
//! Always evaluates to 1 so you may use it easily in comparisons.
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
//! We do the locked ops that don't return the old value as
//! a mask operation on a byte.
//! Defines x86 CPU feature bits
//! Compile time versions of __arch_hweightN()
//! instrument_get_user() - add instrumentation to get_user()-like macros
//! @to: destination variable, may not be address-taken
//!
//! get_user() and friends are fragile, so it may depend on the implementation
//! whether the instrumentation happens before or after the data is copied from
//! the userspace.
//! The __sme_pa() and __sme_pa_nodebug() macros are meant for use when
//! writing to or comparing values from the cr3 register.  Having the
//! encryption mask set in cr3 enables the PGD entry to be encrypted and
//! avoid special case handling of PGD allocations.
//! PAGE_SHIFT determines the page size.
//!
//! Note: This definition is required because PAGE_SHIFT is used
//! in several places throughout the codebase.
//! Physical address where kernel should be loaded.
//! ilog2 - log base 2 of 32-bit or a 64-bit unsigned value
//! @n: parameter
//!
//! constant-capable log of base 2 calculation
//! - this can be used to initialise global variables from constant data, hence
//! the massive ternary operator construction
//!
//! selects the appropriately-sized optimised version depending on sizeof(n)
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
//! Ensure that leaf 0x2 cache and TLB type values do not intersect,
//! since they share the same type field at struct cpuid_0x2_table.
//! writeable
//! 4th level page in 5-level paging case
//! Boot IDT definitions
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
//! An exchange-type operation, which takes a value and a pointer, and
//! returns the old value.
//! signal pending
//! 32-bit address space on 64 bits
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
//! We use the MSB for PREEMPT_NEED_RESCHED mostly because it is available.
//! Macros to retrieve the current execution context:
//!
//! in_nmi()		- We're in NMI context
//! in_hardirq()		- We're in hard IRQ context
//! in_serving_softirq()	- We're in softirq context
//! in_task()		- We're in task context
//! SCHED_ISO: reserved but not implemented yet
//! These constants are defined in the MSan LLVM instrumentation pass.
//! ignore signal
//! bit-flags
//! If !LOCKDEP, the value is meaningless
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
//! mutex_init - initialize the mutex
//! @mutex: the mutex to be initialized
//!
//! Initialize the mutex to unlocked state.
//!
//! It is not allowed to initialize an already locked mutex.
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
//! wait_queue_entry::flags
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
//! Values for SRCU Tree srcu_data ->srcu_reader_flavor, but also used by rcutorture.
//! Suits me
//! __MAX_NR_ZONES
//! percpu_counter batch for local add or sub
//! Opaque type representing current VMA (vm_area_struct) flag state. Must be
//! accessed via vma_flags_xxx() helper functions.
//! Desired design of maximum size and alignment (see RFC2553)
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
//! number of edd_info structs starting at EDDBUF
//! Resource type
//! The IDR API does not expose the tagging functionality of the radix tree
//! to users.  Use tag 0 to track whether a node has free space below it.
//! number of env pointers
//! Generate SD flag indexes
//! Only touch one cacheline for common rbtree-for-core-layout case.
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
//! See the comment in asm-generic/io.h about ioremap_np().
//! DWORD 3
//! 32K LZ77 window
//! largest prime smaller than 65536
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
// Operation not supported on transport endpoint
// Operation is not supported
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
// __ffs - find first set bit in word
// @word: The word to search
//
// Undefined if no bit exists, so code should check against 0 first.
// ffz - find first zero bit in word
// @word: The word to search
//
// Undefined if no zero exists, so code should check against ~0UL first.
// N 32-bit words worth of info
// N 32-bit bug flags
// "tsc" Time Stamp Counter
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
// "rdpid" RDPID instruction
// "fsrm" Fast Short Rep Mov
// "serialize" SERIALIZE instruction
// LFENCE always serializing / synchronizes RDTSC
// BUG word(s)
// "apic_c1e" AMD Erratum 400
// "l1tf" CPU is affected by L1 Terminal Fault
// "div0" AMD DIV0 speculation bug
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
// IS_ERR_VALUE - Detect an error pointer.
// @x: The pointer to check.
//
// Like IS_ERR(), but does not generate a compiler warning if result is unused.
// N 32-bit words worth of info
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
// signal notifications exist
// rescheduling necessary
// idle is polling for TIF_NEED_RESCHED
// re-arm the timer
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
// to_cpumask - convert a NR_CPUS bitmap to a struct cpumask *
// @bitmap: the bitmap
//
// There are a few places where cpumask_var_t isn't appropriate and
// static cpumasks must be used (eg. very early boot), yet we don't
// expose the definition of 'struct cpumask'.
//
// This does the conversion, and can be used as a constant initializer.
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
// Opaque type representing current mm_struct flag state. Must be accessed via
// mm_flags_xxx() helper functions.
// Implementation specific max size
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
// Resource extended types
// No address assigned yet
// Set the IDR flag and the IDR_FREE tag
// buffer for the variables
// Generate SD flag bits
// Flexible array members are not allowed to be part of a union under
// C99, but this is not for any technical reason. Work around the
// limitation.
// Fixed by ACPI spec
// DWORD 5
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
macro_rules! ALIGN {
($x:expr, $a:expr) => {
__ALIGN_KERNEL!($x, $a)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE {
() => {
((core::mem::size_of::<[boot_params_to_save; 20]>() as u64) / (core::mem::size_of::<boot_params_to_save>() as u64)).wrapping_add(__must_be_array!() as u64)
};
}

/// Return the low n bits of the bit accumulator (n < 16)
macro_rules! BITS {
($n:expr, $hold:expr) => {
(($hold) as u32) & (1u32 << $n).wrapping_sub(1)
};
}

macro_rules! BIT_ULL {
($nr:expr) => {
ULL!() << $nr
};
}

/// Remove zero to seven bits as needed to go to a byte boundary
macro_rules! BYTEBITS {
($hold:expr, $bits:expr) => {
loop {
let __v: u64 = $hold;
let __v2: u64 = __v >> (($bits) & 7);
$hold = __v2;
let __v3: u32 = $bits;
let __v4: u32 = __v3.wrapping_sub(($bits) & 7);
$bits = __v4;
if true {
break;
}
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
let __v2: u64 = __v.wrapping_add(((unsafe { *$buf.offset(($i) as isize) }) as i32) as u64);
$s1 = __v2;
let __v3: u64 = $s2;
let __v4: u64 = __v3.wrapping_add($s1);
$s2 = __v4;
};
}

macro_rules! DO2 {
($buf:expr, $i:expr, $s1:expr, $s2:expr) => {
DO1!($buf, $i, $s1, $s2);
DO1!($buf, ($i) + 1, $s1, $s2);
};
}

macro_rules! DO4 {
($buf:expr, $i:expr, $s1:expr, $s2:expr) => {
DO2!($buf, $i, $s1, $s2);
DO2!($buf, ($i) + 2, $s1, $s2);
};
}

macro_rules! DO8 {
($buf:expr, $i:expr, $s1:expr, $s2:expr) => {
DO4!($buf, $i, $s1, $s2);
DO4!($buf, ($i) + 4, $s1, $s2);
};
}

/// Remove n bits from the bit accumulator
macro_rules! DROPBITS__2 {
($n:expr, $hold:expr, $bits:expr) => {
loop {
let __v: u64 = $hold;
let __v2: u64 = __v >> (($n) as i32);
$hold = __v2;
let __v3: u32 = $bits;
let __v4: u32 = __v3.wrapping_sub(($n) as u32);
$bits = __v4;
if true {
break;
}
}
};
}

/// Remove n bits from the bit accumulator
macro_rules! DROPBITS {
($n:expr, $hold:expr, $bits:expr) => {
loop {
let __v: u64 = $hold;
let __v2: u64 = __v >> $n;
$hold = __v2;
let __v3: u32 = $bits;
let __v4: u32 = __v3.wrapping_sub(($n) as u32);
$bits = __v4;
if true {
break;
}
}
};
}

macro_rules! INFLATE_NEED_CHECKSUM {
() => {
1i32
};
}

macro_rules! INFLATE_NEED_UPDATEWINDOW {
() => {
1i32
};
}

macro_rules! INFLATE_RESET_HOOK {
() => {
while false {
}
};
}

macro_rules! INFLATE_TYPEDO_HOOK {
() => {
while false {
}
};
}

/// Clear the input bit accumulator
macro_rules! INITBITS {
($hold:expr, $bits:expr) => {
loop {
$hold = 0;
$bits = 0;
if true {
break;
}
}
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

/// Load registers with state in inflate() for speed
macro_rules! LOAD {
($put:expr, $strm:expr, $left:expr, $next:expr, $have:expr, $hold:expr, $state:expr, $bits:expr) => {
loop {
$put = unsafe { (*$strm).next_out };
$left = (unsafe { (*$strm).avail_out }) as u32;
$next = unsafe { (*$strm).next_in };
$have = (unsafe { (*$strm).avail_in }) as u32;
$hold = unsafe { (*$state).hold };
$bits = unsafe { (*$state).bits };
if true {
break;
}
}
};
}

/// Restore state from registers in inflate()
macro_rules! RESTORE {
($strm:expr, $put:expr, $left:expr, $next:expr, $have:expr, $state:expr, $hold:expr, $bits:expr) => {
loop {
unsafe {
(*$strm).next_out = $put;
}
unsafe {
(*$strm).avail_out = ($left) as u64;
}
unsafe {
(*$strm).next_in = $next;
}
unsafe {
(*$strm).avail_in = ($have) as u64;
}
unsafe {
(*$state).hold = $hold;
}
unsafe {
(*$state).bits = $bits;
}
if true {
break;
}
}
};
}

/// Reverse the bytes in a 32-bit value
macro_rules! REVERSE {
($q:expr) => {
(($q) >> 24i32 & 255).wrapping_add(($q) >> 8i32 & 65280).wrapping_add((($q) & 65280) << 8i32).wrapping_add((($q) & 255) << 24i32)
};
}

macro_rules! ULL {
() => {
_ULL!()
};
}

/// check function to use adler32() for zlib or crc32() for gzip
macro_rules! UPDATE__2 {
($check:expr, $buf:expr, $len:expr) => {
zlib_adler32($check, $buf, ($len) as u32)
};
}

/// check function to use adler32() for zlib or crc32() for gzip
macro_rules! UPDATE {
($check:expr, $buf:expr, $len:expr) => {
zlib_adler32($check, ($buf) as *const u8, $len)
};
}

macro_rules! WS {
($strm:expr) => {
(unsafe { (*$strm).workspace }) as *mut inflate_workspace
};
}

macro_rules! _AC__2 {
() => {
18446744071578845184u64
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

macro_rules! _AC__4 {
() => {
18446744071620108288u64
};
}

macro_rules! _ULL {
() => {
_AC!()
};
}

macro_rules! __ALIGN_KERNEL_MASK {
($x:expr, $mask:expr) => {
($x) + $mask & !$mask
};
}

macro_rules! __ALIGN_KERNEL_MASK__2 {
($x:expr, $mask:expr) => {
$x.wrapping_add($mask) & !$mask
};
}

macro_rules! __ALIGN_KERNEL {
($x:expr, $a:expr) => {
__ALIGN_KERNEL_MASK__2!($x, (($a) as u64).wrapping_sub(1))
};
}

macro_rules! ____or {
() => {
1i32
};
}

macro_rules! __cmp_once {
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

macro_rules! __must_be_array {
() => {
core::mem::size_of::<__SlateRecord_bootparam_utils_h_82>() as i32
};
}

macro_rules! __or {
() => {
____or!()
};
}

/// Boot stub deals with identity mappings, physical and virtual addresses are
/// the same, so override these defines.
///
/// <asm/page.h> will not define them if they are already defined.
macro_rules! __pa {
($x:expr) => {
($x) as u64
};
}

macro_rules! debug_puthex {
($__x:expr) => {
__puthex($__x);
};
}

macro_rules! debug_putstr__2 {
($__x:expr) => {
__putstr(($__x) as *const i8);
};
}

macro_rules! error_putdec {
($__x:expr) => {
__putdec(($__x) as u64);
};
}

macro_rules! error_putstr__3 {
($__x:expr) => {
__putstr(($__x) as *const i8);
};
}

/// max_t - return maximum of two values, using the specified type
/// @type: data type to use
/// @x: first value
/// @y: second value
macro_rules! max_t {
($type:ty, $x:expr, $y:expr) => {
__cmp_once!($type, $x, $y)
};
}

/// Access builtin version by default.
macro_rules! memcpy__2 {
($d:expr, $s:expr, $l:expr) => {
unsafe { __builtin_memcpy(($d) as *mut (), ($s) as *const (), ($l) as u64) };
};
}

/// Access builtin version by default.
macro_rules! memcpy {
($d:expr, $s:expr, $l:expr) => {
unsafe { __builtin_memcpy(($d) as *mut (), ($s) as *const (), $l) };
};
}

macro_rules! memset {
($d:expr, $c:expr, $l:expr) => {
unsafe { __builtin_memset(($d) as *mut (), $c, $l) };
};
}

macro_rules! offsetof {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, eddbuf) as u64
};
}

macro_rules! offsetof__2 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, edd_mbr_sig_buf_entries) as u64
};
}

macro_rules! offsetof__3 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, olpc_ofw_header) as u64
};
}

macro_rules! offsetof__4 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, tboot_addr) as u64
};
}

macro_rules! offsetof__5 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, sys_desc_table) as u64
};
}

macro_rules! offsetof__6 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, e820_entries) as u64
};
}

macro_rules! offsetof__7 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, scratch) as u64
};
}

macro_rules! offsetof__8 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, cc_blob_address) as u64
};
}

macro_rules! offsetof__9 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, alt_mem_k) as u64
};
}

macro_rules! offsetof__10 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, secure_boot) as u64
};
}

macro_rules! offsetof__11 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, ist_info) as u64
};
}

macro_rules! offsetof__12 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, hdr) as u64
};
}

macro_rules! offsetof__13 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, eddbuf_entries) as u64
};
}

macro_rules! offsetof__14 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, efi_info) as u64
};
}

macro_rules! offsetof__15 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, screen_info) as u64
};
}

macro_rules! offsetof__16 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, hd0_info) as u64
};
}

macro_rules! offsetof__17 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, e820_table) as u64
};
}

macro_rules! offsetof__18 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, apm_bios_info) as u64
};
}

macro_rules! offsetof__19 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, edd_mbr_sig_buffer) as u64
};
}

macro_rules! offsetof__20 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, hd1_info) as u64
};
}

macro_rules! sizeof_mbr__4 {
() => {
{
let mut __v: u64;
__v = core::mem::size_of::<screen_info>() as u64;
__v
}
};
}

macro_rules! sizeof_mbr__5 {
() => {
{
let mut __v: u64;
__v = core::mem::size_of::<u64>() as u64;
__v
}
};
}

macro_rules! sizeof_mbr__6 {
() => {
{
let mut __v: u64;
__v = core::mem::size_of::<olpc_ofw_header>() as u64;
__v
}
};
}

macro_rules! sizeof_mbr__7 {
() => {
{
let mut __v: u64;
__v = core::mem::size_of::<sys_desc_table>() as u64;
__v
}
};
}

macro_rules! sizeof_mbr__8 {
() => {
{
let mut __v: u64;
__v = core::mem::size_of::<[boot_e820_entry; 128]>() as u64;
__v
}
};
}

macro_rules! sizeof_mbr__3 {
() => {
{
let mut __v: u64;
__v = core::mem::size_of::<[u8; 16]>() as u64;
__v
}
};
}

macro_rules! sizeof_mbr__9 {
() => {
{
let mut __v: u64;
__v = core::mem::size_of::<[edd_info; 6]>() as u64;
__v
}
};
}

macro_rules! sizeof_mbr__10 {
() => {
{
let mut __v: u64;
__v = core::mem::size_of::<efi_info>() as u64;
__v
}
};
}

macro_rules! sizeof_mbr__2 {
() => {
{
let mut __v: u64;
__v = core::mem::size_of::<u32>() as u64;
__v
}
};
}

macro_rules! sizeof_mbr__11 {
() => {
{
let mut __v: u64;
__v = core::mem::size_of::<apm_bios_info>() as u64;
__v
}
};
}

macro_rules! sizeof_mbr__12 {
() => {
{
let mut __v: u64;
__v = core::mem::size_of::<ist_info>() as u64;
__v
}
};
}

macro_rules! sizeof_mbr__13 {
() => {
{
let mut __v: u64;
__v = core::mem::size_of::<[u32; 16]>() as u64;
__v
}
};
}

macro_rules! sizeof_mbr__14 {
() => {
{
let mut __v: u64;
__v = core::mem::size_of::<setup_header>() as u64;
__v
}
};
}

macro_rules! sizeof_mbr {
() => {
{
let mut __v: u64;
__v = core::mem::size_of::<u8>() as u64;
__v
}
};
}

unsafe extern "C" {
    static mut output_len: u32;
    static mut trampoline_32bit: *mut u64;
    static mut ist_info: ist_info;
    static mut early_serial_base: i32;
    static mut input_data: [u8; 0];
    static mut input_len: u32;
    fn memmove(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn cmdline_find_option_bool(arg0: *const i8) -> i32;
    fn choose_random_location(arg0: u64, arg1: u64, arg2: *mut u64, arg3: u64, arg4: *mut u64);
    fn console_init();
    fn get_rsdp_addr() -> u64;
    fn cleanup_exception_handling();
    fn accept_memory(arg0: u64, arg1: u64);
    fn error(arg0: *mut i8);
    #[link_name = "memset"]
    fn __builtin_memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    #[link_name = "memcpy"]
    fn __builtin_memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct screen_info {
    orig_x: u8,
    orig_y: u8,
    ext_mem_k: u16,
    orig_video_page: u16,
    orig_video_mode: u8,
    orig_video_cols: u8,
    flags: u8,
    unused2: u8,
    orig_video_ega_bx: u16,
    unused3: u16,
    orig_video_lines: u8,
    orig_video_isVGA: u8,
    orig_video_points: u16,
    lfb_width: u16,
    lfb_height: u16,
    lfb_depth: u16,
    lfb_base: u32,
    lfb_size: u32,
    cl_magic: u16,
    cl_offset: u16,
    lfb_linelength: u16,
    red_size: u8,
    red_pos: u8,
    green_size: u8,
    green_pos: u8,
    blue_size: u8,
    blue_pos: u8,
    rsvd_size: u8,
    rsvd_pos: u8,
    vesapm_seg: u16,
    vesapm_off: u16,
    pages: u16,
    vesa_attributes: u16,
    capabilities: u32,
    ext_lfb_base: u32,
    _reserved: [u8; 2],
}


// Monochrome Text Display
// CGA Display
// EGA/VGA in Monochrome Mode
// EGA in Color Mode
// VGA+ in Color Mode
// VESA VGA in graphic mode
// ACER PICA-61 local S3 video
// MIPS Magnum 4000 G364 video
// Various SGI graphics hardware
// DEC TGA
// Sun frame buffer.
// Sun PCI based frame buffer.
// PowerMacintosh frame buffer.
// EFI graphic mode
// Frame buffer base is 64-bit
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
struct elf64_hdr {
    e_ident: [u8; 16],
    e_type: u16,
    e_machine: u16,
    e_version: u32,
    e_entry: u64,
    e_phoff: u64,
    e_shoff: u64,
    e_flags: u32,
    e_ehsize: u16,
    e_phentsize: u16,
    e_phnum: u16,
    e_shentsize: u16,
    e_shnum: u16,
    e_shstrndx: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct elf64_phdr {
    p_type: u32,
    p_flags: u32,
    p_offset: u64,
    p_vaddr: u64,
    p_paddr: u64,
    p_filesz: u64,
    p_memsz: u64,
    p_align: u64,
}


// e_ident[] indexes
// EI_MAG
#[unsafe(no_mangle)]
static mut boot_params_ptr: *mut boot_params = unsafe { core::mem::zeroed() };

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct boot_e820_entry {
    addr: u64,
    size: u64,
    r#type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_edd_h_87 {
    isa: __SlateRecord_edd_h_88,
    pci: __SlateRecord_edd_h_93,
    ibnd: __SlateRecord_edd_h_101,
    xprs: __SlateRecord_edd_h_104,
    htpt: __SlateRecord_edd_h_107,
    unknown: __SlateRecord_edd_h_110,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_88 {
    base_address: u16,
    reserved1: u16,
    reserved2: u32,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_93 {
    bus: u8,
    slot: u8,
    function: u8,
    channel: u8,
    reserved: u32,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_101 {
    reserved: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_104 {
    reserved: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_107 {
    reserved: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_110 {
    reserved: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_edd_h_114 {
    ata: __SlateRecord_edd_h_115,
    atapi: __SlateRecord_edd_h_122,
    scsi: __SlateRecord_edd_h_130,
    usb: __SlateRecord_edd_h_136,
    i1394: __SlateRecord_edd_h_140,
    fibre: __SlateRecord_edd_h_144,
    i2o: __SlateRecord_edd_h_148,
    raid: __SlateRecord_edd_h_152,
    sata: __SlateRecord_edd_h_157,
    unknown: __SlateRecord_edd_h_164,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_115 {
    device: u8,
    reserved1: u8,
    reserved2: u16,
    reserved3: u32,
    reserved4: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_122 {
    device: u8,
    lun: u8,
    reserved1: u8,
    reserved2: u8,
    reserved3: u32,
    reserved4: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_130 {
    id: u16,
    lun: u64,
    reserved1: u16,
    reserved2: u32,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_136 {
    serial_number: u64,
    reserved: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_140 {
    eui: u64,
    reserved: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_144 {
    wwid: u64,
    lun: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_148 {
    identity_tag: u64,
    reserved: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_152 {
    array_number: u32,
    reserved1: u32,
    reserved2: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_157 {
    device: u8,
    reserved1: u8,
    reserved2: u16,
    reserved3: u32,
    reserved4: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_edd_h_164 {
    reserved1: u64,
    reserved2: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct apm_bios_info {
    version: u16,
    cseg: u16,
    offset: u32,
    cseg_16: u16,
    dseg: u16,
    flags: u16,
    cseg_len: u16,
    cseg_16_len: u16,
    dseg_len: u16,
}


// max number of signatures to store
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct edd_device_params {
    length: u16,
    info_flags: u16,
    num_default_cylinders: u32,
    num_default_heads: u32,
    sectors_per_track: u32,
    number_of_sectors: u64,
    bytes_per_sector: u16,
    dpte_ptr: u32,
    key: u16,
    device_path_info_length: u8,
    reserved2: u8,
    reserved3: u16,
    host_bus_type: [u8; 4],
    interface_type: [u8; 8],
    interface_path: __SlateRecord_edd_h_87,
    device_path: __SlateRecord_edd_h_114,
    reserved4: u8,
    checksum: u8,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct edd_info {
    device: u8,
    version: u8,
    interface_support: u16,
    legacy_max_cylinder: u16,
    legacy_max_head: u8,
    legacy_sectors_per_track: u8,
    params: edd_device_params,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ist_info {
    signature: u32,
    command: u32,
    event: u32,
    perf_level: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct edid_info {
    dummy: [u8; 128],
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct setup_header {
    setup_sects: u8,
    root_flags: u16,
    syssize: u32,
    ram_size: u16,
    vid_mode: u16,
    root_dev: u16,
    boot_flag: u16,
    jump: u16,
    header: u32,
    version: u16,
    realmode_swtch: u32,
    start_sys_seg: u16,
    kernel_version: u16,
    type_of_loader: u8,
    loadflags: u8,
    setup_move_size: u16,
    code32_start: u32,
    ramdisk_image: u32,
    ramdisk_size: u32,
    bootsect_kludge: u32,
    heap_end_ptr: u16,
    ext_loader_ver: u8,
    ext_loader_type: u8,
    cmd_line_ptr: u32,
    initrd_addr_max: u32,
    kernel_alignment: u32,
    relocatable_kernel: u8,
    min_alignment: u8,
    xloadflags: u16,
    cmdline_size: u32,
    hardware_subarch: u32,
    hardware_subarch_data: u64,
    payload_offset: u32,
    payload_length: u32,
    setup_data: u64,
    pref_address: u64,
    init_size: u32,
    handover_offset: u32,
    kernel_info_offset: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sys_desc_table {
    length: u16,
    table: [u8; 14],
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct olpc_ofw_header {
    ofw_magic: u32,
    ofw_version: u32,
    cif_handler: u32,
    irq_desc_table: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct efi_info {
    efi_loader_signature: u32,
    efi_systab: u32,
    efi_memdesc_size: u32,
    efi_memdesc_version: u32,
    efi_memmap: u32,
    efi_memmap_size: u32,
    efi_systab_hi: u32,
    efi_memmap_hi: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bootparam_utils_h_82 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct internal_state {
}


// This is the maximum number of entries in struct boot_params::e820_table
// (the zeropage), which is part of the x86 boot protocol ABI:
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct boot_params {
    screen_info: screen_info,
    apm_bios_info: apm_bios_info,
    _pad2: [u8; 4],
    tboot_addr: u64,
    ist_info: ist_info,
    acpi_rsdp_addr: u64,
    _pad3: [u8; 8],
    hd0_info: [u8; 16],
    hd1_info: [u8; 16],
    sys_desc_table: sys_desc_table,
    olpc_ofw_header: olpc_ofw_header,
    ext_ramdisk_image: u32,
    ext_ramdisk_size: u32,
    ext_cmd_line_ptr: u32,
    _pad4: [u8; 112],
    cc_blob_address: u32,
    edid_info: edid_info,
    efi_info: efi_info,
    alt_mem_k: u32,
    scratch: u32,
    e820_entries: u8,
    eddbuf_entries: u8,
    edd_mbr_sig_buf_entries: u8,
    kbd_status: u8,
    secure_boot: u8,
    _pad5: [u8; 2],
    sentinel: u8,
    _pad6: [u8; 1],
    hdr: setup_header,
    _pad7: [u8; 36],
    edd_mbr_sig_buffer: [u32; 16],
    e820_table: [boot_e820_entry; 128],
    _pad8: [u8; 48],
    eddbuf: [edd_info; 6],
    _pad9: [u8; 276],
}


fn early_tdx_detect() {
}

#[inline(always)]
#[unsafe(link_section = ".text.slate_distinct.misc.__outb")]
extern "C" fn __outb(mut value: u8, mut port: u16) {
unsafe {
core::arch::asm!("outb %al, %dx", in("ax") (value as u8) as u32, in("dx") port as u16, options(nostack, att_syntax));
}
}

#[inline(always)]
#[unsafe(link_section = ".text.slate_distinct.misc.__inb")]
extern "C" fn __inb(mut port: u16) -> u8 {
let mut value: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("inb %dx, %al", lateout("ax") __t0, in("dx") port as u16, options(nostack, att_syntax));
value = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
value
}

#[inline(always)]
#[unsafe(link_section = ".text.slate_distinct.misc.__outw")]
extern "C" fn __outw(mut value: u16, mut port: u16) {
unsafe {
core::arch::asm!("outw %ax, %dx", in("ax") value as u16, in("dx") port as u16, options(nostack, att_syntax));
}
}

#[repr(C, align(4))]
struct __SlateAlign4<T>(T);

unsafe impl<T> Sync for __SlateAlign4<T> {
}

#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct port_io_ops {
    f_inb: Option<unsafe extern "C" fn(u16) -> u8>,
    f_outb: Option<unsafe extern "C" fn(u8, u16)>,
    f_outw: Option<unsafe extern "C" fn(u16, u16)>,
}


#[unsafe(no_mangle)]
static mut pio_ops: port_io_ops = unsafe { core::mem::zeroed() };

fn init_default_io_ops() {
unsafe {
pio_ops.f_inb = Some(__inb);
}
unsafe {
pio_ops.f_outb = Some(__outb);
}
unsafe {
pio_ops.f_outw = Some(__outw);
}
}

// Redirect port I/O operations via pio_ops callbacks.
// TDX guests override these callbacks with TDX-specific helpers.
#[unsafe(no_mangle)]
static mut free_mem_ptr: i64 = unsafe { core::mem::zeroed() };

#[unsafe(no_mangle)]
static mut free_mem_end_ptr: i64 = unsafe { core::mem::zeroed() };

#[unsafe(no_mangle)]
static mut spurious_nmi_count: i32 = unsafe { core::mem::zeroed() };

fn init_unaccepted_memory() -> bool {
false
}

#[repr(C)]
#[derive(Clone, Copy)]
struct boot_params_to_save {
    start: u32,
    len: u32,
}


fn sanitize_boot_params(boot_params: *mut boot_params) {
// IMPORTANT NOTE TO BOOTLOADER AUTHORS: do not simply clear
// this field.  The purpose of this field is to guarantee
// compliance with the x86 boot spec located in
// Documentation/arch/x86/boot.rst .  That spec says that the
// *whole* structure should be cleared, after which only the
// portion defined by struct setup_header (boot_params->hdr)
// should be copied in.
//
// If you're having an issue because the sentinel is set, you
// need to change the whole structure to be cleared, not this
// (or any other) individual field, or you will soon have
// problems again.
if (unsafe { (*boot_params).sentinel }) != 0 {
let bp_base: *mut i8 = boot_params as *mut i8;
let save_base: *mut i8 = (unsafe { core::ptr::addr_of_mut!(scratch) }) as *mut i8;
let mut i: i32 = 0;
let to_save: __SlateAlign16<[boot_params_to_save; 20]> = __SlateAlign16([boot_params_to_save { start: offsetof__15!(boot_params) as u32, len: sizeof_mbr__4!() as u32 }, boot_params_to_save { start: offsetof__18!(boot_params) as u32, len: sizeof_mbr__11!() as u32 }, boot_params_to_save { start: offsetof__4!(boot_params) as u32, len: sizeof_mbr__5!() as u32 }, boot_params_to_save { start: offsetof__11!(boot_params) as u32, len: sizeof_mbr__12!() as u32 }, boot_params_to_save { start: offsetof__16!(boot_params) as u32, len: sizeof_mbr__3!() as u32 }, boot_params_to_save { start: offsetof__20!(boot_params) as u32, len: sizeof_mbr__3!() as u32 }, boot_params_to_save { start: offsetof__5!(boot_params) as u32, len: sizeof_mbr__7!() as u32 }, boot_params_to_save { start: offsetof__3!(boot_params) as u32, len: sizeof_mbr__6!() as u32 }, boot_params_to_save { start: offsetof__14!(boot_params) as u32, len: sizeof_mbr__10!() as u32 }, boot_params_to_save { start: offsetof__9!(boot_params) as u32, len: sizeof_mbr__2!() as u32 }, boot_params_to_save { start: offsetof__7!(boot_params) as u32, len: sizeof_mbr__2!() as u32 }, boot_params_to_save { start: offsetof__6!(boot_params) as u32, len: sizeof_mbr!() as u32 }, boot_params_to_save { start: offsetof__13!(boot_params) as u32, len: sizeof_mbr!() as u32 }, boot_params_to_save { start: offsetof__2!(boot_params) as u32, len: sizeof_mbr!() as u32 }, boot_params_to_save { start: offsetof__19!(boot_params) as u32, len: sizeof_mbr__13!() as u32 }, boot_params_to_save { start: offsetof__10!(boot_params) as u32, len: sizeof_mbr!() as u32 }, boot_params_to_save { start: offsetof__12!(boot_params) as u32, len: sizeof_mbr__14!() as u32 }, boot_params_to_save { start: offsetof__17!(boot_params) as u32, len: sizeof_mbr__8!() as u32 }, boot_params_to_save { start: offsetof!(boot_params) as u32, len: sizeof_mbr__9!() as u32 }, boot_params_to_save { start: offsetof__8!(boot_params) as u32, len: sizeof_mbr__2!() as u32 }]);
memset!(unsafe { core::ptr::addr_of_mut!(scratch) }, 0i32, core::mem::size_of::<boot_params>() as u64);
while (i as u64) < ARRAY_SIZE!() {
memcpy__2!(unsafe { save_base.offset(to_save.0[i as usize].start as isize) }, unsafe { bp_base.offset(to_save.0[i as usize].start as isize) }, to_save.0[i as usize].len);
i += 1;
}
memcpy!(boot_params, save_base, core::mem::size_of::<boot_params>() as u64);
}
}

static mut scratch: boot_params = unsafe { core::mem::zeroed() };

static mut malloc_ptr: u64 = unsafe { core::mem::zeroed() };

static mut malloc_count: i32 = unsafe { core::mem::zeroed() };

fn malloc(size: i32) -> *mut () {
if size < 0 {
return core::ptr::null_mut::<()>();
}
if (unsafe { malloc_ptr }) == 0 {
unsafe {
malloc_ptr = free_mem_ptr as u64;
}
}
unsafe {
malloc_ptr = unsafe { malloc_ptr }.wrapping_add(7) & 18446744073709551608u64;
}
// Align
let p: *mut () = (unsafe { malloc_ptr }) as *mut ();
let __v0: u64 = unsafe { malloc_ptr };
unsafe {
malloc_ptr = __v0.wrapping_add(size as u64);
}
if (unsafe { free_mem_end_ptr }) != 0 && (unsafe { malloc_ptr }) >= ((unsafe { free_mem_end_ptr }) as u64) {
return core::ptr::null_mut::<()>();
}
unsafe {
malloc_count = malloc_count + 1;
}
p
}

fn free(r#where: *mut ()) {
unsafe {
malloc_count = malloc_count - 1;
}
if (unsafe { malloc_count }) == 0 {
unsafe {
malloc_ptr = free_mem_ptr as u64;
}
}
}

// WARNING!!
// This code is compiled with -fPIC and it is relocated dynamically at
// run time, but no relocation processing is performed. This means that
// it is not safe to place pointers in static structures.
// Macros used by the included decompressor code below.
// Define an externally visible malloc()/free().
// Provide definitions of memzero and memmove as some of the decompressors will
// try to define their own functions if these are not defined as macros.
// Functions used by the included decompressor code below.
// This is set up by the setup-routine at boot-time
static mut vidmem: *mut i8 = unsafe { core::mem::zeroed() };

static mut vidport: i32 = unsafe { core::mem::zeroed() };

/// These might be accessed before .bss is cleared, so use .data instead.
#[unsafe(link_section = ".data")]
static mut lines: i32 = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".data")]
static mut cols: i32 = unsafe { core::mem::zeroed() };

#[repr(C)]
#[derive(Clone, Copy)]
struct z_stream_s {
    next_in: *const u8,
    avail_in: u64,
    total_in: u64,
    next_out: *mut u8,
    avail_out: u64,
    total_out: u64,
    msg: *mut i8,
    state: *mut internal_state,
    workspace: *mut (),
    data_type: i32,
    adler: u64,
    reserved: u64,
}


// Only for inflate at present
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
buf = unsafe { __v66.add(16) };
k -= 16;
}
if k != 0 {
loop {
let __v71: *const u8 = buf;
buf = unsafe { __v71.add(1) };
let __v73: u8 = unsafe { *__v71 };
s1 = s1.wrapping_add((__v73 as i32) as u64);
s2 = s2.wrapping_add(s1);
let __v80: i32 = k - 1;
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
struct __SlateRecord_inftrees_h_31 {
    op: u8,
    bits: u8,
    val: u16,
}


// Maximum size of dynamic tree.  The maximum found in a long but non-
// exhaustive search was 1444 code structures (852 for length/literals
// and 592 for distances, the latter actually the result of an
// exhaustive search).  The true maximum is not known, but the value
// below is more than safe.
#[unsafe(no_mangle)]
extern "C" fn zlib_inflate_table(r#type: u32, lens: *mut u16, codes: u32, table: *mut *mut __SlateRecord_inftrees_h_31, bits: *mut u32, work: *mut u16) -> i32 {
// accumulate lengths for codes (assumes lens[] all in 0..MAXBITS)
let mut len: u32 = 0; // a code's length in bits
let mut sym: u32 = 0; // index of code symbols
let mut min: u32 = 1;
let mut max: u32 = 15; // minimum and maximum code lengths
let mut root: u32 = 0; // number of index bits for root table
let mut curr: u32 = 0; // number of index bits for current table
// current bits to drop from code for index
let mut drop: u32 = 0; // code bits to drop for sub-table
// check for an over-subscribed or incomplete set of lengths
let mut left: i32 = 1; // number of prefix codes available
let mut used: u32 = 0; // code entries in table used
// initialize state for loop
// starting code
let mut huff: u32 = 0; // Huffman code
let mut incr: u32 = 0; // for incrementing code, index
let mut fill: u32 = 0; // index for replicating entries
// trigger new sub-table when len > root
let mut low: u32 = 4294967295u32; // low bits for current root entry
let mut mask: u32 = 0; // mask for low root bits
let mut this: __SlateRecord_inftrees_h_31 = unsafe { core::mem::zeroed() }; // table entry for duplication
let mut next: *mut __SlateRecord_inftrees_h_31 = unsafe { core::mem::zeroed() }; // next available space in table
let mut base: *const u16 = unsafe { core::mem::zeroed() }; // base value table to use
let mut extra: *const u16 = unsafe { core::mem::zeroed() }; // extra bits table to use
let mut end: i32 = 0; // use base and extra for symbol > end
let mut count: __SlateAlign16<[u16; 16]> = __SlateAlign16([0u16; 16]); // number of codes of each length
let mut offs: __SlateAlign16<[u16; 16]> = __SlateAlign16([0u16; 16]); // offsets in table for each length
// Length codes 257..285 base
// Length codes 257..285 extra
// Distance codes 0..29 base
// Distance codes 0..29 extra
// Process a set of code lengths to create a canonical Huffman code.  The
// code lengths are lens[0..codes-1].  Each length corresponds to the
// symbols 0..codes-1.  The Huffman code is generated by first sorting the
// symbols by length from short to long, and retaining the symbol order
// for codes with equal lengths.  Then the code starts with all zero bits
// for the first code of the shortest length, and the codes are integer
// increments for the same length, and zeros are appended as the length
// increases.  For the deflate format, these bits are stored backwards
// from their more natural integer increment ordering, and so when the
// decoding tables are built in the large loop below, the integer codes
// are incremented backwards.
//
// This routine assumes, but does not check, that all of the entries in
// lens[] are in the range 0..MAXBITS.  The caller must assure this.
// 1..MAXBITS is interpreted as that code length.  zero means that that
// symbol does not occur in this code.
//
// The codes are sorted by computing a count of codes for each length,
// creating from that a table of starting indices for each length in the
// sorted table, and then entering the symbols in order in the sorted
// table.  The sorted table is work[], with that space being provided by
// the caller.
//
// The length counts are used for other purposes as well, i.e. finding
// the minimum and maximum length codes, determining if there are any
// codes at all, checking for a valid set of lengths, and looking ahead
// at length counts to determine sub-table sizes when building the
// decoding tables.
while len <= 15 {
count.0[len as usize] = 0;
len = len.wrapping_add(1);
}
while sym < codes {
let __v3: *mut u16 = unsafe { (count.0.as_mut_ptr() as *mut u16).offset((*lens.add(sym as usize) as i32) as isize) };
unsafe {
*__v3 = ((*__v3 as i32) + 1) as u16;
}
sym = sym.wrapping_add(1);
}
// bound code lengths, force root to be within code lengths
root = unsafe { *bits };
while max >= 1 {
if (count.0[max as usize] as i32) != 0 {
break;
}
max = max.wrapping_sub(1);
}
if root > max {
root = max;
}
if max == 0 {
// no symbols to code at all
this.op = 64; // invalid code marker
this.bits = 1;
this.val = 0;
let __v13: *mut __SlateRecord_inftrees_h_31 = unsafe { *table };
unsafe {
*table = __v13.add(1);
}
unsafe {
*__v13 = this;
}
// make a table to force an error
let __v16: *mut __SlateRecord_inftrees_h_31 = unsafe { *table };
unsafe {
*table = __v16.add(1);
}
unsafe {
*__v16 = this;
}
unsafe {
*bits = 1;
}
return 0; // no symbols, but wait for decoding to report error
}
while min < 15 {
if (count.0[min as usize] as i32) != 0 {
break;
}
min = min.wrapping_add(1);
}
if root < min {
root = min;
}
len = 1;
while len <= 15 {
left <<= 1i32;
left -= count.0[len as usize] as i32;
if left < 0 {
return -1;
}
// over-subscribed
len = len.wrapping_add(1);
}
if left > 0 && ((r#type as u32) == 0 || max != 1) {
return -1;
}
// incomplete set
// generate offsets into symbol table for each length for sorting
offs.0[1] = 0;
len = 1;
while len < 15 {
offs.0[len.wrapping_add(1) as usize] = ((offs.0[len as usize] as i32) + (count.0[len as usize] as i32)) as u16;
len = len.wrapping_add(1);
}
// sort symbols by length, by symbol order within each length
sym = 0;
while sym < codes {
if ((unsafe { *lens.add(sym as usize) }) as i32) != 0 {
let __v31: *mut u16 = unsafe { (offs.0.as_mut_ptr() as *mut u16).offset((*lens.add(sym as usize) as i32) as isize) };
let __v32: u16 = unsafe { *__v31 };
unsafe {
*__v31 = ((__v32 as i32) + 1) as u16;
}
unsafe {
*work.offset((__v32 as i32) as isize) = sym as u16;
}
}
sym = sym.wrapping_add(1);
}
// Create and fill in decoding tables.  In this loop, the table being
// filled is at next and has curr index bits.  The code being used is huff
// with length len.  That code is converted to an index by dropping drop
// bits off of the bottom.  For codes where len is less than drop + curr,
// those top drop + curr - len bits are incremented through all values to
// fill the table with replicated entries.
//
// root is the number of index bits for the root table.  When len exceeds
// root, sub-tables are created pointed to by the root entry with an index
// of the low root bits of huff.  This is saved in low to check for when a
// new sub-table should be started.  drop is zero when the root table is
// being filled, and drop is root when sub-tables are being filled.
//
// When a new sub-table is needed, it is necessary to look ahead in the
// code lengths to determine what size sub-table is needed.  The length
// counts are used for this, and so count[] is decremented as codes are
// entered in the tables.
//
// used keeps track of how many table entries have been allocated from the
// provided *table space.  It is checked when a LENS table is being made
// against the space in *table, ENOUGH, minus the maximum space needed by
// the worst case distance code, MAXD.  This should never happen, but the
// sufficiency of ENOUGH has not been proven exhaustively, hence the check.
// This assumes that when type == LENS, bits == 9.
//
// sym increments through all symbols, and the loop terminates when
// all codes of length max, i.e. all codes, have been processed.  This
// routine permits incomplete codes, so another loop after this one fills
// in the rest of the decoding tables with invalid code markers.
// set up for code type
match r#type as u32 {
0 => {
let __v37: *const u16 = work as *const u16;
extra = __v37;
base = __v37; // dummy value--not used
end = 19;
}
1 => {
base = unsafe { core::ptr::addr_of!(lbase.0) as *const u16 };
let __v39: *const u16 = base;
base = unsafe { __v39.sub(257) };
extra = unsafe { core::ptr::addr_of!(lext.0) as *const u16 };
let __v41: *const u16 = extra;
extra = unsafe { __v41.sub(257) };
end = 256;
}
_ => {
base = unsafe { core::ptr::addr_of!(dbase.0) as *const u16 }; // DISTS
extra = unsafe { core::ptr::addr_of!(dext.0) as *const u16 };
end = -1;
}
}
sym = 0; // starting code symbol
len = min; // starting code length
next = unsafe { *table }; // current table to fill in
curr = root; // current table index bits
used = 1u32 << root; // use root table entries
mask = used.wrapping_sub(1); // mask for comparing low
// check available table space
if (r#type as u32) == 1 && used >= 1456u32 {
return 1;
}
// process all codes and make table entries
loop {
// create table entry
this.bits = len.wrapping_sub(drop) as u8;
if ((unsafe { *work.add(sym as usize) }) as i32) < end {
this.op = 0;
this.val = unsafe { *work.add(sym as usize) };
} else {
if ((unsafe { *work.add(sym as usize) }) as i32) > end {
this.op = (unsafe { *extra.offset((*work.add(sym as usize) as i32) as isize) }) as u8;
this.val = unsafe { *base.offset((*work.add(sym as usize) as i32) as isize) };
} else {
this.op = 96u8; // end of block
this.val = 0;
}
}
// replicate for those indices with low len bits equal to huff
incr = 1u32 << len.wrapping_sub(drop);
fill = 1u32 << curr;
min = fill; // save offset to next table
loop {
fill = fill.wrapping_sub(incr);
unsafe {
*next.offset((huff >> drop).wrapping_add(fill) as isize) = this;
}
if fill == 0 {
break;
}
}
// backwards increment the len-bit code huff
incr = 1u32 << len.wrapping_sub(1);
while huff & incr != 0 {
incr >>= 1i32;
}
if incr != 0 {
huff &= incr.wrapping_sub(1);
huff = huff.wrapping_add(incr);
} else {
huff = 0;
}
// go to next symbol, update count, len
sym = sym.wrapping_add(1);
let __v55: *mut u16 = unsafe { (count.0.as_mut_ptr() as *mut u16).add(len as usize) };
let __v57: u16 = (((unsafe { *__v55 }) as i32) - 1) as u16;
unsafe {
*__v55 = __v57;
}
if (__v57 as i32) == 0 {
if len == max {
break;
}
len = (unsafe { *lens.offset((*work.add(sym as usize) as i32) as isize) }) as u32;
}
// create new sub-table if needed
if len > root && huff & mask != low {
// if first time, transition to sub-tables
if drop == 0 {
drop = root;
}
// increment past last table
let __v59: *mut __SlateRecord_inftrees_h_31 = next;
next = unsafe { __v59.add(min as usize) }; // here min is 1 << curr
// determine length of next table
curr = len.wrapping_sub(drop);
left = 1i32 << curr;
while curr.wrapping_add(drop) < max {
left -= count.0[curr.wrapping_add(drop) as usize] as i32;
if left <= 0 {
break;
}
curr = curr.wrapping_add(1);
left <<= 1i32;
}
// check for enough space
used = used.wrapping_add(1u32 << curr);
if (r#type as u32) == 1 && used >= 1456u32 {
return 1;
}
// point entry in root table to sub-table
low = huff & mask;
unsafe {
(*unsafe { *table }.add(low as usize)).op = curr as u8;
}
unsafe {
(*unsafe { *table }.add(low as usize)).bits = root as u8;
}
unsafe {
(*unsafe { *table }.add(low as usize)).val = (next.offset_from(*table as *mut __SlateRecord_inftrees_h_31) as i64) as u16;
}
}
}
// Fill in rest of table for incomplete codes.  This loop is similar to the
// loop above in incrementing huff for table indices.  It is assumed that
// len is equal to curr + drop, so there is no loop needed to increment
// through high index bits.  When the current sub-table is filled, the loop
// drops back to the root table to fill in any remaining entries there.
this.op = 64; // invalid code marker
this.bits = len.wrapping_sub(drop) as u8;
this.val = 0;
while huff != 0 {
// when done with sub-table, drop back to root table
if drop != 0 && huff & mask != low {
drop = 0;
len = root;
next = unsafe { *table };
this.bits = len as u8;
}
// put invalid code marker in table
unsafe {
*next.add((huff >> drop) as usize) = this;
}
// backwards increment the len-bit code huff
incr = 1u32 << len.wrapping_sub(1);
while huff & incr != 0 {
incr >>= 1i32;
}
if incr != 0 {
huff &= incr.wrapping_sub(1);
huff = huff.wrapping_add(incr);
} else {
huff = 0;
}
}
// set return parameters
let __v79: *mut __SlateRecord_inftrees_h_31 = unsafe { *table };
unsafe {
*table = __v79.add(used as usize);
}
unsafe {
*bits = root;
}
0
}

static lbase: __SlateAlign16<[u16; 31]> = __SlateAlign16([3u16, 4u16, 5u16, 6u16, 7u16, 8u16, 9u16, 10u16, 11u16, 13u16, 15u16, 17u16, 19u16, 23u16, 27u16, 31u16, 35u16, 43u16, 51u16, 59u16, 67u16, 83u16, 99u16, 115u16, 131u16, 163u16, 195u16, 227u16, 258u16, 0u16, 0u16]);

static lext: __SlateAlign16<[u16; 31]> = __SlateAlign16([16u16, 16u16, 16u16, 16u16, 16u16, 16u16, 16u16, 16u16, 17u16, 17u16, 17u16, 17u16, 18u16, 18u16, 18u16, 18u16, 19u16, 19u16, 19u16, 19u16, 20u16, 20u16, 20u16, 20u16, 21u16, 21u16, 21u16, 21u16, 16u16, 201u16, 196u16]);

static dbase: __SlateAlign16<[u16; 32]> = __SlateAlign16([1u16, 2u16, 3u16, 4u16, 5u16, 7u16, 9u16, 13u16, 17u16, 25u16, 33u16, 49u16, 65u16, 97u16, 129u16, 193u16, 257u16, 385u16, 513u16, 769u16, 1025u16, 1537u16, 2049u16, 3073u16, 4097u16, 6145u16, 8193u16, 12289u16, 16385u16, 24577u16, 0u16, 0u16]);

static dext: __SlateAlign16<[u16; 32]> = __SlateAlign16([16u16, 16u16, 16u16, 16u16, 17u16, 17u16, 18u16, 18u16, 19u16, 19u16, 20u16, 20u16, 21u16, 21u16, 22u16, 22u16, 23u16, 23u16, 24u16, 24u16, 25u16, 25u16, 26u16, 26u16, 27u16, 27u16, 28u16, 28u16, 29u16, 29u16, 64u16, 64u16]);

#[repr(C)]
#[derive(Clone, Copy)]
struct inflate_state {
    mode: u32,
    last: i32,
    wrap: i32,
    havedict: i32,
    flags: i32,
    dmax: u32,
    check: u64,
    total: u64,
    wbits: u32,
    wsize: u32,
    whave: u32,
    write: u32,
    window: *mut u8,
    hold: u64,
    bits: u32,
    length: u32,
    offset: u32,
    extra: u32,
    lencode: *const __SlateRecord_inftrees_h_31,
    distcode: *const __SlateRecord_inftrees_h_31,
    lenbits: u32,
    distbits: u32,
    ncode: u32,
    nlen: u32,
    ndist: u32,
    have: u32,
    next: *mut __SlateRecord_inftrees_h_31,
    lens: [u16; 320],
    work: [u16; 288],
    codes: [__SlateRecord_inftrees_h_31; 2048],
}


#[repr(C)]
#[derive(Clone, Copy)]
union uu {
    us: u16,
    b: [u8; 2],
}


fn get_unaligned16(p: *const u16) -> u16 {
let mut mm: uu = unsafe { core::mem::zeroed() };
let b: *mut u8 = p as *mut u8;
unsafe {
mm.b[0] = *b.add(0);
}
unsafe {
mm.b[1] = *b.add(1);
}
unsafe { mm.us }
}

#[unsafe(no_mangle)]
extern "C" fn inflate_fast(strm: *mut z_stream_s, start: u32) {
let mut sfrom: *mut u16 = unsafe { core::mem::zeroed() };
let mut mm: core::mem::MaybeUninit<uu> = core::mem::MaybeUninit::uninit();
let __slate_slot_89: *mut uu = core::ptr::addr_of_mut!(mm) as *mut uu;
let mut pat16: u16 = unsafe { core::mem::zeroed() };
let mut loops: u64 = unsafe { core::mem::zeroed() };
let mut sout: *mut u16 = unsafe { core::mem::zeroed() };
// where to copy match from
let mut from: *mut u8 = unsafe { core::mem::zeroed() };
// match distance
let mut dist: u32 = unsafe { core::mem::zeroed() };
// match length, unused bytes
let mut len: u32 = unsafe { core::mem::zeroed() };
// code bits, operation, extra bits, or
//
//  window position, window bytes to copy
let mut op: u32 = unsafe { core::mem::zeroed() };
// retrieved table entry
let mut this: core::mem::MaybeUninit<__SlateRecord_inftrees_h_31> = core::mem::MaybeUninit::uninit();
let __slate_slot_176: *mut __SlateRecord_inftrees_h_31 = core::ptr::addr_of_mut!(this) as *mut __SlateRecord_inftrees_h_31; // mask for first level of distance codes
let mut dmask: u32 = unsafe { core::mem::zeroed() };
// mask for first level of length codes
let mut lmask: u32 = unsafe { core::mem::zeroed() };
// local strm->distcode
let mut dcode: *const __SlateRecord_inftrees_h_31 = unsafe { core::mem::zeroed() };
// local strm->lencode
let mut lcode: *const __SlateRecord_inftrees_h_31 = unsafe { core::mem::zeroed() };
// local strm->bits
let mut bits: u32 = unsafe { core::mem::zeroed() };
// local strm->hold
let mut hold: u64 = unsafe { core::mem::zeroed() };
// allocated sliding window, if wsize != 0
let mut window: *mut u8 = unsafe { core::mem::zeroed() };
// window write index
let mut write: u32 = unsafe { core::mem::zeroed() };
// valid bytes in the window
let mut whave: u32 = unsafe { core::mem::zeroed() };
// window size or zero if not using window
let mut wsize: u32 = unsafe { core::mem::zeroed() };
// while out < end, enough space available
let mut end: *mut u8 = unsafe { core::mem::zeroed() };
// inflate()'s initial strm->next_out
let mut beg: *mut u8 = unsafe { core::mem::zeroed() };
// local strm->next_out
let mut out: *mut u8 = unsafe { core::mem::zeroed() };
// while in < last, enough input available
let mut last: *const u8 = unsafe { core::mem::zeroed() };
// local strm->next_in
let mut r#in: *const u8 = unsafe { core::mem::zeroed() };
let mut state: *mut inflate_state = unsafe { core::mem::zeroed() };
unsafe {
// copy state to local variables
state = (*strm).state as *mut inflate_state;
r#in = (*strm).next_in;
last = r#in.offset(unsafe { (*strm).avail_in }.wrapping_sub(5) as isize);
out = (*strm).next_out;
beg = out.offset(-((start as u64).wrapping_sub((*strm).avail_out) as isize));
end = out.offset(unsafe { (*strm).avail_out }.wrapping_sub(257) as isize);
wsize = (*state).wsize;
whave = (*state).whave;
write = (*state).write;
window = (*state).window;
hold = (*state).hold;
bits = (*state).bits;
lcode = (*state).lencode;
dcode = (*state).distcode;
lmask = (1u32 << (*state).lenbits).wrapping_sub(1);
dmask = (1u32 << (*state).distbits).wrapping_sub(1);
// decode literals and length/distances until end-of-block or not enough
// input data or output space
'__join_0: {
'__join_25: {
'__join_45: {
'__loop_64: loop {
if bits < 15 {
let __v171: *const u8 = r#in;
r#in = __v171.add(1);
hold = hold.wrapping_add((*__v171 as u64) << bits);
bits = bits.wrapping_add(8);
let __v164: *const u8 = r#in;
r#in = __v164.add(1);
hold = hold.wrapping_add((*__v164 as u64) << bits);
bits = bits.wrapping_add(8);
}
*__slate_slot_176 = *lcode.add((hold & (lmask as u64)) as usize);
'__join_1: {
'__join_2: {
loop {
op = (*__slate_slot_176).bits as u32;
hold >>= op;
bits = bits.wrapping_sub(op);
op = (*__slate_slot_176).op as u32;
if op == 0 {
break '__join_2;
} else {
if op & 16 != 0 {
break;
} else {
if op & 64 == 0 {
// 2nd level length code
*__slate_slot_176 = *lcode.offset((((*__slate_slot_176).val as i32) as u64).wrapping_add(hold & ((1u32 << op).wrapping_sub(1) as u64)) as isize);
} else {
break '__loop_64;
}
}
}
}
// length base
len = (*__slate_slot_176).val as u32;
op &= 15; // number of extra bits
if op != 0 {
if bits < op {
let __v151: *const u8 = r#in;
r#in = __v151.add(1);
hold = hold.wrapping_add((*__v151 as u64) << bits);
bits = bits.wrapping_add(8);
}
len = len.wrapping_add((hold as u32) & (1u32 << op).wrapping_sub(1));
hold >>= op;
bits = bits.wrapping_sub(op);
}
if bits < 15 {
let __v138: *const u8 = r#in;
r#in = __v138.add(1);
hold = hold.wrapping_add((*__v138 as u64) << bits);
bits = bits.wrapping_add(8);
let __v131: *const u8 = r#in;
r#in = __v131.add(1);
hold = hold.wrapping_add((*__v131 as u64) << bits);
bits = bits.wrapping_add(8);
}
*__slate_slot_176 = *dcode.add((hold & (dmask as u64)) as usize);
loop {
op = (*__slate_slot_176).bits as u32;
hold >>= op;
bits = bits.wrapping_sub(op);
op = (*__slate_slot_176).op as u32;
if op & 16 != 0 {
break;
} else {
if op & 64 == 0 {
// 2nd level distance code
*__slate_slot_176 = *dcode.offset((((*__slate_slot_176).val as i32) as u64).wrapping_add(hold & ((1u32 << op).wrapping_sub(1) as u64)) as isize);
} else {
break '__join_45;
}
}
}
// distance base
dist = (*__slate_slot_176).val as u32;
op &= 15; // number of extra bits
if bits < op {
let __v118: *const u8 = r#in;
r#in = __v118.add(1);
hold = hold.wrapping_add((*__v118 as u64) << bits);
bits = bits.wrapping_add(8);
if bits < op {
let __v111: *const u8 = r#in;
r#in = __v111.add(1);
hold = hold.wrapping_add((*__v111 as u64) << bits);
bits = bits.wrapping_add(8);
}
}
dist = dist.wrapping_add((hold as u32) & (1u32 << op).wrapping_sub(1));
hold >>= op;
bits = bits.wrapping_sub(op);
op = (out.offset_from(beg as *mut u8) as i64) as u32; // max distance in output
if dist > op {
// see if copy from window
op = dist.wrapping_sub(op); // distance back in window
if op > whave {
break '__join_25;
} else {
from = window;
if write == 0 {
// very common case
let __v39: *mut u8 = from;
from = __v39.offset(wsize.wrapping_sub(op) as isize);
if op < len {
// some from window
len = len.wrapping_sub(op);
loop {
let __v35: *mut u8 = from;
from = __v35.add(1);
let __v33: *mut u8 = out;
out = __v33.add(1);
*__v33 = *__v35;
let __v30: u32 = op.wrapping_sub(1);
op = __v30;
if __v30 == 0 {
break;
}
}
from = out.sub(dist as usize); // rest from output
}
} else {
if write < op {
// wrap around window
let __v59: *mut u8 = from;
from = __v59.offset(wsize.wrapping_add(write).wrapping_sub(op) as isize);
op = op.wrapping_sub(write);
if op < len {
// some from end of window
len = len.wrapping_sub(op);
loop {
let __v53: *mut u8 = from;
from = __v53.add(1);
let __v51: *mut u8 = out;
out = __v51.add(1);
*__v51 = *__v53;
let __v48: u32 = op.wrapping_sub(1);
op = __v48;
if __v48 == 0 {
break;
}
}
from = window;
if write < len {
// some from start of window
op = write;
len = len.wrapping_sub(op);
loop {
let __v45: *mut u8 = from;
from = __v45.add(1);
let __v43: *mut u8 = out;
out = __v43.add(1);
*__v43 = *__v45;
let __v40: u32 = op.wrapping_sub(1);
op = __v40;
if __v40 == 0 {
break;
}
}
from = out.sub(dist as usize); // rest from output
}
}
} else {
// contiguous in window
let __v69: *mut u8 = from;
from = __v69.offset(write.wrapping_sub(op) as isize);
if op < len {
// some from window
len = len.wrapping_sub(op);
loop {
let __v65: *mut u8 = from;
from = __v65.add(1);
let __v63: *mut u8 = out;
out = __v63.add(1);
*__v63 = *__v65;
let __v60: u32 = op.wrapping_sub(1);
op = __v60;
if __v60 == 0 {
break;
}
}
from = out.sub(dist as usize); // rest from output
}
}
}
while len > 2 {
let __v29: *mut u8 = from;
from = __v29.add(1);
let __v27: *mut u8 = out;
out = __v27.add(1);
*__v27 = *__v29;
let __v25: *mut u8 = from;
from = __v25.add(1);
let __v23: *mut u8 = out;
out = __v23.add(1);
*__v23 = *__v25;
let __v21: *mut u8 = from;
from = __v21.add(1);
let __v19: *mut u8 = out;
out = __v19.add(1);
*__v19 = *__v21;
len = len.wrapping_sub(3);
}
if len != 0 {
let __v15: *mut u8 = from;
from = __v15.add(1);
let __v13: *mut u8 = out;
out = __v13.add(1);
*__v13 = *__v15;
if len > 1 {
let __v11: *mut u8 = from;
from = __v11.add(1);
let __v9: *mut u8 = out;
out = __v9.add(1);
*__v9 = *__v11;
break '__join_1;
} else {
break '__join_1;
}
} else {
break '__join_1;
}
}
} else {
from = out.sub(dist as usize); // copy direct from output
// minimum length is three
//
// Align out addr
if (out.sub(1) as i64) & 1 == 0 {
let __v96: *mut u8 = from;
from = __v96.add(1);
let __v94: *mut u8 = out;
out = __v94.add(1);
*__v94 = *__v96;
len = len.wrapping_sub(1);
}
sout = out as *mut u16;
if dist > 2 {
sfrom = from as *mut u16;
loops = (len >> 1i32) as u64;
loop {
let __v79: *mut u16 = sfrom;
sfrom = __v79.add(1);
let __v77: *mut u16 = sout;
sout = __v77.add(1);
*__v77 = *__v79;
let __v74: u64 = loops.wrapping_sub(1);
loops = __v74;
if __v74 == 0 {
break;
}
}
out = sout as *mut u8;
from = sfrom as *mut u8;
} else {
// dist == 1 or dist == 2
pat16 = *sout.sub(1);
if dist == 1 {
// copy one char pattern to both bytes
(*__slate_slot_89).us = pat16;
(*__slate_slot_89).b[0] = (*__slate_slot_89).b[1];
pat16 = (*__slate_slot_89).us;
}
loops = (len >> 1i32) as u64;
loop {
let __v88: *mut u16 = sout;
sout = __v88.add(1);
*__v88 = pat16;
let __v85: u64 = loops.wrapping_sub(1);
loops = __v85;
if __v85 == 0 {
break;
}
}
out = sout as *mut u8;
}
if len & 1 != 0 {
let __v73: *mut u8 = from;
from = __v73.add(1);
let __v71: *mut u8 = out;
out = __v71.add(1);
*__v71 = *__v73;
break '__join_1;
} else {
break '__join_1;
}
}
}
// literal
let __v7: *mut u8 = out;
out = __v7.add(1);
*__v7 = (*__slate_slot_176).val as u8;
}
if !(r#in < last && out < end) {
break '__join_0;
}
}
if op & 32 != 0 {
// end-of-block
(*state).mode = 11;
break '__join_0;
} else {
(*strm).msg = c"invalid literal/length code".as_ptr() as *mut i8;
(*state).mode = 27;
break '__join_0;
}
}
(*strm).msg = c"invalid distance code".as_ptr() as *mut i8;
(*state).mode = 27;
break '__join_0;
}
(*strm).msg = c"invalid distance too far back".as_ptr() as *mut i8;
(*state).mode = 27;
}
// return unused bytes (on entry, bits < 8, so in won't go too far back)
len = bits >> 3i32;
let __v5: *const u8 = r#in;
r#in = __v5.sub(len as usize);
bits = bits.wrapping_sub(len << 3i32);
hold &= (1u32 << bits).wrapping_sub(1) as u64;
// update state and return
(*strm).next_in = r#in;
(*strm).next_out = out;
(*strm).avail_in = ((if r#in < last { 5 + (last.offset_from(r#in as *const u8) as i64) } else { 5 - (r#in.offset_from(last as *const u8) as i64) }) as u32) as u64;
(*strm).avail_out = ((if out < end { 257 + (end.offset_from(out as *mut u8) as i64) } else { 257 - (out.offset_from(end as *mut u8) as i64) }) as u32) as u64;
(*state).hold = hold;
(*state).bits = bits;
return;
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct inflate_workspace {
    inflate_state: inflate_state,
    working_window: [u8; 32768],
}


#[unsafe(no_mangle)]
extern "C" fn zlib_inflate_workspacesize() -> i32 {
core::mem::size_of::<inflate_workspace>() as i32
}

#[unsafe(export_name = "malloc")]
unsafe extern "C" fn __slate_export_malloc(size: usize) -> *mut core::ffi::c_void {
malloc(size as i32) as *mut core::ffi::c_void
}

#[unsafe(no_mangle)]
extern "C" fn zlib_inflateReset(strm: *mut z_stream_s) -> i32 {
if strm.is_null() || unsafe { (*strm).state }.is_null() {
return -2;
}
let state: *mut inflate_state = (unsafe { (*strm).state }) as *mut inflate_state;
unsafe {
(*state).total = 0;
}
unsafe {
(*strm).total_out = 0;
}
unsafe {
(*strm).total_in = 0;
}
unsafe {
(*strm).msg = core::ptr::null_mut::<i8>();
}
unsafe {
(*strm).adler = 1;
}
// to support ill-conceived Java test suite
unsafe {
(*state).mode = 0;
}
unsafe {
(*state).last = 0;
}
unsafe {
(*state).havedict = 0;
}
unsafe {
(*state).dmax = 32768;
}
unsafe {
(*state).hold = 0;
}
unsafe {
(*state).bits = 0;
}
let __v0: *mut __SlateRecord_inftrees_h_31 = unsafe { (*state).codes.as_mut_ptr() as *mut __SlateRecord_inftrees_h_31 };
unsafe {
(*state).next = __v0;
}
let __v1: *const __SlateRecord_inftrees_h_31 = __v0 as *const __SlateRecord_inftrees_h_31;
unsafe {
(*state).distcode = __v1;
}
unsafe {
(*state).lencode = __v1;
}
// Initialise Window
unsafe {
(*state).wsize = 1u32 << (*state).wbits;
}
unsafe {
(*state).write = 0;
}
unsafe {
(*state).whave = 0;
}
INFLATE_RESET_HOOK!();
0
}

#[unsafe(export_name = "free")]
unsafe extern "C" fn __slate_export_free(r#where: *mut core::ffi::c_void) {
free(r#where as *mut ());
}

#[unsafe(no_mangle)]
extern "C" fn zlib_inflateInit2(strm: *mut z_stream_s, mut windowBits: i32) -> i32 {
if strm.is_null() {
return -2;
}
unsafe {
(*strm).msg = core::ptr::null_mut::<i8>();
}
// in case we return an error
let state: *mut inflate_state = unsafe { core::ptr::addr_of_mut!((*WS!(strm)).inflate_state) };
unsafe {
(*strm).state = state as *mut internal_state;
}
if windowBits < 0 {
unsafe {
(*state).wrap = 0;
}
windowBits = -windowBits;
} else {
unsafe {
(*state).wrap = (windowBits >> 4i32) + 1;
}
}
if windowBits < 8 || windowBits > 15 {
return -2;
}
unsafe {
(*state).wbits = windowBits as u32;
}
unsafe {
(*state).window = ((*WS!(strm)).working_window.as_mut_ptr() as *mut u8).add(0);
}
zlib_inflateReset(strm)
}

fn zlib_fixedtables(state: *mut inflate_state) {
// inffixed.h -- table for decoding fixed codes
// Generated automatically by makefixed().
// WARNING: this file should *not* be used by applications. It
// is part of the implementation of the compression library and
// is subject to change. Applications should only use zlib.h.
unsafe {
(*state).lencode = core::ptr::addr_of!(lenfix.0) as *const __SlateRecord_inftrees_h_31;
}
unsafe {
(*state).lenbits = 9;
}
unsafe {
(*state).distcode = core::ptr::addr_of!(distfix.0) as *const __SlateRecord_inftrees_h_31;
}
unsafe {
(*state).distbits = 5;
}
}

static lenfix: __SlateAlign16<[__SlateRecord_inftrees_h_31; 512]> = __SlateAlign16([__SlateRecord_inftrees_h_31 { op: 96u8, bits: 7u8, val: 0u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 80u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 16u16 }, __SlateRecord_inftrees_h_31 { op: 20u8, bits: 8u8, val: 115u16 }, __SlateRecord_inftrees_h_31 { op: 18u8, bits: 7u8, val: 31u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 112u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 48u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 192u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 10u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 96u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 32u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 160u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 0u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 128u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 64u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 224u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 6u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 88u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 24u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 144u16 }, __SlateRecord_inftrees_h_31 { op: 19u8, bits: 7u8, val: 59u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 120u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 56u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 208u16 }, __SlateRecord_inftrees_h_31 { op: 17u8, bits: 7u8, val: 17u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 104u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 40u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 176u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 8u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 136u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 72u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 240u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 4u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 84u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 20u16 }, __SlateRecord_inftrees_h_31 { op: 21u8, bits: 8u8, val: 227u16 }, __SlateRecord_inftrees_h_31 { op: 19u8, bits: 7u8, val: 43u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 116u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 52u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 200u16 }, __SlateRecord_inftrees_h_31 { op: 17u8, bits: 7u8, val: 13u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 100u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 36u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 168u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 4u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 132u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 68u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 232u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 8u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 92u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 28u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 152u16 }, __SlateRecord_inftrees_h_31 { op: 20u8, bits: 7u8, val: 83u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 124u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 60u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 216u16 }, __SlateRecord_inftrees_h_31 { op: 18u8, bits: 7u8, val: 23u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 108u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 44u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 184u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 12u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 140u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 76u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 248u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 3u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 82u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 18u16 }, __SlateRecord_inftrees_h_31 { op: 21u8, bits: 8u8, val: 163u16 }, __SlateRecord_inftrees_h_31 { op: 19u8, bits: 7u8, val: 35u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 114u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 50u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 196u16 }, __SlateRecord_inftrees_h_31 { op: 17u8, bits: 7u8, val: 11u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 98u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 34u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 164u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 2u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 130u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 66u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 228u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 7u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 90u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 26u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 148u16 }, __SlateRecord_inftrees_h_31 { op: 20u8, bits: 7u8, val: 67u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 122u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 58u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 212u16 }, __SlateRecord_inftrees_h_31 { op: 18u8, bits: 7u8, val: 19u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 106u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 42u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 180u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 10u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 138u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 74u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 244u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 5u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 86u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 22u16 }, __SlateRecord_inftrees_h_31 { op: 64u8, bits: 8u8, val: 0u16 }, __SlateRecord_inftrees_h_31 { op: 19u8, bits: 7u8, val: 51u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 118u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 54u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 204u16 }, __SlateRecord_inftrees_h_31 { op: 17u8, bits: 7u8, val: 15u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 102u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 38u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 172u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 6u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 134u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 70u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 236u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 9u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 94u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 30u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 156u16 }, __SlateRecord_inftrees_h_31 { op: 20u8, bits: 7u8, val: 99u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 126u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 62u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 220u16 }, __SlateRecord_inftrees_h_31 { op: 18u8, bits: 7u8, val: 27u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 110u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 46u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 188u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 14u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 142u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 78u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 252u16 }, __SlateRecord_inftrees_h_31 { op: 96u8, bits: 7u8, val: 0u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 81u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 17u16 }, __SlateRecord_inftrees_h_31 { op: 21u8, bits: 8u8, val: 131u16 }, __SlateRecord_inftrees_h_31 { op: 18u8, bits: 7u8, val: 31u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 113u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 49u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 194u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 10u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 97u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 33u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 162u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 1u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 129u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 65u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 226u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 6u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 89u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 25u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 146u16 }, __SlateRecord_inftrees_h_31 { op: 19u8, bits: 7u8, val: 59u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 121u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 57u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 210u16 }, __SlateRecord_inftrees_h_31 { op: 17u8, bits: 7u8, val: 17u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 105u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 41u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 178u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 9u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 137u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 73u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 242u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 4u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 85u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 21u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 8u8, val: 258u16 }, __SlateRecord_inftrees_h_31 { op: 19u8, bits: 7u8, val: 43u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 117u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 53u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 202u16 }, __SlateRecord_inftrees_h_31 { op: 17u8, bits: 7u8, val: 13u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 101u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 37u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 170u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 5u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 133u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 69u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 234u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 8u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 93u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 29u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 154u16 }, __SlateRecord_inftrees_h_31 { op: 20u8, bits: 7u8, val: 83u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 125u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 61u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 218u16 }, __SlateRecord_inftrees_h_31 { op: 18u8, bits: 7u8, val: 23u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 109u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 45u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 186u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 13u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 141u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 77u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 250u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 3u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 83u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 19u16 }, __SlateRecord_inftrees_h_31 { op: 21u8, bits: 8u8, val: 195u16 }, __SlateRecord_inftrees_h_31 { op: 19u8, bits: 7u8, val: 35u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 115u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 51u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 198u16 }, __SlateRecord_inftrees_h_31 { op: 17u8, bits: 7u8, val: 11u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 99u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 35u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 166u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 3u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 131u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 67u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 230u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 7u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 91u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 27u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 150u16 }, __SlateRecord_inftrees_h_31 { op: 20u8, bits: 7u8, val: 67u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 123u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 59u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 214u16 }, __SlateRecord_inftrees_h_31 { op: 18u8, bits: 7u8, val: 19u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 107u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 43u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 182u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 11u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 139u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 75u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 246u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 5u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 87u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 23u16 }, __SlateRecord_inftrees_h_31 { op: 64u8, bits: 8u8, val: 0u16 }, __SlateRecord_inftrees_h_31 { op: 19u8, bits: 7u8, val: 51u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 119u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 55u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 206u16 }, __SlateRecord_inftrees_h_31 { op: 17u8, bits: 7u8, val: 15u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 103u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 39u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 174u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 7u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 135u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 71u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 238u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 9u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 95u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 31u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 158u16 }, __SlateRecord_inftrees_h_31 { op: 20u8, bits: 7u8, val: 99u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 127u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 63u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 222u16 }, __SlateRecord_inftrees_h_31 { op: 18u8, bits: 7u8, val: 27u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 111u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 47u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 190u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 15u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 143u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 79u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 254u16 }, __SlateRecord_inftrees_h_31 { op: 96u8, bits: 7u8, val: 0u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 80u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 16u16 }, __SlateRecord_inftrees_h_31 { op: 20u8, bits: 8u8, val: 115u16 }, __SlateRecord_inftrees_h_31 { op: 18u8, bits: 7u8, val: 31u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 112u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 48u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 193u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 10u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 96u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 32u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 161u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 0u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 128u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 64u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 225u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 6u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 88u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 24u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 145u16 }, __SlateRecord_inftrees_h_31 { op: 19u8, bits: 7u8, val: 59u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 120u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 56u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 209u16 }, __SlateRecord_inftrees_h_31 { op: 17u8, bits: 7u8, val: 17u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 104u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 40u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 177u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 8u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 136u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 72u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 241u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 4u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 84u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 20u16 }, __SlateRecord_inftrees_h_31 { op: 21u8, bits: 8u8, val: 227u16 }, __SlateRecord_inftrees_h_31 { op: 19u8, bits: 7u8, val: 43u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 116u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 52u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 201u16 }, __SlateRecord_inftrees_h_31 { op: 17u8, bits: 7u8, val: 13u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 100u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 36u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 169u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 4u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 132u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 68u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 233u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 8u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 92u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 28u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 153u16 }, __SlateRecord_inftrees_h_31 { op: 20u8, bits: 7u8, val: 83u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 124u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 60u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 217u16 }, __SlateRecord_inftrees_h_31 { op: 18u8, bits: 7u8, val: 23u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 108u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 44u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 185u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 12u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 140u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 76u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 249u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 3u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 82u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 18u16 }, __SlateRecord_inftrees_h_31 { op: 21u8, bits: 8u8, val: 163u16 }, __SlateRecord_inftrees_h_31 { op: 19u8, bits: 7u8, val: 35u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 114u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 50u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 197u16 }, __SlateRecord_inftrees_h_31 { op: 17u8, bits: 7u8, val: 11u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 98u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 34u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 165u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 2u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 130u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 66u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 229u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 7u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 90u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 26u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 149u16 }, __SlateRecord_inftrees_h_31 { op: 20u8, bits: 7u8, val: 67u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 122u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 58u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 213u16 }, __SlateRecord_inftrees_h_31 { op: 18u8, bits: 7u8, val: 19u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 106u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 42u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 181u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 10u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 138u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 74u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 245u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 5u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 86u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 22u16 }, __SlateRecord_inftrees_h_31 { op: 64u8, bits: 8u8, val: 0u16 }, __SlateRecord_inftrees_h_31 { op: 19u8, bits: 7u8, val: 51u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 118u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 54u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 205u16 }, __SlateRecord_inftrees_h_31 { op: 17u8, bits: 7u8, val: 15u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 102u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 38u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 173u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 6u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 134u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 70u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 237u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 9u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 94u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 30u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 157u16 }, __SlateRecord_inftrees_h_31 { op: 20u8, bits: 7u8, val: 99u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 126u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 62u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 221u16 }, __SlateRecord_inftrees_h_31 { op: 18u8, bits: 7u8, val: 27u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 110u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 46u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 189u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 14u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 142u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 78u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 253u16 }, __SlateRecord_inftrees_h_31 { op: 96u8, bits: 7u8, val: 0u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 81u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 17u16 }, __SlateRecord_inftrees_h_31 { op: 21u8, bits: 8u8, val: 131u16 }, __SlateRecord_inftrees_h_31 { op: 18u8, bits: 7u8, val: 31u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 113u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 49u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 195u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 10u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 97u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 33u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 163u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 1u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 129u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 65u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 227u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 6u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 89u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 25u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 147u16 }, __SlateRecord_inftrees_h_31 { op: 19u8, bits: 7u8, val: 59u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 121u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 57u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 211u16 }, __SlateRecord_inftrees_h_31 { op: 17u8, bits: 7u8, val: 17u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 105u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 41u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 179u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 9u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 137u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 73u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 243u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 4u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 85u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 21u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 8u8, val: 258u16 }, __SlateRecord_inftrees_h_31 { op: 19u8, bits: 7u8, val: 43u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 117u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 53u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 203u16 }, __SlateRecord_inftrees_h_31 { op: 17u8, bits: 7u8, val: 13u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 101u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 37u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 171u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 5u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 133u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 69u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 235u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 8u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 93u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 29u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 155u16 }, __SlateRecord_inftrees_h_31 { op: 20u8, bits: 7u8, val: 83u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 125u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 61u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 219u16 }, __SlateRecord_inftrees_h_31 { op: 18u8, bits: 7u8, val: 23u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 109u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 45u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 187u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 13u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 141u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 77u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 251u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 3u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 83u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 19u16 }, __SlateRecord_inftrees_h_31 { op: 21u8, bits: 8u8, val: 195u16 }, __SlateRecord_inftrees_h_31 { op: 19u8, bits: 7u8, val: 35u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 115u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 51u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 199u16 }, __SlateRecord_inftrees_h_31 { op: 17u8, bits: 7u8, val: 11u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 99u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 35u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 167u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 3u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 131u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 67u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 231u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 7u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 91u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 27u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 151u16 }, __SlateRecord_inftrees_h_31 { op: 20u8, bits: 7u8, val: 67u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 123u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 59u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 215u16 }, __SlateRecord_inftrees_h_31 { op: 18u8, bits: 7u8, val: 19u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 107u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 43u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 183u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 11u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 139u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 75u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 247u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 5u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 87u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 23u16 }, __SlateRecord_inftrees_h_31 { op: 64u8, bits: 8u8, val: 0u16 }, __SlateRecord_inftrees_h_31 { op: 19u8, bits: 7u8, val: 51u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 119u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 55u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 207u16 }, __SlateRecord_inftrees_h_31 { op: 17u8, bits: 7u8, val: 15u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 103u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 39u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 175u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 7u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 135u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 71u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 239u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 7u8, val: 9u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 95u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 31u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 159u16 }, __SlateRecord_inftrees_h_31 { op: 20u8, bits: 7u8, val: 99u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 127u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 63u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 223u16 }, __SlateRecord_inftrees_h_31 { op: 18u8, bits: 7u8, val: 27u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 111u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 47u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 191u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 15u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 143u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 8u8, val: 79u16 }, __SlateRecord_inftrees_h_31 { op: 0u8, bits: 9u8, val: 255u16 }]);

static distfix: __SlateAlign16<[__SlateRecord_inftrees_h_31; 32]> = __SlateAlign16([__SlateRecord_inftrees_h_31 { op: 16u8, bits: 5u8, val: 1u16 }, __SlateRecord_inftrees_h_31 { op: 23u8, bits: 5u8, val: 257u16 }, __SlateRecord_inftrees_h_31 { op: 19u8, bits: 5u8, val: 17u16 }, __SlateRecord_inftrees_h_31 { op: 27u8, bits: 5u8, val: 4097u16 }, __SlateRecord_inftrees_h_31 { op: 17u8, bits: 5u8, val: 5u16 }, __SlateRecord_inftrees_h_31 { op: 25u8, bits: 5u8, val: 1025u16 }, __SlateRecord_inftrees_h_31 { op: 21u8, bits: 5u8, val: 65u16 }, __SlateRecord_inftrees_h_31 { op: 29u8, bits: 5u8, val: 16385u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 5u8, val: 3u16 }, __SlateRecord_inftrees_h_31 { op: 24u8, bits: 5u8, val: 513u16 }, __SlateRecord_inftrees_h_31 { op: 20u8, bits: 5u8, val: 33u16 }, __SlateRecord_inftrees_h_31 { op: 28u8, bits: 5u8, val: 8193u16 }, __SlateRecord_inftrees_h_31 { op: 18u8, bits: 5u8, val: 9u16 }, __SlateRecord_inftrees_h_31 { op: 26u8, bits: 5u8, val: 2049u16 }, __SlateRecord_inftrees_h_31 { op: 22u8, bits: 5u8, val: 129u16 }, __SlateRecord_inftrees_h_31 { op: 64u8, bits: 5u8, val: 0u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 5u8, val: 2u16 }, __SlateRecord_inftrees_h_31 { op: 23u8, bits: 5u8, val: 385u16 }, __SlateRecord_inftrees_h_31 { op: 19u8, bits: 5u8, val: 25u16 }, __SlateRecord_inftrees_h_31 { op: 27u8, bits: 5u8, val: 6145u16 }, __SlateRecord_inftrees_h_31 { op: 17u8, bits: 5u8, val: 7u16 }, __SlateRecord_inftrees_h_31 { op: 25u8, bits: 5u8, val: 1537u16 }, __SlateRecord_inftrees_h_31 { op: 21u8, bits: 5u8, val: 97u16 }, __SlateRecord_inftrees_h_31 { op: 29u8, bits: 5u8, val: 24577u16 }, __SlateRecord_inftrees_h_31 { op: 16u8, bits: 5u8, val: 4u16 }, __SlateRecord_inftrees_h_31 { op: 24u8, bits: 5u8, val: 769u16 }, __SlateRecord_inftrees_h_31 { op: 20u8, bits: 5u8, val: 49u16 }, __SlateRecord_inftrees_h_31 { op: 28u8, bits: 5u8, val: 12289u16 }, __SlateRecord_inftrees_h_31 { op: 18u8, bits: 5u8, val: 13u16 }, __SlateRecord_inftrees_h_31 { op: 26u8, bits: 5u8, val: 3073u16 }, __SlateRecord_inftrees_h_31 { op: 22u8, bits: 5u8, val: 193u16 }, __SlateRecord_inftrees_h_31 { op: 64u8, bits: 5u8, val: 0u16 }]);

fn zlib_updatewindow(strm: *mut z_stream_s, out: u32) {
let mut copy: u32 = 0;
let mut dist: u32 = 0;
let state: *mut inflate_state = (unsafe { (*strm).state }) as *mut inflate_state;
// copy state->wsize or less output bytes into the circular window
copy = (out as u64).wrapping_sub(unsafe { (*strm).avail_out }) as u32;
if copy >= unsafe { (*state).wsize } {
memcpy__2!(unsafe { (*state).window }, unsafe { unsafe { (*strm).next_out }.offset(-((*state).wsize as isize)) }, unsafe { (*state).wsize });
unsafe {
(*state).write = 0;
}
unsafe {
(*state).whave = (*state).wsize;
}
} else {
dist = unsafe { (*state).wsize }.wrapping_sub(unsafe { (*state).write });
if dist > copy {
dist = copy;
}
memcpy__2!(unsafe { unsafe { (*state).window }.offset((*state).write as isize) }, unsafe { unsafe { (*strm).next_out }.sub(copy as usize) }, dist);
copy = copy.wrapping_sub(dist);
if copy != 0 {
memcpy__2!(unsafe { (*state).window }, unsafe { unsafe { (*strm).next_out }.sub(copy as usize) }, copy);
unsafe {
(*state).write = copy;
}
unsafe {
(*state).whave = (*state).wsize;
}
} else {
let __v3: u32 = unsafe { (*state).write };
unsafe {
(*state).write = __v3.wrapping_add(dist);
}
if (unsafe { (*state).write }) == unsafe { (*state).wsize } {
unsafe {
(*state).write = 0;
}
}
if (unsafe { (*state).whave }) < unsafe { (*state).wsize } {
let __v6: u32 = unsafe { (*state).whave };
unsafe {
(*state).whave = __v6.wrapping_add(dist);
}
}
}
}
}

fn zlib_inflateSyncPacket(strm: *mut z_stream_s) -> i32 {
if strm.is_null() || unsafe { (*strm).state }.is_null() {
return -2;
}
let state: *mut inflate_state = (unsafe { (*strm).state }) as *mut inflate_state;
if ((unsafe { (*state).mode }) as u32) == 13 && (unsafe { (*state).bits }) == 0 {
unsafe {
(*state).mode = 11;
}
return 0;
}
-3
}

// Get a byte of input into the bit accumulator, or return from inflate()
// if there is no input available.
// Assure that there are at least n bits in the bit accumulator.  If there is
// not enough available input to do that, then return from inflate().
#[unsafe(no_mangle)]
extern "C" fn zlib_inflate(strm: *mut z_stream_s, flush: i32) -> i32 {
// return code
// permutation of code lengths
let mut ret: i32 = unsafe { core::mem::zeroed() };
// length to copy for repeats, bits to drop
let mut len: u32 = unsafe { core::mem::zeroed() };
// parent table entry
let mut last: core::mem::MaybeUninit<__SlateRecord_inftrees_h_31> = core::mem::MaybeUninit::uninit();
let __slate_slot_238: *mut __SlateRecord_inftrees_h_31 = core::ptr::addr_of_mut!(last) as *mut __SlateRecord_inftrees_h_31; // current decoding table entry
let mut this: core::mem::MaybeUninit<__SlateRecord_inftrees_h_31> = core::mem::MaybeUninit::uninit();
let __slate_slot_239: *mut __SlateRecord_inftrees_h_31 = core::ptr::addr_of_mut!(this) as *mut __SlateRecord_inftrees_h_31; // where to copy match bytes from
let mut from: *mut u8 = unsafe { core::mem::zeroed() };
// number of stored or match bytes to copy
let mut copy: u32 = unsafe { core::mem::zeroed() };
// save starting available input and output
let mut out: u32 = unsafe { core::mem::zeroed() };
let mut r#in: u32 = unsafe { core::mem::zeroed() };
// bits in bit buffer
let mut bits: u32 = unsafe { core::mem::zeroed() };
// bit buffer
let mut hold: u64 = unsafe { core::mem::zeroed() };
// available input and output
let mut left: u32 = unsafe { core::mem::zeroed() };
let mut have: u32 = unsafe { core::mem::zeroed() };
// next output
let mut put: *mut u8 = unsafe { core::mem::zeroed() };
// next input
let mut next: *const u8 = unsafe { core::mem::zeroed() };
let mut state: *mut inflate_state = unsafe { core::mem::zeroed() };
unsafe {
// Do not check for strm->next_out == NULL here as ppc zImage
// inflates to strm->next_out = 0
if strm.is_null() || unsafe { (*strm).state }.is_null() || unsafe { (*strm).next_in }.is_null() && (*strm).avail_in != 0 {
return -2;
} else {
state = (*strm).state as *mut inflate_state;
if ((*state).mode as u32) == 11 {
(*state).mode = 12;
}
// skip check
LOAD!(put, strm, left, next, have, hold, state, bits);
r#in = have;
out = left;
ret = 0;
'__join_8: {
'__join_12: {
'__join_13: {
'__join_14: {
'__join_11: {
'__loop_194: loop {
'__join_170: {
'__join_172: {
'__join_175: {
'__join_150: {
'__join_39: {
'__join_48: {
'__join_60: {
'__join_67: {
'__join_85: {
'__join_92: {
let __t1: u32 = (*state).mode as u32;
if __t1 == 0 {
if (*state).wrap == 0 {
(*state).mode = 12;
continue '__loop_194;
} else {
'__loop_189: loop {
if bits < 16 {
loop {
if have == 0 {
break '__join_8;
} else {
have = have.wrapping_sub(1);
let __v233: *const u8 = next;
next = __v233.add(1);
hold = hold.wrapping_add((*__v233 as u64) << bits);
bits = bits.wrapping_add(8);
if true {
continue '__loop_189;
}
}
}
} else {
if true {
break;
}
}
}
if ((BITS!(8i32, hold) << 8i32) as u64).wrapping_add(hold >> 8i32) % 31 != 0 {
(*strm).msg = c"incorrect header check".as_ptr() as *mut i8;
(*state).mode = 27;
continue '__loop_194;
} else {
if BITS!(4i32, hold) != 8 {
(*strm).msg = c"unknown compression method".as_ptr() as *mut i8;
(*state).mode = 27;
continue '__loop_194;
} else {
DROPBITS!(4i32, hold, bits);
len = BITS!(4i32, hold).wrapping_add(8);
if len > (*state).wbits {
(*strm).msg = c"invalid window size".as_ptr() as *mut i8;
(*state).mode = 27;
continue '__loop_194;
} else {
(*state).dmax = 1u32 << len;
let __v226: u64 = zlib_adler32(0u64, core::ptr::null::<u8>(), 0u32);
(*state).check = __v226;
(*strm).adler = __v226;
(*state).mode = (if hold & 512 != 0 { 9i32 } else { 11i32 }) as u32;
INITBITS!(hold, bits);
continue '__loop_194;
}
}
}
}
} else {
if __t1 == 9 {
'__loop_178: loop {
if bits < 32 {
loop {
if have == 0 {
break '__join_8;
} else {
have = have.wrapping_sub(1);
let __v223: *const u8 = next;
next = __v223.add(1);
hold = hold.wrapping_add((*__v223 as u64) << bits);
bits = bits.wrapping_add(8);
if true {
continue '__loop_178;
}
}
}
} else {
if true {
break;
}
}
}
let __v216: u64 = REVERSE!(hold);
(*state).check = __v216;
(*strm).adler = __v216;
INITBITS!(hold, bits);
(*state).mode = 10;
break '__join_175;
} else {
if __t1 == 10 {
break '__join_175;
} else {
if __t1 == 11 {
break '__join_172;
} else {
if __t1 == 12 {
break '__join_170;
} else {
if __t1 == 13 {
BYTEBITS!(hold, bits); // go to byte boundary
'__loop_155: loop {
if bits < 32 {
loop {
if have == 0 {
break '__join_8;
} else {
have = have.wrapping_sub(1);
let __v203: *const u8 = next;
next = __v203.add(1);
hold = hold.wrapping_add((*__v203 as u64) << bits);
bits = bits.wrapping_add(8);
if true {
continue '__loop_155;
}
}
}
} else {
if true {
break;
}
}
}
if hold & 65535 != hold >> 16i32 ^ 65535 {
(*strm).msg = c"invalid stored block lengths".as_ptr() as *mut i8;
(*state).mode = 27;
continue '__loop_194;
} else {
(*state).length = (hold as u32) & 65535;
INITBITS!(hold, bits);
(*state).mode = 14;
break '__join_150;
}
} else {
if __t1 == 14 {
break '__join_150;
} else {
if __t1 == 15 {
'__loop_140: loop {
if bits < 14 {
loop {
if have == 0 {
break '__join_8;
} else {
have = have.wrapping_sub(1);
let __v183: *const u8 = next;
next = __v183.add(1);
hold = hold.wrapping_add((*__v183 as u64) << bits);
bits = bits.wrapping_add(8);
if true {
continue '__loop_140;
}
}
}
} else {
if true {
break;
}
}
}
(*state).nlen = BITS!(5i32, hold).wrapping_add(257);
DROPBITS!(5i32, hold, bits);
(*state).ndist = BITS!(5i32, hold).wrapping_add(1);
DROPBITS!(5i32, hold, bits);
(*state).ncode = BITS!(4i32, hold).wrapping_add(4);
DROPBITS!(4i32, hold, bits);
if (*state).nlen > 286 || (*state).ndist > 30 {
(*strm).msg = c"too many length or distance symbols".as_ptr() as *mut i8;
(*state).mode = 27;
continue '__loop_194;
} else {
(*state).have = 0;
(*state).mode = 16;
}
} else {
if __t1 != 16 {
if __t1 == 17 {
break '__join_92;
} else {
if __t1 == 18 {
break '__join_85;
} else {
if __t1 == 19 {
break '__join_67;
} else {
if __t1 == 20 {
break '__join_60;
} else {
if __t1 == 21 {
break '__join_48;
} else {
if __t1 == 22 {
break '__join_39;
} else {
if __t1 == 23 {
if left == 0 {
break '__join_8;
} else {
let __v35: *mut u8 = put;
put = __v35.add(1);
*__v35 = (*state).length as u8;
left = left.wrapping_sub(1);
(*state).mode = 18;
continue '__loop_194;
}
} else {
if __t1 == 24 {
if (*state).wrap != 0 {
'__loop_20: loop {
if bits < 32 {
loop {
if have == 0 {
break '__join_8;
} else {
have = have.wrapping_sub(1);
let __v29: *const u8 = next;
next = __v29.add(1);
hold = hold.wrapping_add((*__v29 as u64) << bits);
bits = bits.wrapping_add(8);
if true {
continue '__loop_20;
}
}
}
} else {
if true {
break;
}
}
}
out = out.wrapping_sub(left);
let __v19: u64 = (*strm).total_out;
(*strm).total_out = __v19.wrapping_add(out as u64);
let __v16: u64 = (*state).total;
(*state).total = __v16.wrapping_add(out as u64);
if INFLATE_NEED_CHECKSUM!() != 0 && out != 0 {
let __v14: u64 = UPDATE!(unsafe { (*state).check }, put.sub(out as usize), out);
(*state).check = __v14;
(*strm).adler = __v14;
}
out = left;
if REVERSE!(hold) != (*state).check {
(*strm).msg = c"incorrect data check".as_ptr() as *mut i8;
(*state).mode = 27;
continue '__loop_194;
} else {
break '__join_14;
}
} else {
break '__join_13;
}
} else {
if __t1 == 26 {
break '__join_12;
} else {
if __t1 == 27 {
break '__join_11;
} else {
if __t1 == 28 {
return -4;
} else {
if __t1 == 29 {
break '__loop_194;
} else {
break '__loop_194;
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
}
}
}
}
}
}
}
}
}
while (*state).have < (*state).ncode {
'__loop_133: loop {
if bits < 3 {
loop {
if have == 0 {
break '__join_8;
} else {
have = have.wrapping_sub(1);
let __v174: *const u8 = next;
next = __v174.add(1);
hold = hold.wrapping_add((*__v174 as u64) << bits);
bits = bits.wrapping_add(8);
if true {
continue '__loop_133;
}
}
}
} else {
if true {
break;
}
}
}
let __v166: u32 = (*state).have;
(*state).have = __v166.wrapping_add(1);
*((*state).lens.as_mut_ptr() as *mut u16).offset((order.0[__v166 as usize] as i32) as isize) = BITS!(3i32, hold) as u16;
DROPBITS!(3i32, hold, bits);
}
while (*state).have < 19 {
let __v163: u32 = (*state).have;
(*state).have = __v163.wrapping_add(1);
*((*state).lens.as_mut_ptr() as *mut u16).offset((order.0[__v163 as usize] as i32) as isize) = 0;
}
(*state).next = (*state).codes.as_mut_ptr() as *mut __SlateRecord_inftrees_h_31;
(*state).lencode = (*state).next as *const __SlateRecord_inftrees_h_31;
(*state).lenbits = 7;
ret = zlib_inflate_table(0u32, (*state).lens.as_mut_ptr() as *mut u16, 19u32, core::ptr::addr_of_mut!((*state).next), core::ptr::addr_of_mut!((*state).lenbits), (*state).work.as_mut_ptr() as *mut u16);
if ret != 0 {
(*strm).msg = c"invalid code lengths set".as_ptr() as *mut i8;
(*state).mode = 27;
continue '__loop_194;
} else {
(*state).have = 0;
(*state).mode = 17;
}
}
'__join_91: {
'__join_103: {
'__loop_92: loop {
if (*state).have < unsafe { (*state).nlen }.wrapping_add((*state).ndist) {
'__loop_124: loop {
*__slate_slot_239 = *unsafe { (*state).lencode }.offset(BITS!(unsafe { (*state).lenbits }, hold) as isize);
if ((*__slate_slot_239).bits as u32) <= bits {
break;
} else {
loop {
if have == 0 {
break '__join_8;
} else {
have = have.wrapping_sub(1);
let __v159: *const u8 = next;
next = __v159.add(1);
hold = hold.wrapping_add((*__v159 as u64) << bits);
bits = bits.wrapping_add(8);
if true {
continue '__loop_124;
}
}
}
}
}
if ((*__slate_slot_239).val as i32) < 16 {
'__loop_95: loop {
if bits < ((*__slate_slot_239).bits as u32) {
loop {
if have == 0 {
break '__join_8;
} else {
have = have.wrapping_sub(1);
let __v118: *const u8 = next;
next = __v118.add(1);
hold = hold.wrapping_add((*__v118 as u64) << bits);
bits = bits.wrapping_add(8);
if true {
continue '__loop_95;
}
}
}
} else {
if true {
break;
}
}
}
DROPBITS__2!((*__slate_slot_239).bits, hold, bits);
let __v110: u32 = (*state).have;
(*state).have = __v110.wrapping_add(1);
*((*state).lens.as_mut_ptr() as *mut u16).add(__v110 as usize) = (*__slate_slot_239).val;
} else {
if ((*__slate_slot_239).val as i32) == 16 {
'__loop_106: loop {
if bits < ((((*__slate_slot_239).bits as i32) + 2) as u32) {
loop {
if have == 0 {
break '__join_8;
} else {
have = have.wrapping_sub(1);
let __v132: *const u8 = next;
next = __v132.add(1);
hold = hold.wrapping_add((*__v132 as u64) << bits);
bits = bits.wrapping_add(8);
if true {
continue '__loop_106;
}
}
}
} else {
if true {
break;
}
}
}
DROPBITS__2!((*__slate_slot_239).bits, hold, bits);
if (*state).have == 0 {
break '__join_103;
} else {
len = *((*state).lens.as_mut_ptr() as *mut u16).offset(unsafe { (*state).have }.wrapping_sub(1) as isize) as u32;
copy = 3u32.wrapping_add(BITS!(2i32, hold));
DROPBITS!(2i32, hold, bits);
}
} else {
if ((*__slate_slot_239).val as i32) == 17 {
'__loop_111: loop {
if bits < ((((*__slate_slot_239).bits as i32) + 3) as u32) {
loop {
if have == 0 {
break '__join_8;
} else {
have = have.wrapping_sub(1);
let __v141: *const u8 = next;
next = __v141.add(1);
hold = hold.wrapping_add((*__v141 as u64) << bits);
bits = bits.wrapping_add(8);
if true {
continue '__loop_111;
}
}
}
} else {
if true {
break;
}
}
}
DROPBITS__2!((*__slate_slot_239).bits, hold, bits);
len = 0;
copy = 3u32.wrapping_add(BITS!(3i32, hold));
DROPBITS!(3i32, hold, bits);
} else {
'__loop_116: loop {
if bits < ((((*__slate_slot_239).bits as i32) + 7) as u32) {
loop {
if have == 0 {
break '__join_8;
} else {
have = have.wrapping_sub(1);
let __v150: *const u8 = next;
next = __v150.add(1);
hold = hold.wrapping_add((*__v150 as u64) << bits);
bits = bits.wrapping_add(8);
if true {
continue '__loop_116;
}
}
}
} else {
if true {
break;
}
}
}
DROPBITS__2!((*__slate_slot_239).bits, hold, bits);
len = 0;
copy = 11u32.wrapping_add(BITS!(7i32, hold));
DROPBITS!(7i32, hold, bits);
}
}
if unsafe { (*state).have }.wrapping_add(copy) > unsafe { (*state).nlen }.wrapping_add((*state).ndist) {
break;
} else {
loop {
let __v122: u32 = copy;
copy = __v122.wrapping_sub(1);
if __v122 != 0 {
let __v124: u32 = (*state).have;
(*state).have = __v124.wrapping_add(1);
*((*state).lens.as_mut_ptr() as *mut u16).add(__v124 as usize) = len as u16;
} else {
continue '__loop_92;
}
}
}
}
} else {
break '__join_91;
}
}
(*strm).msg = c"invalid bit length repeat".as_ptr() as *mut i8;
(*state).mode = 27;
break '__join_91;
}
(*strm).msg = c"invalid bit length repeat".as_ptr() as *mut i8;
(*state).mode = 27;
}
// handle error breaks in while
if ((*state).mode as u32) == 27 {
continue '__loop_194;
} else {
// build code tables
(*state).next = (*state).codes.as_mut_ptr() as *mut __SlateRecord_inftrees_h_31;
(*state).lencode = (*state).next as *const __SlateRecord_inftrees_h_31;
(*state).lenbits = 9;
ret = zlib_inflate_table(1u32, (*state).lens.as_mut_ptr() as *mut u16, (*state).nlen, core::ptr::addr_of_mut!((*state).next), core::ptr::addr_of_mut!((*state).lenbits), (*state).work.as_mut_ptr() as *mut u16);
if ret != 0 {
(*strm).msg = c"invalid literal/lengths set".as_ptr() as *mut i8;
(*state).mode = 27;
continue '__loop_194;
} else {
(*state).distcode = (*state).next as *const __SlateRecord_inftrees_h_31;
(*state).distbits = 6;
ret = zlib_inflate_table(2u32, ((*state).lens.as_mut_ptr() as *mut u16).offset((*state).nlen as isize), (*state).ndist, core::ptr::addr_of_mut!((*state).next), core::ptr::addr_of_mut!((*state).distbits), (*state).work.as_mut_ptr() as *mut u16);
if ret != 0 {
(*strm).msg = c"invalid distances set".as_ptr() as *mut i8;
(*state).mode = 27;
continue '__loop_194;
} else {
(*state).mode = 18;
}
}
}
}
if have >= 6 && left >= 258 {
RESTORE!(strm, put, left, next, have, state, hold, bits);
inflate_fast(strm, out);
LOAD!(put, strm, left, next, have, hold, state, bits);
continue '__loop_194;
} else {
'__loop_83: loop {
*__slate_slot_239 = *unsafe { (*state).lencode }.offset(BITS!(unsafe { (*state).lenbits }, hold) as isize);
if ((*__slate_slot_239).bits as u32) <= bits {
break;
} else {
loop {
if have == 0 {
break '__join_8;
} else {
have = have.wrapping_sub(1);
let __v106: *const u8 = next;
next = __v106.add(1);
hold = hold.wrapping_add((*__v106 as u64) << bits);
bits = bits.wrapping_add(8);
if true {
continue '__loop_83;
}
}
}
}
}
if (*__slate_slot_239).op != 0 && ((*__slate_slot_239).op as i32) & 240 == 0 {
*__slate_slot_238 = *__slate_slot_239;
'__loop_78: loop {
*__slate_slot_239 = *unsafe { (*state).lencode }.offset((((*__slate_slot_238).val as i32) as u32).wrapping_add(BITS!(((*__slate_slot_238).bits as i32) + ((*__slate_slot_238).op as i32), hold) >> ((*__slate_slot_238).bits as i32)) as isize);
if ((((*__slate_slot_238).bits as i32) + ((*__slate_slot_239).bits as i32)) as u32) <= bits {
break;
} else {
loop {
if have == 0 {
break '__join_8;
} else {
have = have.wrapping_sub(1);
let __v97: *const u8 = next;
next = __v97.add(1);
hold = hold.wrapping_add((*__v97 as u64) << bits);
bits = bits.wrapping_add(8);
if true {
continue '__loop_78;
}
}
}
}
}
DROPBITS__2!((*__slate_slot_238).bits, hold, bits);
}
DROPBITS__2!((*__slate_slot_239).bits, hold, bits);
(*state).length = (*__slate_slot_239).val as u32;
if ((*__slate_slot_239).op as i32) == 0 {
(*state).mode = 23;
continue '__loop_194;
} else {
if ((*__slate_slot_239).op as i32) & 32 != 0 {
(*state).mode = 11;
continue '__loop_194;
} else {
if ((*__slate_slot_239).op as i32) & 64 != 0 {
(*strm).msg = c"invalid literal/length code".as_ptr() as *mut i8;
(*state).mode = 27;
continue '__loop_194;
} else {
(*state).extra = ((*__slate_slot_239).op as u32) & 15;
(*state).mode = 19;
}
}
}
}
}
if (*state).extra != 0 {
'__loop_64: loop {
if bits < ((*state).extra as u32) {
loop {
if have == 0 {
break '__join_8;
} else {
have = have.wrapping_sub(1);
let __v88: *const u8 = next;
next = __v88.add(1);
hold = hold.wrapping_add((*__v88 as u64) << bits);
bits = bits.wrapping_add(8);
if true {
continue '__loop_64;
}
}
}
} else {
if true {
break;
}
}
}
let __v80: u32 = (*state).length;
let __v79: u32 = __v80.wrapping_add(BITS!(unsafe { (*state).extra }, hold));
(*state).length = __v79;
DROPBITS!(unsafe { (*state).extra }, hold, bits);
}
(*state).mode = 20;
}
'__loop_60: loop {
*__slate_slot_239 = *unsafe { (*state).distcode }.offset(BITS!(unsafe { (*state).distbits }, hold) as isize);
if ((*__slate_slot_239).bits as u32) <= bits {
break;
} else {
loop {
if have == 0 {
break '__join_8;
} else {
have = have.wrapping_sub(1);
let __v76: *const u8 = next;
next = __v76.add(1);
hold = hold.wrapping_add((*__v76 as u64) << bits);
bits = bits.wrapping_add(8);
if true {
continue '__loop_60;
}
}
}
}
}
if ((*__slate_slot_239).op as i32) & 240 == 0 {
*__slate_slot_238 = *__slate_slot_239;
'__loop_55: loop {
*__slate_slot_239 = *unsafe { (*state).distcode }.offset((((*__slate_slot_238).val as i32) as u32).wrapping_add(BITS!(((*__slate_slot_238).bits as i32) + ((*__slate_slot_238).op as i32), hold) >> ((*__slate_slot_238).bits as i32)) as isize);
if ((((*__slate_slot_238).bits as i32) + ((*__slate_slot_239).bits as i32)) as u32) <= bits {
break;
} else {
loop {
if have == 0 {
break '__join_8;
} else {
have = have.wrapping_sub(1);
let __v67: *const u8 = next;
next = __v67.add(1);
hold = hold.wrapping_add((*__v67 as u64) << bits);
bits = bits.wrapping_add(8);
if true {
continue '__loop_55;
}
}
}
}
}
DROPBITS__2!((*__slate_slot_238).bits, hold, bits);
}
DROPBITS__2!((*__slate_slot_239).bits, hold, bits);
if ((*__slate_slot_239).op as i32) & 64 != 0 {
(*strm).msg = c"invalid distance code".as_ptr() as *mut i8;
(*state).mode = 27;
continue '__loop_194;
} else {
(*state).offset = (*__slate_slot_239).val as u32;
(*state).extra = ((*__slate_slot_239).op as u32) & 15;
(*state).mode = 21;
}
}
if (*state).extra != 0 {
'__loop_45: loop {
if bits < ((*state).extra as u32) {
loop {
if have == 0 {
break '__join_8;
} else {
have = have.wrapping_sub(1);
let __v58: *const u8 = next;
next = __v58.add(1);
hold = hold.wrapping_add((*__v58 as u64) << bits);
bits = bits.wrapping_add(8);
if true {
continue '__loop_45;
}
}
}
} else {
if true {
break;
}
}
}
let __v50: u32 = (*state).offset;
let __v49: u32 = __v50.wrapping_add(BITS!(unsafe { (*state).extra }, hold));
(*state).offset = __v49;
DROPBITS!(unsafe { (*state).extra }, hold, bits);
}
if (*state).offset > unsafe { (*state).whave }.wrapping_add(out).wrapping_sub(left) {
(*strm).msg = c"invalid distance too far back".as_ptr() as *mut i8;
(*state).mode = 27;
continue '__loop_194;
} else {
(*state).mode = 22;
}
}
if left == 0 {
break '__join_8;
} else {
copy = out.wrapping_sub(left);
if (*state).offset > copy {
// copy from window
copy = unsafe { (*state).offset }.wrapping_sub(copy);
if copy > (*state).write {
copy = copy.wrapping_sub((*state).write);
from = unsafe { (*state).window }.offset(unsafe { (*state).wsize }.wrapping_sub(copy) as isize);
} else {
from = unsafe { (*state).window }.offset(unsafe { (*state).write }.wrapping_sub(copy) as isize);
}
if copy > (*state).length {
copy = (*state).length;
}
} else {
// copy from output
from = put.offset(-((*state).offset as isize));
copy = (*state).length;
}
if copy > left {
copy = left;
}
left = left.wrapping_sub(copy);
let __v43: u32 = (*state).length;
(*state).length = __v43.wrapping_sub(copy);
loop {
let __v41: *mut u8 = from;
from = __v41.add(1);
let __v39: *mut u8 = put;
put = __v39.add(1);
*__v39 = *__v41;
let __v36: u32 = copy.wrapping_sub(1);
copy = __v36;
if __v36 == 0 {
break;
}
}
if (*state).length == 0 {
(*state).mode = 18;
continue '__loop_194;
} else {
continue '__loop_194;
}
}
}
copy = (*state).length;
if copy != 0 {
if copy > have {
copy = have;
}
if copy > left {
copy = left;
}
if copy == 0 {
break '__join_8;
} else {
memcpy__2!(put, next, copy);
have = have.wrapping_sub(copy);
let __v194: *const u8 = next;
next = __v194.add(copy as usize);
left = left.wrapping_sub(copy);
let __v190: *mut u8 = put;
put = __v190.add(copy as usize);
let __v187: u32 = (*state).length;
(*state).length = __v187.wrapping_sub(copy);
continue '__loop_194;
}
} else {
(*state).mode = 11;
continue '__loop_194;
}
}
if (*state).havedict == 0 {
RESTORE!(strm, put, left, next, have, state, hold, bits);
return 2;
} else {
let __v215: u64 = zlib_adler32(0u64, core::ptr::null::<u8>(), 0u32);
(*state).check = __v215;
(*strm).adler = __v215;
(*state).mode = 11;
}
}
if flush == 6 {
break '__join_8;
}
}
INFLATE_TYPEDO_HOOK!();
if (*state).last != 0 {
BYTEBITS!(hold, bits);
(*state).mode = 24;
} else {
'__loop_166: loop {
if bits < 3 {
loop {
if have == 0 {
break '__join_8;
} else {
have = have.wrapping_sub(1);
let __v212: *const u8 = next;
next = __v212.add(1);
hold = hold.wrapping_add((*__v212 as u64) << bits);
bits = bits.wrapping_add(8);
if true {
continue '__loop_166;
}
}
}
} else {
if true {
break;
}
}
}
(*state).last = BITS!(1i32, hold) as i32;
DROPBITS!(1i32, hold, bits);
let __t0: u32 = BITS!(2i32, hold);
if __t0 == 0 {
(*state).mode = 13;
// stored block
} else {
if __t0 == 1 {
zlib_fixedtables(state); // fixed block
(*state).mode = 18;
// decode codes
} else {
if __t0 == 2 {
(*state).mode = 15;
// dynamic block
} else {
if __t0 == 3 {
(*strm).msg = c"invalid block type".as_ptr() as *mut i8;
(*state).mode = 27;
}
}
}
}
DROPBITS!(2i32, hold, bits);
}
}
return -2;
}
ret = -3;
break '__join_8;
}
INITBITS!(hold, bits);
}
(*state).mode = 26;
}
ret = 1;
}
RESTORE!(strm, put, left, next, have, state, hold, bits);
if INFLATE_NEED_UPDATEWINDOW!() != 0 && ((*state).wsize != 0 || ((*state).mode as u32) < 24 && (out as u64) != (*strm).avail_out) {
zlib_updatewindow(strm, out);
}
r#in = (r#in as u64).wrapping_sub((*strm).avail_in) as u32;
out = (out as u64).wrapping_sub((*strm).avail_out) as u32;
let __v8: u64 = (*strm).total_in;
(*strm).total_in = __v8.wrapping_add(r#in as u64);
let __v5: u64 = (*strm).total_out;
(*strm).total_out = __v5.wrapping_add(out as u64);
let __v2: u64 = (*state).total;
(*state).total = __v2.wrapping_add(out as u64);
if INFLATE_NEED_CHECKSUM!() != 0 && (*state).wrap != 0 && out != 0 {
let __v0: u64 = UPDATE!(unsafe { (*state).check }, unsafe { (*strm).next_out }.sub(out as usize), out);
(*state).check = __v0;
(*strm).adler = __v0;
}
(*strm).data_type = unsafe { (*state).bits }.wrapping_add((if (*state).last != 0 { 64i32 } else { 0i32 }) as u32).wrapping_add((if ((*state).mode as u32) == 11 { 128i32 } else { 0i32 }) as u32) as i32;
if flush == 2 && ret == 0 && (*strm).avail_out != 0 && (*strm).avail_in == 0 {
return zlib_inflateSyncPacket(strm);
} else {
if (r#in == 0 && out == 0 || flush == 5) && ret == 0 {
ret = -5;
}
return ret;
}
}
}
// Return from inflate(), updating the total counts and the check value.
// If there was no progress during the inflate() call, return a buffer
// error.  Call zlib_updatewindow() to create and/or update the window state.
}

static order: __SlateAlign16<[u16; 19]> = __SlateAlign16([16u16, 17u16, 18u16, 0u16, 8u16, 7u16, 9u16, 6u16, 10u16, 5u16, 11u16, 4u16, 12u16, 3u16, 13u16, 2u16, 14u16, 1u16, 15u16]);

#[unsafe(no_mangle)]
extern "C" fn zlib_inflateEnd(strm: *mut z_stream_s) -> i32 {
if strm.is_null() || unsafe { (*strm).state }.is_null() {
return -2;
}
0
}

#[unsafe(no_mangle)]
extern "C" fn zlib_inflateIncomp(z: *mut z_stream_s) -> i32 {
let state: *mut inflate_state = (unsafe { (*z).state }) as *mut inflate_state;
let saved_no: *mut u8 = unsafe { (*z).next_out };
let saved_ao: u32 = (unsafe { (*z).avail_out }) as u32;
if ((unsafe { (*state).mode }) as u32) != 11 && ((unsafe { (*state).mode }) as u32) != 0 {
return -3;
}
// Setup some variables to allow misuse of updateWindow
unsafe {
(*z).avail_out = 0;
}
unsafe {
(*z).next_out = ((*z).next_in as *mut u8).offset((*z).avail_in as isize);
}
zlib_updatewindow(z, (unsafe { (*z).avail_in }) as u32);
// Restore saved variables
unsafe {
(*z).avail_out = saved_ao as u64;
}
unsafe {
(*z).next_out = saved_no;
}
let __v0: u64 = UPDATE__2!(unsafe { (*state).check }, unsafe { (*z).next_in }, unsafe { (*z).avail_in });
unsafe {
(*state).check = __v0;
}
unsafe {
(*z).adler = __v0;
}
let __v2: u64 = unsafe { (*z).total_out };
unsafe {
(*z).total_out = __v2.wrapping_add((*z).avail_in);
}
let __v5: u64 = unsafe { (*z).total_in };
unsafe {
(*z).total_in = __v5.wrapping_add((*z).avail_in);
}
let __v8: *const u8 = unsafe { (*z).next_in };
unsafe {
(*z).next_in = __v8.offset((*z).avail_in as isize);
}
let __v11: u64 = unsafe { (*state).total };
unsafe {
(*state).total = __v11.wrapping_add((*z).avail_in);
}
unsafe {
(*z).avail_in = 0;
}
0
}

#[unsafe(link_section = ".text.slate_distinct.misc.nofill")]
extern "C" fn nofill(buffer: *mut (), len: u64) -> i64 {
-1
}

fn __gunzip(buf: *mut u8, mut len: i64, mut fill: Option<unsafe extern "C" fn(*mut (), u64) -> i64>, flush: Option<unsafe extern "C" fn(*mut (), u64) -> i64>, mut out_buf: *mut u8, mut out_len: i64, pos: *mut i64, error: Option<unsafe extern "C" fn(*mut i8)>) -> i32 {
let mut __v7: u64 = unsafe { core::mem::zeroed() };
let mut rc: i32 = unsafe { core::mem::zeroed() };
let mut strm: *mut z_stream_s = unsafe { core::mem::zeroed() };
let mut zbuf: *mut u8 = unsafe { core::mem::zeroed() };
unsafe {
rc = -1;
if flush != None {
out_len = 32768; // 32 K
out_buf = malloc(out_len as i32) as *mut u8;
} else {
if out_len == 0 {
out_len = 18446744073709551615u64.wrapping_sub(out_buf as u64) as i64;
}
// no limit
}
if out_buf.is_null() {
error.unwrap()(c"Out of memory while allocating output buffer".as_ptr() as *mut i8);
} else {
if !buf.is_null() {
zbuf = buf;
} else {
zbuf = malloc(16384i32) as *mut u8;
len = 0;
}
if zbuf.is_null() {
error.unwrap()(c"Out of memory while allocating input buffer".as_ptr() as *mut i8);
} else {
strm = malloc(core::mem::size_of::<z_stream_s>() as i32) as *mut z_stream_s;
if strm.is_null() {
error.unwrap()(c"Out of memory while allocating z_stream".as_ptr() as *mut i8);
} else {
if flush != None {
__v7 = zlib_inflate_workspacesize() as u64;
} else {
__v7 = core::mem::size_of::<inflate_state>() as u64;
}
(*strm).workspace = malloc(__v7 as i32);
if unsafe { (*strm).workspace }.is_null() {
error.unwrap()(c"Out of memory while allocating workspace".as_ptr() as *mut i8);
} else {
if fill == None {
fill = Some(nofill);
}
if len == 0 {
len = fill.unwrap()(zbuf as *mut (), 16384u64);
}
'__join_6: {
// verify the gzip header
if len < 10 || (*zbuf.add(0) as i32) != 31 || (*zbuf.add(1) as i32) != 139 || (*zbuf.add(2) as i32) != 8 {
if !pos.is_null() {
*pos = 0;
}
error.unwrap()(c"Not a gzip file".as_ptr() as *mut i8);
} else {
'__join_24: {
// skip over gzip header (1f,8b,08... 10 bytes total +
// possible asciz filename)
(*strm).next_in = zbuf.add(10) as *const u8;
(*strm).avail_in = (len - 10) as u64;
// skip over asciz filename
if (*zbuf.add(3) as i32) & 8 != 0 {
loop {
// If the filename doesn't fit into the buffer,
// the file is very probably corrupt. Don't try
// to read more data.
if (*strm).avail_in == 0 {
break;
} else {
let __v5: u64 = (*strm).avail_in;
(*strm).avail_in = __v5.wrapping_sub(1);
let __v2: *const u8 = (*strm).next_in;
(*strm).next_in = __v2.add(1);
if *__v2 == 0 {
break '__join_24;
}
}
}
error.unwrap()(c"header error".as_ptr() as *mut i8);
break '__join_6;
}
}
(*strm).next_out = out_buf;
(*strm).avail_out = out_len as u64;
rc = zlib_inflateInit2(strm, -15i32);
if flush == None {
(*WS!(strm)).inflate_state.wsize = 0;
(*WS!(strm)).inflate_state.window = core::ptr::null_mut::<u8>();
}
'__join_9: {
'__join_20: {
'__join_16: {
loop {
if rc == 0 {
if (*strm).avail_in == 0 {
// TODO: handle case where both pos and fill are set
len = fill.unwrap()(zbuf as *mut (), 16384u64);
if len < 0 {
break '__join_20;
} else {
(*strm).next_in = zbuf as *const u8;
(*strm).avail_in = len as u64;
}
}
rc = zlib_inflate(strm, 0i32);
// Write any data generated
if flush != None && (*strm).next_out > out_buf {
let l: i64 = unsafe { (*strm).next_out }.offset_from(out_buf as *mut u8) as i64;
if l != flush.unwrap()(out_buf as *mut (), l as u64) {
break '__join_16;
} else {
(*strm).next_out = out_buf;
(*strm).avail_out = out_len as u64;
}
}
// after Z_FINISH, only Z_STREAM_END is "we unpacked it all"
if rc == 1 {
break;
} else {
if rc != 0 {
error.unwrap()(c"uncompression error".as_ptr() as *mut i8);
rc = -1;
}
}
} else {
break '__join_9;
}
}
rc = 0;
break '__join_9;
}
rc = -1;
error.unwrap()(c"write error".as_ptr() as *mut i8);
break '__join_9;
}
rc = -1;
error.unwrap()(c"read error".as_ptr() as *mut i8);
}
zlib_inflateEnd(strm);
if !pos.is_null() {
*pos = (unsafe { (*strm).next_in }.offset_from(zbuf as *const u8) as i64) + 8;
}
// add + 8 to skip over trailer
}
}
free((*strm).workspace);
}
free(strm as *mut ());
}
if buf.is_null() {
free(zbuf as *mut ());
}
}
if flush != None {
free(out_buf as *mut ());
}
}
return rc;
}
// returns Z_OK (0) if successful
}

fn __decompress(buf: *mut u8, len: i64, fill: Option<unsafe extern "C" fn(*mut (), u64) -> i64>, flush: Option<unsafe extern "C" fn(*mut (), u64) -> i64>, out_buf: *mut u8, out_len: i64, pos: *mut i64, error: Option<unsafe extern "C" fn(*mut i8)>) -> i32 {
__gunzip(buf, len, fill, flush, out_buf, out_len, pos, error)
}

// NOTE: When adding a new decompressor, please update the analysis in
// ../header.S.
fn scroll() {
let mut i: i32 = 0;
unsafe { memmove(vidmem as *mut (), unsafe { vidmem }.offset((cols * 2) as isize) as *const (), ((lines - 1) * cols * 2) as u64) };
i = ((unsafe { lines }) - 1) * unsafe { cols } * 2;
while i < (unsafe { lines }) * unsafe { cols } * 2 {
unsafe {
*unsafe { vidmem }.offset(i as isize) = 32;
}
i += 2;
}
}

//  Transmit register (WRITE)
//  Line Status
fn serial_putchar(ch: i32) {
let mut timeout: u32 = 65535;
loop {
let __v0: bool = if ((unsafe { unsafe { pio_ops.f_inb }.unwrap()((early_serial_base + 5) as u16) }) as i32) & 32 == 0 {
let __v2: u32 = timeout.wrapping_sub(1);
timeout = __v2;
__v2 != 0
} else {
false
};
if !__v0 {
break;
}
cpu_relax();
}
unsafe { unsafe { pio_ops.f_outb }.unwrap()(ch as u8, (early_serial_base + 0) as u16) };
}

#[unsafe(no_mangle)]
extern "C" fn __putstr(mut s: *const i8) {
let mut x: i32 = 0;
let mut y: i32 = 0;
let mut pos: i32 = 0;
let mut c: i8 = 0;
if (unsafe { early_serial_base }) != 0 {
let mut str: *const i8 = s;
while (unsafe { *str }) != 0 {
if ((unsafe { *str }) as i32) == 10 {
serial_putchar(13i32);
}
let __v0: *const i8 = str;
str = unsafe { __v0.add(1) };
serial_putchar((unsafe { *__v0 }) as i32);
}
}
if (unsafe { lines }) == 0 || (unsafe { cols }) == 0 {
return;
}
x = (unsafe { (*boot_params_ptr).screen_info.orig_x }) as i32;
y = (unsafe { (*boot_params_ptr).screen_info.orig_y }) as i32;
loop {
let __v13: *const i8 = s;
s = unsafe { __v13.add(1) };
let __v15: i8 = unsafe { *__v13 };
c = __v15;
if (__v15 as i32) == 0 {
break;
}
if (c as i32) == 10 {
x = 0;
let __v4: i32 = y + 1;
y = __v4;
if __v4 >= unsafe { lines } {
scroll();
y -= 1;
}
} else {
unsafe {
*unsafe { vidmem }.offset(((x + cols * y) * 2) as isize) = c;
}
let __v8: i32 = x + 1;
x = __v8;
if __v8 >= unsafe { cols } {
x = 0;
let __v10: i32 = y + 1;
y = __v10;
if __v10 >= unsafe { lines } {
scroll();
y -= 1;
}
}
}
}
unsafe {
(*boot_params_ptr).screen_info.orig_x = x as u8;
}
unsafe {
(*boot_params_ptr).screen_info.orig_y = y as u8;
}
pos = (x + (unsafe { cols }) * y) * 2; // Update cursor position
unsafe { unsafe { pio_ops.f_outb }.unwrap()(14u8, vidport as u16) };
unsafe { unsafe { pio_ops.f_outb }.unwrap()((255 & pos >> 9i32) as u8, (vidport + 1) as u16) };
unsafe { unsafe { pio_ops.f_outb }.unwrap()(15u8, vidport as u16) };
unsafe { unsafe { pio_ops.f_outb }.unwrap()((255 & pos >> 1i32) as u8, (vidport + 1) as u16) };
}

#[inline(never)]
fn __putnum(mut value: u64, base: u32, mut mindig: i32) {
let mut buf: __SlateAlign16<[i8; 65]> = __SlateAlign16([0i8; 65]);
let mut p: *mut i8 = unsafe { core::mem::zeroed() };
p = unsafe { (buf.0.as_mut_ptr() as *mut i8).offset(core::mem::size_of::<[i8; 65]>() as isize) };
let __v0: *mut i8 = p;
let __v1: *mut i8 = unsafe { __v0.sub(1) };
p = __v1;
unsafe {
*__v1 = 0;
}
loop {
let __v8: i32 = mindig;
mindig = __v8 - 1;
if !(__v8 > 0 || value != 0) {
break;
}
let mut digit: u8 = (value % (base as u64)) as u8;
digit = ((digit as i32) + if (digit as i32) >= 10 { 97 - 10 } else { 48 }) as u8;
let __v4: *mut i8 = p;
let __v5: *mut i8 = unsafe { __v4.sub(1) };
p = __v5;
unsafe {
*__v5 = digit as i8;
}
value /= base as u64;
}
__putstr(p as *const i8);
}

#[unsafe(no_mangle)]
extern "C" fn __puthex(value: u64) {
__putnum(value, 16u32, (core::mem::size_of::<u64>() as u64).wrapping_mul(2) as i32);
}

#[unsafe(no_mangle)]
extern "C" fn __putdec(value: u64) {
__putnum(value, 10u32, 1i32);
}

fn handle_relocations(output: *mut (), output_len__0: u64, virt_addr: u64) {
let mut reloc: *mut i32 = unsafe { core::mem::zeroed() };
let mut delta: u64 = 0;
let mut map: u64 = 0;
let mut ptr: u64 = 0;
let min_addr: u64 = output as u64;
let max_addr: u64 = min_addr.wrapping_add(_AC__4!().wrapping_sub(_AC__2!()));
// Calculate the delta between where vmlinux was linked to load
// and where it was actually loaded.
delta = min_addr.wrapping_sub(__ALIGN_KERNEL_MASK!(16777216i32, 2097151i32) as u64);
// The kernel contains a table of relocation addresses. Those
// addresses have the final load address of the kernel in virtual
// memory. We are currently working in the self map. So we need to
// create an adjustment for kernel memory addresses to the self map.
// This will involve subtracting out the base address of the kernel.
map = delta.wrapping_sub(_AC__3!());
// 32-bit always performs relocations. 64-bit relocations are only
// needed if KASLR has chosen a different starting address offset
// from __START_KERNEL_map.
if IS_ENABLED__2!() != 0 {
delta = virt_addr.wrapping_sub(__ALIGN_KERNEL_MASK!(16777216i32, 2097151i32) as u64);
}
if delta == 0 {
debug_putstr__2!(c"No relocation needed... ".as_ptr() as *mut i8);
return;
}
debug_putstr__2!(c"Performing relocations... ".as_ptr() as *mut i8);
// Process relocations: 32 bit relocations first then 64 bit after.
// Two sets of binary relocations are added to the end of the kernel
// before compression. Each relocation table entry is the kernel
// address of the location which needs to be updated stored as a
// 32-bit value which is sign extended to 64 bits.
//
// Format is:
//
// kernel bits...
// 0 - zero terminator for 64 bit relocations
// 64 bit relocation repeated
// 0 - zero terminator for 32 bit relocations
// 32 bit relocation repeated
//
// So we work backwards from the end of the decompressed image.
reloc = ((unsafe { (((output as *const u8).add(output_len__0 as usize) as *mut ()) as *const u8).offset(-(core::mem::size_of::<i32>() as isize)) }) as *mut ()) as *mut i32;
while (unsafe { *reloc }) != 0 {
let mut extended: i64 = (unsafe { *reloc }) as i64;
extended = (extended as u64).wrapping_add(map) as i64;
ptr = extended as u64;
if ptr < min_addr || ptr > max_addr {
unsafe { error(c"32-bit relocation outside of kernel!\n".as_ptr() as *mut i8) };
unsafe { core::hint::unreachable_unchecked() };
}
unsafe {
*(ptr as *mut u32) = (*(ptr as *mut u32) as u64).wrapping_add(delta) as u32;
}
let __v6: *mut i32 = reloc;
reloc = unsafe { __v6.sub(1) };
}
let __v8: *mut i32 = reloc;
reloc = unsafe { __v8.sub(1) };
while (unsafe { *reloc }) != 0 {
let mut extended: i64 = (unsafe { *reloc }) as i64;
extended = (extended as u64).wrapping_add(map) as i64;
ptr = extended as u64;
if ptr < min_addr || ptr > max_addr {
unsafe { error(c"64-bit relocation outside of kernel!\n".as_ptr() as *mut i8) };
unsafe { core::hint::unreachable_unchecked() };
}
let __v13: u64 = unsafe { *(ptr as *mut u64) };
unsafe {
*(ptr as *mut u64) = __v13.wrapping_add(delta);
}
let __v16: *mut i32 = reloc;
reloc = unsafe { __v16.sub(1) };
}
}

fn parse_elf(output: *mut ()) -> u64 {
let mut ehdr: elf64_hdr = unsafe { core::mem::zeroed() };
let mut dest: *mut () = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
memcpy!(core::ptr::addr_of_mut!(ehdr), output, core::mem::size_of::<elf64_hdr>() as u64);
if (ehdr.e_ident[0] as i32) != 127 || (ehdr.e_ident[1] as i32) != 69 || (ehdr.e_ident[2] as i32) != 76 || (ehdr.e_ident[3] as i32) != 70 {
unsafe { error(c"Kernel is not a valid ELF file".as_ptr() as *mut i8) };
unsafe { core::hint::unreachable_unchecked() };
}
debug_putstr__2!(c"Parsing ELF... ".as_ptr() as *mut i8);
let phdrs: *mut elf64_phdr = malloc((core::mem::size_of::<elf64_phdr>() as u64).wrapping_mul((ehdr.e_phnum as i32) as u64) as i32) as *mut elf64_phdr;
if phdrs.is_null() {
unsafe { error(c"Failed to allocate space for phdrs".as_ptr() as *mut i8) };
unsafe { core::hint::unreachable_unchecked() };
}
memcpy!(phdrs, (unsafe { (output as *const u8).offset(ehdr.e_phoff as isize) }) as *mut (), (core::mem::size_of::<elf64_phdr>() as u64).wrapping_mul((ehdr.e_phnum as i32) as u64));
while i < (ehdr.e_phnum as i32) {
let phdr: *mut elf64_phdr = unsafe { phdrs.offset(i as isize) };
'__slate_break_2: {
match unsafe { (*phdr).p_type } {
1 => {
if (unsafe { (*phdr).p_align }) % 2097152 != 0 {
unsafe { error(c"Alignment of LOAD segment isn't multiple of 2MB".as_ptr() as *mut i8) };
unsafe { core::hint::unreachable_unchecked() };
}
dest = output;
let __v0: *mut () = dest;
dest = (unsafe { (__v0 as *const u8).offset(unsafe { (*phdr).p_paddr }.wrapping_sub(__ALIGN_KERNEL_MASK!(16777216i32, 2097151i32) as u64) as isize) }) as *mut ();
unsafe { memmove(dest, ((output as *const u8).offset((*phdr).p_offset as isize) as *mut ()) as *const (), (*phdr).p_filesz) };
}
_ => {
break '__slate_break_2; // Ignore other PT_*
}
}
}
i += 1;
}
free(phdrs as *mut ());
ehdr.e_entry.wrapping_sub(__ALIGN_KERNEL_MASK!(16777216i32, 2097151i32) as u64)
}

#[unsafe(no_mangle)]
static kernel_text_size: u64 = 20971520;

#[unsafe(no_mangle)]
static kernel_inittext_offset: u64 = 38240256;

#[unsafe(no_mangle)]
static kernel_inittext_size: u64 = 577536;

#[unsafe(no_mangle)]
static kernel_total_size: u64 = 42065920;

static mut boot_heap: __SlateAlign4<[u8; 65536]> = __SlateAlign4(unsafe { core::mem::zeroed() });

#[unsafe(no_mangle)]
extern "C" fn decompress_kernel(outbuf: *mut u8, virt_addr: u64, error: Option<unsafe extern "C" fn(*mut i8)>) -> u64 {
let mut entry: u64 = 0;
if (unsafe { free_mem_ptr }) == 0 {
unsafe {
free_mem_ptr = ((core::ptr::addr_of_mut!(boot_heap.0) as *mut u8) as u64) as i64;
}
unsafe {
free_mem_end_ptr = ((core::ptr::addr_of_mut!(boot_heap.0) as *mut u8) as u64).wrapping_add(core::mem::size_of::<[u8; 65536]>() as u64) as i64;
}
}
if __decompress(unsafe { core::ptr::addr_of_mut!(input_data) as *mut u8 }, (unsafe { input_len }) as i64, None, None, outbuf, (unsafe { output_len }) as i64, core::ptr::null_mut::<i64>(), error) < 0 {
return !0;
}
entry = parse_elf(outbuf as *mut ());
handle_relocations(outbuf as *mut (), (unsafe { output_len }) as u64, virt_addr);
entry
}

/// Set the memory encryption xloadflag based on the mem_encrypt= command line
/// parameter, if provided.
fn parse_mem_encrypt(hdr: *mut setup_header) {
let on: i32 = unsafe { cmdline_find_option_bool(c"mem_encrypt=on".as_ptr()) };
let off: i32 = unsafe { cmdline_find_option_bool(c"mem_encrypt=off".as_ptr()) };
if on > off {
unsafe {
(*hdr).xloadflags = (((*hdr).xloadflags as i32) | 128i32) as u16;
}
}
}

fn early_sev_detect() {
// Accessing video memory causes guest termination because
// the boot stage2 #VC handler of SEV-ES/SNP guests does not
// support MMIO handling and kexec -c adds screen_info to the
// boot parameters passed to the kexec kernel, which causes
// console output to be dumped to both video and serial.
if 0 & BIT_ULL!(1i32) != 0 {
unsafe {
cols = 0;
}
unsafe {
lines = 0;
}
}
}

/// The compressed kernel image (ZO), has been moved so that its position
/// is against the end of the buffer used to hold the uncompressed kernel
/// image (VO) and the execution environment (.bss, .brk), which makes sure
/// there is room to do the in-place decompression. (See header.S for the
/// calculations.)
///
///                             |-----compressed kernel image------|
///                             V                                  V
/// 0                       extract_offset                      +INIT_SIZE
/// |-----------|---------------|-------------------------|--------|
///             |               |                         |        |
///           VO__text      startup_32 of ZO          VO__end    ZO__end
///             ^                                         ^
///             |-------uncompressed kernel image---------|
#[unsafe(no_mangle)]
extern "C" fn extract_kernel(rmode: *mut (), mut output: *mut u8) -> *mut () {
let mut virt_addr: u64 = __ALIGN_KERNEL_MASK!(16777216i32, 2097151i32) as u64;
let heap: i64 = (unsafe { core::ptr::addr_of_mut!(boot_heap.0) as *mut u8 }) as i64;
let mut needed_size: u64 = 0;
let mut entry_offset: u64 = 0;
// Retain x86 boot parameters pointer passed from startup_32/64.
unsafe {
boot_params_ptr = rmode as *mut boot_params;
}
// Clear flags intended for solely in-kernel use.
let __v0: *mut boot_params = unsafe { boot_params_ptr };
let __v2: u8 = (((unsafe { (*__v0).hdr.loadflags }) as i32) & -3i32) as u8;
unsafe {
(*__v0).hdr.loadflags = __v2;
}
parse_mem_encrypt(unsafe { core::ptr::addr_of_mut!((*boot_params_ptr).hdr) });
sanitize_boot_params(unsafe { boot_params_ptr });
if ((unsafe { (*boot_params_ptr).screen_info.orig_video_mode }) as i32) == 7 {
unsafe {
vidmem = 720896i32 as *mut i8;
}
unsafe {
vidport = 948;
}
} else {
unsafe {
vidmem = 753664i32 as *mut i8;
}
unsafe {
vidport = 980;
}
}
unsafe {
lines = (*boot_params_ptr).screen_info.orig_video_lines as i32;
}
unsafe {
cols = (*boot_params_ptr).screen_info.orig_video_cols as i32;
}
init_default_io_ops();
// Detect TDX guest environment.
//
// It has to be done before console_init() in order to use
// paravirtualized port I/O operations if needed.
early_tdx_detect();
early_sev_detect();
unsafe { console_init() };
// Save RSDP address for later use. Have this after console_init()
// so that early debugging output from the RSDP parsing code can be
// collected.
unsafe {
(*boot_params_ptr).acpi_rsdp_addr = get_rsdp_addr();
}
debug_putstr__2!(c"early console in extract_kernel\n".as_ptr() as *mut i8);
unsafe {
free_mem_ptr = heap;
}
// Heap
unsafe {
free_mem_end_ptr = heap + 65536;
}
// The memory hole needed for the kernel is the larger of either
// the entire decompressed kernel plus relocation table, or the
// entire decompressed kernel plus .bss and .brk sections.
//
// On X86_64, the memory is mapped with PMD pages. Round the
// size up so that the full extent of PMD pages mapped is
// included in the check against the valid memory table
// entries. This ensures the full mapped area is usable RAM
// and doesn't include any reserved areas.
needed_size = max_t!(u64, unsafe { output_len }, unsafe { kernel_total_size });
needed_size = ALIGN!(needed_size, 2097152u64);
// Report initial kernel position details.
debug_putstr__2!(c"input_data: 0x".as_ptr() as *mut i8);
debug_puthex!((unsafe { core::ptr::addr_of_mut!(input_data) as *mut u8 }) as u64);
debug_putstr__2!(c"\n".as_ptr() as *mut i8);
debug_putstr__2!(c"input_len: 0x".as_ptr() as *mut i8);
debug_puthex!((unsafe { input_len }) as u64);
debug_putstr__2!(c"\n".as_ptr() as *mut i8);
debug_putstr__2!(c"output: 0x".as_ptr() as *mut i8);
debug_puthex!(output as u64);
debug_putstr__2!(c"\n".as_ptr() as *mut i8);
debug_putstr__2!(c"output_len: 0x".as_ptr() as *mut i8);
debug_puthex!((unsafe { output_len }) as u64);
debug_putstr__2!(c"\n".as_ptr() as *mut i8);
debug_putstr__2!(c"kernel_total_size: 0x".as_ptr() as *mut i8);
debug_puthex!((unsafe { kernel_total_size }) as u64);
debug_putstr__2!(c"\n".as_ptr() as *mut i8);
debug_putstr__2!(c"needed_size: 0x".as_ptr() as *mut i8);
debug_puthex!(needed_size as u64);
debug_putstr__2!(c"\n".as_ptr() as *mut i8);
// Report address of 32-bit trampoline
debug_putstr__2!(c"trampoline_32bit: 0x".as_ptr() as *mut i8);
debug_puthex!((unsafe { trampoline_32bit }) as u64);
debug_putstr__2!(c"\n".as_ptr() as *mut i8);
unsafe { choose_random_location((core::ptr::addr_of_mut!(input_data) as *mut u8) as u64, input_len as u64, core::ptr::addr_of_mut!(output) as *mut u64, needed_size, core::ptr::addr_of_mut!(virt_addr)) };
// Validate memory location choices.
if (output as u64) & (_AC!() << 21i32).wrapping_sub(1) != 0 {
unsafe { error(c"Destination physical address inappropriately aligned".as_ptr() as *mut i8) };
unsafe { core::hint::unreachable_unchecked() };
}
if virt_addr & (_AC!() << 21i32).wrapping_sub(1) != 0 {
unsafe { error(c"Destination virtual address inappropriately aligned".as_ptr() as *mut i8) };
unsafe { core::hint::unreachable_unchecked() };
}
if (heap as u64) > 70368744177663 {
unsafe { error(c"Destination address too large".as_ptr() as *mut i8) };
unsafe { core::hint::unreachable_unchecked() };
}
if virt_addr.wrapping_add(needed_size) > 1073741824u64 {
unsafe { error(c"Destination virtual address is beyond the kernel mapping area".as_ptr() as *mut i8) };
unsafe { core::hint::unreachable_unchecked() };
}
debug_putstr__2!(c"\nDecompressing Linux... ".as_ptr() as *mut i8);
if init_unaccepted_memory() {
debug_putstr__2!(c"Accepting memory... ".as_ptr() as *mut i8);
unsafe { accept_memory(__pa!(output), needed_size) };
}
entry_offset = decompress_kernel(output, virt_addr, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut i8)>>(error as *const ()) });
debug_putstr__2!(c"done.\nBooting the kernel (entry_offset: 0x".as_ptr() as *mut i8);
debug_puthex!(entry_offset);
debug_putstr__2!(c").\n".as_ptr() as *mut i8);
// Disable exception handling before booting the kernel
unsafe { cleanup_exception_handling() };
if (unsafe { spurious_nmi_count }) != 0 {
error_putstr__3!(c"Spurious early NMIs ignored: ".as_ptr() as *mut i8);
error_putdec!(unsafe { spurious_nmi_count });
error_putstr__3!(c"\n".as_ptr() as *mut i8);
}
(unsafe { output.add(entry_offset as usize) }) as *mut ()
}

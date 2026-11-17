
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
//! The type of an index into the pagecache.
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
//! abs - return absolute value of an argument
//! @x: the value.
//!
//! If it is unsigned type, @x is converted to signed type first.
//! char is treated as if it was signed (regardless of whether it really is)
//! but the macro's return type is preserved as char.
//!
//! NOTE, for signed type if @x is the minimum, the returned result is undefined
//! as there is not enough bits to represent it as a positive number.
//!
//! Return: an absolute value of @x.
//! DIV_U64_ROUND_UP - unsigned 64bit divide with 32bit divisor rounded up
//! @ll: unsigned 64bit dividend
//! @d: unsigned 32bit divisor
//!
//! Divide unsigned 64bit dividend by unsigned 32bit divisor
//! and round up.
//!
//! Return: dividend / divisor rounded up
//! Parameters used to convert the timespec values:
//! Internal kernel timer frequency
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
//! Atomic operations are already serializing on x86
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
//! ilog2 - log base 2 of 32-bit or a 64-bit unsigned value
//! @n: parameter
//!
//! constant-capable log of base 2 calculation
//! - this can be used to initialise global variables from constant data, hence
//! the massive ternary operator construction
//!
//! selects the appropriately-sized optimised version depending on sizeof(n)
//! on 4KB pages
//! Operation not permitted
//! Value too large for defined data type
//! restart by calling sys_restart_syscall
//! IS_ERR_VALUE - Detect an error pointer.
//! @x: The pointer to check.
//!
//! Like IS_ERR(), but does not generate a compiler warning if result is unused.
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
//! list_entry - get the struct for this entry
//! @ptr:	the &struct list_head pointer.
//! @type:	the type of the struct this is embedded in.
//! @member:	the name of the list_head within the struct.
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
//! Macros to retrieve the current execution context:
//!
//! in_nmi()		- We're in NMI context
//! in_hardirq()		- We're in hard IRQ context
//! in_serving_softirq()	- We're in softirq context
//! in_task()		- We're in task context
//! rescheduling necessary
//! New user namespace
//! These constants are defined in the MSan LLVM instrumentation pass.
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
//! get_task_state():
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
//! ***DANGEROUS*** don't add guard page
//! hlist_nulls_first_rcu - returns the first element of the hash list.
//! @head: the head of the list.
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
// "ibpb" Indirect Branch Prediction Barrier without a guaranteed RSB flush
// "erms" Enhanced REP MOVSB/STOSB instructions
// "clflushopt" CLFLUSHOPT instruction
// "clwb" CLWB instruction
// IA32_TSC_DEADLINE and X2APIC MSRs need fencing
// Non-serializing WRMSR
// "wbnoinvd" WBNOINVD instruction
// "la57" 5-level page tables
// LFENCE always serializing / synchronizes RDTSC
// BUG word(s)
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
// On 2MB or 1GB pages
// Extracts the PFN from a (pte|pmd|pud|pgd)val_t of a 4KB page
// Extracts the flags from a (pte|pmd|pud|pgd)val_t
// This includes the protection key value.
// No need to mask any bits for !PAE
// No such file or directory
// Try again
// Out of memory
// Device or resource busy
// Invalid argument
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
// cpumask_bits - get the bits in a cpumask
// @maskp: the struct cpumask *
//
// You should only assume nr_cpu_ids bits of this mask are valid.  This is
// a macro so it's const-correct.
// Physical address zone modifiers (see linux/mmzone.h - low four bits)
//
// Do not put any conditional on these. If necessary modify the definitions
// without the underscores and use them consistently. The definitions here may
// be used in bit comparisons.
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
// For trivial one-depth nesting of a lock-class, the following
// global define can be used. (Subsystems with multiple levels
// of nesting should define their own lock-nesting subclasses.)
// Writer mask
// Remapping rwlock architecture specific functions to the corresponding
// queued rwlock functions.
// Number of contexts where an event can trigger:
//      task, softirq, hardirq, nmi.
// I am an IDLE thread
// Kernel thread cloned from userspace thread
// I am a kernel thread
// Userland is not allowed to meddle with cpus_mask
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
// Opaque type representing current VMA (vm_area_struct) flag state. Must be
// accessed via vma_flags_xxx() helper functions.
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
macro_rules! ARRAY_SIZE__9 {
() => {
(((const { core::mem::size_of::<[ctl_table; 11]>() }) as u64) / ((const { core::mem::size_of::<ctl_table>() }) as u64)).wrapping_add(__must_be_array__50!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__35 {
() => {
((core::mem::size_of::<[ctl_table; 11]>() as u64) / (core::mem::size_of::<ctl_table>() as u64)).wrapping_add(__must_be_array__4!() as u64)
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

macro_rules! __READ_ONCE__2 {
($x:expr) => {
unsafe { core::ptr::read_volatile(core::ptr::addr_of!($x) as *const i64) }
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

macro_rules! __WRITE_ONCE__18 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut *mut *mut hlist_nulls_node, $val) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE__14 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut *mut hlist_nulls_node, $val) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE__37 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut *mut *mut hlist_nulls_node, ($val) as *mut *mut hlist_nulls_node) };
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

macro_rules! __must_be_array__4 {
() => {
core::mem::size_of::<__SlateRecord_ucount_c_113>() as i32
};
}

macro_rules! __must_be_array__50 {
() => {
((const { core::mem::size_of::<__SlateRecord_ucount_c_104>() }) as u64) as i32
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__19 {
() => {
((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__27 {
() => {
((const { core::mem::size_of::<*mut *mut hlist_nulls_node>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut *mut hlist_nulls_node>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut *mut hlist_nulls_node>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut *mut hlist_nulls_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__18 {
() => {
((const { core::mem::size_of::<*mut hlist_nulls_node>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut hlist_nulls_node>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut hlist_nulls_node>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut hlist_nulls_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

macro_rules! __or {
() => {
____or!()
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

macro_rules! arch_atomic_sub_return {
($i:expr, $v:expr) => {
arch_atomic_add_return($i.wrapping_neg(), $v)
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
macro_rules! container_of__63 {
($ptr:expr) => {
{
let mut __v: *mut ucounts;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__48!() as isize)) }) as *mut ()) as *mut ucounts;
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
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__36!($type) as isize)) }) as *mut ()) as *mut $type;
__v
}
};
}

macro_rules! hash_long {
($val:expr, $bits:expr) => {
hash_64_generic($val, ($bits) as u32)
};
}

macro_rules! hlist_nulls_entry {
($ptr:expr) => {
container_of__63!($ptr)
};
}

macro_rules! irqs_disabled {
() => {
{
let mut __v: i32;
let mut __v2: u64 = 0;
raw_local_save_flags!(__v2);
let mut __v3: i32;
typecheck__2!(u64);
__v3 = arch_irqs_disabled_flags(__v2);
__v = __v3;
__v
}
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

macro_rules! kmalloc_index {
($s:expr) => {
__kmalloc_index($s, true)
};
}

macro_rules! likely {
($x:expr) => {
!(!$x) as i64
};
}

macro_rules! offsetof__48 {
() => {
core::mem::offset_of!(ucounts, node) as u64
};
}

macro_rules! offsetof__36 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, set) as u64
};
}

macro_rules! offsetof__113 {
() => {
(const { core::mem::offset_of!(ucounts, rcu) }) as u64
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

macro_rules! raw_local_save_flags {
($flags:expr) => {
loop {
typecheck__2!(u64);
$flags = arch_local_save_flags();
if true {
break;
}
}
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
let mut __v: i32;
let mut __v2: $type = 0;
let mut __v3: $type = 0;
core::ptr::addr_of_mut!(__v2) == core::ptr::addr_of_mut!(__v3);
__v = 1;
};
}

macro_rules! ucounts_hashentry {
($ns:expr, $uid:expr) => {
unsafe { core::ptr::addr_of_mut!(ucounts_hashtable.0) as *mut hlist_nulls_head }.wrapping_offset(ucounts_hashfn!($ns, $uid) as isize)
};
}

macro_rules! ucounts_hashfn {
($ns:expr, $uid:expr) => {
hash_long!((__kuid_val($uid) as u64).wrapping_add(($ns) as u64), 10i32)
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

unsafe extern "C" {
    static mut __preempt_count: u64;
    static mut __SCK__preempt_schedule: static_call_key;
    static mut init_user_ns: user_namespace;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn __cmpxchg_wrong_size();
    fn __xadd_wrong_size();
    fn kmemdup_noprof(arg0: *const (), arg1: u64, arg2: u32) -> *mut ();
    fn _raw_spin_lock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irqsave(arg0: *mut raw_spinlock) -> u64;
    fn _raw_spin_unlock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irqrestore(arg0: *mut raw_spinlock, arg1: u64);
    fn __rcu_read_lock();
    fn __rcu_read_unlock();
    fn rcu_is_watching() -> bool;
    fn kvfree_call_rcu(arg0: *mut kvfree_rcu_head, arg1: *mut ());
    fn proc_doulongvec_minmax(arg0: *const ctl_table, arg1: i32, arg2: *mut (), arg3: *mut u64, arg4: *mut i64) -> i32;
    fn setup_sysctl_set(arg0: *mut ctl_table_set, arg1: *mut ctl_table_root, arg2: Option<unsafe extern "C" fn(*mut ctl_table_set) -> i32>);
    fn retire_sysctl_set(arg0: *mut ctl_table_set);
    fn __register_sysctl_table(arg0: *mut ctl_table_set, arg1: *const u8, arg2: *const ctl_table, arg3: u64) -> *mut ctl_table_header;
    fn register_sysctl_sz(arg0: *const u8, arg1: *const ctl_table, arg2: u64) -> *mut ctl_table_header;
    fn unregister_sysctl_table(arg0: *mut ctl_table_header);
    fn __compiletime_assert_121();
    fn __compiletime_assert_122();
    fn kfree(arg0: *const ());
    fn __compiletime_assert_208();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn ns_capable_noaudit(arg0: *mut user_namespace, arg1: i32) -> bool;
    fn __compiletime_assert_221();
    fn __compiletime_assert_222();
    fn __compiletime_assert_223();
    fn __compiletime_assert_224();
    fn __compiletime_assert_225();
    fn __compiletime_assert_226();
    fn __compiletime_assert_227();
    fn rcuref_get_slowpath(arg0: *mut __SlateRecord_types_h_202) -> bool;
    fn rcuref_put_slowpath(arg0: *mut __SlateRecord_types_h_202, arg1: u32) -> bool;
    fn __compiletime_assert_243();
    fn __compiletime_assert_244();
    fn __compiletime_assert_247();
    fn __compiletime_assert_249();
    fn __compiletime_assert_250();
    fn __compiletime_assert_251();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_a39e427d3b6c2df4(arg0: i32) -> bool;
    #[link_name = "llvm.is.constant.i64"]
    fn __slate_intrinsic_f217c4720f04f6db(arg0: u64) -> bool;
}

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


fn kasan_check_read(p: *const (), size: u32) -> bool {
true
}

fn kasan_check_write(p: *const (), size: u32) -> bool {
true
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
struct static_call_site {
    addr: i32,
    key: i32,
}


#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
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
struct raw_spinlock {
    raw_lock: qspinlock,
}


#[inline(always)]
fn instrument_read_write(v: *const (), size: u64) {
kasan_check_write(v, size as u32);
kcsan_check_read_write!(v, size);
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
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
fn arch_atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
__WRITE_ONCE__2!((*v).counter, i);
}

#[inline(always)]
fn arch_atomic_add_negative(mut i: i32, mut v: *mut __SlateRecord_types_h_190) -> bool {
let mut __v0: bool;
let mut c: bool = false;
unsafe {
let __t0: u8;
core::arch::asm!("lock addl {2:e}, ({0:r})\nsets {1}", in(reg) core::ptr::addr_of_mut!((*v).counter), lateout(reg_byte) __t0, in(reg) i as u32, options(nostack, att_syntax));
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
fn arch_atomic64_read(v: *const __SlateRecord_types_h_197) -> i64 {
__READ_ONCE__2!((*v).counter)
}

#[inline(always)]
fn arch_atomic64_dec(mut v: *mut __SlateRecord_types_h_197) {
unsafe {
core::arch::asm!("lock decq ({0:r})\n# {1}", in(reg) core::ptr::addr_of_mut!((*v).counter), in(reg) core::ptr::addr_of!((*v).counter), options(nostack, att_syntax));
}
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

#[inline(always)]
fn raw_atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
arch_atomic_set(v, i);
}

#[inline(always)]
fn raw_atomic_sub_return_release(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
arch_atomic_sub_return!(i, v)
}

#[inline(always)]
fn raw_atomic_add_negative_relaxed(i: i32, v: *mut __SlateRecord_types_h_190) -> bool {
arch_atomic_add_negative(i, v)
}

#[inline(always)]
fn raw_atomic64_read(v: *const __SlateRecord_types_h_197) -> i64 {
arch_atomic64_read(v)
}

#[inline(always)]
fn raw_atomic64_add_return(i: i64, v: *mut __SlateRecord_types_h_197) -> i64 {
arch_atomic64_add_return(i, v)
}

#[inline(always)]
fn raw_atomic64_sub_return(i: i64, v: *mut __SlateRecord_types_h_197) -> i64 {
arch_atomic64_sub_return!(i, v)
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
fn raw_atomic64_dec(v: *mut __SlateRecord_types_h_197) {
arch_atomic64_dec(v);
}

#[inline(always)]
fn raw_atomic64_try_cmpxchg(v: *mut __SlateRecord_types_h_197, old: *mut i64, new: i64) -> bool {
arch_atomic64_try_cmpxchg(v, old, new)
}

#[inline(always)]
fn raw_atomic64_dec_if_positive(v: *mut __SlateRecord_types_h_197) -> i64 {
let mut dec: i64 = 0;
let mut c: i64 = raw_atomic64_read(v as *const __SlateRecord_types_h_197);
loop {
dec = c.wrapping_sub(1);
if unlikely!(dec < 0) != 0 {
break;
}
if !(!raw_atomic64_try_cmpxchg(v, core::ptr::addr_of_mut!(c), dec)) {
break;
}
}
dec
}

#[inline(always)]
fn raw_atomic_long_read(v: *const __SlateRecord_types_h_197) -> i64 {
raw_atomic64_read(v)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
}


#[inline(always)]
fn raw_atomic_long_add_return(i: i64, v: *mut __SlateRecord_types_h_197) -> i64 {
raw_atomic64_add_return(i, v)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache {
}


#[inline(always)]
fn raw_atomic_long_sub_return(i: i64, v: *mut __SlateRecord_types_h_197) -> i64 {
raw_atomic64_sub_return(i, v)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct dentry {
}


#[inline(always)]
fn raw_atomic_long_dec(v: *mut __SlateRecord_types_h_197) {
raw_atomic64_dec(v);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array_ptr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key_type {
}


#[inline(always)]
fn raw_atomic_long_try_cmpxchg(v: *mut __SlateRecord_types_h_197, old: *mut i64, new: i64) -> bool {
raw_atomic64_try_cmpxchg(v, old as *mut i64, new)
}

#[inline(always)]
fn raw_atomic_long_dec_if_positive(v: *mut __SlateRecord_types_h_197) -> i64 {
raw_atomic64_dec_if_positive(v)
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
fn atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
instrument_atomic_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_set(v, i);
}

#[inline(always)]
fn atomic_sub_return_release(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
kcsan_release!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_sub_return_release(i, v)
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


#[inline(always)]
fn atomic_add_negative_relaxed(i: i32, v: *mut __SlateRecord_types_h_190) -> bool {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_add_negative_relaxed(i, v)
}

#[inline(always)]
fn atomic_long_read(v: *const __SlateRecord_types_h_197) -> i64 {
instrument_atomic_read(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_read(v)
}

#[inline(always)]
fn atomic_long_add_return(i: i64, v: *mut __SlateRecord_types_h_197) -> i64 {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_add_return(i, v)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct proc_ns_operations {
}


#[inline(always)]
fn atomic_long_sub_return(i: i64, v: *mut __SlateRecord_types_h_197) -> i64 {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_sub_return(i, v)
}

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct __SlateRecord_ns_common_types_h_112 {
    __ns_ref: refcount_struct,
}


#[inline(always)]
fn atomic_long_dec(v: *mut __SlateRecord_types_h_197) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_dec(v);
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
struct __SlateRecord_ucount_c_104 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_ucount_c_113 {
}


#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

#[inline(always)]
fn atomic_long_try_cmpxchg(v: *mut __SlateRecord_types_h_197, old: *mut i64, new: i64) -> bool {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
instrument_read_write(old as *const (), core::mem::size_of::<i64>() as u64);
raw_atomic_long_try_cmpxchg(v, old, new)
}

#[inline(always)]
fn atomic_long_dec_if_positive(v: *mut __SlateRecord_types_h_197) -> i64 {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_dec_if_positive(v)
}

#[inline(always)]
fn arch_local_save_flags() -> u64 {
native_save_fl()
}

#[inline(always)]
fn arch_irqs_disabled_flags(flags: u64) -> i32 {
(flags & _BITUL!(9i32) == 0) as i32
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


core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_inc_ucount: ; .asciz \"GPL\" ; .ascii \"module:\" \"binfmt_misc\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_dec_ucount: ; .asciz \"GPL\" ; .ascii \"module:\" \"binfmt_misc\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section\t\".initcall4.init\", \"a\"\t\t\n__initcall__kmod_ucount__264_368_user_namespace_sysctl_init4:\t\t\t\n.long\t{2} - .\t\n.previous\t\t\t\t\t\n", sym inc_ucount, sym dec_ucount, sym user_namespace_sysctl_init, options(att_syntax));

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


#[repr(C)]
#[derive(Clone, Copy)]
struct refcount_struct {
    refs: __SlateRecord_types_h_190,
}


// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
#[inline(always)]
fn spinlock_check(lock: *mut spinlock) -> *mut raw_spinlock {
unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) }
}

#[inline(always)]
fn spin_lock_irq(lock: *mut spinlock) {
raw_spin_lock_irq!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
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
struct work_struct {
    data: __SlateRecord_types_h_197,
    entry: list_head,
    func: Option<unsafe extern "C" fn(*mut work_struct)>,
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


fn __kuid_val(uid: __SlateRecord_uidgid_types_h_9) -> u32 {
uid.val
}

fn uid_eq(left: __SlateRecord_uidgid_types_h_9, right: __SlateRecord_uidgid_types_h_9) -> bool {
__kuid_val(left) == __kuid_val(right)
}

fn rcu_read_lock_held() -> i32 {
1
}

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
#[inline(always)]
fn rcu_read_lock() {
unsafe { __rcu_read_lock() };
__acquire_shared!();
rcu_lock_acquire!();
RCU_LOCKDEP_WARN!(!unsafe { rcu_is_watching() });
}

fn rcu_read_unlock() {
RCU_LOCKDEP_WARN!(!unsafe { rcu_is_watching() });
rcu_lock_release!(); // Keep acq info for rls diags.
__release_shared!();
unsafe { __rcu_read_unlock() };
}

// kfree_rcu() - kfree an object after a grace period.
// @ptr: pointer to kfree for double-argument invocations.
// @rhf: the name of the struct rcu_head within the type of @ptr.
//
// Many rcu callbacks functions just call kfree() on the base structure.
// These functions are trivial, but their size adds up, and furthermore
// when they are used in a kernel module, that module must invoke the
// high-latency rcu_barrier() function at module-unload time.
//
// The kfree_rcu() function handles this issue. In order to have a universal
// callback function handling different offsets of rcu_head, the callback needs
// to determine the starting address of the freed object, which can be a large
// kmalloc or vmalloc allocation. To allow simply aligning the pointer down to
// page boundary for those, only offsets up to 4095 bytes can be accommodated.
// If the offset is larger than 4095 bytes, a compile-time error will
// be generated in kvfree_rcu_arg_2(). If this error is triggered, you can
// either fall back to use of call_rcu() or rearrange the structure to
// position the rcu_head structure into the first 4096 bytes.
//
// The object to be freed can be allocated either by kmalloc(),
// kmalloc_nolock(), or kmem_cache_alloc().
//
// Note that the allowable offset might decrease in the future.
//
// The BUILD_BUG_ON check must not involve any function calls, hence the
// checks are done in macros here.
// The BUILD_BUG_ON() makes sure the rcu_head offset can be handled. See the
// comment of kfree_rcu() for details.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_rcupdate_h_1216 {
    lock: *mut (),
}


#[inline(always)]
fn class_rcu_destructor(_T: *mut __SlateRecord_rcupdate_h_1216) {
rcu_read_unlock();
}

#[inline(always)]
fn class_rcu_constructor() -> __SlateRecord_rcupdate_h_1216 {
let _t: __SlateRecord_rcupdate_h_1216 = __SlateRecord_rcupdate_h_1216 { lock: 1i32 as *mut () };
rcu_read_lock();
return _t;
}

#[repr(C)]
#[derive(Clone, Copy)]
struct wait_queue_head {
    lock: spinlock,
    head: list_head,
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
struct hlist_nulls_head {
    first: *mut hlist_nulls_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hlist_nulls_node {
    next: *mut hlist_nulls_node,
    pprev: *mut *mut hlist_nulls_node,
}


fn is_a_nulls(ptr: *const hlist_nulls_node) -> i32 {
((ptr as u64) & 1) as i32
}

fn __hlist_nulls_del(n: *mut hlist_nulls_node) {
let next: *mut hlist_nulls_node = unsafe { (*n).next };
let pprev: *mut *mut hlist_nulls_node = unsafe { (*n).pprev };
loop {
loop {
if !(__native_word__18!() || ((const { core::mem::size_of::<*mut hlist_nulls_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_121() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__14!(*pprev, next);
if true {
break;
}
}
if is_a_nulls(next as *const hlist_nulls_node) == 0 {
loop {
loop {
if !(__native_word__27!() || ((const { core::mem::size_of::<*mut *mut hlist_nulls_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_122() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__18!((*next).pprev, pprev);
if true {
break;
}
}
}
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


#[inline(always)]
fn hash_64_generic(val: u64, bits: u32) -> u32 {
// 64x64-bit multiply is efficient on all 64-bit processors
(val.wrapping_mul(7046029254386353131) >> 64u32.wrapping_sub(bits)) as u32
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
// All kzalloc aliases for kmalloc_(obj|objs|flex).
fn _kzalloc_noprof(size: u64, flags: u32, token: __SlateRecord_slab_h_532) -> *mut () {
_kmalloc_noprof(size, flags | (BIT!(8i32) as u32), token)
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


fn current_user_ns() -> *mut user_namespace {
unsafe { core::ptr::addr_of_mut!(init_user_ns) }
}

fn kmemleak_ignore(ptr: *const ()) {
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


// hlist_nulls_next_rcu - returns the element of the list after @node.
// @node: element of the list.
// hlist_nulls_pprev_rcu - returns the dereferenced pprev of @node.
// @node: element of the list.
fn hlist_nulls_del_rcu(n: *mut hlist_nulls_node) {
__hlist_nulls_del(n);
loop {
loop {
if !(__native_word__27!() || ((const { core::mem::size_of::<*mut *mut hlist_nulls_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_221() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__37!((*n).pprev, (unsafe { (290i32 as *const u8).wrapping_offset(_AC__2!() as isize) }) as *mut ());
if true {
break;
}
}
}

fn hlist_nulls_add_head_rcu(n: *mut hlist_nulls_node, h: *mut hlist_nulls_head) {
let first: *mut hlist_nulls_node = unsafe { (*h).first };
loop {
loop {
if !(__native_word__18!() || ((const { core::mem::size_of::<*mut hlist_nulls_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_222() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__14!((*n).next, first);
if true {
break;
}
}
loop {
loop {
if !(__native_word__27!() || ((const { core::mem::size_of::<*mut *mut hlist_nulls_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_223() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__18!((*n).pprev, unsafe { core::ptr::addr_of_mut!((*h).first) });
if true {
break;
}
}
let _r_a_p__v: u64 = n as u64;
if false && _r_a_p__v == 0 {
loop {
loop {
if !(__native_word__18!() || ((const { core::mem::size_of::<*mut hlist_nulls_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_224() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__14!(*unsafe { core::ptr::addr_of_mut!((*h).first) }, _r_a_p__v as *mut hlist_nulls_node);
if true {
break;
}
}
} else {
loop {
kcsan_release!();
loop {
loop {
if !__native_word__18!() {
unsafe { __compiletime_assert_225() };
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
if !(__native_word__18!() || ((const { core::mem::size_of::<*mut hlist_nulls_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_226() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__14!(*unsafe { core::ptr::addr_of_mut!((*h).first) }, RCU_INITIALIZER!(_r_a_p__v as *mut hlist_nulls_node));
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
if is_a_nulls(first as *const hlist_nulls_node) == 0 {
loop {
loop {
if !(__native_word__27!() || ((const { core::mem::size_of::<*mut *mut hlist_nulls_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_227() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__18!((*first).pprev, unsafe { core::ptr::addr_of_mut!((*n).next) });
if true {
break;
}
}
}
}

// hlist_nulls_for_each_entry_rcu - iterate over rcu list of given type
// @tpos:	the type * to use as a loop cursor.
// @pos:	the &struct hlist_nulls_node to use as a loop cursor.
// @head:	the head of the list.
// @member:	the name of the hlist_nulls_node within the struct.
//
// The barrier() is needed to make sure compiler doesn't cache first element [1],
// as this loop can be restarted [2]
// [1] Documentation/memory-barriers.txt around line 1533
// [2] Documentation/RCU/rculist_nulls.rst around line 146
fn rcuref_init(r#ref: *mut __SlateRecord_types_h_202, cnt: u32) {
atomic_set(unsafe { core::ptr::addr_of_mut!((*r#ref).refcnt) }, cnt.wrapping_sub(1) as i32);
}

fn rcuref_get(r#ref: *mut __SlateRecord_types_h_202) -> bool {
// Unconditionally increase the reference count. The saturation and
// dead zones provide enough tolerance for this.
if likely!(!atomic_add_negative_relaxed(1i32, unsafe { core::ptr::addr_of_mut!((*r#ref).refcnt) })) != 0 {
return true;
}
// Handle the cases inside the saturation and dead zones
unsafe { rcuref_get_slowpath(r#ref) }
}

#[inline(always)]
fn __rcuref_put(r#ref: *mut __SlateRecord_types_h_202) -> bool {
let mut cnt: i32 = 0;
loop {
let __v1: bool = false && (rcu_read_lock_held() == 0 && preemptible!());
if !__v1 {
break;
}
}
// Unconditionally decrease the reference count. The saturation and
// dead zones provide enough tolerance for this.
cnt = atomic_sub_return_release(1i32, unsafe { core::ptr::addr_of_mut!((*r#ref).refcnt) });
if likely!(cnt >= 0) != 0 {
return false;
}
// Handle the last reference drop and cases inside the saturation
// and dead zones.
unsafe { rcuref_put_slowpath(r#ref, cnt as u32) }
}

fn rcuref_put(r#ref: *mut __SlateRecord_types_h_202) -> bool {
let mut released: bool = false;
loop {
preempt_count_inc!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
released = __rcuref_put(r#ref);
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
released
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_242: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

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


#[unsafe(no_mangle)]
extern "C" fn setup_userns_sysctls(ns: *mut user_namespace) -> bool {
loop {
if ARRAY_SIZE__9!() != 11 {
unsafe { __compiletime_assert_244() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe { setup_sysctl_set(core::ptr::addr_of_mut!((*ns).set), core::ptr::addr_of_mut!(set_root), Some(set_is_seen)) };
let tbl: *mut ctl_table = alloc_hooks!(unsafe { kmemdup_noprof((core::ptr::addr_of!(user_table.0) as *const ctl_table) as *const (), core::mem::size_of::<[ctl_table; 11]>() as u64, 3264u32) }) as *mut ctl_table;
if !tbl.is_null() {
for i in 0i32..11i32 {
unsafe {
(*tbl.wrapping_offset(i as isize)).data = ((*ns).ucount_max.as_mut_ptr() as *mut i64).wrapping_offset(i as isize) as *mut ();
}
}
unsafe {
(*ns).sysctls = __register_sysctl_table(core::ptr::addr_of_mut!((*ns).set), c"user".as_ptr() as *const u8, tbl as *const ctl_table, ARRAY_SIZE__35!());
}
}
if unsafe { (*ns).sysctls }.is_null() {
unsafe { kfree(tbl as *const ()) };
unsafe { retire_sysctl_set(core::ptr::addr_of_mut!((*ns).set)) };
return false;
}
true
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.ucount.inc_ucount")]
extern "C" fn inc_ucount(ns: *mut user_namespace, uid: __SlateRecord_uidgid_types_h_9, r#type: u32) -> *mut ucounts {
let mut max: core::mem::MaybeUninit<i64> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut i64 = core::ptr::addr_of_mut!(max) as *mut i64;
let mut tns: core::mem::MaybeUninit<*mut user_namespace> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut user_namespace = core::ptr::addr_of_mut!(tns) as *mut *mut user_namespace;
let mut bad: core::mem::MaybeUninit<*mut ucounts> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut ucounts = core::ptr::addr_of_mut!(bad) as *mut *mut ucounts;
let mut iter: core::mem::MaybeUninit<*mut ucounts> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut ucounts = core::ptr::addr_of_mut!(iter) as *mut *mut ucounts;
let mut ucounts: core::mem::MaybeUninit<*mut ucounts> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut ucounts = core::ptr::addr_of_mut!(ucounts) as *mut *mut ucounts;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
*__slate_slot_5 = alloc_ucounts(ns, uid);
*__slate_slot_4 = *__slate_slot_5;
'__join_3: {
while !(*__slate_slot_4).is_null() {
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_2 = (*(*__slate_slot_4)).ns;
while false {
}
let __v0: i64 = core::ptr::read_volatile(((*(*__slate_slot_2)).ucount_max.as_mut_ptr() as *mut i64).wrapping_offset((r#type as u32) as isize) as *const i64);
*__slate_slot_1 = __v0;
if !atomic_long_inc_below(((*(*__slate_slot_4)).ucount.as_mut_ptr() as *mut __SlateRecord_types_h_197).wrapping_offset((r#type as u32) as isize), *__slate_slot_1) {
break '__join_3;
} else {
*__slate_slot_4 = (*(*__slate_slot_2)).ucounts;
}
}
return *__slate_slot_5;
}
*__slate_slot_3 = *__slate_slot_4;
*__slate_slot_4 = *__slate_slot_5;
while *__slate_slot_4 != *__slate_slot_3 {
atomic_long_dec(((*(*__slate_slot_4)).ucount.as_mut_ptr() as *mut __SlateRecord_types_h_197).wrapping_offset((r#type as u32) as isize));
*__slate_slot_4 = (*(*(*__slate_slot_4)).ns).ucounts;
}
put_ucounts(*__slate_slot_5);
return core::ptr::null_mut::<ucounts>();
}
}

#[unsafe(no_mangle)]
extern "C" fn put_ucounts(ucounts: *mut ucounts) {
let mut flags: u64 = 0;
if rcuref_put(unsafe { core::ptr::addr_of_mut!((*ucounts).count) }) {
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!(ucounts_lock) }, flags);
hlist_nulls_del_rcu(unsafe { core::ptr::addr_of_mut!((*ucounts).node) });
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!(ucounts_lock) }, flags);
put_user_ns(unsafe { (*ucounts).ns });
loop {
let ___p: *mut ucounts = ucounts;
if !___p.is_null() {
loop {
if offsetof__113!() >= 4096 {
unsafe { __compiletime_assert_250() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let ___head: *mut kvfree_rcu_head = (unsafe { core::ptr::addr_of_mut!((*___p).rcu) }) as *mut kvfree_rcu_head;
unsafe { kvfree_call_rcu(___head, ___p as *mut ()) };
}
if true {
break;
}
}
}
}

fn get_ucounts(ucounts: *mut ucounts) -> *mut ucounts {
if rcuref_get(unsafe { core::ptr::addr_of_mut!((*ucounts).count) }) {
return ucounts;
}
core::ptr::null_mut::<ucounts>()
}

fn get_rlimit_value(ucounts: *mut ucounts, r#type: u32) -> i64 {
atomic_long_read(unsafe { (*ucounts).rlimit.as_mut_ptr() as *mut __SlateRecord_types_h_197 }.wrapping_offset((r#type as u32) as isize) as *const __SlateRecord_types_h_197)
}

fn get_userns_rlimit_max(ns: *mut user_namespace, r#type: u32) -> i64 {
let mut __v0: i64;
loop {
if !(__native_word__19!() || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_243() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(((*ns).rlimit_max.as_mut_ptr() as *mut i64).wrapping_offset((r#type as u32) as isize) as *const i64) };
__v0
}

fn get_user_ns(ns: *mut user_namespace) -> *mut user_namespace {
unsafe { core::ptr::addr_of_mut!(init_user_ns) }
}

fn put_user_ns(ns: *mut user_namespace) {
}

#[unsafe(no_mangle)]
static mut init_ucounts: ucounts = {
let mut __t0: ucounts = unsafe { core::mem::zeroed() };
__t0.ns = unsafe { core::ptr::addr_of_mut!(init_user_ns) };
__t0.uid = __SlateRecord_uidgid_types_h_9 { val: 0u32 };
__t0.count = __SlateRecord_types_h_202 { refcnt: __SlateRecord_types_h_190 { counter: 0i32 } };
__t0
};

static mut ucounts_hashtable: __SlateAlign16<[hlist_nulls_head; 1024]> = __SlateAlign16({
let mut __t0: [hlist_nulls_head; 1024] = unsafe { core::mem::zeroed() };
let __t1 = hlist_nulls_head { first: 1u64 as *mut hlist_nulls_node };
let mut __t2: usize = 0;
while __t2 <= 1023 {
__t0[__t2] = __t1;
__t2 += 1;
}
__t0
});

static mut ucounts_lock: spinlock = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = unsafe { core::mem::zeroed() };
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0
} };

#[unsafe(link_section = ".text.slate_distinct.ucount.set_lookup")]
extern "C" fn set_lookup(root: *mut ctl_table_root) -> *mut ctl_table_set {
unsafe { core::ptr::addr_of_mut!((*current_user_ns()).set) }
}

#[unsafe(link_section = ".text.slate_distinct.ucount.set_is_seen")]
extern "C" fn set_is_seen(set: *mut ctl_table_set) -> i32 {
((unsafe { core::ptr::addr_of_mut!((*current_user_ns()).set) }) == set) as i32
}

#[unsafe(link_section = ".text.slate_distinct.ucount.set_permissions")]
extern "C" fn set_permissions(head: *mut ctl_table_header, table: *const ctl_table) -> i32 {
let user_ns: *mut user_namespace = container_of__11!(unsafe { (*head).set }, user_namespace);
let mut mode: i32 = 0;
// Allow users with CAP_SYS_RESOURCE unrestrained access
if unsafe { ns_capable_noaudit(user_ns, 24i32) } {
mode = (((unsafe { (*table).mode }) as i32) & 448) >> 6i32;
} else {
mode = ((unsafe { (*table).mode }) as i32) & 4;
}
// Allow all others at most read-only access
mode << 6i32 | mode << 3i32 | mode
}

static mut set_root: ctl_table_root = {
let mut __t0: ctl_table_root = unsafe { core::mem::zeroed() };
__t0.lookup = Some(set_lookup);
__t0.permissions = Some(set_permissions);
__t0
};

static mut ue_zero: i64 = 0;

static mut ue_int_max: i64 = 2147483647;

static user_table: __SlateAlign16<[ctl_table; 11]> = __SlateAlign16([{
let mut __t0: ctl_table = unsafe { core::mem::zeroed() };
__t0.procname = (b"max_user_namespaces\0".as_ptr() as *mut u8) as *const u8;
__t0.maxlen = (core::mem::size_of::<i64>() as u64) as i32;
__t0.mode = 420u16;
__t0.proc_handler = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*const ctl_table, i32, *mut (), *mut u64, *mut i64) -> i32>>(proc_doulongvec_minmax as *const ()) };
__t0.extra1 = (unsafe { core::ptr::addr_of_mut!(ue_zero) }) as *mut ();
__t0.extra2 = (unsafe { core::ptr::addr_of_mut!(ue_int_max) }) as *mut ();
__t0
}, {
let mut __t1: ctl_table = unsafe { core::mem::zeroed() };
__t1.procname = (b"max_pid_namespaces\0".as_ptr() as *mut u8) as *const u8;
__t1.maxlen = (core::mem::size_of::<i64>() as u64) as i32;
__t1.mode = 420u16;
__t1.proc_handler = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*const ctl_table, i32, *mut (), *mut u64, *mut i64) -> i32>>(proc_doulongvec_minmax as *const ()) };
__t1.extra1 = (unsafe { core::ptr::addr_of_mut!(ue_zero) }) as *mut ();
__t1.extra2 = (unsafe { core::ptr::addr_of_mut!(ue_int_max) }) as *mut ();
__t1
}, {
let mut __t2: ctl_table = unsafe { core::mem::zeroed() };
__t2.procname = (b"max_uts_namespaces\0".as_ptr() as *mut u8) as *const u8;
__t2.maxlen = (core::mem::size_of::<i64>() as u64) as i32;
__t2.mode = 420u16;
__t2.proc_handler = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*const ctl_table, i32, *mut (), *mut u64, *mut i64) -> i32>>(proc_doulongvec_minmax as *const ()) };
__t2.extra1 = (unsafe { core::ptr::addr_of_mut!(ue_zero) }) as *mut ();
__t2.extra2 = (unsafe { core::ptr::addr_of_mut!(ue_int_max) }) as *mut ();
__t2
}, {
let mut __t3: ctl_table = unsafe { core::mem::zeroed() };
__t3.procname = (b"max_ipc_namespaces\0".as_ptr() as *mut u8) as *const u8;
__t3.maxlen = (core::mem::size_of::<i64>() as u64) as i32;
__t3.mode = 420u16;
__t3.proc_handler = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*const ctl_table, i32, *mut (), *mut u64, *mut i64) -> i32>>(proc_doulongvec_minmax as *const ()) };
__t3.extra1 = (unsafe { core::ptr::addr_of_mut!(ue_zero) }) as *mut ();
__t3.extra2 = (unsafe { core::ptr::addr_of_mut!(ue_int_max) }) as *mut ();
__t3
}, {
let mut __t4: ctl_table = unsafe { core::mem::zeroed() };
__t4.procname = (b"max_net_namespaces\0".as_ptr() as *mut u8) as *const u8;
__t4.maxlen = (core::mem::size_of::<i64>() as u64) as i32;
__t4.mode = 420u16;
__t4.proc_handler = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*const ctl_table, i32, *mut (), *mut u64, *mut i64) -> i32>>(proc_doulongvec_minmax as *const ()) };
__t4.extra1 = (unsafe { core::ptr::addr_of_mut!(ue_zero) }) as *mut ();
__t4.extra2 = (unsafe { core::ptr::addr_of_mut!(ue_int_max) }) as *mut ();
__t4
}, {
let mut __t5: ctl_table = unsafe { core::mem::zeroed() };
__t5.procname = (b"max_mnt_namespaces\0".as_ptr() as *mut u8) as *const u8;
__t5.maxlen = (core::mem::size_of::<i64>() as u64) as i32;
__t5.mode = 420u16;
__t5.proc_handler = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*const ctl_table, i32, *mut (), *mut u64, *mut i64) -> i32>>(proc_doulongvec_minmax as *const ()) };
__t5.extra1 = (unsafe { core::ptr::addr_of_mut!(ue_zero) }) as *mut ();
__t5.extra2 = (unsafe { core::ptr::addr_of_mut!(ue_int_max) }) as *mut ();
__t5
}, {
let mut __t6: ctl_table = unsafe { core::mem::zeroed() };
__t6.procname = (b"max_cgroup_namespaces\0".as_ptr() as *mut u8) as *const u8;
__t6.maxlen = (core::mem::size_of::<i64>() as u64) as i32;
__t6.mode = 420u16;
__t6.proc_handler = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*const ctl_table, i32, *mut (), *mut u64, *mut i64) -> i32>>(proc_doulongvec_minmax as *const ()) };
__t6.extra1 = (unsafe { core::ptr::addr_of_mut!(ue_zero) }) as *mut ();
__t6.extra2 = (unsafe { core::ptr::addr_of_mut!(ue_int_max) }) as *mut ();
__t6
}, {
let mut __t7: ctl_table = unsafe { core::mem::zeroed() };
__t7.procname = (b"max_time_namespaces\0".as_ptr() as *mut u8) as *const u8;
__t7.maxlen = (core::mem::size_of::<i64>() as u64) as i32;
__t7.mode = 420u16;
__t7.proc_handler = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*const ctl_table, i32, *mut (), *mut u64, *mut i64) -> i32>>(proc_doulongvec_minmax as *const ()) };
__t7.extra1 = (unsafe { core::ptr::addr_of_mut!(ue_zero) }) as *mut ();
__t7.extra2 = (unsafe { core::ptr::addr_of_mut!(ue_int_max) }) as *mut ();
__t7
}, {
let mut __t8: ctl_table = unsafe { core::mem::zeroed() };
__t8.procname = (b"max_inotify_instances\0".as_ptr() as *mut u8) as *const u8;
__t8.maxlen = (core::mem::size_of::<i64>() as u64) as i32;
__t8.mode = 420u16;
__t8.proc_handler = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*const ctl_table, i32, *mut (), *mut u64, *mut i64) -> i32>>(proc_doulongvec_minmax as *const ()) };
__t8.extra1 = (unsafe { core::ptr::addr_of_mut!(ue_zero) }) as *mut ();
__t8.extra2 = (unsafe { core::ptr::addr_of_mut!(ue_int_max) }) as *mut ();
__t8
}, {
let mut __t9: ctl_table = unsafe { core::mem::zeroed() };
__t9.procname = (b"max_inotify_watches\0".as_ptr() as *mut u8) as *const u8;
__t9.maxlen = (core::mem::size_of::<i64>() as u64) as i32;
__t9.mode = 420u16;
__t9.proc_handler = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*const ctl_table, i32, *mut (), *mut u64, *mut i64) -> i32>>(proc_doulongvec_minmax as *const ()) };
__t9.extra1 = (unsafe { core::ptr::addr_of_mut!(ue_zero) }) as *mut ();
__t9.extra2 = (unsafe { core::ptr::addr_of_mut!(ue_int_max) }) as *mut ();
__t9
}, {
let mut __t10: ctl_table = unsafe { core::mem::zeroed() };
__t10.procname = (b"max_binfmt_misc_interpreters\0".as_ptr() as *mut u8) as *const u8;
__t10.maxlen = (core::mem::size_of::<i64>() as u64) as i32;
__t10.mode = 420u16;
__t10.proc_handler = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*const ctl_table, i32, *mut (), *mut u64, *mut i64) -> i32>>(proc_doulongvec_minmax as *const ()) };
__t10.extra1 = (unsafe { core::ptr::addr_of_mut!(ue_zero) }) as *mut ();
__t10.extra2 = (unsafe { core::ptr::addr_of_mut!(ue_int_max) }) as *mut ();
__t10
}]);

#[unsafe(no_mangle)]
extern "C" fn retire_userns_sysctls(ns: *mut user_namespace) {
let tbl: *const ctl_table = unsafe { (*(*ns).sysctls).ctl_table_arg };
unsafe { unregister_sysctl_table((*ns).sysctls) };
unsafe { retire_sysctl_set(core::ptr::addr_of_mut!((*ns).set)) };
unsafe { kfree(tbl as *const ()) };
}

fn find_ucounts(ns: *mut user_namespace, uid: __SlateRecord_uidgid_types_h_9, hashent: *mut hlist_nulls_head) -> *mut ucounts {
let mut ucounts: *mut ucounts = unsafe { core::mem::zeroed() };
let mut pos: *mut hlist_nulls_node = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_guard_245: __SlateRecord_rcupdate_h_1216 = class_rcu_constructor();
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_245) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || class_rcu_destructor(__slate_cleanup_ptr_1 as *mut _));
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
let mut __v2: *mut hlist_nulls_node;
let mut __v3: *mut hlist_nulls_node;
loop {
if !(__native_word__18!() || ((const { core::mem::size_of::<*mut hlist_nulls_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_247() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*hashent).first) as *const *mut hlist_nulls_node) };
let __UNIQUE_ID_rcu_246: *mut hlist_nulls_node = __v3;
__v2 = __UNIQUE_ID_rcu_246;
pos = __v2;
loop {
let __v6: bool = if is_a_nulls(pos as *const hlist_nulls_node) == 0 {
let mut __v7: i32;
ucounts = hlist_nulls_entry!(pos);
__v7 = 1;
__v7 != 0
} else {
false
};
if !__v6 {
break;
}
if uid_eq(unsafe { (*ucounts).uid }, uid) && (unsafe { (*ucounts).ns }) == ns {
if rcuref_get(unsafe { core::ptr::addr_of_mut!((*ucounts).count) }) {
return ucounts;
}
}
let mut __v10: *mut hlist_nulls_node;
let mut __v11: *mut hlist_nulls_node;
loop {
if !(__native_word__18!() || ((const { core::mem::size_of::<*mut hlist_nulls_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_249() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v11 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*pos).next) as *const *mut hlist_nulls_node) };
__v10 = __v11;
pos = __v10;
}
core::ptr::null_mut::<ucounts>()
}

fn hlist_add_ucounts(ucounts: *mut ucounts) {
let hashent: *mut hlist_nulls_head = ucounts_hashentry!(unsafe { (*ucounts).ns }, unsafe { (*ucounts).uid });
spin_lock_irq(unsafe { core::ptr::addr_of_mut!(ucounts_lock) });
hlist_nulls_add_head_rcu(unsafe { core::ptr::addr_of_mut!((*ucounts).node) }, hashent);
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!(ucounts_lock) });
}

#[unsafe(no_mangle)]
extern "C" fn alloc_ucounts(ns: *mut user_namespace, uid: __SlateRecord_uidgid_types_h_9) -> *mut ucounts {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let hashent: *mut hlist_nulls_head = ucounts_hashentry!(ns, uid);
let mut ucounts: *mut ucounts = unsafe { core::mem::zeroed() };
let mut new: *mut ucounts = unsafe { core::mem::zeroed() };
ucounts = find_ucounts(ns, uid, hashent);
if !ucounts.is_null() {
return ucounts;
}
let mut __v0: *mut ucounts;
let __obj_size: u64 = size_mul(core::mem::size_of::<ucounts>() as u64, 1u64);
let mut __v1: *mut ();
let mut __v2: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_3 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_3)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_5 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_5)
} });
}
__v2 = _res;
__v1 = __v2;
__v0 = __v1 as *mut ucounts;
new = __v0;
if new.is_null() {
return core::ptr::null_mut::<ucounts>();
}
unsafe {
(*new).ns = ns;
}
unsafe {
(*new).uid = uid;
}
rcuref_init(unsafe { core::ptr::addr_of_mut!((*new).count) }, 1u32);
spin_lock_irq(unsafe { core::ptr::addr_of_mut!(ucounts_lock) });
ucounts = find_ucounts(ns, uid, hashent);
if !ucounts.is_null() {
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!(ucounts_lock) });
unsafe { kfree(new as *const ()) };
return ucounts;
}
hlist_nulls_add_head_rcu(unsafe { core::ptr::addr_of_mut!((*new).node) }, hashent);
get_user_ns(unsafe { (*new).ns });
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!(ucounts_lock) });
new
}

fn atomic_long_inc_below(v: *mut __SlateRecord_types_h_197, u: i64) -> bool {
let mut c: i64 = atomic_long_read(v as *const __SlateRecord_types_h_197);
loop {
if unlikely!(c >= u) != 0 {
return false;
}
if !(!atomic_long_try_cmpxchg(v, core::ptr::addr_of_mut!(c), c.wrapping_add(1))) {
break;
}
}
true
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_inc_ucount_252: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut user_namespace, __SlateRecord_uidgid_types_h_9, u32) -> *mut ucounts>, *mut ()>(Some(inc_ucount)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.ucount.dec_ucount")]
extern "C" fn dec_ucount(ucounts: *mut ucounts, r#type: u32) {
let mut iter: *mut ucounts = unsafe { core::mem::zeroed() };
iter = ucounts;
while !iter.is_null() {
let dec: i64 = atomic_long_dec_if_positive(unsafe { (*iter).ucount.as_mut_ptr() as *mut __SlateRecord_types_h_197 }.wrapping_offset((r#type as u32) as isize));
let mut __v0: i64;
let __ret_warn_on: i32 = (dec < 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("253: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 253b - ., 3; .popsection\n# {0}", const 253 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_dec_ucount_1, sym __slate_asm_string_dec_ucount_2, const 247 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("254: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 254b - ., 4; .popsection\n# {0}", const 254 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
iter = unsafe { (*(*iter).ns).ucounts };
}
put_ucounts(ucounts);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_dec_ucount_255: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ucounts, u32)>, *mut ()>(Some(dec_ucount)) };

#[unsafe(no_mangle)]
extern "C" fn inc_rlimit_ucounts(ucounts: *mut ucounts, r#type: u32, v: i64) -> i64 {
let mut iter: *mut ucounts = unsafe { core::mem::zeroed() };
let mut max: i64 = 9223372036854775807i64;
let mut ret: i64 = 0;
iter = ucounts;
while !iter.is_null() {
let new: i64 = atomic_long_add_return(v, unsafe { (*iter).rlimit.as_mut_ptr() as *mut __SlateRecord_types_h_197 }.wrapping_offset((r#type as u32) as isize));
if new < 0 || new > max {
ret = 9223372036854775807i64;
} else {
if iter == ucounts {
ret = new;
}
}
max = get_userns_rlimit_max(unsafe { (*iter).ns }, r#type);
iter = unsafe { (*(*iter).ns).ucounts };
}
ret
}

#[unsafe(no_mangle)]
extern "C" fn dec_rlimit_ucounts(ucounts: *mut ucounts, r#type: u32, v: i64) -> bool {
let mut iter: *mut ucounts = unsafe { core::mem::zeroed() };
let mut new: i64 = 1i32.wrapping_neg() as i64; // Silence compiler warning
iter = ucounts;
while !iter.is_null() {
let dec: i64 = atomic_long_sub_return(v, unsafe { (*iter).rlimit.as_mut_ptr() as *mut __SlateRecord_types_h_197 }.wrapping_offset((r#type as u32) as isize));
let mut __v0: i64;
let __ret_warn_on: i32 = (dec < 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("256: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 256b - ., 3; .popsection\n# {0}", const 256 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_dec_rlimit_ucounts_1, sym __slate_asm_string_dec_rlimit_ucounts_2, const 276 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("257: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 257b - ., 4; .popsection\n# {0}", const 257 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if iter == ucounts {
new = dec;
}
iter = unsafe { (*(*iter).ns).ucounts };
}
new == 0
}

fn do_dec_rlimit_put_ucounts(ucounts: *mut ucounts, last: *mut ucounts, r#type: u32) {
let mut iter: *mut ucounts = unsafe { core::mem::zeroed() };
let mut next: *mut ucounts = unsafe { core::mem::zeroed() };
iter = ucounts;
while iter != last {
let dec: i64 = atomic_long_sub_return(1i64, unsafe { (*iter).rlimit.as_mut_ptr() as *mut __SlateRecord_types_h_197 }.wrapping_offset((r#type as u32) as isize));
let mut __v0: i64;
let __ret_warn_on: i32 = (dec < 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("258: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 258b - ., 3; .popsection\n# {0}", const 258 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_do_dec_rlimit_put_ucounts_1, sym __slate_asm_string_do_dec_rlimit_put_ucounts_2, const 289 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("259: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 259b - ., 4; .popsection\n# {0}", const 259 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
next = unsafe { (*(*iter).ns).ucounts };
if dec == 0 {
put_ucounts(iter);
}
iter = next;
}
}

#[unsafe(no_mangle)]
extern "C" fn dec_rlimit_put_ucounts(ucounts: *mut ucounts, r#type: u32) {
do_dec_rlimit_put_ucounts(ucounts, core::ptr::null_mut::<ucounts>(), r#type);
}

#[unsafe(no_mangle)]
extern "C" fn inc_rlimit_get_ucounts(ucounts: *mut ucounts, r#type: u32, override_rlimit: bool) -> i64 {
let mut new: i64 = unsafe { core::mem::zeroed() };
let mut ret: i64 = unsafe { core::mem::zeroed() };
let mut dec: core::mem::MaybeUninit<i64> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i64 = core::ptr::addr_of_mut!(dec) as *mut i64;
let mut max: i64 = unsafe { core::mem::zeroed() };
// Caller must hold a reference to ucounts
let mut iter: core::mem::MaybeUninit<*mut ucounts> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut ucounts = core::ptr::addr_of_mut!(iter) as *mut *mut ucounts;
unsafe {
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
max = 9223372036854775807i64;
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
ret = 0;
*__slate_slot_7 = ucounts;
'__join_4: {
while !(*__slate_slot_7).is_null() {
new = atomic_long_add_return(1i64, ((*(*__slate_slot_7)).rlimit.as_mut_ptr() as *mut __SlateRecord_types_h_197).wrapping_offset((r#type as u32) as isize));
if new < 0 || new > max {
break '__join_4;
} else {
if *__slate_slot_7 == ucounts {
ret = new;
}
if !override_rlimit {
max = get_userns_rlimit_max((*(*__slate_slot_7)).ns, r#type);
}
// Grab an extra ucount reference for the caller when
// the rlimit count was previously 0.
if new == 1 {
if get_ucounts(*__slate_slot_7).is_null() {
break '__join_4;
}
}
*__slate_slot_7 = (*(*(*__slate_slot_7)).ns).ucounts;
}
}
return ret;
}
*__slate_slot_5 = atomic_long_sub_return(1i64, ((*(*__slate_slot_7)).rlimit.as_mut_ptr() as *mut __SlateRecord_types_h_197).wrapping_offset((r#type as u32) as isize));
let __ret_warn_on: i32 = (*__slate_slot_5 < 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("260: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 260b - ., 3; .popsection\n# {0}", const 260 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_inc_rlimit_get_ucounts_8, sym __slate_asm_string_inc_rlimit_get_ucounts_9, const 329 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("261: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 261b - ., 4; .popsection\n# {0}", const 261 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
do_dec_rlimit_put_ucounts(ucounts, *__slate_slot_7, r#type);
return 0;
}
}

#[unsafe(no_mangle)]
extern "C" fn is_rlimit_overlimit(ucounts: *mut ucounts, r#type: u32, rlimit: u64) -> bool {
let mut iter: *mut ucounts = unsafe { core::mem::zeroed() };
let mut max: i64 = rlimit as i64;
if rlimit > 9223372036854775807u64 {
max = 9223372036854775807i64;
}
iter = ucounts;
while !iter.is_null() {
let val: i64 = get_rlimit_value(iter, r#type);
if val < 0 || val > max {
return true;
}
max = get_userns_rlimit_max(unsafe { (*iter).ns }, r#type);
iter = unsafe { (*(*iter).ns).ucounts };
}
false
}

#[unsafe(link_section = ".init.text")]
extern "C" fn user_namespace_sysctl_init() -> i32 {
// It is necessary to register the user directory in the
// default set so that registrations in the child sets work
// properly.
unsafe {
user_header = register_sysctl_sz(c"user".as_ptr() as *const u8, (core::ptr::addr_of_mut!(empty.0) as *mut ctl_table) as *const ctl_table, 0u64);
}
kmemleak_ignore((unsafe { user_header }) as *const ());
loop {
if unlikely!(unsafe { user_header }.is_null()) != 0 {
loop {
unsafe {
core::arch::asm!("262: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 262b - ., 3; .popsection\n# {0}", const 262 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_user_namespace_sysctl_init_0, sym __slate_asm_string_user_namespace_sysctl_init_1, const 361 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
if unlikely!(!setup_userns_sysctls(unsafe { core::ptr::addr_of_mut!(init_user_ns) })) != 0 {
loop {
unsafe {
core::arch::asm!("263: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 263b - ., 3; .popsection\n# {0}", const 263 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_user_namespace_sysctl_init_5, sym __slate_asm_string_user_namespace_sysctl_init_6, const 362 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
hlist_add_ucounts(unsafe { core::ptr::addr_of_mut!(init_ucounts) });
inc_rlimit_ucounts(unsafe { core::ptr::addr_of_mut!(init_ucounts) }, 0u32, 1i64);
0
}

static mut user_header: *mut ctl_table_header = unsafe { core::mem::zeroed() };

static mut empty: __SlateAlign16<[ctl_table; 1]> = __SlateAlign16(unsafe { core::mem::zeroed() });

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_user_namespace_sysctl_init_265: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i32>, *mut ()>(Some(user_namespace_sysctl_init)) };

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_dec_rlimit_ucounts_1: [u8; 1] = *b"\0";

static __slate_asm_string_dec_rlimit_ucounts_2: [u8; 45] = *b"/home/takashi/c-corpus/linux/kernel/ucount.c\0";

static __slate_asm_string_dec_ucount_1: [u8; 1] = *b"\0";

static __slate_asm_string_dec_ucount_2: [u8; 45] = *b"/home/takashi/c-corpus/linux/kernel/ucount.c\0";

static __slate_asm_string_do_dec_rlimit_put_ucounts_1: [u8; 1] = *b"\0";

static __slate_asm_string_do_dec_rlimit_put_ucounts_2: [u8; 45] = *b"/home/takashi/c-corpus/linux/kernel/ucount.c\0";

static __slate_asm_string_inc_rlimit_get_ucounts_8: [u8; 1] = *b"\0";

static __slate_asm_string_inc_rlimit_get_ucounts_9: [u8; 45] = *b"/home/takashi/c-corpus/linux/kernel/ucount.c\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

static __slate_asm_string_user_namespace_sysctl_init_0: [u8; 1] = *b"\0";

static __slate_asm_string_user_namespace_sysctl_init_1: [u8; 45] = *b"/home/takashi/c-corpus/linux/kernel/ucount.c\0";

static __slate_asm_string_user_namespace_sysctl_init_5: [u8; 1] = *b"\0";

static __slate_asm_string_user_namespace_sysctl_init_6: [u8; 45] = *b"/home/takashi/c-corpus/linux/kernel/ucount.c\0";

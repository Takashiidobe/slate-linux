
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
//! SPDX-License-Identifier: BSD-3-Clause OR GPL-2.0
//!
//! Module Name: dsmethod - Parser/Interpreter interface - control method parsing
//!
//! Copyright (C) 2000 - 2026, Intel Corp.
//! This macro is used to tag functions as "printf-like" because
//! some compilers (like GCC) can catch printf format string problems.
//! This counts to 15. Any more, it will return 16th argument.
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
//! The type of an index into the pagecache.
//! kcsan_check_read - check regular read access for races
//!
//! @ptr: address of access
//! @size: size of access
//! Yes, this permits 64-bit accesses on 32-bit architectures. These will
//! actually be atomic in some cases (namely Armv7 + LPAE), but for others we
//! rely on the access being split into 2x32-bit accesses for a 32-bit quantity
//! (e.g. a virtual address) and a strong prevailing wind.
//! No such file or directory
//! Value too large for defined data type
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
//! restart by calling sys_restart_syscall
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
//! See linux/instrumentation.h
//! 64 bit
//! unused, was: #define EX_TYPE_COPY	 4
//! @a is a power of 2 value
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
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
//! Generate a mask for the specified type @t. Additional checks are made to
//! guarantee the value returned fits in that type, relying on
//! -Wshift-count-overflow compiler check to detect incompatible arguments.
//! For example, all these create build errors or warnings:
//!
//! - GENMASK(15, 20): wrong argument order
//! - GENMASK(72, 15): doesn't fit unsigned long
//! - GENMASK_U32(33, 15): doesn't fit in a u32
//! Check at compile time that something is of a particular type.
//! Always evaluates to 1 so you may use it easily in comparisons.
//! Begin/end of an instrumentation safe region
//! __ADDRESSABLE() is used to ensure the key symbol doesn't get stripped from
//! the symbol table so that objtool can reference it when it generates the
//! .static_call_sites section.
//! These are for everybody (although not all archs will actually
//! discard it in modules)
//! Internal kernel timer frequency
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! strlen("ratelimit") + 1
//! label
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
//! Defines x86 CPU feature bits
//! Compile time versions of __arch_hweightN()
//! ilog2 - log base 2 of 32-bit or a 64-bit unsigned value
//! @n: parameter
//!
//! constant-capable log of base 2 calculation
//! - this can be used to initialise global variables from constant data, hence
//! the massive ternary operator construction
//!
//! selects the appropriately-sized optimised version depending on sizeof(n)
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
//! Ensure that leaf 0x2 cache and TLB type values do not intersect,
//! since they share the same type field at struct cpuid_0x2_table.
//! on 4KB pages
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
//! DIV_U64_ROUND_UP - unsigned 64bit divide with 32bit divisor rounded up
//! @ll: unsigned 64bit dividend
//! @d: unsigned 32bit divisor
//!
//! Divide unsigned 64bit dividend by unsigned 32bit divisor
//! and round up.
//!
//! Return: dividend / divisor rounded up
//! Parameters used to convert the timespec values:
//! This is the default CPU features testing macro to use in code.
//!
//! It is for detection of features which need kernel infrastructure to be
//! used.  It may *not* directly test the CPU itself.  Use the cpu_has() family
//! if you want true runtime testing of CPU features, like in hypervisor code
//! where you are supporting a possible guest feature where host support for it
//! is not relevant.
//! rescheduling necessary
//! We use the MSB for PREEMPT_NEED_RESCHED mostly because it is available.
//! Macros to retrieve the current execution context:
//!
//! in_nmi()		- We're in NMI context
//! in_hardirq()		- We're in hard IRQ context
//! in_serving_softirq()	- We're in softirq context
//! in_task()		- We're in task context
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
//! Data type ranges
//! Note: These macros are designed to be compiler independent as well as
//! working around problems that some 32-bit compilers have with 64-bit
//! constants.
//! Internal control codes
//! For values, see Producer/Consumer above
//! Debug level macros that are used in the DEBUG_PRINT macros
//! Operand Stack (in WALK_STATE), Must be large enough to contain METHOD_MAX_ARG
//! Common area for all objects.
//!
//! descriptor_type is used to differentiate between internal descriptors, and
//! must be in the same place across all descriptors
//!
//! Note: The descriptor_type and Type fields must appear in the identical
//! position in both the struct acpi_namespace_node and union acpi_operand_object
//! structures.
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
// Flexible array members are not allowed to be part of a union under
// C99, but this is not for any technical reason. Work around the
// limitation.
// Concatenate two parameters, but allow them to be expanded beforehand.
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
// kcsan_check_write - check regular write access for races
//
// @ptr: address of access
// @size: size of access
// kcsan_check_read_write - check regular read-write access for races
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
// Try again
// Out of memory
// Device or resource busy
// Invalid argument
// Operation is not supported
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
// This should be used to refer to an instruction that is considered
// terminating, like a noreturn CALL or UD2 when we know they are not -- eg
// WARN using UD2.
// Annotate a special section entry.  This emables livepatch module generation
// to find and extract individual special section entries as needed.
// reg := -EIO
// reg := -EIO
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
// "sse"
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
// "rdpid" RDPID instruction
// LFENCE always serializing / synchronizes RDTSC
// BUG word(s)
// "div0" AMD DIV0 speculation bug
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
// min - return minimum of two values of the same or compatible types
// @x: first value
// @y: second value
// max - return maximum of two values of the same or compatible types
// @x: first value
// @y: second value
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
// 0xFF
// Value returned by acpi_os_get_thread_id. There is no standard "thread_id"
// across operating systems or even the various UNIX systems. Since ACPICA
// only needs the thread ID as a unique thread identifier, we use a u64
// as the only common data type - it will accommodate any type of pointer or
// any type of integer. It is up to the host-dependent OSL to cast the
// native thread ID type to a u64 (in acpi_os_get_thread_id).
// Synchronization objects - Mutexes, Semaphores, and spin_locks
//
// These macros are used if the host OS does not support a mutex object.
// Map the OSL Mutex interfaces to binary semaphores.
// Fixed by ACPI spec
// u16, as per ACPI spec
// Arg#, Local#, Name, Debug, ref_of, Index
// The module name is used primarily for error and debug messages.
// The __FILE__ macro is not very useful for this, because it
// usually includes the entire pathname to the module making the
// debug output difficult to read.
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
macro_rules! ACPI_CAST_PTR {
($t:ty, $p:expr) => {
(($p) as u64) as *mut $t
};
}

macro_rules! ACPI_ERROR {
($plist__0:expr, $plist__1:expr, $plist__2:expr $(, $plist:expr)*) => {
unsafe { acpi_error($plist__0, ($plist__1) as u32, ($plist__2) as *const u8, $($plist),*) };
};
}

macro_rules! ACPI_FAILURE {
($a:expr) => {
$a
};
}

macro_rules! ACPI_FREE {
($a:expr) => {
acpi_os_free(($a) as *mut ());
};
}

macro_rules! ACPI_FUNCTION_ENTRY {
() => {
unsafe { acpi_ut_track_stack_ptr() };
};
}

macro_rules! ACPI_FUNCTION_TRACE_PTR__7 {
($pointer:expr) => {
ACPI_TRACE_ENTRY__28!(*mut (), $pointer);
};
}

macro_rules! ACPI_FUNCTION_TRACE_PTR__11 {
($pointer:expr) => {
ACPI_TRACE_ENTRY__77!(*mut (), $pointer);
};
}

macro_rules! ACPI_FUNCTION_TRACE_PTR__49 {
($pointer:expr) => {
ACPI_TRACE_ENTRY__6!(*mut (), $pointer);
};
}

macro_rules! ACPI_FUNCTION_TRACE_PTR__58 {
($pointer:expr) => {
ACPI_TRACE_ENTRY__14!(*mut (), $pointer);
};
}

macro_rules! ACPI_FUNCTION_TRACE_PTR__82 {
($pointer:expr) => {
ACPI_TRACE_ENTRY__59!(*mut (), $pointer);
};
}

macro_rules! ACPI_FUNCTION_TRACE__52 {
() => {
unsafe { acpi_ut_trace(252u32, c"acpi_ds_create_method_mutex".as_ptr() as *const u8, core::ptr::addr_of!(_acpi_module_name) as *const u8, 64u32) };
};
}

/// Error reporting. Callers module and line number are inserted by AE_INFO,
/// the plist contains a set of parens to allow variable-length lists.
/// These macros are used for both the debug and non-debug versions of the code.
macro_rules! ACPI_INFO {
($plist__0:expr $(, $plist:expr)*) => {
unsafe { acpi_info(($plist__0) as *const u8, $($plist),*) };
};
}

macro_rules! ACPI_IS_DEBUG_ENABLED {
($level:expr, $component:expr) => {
($level) & unsafe { acpi_dbg_level } != 0 && (($component) as u32) & unsafe { acpi_dbg_layer } != 0
};
}

macro_rules! ACPI_PTR_DIFF {
($a:expr, $b:expr) => {
((unsafe { ACPI_CAST_PTR!(u8, $a).offset_from(ACPI_CAST_PTR!(u8, $b) as *mut u8) }) as i64) as u64
};
}

/// Success is always zero, failure is non-zero
macro_rules! ACPI_SUCCESS {
($a:expr) => {
($a) == 0
};
}

macro_rules! ACPI_TRACE_ENTRY__6 {
($type:ty, $param:expr) => {
unsafe { acpi_ut_trace_ptr(298u32, c"acpi_ds_begin_method_execution".as_ptr() as *const u8, core::ptr::addr_of!(_acpi_module_name) as *const u8, 64u32, (($param) as $type) as *const ()) };
};
}

macro_rules! ACPI_TRACE_ENTRY__14 {
($type:ty, $param:expr) => {
unsafe { acpi_ut_trace_ptr(466u32, c"acpi_ds_call_control_method".as_ptr() as *const u8, core::ptr::addr_of!(_acpi_module_name) as *const u8, 64u32, (($param) as $type) as *const ()) };
};
}

macro_rules! ACPI_TRACE_ENTRY__28 {
($type:ty, $param:expr) => {
unsafe { acpi_ut_trace_ptr(621u32, c"acpi_ds_restart_control_method".as_ptr() as *const u8, core::ptr::addr_of!(_acpi_module_name) as *const u8, 64u32, (($param) as $type) as *const ()) };
};
}

macro_rules! ACPI_TRACE_ENTRY__59 {
($type:ty, $param:expr) => {
unsafe { acpi_ut_trace_ptr(62u32, c"acpi_ds_auto_serialize_method".as_ptr() as *const u8, core::ptr::addr_of!(_acpi_module_name) as *const u8, 64u32, (($param) as $type) as *const ()) };
};
}

macro_rules! ACPI_TRACE_ENTRY__77 {
($type:ty, $param:expr) => {
unsafe { acpi_ut_trace_ptr(711u32, c"acpi_ds_terminate_control_method".as_ptr() as *const u8, core::ptr::addr_of!(_acpi_module_name) as *const u8, 64u32, (($param) as $type) as *const ()) };
};
}

macro_rules! ACPI_TRACE_EXIT__213 {
($type:ty, $param:expr) => {
{
loop {
let _param: $type = ($param) as $type;
unsafe { acpi_ut_status_exit(505i32 as $type, c"acpi_ds_call_control_method".as_ptr() as *const u8, core::ptr::addr_of!(_acpi_module_name) as *const u8, 64i32 as $type, _param) };
return _param;
}
}
};
}

macro_rules! ACPI_TRACE_EXIT__266 {
($type:ty, $param:expr) => {
{
loop {
let _param: $type = ($param) as $type;
unsafe { acpi_ut_status_exit(271i32 as $type, c"acpi_ds_create_method_mutex".as_ptr() as *const u8, core::ptr::addr_of!(_acpi_module_name) as *const u8, 64i32 as $type, _param) };
return _param;
}
}
};
}

macro_rules! ACPI_TRACE_EXIT__302 {
($type:ty, $param:expr) => {
{
loop {
let _param: $type = ($param) as $type;
unsafe { acpi_ut_status_exit(93i32 as $type, c"acpi_ds_auto_serialize_method".as_ptr() as *const u8, core::ptr::addr_of!(_acpi_module_name) as *const u8, 64i32 as $type, _param) };
return _param;
}
}
};
}

macro_rules! ACPI_TRACE_EXIT__377 {
($type:ty, $param:expr) => {
{
loop {
let _param: $type = ($param) as $type;
unsafe { acpi_ut_status_exit(651i32 as $type, c"acpi_ds_restart_control_method".as_ptr() as *const u8, core::ptr::addr_of!(_acpi_module_name) as *const u8, 64i32 as $type, _param) };
return _param;
}
}
};
}

macro_rules! ACPI_TRACE_EXIT__391 {
($type:ty, $param:expr) => {
{
loop {
let _param: $type = ($param) as $type;
unsafe { acpi_ut_status_exit(496i32 as $type, c"acpi_ds_call_control_method".as_ptr() as *const u8, core::ptr::addr_of!(_acpi_module_name) as *const u8, 64i32 as $type, _param) };
return _param;
}
}
};
}

macro_rules! ACPI_TRACE_EXIT__402 {
($type:ty, $param:expr) => {
{
loop {
let _param: $type = ($param) as $type;
unsafe { acpi_ut_status_exit(597i32 as $type, c"acpi_ds_call_control_method".as_ptr() as *const u8, core::ptr::addr_of!(_acpi_module_name) as *const u8, 64i32 as $type, _param) };
return _param;
}
}
};
}

macro_rules! ACPI_TRACE_EXIT__550 {
($type:ty, $param:expr) => {
{
loop {
let _param: $type = ($param) as $type;
unsafe { acpi_ut_status_exit(266i32 as $type, c"acpi_ds_create_method_mutex".as_ptr() as *const u8, core::ptr::addr_of!(_acpi_module_name) as *const u8, 64i32 as $type, _param) };
return _param;
}
}
};
}

macro_rules! ACPI_TRACE_EXIT__591 {
($type:ty, $param:expr) => {
{
loop {
let _param: $type = ($param) as $type;
unsafe { acpi_ut_status_exit(301i32 as $type, c"acpi_ds_begin_method_execution".as_ptr() as *const u8, core::ptr::addr_of!(_acpi_module_name) as *const u8, 64i32 as $type, _param) };
return _param;
}
}
};
}

macro_rules! ACPI_TRACE_EXIT__612 {
($type:ty, $param:expr) => {
{
loop {
let _param: $type = ($param) as $type;
unsafe { acpi_ut_status_exit(582i32 as $type, c"acpi_ds_call_control_method".as_ptr() as *const u8, core::ptr::addr_of!(_acpi_module_name) as *const u8, 64i32 as $type, _param) };
return _param;
}
}
};
}

macro_rules! ACPI_TRACE_EXIT__669 {
($type:ty, $param:expr) => {
{
loop {
let _param: $type = ($param) as $type;
unsafe { acpi_ut_status_exit(311i32 as $type, c"acpi_ds_begin_method_execution".as_ptr() as *const u8, core::ptr::addr_of!(_acpi_module_name) as *const u8, 64i32 as $type, _param) };
return _param;
}
}
};
}

macro_rules! ACPI_TRACE_EXIT__742 {
($type:ty, $param:expr) => {
{
loop {
let _param: $type = ($param) as $type;
unsafe { acpi_ut_status_exit(489i32 as $type, c"acpi_ds_call_control_method".as_ptr() as *const u8, core::ptr::addr_of!(_acpi_module_name) as *const u8, 64i32 as $type, _param) };
return _param;
}
}
};
}

macro_rules! ACPI_TRACE_EXIT__770 {
($type:ty, $param:expr) => {
{
loop {
let _param: $type = ($param) as $type;
unsafe { acpi_ut_status_exit(258i32 as $type, c"acpi_ds_create_method_mutex".as_ptr() as *const u8, core::ptr::addr_of!(_acpi_module_name) as *const u8, 64i32 as $type, _param) };
return _param;
}
}
};
}

macro_rules! ACPI_TRACE_EXIT__772 {
($type:ty, $param:expr) => {
{
loop {
let _param: $type = ($param) as $type;
unsafe { acpi_ut_status_exit(326i32 as $type, c"acpi_ds_begin_method_execution".as_ptr() as *const u8, core::ptr::addr_of!(_acpi_module_name) as *const u8, 64i32 as $type, _param) };
return _param;
}
}
};
}

macro_rules! ACPI_TRACE_EXIT__781 {
($type:ty, $param:expr) => {
{
loop {
let _param: $type = ($param) as $type;
unsafe { acpi_ut_status_exit(438i32 as $type, c"acpi_ds_begin_method_execution".as_ptr() as *const u8, core::ptr::addr_of!(_acpi_module_name) as *const u8, 64i32 as $type, _param) };
return _param;
}
}
};
}

macro_rules! ACPI_TRACE_EXIT__797 {
($type:ty, $param:expr) => {
{
loop {
let _param: $type = ($param) as $type;
unsafe { acpi_ut_status_exit(372i32 as $type, c"acpi_ds_begin_method_execution".as_ptr() as *const u8, core::ptr::addr_of!(_acpi_module_name) as *const u8, 64i32 as $type, _param) };
return _param;
}
}
};
}

macro_rules! ACPI_TRACE_EXIT__880 {
($type:ty, $param:expr) => {
{
loop {
let _param: $type = ($param) as $type;
unsafe { acpi_ut_status_exit(482i32 as $type, c"acpi_ds_call_control_method".as_ptr() as *const u8, core::ptr::addr_of!(_acpi_module_name) as *const u8, 64i32 as $type, _param) };
return _param;
}
}
};
}

macro_rules! ACPI_TRACE_EXIT__894 {
($type:ty, $param:expr) => {
{
loop {
let _param: $type = ($param) as $type;
unsafe { acpi_ut_status_exit(430i32 as $type, c"acpi_ds_begin_method_execution".as_ptr() as *const u8, core::ptr::addr_of!(_acpi_module_name) as *const u8, 64i32 as $type, _param) };
return _param;
}
}
};
}

macro_rules! ACPI_TRACE_EXIT__926 {
($type:ty, $param:expr) => {
{
loop {
let _param: $type = ($param) as $type;
unsafe { acpi_ut_status_exit(103i32 as $type, c"acpi_ds_auto_serialize_method".as_ptr() as *const u8, core::ptr::addr_of!(_acpi_module_name) as *const u8, 64i32 as $type, _param) };
return _param;
}
}
};
}

macro_rules! ACPI_TRACE_EXIT__939 {
($type:ty, $param:expr) => {
{
loop {
let _param: $type = ($param) as $type;
unsafe { acpi_ut_status_exit(477i32 as $type, c"acpi_ds_call_control_method".as_ptr() as *const u8, core::ptr::addr_of!(_acpi_module_name) as *const u8, 64i32 as $type, _param) };
return _param;
}
}
};
}

macro_rules! ACPI_TRACE_EXIT__951 {
($type:ty, $param:expr) => {
{
loop {
let _param: $type = ($param) as $type;
unsafe { acpi_ut_status_exit(84i32 as $type, c"acpi_ds_auto_serialize_method".as_ptr() as *const u8, core::ptr::addr_of!(_acpi_module_name) as *const u8, 64i32 as $type, _param) };
return _param;
}
}
};
}

macro_rules! ACPI_TRACE_EXIT__969 {
($type:ty, $param:expr) => {
{
loop {
let _param: $type = ($param) as $type;
unsafe { acpi_ut_status_exit(72i32 as $type, c"acpi_ds_auto_serialize_method".as_ptr() as *const u8, core::ptr::addr_of!(_acpi_module_name) as *const u8, 64i32 as $type, _param) };
return _param;
}
}
};
}

macro_rules! ACPI_TRACE_EXIT__1034 {
($type:ty, $param:expr) => {
{
loop {
let _param: $type = ($param) as $type;
unsafe { acpi_ut_status_exit(684i32 as $type, c"acpi_ds_restart_control_method".as_ptr() as *const u8, core::ptr::addr_of!(_acpi_module_name) as *const u8, 64i32 as $type, _param) };
return _param;
}
}
};
}

macro_rules! ACPI_TRACE_EXIT__1056 {
($type:ty, $param:expr) => {
{
loop {
let _param: $type = ($param) as $type;
unsafe { acpi_ut_status_exit(352i32 as $type, c"acpi_ds_begin_method_execution".as_ptr() as *const u8, core::ptr::addr_of!(_acpi_module_name) as *const u8, 64i32 as $type, _param) };
return _param;
}
}
};
}

macro_rules! BIT {
($nr:expr) => {
UL!() << $nr
};
}

macro_rules! EXCEP_AML {
($code:expr) => {
(($code) | 12288) as u32
};
}

macro_rules! EXCEP_CTL {
($code:expr) => {
(($code) | 16384) as u32
};
}

/// Macros to insert the exception code classes
macro_rules! EXCEP_ENV {
($code:expr) => {
(($code) | 0) as u32
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

macro_rules! __or {
() => {
____or!()
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

macro_rules! acpi_os_create_mutex {
($out_handle:expr) => {
unsafe { acpi_os_create_semaphore(1u32, 1u32, $out_handle) }
};
}

macro_rules! acpi_os_release_mutex {
($handle:expr) => {
unsafe { acpi_os_signal_semaphore($handle, 1u32) };
};
}

macro_rules! acpi_ut_create_internal_object__21 {
($t:expr) => {
unsafe { acpi_ut_create_internal_object_dbg(core::ptr::addr_of!(_acpi_module_name) as *const u8, 256u32, 64u32, ($t) as u32) }
};
}

macro_rules! alloc_tag_restore {
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

macro_rules! return_ACPI_STATUS__17 {
($status:expr) => {
{
ACPI_TRACE_EXIT__266!(u32, $status);
}
};
}

macro_rules! return_ACPI_STATUS__64 {
($status:expr) => {
{
ACPI_TRACE_EXIT__772!(u32, $status);
}
};
}

macro_rules! return_ACPI_STATUS__68 {
($status:expr) => {
{
ACPI_TRACE_EXIT__969!(u32, $status);
}
};
}

macro_rules! return_ACPI_STATUS__75 {
($status:expr) => {
{
ACPI_TRACE_EXIT__377!(u32, $status);
}
};
}

macro_rules! return_ACPI_STATUS__149 {
($status:expr) => {
{
ACPI_TRACE_EXIT__1034!(u32, $status);
}
};
}

macro_rules! return_ACPI_STATUS__156 {
($status:expr) => {
{
ACPI_TRACE_EXIT__770!(u32, $status);
}
};
}

macro_rules! return_ACPI_STATUS__262 {
($status:expr) => {
{
ACPI_TRACE_EXIT__391!(u32, $status);
}
};
}

macro_rules! return_ACPI_STATUS__315 {
($status:expr) => {
{
ACPI_TRACE_EXIT__302!(u32, $status);
}
};
}

macro_rules! return_ACPI_STATUS__317 {
($status:expr) => {
{
ACPI_TRACE_EXIT__880!(u32, $status);
}
};
}

macro_rules! return_ACPI_STATUS__319 {
($status:expr) => {
{
ACPI_TRACE_EXIT__797!(u32, $status);
}
};
}

macro_rules! return_ACPI_STATUS__438 {
($status:expr) => {
{
ACPI_TRACE_EXIT__402!(u32, $status);
}
};
}

macro_rules! return_ACPI_STATUS__475 {
($status:expr) => {
{
ACPI_TRACE_EXIT__951!(u32, $status);
}
};
}

macro_rules! return_ACPI_STATUS__493 {
($status:expr) => {
{
ACPI_TRACE_EXIT__550!(u32, $status);
}
};
}

macro_rules! return_ACPI_STATUS__498 {
($status:expr) => {
{
ACPI_TRACE_EXIT__742!(u32, $status);
}
};
}

macro_rules! return_ACPI_STATUS__621 {
($status:expr) => {
{
ACPI_TRACE_EXIT__612!(u32, $status);
}
};
}

macro_rules! return_ACPI_STATUS__767 {
($status:expr) => {
{
ACPI_TRACE_EXIT__1056!(u32, $status);
}
};
}

macro_rules! return_ACPI_STATUS__780 {
($status:expr) => {
{
ACPI_TRACE_EXIT__781!(u32, $status);
}
};
}

macro_rules! return_ACPI_STATUS__923 {
($status:expr) => {
{
ACPI_TRACE_EXIT__894!(u32, $status);
}
};
}

macro_rules! return_ACPI_STATUS__956 {
($status:expr) => {
{
ACPI_TRACE_EXIT__939!(u32, $status);
}
};
}

macro_rules! return_ACPI_STATUS__967 {
($status:expr) => {
{
ACPI_TRACE_EXIT__213!(u32, $status);
}
};
}

macro_rules! return_ACPI_STATUS__1019 {
($status:expr) => {
{
ACPI_TRACE_EXIT__591!(u32, $status);
}
};
}

macro_rules! return_ACPI_STATUS__1025 {
($status:expr) => {
{
ACPI_TRACE_EXIT__926!(u32, $status);
}
};
}

macro_rules! return_ACPI_STATUS__1107 {
($status:expr) => {
{
ACPI_TRACE_EXIT__669!(u32, $status);
}
};
}

unsafe extern "C" {
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut acpi_dbg_level: u32;
    static mut acpi_dbg_layer: u32;
    static mut acpi_gbl_exception_handler: Option<unsafe extern "C" fn(u32, u32, u16, u32, *mut ()) -> u32>;
    static mut acpi_method_count: u32;
    fn __bad_size_call_parameter();
    fn __compiletime_assert_5();
    fn kfree(arg0: *const ());
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn __compiletime_assert_208();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn acpi_os_create_semaphore(arg0: u32, arg1: u32, arg2: *mut *mut ()) -> u32;
    fn acpi_os_signal_semaphore(arg0: *mut (), arg1: u32) -> u32;
    fn acpi_error(arg0: *const u8, arg1: u32, arg2: *const u8, ...);
    fn acpi_info(arg0: *const u8, ...);
    fn acpi_debug_print(arg0: u32, arg1: u32, arg2: *const u8, arg3: *const u8, arg4: u32, arg5: *const u8, ...);
    fn acpi_debug_print_raw(arg0: u32, arg1: u32, arg2: *const u8, arg3: *const u8, arg4: u32, arg5: *const u8, ...);
    fn acpi_ut_get_node_name(arg0: *mut ()) -> *const u8;
    fn acpi_ut_track_stack_ptr();
    fn acpi_ut_trace(arg0: u32, arg1: *const u8, arg2: *const u8, arg3: u32);
    fn acpi_ut_trace_ptr(arg0: u32, arg1: *const u8, arg2: *const u8, arg3: u32, arg4: *const ());
    fn acpi_ut_exit(arg0: u32, arg1: *const u8, arg2: *const u8, arg3: u32);
    fn acpi_ut_status_exit(arg0: u32, arg1: *const u8, arg2: *const u8, arg3: u32, arg4: u32);
    fn acpi_ut_remove_reference(arg0: *mut acpi_operand_object);
    fn acpi_ut_create_internal_object_dbg(arg0: *const u8, arg1: u32, arg2: u32, arg3: u32) -> *mut acpi_operand_object;
    fn acpi_ut_delete_object_desc(arg0: *mut acpi_operand_object);
    fn acpi_ut_allocate_owner_id(arg0: *mut u16) -> u32;
    fn acpi_ut_release_owner_id(arg0: *mut u16);
    fn acpi_ds_method_data_delete_all(arg0: *mut acpi_walk_state);
    fn acpi_ds_clear_implicit_return(arg0: *mut acpi_walk_state);
    fn acpi_ds_do_implicit_return(arg0: *mut acpi_operand_object, arg1: *mut acpi_walk_state, arg2: u8) -> u8;
    fn acpi_ds_clear_operands(arg0: *mut acpi_walk_state);
    fn acpi_ds_create_walk_state(arg0: u16, arg1: *mut acpi_parse_object, arg2: *mut acpi_operand_object, arg3: *mut acpi_thread_state) -> *mut acpi_walk_state;
    fn acpi_ds_init_aml_walk(arg0: *mut acpi_walk_state, arg1: *mut acpi_parse_object, arg2: *mut acpi_namespace_node, arg3: *mut u8, arg4: u32, arg5: *mut acpi_evaluate_info, arg6: u8) -> u32;
    fn acpi_ds_delete_walk_state(arg0: *mut acpi_walk_state);
    fn acpi_ds_pop_walk_state(arg0: *mut acpi_thread_state) -> *mut acpi_walk_state;
    fn acpi_ds_result_push(arg0: *mut acpi_operand_object, arg1: *mut acpi_walk_state) -> u32;
    fn acpi_ds_dump_method_stack(arg0: u32, arg1: *mut acpi_walk_state, arg2: *mut acpi_parse_object);
    fn acpi_ex_start_trace_method(arg0: *mut acpi_namespace_node, arg1: *mut acpi_operand_object, arg2: *mut acpi_walk_state);
    fn acpi_ex_stop_trace_method(arg0: *mut acpi_namespace_node, arg1: *mut acpi_operand_object, arg2: *mut acpi_walk_state);
    fn acpi_ex_system_wait_mutex(arg0: *mut (), arg1: u16) -> u32;
    fn acpi_ex_enter_interpreter();
    fn acpi_ex_exit_interpreter();
    fn acpi_ns_delete_namespace_subtree(arg0: *mut acpi_namespace_node);
    fn acpi_ns_delete_namespace_by_owner(arg0: u16);
    fn acpi_ns_get_normalized_pathname(arg0: *mut acpi_namespace_node, arg1: u8) -> *mut u8;
    fn acpi_ns_get_attached_object(arg0: *mut acpi_namespace_node) -> *mut acpi_operand_object;
    fn acpi_ps_parse_aml(arg0: *mut acpi_walk_state) -> u32;
    fn acpi_ps_delete_parse_tree(arg0: *mut acpi_parse_object);
    fn acpi_ps_alloc_op(arg0: u16, arg1: *mut u8) -> *mut acpi_parse_object;
    fn acpi_ps_free_op(arg0: *mut acpi_parse_object);
    fn acpi_ps_set_name(arg0: *mut acpi_parse_object, arg1: u32);
}

unsafe extern "llvm-intrinsic" {
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
struct raw_spinlock {
    raw_lock: qspinlock,
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
fn get_current() -> *mut task_struct {
if IS_ENABLED!() != 0 {
let mut __v0: *mut task_struct;
loop {
if true {
unsafe { __compiletime_assert_5() };
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

// On 2MB or 1GB pages
// Extracts the PFN from a (pte|pmd|pud|pgd)val_t of a 4KB page
// Extracts the flags from a (pte|pmd|pud|pgd)val_t
// This includes the protection key value.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_types_h_295 {
    pgd: u64,
}


// No need to mask any bits for !PAE
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits63U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_bitmap {
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
struct file {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_nodemask_types_h_17 {
    bits: [u64; 1],
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
    __slate_bits_0: __slate_bits::__SlateBits64U0,
    pkru: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
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
struct pollfd {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct thread_info {
    flags: u64,
    syscall_work: u64,
    status: u32,
    cpu: u32,
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
struct mutex {
    owner: __SlateRecord_types_h_197,
    wait_lock: raw_spinlock,
    osq: optimistic_spin_queue,
    first_waiter: *mut mutex_waiter,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
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
struct mutex_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct compat_robust_list_head {
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
struct futex_private_hash {
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
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sem_undo_list {
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
struct hrtimer_clock_base {
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
struct seccomp_filter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct refcount_struct {
    refs: __SlateRecord_types_h_190,
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
struct sigpending {
    list: list_head,
    signal: __SlateRecord_signal_h_25,
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
struct rseq {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq_ids {
    __slate_anon_0: __SlateRecord_rseq_types_h_74,
    node_id: u32,
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
struct rseq_data {
    usrptr: *mut rseq,
    len: u32,
    sig: u32,
    event: rseq_event,
    ids: rseq_ids,
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
struct __SlateRecord_uidgid_types_h_9 {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct prev_cputime {
    utime: u64,
    stime: u64,
    lock: raw_spinlock,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct io_context {
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
struct sighand_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct signal_struct {
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
    __slate_bits_0: __slate_bits::__SlateBits198U0,
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
struct rq_flags {
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
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
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
    __slate_bits_0: __slate_bits::__SlateBits46U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits46U1,
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
    __slate_bits_2: __slate_bits::__SlateBits46U2,
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
union __SlateRecord_sched_h_826 {
    ipi_mask_ptr: *mut cpumask,
    ipi_mask_val: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sched_class {
}


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
#[repr(C)]
#[derive(Clone, Copy)]
struct sched_cache_time {
    runtime: u64,
    epoch: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pid {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cred {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key {
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
struct gendisk {
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
struct vm_struct {
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
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct completion {
    done: u32,
    wait: swait_queue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
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
struct uprobe {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe_xol_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct old_timespec32 {
    tv_sec: i32,
    tv_nsec: i32,
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
struct srcu_ctr {
    srcu_locks: __SlateRecord_types_h_197,
    srcu_unlocks: __SlateRecord_types_h_197,
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
struct hprobe {
    state: u32,
    srcu_scp: *mut srcu_ctr,
    uprobe: *mut uprobe,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
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
struct ldt_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vdso_image {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct address_space {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobes_state {
    xol_area: *mut xol_area,
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
struct kmem_cache {
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


// Opaque type representing current VMA (vm_area_struct) flag state. Must be
// accessed via vma_flags_xxx() helper functions.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_871 {
    __vma_flags: [u64; 1],
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

fn _kzalloc_noprof(size: u64, flags: u32, token: __SlateRecord_slab_h_532) -> *mut () {
_kmalloc_noprof(size, flags | (BIT!(8i32) as u32), token)
}

#[repr(C)]
#[derive(Clone, Copy)]
union acpi_name_union {
    integer: u32,
    ascii: [u8; 4],
}


fn acpi_os_free(memory: *mut ()) {
unsafe { kfree(memory as *const ()) };
}

fn acpi_os_get_thread_id() -> u64 {
get_current() as u64
}

#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_namespace_node {
    object: *mut acpi_operand_object,
    descriptor_type: u8,
    r#type: u8,
    flags: u16,
    name: acpi_name_union,
    parent: *mut acpi_namespace_node,
    child: *mut acpi_namespace_node,
    peer: *mut acpi_namespace_node,
    owner_id: u16,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_name_info {
    name: [u8; 4],
    argument_list: u16,
    expected_btypes: u8,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_package_info {
    r#type: u8,
    object_type1: u8,
    count1: u8,
    object_type2: u8,
    count2: u8,
    reserved: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_package_info2 {
    r#type: u8,
    count: u8,
    object_type: [u8; 4],
    reserved: u8,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_package_info3 {
    r#type: u8,
    count: u8,
    object_type: [u8; 2],
    tail_object_type: u8,
    reserved: u16,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct acpi_package_info4 {
    r#type: u8,
    object_type1: u8,
    count1: u8,
    sub_object_types: u8,
    pkg_count: u8,
    reserved: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
union acpi_predefined_info {
    info: acpi_name_info,
    ret_info: acpi_package_info,
    ret_info2: acpi_package_info2,
    ret_info3: acpi_package_info3,
    ret_info4: acpi_package_info4,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_gpe_handler_info {
    address: Option<unsafe extern "C" fn(*mut (), u32, *mut ()) -> u32>,
    context: *mut (),
    method_node: *mut acpi_namespace_node,
    original_flags: u8,
    originally_enabled: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_gpe_notify_info {
    device_node: *mut acpi_namespace_node,
    next: *mut acpi_gpe_notify_info,
}


#[repr(C)]
#[derive(Clone, Copy)]
union acpi_gpe_dispatch_info {
    method_node: *mut acpi_namespace_node,
    handler: *mut acpi_gpe_handler_info,
    notify_list: *mut acpi_gpe_notify_info,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_gpe_event_info {
    dispatch: acpi_gpe_dispatch_info,
    register_info: *mut acpi_gpe_register_info,
    flags: u8,
    gpe_number: u8,
    runtime_count: u8,
    disable_for_dispatch: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_gpe_address {
    space_id: u8,
    address: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_gpe_register_info {
    status_address: acpi_gpe_address,
    enable_address: acpi_gpe_address,
    base_gpe_number: u16,
    enable_for_wake: u8,
    enable_for_run: u8,
    mask_for_run: u8,
    enable_mask: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_gpe_block_info {
    node: *mut acpi_namespace_node,
    previous: *mut acpi_gpe_block_info,
    next: *mut acpi_gpe_block_info,
    xrupt_block: *mut acpi_gpe_xrupt_info,
    register_info: *mut acpi_gpe_register_info,
    event_info: *mut acpi_gpe_event_info,
    address: u64,
    register_count: u32,
    gpe_count: u16,
    block_base_number: u16,
    space_id: u8,
    initialized: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_gpe_xrupt_info {
    previous: *mut acpi_gpe_xrupt_info,
    next: *mut acpi_gpe_xrupt_info,
    gpe_block_list_head: *mut acpi_gpe_block_info,
    interrupt_number: u32,
}


// To differentiate various internal objs
#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_common_state {
    next: *mut (),
    descriptor_type: u8,
    flags: u8,
    value: u16,
    state: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_update_state {
    next: *mut (),
    descriptor_type: u8,
    flags: u8,
    value: u16,
    state: u16,
    object: *mut acpi_operand_object,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_pkg_state {
    next: *mut (),
    descriptor_type: u8,
    flags: u8,
    value: u16,
    state: u16,
    index: u32,
    source_object: *mut acpi_operand_object,
    dest_object: *mut acpi_operand_object,
    walk_state: *mut acpi_walk_state,
    this_target_obj: *mut (),
    num_packages: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_control_state {
    next: *mut (),
    descriptor_type: u8,
    flags: u8,
    value: u16,
    state: u16,
    opcode: u16,
    predicate_op: *mut acpi_parse_object,
    aml_predicate_start: *mut u8,
    package_end: *mut u8,
    loop_timeout: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_scope_state {
    next: *mut (),
    descriptor_type: u8,
    flags: u8,
    value: u16,
    state: u16,
    node: *mut acpi_namespace_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_pscope_state {
    next: *mut (),
    descriptor_type: u8,
    flags: u8,
    value: u16,
    state: u16,
    arg_count: u32,
    op: *mut acpi_parse_object,
    arg_end: *mut u8,
    pkg_end: *mut u8,
    arg_list: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_thread_state {
    next: *mut (),
    descriptor_type: u8,
    flags: u8,
    value: u16,
    state: u16,
    current_sync_level: u8,
    walk_state_list: *mut acpi_walk_state,
    acquired_mutex_list: *mut acpi_operand_object,
    thread_id: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_result_values {
    next: *mut (),
    descriptor_type: u8,
    flags: u8,
    value: u16,
    state: u16,
    obj_desc: [*mut acpi_operand_object; 8],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_global_notify_handler {
    handler: Option<unsafe extern "C" fn(*mut (), u32, *mut ())>,
    context: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_notify_info {
    next: *mut (),
    descriptor_type: u8,
    flags: u8,
    value: u16,
    state: u16,
    handler_list_id: u8,
    node: *mut acpi_namespace_node,
    handler_list_head: *mut acpi_operand_object,
    global: *mut acpi_global_notify_handler,
}


#[repr(C)]
#[derive(Clone, Copy)]
union acpi_generic_state {
    common: acpi_common_state,
    control: acpi_control_state,
    update: acpi_update_state,
    scope: acpi_scope_state,
    parse_scope: acpi_pscope_state,
    pkg: acpi_pkg_state,
    thread: acpi_thread_state,
    results: acpi_result_values,
    notify: acpi_notify_info,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_opcode_info {
    name: *mut u8,
    parse_args: u32,
    runtime_args: u32,
    flags: u16,
    object_type: u8,
    class: u8,
    r#type: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union acpi_parse_value {
    integer: u64,
    size: u32,
    string: *mut u8,
    buffer: *mut u8,
    name: *mut u8,
    arg: *mut acpi_parse_object,
}


// Parent op
// To differentiate various internal objs
// Type of Op
// AML opcode
// Address of declaration in AML
// Next op
// For use by interpreter
// Value or args associated with the opcode
// Number of elements in the arg list
// Used during AML disassembly
// Subtype used for disassembly
// Used for C-style operator name strings
// Op name (debug only)
// Inline comment
// End of node comment
// Comment associated with the first parameter of the name node
// Comments that come after } on the same as }
// comments that appears before this node
// comments that at the end of a block but before ) or }
// Filename associated with this node. Used for ASL/ASL+ converter
// Parent filename associated with this node. Used for ASL/ASL+ converter
#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_parse_obj_common {
    parent: *mut acpi_parse_object,
    descriptor_type: u8,
    flags: u8,
    aml_opcode: u16,
    aml: *mut u8,
    next: *mut acpi_parse_object,
    node: *mut acpi_namespace_node,
    value: acpi_parse_value,
    arg_list_length: u8,
    disasm_flags: u16,
    disasm_opcode: u8,
    operator_symbol: *mut u8,
    aml_op_name: [u8; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_parse_obj_named {
    parent: *mut acpi_parse_object,
    descriptor_type: u8,
    flags: u8,
    aml_opcode: u16,
    aml: *mut u8,
    next: *mut acpi_parse_object,
    node: *mut acpi_namespace_node,
    value: acpi_parse_value,
    arg_list_length: u8,
    disasm_flags: u16,
    disasm_opcode: u8,
    operator_symbol: *mut u8,
    aml_op_name: [u8; 16],
    path: *mut u8,
    data: *mut u8,
    length: u32,
    name: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_parse_obj_asl {
    parent: *mut acpi_parse_object,
    descriptor_type: u8,
    flags: u8,
    aml_opcode: u16,
    aml: *mut u8,
    next: *mut acpi_parse_object,
    node: *mut acpi_namespace_node,
    value: acpi_parse_value,
    arg_list_length: u8,
    disasm_flags: u16,
    disasm_opcode: u8,
    operator_symbol: *mut u8,
    aml_op_name: [u8; 16],
    child: *mut acpi_parse_object,
    parent_method: *mut acpi_parse_object,
    filename: *mut u8,
    file_changed: u8,
    parent_filename: *mut u8,
    external_name: *mut u8,
    namepath: *mut u8,
    name_seg: [u8; 4],
    extra_value: u32,
    column: u32,
    line_number: u32,
    logical_line_number: u32,
    logical_byte_offset: u32,
    end_line: u32,
    end_logical_line: u32,
    acpi_btype: u32,
    aml_length: u32,
    aml_subtree_length: u32,
    final_aml_length: u32,
    final_aml_offset: u32,
    compile_flags: u32,
    parse_opcode: u16,
    aml_opcode_length: u8,
    aml_pkg_len_bytes: u8,
    extra: u8,
    parse_op_name: [u8; 20],
}


#[repr(C)]
#[derive(Clone, Copy)]
union acpi_parse_object {
    common: acpi_parse_obj_common,
    named: acpi_parse_obj_named,
    asl: acpi_parse_obj_asl,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_parse_state {
    aml_start: *mut u8,
    aml: *mut u8,
    aml_end: *mut u8,
    pkg_start: *mut u8,
    pkg_end: *mut u8,
    start_op: *mut acpi_parse_object,
    start_node: *mut acpi_namespace_node,
    scope: *mut acpi_generic_state,
    start_scope: *mut acpi_parse_object,
    aml_size: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_acobject_h_160 {
    implementation: Option<unsafe extern "C" fn(*mut acpi_walk_state) -> u32>,
    handler: *mut acpi_operand_object,
}


// Objects linked to parent NS node
// To differentiate various internal objs
// acpi_object_type
// For object deletion management
#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_object_common {
    next_object: *mut acpi_operand_object,
    descriptor_type: u8,
    r#type: u8,
    reference_count: u16,
    flags: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_object_integer {
    next_object: *mut acpi_operand_object,
    descriptor_type: u8,
    r#type: u8,
    reference_count: u16,
    flags: u8,
    fill: [u8; 3],
    value: u64,
}


// Note: The String and Buffer object must be identical through the
// pointer and length elements. There is code that depends on this.
//
// Fields common to both Strings and Buffers
#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_object_string {
    next_object: *mut acpi_operand_object,
    descriptor_type: u8,
    r#type: u8,
    reference_count: u16,
    flags: u8,
    pointer: *mut u8,
    length: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_object_buffer {
    next_object: *mut acpi_operand_object,
    descriptor_type: u8,
    r#type: u8,
    reference_count: u16,
    flags: u8,
    pointer: *mut u8,
    length: u32,
    aml_length: u32,
    aml_start: *mut u8,
    node: *mut acpi_namespace_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_object_package {
    next_object: *mut acpi_operand_object,
    descriptor_type: u8,
    r#type: u8,
    reference_count: u16,
    flags: u8,
    node: *mut acpi_namespace_node,
    elements: *mut *mut acpi_operand_object,
    aml_start: *mut u8,
    aml_length: u32,
    count: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_object_event {
    next_object: *mut acpi_operand_object,
    descriptor_type: u8,
    r#type: u8,
    reference_count: u16,
    flags: u8,
    os_semaphore: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_object_mutex {
    next_object: *mut acpi_operand_object,
    descriptor_type: u8,
    r#type: u8,
    reference_count: u16,
    flags: u8,
    sync_level: u8,
    acquisition_depth: u16,
    os_mutex: *mut (),
    thread_id: u64,
    owner_thread: *mut acpi_thread_state,
    prev: *mut acpi_operand_object,
    next: *mut acpi_operand_object,
    node: *mut acpi_namespace_node,
    original_sync_level: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_object_region {
    next_object: *mut acpi_operand_object,
    descriptor_type: u8,
    r#type: u8,
    reference_count: u16,
    flags: u8,
    space_id: u8,
    node: *mut acpi_namespace_node,
    handler: *mut acpi_operand_object,
    next: *mut acpi_operand_object,
    address: u64,
    length: u32,
    pointer: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_object_method {
    next_object: *mut acpi_operand_object,
    descriptor_type: u8,
    r#type: u8,
    reference_count: u16,
    flags: u8,
    info_flags: u8,
    param_count: u8,
    sync_level: u8,
    mutex: *mut acpi_operand_object,
    node: *mut acpi_operand_object,
    aml_start: *mut u8,
    dispatch: __SlateRecord_acobject_h_160,
    aml_length: u32,
    owner_id: u16,
    thread_count: u8,
}


// Method is actually module-level code
// Method is implemented internally (_OSI)
// Method is serialized
// Method is to be marked serialized
// Method was auto-serialized at table load time
// Method modified the namespace
// Common fields for objects that support ASL notifications
// Handlers for system/device notifies
// Handler for Address space
#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_object_notify_common {
    next_object: *mut acpi_operand_object,
    descriptor_type: u8,
    r#type: u8,
    reference_count: u16,
    flags: u8,
    notify_list: [*mut acpi_operand_object; 2],
    handler: *mut acpi_operand_object,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_object_device {
    next_object: *mut acpi_operand_object,
    descriptor_type: u8,
    r#type: u8,
    reference_count: u16,
    flags: u8,
    notify_list: [*mut acpi_operand_object; 2],
    handler: *mut acpi_operand_object,
    gpe_block: *mut acpi_gpe_block_info,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_object_power_resource {
    next_object: *mut acpi_operand_object,
    descriptor_type: u8,
    r#type: u8,
    reference_count: u16,
    flags: u8,
    notify_list: [*mut acpi_operand_object; 2],
    handler: *mut acpi_operand_object,
    system_level: u32,
    resource_order: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_object_processor {
    next_object: *mut acpi_operand_object,
    descriptor_type: u8,
    r#type: u8,
    reference_count: u16,
    flags: u8,
    proc_id: u8,
    length: u8,
    notify_list: [*mut acpi_operand_object; 2],
    handler: *mut acpi_operand_object,
    address: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_object_thermal_zone {
    next_object: *mut acpi_operand_object,
    descriptor_type: u8,
    r#type: u8,
    reference_count: u16,
    flags: u8,
    notify_list: [*mut acpi_operand_object; 2],
    handler: *mut acpi_operand_object,
}


// Common bitfield for the field objects
// "Field Datum"  -- a datum from the actual field object
// "Buffer Datum" -- a datum from a user buffer, read from or to be written to the field
// Access, update, and lock bits
// From access_as keyword
// Read/Write size in bytes
// Link back to parent node
// Length of field in bits
// Byte offset within containing object
// Value to store into the Bank or Index register
// Bit offset within first field datum (0-63)
// For serial regions/fields
#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_object_field_common {
    next_object: *mut acpi_operand_object,
    descriptor_type: u8,
    r#type: u8,
    reference_count: u16,
    flags: u8,
    field_flags: u8,
    attribute: u8,
    access_byte_width: u8,
    node: *mut acpi_namespace_node,
    bit_length: u32,
    base_byte_offset: u32,
    value: u32,
    start_field_bit_offset: u8,
    access_length: u8,
    region_obj: *mut acpi_operand_object,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_object_region_field {
    next_object: *mut acpi_operand_object,
    descriptor_type: u8,
    r#type: u8,
    reference_count: u16,
    flags: u8,
    field_flags: u8,
    attribute: u8,
    access_byte_width: u8,
    node: *mut acpi_namespace_node,
    bit_length: u32,
    base_byte_offset: u32,
    value: u32,
    start_field_bit_offset: u8,
    access_length: u8,
    resource_length: u16,
    region_obj: *mut acpi_operand_object,
    resource_buffer: *mut u8,
    pin_number_index: u16,
    internal_pcc_buffer: *mut u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_object_bank_field {
    next_object: *mut acpi_operand_object,
    descriptor_type: u8,
    r#type: u8,
    reference_count: u16,
    flags: u8,
    field_flags: u8,
    attribute: u8,
    access_byte_width: u8,
    node: *mut acpi_namespace_node,
    bit_length: u32,
    base_byte_offset: u32,
    value: u32,
    start_field_bit_offset: u8,
    access_length: u8,
    region_obj: *mut acpi_operand_object,
    bank_obj: *mut acpi_operand_object,
}


mod __slate_bits {
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits198U0 {
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
    pub struct __SlateBits46U0 {
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
    pub struct __SlateBits46U1 {
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
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits46U2 {
        #[bits(1)]
        pub mce_ripv: u64,
        #[bits(1)]
        pub mce_whole_page: u64,
        #[bits(62)]
        pub __mce_reserved: u64,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits63U0 {
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
    pub struct __SlateBits64U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_object_index_field {
    next_object: *mut acpi_operand_object,
    descriptor_type: u8,
    r#type: u8,
    reference_count: u16,
    flags: u8,
    field_flags: u8,
    attribute: u8,
    access_byte_width: u8,
    node: *mut acpi_namespace_node,
    bit_length: u32,
    base_byte_offset: u32,
    value: u32,
    start_field_bit_offset: u8,
    access_length: u8,
    index_obj: *mut acpi_operand_object,
    data_obj: *mut acpi_operand_object,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_object_buffer_field {
    next_object: *mut acpi_operand_object,
    descriptor_type: u8,
    r#type: u8,
    reference_count: u16,
    flags: u8,
    field_flags: u8,
    attribute: u8,
    access_byte_width: u8,
    node: *mut acpi_namespace_node,
    bit_length: u32,
    base_byte_offset: u32,
    value: u32,
    start_field_bit_offset: u8,
    access_length: u8,
    is_create_field: u8,
    buffer_obj: *mut acpi_operand_object,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_object_notify_handler {
    next_object: *mut acpi_operand_object,
    descriptor_type: u8,
    r#type: u8,
    reference_count: u16,
    flags: u8,
    node: *mut acpi_namespace_node,
    handler_type: u32,
    handler: Option<unsafe extern "C" fn(*mut (), u32, *mut ())>,
    context: *mut (),
    next: [*mut acpi_operand_object; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_object_addr_handler {
    next_object: *mut acpi_operand_object,
    descriptor_type: u8,
    r#type: u8,
    reference_count: u16,
    flags: u8,
    space_id: u8,
    handler_flags: u8,
    handler: Option<unsafe extern "C" fn(u32, u64, u32, *mut u64, *mut (), *mut ()) -> u32>,
    node: *mut acpi_namespace_node,
    context: *mut (),
    context_mutex: *mut (),
    setup: Option<unsafe extern "C" fn(*mut (), u32, *mut (), *mut *mut ()) -> u32>,
    region_list: *mut acpi_operand_object,
    next: *mut acpi_operand_object,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_object_reference {
    next_object: *mut acpi_operand_object,
    descriptor_type: u8,
    r#type: u8,
    reference_count: u16,
    flags: u8,
    class: u8,
    target_type: u8,
    resolved: u8,
    object: *mut (),
    node: *mut acpi_namespace_node,
    r#where: *mut *mut acpi_operand_object,
    index_pointer: *mut u8,
    aml: *mut u8,
    value: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_object_extra {
    next_object: *mut acpi_operand_object,
    descriptor_type: u8,
    r#type: u8,
    reference_count: u16,
    flags: u8,
    method_REG: *mut acpi_namespace_node,
    scope_node: *mut acpi_namespace_node,
    region_context: *mut (),
    aml_start: *mut u8,
    aml_length: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_object_data {
    next_object: *mut acpi_operand_object,
    descriptor_type: u8,
    r#type: u8,
    reference_count: u16,
    flags: u8,
    handler: Option<unsafe extern "C" fn(*mut (), *mut ())>,
    pointer: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_object_cache_list {
    next_object: *mut acpi_operand_object,
    descriptor_type: u8,
    r#type: u8,
    reference_count: u16,
    flags: u8,
    next: *mut acpi_operand_object,
}


#[repr(C)]
#[derive(Clone, Copy)]
union acpi_operand_object {
    common: acpi_object_common,
    integer: acpi_object_integer,
    string: acpi_object_string,
    buffer: acpi_object_buffer,
    package: acpi_object_package,
    event: acpi_object_event,
    method: acpi_object_method,
    mutex: acpi_object_mutex,
    region: acpi_object_region,
    common_notify: acpi_object_notify_common,
    device: acpi_object_device,
    power_resource: acpi_object_power_resource,
    processor: acpi_object_processor,
    thermal_zone: acpi_object_thermal_zone,
    common_field: acpi_object_field_common,
    field: acpi_object_region_field,
    buffer_field: acpi_object_buffer_field,
    bank_field: acpi_object_bank_field,
    index_field: acpi_object_index_field,
    notify: acpi_object_notify_handler,
    address_space: acpi_object_addr_handler,
    reference: acpi_object_reference,
    extra: acpi_object_extra,
    data: acpi_object_data,
    cache: acpi_object_cache_list,
    node: acpi_namespace_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_walk_state {
    next: *mut acpi_walk_state,
    descriptor_type: u8,
    walk_type: u8,
    opcode: u16,
    next_op_info: u8,
    num_operands: u8,
    operand_index: u8,
    owner_id: u16,
    last_predicate: u8,
    current_result: u8,
    return_used: u8,
    scope_depth: u8,
    pass_number: u8,
    namespace_override: u8,
    result_size: u8,
    result_count: u8,
    aml: *mut u8,
    arg_types: u32,
    method_breakpoint: u32,
    user_breakpoint: u32,
    parse_flags: u32,
    parser_state: acpi_parse_state,
    prev_arg_types: u32,
    arg_count: u32,
    method_nesting_depth: u16,
    method_is_nested: u8,
    arguments: [acpi_namespace_node; 7],
    local_variables: [acpi_namespace_node; 8],
    operands: [*mut acpi_operand_object; 9],
    params: *mut *mut acpi_operand_object,
    aml_last_while: *mut u8,
    caller_return_desc: *mut *mut acpi_operand_object,
    control_state: *mut acpi_generic_state,
    deferred_node: *mut acpi_namespace_node,
    implicit_return_obj: *mut acpi_operand_object,
    method_call_node: *mut acpi_namespace_node,
    method_call_op: *mut acpi_parse_object,
    method_desc: *mut acpi_operand_object,
    method_node: *mut acpi_namespace_node,
    method_pathname: *mut u8,
    op: *mut acpi_parse_object,
    op_info: *const acpi_opcode_info,
    origin: *mut acpi_parse_object,
    result_obj: *mut acpi_operand_object,
    results: *mut acpi_generic_state,
    return_desc: *mut acpi_operand_object,
    scope_info: *mut acpi_generic_state,
    prev_op: *mut acpi_parse_object,
    next_op: *mut acpi_parse_object,
    thread: *mut acpi_thread_state,
    descending_callback: Option<unsafe extern "C" fn(*mut acpi_walk_state, *mut *mut acpi_parse_object) -> u32>,
    ascending_callback: Option<unsafe extern "C" fn(*mut acpi_walk_state) -> u32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct acpi_evaluate_info {
    prefix_node: *mut acpi_namespace_node,
    relative_pathname: *const u8,
    parameters: *mut *mut acpi_operand_object,
    node: *mut acpi_namespace_node,
    obj_desc: *mut acpi_operand_object,
    full_pathname: *mut u8,
    predefined: *const acpi_predefined_info,
    return_object: *mut acpi_operand_object,
    parent_package: *mut acpi_operand_object,
    return_flags: u32,
    return_btype: u32,
    param_count: u16,
    node_flags: u16,
    pass_number: u8,
    return_object_type: u8,
    flags: u8,
}


static _acpi_module_name: [u8; 9] = [100, 115, 109, 101, 116, 104, 111, 100, 0];

// FUNCTION:    acpi_ds_auto_serialize_method
//
// PARAMETERS:  node                        - Namespace Node of the method
//              obj_desc                    - Method object attached to node
//
// RETURN:      Status
//
// DESCRIPTION: Parse a control method AML to scan for control methods that
//              need serialization due to the creation of named objects.
//
// NOTE: It is a bit of overkill to mark all such methods serialized, since
// there is only a problem if the method actually blocks during execution.
// A blocking operation is, for example, a Sleep() operation, or any access
// to an operation region. However, it is probably not possible to easily
// detect whether a method will block or not, so we simply mark all suspicious
// methods as serialized.
//
// NOTE2: This code is essentially a generic routine for parsing a single
// control method.
#[unsafe(no_mangle)]
extern "C" fn acpi_ds_auto_serialize_method(node: *mut acpi_namespace_node, obj_desc: *mut acpi_operand_object) -> u32 {
let mut status: u32 = 0;
let mut op: *mut acpi_parse_object = core::ptr::null_mut::<acpi_parse_object>();
ACPI_FUNCTION_TRACE_PTR__82!(node);
loop {
if ACPI_IS_DEBUG_ENABLED!(64u32, 64i32) {
unsafe { acpi_debug_print(64u32, 64u32, c"acpi_ds_auto_serialize_method".as_ptr() as *const u8, core::ptr::addr_of!(_acpi_module_name) as *const u8, 64u32, c"Method auto-serialization parse [%4.4s] %p\n".as_ptr() as *const u8, acpi_ut_get_node_name(node as *mut ()), node) };
}
if true {
break;
}
}
// Create/Init a root op for the method parse tree
op = unsafe { acpi_ps_alloc_op(20u16, (*obj_desc).method.aml_start) };
if op.is_null() {
return_ACPI_STATUS__68!(EXCEP_ENV!(4i32));
}
unsafe { acpi_ps_set_name(op, (*node).name.integer) };
unsafe {
(*op).common.node = node;
}
// Create and initialize a new walk state
let walk_state: *mut acpi_walk_state = unsafe { acpi_ds_create_walk_state((*node).owner_id, core::ptr::null_mut::<acpi_parse_object>(), core::ptr::null_mut::<acpi_operand_object>(), core::ptr::null_mut::<acpi_thread_state>()) };
if walk_state.is_null() {
unsafe { acpi_ps_free_op(op) };
return_ACPI_STATUS__475!(EXCEP_ENV!(4i32));
}
status = unsafe { acpi_ds_init_aml_walk(walk_state, op, node, (*obj_desc).method.aml_start, (*obj_desc).method.aml_length, core::ptr::null_mut::<acpi_evaluate_info>(), 0u8) };
if ACPI_FAILURE!(status) != 0 {
unsafe { acpi_ds_delete_walk_state(walk_state) };
unsafe { acpi_ps_free_op(op) };
return_ACPI_STATUS__315!(status);
}
unsafe {
(*walk_state).descending_callback = Some(acpi_ds_detect_named_opcodes);
}
// Parse the method, scan for creation of named objects
status = unsafe { acpi_ps_parse_aml(walk_state) };
unsafe { acpi_ps_delete_parse_tree(op) };
return_ACPI_STATUS__1025!(status);
unsafe { core::mem::zeroed() }
}

// FUNCTION:    acpi_ds_detect_named_opcodes
//
// PARAMETERS:  walk_state      - Current state of the parse tree walk
//              out_op          - Unused, required for parser interface
//
// RETURN:      Status
//
// DESCRIPTION: Descending callback used during the loading of ACPI tables.
//              Currently used to detect methods that must be marked serialized
//              in order to avoid problems with the creation of named objects.
/// Local prototypes
#[unsafe(link_section = ".text.slate_distinct.dsmethod.acpi_ds_detect_named_opcodes")]
extern "C" fn acpi_ds_detect_named_opcodes(walk_state: *mut acpi_walk_state, out_op: *mut *mut acpi_parse_object) -> u32 {
// We are only interested in opcodes that create a new name
if ((unsafe { (*(*walk_state).op_info).flags }) as i32) & (64 | 8 | 16) == 0 {
return 0;
}
// At this point, we know we have a Named object opcode.
// Mark the method as serialized. Later code will create a mutex for
// this method to enforce serialization.
//
// Note, ACPI_METHOD_IGNORE_SYNC_LEVEL flag means that we will ignore the
// Sync Level mechanism for this method, even though it is now serialized.
// Otherwise, there can be conflicts with existing ASL code that actually
// uses sync levels.
unsafe {
(*(*walk_state).method_desc).method.sync_level = 0;
}
let __v0: *mut acpi_operand_object = unsafe { (*walk_state).method_desc };
unsafe {
(*__v0).method.info_flags = (((*__v0).method.info_flags as i32) | (4 | 16)) as u8;
}
loop {
if ACPI_IS_DEBUG_ENABLED!(4u32, 64i32) {
unsafe { acpi_debug_print(4u32, 150u32, c"acpi_ds_detect_named_opcodes".as_ptr() as *const u8, core::ptr::addr_of!(_acpi_module_name) as *const u8, 64u32, c"Method serialized [%4.4s] %p - [%s] (%4.4X)\n".as_ptr() as *const u8, (*(*walk_state).method_node).name.ascii.as_mut_ptr() as *mut u8, (*walk_state).method_node, (*(*walk_state).op_info).name, (*walk_state).opcode as i32) };
}
if true {
break;
}
}
// Abort the parse, no need to examine this method any further
EXCEP_CTL!(3i32)
}

// FUNCTION:    acpi_ds_method_error
//
// PARAMETERS:  status          - Execution status
//              walk_state      - Current state
//
// RETURN:      Status
//
// DESCRIPTION: Called on method error. Invoke the global exception handler if
//              present, dump the method data if the debugger is configured
//
//              Note: Allows the exception handler to change the status code
#[unsafe(no_mangle)]
extern "C" fn acpi_ds_method_error(mut status: u32, walk_state: *mut acpi_walk_state) -> u32 {
let mut aml_offset: u32 = 0;
let mut name: u32 = 0;
ACPI_FUNCTION_ENTRY!();
// Ignore AE_OK and control exception codes
if ACPI_SUCCESS!(status) || status & 16384 != 0 {
return status;
}
// Invoke the global exception handler
if (unsafe { acpi_gbl_exception_handler }) != None {
// Exit the interpreter, allow handler to execute methods
unsafe { acpi_ex_exit_interpreter() };
// Handler can map the exception code to anything it wants, including
// AE_OK, in which case the executing method will not be aborted.
aml_offset = ACPI_PTR_DIFF!(unsafe { (*walk_state).aml }, unsafe { (*walk_state).parser_state.aml_start }) as u32;
if !unsafe { (*walk_state).method_node }.is_null() {
name = unsafe { (*(*walk_state).method_node).name.integer };
} else {
if !unsafe { (*walk_state).deferred_node }.is_null() {
name = unsafe { (*(*walk_state).deferred_node).name.integer };
}
}
status = unsafe { unsafe { acpi_gbl_exception_handler }.unwrap()(status, name, (*walk_state).opcode, aml_offset, core::ptr::null_mut::<()>()) };
unsafe { acpi_ex_enter_interpreter() };
}
unsafe { acpi_ds_clear_implicit_return(walk_state) };
if ACPI_FAILURE!(status) != 0 {
unsafe { acpi_ds_dump_method_stack(status, walk_state, (*walk_state).op) };
// Display method locals/args if debugger is present
}
status
}

// FUNCTION:    acpi_ds_create_method_mutex
//
// PARAMETERS:  obj_desc            - The method object
//
// RETURN:      Status
//
// DESCRIPTION: Create a mutex object for a serialized control method
fn acpi_ds_create_method_mutex(method_desc: *mut acpi_operand_object) -> u32 {
let mut mutex_desc: *mut acpi_operand_object = unsafe { core::mem::zeroed() };
let mut status: u32 = 0;
ACPI_FUNCTION_TRACE__52!();
// Create the new mutex object
mutex_desc = acpi_ut_create_internal_object__21!(9i32);
if mutex_desc.is_null() {
return_ACPI_STATUS__156!(EXCEP_ENV!(4i32));
}
// Create the actual OS Mutex
status = acpi_os_create_mutex!(unsafe { core::ptr::addr_of_mut!((*mutex_desc).mutex.os_mutex) });
if ACPI_FAILURE!(status) != 0 {
unsafe { acpi_ut_delete_object_desc(mutex_desc) };
return_ACPI_STATUS__493!(status);
}
unsafe {
(*mutex_desc).mutex.sync_level = (*method_desc).method.sync_level;
}
unsafe {
(*method_desc).method.mutex = mutex_desc;
}
return_ACPI_STATUS__17!(0u32);
unsafe { core::mem::zeroed() }
}

// FUNCTION:    acpi_ds_begin_method_execution
//
// PARAMETERS:  method_node         - Node of the method
//              obj_desc            - The method object
//              walk_state          - current state, NULL if not yet executing
//                                    a method.
//
// RETURN:      Status
//
// DESCRIPTION: Prepare a method for execution. Parses the method if necessary,
//              increments the thread count, and waits at the method semaphore
//              for clearance to execute.
#[unsafe(no_mangle)]
extern "C" fn acpi_ds_begin_method_execution(method_node: *mut acpi_namespace_node, obj_desc: *mut acpi_operand_object, walk_state: *mut acpi_walk_state) -> u32 {
let mut status: u32 = unsafe { core::mem::zeroed() };
unsafe {
status = 0;
ACPI_FUNCTION_TRACE_PTR__49!(method_node);
if method_node.is_null() {
return_ACPI_STATUS__1019!(EXCEP_ENV!(10i32));
}
acpi_ex_start_trace_method(method_node, obj_desc, walk_state);
// Prevent wraparound of thread count
if ((*obj_desc).method.thread_count as i32) == 255i32 {
ACPI_ERROR!(core::ptr::addr_of!(_acpi_module_name) as *const u8, 309i32, c"Method reached maximum reentrancy limit (255)".as_ptr() as *mut u8);
return_ACPI_STATUS__1107!(EXCEP_AML!(19i32));
}
// If this method is serialized, we need to acquire the method mutex.
if ((*obj_desc).method.info_flags as i32) & 4 != 0 {
// Create a mutex for the method if it is defined to be Serialized
// and a mutex has not already been created. We defer the mutex creation
// until a method is actually executed, to minimize the object count
if unsafe { (*obj_desc).method.mutex }.is_null() {
status = acpi_ds_create_method_mutex(obj_desc);
if ACPI_FAILURE!(status) != 0 {
return_ACPI_STATUS__64!(status);
}
}
// The current_sync_level (per-thread) must be less than or equal to
// the sync level of the method. This mechanism provides some
// deadlock prevention.
//
// If the method was auto-serialized, we just ignore the sync level
// mechanism, because auto-serialization of methods can interfere
// with ASL code that actually uses sync levels.
//
// Top-level method invocation has no walk state at this point
if !walk_state.is_null() && ((*obj_desc).method.info_flags as i32) & 16 == 0 && ((*(*walk_state).thread).current_sync_level as i32) > ((*(*obj_desc).method.mutex).mutex.sync_level as i32) {
ACPI_ERROR!(core::ptr::addr_of!(_acpi_module_name) as *const u8, 346i32, c"Cannot acquire Mutex for method [%4.4s], current SyncLevel is too large (%u)".as_ptr() as *mut u8, acpi_ut_get_node_name(method_node as *mut ()), (*(*walk_state).thread).current_sync_level as i32);
return_ACPI_STATUS__767!(EXCEP_AML!(21i32));
}
// Obtain the method mutex if necessary. Do not acquire mutex for a
// recursive call.
if walk_state.is_null() || (*(*obj_desc).method.mutex).mutex.thread_id == 0 || (*(*walk_state).thread).thread_id != (*(*obj_desc).method.mutex).mutex.thread_id {
// Acquire the method mutex. This releases the interpreter if we
// block (and reacquires it before it returns)
status = acpi_ex_system_wait_mutex((*(*obj_desc).method.mutex).mutex.os_mutex, 65535u16);
if ACPI_FAILURE!(status) != 0 {
return_ACPI_STATUS__319!(status);
}
// Update the mutex and walk info and save the original sync_level
if !walk_state.is_null() {
(*(*obj_desc).method.mutex).mutex.original_sync_level = (*(*walk_state).thread).current_sync_level;
(*(*obj_desc).method.mutex).mutex.thread_id = (*(*walk_state).thread).thread_id;
// Update the current sync_level only if this is not an auto-
// serialized method. In the auto case, we have to ignore
// the sync level for the method mutex (created for the
// auto-serialization) because we have no idea of what the
// sync level should be. Therefore, just ignore it.
if ((*obj_desc).method.info_flags as i32) & 16 == 0 {
(*(*walk_state).thread).current_sync_level = (*obj_desc).method.sync_level;
}
} else {
(*(*obj_desc).method.mutex).mutex.original_sync_level = (*(*obj_desc).method.mutex).mutex.sync_level;
(*(*obj_desc).method.mutex).mutex.thread_id = acpi_os_get_thread_id();
}
}
// Always increase acquisition depth
let __v7: *mut acpi_operand_object = (*obj_desc).method.mutex;
(*__v7).mutex.acquisition_depth = ((*__v7).mutex.acquisition_depth as i32).wrapping_add(1) as u16;
}
'__join_2: {
// Allocate an Owner ID for this method, only if this is the first thread
// to begin concurrent execution. We only need one owner_id, even if the
// method is invoked recursively.
if (*obj_desc).method.owner_id == 0 {
status = acpi_ut_allocate_owner_id(core::ptr::addr_of_mut!((*obj_desc).method.owner_id));
if ACPI_FAILURE!(status) != 0 {
break '__join_2;
}
}
// Increment the method parse tree thread count since it has been
// reentered one more time (even if it is the same thread)
(*obj_desc).method.thread_count = ((*obj_desc).method.thread_count as i32).wrapping_add(1) as u8;
let __v1: u32 = acpi_method_count;
acpi_method_count = __v1.wrapping_add(1);
return_ACPI_STATUS__923!(status);
}
if !unsafe { (*obj_desc).method.mutex }.is_null() {
acpi_os_release_mutex!(unsafe { (*(*obj_desc).method.mutex).mutex.os_mutex });
}
// On error, must release the method mutex (if present)
return_ACPI_STATUS__780!(status);
}
unsafe { core::mem::zeroed() }
}

// FUNCTION:    acpi_ds_call_control_method
//
// PARAMETERS:  thread              - Info for this thread
//              this_walk_state     - Current walk state
//              op                  - Current Op to be walked
//
// RETURN:      Status
//
// DESCRIPTION: Transfer execution to a called control method
#[unsafe(no_mangle)]
extern "C" fn acpi_ds_call_control_method(thread: *mut acpi_thread_state, this_walk_state: *mut acpi_walk_state, op: *mut acpi_parse_object) -> u32 {
let mut __slate_compound_29: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_31: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __dummy2: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut u64 = core::ptr::addr_of_mut!(__dummy2) as *mut u64;
let mut __dummy: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut u64 = core::ptr::addr_of_mut!(__dummy) as *mut u64;
let mut __dummy2_1: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u64 = core::ptr::addr_of_mut!(__dummy2_1) as *mut u64;
let mut __dummy_1: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut u64 = core::ptr::addr_of_mut!(__dummy_1) as *mut u64;
let mut _flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u64 = core::ptr::addr_of_mut!(_flags) as *mut u64;
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut __dummy2_2: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut u64 = core::ptr::addr_of_mut!(__dummy2_2) as *mut u64;
let mut __dummy_2: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut u64 = core::ptr::addr_of_mut!(__dummy_2) as *mut u64;
let mut __dummy2: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut u64 = core::ptr::addr_of_mut!(__dummy2) as *mut u64;
let mut __dummy: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut u64 = core::ptr::addr_of_mut!(__dummy) as *mut u64;
let mut _flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut u64 = core::ptr::addr_of_mut!(_flags) as *mut u64;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut info: core::mem::MaybeUninit<*mut acpi_evaluate_info> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut *mut acpi_evaluate_info = core::ptr::addr_of_mut!(info) as *mut *mut acpi_evaluate_info;
let mut obj_desc: core::mem::MaybeUninit<*mut acpi_operand_object> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut *mut acpi_operand_object = core::ptr::addr_of_mut!(obj_desc) as *mut *mut acpi_operand_object;
let mut next_walk_state: *mut acpi_walk_state = unsafe { core::mem::zeroed() };
let mut method_node: core::mem::MaybeUninit<*mut acpi_namespace_node> = core::mem::MaybeUninit::uninit();
let __slate_slot_25: *mut *mut acpi_namespace_node = core::ptr::addr_of_mut!(method_node) as *mut *mut acpi_namespace_node;
let mut status: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_26: *mut u32 = core::ptr::addr_of_mut!(status) as *mut u32;
unsafe {
let __slate_auto_bytes_8 = __slate_slot_26 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_9 = __slate_slot_25 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 8usize);
next_walk_state = core::ptr::null_mut::<acpi_walk_state>();
let __slate_auto_bytes_10 = __slate_slot_23 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_10.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_11 = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_11.add(0usize), 0u8, 8usize);
ACPI_FUNCTION_TRACE_PTR__58!(this_walk_state);
loop {
if ACPI_IS_DEBUG_ENABLED!(256u32, 64i32) {
acpi_debug_print(256u32, 468u32, c"acpi_ds_call_control_method".as_ptr() as *const u8, core::ptr::addr_of!(_acpi_module_name) as *const u8, 64u32, c"Calling method %p, currentstate=%p\n".as_ptr() as *const u8, (*this_walk_state).prev_op, this_walk_state);
}
if true {
break;
}
}
// Get the namespace entry for the control method we are about to call
*__slate_slot_25 = (*this_walk_state).method_call_node;
if (*__slate_slot_25).is_null() {
return_ACPI_STATUS__956!(EXCEP_ENV!(10i32));
}
*__slate_slot_23 = acpi_ns_get_attached_object(*__slate_slot_25);
if (*__slate_slot_23).is_null() {
return_ACPI_STATUS__317!(EXCEP_ENV!(9i32));
}
if ((*this_walk_state).num_operands as i32) < ((*(*__slate_slot_23)).method.param_count as i32) {
ACPI_ERROR!(core::ptr::addr_of!(_acpi_module_name) as *const u8, 486i32, c"Missing argument(s) for method [%4.4s]".as_ptr() as *mut u8, acpi_ut_get_node_name(*__slate_slot_25 as *mut ()));
return_ACPI_STATUS__498!(EXCEP_AML!(38i32));
} else {
if ((*this_walk_state).num_operands as i32) > ((*(*__slate_slot_23)).method.param_count as i32) {
ACPI_ERROR!(core::ptr::addr_of!(_acpi_module_name) as *const u8, 493i32, c"Too many arguments for method [%4.4s]".as_ptr() as *mut u8, acpi_ut_get_node_name(*__slate_slot_25 as *mut ()));
return_ACPI_STATUS__262!(EXCEP_AML!(39i32));
}
}
// Init for new method, possibly wait on method mutex
*__slate_slot_26 = acpi_ds_begin_method_execution(*__slate_slot_25, *__slate_slot_23, this_walk_state);
if ACPI_FAILURE!(*__slate_slot_26) != 0 {
return_ACPI_STATUS__967!(*__slate_slot_26);
}
// Begin method parse/execution. Create a new walk state
next_walk_state = acpi_ds_create_walk_state((*(*__slate_slot_23)).method.owner_id, core::ptr::null_mut::<acpi_parse_object>(), *__slate_slot_23, thread);
if next_walk_state.is_null() {
*__slate_slot_26 = EXCEP_ENV!(4i32);
} else {
// The resolved arguments were put on the previous walk state's operand
// stack. Operands on the previous walk state stack always
// start at index 0. Also, null terminate the list of arguments
*((*this_walk_state).operands.as_mut_ptr() as *mut *mut acpi_operand_object).wrapping_offset(((*this_walk_state).num_operands as i32) as isize) = core::ptr::null_mut::<acpi_operand_object>();
let __slate_auto_bytes_7 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes_2 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
*__slate_slot_9 = core::ptr::null_mut::<alloc_tag>();
let __slate_auto_bytes_3 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
loop {
let __slate_auto_bytes_1 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
core::ptr::addr_of_mut!(*__slate_slot_5) == core::ptr::addr_of_mut!(*__slate_slot_4);
*__slate_slot_7 = arch_local_save_flags();
if true {
break;
}
}
let __slate_auto_bytes_12 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_12.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
core::ptr::addr_of_mut!(*__slate_slot_1) == core::ptr::addr_of_mut!(*__slate_slot_0);
let __v3: i32 = arch_irqs_disabled_flags(*__slate_slot_7);
let __v8: i32 = __v3;
*__slate_slot_19 = _kzalloc_noprof(core::mem::size_of::<acpi_evaluate_info>() as u64, if __v8 != 0 { 2080u32 } else { 3264u32 }, *{
__slate_compound_29 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_29)
});
alloc_tag_restore!();
} else {
let __slate_auto_bytes_6 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
loop {
let __slate_auto_bytes_5 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
core::ptr::addr_of_mut!(*__slate_slot_15) == core::ptr::addr_of_mut!(*__slate_slot_14);
*__slate_slot_17 = arch_local_save_flags();
if true {
break;
}
}
let __slate_auto_bytes_4 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
core::ptr::addr_of_mut!(*__slate_slot_11) == core::ptr::addr_of_mut!(*__slate_slot_10);
let __v13: i32 = arch_irqs_disabled_flags(*__slate_slot_17);
let __v18: i32 = __v13;
*__slate_slot_19 = _kzalloc_noprof(core::mem::size_of::<acpi_evaluate_info>() as u64, if __v18 != 0 { 2080u32 } else { 3264u32 }, *{
__slate_compound_31 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_31)
});
}
let __v20: *mut () = *__slate_slot_19;
// Allocate and initialize the evaluation information block
// TBD: this is somewhat inefficient, should change interface to
// ds_init_aml_walk. For now, keeps this struct off the CPU stack
*__slate_slot_22 = __v20 as *mut acpi_evaluate_info;
if (*__slate_slot_22).is_null() {
*__slate_slot_26 = EXCEP_ENV!(4i32);
} else {
(*(*__slate_slot_22)).parameters = ((*this_walk_state).operands.as_mut_ptr() as *mut *mut acpi_operand_object).wrapping_offset(0 as isize);
*__slate_slot_26 = acpi_ds_init_aml_walk(next_walk_state, core::ptr::null_mut::<acpi_parse_object>(), *__slate_slot_25, (*(*__slate_slot_23)).method.aml_start, (*(*__slate_slot_23)).method.aml_length, *__slate_slot_22, 3u8);
ACPI_FREE!(*__slate_slot_22);
if ACPI_FAILURE!(*__slate_slot_26) == 0 {
(*next_walk_state).method_nesting_depth = ((*this_walk_state).method_nesting_depth as i32).wrapping_add(1) as u16;
// Delete the operands on the previous walkstate operand stack
// (they were copied to new objects)
acpi_ds_clear_operands(this_walk_state);
loop {
if ACPI_IS_DEBUG_ENABLED!(256u32, 64i32) {
acpi_debug_print(256u32, 557u32, c"acpi_ds_call_control_method".as_ptr() as *const u8, core::ptr::addr_of!(_acpi_module_name) as *const u8, 64u32, c"**** Begin nested execution of [%4.4s] **** WalkState=%p\n".as_ptr() as *const u8, (*(*__slate_slot_25)).name.ascii.as_mut_ptr() as *mut u8, next_walk_state);
}
if true {
break;
}
}
(*this_walk_state).method_pathname = acpi_ns_get_normalized_pathname(*__slate_slot_25, true as u8);
(*this_walk_state).method_is_nested = true as u8;
// Optional object evaluation log
loop {
if ACPI_IS_DEBUG_ENABLED!(524288u32, 64i32) {
acpi_debug_print_raw(524288u32, 567u32, c"acpi_ds_call_control_method".as_ptr() as *const u8, core::ptr::addr_of!(_acpi_module_name) as *const u8, 64u32, c"%-26s:  %*s%s\n".as_ptr() as *const u8, c"   Nested method call".as_ptr() as *mut u8, ((*next_walk_state).method_nesting_depth as i32).wrapping_mul(3), c" ".as_ptr() as *mut u8, unsafe { (*this_walk_state).method_pathname }.wrapping_offset(1 as isize));
}
if true {
break;
}
}
// Invoke an internal method if necessary
if ((*(*__slate_slot_23)).method.info_flags as i32) & 2 != 0 {
*__slate_slot_26 = unsafe { (*(*__slate_slot_23)).method.dispatch.implementation }.unwrap()(next_walk_state);
if *__slate_slot_26 == 0 {
*__slate_slot_26 = EXCEP_CTL!(3i32);
}
}
return_ACPI_STATUS__621!(*__slate_slot_26);
}
}
acpi_ds_pop_walk_state(thread); // On error, pop the walk state to be deleted from thread
}
acpi_ds_terminate_control_method(*__slate_slot_23, next_walk_state); // On error, we must terminate the method properly
acpi_ds_delete_walk_state(next_walk_state);
return_ACPI_STATUS__438!(*__slate_slot_26);
}
unsafe { core::mem::zeroed() }
}

// FUNCTION:    acpi_ds_restart_control_method
//
// PARAMETERS:  walk_state          - State for preempted method (caller)
//              return_desc         - Return value from the called method
//
// RETURN:      Status
//
// DESCRIPTION: Restart a method that was preempted by another (nested) method
//              invocation. Handle the return value (if any) from the callee.
#[unsafe(no_mangle)]
extern "C" fn acpi_ds_restart_control_method(walk_state: *mut acpi_walk_state, return_desc: *mut acpi_operand_object) -> u32 {
let mut status: u32 = 0;
let mut same_as_implicit_return: i32 = 0;
ACPI_FUNCTION_TRACE_PTR__7!(walk_state);
loop {
if ACPI_IS_DEBUG_ENABLED!(256u32, 64i32) {
unsafe { acpi_debug_print(256u32, 623u32, c"acpi_ds_restart_control_method".as_ptr() as *const u8, core::ptr::addr_of!(_acpi_module_name) as *const u8, 64u32, c"****Restart [%4.4s] Op %p ReturnValueFromCallee %p\n".as_ptr() as *const u8, acpi_ut_get_node_name((*walk_state).method_node as *mut ()), (*walk_state).method_call_op, return_desc) };
}
if true {
break;
}
}
loop {
if ACPI_IS_DEBUG_ENABLED!(256u32, 64i32) {
unsafe { acpi_debug_print(256u32, 628u32, c"acpi_ds_restart_control_method".as_ptr() as *const u8, core::ptr::addr_of!(_acpi_module_name) as *const u8, 64u32, c"    ReturnFromThisMethodUsed?=%X ResStack %p Walk %p\n".as_ptr() as *const u8, (*walk_state).return_used as i32, (*walk_state).results, walk_state) };
}
if true {
break;
}
}
// Did the called method return a value?
if !return_desc.is_null() {
// Is the implicit return object the same as the return desc?
same_as_implicit_return = ((unsafe { (*walk_state).implicit_return_obj }) == return_desc) as i32;
// Are we actually going to use the return value?
if (unsafe { (*walk_state).return_used }) != 0 {
// Save the return value from the previous method
status = unsafe { acpi_ds_result_push(return_desc, walk_state) };
if ACPI_FAILURE!(status) != 0 {
unsafe { acpi_ut_remove_reference(return_desc) };
return_ACPI_STATUS__75!(status);
}
// Save as THIS method's return value in case it is returned
// immediately to yet another method
unsafe {
(*walk_state).return_desc = return_desc;
}
} else {
if (unsafe { acpi_ds_do_implicit_return(return_desc, walk_state, false as u8) }) == 0 || same_as_implicit_return != 0 {
// Delete the return value if it will not be used by the
// calling method or remove one reference if the explicit return
// is the same as the implicit return value.
unsafe { acpi_ut_remove_reference(return_desc) };
}
}
// The following code is the optional support for the so-called
// "implicit return". Some AML code assumes that the last value of the
// method is "implicitly" returned to the caller, in the absence of an
// explicit return value.
//
// Just save the last result of the method as the return value.
//
// NOTE: this is optional because the ASL language does not actually
// support this behavior.
}
return_ACPI_STATUS__149!(0u32);
unsafe { core::mem::zeroed() }
}

// FUNCTION:    acpi_ds_terminate_control_method
//
// PARAMETERS:  method_desc         - Method object
//              walk_state          - State associated with the method
//
// RETURN:      None
//
// DESCRIPTION: Terminate a control method. Delete everything that the method
//              created, delete all locals and arguments, and delete the parse
//              tree if requested.
//
// MUTEX:       Interpreter is locked
#[unsafe(no_mangle)]
extern "C" fn acpi_ds_terminate_control_method(method_desc: *mut acpi_operand_object, walk_state: *mut acpi_walk_state) {
let mut i: u32 = 0;
ACPI_FUNCTION_TRACE_PTR__11!(walk_state);
// method_desc is required, walk_state is optional
if method_desc.is_null() {
loop {
unsafe { acpi_ut_exit(716u32, c"acpi_ds_terminate_control_method".as_ptr() as *const u8, core::ptr::addr_of!(_acpi_module_name) as *const u8, 64u32) };
return;
}
}
if !walk_state.is_null() {
// Check if the return value is a ref_of reference to a method local
// or argument. If so, clear the reference to avoid use-after-free
// when the walk state is deleted.
if !unsafe { (*walk_state).return_desc }.is_null() && ((unsafe { (*(*walk_state).return_desc).common.r#type }) as i32) == 20 && ((unsafe { (*(*walk_state).return_desc).reference.class }) as i32) == 2 {
let ref_node: *mut acpi_namespace_node = (unsafe { (*(*walk_state).return_desc).reference.object }) as *mut acpi_namespace_node;
if !ref_node.is_null() {
// Check against method locals
i = 0;
while i < 8 {
if ref_node == unsafe { (*walk_state).local_variables.as_mut_ptr() as *mut acpi_namespace_node }.wrapping_offset(i as isize) {
unsafe { acpi_ut_remove_reference((*walk_state).return_desc) };
unsafe {
(*walk_state).return_desc = core::ptr::null_mut::<acpi_operand_object>();
}
break;
}
i = i.wrapping_add(1);
}
// Check against method arguments if not already cleared
if !unsafe { (*walk_state).return_desc }.is_null() {
i = 0;
while i < 7 {
if ref_node == unsafe { (*walk_state).arguments.as_mut_ptr() as *mut acpi_namespace_node }.wrapping_offset(i as isize) {
unsafe { acpi_ut_remove_reference((*walk_state).return_desc) };
unsafe {
(*walk_state).return_desc = core::ptr::null_mut::<acpi_operand_object>();
}
break;
}
i = i.wrapping_add(1);
}
}
}
}
// Delete all arguments and locals
unsafe { acpi_ds_method_data_delete_all(walk_state) };
// Delete any namespace objects created anywhere within the
// namespace by the execution of this method. Unless:
// 1) This method is a module-level executable code method, in which
//    case we want make the objects permanent.
// 2) There are other threads executing the method, in which case we
//    will wait until the last thread has completed.
if ((unsafe { (*method_desc).method.info_flags }) as i32) & 1 == 0 && ((unsafe { (*method_desc).method.thread_count }) as i32) == 1 {
// Delete any direct children of (created by) this method
unsafe { acpi_ex_exit_interpreter() };
unsafe { acpi_ns_delete_namespace_subtree((*walk_state).method_node) };
unsafe { acpi_ex_enter_interpreter() };
// Delete any objects that were created by this method
// elsewhere in the namespace (if any were created).
// Use of the ACPI_METHOD_MODIFIED_NAMESPACE optimizes the
// deletion such that we don't have to perform an entire
// namespace walk for every control method execution.
if ((unsafe { (*method_desc).method.info_flags }) as i32) & 32 != 0 {
unsafe { acpi_ex_exit_interpreter() };
unsafe { acpi_ns_delete_namespace_by_owner((*method_desc).method.owner_id) };
unsafe { acpi_ex_enter_interpreter() };
let __v9: u8 = (((unsafe { (*method_desc).method.info_flags }) as i32) & !32) as u8;
unsafe {
(*method_desc).method.info_flags = __v9;
}
}
}
// If method is serialized, release the mutex and restore the
// current sync level for this thread
if !unsafe { (*method_desc).method.mutex }.is_null() {
// Acquisition Depth handles recursive calls
let __v10: *mut acpi_operand_object = unsafe { (*method_desc).method.mutex };
unsafe {
(*__v10).mutex.acquisition_depth = ((*__v10).mutex.acquisition_depth as i32).wrapping_sub(1) as u16;
}
if (unsafe { (*(*method_desc).method.mutex).mutex.acquisition_depth }) == 0 {
unsafe {
(*(*walk_state).thread).current_sync_level = (*(*method_desc).method.mutex).mutex.original_sync_level;
}
acpi_os_release_mutex!(unsafe { (*(*method_desc).method.mutex).mutex.os_mutex });
unsafe {
(*(*method_desc).method.mutex).mutex.thread_id = 0;
}
}
}
}
// Decrement the thread count on the method
if (unsafe { (*method_desc).method.thread_count }) != 0 {
unsafe {
(*method_desc).method.thread_count = ((*method_desc).method.thread_count as i32).wrapping_sub(1) as u8;
}
} else {
ACPI_ERROR!(unsafe { core::ptr::addr_of!(_acpi_module_name) as *const u8 }, 829i32, c"Invalid zero thread count in method".as_ptr() as *mut u8);
}
// Are there any other threads currently executing this method?
if (unsafe { (*method_desc).method.thread_count }) != 0 {
// Additional threads. Do not release the owner_id in this case,
// we immediately reuse it for the next thread executing this method
loop {
if ACPI_IS_DEBUG_ENABLED!(256u32, 64i32) {
unsafe { acpi_debug_print(256u32, 839u32, c"acpi_ds_terminate_control_method".as_ptr() as *const u8, core::ptr::addr_of!(_acpi_module_name) as *const u8, 64u32, c"*** Completed execution of one thread, %u threads remaining\n".as_ptr() as *const u8, (*method_desc).method.thread_count as i32) };
}
if true {
break;
}
}
} else {
// This is the only executing thread for this method
// Support to dynamically change a method from not_serialized to
// Serialized if it appears that the method is incorrectly written and
// does not support multiple thread execution. The best example of this
// is if such a method creates namespace objects and blocks. A second
// thread will fail with an AE_ALREADY_EXISTS exception.
//
// This code is here because we must wait until the last thread exits
// before marking the method as serialized.
if ((unsafe { (*method_desc).method.info_flags }) as i32) & 8 != 0 {
if !walk_state.is_null() {
ACPI_INFO!(c"Marking method %4.4s as Serialized because of AE_ALREADY_EXISTS error".as_ptr() as *mut u8, unsafe { (*(*walk_state).method_node).name.ascii.as_mut_ptr() as *mut u8 });
}
// Method tried to create an object twice and was marked as
// "pending serialized". The probable cause is that the method
// cannot handle reentrancy.
//
// The method was created as not_serialized, but it tried to create
// a named object and then blocked, causing the second thread
// entrance to begin and then fail. Workaround this problem by
// marking the method permanently as Serialized when the last
// thread exits here.
let __v19: u8 = (((unsafe { (*method_desc).method.info_flags }) as i32) & !8) as u8;
unsafe {
(*method_desc).method.info_flags = __v19;
}
unsafe {
(*method_desc).method.info_flags = (((*method_desc).method.info_flags as i32) | (4 | 16)) as u8;
}
unsafe {
(*method_desc).method.sync_level = 0;
}
}
// No more threads, we can free the owner_id
if ((unsafe { (*method_desc).method.info_flags }) as i32) & 1 == 0 {
unsafe { acpi_ut_release_owner_id(core::ptr::addr_of_mut!((*method_desc).method.owner_id)) };
}
}
unsafe { acpi_ex_stop_trace_method((*method_desc).method.node as *mut acpi_namespace_node, method_desc, walk_state) };
loop {
unsafe { acpi_ut_exit(896u32, c"acpi_ds_terminate_control_method".as_ptr() as *const u8, core::ptr::addr_of!(_acpi_module_name) as *const u8, 64u32) };
return;
}
}

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

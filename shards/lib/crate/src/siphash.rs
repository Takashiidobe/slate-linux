
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
//! SPDX-License-Identifier: (GPL-2.0-only OR BSD-3-Clause)
//!
//! Copyright (C) 2016-2022 Jason A. Donenfeld <Jason@zx2c4.com>. All Rights Reserved.
//!
//! SipHash: a fast short-input PRF
//! https://131002.net/siphash/
//!
//! This implementation is specifically for SipHash2-4 for a secure PRF
//! and HalfSipHash1-3/SipHash1-3 for an insecure PRF only suitable for
//! hashtables.
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
//! @a is a power of 2 value
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
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
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
//! Check at compile time that something is of a particular type.
//! Always evaluates to 1 so you may use it easily in comparisons.
//! See linux/instrumentation.h
//! 64 bit
//! unused, was: #define EX_TYPE_COPY	 4
//! Begin/end of an instrumentation safe region
//! __ADDRESSABLE() is used to ensure the key symbol doesn't get stripped from
//! the symbol table so that objtool can reference it when it generates the
//! .static_call_sites section.
//! These are for everybody (although not all archs will actually
//! discard it in modules)
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
//! This counts to 15. Any more, it will return 16th argument.
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
//! An exchange-type operation, which takes a value and a pointer, and
//! returns the old value.
//! These are non-NULL pointers that will result in page faults
//! under normal circumstances, used to verify that nobody uses
//! non-initialized list entries.
//! list_entry - get the struct for this entry
//! @ptr:	the &struct list_head pointer.
//! @type:	the type of the struct this is embedded in.
//! @member:	the name of the list_head within the struct.
//! Try again
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
//! Bit size and mask of CPU number stored in the per CPU data (and TSC_AUX)
//! PAGE_SHIFT determines the page size.
//!
//! Note: This definition is required because PAGE_SHIFT is used
//! in several places throughout the codebase.
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
//! restart by calling sys_restart_syscall
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
//! RCU_INITIALIZER() - statically initialize an RCU-protected global variable
//! @v: The value to statically initialize with.
//! return the ->next pointer of a list_head in an rcu safe
//! way, we must not access it directly
//! return the first ptr or next element in an RCU protected list
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
//! wait_queue_entry::flags
//! 192 bytes
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
// This should be used to refer to an instruction that is considered
// terminating, like a noreturn CALL or UD2 when we know they are not -- eg
// WARN using UD2.
// Annotate a special section entry.  This emables livepatch module generation
// to find and extract individual special section entries as needed.
// longword load with zeropad on fault
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
// Concatenate two parameters, but allow them to be expanded beforehand.
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
// "sse"
// "rep_good" REP microcode works well
// "popcnt" POPCNT instruction
// "3dnowprefetch" 3DNow prefetch instructions
// "ibpb" Indirect Branch Prediction Barrier without a guaranteed RSB flush
// "erms" Enhanced REP MOVSB/STOSB instructions
// "clflushopt" CLFLUSHOPT instruction
// "clwb" CLWB instruction
// IA32_TSC_DEADLINE and X2APIC MSRs need fencing
// "wbnoinvd" WBNOINVD instruction
// "la57" 5-level page tables
// "rdpid" RDPID instruction
// BUG word(s)
// "div0" AMD DIV0 speculation bug
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
// __put_unaligned_t - write an unaligned value to memory.
// @type:	the type of the value to store.
// @val:	the value to store.
// @ptr:	the pointer to store to.
//
// Use memcpy to affect an unaligned type sized store avoiding undefined
// behavior from approaches like type punning that require -fno-strict-aliasing
// in order to be correct. The void* cast silences ubsan warnings.
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
// Out of memory
// Device or resource busy
// Invalid argument
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
// On 2MB or 1GB pages
// Extracts the PFN from a (pte|pmd|pud|pgd)val_t of a 4KB page
// Extracts the flags from a (pte|pmd|pud|pgd)val_t
// This includes the protection key value.
// No need to mask any bits for !PAE
// Operation is not supported
// cpumask_bits - get the bits in a cpumask
// @maskp: the struct cpumask *
//
// You should only assume nr_cpu_ids bits of this mask are valid.  This is
// a macro so it's const-correct.
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
// return the first or the next element in an RCU protected hlist
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
macro_rules! SIPHASH_PERMUTATION {
($a:expr, $b:expr, $c:expr, $d:expr) => {
let __v: u64 = $a;
let __v2: u64 = __v.wrapping_add($b);
$a = __v2;
let __v3: u64 = rol64($b, 13u32);
$b = __v3;
let __v4: u64 = $b;
let __v5: u64 = __v4 ^ $a;
$b = __v5;
let __v6: u64 = rol64($a, 32u32);
$a = __v6;
let __v7: u64 = $c;
let __v8: u64 = __v7.wrapping_add($d);
$c = __v8;
let __v9: u64 = rol64($d, 16u32);
$d = __v9;
let __v10: u64 = $d;
let __v11: u64 = __v10 ^ $c;
$d = __v11;
let __v12: u64 = $a;
let __v13: u64 = __v12.wrapping_add($d);
$a = __v13;
let __v14: u64 = rol64($d, 21u32);
$d = __v14;
let __v15: u64 = $d;
let __v16: u64 = __v15 ^ $a;
$d = __v16;
let __v17: u64 = $c;
let __v18: u64 = __v17.wrapping_add($b);
$c = __v18;
let __v19: u64 = rol64($b, 17u32);
$b = __v19;
let __v20: u64 = $b;
let __v21: u64 = __v20 ^ $c;
$b = __v21;
let __v22: u64 = rol64($c, 32u32);
$c = __v22;
};
}

/// __get_unaligned_t - read an unaligned value from memory.
/// @type:	the type to load from the pointer.
/// @ptr:	the pointer to load from.
///
/// Use memcpy to affect an unaligned type sized load avoiding undefined behavior
/// from approaches like type punning that require -fno-strict-aliasing in order
/// to be correct. As type may be const, use __unqual_scalar_typeof to map to a
/// non-const type - you can't memcpy into a const type. The
/// __get_unaligned_ctrl_type gives __unqual_scalar_typeof its required
/// expression rather than type, a pointer is used to avoid warnings about mixing
/// the use of 0 and NULL. The void* cast silences ubsan warnings.
macro_rules! __get_unaligned_t__4 {
($type:ty, $ptr:expr) => {
{
let mut __v: $type;
core::ptr::null_mut::<u64>();
let mut __v3: $type = 0;
unsafe { __builtin_memcpy(core::ptr::addr_of_mut!(__v3) as *mut (), (($ptr) as *mut ()) as *const (), core::mem::size_of::<$type>() as u64) };
__v = __v3;
__v
}
};
}

macro_rules! bytemask_from_count {
($cnt:expr) => {
!(18446744073709551615u64 << (($cnt) as i32).wrapping_mul(8))
};
}

unsafe extern "C" {
    #[link_name = "memcpy"]
    fn __builtin_memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
}

// The following macros are non-atomic versions of their non-underscored
// counterparts.
// Check that the bitops prototypes are sane
fn rol64(word: u64, shift: u32) -> u64 {
word << (shift & 63) | word >> (shift.wrapping_neg() & 63)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siphash_h_21 {
    key: [u64; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_siphash_h_95 {
    key: [u64; 2],
}


fn get_unaligned_le64(p: *const ()) -> u64 {
__get_unaligned_t__4!(u64, p)
}

fn load_unaligned_zeropad(mut addr: *const ()) -> u64 {
let mut ret: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("1:\tmov ({1:r}), {0:r}\n2:\n .pushsection __ex_table, \"aM\", @progbits, 12\n .balign 4\n .long (1b) - .\n .long (2b) - .\n .long 20 \n .popsection\n", lateout(reg) __t0, in(reg) addr as *mut u64, options(nostack, att_syntax));
ret = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
ret
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.siphash.__siphash_unaligned")]
extern "C" fn __siphash_unaligned(mut data: *const (), len: u64, key: *const __SlateRecord_siphash_h_21) -> u64 {
let end: *const u8 = ((unsafe { (((data as *const u8).wrapping_offset(len as isize) as *const ()) as *const u8).wrapping_offset(-((len % (core::mem::size_of::<u64>() as u64)) as isize)) }) as *const ()) as *const u8;
let left: u8 = (len & (core::mem::size_of::<u64>() as u64).wrapping_sub(1)) as u8;
let mut m: u64 = 0;
let mut v0: u64 = 8317987319222330741;
let mut v1: u64 = 7237128888997146477;
let mut v2: u64 = 7816392313619706465;
let mut v3: u64 = 8387220255154660723;
let mut b: u64 = (len as u64) << 56i32;
let __v0: u64 = v3;
v3 = __v0 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(1 as isize) };
let __v2: u64 = v2;
v2 = __v2 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(0 as isize) };
let __v4: u64 = v1;
v1 = __v4 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(1 as isize) };
let __v6: u64 = v0;
v0 = __v6 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(0 as isize) };
while data != (end as *const ()) {
m = get_unaligned_le64(data);
v3 ^= m;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
v0 ^= m;
data = (unsafe { (data as *const u8).wrapping_offset(core::mem::size_of::<u64>() as isize) }) as *const ();
}
if left != 0 {
let __v59: u64 = load_unaligned_zeropad(data) & bytemask_from_count!(left);
b |= __v59;
}
v3 ^= b;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
v0 ^= b;
v2 ^= 255;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
v0 ^ v1 ^ (v2 ^ v3)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___siphash_unaligned_194: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const (), u64, *const __SlateRecord_siphash_h_21) -> u64>, *mut ()>(Some(__siphash_unaligned)) };

/// siphash_1u64 - compute 64-bit siphash PRF value of a u64
/// @first: first u64
/// @key: the siphash key
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.siphash.siphash_1u64")]
extern "C" fn siphash_1u64(first: u64, key: *const __SlateRecord_siphash_h_21) -> u64 {
let mut v0: u64 = 8317987319222330741;
let mut v1: u64 = 7237128888997146477;
let mut v2: u64 = 7816392313619706465;
let mut v3: u64 = 8387220255154660723;
let b: u64 = 576460752303423488u64;
let __v0: u64 = v3;
v3 = __v0 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(1 as isize) };
let __v2: u64 = v2;
v2 = __v2 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(0 as isize) };
let __v4: u64 = v1;
v1 = __v4 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(1 as isize) };
let __v6: u64 = v0;
v0 = __v6 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(0 as isize) };
v3 ^= first;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
v0 ^= first;
v3 ^= b;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
v0 ^= b;
v2 ^= 255;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
v0 ^ v1 ^ (v2 ^ v3)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_siphash_1u64_195: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u64, *const __SlateRecord_siphash_h_21) -> u64>, *mut ()>(Some(siphash_1u64)) };

/// siphash_2u64 - compute 64-bit siphash PRF value of 2 u64
/// @first: first u64
/// @second: second u64
/// @key: the siphash key
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.siphash.siphash_2u64")]
extern "C" fn siphash_2u64(first: u64, second: u64, key: *const __SlateRecord_siphash_h_21) -> u64 {
let mut v0: u64 = 8317987319222330741;
let mut v1: u64 = 7237128888997146477;
let mut v2: u64 = 7816392313619706465;
let mut v3: u64 = 8387220255154660723;
let b: u64 = 1152921504606846976u64;
let __v0: u64 = v3;
v3 = __v0 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(1 as isize) };
let __v2: u64 = v2;
v2 = __v2 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(0 as isize) };
let __v4: u64 = v1;
v1 = __v4 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(1 as isize) };
let __v6: u64 = v0;
v0 = __v6 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(0 as isize) };
v3 ^= first;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
v0 ^= first;
v3 ^= second;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
v0 ^= second;
v3 ^= b;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
v0 ^= b;
v2 ^= 255;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
v0 ^ v1 ^ (v2 ^ v3)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_siphash_2u64_196: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u64, u64, *const __SlateRecord_siphash_h_21) -> u64>, *mut ()>(Some(siphash_2u64)) };

/// siphash_3u64 - compute 64-bit siphash PRF value of 3 u64
/// @first: first u64
/// @second: second u64
/// @third: third u64
/// @key: the siphash key
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.siphash.siphash_3u64")]
extern "C" fn siphash_3u64(first: u64, second: u64, third: u64, key: *const __SlateRecord_siphash_h_21) -> u64 {
let mut v0: u64 = 8317987319222330741;
let mut v1: u64 = 7237128888997146477;
let mut v2: u64 = 7816392313619706465;
let mut v3: u64 = 8387220255154660723;
let b: u64 = 1729382256910270464u64;
let __v0: u64 = v3;
v3 = __v0 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(1 as isize) };
let __v2: u64 = v2;
v2 = __v2 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(0 as isize) };
let __v4: u64 = v1;
v1 = __v4 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(1 as isize) };
let __v6: u64 = v0;
v0 = __v6 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(0 as isize) };
v3 ^= first;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
v0 ^= first;
v3 ^= second;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
v0 ^= second;
v3 ^= third;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
v0 ^= third;
v3 ^= b;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
v0 ^= b;
v2 ^= 255;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
v0 ^ v1 ^ (v2 ^ v3)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_siphash_3u64_197: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u64, u64, u64, *const __SlateRecord_siphash_h_21) -> u64>, *mut ()>(Some(siphash_3u64)) };

/// siphash_4u64 - compute 64-bit siphash PRF value of 4 u64
/// @first: first u64
/// @second: second u64
/// @third: third u64
/// @forth: forth u64
/// @key: the siphash key
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.siphash.siphash_4u64")]
extern "C" fn siphash_4u64(first: u64, second: u64, third: u64, forth: u64, key: *const __SlateRecord_siphash_h_21) -> u64 {
let mut v0: u64 = 8317987319222330741;
let mut v1: u64 = 7237128888997146477;
let mut v2: u64 = 7816392313619706465;
let mut v3: u64 = 8387220255154660723;
let b: u64 = 2305843009213693952u64;
let __v0: u64 = v3;
v3 = __v0 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(1 as isize) };
let __v2: u64 = v2;
v2 = __v2 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(0 as isize) };
let __v4: u64 = v1;
v1 = __v4 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(1 as isize) };
let __v6: u64 = v0;
v0 = __v6 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(0 as isize) };
v3 ^= first;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
v0 ^= first;
v3 ^= second;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
v0 ^= second;
v3 ^= third;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
v0 ^= third;
v3 ^= forth;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
v0 ^= forth;
v3 ^= b;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
v0 ^= b;
v2 ^= 255;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
v0 ^ v1 ^ (v2 ^ v3)
}

core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol___siphash_unaligned: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_siphash_1u64: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_siphash_2u64: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_siphash_3u64: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_siphash_4u64: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_siphash_1u32: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_siphash_3u32: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol___hsiphash_unaligned: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {7} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_hsiphash_1u32: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {8} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_hsiphash_2u32: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {9} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_hsiphash_3u32: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {10} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_hsiphash_4u32: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {11} ; .previous", sym __siphash_unaligned, sym siphash_1u64, sym siphash_2u64, sym siphash_3u64, sym siphash_4u64, sym siphash_1u32, sym siphash_3u32, sym __hsiphash_unaligned, sym hsiphash_1u32, sym hsiphash_2u32, sym hsiphash_3u32, sym hsiphash_4u32, options(att_syntax));

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_siphash_4u64_198: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u64, u64, u64, u64, *const __SlateRecord_siphash_h_21) -> u64>, *mut ()>(Some(siphash_4u64)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.siphash.siphash_1u32")]
extern "C" fn siphash_1u32(first: u32, key: *const __SlateRecord_siphash_h_21) -> u64 {
let mut v0: u64 = 8317987319222330741;
let mut v1: u64 = 7237128888997146477;
let mut v2: u64 = 7816392313619706465;
let mut v3: u64 = 8387220255154660723;
let mut b: u64 = 288230376151711744u64;
let __v0: u64 = v3;
v3 = __v0 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(1 as isize) };
let __v2: u64 = v2;
v2 = __v2 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(0 as isize) };
let __v4: u64 = v1;
v1 = __v4 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(1 as isize) };
let __v6: u64 = v0;
v0 = __v6 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(0 as isize) };
b |= first as u64;
v3 ^= b;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
v0 ^= b;
v2 ^= 255;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
v0 ^ v1 ^ (v2 ^ v3)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_siphash_1u32_199: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u32, *const __SlateRecord_siphash_h_21) -> u64>, *mut ()>(Some(siphash_1u32)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.siphash.siphash_3u32")]
extern "C" fn siphash_3u32(first: u32, second: u32, third: u32, key: *const __SlateRecord_siphash_h_21) -> u64 {
let combined: u64 = (second as u64) << 32i32 | (first as u64);
let mut v0: u64 = 8317987319222330741;
let mut v1: u64 = 7237128888997146477;
let mut v2: u64 = 7816392313619706465;
let mut v3: u64 = 8387220255154660723;
let mut b: u64 = 864691128455135232u64;
let __v0: u64 = v3;
v3 = __v0 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(1 as isize) };
let __v2: u64 = v2;
v2 = __v2 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(0 as isize) };
let __v4: u64 = v1;
v1 = __v4 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(1 as isize) };
let __v6: u64 = v0;
v0 = __v6 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(0 as isize) };
v3 ^= combined;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
v0 ^= combined;
b |= third as u64;
v3 ^= b;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
v0 ^= b;
v2 ^= 255;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
v0 ^ v1 ^ (v2 ^ v3)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_siphash_3u32_200: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u32, u32, u32, *const __SlateRecord_siphash_h_21) -> u64>, *mut ()>(Some(siphash_3u32)) };

// Note that on 64-bit, we make HalfSipHash1-3 actually be SipHash1-3, for
// performance reasons. On 32-bit, below, we actually implement HalfSipHash1-3.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.siphash.__hsiphash_unaligned")]
extern "C" fn __hsiphash_unaligned(mut data: *const (), len: u64, key: *const __SlateRecord_siphash_h_95) -> u32 {
let end: *const u8 = ((unsafe { (((data as *const u8).wrapping_offset(len as isize) as *const ()) as *const u8).wrapping_offset(-((len % (core::mem::size_of::<u64>() as u64)) as isize)) }) as *const ()) as *const u8;
let left: u8 = (len & (core::mem::size_of::<u64>() as u64).wrapping_sub(1)) as u8;
let mut m: u64 = 0;
let mut v0: u64 = 8317987319222330741;
let mut v1: u64 = 7237128888997146477;
let mut v2: u64 = 7816392313619706465;
let mut v3: u64 = 8387220255154660723;
let mut b: u64 = (len as u64) << 56i32;
let __v0: u64 = v3;
v3 = __v0 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(1 as isize) };
let __v2: u64 = v2;
v2 = __v2 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(0 as isize) };
let __v4: u64 = v1;
v1 = __v4 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(1 as isize) };
let __v6: u64 = v0;
v0 = __v6 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(0 as isize) };
while data != (end as *const ()) {
m = get_unaligned_le64(data);
v3 ^= m;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
v0 ^= m;
data = (unsafe { (data as *const u8).wrapping_offset(core::mem::size_of::<u64>() as isize) }) as *const ();
}
if left != 0 {
let __v37: u64 = load_unaligned_zeropad(data) & bytemask_from_count!(left);
b |= __v37;
}
v3 ^= b;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
v0 ^= b;
v2 ^= 255;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
(v0 ^ v1 ^ (v2 ^ v3)) as u32
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___hsiphash_unaligned_201: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const (), u64, *const __SlateRecord_siphash_h_95) -> u32>, *mut ()>(Some(__hsiphash_unaligned)) };

/// hsiphash_1u32 - compute 64-bit hsiphash PRF value of a u32
/// @first: first u32
/// @key: the hsiphash key
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.siphash.hsiphash_1u32")]
extern "C" fn hsiphash_1u32(first: u32, key: *const __SlateRecord_siphash_h_95) -> u32 {
let mut v0: u64 = 8317987319222330741;
let mut v1: u64 = 7237128888997146477;
let mut v2: u64 = 7816392313619706465;
let mut v3: u64 = 8387220255154660723;
let mut b: u64 = 288230376151711744u64;
let __v0: u64 = v3;
v3 = __v0 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(1 as isize) };
let __v2: u64 = v2;
v2 = __v2 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(0 as isize) };
let __v4: u64 = v1;
v1 = __v4 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(1 as isize) };
let __v6: u64 = v0;
v0 = __v6 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(0 as isize) };
b |= first as u64;
v3 ^= b;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
v0 ^= b;
v2 ^= 255;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
(v0 ^ v1 ^ (v2 ^ v3)) as u32
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_hsiphash_1u32_202: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u32, *const __SlateRecord_siphash_h_95) -> u32>, *mut ()>(Some(hsiphash_1u32)) };

/// hsiphash_2u32 - compute 32-bit hsiphash PRF value of 2 u32
/// @first: first u32
/// @second: second u32
/// @key: the hsiphash key
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.siphash.hsiphash_2u32")]
extern "C" fn hsiphash_2u32(first: u32, second: u32, key: *const __SlateRecord_siphash_h_95) -> u32 {
let combined: u64 = (second as u64) << 32i32 | (first as u64);
let mut v0: u64 = 8317987319222330741;
let mut v1: u64 = 7237128888997146477;
let mut v2: u64 = 7816392313619706465;
let mut v3: u64 = 8387220255154660723;
let b: u64 = 576460752303423488u64;
let __v0: u64 = v3;
v3 = __v0 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(1 as isize) };
let __v2: u64 = v2;
v2 = __v2 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(0 as isize) };
let __v4: u64 = v1;
v1 = __v4 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(1 as isize) };
let __v6: u64 = v0;
v0 = __v6 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(0 as isize) };
v3 ^= combined;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
v0 ^= combined;
v3 ^= b;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
v0 ^= b;
v2 ^= 255;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
(v0 ^ v1 ^ (v2 ^ v3)) as u32
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_hsiphash_2u32_203: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u32, u32, *const __SlateRecord_siphash_h_95) -> u32>, *mut ()>(Some(hsiphash_2u32)) };

/// hsiphash_3u32 - compute 32-bit hsiphash PRF value of 3 u32
/// @first: first u32
/// @second: second u32
/// @third: third u32
/// @key: the hsiphash key
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.siphash.hsiphash_3u32")]
extern "C" fn hsiphash_3u32(first: u32, second: u32, third: u32, key: *const __SlateRecord_siphash_h_95) -> u32 {
let combined: u64 = (second as u64) << 32i32 | (first as u64);
let mut v0: u64 = 8317987319222330741;
let mut v1: u64 = 7237128888997146477;
let mut v2: u64 = 7816392313619706465;
let mut v3: u64 = 8387220255154660723;
let mut b: u64 = 864691128455135232u64;
let __v0: u64 = v3;
v3 = __v0 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(1 as isize) };
let __v2: u64 = v2;
v2 = __v2 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(0 as isize) };
let __v4: u64 = v1;
v1 = __v4 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(1 as isize) };
let __v6: u64 = v0;
v0 = __v6 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(0 as isize) };
v3 ^= combined;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
v0 ^= combined;
b |= third as u64;
v3 ^= b;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
v0 ^= b;
v2 ^= 255;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
(v0 ^ v1 ^ (v2 ^ v3)) as u32
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_hsiphash_3u32_204: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u32, u32, u32, *const __SlateRecord_siphash_h_95) -> u32>, *mut ()>(Some(hsiphash_3u32)) };

/// hsiphash_4u32 - compute 32-bit hsiphash PRF value of 4 u32
/// @first: first u32
/// @second: second u32
/// @third: third u32
/// @forth: forth u32
/// @key: the hsiphash key
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.siphash.hsiphash_4u32")]
extern "C" fn hsiphash_4u32(first: u32, second: u32, third: u32, forth: u32, key: *const __SlateRecord_siphash_h_95) -> u32 {
let mut combined: u64 = (second as u64) << 32i32 | (first as u64);
let mut v0: u64 = 8317987319222330741;
let mut v1: u64 = 7237128888997146477;
let mut v2: u64 = 7816392313619706465;
let mut v3: u64 = 8387220255154660723;
let b: u64 = 1152921504606846976u64;
let __v0: u64 = v3;
v3 = __v0 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(1 as isize) };
let __v2: u64 = v2;
v2 = __v2 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(0 as isize) };
let __v4: u64 = v1;
v1 = __v4 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(1 as isize) };
let __v6: u64 = v0;
v0 = __v6 ^ unsafe { *((*key).key.as_ptr() as *const u64).wrapping_offset(0 as isize) };
v3 ^= combined;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
v0 ^= combined;
combined = (forth as u64) << 32i32 | (third as u64);
v3 ^= combined;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
v0 ^= combined;
v3 ^= b;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
v0 ^= b;
v2 ^= 255;
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
SIPHASH_PERMUTATION!(v0, v1, v2, v3);
(v0 ^ v1 ^ (v2 ^ v3)) as u32
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_hsiphash_4u32_205: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u32, u32, u32, u32, *const __SlateRecord_siphash_h_95) -> u32>, *mut ()>(Some(hsiphash_4u32)) };

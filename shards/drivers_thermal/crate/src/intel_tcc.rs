
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! __context_unsafe() - function attribute, disable context checking
//! @comment: comment explaining why opt-out is safe
//!
//! Function attribute denoting that context analysis is disabled for the
//! whole function. Forces adding an inline comment as argument.
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! SPDX-License-Identifier: GPL-2.0-only
//!
//! intel_tcc.c - Library for Intel TCC (thermal control circuitry) MSR access
//! Copyright (c) 2022, Intel Corporation.
//! Operation not permitted
//! No data available
//! restart by calling sys_restart_syscall
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
//! Auburndale / Havendale
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
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
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
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
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
//! An exchange-type operation, which takes a value and a pointer, and
//! returns the old value.
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
//! These are non-NULL pointers that will result in page faults
//! under normal circumstances, used to verify that nobody uses
//! non-initialized list entries.
//! list_for_each	-	iterate over a list
//! @pos:	the &struct list_head to use as a loop cursor.
//! @head:	the head for your list.
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
//! X86_MATCH_CPU -  Base macro for CPU matching
//! @_vendor:	The vendor name, e.g. INTEL, AMD, HYGON, ..., ANY
//! 		The name is expanded to X86_VENDOR_@_vendor
//! @_family:	The family number or X86_FAMILY_ANY
//! @_model:	The model number, model constant or X86_MODEL_ANY
//! @_steppings:	Bitmask for steppings, stepping constant or X86_STEPPING_ANY
//! @_feature:	A X86_FEATURE bit or X86_FEATURE_ANY
//! @_data:	Driver specific data or NULL. The internal storage
//! 		format is unsigned long. The supplied value, pointer
//! 		etc. is casted to unsigned long internally.
//!
//! Use only if you need all selectors. Otherwise use one of the shorter
//! macros of the X86_MATCH_* family. If there is no matching shorthand
//! macro, consider to add one. If you really need to wrap one of the macros
//! into another macro at the usage site for good reasons, then please
//! start this local macro with X86_MATCH to allow easy grepping.
//! We use the MSB for PREEMPT_NEED_RESCHED mostly because it is available.
//! Macros to retrieve the current execution context:
//!
//! in_nmi()		- We're in NMI context
//! in_hardirq()		- We're in hard IRQ context
//! in_serving_softirq()	- We're in softirq context
//! in_task()		- We're in task context
//! rescheduling necessary
//! These constants are defined in the MSan LLVM instrumentation pass.
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
//! get_task_state():
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
// Try again
// Out of memory
// Device or resource busy
// No such device
// Invalid argument
// Value too large for defined data type
// Operation is not supported
// Sky Lake
// Sky Lake
// Sky Lake
// Sky Lake
// Sky Lake
// Sky Lake
// Sky Lake
// Palm Cove
// Sunny Cove
// Sunny Cove
// Sunny Cove
// Sunny Cove
// Sunny Cove
// Cypress Cove
// Willow Cove
// Willow Cove
// Golden Cove
// Raptor Cove
// Sunny Cove / Tremont
// Golden Cove / Gracemont
// Golden Cove / Gracemont
// Raptor Cove / Enhanced Gracemont
// Diamondville, Pineview
// Silverthorne, Lincroft
// Cedarview
// Penwell
// Bay Trail, Valleyview
// Avaton, Rangely
// Merriefield
// Anniedale
// Cherry Trail, Braswell
// Lightning Mountain
// Apollo Lake
// Denverton
// Note: the micro-architecture is "Goldmont Plus"
// Gemini Lake
// Jacobsville
// Elkhart Lake
// Jasper Lake
// Alderlake N
// Knights Landing
// Knights Mill
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
// "sse"
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
// min - return minimum of two values of the same or compatible types
// @x: first value
// @y: second value
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
// X86_MATCH_VFM - Match encoded vendor/family/model
// @vfm:	Encoded 8-bits each for vendor, family, model
// @data:	Driver specific data or NULL. The internal storage
// 		format is unsigned long. The supplied value, pointer
// 		etc. is cast to unsigned long internally.
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
macro_rules! BIT {
($nr:expr) => {
UL!() << $nr
};
}

macro_rules! EAX_EDX_VAL {
($low:expr, $high:expr) => {
($low) | ($high) << 32i32
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

macro_rules! _UL {
() => {
_AC!()
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

unsafe extern "C" {
    static mut __tracepoint_read_msr: tracepoint;
    static mut __tracepoint_write_msr: tracepoint;
    fn ____wrong_branch_error() -> bool;
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn x86_match_cpu(arg0: *const x86_cpu_id) -> *const x86_cpu_id;
    fn do_trace_write_msr(arg0: u32, arg1: u64, arg2: i32);
    fn do_trace_read_msr(arg0: u32, arg1: u64, arg2: i32);
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn rdmsrq_safe_on_cpu(arg0: u32, arg1: u32, arg2: *mut u64) -> i32;
    fn wrmsrq_safe_on_cpu(arg0: u32, arg1: u32, arg2: u64) -> i32;
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_types_h_190 {
    counter: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct x86_cpu_id {
    vendor: u16,
    family: u16,
    model: u16,
    steppings: u16,
    feature: u16,
    flags: u16,
    platform_mask: u8,
    r#type: u8,
    driver_data: u64,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct static_key {
    enabled: __SlateRecord_types_h_190,
    __slate_anon_1: __SlateRecord_jump_label_h_95,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_jump_label_h_95 {
    r#type: u64,
    entries: *mut jump_entry,
    next: *mut static_key_mod,
}


#[inline(always)]
fn arch_static_branch__const1(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_read_msr, const 0 as u8, label {
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
struct static_key_mod {
}


#[inline(always)]
fn arch_static_branch__const2(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_write_msr, const 0 as u8, label {
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
union __SlateRecord_msr_h_6 {
    __slate_anon_0: __SlateRecord_msr_h_7,
    q: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_msr_h_7 {
    l: u32,
    h: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct static_key_false {
    key: static_key,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msr {
    __slate_anon_0: __SlateRecord_msr_h_6,
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
    __slate_bits_0: __slate_bits::__SlateBits146U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_53 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_53_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_53_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_54 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_54_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_54_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_55 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_55_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_55_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_56 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_56_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_56_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_57 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_57_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_57_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_58 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_58_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_58_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_59 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_59_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_59_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_60 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_60_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_60_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_61 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_61_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_61_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_62 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_62_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_62_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_63 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_63_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_63_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_64 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_64_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_64_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_65 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_65_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_65_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_66 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_66_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_66_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_67 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_67_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_67_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_68 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_68_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_68_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_69 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_69_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_69_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_70 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_70_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_70_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_71 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_71_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_71_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_72 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_72_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_72_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_73 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_73_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_73_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_74 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_74_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_74_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_75 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_75_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_75_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_76 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_76_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_76_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_77 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_77_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_77_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_78 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_78_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_78_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_79 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_79_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_79_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_80 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_80_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_80_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_81 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_81_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_81_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_82 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_82_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_82_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_83 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_83_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_83_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_84 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_84_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_84_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_85 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_85_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_85_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_86 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_86_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_86_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_87 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_87_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_87_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_88 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_88_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_88_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_89 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_89_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_89_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_90 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_90_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_90_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_91 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_91_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_91_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_92 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_92_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_92_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_93 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_93_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_93_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_94 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_94_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_94_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_95 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_95_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_95_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_96 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_96_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_96_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_97 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_97_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_97_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_98 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_98_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_98_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_99 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_99_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_99_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_100 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_100_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_100_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_101 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_101_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_101_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_102 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_102_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_102_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_103 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_103_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_103_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_104 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_104_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_104_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_105 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_105_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_105_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_106 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_106_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_106_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_107 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_107_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_107_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_108 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_108_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_108_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_109 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_109_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_109_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_110 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_110_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_110_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_111 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_111_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_111_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_112 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_112_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_112_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_113 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_113_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_113_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_114 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_114_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_114_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_115 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_115_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_115_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_116 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_116_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_116_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_117 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_117_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_117_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_118 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_118_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_118_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_119 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_119_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_intel_tcc_c_119_3 {
}


type __SlateImage_intel_tcc_cpu_ids = __SlateAlign16<[x86_cpu_id; 68]>;

#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

mod __slate_bits {
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits146U0 {
        #[bits(1)]
        pub faultable: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
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
fn native_read_msr_safe(mut msr: u32, p: *mut u64) -> i32 {
let mut err: i32 = 0;
let mut low: u64 = 0;
let mut high: u64 = 0;
unsafe {
let __t0: u32;
let __t1: u64;
let __t2: u64;
core::arch::asm!("1: rdmsr ; xor {0:e},{0:e}\n2:\n\t .pushsection __ex_table, \"aM\", @progbits, 12\n .balign 4\n .long (1b) - .\n .long (2b) - .\n.macro extable_type_reg type:req reg:req\n.set .Lfound, 0\n.set .Lregnr, 0\n.irp rs,rax,rcx,rdx,rbx,rsp,rbp,rsi,rdi,r8,r9,r10,r11,r12,r13,r14,r15\n.ifc \\reg, %\\rs\n.set .Lfound, .Lfound+1\n.long \\type + (.Lregnr << 8)\n.endif\n.set .Lregnr, .Lregnr+1\n.endr\n.set .Lregnr, 0\n.irp rs,eax,ecx,edx,ebx,esp,ebp,esi,edi,r8d,r9d,r10d,r11d,r12d,r13d,r14d,r15d\n.ifc \\reg, %\\rs\n.set .Lfound, .Lfound+1\n.long \\type + (.Lregnr << 8)\n.endif\n.set .Lregnr, .Lregnr+1\n.endr\n.if (.Lfound != 1)\n.error \"extable_type_reg: bad register argument\"\n.endif\n.endm\nextable_type_reg reg={0:e}, type=11 \n.purgem extable_type_reg\n .popsection\n", lateout(reg) __t0, lateout("ax") __t1, lateout("dx") __t2, in("cx") msr as u32, options(nostack, att_syntax));
err = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
low = unsafe { core::mem::transmute::<u64, u64>(__t1 as u64) };
high = unsafe { core::mem::transmute::<u64, u64>(__t2 as u64) };
}
if static_branch_unlikely!(unsafe { core::ptr::addr_of_mut!(__tracepoint_read_msr.key) }) != 0 {
unsafe { do_trace_read_msr(msr, EAX_EDX_VAL!(low, high), err) };
}
unsafe {
*p = EAX_EDX_VAL!(low, high);
}
err
}

fn native_write_msr_safe(mut msr: u32, mut val: u64) -> i32 {
let mut err: i32 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("1: wrmsr ; xor %eax,%eax\n2:\n\t .pushsection __ex_table, \"aM\", @progbits, 12\n .balign 4\n .long (1b) - .\n .long (2b) - .\n.macro extable_type_reg type:req reg:req\n.set .Lfound, 0\n.set .Lregnr, 0\n.irp rs,rax,rcx,rdx,rbx,rsp,rbp,rsi,rdi,r8,r9,r10,r11,r12,r13,r14,r15\n.ifc \\reg, %\\rs\n.set .Lfound, .Lfound+1\n.long \\type + (.Lregnr << 8)\n.endif\n.set .Lregnr, .Lregnr+1\n.endr\n.set .Lregnr, 0\n.irp rs,eax,ecx,edx,ebx,esp,ebp,esi,edi,r8d,r9d,r10d,r11d,r12d,r13d,r14d,r15d\n.ifc \\reg, %\\rs\n.set .Lfound, .Lfound+1\n.long \\type + (.Lregnr << 8)\n.endif\n.set .Lregnr, .Lregnr+1\n.endr\n.if (.Lfound != 1)\n.error \"extable_type_reg: bad register argument\"\n.endif\n.endm\nextable_type_reg reg=%eax, type=10 \n.purgem extable_type_reg\n .popsection\n", inlateout("ax") val as u32 => __t0, in("cx") msr as u32, in("dx") (val >> (32 as i32)) as u32, options(nostack, att_syntax));
err = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
}
if static_branch_unlikely__2!(unsafe { core::ptr::addr_of_mut!(__tracepoint_write_msr.key) }) != 0 {
unsafe { do_trace_write_msr(msr, val, err) };
}
err
}

fn wrmsrq_safe(msr: u32, val: u64) -> i32 {
native_write_msr_safe(msr, val)
}

fn rdmsrq_safe(msr: u32, p: *mut u64) -> i32 {
native_read_msr_safe(msr, p)
}

// Instruction opcode for WRMSRNS supported in binutils >= 2.40
/// struct temp_masks - Bitmasks for temperature readings
/// @tcc_offset:			TCC offset in MSR_TEMPERATURE_TARGET
/// @digital_readout:		Digital readout in MSR_IA32_THERM_STATUS
/// @pkg_digital_readout:	Digital readout in MSR_IA32_PACKAGE_THERM_STATUS
///
/// Bitmasks to extract the fields of the MSR_TEMPERATURE and IA32_[PACKAGE]_
/// THERM_STATUS registers for different processor models.
///
/// The bitmask of TjMax is not included in this structure. It is always 0xff.
#[repr(C)]
#[derive(Clone, Copy)]
struct temp_masks {
    tcc_offset: u32,
    digital_readout: u32,
    pkg_digital_readout: u32,
}


#[unsafe(link_section = ".init.rodata")]
static temp_nehalem: temp_masks = temp_masks { tcc_offset: 0u32, digital_readout: 127u32, pkg_digital_readout: 127u32 };

#[unsafe(link_section = ".init.rodata")]
static temp_haswell_x: temp_masks = temp_masks { tcc_offset: 15u32, digital_readout: 127u32, pkg_digital_readout: 127u32 };

#[unsafe(link_section = ".init.rodata")]
static temp_broadwell: temp_masks = temp_masks { tcc_offset: 63u32, digital_readout: 127u32, pkg_digital_readout: 127u32 };

#[unsafe(link_section = ".init.rodata")]
static temp_goldmont: temp_masks = temp_masks { tcc_offset: 127u32, digital_readout: 127u32, pkg_digital_readout: 127u32 };

#[unsafe(link_section = ".init.rodata")]
static temp_tigerlake: temp_masks = temp_masks { tcc_offset: 63u32, digital_readout: 255u32, pkg_digital_readout: 255u32 };

#[unsafe(link_section = ".init.rodata")]
static temp_sapphirerapids: temp_masks = temp_masks { tcc_offset: 63u32, digital_readout: 127u32, pkg_digital_readout: 255u32 };

/// Use these masks for processors not included in @tcc_cpu_ids.
#[unsafe(link_section = ".data..ro_after_init")]
static mut intel_tcc_temp_masks: temp_masks = temp_masks { tcc_offset: 127u32, digital_readout: 255u32, pkg_digital_readout: 255u32 };

#[unsafe(link_section = ".init.rodata")]
static mut intel_tcc_cpu_ids: core::mem::MaybeUninit<__SlateImage_intel_tcc_cpu_ids> = unsafe {
let mut __t0: core::mem::MaybeUninit<__SlateImage_intel_tcc_cpu_ids> = core::mem::MaybeUninit::zeroed();
let __t1 = __t0.as_mut_ptr();
(*__t1).0[0].vendor = ((1550u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_53>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[0].family = ((1550u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_53_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[0].model = ((1550u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_53_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[0].steppings = 0u16;
(*__t1).0[0].feature = 0u16;
(*__t1).0[0].flags = 1u16;
(*__t1).0[0].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[0].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_nehalem) }) });
(*__t1).0[1].vendor = ((1551u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_54>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[1].family = ((1551u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_54_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[1].model = ((1551u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_54_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[1].steppings = 0u16;
(*__t1).0[1].feature = 0u16;
(*__t1).0[1].flags = 1u16;
(*__t1).0[1].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[1].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_nehalem) }) });
(*__t1).0[2].vendor = ((1558u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_55>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[2].family = ((1558u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_55_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[2].model = ((1558u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_55_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[2].steppings = 0u16;
(*__t1).0[2].feature = 0u16;
(*__t1).0[2].flags = 1u16;
(*__t1).0[2].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[2].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_nehalem) }) });
(*__t1).0[3].vendor = ((1559u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_56>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[3].family = ((1559u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_56_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[3].model = ((1559u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_56_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[3].steppings = 0u16;
(*__t1).0[3].feature = 0u16;
(*__t1).0[3].flags = 1u16;
(*__t1).0[3].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[3].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_nehalem) }) });
(*__t1).0[4].vendor = ((1565u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_57>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[4].family = ((1565u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_57_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[4].model = ((1565u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_57_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[4].steppings = 0u16;
(*__t1).0[4].feature = 0u16;
(*__t1).0[4].flags = 1u16;
(*__t1).0[4].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[4].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_nehalem) }) });
(*__t1).0[5].vendor = ((1566u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_58>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[5].family = ((1566u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_58_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[5].model = ((1566u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_58_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[5].steppings = 0u16;
(*__t1).0[5].feature = 0u16;
(*__t1).0[5].flags = 1u16;
(*__t1).0[5].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[5].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_nehalem) }) });
(*__t1).0[6].vendor = ((1567u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_59>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[6].family = ((1567u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_59_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[6].model = ((1567u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_59_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[6].steppings = 0u16;
(*__t1).0[6].feature = 0u16;
(*__t1).0[6].flags = 1u16;
(*__t1).0[6].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[6].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_nehalem) }) });
(*__t1).0[7].vendor = ((1562u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_60>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[7].family = ((1562u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_60_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[7].model = ((1562u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_60_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[7].steppings = 0u16;
(*__t1).0[7].feature = 0u16;
(*__t1).0[7].flags = 1u16;
(*__t1).0[7].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[7].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_nehalem) }) });
(*__t1).0[8].vendor = ((1582u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_61>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[8].family = ((1582u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_61_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[8].model = ((1582u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_61_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[8].steppings = 0u16;
(*__t1).0[8].feature = 0u16;
(*__t1).0[8].flags = 1u16;
(*__t1).0[8].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[8].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_nehalem) }) });
(*__t1).0[9].vendor = ((1573u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_62>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[9].family = ((1573u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_62_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[9].model = ((1573u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_62_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[9].steppings = 0u16;
(*__t1).0[9].feature = 0u16;
(*__t1).0[9].flags = 1u16;
(*__t1).0[9].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[9].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_nehalem) }) });
(*__t1).0[10].vendor = ((1580u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_63>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[10].family = ((1580u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_63_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[10].model = ((1580u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_63_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[10].steppings = 0u16;
(*__t1).0[10].feature = 0u16;
(*__t1).0[10].flags = 1u16;
(*__t1).0[10].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[10].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_nehalem) }) });
(*__t1).0[11].vendor = ((1583u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_64>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[11].family = ((1583u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_64_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[11].model = ((1583u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_64_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[11].steppings = 0u16;
(*__t1).0[11].feature = 0u16;
(*__t1).0[11].flags = 1u16;
(*__t1).0[11].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[11].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_nehalem) }) });
(*__t1).0[12].vendor = ((1578u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_65>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[12].family = ((1578u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_65_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[12].model = ((1578u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_65_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[12].steppings = 0u16;
(*__t1).0[12].feature = 0u16;
(*__t1).0[12].flags = 1u16;
(*__t1).0[12].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[12].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_nehalem) }) });
(*__t1).0[13].vendor = ((1581u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_66>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[13].family = ((1581u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_66_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[13].model = ((1581u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_66_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[13].steppings = 0u16;
(*__t1).0[13].feature = 0u16;
(*__t1).0[13].flags = 1u16;
(*__t1).0[13].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[13].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_nehalem) }) });
(*__t1).0[14].vendor = ((1594u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_67>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[14].family = ((1594u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_67_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[14].model = ((1594u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_67_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[14].steppings = 0u16;
(*__t1).0[14].feature = 0u16;
(*__t1).0[14].flags = 1u16;
(*__t1).0[14].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[14].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_nehalem) }) });
(*__t1).0[15].vendor = ((1598u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_68>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[15].family = ((1598u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_68_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[15].model = ((1598u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_68_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[15].steppings = 0u16;
(*__t1).0[15].feature = 0u16;
(*__t1).0[15].flags = 1u16;
(*__t1).0[15].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[15].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_haswell_x) }) });
(*__t1).0[16].vendor = ((1596u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_69>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[16].family = ((1596u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_69_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[16].model = ((1596u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_69_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[16].steppings = 0u16;
(*__t1).0[16].feature = 0u16;
(*__t1).0[16].flags = 1u16;
(*__t1).0[16].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[16].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_nehalem) }) });
(*__t1).0[17].vendor = ((1599u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_70>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[17].family = ((1599u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_70_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[17].model = ((1599u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_70_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[17].steppings = 0u16;
(*__t1).0[17].feature = 0u16;
(*__t1).0[17].flags = 1u16;
(*__t1).0[17].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[17].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_haswell_x) }) });
(*__t1).0[18].vendor = ((1605u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_71>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[18].family = ((1605u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_71_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[18].model = ((1605u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_71_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[18].steppings = 0u16;
(*__t1).0[18].feature = 0u16;
(*__t1).0[18].flags = 1u16;
(*__t1).0[18].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[18].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_nehalem) }) });
(*__t1).0[19].vendor = ((1606u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_72>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[19].family = ((1606u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_72_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[19].model = ((1606u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_72_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[19].steppings = 0u16;
(*__t1).0[19].feature = 0u16;
(*__t1).0[19].flags = 1u16;
(*__t1).0[19].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[19].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_nehalem) }) });
(*__t1).0[20].vendor = ((1597u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_73>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[20].family = ((1597u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_73_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[20].model = ((1597u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_73_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[20].steppings = 0u16;
(*__t1).0[20].feature = 0u16;
(*__t1).0[20].flags = 1u16;
(*__t1).0[20].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[20].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_broadwell) }) });
(*__t1).0[21].vendor = ((1607u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_74>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[21].family = ((1607u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_74_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[21].model = ((1607u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_74_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[21].steppings = 0u16;
(*__t1).0[21].feature = 0u16;
(*__t1).0[21].flags = 1u16;
(*__t1).0[21].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[21].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_broadwell) }) });
(*__t1).0[22].vendor = ((1615u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_75>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[22].family = ((1615u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_75_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[22].model = ((1615u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_75_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[22].steppings = 0u16;
(*__t1).0[22].feature = 0u16;
(*__t1).0[22].flags = 1u16;
(*__t1).0[22].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[22].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_haswell_x) }) });
(*__t1).0[23].vendor = ((1622u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_76>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[23].family = ((1622u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_76_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[23].model = ((1622u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_76_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[23].steppings = 0u16;
(*__t1).0[23].feature = 0u16;
(*__t1).0[23].flags = 1u16;
(*__t1).0[23].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[23].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_haswell_x) }) });
(*__t1).0[24].vendor = ((1614u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_77>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[24].family = ((1614u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_77_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[24].model = ((1614u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_77_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[24].steppings = 0u16;
(*__t1).0[24].feature = 0u16;
(*__t1).0[24].flags = 1u16;
(*__t1).0[24].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[24].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_broadwell) }) });
(*__t1).0[25].vendor = ((1630u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_78>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[25].family = ((1630u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_78_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[25].model = ((1630u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_78_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[25].steppings = 0u16;
(*__t1).0[25].feature = 0u16;
(*__t1).0[25].flags = 1u16;
(*__t1).0[25].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[25].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_broadwell) }) });
(*__t1).0[26].vendor = ((1621u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_79>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[26].family = ((1621u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_79_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[26].model = ((1621u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_79_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[26].steppings = 0u16;
(*__t1).0[26].feature = 0u16;
(*__t1).0[26].flags = 1u16;
(*__t1).0[26].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[26].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_haswell_x) }) });
(*__t1).0[27].vendor = ((1678u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_80>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[27].family = ((1678u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_80_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[27].model = ((1678u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_80_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[27].steppings = 0u16;
(*__t1).0[27].feature = 0u16;
(*__t1).0[27].flags = 1u16;
(*__t1).0[27].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[27].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_broadwell) }) });
(*__t1).0[28].vendor = ((1694u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_81>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[28].family = ((1694u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_81_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[28].model = ((1694u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_81_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[28].steppings = 0u16;
(*__t1).0[28].feature = 0u16;
(*__t1).0[28].flags = 1u16;
(*__t1).0[28].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[28].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_broadwell) }) });
(*__t1).0[29].vendor = ((1701u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_82>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[29].family = ((1701u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_82_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[29].model = ((1701u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_82_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[29].steppings = 0u16;
(*__t1).0[29].feature = 0u16;
(*__t1).0[29].flags = 1u16;
(*__t1).0[29].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[29].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_broadwell) }) });
(*__t1).0[30].vendor = ((1702u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_83>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[30].family = ((1702u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_83_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[30].model = ((1702u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_83_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[30].steppings = 0u16;
(*__t1).0[30].feature = 0u16;
(*__t1).0[30].flags = 1u16;
(*__t1).0[30].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[30].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_broadwell) }) });
(*__t1).0[31].vendor = ((1638u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_84>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[31].family = ((1638u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_84_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[31].model = ((1638u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_84_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[31].steppings = 0u16;
(*__t1).0[31].feature = 0u16;
(*__t1).0[31].flags = 1u16;
(*__t1).0[31].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[31].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_broadwell) }) });
(*__t1).0[32].vendor = ((1642u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_85>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[32].family = ((1642u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_85_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[32].model = ((1642u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_85_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[32].steppings = 0u16;
(*__t1).0[32].feature = 0u16;
(*__t1).0[32].flags = 1u16;
(*__t1).0[32].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[32].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_broadwell) }) });
(*__t1).0[33].vendor = ((1644u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_86>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[33].family = ((1644u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_86_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[33].model = ((1644u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_86_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[33].steppings = 0u16;
(*__t1).0[33].feature = 0u16;
(*__t1).0[33].flags = 1u16;
(*__t1).0[33].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[33].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_broadwell) }) });
(*__t1).0[34].vendor = ((1661u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_87>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[34].family = ((1661u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_87_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[34].model = ((1661u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_87_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[34].steppings = 0u16;
(*__t1).0[34].feature = 0u16;
(*__t1).0[34].flags = 1u16;
(*__t1).0[34].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[34].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_broadwell) }) });
(*__t1).0[35].vendor = ((1662u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_88>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[35].family = ((1662u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_88_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[35].model = ((1662u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_88_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[35].steppings = 0u16;
(*__t1).0[35].feature = 0u16;
(*__t1).0[35].flags = 1u16;
(*__t1).0[35].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[35].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_broadwell) }) });
(*__t1).0[36].vendor = ((1693u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_89>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[36].family = ((1693u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_89_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[36].model = ((1693u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_89_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[36].steppings = 0u16;
(*__t1).0[36].feature = 0u16;
(*__t1).0[36].flags = 1u16;
(*__t1).0[36].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[36].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_broadwell) }) });
(*__t1).0[37].vendor = ((1703u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_90>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[37].family = ((1703u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_90_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[37].model = ((1703u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_90_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[37].steppings = 0u16;
(*__t1).0[37].feature = 0u16;
(*__t1).0[37].flags = 1u16;
(*__t1).0[37].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[37].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_broadwell) }) });
(*__t1).0[38].vendor = ((1676u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_91>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[38].family = ((1676u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_91_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[38].model = ((1676u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_91_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[38].steppings = 0u16;
(*__t1).0[38].feature = 0u16;
(*__t1).0[38].flags = 1u16;
(*__t1).0[38].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[38].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_tigerlake) }) });
(*__t1).0[39].vendor = ((1677u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_92>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[39].family = ((1677u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_92_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[39].model = ((1677u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_92_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[39].steppings = 0u16;
(*__t1).0[39].feature = 0u16;
(*__t1).0[39].flags = 1u16;
(*__t1).0[39].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[39].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_tigerlake) }) });
(*__t1).0[40].vendor = ((1679u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_93>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[40].family = ((1679u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_93_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[40].model = ((1679u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_93_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[40].steppings = 0u16;
(*__t1).0[40].feature = 0u16;
(*__t1).0[40].flags = 1u16;
(*__t1).0[40].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[40].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_sapphirerapids) }) });
(*__t1).0[41].vendor = ((1743u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_94>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[41].family = ((1743u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_94_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[41].model = ((1743u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_94_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[41].steppings = 0u16;
(*__t1).0[41].feature = 0u16;
(*__t1).0[41].flags = 1u16;
(*__t1).0[41].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[41].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_sapphirerapids) }) });
(*__t1).0[42].vendor = ((1674u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_95>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[42].family = ((1674u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_95_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[42].model = ((1674u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_95_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[42].steppings = 0u16;
(*__t1).0[42].feature = 0u16;
(*__t1).0[42].flags = 1u16;
(*__t1).0[42].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[42].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_broadwell) }) });
(*__t1).0[43].vendor = ((1687u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_96>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[43].family = ((1687u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_96_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[43].model = ((1687u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_96_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[43].steppings = 0u16;
(*__t1).0[43].feature = 0u16;
(*__t1).0[43].flags = 1u16;
(*__t1).0[43].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[43].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_tigerlake) }) });
(*__t1).0[44].vendor = ((1690u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_97>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[44].family = ((1690u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_97_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[44].model = ((1690u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_97_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[44].steppings = 0u16;
(*__t1).0[44].feature = 0u16;
(*__t1).0[44].flags = 1u16;
(*__t1).0[44].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[44].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_tigerlake) }) });
(*__t1).0[45].vendor = ((1719u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_98>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[45].family = ((1719u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_98_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[45].model = ((1719u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_98_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[45].steppings = 0u16;
(*__t1).0[45].feature = 0u16;
(*__t1).0[45].flags = 1u16;
(*__t1).0[45].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[45].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_tigerlake) }) });
(*__t1).0[46].vendor = ((1722u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_99>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[46].family = ((1722u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_99_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[46].model = ((1722u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_99_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[46].steppings = 0u16;
(*__t1).0[46].feature = 0u16;
(*__t1).0[46].flags = 1u16;
(*__t1).0[46].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[46].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_tigerlake) }) });
(*__t1).0[47].vendor = ((1727u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_100>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[47].family = ((1727u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_100_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[47].model = ((1727u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_100_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[47].steppings = 0u16;
(*__t1).0[47].feature = 0u16;
(*__t1).0[47].flags = 1u16;
(*__t1).0[47].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[47].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_tigerlake) }) });
(*__t1).0[48].vendor = ((1564u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_101>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[48].family = ((1564u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_101_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[48].model = ((1564u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_101_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[48].steppings = 0u16;
(*__t1).0[48].feature = 0u16;
(*__t1).0[48].flags = 1u16;
(*__t1).0[48].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[48].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_nehalem) }) });
(*__t1).0[49].vendor = ((1574u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_102>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[49].family = ((1574u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_102_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[49].model = ((1574u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_102_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[49].steppings = 0u16;
(*__t1).0[49].feature = 0u16;
(*__t1).0[49].flags = 1u16;
(*__t1).0[49].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[49].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_nehalem) }) });
(*__t1).0[50].vendor = ((1590u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_103>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[50].family = ((1590u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_103_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[50].model = ((1590u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_103_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[50].steppings = 0u16;
(*__t1).0[50].feature = 0u16;
(*__t1).0[50].flags = 1u16;
(*__t1).0[50].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[50].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_nehalem) }) });
(*__t1).0[51].vendor = ((1575u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_104>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[51].family = ((1575u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_104_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[51].model = ((1575u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_104_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[51].steppings = 0u16;
(*__t1).0[51].feature = 0u16;
(*__t1).0[51].flags = 1u16;
(*__t1).0[51].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[51].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_nehalem) }) });
(*__t1).0[52].vendor = ((1591u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_105>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[52].family = ((1591u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_105_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[52].model = ((1591u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_105_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[52].steppings = 0u16;
(*__t1).0[52].feature = 0u16;
(*__t1).0[52].flags = 1u16;
(*__t1).0[52].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[52].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_broadwell) }) });
(*__t1).0[53].vendor = ((1613u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_106>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[53].family = ((1613u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_106_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[53].model = ((1613u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_106_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[53].steppings = 0u16;
(*__t1).0[53].feature = 0u16;
(*__t1).0[53].flags = 1u16;
(*__t1).0[53].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[53].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_broadwell) }) });
(*__t1).0[54].vendor = ((1610u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_107>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[54].family = ((1610u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_107_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[54].model = ((1610u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_107_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[54].steppings = 0u16;
(*__t1).0[54].feature = 0u16;
(*__t1).0[54].flags = 1u16;
(*__t1).0[54].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[54].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_broadwell) }) });
(*__t1).0[55].vendor = ((1612u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_108>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[55].family = ((1612u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_108_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[55].model = ((1612u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_108_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[55].steppings = 0u16;
(*__t1).0[55].feature = 0u16;
(*__t1).0[55].flags = 1u16;
(*__t1).0[55].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[55].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_broadwell) }) });
(*__t1).0[56].vendor = ((1626u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_109>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[56].family = ((1626u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_109_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[56].model = ((1626u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_109_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[56].steppings = 0u16;
(*__t1).0[56].feature = 0u16;
(*__t1).0[56].flags = 1u16;
(*__t1).0[56].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[56].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_broadwell) }) });
(*__t1).0[57].vendor = ((1653u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_110>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[57].family = ((1653u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_110_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[57].model = ((1653u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_110_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[57].steppings = 0u16;
(*__t1).0[57].feature = 0u16;
(*__t1).0[57].flags = 1u16;
(*__t1).0[57].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[57].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_broadwell) }) });
(*__t1).0[58].vendor = ((1628u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_111>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[58].family = ((1628u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_111_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[58].model = ((1628u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_111_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[58].steppings = 0u16;
(*__t1).0[58].feature = 0u16;
(*__t1).0[58].flags = 1u16;
(*__t1).0[58].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[58].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_goldmont) }) });
(*__t1).0[59].vendor = ((1631u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_112>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[59].family = ((1631u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_112_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[59].model = ((1631u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_112_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[59].steppings = 0u16;
(*__t1).0[59].feature = 0u16;
(*__t1).0[59].flags = 1u16;
(*__t1).0[59].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[59].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_goldmont) }) });
(*__t1).0[60].vendor = ((1658u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_113>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[60].family = ((1658u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_113_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[60].model = ((1658u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_113_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[60].steppings = 0u16;
(*__t1).0[60].feature = 0u16;
(*__t1).0[60].flags = 1u16;
(*__t1).0[60].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[60].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_goldmont) }) });
(*__t1).0[61].vendor = ((1670u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_114>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[61].family = ((1670u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_114_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[61].model = ((1670u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_114_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[61].steppings = 0u16;
(*__t1).0[61].feature = 0u16;
(*__t1).0[61].flags = 1u16;
(*__t1).0[61].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[61].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_broadwell) }) });
(*__t1).0[62].vendor = ((1686u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_115>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[62].family = ((1686u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_115_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[62].model = ((1686u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_115_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[62].steppings = 0u16;
(*__t1).0[62].feature = 0u16;
(*__t1).0[62].flags = 1u16;
(*__t1).0[62].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[62].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_broadwell) }) });
(*__t1).0[63].vendor = ((1692u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_116>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[63].family = ((1692u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_116_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[63].model = ((1692u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_116_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[63].steppings = 0u16;
(*__t1).0[63].feature = 0u16;
(*__t1).0[63].flags = 1u16;
(*__t1).0[63].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[63].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_broadwell) }) });
(*__t1).0[64].vendor = ((1726u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_117>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[64].family = ((1726u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_117_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[64].model = ((1726u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_117_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[64].steppings = 0u16;
(*__t1).0[64].feature = 0u16;
(*__t1).0[64].flags = 1u16;
(*__t1).0[64].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[64].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_tigerlake) }) });
(*__t1).0[65].vendor = ((1623u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_118>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[65].family = ((1623u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_118_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[65].model = ((1623u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_118_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[65].steppings = 0u16;
(*__t1).0[65].feature = 0u16;
(*__t1).0[65].flags = 1u16;
(*__t1).0[65].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[65].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_broadwell) }) });
(*__t1).0[66].vendor = ((1669u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_119>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 16i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(23u64))) >> 16i32) as u16;
(*__t1).0[66].family = ((1669u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_119_2>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 8i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(15u64))) >> 8i32) as u16;
(*__t1).0[66].model = ((1669u64 & (((core::mem::size_of::<__SlateRecord_intel_tcc_c_119_3>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(7u64))) >> 0i32) as u16;
(*__t1).0[66].steppings = 0u16;
(*__t1).0[66].feature = 0u16;
(*__t1).0[66].flags = 1u16;
(*__t1).0[66].r#type = 0u8;
(core::ptr::addr_of_mut!((*__t1).0[66].driver_data) as *mut *const ()).write_unaligned(unsafe { core::mem::transmute::<*const temp_masks, *const ()>(unsafe { core::ptr::addr_of!(temp_broadwell) }) });
__t0
};

core::arch::global_asm!(".section\t\".initcall4.init\", \"a\"\t\t\n__initcall__kmod_intel_tcc__103_138_intel_tcc_init4:\t\t\t\n.long\t{0} - .\t\n.previous\t\t\t\t\t\n\n.section \".export_symbol\",\"a\" ; __export_symbol_intel_tcc_get_offset_mask: ; .asciz \"\" ; .ascii \"INTEL_TCC\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_intel_tcc_get_tjmax: ; .asciz \"GPL\" ; .ascii \"INTEL_TCC\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_intel_tcc_get_offset: ; .asciz \"GPL\" ; .ascii \"INTEL_TCC\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_intel_tcc_set_offset: ; .asciz \"GPL\" ; .ascii \"INTEL_TCC\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_intel_tcc_get_temp: ; .asciz \"GPL\" ; .ascii \"INTEL_TCC\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous", sym intel_tcc_init, sym intel_tcc_get_offset_mask, sym intel_tcc_get_tjmax, sym intel_tcc_get_offset, sym intel_tcc_set_offset, sym intel_tcc_get_temp, options(att_syntax));

#[unsafe(link_section = ".init.text")]
extern "C" fn intel_tcc_init() -> i32 {
let id: *const x86_cpu_id = unsafe { x86_match_cpu(core::ptr::addr_of!((*(core::ptr::addr_of_mut!(intel_tcc_cpu_ids) as *mut __SlateImage_intel_tcc_cpu_ids)).0) as *const x86_cpu_id) };
if !id.is_null() {
unsafe { memcpy(core::ptr::addr_of_mut!(intel_tcc_temp_masks) as *mut (), (*id).driver_data as *const (), core::mem::size_of::<temp_masks>() as u64) };
}
0
}

/// Use subsys_initcall to ensure temperature bitmasks are initialized before
/// the drivers that use this library.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_intel_tcc_init_104: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i32>, *mut ()>(Some(intel_tcc_init)) };

/// intel_tcc_get_offset_mask() - Returns the bitmask to read TCC offset
///
/// Get the model-specific bitmask to extract TCC_OFFSET from the MSR
/// TEMPERATURE_TARGET register. If the mask is 0, it means the processor does
/// not support TCC offset.
///
/// Return: The model-specific bitmask for TCC offset.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.intel_tcc.intel_tcc_get_offset_mask")]
extern "C" fn intel_tcc_get_offset_mask() -> u32 {
unsafe { intel_tcc_temp_masks.tcc_offset }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_intel_tcc_get_offset_mask_105: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> u32>, *mut ()>(Some(intel_tcc_get_offset_mask)) };

/// get_temp_mask() - Returns the model-specific bitmask for temperature
///
/// @pkg: true: Package Thermal Sensor. false: Core Thermal Sensor.
///
/// Get the model-specific bitmask to extract the temperature reading from the
/// MSR_IA32_[PACKAGE]_THERM_STATUS register.
///
/// Callers must check if the thermal status registers are supported.
///
/// Return: The model-specific bitmask for temperature reading
fn get_temp_mask(pkg: bool) -> u32 {
if pkg { unsafe { intel_tcc_temp_masks.pkg_digital_readout } } else { unsafe { intel_tcc_temp_masks.digital_readout } }
}

/// intel_tcc_get_tjmax() - returns the default TCC activation Temperature
/// @cpu: cpu that the MSR should be run on, negative value means any cpu.
///
/// Get the TjMax value, which is the default thermal throttling or TCC
/// activation temperature in degrees C.
///
/// Return: Tjmax value in degrees C on success, negative error code otherwise.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.intel_tcc.intel_tcc_get_tjmax")]
extern "C" fn intel_tcc_get_tjmax(cpu: i32) -> i32 {
let mut msrval: msr = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(msrval) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut val: i32 = 0;
let err: i32 = if cpu < 0 {
rdmsrq_safe(418u32, unsafe { core::ptr::addr_of_mut!(msrval.__slate_anon_0.q) })
} else {
unsafe { rdmsrq_safe_on_cpu(cpu as u32, 418u32, core::ptr::addr_of_mut!(msrval.__slate_anon_0.q)) }
};
if err != 0 {
return err;
}
val = ((unsafe { msrval.__slate_anon_0.__slate_anon_0.l }) >> 16i32 & 255) as i32;
if val != 0 { val } else { 61i32.wrapping_neg() }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_intel_tcc_get_tjmax_106: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(i32) -> i32>, *mut ()>(Some(intel_tcc_get_tjmax)) };

/// intel_tcc_get_offset() - returns the TCC Offset value to Tjmax
/// @cpu: cpu that the MSR should be run on, negative value means any cpu.
///
/// Get the TCC offset value to Tjmax. The effective thermal throttling or TCC
/// activation temperature equals "Tjmax" - "TCC Offset", in degrees C.
///
/// Return: Tcc offset value in degrees C on success, negative error code otherwise.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.intel_tcc.intel_tcc_get_offset")]
extern "C" fn intel_tcc_get_offset(cpu: i32) -> i32 {
let mut val: msr = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(val) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let err: i32 = if cpu < 0 {
rdmsrq_safe(418u32, unsafe { core::ptr::addr_of_mut!(val.__slate_anon_0.q) })
} else {
unsafe { rdmsrq_safe_on_cpu(cpu as u32, 418u32, core::ptr::addr_of_mut!(val.__slate_anon_0.q)) }
};
if err != 0 {
return err;
}
((unsafe { val.__slate_anon_0.__slate_anon_0.l }) >> 24i32 & unsafe { intel_tcc_temp_masks.tcc_offset }) as i32
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_intel_tcc_get_offset_107: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(i32) -> i32>, *mut ()>(Some(intel_tcc_get_offset)) };

// intel_tcc_set_offset() - set the TCC offset value to Tjmax
// @cpu: cpu that the MSR should be run on, negative value means any cpu.
// @offset: TCC offset value in degree C
//
// Set the TCC Offset value to Tjmax. The effective thermal throttling or TCC
// activation temperature equals "Tjmax" - "TCC Offset", in degree C.
//
// Return: On success returns 0, negative error code otherwise.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.intel_tcc.intel_tcc_set_offset")]
extern "C" fn intel_tcc_set_offset(cpu: i32, offset: i32) -> i32 {
let mut val: msr = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(val) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut err: i32 = 0;
if (unsafe { intel_tcc_temp_masks.tcc_offset }) == 0 {
return 19i32.wrapping_neg();
}
if offset < 0 || (offset as u32) > unsafe { intel_tcc_temp_masks.tcc_offset } {
return 22i32.wrapping_neg();
}
if cpu < 0 {
err = rdmsrq_safe(418u32, unsafe { core::ptr::addr_of_mut!(val.__slate_anon_0.q) });
} else {
err = unsafe { rdmsrq_safe_on_cpu(cpu as u32, 418u32, core::ptr::addr_of_mut!(val.__slate_anon_0.q)) };
}
if err != 0 {
return err;
}
// MSR Locked
if ((unsafe { val.__slate_anon_0.__slate_anon_0.l }) as u64) & BIT!(31i32) != 0 {
return 1i32.wrapping_neg();
}
let __v1: u32 = (unsafe { val.__slate_anon_0.__slate_anon_0.l }) & !((unsafe { intel_tcc_temp_masks.tcc_offset }) << 24i32);
unsafe {
val.__slate_anon_0.__slate_anon_0.l = __v1;
}
unsafe {
val.__slate_anon_0.__slate_anon_0.l = val.__slate_anon_0.__slate_anon_0.l | ((offset << 24i32) as u32);
}
if cpu < 0 {
return wrmsrq_safe(418u32, unsafe { val.__slate_anon_0.q });
} else {
return unsafe { wrmsrq_safe_on_cpu(cpu as u32, 418u32, val.__slate_anon_0.q) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_intel_tcc_set_offset_108: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(i32, i32) -> i32>, *mut ()>(Some(intel_tcc_set_offset)) };

/// intel_tcc_get_temp() - returns the current temperature
/// @cpu: cpu that the MSR should be run on, negative value means any cpu.
/// @temp: pointer to the memory for saving cpu temperature.
/// @pkg: true: Package Thermal Sensor. false: Core Thermal Sensor.
///
/// Get the current temperature returned by the CPU core/package level
/// thermal sensor, in degrees C.
///
/// Return: 0 on success, negative error code otherwise.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.intel_tcc.intel_tcc_get_temp")]
extern "C" fn intel_tcc_get_temp(cpu: i32, temp: *mut i32, pkg: bool) -> i32 {
let msr: u32 = (if pkg { 433i32 } else { 412i32 }) as u32;
let mut mask: u32 = 0;
let mut val: msr = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(val) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let tjmax: i32 = intel_tcc_get_tjmax(cpu);
let mut err: i32 = 0;
if tjmax < 0 {
return tjmax;
}
if cpu < 0 {
err = rdmsrq_safe(msr, unsafe { core::ptr::addr_of_mut!(val.__slate_anon_0.q) });
} else {
err = unsafe { rdmsrq_safe_on_cpu(cpu as u32, msr, core::ptr::addr_of_mut!(val.__slate_anon_0.q)) };
}
if err != 0 {
return err;
}
// Temperature is beyond the valid thermal sensor range
if ((unsafe { val.__slate_anon_0.__slate_anon_0.l }) as u64) & BIT!(31i32) == 0 {
return 61i32.wrapping_neg();
}
mask = get_temp_mask(pkg);
unsafe {
*temp = (tjmax as u32).wrapping_sub(val.__slate_anon_0.__slate_anon_0.l >> 16i32 & mask) as i32;
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_intel_tcc_get_temp_109: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(i32, *mut i32, bool) -> i32>, *mut ()>(Some(intel_tcc_get_temp)) };

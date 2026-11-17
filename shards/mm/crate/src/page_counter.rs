
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
//! SPDX-License-Identifier: GPL-2.0
//!
//! Lockless hierarchical page accounting & limiting
//!
//! Copyright (C) 2014 Red Hat, Inc., Johannes Weiner
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
//! Optimization barrier
//!
//! The "volatile" is due to gcc bugs
//! Yes, this permits 64-bit accesses on 32-bit architectures. These will
//! actually be atomic in some cases (namely Armv7 + LPAE), but for others we
//! rely on the access being split into 2x32-bit accesses for a 32-bit quantity
//! (e.g. a virtual address) and a strong prevailing wind.
//! See linux/instrumentation.h
//! 64 bit
//! unused, was: #define EX_TYPE_COPY	 4
//! Begin/end of an instrumentation safe region
//! Force a compilation error if condition is true, but also produce a
//! result (of value 0 and type int), so the expression can be used
//! e.g. in a structure initializer (or where-ever else comma expressions
//! aren't permitted).
//!
//! Take an error message as an optional second argument. If omitted,
//! default to the stringification of the tested expression.
//! These are for everybody (although not all archs will actually
//! discard it in modules)
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
//! strlen("ratelimit") + 1
//! label
//! Defines x86 CPU feature bits
//! This counts to 15. Any more, it will return 16th argument.
//! Atomic operations are already serializing on x86
//! The __sme_pa() and __sme_pa_nodebug() macros are meant for use when
//! writing to or comparing values from the cr3 register.  Having the
//! encryption mask set in cr3 enables the PGD entry to be encrypted and
//! avoid special case handling of PGD allocations.
//! PAGE_SHIFT determines the page size.
//!
//! Note: This definition is required because PAGE_SHIFT is used
//! in several places throughout the codebase.
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
//! ilog2 - log base 2 of 32-bit or a 64-bit unsigned value
//! @n: parameter
//!
//! constant-capable log of base 2 calculation
//! - this can be used to initialise global variables from constant data, hence
//! the massive ternary operator construction
//!
//! selects the appropriately-sized optimised version depending on sizeof(n)
//! @a is a power of 2 value
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
//! restart by calling sys_restart_syscall
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
//! cpumask_bits - get the bits in a cpumask
//! @maskp: the struct cpumask *
//!
//! You should only assume nr_cpu_ids bits of this mask are valid.  This is
//! a macro so it's const-correct.
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
// Concatenate two parameters, but allow them to be expanded beforehand.
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
// min - return minimum of two values of the same or compatible types
// @x: first value
// @y: second value
// Out of memory
// Device or resource busy
// Invalid argument
// Operation is not supported
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
// On 2MB or 1GB pages
// Extracts the PFN from a (pte|pmd|pud|pgd)val_t of a 4KB page
// Extracts the flags from a (pte|pmd|pud|pgd)val_t
// This includes the protection key value.
// No need to mask any bits for !PAE
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
macro_rules! BUGFLAG_TAINT {
($taint:expr) => {
($taint) << 8i32
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

macro_rules! RELOC_HIDE {
($ptr:expr, $off:expr) => {
(($ptr) as u64).wrapping_add(($off) as u64) as *mut ()
};
}

macro_rules! _AC {
() => {
1u64
};
}

macro_rules! __READ_ONCE__2 {
($x:expr) => {
unsafe { core::ptr::read_volatile(core::ptr::addr_of!($x) as *const i64) }
};
}

macro_rules! __WRITE_ONCE__8 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut u64, ($val) as u64) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE__4 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut u64, $val) };
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
macro_rules! __native_word {
() => {
((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
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

macro_rules! __types_ok__2 {
($ux:expr, $uy:expr) => {
__sign_use!($ux) & __sign_use!($uy)
};
}

macro_rules! absolute_pointer {
($val:expr) => {
RELOC_HIDE!(($val) as *mut (), 0i32)
};
}

macro_rules! arch_atomic64_sub_return {
($i:expr, $v:expr) => {
arch_atomic64_add_return($i.wrapping_neg(), $v)
};
}

macro_rules! cond_resched__8 {
() => {
let mut __v: i32;
__might_resched(c"/home/takashi/c-corpus/linux/mm/page_counter.c".as_ptr() as *const u8, 225i32, 0u32);
__v = _cond_resched();
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

macro_rules! unlikely__2 {
($x:expr) => {
(($x) != 0) as i64
};
}

unsafe extern "C" {
    static mut __SCK__WARN_trap: static_call_key;
    static mut __SCK__cond_resched: static_call_key;
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn __xchg_wrong_size();
    fn __xadd_wrong_size();
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn __cond_resched() -> i32;
    fn strcmp(arg0: *const u8, arg1: *const u8) -> i32;
    fn memparse(arg0: *const u8, arg1: *mut *mut u8) -> u64;
    fn __SCT__cond_resched() -> i32;
    fn __compiletime_assert_104();
    fn __compiletime_assert_107();
    fn __compiletime_assert_108();
    fn __compiletime_assert_111();
    fn __compiletime_assert_113();
    fn __compiletime_assert_114();
    fn __compiletime_assert_115();
    fn __compiletime_assert_116();
    fn __compiletime_assert_117();
    fn __compiletime_assert_118();
    fn __compiletime_assert_119();
    fn __compiletime_assert_120();
    fn __compiletime_assert_121();
    fn __compiletime_assert_122();
    fn __compiletime_assert_125();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_types_h_197 {
    counter: i64,
}


fn kasan_check_read(p: *const (), size: u32) -> bool {
true
}

fn kasan_check_write(p: *const (), size: u32) -> bool {
true
}

fn kcsan_check_access(ptr: *const (), size: u64, r#type: i32) {
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


#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
}


fn __kcsan_enable_current() {
}

fn __kcsan_disable_current() {
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
fn raw_atomic64_add_return(i: i64, v: *mut __SlateRecord_types_h_197) -> i64 {
arch_atomic64_add_return(i, v)
}

#[inline(always)]
fn raw_atomic64_sub(i: i64, v: *mut __SlateRecord_types_h_197) {
arch_atomic64_sub(i, v);
}

#[inline(always)]
fn raw_atomic64_sub_return(i: i64, v: *mut __SlateRecord_types_h_197) -> i64 {
arch_atomic64_sub_return!(i, v)
}

#[inline(always)]
fn raw_atomic64_xchg(v: *mut __SlateRecord_types_h_197, new: i64) -> i64 {
arch_atomic64_xchg(v, new)
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
fn raw_atomic_long_add_return(i: i64, v: *mut __SlateRecord_types_h_197) -> i64 {
raw_atomic64_add_return(i, v)
}

#[inline(always)]
fn raw_atomic_long_sub(i: i64, v: *mut __SlateRecord_types_h_197) {
raw_atomic64_sub(i, v);
}

#[inline(always)]
fn raw_atomic_long_sub_return(i: i64, v: *mut __SlateRecord_types_h_197) -> i64 {
raw_atomic64_sub_return(i, v)
}

#[inline(always)]
fn raw_atomic_long_xchg(v: *mut __SlateRecord_types_h_197, new: i64) -> i64 {
raw_atomic64_xchg(v, new)
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
core::arch::asm!("8: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 8b - ., 3; .popsection\n# {0}", const 8 as i32, options(nostack, att_syntax));
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
core::arch::asm!("9: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 9b - ., 4; .popsection\n# {0}", const 9 as i32, options(nostack, att_syntax));
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

#[inline(always)]
fn atomic_long_add_return(i: i64, v: *mut __SlateRecord_types_h_197) -> i64 {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_add_return(i, v)
}

#[inline(always)]
fn atomic_long_sub(i: i64, v: *mut __SlateRecord_types_h_197) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_sub(i, v);
}

#[inline(always)]
fn atomic_long_sub_return(i: i64, v: *mut __SlateRecord_types_h_197) -> i64 {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_sub_return(i, v)
}

#[inline(always)]
fn atomic_long_xchg(v: *mut __SlateRecord_types_h_197, new: i64) -> i64 {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_xchg(v, new)
}

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct cacheline_padding {
    x: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct page_counter {
    usage: __SlateRecord_types_h_197,
    failcnt: u64,
    _pad1_: cacheline_padding,
    emin: u64,
    min_usage: __SlateRecord_types_h_197,
    children_min_usage: __SlateRecord_types_h_197,
    elow: u64,
    low_usage: __SlateRecord_types_h_197,
    children_low_usage: __SlateRecord_types_h_197,
    watermark: u64,
    local_watermark: u64,
    _pad2_: cacheline_padding,
    protection_support: bool,
    track_failcnt: bool,
    min: u64,
    low: u64,
    high: u64,
    max: u64,
    parent: *mut page_counter,
}


fn page_counter_read(counter: *mut page_counter) -> u64 {
atomic_long_read((unsafe { core::ptr::addr_of_mut!((*counter).usage) }) as *const __SlateRecord_types_h_197) as u64
}

#[unsafe(no_mangle)]
extern "C" fn page_counter_charge(counter: *mut page_counter, nr_pages: u64) {
let mut c: *mut page_counter = unsafe { core::mem::zeroed() };
let protection: bool = track_protection(counter);
c = counter;
while !c.is_null() {
let new: i64 = atomic_long_add_return(nr_pages as i64, unsafe { core::ptr::addr_of_mut!((*c).usage) });
if protection {
propagate_protected_usage(c, new as u64);
}
// This is indeed racy, but we can live with some
// inaccuracy in the watermark.
//
// Notably, we have two watermarks to allow for both a globally
// visible peak and one that can be reset at a smaller scope.
//
// Since we reset both watermarks when the global reset occurs,
// we can guarantee that watermark >= local_watermark, so we
// don't need to do both comparisons every time.
//
// On systems with branch predictors, the inner condition should
// be almost free.
let mut __v0: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_113() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*c).local_watermark) as *const u64) };
if (new as u64) > __v0 {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_114() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__8!((*c).local_watermark, new);
if true {
break;
}
}
let mut __v5: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_115() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v5 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*c).watermark) as *const u64) };
if (new as u64) > __v5 {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_116() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__8!((*c).watermark, new);
if true {
break;
}
}
}
}
c = unsafe { (*c).parent };
}
}

#[unsafe(no_mangle)]
extern "C" fn page_counter_try_charge(counter: *mut page_counter, nr_pages: u64, fail: *mut *mut page_counter) -> bool {
let mut new: core::mem::MaybeUninit<i64> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut i64 = core::ptr::addr_of_mut!(new) as *mut i64;
let mut track_failcnt: bool = unsafe { core::mem::zeroed() };
let mut protection: bool = unsafe { core::mem::zeroed() };
let mut c: core::mem::MaybeUninit<*mut page_counter> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut page_counter = core::ptr::addr_of_mut!(c) as *mut *mut page_counter;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
protection = track_protection(counter);
track_failcnt = (*counter).track_failcnt;
*__slate_slot_10 = counter;
'__join_18: {
while !(*__slate_slot_10).is_null() {
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
// Charge speculatively to avoid an expensive CAS.  If
// a bigger charge fails, it might falsely lock out a
// racing smaller charge and send it into reclaim
// early, but the error is limited to the difference
// between the two sizes, which is less than 2M/4M in
// case of a THP locking out a regular page charge.
//
// The atomic_long_add_return() implies a full memory
// barrier between incrementing the count and reading
// the limit.  When racing with page_counter_set_max(),
// we either see the new limit or the setter sees the
// counter has changed and retries.
*__slate_slot_7 = atomic_long_add_return(nr_pages as i64, core::ptr::addr_of_mut!((*(*__slate_slot_10)).usage));
if (*__slate_slot_7 as u64) > (*(*__slate_slot_10)).max {
break '__join_18;
} else {
if protection {
propagate_protected_usage(*__slate_slot_10, *__slate_slot_7 as u64);
}
while false {
}
// see comment on page_counter_charge
let __v1: u64 = core::ptr::read_volatile(core::ptr::addr_of_mut!((*(*__slate_slot_10)).local_watermark) as *const u64);
if (*__slate_slot_7 as u64) > __v1 {
loop {
if true {
__WRITE_ONCE__8!((*(*__slate_slot_10)).local_watermark, *__slate_slot_7);
if true {
break;
}
}
}
while false {
}
let __v0: u64 = core::ptr::read_volatile(core::ptr::addr_of_mut!((*(*__slate_slot_10)).watermark) as *const u64);
if (*__slate_slot_7 as u64) > __v0 {
loop {
if true {
__WRITE_ONCE__8!((*(*__slate_slot_10)).watermark, *__slate_slot_7);
if true {
break;
}
}
}
}
}
*__slate_slot_10 = (*(*__slate_slot_10)).parent;
}
}
return true;
}
atomic_long_sub(nr_pages as i64, core::ptr::addr_of_mut!((*(*__slate_slot_10)).usage));
// This is racy, but we can live with some
// inaccuracy in the failcnt which is only used
// to report stats.
if track_failcnt {
__kcsan_disable_current();
let __v4: *mut page_counter = *__slate_slot_10;
let __v3: u64 = (*__v4).failcnt;
(*__v4).failcnt = __v3.wrapping_add(1);
__kcsan_enable_current();
}
*fail = *__slate_slot_10;
*__slate_slot_10 = counter;
while *__slate_slot_10 != *fail {
page_counter_cancel(*__slate_slot_10, nr_pages);
*__slate_slot_10 = (*(*__slate_slot_10)).parent;
}
return false;
}
}

#[unsafe(no_mangle)]
extern "C" fn page_counter_set_min(counter: *mut page_counter, nr_pages: u64) {
let mut c: *mut page_counter = unsafe { core::mem::zeroed() };
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_121() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__4!((*counter).min, nr_pages);
if true {
break;
}
}
c = counter;
while !c.is_null() {
propagate_protected_usage(c, atomic_long_read((unsafe { core::ptr::addr_of_mut!((*c).usage) }) as *const __SlateRecord_types_h_197) as u64);
c = unsafe { (*c).parent };
}
}

#[unsafe(no_mangle)]
extern "C" fn page_counter_set_low(counter: *mut page_counter, nr_pages: u64) {
let mut c: *mut page_counter = unsafe { core::mem::zeroed() };
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_122() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__4!((*counter).low, nr_pages);
if true {
break;
}
}
c = counter;
while !c.is_null() {
propagate_protected_usage(c, atomic_long_read((unsafe { core::ptr::addr_of_mut!((*c).usage) }) as *const __SlateRecord_types_h_197) as u64);
c = unsafe { (*c).parent };
}
}

#[unsafe(no_mangle)]
extern "C" fn page_counter_memparse(buf: *const u8, max: *const u8, nr_pages: *mut u64) -> i32 {
let mut end: *mut u8 = unsafe { core::mem::zeroed() };
let mut bytes: u64 = 0;
if (unsafe { strcmp(buf, max) }) == 0 {
unsafe {
*nr_pages = 9223372036854775807u64 / (_AC!() << 12i32);
}
return 0;
}
bytes = unsafe { memparse(buf, core::ptr::addr_of_mut!(end)) };
if ((unsafe { *end }) as i32) != 0 {
return 22i32.wrapping_neg();
}
let mut __v0: u64;
let __UNIQUE_ID_x__123: u64 = bytes / (_AC!() << 12i32);
let __UNIQUE_ID_y__124: u64 = 9223372036854775807u64 / (_AC!() << 12i32);
loop {
if __types_ok__2!(__UNIQUE_ID_x__123, __UNIQUE_ID_y__124) == 0 {
unsafe { __compiletime_assert_125() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__123 < __UNIQUE_ID_y__124 { __UNIQUE_ID_x__123 } else { __UNIQUE_ID_y__124 };
unsafe {
*nr_pages = __v0;
}
0
}

fn __might_resched(file: *const u8, line: i32, offsets: u32) {
}

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
#[inline(always)]
fn _cond_resched() -> i32 {
let mut __v0: Option<unsafe extern "C" fn() -> i32>;
__v0 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn() -> i32>>(__SCT__cond_resched as *const ()) };
unsafe { __v0.unwrap()() }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__cond_resched_89: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__cond_resched) }) as *mut ();

fn track_protection(c: *mut page_counter) -> bool {
unsafe { (*c).protection_support }
}

fn propagate_protected_usage(c: *mut page_counter, usage: u64) {
let mut protected: u64 = 0;
let mut old_protected: u64 = 0;
let mut delta: i64 = 0;
if unsafe { (*c).parent }.is_null() {
return;
}
let mut __v0: u64;
let __UNIQUE_ID_x__105: u64 = usage;
let mut __UNIQUE_ID_y__106: u64 = 0;
let mut __v1: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_104() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*c).min) as *const u64) };
__UNIQUE_ID_y__106 = __v1;
loop {
if __types_ok__2!(__UNIQUE_ID_x__105, __UNIQUE_ID_y__106) == 0 {
unsafe { __compiletime_assert_107() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__105 < __UNIQUE_ID_y__106 { __UNIQUE_ID_x__105 } else { __UNIQUE_ID_y__106 };
protected = __v0;
old_protected = atomic_long_read((unsafe { core::ptr::addr_of_mut!((*c).min_usage) }) as *const __SlateRecord_types_h_197) as u64;
if protected != old_protected {
old_protected = atomic_long_xchg(unsafe { core::ptr::addr_of_mut!((*c).min_usage) }, protected as i64) as u64;
delta = protected.wrapping_sub(old_protected) as i64;
if delta != 0 {
atomic_long_add(delta, unsafe { core::ptr::addr_of_mut!((*(*c).parent).children_min_usage) });
}
}
let mut __v4: u64;
let __UNIQUE_ID_x__109: u64 = usage;
let mut __UNIQUE_ID_y__110: u64 = 0;
let mut __v5: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_108() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v5 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*c).low) as *const u64) };
__UNIQUE_ID_y__110 = __v5;
loop {
if __types_ok__2!(__UNIQUE_ID_x__109, __UNIQUE_ID_y__110) == 0 {
unsafe { __compiletime_assert_111() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v4 = if __UNIQUE_ID_x__109 < __UNIQUE_ID_y__110 { __UNIQUE_ID_x__109 } else { __UNIQUE_ID_y__110 };
protected = __v4;
old_protected = atomic_long_read((unsafe { core::ptr::addr_of_mut!((*c).low_usage) }) as *const __SlateRecord_types_h_197) as u64;
if protected != old_protected {
old_protected = atomic_long_xchg(unsafe { core::ptr::addr_of_mut!((*c).low_usage) }, protected as i64) as u64;
delta = protected.wrapping_sub(old_protected) as i64;
if delta != 0 {
atomic_long_add(delta, unsafe { core::ptr::addr_of_mut!((*(*c).parent).children_low_usage) });
}
}
}

/// page_counter_cancel - take pages out of the local counter
/// @counter: counter
/// @nr_pages: number of pages to cancel
#[unsafe(no_mangle)]
extern "C" fn page_counter_cancel(counter: *mut page_counter, nr_pages: u64) {
let mut new: i64 = atomic_long_sub_return(nr_pages as i64, unsafe { core::ptr::addr_of_mut!((*counter).usage) });
// More uncharges than charges?
let mut __v0: i32;
let __ret_warn_on: i32 = (new < 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
2i32 | BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"page_counter underflow: %ld nr_pages=%lu\n".as_ptr() as *const u8, new, nr_pages) };
let mut __v1: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v1 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v2: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_page_counter_cancel_3, sym __slate_asm_string_page_counter_cancel_4, const 61 as i32, const 2323 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v2 = bug;
unsafe { __v1.unwrap()(__v2, new, nr_pages) };
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
new = 0;
atomic_long_set(unsafe { core::ptr::addr_of_mut!((*counter).usage) }, new);
}
if track_protection(counter) {
propagate_protected_usage(counter, new as u64);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_112: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

// page_counter_charge - hierarchically charge pages
// @counter: counter
// @nr_pages: number of pages to charge
//
// NOTE: This does not consider any configured counter limits.
// page_counter_try_charge - try to hierarchically charge pages
// @counter: counter
// @nr_pages: number of pages to charge
// @fail: points first counter to hit its limit, if any
//
// Returns %true on success, or %false and @fail if the counter or one
// of its ancestors has hit its configured limit.
/// page_counter_uncharge - hierarchically uncharge pages
/// @counter: counter
/// @nr_pages: number of pages to uncharge
#[unsafe(no_mangle)]
extern "C" fn page_counter_uncharge(counter: *mut page_counter, nr_pages: u64) {
let mut c: *mut page_counter = unsafe { core::mem::zeroed() };
c = counter;
while !c.is_null() {
page_counter_cancel(c, nr_pages);
c = unsafe { (*c).parent };
}
}

/// page_counter_set_max - set the maximum number of pages allowed
/// @counter: counter
/// @nr_pages: limit to set
///
/// Returns 0 on success, -EBUSY if the current number of pages on the
/// counter already exceeds the specified limit.
///
/// The caller must serialize invocations on the same counter.
#[unsafe(no_mangle)]
extern "C" fn page_counter_set_max(counter: *mut page_counter, nr_pages: u64) -> i32 {
loop {
let mut old: u64 = 0;
let mut usage: i64 = 0;
// Update the limit while making sure that it's not
// below the concurrently-changing counter value.
//
// The xchg implies two full memory barriers before
// and after, so the read-swap-read is ordered and
// ensures coherency with page_counter_try_charge():
// that function modifies the count before checking
// the limit, so if it sees the old limit, we see the
// modified counter and retry.
usage = page_counter_read(counter) as i64;
if (usage as u64) > nr_pages {
return 16i32.wrapping_neg();
}
let mut __v0: u64;
let mut __ai_ptr: *mut u64 = unsafe { core::ptr::addr_of_mut!((*counter).max) };
kcsan_mb!();
instrument_atomic_read_write(__ai_ptr as *const (), core::mem::size_of::<u64>() as u64);
let mut __v2: u64;
let mut __ret: u64 = nr_pages;
match (const { core::mem::size_of::<u64>() }) as u64 {
1 => {
unsafe {
let __t0: u64;
core::arch::asm!("xchgb {0:l}, ({1:r})", inlateout(reg) __ret as u64 => __t0, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
}
2 => {
unsafe {
let __t1: u64;
core::arch::asm!("xchgw {0:x}, ({1:r})", inlateout(reg) __ret as u64 => __t1, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, u64>(__t1 as u64) };
}
}
4 => {
unsafe {
let __t2: u64;
core::arch::asm!("xchgl {0:r}, ({1:r})", inlateout(reg) __ret as u64 => __t2, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, u64>(__t2 as u64) };
}
}
8 => {
unsafe {
let __t3: u64;
core::arch::asm!("xchgq {0:r}, ({1:r})", inlateout(reg) __ret as u64 => __t3, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
}
_ => {
unsafe { __xchg_wrong_size() };
}
}
__v2 = __ret;
__v0 = __v2;
old = __v0;
if page_counter_read(counter) <= (usage as u64) || nr_pages >= old {
return 0;
}
unsafe {
(*counter).max = old;
}
cond_resched__8!();
}
unsafe { core::mem::zeroed() }
}

// page_counter_set_min - set the amount of protected memory
// @counter: counter
// @nr_pages: value to set
//
// The caller must serialize invocations on the same counter.
// page_counter_set_low - set the amount of protected memory
// @counter: counter
// @nr_pages: value to set
//
// The caller must serialize invocations on the same counter.
// page_counter_memparse - memparse() for page counter limits
// @buf: string to parse
// @max: string meaning maximum possible value
// @nr_pages: returns the result in number of pages
//
// Returns -EINVAL, or 0 and @nr_pages on success.  @nr_pages will be
// limited to %PAGE_COUNTER_MAX.
static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

static __slate_asm_string_page_counter_cancel_3: [u8; 42] = *b"page_counter underflow: %ld nr_pages=%lu\n\0";

static __slate_asm_string_page_counter_cancel_4: [u8; 47] = *b"/home/takashi/c-corpus/linux/mm/page_counter.c\0";

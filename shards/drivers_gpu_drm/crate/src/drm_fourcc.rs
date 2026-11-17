
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
//! Copyright (c) 2016 Laurent Pinchart <laurent.pinchart@ideasonboard.com>
//!
//! DRM core format related functions
//!
//! Permission to use, copy, modify, distribute, and sell this software and its
//! documentation for any purpose is hereby granted without fee, provided that
//! the above copyright notice appear in all copies and that both that copyright
//! notice and this permission notice appear in supporting documentation, and
//! that the name of the copyright holders not be used in advertising or
//! publicity pertaining to distribution of the software without specific,
//! written prior permission.  The copyright holders make no representations
//! about the suitability of this software for any purpose.  It is provided "as
//! is" without express or implied warranty.
//!
//! THE COPYRIGHT HOLDERS DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
//! INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO
//! EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY SPECIAL, INDIRECT OR
//! CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE,
//! DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
//! TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
//! OF THIS SOFTWARE.
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
//! __ADDRESSABLE() is used to ensure the key symbol doesn't get stripped from
//! the symbol table so that objtool can reference it when it generates the
//! .static_call_sites section.
//! Force a compilation error if condition is true, but also produce a
//! result (of value 0 and type int), so the expression can be used
//! e.g. in a structure initializer (or where-ever else comma expressions
//! aren't permitted).
//!
//! Take an error message as an optional second argument. If omitted,
//! default to the stringification of the tested expression.
//! These are for everybody (although not all archs will actually
//! discard it in modules)
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
//! Internal kernel timer frequency
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! strlen("ratelimit") + 1
//! @a is a power of 2 value
//! container_of - cast a member of a structure out to the containing structure
//! @ptr:	the pointer to the member.
//! @type:	the type of the container struct this is embedded in.
//! @member:	the name of the member within the struct.
//!
//! WARNING: any const qualifier of @ptr is lost.
//! Do not use container_of() in new code.
//! Check at compile time that something is of a particular type.
//! Always evaluates to 1 so you may use it easily in comparisons.
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
//! These are non-NULL pointers that will result in page faults
//! under normal circumstances, used to verify that nobody uses
//! non-initialized list entries.
//! list_entry - get the struct for this entry
//! @ptr:	the &struct list_head pointer.
//! @type:	the type of the struct this is embedded in.
//! @member:	the name of the list_head within the struct.
//! No such file or directory
//! Value too large for defined data type
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
//! An exchange-type operation, which takes a value and a pointer, and
//! returns the old value.
//! restart by calling sys_restart_syscall
//! ZONE_MOVABLE allowed
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
//! This is the default CPU features testing macro to use in code.
//!
//! It is for detection of features which need kernel infrastructure to be
//! used.  It may *not* directly test the CPU itself.  Use the cpu_has() family
//! if you want true runtime testing of CPU features, like in hypervisor code
//! where you are supporting a possible guest feature where host support for it
//! is not relevant.
//! rescheduling necessary
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
//! mutex_init - initialize the mutex
//! @mutex: the mutex to be initialized
//!
//! Initialize the mutex to unlocked state.
//!
//! It is not allowed to initialize an already locked mutex.
//! These constants are defined in the MSan LLVM instrumentation pass.
//! get_task_state():
//! Dependency order vs. p above.
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
//! Keep unconverted code working
//! Number of bits in the handle that stack depot doesn't use. Users may store
//! information in them via stack_depot_set/get_extra_bits.
//! Reserve 0 for the invalid format specifier
//! DRM_FORMAT_MAX_PLANES - maximum number of planes a DRM format can have
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
// "tsc" Time Stamp Counter
// "sse"
// "rdtscp" RDTSCP
// "rep_good" REP microcode works well
// "popcnt" POPCNT instruction
// "3dnowprefetch" 3DNow prefetch instructions
// "pti" Kernel Page Table Isolation enabled
// "ibpb" Indirect Branch Prediction Barrier without a guaranteed RSB flush
// "erms" Enhanced REP MOVSB/STOSB instructions
// "clflushopt" CLFLUSHOPT instruction
// "clwb" CLWB instruction
// IA32_TSC_DEADLINE and X2APIC MSRs need fencing
// Non-serializing WRMSR
// "wbnoinvd" WBNOINVD instruction
// "la57" 5-level page tables
// "rdpid" RDPID instruction
// "serialize" SERIALIZE instruction
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
// Try again
// Out of memory
// Device or resource busy
// Invalid argument
// IS_ERR_VALUE - Detect an error pointer.
// @x: The pointer to check.
//
// Like IS_ERR(), but does not generate a compiler warning if result is unused.
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
// Operation is not supported
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
// For trivial one-depth nesting of a lock-class, the following
// global define can be used. (Subsystems with multiple levels
// of nesting should define their own lock-nesting subclasses.)
// Writer mask
// Remapping rwlock architecture specific functions to the corresponding
// queued rwlock functions.
// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
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
// color index
// [7:0] C0:C1:C2:C3:C4:C5:C6:C7 1:1:1:1:1:1:1:1 eight pixels/byte
// [7:0] C0:C1:C2:C3 2:2:2:2 four pixels/byte
// [7:0] C0:C1 4:4 two pixels/byte
// [7:0] C
// 1 bpp Darkness (inverse relationship between channel value and brightness)
// [7:0] D0:D1:D2:D3:D4:D5:D6:D7 1:1:1:1:1:1:1:1 eight pixels/byte
// 2 bpp Darkness (inverse relationship between channel value and brightness)
// [7:0] D0:D1:D2:D3 2:2:2:2 four pixels/byte
// 4 bpp Darkness (inverse relationship between channel value and brightness)
// [7:0] D0:D1 4:4 two pixels/byte
// 8 bpp Darkness (inverse relationship between channel value and brightness)
// [7:0] D
// 1 bpp Red (direct relationship between channel value and brightness)
// [7:0] R0:R1:R2:R3:R4:R5:R6:R7 1:1:1:1:1:1:1:1 eight pixels/byte
// 2 bpp Red (direct relationship between channel value and brightness)
// [7:0] R0:R1:R2:R3 2:2:2:2 four pixels/byte
// 4 bpp Red (direct relationship between channel value and brightness)
// [7:0] R0:R1 4:4 two pixels/byte
// 8 bpp Red (direct relationship between channel value and brightness)
// [7:0] R
// 10 bpp Red (direct relationship between channel value and brightness)
// [15:0] x:R 6:10 little endian
// 12 bpp Red (direct relationship between channel value and brightness)
// [15:0] x:R 4:12 little endian
// 8 bpp RGB
// [7:0] R:G:B 3:3:2
// [7:0] B:G:R 2:3:3
// 16 bpp RGB
// [15:0] x:R:G:B 4:4:4:4 little endian
// [15:0] x:B:G:R 4:4:4:4 little endian
// [15:0] R:G:B:x 4:4:4:4 little endian
// [15:0] B:G:R:x 4:4:4:4 little endian
// [15:0] A:R:G:B 4:4:4:4 little endian
// [15:0] A:B:G:R 4:4:4:4 little endian
// [15:0] R:G:B:A 4:4:4:4 little endian
// [15:0] B:G:R:A 4:4:4:4 little endian
// [15:0] x:R:G:B 1:5:5:5 little endian
// [15:0] x:B:G:R 1:5:5:5 little endian
// [15:0] R:G:B:x 5:5:5:1 little endian
// [15:0] B:G:R:x 5:5:5:1 little endian
// [15:0] A:R:G:B 1:5:5:5 little endian
// [15:0] A:B:G:R 1:5:5:5 little endian
// [15:0] R:G:B:A 5:5:5:1 little endian
// [15:0] B:G:R:A 5:5:5:1 little endian
// [15:0] R:G:B 5:6:5 little endian
// [15:0] B:G:R 5:6:5 little endian
// 24 bpp RGB
// [23:0] R:G:B little endian
// [23:0] B:G:R little endian
// 32 bpp RGB
// [31:0] x:R:G:B 8:8:8:8 little endian
// [31:0] x:B:G:R 8:8:8:8 little endian
// [31:0] R:G:B:x 8:8:8:8 little endian
// [31:0] B:G:R:x 8:8:8:8 little endian
// [31:0] A:R:G:B 8:8:8:8 little endian
// [31:0] A:B:G:R 8:8:8:8 little endian
// [31:0] R:G:B:A 8:8:8:8 little endian
// [31:0] B:G:R:A 8:8:8:8 little endian
// [31:0] x:R:G:B 2:10:10:10 little endian
// [31:0] x:B:G:R 2:10:10:10 little endian
// [31:0] R:G:B:x 10:10:10:2 little endian
// [31:0] B:G:R:x 10:10:10:2 little endian
// [31:0] A:R:G:B 2:10:10:10 little endian
// [31:0] A:B:G:R 2:10:10:10 little endian
// [31:0] R:G:B:A 10:10:10:2 little endian
// [31:0] B:G:R:A 10:10:10:2 little endian
// 48 bpp RGB
// [47:0] R:G:B 16:16:16 little endian
// [47:0] B:G:R 16:16:16 little endian
// 64 bpp RGB
// [63:0] x:R:G:B 16:16:16:16 little endian
// [63:0] x:B:G:R 16:16:16:16 little endian
// [63:0] A:R:G:B 16:16:16:16 little endian
// [63:0] A:B:G:R 16:16:16:16 little endian
// Half-Floating point - 16b/component
// IEEE 754-2008 binary16 half-precision float
// [15:0] sign:exponent:mantissa 1:5:10
// [63:0] x:R:G:B 16:16:16:16 little endian
// [63:0] x:B:G:R 16:16:16:16 little endian
// [63:0] A:R:G:B 16:16:16:16 little endian
// [63:0] A:B:G:R 16:16:16:16 little endian
// RGBA format with 10-bit components packed in 64-bit per pixel, with 6 bits
// of unused padding per component:
// [63:0] A:x:B:x:G:x:R:x 10:6:10:6:10:6:10:6 little endian
// packed YCbCr
// [31:0] Cr0:Y1:Cb0:Y0 8:8:8:8 little endian
// [31:0] Cb0:Y1:Cr0:Y0 8:8:8:8 little endian
// [31:0] Y1:Cr0:Y0:Cb0 8:8:8:8 little endian
// [31:0] Y1:Cb0:Y0:Cr0 8:8:8:8 little endian
// [31:0] A:Y:Cb:Cr 8:8:8:8 little endian
// [31:0] X:Y:Cb:Cr 8:8:8:8 little endian
// [23:0] Cr:Cb:Y 8:8:8 little endian
// Y followed by U then V, 10:10:10. Non-linear modifier only
// [31:0] x:Cr:Cb:Y 2:10:10:10 little endian
// packed Y2xx indicate for each component, xx valid data occupy msb
// 16-xx padding occupy lsb
// [63:0] Cr0:0:Y1:0:Cb0:0:Y0:0 10:6:10:6:10:6:10:6 little endian per 2 Y pixels
// [63:0] Cr0:0:Y1:0:Cb0:0:Y0:0 12:4:12:4:12:4:12:4 little endian per 2 Y pixels
// [63:0] Cr0:Y1:Cb0:Y0 16:16:16:16 little endian per 2 Y pixels
// packed Y4xx indicate for each component, xx valid data occupy msb
// 16-xx padding occupy lsb except Y410
// [31:0] A:Cr:Y:Cb 2:10:10:10 little endian
// [63:0] A:0:Cr:0:Y:0:Cb:0 12:4:12:4:12:4:12:4 little endian
// [63:0] A:Cr:Y:Cb 16:16:16:16 little endian
// [31:0] X:Cr:Y:Cb 2:10:10:10 little endian
// [63:0] X:0:Cr:0:Y:0:Cb:0 12:4:12:4:12:4:12:4 little endian
// [63:0] X:Cr:Y:Cb 16:16:16:16 little endian
// packed YCbCr420 2x2 tiled formats
// first 64 bits will contain Y,Cb,Cr components for a 2x2 tile
//
// [63:0]   A3:A2:Y3:0:Cr0:0:Y2:0:A1:A0:Y1:0:Cb0:0:Y0:0  1:1:8:2:8:2:8:2:1:1:8:2:8:2:8:2 little endian
// [63:0]   X3:X2:Y3:0:Cr0:0:Y2:0:X1:X0:Y1:0:Cb0:0:Y0:0  1:1:8:2:8:2:8:2:1:1:8:2:8:2:8:2 little endian
// [63:0]   A3:A2:Y3:Cr0:Y2:A1:A0:Y1:Cb0:Y0  1:1:10:10:10:1:1:10:10:10 little endian
// [63:0]   X3:X2:Y3:Cr0:Y2:X1:X0:Y1:Cb0:Y0  1:1:10:10:10:1:1:10:10:10 little endian
// 1-plane YUV 4:2:0
// In these formats, the component ordering is specified (Y, followed by U
// then V), but the exact Linear layout is undefined.
// These formats can only be used with a non-Linear modifier.
// 2 plane RGB + A
// index 0 = RGB plane, same format as the corresponding non _A8 format has
// index 1 = A plane, [7:0] A
// 2 plane YCbCr
// index 0 = Y plane, [7:0] Y
// index 1 = Cr:Cb plane, [15:0] Cr:Cb little endian
// or
// index 1 = Cb:Cr plane, [15:0] Cb:Cr little endian
// 2x2 subsampled Cr:Cb plane
// 2x2 subsampled Cb:Cr plane
// 2x1 subsampled Cr:Cb plane
// 2x1 subsampled Cb:Cr plane
// non-subsampled Cr:Cb plane
// non-subsampled Cb:Cr plane
// 2 plane YCbCr
// index 0 = Y plane, [39:0] Y3:Y2:Y1:Y0 little endian
// index 1 = Cr:Cb plane, [39:0] Cr1:Cb1:Cr0:Cb0 little endian
// 2x2 subsampled Cr:Cb plane
// 2x1 subsampled Cr:Cb plane
// non-subsampled Cr:Cb plane
// 2 plane YCbCr MSB aligned
// index 0 = Y plane, [15:0] Y:x [10:6] little endian
// index 1 = Cr:Cb plane, [31:0] Cr:x:Cb:x [10:6:10:6] little endian
// 2x1 subsampled Cr:Cb plane, 10 bit per channel
// 2 plane YCbCr MSB aligned
// index 0 = Y plane, [15:0] Y:x [10:6] little endian
// index 1 = Cr:Cb plane, [31:0] Cr:x:Cb:x [10:6:10:6] little endian
// 2x2 subsampled Cr:Cb plane 10 bits per channel
// 2 plane YCbCr MSB aligned
// index 0 = Y plane, [15:0] Y:x [12:4] little endian
// index 1 = Cr:Cb plane, [31:0] Cr:x:Cb:x [12:4:12:4] little endian
// 2x2 subsampled Cr:Cb plane 12 bits per channel
// 2 plane YCbCr MSB aligned
// index 0 = Y plane, [15:0] Y little endian
// index 1 = Cr:Cb plane, [31:0] Cr:Cb [16:16] little endian
// 2x2 subsampled Cr:Cb plane 16 bits per channel
// 2 plane YCbCr420.
// 3 10 bit components and 2 padding bits packed into 4 bytes.
// index 0 = Y plane, [31:0] x:Y2:Y1:Y0 2:10:10:10 little endian
// index 1 = Cr:Cb plane, [63:0] x:Cr2:Cb2:Cr1:x:Cb1:Cr0:Cb0 [2:10:10:10:2:10:10:10] little endian
// 2x2 subsampled Cr:Cb plane 10 bits per channel packed
// 2 plane YCbCr422.
// 3 10 bit components and 2 padding bits packed into 4 bytes.
// index 0 = Y plane, [31:0] x:Y2:Y1:Y0 2:10:10:10 little endian
// index 1 = Cr:Cb plane, [63:0] x:Cr2:Cb2:Cr1:x:Cb1:Cr0:Cb0 [2:10:10:10:2:10:10:10] little endian
// 2x1 subsampled Cr:Cb plane 10 bits per channel packed
// 3 plane non-subsampled (444) YCbCr
// 16 bits per component, but only 10 bits are used and 6 bits are padded
// index 0: Y plane, [15:0] Y:x [10:6] little endian
// index 1: Cb plane, [15:0] Cb:x [10:6] little endian
// index 2: Cr plane, [15:0] Cr:x [10:6] little endian
// 3 plane non-subsampled (444) YCrCb
// 16 bits per component, but only 10 bits are used and 6 bits are padded
// index 0: Y plane, [15:0] Y:x [10:6] little endian
// index 1: Cr plane, [15:0] Cr:x [10:6] little endian
// index 2: Cb plane, [15:0] Cb:x [10:6] little endian
// 3 plane non-subsampled (444) YCbCr LSB aligned
// 10 bpc, 30 bits per sample image data in a single contiguous buffer.
// index 0: Y plane,  [31:0] x:Y2:Y1:Y0    [2:10:10:10] little endian
// index 1: Cb plane, [31:0] x:Cb2:Cb1:Cb0 [2:10:10:10] little endian
// index 2: Cr plane, [31:0] x:Cr2:Cr1:Cr0 [2:10:10:10] little endian
// 3 plane YCbCr LSB aligned
// In order to use these formats in a similar fashion to MSB aligned ones
// implementation can multiply the values by 2^6=64. For that reason the padding
// must only contain zeros.
// index 0 = Y plane, [15:0] z:Y [6:10] little endian
// index 1 = Cb plane, [15:0] z:Cb [6:10] little endian
// index 2 = Cr plane, [15:0] z:Cr [6:10] little endian
// 2x2 subsampled Cb (1) and Cr (2) planes 10 bits per channel
// 2x1 subsampled Cb (1) and Cr (2) planes 10 bits per channel
// non-subsampled Cb (1) and Cr (2) planes 10 bits per channel
// 3 plane YCbCr LSB aligned
// In order to use these formats in a similar fashion to MSB aligned ones
// implementation can multiply the values by 2^4=16. For that reason the padding
// must only contain zeros.
// index 0 = Y plane, [15:0] z:Y [4:12] little endian
// index 1 = Cb plane, [15:0] z:Cb [4:12] little endian
// index 2 = Cr plane, [15:0] z:Cr [4:12] little endian
// 2x2 subsampled Cb (1) and Cr (2) planes 12 bits per channel
// 2x1 subsampled Cb (1) and Cr (2) planes 12 bits per channel
// non-subsampled Cb (1) and Cr (2) planes 12 bits per channel
// 3 plane YCbCr
// index 0 = Y plane, [15:0] Y little endian
// index 1 = Cb plane, [15:0] Cb little endian
// index 2 = Cr plane, [15:0] Cr little endian
// 2x2 subsampled Cb (1) and Cr (2) planes 16 bits per channel
// 2x1 subsampled Cb (1) and Cr (2) planes 16 bits per channel
// non-subsampled Cb (1) and Cr (2) planes 16 bits per channel
// 3 plane YCbCr
// index 0: Y plane, [7:0] Y
// index 1: Cb plane, [7:0] Cb
// index 2: Cr plane, [7:0] Cr
// or
// index 1: Cr plane, [7:0] Cr
// index 2: Cb plane, [7:0] Cb
// 4x4 subsampled Cb (1) and Cr (2) planes
// 4x4 subsampled Cr (1) and Cb (2) planes
// 4x1 subsampled Cb (1) and Cr (2) planes
// 4x1 subsampled Cr (1) and Cb (2) planes
// 2x2 subsampled Cb (1) and Cr (2) planes
// 2x2 subsampled Cr (1) and Cb (2) planes
// 2x1 subsampled Cb (1) and Cr (2) planes
// 2x1 subsampled Cr (1) and Cb (2) planes
// non-subsampled Cb (1) and Cr (2) planes
// non-subsampled Cr (1) and Cb (2) planes
// 8-bit Y-only
// [31:0] x:Y2:Y1:Y0 2:10:10:10 little endian
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
macro_rules! ARRAY_SIZE__12 {
() => {
((core::mem::size_of::<[drm_format_info; 134]>() as u64) / (core::mem::size_of::<drm_format_info>() as u64)).wrapping_add(__must_be_array__19!() as u64)
};
}

macro_rules! BUGFLAG_TAINT {
($taint:expr) => {
($taint) << 8i32
};
}

macro_rules! DIV_ROUND_DOWN_ULL__2 {
($ll:expr, $d:expr) => {
{
let mut __v: u64;
let mut __v2: u64 = $ll;
do_div__2!(__v2, $d);
__v = __v2;
__v
}
};
}

macro_rules! DIV_ROUND_UP_ULL {
($ll:expr, $d:expr) => {
DIV_ROUND_DOWN_ULL__2!((($ll) as u64).wrapping_add(($d) as u64).wrapping_sub(1), $d)
};
}

macro_rules! __must_be_array__19 {
() => {
core::mem::size_of::<__SlateRecord_drm_fourcc_c_399>() as i32
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
macro_rules! do_div__2 {
($n:expr, $base:expr) => {
let mut __v: u32;
let __v2: u32 = $base;
let __v3: u32 = ((($n) as u64) % (__v2 as u64)) as u32;
$n = (($n) as u64) / (__v2 as u64);
__v = __v3;
};
}

macro_rules! fourcc_code {
($a:expr, $b:expr, $c:expr, $d:expr) => {
(($a) as u32) | (($b) as u32) << 8i32 | (($c) as u32) << 16i32 | (($d) as u32) << 24i32
};
}

macro_rules! unlikely {
($x:expr) => {
(($x) != 0) as i64
};
}

unsafe extern "C" {
    fn _printk(arg0: *const u8, ...) -> i32;
    fn drm_modeset_lock(arg0: *mut drm_modeset_lock, arg1: *mut drm_modeset_acquire_ctx) -> i32;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
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


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits58U0,
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
    __slate_bits_0: __slate_bits::__SlateBits65U0,
    pkru: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct file {
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
struct device {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __kernel_timespec {
    tv_sec: i64,
    tv_nsec: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_bitmap {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
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


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct spinlock {
    __slate_anon_0: __SlateRecord_spinlock_types_h_18,
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
struct pollfd {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct optimistic_spin_queue {
    tail: __SlateRecord_types_h_190,
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
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
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
struct futex_private_hash {
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
struct sem_undo_list {
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
struct hrtimer_clock_base {
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
struct seccomp_filter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sigpending {
    list: list_head,
    signal: __SlateRecord_signal_h_25,
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
struct tlbflush_unmap_batch {
    arch: arch_tlbflush_unmap_batch,
    flush_required: bool,
    writable: bool,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct workqueue_struct {
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct mm_cid_pcpu {
    cid: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
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
struct seqcount {
    sequence: u32,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
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
    __slate_bits_0: __slate_bits::__SlateBits203U0,
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
    __slate_bits_0: __slate_bits::__SlateBits48U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits48U1,
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
    __slate_bits_2: __slate_bits::__SlateBits48U2,
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
struct rq_flags {
}


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
#[repr(C)]
#[derive(Clone, Copy)]
struct sched_cache_time {
    runtime: u64,
    epoch: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
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
struct swait_queue_head {
    lock: raw_spinlock,
    task_list: list_head,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct completion {
    done: u32,
    wait: swait_queue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gendisk {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct old_timespec32 {
    tv_sec: i32,
    tv_nsec: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_struct {
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
struct srcu_ctr {
    srcu_locks: __SlateRecord_types_h_197,
    srcu_unlocks: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
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
struct rwsem_waiter {
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
struct uprobe {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct inode {
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
struct uprobe_xol_ops {
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
struct percpu_counter {
    lock: raw_spinlock,
    count: i64,
    list: list_head,
    counters: *mut i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
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


// Opaque type representing current VMA (vm_area_struct) flag state. Must be
// accessed via vma_flags_xxx() helper functions.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_871 {
    __vma_flags: [u64; 1],
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
struct xarray {
    xa_lock: spinlock,
    xa_flags: u32,
    xa_head: *mut (),
}


// We encode errnos in the xas->xa_node.  If an error has happened, we need to
// drop the lock to fix it, and once we've done so the xa_state is invalid.
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
struct drm_file {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct drm_atomic_commit {
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
struct drm_display_mode {
}


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
struct drm_framebuffer {
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
struct drm_property {
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
struct __SlateRecord_drm_device_h_113 {
    resources: list_head,
    final_kfree: *mut (),
    lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dentry {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_drm_fourcc_h_77 {
    cpp: [u8; 4],
    char_per_block: [u8; 4],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_drm_fourcc_c_399 {
}


#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

mod __slate_bits {
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits203U0 {
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
    pub struct __SlateBits48U0 {
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
    pub struct __SlateBits48U1 {
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
    pub struct __SlateBits48U2 {
        #[bits(1)]
        pub mce_ripv: u64,
        #[bits(1)]
        pub mce_whole_page: u64,
        #[bits(62)]
        pub __mce_reserved: u64,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits58U0 {
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
    pub struct __SlateBits65U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_fourcc.drm_format_info")]
extern "C" fn drm_format_info(format: u32) -> *const drm_format_info {
let info: *const drm_format_info = __drm_format_info(format);
let mut __v0: i64;
let __ret_warn_on: i32 = info.is_null() as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("230: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 230b - ., 3; .popsection\n# {0}", const 230 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_drm_format_info_1, sym __slate_asm_string_drm_format_info_2, const 423 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("231: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 231b - ., 4; .popsection\n# {0}", const 231 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
info
}

/// drm_mode_legacy_fb_format - compute drm fourcc code from legacy description
/// @bpp: bits per pixels
/// @depth: bit depth per pixel
///
/// Computes a drm fourcc pixel format code for the given @bpp/@depth values.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_fourcc.drm_mode_legacy_fb_format")]
extern "C" fn drm_mode_legacy_fb_format(bpp: u32, depth: u32) -> u32 {
let mut fmt: u32 = 0;
match bpp {
1 => {
if depth == 1 {
fmt = fourcc_code!(67i32, 49i32, 32i32, 32i32);
}
}
2 => {
if depth == 2 {
fmt = fourcc_code!(67i32, 50i32, 32i32, 32i32);
}
}
4 => {
if depth == 4 {
fmt = fourcc_code!(67i32, 52i32, 32i32, 32i32);
}
}
8 => {
if depth == 8 {
fmt = fourcc_code!(67i32, 56i32, 32i32, 32i32);
}
}
16 => {
match depth {
15 => {
fmt = fourcc_code!(88i32, 82i32, 49i32, 53i32);
}
16 => {
fmt = fourcc_code!(82i32, 71i32, 49i32, 54i32);
}
_ => {
}
}
}
24 => {
if depth == 24 {
fmt = fourcc_code!(82i32, 71i32, 50i32, 52i32);
}
}
32 => {
match depth {
24 => {
fmt = fourcc_code!(88i32, 82i32, 50i32, 52i32);
}
30 => {
fmt = fourcc_code!(88i32, 82i32, 51i32, 48i32);
}
32 => {
fmt = fourcc_code!(65i32, 82i32, 50i32, 52i32);
}
_ => {
}
}
}
_ => {
}
}
fmt
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_mode_legacy_fb_format_227: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u32, u32) -> u32>, *mut ()>(Some(drm_mode_legacy_fb_format)) };

/// drm_driver_legacy_fb_format - compute drm fourcc code from legacy description
/// @dev: DRM device
/// @bpp: bits per pixels
/// @depth: bit depth per pixel
///
/// Computes a drm fourcc pixel format code for the given @bpp/@depth values.
/// Unlike drm_mode_legacy_fb_format() this looks at the drivers mode_config,
/// and depending on the &drm_mode_config.quirk_addfb_prefer_host_byte_order flag
/// it returns little endian byte order or host byte order framebuffer formats.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_fourcc.drm_driver_legacy_fb_format")]
extern "C" fn drm_driver_legacy_fb_format(dev: *mut drm_device, bpp: u32, depth: u32) -> u32 {
let mut fmt: u32 = drm_mode_legacy_fb_format(bpp, depth);
if unsafe { (*dev).mode_config.quirk_addfb_prefer_host_byte_order } {
if fmt == fourcc_code!(88i32, 82i32, 50i32, 52i32) {
fmt = fourcc_code!(88i32, 82i32, 50i32, 52i32);
}
if fmt == fourcc_code!(65i32, 82i32, 50i32, 52i32) {
fmt = fourcc_code!(65i32, 82i32, 50i32, 52i32);
}
if fmt == fourcc_code!(82i32, 71i32, 49i32, 54i32) {
fmt = fourcc_code!(82i32, 71i32, 49i32, 54i32);
}
if fmt == fourcc_code!(88i32, 82i32, 49i32, 53i32) {
fmt = fourcc_code!(88i32, 82i32, 49i32, 53i32);
}
}
if (unsafe { (*dev).mode_config.quirk_addfb_prefer_xbgr_30bpp }) && fmt == fourcc_code!(88i32, 82i32, 51i32, 48i32) {
fmt = fourcc_code!(88i32, 66i32, 51i32, 48i32);
}
fmt
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_driver_legacy_fb_format_228: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, u32, u32) -> u32>, *mut ()>(Some(drm_driver_legacy_fb_format)) };

/// drm_driver_color_mode_format - Compute DRM 4CC code from color mode
/// @dev: DRM device
/// @color_mode: command-line color mode
///
/// Computes a DRM 4CC pixel format code for the given color mode using
/// drm_driver_color_mode(). The color mode is in the format used and the
/// kernel command line. It specifies the number of bits per pixel
/// and color depth in a single value.
///
/// Useful in fbdev emulation code, since that deals in those values. The
/// helper does not consider YUV or other complicated formats. This means
/// only legacy formats are supported (fmt->depth is a legacy field), but
/// the framebuffer emulation can only deal with such formats, specifically
/// RGB/BGA formats.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_fourcc.drm_driver_color_mode_format")]
extern "C" fn drm_driver_color_mode_format(dev: *mut drm_device, color_mode: u32) -> u32 {
match color_mode {
15 => {
return drm_driver_legacy_fb_format(dev, 16u32, 15u32);
}
32 => {
return drm_driver_legacy_fb_format(dev, 32u32, 24u32);
}
_ => {
return drm_driver_legacy_fb_format(dev, color_mode, color_mode);
}
}
unsafe { core::mem::zeroed() }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_driver_color_mode_format_229: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, u32) -> u32>, *mut ()>(Some(drm_driver_color_mode_format)) };

/// Internal function to query information for a given format. See
/// drm_format_info() for the public API.
#[unsafe(no_mangle)]
extern "C" fn __drm_format_info(format: u32) -> *const drm_format_info {
let mut i: u32 = 0;
while (i as u64) < ARRAY_SIZE__12!() {
if (unsafe { formats.0[i as usize].format }) == format {
return unsafe { core::ptr::addr_of!(formats.0) as *const drm_format_info }.wrapping_offset(i as isize);
}
i = i.wrapping_add(1);
}
core::ptr::null::<drm_format_info>()
}

static formats: __SlateAlign16<[drm_format_info; 134]> = __SlateAlign16([{
let mut __t0: drm_format_info = unsafe { core::mem::zeroed() };
__t0.format = 538980675u32;
__t0.depth = 1u8;
__t0.num_planes = 1u8;
__t0.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { char_per_block: {
let mut __t1: [u8; 4] = unsafe { core::mem::zeroed() };
__t1[0] = 1u8;
__t1
} };
__t0.block_w = {
let mut __t2: [u8; 4] = unsafe { core::mem::zeroed() };
__t2[0] = 8u8;
__t2
};
__t0.block_h = {
let mut __t3: [u8; 4] = unsafe { core::mem::zeroed() };
__t3[0] = 1u8;
__t3
};
__t0.hsub = 1u8;
__t0.vsub = 1u8;
__t0.is_color_indexed = 1i32 != 0i32;
__t0
}, {
let mut __t4: drm_format_info = unsafe { core::mem::zeroed() };
__t4.format = 538980931u32;
__t4.depth = 2u8;
__t4.num_planes = 1u8;
__t4.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { char_per_block: {
let mut __t5: [u8; 4] = unsafe { core::mem::zeroed() };
__t5[0] = 1u8;
__t5
} };
__t4.block_w = {
let mut __t6: [u8; 4] = unsafe { core::mem::zeroed() };
__t6[0] = 4u8;
__t6
};
__t4.block_h = {
let mut __t7: [u8; 4] = unsafe { core::mem::zeroed() };
__t7[0] = 1u8;
__t7
};
__t4.hsub = 1u8;
__t4.vsub = 1u8;
__t4.is_color_indexed = 1i32 != 0i32;
__t4
}, {
let mut __t8: drm_format_info = unsafe { core::mem::zeroed() };
__t8.format = 538981443u32;
__t8.depth = 4u8;
__t8.num_planes = 1u8;
__t8.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { char_per_block: {
let mut __t9: [u8; 4] = unsafe { core::mem::zeroed() };
__t9[0] = 1u8;
__t9
} };
__t8.block_w = {
let mut __t10: [u8; 4] = unsafe { core::mem::zeroed() };
__t10[0] = 2u8;
__t10
};
__t8.block_h = {
let mut __t11: [u8; 4] = unsafe { core::mem::zeroed() };
__t11[0] = 1u8;
__t11
};
__t8.hsub = 1u8;
__t8.vsub = 1u8;
__t8.is_color_indexed = 1i32 != 0i32;
__t8
}, {
let mut __t12: drm_format_info = unsafe { core::mem::zeroed() };
__t12.format = 538982467u32;
__t12.depth = 8u8;
__t12.num_planes = 1u8;
__t12.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t13: [u8; 4] = unsafe { core::mem::zeroed() };
__t13[0] = 1u8;
__t13[1] = 0u8;
__t13[2] = 0u8;
__t13
} };
__t12.hsub = 1u8;
__t12.vsub = 1u8;
__t12.is_color_indexed = 1i32 != 0i32;
__t12
}, {
let mut __t14: drm_format_info = unsafe { core::mem::zeroed() };
__t14.format = 538980676u32;
__t14.depth = 1u8;
__t14.num_planes = 1u8;
__t14.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { char_per_block: {
let mut __t15: [u8; 4] = unsafe { core::mem::zeroed() };
__t15[0] = 1u8;
__t15
} };
__t14.block_w = {
let mut __t16: [u8; 4] = unsafe { core::mem::zeroed() };
__t16[0] = 8u8;
__t16
};
__t14.block_h = {
let mut __t17: [u8; 4] = unsafe { core::mem::zeroed() };
__t17[0] = 1u8;
__t17
};
__t14.hsub = 1u8;
__t14.vsub = 1u8;
__t14
}, {
let mut __t18: drm_format_info = unsafe { core::mem::zeroed() };
__t18.format = 538980932u32;
__t18.depth = 2u8;
__t18.num_planes = 1u8;
__t18.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { char_per_block: {
let mut __t19: [u8; 4] = unsafe { core::mem::zeroed() };
__t19[0] = 1u8;
__t19
} };
__t18.block_w = {
let mut __t20: [u8; 4] = unsafe { core::mem::zeroed() };
__t20[0] = 4u8;
__t20
};
__t18.block_h = {
let mut __t21: [u8; 4] = unsafe { core::mem::zeroed() };
__t21[0] = 1u8;
__t21
};
__t18.hsub = 1u8;
__t18.vsub = 1u8;
__t18
}, {
let mut __t22: drm_format_info = unsafe { core::mem::zeroed() };
__t22.format = 538981444u32;
__t22.depth = 4u8;
__t22.num_planes = 1u8;
__t22.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { char_per_block: {
let mut __t23: [u8; 4] = unsafe { core::mem::zeroed() };
__t23[0] = 1u8;
__t23
} };
__t22.block_w = {
let mut __t24: [u8; 4] = unsafe { core::mem::zeroed() };
__t24[0] = 2u8;
__t24
};
__t22.block_h = {
let mut __t25: [u8; 4] = unsafe { core::mem::zeroed() };
__t25[0] = 1u8;
__t25
};
__t22.hsub = 1u8;
__t22.vsub = 1u8;
__t22
}, {
let mut __t26: drm_format_info = unsafe { core::mem::zeroed() };
__t26.format = 538982468u32;
__t26.depth = 8u8;
__t26.num_planes = 1u8;
__t26.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t27: [u8; 4] = unsafe { core::mem::zeroed() };
__t27[0] = 1u8;
__t27[1] = 0u8;
__t27[2] = 0u8;
__t27
} };
__t26.hsub = 1u8;
__t26.vsub = 1u8;
__t26
}, {
let mut __t28: drm_format_info = unsafe { core::mem::zeroed() };
__t28.format = 538980690u32;
__t28.depth = 1u8;
__t28.num_planes = 1u8;
__t28.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { char_per_block: {
let mut __t29: [u8; 4] = unsafe { core::mem::zeroed() };
__t29[0] = 1u8;
__t29
} };
__t28.block_w = {
let mut __t30: [u8; 4] = unsafe { core::mem::zeroed() };
__t30[0] = 8u8;
__t30
};
__t28.block_h = {
let mut __t31: [u8; 4] = unsafe { core::mem::zeroed() };
__t31[0] = 1u8;
__t31
};
__t28.hsub = 1u8;
__t28.vsub = 1u8;
__t28
}, {
let mut __t32: drm_format_info = unsafe { core::mem::zeroed() };
__t32.format = 538980946u32;
__t32.depth = 2u8;
__t32.num_planes = 1u8;
__t32.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { char_per_block: {
let mut __t33: [u8; 4] = unsafe { core::mem::zeroed() };
__t33[0] = 1u8;
__t33
} };
__t32.block_w = {
let mut __t34: [u8; 4] = unsafe { core::mem::zeroed() };
__t34[0] = 4u8;
__t34
};
__t32.block_h = {
let mut __t35: [u8; 4] = unsafe { core::mem::zeroed() };
__t35[0] = 1u8;
__t35
};
__t32.hsub = 1u8;
__t32.vsub = 1u8;
__t32
}, {
let mut __t36: drm_format_info = unsafe { core::mem::zeroed() };
__t36.format = 538981458u32;
__t36.depth = 4u8;
__t36.num_planes = 1u8;
__t36.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { char_per_block: {
let mut __t37: [u8; 4] = unsafe { core::mem::zeroed() };
__t37[0] = 1u8;
__t37
} };
__t36.block_w = {
let mut __t38: [u8; 4] = unsafe { core::mem::zeroed() };
__t38[0] = 2u8;
__t38
};
__t36.block_h = {
let mut __t39: [u8; 4] = unsafe { core::mem::zeroed() };
__t39[0] = 1u8;
__t39
};
__t36.hsub = 1u8;
__t36.vsub = 1u8;
__t36
}, {
let mut __t40: drm_format_info = unsafe { core::mem::zeroed() };
__t40.format = 538982482u32;
__t40.depth = 8u8;
__t40.num_planes = 1u8;
__t40.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t41: [u8; 4] = unsafe { core::mem::zeroed() };
__t41[0] = 1u8;
__t41[1] = 0u8;
__t41[2] = 0u8;
__t41
} };
__t40.hsub = 1u8;
__t40.vsub = 1u8;
__t40
}, {
let mut __t42: drm_format_info = unsafe { core::mem::zeroed() };
__t42.format = 540029266u32;
__t42.depth = 10u8;
__t42.num_planes = 1u8;
__t42.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t43: [u8; 4] = unsafe { core::mem::zeroed() };
__t43[0] = 2u8;
__t43[1] = 0u8;
__t43[2] = 0u8;
__t43
} };
__t42.hsub = 1u8;
__t42.vsub = 1u8;
__t42
}, {
let mut __t44: drm_format_info = unsafe { core::mem::zeroed() };
__t44.format = 540160338u32;
__t44.depth = 12u8;
__t44.num_planes = 1u8;
__t44.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t45: [u8; 4] = unsafe { core::mem::zeroed() };
__t45[0] = 2u8;
__t45[1] = 0u8;
__t45[2] = 0u8;
__t45
} };
__t44.hsub = 1u8;
__t44.vsub = 1u8;
__t44
}, {
let mut __t46: drm_format_info = unsafe { core::mem::zeroed() };
__t46.format = 943867730u32;
__t46.depth = 8u8;
__t46.num_planes = 1u8;
__t46.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t47: [u8; 4] = unsafe { core::mem::zeroed() };
__t47[0] = 1u8;
__t47[1] = 0u8;
__t47[2] = 0u8;
__t47
} };
__t46.hsub = 1u8;
__t46.vsub = 1u8;
__t46
}, {
let mut __t48: drm_format_info = unsafe { core::mem::zeroed() };
__t48.format = 944916290u32;
__t48.depth = 8u8;
__t48.num_planes = 1u8;
__t48.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t49: [u8; 4] = unsafe { core::mem::zeroed() };
__t49[0] = 1u8;
__t49[1] = 0u8;
__t49[2] = 0u8;
__t49
} };
__t48.hsub = 1u8;
__t48.vsub = 1u8;
__t48
}, {
let mut __t50: drm_format_info = unsafe { core::mem::zeroed() };
__t50.format = 842093144u32;
__t50.depth = 0u8;
__t50.num_planes = 1u8;
__t50.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t51: [u8; 4] = unsafe { core::mem::zeroed() };
__t51[0] = 2u8;
__t51[1] = 0u8;
__t51[2] = 0u8;
__t51
} };
__t50.hsub = 1u8;
__t50.vsub = 1u8;
__t50
}, {
let mut __t52: drm_format_info = unsafe { core::mem::zeroed() };
__t52.format = 842089048u32;
__t52.depth = 0u8;
__t52.num_planes = 1u8;
__t52.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t53: [u8; 4] = unsafe { core::mem::zeroed() };
__t53[0] = 2u8;
__t53[1] = 0u8;
__t53[2] = 0u8;
__t53
} };
__t52.hsub = 1u8;
__t52.vsub = 1u8;
__t52
}, {
let mut __t54: drm_format_info = unsafe { core::mem::zeroed() };
__t54.format = 842094674u32;
__t54.depth = 0u8;
__t54.num_planes = 1u8;
__t54.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t55: [u8; 4] = unsafe { core::mem::zeroed() };
__t55[0] = 2u8;
__t55[1] = 0u8;
__t55[2] = 0u8;
__t55
} };
__t54.hsub = 1u8;
__t54.vsub = 1u8;
__t54
}, {
let mut __t56: drm_format_info = unsafe { core::mem::zeroed() };
__t56.format = 842094658u32;
__t56.depth = 0u8;
__t56.num_planes = 1u8;
__t56.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t57: [u8; 4] = unsafe { core::mem::zeroed() };
__t57[0] = 2u8;
__t57[1] = 0u8;
__t57[2] = 0u8;
__t57
} };
__t56.hsub = 1u8;
__t56.vsub = 1u8;
__t56
}, {
let mut __t58: drm_format_info = unsafe { core::mem::zeroed() };
__t58.format = 842093121u32;
__t58.depth = 0u8;
__t58.num_planes = 1u8;
__t58.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t59: [u8; 4] = unsafe { core::mem::zeroed() };
__t59[0] = 2u8;
__t59[1] = 0u8;
__t59[2] = 0u8;
__t59
} };
__t58.hsub = 1u8;
__t58.vsub = 1u8;
__t58.has_alpha = 1i32 != 0i32;
__t58
}, {
let mut __t60: drm_format_info = unsafe { core::mem::zeroed() };
__t60.format = 842089025u32;
__t60.depth = 0u8;
__t60.num_planes = 1u8;
__t60.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t61: [u8; 4] = unsafe { core::mem::zeroed() };
__t61[0] = 2u8;
__t61[1] = 0u8;
__t61[2] = 0u8;
__t61
} };
__t60.hsub = 1u8;
__t60.vsub = 1u8;
__t60.has_alpha = 1i32 != 0i32;
__t60
}, {
let mut __t62: drm_format_info = unsafe { core::mem::zeroed() };
__t62.format = 842088786u32;
__t62.depth = 0u8;
__t62.num_planes = 1u8;
__t62.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t63: [u8; 4] = unsafe { core::mem::zeroed() };
__t63[0] = 2u8;
__t63[1] = 0u8;
__t63[2] = 0u8;
__t63
} };
__t62.hsub = 1u8;
__t62.vsub = 1u8;
__t62.has_alpha = 1i32 != 0i32;
__t62
}, {
let mut __t64: drm_format_info = unsafe { core::mem::zeroed() };
__t64.format = 842088770u32;
__t64.depth = 0u8;
__t64.num_planes = 1u8;
__t64.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t65: [u8; 4] = unsafe { core::mem::zeroed() };
__t65[0] = 2u8;
__t65[1] = 0u8;
__t65[2] = 0u8;
__t65
} };
__t64.hsub = 1u8;
__t64.vsub = 1u8;
__t64.has_alpha = 1i32 != 0i32;
__t64
}, {
let mut __t66: drm_format_info = unsafe { core::mem::zeroed() };
__t66.format = 892424792u32;
__t66.depth = 15u8;
__t66.num_planes = 1u8;
__t66.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t67: [u8; 4] = unsafe { core::mem::zeroed() };
__t67[0] = 2u8;
__t67[1] = 0u8;
__t67[2] = 0u8;
__t67
} };
__t66.hsub = 1u8;
__t66.vsub = 1u8;
__t66
}, {
let mut __t68: drm_format_info = unsafe { core::mem::zeroed() };
__t68.format = 892420696u32;
__t68.depth = 15u8;
__t68.num_planes = 1u8;
__t68.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t69: [u8; 4] = unsafe { core::mem::zeroed() };
__t69[0] = 2u8;
__t69[1] = 0u8;
__t69[2] = 0u8;
__t69
} };
__t68.hsub = 1u8;
__t68.vsub = 1u8;
__t68
}, {
let mut __t70: drm_format_info = unsafe { core::mem::zeroed() };
__t70.format = 892426322u32;
__t70.depth = 15u8;
__t70.num_planes = 1u8;
__t70.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t71: [u8; 4] = unsafe { core::mem::zeroed() };
__t71[0] = 2u8;
__t71[1] = 0u8;
__t71[2] = 0u8;
__t71
} };
__t70.hsub = 1u8;
__t70.vsub = 1u8;
__t70
}, {
let mut __t72: drm_format_info = unsafe { core::mem::zeroed() };
__t72.format = 892426306u32;
__t72.depth = 15u8;
__t72.num_planes = 1u8;
__t72.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t73: [u8; 4] = unsafe { core::mem::zeroed() };
__t73[0] = 2u8;
__t73[1] = 0u8;
__t73[2] = 0u8;
__t73
} };
__t72.hsub = 1u8;
__t72.vsub = 1u8;
__t72
}, {
let mut __t74: drm_format_info = unsafe { core::mem::zeroed() };
__t74.format = 892424769u32;
__t74.depth = 15u8;
__t74.num_planes = 1u8;
__t74.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t75: [u8; 4] = unsafe { core::mem::zeroed() };
__t75[0] = 2u8;
__t75[1] = 0u8;
__t75[2] = 0u8;
__t75
} };
__t74.hsub = 1u8;
__t74.vsub = 1u8;
__t74.has_alpha = 1i32 != 0i32;
__t74
}, {
let mut __t76: drm_format_info = unsafe { core::mem::zeroed() };
__t76.format = 892420673u32;
__t76.depth = 15u8;
__t76.num_planes = 1u8;
__t76.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t77: [u8; 4] = unsafe { core::mem::zeroed() };
__t77[0] = 2u8;
__t77[1] = 0u8;
__t77[2] = 0u8;
__t77
} };
__t76.hsub = 1u8;
__t76.vsub = 1u8;
__t76.has_alpha = 1i32 != 0i32;
__t76
}, {
let mut __t78: drm_format_info = unsafe { core::mem::zeroed() };
__t78.format = 892420434u32;
__t78.depth = 15u8;
__t78.num_planes = 1u8;
__t78.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t79: [u8; 4] = unsafe { core::mem::zeroed() };
__t79[0] = 2u8;
__t79[1] = 0u8;
__t79[2] = 0u8;
__t79
} };
__t78.hsub = 1u8;
__t78.vsub = 1u8;
__t78.has_alpha = 1i32 != 0i32;
__t78
}, {
let mut __t80: drm_format_info = unsafe { core::mem::zeroed() };
__t80.format = 892420418u32;
__t80.depth = 15u8;
__t80.num_planes = 1u8;
__t80.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t81: [u8; 4] = unsafe { core::mem::zeroed() };
__t81[0] = 2u8;
__t81[1] = 0u8;
__t81[2] = 0u8;
__t81
} };
__t80.hsub = 1u8;
__t80.vsub = 1u8;
__t80.has_alpha = 1i32 != 0i32;
__t80
}, {
let mut __t82: drm_format_info = unsafe { core::mem::zeroed() };
__t82.format = 909199186u32;
__t82.depth = 16u8;
__t82.num_planes = 1u8;
__t82.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t83: [u8; 4] = unsafe { core::mem::zeroed() };
__t83[0] = 2u8;
__t83[1] = 0u8;
__t83[2] = 0u8;
__t83
} };
__t82.hsub = 1u8;
__t82.vsub = 1u8;
__t82
}, {
let mut __t84: drm_format_info = unsafe { core::mem::zeroed() };
__t84.format = 909199170u32;
__t84.depth = 16u8;
__t84.num_planes = 1u8;
__t84.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t85: [u8; 4] = unsafe { core::mem::zeroed() };
__t85[0] = 2u8;
__t85[1] = 0u8;
__t85[2] = 0u8;
__t85
} };
__t84.hsub = 1u8;
__t84.vsub = 1u8;
__t84
}, {
let mut __t86: drm_format_info = unsafe { core::mem::zeroed() };
__t86.format = 875710290u32;
__t86.depth = 24u8;
__t86.num_planes = 1u8;
__t86.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t87: [u8; 4] = unsafe { core::mem::zeroed() };
__t87[0] = 3u8;
__t87[1] = 0u8;
__t87[2] = 0u8;
__t87
} };
__t86.hsub = 1u8;
__t86.vsub = 1u8;
__t86
}, {
let mut __t88: drm_format_info = unsafe { core::mem::zeroed() };
__t88.format = 875710274u32;
__t88.depth = 24u8;
__t88.num_planes = 1u8;
__t88.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t89: [u8; 4] = unsafe { core::mem::zeroed() };
__t89[0] = 3u8;
__t89[1] = 0u8;
__t89[2] = 0u8;
__t89
} };
__t88.hsub = 1u8;
__t88.vsub = 1u8;
__t88
}, {
let mut __t90: drm_format_info = unsafe { core::mem::zeroed() };
__t90.format = 875713112u32;
__t90.depth = 24u8;
__t90.num_planes = 1u8;
__t90.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t91: [u8; 4] = unsafe { core::mem::zeroed() };
__t91[0] = 4u8;
__t91[1] = 0u8;
__t91[2] = 0u8;
__t91
} };
__t90.hsub = 1u8;
__t90.vsub = 1u8;
__t90
}, {
let mut __t92: drm_format_info = unsafe { core::mem::zeroed() };
__t92.format = 875709016u32;
__t92.depth = 24u8;
__t92.num_planes = 1u8;
__t92.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t93: [u8; 4] = unsafe { core::mem::zeroed() };
__t93[0] = 4u8;
__t93[1] = 0u8;
__t93[2] = 0u8;
__t93
} };
__t92.hsub = 1u8;
__t92.vsub = 1u8;
__t92
}, {
let mut __t94: drm_format_info = unsafe { core::mem::zeroed() };
__t94.format = 875714642u32;
__t94.depth = 24u8;
__t94.num_planes = 1u8;
__t94.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t95: [u8; 4] = unsafe { core::mem::zeroed() };
__t95[0] = 4u8;
__t95[1] = 0u8;
__t95[2] = 0u8;
__t95
} };
__t94.hsub = 1u8;
__t94.vsub = 1u8;
__t94
}, {
let mut __t96: drm_format_info = unsafe { core::mem::zeroed() };
__t96.format = 875714626u32;
__t96.depth = 24u8;
__t96.num_planes = 1u8;
__t96.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t97: [u8; 4] = unsafe { core::mem::zeroed() };
__t97[0] = 4u8;
__t97[1] = 0u8;
__t97[2] = 0u8;
__t97
} };
__t96.hsub = 1u8;
__t96.vsub = 1u8;
__t96
}, {
let mut __t98: drm_format_info = unsafe { core::mem::zeroed() };
__t98.format = 943797586u32;
__t98.depth = 24u8;
__t98.num_planes = 2u8;
__t98.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t99: [u8; 4] = unsafe { core::mem::zeroed() };
__t99[0] = 2u8;
__t99[1] = 1u8;
__t99[2] = 0u8;
__t99
} };
__t98.hsub = 1u8;
__t98.vsub = 1u8;
__t98.has_alpha = 1i32 != 0i32;
__t98
}, {
let mut __t100: drm_format_info = unsafe { core::mem::zeroed() };
__t100.format = 943797570u32;
__t100.depth = 24u8;
__t100.num_planes = 2u8;
__t100.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t101: [u8; 4] = unsafe { core::mem::zeroed() };
__t101[0] = 2u8;
__t101[1] = 1u8;
__t101[2] = 0u8;
__t101
} };
__t100.hsub = 1u8;
__t100.vsub = 1u8;
__t100.has_alpha = 1i32 != 0i32;
__t100
}, {
let mut __t102: drm_format_info = unsafe { core::mem::zeroed() };
__t102.format = 808669784u32;
__t102.depth = 30u8;
__t102.num_planes = 1u8;
__t102.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t103: [u8; 4] = unsafe { core::mem::zeroed() };
__t103[0] = 4u8;
__t103[1] = 0u8;
__t103[2] = 0u8;
__t103
} };
__t102.hsub = 1u8;
__t102.vsub = 1u8;
__t102
}, {
let mut __t104: drm_format_info = unsafe { core::mem::zeroed() };
__t104.format = 808665688u32;
__t104.depth = 30u8;
__t104.num_planes = 1u8;
__t104.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t105: [u8; 4] = unsafe { core::mem::zeroed() };
__t105[0] = 4u8;
__t105[1] = 0u8;
__t105[2] = 0u8;
__t105
} };
__t104.hsub = 1u8;
__t104.vsub = 1u8;
__t104
}, {
let mut __t106: drm_format_info = unsafe { core::mem::zeroed() };
__t106.format = 808671314u32;
__t106.depth = 30u8;
__t106.num_planes = 1u8;
__t106.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t107: [u8; 4] = unsafe { core::mem::zeroed() };
__t107[0] = 4u8;
__t107[1] = 0u8;
__t107[2] = 0u8;
__t107
} };
__t106.hsub = 1u8;
__t106.vsub = 1u8;
__t106
}, {
let mut __t108: drm_format_info = unsafe { core::mem::zeroed() };
__t108.format = 808671298u32;
__t108.depth = 30u8;
__t108.num_planes = 1u8;
__t108.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t109: [u8; 4] = unsafe { core::mem::zeroed() };
__t109[0] = 4u8;
__t109[1] = 0u8;
__t109[2] = 0u8;
__t109
} };
__t108.hsub = 1u8;
__t108.vsub = 1u8;
__t108
}, {
let mut __t110: drm_format_info = unsafe { core::mem::zeroed() };
__t110.format = 808669761u32;
__t110.depth = 30u8;
__t110.num_planes = 1u8;
__t110.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t111: [u8; 4] = unsafe { core::mem::zeroed() };
__t111[0] = 4u8;
__t111[1] = 0u8;
__t111[2] = 0u8;
__t111
} };
__t110.hsub = 1u8;
__t110.vsub = 1u8;
__t110.has_alpha = 1i32 != 0i32;
__t110
}, {
let mut __t112: drm_format_info = unsafe { core::mem::zeroed() };
__t112.format = 808665665u32;
__t112.depth = 30u8;
__t112.num_planes = 1u8;
__t112.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t113: [u8; 4] = unsafe { core::mem::zeroed() };
__t113[0] = 4u8;
__t113[1] = 0u8;
__t113[2] = 0u8;
__t113
} };
__t112.hsub = 1u8;
__t112.vsub = 1u8;
__t112.has_alpha = 1i32 != 0i32;
__t112
}, {
let mut __t114: drm_format_info = unsafe { core::mem::zeroed() };
__t114.format = 808665426u32;
__t114.depth = 30u8;
__t114.num_planes = 1u8;
__t114.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t115: [u8; 4] = unsafe { core::mem::zeroed() };
__t115[0] = 4u8;
__t115[1] = 0u8;
__t115[2] = 0u8;
__t115
} };
__t114.hsub = 1u8;
__t114.vsub = 1u8;
__t114.has_alpha = 1i32 != 0i32;
__t114
}, {
let mut __t116: drm_format_info = unsafe { core::mem::zeroed() };
__t116.format = 808665410u32;
__t116.depth = 30u8;
__t116.num_planes = 1u8;
__t116.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t117: [u8; 4] = unsafe { core::mem::zeroed() };
__t117[0] = 4u8;
__t117[1] = 0u8;
__t117[2] = 0u8;
__t117
} };
__t116.hsub = 1u8;
__t116.vsub = 1u8;
__t116.has_alpha = 1i32 != 0i32;
__t116
}, {
let mut __t118: drm_format_info = unsafe { core::mem::zeroed() };
__t118.format = 942950226u32;
__t118.depth = 0u8;
__t118.num_planes = 1u8;
__t118.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { char_per_block: {
let mut __t119: [u8; 4] = unsafe { core::mem::zeroed() };
__t119[0] = 6u8;
__t119[1] = 0u8;
__t119[2] = 0u8;
__t119
} };
__t118.block_w = {
let mut __t120: [u8; 4] = unsafe { core::mem::zeroed() };
__t120[0] = 1u8;
__t120[1] = 0u8;
__t120[2] = 0u8;
__t120
};
__t118.block_h = {
let mut __t121: [u8; 4] = unsafe { core::mem::zeroed() };
__t121[0] = 1u8;
__t121[1] = 0u8;
__t121[2] = 0u8;
__t121
};
__t118.hsub = 1u8;
__t118.vsub = 1u8;
__t118.has_alpha = 0i32 != 0i32;
__t118
}, {
let mut __t122: drm_format_info = unsafe { core::mem::zeroed() };
__t122.format = 942950210u32;
__t122.depth = 0u8;
__t122.num_planes = 1u8;
__t122.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { char_per_block: {
let mut __t123: [u8; 4] = unsafe { core::mem::zeroed() };
__t123[0] = 6u8;
__t123[1] = 0u8;
__t123[2] = 0u8;
__t123
} };
__t122.block_w = {
let mut __t124: [u8; 4] = unsafe { core::mem::zeroed() };
__t124[0] = 1u8;
__t124[1] = 0u8;
__t124[2] = 0u8;
__t124
};
__t122.block_h = {
let mut __t125: [u8; 4] = unsafe { core::mem::zeroed() };
__t125[0] = 1u8;
__t125[1] = 0u8;
__t125[2] = 0u8;
__t125
};
__t122.hsub = 1u8;
__t122.vsub = 1u8;
__t122.has_alpha = 0i32 != 0i32;
__t122
}, {
let mut __t126: drm_format_info = unsafe { core::mem::zeroed() };
__t126.format = 875713089u32;
__t126.depth = 32u8;
__t126.num_planes = 1u8;
__t126.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t127: [u8; 4] = unsafe { core::mem::zeroed() };
__t127[0] = 4u8;
__t127[1] = 0u8;
__t127[2] = 0u8;
__t127
} };
__t126.hsub = 1u8;
__t126.vsub = 1u8;
__t126.has_alpha = 1i32 != 0i32;
__t126
}, {
let mut __t128: drm_format_info = unsafe { core::mem::zeroed() };
__t128.format = 875708993u32;
__t128.depth = 32u8;
__t128.num_planes = 1u8;
__t128.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t129: [u8; 4] = unsafe { core::mem::zeroed() };
__t129[0] = 4u8;
__t129[1] = 0u8;
__t129[2] = 0u8;
__t129
} };
__t128.hsub = 1u8;
__t128.vsub = 1u8;
__t128.has_alpha = 1i32 != 0i32;
__t128
}, {
let mut __t130: drm_format_info = unsafe { core::mem::zeroed() };
__t130.format = 875708754u32;
__t130.depth = 32u8;
__t130.num_planes = 1u8;
__t130.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t131: [u8; 4] = unsafe { core::mem::zeroed() };
__t131[0] = 4u8;
__t131[1] = 0u8;
__t131[2] = 0u8;
__t131
} };
__t130.hsub = 1u8;
__t130.vsub = 1u8;
__t130.has_alpha = 1i32 != 0i32;
__t130
}, {
let mut __t132: drm_format_info = unsafe { core::mem::zeroed() };
__t132.format = 875708738u32;
__t132.depth = 32u8;
__t132.num_planes = 1u8;
__t132.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t133: [u8; 4] = unsafe { core::mem::zeroed() };
__t133[0] = 4u8;
__t133[1] = 0u8;
__t133[2] = 0u8;
__t133
} };
__t132.hsub = 1u8;
__t132.vsub = 1u8;
__t132.has_alpha = 1i32 != 0i32;
__t132
}, {
let mut __t134: drm_format_info = unsafe { core::mem::zeroed() };
__t134.format = 1211388504u32;
__t134.depth = 0u8;
__t134.num_planes = 1u8;
__t134.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t135: [u8; 4] = unsafe { core::mem::zeroed() };
__t135[0] = 8u8;
__t135[1] = 0u8;
__t135[2] = 0u8;
__t135
} };
__t134.hsub = 1u8;
__t134.vsub = 1u8;
__t134
}, {
let mut __t136: drm_format_info = unsafe { core::mem::zeroed() };
__t136.format = 1211384408u32;
__t136.depth = 0u8;
__t136.num_planes = 1u8;
__t136.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t137: [u8; 4] = unsafe { core::mem::zeroed() };
__t137[0] = 8u8;
__t137[1] = 0u8;
__t137[2] = 0u8;
__t137
} };
__t136.hsub = 1u8;
__t136.vsub = 1u8;
__t136
}, {
let mut __t138: drm_format_info = unsafe { core::mem::zeroed() };
__t138.format = 1211388481u32;
__t138.depth = 0u8;
__t138.num_planes = 1u8;
__t138.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t139: [u8; 4] = unsafe { core::mem::zeroed() };
__t139[0] = 8u8;
__t139[1] = 0u8;
__t139[2] = 0u8;
__t139
} };
__t138.hsub = 1u8;
__t138.vsub = 1u8;
__t138.has_alpha = 1i32 != 0i32;
__t138
}, {
let mut __t140: drm_format_info = unsafe { core::mem::zeroed() };
__t140.format = 1211384385u32;
__t140.depth = 0u8;
__t140.num_planes = 1u8;
__t140.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t141: [u8; 4] = unsafe { core::mem::zeroed() };
__t141[0] = 8u8;
__t141[1] = 0u8;
__t141[2] = 0u8;
__t141
} };
__t140.hsub = 1u8;
__t140.vsub = 1u8;
__t140.has_alpha = 1i32 != 0i32;
__t140
}, {
let mut __t142: drm_format_info = unsafe { core::mem::zeroed() };
__t142.format = 808534593u32;
__t142.depth = 0u8;
__t142.num_planes = 1u8;
__t142.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t143: [u8; 4] = unsafe { core::mem::zeroed() };
__t143[0] = 8u8;
__t143[1] = 0u8;
__t143[2] = 0u8;
__t143
} };
__t142.hsub = 1u8;
__t142.vsub = 1u8;
__t142.has_alpha = 1i32 != 0i32;
__t142
}, {
let mut __t144: drm_format_info = unsafe { core::mem::zeroed() };
__t144.format = 942953048u32;
__t144.depth = 0u8;
__t144.num_planes = 1u8;
__t144.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t145: [u8; 4] = unsafe { core::mem::zeroed() };
__t145[0] = 8u8;
__t145[1] = 0u8;
__t145[2] = 0u8;
__t145
} };
__t144.hsub = 1u8;
__t144.vsub = 1u8;
__t144
}, {
let mut __t146: drm_format_info = unsafe { core::mem::zeroed() };
__t146.format = 942948952u32;
__t146.depth = 0u8;
__t146.num_planes = 1u8;
__t146.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t147: [u8; 4] = unsafe { core::mem::zeroed() };
__t147[0] = 8u8;
__t147[1] = 0u8;
__t147[2] = 0u8;
__t147
} };
__t146.hsub = 1u8;
__t146.vsub = 1u8;
__t146
}, {
let mut __t148: drm_format_info = unsafe { core::mem::zeroed() };
__t148.format = 942953025u32;
__t148.depth = 0u8;
__t148.num_planes = 1u8;
__t148.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t149: [u8; 4] = unsafe { core::mem::zeroed() };
__t149[0] = 8u8;
__t149[1] = 0u8;
__t149[2] = 0u8;
__t149
} };
__t148.hsub = 1u8;
__t148.vsub = 1u8;
__t148.has_alpha = 1i32 != 0i32;
__t148
}, {
let mut __t150: drm_format_info = unsafe { core::mem::zeroed() };
__t150.format = 942948929u32;
__t150.depth = 0u8;
__t150.num_planes = 1u8;
__t150.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t151: [u8; 4] = unsafe { core::mem::zeroed() };
__t151[0] = 8u8;
__t151[1] = 0u8;
__t151[2] = 0u8;
__t151
} };
__t150.hsub = 1u8;
__t150.vsub = 1u8;
__t150.has_alpha = 1i32 != 0i32;
__t150
}, {
let mut __t152: drm_format_info = unsafe { core::mem::zeroed() };
__t152.format = 943798354u32;
__t152.depth = 32u8;
__t152.num_planes = 2u8;
__t152.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t153: [u8; 4] = unsafe { core::mem::zeroed() };
__t153[0] = 3u8;
__t153[1] = 1u8;
__t153[2] = 0u8;
__t153
} };
__t152.hsub = 1u8;
__t152.vsub = 1u8;
__t152.has_alpha = 1i32 != 0i32;
__t152
}, {
let mut __t154: drm_format_info = unsafe { core::mem::zeroed() };
__t154.format = 943798338u32;
__t154.depth = 32u8;
__t154.num_planes = 2u8;
__t154.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t155: [u8; 4] = unsafe { core::mem::zeroed() };
__t155[0] = 3u8;
__t155[1] = 1u8;
__t155[2] = 0u8;
__t155
} };
__t154.hsub = 1u8;
__t154.vsub = 1u8;
__t154.has_alpha = 1i32 != 0i32;
__t154
}, {
let mut __t156: drm_format_info = unsafe { core::mem::zeroed() };
__t156.format = 943805016u32;
__t156.depth = 32u8;
__t156.num_planes = 2u8;
__t156.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t157: [u8; 4] = unsafe { core::mem::zeroed() };
__t157[0] = 4u8;
__t157[1] = 1u8;
__t157[2] = 0u8;
__t157
} };
__t156.hsub = 1u8;
__t156.vsub = 1u8;
__t156.has_alpha = 1i32 != 0i32;
__t156
}, {
let mut __t158: drm_format_info = unsafe { core::mem::zeroed() };
__t158.format = 943800920u32;
__t158.depth = 32u8;
__t158.num_planes = 2u8;
__t158.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t159: [u8; 4] = unsafe { core::mem::zeroed() };
__t159[0] = 4u8;
__t159[1] = 1u8;
__t159[2] = 0u8;
__t159
} };
__t158.hsub = 1u8;
__t158.vsub = 1u8;
__t158.has_alpha = 1i32 != 0i32;
__t158
}, {
let mut __t160: drm_format_info = unsafe { core::mem::zeroed() };
__t160.format = 943806546u32;
__t160.depth = 32u8;
__t160.num_planes = 2u8;
__t160.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t161: [u8; 4] = unsafe { core::mem::zeroed() };
__t161[0] = 4u8;
__t161[1] = 1u8;
__t161[2] = 0u8;
__t161
} };
__t160.hsub = 1u8;
__t160.vsub = 1u8;
__t160.has_alpha = 1i32 != 0i32;
__t160
}, {
let mut __t162: drm_format_info = unsafe { core::mem::zeroed() };
__t162.format = 943806530u32;
__t162.depth = 32u8;
__t162.num_planes = 2u8;
__t162.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t163: [u8; 4] = unsafe { core::mem::zeroed() };
__t163[0] = 4u8;
__t163[1] = 1u8;
__t163[2] = 0u8;
__t163
} };
__t162.hsub = 1u8;
__t162.vsub = 1u8;
__t162.has_alpha = 1i32 != 0i32;
__t162
}, {
let mut __t164: drm_format_info = unsafe { core::mem::zeroed() };
__t164.format = 961959257u32;
__t164.depth = 0u8;
__t164.num_planes = 3u8;
__t164.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t165: [u8; 4] = unsafe { core::mem::zeroed() };
__t165[0] = 1u8;
__t165[1] = 1u8;
__t165[2] = 1u8;
__t165
} };
__t164.hsub = 4u8;
__t164.vsub = 4u8;
__t164.is_yuv = 1i32 != 0i32;
__t164
}, {
let mut __t166: drm_format_info = unsafe { core::mem::zeroed() };
__t166.format = 961893977u32;
__t166.depth = 0u8;
__t166.num_planes = 3u8;
__t166.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t167: [u8; 4] = unsafe { core::mem::zeroed() };
__t167[0] = 1u8;
__t167[1] = 1u8;
__t167[2] = 1u8;
__t167
} };
__t166.hsub = 4u8;
__t166.vsub = 4u8;
__t166.is_yuv = 1i32 != 0i32;
__t166
}, {
let mut __t168: drm_format_info = unsafe { core::mem::zeroed() };
__t168.format = 825316697u32;
__t168.depth = 0u8;
__t168.num_planes = 3u8;
__t168.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t169: [u8; 4] = unsafe { core::mem::zeroed() };
__t169[0] = 1u8;
__t169[1] = 1u8;
__t169[2] = 1u8;
__t169
} };
__t168.hsub = 4u8;
__t168.vsub = 1u8;
__t168.is_yuv = 1i32 != 0i32;
__t168
}, {
let mut __t170: drm_format_info = unsafe { core::mem::zeroed() };
__t170.format = 825316953u32;
__t170.depth = 0u8;
__t170.num_planes = 3u8;
__t170.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t171: [u8; 4] = unsafe { core::mem::zeroed() };
__t171[0] = 1u8;
__t171[1] = 1u8;
__t171[2] = 1u8;
__t171
} };
__t170.hsub = 4u8;
__t170.vsub = 1u8;
__t170.is_yuv = 1i32 != 0i32;
__t170
}, {
let mut __t172: drm_format_info = unsafe { core::mem::zeroed() };
__t172.format = 842093913u32;
__t172.depth = 0u8;
__t172.num_planes = 3u8;
__t172.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t173: [u8; 4] = unsafe { core::mem::zeroed() };
__t173[0] = 1u8;
__t173[1] = 1u8;
__t173[2] = 1u8;
__t173
} };
__t172.hsub = 2u8;
__t172.vsub = 2u8;
__t172.is_yuv = 1i32 != 0i32;
__t172
}, {
let mut __t174: drm_format_info = unsafe { core::mem::zeroed() };
__t174.format = 842094169u32;
__t174.depth = 0u8;
__t174.num_planes = 3u8;
__t174.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t175: [u8; 4] = unsafe { core::mem::zeroed() };
__t175[0] = 1u8;
__t175[1] = 1u8;
__t175[2] = 1u8;
__t175
} };
__t174.hsub = 2u8;
__t174.vsub = 2u8;
__t174.is_yuv = 1i32 != 0i32;
__t174
}, {
let mut __t176: drm_format_info = unsafe { core::mem::zeroed() };
__t176.format = 909202777u32;
__t176.depth = 0u8;
__t176.num_planes = 3u8;
__t176.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t177: [u8; 4] = unsafe { core::mem::zeroed() };
__t177[0] = 1u8;
__t177[1] = 1u8;
__t177[2] = 1u8;
__t177
} };
__t176.hsub = 2u8;
__t176.vsub = 1u8;
__t176.is_yuv = 1i32 != 0i32;
__t176
}, {
let mut __t178: drm_format_info = unsafe { core::mem::zeroed() };
__t178.format = 909203033u32;
__t178.depth = 0u8;
__t178.num_planes = 3u8;
__t178.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t179: [u8; 4] = unsafe { core::mem::zeroed() };
__t179[0] = 1u8;
__t179[1] = 1u8;
__t179[2] = 1u8;
__t179
} };
__t178.hsub = 2u8;
__t178.vsub = 1u8;
__t178.is_yuv = 1i32 != 0i32;
__t178
}, {
let mut __t180: drm_format_info = unsafe { core::mem::zeroed() };
__t180.format = 875713881u32;
__t180.depth = 0u8;
__t180.num_planes = 3u8;
__t180.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t181: [u8; 4] = unsafe { core::mem::zeroed() };
__t181[0] = 1u8;
__t181[1] = 1u8;
__t181[2] = 1u8;
__t181
} };
__t180.hsub = 1u8;
__t180.vsub = 1u8;
__t180.is_yuv = 1i32 != 0i32;
__t180
}, {
let mut __t182: drm_format_info = unsafe { core::mem::zeroed() };
__t182.format = 875714137u32;
__t182.depth = 0u8;
__t182.num_planes = 3u8;
__t182.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t183: [u8; 4] = unsafe { core::mem::zeroed() };
__t183[0] = 1u8;
__t183[1] = 1u8;
__t183[2] = 1u8;
__t183
} };
__t182.hsub = 1u8;
__t182.vsub = 1u8;
__t182.is_yuv = 1i32 != 0i32;
__t182
}, {
let mut __t184: drm_format_info = unsafe { core::mem::zeroed() };
__t184.format = 1497715271u32;
__t184.depth = 8u8;
__t184.num_planes = 1u8;
__t184.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t185: [u8; 4] = unsafe { core::mem::zeroed() };
__t185[0] = 1u8;
__t185[1] = 0u8;
__t185[2] = 0u8;
__t185
} };
__t184.hsub = 1u8;
__t184.vsub = 1u8;
__t184.is_yuv = 1i32 != 0i32;
__t184
}, {
let mut __t186: drm_format_info = unsafe { core::mem::zeroed() };
__t186.format = 842094158u32;
__t186.depth = 0u8;
__t186.num_planes = 2u8;
__t186.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t187: [u8; 4] = unsafe { core::mem::zeroed() };
__t187[0] = 1u8;
__t187[1] = 2u8;
__t187[2] = 0u8;
__t187
} };
__t186.hsub = 2u8;
__t186.vsub = 2u8;
__t186.is_yuv = 1i32 != 0i32;
__t186
}, {
let mut __t188: drm_format_info = unsafe { core::mem::zeroed() };
__t188.format = 825382478u32;
__t188.depth = 0u8;
__t188.num_planes = 2u8;
__t188.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t189: [u8; 4] = unsafe { core::mem::zeroed() };
__t189[0] = 1u8;
__t189[1] = 2u8;
__t189[2] = 0u8;
__t189
} };
__t188.hsub = 2u8;
__t188.vsub = 2u8;
__t188.is_yuv = 1i32 != 0i32;
__t188
}, {
let mut __t190: drm_format_info = unsafe { core::mem::zeroed() };
__t190.format = 909203022u32;
__t190.depth = 0u8;
__t190.num_planes = 2u8;
__t190.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t191: [u8; 4] = unsafe { core::mem::zeroed() };
__t191[0] = 1u8;
__t191[1] = 2u8;
__t191[2] = 0u8;
__t191
} };
__t190.hsub = 2u8;
__t190.vsub = 1u8;
__t190.is_yuv = 1i32 != 0i32;
__t190
}, {
let mut __t192: drm_format_info = unsafe { core::mem::zeroed() };
__t192.format = 825644622u32;
__t192.depth = 0u8;
__t192.num_planes = 2u8;
__t192.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t193: [u8; 4] = unsafe { core::mem::zeroed() };
__t193[0] = 1u8;
__t193[1] = 2u8;
__t193[2] = 0u8;
__t193
} };
__t192.hsub = 2u8;
__t192.vsub = 1u8;
__t192.is_yuv = 1i32 != 0i32;
__t192
}, {
let mut __t194: drm_format_info = unsafe { core::mem::zeroed() };
__t194.format = 875714126u32;
__t194.depth = 0u8;
__t194.num_planes = 2u8;
__t194.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t195: [u8; 4] = unsafe { core::mem::zeroed() };
__t195[0] = 1u8;
__t195[1] = 2u8;
__t195[2] = 0u8;
__t195
} };
__t194.hsub = 1u8;
__t194.vsub = 1u8;
__t194.is_yuv = 1i32 != 0i32;
__t194
}, {
let mut __t196: drm_format_info = unsafe { core::mem::zeroed() };
__t196.format = 842290766u32;
__t196.depth = 0u8;
__t196.num_planes = 2u8;
__t196.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t197: [u8; 4] = unsafe { core::mem::zeroed() };
__t197[0] = 1u8;
__t197[1] = 2u8;
__t197[2] = 0u8;
__t197
} };
__t196.hsub = 1u8;
__t196.vsub = 1u8;
__t196.is_yuv = 1i32 != 0i32;
__t196
}, {
let mut __t198: drm_format_info = unsafe { core::mem::zeroed() };
__t198.format = 1448695129u32;
__t198.depth = 0u8;
__t198.num_planes = 1u8;
__t198.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t199: [u8; 4] = unsafe { core::mem::zeroed() };
__t199[0] = 2u8;
__t199[1] = 0u8;
__t199[2] = 0u8;
__t199
} };
__t198.hsub = 2u8;
__t198.vsub = 1u8;
__t198.is_yuv = 1i32 != 0i32;
__t198
}, {
let mut __t200: drm_format_info = unsafe { core::mem::zeroed() };
__t200.format = 1431918169u32;
__t200.depth = 0u8;
__t200.num_planes = 1u8;
__t200.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t201: [u8; 4] = unsafe { core::mem::zeroed() };
__t201[0] = 2u8;
__t201[1] = 0u8;
__t201[2] = 0u8;
__t201
} };
__t200.hsub = 2u8;
__t200.vsub = 1u8;
__t200.is_yuv = 1i32 != 0i32;
__t200
}, {
let mut __t202: drm_format_info = unsafe { core::mem::zeroed() };
__t202.format = 1498831189u32;
__t202.depth = 0u8;
__t202.num_planes = 1u8;
__t202.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t203: [u8; 4] = unsafe { core::mem::zeroed() };
__t203[0] = 2u8;
__t203[1] = 0u8;
__t203[2] = 0u8;
__t203
} };
__t202.hsub = 2u8;
__t202.vsub = 1u8;
__t202.is_yuv = 1i32 != 0i32;
__t202
}, {
let mut __t204: drm_format_info = unsafe { core::mem::zeroed() };
__t204.format = 1498765654u32;
__t204.depth = 0u8;
__t204.num_planes = 1u8;
__t204.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t205: [u8; 4] = unsafe { core::mem::zeroed() };
__t205[0] = 2u8;
__t205[1] = 0u8;
__t205[2] = 0u8;
__t205
} };
__t204.hsub = 2u8;
__t204.vsub = 1u8;
__t204.is_yuv = 1i32 != 0i32;
__t204
}, {
let mut __t206: drm_format_info = unsafe { core::mem::zeroed() };
__t206.format = 1448434008u32;
__t206.depth = 0u8;
__t206.num_planes = 1u8;
__t206.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t207: [u8; 4] = unsafe { core::mem::zeroed() };
__t207[0] = 4u8;
__t207[1] = 0u8;
__t207[2] = 0u8;
__t207
} };
__t206.hsub = 1u8;
__t206.vsub = 1u8;
__t206.is_yuv = 1i32 != 0i32;
__t206
}, {
let mut __t208: drm_format_info = unsafe { core::mem::zeroed() };
__t208.format = 875713878u32;
__t208.depth = 0u8;
__t208.num_planes = 1u8;
__t208.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t209: [u8; 4] = unsafe { core::mem::zeroed() };
__t209[0] = 3u8;
__t209[1] = 0u8;
__t209[2] = 0u8;
__t209
} };
__t208.hsub = 1u8;
__t208.vsub = 1u8;
__t208.is_yuv = 1i32 != 0i32;
__t208
}, {
let mut __t210: drm_format_info = unsafe { core::mem::zeroed() };
__t210.format = 808671576u32;
__t210.depth = 0u8;
__t210.num_planes = 1u8;
__t210.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t211: [u8; 4] = unsafe { core::mem::zeroed() };
__t211[0] = 4u8;
__t211[1] = 0u8;
__t211[2] = 0u8;
__t211
} };
__t210.hsub = 1u8;
__t210.vsub = 1u8;
__t210.is_yuv = 1i32 != 0i32;
__t210
}, {
let mut __t212: drm_format_info = unsafe { core::mem::zeroed() };
__t212.format = 1448433985u32;
__t212.depth = 0u8;
__t212.num_planes = 1u8;
__t212.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t213: [u8; 4] = unsafe { core::mem::zeroed() };
__t213[0] = 4u8;
__t213[1] = 0u8;
__t213[2] = 0u8;
__t213
} };
__t212.hsub = 1u8;
__t212.vsub = 1u8;
__t212.has_alpha = 1i32 != 0i32;
__t212.is_yuv = 1i32 != 0i32;
__t212
}, {
let mut __t214: drm_format_info = unsafe { core::mem::zeroed() };
__t214.format = 808530521u32;
__t214.depth = 0u8;
__t214.num_planes = 1u8;
__t214.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t215: [u8; 4] = unsafe { core::mem::zeroed() };
__t215[0] = 4u8;
__t215[1] = 0u8;
__t215[2] = 0u8;
__t215
} };
__t214.hsub = 2u8;
__t214.vsub = 1u8;
__t214.is_yuv = 1i32 != 0i32;
__t214
}, {
let mut __t216: drm_format_info = unsafe { core::mem::zeroed() };
__t216.format = 842084953u32;
__t216.depth = 0u8;
__t216.num_planes = 1u8;
__t216.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t217: [u8; 4] = unsafe { core::mem::zeroed() };
__t217[0] = 4u8;
__t217[1] = 0u8;
__t217[2] = 0u8;
__t217
} };
__t216.hsub = 2u8;
__t216.vsub = 1u8;
__t216.is_yuv = 1i32 != 0i32;
__t216
}, {
let mut __t218: drm_format_info = unsafe { core::mem::zeroed() };
__t218.format = 909193817u32;
__t218.depth = 0u8;
__t218.num_planes = 1u8;
__t218.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t219: [u8; 4] = unsafe { core::mem::zeroed() };
__t219[0] = 4u8;
__t219[1] = 0u8;
__t219[2] = 0u8;
__t219
} };
__t218.hsub = 2u8;
__t218.vsub = 1u8;
__t218.is_yuv = 1i32 != 0i32;
__t218
}, {
let mut __t220: drm_format_info = unsafe { core::mem::zeroed() };
__t220.format = 808531033u32;
__t220.depth = 0u8;
__t220.num_planes = 1u8;
__t220.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t221: [u8; 4] = unsafe { core::mem::zeroed() };
__t221[0] = 4u8;
__t221[1] = 0u8;
__t221[2] = 0u8;
__t221
} };
__t220.hsub = 1u8;
__t220.vsub = 1u8;
__t220.has_alpha = 1i32 != 0i32;
__t220.is_yuv = 1i32 != 0i32;
__t220
}, {
let mut __t222: drm_format_info = unsafe { core::mem::zeroed() };
__t222.format = 842085465u32;
__t222.depth = 0u8;
__t222.num_planes = 1u8;
__t222.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t223: [u8; 4] = unsafe { core::mem::zeroed() };
__t223[0] = 8u8;
__t223[1] = 0u8;
__t223[2] = 0u8;
__t223
} };
__t222.hsub = 1u8;
__t222.vsub = 1u8;
__t222.has_alpha = 1i32 != 0i32;
__t222.is_yuv = 1i32 != 0i32;
__t222
}, {
let mut __t224: drm_format_info = unsafe { core::mem::zeroed() };
__t224.format = 909194329u32;
__t224.depth = 0u8;
__t224.num_planes = 1u8;
__t224.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t225: [u8; 4] = unsafe { core::mem::zeroed() };
__t225[0] = 8u8;
__t225[1] = 0u8;
__t225[2] = 0u8;
__t225
} };
__t224.hsub = 1u8;
__t224.vsub = 1u8;
__t224.has_alpha = 1i32 != 0i32;
__t224.is_yuv = 1i32 != 0i32;
__t224
}, {
let mut __t226: drm_format_info = unsafe { core::mem::zeroed() };
__t226.format = 808670808u32;
__t226.depth = 0u8;
__t226.num_planes = 1u8;
__t226.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t227: [u8; 4] = unsafe { core::mem::zeroed() };
__t227[0] = 4u8;
__t227[1] = 0u8;
__t227[2] = 0u8;
__t227
} };
__t226.hsub = 1u8;
__t226.vsub = 1u8;
__t226.is_yuv = 1i32 != 0i32;
__t226
}, {
let mut __t228: drm_format_info = unsafe { core::mem::zeroed() };
__t228.format = 909334104u32;
__t228.depth = 0u8;
__t228.num_planes = 1u8;
__t228.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t229: [u8; 4] = unsafe { core::mem::zeroed() };
__t229[0] = 8u8;
__t229[1] = 0u8;
__t229[2] = 0u8;
__t229
} };
__t228.hsub = 1u8;
__t228.vsub = 1u8;
__t228.is_yuv = 1i32 != 0i32;
__t228
}, {
let mut __t230: drm_format_info = unsafe { core::mem::zeroed() };
__t230.format = 942954072u32;
__t230.depth = 0u8;
__t230.num_planes = 1u8;
__t230.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t231: [u8; 4] = unsafe { core::mem::zeroed() };
__t231[0] = 8u8;
__t231[1] = 0u8;
__t231[2] = 0u8;
__t231
} };
__t230.hsub = 1u8;
__t230.vsub = 1u8;
__t230.is_yuv = 1i32 != 0i32;
__t230
}, {
let mut __t232: drm_format_info = unsafe { core::mem::zeroed() };
__t232.format = 810299481u32;
__t232.depth = 0u8;
__t232.num_planes = 1u8;
__t232.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { char_per_block: {
let mut __t233: [u8; 4] = unsafe { core::mem::zeroed() };
__t233[0] = 8u8;
__t233[1] = 0u8;
__t233[2] = 0u8;
__t233
} };
__t232.block_w = {
let mut __t234: [u8; 4] = unsafe { core::mem::zeroed() };
__t234[0] = 2u8;
__t234[1] = 0u8;
__t234[2] = 0u8;
__t234
};
__t232.block_h = {
let mut __t235: [u8; 4] = unsafe { core::mem::zeroed() };
__t235[0] = 2u8;
__t235[1] = 0u8;
__t235[2] = 0u8;
__t235
};
__t232.hsub = 2u8;
__t232.vsub = 2u8;
__t232.has_alpha = 1i32 != 0i32;
__t232.is_yuv = 1i32 != 0i32;
__t232
}, {
let mut __t236: drm_format_info = unsafe { core::mem::zeroed() };
__t236.format = 810299480u32;
__t236.depth = 0u8;
__t236.num_planes = 1u8;
__t236.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { char_per_block: {
let mut __t237: [u8; 4] = unsafe { core::mem::zeroed() };
__t237[0] = 8u8;
__t237[1] = 0u8;
__t237[2] = 0u8;
__t237
} };
__t236.block_w = {
let mut __t238: [u8; 4] = unsafe { core::mem::zeroed() };
__t238[0] = 2u8;
__t238[1] = 0u8;
__t238[2] = 0u8;
__t238
};
__t236.block_h = {
let mut __t239: [u8; 4] = unsafe { core::mem::zeroed() };
__t239[0] = 2u8;
__t239[1] = 0u8;
__t239[2] = 0u8;
__t239
};
__t236.hsub = 2u8;
__t236.vsub = 2u8;
__t236.is_yuv = 1i32 != 0i32;
__t236
}, {
let mut __t240: drm_format_info = unsafe { core::mem::zeroed() };
__t240.format = 843853913u32;
__t240.depth = 0u8;
__t240.num_planes = 1u8;
__t240.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { char_per_block: {
let mut __t241: [u8; 4] = unsafe { core::mem::zeroed() };
__t241[0] = 8u8;
__t241[1] = 0u8;
__t241[2] = 0u8;
__t241
} };
__t240.block_w = {
let mut __t242: [u8; 4] = unsafe { core::mem::zeroed() };
__t242[0] = 2u8;
__t242[1] = 0u8;
__t242[2] = 0u8;
__t242
};
__t240.block_h = {
let mut __t243: [u8; 4] = unsafe { core::mem::zeroed() };
__t243[0] = 2u8;
__t243[1] = 0u8;
__t243[2] = 0u8;
__t243
};
__t240.hsub = 2u8;
__t240.vsub = 2u8;
__t240.has_alpha = 1i32 != 0i32;
__t240.is_yuv = 1i32 != 0i32;
__t240
}, {
let mut __t244: drm_format_info = unsafe { core::mem::zeroed() };
__t244.format = 843853912u32;
__t244.depth = 0u8;
__t244.num_planes = 1u8;
__t244.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { char_per_block: {
let mut __t245: [u8; 4] = unsafe { core::mem::zeroed() };
__t245[0] = 8u8;
__t245[1] = 0u8;
__t245[2] = 0u8;
__t245
} };
__t244.block_w = {
let mut __t246: [u8; 4] = unsafe { core::mem::zeroed() };
__t246[0] = 2u8;
__t246[1] = 0u8;
__t246[2] = 0u8;
__t246
};
__t244.block_h = {
let mut __t247: [u8; 4] = unsafe { core::mem::zeroed() };
__t247[0] = 2u8;
__t247[1] = 0u8;
__t247[2] = 0u8;
__t247
};
__t244.hsub = 2u8;
__t244.vsub = 2u8;
__t244.is_yuv = 1i32 != 0i32;
__t244
}, {
let mut __t248: drm_format_info = unsafe { core::mem::zeroed() };
__t248.format = 808530000u32;
__t248.depth = 0u8;
__t248.num_planes = 2u8;
__t248.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { char_per_block: {
let mut __t249: [u8; 4] = unsafe { core::mem::zeroed() };
__t249[0] = 2u8;
__t249[1] = 4u8;
__t249[2] = 0u8;
__t249
} };
__t248.block_w = {
let mut __t250: [u8; 4] = unsafe { core::mem::zeroed() };
__t250[0] = 1u8;
__t250[1] = 1u8;
__t250[2] = 0u8;
__t250
};
__t248.block_h = {
let mut __t251: [u8; 4] = unsafe { core::mem::zeroed() };
__t251[0] = 1u8;
__t251[1] = 1u8;
__t251[2] = 0u8;
__t251
};
__t248.hsub = 2u8;
__t248.vsub = 2u8;
__t248.is_yuv = 1i32 != 0i32;
__t248
}, {
let mut __t252: drm_format_info = unsafe { core::mem::zeroed() };
__t252.format = 842084432u32;
__t252.depth = 0u8;
__t252.num_planes = 2u8;
__t252.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { char_per_block: {
let mut __t253: [u8; 4] = unsafe { core::mem::zeroed() };
__t253[0] = 2u8;
__t253[1] = 4u8;
__t253[2] = 0u8;
__t253
} };
__t252.block_w = {
let mut __t254: [u8; 4] = unsafe { core::mem::zeroed() };
__t254[0] = 1u8;
__t254[1] = 1u8;
__t254[2] = 0u8;
__t254
};
__t252.block_h = {
let mut __t255: [u8; 4] = unsafe { core::mem::zeroed() };
__t255[0] = 1u8;
__t255[1] = 1u8;
__t255[2] = 0u8;
__t255
};
__t252.hsub = 2u8;
__t252.vsub = 2u8;
__t252.is_yuv = 1i32 != 0i32;
__t252
}, {
let mut __t256: drm_format_info = unsafe { core::mem::zeroed() };
__t256.format = 909193296u32;
__t256.depth = 0u8;
__t256.num_planes = 2u8;
__t256.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { char_per_block: {
let mut __t257: [u8; 4] = unsafe { core::mem::zeroed() };
__t257[0] = 2u8;
__t257[1] = 4u8;
__t257[2] = 0u8;
__t257
} };
__t256.block_w = {
let mut __t258: [u8; 4] = unsafe { core::mem::zeroed() };
__t258[0] = 1u8;
__t258[1] = 1u8;
__t258[2] = 0u8;
__t258
};
__t256.block_h = {
let mut __t259: [u8; 4] = unsafe { core::mem::zeroed() };
__t259[0] = 1u8;
__t259[1] = 1u8;
__t259[2] = 0u8;
__t259
};
__t256.hsub = 2u8;
__t256.vsub = 2u8;
__t256.is_yuv = 1i32 != 0i32;
__t256
}, {
let mut __t260: drm_format_info = unsafe { core::mem::zeroed() };
__t260.format = 808530512u32;
__t260.depth = 0u8;
__t260.num_planes = 2u8;
__t260.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { char_per_block: {
let mut __t261: [u8; 4] = unsafe { core::mem::zeroed() };
__t261[0] = 2u8;
__t261[1] = 4u8;
__t261[2] = 0u8;
__t261
} };
__t260.block_w = {
let mut __t262: [u8; 4] = unsafe { core::mem::zeroed() };
__t262[0] = 1u8;
__t262[1] = 1u8;
__t262[2] = 0u8;
__t262
};
__t260.block_h = {
let mut __t263: [u8; 4] = unsafe { core::mem::zeroed() };
__t263[0] = 1u8;
__t263[1] = 1u8;
__t263[2] = 0u8;
__t263
};
__t260.hsub = 2u8;
__t260.vsub = 1u8;
__t260.is_yuv = 1i32 != 0i32;
__t260
}, {
let mut __t264: drm_format_info = unsafe { core::mem::zeroed() };
__t264.format = 808670550u32;
__t264.depth = 0u8;
__t264.num_planes = 1u8;
__t264.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t265: [u8; 4] = unsafe { core::mem::zeroed() };
__t265[0] = 0u8;
__t265[1] = 0u8;
__t265[2] = 0u8;
__t265
} };
__t264.hsub = 1u8;
__t264.vsub = 1u8;
__t264.is_yuv = 1i32 != 0i32;
__t264
}, {
let mut __t266: drm_format_info = unsafe { core::mem::zeroed() };
__t266.format = 942691673u32;
__t266.depth = 0u8;
__t266.num_planes = 1u8;
__t266.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t267: [u8; 4] = unsafe { core::mem::zeroed() };
__t267[0] = 0u8;
__t267[1] = 0u8;
__t267[2] = 0u8;
__t267
} };
__t266.hsub = 2u8;
__t266.vsub = 2u8;
__t266.is_yuv = 1i32 != 0i32;
__t266
}, {
let mut __t268: drm_format_info = unsafe { core::mem::zeroed() };
__t268.format = 808539481u32;
__t268.depth = 0u8;
__t268.num_planes = 1u8;
__t268.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { cpp: {
let mut __t269: [u8; 4] = unsafe { core::mem::zeroed() };
__t269[0] = 0u8;
__t269[1] = 0u8;
__t269[2] = 0u8;
__t269
} };
__t268.hsub = 2u8;
__t268.vsub = 2u8;
__t268.is_yuv = 1i32 != 0i32;
__t268
}, {
let mut __t270: drm_format_info = unsafe { core::mem::zeroed() };
__t270.format = 892425806u32;
__t270.depth = 0u8;
__t270.num_planes = 2u8;
__t270.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { char_per_block: {
let mut __t271: [u8; 4] = unsafe { core::mem::zeroed() };
__t271[0] = 5u8;
__t271[1] = 5u8;
__t271[2] = 0u8;
__t271
} };
__t270.block_w = {
let mut __t272: [u8; 4] = unsafe { core::mem::zeroed() };
__t272[0] = 4u8;
__t272[1] = 2u8;
__t272[2] = 0u8;
__t272
};
__t270.block_h = {
let mut __t273: [u8; 4] = unsafe { core::mem::zeroed() };
__t273[0] = 1u8;
__t273[1] = 1u8;
__t273[2] = 0u8;
__t273
};
__t270.hsub = 2u8;
__t270.vsub = 2u8;
__t270.is_yuv = 1i32 != 0i32;
__t270
}, {
let mut __t274: drm_format_info = unsafe { core::mem::zeroed() };
__t274.format = 808605262u32;
__t274.depth = 0u8;
__t274.num_planes = 2u8;
__t274.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { char_per_block: {
let mut __t275: [u8; 4] = unsafe { core::mem::zeroed() };
__t275[0] = 5u8;
__t275[1] = 5u8;
__t275[2] = 0u8;
__t275
} };
__t274.block_w = {
let mut __t276: [u8; 4] = unsafe { core::mem::zeroed() };
__t276[0] = 4u8;
__t276[1] = 2u8;
__t276[2] = 0u8;
__t276
};
__t274.block_h = {
let mut __t277: [u8; 4] = unsafe { core::mem::zeroed() };
__t277[0] = 1u8;
__t277[1] = 1u8;
__t277[2] = 0u8;
__t277
};
__t274.hsub = 2u8;
__t274.vsub = 1u8;
__t274.is_yuv = 1i32 != 0i32;
__t274
}, {
let mut __t278: drm_format_info = unsafe { core::mem::zeroed() };
__t278.format = 808670798u32;
__t278.depth = 0u8;
__t278.num_planes = 2u8;
__t278.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { char_per_block: {
let mut __t279: [u8; 4] = unsafe { core::mem::zeroed() };
__t279[0] = 5u8;
__t279[1] = 5u8;
__t279[2] = 0u8;
__t279
} };
__t278.block_w = {
let mut __t280: [u8; 4] = unsafe { core::mem::zeroed() };
__t280[0] = 4u8;
__t280[1] = 2u8;
__t280[2] = 0u8;
__t280
};
__t278.block_h = {
let mut __t281: [u8; 4] = unsafe { core::mem::zeroed() };
__t281[0] = 1u8;
__t281[1] = 1u8;
__t281[2] = 0u8;
__t281
};
__t278.hsub = 1u8;
__t278.vsub = 1u8;
__t278.is_yuv = 1i32 != 0i32;
__t278
}, {
let mut __t282: drm_format_info = unsafe { core::mem::zeroed() };
__t282.format = 808531025u32;
__t282.depth = 0u8;
__t282.num_planes = 3u8;
__t282.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { char_per_block: {
let mut __t283: [u8; 4] = unsafe { core::mem::zeroed() };
__t283[0] = 2u8;
__t283[1] = 2u8;
__t283[2] = 2u8;
__t283
} };
__t282.block_w = {
let mut __t284: [u8; 4] = unsafe { core::mem::zeroed() };
__t284[0] = 1u8;
__t284[1] = 1u8;
__t284[2] = 1u8;
__t284
};
__t282.block_h = {
let mut __t285: [u8; 4] = unsafe { core::mem::zeroed() };
__t285[0] = 1u8;
__t285[1] = 1u8;
__t285[2] = 1u8;
__t285
};
__t282.hsub = 1u8;
__t282.vsub = 1u8;
__t282.is_yuv = 1i32 != 0i32;
__t282
}, {
let mut __t286: drm_format_info = unsafe { core::mem::zeroed() };
__t286.format = 825242705u32;
__t286.depth = 0u8;
__t286.num_planes = 3u8;
__t286.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { char_per_block: {
let mut __t287: [u8; 4] = unsafe { core::mem::zeroed() };
__t287[0] = 2u8;
__t287[1] = 2u8;
__t287[2] = 2u8;
__t287
} };
__t286.block_w = {
let mut __t288: [u8; 4] = unsafe { core::mem::zeroed() };
__t288[0] = 1u8;
__t288[1] = 1u8;
__t288[2] = 1u8;
__t288
};
__t286.block_h = {
let mut __t289: [u8; 4] = unsafe { core::mem::zeroed() };
__t289[0] = 1u8;
__t289[1] = 1u8;
__t289[2] = 1u8;
__t289
};
__t286.hsub = 1u8;
__t286.vsub = 1u8;
__t286.is_yuv = 1i32 != 0i32;
__t286
}, {
let mut __t290: drm_format_info = unsafe { core::mem::zeroed() };
__t290.format = 808661072u32;
__t290.depth = 0u8;
__t290.num_planes = 2u8;
__t290.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { char_per_block: {
let mut __t291: [u8; 4] = unsafe { core::mem::zeroed() };
__t291[0] = 4u8;
__t291[1] = 8u8;
__t291[2] = 0u8;
__t291
} };
__t290.block_w = {
let mut __t292: [u8; 4] = unsafe { core::mem::zeroed() };
__t292[0] = 3u8;
__t292[1] = 3u8;
__t292[2] = 0u8;
__t292
};
__t290.block_h = {
let mut __t293: [u8; 4] = unsafe { core::mem::zeroed() };
__t293[0] = 1u8;
__t293[1] = 1u8;
__t293[2] = 0u8;
__t293
};
__t290.hsub = 2u8;
__t290.vsub = 2u8;
__t290.is_yuv = 1i32 != 0i32;
__t290
}, {
let mut __t294: drm_format_info = unsafe { core::mem::zeroed() };
__t294.format = 808661584u32;
__t294.depth = 0u8;
__t294.num_planes = 2u8;
__t294.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { char_per_block: {
let mut __t295: [u8; 4] = unsafe { core::mem::zeroed() };
__t295[0] = 4u8;
__t295[1] = 8u8;
__t295[2] = 0u8;
__t295
} };
__t294.block_w = {
let mut __t296: [u8; 4] = unsafe { core::mem::zeroed() };
__t296[0] = 3u8;
__t296[1] = 3u8;
__t296[2] = 0u8;
__t296
};
__t294.block_h = {
let mut __t297: [u8; 4] = unsafe { core::mem::zeroed() };
__t297[0] = 1u8;
__t297[1] = 1u8;
__t297[2] = 0u8;
__t297
};
__t294.hsub = 2u8;
__t294.vsub = 1u8;
__t294.is_yuv = 1i32 != 0i32;
__t294
}, {
let mut __t298: drm_format_info = unsafe { core::mem::zeroed() };
__t298.format = 808530003u32;
__t298.depth = 0u8;
__t298.num_planes = 3u8;
__t298.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { char_per_block: {
let mut __t299: [u8; 4] = unsafe { core::mem::zeroed() };
__t299[0] = 2u8;
__t299[1] = 2u8;
__t299[2] = 2u8;
__t299
} };
__t298.block_w = {
let mut __t300: [u8; 4] = unsafe { core::mem::zeroed() };
__t300[0] = 1u8;
__t300[1] = 1u8;
__t300[2] = 1u8;
__t300
};
__t298.block_h = {
let mut __t301: [u8; 4] = unsafe { core::mem::zeroed() };
__t301[0] = 1u8;
__t301[1] = 1u8;
__t301[2] = 1u8;
__t301
};
__t298.hsub = 2u8;
__t298.vsub = 2u8;
__t298.is_yuv = 1i32 != 0i32;
__t298
}, {
let mut __t302: drm_format_info = unsafe { core::mem::zeroed() };
__t302.format = 808530515u32;
__t302.depth = 0u8;
__t302.num_planes = 3u8;
__t302.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { char_per_block: {
let mut __t303: [u8; 4] = unsafe { core::mem::zeroed() };
__t303[0] = 2u8;
__t303[1] = 2u8;
__t303[2] = 2u8;
__t303
} };
__t302.block_w = {
let mut __t304: [u8; 4] = unsafe { core::mem::zeroed() };
__t304[0] = 1u8;
__t304[1] = 1u8;
__t304[2] = 1u8;
__t304
};
__t302.block_h = {
let mut __t305: [u8; 4] = unsafe { core::mem::zeroed() };
__t305[0] = 1u8;
__t305[1] = 1u8;
__t305[2] = 1u8;
__t305
};
__t302.hsub = 2u8;
__t302.vsub = 1u8;
__t302.is_yuv = 1i32 != 0i32;
__t302
}, {
let mut __t306: drm_format_info = unsafe { core::mem::zeroed() };
__t306.format = 808531027u32;
__t306.depth = 0u8;
__t306.num_planes = 3u8;
__t306.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { char_per_block: {
let mut __t307: [u8; 4] = unsafe { core::mem::zeroed() };
__t307[0] = 2u8;
__t307[1] = 2u8;
__t307[2] = 2u8;
__t307
} };
__t306.block_w = {
let mut __t308: [u8; 4] = unsafe { core::mem::zeroed() };
__t308[0] = 1u8;
__t308[1] = 1u8;
__t308[2] = 1u8;
__t308
};
__t306.block_h = {
let mut __t309: [u8; 4] = unsafe { core::mem::zeroed() };
__t309[0] = 1u8;
__t309[1] = 1u8;
__t309[2] = 1u8;
__t309
};
__t306.hsub = 1u8;
__t306.vsub = 1u8;
__t306.is_yuv = 1i32 != 0i32;
__t306
}, {
let mut __t310: drm_format_info = unsafe { core::mem::zeroed() };
__t310.format = 842084435u32;
__t310.depth = 0u8;
__t310.num_planes = 3u8;
__t310.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { char_per_block: {
let mut __t311: [u8; 4] = unsafe { core::mem::zeroed() };
__t311[0] = 2u8;
__t311[1] = 2u8;
__t311[2] = 2u8;
__t311
} };
__t310.block_w = {
let mut __t312: [u8; 4] = unsafe { core::mem::zeroed() };
__t312[0] = 1u8;
__t312[1] = 1u8;
__t312[2] = 1u8;
__t312
};
__t310.block_h = {
let mut __t313: [u8; 4] = unsafe { core::mem::zeroed() };
__t313[0] = 1u8;
__t313[1] = 1u8;
__t313[2] = 1u8;
__t313
};
__t310.hsub = 2u8;
__t310.vsub = 2u8;
__t310.is_yuv = 1i32 != 0i32;
__t310
}, {
let mut __t314: drm_format_info = unsafe { core::mem::zeroed() };
__t314.format = 842084947u32;
__t314.depth = 0u8;
__t314.num_planes = 3u8;
__t314.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { char_per_block: {
let mut __t315: [u8; 4] = unsafe { core::mem::zeroed() };
__t315[0] = 2u8;
__t315[1] = 2u8;
__t315[2] = 2u8;
__t315
} };
__t314.block_w = {
let mut __t316: [u8; 4] = unsafe { core::mem::zeroed() };
__t316[0] = 1u8;
__t316[1] = 1u8;
__t316[2] = 1u8;
__t316
};
__t314.block_h = {
let mut __t317: [u8; 4] = unsafe { core::mem::zeroed() };
__t317[0] = 1u8;
__t317[1] = 1u8;
__t317[2] = 1u8;
__t317
};
__t314.hsub = 2u8;
__t314.vsub = 1u8;
__t314.is_yuv = 1i32 != 0i32;
__t314
}, {
let mut __t318: drm_format_info = unsafe { core::mem::zeroed() };
__t318.format = 842085459u32;
__t318.depth = 0u8;
__t318.num_planes = 3u8;
__t318.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { char_per_block: {
let mut __t319: [u8; 4] = unsafe { core::mem::zeroed() };
__t319[0] = 2u8;
__t319[1] = 2u8;
__t319[2] = 2u8;
__t319
} };
__t318.block_w = {
let mut __t320: [u8; 4] = unsafe { core::mem::zeroed() };
__t320[0] = 1u8;
__t320[1] = 1u8;
__t320[2] = 1u8;
__t320
};
__t318.block_h = {
let mut __t321: [u8; 4] = unsafe { core::mem::zeroed() };
__t321[0] = 1u8;
__t321[1] = 1u8;
__t321[2] = 1u8;
__t321
};
__t318.hsub = 1u8;
__t318.vsub = 1u8;
__t318.is_yuv = 1i32 != 0i32;
__t318
}, {
let mut __t322: drm_format_info = unsafe { core::mem::zeroed() };
__t322.format = 909193299u32;
__t322.depth = 0u8;
__t322.num_planes = 3u8;
__t322.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { char_per_block: {
let mut __t323: [u8; 4] = unsafe { core::mem::zeroed() };
__t323[0] = 2u8;
__t323[1] = 2u8;
__t323[2] = 2u8;
__t323
} };
__t322.block_w = {
let mut __t324: [u8; 4] = unsafe { core::mem::zeroed() };
__t324[0] = 1u8;
__t324[1] = 1u8;
__t324[2] = 1u8;
__t324
};
__t322.block_h = {
let mut __t325: [u8; 4] = unsafe { core::mem::zeroed() };
__t325[0] = 1u8;
__t325[1] = 1u8;
__t325[2] = 1u8;
__t325
};
__t322.hsub = 2u8;
__t322.vsub = 2u8;
__t322.is_yuv = 1i32 != 0i32;
__t322
}, {
let mut __t326: drm_format_info = unsafe { core::mem::zeroed() };
__t326.format = 909193811u32;
__t326.depth = 0u8;
__t326.num_planes = 3u8;
__t326.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { char_per_block: {
let mut __t327: [u8; 4] = unsafe { core::mem::zeroed() };
__t327[0] = 2u8;
__t327[1] = 2u8;
__t327[2] = 2u8;
__t327
} };
__t326.block_w = {
let mut __t328: [u8; 4] = unsafe { core::mem::zeroed() };
__t328[0] = 1u8;
__t328[1] = 1u8;
__t328[2] = 1u8;
__t328
};
__t326.block_h = {
let mut __t329: [u8; 4] = unsafe { core::mem::zeroed() };
__t329[0] = 1u8;
__t329[1] = 1u8;
__t329[2] = 1u8;
__t329
};
__t326.hsub = 2u8;
__t326.vsub = 1u8;
__t326.is_yuv = 1i32 != 0i32;
__t326
}, {
let mut __t330: drm_format_info = unsafe { core::mem::zeroed() };
__t330.format = 909194323u32;
__t330.depth = 0u8;
__t330.num_planes = 3u8;
__t330.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { char_per_block: {
let mut __t331: [u8; 4] = unsafe { core::mem::zeroed() };
__t331[0] = 2u8;
__t331[1] = 2u8;
__t331[2] = 2u8;
__t331
} };
__t330.block_w = {
let mut __t332: [u8; 4] = unsafe { core::mem::zeroed() };
__t332[0] = 1u8;
__t332[1] = 1u8;
__t332[2] = 1u8;
__t332
};
__t330.block_h = {
let mut __t333: [u8; 4] = unsafe { core::mem::zeroed() };
__t333[0] = 1u8;
__t333[1] = 1u8;
__t333[2] = 1u8;
__t333
};
__t330.hsub = 1u8;
__t330.vsub = 1u8;
__t330.is_yuv = 1i32 != 0i32;
__t330
}, {
let mut __t334: drm_format_info = unsafe { core::mem::zeroed() };
__t334.format = 876695641u32;
__t334.depth = 0u8;
__t334.num_planes = 1u8;
__t334.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { char_per_block: {
let mut __t335: [u8; 4] = unsafe { core::mem::zeroed() };
__t335[0] = 4u8;
__t335[1] = 0u8;
__t335[2] = 0u8;
__t335
} };
__t334.block_w = {
let mut __t336: [u8; 4] = unsafe { core::mem::zeroed() };
__t336[0] = 3u8;
__t336[1] = 0u8;
__t336[2] = 0u8;
__t336
};
__t334.block_h = {
let mut __t337: [u8; 4] = unsafe { core::mem::zeroed() };
__t337[0] = 1u8;
__t337[1] = 0u8;
__t337[2] = 0u8;
__t337
};
__t334.hsub = 1u8;
__t334.vsub = 1u8;
__t334.is_yuv = 1i32 != 0i32;
__t334
}, {
let mut __t338: drm_format_info = unsafe { core::mem::zeroed() };
__t338.format = 808662100u32;
__t338.depth = 0u8;
__t338.num_planes = 3u8;
__t338.__slate_anon_3 = __SlateRecord_drm_fourcc_h_77 { char_per_block: {
let mut __t339: [u8; 4] = unsafe { core::mem::zeroed() };
__t339[0] = 4u8;
__t339[1] = 4u8;
__t339[2] = 4u8;
__t339
} };
__t338.block_w = {
let mut __t340: [u8; 4] = unsafe { core::mem::zeroed() };
__t340[0] = 3u8;
__t340[1] = 3u8;
__t340[2] = 3u8;
__t340
};
__t338.block_h = {
let mut __t341: [u8; 4] = unsafe { core::mem::zeroed() };
__t341[0] = 1u8;
__t341[1] = 1u8;
__t341[2] = 1u8;
__t341
};
__t338.hsub = 1u8;
__t338.vsub = 1u8;
__t338.is_yuv = 1i32 != 0i32;
__t338
}]);

// drm_format_info - query information for a given format
// @format: pixel format (DRM_FORMAT_*)
//
// The caller should only pass a supported pixel format to this function.
// Unsupported pixel formats will generate a warning in the kernel log.
//
// Returns:
// The instance of struct drm_format_info that describes the pixel format, or
// NULL if the format is unsupported.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_format_info_232: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u32) -> *const drm_format_info>, *mut ()>(Some(drm_format_info)) };

core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_drm_mode_legacy_fb_format: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_driver_legacy_fb_format: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_driver_color_mode_format: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_format_info: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_get_format_info: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_format_info_block_width: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_format_info_block_height: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_format_info_bpp: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {7} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_drm_format_info_min_pitch: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {8} ; .previous", sym drm_mode_legacy_fb_format, sym drm_driver_legacy_fb_format, sym drm_driver_color_mode_format, sym drm_format_info, sym drm_get_format_info, sym drm_format_info_block_width, sym drm_format_info_block_height, sym drm_format_info_bpp, sym drm_format_info_min_pitch, options(att_syntax));

/// drm_get_format_info - query information for a given framebuffer configuration
/// @dev: DRM device
/// @pixel_format: pixel format (DRM_FORMAT_*)
/// @modifier: modifier
///
/// Returns:
/// The instance of struct drm_format_info that describes the pixel format, or
/// NULL if the format is unsupported.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_fourcc.drm_get_format_info")]
extern "C" fn drm_get_format_info(dev: *mut drm_device, pixel_format: u32, modifier: u64) -> *const drm_format_info {
let mut info: *const drm_format_info = core::ptr::null::<drm_format_info>();
if (unsafe { (*(*dev).mode_config.funcs).get_format_info }) != None {
info = unsafe { unsafe { (*(*dev).mode_config.funcs).get_format_info }.unwrap()(pixel_format, modifier) };
}
if info.is_null() {
info = drm_format_info(pixel_format);
}
info
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_get_format_info_233: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut drm_device, u32, u64) -> *const drm_format_info>, *mut ()>(Some(drm_get_format_info)) };

/// drm_format_info_block_width - width in pixels of block.
/// @info: pixel format info
/// @plane: plane index
///
/// Returns:
/// The width in pixels of a block, depending on the plane index.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_fourcc.drm_format_info_block_width")]
extern "C" fn drm_format_info_block_width(info: *const drm_format_info, plane: i32) -> u32 {
if info.is_null() || plane < 0 || plane >= ((unsafe { (*info).num_planes }) as i32) {
return 0;
}
if (unsafe { *((*info).block_w.as_ptr() as *const u8).wrapping_offset(plane as isize) }) == 0 {
return 1;
}
(unsafe { *((*info).block_w.as_ptr() as *const u8).wrapping_offset(plane as isize) }) as u32
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_format_info_block_width_234: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const drm_format_info, i32) -> u32>, *mut ()>(Some(drm_format_info_block_width)) };

/// drm_format_info_block_height - height in pixels of a block
/// @info: pixel format info
/// @plane: plane index
///
/// Returns:
/// The height in pixels of a block, depending on the plane index.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_fourcc.drm_format_info_block_height")]
extern "C" fn drm_format_info_block_height(info: *const drm_format_info, plane: i32) -> u32 {
if info.is_null() || plane < 0 || plane >= ((unsafe { (*info).num_planes }) as i32) {
return 0;
}
if (unsafe { *((*info).block_h.as_ptr() as *const u8).wrapping_offset(plane as isize) }) == 0 {
return 1;
}
(unsafe { *((*info).block_h.as_ptr() as *const u8).wrapping_offset(plane as isize) }) as u32
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_format_info_block_height_235: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const drm_format_info, i32) -> u32>, *mut ()>(Some(drm_format_info_block_height)) };

/// drm_format_info_bpp - number of bits per pixel
/// @info: pixel format info
/// @plane: plane index
///
/// Returns:
/// The actual number of bits per pixel, depending on the plane index.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_fourcc.drm_format_info_bpp")]
extern "C" fn drm_format_info_bpp(info: *const drm_format_info, plane: i32) -> u32 {
let mut block_size: u32 = 0;
if info.is_null() || plane < 0 || plane >= ((unsafe { (*info).num_planes }) as i32) {
return 0;
}
block_size = drm_format_info_block_width(info, plane).wrapping_mul(drm_format_info_block_height(info, plane));
if (((unsafe { *((*info).__slate_anon_3.char_per_block.as_ptr() as *const u8).wrapping_offset(plane as isize) }) as i32).wrapping_mul(8) as u32) % block_size != 0 {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x014unable to return an integer bpp\n".as_ptr() as *const u8) };
return 0;
}
(((unsafe { *((*info).__slate_anon_3.char_per_block.as_ptr() as *const u8).wrapping_offset(plane as isize) }) as i32).wrapping_mul(8) as u32) / block_size
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_format_info_bpp_236: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const drm_format_info, i32) -> u32>, *mut ()>(Some(drm_format_info_bpp)) };

/// drm_format_info_min_pitch - computes the minimum required pitch in bytes
/// @info: pixel format info
/// @plane: plane index
/// @buffer_width: buffer width in pixels
///
/// Returns:
/// The minimum required pitch in bytes for a buffer by taking into consideration
/// the pixel format information and the buffer width.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.drm_fourcc.drm_format_info_min_pitch")]
extern "C" fn drm_format_info_min_pitch(info: *const drm_format_info, plane: i32, buffer_width: u32) -> u64 {
if info.is_null() || plane < 0 || plane >= ((unsafe { (*info).num_planes }) as i32) {
return 0;
}
DIV_ROUND_UP_ULL!((buffer_width as u64).wrapping_mul(((unsafe { *((*info).__slate_anon_3.char_per_block.as_ptr() as *const u8).wrapping_offset(plane as isize) }) as i32) as u64), drm_format_info_block_width(info, plane).wrapping_mul(drm_format_info_block_height(info, plane)))
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_drm_format_info_min_pitch_237: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const drm_format_info, i32, u32) -> u64>, *mut ()>(Some(drm_format_info_min_pitch)) };

static __slate_asm_string_drm_format_info_1: [u8; 1] = *b"\0";

static __slate_asm_string_drm_format_info_2: [u8; 58] = *b"/home/takashi/c-corpus/linux/drivers/gpu/drm/drm_fourcc.c\0";

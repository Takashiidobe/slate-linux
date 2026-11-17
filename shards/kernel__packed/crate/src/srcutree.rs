
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
//! SPDX-License-Identifier: GPL-2.0+
//!
//! Sleepable Read-Copy Update mechanism for mutual exclusion.
//!
//! Copyright (C) IBM Corporation, 2006
//! Copyright (C) Fujitsu, 2012
//!
//! Authors: Paul McKenney <paulmck@linux.ibm.com>
//! 	   Lai Jiangshan <laijs@cn.fujitsu.com>
//!
//! For detailed explanation of Read-Copy Update mechanism see -
//! 		Documentation/RCU/ *.txt
//! Optimization barrier
//!
//! The "volatile" is due to gcc bugs
//! aligned_u64 should be used in defining kernel<->userspace ABIs to avoid
//! common 32/64-bit compat problems.
//! 64-bit values align to 4-byte boundaries on x86_32 (and possibly other
//! architectures) and to 8-byte boundaries on 64-bit architectures.  The new
//! aligned_64 type enforces 8-byte alignment so that structs containing
//! aligned_64 values have the same alignment on 32-bit and 64-bit architectures.
//! No conversions are necessary between 32-bit user-space and a 64-bit kernel.
//! These are the special 64-bit data types that are 8-byte aligned
//! Yes, this permits 64-bit accesses on 32-bit architectures. These will
//! actually be atomic in some cases (namely Armv7 + LPAE), but for others we
//! rely on the access being split into 2x32-bit accesses for a 32-bit quantity
//! (e.g. a virtual address) and a strong prevailing wind.
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
//! Force a compilation error if condition is true, but also produce a
//! result (of value 0 and type int), so the expression can be used
//! e.g. in a structure initializer (or where-ever else comma expressions
//! aren't permitted).
//!
//! Take an error message as an optional second argument. If omitted,
//! default to the stringification of the tested expression.
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
//! These are non-NULL pointers that will result in page faults
//! under normal circumstances, used to verify that nobody uses
//! non-initialized list entries.
//! Begin/end of an instrumentation safe region
//! Call a function once. Similar to DO_ONCE(), but does not use jump label
//! patching via static keys.
//! These are for everybody (although not all archs will actually
//! discard it in modules)
//! warning conditions
//! Internal kernel timer frequency
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! issue num suppressed message on exit
//! strlen("ratelimit") + 1
//! label
//! Prevent speculative execution past this barrier.
//! LIST_HEAD_INIT - initialize a &struct list_head's links to point to itself
//! @name: name of the list_head
//! Operation not permitted
//! Value too large for defined data type
//! restart by calling sys_restart_syscall
//! Defines x86 CPU feature bits
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
//! Compile time versions of __arch_hweightN()
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
//! ilog2 - log base 2 of 32-bit or a 64-bit unsigned value
//! @n: parameter
//!
//! constant-capable log of base 2 calculation
//! - this can be used to initialise global variables from constant data, hence
//! the massive ternary operator construction
//!
//! selects the appropriately-sized optimised version depending on sizeof(n)
//! This looks more complex than it should be. But we need to
//! get the type for the ~ right in round_down (it needs to be
//! as wide as the result!), and we want to evaluate the macro
//! arguments just once each.
//! used to decode ioctl numbers..
//! The inline keyword gives the compiler room to decide to inline, or
//! not inline a function as it sees best.  However, as these functions
//! are called in both __init and non-__init functions, if they are not
//! inlined we will end up with a section mismatch error (of the type of
//! freeable items not being freed).  So we must use __always_inline here
//! to fix the problem.  If other functions in the future also end up in
//! this situation they will also need to be annotated as __always_inline
//! PAGE_SHIFT determines the page size.
//!
//! Note: This definition is required because PAGE_SHIFT is used
//! in several places throughout the codebase.
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
//! Sign Flag
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
//! signal pending
//! mutex_init - initialize the mutex
//! @mutex: the mutex to be initialized
//!
//! Initialize the mutex to unlocked state.
//!
//! It is not allowed to initialize an already locked mutex.
//! SCHED_ISO: reserved but not implemented yet
//! These constants are defined in the MSan LLVM instrumentation pass.
//! ignore signal
//! bit-flags
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
//! return the ->next pointer of a list_head in an rcu safe
//! way, we must not access it directly
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
//! Instruction opcode for WRMSRNS supported in binutils >= 2.40
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
//! __MAX_NR_ZONES
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
//! Desired design of maximum size and alignment (see RFC2553)
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
//! Keep unconverted code working
//! used to be O_SYNC, see below
//! Valid hint values for F_{GET,SET}_RW_HINT. 0 is "not set", or can be
//! used to clear any hints previously set.
//! Gives us 8 prio classes with 13-bits of data for each class
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
//! Generate SD flag indexes
//! Resource type
//! Currently, the max observed size in the kernel code is
//! JUMP_LABEL_NOP_SIZE/RELATIVEJUMP_SIZE, which are 5.
//! Raise it if needed.
//! A complete grace period count
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
// Format: <modname>__<counter>_<line>_<fn>
// Format: __<prefix>__<iid><id>
// Early initcalls run before initializing SMP.
//
// Only for built-in code, not modules.
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
// pr_alert - Print an alert-level message
// @fmt: format string
// @...: arguments for the format string
//
// This macro expands to a printk with KERN_ALERT loglevel. It uses pr_fmt() to
// generate the format string.
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
// pr_cont - Continues a previous log message in the same line.
// @fmt: format string
// @...: arguments for the format string
//
// This macro expands to a printk with KERN_CONT loglevel. It should only be
// used when continuing a log message with no newline ('\n') enclosed. Otherwise
// it defaults back to KERN_DEFAULT loglevel.
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
// N 32-bit words worth of info
// N 32-bit bug flags
// "tsc" Time Stamp Counter
// "apic" Onboard APIC
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
// "rdpid" RDPID instruction
// "fsrm" Fast Short Rep Mov
// "serialize" SERIALIZE instruction
// LFENCE always serializing / synchronizes RDTSC
// BUG word(s)
// "l1tf" CPU is affected by L1 Terminal Fault
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
// The following macros are non-atomic versions of their non-underscored
// counterparts.
// Check that the bitops prototypes are sane
// assign_bit - Assign value to a bit in memory
// @nr: the bit to set
// @addr: the address to start counting from
// @value: the value to assign
// IS_ERR_VALUE - Detect an error pointer.
// @x: The pointer to check.
//
// Like IS_ERR(), but does not generate a compiler warning if result is unused.
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
// min - return minimum of two values of the same or compatible types
// @x: first value
// @y: second value
// max - return maximum of two values of the same or compatible types
// @x: first value
// @y: second value
// Use these carefully: no type checking, and uses the arguments
// multiple times. Use for obvious constants only.
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
// No static inline type checking - see Subtlety (1) above.
// for_each_cpu - iterate over every cpu in a mask
// @cpu: the (optionally unsigned) integer iterator
// @mask: the cpumask pointer
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
// Overflow Flag
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
// entries per page directory level
// We borrow bit 3 to remember PG_anon_exclusive.
// N 32-bit words worth of info
// signal notifications exist
// rescheduling necessary
// idle is polling for TIF_NEED_RESCHED
// re-arm the timer
// This function is needed by all SMP systems. It must _always_ be valid
// from the initial startup.
// If !LOCKDEP, the value is meaningless
// For trivial one-depth nesting of a lock-class, the following
// global define can be used. (Subsystems with multiple levels
// of nesting should define their own lock-nesting subclasses.)
// disable sas during sighandling
// Writer mask
// Remapping rwlock architecture specific functions to the corresponding
// queued rwlock functions.
// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
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
// Same as rb_first(), but O(1)
// Change timeval to jiffies, trying to avoid the
// most obvious overflows..
//
// And some not so obvious.
//
// Note that we don't want to return LONG_MAX, because
// for various timeout reasons we often end up having
// to wait "jiffies+1" in order to guarantee that we wait
// at _least_ "jiffies" - so "jiffies+1" had better still
// be positive.
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
// Implementation specific max size
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
// per-IO, return -EAGAIN if operation would block
// per-IO O_APPEND
// per-IO negation of O_APPEND
// Atomic Write
// buffered IO that drops the cache after reading or writing data
// mask of flags supported by the kernel
// Set the IDR flag and the IDR_FREE tag
// Generate SD flag bits
// Resource extended types
// No address assigned yet
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
macro_rules! ARRAY_SIZE__24 {
() => {
(((const { core::mem::size_of::<[u64; 4]>() }) as u64) / ((const { core::mem::size_of::<u64>() }) as u64)).wrapping_add(__must_be_array__116!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__53 {
() => {
(((const { core::mem::size_of::<[u64; 4]>() }) as u64) / ((const { core::mem::size_of::<u64>() }) as u64)).wrapping_add(__must_be_array__32!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__68 {
() => {
((core::mem::size_of::<[*const u8; 10]>() as u64) / (core::mem::size_of::<*const u8>() as u64)).wrapping_add(__must_be_array__69!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__91 {
() => {
((core::mem::size_of::<[u64; 4]>() as u64) / (core::mem::size_of::<u64>() as u64)).wrapping_add(__must_be_array__6!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__94 {
() => {
((core::mem::size_of::<[u64; 4]>() as u64) / (core::mem::size_of::<u64>() as u64)).wrapping_add(__must_be_array__39!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__106 {
() => {
((core::mem::size_of::<[u64; 4]>() as u64) / (core::mem::size_of::<u64>() as u64)).wrapping_add(__must_be_array__84!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__114 {
() => {
((core::mem::size_of::<[*const u8; 10]>() as u64) / (core::mem::size_of::<*const u8>() as u64)).wrapping_add(__must_be_array__138!() as u64)
};
}

/// ASSERT_EXCLUSIVE_WRITER - assert no concurrent writes to @var
///
/// Assert that there are no concurrent writes to @var; other readers are
/// allowed. This assertion can be used to specify properties of concurrent code,
/// where violation cannot be detected as a normal data race.
///
/// For example, if we only have a single writer, but multiple concurrent
/// readers, to avoid data races, all these accesses must be marked; even
/// concurrent marked writes racing with the single writer are bugs.
/// Unfortunately, due to being marked, they are no longer data races. For cases
/// like these, we can use the macro as follows:
///
/// .. code-block:: c
///
/// 	void writer(void) {
/// 		spin_lock(&update_foo_lock);
/// 		ASSERT_EXCLUSIVE_WRITER(shared_foo);
/// 		WRITE_ONCE(shared_foo, ...);
/// 		spin_unlock(&update_foo_lock);
/// 	}
/// 	void reader(void) {
/// 		// update_foo_lock does not need to be held!
/// 		... = READ_ONCE(shared_foo);
/// 	}
///
/// Note: ASSERT_EXCLUSIVE_WRITER_SCOPED(), if applicable, performs more thorough
/// checking if a clear scope where no concurrent writes are expected exists.
///
/// @var: variable to assert on
macro_rules! ASSERT_EXCLUSIVE_WRITER__2 {
($var:expr) => {
__kcsan_check_access((unsafe { core::ptr::addr_of_mut!($var) }) as *const (), core::mem::size_of::<u64>() as u64, 8i32);
};
}

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

macro_rules! GENMASK_INPUT_CHECK {
() => {
core::mem::size_of::<__SlateRecord_find_h_67>() as i32
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
macro_rules! GENMASK_TYPE {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub($h))
};
}

macro_rules! GENMASK {
($h:expr, $l:expr) => {
GENMASK_TYPE!(u64, $h, $l)
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

macro_rules! RELOC_HIDE {
($ptr:expr, $off:expr) => {
(($ptr) as u64).wrapping_add(($off) as u64) as *mut ()
};
}

macro_rules! SRCU_SIZING_IS_CONTEND {
() => {
(unsafe { convert_to_big }) & 16
};
}

macro_rules! SRCU_SIZING_IS_INIT {
() => {
SRCU_SIZING_IS!(1i32)
};
}

macro_rules! SRCU_SIZING_IS_TORTURE {
() => {
SRCU_SIZING_IS!(2i32)
};
}

macro_rules! SRCU_SIZING_IS {
($x:expr) => {
(unsafe { convert_to_big }) & !16 == $x
};
}

macro_rules! SRCU_UL_CLAMP_HI {
($val:expr, $high:expr) => {
if ($val) < $high { $val } else { $high }
};
}

macro_rules! SRCU_UL_CLAMP {
($val:expr, $low:expr, $high:expr) => {
SRCU_UL_CLAMP_HI!(if ($val) > $low { $val } else { $low }, $high)
};
}

macro_rules! ULONG_CMP_GE {
($a:expr, $b:expr) => {
!0 / 2 >= $a.wrapping_sub($b)
};
}

macro_rules! ULONG_CMP_LT {
($a:expr, $b:expr) => {
!0 / 2 < $a.wrapping_sub($b)
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

macro_rules! __IRQ_MASK {
($x:expr) => {
(1u64 << $x).wrapping_sub(1)
};
}

macro_rules! __READ_ONCE__2 {
($x:expr) => {
unsafe { core::ptr::read_volatile(core::ptr::addr_of!($x) as *const i64) }
};
}

macro_rules! __WRITE_ONCE__13 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut u64, ($val) as u64) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE__55 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut *mut srcu_ctr, $val) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE__3 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut i32, $val) };
if true {
break;
}
}
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

macro_rules! __WRITE_ONCE__2 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut u64, $val) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE__37 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut Option<unsafe extern "C" fn(*mut callback_head)>, $val) };
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

macro_rules! __alloc_percpu {
($_size:expr, $_align:expr) => {
alloc_hooks!(unsafe { pcpu_alloc_noprof($_size, $_align, false, 3264u32) })
};
}

macro_rules! __assume_ctx_lock {
($var:expr) => {
loop {
$var;
if true {
break;
}
}
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

macro_rules! __must_be_array__6 {
() => {
core::mem::size_of::<__SlateRecord_srcutree_c_935>() as i32
};
}

macro_rules! __must_be_array__32 {
() => {
((const { core::mem::size_of::<__SlateRecord_srcutree_c_157>() }) as u64) as i32
};
}

macro_rules! __must_be_array__39 {
() => {
core::mem::size_of::<__SlateRecord_srcutree_c_1045>() as i32
};
}

macro_rules! __must_be_array__69 {
() => {
core::mem::size_of::<__SlateRecord_srcutree_c_2022>() as i32
};
}

macro_rules! __must_be_array__84 {
() => {
core::mem::size_of::<__SlateRecord_srcutree_c_158>() as i32
};
}

macro_rules! __must_be_array__116 {
() => {
((const { core::mem::size_of::<__SlateRecord_srcutree_c_156>() }) as u64) as i32
};
}

macro_rules! __must_be_array__138 {
() => {
core::mem::size_of::<__SlateRecord_srcutree_c_2023>() as i32
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__4 {
() => {
((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__14 {
() => {
((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__7 {
() => {
((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__63 {
() => {
((const { core::mem::size_of::<Option<unsafe extern "C" fn(*mut callback_head)>>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<Option<unsafe extern "C" fn(*mut callback_head)>>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<Option<unsafe extern "C" fn(*mut callback_head)>>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<Option<unsafe extern "C" fn(*mut callback_head)>>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__10 {
() => {
((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__24 {
() => {
((const { core::mem::size_of::<*mut callback_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut callback_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut callback_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut callback_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__5 {
() => {
((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word {
() => {
((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

macro_rules! __or {
() => {
____or!()
};
}

macro_rules! __timer_init {
($_timer:expr, $_fn:expr, $_flags:expr) => {
unsafe { timer_init_key($_timer, $_fn, ($_flags) as u32, core::ptr::null::<u8>(), core::ptr::null_mut::<lock_class_key>()) };
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

/// __verify_pcpu_ptr() verifies @ptr is a percpu pointer without evaluating
/// @ptr and is invoked once before a percpu area is accessed by all
/// accessors and operations.  This is performed in the generic part of
/// percpu and arch overrides don't need to worry about it; however, if an
/// arch wants to implement an arch-specific percpu accessor or operation,
/// it may use __verify_pcpu_ptr() to verify the parameters.
///
/// + 0 is required in order to convert the pointer type from a
/// potential array type to a pointer to a single item of the array.
macro_rules! __verify_pcpu_ptr__43 {
() => {
loop {
let __v: *const () = core::ptr::null_mut::<srcu_data>() as *const ();
if true {
break;
}
}
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
macro_rules! __verify_pcpu_ptr__27 {
() => {
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<srcu_data>() as *const ();
if true {
break;
}
}
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
macro_rules! __verify_pcpu_ptr__4 {
() => {
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i64>() as *const ();
if true {
break;
}
}
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
macro_rules! __verify_pcpu_ptr__2 {
() => {
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<*mut task_struct>() as *const ();
if true {
break;
}
}
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
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
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

macro_rules! alloc_percpu {
($type:ty) => {
__alloc_percpu!(core::mem::size_of::<$type>() as u64, core::mem::align_of::<$type>() as u64) as *mut $type
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

/// container_of - cast a member of a structure out to the containing structure
/// @ptr:	the pointer to the member.
/// @type:	the type of the container struct this is embedded in.
/// @member:	the name of the member within the struct.
///
/// WARNING: any const qualifier of @ptr is lost.
/// Do not use container_of() in new code.
macro_rules! container_of__5 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__15!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__14 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__34!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__98 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__140!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__99 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__169!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__145 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__207!($type) as isize)) }) as *mut ()) as *mut $type;
__v
}
};
}

/// cpumask_bits - get the bits in a cpumask
/// @maskp: the struct cpumask *
///
/// You should only assume nr_cpu_ids bits of this mask are valid.  This is
/// a macro so it's const-correct.
macro_rules! cpumask_bits {
($maskp:expr) => {
unsafe { (*$maskp).bits.as_ptr() as *const u64 }
};
}

/// data_race - mark an expression as containing intentional data races
///
/// This data_race() macro is useful for situations in which data races
/// should be forgiven.  One example is diagnostic code that accesses
/// shared variables but is not a part of the core synchronization design.
/// For example, if accesses to a given variable are protected by a lock,
/// except for diagnostic code, then the accesses under the lock should
/// be plain C-language accesses and those in the diagnostic code should
/// use data_race().  This way, KCSAN will complain if buggy lockless
/// accesses to that variable are introduced, even if the buggy accesses
/// are protected by READ_ONCE() or WRITE_ONCE().
///
/// This macro *does not* affect normal code generation, but is a hint
/// to tooling that data races here are to be ignored.  If the access must
/// be atomic *and* KCSAN should ignore the access, use both data_race()
/// and READ_ONCE(), for example, data_race(READ_ONCE(x)).
macro_rules! data_race__4 {
($expr:expr) => {
{
let mut __v: i64;
__kcsan_disable_current();
let __v2: i64 = $expr;
__kcsan_enable_current();
__v = __v2;
__v
}
};
}

/// Macros to retrieve the current execution context:
///
/// in_nmi()		- We're in NMI context
/// in_hardirq()		- We're in hard IRQ context
/// in_serving_softirq()	- We're in softirq context
/// in_task()		- We're in task context
macro_rules! in_nmi {
() => {
nmi_count!()
};
}

macro_rules! irqs_disabled {
() => {
{
let mut __v: i32;
let mut __v2: u64 = 0;
raw_local_save_flags!(__v2);
let mut __v3: i32;
typecheck__3!(u64);
__v3 = arch_irqs_disabled_flags(__v2);
__v = __v3;
__v
}
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

macro_rules! kcsan_release {
() => {
while false {
}
};
}

macro_rules! kcsan_rmb {
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

/// Subtract two ktime_t variables. rem = lhs -rhs:
macro_rules! ktime_sub {
($lhs:expr, $rhs:expr) => {
$lhs.wrapping_sub($rhs)
};
}

macro_rules! likely {
($x:expr) => {
!(!$x) as i64
};
}

macro_rules! local_irq_restore {
($flags:expr) => {
loop {
raw_local_irq_restore!($flags);
if true {
break;
}
}
};
}

macro_rules! local_irq_save__2 {
($flags:expr) => {
loop {
raw_local_irq_save__2!($flags);
if true {
break;
}
}
};
}

macro_rules! lockdep_assert_held {
($l:expr) => {
__assume_ctx_lock!($l);
};
}

macro_rules! lockdep_assert_irqs_disabled {
() => {
while false {
}
};
}

macro_rules! might_sleep {
() => {
loop {
might_resched();
if true {
break;
}
}
};
}

macro_rules! nmi_count {
() => {
(preempt_count() as u64) & __IRQ_MASK!(1i32.wrapping_add(3i32.wrapping_mul(IS_ENABLED__2!()))) << 28i32
};
}

macro_rules! offsetof__106 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, work.work.entry) as u64
};
}

macro_rules! offsetof__140 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, srcu_barrier_head) as u64
};
}

macro_rules! offsetof__141 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, work.work) as u64
};
}

macro_rules! offsetof__169 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, srcu_ec_head) as u64
};
}

macro_rules! offsetof__34 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, irq_work) as u64
};
}

macro_rules! offsetof__15 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, work) as u64
};
}

macro_rules! offsetof__207 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, delay_work) as u64
};
}

macro_rules! per_cpu_offset {
($x:expr) => {
unsafe { __per_cpu_offset[($x) as usize] }
};
}

macro_rules! per_cpu_ptr__4 {
($ptr:expr, $cpu:expr) => {
{
let mut __v: *mut srcu_data;
__verify_pcpu_ptr__43!();
__v = (((($ptr) as u64) as *mut srcu_data) as u64).wrapping_add(per_cpu_offset!($cpu)) as *mut srcu_data;
__v
}
};
}

macro_rules! preempt_count_add__2 {
($val:expr) => {
__preempt_count_add(($val) as i32);
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

macro_rules! raw_check_bogus_irq_restore {
() => {
while false {
}
};
}

macro_rules! raw_local_irq_restore {
($flags:expr) => {
loop {
typecheck!(u64);
raw_check_bogus_irq_restore!();
arch_local_irq_restore($flags);
if true {
break;
}
}
};
}

macro_rules! raw_local_irq_save__2 {
($flags:expr) => {
loop {
typecheck!(u64);
unsafe {
$flags = arch_local_irq_save();
}
if true {
break;
}
}
};
}

macro_rules! raw_local_save_flags {
($flags:expr) => {
loop {
typecheck__3!(u64);
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

macro_rules! raw_spin_lock_irq_rcu_node {
($p:expr) => {
loop {
raw_spin_lock_irq!(unsafe { core::ptr::addr_of_mut!((*$p).lock) });
smp_mb__after_unlock_lock!();
if true {
break;
}
}
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

macro_rules! raw_spin_lock_irqsave__2 {
($lock:expr, $flags:expr) => {
loop {
typecheck!(u64);
unsafe {
$flags = _raw_spin_lock_irqsave($lock);
}
if true {
break;
}
}
};
}

macro_rules! raw_spin_lock_irqsave_rcu_node__2 {
($p:expr, $flags:expr) => {
loop {
raw_spin_lock_irqsave__2!(unsafe { core::ptr::addr_of_mut!((*$p).lock) }, $flags);
smp_mb__after_unlock_lock!();
if true {
break;
}
}
};
}

macro_rules! raw_spin_lock_irqsave_rcu_node {
($p:expr, $flags:expr) => {
loop {
raw_spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*$p).lock) }, $flags);
smp_mb__after_unlock_lock!();
if true {
break;
}
}
};
}

macro_rules! raw_spin_trylock_irqsave {
($lock:expr, $flags:expr) => {
_raw_spin_trylock_irqsave($lock, unsafe { core::ptr::addr_of_mut!($flags) })
};
}

macro_rules! raw_spin_trylock_irqsave_rcu_node {
($p:expr, $flags:expr) => {
{
let mut __v: bool;
let __v2: bool = raw_spin_trylock_irqsave!(unsafe { core::ptr::addr_of_mut!((*$p).lock) }, $flags);
if __v2 {
smp_mb__after_unlock_lock!();
}
__v = __v2;
__v
}
};
}

macro_rules! raw_spin_unlock_irq {
($lock:expr) => {
unsafe { _raw_spin_unlock_irq($lock) };
};
}

macro_rules! raw_spin_unlock_irq_rcu_node {
($p:expr) => {
loop {
lockdep_assert_irqs_disabled!();
raw_spin_unlock_irq!(unsafe { core::ptr::addr_of_mut!((*$p).lock) });
if true {
break;
}
}
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

macro_rules! raw_spin_unlock_irqrestore_rcu_node {
($p:expr, $flags:expr) => {
loop {
lockdep_assert_irqs_disabled!();
raw_spin_unlock_irqrestore!(unsafe { core::ptr::addr_of_mut!((*$p).lock) }, $flags);
if true {
break;
}
}
};
}

/// small_const_nbits(n) is true precisely when it is known at compile-time
/// that BITMAP_SIZE(n) is 1, i.e. 1 <= n <= BITS_PER_LONG. This allows
/// various bit/bitmap APIs to provide a fast inline implementation. Bitmaps
/// of size 0 are very rare, and a compile-time-known-size 0 is most likely
/// a sign of error. They will be handled correctly by the bit/bitmap APIs,
/// but using the out-of-line functions, so that the inline implementations
/// can unconditionally dereference the pointer(s).
macro_rules! small_const_nbits__2 {
($nbits:expr) => {
(unsafe { __slate_intrinsic_f217c4720f04f6db($nbits) }) && ($nbits) <= 64 && ($nbits) > 0
};
}

macro_rules! smp_mb__after_unlock_lock {
() => {
while false {
}
};
}

macro_rules! srcu_lock_sync {
() => {
while false {
}
};
}

/// time_after - returns true if the time a is after time b.
/// @a: first comparable as unsigned long
/// @b: second comparable as unsigned long
///
/// Do this with "<0" and ">=0" to only test the sign of the result. A
/// good compiler would generate better code (and a really good compiler
/// wouldn't care). Gcc is currently neither.
///
/// Return: %true is time a is after time b, otherwise %false.
macro_rules! time_after {
($a:expr, $b:expr) => {
{
let __v: bool = typecheck__2!(u64) != 0 && typecheck__2!(u64) != 0;
__v && ($b.wrapping_sub($a) as i64) < 0
}
};
}

/// time_after_eq - returns true if the time a is after or the same as time b.
/// @a: first comparable as unsigned long
/// @b: second comparable as unsigned long
///
/// Return: %true is time a is after or the same as time b, otherwise %false.
macro_rules! time_after_eq__2 {
($a:expr, $b:expr) => {
{
let __v: bool = typecheck__2!(u64) != 0 && typecheck__2!(u64) != 0;
__v && ($a.wrapping_sub($b) as i64) >= 0
}
};
}

/// time_before - returns true if the time a is before time b.
/// @a: first comparable as unsigned long
/// @b: second comparable as unsigned long
///
/// Return: %true is time a is before time b, otherwise %false.
macro_rules! time_before {
($a:expr, $b:expr) => {
time_after!($b, $a)
};
}

/// time_in_range_open - Calculate whether a is in the range of [b, c).
/// @a: time to test
/// @b: beginning of the range
/// @c: end of the range
///
/// Return: %true is time a is in the range [b, c), otherwise %false.
macro_rules! time_in_range_open {
($a:expr, $b:expr, $c:expr) => {
{
let __v: bool = time_after_eq__2!($a, $b) && time_before!($a, $c);
__v
}
};
}

/// timer_setup - prepare a timer for first use
/// @timer: the timer in question
/// @callback: the function to call when timer expires
/// @flags: any TIMER_* flags
///
/// Regular timer initialization should use either DEFINE_TIMER() above,
/// or timer_setup(). For timers on the stack, timer_setup_on_stack() must
/// be used and must be balanced with a call to timer_destroy_on_stack().
macro_rules! timer_setup {
($timer:expr, $callback:expr, $flags:expr) => {
__timer_init!($timer, $callback, $flags);
};
}

macro_rules! type_max {
() => {
__type_max!()
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
{
let mut __v: i32;
let mut __v2: $type = 0;
let mut __v3: $type = 0;
core::ptr::addr_of_mut!(__v2) == core::ptr::addr_of_mut!(__v3);
__v = 1;
__v
}
};
}

/// Check at compile time that something is of a particular type.
/// Always evaluates to 1 so you may use it easily in comparisons.
macro_rules! typecheck__3 {
($type:ty) => {
let mut __v: i32;
let mut __v2: $type = 0;
let mut __v3: $type = 0;
core::ptr::addr_of_mut!(__v2) == core::ptr::addr_of_mut!(__v3);
__v = 1;
};
}

macro_rules! unlikely {
($x:expr) => {
(($x) != 0) as i64
};
}

macro_rules! unlikely__2 {
($x:expr) => {
!(!$x) as i64
};
}

unsafe extern "C" {
    static mut __per_cpu_offset: [u64; 64];
    static mut this_cpu_off: u64;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut __SCK__WARN_trap: static_call_key;
    static mut kmsg_fops: file_operations;
    static mut __SCK__might_resched: static_call_key;
    static mut nr_cpu_ids: u32;
    static mut __cpu_possible_mask: cpumask;
    static mut __preempt_count: u64;
    static mut __SCK__preempt_schedule: static_call_key;
    static mut cpu_number: i32;
    static mut __boot_cpu_id: i32;
    static mut runqueues: rq;
    static mut rcu_scheduler_active: i32;
    static mut rcu_lock_map: lockdep_map;
    static mut rcu_bh_lock_map: lockdep_map;
    static mut rcu_sched_lock_map: lockdep_map;
    static mut sighand_cachep: *mut kmem_cache;
    static mut jiffies: u64;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut param_ops_int: kernel_param_ops;
    static mut param_ops_ulong: kernel_param_ops;
    static mut rcu_num_lvls: i32;
    static mut num_rcu_lvl: [i32; 0];
    static mut rcu_num_nodes: i32;
    static mut rcu_gp_wq: *mut workqueue_struct;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn __bad_size_call_parameter();
    fn __compiletime_assert_0();
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn _printk(arg0: *const u8, ...) -> i32;
    fn __compiletime_assert_1();
    fn __compiletime_assert_2();
    fn __compiletime_assert_3();
    fn __compiletime_assert_6();
    fn __compiletime_assert_7();
    fn __cmpxchg_wrong_size();
    fn _find_next_bit(arg0: *const u64, arg1: u64, arg2: u64) -> u64;
    fn __cond_resched() -> i32;
    fn __SCT__might_resched() -> i32;
    fn lock_is_held(arg0: *const ()) -> i32;
    fn lockdep_is_held(arg0: *const ()) -> i32;
    fn mutex_init_generic(arg0: *mut mutex);
    fn mutex_lock(arg0: *mut mutex);
    fn mutex_unlock(arg0: *mut mutex);
    fn __local_bh_enable_ip(arg0: u64, arg1: u32);
    fn _raw_spin_lock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irqsave(arg0: *mut raw_spinlock) -> u64;
    fn _raw_spin_trylock(arg0: *mut raw_spinlock) -> i32;
    fn _raw_spin_unlock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irqrestore(arg0: *mut raw_spinlock, arg1: u64);
    fn delayed_work_timer_fn(arg0: *mut timer_list);
    fn ___migrate_enable();
    fn pcpu_alloc_noprof(arg0: u64, arg1: u64, arg2: bool, arg3: u32) -> *mut ();
    fn free_percpu(arg0: *mut ());
    fn synchronize_rcu();
    fn synchronize_rcu_expedited();
    fn __init_swait_queue_head(arg0: *mut swait_queue_head, arg1: *const u8, arg2: *mut lock_class_key);
    fn wait_for_completion(arg0: *mut completion);
    fn complete(arg0: *mut completion);
    fn ktime_get_mono_fast_ns() -> u64;
    fn timer_init_key(arg0: *mut timer_list, arg1: Option<unsafe extern "C" fn(*mut timer_list)>, arg2: u32, arg3: *const u8, arg4: *mut lock_class_key);
    fn timer_reduce(arg0: *mut timer_list, arg1: u64) -> i32;
    fn timer_delete_sync(arg0: *mut timer_list) -> i32;
    fn queue_work_on(arg0: i32, arg1: *mut workqueue_struct, arg2: *mut work_struct) -> bool;
    fn queue_delayed_work_on(arg0: i32, arg1: *mut workqueue_struct, arg2: *mut delayed_work, arg3: u64) -> bool;
    fn flush_work(arg0: *mut work_struct) -> bool;
    fn flush_delayed_work(arg0: *mut delayed_work) -> bool;
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_344();
    fn irq_work_queue(arg0: *mut irq_work) -> bool;
    fn irq_work_sync(arg0: *mut irq_work);
    fn wakeme_after_rcu(arg0: *mut callback_head);
    fn __bad_udelay();
    fn __udelay(arg0: u64);
    fn __const_udelay(arg0: u64);
    fn __compiletime_assert_386();
    fn kfree(arg0: *const ());
    fn kmem_dump_obj(arg0: *mut ()) -> bool;
    fn __compiletime_assert_392();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn register_module_notifier(arg0: *mut notifier_block) -> i32;
    fn __compiletime_assert_622();
    fn __compiletime_assert_623();
    fn __compiletime_assert_628();
    fn __compiletime_assert_629();
    fn __compiletime_assert_630();
    fn __compiletime_assert_632();
    fn __compiletime_assert_633();
    fn rcu_init_geometry();
    fn rcu_gp_is_normal() -> bool;
    fn rcu_gp_is_expedited() -> bool;
    fn rcu_cpu_beenfullyonline(arg0: i32) -> bool;
    fn rcu_cblist_init(arg0: *mut rcu_cblist);
    fn rcu_cblist_dequeue(arg0: *mut rcu_cblist) -> *mut callback_head;
    fn __compiletime_assert_635();
    fn __compiletime_assert_636();
    fn rcu_segcblist_add_len(arg0: *mut rcu_segcblist, arg1: i64);
    fn rcu_segcblist_init(arg0: *mut rcu_segcblist);
    fn rcu_segcblist_ready_cbs(arg0: *mut rcu_segcblist) -> bool;
    fn rcu_segcblist_pend_cbs(arg0: *mut rcu_segcblist) -> bool;
    fn rcu_segcblist_enqueue(arg0: *mut rcu_segcblist, arg1: *mut callback_head);
    fn rcu_segcblist_entrain(arg0: *mut rcu_segcblist, arg1: *mut callback_head) -> bool;
    fn rcu_segcblist_extract_done_cbs(arg0: *mut rcu_segcblist, arg1: *mut rcu_cblist);
    fn srcu_segcblist_advance(arg0: *mut rcu_segcblist, arg1: u64);
    fn srcu_segcblist_accelerate(arg0: *mut rcu_segcblist, arg1: u64) -> bool;
    fn __compiletime_assert_647();
    fn __compiletime_assert_648();
    fn __compiletime_assert_649();
    fn __compiletime_assert_650();
    fn __compiletime_assert_651();
    fn __compiletime_assert_652();
    fn __compiletime_assert_653();
    fn __compiletime_assert_654();
    fn __compiletime_assert_658();
    fn __compiletime_assert_659();
    fn __compiletime_assert_660();
    fn __compiletime_assert_661();
    fn __compiletime_assert_662();
    fn __compiletime_assert_663();
    fn __compiletime_assert_664();
    fn __compiletime_assert_665();
    fn __compiletime_assert_666();
    fn __compiletime_assert_667();
    fn __compiletime_assert_668();
    fn __compiletime_assert_670();
    fn __compiletime_assert_675();
    fn __compiletime_assert_676();
    fn __compiletime_assert_678();
    fn __compiletime_assert_677();
    fn __compiletime_assert_679();
    fn __compiletime_assert_688();
    fn __compiletime_assert_695();
    fn __compiletime_assert_701();
    fn __compiletime_assert_710();
    fn __compiletime_assert_715();
    fn __compiletime_assert_716();
    fn __compiletime_assert_721();
    fn __compiletime_assert_722();
    fn __compiletime_assert_723();
    fn __compiletime_assert_724();
    fn __compiletime_assert_725();
    fn __compiletime_assert_726();
    fn __compiletime_assert_727();
    fn __compiletime_assert_728();
    fn __compiletime_assert_731();
    fn __compiletime_assert_732();
    fn __compiletime_assert_733();
    fn __compiletime_assert_734();
    fn __compiletime_assert_737();
    fn __compiletime_assert_738();
    fn __compiletime_assert_739();
    fn __compiletime_assert_740();
    fn __compiletime_assert_743();
    fn __compiletime_assert_744();
    fn __compiletime_assert_745();
    fn __compiletime_assert_746();
    fn __compiletime_assert_747();
    fn __compiletime_assert_750();
    fn __compiletime_assert_758();
    fn __compiletime_assert_759();
    fn __compiletime_assert_768();
    fn __compiletime_assert_770();
    fn __compiletime_assert_771();
    fn __compiletime_assert_774();
    fn __compiletime_assert_775();
    fn __compiletime_assert_776();
    fn __compiletime_assert_783();
    fn __compiletime_assert_784();
    fn __compiletime_assert_786();
    fn __compiletime_assert_785();
    fn __compiletime_assert_787();
    fn __compiletime_assert_788();
    fn __compiletime_assert_789();
    fn __compiletime_assert_791();
    fn __compiletime_assert_797();
    fn __compiletime_assert_798();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_a39e427d3b6c2df4(arg0: i32) -> bool;
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


fn kasan_check_read(p: *const (), size: u32) -> bool {
true
}

fn kasan_check_write(p: *const (), size: u32) -> bool {
true
}

fn __kcsan_check_access(ptr: *const (), size: u64, r#type: i32) {
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
#[inline(always)]
fn __must_check_overflow(overflow: bool) -> bool {
unlikely__2!(overflow) != 0
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
__verify_pcpu_ptr__2!();
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
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct lockdep_map {
}


#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_find_h_67 {
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
struct bug_entry {
    bug_addr_disp: i32,
    format_disp: i32,
    file_disp: i32,
    line: u16,
    flags: u16,
}


// LIST_HEAD - definition of a &struct list_head with initialization values
// @name: name of the list_head
fn INIT_LIST_HEAD(list: *mut list_head) {
loop {
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_1() };
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
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_2() };
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
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_3() };
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

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


fn __list_del(prev: *mut list_head, next: *mut list_head) {
unsafe {
(*next).prev = prev;
}
loop {
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_6() };
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

fn list_del_init(entry: *mut list_head) {
__list_del_entry(entry);
INIT_LIST_HEAD(entry);
}

fn list_empty(head: *const list_head) -> i32 {
let mut __v0: *mut list_head;
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_7() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*head).next) as *const *mut list_head) };
(__v0 == (head as *mut list_head)) as i32
}

// list_entry - get the struct for this entry
// @ptr:	the &struct list_head pointer.
// @type:	the type of the struct this is embedded in.
// @member:	the name of the list_head within the struct.
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
#[repr(C)]
#[derive(Clone, Copy)]
struct pollfd {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct spinlock {
    __slate_anon_0: __SlateRecord_spinlock_types_h_18,
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
struct rwlock {
    raw_lock: qrwlock,
}


#[inline(always)]
fn arch_atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
__WRITE_ONCE__3!((*v).counter, i);
}

#[inline(always)]
fn arch_atomic_inc(mut v: *mut __SlateRecord_types_h_190) {
unsafe {
core::arch::asm!("lock incl ({0:r})", in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct io_bitmap {
}


#[inline(always)]
fn arch_atomic_dec(mut v: *mut __SlateRecord_types_h_190) {
unsafe {
core::arch::asm!("lock decl ({0:r})", in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
}
}

#[inline(always)]
fn arch_atomic_dec_and_test(mut v: *mut __SlateRecord_types_h_190) -> bool {
let mut __v0: bool;
let mut c: bool = false;
unsafe {
let __t0: u8;
core::arch::asm!("lock decl ({0:r})\nsete {1}", in(reg) core::ptr::addr_of_mut!((*v).counter), lateout(reg_byte) __t0, options(nostack, att_syntax));
c = (__t0 as u8) != 0;
}
__v0 = c;
__v0
}

#[inline(always)]
fn arch_atomic64_read(v: *const __SlateRecord_types_h_197) -> i64 {
__READ_ONCE__2!((*v).counter)
}

#[inline(always)]
fn raw_atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
arch_atomic_set(v, i);
}

#[inline(always)]
fn raw_atomic_inc(v: *mut __SlateRecord_types_h_190) {
arch_atomic_inc(v);
}

#[inline(always)]
fn raw_atomic_dec(v: *mut __SlateRecord_types_h_190) {
arch_atomic_dec(v);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
}


#[inline(always)]
fn raw_atomic_dec_and_test(v: *mut __SlateRecord_types_h_190) -> bool {
arch_atomic_dec_and_test(v)
}

#[inline(always)]
fn raw_atomic64_read(v: *const __SlateRecord_types_h_197) -> i64 {
arch_atomic64_read(v)
}

#[inline(always)]
fn raw_atomic_long_read(v: *const __SlateRecord_types_h_197) -> i64 {
raw_atomic64_read(v)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
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
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("38: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 38b - ., 3; .popsection\n# {0}", const 38 as i32, options(nostack, att_syntax));
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
core::arch::asm!("39: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 39b - ., 4; .popsection\n# {0}", const 39 as i32, options(nostack, att_syntax));
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

// instrument_get_user() - add instrumentation to get_user()-like macros
// @to: destination variable, may not be address-taken
//
// get_user() and friends are fragile, so it may depend on the implementation
// whether the instrumentation happens before or after the data is copied from
// the userspace.
#[repr(C)]
#[derive(Clone, Copy)]
struct futex_pi_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct robust_list_head {
}


#[inline(always)]
fn atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
instrument_atomic_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_set(v, i);
}

#[inline(always)]
fn atomic_inc(v: *mut __SlateRecord_types_h_190) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_inc(v);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct futex_private_hash {
}


#[inline(always)]
fn atomic_dec(v: *mut __SlateRecord_types_h_190) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_dec(v);
}

#[inline(always)]
fn atomic_dec_and_test(v: *mut __SlateRecord_types_h_190) -> bool {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_dec_and_test(v)
}

#[inline(always)]
fn atomic_long_read(v: *const __SlateRecord_types_h_197) -> i64 {
instrument_atomic_read(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_read(v)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pid_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sem_undo_list {
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

// ffz - find first zero bit in word
// @word: The word to search
//
// Undefined if no zero exists, so code should check against ~0UL first.
#[inline(always)]
fn find_next_bit(addr: *const u64, size: u64, offset: u64) -> u64 {
if small_const_nbits__2!(size) {
let mut val: u64 = 0;
if unlikely__2!(offset >= size) != 0 {
return size;
}
val = (unsafe { *addr }) & GENMASK!(size.wrapping_sub(1), offset);
return if val != 0 { __ffs!(val) } else { size };
}
unsafe { _find_next_bit(addr, size, offset) }
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
struct cpumask {
    bits: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_nodemask_types_h_17 {
    bits: [u64; 1],
}


#[inline(always)]
fn might_resched() {
let mut __v0: Option<unsafe extern "C" fn() -> i32>;
__v0 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn() -> i32>>(__SCT__might_resched as *const ()) };
unsafe { __v0.unwrap()() };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__might_resched_42: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__might_resched) }) as *mut ();

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
struct hrtimer_clock_base {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __call_single_node {
    llist: llist_node,
    __slate_anon_1: __SlateRecord_smp_types_h_60,
    src: u16,
    dst: u16,
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

#[repr(C)]
#[derive(Clone, Copy)]
struct seccomp_filter {
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
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_preempt_h_496 {
    lock: *mut (),
}


#[inline(always)]
fn class_preempt_destructor(_T: *mut __SlateRecord_preempt_h_496) {
loop {
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if unlikely__2!(preempt_count_dec_and_test!()) != 0 {
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
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_61: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

#[inline(always)]
fn class_preempt_constructor() -> __SlateRecord_preempt_h_496 {
let _t: __SlateRecord_preempt_h_496 = __SlateRecord_preempt_h_496 { lock: 1i32 as *mut () };
loop {
preempt_count_inc!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
return _t;
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
struct __kernel_timespec {
    tv_sec: i64,
    tv_nsec: i64,
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
struct ucounts {
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
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits140U0,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct orc_entry {
    sp_offset: i16,
    bp_offset: i16,
    __slate_bits_0: __slate_bits::__SlateBits141U0,
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
fn native_irq_disable() {
unsafe {
core::arch::asm!("cli", options(nostack, att_syntax));
}
}

#[inline(always)]
fn native_irq_enable() {
unsafe {
core::arch::asm!("sti", options(nostack, att_syntax));
}
}

#[inline(always)]
fn arch_local_save_flags() -> u64 {
native_save_fl()
}

#[inline(always)]
fn arch_local_irq_disable() {
native_irq_disable();
}

#[inline(always)]
fn arch_local_irq_enable() {
native_irq_enable();
}

#[repr(C)]
#[derive(Clone, Copy)]
struct workqueue_struct {
}


#[inline(always)]
fn arch_local_irq_save() -> u64 {
let flags: u64 = arch_local_save_flags();
arch_local_irq_disable();
flags
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
}


#[inline(always)]
fn arch_irqs_disabled_flags(flags: u64) -> i32 {
(flags & _BITUL!(9i32) == 0) as i32
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


#[inline(always)]
fn arch_local_irq_restore(flags: u64) {
if arch_irqs_disabled_flags(flags) == 0 {
arch_local_irq_enable();
}
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
    __slate_bits_0: __slate_bits::__SlateBits142U0,
    pkru: u32,
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
fn get_boot_cpu_id() -> i32 {
unsafe { __boot_cpu_id }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct optimistic_spin_queue {
    tail: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mutex {
    owner: __SlateRecord_types_h_197,
    wait_lock: raw_spinlock,
    osq: optimistic_spin_queue,
    first_waiter: *mut mutex_waiter,
}


fn __mutex_init(lock: *mut mutex, name: *const u8, key: *mut lock_class_key) {
unsafe { mutex_init_generic(lock) };
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
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
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
struct futex_unlock_cs_range {
    start_ip: u64,
    len: u32,
    pop_size32: u32,
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
struct futex_unlock_cs_ranges {
    cs_ranges: [futex_unlock_cs_range; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seq_file {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_mm_data {
    phash: futex_mm_phash,
    unlock: futex_unlock_cs_ranges,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sysv_sem {
    undo_list: *mut sem_undo_list,
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
struct rq_flags {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct timerqueue_linked_node {
    node: rb_node_linked,
    expires: i64,
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
struct gendisk {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_siginfo {
    __slate_anon_0: __SlateRecord_signal_types_h_14,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_struct {
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
struct kmem_cache {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct k_sigaction {
    sa: sigaction,
}


#[inline(always)]
fn __local_bh_disable_ip(ip: u64, cnt: u32) {
preempt_count_add__2!(cnt);
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


fn local_bh_disable() {
let mut __v0: u64;
let mut __here: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("lea 0(%rip), {0:r}", lateout(reg) __t0, options(pure, nomem, nostack, att_syntax));
__here = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
__v0 = __here;
__local_bh_disable_ip(__v0, 512u32);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
}


fn local_bh_enable() {
let mut __v0: u64;
let mut __here: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("lea 0(%rip), {0:r}", lateout(reg) __t0, options(pure, nomem, nostack, att_syntax));
__here = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
__v0 = __here;
unsafe { __local_bh_enable_ip(__v0, 512u32) };
}

#[inline(always)]
fn _raw_spin_trylock_irqsave(lock: *mut raw_spinlock, flags: *mut u64) -> bool {
local_irq_save__2!(*flags);
if (unsafe { _raw_spin_trylock(lock) }) != 0 {
return true;
}
local_irq_restore!(unsafe { *flags });
false
}

#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe {
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
struct uprobe_xol_ops {
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
struct rseq_data {
    usrptr: *mut rseq,
    len: u32,
    sig: u32,
    event: rseq_event,
    ids: rseq_ids,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
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
struct ldt_struct {
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
struct mem_cgroup {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqlock {
    seqcount: seqcount_spinlock,
    lock: spinlock,
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
struct dev_pagemap {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_operations_struct {
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
struct user_namespace {
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
struct mnt_idmap {
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
    __slate_bits_0: __slate_bits::__SlateBits269U0,
    dl_timer: hrtimer,
    inactive_timer: hrtimer,
    rq: *mut rq,
    server_pick_task: Option<unsafe extern "C" fn(*mut sched_dl_entity, *mut rq_flags) -> *mut task_struct>,
    pi_se: *mut sched_dl_entity,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array_ptr {
}


#[repr(C)]
#[derive(Clone, Copy)]
union rcu_special {
    b: __SlateRecord_sched_h_791,
    s: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct key_type {
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
    __slate_bits_0: __slate_bits::__SlateBits67U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits67U1,
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
    __slate_bits_2: __slate_bits::__SlateBits67U2,
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
union __SlateRecord_cred_h_147 {
    non_rcu: i32,
    rcu: callback_head,
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
struct pidfs_attr {
}


fn __migrate_enable() {
let p: *mut task_struct = get_current();
if ((unsafe { (*p).migration_disabled }) as i32) > 1 {
unsafe {
(*p).migration_disabled = ((*p).migration_disabled as i32).wrapping_sub(1) as u16;
}
return;
}
// Ensure stop_task runs either before or after this, and that
// __set_cpus_allowed_ptr(SCA_MIGRATE_ENABLE) doesn't schedule().
let mut __UNIQUE_ID_guard_101: __SlateRecord_preempt_h_496 = class_preempt_constructor();
let __slate_cleanup_ptr_3 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_101) as *mut _;
let __slate_cleanup_3 = ::slate_support::Guard(&mut || class_preempt_destructor(__slate_cleanup_ptr_3 as *mut _));
if unlikely__2!((unsafe { (*p).cpus_ptr }) != ((unsafe { core::ptr::addr_of_mut!((*p).cpus_mask) }) as *const cpumask)) != 0 {
unsafe { ___migrate_enable() };
}
// Mustn't clear migration_disabled() until cpus_ptr points back at the
// regular cpus_mask, otherwise things that race (eg.
// select_fallback_rq) get confused.
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
unsafe {
(*p).migration_disabled = 0;
}
let mut __v4: *mut rq;
let mut tcp_ptr__: u64 = 0;
let mut __v5: u64;
let mut pfo_val__: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t0, sym this_cpu_off, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
__v5 = pfo_val__;
tcp_ptr__ = __v5;
let __v7: u64 = tcp_ptr__.wrapping_add((unsafe { core::ptr::addr_of_mut!(runqueues) }) as u64);
tcp_ptr__ = __v7;
__v4 = tcp_ptr__ as *mut rq;
let __v8: *mut u32 = ((unsafe { (__v4 as *const u8).wrapping_offset(3312 as isize) }) as *mut ()) as *mut u32;
let __v9: u32 = unsafe { *__v8 };
unsafe {
*__v8 = __v9.wrapping_sub(1);
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pid_h_62 {
    ino: u64,
    pidfs_hash: rhash_head,
    stashed: *mut dentry,
    attr: *mut pidfs_attr,
}


fn __migrate_disable() {
let p: *mut task_struct = get_current();
if (unsafe { (*p).migration_disabled }) != 0 {
unsafe {
(*p).migration_disabled = ((*p).migration_disabled as i32).wrapping_add(1) as u16;
}
return;
}
let mut __UNIQUE_ID_guard_102: __SlateRecord_preempt_h_496 = class_preempt_constructor();
let __slate_cleanup_ptr_3 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_102) as *mut _;
let __slate_cleanup_3 = ::slate_support::Guard(&mut || class_preempt_destructor(__slate_cleanup_ptr_3 as *mut _));
let mut __v4: *mut rq;
let mut tcp_ptr__: u64 = 0;
let mut __v5: u64;
let mut pfo_val__: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t0, sym this_cpu_off, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
__v5 = pfo_val__;
tcp_ptr__ = __v5;
let __v7: u64 = tcp_ptr__.wrapping_add((unsafe { core::ptr::addr_of_mut!(runqueues) }) as u64);
tcp_ptr__ = __v7;
__v4 = tcp_ptr__ as *mut rq;
let __v8: *mut u32 = ((unsafe { (__v4 as *const u8).wrapping_offset(3312 as isize) }) as *mut ()) as *mut u32;
let __v9: u32 = unsafe { *__v8 };
unsafe {
*__v8 = __v9.wrapping_add(1);
}
unsafe {
(*p).migration_disabled = 1;
}
}

#[inline(always)]
fn migrate_disable() {
__migrate_disable();
}

#[inline(always)]
fn migrate_enable() {
__migrate_enable();
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rcu_gp_seq {
    norm: u64,
    exp: u64,
}


fn init_rcu_head_on_stack(head: &callback_head) {
}

fn destroy_rcu_head_on_stack(head: &callback_head) {
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
struct tty_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct taskstats {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_audit_buf {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kref {
    refcount: refcount_struct,
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
struct folio_queue {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wait_queue_head {
    lock: spinlock,
    head: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swait_queue_head {
    lock: raw_spinlock,
    task_list: list_head,
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
struct bio_vec {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct completion {
    done: u32,
    wait: swait_queue_head,
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
struct sock {
}


fn init_completion(x: *mut completion) {
unsafe {
(*x).done = 0;
}
loop {
unsafe { __init_swait_queue_head(core::ptr::addr_of_mut!((*x).wait), c"&x->wait".as_ptr() as *const u8, core::ptr::addr_of_mut!(__key__87)) };
if true {
break;
}
}
}

static mut __key__87: lock_class_key = unsafe { core::mem::zeroed() };

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


fn __init_work(work: *mut work_struct, onstack: i32) {
}

fn queue_work(wq: *mut workqueue_struct, work: *mut work_struct) -> bool {
unsafe { queue_work_on(64i32, wq, work) }
}

fn queue_delayed_work(wq: *mut workqueue_struct, dwork: *mut delayed_work, delay: u64) -> bool {
unsafe { queue_delayed_work_on(64i32, wq, dwork, delay) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rcu_cblist {
    head: *mut callback_head,
    tail: *mut *mut callback_head,
    len: i64,
}


// Index values for segments in rcu_segcblist structure.
//
// The segments are as follows:
//
// [head, *tails[RCU_DONE_TAIL]):
// 	Callbacks whose grace period has elapsed, and thus can be invoked.
// [*tails[RCU_DONE_TAIL], *tails[RCU_WAIT_TAIL]):
// 	Callbacks waiting for the current GP from the current CPU's viewpoint.
// [*tails[RCU_WAIT_TAIL], *tails[RCU_NEXT_READY_TAIL]):
// 	Callbacks that arrived before the next GP started, again from
// 	the current CPU's viewpoint.  These can be handled by the next GP.
// [*tails[RCU_NEXT_READY_TAIL], *tails[RCU_NEXT_TAIL]):
// 	Callbacks that might have arrived after the next GP started.
// 	There is some uncertainty as to when a given GP starts and
// 	ends, but a CPU knows the exact times if it is the one starting
// 	or ending the GP.  Other CPUs know that the previous GP ends
// 	before the next one starts.
//
// Note that RCU_WAIT_TAIL cannot be empty unless RCU_NEXT_READY_TAIL is also
// empty.
//
// The ->gp_seq[] array contains the grace-period state at which the
// corresponding segment of callbacks will be ready to invoke.  This tracks
// both normal and expedited grace periods, allowing callbacks to complete
// when either type of GP finishes.  A given element of this array is
// meaningful only when the corresponding segment is non-empty, and it is
// never valid for RCU_DONE_TAIL (whose callbacks are already ready to
// invoke) or for RCU_NEXT_TAIL (whose callbacks have not yet been assigned
// a grace-period state).
// Also RCU_WAIT head.
//                      ==NOCB De-Offloading state machine==
//
//
// |--------------------------------------------------------------------------|
// |                           SEGCBLIST_LOCKING    |                         |
// |                           SEGCBLIST_OFFLOADED  |                         |
// |                           SEGCBLIST_KTHREAD_GP                           |
// |                           + unparked CB kthread                          |
// |                                                                          |
// |   CB/GP kthreads handle callbacks holding nocb_lock, local rcu_core()    |
// |   ignores callbacks. Bypass enqueue is enabled.                          |
//                                     |
//                                     v
// |--------------------------------------------------------------------------|
// |                           SEGCBLIST_RCU_CORE   |                         |
// |                           SEGCBLIST_LOCKING    |                         |
// |                           SEGCBLIST_OFFLOADED  |                         |
// |                           SEGCBLIST_KTHREAD_GP                           |
// |                           + unparked CB kthread                          |
// |                                                                          |
// |   CB/GP kthreads handle callbacks holding nocb_lock, local rcu_core()    |
// |   handles callbacks concurrently. Bypass enqueue is disabled.            |
// |   Invoke RCU core so we make sure not to preempt it in the middle with   |
// |   leaving some urgent work unattended within a jiffy.                    |
//                                     |
//                                     v
// |--------------------------------------------------------------------------|
// |                           SEGCBLIST_RCU_CORE   |                         |
// |                           SEGCBLIST_LOCKING    |                         |
// |                           SEGCBLIST_KTHREAD_GP                           |
// |                           + unparked CB kthread                          |
// |                                                                          |
// |   CB/GP kthreads and local rcu_core() handle callbacks concurrently      |
// |   holding nocb_lock. Wake up GP kthread if necessary.                    |
//                                     |
//                                     v
// |--------------------------------------------------------------------------|
// |                           SEGCBLIST_RCU_CORE   |                         |
// |                           SEGCBLIST_LOCKING    |                         |
// |                           + unparked CB kthread                          |
// |                                                                          |
// |   GP kthread woke up and acknowledged the fact that SEGCBLIST_OFFLOADED  |
// |   got cleared. The callbacks from the target CPU will be ignored from the|
// |   GP kthread loop.                                                       |
//                                     |
//                                     v
// |                SEGCBLIST_RCU_CORE | SEGCBLIST_LOCKING                    |
// |                          + parked CB kthread                             |
// |                                                                          |
// | CB kthread is parked. Callbacks processed by rcu_core() from softirqs or |
// | local rcuc kthread, while holding nocb_lock.                             |
//                                     |
//                                     v
// |                         SEGCBLIST_RCU_CORE                               |
// |                                                                          |
// |  Callbacks processed by rcu_core() from softirqs or local                |
// |  rcuc kthread, without holding nocb_lock.                                |
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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


// Values for size state variable (->srcu_size_state).  Once the state
//
// has been set to SRCU_SIZE_ALLOC, the grace-period code advances through
//
// this state machine one step per grace period until the SRCU_SIZE_BIG state
//
// is reached.  Otherwise, the state machine remains in the SRCU_SIZE_SMALL
//
// state indefinitely.
// No srcu_node combining tree, ->node == NULL
// An srcu_node tree is being allocated, initialized,
//  and then referenced by ->node.  It will not be used.
// The srcu_node tree starts being used by everything
//  except call_srcu(), especially by srcu_barrier().
//
//  By the end of this state, all CPUs and threads
//
//  are aware of this tree's existence.
// The srcu_node tree starts being used by call_srcu().
// The srcu_node combining tree is fully initialized
// Values for state variable (bottom bits of ->srcu_gp_seq).
// Values for srcu_expedite_current() state (->srcu_ec_state).
// Values for initializing gp sequence fields. Higher values allow wrap arounds to
// occur earlier.
// The second value with state is useful in the case of static initialization of
// srcu_usage where srcu_gp_seq_needed is expected to have some state value in its
// lower bits (or else it will appear to be already initialized within
// the call check_init_srcu_struct()).
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.srcutree.__srcu_read_lock")]
extern "C" fn __srcu_read_lock(ssp: *mut srcu_struct) -> i32 {
let mut scp: *mut srcu_ctr = unsafe { core::mem::zeroed() };
let mut __v0: *mut srcu_ctr;
loop {
if !(__native_word__10!() || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_710() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*ssp).srcu_ctrp) as *const *mut srcu_ctr) };
scp = __v0;
loop {
__verify_pcpu_ptr__4!();
match (const { core::mem::size_of::<i64>() }) as u64 {
1 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incb %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decb %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u8 = 1u8;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addb {1:l}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
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
2 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incw %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decw %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u16 = 1u16;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addw {1:x}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, in(reg) pto_val__ as u16, options(nostack, att_syntax));
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
4 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incl %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decl %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u32 = 1u32;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addl {1:e}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, in(reg) pto_val__ as u32, options(nostack, att_syntax));
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
8 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incq %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decq %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u64 = 1;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addq {1:r}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_locks.counter) as u64) as *mut i64, in(reg) pto_val__ as u64, options(nostack, att_syntax));
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
_ => {
unsafe { __bad_size_call_parameter() };
}
}
if true {
break;
}
}
loop {
kcsan_mb!();
unsafe {
core::arch::asm!("lock addl $0,-4(%rsp)", options(nostack, att_syntax));
}
if true {
break;
}
}
// B
//
// Avoid leaking the critical section.
__srcu_ptr_to_ctr(ssp, scp) as i32
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.srcutree.srcu_barrier")]
extern "C" fn srcu_barrier(ssp: *mut srcu_struct) {
let mut cpu: i32 = 0;
let mut idx: i32 = 0;
let s: u64 = rcu_seq_snap(unsafe { core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_barrier_seq) });
check_init_srcu_struct(ssp);
unsafe { mutex_lock(core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_barrier_mutex)) };
if rcu_seq_done(unsafe { core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_barrier_seq) }, s) {
loop {
kcsan_mb!();
unsafe {
core::arch::asm!("lock addl $0,-4(%rsp)", options(nostack, att_syntax));
}
if true {
break;
}
}
// Force ordering following return.
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_barrier_mutex)) };
return; // Someone else did our work for us.
}
rcu_seq_start(unsafe { core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_barrier_seq) });
init_completion(unsafe { core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_barrier_completion) });
// Initial count prevents reaching zero until all CBs are posted.
atomic_set(unsafe { core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_barrier_cpu_cnt) }, 1i32);
idx = __srcu_read_lock_nmisafe(ssp);
let mut __v2: i32;
let mut ___p1: i32 = 0;
let mut __v3: i32;
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_758() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_size_state) as *const i32) };
___p1 = __v3;
loop {
if !__native_word__5!() {
unsafe { __compiletime_assert_759() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
__v2 = ___p1;
if __v2 < 2 {
srcu_barrier_one_cpu(ssp, per_cpu_ptr__4!(unsafe { (*ssp).sda }, get_boot_cpu_id()));
} else {
cpu = 0;
loop {
cpu = find_next_bit(cpumask_bits!((unsafe { core::ptr::addr_of_mut!(__cpu_possible_mask) }) as *const cpumask), 64u64, cpu as u64) as i32;
if (cpu as u32) >= 64 {
break;
}
srcu_barrier_one_cpu(ssp, per_cpu_ptr__4!(unsafe { (*ssp).sda }, cpu));
cpu = cpu.wrapping_add(1);
}
}
__srcu_read_unlock_nmisafe(ssp, idx);
// Remove the initial count, at which point reaching zero can happen.
if atomic_dec_and_test(unsafe { core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_barrier_cpu_cnt) }) {
unsafe { complete(core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_barrier_completion)) };
}
unsafe { wait_for_completion(core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_barrier_completion)) };
rcu_seq_end(unsafe { core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_barrier_seq) });
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_barrier_mutex)) };
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.srcutree.srcu_torture_stats_print")]
extern "C" fn srcu_torture_stats_print(ssp: *mut srcu_struct, tt: *mut u8, tf: *mut u8) {
let mut cpu: i32 = 0;
let mut idx: i32 = 0;
let mut s0: u64 = 0;
let mut s1: u64 = 0;
let mut ss_state: i32 = 0;
let mut __v0: i32;
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_791() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_size_state) as *const i32) };
ss_state = __v0;
let mut ss_state_idx: i32 = ss_state;
idx = ((unsafe { unsafe { (*ssp).srcu_ctrp }.offset_from(((*(*ssp).sda).srcu_ctrs.as_mut_ptr() as *mut srcu_ctr).wrapping_offset(0 as isize) as *mut srcu_ctr) }) as i64) as i32;
if ss_state < 0 || (ss_state as u64) >= ARRAY_SIZE__68!() {
ss_state_idx = ARRAY_SIZE__114!().wrapping_sub(1) as i32;
}
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x011rcu: %s%s Tree SRCU g%ld state %d (%s)".as_ptr() as *const u8, tt, tf, rcu_seq_current(core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_gp_seq)), ss_state, srcu_size_state_name.0[ss_state_idx as usize]) };
if unsafe { (*ssp).sda }.is_null() {
// Called after cleanup_srcu_struct(), perhaps.
let mut __v4: i32;
while false {
}
__v4 = unsafe { _printk(c"\x01c No per-CPU srcu_data structures (->sda == NULL).\n".as_ptr() as *const u8) };
} else {
let mut __v6: i32;
while false {
}
__v6 = unsafe { _printk(c"\x01c per-CPU(idx=%d):".as_ptr() as *const u8, idx) };
cpu = 0;
loop {
cpu = find_next_bit(cpumask_bits!((unsafe { core::ptr::addr_of_mut!(__cpu_possible_mask) }) as *const cpumask), 64u64, cpu as u64) as i32;
if (cpu as u32) >= 64 {
break;
}
let mut l0: u64 = 0;
let mut l1: u64 = 0;
let mut u0: u64 = 0;
let mut u1: u64 = 0;
let mut c0: i64 = 0;
let mut c1: i64 = 0;
let mut sdp: *mut srcu_data = unsafe { core::mem::zeroed() };
sdp = per_cpu_ptr__4!(unsafe { (*ssp).sda }, cpu);
u0 = data_race__4!(atomic_long_read((unsafe { core::ptr::addr_of_mut!((*((*sdp).srcu_ctrs.as_mut_ptr() as *mut srcu_ctr).wrapping_offset(((idx == 0) as i32) as isize)).srcu_unlocks) }) as *const __SlateRecord_types_h_197)) as u64;
u1 = data_race__4!(atomic_long_read((unsafe { core::ptr::addr_of_mut!((*((*sdp).srcu_ctrs.as_mut_ptr() as *mut srcu_ctr).wrapping_offset(idx as isize)).srcu_unlocks) }) as *const __SlateRecord_types_h_197)) as u64;
// Make sure that a lock is always counted if the corresponding
// unlock is counted.
loop {
kcsan_rmb!();
loop {
kcsan_rmb!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
if true {
break;
}
}
l0 = data_race__4!(atomic_long_read((unsafe { core::ptr::addr_of_mut!((*((*sdp).srcu_ctrs.as_mut_ptr() as *mut srcu_ctr).wrapping_offset(((idx == 0) as i32) as isize)).srcu_locks) }) as *const __SlateRecord_types_h_197)) as u64;
l1 = data_race__4!(atomic_long_read((unsafe { core::ptr::addr_of_mut!((*((*sdp).srcu_ctrs.as_mut_ptr() as *mut srcu_ctr).wrapping_offset(idx as isize)).srcu_locks) }) as *const __SlateRecord_types_h_197)) as u64;
c0 = l0.wrapping_sub(u0) as i64;
c1 = l1.wrapping_sub(u1) as i64;
let mut __v23: i32;
while false {
}
__v23 = unsafe { _printk(c"\x01c %d(%ld,%ld %c)".as_ptr() as *const u8, cpu, c0, c1, *(c"C.".as_ptr() as *mut u8).wrapping_offset((rcu_segcblist_empty(core::ptr::addr_of_mut!((*sdp).srcu_cblist)) as i32) as isize) as i32) };
s0 = s0.wrapping_add(c0 as u64);
s1 = s1.wrapping_add(c1 as u64);
cpu = cpu.wrapping_add(1);
}
let mut __v32: i32;
while false {
}
__v32 = unsafe { _printk(c"\x01c T(%ld,%ld)\n".as_ptr() as *const u8, s0, s1) };
}
if SRCU_SIZING_IS_TORTURE!() {
srcu_transition_to_big(ssp);
}
}

fn __srcu_ptr_to_ctr(ssp: *mut srcu_struct, scpp: *mut srcu_ctr) -> bool {
((unsafe { scpp.offset_from(((*(*ssp).sda).srcu_ctrs.as_mut_ptr() as *mut srcu_ctr).wrapping_offset(0 as isize) as *mut srcu_ctr) }) as i64) != 0
}

fn __srcu_ctr_to_ptr(ssp: &srcu_struct, idx: i32) -> *mut srcu_ctr {
unsafe { (*(*(ssp as *const srcu_struct)).sda).srcu_ctrs.as_mut_ptr() as *mut srcu_ctr }.wrapping_offset(idx as isize)
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.srcutree.__srcu_check_read_flavor")]
extern "C" fn __srcu_check_read_flavor(ssp: *mut srcu_struct, read_flavor: i32) {
let mut old_read_flavor: i32 = 0;
let mut sdp: *mut srcu_data = unsafe { core::mem::zeroed() };
// NMI-unsafe use in NMI is a bad sign, as is multi-bit read_flavor values.
let mut __v0: i64;
{
let mut __ret_warn_on: i32 = 0;
let __v1: bool = read_flavor != 2 && read_flavor != 4 && in_nmi!() != 0;
__ret_warn_on = __v1 as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("697: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 697b - ., 3; .popsection\n# {0}", const 697 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___srcu_check_read_flavor_2, sym __slate_asm_string___srcu_check_read_flavor_3, const 747 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("698: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 698b - ., 4; .popsection\n# {0}", const 698 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
}
let mut __v6: i64;
{
let __ret_warn_on: i32 = (read_flavor & read_flavor.wrapping_sub(1) != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("699: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 699b - ., 3; .popsection\n# {0}", const 699 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___srcu_check_read_flavor_7, sym __slate_asm_string___srcu_check_read_flavor_8, const 748 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("700: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 700b - ., 4; .popsection\n# {0}", const 700 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v6 = (__ret_warn_on != 0) as i64;
}
let mut __v11: *mut srcu_data;
__verify_pcpu_ptr__27!();
let mut __v13: *mut srcu_data;
let mut tcp_ptr__: u64 = 0;
let mut __v14: u64;
let mut pfo_val__: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t0, sym this_cpu_off, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
__v14 = pfo_val__;
tcp_ptr__ = __v14;
tcp_ptr__ = tcp_ptr__.wrapping_add((unsafe { (*ssp).sda }) as u64);
__v13 = tcp_ptr__ as *mut srcu_data;
__v11 = __v13;
sdp = __v11;
let mut __v17: i32;
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_701() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v17 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*sdp).srcu_reader_flavor) as *const i32) };
old_read_flavor = __v17;
let mut __v19: i64;
{
let __ret_warn_on: i32 = ((unsafe { (*ssp).srcu_reader_flavor }) != 0 && read_flavor != ((unsafe { (*ssp).srcu_reader_flavor }) as i32)) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("702: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 702b - ., 3; .popsection\n# {0}", const 702 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___srcu_check_read_flavor_20, sym __slate_asm_string___srcu_check_read_flavor_21, const 752 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("703: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 703b - ., 4; .popsection\n# {0}", const 703 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v19 = (__ret_warn_on != 0) as i64;
}
let mut __v24: i64;
{
let __ret_warn_on: i32 = (old_read_flavor != 0 && (unsafe { (*ssp).srcu_reader_flavor }) != 0 && old_read_flavor != ((unsafe { (*ssp).srcu_reader_flavor }) as i32)) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("704: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 704b - ., 3; .popsection\n# {0}", const 704 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___srcu_check_read_flavor_25, sym __slate_asm_string___srcu_check_read_flavor_26, const 754 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("705: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 705b - ., 4; .popsection\n# {0}", const 705 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v24 = (__ret_warn_on != 0) as i64;
}
let mut __v29: i64;
{
let __ret_warn_on: i32 = (read_flavor == 4 && (unsafe { (*ssp).srcu_reader_flavor }) == 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("706: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 706b - ., 3; .popsection\n# {0}", const 706 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___srcu_check_read_flavor_30, sym __slate_asm_string___srcu_check_read_flavor_31, const 755 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("707: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 707b - ., 4; .popsection\n# {0}", const 707 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v29 = (__ret_warn_on != 0) as i64;
}
if old_read_flavor == 0 {
let mut __v34: i32;
let __ai_ptr: *mut i32 = unsafe { core::ptr::addr_of_mut!((*sdp).srcu_reader_flavor) };
kcsan_mb!();
instrument_atomic_read_write(__ai_ptr as *const (), core::mem::size_of::<i32>() as u64);
let mut __v36: i32;
let mut __ret: i32 = 0;
let mut __old: i32 = 0;
let mut __new: i32 = read_flavor;
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __ptr: *mut u8 = __ai_ptr as *mut u8;
unsafe {
let __t1: u32;
core::arch::asm!("lock cmpxchgb {1:e}, ({0:r})", in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t1, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t1 as u32) };
}
}
2 => {
let mut __ptr: *mut u16 = __ai_ptr as *mut u16;
unsafe {
let __t2: u32;
core::arch::asm!("lock cmpxchgw {1:e}, ({0:r})", in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t2, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t2 as u32) };
}
}
4 => {
let mut __ptr: *mut u32 = __ai_ptr as *mut u32;
unsafe {
let __t3: u32;
core::arch::asm!("lock cmpxchgl {1:e}, ({0:r})", in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t3, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t3 as u32) };
}
}
8 => {
let mut __ptr: *mut u64 = __ai_ptr as *mut u64;
unsafe {
let __t4: u32;
core::arch::asm!("lock cmpxchgq {1:e}, ({0:r})", in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t4, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t4 as u32) };
}
}
_ => {
unsafe { __cmpxchg_wrong_size() };
}
}
__v36 = __ret;
__v34 = __v36;
old_read_flavor = __v34;
if old_read_flavor == 0 {
return;
}
}
let mut __v38: i32;
let __ret_warn_on: i32 = (old_read_flavor != read_flavor) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
2i32 | BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"CPU %d old state %d new state %d\n".as_ptr() as *const u8, (*sdp).cpu, old_read_flavor, read_flavor) };
let mut __v39: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v39 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v40: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t5: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t5, sym __slate_asm_string___srcu_check_read_flavor_41, sym __slate_asm_string___srcu_check_read_flavor_42, const 761 as i32, const 2323 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t5 as u64) };
}
__v40 = bug;
unsafe { __v39.unwrap()(__v40, (*sdp).cpu, old_read_flavor, read_flavor) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v38 = __ret_warn_on;
}

fn srcu_readers_active(ssp: *mut srcu_struct) -> bool {
let mut cpu: i32 = 0;
let mut sum: u64 = 0;
loop {
cpu = find_next_bit(cpumask_bits!((unsafe { core::ptr::addr_of_mut!(__cpu_possible_mask) }) as *const cpumask), 64u64, cpu as u64) as i32;
if (cpu as u32) >= 64 {
break;
}
let sdp: *mut srcu_data = per_cpu_ptr__4!(unsafe { (*ssp).sda }, cpu);
let __v3: i64 = atomic_long_read((unsafe { core::ptr::addr_of_mut!((*((*sdp).srcu_ctrs.as_mut_ptr() as *mut srcu_ctr).wrapping_offset(0 as isize)).srcu_locks) }) as *const __SlateRecord_types_h_197);
sum = sum.wrapping_add(__v3 as u64);
let __v6: i64 = atomic_long_read((unsafe { core::ptr::addr_of_mut!((*((*sdp).srcu_ctrs.as_mut_ptr() as *mut srcu_ctr).wrapping_offset(1 as isize)).srcu_locks) }) as *const __SlateRecord_types_h_197);
sum = sum.wrapping_add(__v6 as u64);
let __v9: i64 = atomic_long_read((unsafe { core::ptr::addr_of_mut!((*((*sdp).srcu_ctrs.as_mut_ptr() as *mut srcu_ctr).wrapping_offset(0 as isize)).srcu_unlocks) }) as *const __SlateRecord_types_h_197);
sum = sum.wrapping_sub(__v9 as u64);
let __v12: i64 = atomic_long_read((unsafe { core::ptr::addr_of_mut!((*((*sdp).srcu_ctrs.as_mut_ptr() as *mut srcu_ctr).wrapping_offset(1 as isize)).srcu_unlocks) }) as *const __SlateRecord_types_h_197);
sum = sum.wrapping_sub(__v12 as u64);
cpu = cpu.wrapping_add(1);
}
sum != 0
}

// srcu_read_lock().
// srcu_read_lock_nmisafe().
// 					0x4		// SRCU-lite is no longer with us.
// srcu_read_lock_fast(), also NMI-safe.
// srcu_read_lock_fast_updown().
// All of the above.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.srcutree.cleanup_srcu_struct")]
extern "C" fn cleanup_srcu_struct(ssp: *mut srcu_struct) {
let mut cpu: i32 = 0;
let mut delay: u64 = 0;
let sup: *mut srcu_usage = unsafe { (*ssp).srcu_sup };
raw_spin_lock_irq_rcu_node!(unsafe { (*ssp).srcu_sup });
delay = srcu_get_delay(ssp);
raw_spin_unlock_irq_rcu_node!(unsafe { (*ssp).srcu_sup });
let mut __v4: i64;
{
let __ret_warn_on: i32 = (delay == 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("680: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 680b - ., 3; .popsection\n# {0}", const 680 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_cleanup_srcu_struct_5, sym __slate_asm_string_cleanup_srcu_struct_6, const 694 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("681: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 681b - ., 4; .popsection\n# {0}", const 681 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v4 = (__ret_warn_on != 0) as i64;
}
if __v4 != 0 {
return;
}
// Just leak it!
let mut __v9: i64;
{
let __ret_warn_on: i32 = !(!srcu_readers_active(ssp)) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("682: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 682b - ., 3; .popsection\n# {0}", const 682 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_cleanup_srcu_struct_10, sym __slate_asm_string_cleanup_srcu_struct_11, const 696 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("683: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 683b - ., 4; .popsection\n# {0}", const 683 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v9 = (__ret_warn_on != 0) as i64;
}
if __v9 != 0 {
return;
}
// Just leak it!
// Wait for irq_work to finish first as it may queue a new work.
unsafe { irq_work_sync(core::ptr::addr_of_mut!((*sup).irq_work)) };
unsafe { flush_delayed_work(core::ptr::addr_of_mut!((*sup).work)) };
loop {
cpu = find_next_bit(cpumask_bits!((unsafe { core::ptr::addr_of_mut!(__cpu_possible_mask) }) as *const cpumask), 64u64, cpu as u64) as i32;
if (cpu as u32) >= 64 {
break;
}
let sdp: *mut srcu_data = per_cpu_ptr__4!(unsafe { (*ssp).sda }, cpu);
// Call srcu_barrier() before this cleanup_srcu_struct()
//
// to avoid triggering this WARN_ON().
let mut __v17: i64;
{
let mut __ret_warn_on: i32 = 0;
let __v18: bool = (unsafe { timer_delete_sync(core::ptr::addr_of_mut!((*sdp).delay_work)) }) != 0 && rcu_segcblist_n_cbs(unsafe { core::ptr::addr_of_mut!((*sdp).srcu_cblist) }) != 0;
__ret_warn_on = __v18 as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("684: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 684b - ., 3; .popsection\n# {0}", const 684 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_cleanup_srcu_struct_19, sym __slate_asm_string_cleanup_srcu_struct_20, const 707 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("685: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 685b - ., 4; .popsection\n# {0}", const 685 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v17 = (__ret_warn_on != 0) as i64;
}
let __v23: bool = __v17 != 0 && unsafe { rcu_cpu_beenfullyonline((*sdp).cpu) };
if __v23 {
unsafe { queue_work_on((*sdp).cpu, rcu_gp_wq, core::ptr::addr_of_mut!((*sdp).work)) };
}
unsafe { flush_work(core::ptr::addr_of_mut!((*sdp).work)) };
let mut __v24: i64;
let __ret_warn_on: i32 = (rcu_segcblist_n_cbs(unsafe { core::ptr::addr_of_mut!((*sdp).srcu_cblist) }) != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("686: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 686b - ., 3; .popsection\n# {0}", const 686 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_cleanup_srcu_struct_25, sym __slate_asm_string_cleanup_srcu_struct_26, const 711 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("687: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 687b - ., 4; .popsection\n# {0}", const 687 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v24 = (__ret_warn_on != 0) as i64;
if __v24 != 0 {
return;
}
// Forgot srcu_barrier(), so just leak it!
cpu = cpu.wrapping_add(1);
}
let mut __v32: i64;
{
let mut __ret_warn_on: i32 = 0;
let mut __v33: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_688() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v33 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*sup).srcu_gp_seq) as *const u64) };
__ret_warn_on = (rcu_seq_state(__v33) != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("689: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 689b - ., 3; .popsection\n# {0}", const 689 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_cleanup_srcu_struct_35, sym __slate_asm_string_cleanup_srcu_struct_36, const 714 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("690: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 690b - ., 4; .popsection\n# {0}", const 690 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v32 = (__ret_warn_on != 0) as i64;
}
let __v39: bool = if __v32 != 0 {
true
} else {
let mut __v40: i64;
let __ret_warn_on: i32 = (rcu_seq_current(unsafe { core::ptr::addr_of_mut!((*sup).srcu_gp_seq) }) != unsafe { (*sup).srcu_gp_seq_needed }) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("691: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 691b - ., 3; .popsection\n# {0}", const 691 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_cleanup_srcu_struct_41, sym __slate_asm_string_cleanup_srcu_struct_42, const 715 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("692: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 692b - ., 4; .popsection\n# {0}", const 692 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v40 = (__ret_warn_on != 0) as i64;
__v40 != 0
};
let __v45: bool = if __v39 {
true
} else {
let mut __v46: i64;
let __ret_warn_on: i32 = !(!srcu_readers_active(ssp)) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("693: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 693b - ., 3; .popsection\n# {0}", const 693 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_cleanup_srcu_struct_47, sym __slate_asm_string_cleanup_srcu_struct_48, const 716 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("694: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 694b - ., 4; .popsection\n# {0}", const 694 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v46 = (__ret_warn_on != 0) as i64;
__v46 != 0
};
if __v45 {
let mut __v51: i32;
while false {
}
let mut __v53: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_695() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v53 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*sup).srcu_gp_seq) as *const u64) };
__v51 = unsafe { _printk(c"\x016rcu: %s: Active srcu_struct %p read state: %d gp state: %lu/%lu\n".as_ptr() as *const u8, c"cleanup_srcu_struct".as_ptr() as *mut u8, ssp, rcu_seq_state(__v53), rcu_seq_current(core::ptr::addr_of_mut!((*sup).srcu_gp_seq)), (*sup).srcu_gp_seq_needed) };
return; // Caller forgot to stop doing call_srcu()?
// Or caller invoked start_poll_synchronize_srcu()
//
// and then cleanup_srcu_struct() before that grace
//
// period ended?
}
unsafe { kfree((*sup).node as *const ()) };
unsafe {
(*sup).node = core::ptr::null_mut::<srcu_node>();
}
unsafe {
(*sup).srcu_size_state = 0;
}
if !unsafe { (*sup).sda_is_static } {
unsafe { free_percpu((*ssp).sda as *mut ()) };
unsafe {
(*ssp).sda = core::ptr::null_mut::<srcu_data>();
}
unsafe { kfree(sup as *const ()) };
unsafe {
(*ssp).srcu_sup = core::ptr::null_mut::<srcu_usage>();
}
}
}

fn __srcu_read_lock_nmisafe(ssp: *mut srcu_struct) -> i32 {
__srcu_read_lock(ssp)
}

fn __srcu_read_unlock_nmisafe(ssp: *mut srcu_struct, idx: i32) {
__srcu_read_unlock(ssp, idx);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct backing_dev_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct block_device {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct notifier_block {
    notifier_call: Option<unsafe extern "C" fn(*mut notifier_block, u64, *mut ()) -> i32>,
    next: *mut notifier_block,
    priority: i32,
}


// Don't care
// Suits me
// Don't call further
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
struct writeback_control {
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
struct hprobe {
    state: u32,
    srcu_scp: *mut srcu_ctr,
    uprobe: *mut uprobe,
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
struct uprobes_state {
    xol_area: *mut xol_area,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_context {
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
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_288 {
    val: u64,
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
struct poll_table_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_info_struct {
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
struct __SlateRecord_mm_types_h_701 {
    v: u64,
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
struct readahead_control {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wait_page_queue {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_userfaultfd_ctx {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_acl {
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
    __slate_bits_0: __slate_bits::__SlateBits393U0,
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
struct cdev {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_cmd {
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
struct exception_table_entry {
    insn: i32,
    fixup: i32,
    data: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ns_common {
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
struct __SlateRecord_capability_h_24 {
    val: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
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
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
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
struct rhash_head {
    next: *mut rhash_head,
}


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
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
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
struct module_param_attrs {
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
    __slate_bits_0: __slate_bits::__SlateBits272U0,
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
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_344() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*((*(*task).signal).rlim.as_mut_ptr() as *mut rlimit).wrapping_offset(limit as isize)).rlim_cur) as *const u64) };
__v0
}

fn rlimit(limit: u32) -> u64 {
task_rlimit(get_current() as *const task_struct, limit)
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


fn init_irq_work(work: *mut irq_work, func: Option<unsafe extern "C" fn(*mut irq_work)>) {
let mut __slate_compound_0: irq_work = unsafe { core::mem::zeroed() };
unsafe {
*work = *{
__slate_compound_0 = irq_work { node: {
let mut __t0: __call_single_node = core::mem::zeroed();
__t0.__slate_anon_1 = __SlateRecord_smp_types_h_60 { u_flags: 0 };
__t0
}, func: func, irqwait: rcuwait { task: core::ptr::null_mut::<task_struct>() } };
core::ptr::addr_of_mut!(__slate_compound_0)
};
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_37 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_37_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_37_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_37_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_37_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_37_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_41 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_41_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_41_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_41_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_41_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_41_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_62 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_62_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_62_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_62_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_62_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_62_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_66 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_66_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_66_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_66_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_66_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_66_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_70 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_70_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_70_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_70_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_70_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_70_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_156 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_157 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_158 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_614 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_614_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_614_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_614_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_614_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_614_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_640 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_640_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_640_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_640_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_640_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_640_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_647 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_647_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_647_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_647_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_647_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_647_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_935 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_1045 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_2022 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcutree_c_2023 {
}


#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

#[repr(transparent)]
struct __SlateConst<T>(T);

unsafe impl<T> Sync for __SlateConst<T> {
}

mod __slate_bits {
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits269U0 {
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
    pub struct __SlateBits609U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits553U0 {
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
    pub struct __SlateBits141U0 {
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
    pub struct __SlateBits393U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits67U0 {
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
    pub struct __SlateBits67U1 {
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
    pub struct __SlateBits272U0 {
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
    pub struct __SlateBits67U2 {
        #[bits(1)]
        pub mce_ripv: u64,
        #[bits(1)]
        pub mce_whole_page: u64,
        #[bits(62)]
        pub __mce_reserved: u64,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits140U0 {
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
    pub struct __SlateBits142U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rcu_synchronize {
    head: callback_head,
    completion: completion,
    oldstate: rcu_gp_seq,
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
struct iovec {
    iov_base: *mut (),
    iov_len: u64,
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
struct xarray {
    xa_lock: spinlock,
    xa_flags: u32,
    xa_head: *mut (),
}


// We encode errnos in the xas->xa_node.  If an error has happened, we need to
// drop the lock to fix it, and once we've done so the xa_state is invalid.
#[repr(C)]
#[derive(Clone, Copy)]
struct list_lru_one {
    list: list_head,
    nr_items: i64,
    lock: spinlock,
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
struct __SlateRecord_uuid_h_21 {
    b: [u8; 16],
}


// The length of a UUID string ("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
// not including trailing NUL.
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
struct percpu_rw_semaphore {
    rss: rcu_sync,
    read_count: *mut u32,
    writer: rcuwait,
    waiters: wait_queue_head,
    block: __SlateRecord_types_h_190,
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
struct lockref {
    __slate_anon_0: __SlateRecord_lockref_h_26,
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
struct io_context {
    refcount: __SlateRecord_types_h_197,
    active_ref: __SlateRecord_types_h_190,
    ioprio: u16,
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
if !(__native_word__7!() || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_386() };
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
unsafe { __compiletime_assert_392() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
} else {
loop {
unsafe {
core::arch::asm!("393: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 393b - ., 3; .popsection\n# {0}", const 393 as i32, options(nostack, att_syntax));
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
struct offset_ctx {
    mt: maple_tree,
    next_offset: u64,
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
struct kernfs_elem_dir {
    subdirs: u64,
    children: rb_root,
    root: *mut kernfs_root,
    rev: u64,
}


core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_init_srcu_struct_generic: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_init_srcu_struct_fast: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_init_srcu_struct_fast_updown: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_cleanup_srcu_struct: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol___srcu_check_read_flavor: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol___srcu_read_lock: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol___srcu_read_unlock: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_call_srcu: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {7} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_synchronize_srcu_expedited: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {8} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_synchronize_srcu: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {9} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_get_state_synchronize_srcu: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {10} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_start_poll_synchronize_srcu: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {11} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_poll_state_synchronize_srcu: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {12} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_srcu_barrier: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {13} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_srcu_expedite_current: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {14} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_srcu_batches_completed: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {15} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_srcutorture_get_gp_data: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {16} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_srcu_torture_stats_print: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {17} ; .previous\n.section\t\".initcallearly.init\", \"a\"\t\t\n__initcall__kmod_srcutree__793_2078_srcu_bootup_announceearly:\t\t\t\n.long\t{18} - .\t\n.previous\t\t\t\t\t\n\n.section\t\".initcall7.init\", \"a\"\t\t\n__initcall__kmod_srcutree__803_2182_init_srcu_module_notifier7:\t\t\t\n.long\t{19} - .\t\n.previous\t\t\t\t\t\n", sym init_srcu_struct_generic, sym init_srcu_struct_fast, sym init_srcu_struct_fast_updown, sym cleanup_srcu_struct, sym __srcu_check_read_flavor, sym __srcu_read_lock, sym __srcu_read_unlock, sym call_srcu, sym synchronize_srcu_expedited, sym synchronize_srcu, sym get_state_synchronize_srcu, sym start_poll_synchronize_srcu, sym poll_state_synchronize_srcu, sym srcu_barrier, sym srcu_expedite_current, sym srcu_batches_completed, sym srcutorture_get_gp_data, sym srcu_torture_stats_print, sym srcu_bootup_announce, sym init_srcu_module_notifier, options(att_syntax));

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
    __slate_bits_0: __slate_bits::__SlateBits609U0,
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
struct attribute {
    name: *const u8,
    mode: u16,
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


// Permissions on a sysfs file: you didn't miss the 0 prefix did you?
// USER_READABLE >= GROUP_READABLE >= OTHER_READABLE
// USER_WRITABLE >= GROUP_WRITABLE
// OTHER_WRITABLE?  Generally considered a bad idea.
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
    __slate_bits_0: __slate_bits::__SlateBits553U0,
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
struct kset {
    list: list_head,
    list_lock: spinlock,
    kobj: kobject,
    uevent_ops: *const kset_uevent_ops,
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
struct kparam_string {
    maxlen: u32,
    string: *mut u8,
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


// module_param - typesafe helper for a module/cmdline parameter
// @name: the variable to alter, and exposed parameter name.
// @type: the type of the parameter
// @perm: visibility in sysfs.
//
// @name becomes the module parameter, or (prefixed by KBUILD_MODNAME and a
// ".") the kernel commandline parameter.  Note that - is changed to _, so
// the user can use "foo-bar=1" even for variable "foo_bar".
//
// @perm is 0 if the variable is not to appear in sysfs, or 0444
// for world-readable, 0644 for root-writable, etc.  Note that if it
// is writable, you may need to use kernel_param_lock() around
// accesses (esp. charp, which can be kfreed when it changes).
//
// The @type is simply pasted to refer to a param_ops_##type and a
// param_check_##type: for convenience many standard types are provided but
// you can create your own by defining those variables.
//
// Standard types are:
// 	byte, hexint, short, ushort, int, uint, long, ulong
// 	charp: a character pointer
// 	bool: a bool, values 0/1, y/n, Y/N.
// 	invbool: the above, only sense-reversed (N = true).
// module_param_named - typesafe helper for a renamed module/cmdline parameter
// @name: a valid C identifier which is the parameter name.
// @value: the actual lvalue to alter.
// @type: the type of the parameter
// @perm: visibility in sysfs.
//
// Usually it's a good idea to have variable names and user-exposed names the
// same, but that's harder if the variable must be non-static or is inside a
// structure.  This allows exposure under a different name.
// module_param_cb - general callback for a module/cmdline parameter
// @name: a valid C identifier which is the parameter name.
// @ops: the set & get operations for this parameter.
// @arg: args for @ops
// @perm: visibility in sysfs.
//
// The ops can have NULL set or get functions.
// This is the fundamental function for registering boot/module parameters.
// All the helper functions
//
// The macros to do compile-time type checking stolen from Jakub
// Jelinek, who IIRC came up with this idea for the 2.4 module init code.
#[repr(C)]
#[derive(Clone, Copy)]
struct latch_tree_node {
    node: [rb_node; 2],
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
fn rcu_seq_ctr(s: u64) -> u64 {
s >> 2i32
}

fn rcu_seq_state(s: u64) -> i32 {
(s & 3u64) as i32
}

fn rcu_seq_set_state(sp: *mut u64, newstate: i32) {
let mut __v0: i64;
let __ret_warn_on: i32 = (newstate & -4i32 != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("620: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 620b - ., 3; .popsection\n# {0}", const 620 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_rcu_seq_set_state_1, sym __slate_asm_string_rcu_seq_set_state_2, const 98 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("621: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 621b - ., 4; .popsection\n# {0}", const 621 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_622() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__2!(*sp, ((unsafe { *sp }) & 18446744073709551612u64).wrapping_add(newstate as u64));
if true {
break;
}
}
}

fn rcu_seq_start(sp: *mut u64) {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_623() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__2!(*sp, unsafe { *sp }.wrapping_add(1));
if true {
break;
}
}
loop {
kcsan_mb!();
unsafe {
core::arch::asm!("lock addl $0,-4(%rsp)", options(nostack, att_syntax));
}
if true {
break;
}
}
// Ensure update-side operation after counter increment.
let mut __v5: i64;
let __ret_warn_on: i32 = (rcu_seq_state(unsafe { *sp }) != 1) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("624: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 624b - ., 3; .popsection\n# {0}", const 624 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_rcu_seq_start_6, sym __slate_asm_string_rcu_seq_start_7, const 107 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("625: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 625b - ., 4; .popsection\n# {0}", const 625 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v5 = (__ret_warn_on != 0) as i64;
}

fn rcu_seq_endval(sp: *mut u64) -> u64 {
((unsafe { *sp }) | 3u64).wrapping_add(1)
}

fn rcu_seq_end(sp: *mut u64) {
loop {
kcsan_mb!();
unsafe {
core::arch::asm!("lock addl $0,-4(%rsp)", options(nostack, att_syntax));
}
if true {
break;
}
}
// Ensure update-side operation before counter increment.
let mut __v2: i64;
let __ret_warn_on: i32 = (rcu_seq_state(unsafe { *sp }) == 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("626: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 626b - ., 3; .popsection\n# {0}", const 626 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_rcu_seq_end_3, sym __slate_asm_string_rcu_seq_end_4, const 120 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("627: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 627b - ., 4; .popsection\n# {0}", const 627 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v2 = (__ret_warn_on != 0) as i64;
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_628() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__2!(*sp, rcu_seq_endval(sp));
if true {
break;
}
}
}

fn rcu_seq_snap(sp: *mut u64) -> u64 {
let mut s: u64 = 0;
let mut __v0: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_629() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(sp as *const u64) };
s = __v0.wrapping_add(6u64).wrapping_add(1) & 18446744073709551612u64;
loop {
kcsan_mb!();
unsafe {
core::arch::asm!("lock addl $0,-4(%rsp)", options(nostack, att_syntax));
}
if true {
break;
}
}
// Above access must not bleed into critical section.
s
}

fn rcu_seq_current(sp: *mut u64) -> u64 {
let mut __v0: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_630() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(sp as *const u64) };
__v0
}

fn rcu_seq_done(sp: *mut u64, s: u64) -> bool {
let mut __v0: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_632() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(sp as *const u64) };
!0 / 2 >= __v0.wrapping_sub(s)
}

fn rcu_seq_done_exact(sp: *mut u64, s: u64) -> bool {
let mut cur_s: u64 = 0;
let mut __v0: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_633() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(sp as *const u64) };
cur_s = __v0;
ULONG_CMP_GE!(cur_s, s) || ULONG_CMP_LT!(cur_s, s.wrapping_sub(8u64))
}

fn debug_rcu_head_queue(head: *mut callback_head) -> i32 {
0
}

fn debug_rcu_head_unqueue(head: *mut callback_head) {
}

fn debug_rcu_head_callback(rhp: *mut callback_head) {
if unlikely__2!((unsafe { (*rhp).func }) == None) != 0 {
unsafe { kmem_dump_obj(rhp as *mut ()) };
}
}

static mut rcu_fanout_exact: bool = unsafe { core::mem::zeroed() };

static mut rcu_fanout_leaf: i32 = unsafe { core::mem::zeroed() };

fn rcu_init_levelspread(levelspread: *mut i32, levelcnt: *const i32) {
let mut i: i32 = 0;
while i < 2 {
unsafe {
*levelspread.wrapping_offset(i as isize) = 2147483647i32.wrapping_neg().wrapping_sub(1);
}
i = i.wrapping_add(1);
}
if unsafe { rcu_fanout_exact } {
unsafe {
*levelspread.wrapping_offset(unsafe { rcu_num_lvls }.wrapping_sub(1) as isize) = rcu_fanout_leaf;
}
i = unsafe { rcu_num_lvls }.wrapping_sub(2);
while i >= 0 {
unsafe {
*levelspread.wrapping_offset(i as isize) = 64;
}
i = i.wrapping_sub(1);
}
} else {
let mut ccur: i32 = 0;
let mut cprv: i32 = (unsafe { nr_cpu_ids }) as i32;
i = unsafe { rcu_num_lvls }.wrapping_sub(1);
while i >= 0 {
ccur = unsafe { *levelcnt.wrapping_offset(i as isize) };
unsafe {
*levelspread.wrapping_offset(i as isize) = cprv.wrapping_add(ccur).wrapping_sub(1) / ccur;
}
cprv = ccur;
i = i.wrapping_sub(1);
}
}
}

// Do a full breadth-first scan of the {s,}rcu_node structures for the
// specified state structure (for SRCU) or the only rcu_state structure
// (for RCU).
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.srcutree.srcu_batches_completed")]
extern "C" fn srcu_batches_completed(ssp: *mut srcu_struct) -> u64 {
let mut __v0: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_768() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_gp_seq) as *const u64) };
__v0
}

fn rcu_segcblist_empty(rsclp: *mut rcu_segcblist) -> bool {
let mut __v0: *mut callback_head;
loop {
if !(__native_word__24!() || ((const { core::mem::size_of::<*mut callback_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_635() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*rsclp).head) as *const *mut callback_head) };
__v0.is_null()
}

fn rcu_segcblist_n_cbs(rsclp: *mut rcu_segcblist) -> i64 {
let mut __v0: i64;
loop {
if !(__native_word__14!() || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_636() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*rsclp).len) as *const i64) };
__v0
}

fn rcu_segcblist_segempty(rsclp: *mut rcu_segcblist, seg: i32) -> bool {
if seg == 0 {
return (unsafe { core::ptr::addr_of_mut!((*rsclp).head) }) == unsafe { *((*rsclp).tails.as_mut_ptr() as *mut *mut *mut callback_head).wrapping_offset(0 as isize) };
}
(unsafe { *((*rsclp).tails.as_mut_ptr() as *mut *mut *mut callback_head).wrapping_offset(seg.wrapping_sub(1) as isize) }) == unsafe { *((*rsclp).tails.as_mut_ptr() as *mut *mut *mut callback_head).wrapping_offset(seg as isize) }
}

// Holdoff in nanoseconds for auto-expediting.
static mut exp_holdoff: u64 = 25000;

fn __check_exp_holdoff() -> *mut u64 {
unsafe { core::ptr::addr_of_mut!(exp_holdoff) }
}

static __param_str_exp_holdoff: __SlateAlign16<[u8; 21]> = __SlateAlign16([115u8, 114u8, 99u8, 117u8, 116u8, 114u8, 101u8, 101u8, 46u8, 101u8, 120u8, 112u8, 95u8, 104u8, 111u8, 108u8, 100u8, 111u8, 102u8, 102u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_exp_holdoff: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_exp_holdoff.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_ulong) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_srcutree_c_37>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_37_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_37_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_37_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_37_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_37_6>() as u64) as i32).wrapping_add(292i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(exp_holdoff) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_642: [u8; 36] = [115, 114, 99, 117, 116, 114, 101, 101, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 101, 120, 112, 95, 104, 111, 108, 100, 111, 102, 102, 58, 117, 108, 111, 110, 103, 0];

/// Overflow-check frequency.  N bits roughly says every 2**N grace periods.
static mut counter_wrap_check: u64 = 4611686018427387903;

fn __check_counter_wrap_check() -> *mut u64 {
unsafe { core::ptr::addr_of_mut!(counter_wrap_check) }
}

static __param_str_counter_wrap_check: __SlateAlign16<[u8; 28]> = __SlateAlign16([115u8, 114u8, 99u8, 117u8, 116u8, 114u8, 101u8, 101u8, 46u8, 99u8, 111u8, 117u8, 110u8, 116u8, 101u8, 114u8, 95u8, 119u8, 114u8, 97u8, 112u8, 95u8, 99u8, 104u8, 101u8, 99u8, 107u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_counter_wrap_check: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_counter_wrap_check.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_ulong) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_srcutree_c_41>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_41_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_41_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_41_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_41_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_41_6>() as u64) as i32).wrapping_add(292i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(counter_wrap_check) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_643: [u8; 43] = [115, 114, 99, 117, 116, 114, 101, 101, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 99, 111, 117, 110, 116, 101, 114, 95, 119, 114, 97, 112, 95, 99, 104, 101, 99, 107, 58, 117, 108, 111, 110, 103, 0];

// Control conversion to SRCU_SIZE_BIG:
//    0: Don't convert at all.
//    1: Convert at init_srcu_struct() time.
//    2: Convert when rcutorture invokes srcu_torture_stats_print().
//    3: Decide at boot time based on system shape (default).
// 0x1x: Convert when excessive contention encountered.
static mut convert_to_big: i32 = 3;

fn __check_convert_to_big() -> *mut i32 {
unsafe { core::ptr::addr_of_mut!(convert_to_big) }
}

static __param_str_convert_to_big: __SlateAlign16<[u8; 24]> = __SlateAlign16([115u8, 114u8, 99u8, 117u8, 116u8, 114u8, 101u8, 101u8, 46u8, 99u8, 111u8, 110u8, 118u8, 101u8, 114u8, 116u8, 95u8, 116u8, 111u8, 95u8, 98u8, 105u8, 103u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_convert_to_big: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_convert_to_big.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_int) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_srcutree_c_62>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_62_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_62_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_62_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_62_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_62_6>() as u64) as i32).wrapping_add(292i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(convert_to_big) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_644: [u8; 37] = [115, 114, 99, 117, 116, 114, 101, 101, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 99, 111, 110, 118, 101, 114, 116, 95, 116, 111, 95, 98, 105, 103, 58, 105, 110, 116, 0];

/// Number of CPUs to trigger init_srcu_struct()-time transition to big.
#[unsafe(link_section = ".data..read_mostly")]
static mut big_cpu_lim: i32 = 128;

fn __check_big_cpu_lim() -> *mut i32 {
unsafe { core::ptr::addr_of_mut!(big_cpu_lim) }
}

static __param_str_big_cpu_lim: __SlateAlign16<[u8; 21]> = __SlateAlign16([115u8, 114u8, 99u8, 117u8, 116u8, 114u8, 101u8, 101u8, 46u8, 98u8, 105u8, 103u8, 95u8, 99u8, 112u8, 117u8, 95u8, 108u8, 105u8, 109u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_big_cpu_lim: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_big_cpu_lim.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_int) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_srcutree_c_66>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_66_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_66_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_66_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_66_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_66_6>() as u64) as i32).wrapping_add(292i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(big_cpu_lim) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_645: [u8; 34] = [115, 114, 99, 117, 116, 114, 101, 101, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 98, 105, 103, 95, 99, 112, 117, 95, 108, 105, 109, 58, 105, 110, 116, 0];

/// Contention events per jiffy to initiate transition to big.
#[unsafe(link_section = ".data..read_mostly")]
static mut small_contention_lim: i32 = 100;

fn __check_small_contention_lim() -> *mut i32 {
unsafe { core::ptr::addr_of_mut!(small_contention_lim) }
}

static __param_str_small_contention_lim: __SlateAlign16<[u8; 30]> = __SlateAlign16([115u8, 114u8, 99u8, 117u8, 116u8, 114u8, 101u8, 101u8, 46u8, 115u8, 109u8, 97u8, 108u8, 108u8, 95u8, 99u8, 111u8, 110u8, 116u8, 101u8, 110u8, 116u8, 105u8, 111u8, 110u8, 95u8, 108u8, 105u8, 109u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_small_contention_lim: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_small_contention_lim.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_int) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_srcutree_c_70>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_70_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_70_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_70_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_70_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_70_6>() as u64) as i32).wrapping_add(292i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(small_contention_lim) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_646: [u8; 43] = [115, 114, 99, 117, 116, 114, 101, 101, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 115, 109, 97, 108, 108, 95, 99, 111, 110, 116, 101, 110, 116, 105, 111, 110, 95, 108, 105, 109, 58, 105, 110, 116, 0];

/// Early-boot callback-management, so early that no lock is required!
static mut srcu_boot_list: list_head = list_head { next: unsafe { core::ptr::addr_of_mut!(srcu_boot_list) }, prev: unsafe { core::ptr::addr_of_mut!(srcu_boot_list) } };

#[unsafe(link_section = ".data..read_mostly")]
static mut srcu_init_done: bool = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".text.slate_distinct.srcutree.process_srcu")]
extern "C" fn process_srcu(work: *mut work_struct) {
let mut curdelay: u64 = 0;
let mut j: u64 = 0;
let mut sup: *mut srcu_usage = unsafe { core::mem::zeroed() };
let mut __v0: *mut srcu_usage;
__v0 = ((unsafe { (work as *const u8).wrapping_offset(-(offsetof__141!(srcu_usage) as isize)) }) as *mut ()) as *mut srcu_usage;
sup = __v0;
let ssp: *mut srcu_struct = unsafe { (*sup).srcu_ssp };
srcu_advance_state(ssp);
raw_spin_lock_irq_rcu_node!(unsafe { (*ssp).srcu_sup });
curdelay = srcu_get_delay(ssp);
raw_spin_unlock_irq_rcu_node!(unsafe { (*ssp).srcu_sup });
if curdelay != 0 {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_783() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__13!((*sup).reschedule_count, 0i32);
if true {
break;
}
}
} else {
j = unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) };
let mut __v8: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_784() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v8 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*sup).reschedule_jiffies) as *const u64) };
if __v8 == j {
ASSERT_EXCLUSIVE_WRITER__2!((*sup).reschedule_count);
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_786() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
let mut __v11: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_785() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v11 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*sup).reschedule_count) as *const u64) };
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!((*sup).reschedule_count) as *mut u64, __v11.wrapping_add(1)) };
if true {
break;
}
}
if true {
break;
}
}
let mut __v15: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_787() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v15 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*sup).reschedule_count) as *const u64) };
if __v15 > unsafe { srcu_max_nodelay } {
curdelay = 1;
}
} else {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_788() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__13!((*sup).reschedule_count, 1i32);
if true {
break;
}
}
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_789() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__2!((*sup).reschedule_jiffies, j);
if true {
break;
}
}
}
}
srcu_reschedule(ssp, curdelay);
}

/// Initialize SRCU per-CPU data.  Note that statically allocated
/// srcu_struct structures might already have srcu_read_lock() and
/// srcu_read_unlock() running against them.  So if the is_static
/// parameter is set, don't initialize ->srcu_ctrs[].srcu_locks and
/// ->srcu_ctrs[].srcu_unlocks.
fn init_srcu_struct_data(ssp: *mut srcu_struct) {
let mut __slate_compound_3: raw_spinlock = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_types_h_197 = unsafe { core::mem::zeroed() };
let mut cpu: i32 = 0;
let mut sdp: *mut srcu_data = unsafe { core::mem::zeroed() };
// Initialize the per-CPU srcu_data array, which feeds into the
// leaves of the srcu_node tree.
'__slate_break_8: loop {
cpu = find_next_bit(cpumask_bits!((unsafe { core::ptr::addr_of_mut!(__cpu_possible_mask) }) as *const cpumask), 64u64, cpu as u64) as i32;
if (cpu as u32) >= 64 {
break;
}
sdp = per_cpu_ptr__4!(unsafe { (*ssp).sda }, cpu);
'__slate_break_4: loop {
unsafe {
*core::ptr::addr_of_mut!((*sdp).lock) = *{
__slate_compound_3 = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_qspinlock_types_h_44 = core::mem::zeroed();
__t0.val = __SlateRecord_types_h_190 { counter: 0 };
__t0
} } };
core::ptr::addr_of_mut!(__slate_compound_3)
};
}
if true {
break;
}
}
unsafe { rcu_segcblist_init(core::ptr::addr_of_mut!((*sdp).srcu_cblist)) };
unsafe {
(*sdp).srcu_cblist_invoking = false;
}
unsafe {
(*sdp).srcu_gp_seq_needed = (*(*ssp).srcu_sup).srcu_gp_seq;
}
unsafe {
(*sdp).srcu_gp_seq_needed_exp = (*(*ssp).srcu_sup).srcu_gp_seq;
}
unsafe {
(*sdp).srcu_barrier_head.next = core::ptr::addr_of_mut!((*sdp).srcu_barrier_head);
}
unsafe {
(*sdp).mynode = core::ptr::null_mut::<srcu_node>();
}
unsafe {
(*sdp).cpu = cpu;
}
'__slate_break_7: loop {
'__slate_break_6: loop {
__init_work(unsafe { core::ptr::addr_of_mut!((*sdp).work) }, 0i32);
unsafe {
(*core::ptr::addr_of_mut!((*sdp).work)).data = *{
__slate_compound_5 = __SlateRecord_types_h_197 { counter: 4503599625273344i64 };
core::ptr::addr_of_mut!(__slate_compound_5)
};
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*sdp).work)).entry) });
unsafe {
(*core::ptr::addr_of_mut!((*sdp).work)).func = Some(srcu_invoke_callbacks);
}
if true {
break;
}
}
if true {
break;
}
}
timer_setup!(unsafe { core::ptr::addr_of_mut!((*sdp).delay_work) }, Some(srcu_delay_timer), 0i32);
unsafe {
(*sdp).ssp = ssp;
}
cpu = cpu.wrapping_add(1);
}
}

static mut __key__108: lock_class_key = unsafe { core::mem::zeroed() };

// Invalid seq state, used during snp node initialization
/// Check whether sequence number corresponding to snp node,
/// is invalid.
fn srcu_invl_snp_seq(s: u64) -> bool {
s == 2
}

// Allocated and initialize SRCU combining tree.  Returns @true if
// allocation succeeded and @false otherwise.
fn init_srcu_struct_nodes(ssp: *mut srcu_struct, gfp_flags: u32) -> bool {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_9: raw_spinlock = unsafe { core::mem::zeroed() };
let mut cpu: i32 = 0;
let mut i: i32 = 1;
let mut level: i32 = 0;
let mut levelspread: [i32; 2] = [0i32; 2];
let mut snp: *mut srcu_node = unsafe { core::mem::zeroed() };
// Initialize geometry if it has not already been initialized.
unsafe { rcu_init_geometry() };
let mut __v0: *mut srcu_node;
let __obj_size: u64 = size_mul(core::mem::size_of::<srcu_node>() as u64, (unsafe { rcu_num_nodes }) as u64);
let mut __v1: *mut ();
let mut __v2: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, gfp_flags, unsafe { *{
__slate_compound_3 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_3)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, gfp_flags, unsafe { *{
__slate_compound_5 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_5)
} });
}
__v2 = _res;
__v1 = __v2;
__v0 = __v1 as *mut srcu_node;
unsafe {
(*(*ssp).srcu_sup).node = __v0;
}
if unsafe { (*(*ssp).srcu_sup).node }.is_null() {
return false;
}
// Work out the overall tree geometry.
unsafe {
*((*(*ssp).srcu_sup).level.as_mut_ptr() as *mut *mut srcu_node).wrapping_offset(0 as isize) = unsafe { (*(*ssp).srcu_sup).node }.wrapping_offset(0 as isize);
}
while i < unsafe { rcu_num_lvls } {
unsafe {
*((*(*ssp).srcu_sup).level.as_mut_ptr() as *mut *mut srcu_node).wrapping_offset(i as isize) = unsafe { *((*(*ssp).srcu_sup).level.as_mut_ptr() as *mut *mut srcu_node).wrapping_offset(i.wrapping_sub(1) as isize) }.wrapping_offset(*(core::ptr::addr_of_mut!(num_rcu_lvl) as *mut i32).wrapping_offset(i.wrapping_sub(1) as isize) as isize);
}
i = i.wrapping_add(1);
}
rcu_init_levelspread(levelspread.as_mut_ptr() as *mut i32, (unsafe { core::ptr::addr_of_mut!(num_rcu_lvl) as *mut i32 }) as *const i32);
// Each pass through this loop initializes one srcu_node structure.
snp = unsafe { (*(*ssp).srcu_sup).node }.wrapping_offset(0 as isize);
'__slate_break_15: while snp < unsafe { (*(*ssp).srcu_sup).node }.wrapping_offset((unsafe { rcu_num_nodes }) as isize) {
'__slate_break_10: loop {
unsafe {
*core::ptr::addr_of_mut!((*snp).lock) = *{
__slate_compound_9 = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_qspinlock_types_h_44 = core::mem::zeroed();
__t0.val = __SlateRecord_types_h_190 { counter: 0 };
__t0
} } };
core::ptr::addr_of_mut!(__slate_compound_9)
};
}
if true {
break;
}
}
loop {
if ARRAY_SIZE__24!() != ARRAY_SIZE__53!() {
unsafe { __compiletime_assert_647() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
i = 0;
while (i as u64) < ARRAY_SIZE__106!() {
unsafe {
*((*snp).srcu_have_cbs.as_mut_ptr() as *mut u64).wrapping_offset(i as isize) = 2;
}
unsafe {
*((*snp).srcu_data_have_cbs.as_mut_ptr() as *mut u64).wrapping_offset(i as isize) = 0;
}
i = i.wrapping_add(1);
}
unsafe {
(*snp).srcu_gp_seq_needed_exp = 2;
}
unsafe {
(*snp).grplo = 1i32.wrapping_neg();
}
unsafe {
(*snp).grphi = 1i32.wrapping_neg();
}
if snp == unsafe { (*(*ssp).srcu_sup).node }.wrapping_offset(0 as isize) {
// Root node, special case.
unsafe {
(*snp).srcu_parent = core::ptr::null_mut::<srcu_node>();
}
} else {
// Non-root node.
if snp == unsafe { *((*(*ssp).srcu_sup).level.as_mut_ptr() as *mut *mut srcu_node).wrapping_offset(level.wrapping_add(1) as isize) } {
level = level.wrapping_add(1);
}
unsafe {
(*snp).srcu_parent = unsafe { *((*(*ssp).srcu_sup).level.as_mut_ptr() as *mut *mut srcu_node).wrapping_offset(level.wrapping_sub(1) as isize) }.wrapping_offset(((snp.offset_from(*((*(*ssp).srcu_sup).level.as_mut_ptr() as *mut *mut srcu_node).wrapping_offset(level as isize) as *mut srcu_node) as i64) / (levelspread[level.wrapping_sub(1) as usize] as i64)) as isize);
}
}
let __v18: *mut srcu_node = snp;
snp = __v18.wrapping_offset(1 as isize);
}
// Initialize the per-CPU srcu_data array, which feeds into the
// leaves of the srcu_node tree.
level = unsafe { rcu_num_lvls }.wrapping_sub(1);
let snp_first: *mut srcu_node = unsafe { *((*(*ssp).srcu_sup).level.as_mut_ptr() as *mut *mut srcu_node).wrapping_offset(level as isize) };
loop {
cpu = find_next_bit(cpumask_bits!((unsafe { core::ptr::addr_of_mut!(__cpu_possible_mask) }) as *const cpumask), 64u64, cpu as u64) as i32;
if (cpu as u32) >= 64 {
break;
}
let sdp: *mut srcu_data = per_cpu_ptr__4!(unsafe { (*ssp).sda }, cpu);
unsafe {
(*sdp).mynode = snp_first.wrapping_offset((cpu / levelspread[level as usize]) as isize);
}
snp = unsafe { (*sdp).mynode };
while !snp.is_null() {
if (unsafe { (*snp).grplo }) < 0 {
unsafe {
(*snp).grplo = cpu;
}
}
unsafe {
(*snp).grphi = cpu;
}
snp = unsafe { (*snp).srcu_parent };
}
unsafe {
(*sdp).grpmask = 1u64 << cpu.wrapping_sub((*(*sdp).mynode).grplo);
}
cpu = cpu.wrapping_add(1);
}
loop {
kcsan_release!();
loop {
loop {
if !__native_word__5!() {
unsafe { __compiletime_assert_648() };
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
if !(__native_word__5!() || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_649() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__3!(*unsafe { core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_size_state) }, 2i32);
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
true
}

// Initialize non-compile-time initialized fields, including the
// associated srcu_node and srcu_data structures.  The is_static parameter
// tells us that ->sda has already been wired up to srcu_data.
fn init_srcu_struct_fields(ssp: *mut srcu_struct, is_static: bool) -> i32 {
let mut __slate_compound_27: __SlateRecord_types_h_197 = unsafe { core::mem::zeroed() };
let mut __slate_compound_28: raw_spinlock = unsafe { core::mem::zeroed() };
let mut __slate_compound_29: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_31: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
unsafe {
if !is_static {
__obj_size = size_mul(core::mem::size_of::<srcu_usage>() as u64, 1u64);
let __slate_auto_bytes_1 = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_1 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_2 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_29 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_29)
});
alloc_tag_restore!();
} else {
*__slate_slot_2 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_31 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_31)
});
}
(*ssp).srcu_sup = *__slate_slot_2 as *mut srcu_usage;
}
if unsafe { (*ssp).srcu_sup }.is_null() {
return 12i32.wrapping_neg();
} else {
if !is_static {
loop {
*core::ptr::addr_of_mut!((*(*ssp).srcu_sup).lock) = *{
__slate_compound_28 = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_qspinlock_types_h_44 = core::mem::zeroed();
__t0.val = __SlateRecord_types_h_190 { counter: 0 };
__t0
} } };
core::ptr::addr_of_mut!(__slate_compound_28)
};
if true {
break;
}
}
}
(*(*ssp).srcu_sup).srcu_size_state = 0;
(*(*ssp).srcu_sup).node = core::ptr::null_mut::<srcu_node>();
loop {
__mutex_init(core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_cb_mutex), c"&ssp->srcu_sup->srcu_cb_mutex".as_ptr() as *const u8, core::ptr::addr_of_mut!(__key__214));
if true {
break;
}
}
loop {
__mutex_init(core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_gp_mutex), c"&ssp->srcu_sup->srcu_gp_mutex".as_ptr() as *const u8, core::ptr::addr_of_mut!(__key__215));
if true {
break;
}
}
(*(*ssp).srcu_sup).srcu_gp_seq = 18446744073709551216u64;
(*(*ssp).srcu_sup).srcu_barrier_seq = 0;
loop {
__mutex_init(core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_barrier_mutex), c"&ssp->srcu_sup->srcu_barrier_mutex".as_ptr() as *const u8, core::ptr::addr_of_mut!(__key__218));
if true {
break;
}
}
atomic_set(core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_barrier_cpu_cnt), 0i32);
loop {
__init_work(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*(*ssp).srcu_sup).work)).work), 0i32);
(*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*(*ssp).srcu_sup).work)).work)).data = *{
__slate_compound_27 = __SlateRecord_types_h_197 { counter: 4503599625273344i64 };
core::ptr::addr_of_mut!(__slate_compound_27)
};
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*(*ssp).srcu_sup).work)).work)).entry));
(*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*(*ssp).srcu_sup).work)).work)).func = Some(process_srcu);
if true {
if true {
__timer_init!(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*(*ssp).srcu_sup).work)).timer), unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut timer_list)>>(delayed_work_timer_fn as *const ()) }, 2097152i32);
if true {
break;
}
}
}
}
init_irq_work(core::ptr::addr_of_mut!((*(*ssp).srcu_sup).irq_work), Some(srcu_irq_work));
(*(*ssp).srcu_sup).sda_is_static = is_static;
if !is_static {
(*ssp).sda = alloc_percpu!(srcu_data);
(*ssp).srcu_ctrp = ((*(*ssp).sda).srcu_ctrs.as_mut_ptr() as *mut srcu_ctr).wrapping_offset(0 as isize);
}
'__join_2: {
if !unsafe { (*ssp).sda }.is_null() {
init_srcu_struct_data(ssp);
(*(*ssp).srcu_sup).srcu_gp_seq_needed_exp = 18446744073709551216u64;
(*(*ssp).srcu_sup).srcu_last_gp_end = ktime_get_mono_fast_ns();
while false {
}
let __v0: i32 = core::ptr::read_volatile(core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_size_state) as *const i32);
if __v0 == 0 && SRCU_SIZING_IS_INIT!() {
if !preemptible!() {
loop {
if true {
__WRITE_ONCE__3!((*(*ssp).srcu_sup).srcu_size_state, 1i32);
if true {
break;
}
}
}
} else {
if init_srcu_struct_nodes(ssp, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)) {
loop {
if true {
__WRITE_ONCE__3!((*(*ssp).srcu_sup).srcu_size_state, 8i32);
if true {
break;
}
}
}
} else {
if !is_static {
free_percpu((*ssp).sda as *mut ());
(*ssp).sda = core::ptr::null_mut::<srcu_data>();
break '__join_2;
} else {
break '__join_2;
}
}
}
}
(*(*ssp).srcu_sup).srcu_ssp = ssp;
loop {
kcsan_release!();
loop {
if true {
core::arch::asm!("", options(nostack, att_syntax));
loop {
if true {
__WRITE_ONCE__2!(*core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_gp_seq_needed), 18446744073709551216u64);
if true {
break;
}
}
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
// Init done.
return 0;
}
}
if !is_static {
kfree((*ssp).srcu_sup as *const ());
(*ssp).srcu_sup = core::ptr::null_mut::<srcu_usage>();
}
return 12i32.wrapping_neg();
}
}
}

static mut __key__214: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__215: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__218: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__220: lock_class_key = unsafe { core::mem::zeroed() };

/// init_srcu_struct_generic - initialize a sleep-RCU structure
/// @ssp: structure to initialize.
///
/// Use this in place of DEFINE_SRCU() and DEFINE_STATIC_SRCU()
/// for non-static srcu_struct structures that are to be passed to
/// srcu_read_lock(), srcu_read_lock_nmisafe(), and friends.  It is necessary
/// to invoke this on a given srcu_struct before passing that srcu_struct
/// to any other function.  Each srcu_struct represents a separate domain
/// of SRCU protection.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.srcutree.init_srcu_struct_generic")]
extern "C" fn init_srcu_struct_generic(ssp: *mut srcu_struct) -> i32 {
unsafe {
(*ssp).srcu_reader_flavor = 0;
}
init_srcu_struct_fields(ssp, false)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_init_srcu_struct_generic_655: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut srcu_struct) -> i32>, *mut ()>(Some(init_srcu_struct_generic)) };

/// init_srcu_struct_fast - initialize a fast-reader sleep-RCU structure
/// @ssp: structure to initialize.
///
/// Use this in place of DEFINE_SRCU_FAST() and DEFINE_STATIC_SRCU_FAST()
/// for non-static srcu_struct structures that are to be passed to
/// srcu_read_lock_fast() and friends.  It is necessary to invoke this on a
/// given srcu_struct before passing that srcu_struct to any other function.
/// Each srcu_struct represents a separate domain of SRCU protection.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.srcutree.init_srcu_struct_fast")]
extern "C" fn init_srcu_struct_fast(ssp: *mut srcu_struct) -> i32 {
unsafe {
(*ssp).srcu_reader_flavor = 4;
}
init_srcu_struct_fields(ssp, false)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_init_srcu_struct_fast_656: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut srcu_struct) -> i32>, *mut ()>(Some(init_srcu_struct_fast)) };

/// init_srcu_struct_fast_updown - initialize a fast-reader up/down sleep-RCU structure
/// @ssp: structure to initialize.
///
/// Use this function in place of DEFINE_SRCU_FAST_UPDOWN() and
/// DEFINE_STATIC_SRCU_FAST_UPDOWN() for non-static srcu_struct
/// structures that are to be passed to srcu_read_lock_fast_updown(),
/// srcu_down_read_fast(), and friends.  It is necessary to invoke this on a
/// given srcu_struct before passing that srcu_struct to any other function.
/// Each srcu_struct represents a separate domain of SRCU protection.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.srcutree.init_srcu_struct_fast_updown")]
extern "C" fn init_srcu_struct_fast_updown(ssp: *mut srcu_struct) -> i32 {
unsafe {
(*ssp).srcu_reader_flavor = 8;
}
init_srcu_struct_fields(ssp, false)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_init_srcu_struct_fast_updown_657: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut srcu_struct) -> i32>, *mut ()>(Some(init_srcu_struct_fast_updown)) };

// Initiate a transition to SRCU_SIZE_BIG with lock held.
fn __srcu_transition_to_big(ssp: *mut srcu_struct) {
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!((*(*ssp).srcu_sup).lock) });
loop {
kcsan_release!();
loop {
loop {
if !__native_word__5!() {
unsafe { __compiletime_assert_658() };
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
if !(__native_word__5!() || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_659() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__3!(*unsafe { core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_size_state) }, 1i32);
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

// Initiate an idempotent transition to SRCU_SIZE_BIG.
fn srcu_transition_to_big(ssp: *mut srcu_struct) {
let mut flags: u64 = 0;
// Double-checked locking on ->srcu_size-state.
let mut __v0: i32;
{
let mut ___p1: i32 = 0;
let mut __v1: i32;
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_660() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_size_state) as *const i32) };
___p1 = __v1;
loop {
if !__native_word__5!() {
unsafe { __compiletime_assert_661() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
__v0 = ___p1;
}
if __v0 != 0 {
return;
}
raw_spin_lock_irqsave_rcu_node!(unsafe { (*ssp).srcu_sup }, flags);
let mut __v8: i32;
let mut ___p1: i32 = 0;
let mut __v9: i32;
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_662() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v9 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_size_state) as *const i32) };
___p1 = __v9;
loop {
if !__native_word__5!() {
unsafe { __compiletime_assert_663() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
__v8 = ___p1;
if __v8 != 0 {
raw_spin_unlock_irqrestore_rcu_node!(unsafe { (*ssp).srcu_sup }, flags);
return;
}
__srcu_transition_to_big(ssp);
raw_spin_unlock_irqrestore_rcu_node!(unsafe { (*ssp).srcu_sup }, flags);
}

/// Check to see if the just-encountered contention event justifies
/// a transition to SRCU_SIZE_BIG.
fn raw_spin_lock_irqsave_check_contention(ssp: *mut srcu_struct) {
let mut j: u64 = 0;
if SRCU_SIZING_IS_CONTEND!() == 0 || (unsafe { (*(*ssp).srcu_sup).srcu_size_state }) != 0 {
return;
}
j = unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) };
if (unsafe { (*(*ssp).srcu_sup).srcu_size_jiffies }) != j {
unsafe {
(*(*ssp).srcu_sup).srcu_size_jiffies = j;
}
unsafe {
(*(*ssp).srcu_sup).srcu_n_lock_retries = 0;
}
}
let __v0: *mut srcu_usage = unsafe { (*ssp).srcu_sup };
let __v1: u64 = unsafe { (*__v0).srcu_n_lock_retries };
let __v2: u64 = __v1.wrapping_add(1);
unsafe {
(*__v0).srcu_n_lock_retries = __v2;
}
if __v2 <= ((unsafe { small_contention_lim }) as u64) {
return;
}
__srcu_transition_to_big(ssp);
}

/// Acquire the specified srcu_data structure's ->lock, but check for
/// excessive contention, which results in initiation of a transition
/// to SRCU_SIZE_BIG.  But only if the srcutree.convert_to_big module
/// parameter permits this.
fn raw_spin_lock_irqsave_sdp_contention(sdp: *mut srcu_data, flags: *mut u64) {
let ssp: *mut srcu_struct = unsafe { (*sdp).ssp };
if raw_spin_trylock_irqsave_rcu_node!(sdp, *flags) {
return;
}
raw_spin_lock_irqsave_rcu_node__2!(unsafe { (*ssp).srcu_sup }, *flags);
raw_spin_lock_irqsave_check_contention(ssp);
raw_spin_unlock_irqrestore_rcu_node!(unsafe { (*ssp).srcu_sup }, unsafe { *flags });
raw_spin_lock_irqsave_rcu_node__2!(sdp, *flags);
}

/// Acquire the specified srcu_struct structure's ->lock, but check for
/// excessive contention, which results in initiation of a transition
/// to SRCU_SIZE_BIG.  But only if the srcutree.convert_to_big module
/// parameter permits this.
fn raw_spin_lock_irqsave_ssp_contention(ssp: *mut srcu_struct, flags: *mut u64) {
if raw_spin_trylock_irqsave_rcu_node!(unsafe { (*ssp).srcu_sup }, *flags) {
return;
}
raw_spin_lock_irqsave_rcu_node__2!(unsafe { (*ssp).srcu_sup }, *flags);
raw_spin_lock_irqsave_check_contention(ssp);
}

// First-use initialization of statically allocated srcu_struct
// structure.  Wiring up the combining tree is more than can be
// done with compile-time initialization, so this check is added
// to each update-side SRCU primitive.  Use ssp->lock, which -is-
// compile-time initialized, to resolve races involving multiple
// CPUs trying to garner first-use privileges.
fn check_init_srcu_struct(ssp: *mut srcu_struct) {
let mut flags: u64 = 0;
// The smp_load_acquire() pairs with the smp_store_release().
let mut __v0: u64;
let mut ___p1: u64 = 0;
let mut __v1: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_664() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_gp_seq_needed) as *const u64) };
___p1 = __v1;
loop {
if !__native_word!() {
unsafe { __compiletime_assert_665() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
__v0 = ___p1;
if rcu_seq_state(__v0) == 0 {
return;
}
// ^^^
// Already initialized.
raw_spin_lock_irqsave_rcu_node!(unsafe { (*ssp).srcu_sup }, flags);
if rcu_seq_state(unsafe { (*(*ssp).srcu_sup).srcu_gp_seq_needed }) == 0 {
raw_spin_unlock_irqrestore_rcu_node!(unsafe { (*ssp).srcu_sup }, flags);
return;
}
init_srcu_struct_fields(ssp, true);
raw_spin_unlock_irqrestore_rcu_node!(unsafe { (*ssp).srcu_sup }, flags);
}

// Is the current or any upcoming grace period to be expedited?
fn srcu_gp_is_expedited(ssp: *mut srcu_struct) -> bool {
let sup: *mut srcu_usage = unsafe { (*ssp).srcu_sup };
let mut __v0: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_666() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*sup).srcu_gp_seq) as *const u64) };
let mut __v2: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_667() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*sup).srcu_gp_seq_needed_exp) as *const u64) };
!0 / 2 < __v0.wrapping_sub(__v2)
}

// Computes approximate total of the readers' ->srcu_ctrs[].srcu_locks
// values for the rank of per-CPU counters specified by idx, and returns
// true if the caller did the proper barrier (gp), and if the count of
// the locks matches that of the unlocks passed in.
fn srcu_readers_lock_idx(ssp: *mut srcu_struct, idx: i32, gp: bool, unlocks: u64) -> bool {
let mut cpu: i32 = 0;
let mut mask: u64 = 0;
let mut sum: u64 = 0;
loop {
cpu = find_next_bit(cpumask_bits!((unsafe { core::ptr::addr_of_mut!(__cpu_possible_mask) }) as *const cpumask), 64u64, cpu as u64) as i32;
if (cpu as u32) >= 64 {
break;
}
let sdp: *mut srcu_data = per_cpu_ptr__4!(unsafe { (*ssp).sda }, cpu);
let __v3: i64 = atomic_long_read((unsafe { core::ptr::addr_of_mut!((*((*sdp).srcu_ctrs.as_mut_ptr() as *mut srcu_ctr).wrapping_offset(idx as isize)).srcu_locks) }) as *const __SlateRecord_types_h_197);
sum = sum.wrapping_add(__v3 as u64);
if IS_ENABLED!() != 0 {
let mut __v6: i32;
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_668() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v6 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*sdp).srcu_reader_flavor) as *const i32) };
mask |= __v6 as u64;
}
cpu = cpu.wrapping_add(1);
}
let mut __v11: i32;
let __ret_warn_on: i32 = (IS_ENABLED!() != 0 && mask & mask.wrapping_sub(1) != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
2i32 | BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"Mixed reader flavors for srcu_struct at %ps.\n".as_ptr() as *const u8, ssp) };
let mut __v12: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v13: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_srcu_readers_lock_idx_14, sym __slate_asm_string_srcu_readers_lock_idx_15, const 483 as i32, const 2323 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v13 = bug;
unsafe { __v12.unwrap()(__v13, ssp) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v11 = __ret_warn_on;
if mask & 12u64 != 0 && !gp {
return false;
}
sum == unlocks
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_669: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

// Returns approximate total of the readers' ->srcu_ctrs[].srcu_unlocks
// values for the rank of per-CPU counters specified by idx.
fn srcu_readers_unlock_idx(ssp: *mut srcu_struct, idx: i32, rdm: &mut u64) -> u64 {
let mut cpu: i32 = 0;
let mut mask: u64 = (unsafe { (*ssp).srcu_reader_flavor }) as u64;
let mut sum: u64 = 0;
loop {
cpu = find_next_bit(cpumask_bits!((unsafe { core::ptr::addr_of_mut!(__cpu_possible_mask) }) as *const cpumask), 64u64, cpu as u64) as i32;
if (cpu as u32) >= 64 {
break;
}
let sdp: *mut srcu_data = per_cpu_ptr__4!(unsafe { (*ssp).sda }, cpu);
let __v3: i64 = atomic_long_read((unsafe { core::ptr::addr_of_mut!((*((*sdp).srcu_ctrs.as_mut_ptr() as *mut srcu_ctr).wrapping_offset(idx as isize)).srcu_unlocks) }) as *const __SlateRecord_types_h_197);
sum = sum.wrapping_add(__v3 as u64);
let mut __v6: i32;
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_670() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v6 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*sdp).srcu_reader_flavor) as *const i32) };
mask |= __v6 as u64;
cpu = cpu.wrapping_add(1);
}
let mut __v11: i32;
let __ret_warn_on: i32 = (IS_ENABLED!() != 0 && mask & mask.wrapping_sub(1) != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
2i32 | BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"Mixed reader flavors for srcu_struct at %ps.\n".as_ptr() as *const u8, ssp) };
let mut __v12: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v13: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_srcu_readers_unlock_idx_14, sym __slate_asm_string_srcu_readers_unlock_idx_15, const 506 as i32, const 2323 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v13 = bug;
unsafe { __v12.unwrap()(__v13, ssp) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v11 = __ret_warn_on;
unsafe {
*(rdm as *mut u64) = mask;
}
sum
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_671: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

/// Return true if the number of pre-existing readers is determined to
/// be zero.
fn srcu_readers_active_idx_check(ssp: *mut srcu_struct, idx: i32) -> bool {
let mut did_gp: bool = false;
let mut rdm: u64 = 0;
let unlocks: u64 = srcu_readers_unlock_idx(ssp, idx, unsafe { &mut (*core::ptr::addr_of_mut!(rdm)) });
did_gp = rdm & 12u64 != 0;
// Make sure that a lock is always counted if the corresponding
// unlock is counted. Needs to be a smp_mb() as the read side may
// contain a read from a variable that is written to before the
// synchronize_srcu() in the write side. In this case smp_mb()s
// A and B (or X and Y) act like the store buffering pattern.
//
// This smp_mb() also pairs with smp_mb() C (or, in the case of X,
// Z) to prevent accesses after the synchronize_srcu() from being
// executed before the grace period ends.
if !did_gp {
loop {
kcsan_mb!();
unsafe {
core::arch::asm!("lock addl $0,-4(%rsp)", options(nostack, att_syntax));
}
if true {
break;
}
}
} else {
if srcu_gp_is_expedited(ssp) {
unsafe { synchronize_rcu_expedited() };
} else {
unsafe { synchronize_rcu() };
}
}
// A
// X
// X
// * If the locks are the same as the unlocks, then there must have
// * been no readers on this index at some point in this function.
// * But there might be more readers, as a task might have read
// * the current ->srcu_ctrp but not yet have incremented its CPU's
// * ->srcu_ctrs[idx].srcu_locks counter.  In fact, it is possible
// * that most of the tasks have been preempted between fetching
// * ->srcu_ctrp and incrementing ->srcu_ctrs[idx].srcu_locks.  And
// * there could be almost (ULONG_MAX / sizeof(struct task_struct))
// * tasks in a system whose address space was fully populated
// * with memory.  Call this quantity Nt.
// *
// * So suppose that the updater is preempted at this
// * point in the code for a long time.  That now-preempted
// * updater has already flipped ->srcu_ctrp (possibly during
// * the preceding grace period), done an smp_mb() (again,
// * possibly during the preceding grace period), and summed up
// * the ->srcu_ctrs[idx].srcu_unlocks counters.  How many times
// * can a given one of the aforementioned Nt tasks increment the
// * old ->srcu_ctrp value's ->srcu_ctrs[idx].srcu_locks counter,
// * in the absence of nesting?
// *
// * It can clearly do so once, given that it has already fetched
// * the old value of ->srcu_ctrp and is just about to use that
// * value to index its increment of ->srcu_ctrs[idx].srcu_locks.
// * But as soon as it leaves that SRCU read-side critical section,
// * it will increment ->srcu_ctrs[idx].srcu_unlocks, which must
// * follow the updater's above read from that same value.  Thus,
//   as soon the reading task does an smp_mb() and a later fetch from
// * ->srcu_ctrp, that task will be guaranteed to get the new index.
// * Except that the increment of ->srcu_ctrs[idx].srcu_unlocks
// * in __srcu_read_unlock() is after the smp_mb(), and the fetch
// * from ->srcu_ctrp in __srcu_read_lock() is before the smp_mb().
// * Thus, that task might not see the new value of ->srcu_ctrp until
// * the -second- __srcu_read_lock(), which in turn means that this
// * task might well increment ->srcu_ctrs[idx].srcu_locks for the
// * old value of ->srcu_ctrp twice, not just once.
// *
// * However, it is important to note that a given smp_mb() takes
// * effect not just for the task executing it, but also for any
// * later task running on that same CPU.
// *
// * That is, there can be almost Nt + Nc further increments
// * of ->srcu_ctrs[idx].srcu_locks for the old index, where Nc
// * is the number of CPUs.  But this is OK because the size of
// * the task_struct structure limits the value of Nt and current
// * systems limit Nc to a few thousand.
// *
// * OK, but what about nesting?  This does impose a limit on
// * nesting of half of the size of the task_struct structure
// * (measured in bytes), which should be sufficient.  A late 2022
// * TREE01 rcutorture run reported this size to be no less than
// * 9408 bytes, allowing up to 4704 levels of nesting, which is
// * comfortably beyond excessive.  Especially on 64-bit systems,
// * which are unlikely to be configured with an address space fully
// * populated with memory, at least not anytime soon.
srcu_readers_lock_idx(ssp, idx, did_gp, unlocks)
}

// We use an adaptive strategy for synchronize_srcu() and especially for
// synchronize_srcu_expedited().  We spin for a fixed time period
// (defined below, boot time configurable) to allow SRCU readers to exit
// their read-side critical sections.  If there are still some readers
// after one jiffy, we repeatedly block for one jiffy time periods.
// The blocking time is increased as the grace-period age increases,
// with max blocking time capped at 10 jiffies.
static mut srcu_retry_check_delay: u64 = 5;

fn __check_srcu_retry_check_delay() -> *mut u64 {
unsafe { core::ptr::addr_of_mut!(srcu_retry_check_delay) }
}

static __param_str_srcu_retry_check_delay: __SlateAlign16<[u8; 32]> = __SlateAlign16([115u8, 114u8, 99u8, 117u8, 116u8, 114u8, 101u8, 101u8, 46u8, 115u8, 114u8, 99u8, 117u8, 95u8, 114u8, 101u8, 116u8, 114u8, 121u8, 95u8, 99u8, 104u8, 101u8, 99u8, 107u8, 95u8, 100u8, 101u8, 108u8, 97u8, 121u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_srcu_retry_check_delay: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_srcu_retry_check_delay.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_ulong) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_srcutree_c_614>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_614_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_614_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_614_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_614_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_614_6>() as u64) as i32).wrapping_add(292i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(srcu_retry_check_delay) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_672: [u8; 47] = [115, 114, 99, 117, 116, 114, 101, 101, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 115, 114, 99, 117, 95, 114, 101, 116, 114, 121, 95, 99, 104, 101, 99, 107, 95, 100, 101, 108, 97, 121, 58, 117, 108, 111, 110, 103, 0];

// Base delay if no expedited GPs pending.
// Maximum incremental delay from slow readers.
// Lowmark on default per-GP-phase
// no-delay instances.
// Highmark on default per-GP-phase
// no-delay instances.
// per-GP-phase no-delay instances adjusted to allow non-sleeping poll upto
//
// one jiffies time duration. Mult by 2 is done to factor in the srcu_get_delay()
//
// called from process_srcu().
// Maximum per-GP-phase consecutive no-delay instances.
static mut srcu_max_nodelay_phase: u64 = 400;

fn __check_srcu_max_nodelay_phase() -> *mut u64 {
unsafe { core::ptr::addr_of_mut!(srcu_max_nodelay_phase) }
}

static __param_str_srcu_max_nodelay_phase: __SlateAlign16<[u8; 32]> = __SlateAlign16([115u8, 114u8, 99u8, 117u8, 116u8, 114u8, 101u8, 101u8, 46u8, 115u8, 114u8, 99u8, 117u8, 95u8, 109u8, 97u8, 120u8, 95u8, 110u8, 111u8, 100u8, 101u8, 108u8, 97u8, 121u8, 95u8, 112u8, 104u8, 97u8, 115u8, 101u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_srcu_max_nodelay_phase: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_srcu_max_nodelay_phase.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_ulong) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_srcutree_c_640>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_640_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_640_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_640_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_640_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_640_6>() as u64) as i32).wrapping_add(292i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(srcu_max_nodelay_phase) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_673: [u8; 47] = [115, 114, 99, 117, 116, 114, 101, 101, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 115, 114, 99, 117, 95, 109, 97, 120, 95, 110, 111, 100, 101, 108, 97, 121, 95, 112, 104, 97, 115, 101, 58, 117, 108, 111, 110, 103, 0];

// Maximum consecutive no-delay instances.
static mut srcu_max_nodelay: u64 = 400;

fn __check_srcu_max_nodelay() -> *mut u64 {
unsafe { core::ptr::addr_of_mut!(srcu_max_nodelay) }
}

static __param_str_srcu_max_nodelay: __SlateAlign16<[u8; 26]> = __SlateAlign16([115u8, 114u8, 99u8, 117u8, 116u8, 114u8, 101u8, 101u8, 46u8, 115u8, 114u8, 99u8, 117u8, 95u8, 109u8, 97u8, 120u8, 95u8, 110u8, 111u8, 100u8, 101u8, 108u8, 97u8, 121u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_srcu_max_nodelay: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_srcu_max_nodelay.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_ulong) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_srcutree_c_647>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_647_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_647_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_647_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_647_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_srcutree_c_647_6>() as u64) as i32).wrapping_add(292i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(srcu_max_nodelay) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_674: [u8; 41] = [115, 114, 99, 117, 116, 114, 101, 101, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 115, 114, 99, 117, 95, 109, 97, 120, 95, 110, 111, 100, 101, 108, 97, 121, 58, 117, 108, 111, 110, 103, 0];

// Return grace-period delay, zero if there are expedited grace
// periods pending, SRCU_INTERVAL otherwise.
fn srcu_get_delay(ssp: *mut srcu_struct) -> u64 {
let mut gpstart: u64 = 0;
let mut j: u64 = 0;
let mut jbase: u64 = 1;
let sup: *mut srcu_usage = unsafe { (*ssp).srcu_sup };
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!((*(*ssp).srcu_sup).lock) });
if srcu_gp_is_expedited(ssp) {
jbase = 0;
}
let mut __v1: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_675() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*sup).srcu_gp_seq) as *const u64) };
if rcu_seq_state(__v1) != 0 {
j = unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) }.wrapping_sub(1);
let mut __v3: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_676() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*sup).srcu_gp_start) as *const u64) };
gpstart = __v3;
if time_after!(j, gpstart) {
jbase = jbase.wrapping_add(j.wrapping_sub(gpstart));
}
if jbase == 0 {
ASSERT_EXCLUSIVE_WRITER__2!((*sup).srcu_n_exp_nodelay);
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_678() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
loop {
let mut __v15: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_677() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v15 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*sup).srcu_n_exp_nodelay) as *const u64) };
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!((*sup).srcu_n_exp_nodelay) as *mut u64, __v15.wrapping_add(1)) };
if true {
break;
}
}
if true {
break;
}
}
let mut __v19: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_679() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v19 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*sup).srcu_n_exp_nodelay) as *const u64) };
if __v19 > unsafe { srcu_max_nodelay_phase } {
jbase = 1;
}
}
}
if jbase > 10 { 10 } else { jbase }
}

// cleanup_srcu_struct - deconstruct a sleep-RCU structure
// @ssp: structure to clean up.
//
// Must invoke this after you are finished using a given srcu_struct that
// was initialized via init_srcu_struct(), else you leak memory.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_cleanup_srcu_struct_696: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut srcu_struct)>, *mut ()>(Some(cleanup_srcu_struct)) };

// Check for consistent reader flavor.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_708: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___srcu_check_read_flavor_709: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut srcu_struct, i32)>, *mut ()>(Some(__srcu_check_read_flavor)) };

// Counts the new reader in the appropriate per-CPU element of the
// srcu_struct.
// Returns a guaranteed non-negative index that must be passed to the
// matching __srcu_read_unlock().
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___srcu_read_lock_711: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut srcu_struct) -> i32>, *mut ()>(Some(__srcu_read_lock)) };

/// Removes the count for the old reader from the appropriate per-CPU
/// element of the srcu_struct.  Note that this may well be a different
/// CPU than that which was incremented by the corresponding srcu_read_lock().
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.srcutree.__srcu_read_unlock")]
extern "C" fn __srcu_read_unlock(mut ssp: *mut srcu_struct, mut idx: i32) {
loop {
kcsan_mb!();
unsafe {
core::arch::asm!("lock addl $0,-4(%rsp)", options(nostack, att_syntax));
}
if true {
break;
}
}
// C
//
// Avoid leaking the critical section.
loop {
__verify_pcpu_ptr__4!();
match (const { core::mem::size_of::<i64>() }) as u64 {
1 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incb %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*__srcu_ctr_to_ptr(&(*ssp), idx)).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decb %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*__srcu_ctr_to_ptr(&(*ssp), idx)).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u8 = 1u8;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addb {1:l}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*__srcu_ctr_to_ptr(&(*ssp), idx)).srcu_unlocks.counter) as u64) as *mut i64, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
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
2 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incw %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*__srcu_ctr_to_ptr(&(*ssp), idx)).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decw %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*__srcu_ctr_to_ptr(&(*ssp), idx)).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u16 = 1u16;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addw {1:x}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*__srcu_ctr_to_ptr(&(*ssp), idx)).srcu_unlocks.counter) as u64) as *mut i64, in(reg) pto_val__ as u16, options(nostack, att_syntax));
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
4 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incl %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*__srcu_ctr_to_ptr(&(*ssp), idx)).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decl %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*__srcu_ctr_to_ptr(&(*ssp), idx)).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u32 = 1u32;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addl {1:e}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*__srcu_ctr_to_ptr(&(*ssp), idx)).srcu_unlocks.counter) as u64) as *mut i64, in(reg) pto_val__ as u32, options(nostack, att_syntax));
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
8 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incq %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*__srcu_ctr_to_ptr(&(*ssp), idx)).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decq %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*__srcu_ctr_to_ptr(&(*ssp), idx)).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u64 = 1;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addq {1:r}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*__srcu_ctr_to_ptr(&(*ssp), idx)).srcu_unlocks.counter) as u64) as *mut i64, in(reg) pto_val__ as u64, options(nostack, att_syntax));
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
_ => {
unsafe { __bad_size_call_parameter() };
}
}
if true {
break;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___srcu_read_unlock_712: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut srcu_struct, i32)>, *mut ()>(Some(__srcu_read_unlock)) };

// Start an SRCU grace period.
fn srcu_gp_start(ssp: *mut srcu_struct) {
let mut state: i32 = 0;
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!((*(*ssp).srcu_sup).lock) });
let mut __v1: i64;
{
let __ret_warn_on: i32 = !(!ULONG_CMP_GE!(unsafe { (*(*ssp).srcu_sup).srcu_gp_seq }, unsafe { (*(*ssp).srcu_sup).srcu_gp_seq_needed })) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("713: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 713b - ., 3; .popsection\n# {0}", const 713 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_srcu_gp_start_2, sym __slate_asm_string_srcu_gp_start_3, const 833 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("714: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 714b - ., 4; .popsection\n# {0}", const 714 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
}
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_715() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__2!((*unsafe { (*ssp).srcu_sup }).srcu_gp_start, unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) });
if true {
break;
}
}
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_716() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__13!((*unsafe { (*ssp).srcu_sup }).srcu_n_exp_nodelay, 0i32);
if true {
break;
}
}
loop {
kcsan_mb!();
unsafe {
core::arch::asm!("lock addl $0,-4(%rsp)", options(nostack, att_syntax));
}
if true {
break;
}
}
// Order prior store to ->srcu_gp_seq_needed vs. GP start.
rcu_seq_start(unsafe { core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_gp_seq) });
state = rcu_seq_state(unsafe { (*(*ssp).srcu_sup).srcu_gp_seq });
let mut __v14: i64;
let __ret_warn_on: i32 = (state != 1) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("717: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 717b - ., 3; .popsection\n# {0}", const 717 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_srcu_gp_start_15, sym __slate_asm_string_srcu_gp_start_16, const 839 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("718: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 718b - ., 4; .popsection\n# {0}", const 718 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v14 = (__ret_warn_on != 0) as i64;
}

#[unsafe(link_section = ".text.slate_distinct.srcutree.srcu_delay_timer")]
extern "C" fn srcu_delay_timer(t: *mut timer_list) {
let sdp: *mut srcu_data = container_of__145!(t, srcu_data);
unsafe { queue_work_on((*sdp).cpu, rcu_gp_wq, core::ptr::addr_of_mut!((*sdp).work)) };
}

fn srcu_queue_delayed_work_on(sdp: *mut srcu_data, delay: u64) {
if delay == 0 {
unsafe { queue_work_on((*sdp).cpu, rcu_gp_wq, core::ptr::addr_of_mut!((*sdp).work)) };
return;
}
unsafe { timer_reduce(core::ptr::addr_of_mut!((*sdp).delay_work), core::ptr::read_volatile(core::ptr::addr_of!(jiffies)).wrapping_add(delay)) };
}

/// Schedule callback invocation for the specified srcu_data structure,
/// if possible, on the corresponding CPU.
fn srcu_schedule_cbs_sdp(sdp: *mut srcu_data, delay: u64) {
srcu_queue_delayed_work_on(sdp, delay);
}

/// Schedule callback invocation for all srcu_data structures associated
/// with the specified srcu_node structure that have callbacks for the
/// just-completed grace period, the one corresponding to idx.  If possible,
/// schedule this invocation on the corresponding CPUs.
fn srcu_schedule_cbs_snp(ssp: *mut srcu_struct, snp: *mut srcu_node, mask: u64, delay: u64) {
let mut cpu: i32 = unsafe { (*snp).grplo };
while cpu <= unsafe { (*snp).grphi } {
let __v0: bool = mask & 1u64 << cpu.wrapping_sub(unsafe { (*snp).grplo }) != 0 && unsafe { rcu_cpu_beenfullyonline(cpu) };
if __v0 {
srcu_schedule_cbs_sdp(per_cpu_ptr__4!(unsafe { (*ssp).sda }, cpu), delay);
}
cpu = cpu.wrapping_add(1);
}
}

// Note the end of an SRCU grace period.  Initiates callback invocation
// and starts a new grace period if needed.
//
// The ->srcu_cb_mutex acquisition does not protect any data, but
// instead prevents more than one grace period from starting while we
// are initiating callback invocation.  This allows the ->srcu_have_cbs[]
// array to have a finite number of elements.
fn srcu_gp_end(ssp: *mut srcu_struct) {
let mut cbdelay: u64 = 1;
let mut cbs: bool = false;
let mut last_lvl: bool = false;
let mut cpu: i32 = 0;
let mut gpseq: u64 = 0;
let mut idx: i32 = 0;
let mut mask: u64 = 0;
let mut sgsne: u64 = 0;
let mut snp: *mut srcu_node = unsafe { core::mem::zeroed() };
let mut ss_state: i32 = 0;
let sup: *mut srcu_usage = unsafe { (*ssp).srcu_sup };
// Prevent more than one additional grace period.
unsafe { mutex_lock(core::ptr::addr_of_mut!((*sup).srcu_cb_mutex)) };
// End the current grace period.
raw_spin_lock_irq_rcu_node!(sup);
idx = rcu_seq_state(unsafe { (*sup).srcu_gp_seq });
let mut __v2: i64;
let __ret_warn_on: i32 = (idx != 2) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("719: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 719b - ., 3; .popsection\n# {0}", const 719 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_srcu_gp_end_3, sym __slate_asm_string_srcu_gp_end_4, const 916 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("720: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 720b - ., 4; .popsection\n# {0}", const 720 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v2 = (__ret_warn_on != 0) as i64;
if srcu_gp_is_expedited(ssp) {
cbdelay = 0;
}
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_721() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__2!((*sup).srcu_last_gp_end, unsafe { ktime_get_mono_fast_ns() });
if true {
break;
}
}
rcu_seq_end(unsafe { core::ptr::addr_of_mut!((*sup).srcu_gp_seq) });
gpseq = rcu_seq_current(unsafe { core::ptr::addr_of_mut!((*sup).srcu_gp_seq) });
if ULONG_CMP_LT!(unsafe { (*sup).srcu_gp_seq_needed_exp }, gpseq) {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_722() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__2!((*sup).srcu_gp_seq_needed_exp, gpseq);
if true {
break;
}
}
}
raw_spin_unlock_irq_rcu_node!(sup);
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*sup).srcu_gp_mutex)) };
// A new grace period can start at this point.  But only one.
// Initiate callback invocation as needed.
let mut __v15: i32;
let mut ___p1: i32 = 0;
let mut __v16: i32;
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_723() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v16 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*sup).srcu_size_state) as *const i32) };
___p1 = __v16;
loop {
if !__native_word__5!() {
unsafe { __compiletime_assert_724() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
__v15 = ___p1;
ss_state = __v15;
if ss_state < 2 {
srcu_schedule_cbs_sdp(per_cpu_ptr__4!(unsafe { (*ssp).sda }, get_boot_cpu_id()), cbdelay);
} else {
idx = (rcu_seq_ctr(gpseq) % ARRAY_SIZE__91!()) as i32;
snp = unsafe { (*(*ssp).srcu_sup).node }.wrapping_offset(0 as isize);
while snp < unsafe { (*(*ssp).srcu_sup).node }.wrapping_offset((unsafe { rcu_num_nodes }) as isize) {
raw_spin_lock_irq_rcu_node!(snp);
cbs = false;
last_lvl = snp >= unsafe { *((*sup).level.as_mut_ptr() as *mut *mut srcu_node).wrapping_offset(unsafe { rcu_num_lvls }.wrapping_sub(1) as isize) };
if last_lvl {
cbs = ss_state < 8 || (unsafe { *((*snp).srcu_have_cbs.as_mut_ptr() as *mut u64).wrapping_offset(idx as isize) }) == gpseq;
}
unsafe {
*((*snp).srcu_have_cbs.as_mut_ptr() as *mut u64).wrapping_offset(idx as isize) = gpseq;
}
rcu_seq_set_state(unsafe { (*snp).srcu_have_cbs.as_mut_ptr() as *mut u64 }.wrapping_offset(idx as isize), 1i32);
sgsne = unsafe { (*snp).srcu_gp_seq_needed_exp };
if srcu_invl_snp_seq(sgsne) || ULONG_CMP_LT!(sgsne, gpseq) {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_725() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__2!((*snp).srcu_gp_seq_needed_exp, gpseq);
if true {
break;
}
}
}
if ss_state < 8 {
mask = 18446744073709551615u64;
} else {
mask = unsafe { *((*snp).srcu_data_have_cbs.as_mut_ptr() as *mut u64).wrapping_offset(idx as isize) };
}
unsafe {
*((*snp).srcu_data_have_cbs.as_mut_ptr() as *mut u64).wrapping_offset(idx as isize) = 0;
}
raw_spin_unlock_irq_rcu_node!(snp);
if cbs {
srcu_schedule_cbs_snp(ssp, snp, mask, cbdelay);
}
let __v30: *mut srcu_node = snp;
snp = __v30.wrapping_offset(1 as isize);
}
}
// Occasionally prevent srcu_data counter wrap.
if gpseq & unsafe { counter_wrap_check } == 0 {
cpu = 0;
loop {
cpu = find_next_bit(cpumask_bits!((unsafe { core::ptr::addr_of_mut!(__cpu_possible_mask) }) as *const cpumask), 64u64, cpu as u64) as i32;
if (cpu as u32) >= 64 {
break;
}
let sdp: *mut srcu_data = per_cpu_ptr__4!(unsafe { (*ssp).sda }, cpu);
raw_spin_lock_irq_rcu_node!(sdp);
if ULONG_CMP_GE!(gpseq, unsafe { (*sdp).srcu_gp_seq_needed }.wrapping_add(100)) {
unsafe {
(*sdp).srcu_gp_seq_needed = gpseq;
}
}
if ULONG_CMP_GE!(gpseq, unsafe { (*sdp).srcu_gp_seq_needed_exp }.wrapping_add(100)) {
unsafe {
(*sdp).srcu_gp_seq_needed_exp = gpseq;
}
}
raw_spin_unlock_irq_rcu_node!(sdp);
cpu = cpu.wrapping_add(1);
}
}
// Callback initiation done, allow grace periods after next.
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*sup).srcu_cb_mutex)) };
// Start a new grace period if needed.
raw_spin_lock_irq_rcu_node!(sup);
gpseq = rcu_seq_current(unsafe { core::ptr::addr_of_mut!((*sup).srcu_gp_seq) });
if rcu_seq_state(gpseq) == 0 && ULONG_CMP_LT!(gpseq, unsafe { (*sup).srcu_gp_seq_needed }) {
srcu_gp_start(ssp);
raw_spin_unlock_irq_rcu_node!(sup);
srcu_reschedule(ssp, 0u64);
} else {
raw_spin_unlock_irq_rcu_node!(sup);
}
// Transition to big if needed.
if ss_state != 0 && ss_state != 8 {
if ss_state == 1 {
init_srcu_struct_nodes(ssp, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32));
} else {
loop {
kcsan_release!();
loop {
loop {
if !__native_word__5!() {
unsafe { __compiletime_assert_726() };
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
if !(__native_word__5!() || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_727() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__3!(*unsafe { core::ptr::addr_of_mut!((*sup).srcu_size_state) }, ss_state.wrapping_add(1));
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
}
}

// Funnel-locking scheme to scalably mediate many concurrent expedited
// grace-period requests.  This function is invoked for the first known
// expedited request for a grace period that has already been requested,
// but without expediting.  To start a completely new grace period,
// whether expedited or not, use srcu_funnel_gp_start() instead.
fn srcu_funnel_exp_start(ssp: *mut srcu_struct, mut snp: *mut srcu_node, s: u64) {
let mut flags: u64 = 0;
let mut sgsne: u64 = 0;
if !snp.is_null() {
while !snp.is_null() {
let mut __v0: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_728() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*snp).srcu_gp_seq_needed_exp) as *const u64) };
sgsne = __v0;
let mut __v2: i64;
let __ret_warn_on: i32 = !(!rcu_seq_done(unsafe { core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_gp_seq) }, s)) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("729: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 729b - ., 3; .popsection\n# {0}", const 729 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_srcu_funnel_exp_start_3, sym __slate_asm_string_srcu_funnel_exp_start_4, const 1010 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("730: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 730b - ., 4; .popsection\n# {0}", const 730 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v2 = (__ret_warn_on != 0) as i64;
let __v7: bool = __v2 != 0 || !srcu_invl_snp_seq(sgsne) && ULONG_CMP_GE!(sgsne, s);
if __v7 {
return;
}
raw_spin_lock_irqsave_rcu_node!(snp, flags);
sgsne = unsafe { (*snp).srcu_gp_seq_needed_exp };
if !srcu_invl_snp_seq(sgsne) && ULONG_CMP_GE!(sgsne, s) {
raw_spin_unlock_irqrestore_rcu_node!(snp, flags);
return;
}
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_731() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__2!((*snp).srcu_gp_seq_needed_exp, s);
if true {
break;
}
}
raw_spin_unlock_irqrestore_rcu_node!(snp, flags);
snp = unsafe { (*snp).srcu_parent };
}
}
raw_spin_lock_irqsave_ssp_contention(ssp, core::ptr::addr_of_mut!(flags));
if ULONG_CMP_LT!(unsafe { (*(*ssp).srcu_sup).srcu_gp_seq_needed_exp }, s) {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_732() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__2!((*unsafe { (*ssp).srcu_sup }).srcu_gp_seq_needed_exp, s);
if true {
break;
}
}
}
raw_spin_unlock_irqrestore_rcu_node!(unsafe { (*ssp).srcu_sup }, flags);
}

// Funnel-locking scheme to scalably mediate many concurrent grace-period
// requests.  The winner has to do the work of actually starting grace
// period s.  Losers must either ensure that their desired grace-period
// number is recorded on at least their leaf srcu_node structure, or they
// must take steps to invoke their own callbacks.
//
// Note that this function also does the work of srcu_funnel_exp_start(),
// in some cases by directly invoking it.
//
// The srcu read lock should be hold around this function. And s is a seq snap
// after holding that lock.
fn srcu_funnel_gp_start(ssp: *mut srcu_struct, sdp: *mut srcu_data, s: u64, do_norm: bool) {
let mut flags: u64 = 0;
let idx: i32 = (rcu_seq_ctr(s) % ARRAY_SIZE__94!()) as i32;
let mut sgsne: u64 = 0;
let mut snp: *mut srcu_node = unsafe { core::mem::zeroed() };
let mut snp_leaf: *mut srcu_node = unsafe { core::mem::zeroed() };
let mut snp_seq: u64 = 0;
let sup: *mut srcu_usage = unsafe { (*ssp).srcu_sup };
// Ensure that snp node tree is fully initialized before traversing it
let mut __v0: i32;
let mut ___p1: i32 = 0;
let mut __v1: i32;
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_733() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*sup).srcu_size_state) as *const i32) };
___p1 = __v1;
loop {
if !__native_word__5!() {
unsafe { __compiletime_assert_734() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
__v0 = ___p1;
if __v0 < 2 {
snp_leaf = core::ptr::null_mut::<srcu_node>();
} else {
snp_leaf = unsafe { (*sdp).mynode };
}
if !snp_leaf.is_null() {
snp = snp_leaf;
while !snp.is_null() {
let mut __v4: i64;
let __ret_warn_on: i32 = !(!rcu_seq_done(unsafe { core::ptr::addr_of_mut!((*sup).srcu_gp_seq) }, s)) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("735: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 735b - ., 3; .popsection\n# {0}", const 735 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_srcu_funnel_gp_start_5, sym __slate_asm_string_srcu_funnel_gp_start_6, const 1061 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("736: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 736b - ., 4; .popsection\n# {0}", const 736 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v4 = (__ret_warn_on != 0) as i64;
if __v4 != 0 && snp != snp_leaf {
return;
}
// GP already done and CBs recorded.
raw_spin_lock_irqsave_rcu_node!(snp, flags);
snp_seq = unsafe { *((*snp).srcu_have_cbs.as_mut_ptr() as *mut u64).wrapping_offset(idx as isize) };
if !srcu_invl_snp_seq(snp_seq) && ULONG_CMP_GE!(snp_seq, s) {
if snp == snp_leaf && snp_seq == s {
let __v13: *mut u64 = unsafe { (*snp).srcu_data_have_cbs.as_mut_ptr() as *mut u64 }.wrapping_offset(idx as isize);
unsafe {
*__v13 = *__v13 | (*sdp).grpmask;
}
}
raw_spin_unlock_irqrestore_rcu_node!(snp, flags);
if snp == snp_leaf && snp_seq != s {
srcu_schedule_cbs_sdp(sdp, (if do_norm { 1i32 } else { 0i32 }) as u64);
return;
}
if !do_norm {
srcu_funnel_exp_start(ssp, snp, s);
}
return;
}
unsafe {
*((*snp).srcu_have_cbs.as_mut_ptr() as *mut u64).wrapping_offset(idx as isize) = s;
}
if snp == snp_leaf {
let __v20: *mut u64 = unsafe { (*snp).srcu_data_have_cbs.as_mut_ptr() as *mut u64 }.wrapping_offset(idx as isize);
unsafe {
*__v20 = *__v20 | (*sdp).grpmask;
}
}
sgsne = unsafe { (*snp).srcu_gp_seq_needed_exp };
let __v23: bool = !do_norm && (srcu_invl_snp_seq(sgsne) || ULONG_CMP_LT!(sgsne, s));
if __v23 {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_737() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__2!((*snp).srcu_gp_seq_needed_exp, s);
if true {
break;
}
}
}
raw_spin_unlock_irqrestore_rcu_node!(snp, flags);
snp = unsafe { (*snp).srcu_parent };
}
}
// Each pass through the loop does one level of the srcu_node tree.
// Top of tree, must ensure the grace period will be started.
raw_spin_lock_irqsave_ssp_contention(ssp, core::ptr::addr_of_mut!(flags));
if ULONG_CMP_LT!(unsafe { (*sup).srcu_gp_seq_needed }, s) {
// Record need for grace period s.  Pair with load
// acquire setting up for initialization.
loop {
kcsan_release!();
loop {
loop {
if !__native_word!() {
unsafe { __compiletime_assert_738() };
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
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_739() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__2!(*unsafe { core::ptr::addr_of_mut!((*sup).srcu_gp_seq_needed) }, s);
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
// ^^^
}
if !do_norm && ULONG_CMP_LT!(unsafe { (*sup).srcu_gp_seq_needed_exp }, s) {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_740() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__2!((*sup).srcu_gp_seq_needed_exp, s);
if true {
break;
}
}
}
// If grace period not already in progress, start it.
let mut __v42: i64;
let __ret_warn_on: i32 = !(!rcu_seq_done(unsafe { core::ptr::addr_of_mut!((*sup).srcu_gp_seq) }, s)) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("741: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 741b - ., 3; .popsection\n# {0}", const 741 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_srcu_funnel_gp_start_43, sym __slate_asm_string_srcu_funnel_gp_start_44, const 1099 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("742: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 742b - ., 4; .popsection\n# {0}", const 742 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v42 = (__ret_warn_on != 0) as i64;
let __v47: bool = __v42 == 0 && rcu_seq_state(unsafe { (*sup).srcu_gp_seq }) == 0;
if __v47 {
srcu_gp_start(ssp);
// And how can that list_add() in the "else" clause
//
// possibly be safe for concurrent execution?  Well,
//
// it isn't.  And it does not have to be.  After all, it
//
// can only be executed during early boot when there is only
//
// the one boot CPU running with interrupts still disabled.
//
//
//
// Use an irq_work here to avoid acquiring runqueue lock with
//
// srcu rcu_node::lock held. BPF instrument could introduce the
//
// opposite dependency, hence we need to break the possible
//
// locking dependency here.
if likely!(unsafe { srcu_init_done }) != 0 {
unsafe { irq_work_queue(core::ptr::addr_of_mut!((*sup).irq_work)) };
} else {
if list_empty((unsafe { core::ptr::addr_of_mut!((*sup).work.work.entry) }) as *const list_head) != 0 {
list_add(unsafe { core::ptr::addr_of_mut!((*sup).work.work.entry) }, unsafe { core::ptr::addr_of_mut!(srcu_boot_list) });
}
}
}
raw_spin_unlock_irqrestore_rcu_node!(sup, flags);
}

/// Wait until all readers counted by array index idx complete, but
/// loop an additional time if there is an expedited grace period pending.
/// The caller must ensure that ->srcu_ctrp is not changed while checking.
fn try_check_zero(ssp: *mut srcu_struct, idx: i32, mut trycount: i32) -> bool {
let mut curdelay: u64 = 0;
raw_spin_lock_irq_rcu_node!(unsafe { (*ssp).srcu_sup });
curdelay = (srcu_get_delay(ssp) == 0) as u64;
raw_spin_unlock_irq_rcu_node!(unsafe { (*ssp).srcu_sup });
loop {
if srcu_readers_active_idx_check(ssp, idx) {
return true;
}
let __v5: i32 = trycount.wrapping_sub(1);
trycount = __v5;
if (__v5 as u64).wrapping_add(curdelay) <= 0 {
return false;
}
udelay(unsafe { srcu_retry_check_delay });
}
unsafe { core::mem::zeroed() }
}

// Increment the ->srcu_ctrp counter so that future SRCU readers will
// use the other rank of the ->srcu_(un)lock_count[] arrays.  This allows
// us to wait for pre-existing readers in a starvation-free manner.
fn srcu_flip(ssp: *mut srcu_struct) {
// Because the flip of ->srcu_ctrp is executed only if the
// preceding call to srcu_readers_active_idx_check() found that
// the ->srcu_ctrs[].srcu_unlocks and ->srcu_ctrs[].srcu_locks sums
// matched and because that summing uses atomic_long_read(),
// there is ordering due to a control dependency between that
// summing and the WRITE_ONCE() in this call to srcu_flip().
// This ordering ensures that if this updater saw a given reader's
// increment from __srcu_read_lock(), that reader was using a value
// of ->srcu_ctrp from before the previous call to srcu_flip(),
// which should be quite rare.  This ordering thus helps forward
// progress because the grace period could otherwise be delayed
// by additional calls to __srcu_read_lock() using that old (soon
// to be new) value of ->srcu_ctrp.
//
// This sum-equality check and ordering also ensures that if
// a given call to __srcu_read_lock() uses the new value of
// ->srcu_ctrp, this updater's earlier scans cannot have seen
// that reader's increments, which is all to the good, because
// this grace period need not wait on that reader.  After all,
// if those earlier scans had seen that reader, there would have
// been a sum mismatch and this code would not be reached.
//
// This means that the following smp_mb() is redundant, but
// it stays until either (1) Compilers learn about this sort of
// control dependency or (2) Some production workload running on
// a production system is unduly delayed by this slowpath smp_mb().
// Except for _lite() readers, where it is inoperative, which
// means that it is a good thing that it is redundant.
loop {
kcsan_mb!();
unsafe {
core::arch::asm!("lock addl $0,-4(%rsp)", options(nostack, att_syntax));
}
if true {
break;
}
}
// E
//
// Pairs with B and C.
loop {
loop {
if !(__native_word__10!() || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_743() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__55!((*ssp).srcu_ctrp, unsafe { (*(*ssp).sda).srcu_ctrs.as_mut_ptr() as *mut srcu_ctr }.wrapping_offset(((((unsafe { unsafe { (*ssp).srcu_ctrp }.offset_from(((*(*ssp).sda).srcu_ctrs.as_mut_ptr() as *mut srcu_ctr).wrapping_offset(0 as isize) as *mut srcu_ctr) }) as i64) == 0) as i32) as isize));
if true {
break;
}
}
// Ensure that if the updater misses an __srcu_read_unlock()
// increment, that task's __srcu_read_lock() following its next
// __srcu_read_lock() or __srcu_read_unlock() will see the above
// counter update.  Note that both this memory barrier and the
// one in srcu_readers_active_idx_check() provide the guarantee
// for __srcu_read_lock().
//
// Note that this is a performance optimization, in which we spend
// an otherwise unnecessary smp_mb() in order to reduce the number
// of full per-CPU-variable scans in srcu_readers_lock_idx() and
// srcu_readers_unlock_idx().  But this performance optimization
// is not so optimal for SRCU-fast, where we would be spending
// not smp_mb(), but rather synchronize_rcu().  At the same time,
// the overhead of the smp_mb() is in the noise, so there is no
// point in omitting it in the SRCU-fast case.  So the same code
// is executed either way.
loop {
kcsan_mb!();
unsafe {
core::arch::asm!("lock addl $0,-4(%rsp)", options(nostack, att_syntax));
}
if true {
break;
}
}
// D
//
// Pairs with C.
}

// If SRCU is likely idle, in other words, the next SRCU grace period
// should be expedited, return true, otherwise return false.  Except that
// in the presence of _lite() readers, always return false.
//
// Note that it is OK for several current from-idle requests for a new
// grace period from idle to specify expediting because they will all end
// up requesting the same grace period anyhow.  So no loss.
//
// Note also that if any CPU (including the current one) is still invoking
// callbacks, this function will nevertheless say "idle".  This is not
// ideal, but the overhead of checking all CPUs' callback lists is even
// less ideal, especially on large systems.  Furthermore, the wakeup
// can happen before the callback is fully removed, so we have no choice
// but to accept this type of error.
//
// This function is also subject to counter-wrap errors, but let's face
// it, if this function was preempted for enough time for the counters
// to wrap, it really doesn't matter whether or not we expedite the grace
// period.  The extra overhead of a needlessly expedited grace period is
// negligible when amortized over that time period, and the extra latency
// of a needlessly non-expedited grace period is similarly negligible.
fn srcu_should_expedite(mut ssp: *mut srcu_struct) -> bool {
let mut curseq: u64 = 0;
let mut flags: u64 = 0;
let mut sdp: *mut srcu_data = unsafe { core::mem::zeroed() };
let mut t: u64 = 0;
let mut tlast: u64 = 0;
check_init_srcu_struct(ssp);
// If _lite() readers, don't do unsolicited expediting.
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
__verify_pcpu_ptr!();
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:({1:r}), {0:l}", lateout(reg) __t0, in(reg) (core::ptr::addr_of_mut!((*(*ssp).sda).srcu_reader_flavor) as u64) as *mut i32, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v2 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v2;
}
2 => {
let mut __v4: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:({1:r}), {0:x}", lateout(reg) __t1, in(reg) (core::ptr::addr_of_mut!((*(*ssp).sda).srcu_reader_flavor) as u64) as *mut i32, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v4 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v4;
}
4 => {
let mut __v5: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:({1:r}), {0:e}", lateout(reg) __t2, in(reg) (core::ptr::addr_of_mut!((*(*ssp).sda).srcu_reader_flavor) as u64) as *mut i32, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v5 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v5;
}
8 => {
let mut __v6: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:({1:r}), {0:r}", lateout(reg) __t3, in(reg) (core::ptr::addr_of_mut!((*(*ssp).sda).srcu_reader_flavor) as u64) as *mut i32, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v6 = pfo_val__ as i32;
pscr_ret__ = __v6;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v0 = pscr_ret__;
if __v0 & (4 | 8) != 0 {
return false;
}
// If the local srcu_data structure has callbacks, not idle.
let mut __v7: *mut srcu_data;
__verify_pcpu_ptr__27!();
let mut __v9: *mut srcu_data;
let mut tcp_ptr__: u64 = 0;
let mut __v10: u64;
let mut pfo_val__: u64 = 0;
unsafe {
let __t4: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t4, sym this_cpu_off, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t4 as u64) };
}
__v10 = pfo_val__;
tcp_ptr__ = __v10;
tcp_ptr__ = tcp_ptr__.wrapping_add((unsafe { (*ssp).sda }) as u64);
__v9 = tcp_ptr__ as *mut srcu_data;
__v7 = __v9;
sdp = __v7;
raw_spin_lock_irqsave_rcu_node!(sdp, flags);
if unsafe { rcu_segcblist_pend_cbs(core::ptr::addr_of_mut!((*sdp).srcu_cblist)) } {
raw_spin_unlock_irqrestore_rcu_node!(sdp, flags);
return false; // Callbacks already present, so not idle.
}
raw_spin_unlock_irqrestore_rcu_node!(sdp, flags);
// No local callbacks, so probabilistically probe global state.
// Exact information would require acquiring locks, which would
// kill scalability, hence the probabilistic nature of the probe.
// First, see if enough time has passed since the last GP.
t = unsafe { ktime_get_mono_fast_ns() };
let mut __v25: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_744() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v25 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_last_gp_end) as *const u64) };
tlast = __v25;
let __v27: bool = (unsafe { exp_holdoff }) == 0 || time_in_range_open!(t, tlast, tlast.wrapping_add(unsafe { exp_holdoff }));
if __v27 {
return false;
}
// Too soon after last GP.
// Next, check for probable idleness.
curseq = rcu_seq_current(unsafe { core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_gp_seq) });
loop {
kcsan_mb!();
unsafe {
core::arch::asm!("lock addl $0,-4(%rsp)", options(nostack, att_syntax));
}
if true {
break;
}
}
// Order ->srcu_gp_seq with ->srcu_gp_seq_needed.
let mut __v45: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_745() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v45 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_gp_seq_needed) as *const u64) };
if !0 / 2 < curseq.wrapping_sub(__v45) {
return false;
}
// Grace period in progress, so not idle.
loop {
kcsan_mb!();
unsafe {
core::arch::asm!("lock addl $0,-4(%rsp)", options(nostack, att_syntax));
}
if true {
break;
}
}
// Order ->srcu_gp_seq with prior access.
if curseq != rcu_seq_current(unsafe { core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_gp_seq) }) {
return false;
}
// GP # changed, so not idle.
return true; // With reasonable probability, idle!
}

/// SRCU callback function to leak a callback.
#[unsafe(link_section = ".text.slate_distinct.srcutree.srcu_leak_callback")]
extern "C" fn srcu_leak_callback(rhp: *mut callback_head) {
}

// Start an SRCU grace period, and also queue the callback if non-NULL.
fn srcu_gp_start_if_needed(ssp: *mut srcu_struct, rhp: *mut callback_head, do_norm: bool) -> u64 {
let mut flags: u64 = 0;
let mut idx: i32 = 0;
let mut needexp: bool = false;
let mut needgp: bool = false;
let mut s: u64 = 0;
let mut sdp: *mut srcu_data = unsafe { core::mem::zeroed() };
let mut sdp_mynode: *mut srcu_node = unsafe { core::mem::zeroed() };
let mut ss_state: i32 = 0;
check_init_srcu_struct(ssp);
// While starting a new grace period, make sure we are in an
// SRCU read-side critical section so that the grace-period
// sequence number cannot wrap around in the meantime.
idx = __srcu_read_lock_nmisafe(ssp);
let mut __v0: i32;
let mut ___p1: i32 = 0;
let mut __v1: i32;
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<i32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_746() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_size_state) as *const i32) };
___p1 = __v1;
loop {
if !__native_word__5!() {
unsafe { __compiletime_assert_747() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
__v0 = ___p1;
ss_state = __v0;
// If !rcu_cpu_beenfullyonline(), interrupts are still disabled,
//
// so no migration is possible in either direction from this CPU.
let __v4: bool = if ss_state < 3 {
true
} else {
let mut __v5: i32;
let mut pscr_ret__: i32 = 0;
__verify_pcpu_ptr!();
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v7: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
__v7 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v7;
}
2 => {
let mut __v9: i32;
let mut pfo_val__: u16 = 0;
unsafe {
let __t1: u16;
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
__v9 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v9;
}
4 => {
let mut __v10: i32;
let mut pfo_val__: u32 = 0;
unsafe {
let __t2: u32;
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
__v10 = (pfo_val__ as u64) as i32;
pscr_ret__ = __v10;
}
8 => {
let mut __v11: i32;
let mut pfo_val__: u64 = 0;
unsafe {
let __t3: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
__v11 = pfo_val__ as i32;
pscr_ret__ = __v11;
}
_ => {
unsafe { __bad_size_call_parameter() };
}
}
__v5 = pscr_ret__;
!unsafe { rcu_cpu_beenfullyonline(__v5) }
};
if __v4 {
sdp = per_cpu_ptr__4!(unsafe { (*ssp).sda }, get_boot_cpu_id());
} else {
let mut __v15: *mut srcu_data;
__verify_pcpu_ptr__27!();
let mut __v17: *mut srcu_data;
let mut tcp_ptr__: u64 = 0;
let mut __v18: u64;
let mut pfo_val__: u64 = 0;
unsafe {
let __t4: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t4, sym this_cpu_off, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t4 as u64) };
}
__v18 = pfo_val__;
tcp_ptr__ = __v18;
tcp_ptr__ = tcp_ptr__.wrapping_add((unsafe { (*ssp).sda }) as u64);
__v17 = tcp_ptr__ as *mut srcu_data;
__v15 = __v17;
sdp = __v15;
}
raw_spin_lock_irqsave_sdp_contention(sdp, core::ptr::addr_of_mut!(flags));
if !rhp.is_null() {
unsafe { rcu_segcblist_enqueue(core::ptr::addr_of_mut!((*sdp).srcu_cblist), rhp) };
}
// It's crucial to capture the snapshot 's' for acceleration before
// reading the current gp_seq that is used for advancing. This is
// essential because if the acceleration snapshot is taken after a
// failed advancement attempt, there's a risk that a grace period may
// conclude and a new one may start in the interim. If the snapshot is
// captured after this sequence of events, the acceleration snapshot 's'
// could be excessively advanced, leading to acceleration failure.
// In such a scenario, an 'acceleration leak' can occur, where new
// callbacks become indefinitely stuck in the RCU_NEXT_TAIL segment.
// Also note that encountering advancing failures is a normal
// occurrence when the grace period for RCU_WAIT_TAIL is in progress.
//
// To see this, consider the following events which occur if
// rcu_seq_snap() were to be called after advance:
//
//  1) The RCU_WAIT_TAIL segment has callbacks (gp_num = X + 4) and the
//     RCU_NEXT_READY_TAIL also has callbacks (gp_num = X + 8).
//
//  2) The grace period for RCU_WAIT_TAIL is seen as started but not
//     completed so rcu_seq_current() returns X + SRCU_STATE_SCAN1.
//
//  3) This value is passed to srcu_segcblist_advance() which can't move
//     any segment forward and fails.
//
//  4) srcu_gp_start_if_needed() still proceeds with callback acceleration.
//     But then the call to rcu_seq_snap() observes the grace period for the
//     RCU_WAIT_TAIL segment as completed and the subsequent one for the
//     RCU_NEXT_READY_TAIL segment as started (ie: X + 4 + SRCU_STATE_SCAN1)
//     so it returns a snapshot of the next grace period, which is X + 12.
//
//  5) The value of X + 12 is passed to srcu_segcblist_accelerate() but the
//     freshly enqueued callback in RCU_NEXT_TAIL can't move to
//     RCU_NEXT_READY_TAIL which already has callbacks for a previous grace
//     period (gp_num = X + 8). So acceleration fails.
s = rcu_seq_snap(unsafe { core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_gp_seq) });
if !rhp.is_null() {
unsafe { srcu_segcblist_advance(core::ptr::addr_of_mut!((*sdp).srcu_cblist), rcu_seq_current(core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_gp_seq))) };
// Acceleration can never fail because the base current gp_seq
// used for acceleration is <= the value of gp_seq used for
// advancing. This means that RCU_NEXT_TAIL segment will
// always be able to be emptied by the acceleration into the
// RCU_NEXT_READY_TAIL or RCU_WAIT_TAIL segments.
let mut __v21: i64;
let __ret_warn_on: i32 = !(!(!unsafe { srcu_segcblist_accelerate(core::ptr::addr_of_mut!((*sdp).srcu_cblist), s) })) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("748: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 748b - ., 3; .popsection\n# {0}", const 748 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_srcu_gp_start_if_needed_22, sym __slate_asm_string_srcu_gp_start_if_needed_23, const 1360 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("749: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 749b - ., 4; .popsection\n# {0}", const 749 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v21 = (__ret_warn_on != 0) as i64;
}
if ULONG_CMP_LT!(unsafe { (*sdp).srcu_gp_seq_needed }, s) {
unsafe {
(*sdp).srcu_gp_seq_needed = s;
}
needgp = true;
}
if !do_norm && ULONG_CMP_LT!(unsafe { (*sdp).srcu_gp_seq_needed_exp }, s) {
unsafe {
(*sdp).srcu_gp_seq_needed_exp = s;
}
needexp = true;
}
raw_spin_unlock_irqrestore_rcu_node!(sdp, flags);
// Ensure that snp node tree is fully initialized before traversing it
if ss_state < 2 {
sdp_mynode = core::ptr::null_mut::<srcu_node>();
} else {
sdp_mynode = unsafe { (*sdp).mynode };
}
if needgp {
srcu_funnel_gp_start(ssp, sdp, s, do_norm);
} else {
if needexp {
srcu_funnel_exp_start(ssp, sdp_mynode, s);
}
}
__srcu_read_unlock_nmisafe(ssp, idx);
s
}

// Enqueue an SRCU callback on the srcu_data structure associated with
// the current CPU and the specified srcu_struct structure, initiating
// grace-period processing if it is not already running.
//
// Note that all CPUs must agree that the grace period extended beyond
// all pre-existing SRCU read-side critical section.  On systems with
// more than one CPU, this means that when "func()" is invoked, each CPU
// is guaranteed to have executed a full memory barrier since the end of
// its last corresponding SRCU read-side critical section whose beginning
// preceded the call to call_srcu().  It also means that each CPU executing
// an SRCU read-side critical section that continues beyond the start of
// "func()" must have executed a memory barrier after the call_srcu()
// but before the beginning of that SRCU read-side critical section.
// Note that these guarantees include CPUs that are offline, idle, or
// executing in user mode, as well as CPUs that are executing in the kernel.
//
// Furthermore, if CPU A invoked call_srcu() and CPU B invoked the
// resulting SRCU callback function "func()", then both CPU A and CPU
// B are guaranteed to execute a full memory barrier during the time
// interval between the call to call_srcu() and the invocation of "func()".
// This guarantee applies even if CPU A and CPU B are the same CPU (but
// again only if the system has more than one CPU).
//
// Of course, these guarantees apply only for invocations of call_srcu(),
// srcu_read_lock(), and srcu_read_unlock() that are all passed the same
// srcu_struct structure.
fn __call_srcu(ssp: *mut srcu_struct, rhp: *mut callback_head, func: Option<unsafe extern "C" fn(*mut callback_head)>, do_norm: bool) {
if debug_rcu_head_queue(rhp) != 0 {
// Probable double call_srcu(), so leak the callback.
loop {
loop {
if !(__native_word__63!() || ((const { core::mem::size_of::<Option<unsafe extern "C" fn(*mut callback_head)>>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_750() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__37!((*rhp).func, Some(srcu_leak_callback));
if true {
break;
}
}
let mut __v3: i32;
let __ret_warn_on: i32 = true as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
2i32 | BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"call_srcu(): Leaked duplicate callback\n".as_ptr() as *const u8) };
let mut __v4: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v4 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v5: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string___call_srcu_6, sym __slate_asm_string___call_srcu_7, const 1420 as i32, const 2323 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v5 = bug;
unsafe { __v4.unwrap()(__v5) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v3 = __ret_warn_on;
return;
}
unsafe {
(*rhp).func = func;
}
srcu_gp_start_if_needed(ssp, rhp, do_norm);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_751: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

/// call_srcu() - Queue a callback for invocation after an SRCU grace period
/// @ssp: srcu_struct in queue the callback
/// @rhp: structure to be used for queueing the SRCU callback.
/// @func: function to be invoked after the SRCU grace period
///
/// The callback function will be invoked some time after a full SRCU
/// grace period elapses, in other words after all pre-existing SRCU
/// read-side critical sections have completed.  However, the callback
/// function might well execute concurrently with other SRCU read-side
/// critical sections that started after call_srcu() was invoked.  SRCU
/// read-side critical sections are delimited by srcu_read_lock() and
/// srcu_read_unlock(), and may be nested.
///
/// The callback will be invoked from process context, but with bh
/// disabled.  The callback function must therefore be fast and must
/// not block.
///
/// See the description of call_rcu() for more detailed information on
/// memory ordering guarantees.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.srcutree.call_srcu")]
extern "C" fn call_srcu(ssp: *mut srcu_struct, rhp: *mut callback_head, func: Option<unsafe extern "C" fn(*mut callback_head)>) {
__call_srcu(ssp, rhp, func, true);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_call_srcu_752: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut srcu_struct, *mut callback_head, Option<unsafe extern "C" fn(*mut callback_head)>)>, *mut ()>(Some(call_srcu)) };

/// Helper function for synchronize_srcu() and synchronize_srcu_expedited().
fn __synchronize_srcu(ssp: *mut srcu_struct, do_norm: bool) {
let mut rcu: rcu_synchronize = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(rcu) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 64usize);
}
srcu_lock_sync!();
loop {
let __v2: bool = if false {
let __v3: bool = (unsafe { lockdep_is_held(ssp as *const ()) }) != 0 || (unsafe { lock_is_held(core::ptr::addr_of_mut!(rcu_bh_lock_map) as *const ()) }) != 0;
let __v4: bool = __v3 || (unsafe { lock_is_held(core::ptr::addr_of_mut!(rcu_lock_map) as *const ()) }) != 0;
let __v5: bool = __v4 || (unsafe { lock_is_held(core::ptr::addr_of_mut!(rcu_sched_lock_map) as *const ()) }) != 0;
__v5
} else {
false
};
if !__v2 {
break;
}
}
if (unsafe { rcu_scheduler_active }) == 0 {
return;
}
might_sleep!();
check_init_srcu_struct(ssp);
init_completion(core::ptr::addr_of_mut!(rcu.completion));
init_rcu_head_on_stack(unsafe { &(*core::ptr::addr_of_mut!(rcu.head)) });
__call_srcu(ssp, core::ptr::addr_of_mut!(rcu.head), unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut callback_head)>>(wakeme_after_rcu as *const ()) }, do_norm);
unsafe { wait_for_completion(core::ptr::addr_of_mut!(rcu.completion)) };
destroy_rcu_head_on_stack(unsafe { &(*core::ptr::addr_of_mut!(rcu.head)) });
// Make sure that later code is ordered after the SRCU grace
// period.  This pairs with the raw_spin_lock_irq_rcu_node()
// in srcu_invoke_callbacks().  Unlike Tree RCU, this is needed
// because the current CPU might have been totally uninvolved with
// (and thus unordered against) that grace period.
loop {
kcsan_mb!();
unsafe {
core::arch::asm!("lock addl $0,-4(%rsp)", options(nostack, att_syntax));
}
if true {
break;
}
}
}

/// synchronize_srcu_expedited - Brute-force SRCU grace period
/// @ssp: srcu_struct with which to synchronize.
///
/// Wait for an SRCU grace period to elapse, but be more aggressive about
/// spinning rather than blocking when waiting.
///
/// Note that synchronize_srcu_expedited() has the same deadlock and
/// memory-ordering properties as does synchronize_srcu().
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.srcutree.synchronize_srcu_expedited")]
extern "C" fn synchronize_srcu_expedited(ssp: *mut srcu_struct) {
__synchronize_srcu(ssp, unsafe { rcu_gp_is_normal() });
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_synchronize_srcu_expedited_753: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut srcu_struct)>, *mut ()>(Some(synchronize_srcu_expedited)) };

/// synchronize_srcu - wait for prior SRCU read-side critical-section completion
/// @ssp: srcu_struct with which to synchronize.
///
/// Wait for the count to drain to zero of both indexes. To avoid the
/// possible starvation of synchronize_srcu(), it waits for the count of
/// the index=!(ssp->srcu_ctrp - &ssp->sda->srcu_ctrs[0]) to drain to zero
/// at first, and then flip the ->srcu_ctrp and wait for the count of the
/// other index.
///
/// Can block; must be called from process context.
///
/// Note that it is illegal to call synchronize_srcu() from the corresponding
/// SRCU read-side critical section; doing so will result in deadlock.
/// However, it is perfectly legal to call synchronize_srcu() on one
/// srcu_struct from some other srcu_struct's read-side critical section,
/// as long as the resulting graph of srcu_structs is acyclic.
///
/// There are memory-ordering constraints implied by synchronize_srcu().
/// On systems with more than one CPU, when synchronize_srcu() returns,
/// each CPU is guaranteed to have executed a full memory barrier since
/// the end of its last corresponding SRCU read-side critical section
/// whose beginning preceded the call to synchronize_srcu().  In addition,
/// each CPU having an SRCU read-side critical section that extends beyond
/// the return from synchronize_srcu() is guaranteed to have executed a
/// full memory barrier after the beginning of synchronize_srcu() and before
/// the beginning of that SRCU read-side critical section.  Note that these
/// guarantees include CPUs that are offline, idle, or executing in user mode,
/// as well as CPUs that are executing in the kernel.
///
/// Furthermore, if CPU A invoked synchronize_srcu(), which returned
/// to its caller on CPU B, then both CPU A and CPU B are guaranteed
/// to have executed a full memory barrier during the execution of
/// synchronize_srcu().  This guarantee applies even if CPU A and CPU B
/// are the same CPU, but again only if the system has more than one CPU.
///
/// Of course, these memory-ordering guarantees apply only when
/// synchronize_srcu(), srcu_read_lock(), and srcu_read_unlock() are
/// passed the same srcu_struct structure.
///
/// Implementation of these memory-ordering guarantees is similar to
/// that of synchronize_rcu().
///
/// If SRCU is likely idle as determined by srcu_should_expedite(),
/// expedite the first request.  This semantic was provided by Classic SRCU,
/// and is relied upon by its users, so TREE SRCU must also provide it.
/// Note that detecting idleness is heuristic and subject to both false
/// positives and negatives.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.srcutree.synchronize_srcu")]
extern "C" fn synchronize_srcu(ssp: *mut srcu_struct) {
let __v0: bool = srcu_should_expedite(ssp) || unsafe { rcu_gp_is_expedited() };
if __v0 {
synchronize_srcu_expedited(ssp);
} else {
__synchronize_srcu(ssp, true);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_synchronize_srcu_754: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut srcu_struct)>, *mut ()>(Some(synchronize_srcu)) };

/// get_state_synchronize_srcu - Provide an end-of-grace-period cookie
/// @ssp: srcu_struct to provide cookie for.
///
/// This function returns a cookie that can be passed to
/// poll_state_synchronize_srcu(), which will return true if a full grace
/// period has elapsed in the meantime.  It is the caller's responsibility
/// to make sure that grace period happens, for example, by invoking
/// call_srcu() after return from get_state_synchronize_srcu().
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.srcutree.get_state_synchronize_srcu")]
extern "C" fn get_state_synchronize_srcu(ssp: *mut srcu_struct) -> u64 {
// Any prior manipulation of SRCU-protected data must happen
//
// before the load from ->srcu_gp_seq.
loop {
kcsan_mb!();
unsafe {
core::arch::asm!("lock addl $0,-4(%rsp)", options(nostack, att_syntax));
}
if true {
break;
}
}
rcu_seq_snap(unsafe { core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_gp_seq) })
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_get_state_synchronize_srcu_755: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut srcu_struct) -> u64>, *mut ()>(Some(get_state_synchronize_srcu)) };

/// start_poll_synchronize_srcu - Provide cookie and start grace period
/// @ssp: srcu_struct to provide cookie for.
///
/// This function returns a cookie that can be passed to
/// poll_state_synchronize_srcu(), which will return true if a full grace
/// period has elapsed in the meantime.  Unlike get_state_synchronize_srcu(),
/// this function also ensures that any needed SRCU grace period will be
/// started.  This convenience does come at a cost in terms of CPU overhead.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.srcutree.start_poll_synchronize_srcu")]
extern "C" fn start_poll_synchronize_srcu(ssp: *mut srcu_struct) -> u64 {
srcu_gp_start_if_needed(ssp, core::ptr::null_mut::<callback_head>(), true)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_start_poll_synchronize_srcu_756: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut srcu_struct) -> u64>, *mut ()>(Some(start_poll_synchronize_srcu)) };

/// poll_state_synchronize_srcu - Has cookie's grace period ended?
/// @ssp: srcu_struct to provide cookie for.
/// @cookie: Return value from get_state_synchronize_srcu() or start_poll_synchronize_srcu().
///
/// This function takes the cookie that was returned from either
/// get_state_synchronize_srcu() or start_poll_synchronize_srcu(), and
/// returns @true if an SRCU grace period elapsed since the time that the
/// cookie was created.
///
/// Because cookies are finite in size, wrapping/overflow is possible.
/// This is more pronounced on 32-bit systems where cookies are 32 bits,
/// where in theory wrapping could happen in about 14 hours assuming
/// 25-microsecond expedited SRCU grace periods.  However, a more likely
/// overflow lower bound is on the order of 24 days in the case of
/// one-millisecond SRCU grace periods.  Of course, wrapping in a 64-bit
/// system requires geologic timespans, as in more than seven million years
/// even for expedited SRCU grace periods.
///
/// Wrapping/overflow is much more of an issue for CONFIG_SMP=n systems
/// that also have CONFIG_PREEMPTION=n, which selects Tiny SRCU.  This uses
/// a 16-bit cookie, which rcutorture routinely wraps in a matter of a
/// few minutes.  If this proves to be a problem, this counter will be
/// expanded to the same size as for Tree SRCU.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.srcutree.poll_state_synchronize_srcu")]
extern "C" fn poll_state_synchronize_srcu(ssp: *mut srcu_struct, cookie: u64) -> bool {
let __v0: bool = cookie != 1 && !rcu_seq_done_exact(unsafe { core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_gp_seq) }, cookie);
if __v0 {
return false;
}
// Ensure that the end of the SRCU grace period happens before
//
// any subsequent code that the caller might execute.
loop {
kcsan_mb!();
unsafe {
core::arch::asm!("lock addl $0,-4(%rsp)", options(nostack, att_syntax));
}
if true {
break;
}
}
// ^^^
true
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_poll_state_synchronize_srcu_757: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut srcu_struct, u64) -> bool>, *mut ()>(Some(poll_state_synchronize_srcu)) };

/// Callback function for srcu_barrier() use.
#[unsafe(link_section = ".text.slate_distinct.srcutree.srcu_barrier_cb")]
extern "C" fn srcu_barrier_cb(rhp: *mut callback_head) {
let mut ssp: *mut srcu_struct = unsafe { core::mem::zeroed() };
unsafe {
(*rhp).next = rhp;
}
// Mark the callback as having been invoked.
let sdp: *mut srcu_data = container_of__98!(rhp, srcu_data);
ssp = unsafe { (*sdp).ssp };
if atomic_dec_and_test(unsafe { core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_barrier_cpu_cnt) }) {
unsafe { complete(core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_barrier_completion)) };
}
}

/// Enqueue an srcu_barrier() callback on the specified srcu_data
/// structure's ->cblist.  but only if that ->cblist already has at least one
/// callback enqueued.  Note that if a CPU already has callbacks enqueue,
/// it must have already registered the need for a future grace period,
/// so all we need do is enqueue a callback that will use the same grace
/// period as the last callback already in the queue.
fn srcu_barrier_one_cpu(ssp: *mut srcu_struct, sdp: *mut srcu_data) {
raw_spin_lock_irq_rcu_node!(sdp);
atomic_inc(unsafe { core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_barrier_cpu_cnt) });
unsafe {
(*sdp).srcu_barrier_head.func = Some(srcu_barrier_cb);
}
debug_rcu_head_queue(unsafe { core::ptr::addr_of_mut!((*sdp).srcu_barrier_head) });
if !unsafe { rcu_segcblist_entrain(core::ptr::addr_of_mut!((*sdp).srcu_cblist), core::ptr::addr_of_mut!((*sdp).srcu_barrier_head)) } {
debug_rcu_head_unqueue(unsafe { core::ptr::addr_of_mut!((*sdp).srcu_barrier_head) });
atomic_dec(unsafe { core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_barrier_cpu_cnt) });
}
raw_spin_unlock_irq_rcu_node!(sdp);
}

// srcu_barrier - Wait until all in-flight call_srcu() callbacks complete.
// @ssp: srcu_struct on which to wait for in-flight callbacks.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_srcu_barrier_760: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut srcu_struct)>, *mut ()>(Some(srcu_barrier)) };

/// Callback for srcu_expedite_current() usage.
#[unsafe(link_section = ".text.slate_distinct.srcutree.srcu_expedite_current_cb")]
extern "C" fn srcu_expedite_current_cb(rhp: *mut callback_head) {
let mut flags: u64 = 0;
let mut needcb: bool = false;
let sdp: *mut srcu_data = container_of__99!(rhp, srcu_data);
raw_spin_lock_irqsave_sdp_contention(sdp, core::ptr::addr_of_mut!(flags));
if (unsafe { (*sdp).srcu_ec_state }) == 0 {
let mut __v1: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("761: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 761b - ., 3; .popsection\n# {0}", const 761 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_srcu_expedite_current_cb_2, sym __slate_asm_string_srcu_expedite_current_cb_3, const 1723 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("762: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 762b - ., 4; .popsection\n# {0}", const 762 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
} else {
if (unsafe { (*sdp).srcu_ec_state }) == 1 {
unsafe {
(*sdp).srcu_ec_state = 0;
}
} else {
let mut __v6: i64;
let __ret_warn_on: i32 = ((unsafe { (*sdp).srcu_ec_state }) != 2) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("763: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 763b - ., 3; .popsection\n# {0}", const 763 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_srcu_expedite_current_cb_7, sym __slate_asm_string_srcu_expedite_current_cb_8, const 1727 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("764: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 764b - ., 4; .popsection\n# {0}", const 764 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v6 = (__ret_warn_on != 0) as i64;
unsafe {
(*sdp).srcu_ec_state = 1;
}
needcb = true;
}
}
raw_spin_unlock_irqrestore_rcu_node!(sdp, flags);
// If needed, requeue ourselves as an expedited SRCU callback.
if needcb {
__call_srcu(unsafe { (*sdp).ssp }, unsafe { core::ptr::addr_of_mut!((*sdp).srcu_ec_head) }, Some(srcu_expedite_current_cb), false);
}
}

/// srcu_expedite_current - Expedite the current SRCU grace period
/// @ssp: srcu_struct to expedite.
///
/// Cause the current SRCU grace period to become expedited.  The grace
/// period following the current one might also be expedited.  If there is
/// no current grace period, one might be created.  If the current grace
/// period is currently sleeping, that sleep will complete before expediting
/// will take effect.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.srcutree.srcu_expedite_current")]
extern "C" fn srcu_expedite_current(ssp: *mut srcu_struct) {
let mut flags: u64 = 0;
let mut needcb: bool = false;
let mut sdp: *mut srcu_data = unsafe { core::mem::zeroed() };
migrate_disable();
let mut __v0: *mut srcu_data;
__verify_pcpu_ptr__27!();
let mut __v2: *mut srcu_data;
let mut tcp_ptr__: u64 = 0;
let mut __v3: u64;
let mut pfo_val__: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t0, sym this_cpu_off, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
__v3 = pfo_val__;
tcp_ptr__ = __v3;
tcp_ptr__ = tcp_ptr__.wrapping_add((unsafe { (*ssp).sda }) as u64);
__v2 = tcp_ptr__ as *mut srcu_data;
__v0 = __v2;
sdp = __v0;
raw_spin_lock_irqsave_sdp_contention(sdp, core::ptr::addr_of_mut!(flags));
if (unsafe { (*sdp).srcu_ec_state }) == 0 {
unsafe {
(*sdp).srcu_ec_state = 1;
}
needcb = true;
} else {
if (unsafe { (*sdp).srcu_ec_state }) == 1 {
unsafe {
(*sdp).srcu_ec_state = 2;
}
} else {
let mut __v6: i64;
let __ret_warn_on: i32 = ((unsafe { (*sdp).srcu_ec_state }) != 2) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("765: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 765b - ., 3; .popsection\n# {0}", const 765 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_srcu_expedite_current_7, sym __slate_asm_string_srcu_expedite_current_8, const 1762 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("766: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 766b - ., 4; .popsection\n# {0}", const 766 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v6 = (__ret_warn_on != 0) as i64;
}
}
raw_spin_unlock_irqrestore_rcu_node!(sdp, flags);
// If needed, queue an expedited SRCU callback.
if needcb {
__call_srcu(ssp, unsafe { core::ptr::addr_of_mut!((*sdp).srcu_ec_head) }, Some(srcu_expedite_current_cb), false);
}
migrate_enable();
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_srcu_expedite_current_767: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut srcu_struct)>, *mut ()>(Some(srcu_expedite_current)) };

// srcu_batches_completed - return batches completed.
// @ssp: srcu_struct on which to report batch completion.
//
// Report the number of batches, correlated with, but not necessarily
// precisely the same as, the number of grace periods that have elapsed.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_srcu_batches_completed_769: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut srcu_struct) -> u64>, *mut ()>(Some(srcu_batches_completed)) };

// Core SRCU state machine.  Push state bits of ->srcu_gp_seq
// to SRCU_STATE_SCAN2, and invoke srcu_gp_end() when scan has
// completed in that state.
fn srcu_advance_state(ssp: *mut srcu_struct) {
let mut idx: i32 = 0;
unsafe { mutex_lock(core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_gp_mutex)) };
// Because readers might be delayed for an extended period after
// fetching ->srcu_ctrp for their index, at any point in time there
// might well be readers using both idx=0 and idx=1.  We therefore
// need to wait for readers to clear from both index values before
// invoking a callback.
//
// The load-acquire ensures that we see the accesses performed
// by the prior grace period.
let mut __v0: u64;
let mut ___p1: u64 = 0;
let mut __v1: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_770() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_gp_seq) as *const u64) };
___p1 = __v1;
loop {
if !__native_word!() {
unsafe { __compiletime_assert_771() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
__v0 = ___p1;
idx = rcu_seq_state(__v0); // ^^^
if idx == 0 {
raw_spin_lock_irq_rcu_node!(unsafe { (*ssp).srcu_sup });
if ULONG_CMP_GE!(unsafe { (*(*ssp).srcu_sup).srcu_gp_seq }, unsafe { (*(*ssp).srcu_sup).srcu_gp_seq_needed }) {
let mut __v6: i64;
let __ret_warn_on: i32 = (rcu_seq_state(unsafe { (*(*ssp).srcu_sup).srcu_gp_seq }) != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("772: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 772b - ., 3; .popsection\n# {0}", const 772 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_srcu_advance_state_7, sym __slate_asm_string_srcu_advance_state_8, const 1810 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("773: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 773b - ., 4; .popsection\n# {0}", const 773 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v6 = (__ret_warn_on != 0) as i64;
raw_spin_unlock_irq_rcu_node!(unsafe { (*ssp).srcu_sup });
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_gp_mutex)) };
return;
}
let mut __v13: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_774() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v13 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_gp_seq) as *const u64) };
idx = rcu_seq_state(__v13);
if idx == 0 {
srcu_gp_start(ssp);
}
raw_spin_unlock_irq_rcu_node!(unsafe { (*ssp).srcu_sup });
if idx != 0 {
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_gp_mutex)) };
return; // Someone else started the grace period.
}
}
let mut __v17: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_775() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v17 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_gp_seq) as *const u64) };
if rcu_seq_state(__v17) == 1 {
idx = (((unsafe { unsafe { (*ssp).srcu_ctrp }.offset_from(((*(*ssp).sda).srcu_ctrs.as_mut_ptr() as *mut srcu_ctr).wrapping_offset(0 as isize) as *mut srcu_ctr) }) as i64) == 0) as i32;
if !try_check_zero(ssp, idx, 1i32) {
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_gp_mutex)) };
return; // readers present, retry later.
}
srcu_flip(ssp);
raw_spin_lock_irq_rcu_node!(unsafe { (*ssp).srcu_sup });
rcu_seq_set_state(unsafe { core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_gp_seq) }, 2i32);
unsafe {
(*(*ssp).srcu_sup).srcu_n_exp_nodelay = 0;
}
raw_spin_unlock_irq_rcu_node!(unsafe { (*ssp).srcu_sup });
}
let mut __v23: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_776() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v23 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_gp_seq) as *const u64) };
if rcu_seq_state(__v23) == 2 {
// SRCU read-side critical sections are normally short,
// so check at least twice in quick succession after a flip.
idx = (((unsafe { unsafe { (*ssp).srcu_ctrp }.offset_from(((*(*ssp).sda).srcu_ctrs.as_mut_ptr() as *mut srcu_ctr).wrapping_offset(0 as isize) as *mut srcu_ctr) }) as i64) == 0) as i32;
if !try_check_zero(ssp, idx, 2i32) {
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_gp_mutex)) };
return; // readers present, retry later.
}
unsafe {
(*(*ssp).srcu_sup).srcu_n_exp_nodelay = 0;
}
srcu_gp_end(ssp); // Releases ->srcu_gp_mutex.
}
}

/// Invoke a limited number of SRCU callbacks that have passed through
/// their grace period.  If there are more to do, SRCU will reschedule
/// the workqueue.  Note that needed memory barriers have been executed
/// in this task's context by srcu_readers_active_idx_check().
#[unsafe(link_section = ".text.slate_distinct.srcutree.srcu_invoke_callbacks")]
extern "C" fn srcu_invoke_callbacks(work: *mut work_struct) {
let mut len: i64 = 0;
let mut more: bool = false;
let mut ready_cbs: rcu_cblist = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(ready_cbs) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 24usize);
}
let mut rhp: *mut callback_head = unsafe { core::mem::zeroed() };
let mut sdp: *mut srcu_data = unsafe { core::mem::zeroed() };
let mut ssp: *mut srcu_struct = unsafe { core::mem::zeroed() };
sdp = container_of__5!(work, srcu_data);
ssp = unsafe { (*sdp).ssp };
unsafe { rcu_cblist_init(core::ptr::addr_of_mut!(ready_cbs)) };
raw_spin_lock_irq_rcu_node!(sdp);
let mut __v3: i64;
{
let __ret_warn_on: i32 = !(!(!rcu_segcblist_segempty(unsafe { core::ptr::addr_of_mut!((*sdp).srcu_cblist) }, 3i32))) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("777: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 777b - ., 3; .popsection\n# {0}", const 777 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_srcu_invoke_callbacks_4, sym __slate_asm_string_srcu_invoke_callbacks_5, const 1874 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("778: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 778b - ., 4; .popsection\n# {0}", const 778 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v3 = (__ret_warn_on != 0) as i64;
}
unsafe { srcu_segcblist_advance(core::ptr::addr_of_mut!((*sdp).srcu_cblist), rcu_seq_current(core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_gp_seq))) };
// Although this function is theoretically re-entrant, concurrent
// callbacks invocation is disallowed to avoid executing an SRCU barrier
// too early.
let __v8: bool = (unsafe { (*sdp).srcu_cblist_invoking }) || !unsafe { rcu_segcblist_ready_cbs(core::ptr::addr_of_mut!((*sdp).srcu_cblist)) };
if __v8 {
raw_spin_unlock_irq_rcu_node!(sdp);
return; // Someone else on the job or nothing to do.
}
// We are on the job!  Extract and invoke ready callbacks.
unsafe {
(*sdp).srcu_cblist_invoking = true;
}
unsafe { rcu_segcblist_extract_done_cbs(core::ptr::addr_of_mut!((*sdp).srcu_cblist), core::ptr::addr_of_mut!(ready_cbs)) };
len = ready_cbs.len;
raw_spin_unlock_irq_rcu_node!(sdp);
rhp = unsafe { rcu_cblist_dequeue(core::ptr::addr_of_mut!(ready_cbs)) };
while !rhp.is_null() {
debug_rcu_head_unqueue(rhp);
debug_rcu_head_callback(rhp);
local_bh_disable();
unsafe { unsafe { (*rhp).func }.unwrap()(rhp) };
local_bh_enable();
rhp = unsafe { rcu_cblist_dequeue(core::ptr::addr_of_mut!(ready_cbs)) };
}
let mut __v14: i64;
let __ret_warn_on: i32 = (ready_cbs.len != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("779: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 779b - ., 3; .popsection\n# {0}", const 779 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_srcu_invoke_callbacks_15, sym __slate_asm_string_srcu_invoke_callbacks_16, const 1901 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("780: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 780b - ., 4; .popsection\n# {0}", const 780 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v14 = (__ret_warn_on != 0) as i64;
// Update counts, accelerate new callbacks, and if needed,
// schedule another round of callback invocation.
raw_spin_lock_irq_rcu_node!(sdp);
unsafe { rcu_segcblist_add_len(core::ptr::addr_of_mut!((*sdp).srcu_cblist), len.wrapping_neg()) };
unsafe {
(*sdp).srcu_cblist_invoking = false;
}
more = unsafe { rcu_segcblist_ready_cbs(core::ptr::addr_of_mut!((*sdp).srcu_cblist)) };
raw_spin_unlock_irq_rcu_node!(sdp);
// An SRCU barrier or callbacks from previous nesting work pending
if more {
srcu_schedule_cbs_sdp(sdp, 0u64);
}
}

/// Finished one round of SRCU grace period.  Start another if there are
/// more SRCU callbacks queued, otherwise put SRCU into not-running state.
fn srcu_reschedule(ssp: *mut srcu_struct, delay: u64) {
let mut pushgp: bool = true;
raw_spin_lock_irq_rcu_node!(unsafe { (*ssp).srcu_sup });
if ULONG_CMP_GE!(unsafe { (*(*ssp).srcu_sup).srcu_gp_seq }, unsafe { (*(*ssp).srcu_sup).srcu_gp_seq_needed }) {
let mut __v2: i64;
let __ret_warn_on: i32 = (rcu_seq_state(unsafe { (*(*ssp).srcu_sup).srcu_gp_seq }) != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("781: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 781b - ., 3; .popsection\n# {0}", const 781 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_srcu_reschedule_3, sym __slate_asm_string_srcu_reschedule_4, const 1927 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("782: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 782b - ., 4; .popsection\n# {0}", const 782 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v2 = (__ret_warn_on != 0) as i64;
if __v2 == 0 {
// All requests fulfilled, time to go idle.
pushgp = false;
}
} else {
if rcu_seq_state(unsafe { (*(*ssp).srcu_sup).srcu_gp_seq }) == 0 {
// Outstanding request and no GP.  Start one.
srcu_gp_start(ssp);
}
}
raw_spin_unlock_irq_rcu_node!(unsafe { (*ssp).srcu_sup });
if pushgp {
queue_delayed_work(unsafe { rcu_gp_wq }, unsafe { core::ptr::addr_of_mut!((*(*ssp).srcu_sup).work) }, delay);
}
}

// This is the work-queue function that handles SRCU grace periods.
#[unsafe(link_section = ".text.slate_distinct.srcutree.srcu_irq_work")]
extern "C" fn srcu_irq_work(work: *mut irq_work) {
let mut sup: *mut srcu_usage = unsafe { core::mem::zeroed() };
let mut delay: u64 = 0;
let mut flags: u64 = 0;
sup = container_of__14!(work, srcu_usage);
let ssp: *mut srcu_struct = unsafe { (*sup).srcu_ssp };
raw_spin_lock_irqsave_rcu_node!(unsafe { (*ssp).srcu_sup }, flags);
delay = srcu_get_delay(ssp);
raw_spin_unlock_irqrestore_rcu_node!(unsafe { (*ssp).srcu_sup }, flags);
queue_delayed_work(unsafe { rcu_gp_wq }, unsafe { core::ptr::addr_of_mut!((*sup).work) }, (delay != 0) as u64);
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.srcutree.srcutorture_get_gp_data")]
extern "C" fn srcutorture_get_gp_data(ssp: *mut srcu_struct, flags: *mut i32, gp_seq: *mut u64) {
unsafe {
*flags = 0;
}
unsafe {
*gp_seq = rcu_seq_current(core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_gp_seq));
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_srcutorture_get_gp_data_790: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut srcu_struct, *mut i32, *mut u64)>, *mut ()>(Some(srcutorture_get_gp_data)) };

static srcu_size_state_name: __SlateAlign16<[*const u8; 10]> = __SlateAlign16([(b"SRCU_SIZE_SMALL\0".as_ptr() as *mut u8) as *const u8, (b"SRCU_SIZE_ALLOC\0".as_ptr() as *mut u8) as *const u8, (b"SRCU_SIZE_WAIT_BARRIER\0".as_ptr() as *mut u8) as *const u8, (b"SRCU_SIZE_WAIT_CALL\0".as_ptr() as *mut u8) as *const u8, (b"SRCU_SIZE_WAIT_CBS1\0".as_ptr() as *mut u8) as *const u8, (b"SRCU_SIZE_WAIT_CBS2\0".as_ptr() as *mut u8) as *const u8, (b"SRCU_SIZE_WAIT_CBS3\0".as_ptr() as *mut u8) as *const u8, (b"SRCU_SIZE_WAIT_CBS4\0".as_ptr() as *mut u8) as *const u8, (b"SRCU_SIZE_BIG\0".as_ptr() as *mut u8) as *const u8, (b"SRCU_SIZE_???\0".as_ptr() as *mut u8) as *const u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_srcu_torture_stats_print_792: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut srcu_struct, *mut u8, *mut u8)>, *mut ()>(Some(srcu_torture_stats_print)) };

#[unsafe(link_section = ".init.text")]
extern "C" fn srcu_bootup_announce() -> i32 {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x016rcu: Hierarchical SRCU implementation.\n".as_ptr() as *const u8) };
if (unsafe { exp_holdoff }) != 25000u64 {
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x016rcu: \tNon-default auto-expedite holdoff of %lu ns.\n".as_ptr() as *const u8, exp_holdoff) };
}
if (unsafe { srcu_retry_check_delay }) != 5 {
let mut __v4: i32;
while false {
}
__v4 = unsafe { _printk(c"\x016rcu: \tNon-default retry check delay of %lu us.\n".as_ptr() as *const u8, srcu_retry_check_delay) };
}
if (unsafe { srcu_max_nodelay }) != if SRCU_UL_CLAMP!(400u64, 3u64, 1000u64) > 100 { SRCU_UL_CLAMP!(400u64, 3u64, 1000u64) } else { 100 } {
let mut __v6: i32;
while false {
}
__v6 = unsafe { _printk(c"\x016rcu: \tNon-default max no-delay of %lu.\n".as_ptr() as *const u8, srcu_max_nodelay) };
}
let mut __v8: i32;
while false {
}
__v8 = unsafe { _printk(c"\x016rcu: \tMax phase no-delay instances is %lu.\n".as_ptr() as *const u8, srcu_max_nodelay_phase) };
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_srcu_bootup_announce_794: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i32>, *mut ()>(Some(srcu_bootup_announce)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn srcu_init() {
let mut sup: *mut srcu_usage = unsafe { core::mem::zeroed() };
// Decide on srcu_struct-size strategy.
if SRCU_SIZING_IS!(3i32) {
if (unsafe { nr_cpu_ids }) >= ((unsafe { big_cpu_lim }) as u32) {
unsafe {
convert_to_big = 1;
}
// Don't bother waiting for contention.
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x016rcu: %s: Setting srcu_struct sizes to big.\n".as_ptr() as *const u8, c"srcu_init".as_ptr() as *mut u8) };
} else {
unsafe {
convert_to_big = 0 | 16;
}
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x016rcu: %s: Setting srcu_struct sizes based on contention.\n".as_ptr() as *const u8, c"srcu_init".as_ptr() as *mut u8) };
}
}
// Once that is set, call_srcu() can follow the normal path and
// queue delayed work. This must follow RCU workqueues creation
// and timers initialization.
unsafe {
srcu_init_done = true;
}
while list_empty((unsafe { core::ptr::addr_of_mut!(srcu_boot_list) }) as *const list_head) == 0 {
let mut __v4: *mut srcu_usage;
__v4 = ((unsafe { (((*core::ptr::addr_of_mut!(srcu_boot_list)).next as *mut ()) as *const u8).wrapping_offset(-(offsetof__106!(srcu_usage) as isize)) }) as *mut ()) as *mut srcu_usage;
sup = __v4;
list_del_init(unsafe { core::ptr::addr_of_mut!((*sup).work.work.entry) });
if SRCU_SIZING_IS!(1i32) && (unsafe { (*sup).srcu_size_state }) == 0 {
unsafe {
(*sup).srcu_size_state = 1;
}
}
queue_work(unsafe { rcu_gp_wq }, unsafe { core::ptr::addr_of_mut!((*sup).work.work) });
}
}

/// Initialize any global-scope srcu_struct structures used by this module.
fn srcu_module_coming(r#mod: *mut module) -> i32 {
let mut i: i32 = 0;
let mut sspp: *mut *mut srcu_struct = unsafe { (*r#mod).srcu_struct_ptrs };
while (i as u32) < unsafe { (*r#mod).num_srcu_structs } {
let __v0: *mut *mut srcu_struct = sspp;
sspp = __v0.wrapping_offset(1 as isize);
let ssp: *mut srcu_struct = unsafe { *__v0 };
unsafe {
(*ssp).sda = alloc_percpu!(srcu_data);
}
let mut __v7: i64;
let __ret_warn_on: i32 = unsafe { (*ssp).sda }.is_null() as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("795: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 795b - ., 3; .popsection\n# {0}", const 795 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_srcu_module_coming_8, sym __slate_asm_string_srcu_module_coming_9, const 2124 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("796: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 796b - ., 4; .popsection\n# {0}", const 796 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v7 = (__ret_warn_on != 0) as i64;
if __v7 != 0 {
return 12i32.wrapping_neg();
}
unsafe {
(*ssp).srcu_ctrp = ((*(*ssp).sda).srcu_ctrs.as_mut_ptr() as *mut srcu_ctr).wrapping_offset(0 as isize);
}
i = i.wrapping_add(1);
}
0
}

// Clean up any global-scope srcu_struct structures used by this module.
fn srcu_module_going(r#mod: *mut module) {
let mut i: i32 = 0;
let mut ssp: *mut srcu_struct = unsafe { core::mem::zeroed() };
let mut sspp: *mut *mut srcu_struct = unsafe { (*r#mod).srcu_struct_ptrs };
while (i as u32) < unsafe { (*r#mod).num_srcu_structs } {
let __v0: *mut *mut srcu_struct = sspp;
sspp = __v0.wrapping_offset(1 as isize);
ssp = unsafe { *__v0 };
let mut __v2: u64;
let mut ___p1: u64 = 0;
let mut __v3: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_797() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*(*ssp).srcu_sup).srcu_gp_seq_needed) as *const u64) };
___p1 = __v3;
loop {
if !__native_word!() {
unsafe { __compiletime_assert_798() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
__v2 = ___p1;
let __v6: bool = if rcu_seq_state(__v2) == 0 {
let mut __v7: i64;
let __ret_warn_on: i32 = !(!(!unsafe { (*(*ssp).srcu_sup).sda_is_static })) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("799: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 799b - ., 3; .popsection\n# {0}", const 799 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_srcu_module_going_8, sym __slate_asm_string_srcu_module_going_9, const 2141 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("800: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 800b - ., 4; .popsection\n# {0}", const 800 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v7 = (__ret_warn_on != 0) as i64;
__v7 == 0
} else {
false
};
if __v6 {
cleanup_srcu_struct(ssp);
}
let mut __v12: i64;
let __ret_warn_on: i32 = !(!srcu_readers_active(ssp)) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("801: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 801b - ., 3; .popsection\n# {0}", const 801 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_srcu_module_going_13, sym __slate_asm_string_srcu_module_going_14, const 2143 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("802: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 802b - ., 4; .popsection\n# {0}", const 802 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v12 = (__ret_warn_on != 0) as i64;
if __v12 == 0 {
unsafe { free_percpu((*ssp).sda as *mut ()) };
}
i = i.wrapping_add(1);
}
}

/// Handle one module, either coming or going.
#[unsafe(link_section = ".text.slate_distinct.srcutree.srcu_module_notify")]
extern "C" fn srcu_module_notify(self_: *mut notifier_block, val: u64, data: *mut ()) -> i32 {
let r#mod: *mut module = data as *mut module;
let mut ret: i32 = 0;
match val {
1 => {
ret = srcu_module_coming(r#mod);
}
2 => {
srcu_module_going(r#mod);
}
_ => {
}
}
ret
}

static mut srcu_module_nb: notifier_block = {
let mut __t0: notifier_block = unsafe { core::mem::zeroed() };
__t0.notifier_call = Some(srcu_module_notify);
__t0.priority = 0i32;
__t0
};

#[unsafe(link_section = ".init.text")]
extern "C" fn init_srcu_module_notifier() -> i32 {
let ret: i32 = unsafe { register_module_notifier(core::ptr::addr_of_mut!(srcu_module_nb)) };
if ret != 0 {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x014rcu: Failed to register srcu module notifier\n".as_ptr() as *const u8) };
}
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_init_srcu_module_notifier_804: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i32>, *mut ()>(Some(init_srcu_module_notifier)) };

static __slate_asm_string___call_srcu_6: [u8; 40] = *b"call_srcu(): Leaked duplicate callback\n\0";

static __slate_asm_string___call_srcu_7: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/rcu/srcutree.c\0";

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string___srcu_check_read_flavor_2: [u8; 1] = *b"\0";

static __slate_asm_string___srcu_check_read_flavor_20: [u8; 1] = *b"\0";

static __slate_asm_string___srcu_check_read_flavor_21: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/rcu/srcutree.c\0";

static __slate_asm_string___srcu_check_read_flavor_25: [u8; 1] = *b"\0";

static __slate_asm_string___srcu_check_read_flavor_26: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/rcu/srcutree.c\0";

static __slate_asm_string___srcu_check_read_flavor_3: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/rcu/srcutree.c\0";

static __slate_asm_string___srcu_check_read_flavor_30: [u8; 1] = *b"\0";

static __slate_asm_string___srcu_check_read_flavor_31: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/rcu/srcutree.c\0";

static __slate_asm_string___srcu_check_read_flavor_41: [u8; 34] = *b"CPU %d old state %d new state %d\n\0";

static __slate_asm_string___srcu_check_read_flavor_42: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/rcu/srcutree.c\0";

static __slate_asm_string___srcu_check_read_flavor_7: [u8; 1] = *b"\0";

static __slate_asm_string___srcu_check_read_flavor_8: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/rcu/srcutree.c\0";

static __slate_asm_string_cleanup_srcu_struct_10: [u8; 1] = *b"\0";

static __slate_asm_string_cleanup_srcu_struct_11: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/rcu/srcutree.c\0";

static __slate_asm_string_cleanup_srcu_struct_19: [u8; 1] = *b"\0";

static __slate_asm_string_cleanup_srcu_struct_20: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/rcu/srcutree.c\0";

static __slate_asm_string_cleanup_srcu_struct_25: [u8; 1] = *b"\0";

static __slate_asm_string_cleanup_srcu_struct_26: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/rcu/srcutree.c\0";

static __slate_asm_string_cleanup_srcu_struct_35: [u8; 1] = *b"\0";

static __slate_asm_string_cleanup_srcu_struct_36: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/rcu/srcutree.c\0";

static __slate_asm_string_cleanup_srcu_struct_41: [u8; 1] = *b"\0";

static __slate_asm_string_cleanup_srcu_struct_42: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/rcu/srcutree.c\0";

static __slate_asm_string_cleanup_srcu_struct_47: [u8; 1] = *b"\0";

static __slate_asm_string_cleanup_srcu_struct_48: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/rcu/srcutree.c\0";

static __slate_asm_string_cleanup_srcu_struct_5: [u8; 1] = *b"\0";

static __slate_asm_string_cleanup_srcu_struct_6: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/rcu/srcutree.c\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

static __slate_asm_string_rcu_seq_end_3: [u8; 1] = *b"\0";

static __slate_asm_string_rcu_seq_end_4: [u8; 46] = *b"/home/takashi/c-corpus/linux/kernel/rcu/rcu.h\0";

static __slate_asm_string_rcu_seq_set_state_1: [u8; 1] = *b"\0";

static __slate_asm_string_rcu_seq_set_state_2: [u8; 46] = *b"/home/takashi/c-corpus/linux/kernel/rcu/rcu.h\0";

static __slate_asm_string_rcu_seq_start_6: [u8; 1] = *b"\0";

static __slate_asm_string_rcu_seq_start_7: [u8; 46] = *b"/home/takashi/c-corpus/linux/kernel/rcu/rcu.h\0";

static __slate_asm_string_srcu_advance_state_7: [u8; 1] = *b"\0";

static __slate_asm_string_srcu_advance_state_8: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/rcu/srcutree.c\0";

static __slate_asm_string_srcu_expedite_current_7: [u8; 1] = *b"\0";

static __slate_asm_string_srcu_expedite_current_8: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/rcu/srcutree.c\0";

static __slate_asm_string_srcu_expedite_current_cb_2: [u8; 1] = *b"\0";

static __slate_asm_string_srcu_expedite_current_cb_3: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/rcu/srcutree.c\0";

static __slate_asm_string_srcu_expedite_current_cb_7: [u8; 1] = *b"\0";

static __slate_asm_string_srcu_expedite_current_cb_8: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/rcu/srcutree.c\0";

static __slate_asm_string_srcu_funnel_exp_start_3: [u8; 1] = *b"\0";

static __slate_asm_string_srcu_funnel_exp_start_4: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/rcu/srcutree.c\0";

static __slate_asm_string_srcu_funnel_gp_start_43: [u8; 1] = *b"\0";

static __slate_asm_string_srcu_funnel_gp_start_44: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/rcu/srcutree.c\0";

static __slate_asm_string_srcu_funnel_gp_start_5: [u8; 1] = *b"\0";

static __slate_asm_string_srcu_funnel_gp_start_6: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/rcu/srcutree.c\0";

static __slate_asm_string_srcu_gp_end_3: [u8; 1] = *b"\0";

static __slate_asm_string_srcu_gp_end_4: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/rcu/srcutree.c\0";

static __slate_asm_string_srcu_gp_start_15: [u8; 1] = *b"\0";

static __slate_asm_string_srcu_gp_start_16: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/rcu/srcutree.c\0";

static __slate_asm_string_srcu_gp_start_2: [u8; 1] = *b"\0";

static __slate_asm_string_srcu_gp_start_3: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/rcu/srcutree.c\0";

static __slate_asm_string_srcu_gp_start_if_needed_22: [u8; 1] = *b"\0";

static __slate_asm_string_srcu_gp_start_if_needed_23: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/rcu/srcutree.c\0";

static __slate_asm_string_srcu_invoke_callbacks_15: [u8; 1] = *b"\0";

static __slate_asm_string_srcu_invoke_callbacks_16: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/rcu/srcutree.c\0";

static __slate_asm_string_srcu_invoke_callbacks_4: [u8; 1] = *b"\0";

static __slate_asm_string_srcu_invoke_callbacks_5: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/rcu/srcutree.c\0";

static __slate_asm_string_srcu_module_coming_8: [u8; 1] = *b"\0";

static __slate_asm_string_srcu_module_coming_9: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/rcu/srcutree.c\0";

static __slate_asm_string_srcu_module_going_13: [u8; 1] = *b"\0";

static __slate_asm_string_srcu_module_going_14: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/rcu/srcutree.c\0";

static __slate_asm_string_srcu_module_going_8: [u8; 1] = *b"\0";

static __slate_asm_string_srcu_module_going_9: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/rcu/srcutree.c\0";

static __slate_asm_string_srcu_readers_lock_idx_14: [u8; 46] = *b"Mixed reader flavors for srcu_struct at %ps.\n\0";

static __slate_asm_string_srcu_readers_lock_idx_15: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/rcu/srcutree.c\0";

static __slate_asm_string_srcu_readers_unlock_idx_14: [u8; 46] = *b"Mixed reader flavors for srcu_struct at %ps.\n\0";

static __slate_asm_string_srcu_readers_unlock_idx_15: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/rcu/srcutree.c\0";

static __slate_asm_string_srcu_reschedule_3: [u8; 1] = *b"\0";

static __slate_asm_string_srcu_reschedule_4: [u8; 51] = *b"/home/takashi/c-corpus/linux/kernel/rcu/srcutree.c\0";

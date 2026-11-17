
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
//! SPDX-License-Identifier: GPL-2.0-or-later
//!
//! ALSA sequencer Client Manager
//! Copyright (c) 1998-2001 by Frank van de Pol <fvdpol@coil.demon.nl>
//!                            Jaroslav Kysela <perex@perex.cz>
//!                            Takashi Iwai <tiwai@suse.de>
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
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
//! See linux/instrumentation.h
//! Begin/end of an instrumentation safe region
//! 64 bit
//! unused, was: #define EX_TYPE_COPY	 4
//! inhibit tail-call optimization
//! Call a function once. Similar to DO_ONCE(), but does not use jump label
//! patching via static keys.
//! warning conditions
//! Internal kernel timer frequency
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
//! Prevent speculative execution past this barrier.
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
//! No ioctl command
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
//! These are non-NULL pointers that will result in page faults
//! under normal circumstances, used to verify that nobody uses
//! non-initialized list entries.
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
//! Keep unconverted code working
//! The IDR API does not expose the tagging functionality of the radix tree
//! to users.  Use tag 0 to track whether a node has free space below it.
//! number of env pointers
//! Generate SD flag indexes
//! Resource type
//! Desired design of maximum size and alignment (see RFC2553)
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
//! return the first ptr or next element in an RCU protected list
//! used to be O_SYNC, see below
//! Valid hint values for F_{GET,SET}_RW_HINT. 0 is "not set", or can be
//! used to clear any hints previously set.
//! Gives us 8 prio classes with 13-bits of data for each class
//! per-IO O_SYNC
//! file is open for reading
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
//! don't change - minor numbers
//! UMP groups and blocks
//! version of the sequencer
//! maximum number of queues
//! These are specified by iBCS2
//! Epoll event masks
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
// __swahw32 - return a word-swapped 32-bit value
// @x: value to wordswap
//
// __swahw32(0x12340000) is 0x00001234
// __swahb32 - return a high and low byte-swapped 32-bit value
// @x: value to byteswap
//
// __swahb32(0x12345678) is 0x34127856
// Dummy printk for disabled debugging statements to use whilst maintaining
// gcc's format checking.
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
// pr_err - Print an error-level message
// @fmt: format string
// @...: arguments for the format string
//
// This macro expands to a printk with KERN_ERR loglevel. It uses pr_fmt() to
// generate the format string.
// pr_warn - Print a warning-level message
// @fmt: format string
// @...: arguments for the format string
//
// This macro expands to a printk with KERN_WARNING loglevel. It uses pr_fmt()
// to generate the format string.
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
// Not a typewriter
// Text file busy
// No space left on device
// Too many links
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
// entries per page directory level
// We borrow bit 3 to remember PG_anon_exclusive.
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
// Atomic operations are already serializing on x86
// The following macros are non-atomic versions of their non-underscored
// counterparts.
// Check that the bitops prototypes are sane
// assign_bit - Assign value to a bit in memory
// @nr: the bit to set
// @addr: the address to start counting from
// @value: the value to assign
// restart by calling sys_restart_syscall
// Operation is not supported
// strscpy - Copy a C-string into a sized buffer
// @dst: Where to copy the string to
// @src: Where to copy the string from
// @...: Size of destination buffer (optional)
//
// Copy the source string @src, or as much of it as fits, into the
// destination @dst buffer. The behavior is undefined if the string
// buffers overlap. The destination @dst buffer is always NUL terminated,
// unless it's zero-sized.
//
// The size argument @... is only required when @dst is not an array, or
// when the copy needs to be smaller than sizeof(@dst).
//
// Preferred to strncpy() since it always returns a valid string, and
// doesn't unnecessarily force the tail of the destination buffer to be
// zero padded. If padding is desired please use strscpy_pad().
//
// Returns the number of characters copied in @dst (not including the
// trailing %NUL) or -E2BIG if @size is 0 or the copy from @src was
// truncated.
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
// bits_per - calculate the number of bits required for the argument
// @n: parameter
//
// This is constant-capable and can be used for compile time
// initializations, e.g bitfields.
//
// The first few values calculated by this routine:
// bf(0) = 1
// bf(1) = 1
// bf(2) = 2
// bf(3) = 2
// bf(4) = 3
// ... and so on.
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
// idle is polling for TIF_NEED_RESCHED
// re-arm the timer
// If !LOCKDEP, the value is meaningless
// For trivial one-depth nesting of a lock-class, the following
// global define can be used. (Subsystems with multiple levels
// of nesting should define their own lock-nesting subclasses.)
// Writer mask
// Remapping rwlock architecture specific functions to the corresponding
// queued rwlock functions.
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
// Same as rb_first(), but O(1)
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
// GFP_ZONE_BAD is a bitmap for all combinations of __GFP_DMA, __GFP_DMA32
// __GFP_HIGHMEM and __GFP_MOVABLE that are not permitted. One flag per
// entry starting with bit 0. Bit is set if the combination is not
// allowed.
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
// Set the IDR flag and the IDR_FREE tag
// Generate SD flag bits
// Resource extended types
// No address assigned yet
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
// hlist_for_each_entry_rcu - iterate over rcu list of given type
// @pos:	the type * to use as a loop cursor.
// @head:	the head for your list.
// @member:	the name of the hlist_node within the struct.
// @cond:	optional lockdep expression if called from non-RCU protection.
//
// This list-traversal primitive may safely run concurrently with
// the _rcu list-mutation primitives such as hlist_add_head_rcu()
// as long as the traversal is guarded by rcu_read_lock().
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
// ptr must be evaluated and assigned to the temporary __ptr_pu before
// the assignment of x to __val_pu, to avoid any function calls
// involved in the ptr expression (possibly implicitly generated due
// to KASAN) from clobbering %ax.
// eval x once
// eval ptr once
// put_user - Write a simple value into user space.
// @x:   Value to copy to user space.
// @ptr: Destination address, in user space.
//
// Context: User context only. This function may sleep if pagefaults are
//          enabled.
//
// This macro copies a single simple value from kernel space to user
// space.  It supports simple types like char and int, but not larger
// data types like structures or arrays.
//
// @ptr must have pointer-to-simple-variable type, and @x must be assignable
// to the result of dereferencing @ptr.
//
// Return: zero on success, or -EFAULT on error.
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

macro_rules! BIT_MASK {
($nr:expr) => {
UL!() << ($nr) % 64
};
}

macro_rules! BIT_WORD {
($nr:expr) => {
($nr) / 64
};
}

macro_rules! BIT_WORD__2 {
($nr:expr) => {
($nr) / 64
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

macro_rules! FLAG_PERM_DUPLEX {
($perm:expr) => {
if ($perm) & 16u32 != 0 { 88i32 } else { 45i32 }
};
}

macro_rules! FLAG_PERM_EX {
($perm:expr) => {
if ($perm) & 128u32 != 0 { 45i32 } else { 101i32 }
};
}

macro_rules! FLAG_PERM_RD {
($perm:expr) => {
if ($perm) & 1u32 != 0 { if ($perm) & 32u32 != 0 { 82i32 } else { 114i32 } } else { 45i32 }
};
}

macro_rules! FLAG_PERM_WR {
($perm:expr) => {
if ($perm) & 2u32 != 0 { if ($perm) & 64u32 != 0 { 87i32 } else { 119i32 } } else { 45i32 }
};
}

macro_rules! GENMASK_INPUT_CHECK__9 {
() => {
core::mem::size_of::<__SlateRecord_seq_clientmgr_c_1300>() as i32
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
macro_rules! GENMASK_TYPE__4 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__9!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub(($h) as $t))
};
}

macro_rules! GENMASK__3 {
($h:expr, $l:expr) => {
GENMASK_TYPE__4!(u64, $h, $l)
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

macro_rules! SNDRV_PROTOCOL_VERSION {
($major:expr, $minor:expr, $subminor:expr) => {
($major) << 16i32 | ($minor) << 8i32 | $subminor
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

macro_rules! _IOC_NR {
($nr:expr) => {
($nr) >> 0i32 & 255u32
};
}

macro_rules! _IOC_SIZE {
($nr:expr) => {
($nr) >> 16i32 & 16383u32
};
}

macro_rules! _IOC_TYPECHECK {
($t:ty) => {
if (core::mem::size_of::<$t>() as u64) == (core::mem::size_of::<[$t; 1]>() as u64) && (core::mem::size_of::<$t>() as u64) < 16384u64 { core::mem::size_of::<$t>() as u64 } else { (unsafe { __invalid_size_argument_for_IOC }) as u64 }
};
}

macro_rules! _IOC_TYPE {
($nr:expr) => {
($nr) >> 8i32 & 255u32
};
}

macro_rules! _IOC {
($dir:expr, $type:expr, $nr:expr, $size:expr) => {
((($dir) << 30i32 | ((($type) << 8i32) as u32) | ((($nr) << 0i32) as u32)) as u64) | ($size) << 16i32
};
}

macro_rules! _IOWR {
($type:expr, $nr:expr, $argtype:ty) => {
_IOC!(3u32, $type, $nr, _IOC_TYPECHECK!($argtype))
};
}

macro_rules! _IOW {
($type:expr, $nr:expr, $argtype:ty) => {
_IOC!(1u32, $type, $nr, _IOC_TYPECHECK!($argtype))
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

macro_rules! __READ_ONCE {
($x:expr) => {
unsafe { core::ptr::read_volatile(core::ptr::addr_of!($x) as *const i32) }
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

macro_rules! __WRITE_ONCE__5 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut *mut snd_seq_client, $val) };
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

macro_rules! __chk_user_ptr {
() => {
};
}

macro_rules! __get_and_null__2 {
($p:expr) => {
{
let mut __v: *mut snd_seq_client;
let __v2: *mut *mut snd_seq_client = core::ptr::addr_of_mut!($p);
let __v3: *mut snd_seq_client = unsafe { *__v2 };
unsafe {
*__v2 = core::ptr::null_mut::<snd_seq_client>();
}
__v = __v3;
__v
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

macro_rules! __must_be_array__19 {
() => {
core::mem::size_of::<__SlateRecord_seq_clientmgr_c_1230>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__6 {
() => {
core::mem::size_of::<__SlateRecord_seq_clientmgr_c_1612>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__19 {
() => {
core::mem::size_of::<__SlateRecord_seq_clientmgr_c_1230_3>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__37 {
() => {
core::mem::size_of::<__SlateRecord_seq_clientmgr_c_1230_2>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__48 {
() => {
core::mem::size_of::<__SlateRecord_seq_clientmgr_c_1293>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__56 {
() => {
core::mem::size_of::<__SlateRecord_seq_clientmgr_c_1642>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__68 {
() => {
core::mem::size_of::<__SlateRecord_seq_clientmgr_c_1584>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__75 {
() => {
core::mem::size_of::<__SlateRecord_seq_clientmgr_c_1584_2>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__83 {
() => {
core::mem::size_of::<__SlateRecord_seq_clientmgr_c_1612_2>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__89 {
() => {
core::mem::size_of::<__SlateRecord_seq_clientmgr_c_1642_2>() as i32
};
}

/// Require C Strings (i.e. NUL-terminated) lack the "nonstring" attribute.
macro_rules! __must_be_cstr__102 {
() => {
core::mem::size_of::<__SlateRecord_seq_clientmgr_c_1293_2>() as i32
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word {
() => {
((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__3 {
() => {
((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__11 {
() => {
((const { core::mem::size_of::<*mut snd_seq_client>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut snd_seq_client>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut snd_seq_client>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut snd_seq_client>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__6 {
() => {
((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__2 {
() => {
((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

macro_rules! __or {
() => {
____or!()
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
1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((is_signed_type__2!() as i32) as u64)
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

macro_rules! absolute_pointer {
($val:expr) => {
RELOC_HIDE!(($val) as *mut (), 0i32)
};
}

macro_rules! access_ok {
($addr:expr, $size:expr) => {
likely!(__access_ok(($addr) as *const (), $size))
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
macro_rules! container_of__6 {
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
macro_rules! container_of__7 {
($ptr:expr) => {
{
let mut __v: *mut snd_seq_client_port;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__3!() as isize)) }) as *mut ()) as *mut snd_seq_client_port;
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
macro_rules! container_of__9 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__18!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__12 {
($ptr:expr) => {
{
let mut __v: *mut snd_seq_subscribers;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__10!() as isize)) }) as *mut ()) as *mut snd_seq_subscribers;
__v
}
};
}

macro_rules! down_read_nested {
($sem:expr) => {
unsafe { down_read($sem) };
};
}

macro_rules! hlist_entry {
($ptr:expr, $type:ty) => {
container_of__6!($ptr, $type)
};
}

macro_rules! hlist_entry__2 {
($ptr:expr) => {
container_of__12!($ptr)
};
}

macro_rules! hlist_entry__3 {
($ptr:expr, $type:ty) => {
container_of__9!($ptr, $type)
};
}

macro_rules! in_interrupt {
() => {
irq_count!()
};
}

/// instrument_get_user() - add instrumentation to get_user()-like macros
/// @to: destination variable, may not be address-taken
///
/// get_user() and friends are fragile, so it may depend on the implementation
/// whether the instrumentation happens before or after the data is copied from
/// the userspace.
macro_rules! instrument_get_user__3 {
($to:expr) => {
let mut __v: u8;
let mut __tmp: u64 = ($to) as u64;
kmsan_unpoison_memory(core::ptr::addr_of_mut!(__tmp) as *const (), core::mem::size_of::<u64>() as u64);
let __v2: u8 = __tmp as u8;
$to = __v2;
__v = __v2;
};
}

/// instrument_get_user() - add instrumentation to get_user()-like macros
/// @to: destination variable, may not be address-taken
///
/// get_user() and friends are fragile, so it may depend on the implementation
/// whether the instrumentation happens before or after the data is copied from
/// the userspace.
macro_rules! instrument_get_user {
($to:expr) => {
let mut __v: u32;
let mut __tmp: u64 = ($to) as u64;
kmsan_unpoison_memory(core::ptr::addr_of_mut!(__tmp) as *const (), core::mem::size_of::<u64>() as u64);
let __v2: u32 = __tmp as u32;
$to = __v2;
__v = __v2;
};
}

/// instrument_put_user() - add instrumentation to put_user()-like macros
/// @from: source address
/// @ptr: userspace pointer to copy to
/// @size: number of bytes to copy
///
/// put_user() and friends are fragile, so it may depend on the implementation
/// whether the instrumentation happens before or after the data is copied from
/// the userspace.
macro_rules! instrument_put_user__2 {
($from:expr, $ptr:expr) => {
kmsan_copy_to_user(($ptr) as *mut (), core::ptr::addr_of_mut!($from) as *const (), core::mem::size_of::<u32>() as u64, 0u64);
};
}

/// instrument_put_user() - add instrumentation to put_user()-like macros
/// @from: source address
/// @ptr: userspace pointer to copy to
/// @size: number of bytes to copy
///
/// put_user() and friends are fragile, so it may depend on the implementation
/// whether the instrumentation happens before or after the data is copied from
/// the userspace.
macro_rules! instrument_put_user__5 {
($from:expr, $ptr:expr) => {
kmsan_copy_to_user(($ptr) as *mut (), core::ptr::addr_of_mut!($from) as *const (), core::mem::size_of::<u8>() as u64, 0u64);
};
}

macro_rules! irq_count {
() => {
(preempt_count() as u64) & (__IRQ_MASK!(1i32.wrapping_add(3i32.wrapping_mul(IS_ENABLED__2!()))) << 28i32 | __IRQ_MASK!(4i32) << 24i32 | __IRQ_MASK!(8i32) << 8i32)
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type__2 {
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

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__4 {
($ptr:expr) => {
container_of__7!($ptr)
};
}

/// list_entry_is_head - test if the entry points to the head of the list
/// @pos:	the type * to cursor
/// @head:	the head for your list.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry_is_head {
($pos:expr, $head:expr, $member:ident) => {
list_is_head((unsafe { core::ptr::addr_of_mut!((*$pos).$member) }) as *const list_head, ($head) as *const list_head)
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__9 {
($ptr:expr) => {
list_entry__4!(unsafe { (*$ptr).next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__6 {
($pos:expr, $member:ident) => {
list_entry__4!(unsafe { (*$pos).$member.next })
};
}

macro_rules! no_free_ptr {
($p:expr) => {
__must_check_fn(__get_and_null__2!($p) as *const ()) as *mut snd_seq_client
};
}

/// normal system notification event broadcast
macro_rules! notify_event {
($client:expr, $port:expr, $type:expr) => {
unsafe { snd_seq_system_broadcast($client, $port, $type, false) };
};
}

/// normal system notification event broadcast
macro_rules! notify_event__2 {
($client:expr, $port:expr, $type:expr) => {
unsafe { snd_seq_system_broadcast(($client) as i32, ($port) as i32, $type, false) };
};
}

/// normal system notification event broadcast
macro_rules! notify_event__3 {
($client:expr, $port:expr, $type:expr) => {
unsafe { snd_seq_system_broadcast($client, ($port) as i32, $type, false) };
};
}

macro_rules! offsetof__10 {
() => {
core::mem::offset_of!(snd_seq_subscribers, src_list) as u64
};
}

macro_rules! offsetof__15 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, src_list) as u64
};
}

macro_rules! offsetof__3 {
() => {
core::mem::offset_of!(snd_seq_client_port, list) as u64
};
}

macro_rules! offsetof__18 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, dest_list) as u64
};
}

/// unlock
macro_rules! queuefree {
($q:expr) => {
snd_use_lock_free!(unsafe { core::ptr::addr_of_mut!((*$q).use_lock) });
};
}

macro_rules! raw_spin_lock_irq {
($lock:expr) => {
unsafe { _raw_spin_lock_irq($lock) };
};
}

macro_rules! raw_spin_unlock_irq {
($lock:expr) => {
unsafe { _raw_spin_unlock_irq($lock) };
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

macro_rules! snd_BUG_ON__4 {
($condition:expr) => {
{
let mut __v: i64;
let __v2: i32 = (($condition) != 0) as i32;
__v = (__v2 != 0) as i64;
__v
}
};
}

macro_rules! snd_BUG_ON {
($condition:expr) => {
{
let mut __v: i64;
let __v2: i32 = !(!$condition) as i32;
__v = (__v2 != 0) as i64;
__v
}
};
}

/// direct dispatched events
macro_rules! snd_seq_ev_is_direct {
($ev:expr) => {
((unsafe { (*$ev).queue }) as i32) == 253
};
}

/// reserved for kernel
macro_rules! snd_seq_ev_is_reserved {
($ev:expr) => {
((unsafe { (*$ev).r#type }) as i32) >= 150
};
}

/// check whether the given event is a UMP event
macro_rules! snd_seq_ev_is_ump {
($ev:expr) => {
IS_ENABLED!() != 0 && ((unsafe { (*$ev).flags }) as i32) & 32i32 != 0
};
}

macro_rules! snd_seq_ev_is_variable {
($ev:expr) => {
snd_seq_ev_length_type!($ev) == 4i32
};
}

/// variable length events: 130-139
macro_rules! snd_seq_ev_is_variable_type {
($ev:expr) => {
((unsafe { (*$ev).r#type }) as i32) >= 130 && ((unsafe { (*$ev).r#type }) as i32) < 140
};
}

macro_rules! snd_seq_ev_is_varusr {
($ev:expr) => {
snd_seq_ev_length_type!($ev) == 8i32
};
}

/// event length type
macro_rules! snd_seq_ev_length_type {
($ev:expr) => {
((unsafe { (*$ev).flags }) as i32) & 12i32
};
}

/// lock fifo from release
macro_rules! snd_seq_fifo_lock {
($fifo:expr) => {
snd_use_lock_use!(unsafe { core::ptr::addr_of_mut!((*$fifo).use_lock) });
};
}

macro_rules! snd_seq_fifo_unlock {
($fifo:expr) => {
snd_use_lock_free!(unsafe { core::ptr::addr_of_mut!((*$fifo).use_lock) });
};
}

/// unlock the port
macro_rules! snd_seq_port_unlock {
($port:expr) => {
snd_use_lock_free!(unsafe { core::ptr::addr_of_mut!((*$port).use_lock) });
};
}

macro_rules! snd_seq_system_client_ev_client_change {
($client:expr) => {
notify_event!($client, 0i32, 62i32);
};
}

macro_rules! snd_seq_system_client_ev_client_exit {
($client:expr) => {
notify_event!($client, 0i32, 61i32);
};
}

macro_rules! snd_seq_system_client_ev_client_start {
($client:expr) => {
notify_event!($client, 0i32, 60i32);
};
}

macro_rules! snd_seq_system_client_ev_port_change {
($client:expr, $port:expr) => {
notify_event__2!($client, $port, 65i32);
};
}

macro_rules! snd_seq_system_client_ev_port_exit {
($client:expr, $port:expr) => {
notify_event__3!($client, $port, 64i32);
};
}

macro_rules! snd_seq_system_client_ev_port_start {
($client:expr, $port:expr) => {
notify_event__2!($client, $port, 63i32);
};
}

/// release lock
macro_rules! snd_use_lock_free {
($lockp:expr) => {
atomic_dec($lockp);
};
}

/// initialize lock
macro_rules! snd_use_lock_init {
($lockp:expr) => {
atomic_set($lockp, 0i32);
};
}

macro_rules! snd_use_lock_sync {
($lockp:expr) => {
unsafe { snd_use_lock_sync_helper($lockp, c"/home/takashi/c-corpus/linux/sound/core/seq/seq_clientmgr.c".as_ptr() as *const u8, 272i32) };
};
}

/// increment lock
macro_rules! snd_use_lock_use {
($lockp:expr) => {
atomic_inc($lockp);
};
}

macro_rules! test_bit__3 {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if (unsafe { __slate_intrinsic_70b71f185ebb8c4a($nr) }) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
const_test_bit(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit(($nr) as u64, ($addr) as *const u64) as i32
};
__v
}
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

macro_rules! va_end {
() => {
};
}

unsafe extern "C" {
    static mut kmsg_fops: file_operations;
    static mut __preempt_count: u64;
    static mut __invalid_size_argument_for_IOC: u32;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut cpuinfo_op: seq_operations;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut USER_PTR_MAX: u64;
    static mut snd_ecards_limit: i32;
    static mut seq_client_load: [i32; 15];
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn _printk(arg0: *const u8, ...) -> i32;
    fn __bad_size_call_parameter();
    fn __xadd_wrong_size();
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn sized_strscpy(arg0: *mut u8, arg1: *const u8, arg2: u64) -> i64;
    fn sprintf(arg0: *mut u8, arg1: *const u8, ...) -> i32;
    fn snprintf(arg0: *mut u8, arg1: u64, arg2: *const u8, ...) -> i32;
    fn vsnprintf(arg0: *mut u8, arg1: u64, arg2: *const u8, arg3: core::ffi::VaList<'static>) -> i32;
    fn __compiletime_assert_16();
    fn __compiletime_assert_17();
    fn __compiletime_assert_30();
    fn __compiletime_assert_57();
    fn lockdep_is_held(arg0: *const ()) -> i32;
    fn _raw_spin_lock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irq(arg0: *mut raw_spinlock);
    fn mutex_init_generic(arg0: *mut mutex);
    fn mutex_lock(arg0: *mut mutex);
    fn mutex_unlock(arg0: *mut mutex);
    fn refcount_warn_saturate(arg0: *mut refcount_struct, arg1: u32);
    fn synchronize_rcu();
    fn __rcu_read_lock();
    fn __rcu_read_unlock();
    fn rcu_is_watching() -> bool;
    fn down_read(arg0: *mut rw_semaphore);
    fn up_read(arg0: *mut rw_semaphore);
    fn kfree(arg0: *const ());
    fn __compiletime_assert_208();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __request_module(arg0: bool, arg1: *const u8, ...) -> i32;
    fn __bad_copy_from();
    fn __bad_copy_to();
    fn __copy_overflow(arg0: i32, arg1: u64);
    fn _copy_from_user(arg0: *mut (), arg1: *const (), arg2: u64) -> u64;
    fn _copy_to_user(arg0: *mut (), arg1: *const (), arg2: u64) -> u64;
    fn put_pid(arg0: *mut pid);
    fn pid_vnr(arg0: *mut pid) -> i32;
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_395();
    fn __compiletime_assert_416();
    fn stream_open(arg0: *mut inode, arg1: *mut file) -> i32;
    fn dev_set_name(arg0: *mut device, arg1: *const u8, ...) -> i32;
    fn put_device(arg0: *mut device);
    fn snd_request_card(arg0: i32);
    fn snd_device_alloc(arg0: *mut *mut device, arg1: *mut snd_card) -> i32;
    fn snd_register_device(arg0: i32, arg1: *mut snd_card, arg2: i32, arg3: *const file_operations, arg4: *mut (), arg5: *mut device) -> i32;
    fn snd_unregister_device(arg0: *mut device) -> i32;
    fn snd_seq_expand_var_event(arg0: *const snd_seq_event, arg1: i32, arg2: *mut u8, arg3: i32, arg4: i32) -> i32;
    fn snd_seq_device_load_drivers();
    fn seq_printf(arg0: *mut seq_file, arg1: *const u8, ...);
    fn snd_seq_cell_free(arg0: *mut snd_seq_event_cell);
    fn snd_seq_event_dup(arg0: *mut snd_seq_pool, arg1: *mut snd_seq_event, arg2: *mut *mut snd_seq_event_cell, arg3: i32, arg4: *mut file, arg5: *mut mutex) -> i32;
    fn snd_seq_pool_init(arg0: *mut snd_seq_pool) -> i32;
    fn snd_seq_pool_mark_closing(arg0: *mut snd_seq_pool);
    fn snd_seq_pool_done(arg0: *mut snd_seq_pool) -> i32;
    fn snd_seq_pool_new(arg0: i32) -> *mut snd_seq_pool;
    fn snd_seq_pool_delete(arg0: *mut *mut snd_seq_pool) -> i32;
    fn snd_seq_pool_poll_wait(arg0: *mut snd_seq_pool, arg1: *mut file, arg2: *mut poll_table_struct) -> i32;
    fn snd_seq_info_pool(arg0: *mut snd_info_buffer, arg1: *mut snd_seq_pool, arg2: *mut u8);
    fn snd_use_lock_sync_helper(arg0: *mut __SlateRecord_types_h_190, arg1: *const u8, arg2: i32);
    fn snd_seq_fifo_new(arg0: i32) -> *mut snd_seq_fifo;
    fn snd_seq_fifo_delete(arg0: *mut *mut snd_seq_fifo);
    fn snd_seq_fifo_event_in(arg0: *mut snd_seq_fifo, arg1: *mut snd_seq_event) -> i32;
    fn snd_seq_fifo_cell_out(arg0: *mut snd_seq_fifo, arg1: *mut *mut snd_seq_event_cell, arg2: i32) -> i32;
    fn snd_seq_fifo_cell_putback(arg0: *mut snd_seq_fifo, arg1: *mut snd_seq_event_cell);
    fn snd_seq_fifo_clear(arg0: *mut snd_seq_fifo);
    fn snd_seq_fifo_poll_wait(arg0: *mut snd_seq_fifo, arg1: *mut file, arg2: *mut poll_table_struct) -> i32;
    fn snd_seq_fifo_resize(arg0: *mut snd_seq_fifo, arg1: i32) -> i32;
    fn snd_seq_fifo_unused_cells(arg0: *mut snd_seq_fifo) -> i32;
    fn snd_seq_port_use_ptr(arg0: *mut snd_seq_client, arg1: i32) -> *mut snd_seq_client_port;
    fn snd_seq_port_query_nearest(arg0: *mut snd_seq_client, arg1: *mut snd_seq_port_info) -> *mut snd_seq_client_port;
    fn snd_seq_create_port(arg0: *mut snd_seq_client, arg1: *mut *mut snd_seq_client_port) -> i32;
    fn snd_seq_insert_port(arg0: *mut snd_seq_client, arg1: i32, arg2: *mut snd_seq_client_port) -> i32;
    fn snd_seq_delete_port(arg0: *mut snd_seq_client, arg1: i32) -> i32;
    fn snd_seq_delete_all_ports(arg0: *mut snd_seq_client) -> i32;
    fn snd_seq_set_port_info(arg0: *mut snd_seq_client_port, arg1: *mut snd_seq_port_info) -> i32;
    fn snd_seq_get_port_info(arg0: *mut snd_seq_client_port, arg1: *mut snd_seq_port_info) -> i32;
    fn snd_seq_port_connect(arg0: *mut snd_seq_client, arg1: *mut snd_seq_client, arg2: *mut snd_seq_client_port, arg3: *mut snd_seq_client, arg4: *mut snd_seq_client_port, arg5: *mut snd_seq_port_subscribe) -> i32;
    fn snd_seq_port_disconnect(arg0: *mut snd_seq_client, arg1: *mut snd_seq_client, arg2: *mut snd_seq_client_port, arg3: *mut snd_seq_client, arg4: *mut snd_seq_client_port, arg5: *mut snd_seq_port_subscribe) -> i32;
    fn snd_seq_port_subscribe(arg0: *mut snd_seq_client_port, arg1: *mut snd_seq_port_subscribe) -> i32;
    fn snd_seq_port_get_subscription(arg0: *mut snd_seq_port_subs_info, arg1: *mut snd_seq_addr, arg2: *mut snd_seq_port_subscribe) -> i32;
    fn snd_seq_timer_get_cur_time(arg0: *mut snd_seq_timer, arg1: bool) -> snd_seq_real_time;
    fn snd_seq_timer_get_cur_tick(arg0: *mut snd_seq_timer) -> u32;
    fn snd_seq_queue_get_cur_queues() -> i32;
    fn snd_seq_queue_alloc(arg0: i32, arg1: i32, arg2: u32) -> *mut snd_seq_queue;
    fn snd_seq_queue_delete(arg0: i32, arg1: i32) -> i32;
    fn snd_seq_queue_client_leave(arg0: i32);
    fn snd_seq_enqueue_event(arg0: *mut snd_seq_event_cell, arg1: i32, arg2: i32) -> i32;
    fn snd_seq_queue_remove_cells(arg0: i32, arg1: *mut snd_seq_remove_events);
    fn queueptr(arg0: i32) -> *mut snd_seq_queue;
    fn snd_seq_queue_find_name(arg0: *mut u8) -> *mut snd_seq_queue;
    fn snd_seq_queue_check_access(arg0: i32, arg1: i32) -> i32;
    fn snd_seq_queue_timer_set_tempo(arg0: i32, arg1: i32, arg2: *mut snd_seq_queue_tempo) -> i32;
    fn snd_seq_queue_set_owner(arg0: i32, arg1: i32, arg2: i32) -> i32;
    fn snd_seq_queue_timer_open(arg0: i32) -> i32;
    fn snd_seq_queue_timer_close(arg0: i32) -> i32;
    fn snd_seq_queue_use(arg0: i32, arg1: i32, arg2: i32) -> i32;
    fn snd_seq_queue_is_used(arg0: i32, arg1: i32) -> i32;
    fn snd_seq_system_broadcast(arg0: i32, arg1: i32, arg2: i32, arg3: bool);
    fn snd_seq_system_notify(arg0: i32, arg1: i32, arg2: *mut snd_seq_event, arg3: bool) -> i32;
    fn __compiletime_assert_464();
    fn __compiletime_assert_471();
    fn __compiletime_assert_472();
    fn __compiletime_assert_473();
    fn __compiletime_assert_474();
    fn __compiletime_assert_476();
    fn __compiletime_assert_477();
    fn __compiletime_assert_478();
    fn __compiletime_assert_479();
    fn __compiletime_assert_482();
    fn __compiletime_assert_483();
    fn __compiletime_assert_484();
    fn __compiletime_assert_493();
    fn __compiletime_assert_495();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
    #[link_name = "llvm.objectsize.i64.p0"]
    fn __slate_intrinsic_2913c0303de876b3(arg0: *const (), arg1: bool, arg2: bool, arg3: bool) -> u64;
    #[link_name = "llvm.is.constant.i8"]
    fn __slate_intrinsic_70b71f185ebb8c4a(arg0: u8) -> bool;
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
#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_site {
    addr: i32,
    key: i32,
}


// __ADDRESSABLE() is used to ensure the key symbol doesn't get stripped from
// the symbol table so that objtool can reference it when it generates the
// .static_call_sites section.
#[inline(always)]
fn __must_check_overflow(overflow: bool) -> bool {
unlikely!(overflow) != 0
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
struct lockdep_map {
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


#[inline(always)]
fn PTR_ERR(ptr: *const ()) -> i64 {
ptr as i64
}

#[inline(always)]
fn IS_ERR(ptr: *const ()) -> bool {
IS_ERR_VALUE!(ptr as u64) != 0
}

#[inline(always)]
fn IS_ERR_OR_NULL(ptr: *const ()) -> bool {
unlikely!(ptr.is_null()) != 0 || IS_ERR_VALUE!(ptr as u64) != 0
}

#[inline(always)]
fn __must_check_fn(val: *const ()) -> *const () {
val
}

// Helper macro for scoped_guard().
//
// Note that the "!__is_cond_ptr(_name)" part of the condition ensures that
// compiler would be sure that for the unconditional locks the body of the
// loop (caller-provided code glued to the else clause) could not be skipped.
// It is needed because the other part - "__guard_ptr(_name)(&scope)" - is too
// hard to deduce (even if could be proven true for unconditional locks).
// To support Context Analysis, we need to allow the compiler to see the
// acquisition and release of the context lock. However, the "cleanup" helpers
// wrap the lock in a struct passed through separate helper functions, which
// hides the lock alias from the compiler (no inter-procedural analysis).
//
// To make it work, we introduce an explicit alias to the context lock instance
// that is "cleaned" up with a separate cleanup helper. This helper is a dummy
// function that does nothing at runtime, but has the "_unlock" attribute to
// tell the compiler what happens at the end of the scope.
//
// To generalize the pattern, the WITH_LOCK_GUARD_1_ATTRS() macro should be used
// to redefine the constructor, which then also creates the alias variable with
// the right "cleanup" attribute, *after* DECLARE_LOCK_GUARD_1_ATTRS() has been
// used.
//
// Example usage:
//
//   DECLARE_LOCK_GUARD_1_ATTRS(mutex, __acquires(_T), __releases(*(struct mutex **)_T))
//   #define class_mutex_constructor(_T) WITH_LOCK_GUARD_1_ATTRS(mutex, _T)
//
// Note: To support the for-loop based scoped helpers, the auxiliary variable
// must be a pointer to the "class" type because it is defined in the same
// statement as the guard variable. However, we initialize it with the lock
// pointer (despite the type mismatch, the compiler's alias analysis still works
// as expected). The "_unlock" attribute receives a pointer to the auxiliary
// variable (a double pointer to the class type), and must be cast and
// dereferenced appropriately.
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
struct jump_entry {
    code: i32,
    target: i32,
    key: i64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct orc_entry {
    sp_offset: i16,
    bp_offset: i16,
    __slate_bits_0: __slate_bits::__SlateBits91U0,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits92U0,
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
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
}


#[inline(always)]
fn arch_atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
// Note for KASAN: we deliberately don't use READ_ONCE_NOCHECK() here,
// it's non-inlined function that increases binary size and stack usage.
__READ_ONCE!((*v).counter)
}

#[inline(always)]
fn arch_atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
__WRITE_ONCE__2!((*v).counter, i);
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

#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
}


#[inline(always)]
fn arch_atomic_fetch_add(mut __ret: i32, mut v: *mut __SlateRecord_types_h_190) -> i32 {
let mut __v0: i32;
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
__v0
}

#[inline(always)]
fn raw_atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
arch_atomic_read(v)
}

#[inline(always)]
fn raw_atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
arch_atomic_set(v, i);
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


#[inline(always)]
fn raw_atomic_fetch_add_relaxed(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
arch_atomic_fetch_add(i, v)
}

#[inline(always)]
fn raw_atomic_inc(v: *mut __SlateRecord_types_h_190) {
arch_atomic_inc(v);
}

#[inline(always)]
fn raw_atomic_dec(v: *mut __SlateRecord_types_h_190) {
arch_atomic_dec(v);
}

fn kmsan_unpoison_memory(address: *const (), size: u64) {
}

fn kmsan_copy_to_user(to: *mut (), from: *const (), to_copy: u64, left: u64) {
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
core::arch::asm!("11: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 11b - ., 3; .popsection\n# {0}", const 11 as i32, options(nostack, att_syntax));
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
core::arch::asm!("12: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 12b - ., 4; .popsection\n# {0}", const 12 as i32, options(nostack, att_syntax));
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

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
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
fn atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
instrument_atomic_read(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_read(v)
}

#[inline(always)]
fn atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
instrument_atomic_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_set(v, i);
}

#[inline(always)]
fn atomic_fetch_add_relaxed(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_fetch_add_relaxed(i, v)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
}


#[inline(always)]
fn atomic_inc(v: *mut __SlateRecord_types_h_190) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_inc(v);
}

#[inline(always)]
fn atomic_dec(v: *mut __SlateRecord_types_h_190) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_dec(v);
}

#[inline(always)]
fn const_test_bit(nr: u64, addr: *const u64) -> bool {
let p: *const u64 = (addr as *const u64).wrapping_offset(BIT_WORD!(nr) as isize);
let mask: u64 = BIT_MASK!(nr);
let val: u64 = unsafe { *p };
val & mask != 0
}

#[inline(always)]
fn arch_test_and_set_bit(mut nr: i64, mut addr: *mut u64) -> bool {
let mut __v0: bool;
let mut c: bool = false;
unsafe {
let __t0: u8;
core::arch::asm!("lock  btsq  {2:r}, ({0:r})\nsetc {1}", in(reg) addr, lateout(reg_byte) __t0, in(reg) nr as u64, options(nostack, att_syntax));
c = (__t0 as u8) != 0;
}
__v0 = c;
__v0
}

#[inline(always)]
fn constant_test_bit(nr: i64, addr: *const u64) -> bool {
1u64 << (nr & 63i64) & unsafe { core::ptr::read_volatile(addr.wrapping_offset((nr >> 6i32) as isize)) } != 0
}

#[inline(always)]
fn variable_test_bit(mut nr: i64, mut addr: *const u64) -> bool {
let mut oldbit: bool = false;
unsafe {
let __t0: u8;
core::arch::asm!(" btq  {2:r},({1:r})\nsetc {0}", lateout(reg_byte) __t0, in(reg) addr as *mut u64, in(reg) nr as u64, options(nostack, att_syntax));
oldbit = (__t0 as u8) != 0;
}
oldbit
}

#[inline(always)]
fn arch_test_bit(nr: u64, addr: *const u64) -> bool {
let __v0: i32 = if unsafe { __slate_intrinsic_f217c4720f04f6db(nr) } {
constant_test_bit(nr as i64, addr) as i32
} else {
variable_test_bit(nr as i64, addr) as i32
};
__v0 != 0
}

// __ffs - find first set bit in word
// @word: The word to search
//
// Undefined if no bit exists, so code should check against 0 first.
// ffz - find first zero bit in word
// @word: The word to search
//
// Undefined if no zero exists, so code should check against ~0UL first.
#[inline(always)]
fn test_and_set_bit(nr: i64, addr: *mut u64) -> bool {
kcsan_mb!();
instrument_atomic_read_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_and_set_bit(nr, addr)
}

#[inline(always)]
fn _test_bit(nr: u64, addr: *const u64) -> bool {
instrument_atomic_read(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_bit(nr, addr)
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
struct __SlateRecord_nodemask_types_h_17 {
    bits: [u64; 1],
}


fn might_fault() {
}

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

fn list_is_head(list: *const list_head, head: *const list_head) -> i32 {
(list == head) as i32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct futex_pi_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct robust_list_head {
}


// list_for_each	-	iterate over a list
// @pos:	the &struct list_head to use as a loop cursor.
// @head:	the head for your list.
// list_for_each_entry	-	iterate over list of given type
// @pos:	the type * to use as a loop cursor.
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
fn hlist_empty(h: *const hlist_head) -> i32 {
let mut __v0: *mut hlist_node;
loop {
if !(__native_word__6!() || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_30() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*h).first) as *const *mut hlist_node) };
__v0.is_null() as i32
}

#[inline(always)]
fn get_current() -> *mut task_struct {
if IS_ENABLED!() != 0 {
let mut __v0: *mut task_struct;
loop {
if true {
unsafe { __compiletime_assert_57() };
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

#[repr(C)]
#[derive(Clone, Copy)]
struct futex_private_hash {
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
    __slate_bits_0: __slate_bits::__SlateBits102U0,
    pkru: u32,
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
struct pid_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sem_undo_list {
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
struct thread_info {
    flags: u64,
    syscall_work: u64,
    status: u32,
    cpu: u32,
}


// 32-bit address space on 64 bits
#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer_clock_base {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct spinlock {
    __slate_anon_0: __SlateRecord_spinlock_types_h_18,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwlock {
    raw_lock: qrwlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seccomp_filter {
}


// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
#[inline(always)]
fn spin_lock_irq(lock: *mut spinlock) {
raw_spin_lock_irq!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[inline(always)]
fn spin_unlock_irq(lock: *mut spinlock) {
raw_spin_unlock_irq!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

static class_spinlock_irq_is_conditional: bool = 0i32 != 0i32;

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_spinlock_h_620 {
    lock: *mut spinlock,
}


#[inline(always)]
fn class_spinlock_irq_destructor(_T: *mut __SlateRecord_spinlock_h_620) {
spin_unlock_irq(unsafe { (*_T).lock });
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


#[inline(always)]
fn class_spinlock_irq_lock_ptr(_T: &mut __SlateRecord_spinlock_h_620) -> *mut () {
let mut _ptr: *mut () = ((unsafe { *core::ptr::addr_of_mut!((*(_T as *mut __SlateRecord_spinlock_h_620)).lock) }) as u64) as *mut ();
if IS_ERR(_ptr as *const ()) {
_ptr = core::ptr::null_mut::<()>();
}
_ptr
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


#[inline(always)]
fn class_spinlock_irq_constructor(l: *mut spinlock) -> __SlateRecord_spinlock_h_620 {
let mut _t: __SlateRecord_spinlock_h_620 = __SlateRecord_spinlock_h_620 { lock: l };
let _T: *mut __SlateRecord_spinlock_h_620 = core::ptr::addr_of_mut!(_t);
spin_lock_irq(unsafe { (*_T).lock });
return _t;
}

#[inline(always)]
fn __class_spinlock_irq_cleanup_ctx(_T: *mut *mut __SlateRecord_spinlock_h_620) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct wait_queue_head {
    lock: spinlock,
    head: list_head,
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

static class_mutex_is_conditional: bool = 0i32 != 0i32;

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mutex_h_253 {
    lock: *mut mutex,
}


#[inline(always)]
fn class_mutex_destructor(_T: *mut __SlateRecord_mutex_h_253) {
unsafe { mutex_unlock((*_T).lock) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct workqueue_struct {
}


#[inline(always)]
fn class_mutex_lock_ptr(_T: &mut __SlateRecord_mutex_h_253) -> *mut () {
let mut _ptr: *mut () = ((unsafe { *core::ptr::addr_of_mut!((*(_T as *mut __SlateRecord_mutex_h_253)).lock) }) as u64) as *mut ();
if IS_ERR(_ptr as *const ()) {
_ptr = core::ptr::null_mut::<()>();
}
_ptr
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
}


#[inline(always)]
fn class_mutex_constructor(l: *mut mutex) -> __SlateRecord_mutex_h_253 {
let mut _t: __SlateRecord_mutex_h_253 = __SlateRecord_mutex_h_253 { lock: l };
let _T: *mut __SlateRecord_mutex_h_253 = core::ptr::addr_of_mut!(_t);
unsafe { mutex_lock((*_T).lock) };
return _t;
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
fn __class_mutex_cleanup_ctx(_T: *mut *mut __SlateRecord_mutex_h_253) {
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
struct seqlock {
    seqcount: seqcount_spinlock,
    lock: spinlock,
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
struct refcount_struct {
    refs: __SlateRecord_types_h_190,
}


fn __refcount_add(i: i32, r: *mut refcount_struct, oldp: *mut i32) {
let old: i32 = atomic_fetch_add_relaxed(i, unsafe { core::ptr::addr_of_mut!((*r).refs) });
if !oldp.is_null() {
unsafe {
*oldp = old;
}
}
if unlikely!(old == 0) != 0 {
unsafe { refcount_warn_saturate(r, 2u32) };
} else {
if unlikely!(old < 0 || old.wrapping_add(i) < 0) != 0 {
unsafe { refcount_warn_saturate(r, 1u32) };
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


fn __refcount_inc(r: *mut refcount_struct, oldp: *mut i32) {
__refcount_add(1i32, r, oldp);
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


fn refcount_inc(r: *mut refcount_struct) {
__refcount_inc(r, core::ptr::null_mut::<i32>());
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
struct kref {
    refcount: refcount_struct,
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
struct rb_root_cached {
    rb_root: rb_root,
    rb_leftmost: *mut rb_node,
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
struct sysv_sem {
    undo_list: *mut sem_undo_list,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rq_flags {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sysv_shm {
    shm_clist: list_head,
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
struct plist_node {
    prio: i32,
    prio_list: list_head,
    node_list: list_head,
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
struct rlimit {
    rlim_cur: u64,
    rlim_max: u64,
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
struct __SlateRecord_signal_h_25 {
    sig: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gendisk {
}


#[repr(C)]
#[derive(Clone, Copy)]
union sigval {
    sival_int: i32,
    sival_ptr: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_struct {
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
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct k_sigaction {
    sa: sigaction,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
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
struct uprobe {
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


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct mm_cid_pcpu {
    cid: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe_xol_ops {
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
struct unwind_cache {
    unwind_completed: u64,
    nr_entries: u32,
    entries: [u64; 0],
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
union unwind_task_id {
    __slate_anon_0: __SlateRecord_unwind_deferred_types_h_28,
    id: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
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
struct ldt_struct {
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
struct mem_cgroup {
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
    __slate_bits_0: __slate_bits::__SlateBits270U0,
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
    __slate_bits_0: __slate_bits::__SlateBits65U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits65U1,
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
    __slate_bits_2: __slate_bits::__SlateBits65U2,
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


// Per process flags
// I'm a virtual CPU
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
struct device_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache {
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
struct rcu_gp_seq {
    norm: u64,
    exp: u64,
}


fn rcu_read_lock_held() -> i32 {
1
}

#[repr(C)]
#[derive(Clone, Copy)]
struct user_namespace {
}


// Dependency order vs. p above.
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

#[repr(C)]
#[derive(Clone, Copy)]
struct ns_common {
}


static class_rcu_is_conditional: bool = 0i32 != 0i32;

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_rcupdate_h_1216 {
    lock: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_context {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_open_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_iattrs {
}


#[inline(always)]
fn class_rcu_destructor(_T: *mut __SlateRecord_rcupdate_h_1216) {
rcu_read_unlock();
}

#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
}


#[inline(always)]
fn class_rcu_lock_ptr(_T: &mut __SlateRecord_rcupdate_h_1216) -> *mut () {
let mut _ptr: *mut () = ((unsafe { *core::ptr::addr_of_mut!((*(_T as *mut __SlateRecord_rcupdate_h_1216)).lock) }) as u64) as *mut ();
if IS_ERR(_ptr as *const ()) {
_ptr = core::ptr::null_mut::<()>();
}
_ptr
}

#[inline(always)]
fn class_rcu_constructor() -> __SlateRecord_rcupdate_h_1216 {
let _t: __SlateRecord_rcupdate_h_1216 = __SlateRecord_rcupdate_h_1216 { lock: 1i32 as *mut () };
rcu_read_lock();
return _t;
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
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_rwsem_h_259 {
    lock: *mut rw_semaphore,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sock {
}


#[inline(always)]
fn class_rwsem_read_destructor(_T: *mut __SlateRecord_rwsem_h_259) {
unsafe { up_read((*_T).lock) };
}

#[inline(always)]
fn class_rwsem_read_constructor(l: *mut rw_semaphore) -> __SlateRecord_rwsem_h_259 {
let mut _t: __SlateRecord_rwsem_h_259 = __SlateRecord_rwsem_h_259 { lock: l };
let _T: *mut __SlateRecord_rwsem_h_259 = core::ptr::addr_of_mut!(_t);
unsafe { down_read((*_T).lock) };
return _t;
}

#[inline(always)]
fn __class_rwsem_read_cleanup_ctx(_T: *mut *mut __SlateRecord_rwsem_h_259) {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct wake_irq {
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
struct dev_pm_qos {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct fwnode_handle {
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
struct folio_queue {
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
union __SlateRecord_uio_h_78 {
    nr_segs: u64,
    folioq_slot: u8,
    xarray_start: i64,
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
struct __SlateRecord_mm_types_h_701 {
    v: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_userfaultfd_ctx {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mnt_idmap {
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
struct assoc_array_ptr {
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
struct key_type {
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


// Opaque type representing current mm_struct flag state. Must be accessed via
// mm_flags_xxx() helper functions.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_1171 {
    __mm_flags: [u64; 1],
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


#[repr(C)]
#[derive(Clone, Copy)]
struct mm_struct {
    __slate_anon_0: __SlateRecord_mm_types_h_1176,
    flexible_array: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_cred_h_147 {
    non_rcu: i32,
    rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pidfs_attr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_slab_h_532 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pid_h_62 {
    ino: u64,
    pidfs_hash: rhash_head,
    stashed: *mut dentry,
    attr: *mut pidfs_attr,
}


// no-op
#[inline(always)]
fn __free_kfree(p: *mut ()) {
let _T: *mut () = unsafe { *(p as *mut *mut ()) };
if !IS_ERR_OR_NULL(_T as *const ()) {
unsafe { kfree(_T as *const ()) };
}
}

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
// kmalloc_obj - Allocate a single instance of the given type
// @VAR_OR_TYPE: Variable or type to allocate.
// @...: optional GFP flags for the allocation (GFP_KERNEL when not specified).
//
// Returns: newly allocated pointer to a @VAR_OR_TYPE on success, or NULL
// on failure.
// All kzalloc aliases for kmalloc_(obj|objs|flex).
fn _kzalloc_noprof(size: u64, flags: u32, token: __SlateRecord_slab_h_532) -> *mut () {
_kmalloc_noprof(size, flags | (BIT!(8i32) as u32), token)
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
struct kernfs_elem_dir {
    subdirs: u64,
    children: rb_root,
    root: *mut kernfs_root,
    rev: u64,
}


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
    __slate_bits_0: __slate_bits::__SlateBits441U0,
    vm_ops: *const vm_operations_struct,
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
    __slate_bits_0: __slate_bits::__SlateBits443U0,
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
struct backing_dev_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct block_device {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pm_message {
    event: i32,
}


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
struct dev_pm_ops {
    prepare: Option<unsafe extern "C" fn(*mut device) -> i32>,
    complete: Option<unsafe extern "C" fn(*mut device)>,
    suspend: Option<unsafe extern "C" fn(*mut device) -> i32>,
    resume: Option<unsafe extern "C" fn(*mut device) -> i32>,
    freeze: Option<unsafe extern "C" fn(*mut device) -> i32>,
    thaw: Option<unsafe extern "C" fn(*mut device) -> i32>,
    poweroff: Option<unsafe extern "C" fn(*mut device) -> i32>,
    restore: Option<unsafe extern "C" fn(*mut device) -> i32>,
    suspend_late: Option<unsafe extern "C" fn(*mut device) -> i32>,
    resume_early: Option<unsafe extern "C" fn(*mut device) -> i32>,
    freeze_late: Option<unsafe extern "C" fn(*mut device) -> i32>,
    thaw_early: Option<unsafe extern "C" fn(*mut device) -> i32>,
    poweroff_late: Option<unsafe extern "C" fn(*mut device) -> i32>,
    restore_early: Option<unsafe extern "C" fn(*mut device) -> i32>,
    suspend_noirq: Option<unsafe extern "C" fn(*mut device) -> i32>,
    resume_noirq: Option<unsafe extern "C" fn(*mut device) -> i32>,
    freeze_noirq: Option<unsafe extern "C" fn(*mut device) -> i32>,
    thaw_noirq: Option<unsafe extern "C" fn(*mut device) -> i32>,
    poweroff_noirq: Option<unsafe extern "C" fn(*mut device) -> i32>,
    restore_noirq: Option<unsafe extern "C" fn(*mut device) -> i32>,
    runtime_suspend: Option<unsafe extern "C" fn(*mut device) -> i32>,
    runtime_resume: Option<unsafe extern "C" fn(*mut device) -> i32>,
    runtime_idle: Option<unsafe extern "C" fn(*mut device) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pm_subsys_data {
    lock: spinlock,
    refcount: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_info {
    power_state: pm_message,
    __slate_bits_0: __slate_bits::__SlateBits468U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits468U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits468U2,
    links_count: u32,
    request: u32,
    runtime_status: i32,
    last_status: i32,
    runtime_error: i32,
    autosuspend_delay: i32,
    last_busy: u64,
    active_time: u64,
    suspended_time: u64,
    accounting_timestamp: u64,
    subsys_data: *mut pm_subsys_data,
    set_latency_tolerance: Option<unsafe extern "C" fn(*mut device, i32)>,
    qos: *mut dev_pm_qos,
    __slate_bits_3: __slate_bits::__SlateBits468U3,
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
struct dev_pm_domain {
    ops: dev_pm_ops,
    start: Option<unsafe extern "C" fn(*mut device) -> i32>,
    detach: Option<unsafe extern "C" fn(*mut device, bool)>,
    activate: Option<unsafe extern "C" fn(*mut device) -> i32>,
    sync: Option<unsafe extern "C" fn(*mut device)>,
    dismiss: Option<unsafe extern "C" fn(*mut device)>,
    set_performance_state: Option<unsafe extern "C" fn(*mut device, u32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bus_type {
    name: *const u8,
    dev_name: *const u8,
    bus_groups: *const *const attribute_group,
    dev_groups: *const *const attribute_group,
    drv_groups: *const *const attribute_group,
    r#match: Option<unsafe extern "C" fn(*mut device, *const device_driver) -> i32>,
    uevent: Option<unsafe extern "C" fn(*const device, *mut kobj_uevent_env) -> i32>,
    probe: Option<unsafe extern "C" fn(*mut device) -> i32>,
    sync_state: Option<unsafe extern "C" fn(*mut device)>,
    remove: Option<unsafe extern "C" fn(*mut device)>,
    shutdown: Option<unsafe extern "C" fn(*mut device)>,
    irq_get_affinity: Option<unsafe extern "C" fn(*mut device, u32) -> *const cpumask>,
    online: Option<unsafe extern "C" fn(*mut device) -> i32>,
    offline: Option<unsafe extern "C" fn(*mut device) -> i32>,
    suspend: Option<unsafe extern "C" fn(*mut device, pm_message) -> i32>,
    resume: Option<unsafe extern "C" fn(*mut device) -> i32>,
    num_vf: Option<unsafe extern "C" fn(*mut device) -> i32>,
    dma_configure: Option<unsafe extern "C" fn(*mut device) -> i32>,
    dma_cleanup: Option<unsafe extern "C" fn(*mut device)>,
    pm: *const dev_pm_ops,
    driver_override: bool,
    need_parent_lock: bool,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct class {
    name: *const u8,
    class_groups: *const *const attribute_group,
    dev_groups: *const *const attribute_group,
    dev_uevent: Option<unsafe extern "C" fn(*const device, *mut kobj_uevent_env) -> i32>,
    devnode: Option<unsafe extern "C" fn(*const device, *mut u16) -> *mut u8>,
    class_release: Option<unsafe extern "C" fn(*const class)>,
    dev_release: Option<unsafe extern "C" fn(*mut device)>,
    shutdown_pre: Option<unsafe extern "C" fn(*mut device) -> i32>,
    ns_type: *const kobj_ns_type_operations,
    namespace: Option<unsafe extern "C" fn(*const device) -> *const ns_common>,
    get_ownership: Option<unsafe extern "C" fn(*const device, *mut __SlateRecord_uidgid_types_h_9, *mut __SlateRecord_uidgid_types_h_13)>,
    pm: *const dev_pm_ops,
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
struct rhash_head {
    next: *mut rhash_head,
}


fn check_object_size(ptr: *const (), n: u64, to_user: bool) {
}

fn copy_overflow(size: i32, count: u64) {
if IS_ENABLED__2!() != 0 {
unsafe { __copy_overflow(size, count) };
}
}

#[inline(always)]
fn check_copy_size(addr: *const (), bytes: u64, is_source: bool) -> bool {
let sz: i32 = (unsafe { __slate_intrinsic_2913c0303de876b3(addr, const { false as bool }, const { false as bool }, const { false as bool }) }) as i32;
if unlikely!(sz >= 0 && (sz as u64) < bytes) != 0 {
if !unsafe { __slate_intrinsic_f217c4720f04f6db(bytes) } {
copy_overflow(sz, bytes);
} else {
if is_source {
unsafe { __bad_copy_from() };
} else {
unsafe { __bad_copy_to() };
}
}
return false;
}
let mut __v0: i64;
let __ret_warn_on: i32 = (bytes > 2147483647u64) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("249: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 249b - ., 3; .popsection\n# {0}", const 249 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_check_copy_size_1, sym __slate_asm_string_check_copy_size_2, const 57 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("250: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 250b - ., 4; .popsection\n# {0}", const 250 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
return false;
}
check_object_size(addr, bytes, is_source);
true
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
struct iovec {
    iov_base: *mut (),
    iov_len: u64,
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
struct kvec {
    iov_base: *mut (),
    iov_len: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_acl {
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
struct list_lru_one {
    list: list_head,
    nr_items: i64,
    lock: spinlock,
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
struct list_lru_node {
    lru: list_lru_one,
    nr_items: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_cmd {
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
struct exception_table_entry {
    insn: i32,
    fixup: i32,
    data: i32,
}


fn __access_ok(ptr: *const (), size: u64) -> bool {
if (unsafe { __slate_intrinsic_1cfe0293d0a2af8d(size <= _AC!() << 12i32) }) && size <= _AC!() << 12i32 {
let mut __v0: u64;
let mut __ret: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("mov ${1},{0:r}\n1:\n.pushsection runtime_ptr_USER_PTR_MAX,\"a\"\n\t.long 1b - {2} - .\n.popsection", lateout(reg) __t0, const 81985529216486895 as u64, const 8 as u64, options(pure, nomem, nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
__v0 = __ret;
return (ptr as u64) <= __v0;
} else {
let sum: u64 = size.wrapping_add(ptr as u64);
let mut __v1: u64;
let mut __ret: u64 = 0;
unsafe {
let __t1: u64;
core::arch::asm!("mov ${1},{0:r}\n1:\n.pushsection runtime_ptr_USER_PTR_MAX,\"a\"\n\t.long 1b - {2} - .\n.popsection", lateout(reg) __t1, const 81985529216486895 as u64, const 8 as u64, options(pure, nomem, nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u64, u64>(__t1 as u64) };
}
__v1 = __ret;
return sum <= __v1 && sum >= (ptr as u64);
}
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
}


#[inline(always)]
fn copy_from_user(to: *mut (), from: *const (), n: u64) -> u64 {
if !check_copy_size(to as *const (), n, false) {
return n;
}
unsafe { _copy_from_user(to, from, n) }
}

#[inline(always)]
fn copy_to_user(to: *mut (), from: *const (), n: u64) -> u64 {
if !check_copy_size(from, n, true) {
return n;
}
unsafe { _copy_to_user(to, from, n) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_capability_h_24 {
    val: u64,
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
struct module_param_attrs {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_eval_map {
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
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct upid {
    nr: i32,
    ns: *mut pid_namespace,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iommu_group {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_iommu {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_device_data {
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


fn get_pid(pid: *mut pid) -> *mut pid {
if !pid.is_null() {
refcount_inc(unsafe { core::ptr::addr_of_mut!((*pid).count) });
}
pid
}

fn task_pid(task: *mut task_struct) -> *mut pid {
unsafe { (*task).thread_pid }
}

fn hrtimer_get_expires(timer: *const hrtimer) -> i64 {
unsafe { (*timer).node.expires }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_device_h_719 {
    name: *const u8,
    lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bus_dma_region {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_tlb_mem {
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
struct proc_dir_entry {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_shutdown_f_ops {
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
    __slate_bits_0: __slate_bits::__SlateBits273U0,
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
if !(__native_word__2!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_395() };
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
union __SlateRecord_asequencer_h_242 {
    value: i32,
    time: snd_seq_timestamp,
    position: u32,
    skew: snd_seq_queue_skew,
    d32: [u32; 2],
    d8: [u8; 8],
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
union __SlateRecord_asequencer_h_305 {
    data: snd_seq_event_data,
    ump: [u32; 4],
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


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_asequencer_h_546 {
    alsa: __SlateRecord_asequencer_h_547,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_asequencer_h_547 {
    id: snd_timer_id,
    resolution: u32,
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
union __SlateRecord_info_h_65 {
    text: snd_info_entry_text,
    ops: *const snd_info_entry_ops,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct delayed_call {
    r#fn: Option<unsafe extern "C" fn(*mut ())>,
    arg: *mut (),
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct __SlateRecord_seq_memory_h_20 {
    event: snd_seq_event,
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
union __SlateRecord_seq_memory_h_30 {
    event: snd_seq_event,
    ump: __snd_seq_event,
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
if !(__native_word__3!() || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_416() };
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
struct __SlateRecord_file_ref_h_52 {
    refcnt: __SlateRecord_types_h_197,
}


// file is open for writing
// File supports atomic writes
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
union __SlateRecord_seq_clientmgr_h_60 {
    user: snd_seq_user_client,
    kernel: snd_seq_kernel_client,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_seq_clientmgr_c_1230 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_seq_clientmgr_c_1230_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_seq_clientmgr_c_1230_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_seq_clientmgr_c_1293 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_seq_clientmgr_c_1293_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_seq_clientmgr_c_1300 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_seq_clientmgr_c_1584 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_seq_clientmgr_c_1584_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_seq_clientmgr_c_1612 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_seq_clientmgr_c_1612_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_seq_clientmgr_c_1642 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_seq_clientmgr_c_1642_2 {
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


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_seq_clientmgr_c_2260 {
    pversion: i32,
    client_id: i32,
    system_info: snd_seq_system_info,
    running_info: snd_seq_running_info,
    client_info: snd_seq_client_info,
    port_info: snd_seq_port_info,
    port_subscribe: snd_seq_port_subscribe,
    queue_info: snd_seq_queue_info,
    queue_status: snd_seq_queue_status,
    tempo: snd_seq_queue_tempo,
    queue_timer: snd_seq_queue_timer,
    queue_client: snd_seq_queue_client,
    client_pool: snd_seq_client_pool,
    remove_events: snd_seq_remove_events,
    query_subs: snd_seq_query_subs,
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
    pub struct __SlateBits270U0 {
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
    pub struct __SlateBits65U0 {
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
    pub struct __SlateBits65U1 {
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
    pub struct __SlateBits273U0 {
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
    pub struct __SlateBits65U2 {
        #[bits(1)]
        pub mce_ripv: u64,
        #[bits(1)]
        pub mce_whole_page: u64,
        #[bits(62)]
        pub __mce_reserved: u64,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits92U0 {
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
    pub struct __SlateBits102U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits441U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits443U0 {
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
    pub struct __SlateBits91U0 {
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
    pub struct __SlateBits755U0 {
        #[bits(1)]
        pub accept_input: u32,
        #[bits(1)]
        pub accept_output: u32,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits468U0 {
        #[bits(1)]
        pub can_wakeup: bool,
        #[bits(1)]
        pub async_suspend: bool,
        #[bits(1)]
        pub in_dpm_list: bool,
        #[bits(1)]
        pub is_prepared: bool,
        #[bits(1)]
        pub is_suspended: bool,
        #[bits(1)]
        pub is_noirq_suspended: bool,
        #[bits(1)]
        pub is_late_suspended: bool,
        #[bits(1)]
        pub no_pm: bool,
        #[bits(1)]
        pub early_init: bool,
        #[bits(1)]
        pub direct_complete: bool,
        #[bits(6, access = na)]
        pub __slate_pad_10: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits465U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits468U1 {
        #[bits(1)]
        pub wakeup_path: bool,
        #[bits(1)]
        pub syscore: bool,
        #[bits(1)]
        pub no_pm_callbacks: bool,
        #[bits(1)]
        pub smart_suspend: bool,
        #[bits(1)]
        pub must_resume: bool,
        #[bits(1)]
        pub may_skip_resume: bool,
        #[bits(1)]
        pub out_band_wakeup: bool,
        #[bits(1)]
        pub strict_midlayer: bool,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits468U2 {
        #[bits(3)]
        pub disable_depth: u32,
        #[bits(1)]
        pub idle_notification: bool,
        #[bits(1)]
        pub request_pending: bool,
        #[bits(1)]
        pub deferred_resume: bool,
        #[bits(1)]
        pub needs_force_resume: bool,
        #[bits(1)]
        pub runtime_auto: bool,
        #[bits(1)]
        pub ignore_children: bool,
        #[bits(1)]
        pub no_callbacks: bool,
        #[bits(1)]
        pub irq_safe: bool,
        #[bits(1)]
        pub use_autosuspend: bool,
        #[bits(1)]
        pub timer_autosuspends: bool,
        #[bits(1)]
        pub memalloc_noio: bool,
        #[bits(2, access = na)]
        pub __slate_pad_12: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits468U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits726U0 {
        #[bits(1)]
        pub locked: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits753U0 {
        #[bits(1)]
        pub exclusive: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits754U0 {
        #[bits(1)]
        pub closing: u32,
        #[bits(1)]
        pub timestamping: u32,
        #[bits(1)]
        pub time_real: u32,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits765U0 {
        #[bits(1)]
        pub running: u32,
        #[bits(1)]
        pub initialized: u32,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
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


fn compat_ptr(uptr: u32) -> *mut () {
(uptr as u64) as *mut ()
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
struct acpi_device_id {
    id: [u8; 16],
    driver_data: u64,
    cls: u32,
    cls_msk: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct of_device_id {
    name: [u8; 32],
    r#type: [u8; 32],
    compatible: [u8; 128],
    data: *const (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_driver {
    name: *const u8,
    bus: *const bus_type,
    owner: *mut module,
    mod_name: *const u8,
    suppress_bind_attrs: bool,
    probe_type: u32,
    of_match_table: *const of_device_id,
    acpi_match_table: *const acpi_device_id,
    probe: Option<unsafe extern "C" fn(*mut device) -> i32>,
    sync_state: Option<unsafe extern "C" fn(*mut device)>,
    remove: Option<unsafe extern "C" fn(*mut device) -> i32>,
    shutdown: Option<unsafe extern "C" fn(*mut device)>,
    suspend: Option<unsafe extern "C" fn(*mut device, pm_message) -> i32>,
    resume: Option<unsafe extern "C" fn(*mut device) -> i32>,
    groups: *const *const attribute_group,
    dev_groups: *const *const attribute_group,
    pm: *const dev_pm_ops,
    coredump: Option<unsafe extern "C" fn(*mut device)>,
    p: *mut driver_private,
    p_cb: __SlateRecord_driver_h_126,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_archdata {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_type {
    name: *const u8,
    groups: *mut *const attribute_group,
    uevent: Option<unsafe extern "C" fn(*const device, *mut kobj_uevent_env) -> i32>,
    devnode: Option<unsafe extern "C" fn(*const device, *mut u16, *mut __SlateRecord_uidgid_types_h_9, *mut __SlateRecord_uidgid_types_h_13) -> *mut u8>,
    release: Option<unsafe extern "C" fn(*mut device)>,
    pm: *const dev_pm_ops,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_dma_parameters {
    max_segment_size: u32,
    min_align_mask: u32,
    segment_boundary_mask: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_links_info {
    suppliers: list_head,
    consumers: list_head,
    defer_sync: list_head,
    status: u32,
}


core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_snd_seq_set_queue_tempo: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_seq_create_kernel_client: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_seq_delete_kernel_client: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_seq_kernel_client_enqueue: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_seq_kernel_client_dispatch: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_seq_kernel_client_ctl: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_seq_kernel_client_ioctl: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_seq_kernel_client_write_poll: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {7} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_seq_kernel_client_get: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {8} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_snd_seq_kernel_client_put: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {9} ; .previous", sym snd_seq_set_queue_tempo, sym snd_seq_create_kernel_client, sym snd_seq_delete_kernel_client, sym snd_seq_kernel_client_enqueue, sym snd_seq_kernel_client_dispatch, sym snd_seq_kernel_client_ctl, sym snd_seq_kernel_client_ioctl, sym snd_seq_kernel_client_write_poll, sym snd_seq_kernel_client_get, sym snd_seq_kernel_client_put, options(att_syntax));

#[repr(C)]
#[derive(Clone, Copy)]
struct dev_msi_info {
    domain: *mut irq_domain,
    data: *mut msi_device_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_physical_location {
    panel: u32,
    vertical_position: u32,
    horizontal_position: u32,
    dock: bool,
    lid: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device {
    kobj: kobject,
    parent: *mut device,
    p: *mut device_private,
    init_name: *const u8,
    r#type: *const device_type,
    bus: *const bus_type,
    driver: *mut device_driver,
    platform_data: *mut (),
    driver_data: *mut (),
    driver_override: __SlateRecord_device_h_719,
    mutex: mutex,
    links: dev_links_info,
    power: dev_pm_info,
    pm_domain: *mut dev_pm_domain,
    msi: dev_msi_info,
    dma_mask: *mut u64,
    coherent_dma_mask: u64,
    bus_dma_limit: u64,
    dma_range_map: *const bus_dma_region,
    dma_parms: *mut device_dma_parameters,
    dma_pools: list_head,
    dma_io_tlb_mem: *mut io_tlb_mem,
    archdata: dev_archdata,
    of_node: *mut device_node,
    fwnode: *mut fwnode_handle,
    numa_node: i32,
    devt: u32,
    id: u32,
    devres_lock: spinlock,
    devres_head: list_head,
    class: *const class,
    groups: *mut *const attribute_group,
    release: Option<unsafe extern "C" fn(*mut device)>,
    iommu_group: *mut iommu_group,
    iommu: *mut dev_iommu,
    physical_location: *mut device_physical_location,
    removable: u32,
    flags: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wakeup_source {
    name: *const u8,
    id: i32,
    entry: list_head,
    lock: spinlock,
    wakeirq: *mut wake_irq,
    timer: timer_list,
    timer_expires: u64,
    total_time: i64,
    max_time: i64,
    last_time: i64,
    start_prevent_time: i64,
    prevent_sleep_time: i64,
    event_count: u64,
    active_count: u64,
    relax_count: u64,
    expire_count: u64,
    wakeup_count: u64,
    dev: *mut device,
    __slate_bits_0: __slate_bits::__SlateBits465U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_refcount {
    count: __SlateRecord_types_h_190,
    waiter: wait_queue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_card {
    number: i32,
    id: [u8; 16],
    driver: [u8; 16],
    shortname: [u8; 32],
    longname: [u8; 80],
    irq_descr: [u8; 32],
    mixername: [u8; 80],
    components: *mut u8,
    components_alloc_size: u32,
    module: *mut module,
    private_data: *mut (),
    private_free: Option<unsafe extern "C" fn(*mut snd_card)>,
    devices: list_head,
    ctl_dev: *mut device,
    last_numid: u32,
    controls_rwsem: rw_semaphore,
    controls_rwlock: rwlock,
    controls_count: i32,
    user_ctl_alloc_size: u64,
    controls: list_head,
    ctl_files: list_head,
    ctl_numids: xarray,
    ctl_hash: xarray,
    ctl_hash_collision: bool,
    proc_root: *mut snd_info_entry,
    proc_root_link: *mut proc_dir_entry,
    files_list: list_head,
    s_f_ops: *mut snd_shutdown_f_ops,
    files_lock: spinlock,
    shutdown: i32,
    release_completion: *mut completion,
    dev: *mut device,
    card_dev: device,
    dev_groups: [*const attribute_group; 4],
    registered: bool,
    managed: bool,
    releasing: bool,
    sync_irq: i32,
    remove_sleep: wait_queue_head,
    total_pcm_alloc_bytes: u64,
    memory_mutex: mutex,
    power_state: u32,
    power_sleep: wait_queue_head,
    power_ref: snd_refcount,
    private_data_area: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_timer_id {
    dev_class: i32,
    dev_sclass: i32,
    card: i32,
    device: i32,
    subdevice: i32,
}


// note messages (channel specific)
// event data type = #snd_seq_ev_note
// port connection changes
// event data type = snd_seq_connect
// ports connected
// ports disconnected
// error event
// 150-151: kernel events with quote - DO NOT use in user clients
// 255: special event
#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_addr {
    client: u8,
    port: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_connect {
    sender: snd_seq_addr,
    dest: snd_seq_addr,
}


// unknown source
// send event to all subscribed ports
// direct dispatch
// event mode flag - NOTE: only 8 bits available!
// timestamp in clock ticks
// timestamp in real time
// fixed event size
// variable event size
// variable event size - user memory space
// event should be processed before others
#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_ev_note {
    channel: u8,
    note: u8,
    velocity: u8,
    off_velocity: u8,
    duration: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_ev_ctrl {
    channel: u8,
    unused1: u8,
    unused2: u8,
    unused3: u8,
    param: u32,
    value: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_ev_raw8 {
    d: [u8; 12],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_ev_raw32 {
    d: [u32; 3],
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct snd_seq_ev_ext {
    len: u32,
    ptr: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_result {
    event: i32,
    result: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_real_time {
    tv_sec: u32,
    tv_nsec: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union snd_seq_timestamp {
    tick: u32,
    time: snd_seq_real_time,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_queue_skew {
    value: u32,
    base: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_ev_queue_control {
    queue: u8,
    pad: [u8; 3],
    param: __SlateRecord_asequencer_h_242,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct snd_seq_ev_quote {
    origin: snd_seq_addr,
    value: u16,
    event: *mut snd_seq_event,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_ev_ump_notify {
    client: u8,
    block: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union snd_seq_event_data {
    note: snd_seq_ev_note,
    control: snd_seq_ev_ctrl,
    raw8: snd_seq_ev_raw8,
    raw32: snd_seq_ev_raw32,
    ext: snd_seq_ev_ext,
    queue: snd_seq_ev_queue_control,
    time: snd_seq_timestamp,
    addr: snd_seq_addr,
    connect: snd_seq_connect,
    result: snd_seq_result,
    quote: snd_seq_ev_quote,
    ump_notify: snd_seq_ev_ump_notify,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_event {
    r#type: u8,
    flags: u8,
    tag: u8,
    queue: u8,
    time: snd_seq_timestamp,
    source: snd_seq_addr,
    dest: snd_seq_addr,
    data: snd_seq_event_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_ump_event {
    r#type: u8,
    flags: u8,
    tag: u8,
    queue: u8,
    time: snd_seq_timestamp,
    source: snd_seq_addr,
    dest: snd_seq_addr,
    __slate_anon_7: __SlateRecord_asequencer_h_305,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_system_info {
    queues: i32,
    clients: i32,
    ports: i32,
    channels: i32,
    cur_clients: i32,
    cur_queues: i32,
    reserved: [u8; 24],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_running_info {
    client: u8,
    big_endian: u8,
    cpu_mode: u8,
    pad: u8,
    reserved: [u8; 12],
}


// known client numbers
// accept bounce event in error
// use event filter
#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_client_info {
    client: i32,
    r#type: i32,
    name: [u8; 64],
    filter: u32,
    multicast_filter: [u8; 8],
    event_filter: [u8; 32],
    num_ports: i32,
    event_lost: i32,
    card: i32,
    pid: i32,
    midi_version: u32,
    group_filter: u32,
    reserved: [u8; 48],
}


// MIDI version numbers in client info
// Legacy client
// UMP MIDI 1.0
// UMP MIDI 2.0
#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_client_pool {
    client: i32,
    output_pool: i32,
    input_pool: i32,
    output_room: i32,
    output_free: i32,
    input_free: i32,
    reserved: [u8; 64],
}


// Flush input queues
// Flush output queues
#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_remove_events {
    remove_mode: u32,
    time: snd_seq_timestamp,
    queue: u8,
    dest: snd_seq_addr,
    channel: u8,
    r#type: i32,
    tag: u8,
    reserved: [i32; 10],
}


// writable to this port
// routing not allowed
// inactive port
// MIDI 2.0 UMP Endpoint port
// misc. conditioning flags
#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_port_info {
    addr: snd_seq_addr,
    name: [u8; 64],
    capability: u32,
    r#type: u32,
    midi_channels: i32,
    midi_voices: i32,
    synth_voices: i32,
    read_use: i32,
    write_use: i32,
    kernel: *mut (),
    flags: u32,
    time_queue: u8,
    direction: u8,
    ump_group: u8,
    reserved: [u8; 57],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_queue_info {
    queue: i32,
    owner: i32,
    __slate_bits_0: __slate_bits::__SlateBits726U0,
    name: [u8; 64],
    flags: u32,
    reserved: [u8; 60],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_queue_status {
    queue: i32,
    events: i32,
    tick: u32,
    time: snd_seq_real_time,
    running: i32,
    flags: i32,
    reserved: [u8; 64],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_queue_tempo {
    queue: i32,
    tempo: u32,
    ppq: i32,
    skew_value: u32,
    skew_base: u32,
    tempo_base: u16,
    reserved: [u8; 22],
}


// sequencer timer sources
// ALSA timer
#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_queue_timer {
    queue: i32,
    r#type: i32,
    u: __SlateRecord_asequencer_h_546,
    reserved: [u8; 64],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_queue_client {
    queue: i32,
    client: i32,
    used: i32,
    reserved: [u8; 64],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_port_subscribe {
    sender: snd_seq_addr,
    dest: snd_seq_addr,
    voices: u32,
    flags: u32,
    queue: u8,
    pad: [u8; 3],
    reserved: [u8; 64],
}


// type of query subscription
#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_query_subs {
    root: snd_seq_addr,
    r#type: i32,
    index: i32,
    num_subs: i32,
    addr: snd_seq_addr,
    queue: u8,
    flags: u32,
    reserved: [u8; 64],
}


// max number of concurrent clients
// max number of concurrent ports
// max number of events in memory pool
// default number of events in memory pool
// max number of events in memory pool for one client (outqueue)
// default number of events in memory pool for one client (outqueue)
// max delivery path length
//
// NOTE: this shouldn't be greater than MAX_LOCKDEP_SUBCLASSES
// max size of event size
#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_port_callback {
    owner: *mut module,
    private_data: *mut (),
    subscribe: Option<unsafe extern "C" fn(*mut (), *mut snd_seq_port_subscribe) -> i32>,
    unsubscribe: Option<unsafe extern "C" fn(*mut (), *mut snd_seq_port_subscribe) -> i32>,
    r#use: Option<unsafe extern "C" fn(*mut (), *mut snd_seq_port_subscribe) -> i32>,
    unuse: Option<unsafe extern "C" fn(*mut (), *mut snd_seq_port_subscribe) -> i32>,
    event_input: Option<unsafe extern "C" fn(*mut snd_seq_event, i32, *mut (), i32, i32) -> i32>,
    private_free: Option<unsafe extern "C" fn(*mut ())>,
}


fn snd_seq_event_packet_size(ev: *mut snd_seq_event) -> u64 {
if snd_seq_ev_is_ump!(ev) {
return core::mem::size_of::<snd_seq_ump_event>() as u64;
}
core::mem::size_of::<snd_seq_event>() as u64
}

// The rest seem to be more-or-less nonstandard. Check them!
#[repr(C)]
#[derive(Clone, Copy)]
struct pollfd {
    fd: i32,
    events: i16,
    revents: i16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct poll_table_struct {
    _qproc: Option<unsafe extern "C" fn(*mut file, *mut wait_queue_head, *mut poll_table_struct)>,
    _key: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seq_file {
    buf: *mut u8,
    size: u64,
    from: u64,
    count: u64,
    pad_until: u64,
    index: i64,
    read_pos: i64,
    lock: mutex,
    op: *const seq_operations,
    poll_event: i32,
    file: *const file,
    private: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seq_operations {
    start: Option<unsafe extern "C" fn(*mut seq_file, *mut i64) -> *mut ()>,
    stop: Option<unsafe extern "C" fn(*mut seq_file, *mut ())>,
    next: Option<unsafe extern "C" fn(*mut seq_file, *mut (), *mut i64) -> *mut ()>,
    show: Option<unsafe extern "C" fn(*mut seq_file, *mut ()) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_info_buffer {
    buffer: *mut u8,
    curr: u32,
    size: u32,
    len: u32,
    stop: i32,
    error: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_info_entry_text {
    read: Option<unsafe extern "C" fn(*mut snd_info_entry, *mut snd_info_buffer)>,
    write: Option<unsafe extern "C" fn(*mut snd_info_entry, *mut snd_info_buffer)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_info_entry_ops {
    open: Option<unsafe extern "C" fn(*mut snd_info_entry, u16, *mut *mut ()) -> i32>,
    release: Option<unsafe extern "C" fn(*mut snd_info_entry, u16, *mut ()) -> i32>,
    read: Option<unsafe extern "C" fn(*mut snd_info_entry, *mut (), *mut file, *mut u8, u64, i64) -> i64>,
    write: Option<unsafe extern "C" fn(*mut snd_info_entry, *mut (), *mut file, *const u8, u64, i64) -> i64>,
    llseek: Option<unsafe extern "C" fn(*mut snd_info_entry, *mut (), *mut file, i64, i32) -> i64>,
    poll: Option<unsafe extern "C" fn(*mut snd_info_entry, *mut (), *mut file, *mut poll_table_struct) -> u32>,
    ioctl: Option<unsafe extern "C" fn(*mut snd_info_entry, *mut (), *mut file, u32, u64) -> i32>,
    mmap: Option<unsafe extern "C" fn(*mut snd_info_entry, *mut (), *mut inode, *mut file, *mut vm_area_struct) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_info_entry {
    name: *const u8,
    mode: u16,
    size: i64,
    content: u16,
    c: __SlateRecord_info_h_65,
    parent: *mut snd_info_entry,
    module: *mut module,
    private_data: *mut (),
    private_free: Option<unsafe extern "C" fn(*mut snd_info_entry)>,
    p: *mut proc_dir_entry,
    access: mutex,
    children: list_head,
    list: list_head,
}


// snd_iprintf - printf on the procfs buffer
// @buf: the procfs buffer
// @fmt: the printf format
//
// Outputs the string on the procfs buffer just like printf().
//
// Return: zero for success, or a negative error code.
#[repr(C)]
#[derive(Clone, Copy)]
union __snd_seq_event {
    legacy: snd_seq_event,
    raw: __SlateRecord_seq_memory_h_20,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_event_cell {
    __slate_anon_0: __SlateRecord_seq_memory_h_30,
    pool: *mut snd_seq_pool,
    next: *mut snd_seq_event_cell,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_pool {
    ptr: *mut snd_seq_event_cell,
    free: *mut snd_seq_event_cell,
    total_elements: i32,
    counter: __SlateRecord_types_h_190,
    size: i32,
    room: i32,
    closing: i32,
    max_used: i32,
    event_alloc_nopool: i32,
    event_alloc_failures: i32,
    event_alloc_success: i32,
    output_sleep: wait_queue_head,
    lock: spinlock,
}


fn snd_seq_unused_cells(pool: *mut snd_seq_pool) -> i32 {
let __v0: i32 = if !pool.is_null() {
unsafe { (*pool).total_elements }.wrapping_sub(atomic_read((unsafe { core::ptr::addr_of_mut!((*pool).counter) }) as *const __SlateRecord_types_h_190))
} else {
0
};
__v0
}

fn snd_seq_total_cells(pool: *mut snd_seq_pool) -> i32 {
if !pool.is_null() { unsafe { (*pool).total_elements } } else { 0 }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_fifo {
    pool: *mut snd_seq_pool,
    head: *mut snd_seq_event_cell,
    tail: *mut snd_seq_event_cell,
    cells: i32,
    lock: spinlock,
    use_lock: __SlateRecord_types_h_190,
    input_sleep: wait_queue_head,
    overflow: __SlateRecord_types_h_190,
}


#[inline(always)]
fn class_snd_seq_fifo_destructor(p: *mut *mut snd_seq_fifo) {
let _T: *mut snd_seq_fifo = unsafe { *p };
snd_seq_fifo_unlock!(_T);
}

#[inline(always)]
fn class_snd_seq_fifo_constructor(_T: *mut snd_seq_fifo) -> *mut snd_seq_fifo {
let mut __v0: *mut snd_seq_fifo;
snd_seq_fifo_lock!(_T);
__v0 = _T;
let t: *mut snd_seq_fifo = __v0;
t
}

#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_subscribers {
    info: snd_seq_port_subscribe,
    src_list: hlist_node,
    dest_list: hlist_node,
    ref_count: __SlateRecord_types_h_190,
    rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_port_subs_info {
    list_head: hlist_head,
    count: u32,
    __slate_bits_0: __slate_bits::__SlateBits753U0,
    list_mutex: rw_semaphore,
    open: Option<unsafe extern "C" fn(*mut (), *mut snd_seq_port_subscribe) -> i32>,
    close: Option<unsafe extern "C" fn(*mut (), *mut snd_seq_port_subscribe) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_client_port {
    addr: snd_seq_addr,
    owner: *mut module,
    name: [u8; 64],
    list: list_head,
    use_lock: __SlateRecord_types_h_190,
    c_src: snd_seq_port_subs_info,
    c_dest: snd_seq_port_subs_info,
    event_input: Option<unsafe extern "C" fn(*mut snd_seq_event, i32, *mut (), i32, i32) -> i32>,
    private_free: Option<unsafe extern "C" fn(*mut ())>,
    private_data: *mut (),
    __slate_bits_0: __slate_bits::__SlateBits754U0,
    time_queue: i32,
    capability: u32,
    r#type: u32,
    midi_channels: i32,
    midi_voices: i32,
    synth_voices: i32,
    direction: u8,
    ump_group: u8,
    is_midi1: bool,
}


#[inline(always)]
fn __free_snd_seq_port(p: *mut ()) {
let _T: *mut snd_seq_client_port = unsafe { *(p as *mut *mut snd_seq_client_port) };
if !IS_ERR_OR_NULL(_T as *const ()) {
snd_seq_port_unlock!(_T);
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_user_client {
    file: *mut file,
    owner: *mut pid,
    fifo: *mut snd_seq_fifo,
    fifo_pool_size: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_kernel_client {
    card: *mut snd_card,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_client {
    r#type: i32,
    __slate_bits_0: __slate_bits::__SlateBits755U0,
    midi_version: u32,
    user_pversion: u32,
    name: [u8; 64],
    number: i32,
    filter: u32,
    event_filter: [u64; 4],
    group_filter: u32,
    use_lock: __SlateRecord_types_h_190,
    event_lost: i32,
    num_ports: i32,
    ports_list_head: list_head,
    ports_mutex: mutex,
    ioctl_mutex: mutex,
    convert32: i32,
    ump_endpoint_port: i32,
    pool: *mut snd_seq_pool,
    data: __SlateRecord_seq_clientmgr_h_60,
    ump_info: *mut *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_usage {
    cur: i32,
    peak: i32,
}


fn snd_seq_client_ref(client: *mut snd_seq_client) -> *mut snd_seq_client {
snd_use_lock_use!(unsafe { core::ptr::addr_of_mut!((*client).use_lock) });
client
}

fn snd_seq_client_unref(client: *mut snd_seq_client) {
snd_use_lock_free!(unsafe { core::ptr::addr_of_mut!((*client).use_lock) });
}

#[inline(always)]
fn __free_snd_seq_client(p: *mut ()) {
let _T: *mut snd_seq_client = unsafe { *(p as *mut *mut snd_seq_client) };
if !IS_ERR_OR_NULL(_T as *const ()) {
snd_seq_client_unref(_T);
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_prioq {
    head: *mut snd_seq_event_cell,
    tail: *mut snd_seq_event_cell,
    cells: i32,
    lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_timer_hardware {
    flags: u32,
    resolution: u64,
    resolution_min: u64,
    resolution_max: u64,
    ticks: u64,
    open: Option<unsafe extern "C" fn(*mut snd_timer) -> i32>,
    close: Option<unsafe extern "C" fn(*mut snd_timer) -> i32>,
    c_resolution: Option<unsafe extern "C" fn(*mut snd_timer) -> u64>,
    start: Option<unsafe extern "C" fn(*mut snd_timer) -> i32>,
    stop: Option<unsafe extern "C" fn(*mut snd_timer) -> i32>,
    set_period: Option<unsafe extern "C" fn(*mut snd_timer, u64, u64) -> i32>,
    precise_resolution: Option<unsafe extern "C" fn(*mut snd_timer, *mut u64, *mut u64) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_timer {
    tmr_class: i32,
    card: *mut snd_card,
    module: *mut module,
    tmr_device: i32,
    tmr_subdevice: i32,
    id: [u8; 64],
    name: [u8; 80],
    flags: u32,
    running: i32,
    sticks: u64,
    private_data: *mut (),
    private_free: Option<unsafe extern "C" fn(*mut snd_timer)>,
    hw: snd_timer_hardware,
    lock: spinlock,
    device_list: list_head,
    open_list_head: list_head,
    active_list_head: list_head,
    ack_list_head: list_head,
    sack_list_head: list_head,
    task_work: work_struct,
    kref: kref,
    max_instances: i32,
    num_instances: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_timer_instance {
    timer: *mut snd_timer,
    owner: *mut u8,
    flags: u32,
    private_data: *mut (),
    private_free: Option<unsafe extern "C" fn(*mut snd_timer_instance)>,
    callback: Option<unsafe extern "C" fn(*mut snd_timer_instance, u64, u64)>,
    ccallback: Option<unsafe extern "C" fn(*mut snd_timer_instance, i32, *mut timespec64, u64)>,
    disconnect: Option<unsafe extern "C" fn(*mut snd_timer_instance)>,
    callback_data: *mut (),
    ticks: u64,
    cticks: u64,
    pticks: u64,
    resolution: u64,
    lost: u64,
    slave_class: i32,
    slave_id: u32,
    open_list: list_head,
    active_list: list_head,
    master_list: list_head,
    ack_list: list_head,
    slave_list_head: list_head,
    slave_active_head: list_head,
    master: *mut snd_timer_instance,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_timer_tick {
    cur_tick: u32,
    resolution: u64,
    fraction: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_timer {
    __slate_bits_0: __slate_bits::__SlateBits765U0,
    tempo: u32,
    ppq: i32,
    cur_time: snd_seq_real_time,
    tick: snd_seq_timer_tick,
    tick_updated: i32,
    r#type: i32,
    alsa_id: snd_timer_id,
    timeri: *mut snd_timer_instance,
    ticks: u32,
    preferred_resolution: u64,
    skew: u32,
    skew_base: u32,
    tempo_base: u32,
    last_update: timespec64,
    lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_queue {
    queue: i32,
    name: [u8; 64],
    tickq: *mut snd_seq_prioq,
    timeq: *mut snd_seq_prioq,
    timer: *mut snd_seq_timer,
    owner: i32,
    locked: bool,
    klocked: bool,
    check_again: bool,
    check_blocked: bool,
    flags: u32,
    info_flags: u32,
    owner_lock: spinlock,
    check_lock: spinlock,
    clients_bitmap: [u64; 3],
    clients: u32,
    timer_mutex: mutex,
    use_lock: __SlateRecord_types_h_190,
}


#[inline(always)]
fn __free_snd_seq_queue(p: *mut ()) {
let _T: *mut snd_seq_queue = unsafe { *(p as *mut *mut snd_seq_queue) };
if !IS_ERR_OR_NULL(_T as *const ()) {
queuefree!(_T);
}
}

// Client Manager
//
// this module handles the connections of userland and kernel clients
// There are four ranges of client numbers (last two shared):
// 0..15: global clients
// 16..127: statically allocated client numbers for cards 0..27
// 128..191: dynamically allocated client numbers for cards 28..31
// 128..191: dynamically allocated client numbers for applications
// number of kernel non-card clients
// clients per cards, for static clients
// dynamically allocated client numbers (both kernel drivers and user space)
static mut clients_lock: spinlock = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = unsafe { core::mem::zeroed() };
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0
} };

static mut register_mutex: mutex = {
let mut __t0: mutex = unsafe { core::mem::zeroed() };
__t0.owner = __SlateRecord_types_h_197 { counter: 0i64 };
__t0.wait_lock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0.first_waiter = core::ptr::null_mut::<mutex_waiter>();
__t0
};

/// client table
static mut clienttablock: __SlateAlign16<[u8; 192]> = __SlateAlign16(unsafe { core::mem::zeroed() });

static mut clienttab: __SlateAlign16<[*mut snd_seq_client; 192]> = __SlateAlign16(unsafe { core::mem::zeroed() });

static mut client_usage: snd_seq_usage = unsafe { core::mem::zeroed() };

fn snd_seq_file_flags(file: *mut file) -> u16 {
match (unsafe { (*file).f_mode }) & 3u32 {
2 => {
return 2;
}
1 => {
return 1;
}
_ => {
return 3u16;
}
}
unsafe { core::mem::zeroed() }
}

fn snd_seq_write_pool_allocated(client: *mut snd_seq_client) -> i32 {
(snd_seq_total_cells(unsafe { (*client).pool }) > 0) as i32
}

// return pointer to client structure for specified id; call under RCU read-lock
fn __clientptr(clientid: i32) -> *mut snd_seq_client {
if clientid < 0 || clientid >= 192 {
let mut __v0: i32;
if false {
unsafe { _printk(c"\x017ALSA: seq: oops. Trying to get pointer to client %d\n".as_ptr() as *const u8, clientid) };
}
__v0 = 0;
return core::ptr::null_mut::<snd_seq_client>();
}
let mut __v1: *mut snd_seq_client;
let mut __v2: *mut snd_seq_client;
loop {
if !(__native_word__11!() || ((const { core::mem::size_of::<*mut snd_seq_client>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_464() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = unsafe { core::ptr::read_volatile((core::ptr::addr_of_mut!(clienttab.0) as *mut *mut snd_seq_client).wrapping_offset(clientid as isize) as *const *mut snd_seq_client) };
let __UNIQUE_ID_rcu_463: *mut snd_seq_client = __v2;
loop {
let __v5: bool = if false {
let __v6: bool = (unsafe { lockdep_is_held(core::ptr::addr_of_mut!(clients_lock) as *const ()) }) != 0 || rcu_read_lock_held() != 0;
!__v6
} else {
false
};
if !__v5 {
break;
}
}
__v1 = __UNIQUE_ID_rcu_463;
__v1
}

/// return pointer to client structure for specified id
fn clientptr(clientid: i32) -> *mut snd_seq_client {
let mut __UNIQUE_ID_guard_465: __SlateRecord_rcupdate_h_1216 = class_rcu_constructor();
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_465) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_rcu_destructor(__slate_cleanup_ptr_0 as *mut _));
__clientptr(clientid)
}

fn client_use_ptr(clientid: i32, load_module: bool) -> *mut snd_seq_client {
let mut scope: core::mem::MaybeUninit<__SlateRecord_rcupdate_h_1216> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut __SlateRecord_rcupdate_h_1216 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_rcupdate_h_1216;
let mut idx: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut i32 = core::ptr::addr_of_mut!(idx) as *mut i32;
let mut card: i32 = unsafe { core::mem::zeroed() };
let mut scope: core::mem::MaybeUninit<__SlateRecord_rcupdate_h_1216> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut __SlateRecord_rcupdate_h_1216 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_rcupdate_h_1216;
let mut client: core::mem::MaybeUninit<*mut snd_seq_client> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut snd_seq_client = core::ptr::addr_of_mut!(client) as *mut *mut snd_seq_client;
let __slate_cleanup_ptr_5 = __slate_slot_5;
let __slate_cleanup_armed_5 = core::cell::Cell::new(false);
let __slate_cleanup_5 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_5.get() {
unsafe { class_rcu_destructor(__slate_cleanup_ptr_5 as *mut _) };
}
});
let __slate_cleanup_ptr_0 = __slate_slot_0;
let __slate_cleanup_armed_0 = core::cell::Cell::new(false);
let __slate_cleanup_0 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_0.get() {
unsafe { class_rcu_destructor(__slate_cleanup_ptr_0 as *mut _) };
}
});
unsafe {
let __slate_auto_bytes_1 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if clientid < 0 || clientid >= 192 {
return core::ptr::null_mut::<snd_seq_client>();
} else {
core::ptr::write(__slate_slot_5, class_rcu_constructor());
__slate_cleanup_armed_5.set(true);
if !class_rcu_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_5))).is_null() || !class_rcu_is_conditional {
if true {
*__slate_slot_7 = __clientptr(clientid);
if !(*__slate_slot_7).is_null() {
return snd_seq_client_ref(*__slate_slot_7);
} else {
if clienttablock.0[clientid as usize] != 0 {
return core::ptr::null_mut::<snd_seq_client>();
}
}
}
}
__slate_cleanup_armed_5.set(false);
class_rcu_destructor(__slate_slot_5 as *mut _);
if load_module {
'__join_5: {
if clientid < 16 {
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if !test_and_set_bit(clientid as i64, (core::ptr::addr_of_mut!(client_requested) as *mut u64) as *mut u64) {
*__slate_slot_3 = 0;
loop {
if *__slate_slot_3 < 15 {
if seq_client_load[*__slate_slot_3 as usize] < 0 {
break '__join_5;
} else {
if seq_client_load[*__slate_slot_3 as usize] == clientid {
break;
} else {
let __v2: i32 = *__slate_slot_3;
*__slate_slot_3 = __v2.wrapping_add(1);
}
}
} else {
break '__join_5;
}
}
__request_module(true, c"snd-seq-client-%i".as_ptr() as *const u8, clientid);
}
} else {
if clientid < 128 {
card = clientid.wrapping_sub(16) / 4;
if card < snd_ecards_limit {
if !test_and_set_bit(card as i64, (core::ptr::addr_of_mut!(card_requested) as *mut u64) as *mut u64) {
snd_request_card(card);
}
snd_seq_device_load_drivers();
}
}
}
}
core::ptr::write(__slate_slot_0, class_rcu_constructor());
__slate_cleanup_armed_0.set(true);
if !class_rcu_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_0))).is_null() || !class_rcu_is_conditional {
if true {
*__slate_slot_7 = __clientptr(clientid);
if !(*__slate_slot_7).is_null() {
return snd_seq_client_ref(*__slate_slot_7);
}
}
}
__slate_cleanup_armed_0.set(false);
class_rcu_destructor(__slate_slot_0 as *mut _);
}
return core::ptr::null_mut::<snd_seq_client>();
}
}
}

static mut client_requested: [u64; 1] = unsafe { core::mem::zeroed() };

static mut card_requested: [u64; 1] = unsafe { core::mem::zeroed() };

/// get snd_seq_client object for the given id quickly
#[unsafe(no_mangle)]
extern "C" fn snd_seq_client_use_ptr(clientid: i32) -> *mut snd_seq_client {
client_use_ptr(clientid, false)
}

/// get snd_seq_client object for the given id;
/// if not found, retry after loading the modules
fn client_load_and_use_ptr(clientid: i32) -> *mut snd_seq_client {
client_use_ptr(clientid, IS_ENABLED__2!() != 0)
}

fn usage_alloc(res: *mut snd_seq_usage, num: i32) {
let __v1: i32 = unsafe { (*res).cur };
unsafe {
(*res).cur = __v1.wrapping_add(num);
}
if (unsafe { (*res).cur }) > unsafe { (*res).peak } {
unsafe {
(*res).peak = (*res).cur;
}
}
}

fn usage_free(__v0: *mut snd_seq_usage, num: i32) {
let __v1: i32 = unsafe { (*__v0).cur };
unsafe {
(*__v0).cur = __v1.wrapping_sub(num);
}
}

/// initialise data structures
#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn client_init_data() -> i32 {
// zap out the client table
unsafe { memset(core::ptr::addr_of_mut!(clienttablock.0) as *mut (), 0i32, core::mem::size_of::<[u8; 192]>() as u64) };
unsafe { memset(core::ptr::addr_of_mut!(clienttab.0) as *mut (), 0i32, core::mem::size_of::<[*mut snd_seq_client; 192]>() as u64) };
0
}

fn seq_create_client1(client_index: i32, poolsize: i32) -> *mut snd_seq_client {
let mut __slate_compound_26: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_28: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _r_a_p__v: u64 = unsafe { core::mem::zeroed() };
let mut _r_a_p__v_1: u64 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_unlock_469: core::mem::MaybeUninit<*mut __SlateRecord_spinlock_h_620> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut __SlateRecord_spinlock_h_620 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_469) as *mut *mut __SlateRecord_spinlock_h_620;
// find free slot in the client table
let mut scope: core::mem::MaybeUninit<__SlateRecord_spinlock_h_620> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut __SlateRecord_spinlock_h_620 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_spinlock_h_620;
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut client: core::mem::MaybeUninit<*mut snd_seq_client> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut *mut snd_seq_client = core::ptr::addr_of_mut!(client) as *mut *mut snd_seq_client;
let mut c: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut i32 = core::ptr::addr_of_mut!(c) as *mut i32;
let __slate_cleanup_ptr_13 = __slate_slot_13;
let __slate_cleanup_armed_13 = core::cell::Cell::new(false);
let __slate_cleanup_13 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_13.get() {
unsafe { class_spinlock_irq_destructor(__slate_cleanup_ptr_13 as *mut _) };
}
});
let __slate_cleanup_ptr_12 = __slate_slot_12;
let __slate_cleanup_armed_12 = core::cell::Cell::new(false);
let __slate_cleanup_12 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_12.get() {
unsafe { __class_spinlock_irq_cleanup_ctx(__slate_cleanup_ptr_12 as *mut _) };
}
});
unsafe {
let __slate_auto_bytes_1 = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
__obj_size = size_mul(core::mem::size_of::<snd_seq_client>() as u64, 1u64);
let __slate_auto_bytes_3 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_14 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_15 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_26 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_26)
});
alloc_tag_restore!();
} else {
*__slate_slot_15 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_28 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_28)
});
}
let __v17: *mut () = *__slate_slot_15;
// init client data
*__slate_slot_20 = __v17 as *mut snd_seq_client;
if (*__slate_slot_20).is_null() {
return core::ptr::null_mut::<snd_seq_client>();
} else {
(*(*__slate_slot_20)).pool = snd_seq_pool_new(poolsize);
if unsafe { (*(*__slate_slot_20)).pool }.is_null() {
kfree(*__slate_slot_20 as *const ());
return core::ptr::null_mut::<snd_seq_client>();
} else {
(*(*__slate_slot_20)).r#type = 0;
snd_use_lock_init!(core::ptr::addr_of_mut!((*(*__slate_slot_20)).use_lock));
loop {
__mutex_init(core::ptr::addr_of_mut!((*(*__slate_slot_20)).ports_mutex), c"&client->ports_mutex".as_ptr() as *const u8, core::ptr::addr_of_mut!(__key__223));
if true {
break;
}
}
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*(*__slate_slot_20)).ports_list_head));
loop {
__mutex_init(core::ptr::addr_of_mut!((*(*__slate_slot_20)).ioctl_mutex), c"&client->ioctl_mutex".as_ptr() as *const u8, core::ptr::addr_of_mut!(__key__225));
if true {
break;
}
}
'__join_0: {
(*(*__slate_slot_20)).ump_endpoint_port = 1i32.wrapping_neg();
core::ptr::write(__slate_slot_13, class_spinlock_irq_constructor(core::ptr::addr_of_mut!(clients_lock)));
__slate_cleanup_armed_13.set(true);
core::ptr::write(__slate_slot_12, (core::ptr::addr_of_mut!(clients_lock) as u64) as *mut __SlateRecord_spinlock_h_620);
__slate_cleanup_armed_12.set(true);
if !class_spinlock_irq_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_13))).is_null() || !class_spinlock_irq_is_conditional {
if true {
if client_index < 0 {
*__slate_slot_21 = 128;
loop {
if *__slate_slot_21 < 192 {
while false {
}
let __v3: *mut snd_seq_client = core::ptr::read_volatile((core::ptr::addr_of_mut!(clienttab.0) as *mut *mut snd_seq_client).wrapping_offset(*__slate_slot_21 as isize) as *const *mut snd_seq_client);
let __UNIQUE_ID_rcu_470: *mut snd_seq_client = __v3;
let __v6: *mut snd_seq_client = __UNIQUE_ID_rcu_470;
if !__v6.is_null() || clienttablock.0[*__slate_slot_21 as usize] != 0 {
let __v1: i32 = *__slate_slot_21;
*__slate_slot_21 = __v1.wrapping_add(1);
} else {
break;
}
} else {
break '__join_0;
}
}
(*(*__slate_slot_20)).number = *__slate_slot_21;
_r_a_p__v = *__slate_slot_20 as u64;
loop {
kcsan_release!();
loop {
if true {
core::arch::asm!("", options(nostack, att_syntax));
loop {
if true {
__WRITE_ONCE__5!(*(core::ptr::addr_of_mut!(clienttab.0) as *mut *mut snd_seq_client).wrapping_offset(*__slate_slot_21 as isize), RCU_INITIALIZER!(_r_a_p__v as *mut snd_seq_client));
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
return *__slate_slot_20;
} else {
while false {
}
let __v8: *mut snd_seq_client = core::ptr::read_volatile((core::ptr::addr_of_mut!(clienttab.0) as *mut *mut snd_seq_client).wrapping_offset(client_index as isize) as *const *mut snd_seq_client);
let __UNIQUE_ID_rcu_475: *mut snd_seq_client = __v8;
let __v11: *mut snd_seq_client = __UNIQUE_ID_rcu_475;
if __v11.is_null() && clienttablock.0[client_index as usize] == 0 {
(*(*__slate_slot_20)).number = client_index;
_r_a_p__v_1 = *__slate_slot_20 as u64;
loop {
kcsan_release!();
loop {
if true {
core::arch::asm!("", options(nostack, att_syntax));
loop {
if true {
__WRITE_ONCE__5!(*(core::ptr::addr_of_mut!(clienttab.0) as *mut *mut snd_seq_client).wrapping_offset(client_index as isize), RCU_INITIALIZER!(_r_a_p__v_1 as *mut snd_seq_client));
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
return *__slate_slot_20;
}
}
}
}
}
__slate_cleanup_armed_12.set(false);
__class_spinlock_irq_cleanup_ctx(__slate_slot_12 as *mut _);
__slate_cleanup_armed_13.set(false);
class_spinlock_irq_destructor(__slate_slot_13 as *mut _);
snd_seq_pool_delete(core::ptr::addr_of_mut!((*(*__slate_slot_20)).pool));
kfree(*__slate_slot_20 as *const ());
return core::ptr::null_mut::<snd_seq_client>();
}
}
}
// no free slot found or busy, return failure code
}

static mut __key__223: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__225: lock_class_key = unsafe { core::mem::zeroed() };

fn seq_free_client1(client: *mut snd_seq_client) -> i32 {
let mut __UNIQUE_ID_unlock_486: core::mem::MaybeUninit<*mut __SlateRecord_spinlock_h_620> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut __SlateRecord_spinlock_h_620 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_486) as *mut *mut __SlateRecord_spinlock_h_620;
let mut scope: core::mem::MaybeUninit<__SlateRecord_spinlock_h_620> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut __SlateRecord_spinlock_h_620 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_spinlock_h_620;
let mut _r_a_p__v: u64 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_unlock_481: core::mem::MaybeUninit<*mut __SlateRecord_spinlock_h_620> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut __SlateRecord_spinlock_h_620 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_481) as *mut *mut __SlateRecord_spinlock_h_620;
let mut scope: core::mem::MaybeUninit<__SlateRecord_spinlock_h_620> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut __SlateRecord_spinlock_h_620 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_spinlock_h_620;
let __slate_cleanup_ptr_4 = __slate_slot_4;
let __slate_cleanup_armed_4 = core::cell::Cell::new(false);
let __slate_cleanup_4 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_4.get() {
unsafe { class_spinlock_irq_destructor(__slate_cleanup_ptr_4 as *mut _) };
}
});
let __slate_cleanup_ptr_3 = __slate_slot_3;
let __slate_cleanup_armed_3 = core::cell::Cell::new(false);
let __slate_cleanup_3 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_3.get() {
unsafe { __class_spinlock_irq_cleanup_ctx(__slate_cleanup_ptr_3 as *mut _) };
}
});
let __slate_cleanup_ptr_1 = __slate_slot_1;
let __slate_cleanup_armed_1 = core::cell::Cell::new(false);
let __slate_cleanup_1 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_1.get() {
unsafe { class_spinlock_irq_destructor(__slate_cleanup_ptr_1 as *mut _) };
}
});
let __slate_cleanup_ptr_0 = __slate_slot_0;
let __slate_cleanup_armed_0 = core::cell::Cell::new(false);
let __slate_cleanup_0 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_0.get() {
unsafe { __class_spinlock_irq_cleanup_ctx(__slate_cleanup_ptr_0 as *mut _) };
}
});
unsafe {
if client.is_null() {
return 0;
} else {
core::ptr::write(__slate_slot_4, class_spinlock_irq_constructor(core::ptr::addr_of_mut!(clients_lock)));
__slate_cleanup_armed_4.set(true);
core::ptr::write(__slate_slot_3, (core::ptr::addr_of_mut!(clients_lock) as u64) as *mut __SlateRecord_spinlock_h_620);
__slate_cleanup_armed_3.set(true);
if !class_spinlock_irq_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_4))).is_null() || !class_spinlock_irq_is_conditional {
if true {
clienttablock.0[(*client).number as usize] = 1;
_r_a_p__v = 0;
if true && _r_a_p__v == 0 {
loop {
if true {
__WRITE_ONCE__5!(clienttab.0[(*client).number as usize], _r_a_p__v as *mut snd_seq_client);
if true {
break;
}
}
}
} else {
loop {
kcsan_release!();
loop {
if true {
core::arch::asm!("", options(nostack, att_syntax));
loop {
if true {
__WRITE_ONCE__5!(*(core::ptr::addr_of_mut!(clienttab.0) as *mut *mut snd_seq_client).wrapping_offset((*client).number as isize), RCU_INITIALIZER!(_r_a_p__v as *mut snd_seq_client));
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
}
}
}
__slate_cleanup_armed_3.set(false);
__class_spinlock_irq_cleanup_ctx(__slate_slot_3 as *mut _);
__slate_cleanup_armed_4.set(false);
class_spinlock_irq_destructor(__slate_slot_4 as *mut _);
snd_seq_delete_all_ports(client);
snd_seq_queue_client_leave((*client).number);
// the client has been unpublished from the table; wait for a grace
// period so that lockless readers (snd_seq_client_use_ptr()) that
// observed the old pointer can no longer take a new use_lock
// reference, then drain the outstanding references before freeing
synchronize_rcu();
snd_use_lock_sync!(core::ptr::addr_of_mut!((*client).use_lock));
if !unsafe { (*client).pool }.is_null() {
snd_seq_pool_delete(core::ptr::addr_of_mut!((*client).pool));
}
core::ptr::write(__slate_slot_1, class_spinlock_irq_constructor(core::ptr::addr_of_mut!(clients_lock)));
__slate_cleanup_armed_1.set(true);
core::ptr::write(__slate_slot_0, (core::ptr::addr_of_mut!(clients_lock) as u64) as *mut __SlateRecord_spinlock_h_620);
__slate_cleanup_armed_0.set(true);
if !class_spinlock_irq_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_1))).is_null() || !class_spinlock_irq_is_conditional {
if true {
clienttablock.0[(*client).number as usize] = 0;
}
}
__slate_cleanup_armed_0.set(false);
__class_spinlock_irq_cleanup_ctx(__slate_slot_0 as *mut _);
__slate_cleanup_armed_1.set(false);
class_spinlock_irq_destructor(__slate_slot_1 as *mut _);
return 0;
}
}
}

fn seq_free_client(client: *mut snd_seq_client) {
let mut __UNIQUE_ID_unlock_488: core::mem::MaybeUninit<*mut __SlateRecord_mutex_h_253> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut __SlateRecord_mutex_h_253 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_488) as *mut *mut __SlateRecord_mutex_h_253;
let mut scope: core::mem::MaybeUninit<__SlateRecord_mutex_h_253> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut __SlateRecord_mutex_h_253 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_mutex_h_253;
let __slate_cleanup_ptr_3 = __slate_slot_3;
let __slate_cleanup_armed_3 = core::cell::Cell::new(false);
let __slate_cleanup_3 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_3.get() {
unsafe { class_mutex_destructor(__slate_cleanup_ptr_3 as *mut _) };
}
});
let __slate_cleanup_ptr_2 = __slate_slot_2;
let __slate_cleanup_armed_2 = core::cell::Cell::new(false);
let __slate_cleanup_2 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_2.get() {
unsafe { __class_mutex_cleanup_ctx(__slate_cleanup_ptr_2 as *mut _) };
}
});
unsafe {
'__join_0: {
core::ptr::write(__slate_slot_3, class_mutex_constructor(core::ptr::addr_of_mut!(register_mutex)));
__slate_cleanup_armed_3.set(true);
core::ptr::write(__slate_slot_2, (core::ptr::addr_of_mut!(register_mutex) as u64) as *mut __SlateRecord_mutex_h_253);
__slate_cleanup_armed_2.set(true);
if !class_mutex_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_3))).is_null() || !class_mutex_is_conditional {
if true {
let __t0: i32 = (*client).r#type;
if __t0 == 0 {
while false {
}
_printk(c"\x014ALSA: seq: Trying to free unused client %d\n".as_ptr() as *const u8, (*client).number);
break '__join_0;
} else {
if __t0 != 1 {
if __t0 != 2 {
while false {
}
_printk(c"\x013ALSA: seq: Trying to free client %d with undefined type = %d\n".as_ptr() as *const u8, (*client).number, (*client).r#type);
break '__join_0;
}
}
}
seq_free_client1(client);
usage_free(core::ptr::addr_of_mut!(client_usage), 1i32);
}
}
}
__slate_cleanup_armed_2.set(false);
__class_mutex_cleanup_ctx(__slate_slot_2 as *mut _);
__slate_cleanup_armed_3.set(false);
class_mutex_destructor(__slate_slot_3 as *mut _);
snd_seq_system_client_ev_client_exit!(unsafe { (*client).number });
}
}

/// create a user client
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_open")]
extern "C" fn snd_seq_open(inode: *mut inode, file: *mut file) -> i32 {
let mut __UNIQUE_ID_unlock_490: core::mem::MaybeUninit<*mut __SlateRecord_mutex_h_253> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut __SlateRecord_mutex_h_253 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_490) as *mut *mut __SlateRecord_mutex_h_253;
let mut scope: core::mem::MaybeUninit<__SlateRecord_mutex_h_253> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut __SlateRecord_mutex_h_253 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_mutex_h_253;
let mut user: core::mem::MaybeUninit<*mut snd_seq_user_client> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut snd_seq_user_client = core::ptr::addr_of_mut!(user) as *mut *mut snd_seq_user_client;
let mut client: core::mem::MaybeUninit<*mut snd_seq_client> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut snd_seq_client = core::ptr::addr_of_mut!(client) as *mut *mut snd_seq_client; // client id
let mut mode: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(mode) as *mut i32;
let mut c: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i32 = core::ptr::addr_of_mut!(c) as *mut i32;
let __slate_cleanup_ptr_1 = __slate_slot_1;
let __slate_cleanup_armed_1 = core::cell::Cell::new(false);
let __slate_cleanup_1 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_1.get() {
unsafe { class_mutex_destructor(__slate_cleanup_ptr_1 as *mut _) };
}
});
let __slate_cleanup_ptr_0 = __slate_slot_0;
let __slate_cleanup_armed_0 = core::cell::Cell::new(false);
let __slate_cleanup_0 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_0.get() {
unsafe { __class_mutex_cleanup_ctx(__slate_cleanup_ptr_0 as *mut _) };
}
});
unsafe {
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
stream_open(inode, file);
core::ptr::write(__slate_slot_1, class_mutex_constructor(core::ptr::addr_of_mut!(register_mutex)));
__slate_cleanup_armed_1.set(true);
core::ptr::write(__slate_slot_0, (core::ptr::addr_of_mut!(register_mutex) as u64) as *mut __SlateRecord_mutex_h_253);
__slate_cleanup_armed_0.set(true);
if !class_mutex_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_1))).is_null() || !class_mutex_is_conditional {
if true {
*__slate_slot_3 = seq_create_client1(1i32.wrapping_neg(), 500i32);
if (*__slate_slot_3).is_null() {
return 12i32.wrapping_neg();
} else {
// failure code
*__slate_slot_4 = snd_seq_file_flags(file) as i32;
if *__slate_slot_4 & 1 != 0 {
(*(*__slate_slot_3)).__slate_bits_0.__set_accept_input(1u32);
}
if *__slate_slot_4 & 2 != 0 {
(*(*__slate_slot_3)).__slate_bits_0.__set_accept_output(1u32);
}
*__slate_slot_2 = core::ptr::addr_of_mut!((*(*__slate_slot_3)).data.user);
(*(*__slate_slot_2)).fifo = core::ptr::null_mut::<snd_seq_fifo>();
(*(*__slate_slot_2)).fifo_pool_size = 0;
if *__slate_slot_4 & 1 != 0 {
(*(*__slate_slot_2)).fifo_pool_size = 200;
(*(*__slate_slot_2)).fifo = snd_seq_fifo_new((*(*__slate_slot_2)).fifo_pool_size);
if unsafe { (*(*__slate_slot_2)).fifo }.is_null() {
seq_free_client1(*__slate_slot_3);
kfree(*__slate_slot_3 as *const ());
return 12i32.wrapping_neg();
}
}
usage_alloc(core::ptr::addr_of_mut!(client_usage), 1i32);
(*(*__slate_slot_3)).r#type = 1;
}
}
}
__slate_cleanup_armed_0.set(false);
__class_mutex_cleanup_ctx(__slate_slot_0 as *mut _);
__slate_cleanup_armed_1.set(false);
class_mutex_destructor(__slate_slot_1 as *mut _);
*__slate_slot_5 = (*(*__slate_slot_3)).number;
(*file).private_data = *__slate_slot_3 as *mut ();
// fill client data
(*(*__slate_slot_2)).file = file;
sprintf((*(*__slate_slot_3)).name.as_mut_ptr() as *mut u8, c"Client-%d".as_ptr() as *const u8, *__slate_slot_5);
(*(*__slate_slot_3)).data.user.owner = get_pid(task_pid(get_current()));
// make others aware this new client
snd_seq_system_client_ev_client_start!(*__slate_slot_5);
return 0;
}
}

/// delete a user client
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_release")]
extern "C" fn snd_seq_release(inode: *mut inode, file: *mut file) -> i32 {
let client: *mut snd_seq_client = (unsafe { (*file).private_data }) as *mut snd_seq_client;
if !client.is_null() {
seq_free_client(client);
if !unsafe { (*client).data.user.fifo }.is_null() {
unsafe { snd_seq_fifo_delete(core::ptr::addr_of_mut!((*client).data.user.fifo)) };
}
unsafe { put_pid((*client).data.user.owner) };
unsafe { kfree(client as *const ()) };
}
0
}

fn event_is_compatible(client: *const snd_seq_client, ev: *const snd_seq_event) -> bool {
if snd_seq_ev_is_ump!(ev) && (unsafe { (*client).midi_version }) == 0 {
return false;
}
if snd_seq_ev_is_ump!(ev) && snd_seq_ev_is_variable!(ev) {
return false;
}
true
}

/// handle client read()
///
/// possible error values:
/// -ENXIO	invalid client or file open mode
/// -ENOSPC	FIFO overflow (the flag is cleared after this error report)
/// -EINVAL	no enough user-space buffer to write the whole event
/// -EFAULT	seg. fault during copy to user space
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_read")]
extern "C" fn snd_seq_read(file: *mut file, mut buf: *mut u8, mut count: u64, offset: *mut i64) -> i64 {
let client: *mut snd_seq_client = (unsafe { (*file).private_data }) as *mut snd_seq_client;
let mut fifo: *mut snd_seq_fifo = unsafe { core::mem::zeroed() };
let mut aligned_size: u64 = 0;
let mut err: i32 = 0;
let mut result: i64 = 0;
let mut cell: *mut snd_seq_event_cell = unsafe { core::mem::zeroed() };
if (snd_seq_file_flags(file) as i32) & 1 == 0 {
return 6i32.wrapping_neg() as i64;
}
if access_ok!(buf, count) == 0 {
return 14i32.wrapping_neg() as i64;
}
// check client structures are in place
if snd_BUG_ON!(client.is_null()) != 0 {
return 6i32.wrapping_neg() as i64;
}
if ((unsafe { (*client).__slate_bits_0.__get_accept_input() }) as i32) == 0 {
return 6i32.wrapping_neg() as i64;
}
fifo = unsafe { (*client).data.user.fifo };
if fifo.is_null() {
return 6i32.wrapping_neg() as i64;
}
if atomic_read((unsafe { core::ptr::addr_of_mut!((*fifo).overflow) }) as *const __SlateRecord_types_h_190) > 0 {
// buffer overflow is detected
unsafe { snd_seq_fifo_clear(fifo) };
// return error code
return 28i32.wrapping_neg() as i64;
}
cell = core::ptr::null_mut::<snd_seq_event_cell>();
let mut __UNIQUE_ID_guard_491: *mut snd_seq_fifo = class_snd_seq_fifo_constructor(fifo);
let __slate_cleanup_ptr_2 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_491) as *mut _;
let __slate_cleanup_2 = ::slate_support::Guard(&mut || class_snd_seq_fifo_destructor(__slate_cleanup_ptr_2 as *mut _));
if IS_ENABLED!() != 0 && (unsafe { (*client).midi_version }) > 0 {
aligned_size = core::mem::size_of::<snd_seq_ump_event>() as u64;
} else {
aligned_size = core::mem::size_of::<snd_seq_event>() as u64;
}
// while data available in queue
while count >= aligned_size {
let nonblock: i32 = ((unsafe { (*file).f_flags }) & 2048u32 != 0 || result > 0) as i32;
err = unsafe { snd_seq_fifo_cell_out(fifo, core::ptr::addr_of_mut!(cell), nonblock) };
if err < 0 {
break;
}
if !event_is_compatible(client as *const snd_seq_client, (unsafe { core::ptr::addr_of_mut!((*cell).__slate_anon_0.event) }) as *const snd_seq_event) {
unsafe { snd_seq_cell_free(cell) };
cell = core::ptr::null_mut::<snd_seq_event_cell>();
} else {
if snd_seq_ev_is_variable!(unsafe { core::ptr::addr_of_mut!((*cell).__slate_anon_0.event) }) {
let mut tmpev: snd_seq_ump_event = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(tmpev) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 32usize);
}
unsafe { memcpy(core::ptr::addr_of_mut!(tmpev) as *mut (), core::ptr::addr_of_mut!((*cell).__slate_anon_0.event) as *const (), aligned_size) };
let __v5: u32 = (unsafe { tmpev.__slate_anon_7.data.ext.len }) & !3221225472;
unsafe {
tmpev.__slate_anon_7.data.ext.len = __v5;
}
unsafe {
tmpev.__slate_anon_7.data.ext.ptr = core::ptr::null_mut::<()>();
}
if copy_to_user(buf as *mut (), core::ptr::addr_of_mut!(tmpev) as *const (), aligned_size) != 0 {
err = 14i32.wrapping_neg();
break;
}
count = count.wrapping_sub(aligned_size);
let __v8: *mut u8 = buf;
buf = __v8.wrapping_offset(aligned_size as isize);
err = unsafe { snd_seq_expand_var_event(core::ptr::addr_of_mut!((*cell).__slate_anon_0.event) as *const snd_seq_event, count as i32, buf, 0i32, aligned_size as i32) };
if err < 0 {
break;
}
result = result.wrapping_add(err as i64);
count = count.wrapping_sub(err as u64);
let __v14: *mut u8 = buf;
buf = __v14.wrapping_offset(err as isize);
} else {
if copy_to_user(buf as *mut (), (unsafe { core::ptr::addr_of_mut!((*cell).__slate_anon_0.event) }) as *const (), aligned_size) != 0 {
err = 14i32.wrapping_neg();
break;
}
count = count.wrapping_sub(aligned_size);
let __v18: *mut u8 = buf;
buf = __v18.wrapping_offset(aligned_size as isize);
}
unsafe { snd_seq_cell_free(cell) };
cell = core::ptr::null_mut::<snd_seq_event_cell>(); // to be sure
result = (result as u64).wrapping_add(aligned_size) as i64;
}
}
if err < 0 {
if !cell.is_null() {
unsafe { snd_seq_fifo_cell_putback(fifo, cell) };
}
if err == 11i32.wrapping_neg() && result > 0 {
err = 0;
}
}
if err < 0 { err as i64 } else { result }
}

/// check access permission to the port
fn check_port_perm(port: *mut snd_seq_client_port, flags: u32) -> i32 {
if (unsafe { (*port).capability }) & flags != flags {
return 0;
}
flags as i32
}

/// check if the destination client is available, and return the pointer
fn get_event_dest_client(event: *mut snd_seq_event) -> *mut snd_seq_client {
let mut dest: *mut snd_seq_client = snd_seq_client_use_ptr((unsafe { (*event).dest.client }) as i32);
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(dest) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || __free_snd_seq_client(__slate_cleanup_ptr_0 as *mut _));
if dest.is_null() {
return core::ptr::null_mut::<snd_seq_client>();
}
if ((unsafe { (*dest).__slate_bits_0.__get_accept_input() }) as i32) == 0 {
return core::ptr::null_mut::<snd_seq_client>();
}
if snd_seq_ev_is_ump!(event) {
return no_free_ptr!(dest);
}
// ok - no filter checks
let __v4: bool = (unsafe { (*dest).filter }) & 2147483648u32 != 0 && test_bit__3!(unsafe { (*event).r#type }, unsafe { (*dest).event_filter.as_mut_ptr() as *mut u64 }) == 0;
if __v4 {
return core::ptr::null_mut::<snd_seq_client>();
}
return no_free_ptr!(dest); // ok - accessible
}

/// Return the error event.
///
/// If the receiver client is a user client, the original event is
/// encapsulated in SNDRV_SEQ_EVENT_BOUNCE as variable length event.  The
/// external data of a variable length event is not copied along.
/// If the receiver client is a kernel client, the original event is
/// quoted in SNDRV_SEQ_EVENT_KERNEL_ERROR, since this requires no extra
/// kmalloc.
fn bounce_error_event(client: *mut snd_seq_client, event: *mut snd_seq_event, err: i32, atomic: i32, hop: i32) -> i32 {
let mut bounce_ev: snd_seq_event = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(bounce_ev) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 28usize);
}
let mut quoted: snd_seq_event = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(quoted) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 28usize);
}
let mut result: i32 = 0;
if client.is_null() || (unsafe { (*client).filter }) & 4u32 == 0 || ((unsafe { (*client).__slate_bits_0.__get_accept_input() }) as i32) == 0 {
return 0;
}
// ignored
if ((unsafe { (*event).r#type }) as i32) == 131 || ((unsafe { (*event).r#type }) as i32) == 150 {
return err;
}
// avoid re-bouncing
// set up quoted error
unsafe { memset(core::ptr::addr_of_mut!(bounce_ev) as *mut (), 0i32, core::mem::size_of::<snd_seq_event>() as u64) };
if (unsafe { (*client).r#type }) == 1 {
// For user clients, send SNDRV_SEQ_EVENT_BOUNCE with the
// original event embedded as variable-length data.  This
// avoids exposing data.quote.event (a kernel pointer) to
// userspace.  Sanitise the embedded copy too - a queued
// variable-length event carries the address of its own
// extension cell, and the payload goes out verbatim.
quoted = unsafe { *event };
if snd_seq_ev_is_variable!(core::ptr::addr_of_mut!(quoted)) {
let __v1: u32 = (unsafe { quoted.data.ext.len }) & !3221225472;
unsafe {
quoted.data.ext.len = __v1;
}
unsafe {
quoted.data.ext.ptr = core::ptr::null_mut::<()>();
}
}
bounce_ev.r#type = 131;
bounce_ev.flags = 4u8;
unsafe {
bounce_ev.data.ext.len = core::mem::size_of::<snd_seq_event>() as u32;
}
unsafe {
bounce_ev.data.ext.ptr = (core::ptr::addr_of_mut!(quoted) as *mut u8) as *mut ();
}
} else {
// For kernel clients, quote the event pointer directly.
// Kernel consumers can safely dereference the pointer.
bounce_ev.r#type = 150;
bounce_ev.flags = 0u8;
unsafe {
bounce_ev.data.quote.origin = (*event).dest;
}
unsafe {
bounce_ev.data.quote.event = event;
}
unsafe {
bounce_ev.data.quote.value = err.wrapping_neg() as u16;
}
// use positive value
}
bounce_ev.queue = 253;
bounce_ev.source.client = 0;
bounce_ev.source.port = 1;
bounce_ev.dest.client = (unsafe { (*client).number }) as u8;
bounce_ev.dest.port = unsafe { (*event).source.port };
result = snd_seq_deliver_single_event(core::ptr::null_mut::<snd_seq_client>(), core::ptr::addr_of_mut!(bounce_ev), atomic, hop.wrapping_add(1));
if result < 0 {
let __v3: i32 = unsafe { (*client).event_lost };
unsafe {
(*client).event_lost = __v3.wrapping_add(1);
}
return result;
}
result
}

/// rewrite the time-stamp of the event record with the curren time
/// of the given queue.
/// return non-zero if updated.
fn update_timestamp_of_queue(event: *mut snd_seq_event, queue: i32, real_time: i32) -> i32 {
let mut q: *mut snd_seq_queue = unsafe { queueptr(queue) };
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(q) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || __free_snd_seq_queue(__slate_cleanup_ptr_0 as *mut _));
if q.is_null() {
return 0;
}
unsafe {
(*event).queue = queue as u8;
}
let __v3: u8 = (((unsafe { (*event).flags }) as i32) & -2i32) as u8;
unsafe {
(*event).flags = __v3;
}
if real_time != 0 {
unsafe {
(*event).time.time = snd_seq_timer_get_cur_time((*q).timer, true);
}
unsafe {
(*event).flags = (((*event).flags as i32) | 1i32) as u8;
}
} else {
unsafe {
(*event).time.tick = snd_seq_timer_get_cur_tick((*q).timer);
}
unsafe {
(*event).flags = (((*event).flags as i32) | 0i32) as u8;
}
}
1
}

/// deliver a single event; called from below and UMP converter
#[unsafe(no_mangle)]
extern "C" fn __snd_seq_deliver_single_event(dest: *mut snd_seq_client, dest_port: *mut snd_seq_client_port, event: *mut snd_seq_event, atomic: i32, hop: i32) -> i32 {
match unsafe { (*dest).r#type } {
1 => {
if unsafe { (*dest).data.user.fifo }.is_null() {
return 0;
}
return unsafe { snd_seq_fifo_event_in((*dest).data.user.fifo, event) };
}
2 => {
if (unsafe { (*dest_port).event_input }) == None {
return 0;
}
return unsafe { unsafe { (*dest_port).event_input }.unwrap()(event, snd_seq_ev_is_direct!(event) as i32, (*dest_port).private_data, atomic, hop) };
}
_ => {
}
}
0
}

/// deliver a single event; called from snd_seq_deliver_single_event()
fn _snd_seq_deliver_single_event(client: *mut snd_seq_client, event: *mut snd_seq_event, atomic: i32, hop: i32) -> i32 {
let mut dest: *mut snd_seq_client = get_event_dest_client(event);
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(dest) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || __free_snd_seq_client(__slate_cleanup_ptr_0 as *mut _));
if dest.is_null() {
return 2i32.wrapping_neg();
}
let mut dest_port: *mut snd_seq_client_port = unsafe { snd_seq_port_use_ptr(dest, (*event).dest.port as i32) };
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(dest_port) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __free_snd_seq_port(__slate_cleanup_ptr_1 as *mut _));
if dest_port.is_null() {
return 2i32.wrapping_neg();
}
// check permission
if check_port_perm(dest_port, 2u32) == 0 {
return 1i32.wrapping_neg();
}
if ((unsafe { (*dest_port).__slate_bits_0.__get_timestamping() }) as i32) != 0 {
update_timestamp_of_queue(event, unsafe { (*dest_port).time_queue }, (unsafe { (*dest_port).__slate_bits_0.__get_time_real() }) as i32);
}
__snd_seq_deliver_single_event(dest, dest_port, event, atomic, hop)
}

/// deliver an event to the specified destination.
/// if filter is non-zero, client filter bitmap is tested.
///
///  RETURN VALUE: 0 : if succeeded
/// 		 <0 : error
fn snd_seq_deliver_single_event(client: *mut snd_seq_client, event: *mut snd_seq_event, atomic: i32, hop: i32) -> i32 {
let result: i32 = _snd_seq_deliver_single_event(client, event, atomic, hop);
if result < 0 && !snd_seq_ev_is_direct!(event) {
return bounce_error_event(client, event, result, atomic, hop);
}
result
}

// send the event to all subscribers:
fn __deliver_to_subscribers(client: *mut snd_seq_client, event: *mut snd_seq_event, port: i32, atomic: i32, hop: i32) -> i32 {
let mut subs: *mut snd_seq_subscribers = unsafe { core::mem::zeroed() };
let mut err: i32 = 0;
let mut result: i32 = 0;
let mut num_ev: i32 = 0;
let mut event_saved: __snd_seq_event = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(event_saved) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 28usize);
}
let mut saved_size: u64 = 0;
let mut grp: *mut snd_seq_port_subs_info = unsafe { core::mem::zeroed() };
if port < 0 {
return 0;
}
let mut src_port: *mut snd_seq_client_port = unsafe { snd_seq_port_use_ptr(client, port) };
let __slate_cleanup_ptr_2 = core::ptr::addr_of_mut!(src_port) as *mut _;
let __slate_cleanup_2 = ::slate_support::Guard(&mut || __free_snd_seq_port(__slate_cleanup_ptr_2 as *mut _));
if src_port.is_null() {
return 0;
}
// save original event record
saved_size = snd_seq_event_packet_size(event);
unsafe { memcpy(core::ptr::addr_of_mut!(event_saved) as *mut (), event as *const (), saved_size) };
grp = unsafe { core::ptr::addr_of_mut!((*src_port).c_src) };
// lock list
if atomic != 0 {
rcu_read_lock();
} else {
down_read_nested!(unsafe { core::ptr::addr_of_mut!((*grp).list_mutex) });
}
let mut __v3: *mut snd_seq_subscribers;
let mut __v4: *mut hlist_node;
let mut __v5: *mut hlist_node;
loop {
if !(((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64) || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_493() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v5 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*grp).list_head)).first) as *const *mut hlist_node) };
let __UNIQUE_ID_rcu_492: *mut hlist_node = __v5;
__v4 = __UNIQUE_ID_rcu_492;
let ____ptr: *mut hlist_node = __v4;
let __v7: *mut snd_seq_subscribers = if !____ptr.is_null() {
hlist_entry__2!(____ptr)
} else {
core::ptr::null_mut::<snd_seq_subscribers>()
};
__v3 = __v7;
subs = __v3;
while !subs.is_null() {
// both ports ready?
if atomic_read((unsafe { core::ptr::addr_of_mut!((*subs).ref_count) }) as *const __SlateRecord_types_h_190) == 2 {
unsafe {
(*event).dest = (*subs).info.dest;
}
if (unsafe { (*subs).info.flags }) & 2u32 != 0 {
update_timestamp_of_queue(event, (unsafe { (*subs).info.queue }) as i32, ((unsafe { (*subs).info.flags }) & 4u32) as i32);
}
// convert time according to flag with subscription
err = snd_seq_deliver_single_event(client, event, atomic, hop);
if err < 0 {
// save first error that occurs and continue
if result == 0 {
result = err;
}
} else {
num_ev = num_ev.wrapping_add(1);
// restore original event record
unsafe { memcpy(event as *mut (), core::ptr::addr_of_mut!(event_saved) as *const (), saved_size) };
}
}
let mut __v13: *mut snd_seq_subscribers;
let mut __v14: *mut hlist_node;
let mut __v15: *mut hlist_node;
loop {
if !(((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64) || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_495() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v15 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*subs).src_list)).next) as *const *mut hlist_node) };
__v14 = __v15;
let __v0: *mut hlist_node = __v14;
let __v17: *mut snd_seq_subscribers = if !__v0.is_null() {
hlist_entry__2!(__v0)
} else {
core::ptr::null_mut::<snd_seq_subscribers>()
};
__v13 = __v17;
subs = __v13;
}
if atomic != 0 {
rcu_read_unlock();
} else {
unsafe { up_read(core::ptr::addr_of_mut!((*grp).list_mutex)) };
}
unsafe { memcpy(event as *mut (), core::ptr::addr_of_mut!(event_saved) as *const (), saved_size) };
if result < 0 { result } else { num_ev }
}

fn deliver_to_subscribers(client: *mut snd_seq_client, event: *mut snd_seq_event, atomic: i32, hop: i32) -> i32 {
let ret: i32 = __deliver_to_subscribers(client, event, (unsafe { (*event).source.port }) as i32, atomic, hop);
ret
}

/// deliver an event to the destination port(s).
/// if the event is to subscribers or broadcast, the event is dispatched
/// to multiple targets.
///
/// RETURN VALUE: n > 0  : the number of delivered events.
///               n == 0 : the event was not passed to any client.
///               n < 0  : error - event was not processed.
fn snd_seq_deliver_event(client: *mut snd_seq_client, event: *mut snd_seq_event, atomic: i32, mut hop: i32) -> i32 {
let mut result: i32 = 0;
hop = hop.wrapping_add(1);
if hop >= 8 {
let mut __v2: i32;
if false {
unsafe { _printk(c"\x017ALSA: seq: too long delivery path (%d:%d->%d:%d)\n".as_ptr() as *const u8, (*event).source.client as i32, (*event).source.port as i32, (*event).dest.client as i32, (*event).dest.port as i32) };
}
__v2 = 0;
return 31i32.wrapping_neg();
}
let __v3: bool = snd_seq_ev_is_variable!(event) && snd_BUG_ON!(atomic != 0 && (unsafe { (*event).data.ext.len }) & 2147483648 != 0) != 0;
if __v3 {
return 22i32.wrapping_neg();
}
if ((unsafe { (*event).queue }) as i32) == 254 || ((unsafe { (*event).dest.client }) as i32) == 254 {
result = deliver_to_subscribers(client, event, atomic, hop);
} else {
result = snd_seq_deliver_single_event(client, event, atomic, hop);
}
result
}

/// dispatch an event cell:
/// This function is called only from queue check routines in timer
/// interrupts or after enqueued.
/// The event cell shall be released or re-queued in this function.
///
/// RETURN VALUE: n > 0  : the number of delivered events.
/// 		 n == 0 : the event was not passed to any client.
/// 		 n < 0  : error - event was not processed.
#[unsafe(no_mangle)]
extern "C" fn snd_seq_dispatch_event(cell: *mut snd_seq_event_cell, atomic: i32, hop: i32) -> i32 {
let mut result: i32 = 0;
if snd_BUG_ON!(cell.is_null()) != 0 {
return 22i32.wrapping_neg();
}
let mut client: *mut snd_seq_client = snd_seq_client_use_ptr((unsafe { (*cell).__slate_anon_0.event.source.client }) as i32);
let __slate_cleanup_ptr_2 = core::ptr::addr_of_mut!(client) as *mut _;
let __slate_cleanup_2 = ::slate_support::Guard(&mut || __free_snd_seq_client(__slate_cleanup_ptr_2 as *mut _));
if client.is_null() {
unsafe { snd_seq_cell_free(cell) }; // release this cell
return 22i32.wrapping_neg();
}
if !snd_seq_ev_is_ump!(unsafe { core::ptr::addr_of_mut!((*cell).__slate_anon_0.event) }) && ((unsafe { (*cell).__slate_anon_0.event.r#type }) as i32) == 5 {
// NOTE event:
// the event cell is re-used as a NOTE-OFF event and
// enqueued again.
let mut tmpev: snd_seq_event = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(tmpev) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 28usize);
}
// reserve this event to enqueue note-off later
tmpev = unsafe { (*cell).__slate_anon_0.event };
tmpev.r#type = 6;
result = snd_seq_deliver_event(client, core::ptr::addr_of_mut!(tmpev), atomic, hop);
// This was originally a note event.  We now re-use the
// cell for the note-off event.
let ev: *mut snd_seq_event = unsafe { core::ptr::addr_of_mut!((*cell).__slate_anon_0.event) };
unsafe {
(*ev).r#type = 7;
}
unsafe {
(*ev).flags = (((*ev).flags as i32) | 16i32) as u8;
}
// add the duration time
match ((unsafe { (*ev).flags }) as i32) & 1i32 {
0 => {
let __v7: u32 = unsafe { (*cell).__slate_anon_0.event.time.tick };
unsafe {
(*cell).__slate_anon_0.event.time.tick = __v7.wrapping_add((*ev).data.note.duration);
}
}
1 => {
let __v11: u32 = unsafe { (*ev).time.time.tv_nsec };
unsafe {
(*ev).time.time.tv_nsec = __v11.wrapping_add(1000000u32.wrapping_mul((*ev).data.note.duration % 1000));
}
// unit for duration is ms
let __v14: u32 = unsafe { (*ev).time.time.tv_sec };
unsafe {
(*ev).time.time.tv_sec = __v14.wrapping_add(((*ev).data.note.duration / 1000).wrapping_add((*ev).time.time.tv_nsec / 1000000000));
}
unsafe {
(*ev).time.time.tv_nsec = (*ev).time.time.tv_nsec % 1000000000;
}
}
_ => {
}
}
unsafe {
(*ev).data.note.velocity = (*ev).data.note.off_velocity;
}
// Now queue this cell as the note off event
if (unsafe { snd_seq_enqueue_event(cell, atomic, hop) }) < 0 {
unsafe { snd_seq_cell_free(cell) };
}
// release this cell
} else {
// Normal events:
// event cell is freed after processing the event
result = snd_seq_deliver_event(client, unsafe { core::ptr::addr_of_mut!((*cell).__slate_anon_0.event) }, atomic, hop);
unsafe { snd_seq_cell_free(cell) };
}
result
}

/// Allocate a cell from client pool and enqueue it to queue:
/// if pool is empty and blocking is TRUE, sleep until a new cell is
/// available.
fn snd_seq_client_enqueue_event(client: *mut snd_seq_client, event: *mut snd_seq_event, file: *mut file, blocking: i32, atomic: i32, hop: i32, mutexp: *mut mutex) -> i32 {
let mut cell: *mut snd_seq_event_cell = unsafe { core::mem::zeroed() };
let mut err: i32 = 0;
// special queue values - force direct passing
if ((unsafe { (*event).queue }) as i32) == 254 {
unsafe {
(*event).dest.client = 254;
}
unsafe {
(*event).queue = 253;
}
} else {
if ((unsafe { (*event).dest.client }) as i32) == 254 {
// check presence of source port
let mut src_port: *mut snd_seq_client_port = unsafe { snd_seq_port_use_ptr(client, (*event).source.port as i32) };
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(src_port) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || __free_snd_seq_port(__slate_cleanup_ptr_0 as *mut _));
if src_port.is_null() {
return 22i32.wrapping_neg();
}
}
}
// direct event processing without enqueued
if snd_seq_ev_is_direct!(event) {
if !snd_seq_ev_is_ump!(event) && ((unsafe { (*event).r#type }) as i32) == 5 {
return 22i32.wrapping_neg();
}
// this event must be enqueued!
return snd_seq_deliver_event(client, event, atomic, hop);
}
// Not direct, normal queuing
if (unsafe { snd_seq_queue_is_used((*event).queue as i32, (*client).number) }) <= 0 {
return 22i32.wrapping_neg();
}
// invalid queue
if snd_seq_write_pool_allocated(client) == 0 {
return 6i32.wrapping_neg();
}
// queue is not allocated
// allocate an event cell
err = unsafe { snd_seq_event_dup((*client).pool, event, core::ptr::addr_of_mut!(cell), (blocking == 0 || atomic != 0) as i32, file, mutexp) };
if err < 0 {
return err;
}
// we got a cell. enqueue it.
err = unsafe { snd_seq_enqueue_event(cell, atomic, hop) };
if err < 0 {
unsafe { snd_seq_cell_free(cell) };
return err;
}
0
}

/// check validity of event type and data length.
/// return non-zero if invalid.
fn check_event_type_and_length(ev: *mut snd_seq_event) -> i32 {
match snd_seq_ev_length_type!(ev) {
0 => {
if snd_seq_ev_is_variable_type!(ev) {
return 22i32.wrapping_neg();
}
}
4 => {
if !snd_seq_ev_is_variable_type!(ev) || (unsafe { (*ev).data.ext.len }) & !3221225472 >= 1073741823 {
return 22i32.wrapping_neg();
}
}
8 => {
if !snd_seq_ev_is_direct!(ev) {
return 22i32.wrapping_neg();
}
}
_ => {
}
}
0
}

/// handle write()
///
/// possible error values:
/// -ENXIO	invalid client or file open mode
/// -ENOMEM	malloc failed
/// -EFAULT	seg. fault during copy from user space
/// -EINVAL	invalid event
/// -EAGAIN	no space in output pool
/// -EINTR	interrupts while sleep
/// -EMLINK	too many hops
/// others	depends on return value from driver callback
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_write")]
extern "C" fn snd_seq_write(file: *mut file, mut buf: *const u8, mut count: u64, offset: *mut i64) -> i64 {
let mut extlen: i32 = unsafe { core::mem::zeroed() };
let mut __v13: bool = unsafe { core::mem::zeroed() };
let mut ev: *mut snd_seq_event = unsafe { core::mem::zeroed() };
let mut __event: core::mem::MaybeUninit<__snd_seq_event> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut __snd_seq_event = core::ptr::addr_of_mut!(__event) as *mut __snd_seq_event;
let mut handled: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut i32 = core::ptr::addr_of_mut!(handled) as *mut i32;
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut len: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut i32 = core::ptr::addr_of_mut!(len) as *mut i32;
let mut written: i32 = unsafe { core::mem::zeroed() };
let mut client: *mut snd_seq_client = unsafe { core::mem::zeroed() };
unsafe {
client = (*file).private_data as *mut snd_seq_client;
written = 0;
let __slate_auto_bytes = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 28usize);
ev = core::ptr::addr_of_mut!((*__slate_slot_15).legacy);
if (snd_seq_file_flags(file) as i32) & 2 == 0 {
return 6i32.wrapping_neg() as i64;
} else {
// check client structures are in place
if snd_BUG_ON!(client.is_null()) != 0 {
return 6i32.wrapping_neg() as i64;
} else {
if ((*client).__slate_bits_0.__get_accept_output() as i32) == 0 || unsafe { (*client).pool }.is_null() {
return 6i32.wrapping_neg() as i64;
} else {
'__join_0: {
'__join_19: {
'__join_17: {
'__join_15: {
'__join_12: {
'__loop_26: loop {
*__slate_slot_16 = 0;
// allocate the pool now if the pool is not allocated yet
mutex_lock(core::ptr::addr_of_mut!((*client).ioctl_mutex));
if (*(*client).pool).size > 0 {
__v13 = snd_seq_write_pool_allocated(client) == 0;
} else {
__v13 = false;
}
if __v13 {
*__slate_slot_17 = snd_seq_pool_init((*client).pool);
if *__slate_slot_17 < 0 {
break '__join_0;
}
}
// only process whole events
*__slate_slot_17 = 22i32.wrapping_neg();
loop {
if count >= (core::mem::size_of::<snd_seq_event>() as u64) {
// Read in the event header from the user
*__slate_slot_18 = core::mem::size_of::<snd_seq_event>() as i32;
if copy_from_user(ev as *mut (), buf as *const (), *__slate_slot_18 as u64) != 0 {
break '__join_19;
} else {
// read in the rest bytes for UMP events
(*ev).source.client = (*client).number as u8;
// fill in client number
// Check for extension data length
if check_event_type_and_length(ev) != 0 {
break '__join_17;
} else {
if !event_is_compatible(client as *const snd_seq_client, ev as *const snd_seq_event) {
break '__join_15;
} else {
// check for special events
if ((*ev).r#type as i32) != 255 {
if snd_seq_ev_is_reserved!(ev) {
break '__join_12;
} else {
if snd_seq_ev_is_variable!(ev) {
extlen = ((*ev).data.ext.len & !3221225472) as i32;
if (extlen.wrapping_add(*__slate_slot_18) as u64) > count {
break '__loop_26;
} else {
// set user space pointer
(*ev).data.ext.len = (extlen as u32) | 2147483648;
(*ev).data.ext.ptr = (buf as *mut u8).wrapping_offset(*__slate_slot_18 as isize) as *mut ();
let __v11: i32 = *__slate_slot_18;
*__slate_slot_18 = __v11.wrapping_add(extlen); // increment data length
}
} else {
if (*client).convert32 != 0 && snd_seq_ev_is_varusr!(ev) {
(*ev).data.ext.ptr = compat_ptr(*((*ev).data.raw32.d.as_mut_ptr() as *mut u32).wrapping_offset(1 as isize));
}
}
// ok, enqueue it
*__slate_slot_17 = snd_seq_client_enqueue_event(client, ev, file, ((*file).f_flags & 2048u32 == 0) as i32, 0i32, 0i32, core::ptr::addr_of_mut!((*client).ioctl_mutex));
if *__slate_slot_17 < 0 {
break '__join_0;
} else {
let __v9: i32 = *__slate_slot_16;
*__slate_slot_16 = __v9.wrapping_add(1);
}
}
}
count = count.wrapping_sub(*__slate_slot_18 as u64); // Update pointers and counts
let __v5: *const u8 = buf;
buf = __v5.wrapping_offset(*__slate_slot_18 as isize);
written = written.wrapping_add(*__slate_slot_18);
// let's have a coffee break if too many events are queued
let __v1: i32 = *__slate_slot_16;
let __v0: i32 = __v1.wrapping_add(1);
*__slate_slot_16 = __v0;
if __v0 >= 200 {
break;
}
}
}
}
} else {
break '__join_0;
}
}
mutex_unlock(core::ptr::addr_of_mut!((*client).ioctl_mutex));
}
// back out, will get an error this time or next
*__slate_slot_17 = 22i32.wrapping_neg();
break '__join_0;
}
*__slate_slot_17 = 22i32.wrapping_neg();
break '__join_0;
}
*__slate_slot_17 = 22i32.wrapping_neg();
break '__join_0;
}
*__slate_slot_17 = 22i32.wrapping_neg();
break '__join_0;
}
*__slate_slot_17 = 14i32.wrapping_neg();
}
mutex_unlock(core::ptr::addr_of_mut!((*client).ioctl_mutex));
return (if written != 0 { written } else { *__slate_slot_17 }) as i64;
}
}
}
}
}

/// handle polling
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_poll")]
extern "C" fn snd_seq_poll(file: *mut file, wait: *mut poll_table_struct) -> u32 {
let client: *mut snd_seq_client = (unsafe { (*file).private_data }) as *mut snd_seq_client;
let mut mask: u32 = 0;
// check client structures are in place
if snd_BUG_ON!(client.is_null()) != 0 {
return 8;
}
if (snd_seq_file_flags(file) as i32) & 1 != 0 && !unsafe { (*client).data.user.fifo }.is_null() {
// check if data is available in the outqueue
if (unsafe { snd_seq_fifo_poll_wait((*client).data.user.fifo, file, wait) }) != 0 {
mask |= 1 | 64;
}
}
if (snd_seq_file_flags(file) as i32) & 2 != 0 {
// check if data is available in the pool
if (unsafe { snd_seq_pool_poll_wait((*client).pool, file, wait) }) != 0 {
mask |= 4 | 256;
}
}
mask
}

#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_ioctl_pversion")]
extern "C" fn snd_seq_ioctl_pversion(client: *mut snd_seq_client, arg: *mut ()) -> i32 {
unsafe {
*(arg as *mut i32) = SNDRV_PROTOCOL_VERSION!(1i32, 0i32, 5i32);
}
0
}

#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_ioctl_user_pversion")]
extern "C" fn snd_seq_ioctl_user_pversion(client: *mut snd_seq_client, arg: *mut ()) -> i32 {
unsafe {
(*client).user_pversion = *(arg as *mut u32);
}
0
}

#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_ioctl_client_id")]
extern "C" fn snd_seq_ioctl_client_id(client: *mut snd_seq_client, arg: *mut ()) -> i32 {
unsafe {
*(arg as *mut i32) = (*client).number;
}
0
}

/// SYSTEM_INFO ioctl()
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_ioctl_system_info")]
extern "C" fn snd_seq_ioctl_system_info(client: *mut snd_seq_client, arg: *mut ()) -> i32 {
let info: *mut snd_seq_system_info = arg as *mut snd_seq_system_info;
unsafe { memset(info as *mut (), 0i32, core::mem::size_of::<snd_seq_system_info>() as u64) };
// fill the info fields
unsafe {
(*info).queues = 32;
}
unsafe {
(*info).clients = 192;
}
unsafe {
(*info).ports = 254;
}
unsafe {
(*info).channels = 256;
}
// fixed limit
unsafe {
(*info).cur_clients = client_usage.cur;
}
unsafe {
(*info).cur_queues = snd_seq_queue_get_cur_queues();
}
0
}

/// RUNNING_MODE ioctl()
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_ioctl_running_mode")]
extern "C" fn snd_seq_ioctl_running_mode(client: *mut snd_seq_client, arg: *mut ()) -> i32 {
// requested client number
let mut cptr: *mut snd_seq_client = client_load_and_use_ptr((unsafe { (*(arg as *mut snd_seq_running_info)).client }) as i32);
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(cptr) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || __free_snd_seq_client(__slate_cleanup_ptr_0 as *mut _));
if cptr.is_null() {
return 2i32.wrapping_neg();
}
// don't change !!!
if (unsafe { (*(arg as *mut snd_seq_running_info)).big_endian }) != 0 {
return 22i32.wrapping_neg();
}
if (((unsafe { (*(arg as *mut snd_seq_running_info)).cpu_mode }) as i32) as u64) > (core::mem::size_of::<i64>() as u64) {
return 22i32.wrapping_neg();
}
unsafe {
(*cptr).convert32 = ((((*(arg as *mut snd_seq_running_info)).cpu_mode as i32) as u64) < (core::mem::size_of::<i64>() as u64)) as i32;
}
0
}

/// CLIENT_INFO ioctl()
fn get_client_info(cptr: *mut snd_seq_client, info: *mut snd_seq_client_info) {
unsafe {
(*info).client = (*cptr).number;
}
// fill the info fields
unsafe {
(*info).r#type = (*cptr).r#type;
}
unsafe { sized_strscpy((*info).name.as_mut_ptr() as *mut u8, ((*cptr).name.as_mut_ptr() as *mut u8) as *const u8, (core::mem::size_of::<[u8; 64]>() as u64).wrapping_add(__must_be_array__19!() as u64).wrapping_add(__must_be_cstr__37!() as u64).wrapping_add(__must_be_cstr__19!() as u64)) };
unsafe {
(*info).filter = (*cptr).filter;
}
unsafe {
(*info).event_lost = (*cptr).event_lost;
}
unsafe { memcpy(((*info).event_filter.as_mut_ptr() as *mut u8) as *mut (), ((*cptr).event_filter.as_mut_ptr() as *mut u64) as *const (), 32u64) };
unsafe {
(*info).group_filter = (*cptr).group_filter;
}
unsafe {
(*info).num_ports = (*cptr).num_ports;
}
if (unsafe { (*cptr).r#type }) == 1 {
unsafe {
(*info).pid = pid_vnr((*cptr).data.user.owner);
}
} else {
unsafe {
(*info).pid = 1i32.wrapping_neg();
}
}
if (unsafe { (*cptr).r#type }) == 2 {
unsafe {
(*info).card = if !unsafe { (*cptr).data.kernel.card }.is_null() { (*(*cptr).data.kernel.card).number } else { 1i32.wrapping_neg() };
}
} else {
unsafe {
(*info).card = 1i32.wrapping_neg();
}
}
unsafe {
(*info).midi_version = (*cptr).midi_version;
}
unsafe { memset(((*info).reserved.as_mut_ptr() as *mut u8) as *mut (), 0i32, core::mem::size_of::<[u8; 48]>() as u64) };
}

#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_ioctl_get_client_info")]
extern "C" fn snd_seq_ioctl_get_client_info(client: *mut snd_seq_client, arg: *mut ()) -> i32 {
let client_info: *mut snd_seq_client_info = arg as *mut snd_seq_client_info;
// requested client number
let mut cptr: *mut snd_seq_client = client_load_and_use_ptr(unsafe { (*client_info).client });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(cptr) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || __free_snd_seq_client(__slate_cleanup_ptr_0 as *mut _));
if cptr.is_null() {
return 2i32.wrapping_neg();
}
// don't change !!!
get_client_info(cptr, client_info);
0
}

/// CLIENT_INFO ioctl()
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_ioctl_set_client_info")]
extern "C" fn snd_seq_ioctl_set_client_info(client: *mut snd_seq_client, arg: *mut ()) -> i32 {
// it is not allowed to set the info fields for an another client
if (unsafe { (*client).number }) != unsafe { (*(arg as *mut snd_seq_client_info)).client } {
return 1i32.wrapping_neg();
}
// also client type must be set now
if (unsafe { (*client).r#type }) != unsafe { (*(arg as *mut snd_seq_client_info)).r#type } {
return 22i32.wrapping_neg();
}
if (unsafe { (*client).user_pversion }) >= (SNDRV_PROTOCOL_VERSION!(1i32, 0i32, 3i32) as u32) {
// check validity of midi_version field
if (unsafe { (*(arg as *mut snd_seq_client_info)).midi_version }) > 2 {
return 22i32.wrapping_neg();
}
// check if UMP is supported in kernel
if IS_ENABLED!() == 0 && (unsafe { (*(arg as *mut snd_seq_client_info)).midi_version }) > 0 {
return 22i32.wrapping_neg();
}
}
// fill the info fields
if (unsafe { *((*(arg as *mut snd_seq_client_info)).name.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) }) != 0 {
unsafe { sized_strscpy((*client).name.as_mut_ptr() as *mut u8, ((*(arg as *mut snd_seq_client_info)).name.as_mut_ptr() as *mut u8) as *const u8, (core::mem::size_of::<[u8; 64]>() as u64).wrapping_add(__must_be_cstr__48!() as u64).wrapping_add(__must_be_cstr__102!() as u64)) };
}
unsafe {
(*client).filter = (*(arg as *mut snd_seq_client_info)).filter;
}
unsafe {
(*client).event_lost = (*(arg as *mut snd_seq_client_info)).event_lost;
}
if (unsafe { (*client).user_pversion }) >= (SNDRV_PROTOCOL_VERSION!(1i32, 0i32, 3i32) as u32) {
unsafe {
(*client).midi_version = (*(arg as *mut snd_seq_client_info)).midi_version;
}
}
unsafe { memcpy(((*client).event_filter.as_mut_ptr() as *mut u64) as *mut (), ((*(arg as *mut snd_seq_client_info)).event_filter.as_mut_ptr() as *mut u8) as *const (), 32u64) };
unsafe {
(*client).group_filter = (((*(arg as *mut snd_seq_client_info)).group_filter as u64) & GENMASK__3!(16i32, 0i32)) as u32;
}
// notify the change
snd_seq_system_client_ev_client_change!(unsafe { (*client).number });
0
}

/// CREATE PORT ioctl()
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_ioctl_create_port")]
extern "C" fn snd_seq_ioctl_create_port(client: *mut snd_seq_client, arg: *mut ()) -> i32 {
let info: *mut snd_seq_port_info = arg as *mut snd_seq_port_info;
let mut port: *mut snd_seq_client_port = unsafe { core::mem::zeroed() };
let mut port_idx: i32 = 0;
let mut err: i32 = 0;
// it is not allowed to create the port for an another client
if ((unsafe { (*info).addr.client }) as i32) != unsafe { (*client).number } {
return 1i32.wrapping_neg();
}
if (unsafe { (*client).r#type }) == 1 && !unsafe { (*info).kernel }.is_null() {
return 22i32.wrapping_neg();
}
if (unsafe { (*info).capability }) & 512u32 != 0 && (unsafe { (*client).ump_endpoint_port }) >= 0 {
return 16i32.wrapping_neg();
}
if (unsafe { (*info).flags }) & 1u32 != 0 {
port_idx = (unsafe { (*info).addr.port }) as i32;
} else {
port_idx = 1i32.wrapping_neg();
}
if port_idx >= 253 {
return 22i32.wrapping_neg();
}
err = unsafe { snd_seq_create_port(client, core::ptr::addr_of_mut!(port)) };
if err < 0 {
return err;
}
if (unsafe { (*client).r#type }) == 2 {
let callback: *mut snd_seq_port_callback = (unsafe { (*info).kernel }) as *mut snd_seq_port_callback;
if !callback.is_null() {
if !unsafe { (*callback).owner }.is_null() {
unsafe {
(*port).owner = (*callback).owner;
}
}
unsafe {
(*port).private_data = (*callback).private_data;
}
unsafe {
(*port).private_free = (*callback).private_free;
}
unsafe {
(*port).event_input = (*callback).event_input;
}
unsafe {
(*port).c_src.open = (*callback).subscribe;
}
unsafe {
(*port).c_src.close = (*callback).unsubscribe;
}
unsafe {
(*port).c_dest.open = (*callback).r#use;
}
unsafe {
(*port).c_dest.close = (*callback).unuse;
}
}
}
unsafe { snd_seq_set_port_info(port, info) };
err = unsafe { snd_seq_insert_port(client, port_idx, port) };
if err < 0 {
unsafe { kfree(port as *const ()) };
return err;
}
unsafe {
(*info).addr = (*port).addr;
}
if (unsafe { (*info).capability }) & 512u32 != 0 {
unsafe {
(*client).ump_endpoint_port = (*port).addr.port as i32;
}
}
snd_seq_system_client_ev_port_start!(unsafe { (*port).addr.client }, unsafe { (*port).addr.port });
snd_seq_port_unlock!(port);
0
}

/// DELETE PORT ioctl()
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_ioctl_delete_port")]
extern "C" fn snd_seq_ioctl_delete_port(client: *mut snd_seq_client, arg: *mut ()) -> i32 {
let info: *mut snd_seq_port_info = arg as *mut snd_seq_port_info;
let mut err: i32 = 0;
// it is not allowed to remove the port for an another client
if ((unsafe { (*info).addr.client }) as i32) != unsafe { (*client).number } {
return 1i32.wrapping_neg();
}
err = unsafe { snd_seq_delete_port(client, (*info).addr.port as i32) };
if err >= 0 {
if (unsafe { (*client).ump_endpoint_port }) == ((unsafe { (*info).addr.port }) as i32) {
unsafe {
(*client).ump_endpoint_port = 1i32.wrapping_neg();
}
}
snd_seq_system_client_ev_port_exit!(unsafe { (*client).number }, unsafe { (*info).addr.port });
}
err
}

/// GET_PORT_INFO ioctl() (on any client)
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_ioctl_get_port_info")]
extern "C" fn snd_seq_ioctl_get_port_info(client: *mut snd_seq_client, arg: *mut ()) -> i32 {
let info: *mut snd_seq_port_info = arg as *mut snd_seq_port_info;
let mut cptr: *mut snd_seq_client = client_load_and_use_ptr((unsafe { (*info).addr.client }) as i32);
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(cptr) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || __free_snd_seq_client(__slate_cleanup_ptr_0 as *mut _));
if cptr.is_null() {
return 6i32.wrapping_neg();
}
let mut port: *mut snd_seq_client_port = unsafe { snd_seq_port_use_ptr(cptr, (*info).addr.port as i32) };
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(port) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __free_snd_seq_port(__slate_cleanup_ptr_1 as *mut _));
if port.is_null() {
return 2i32.wrapping_neg();
}
// don't change
// get port info
unsafe { snd_seq_get_port_info(port, info) };
0
}

/// SET_PORT_INFO ioctl() (only ports on this/own client)
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_ioctl_set_port_info")]
extern "C" fn snd_seq_ioctl_set_port_info(client: *mut snd_seq_client, arg: *mut ()) -> i32 {
let info: *mut snd_seq_port_info = arg as *mut snd_seq_port_info;
if ((unsafe { (*info).addr.client }) as i32) != unsafe { (*client).number } {
return 1i32.wrapping_neg();
}
// only set our own ports !
let mut port: *mut snd_seq_client_port = unsafe { snd_seq_port_use_ptr(client, (*info).addr.port as i32) };
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(port) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || __free_snd_seq_port(__slate_cleanup_ptr_0 as *mut _));
if !port.is_null() {
unsafe { snd_seq_set_port_info(port, info) };
// notify the change
snd_seq_system_client_ev_port_change!(unsafe { (*info).addr.client }, unsafe { (*info).addr.port });
}
0
}

// port subscription (connection)
fn check_subscription_permission(client: *mut snd_seq_client, sport: *mut snd_seq_client_port, dport: *mut snd_seq_client_port, subs: *mut snd_seq_port_subscribe) -> i32 {
if (unsafe { (*client).number }) != ((unsafe { (*subs).sender.client }) as i32) && (unsafe { (*client).number }) != ((unsafe { (*subs).dest.client }) as i32) {
// connection by third client - check export permission
if check_port_perm(sport, 128u32) != 0 {
return 1i32.wrapping_neg();
}
if check_port_perm(dport, 128u32) != 0 {
return 1i32.wrapping_neg();
}
}
// check read permission
//
// if sender or receiver is the subscribing client itself,
// no permission check is necessary
if (unsafe { (*client).number }) != ((unsafe { (*subs).sender.client }) as i32) {
if check_port_perm(sport, 33u32) == 0 {
return 1i32.wrapping_neg();
}
}
// check write permission
if (unsafe { (*client).number }) != ((unsafe { (*subs).dest.client }) as i32) {
if check_port_perm(dport, 66u32) == 0 {
return 1i32.wrapping_neg();
}
}
0
}

/// send an subscription notify event to user client:
/// client must be user client.
#[unsafe(no_mangle)]
extern "C" fn snd_seq_client_notify_subscription(client: i32, port: i32, info: *mut snd_seq_port_subscribe, evtype: i32) -> i32 {
let mut event: snd_seq_event = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(event) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 28usize);
}
unsafe { memset(core::ptr::addr_of_mut!(event) as *mut (), 0i32, core::mem::size_of::<snd_seq_event>() as u64) };
event.r#type = evtype as u8;
unsafe {
event.data.connect.dest = (*info).dest;
}
unsafe {
event.data.connect.sender = (*info).sender;
}
return unsafe { snd_seq_system_notify(client, port, core::ptr::addr_of_mut!(event), false) }; // non-atomic
}

/// add to port's subscription list IOCTL interface
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_ioctl_subscribe_port")]
extern "C" fn snd_seq_ioctl_subscribe_port(client: *mut snd_seq_client, arg: *mut ()) -> i32 {
let subs: *mut snd_seq_port_subscribe = arg as *mut snd_seq_port_subscribe;
let mut result: i32 = 0;
let mut receiver: *mut snd_seq_client = client_load_and_use_ptr((unsafe { (*subs).dest.client }) as i32);
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(receiver) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || __free_snd_seq_client(__slate_cleanup_ptr_0 as *mut _));
if receiver.is_null() {
return 22i32.wrapping_neg();
}
let mut sender: *mut snd_seq_client = client_load_and_use_ptr((unsafe { (*subs).sender.client }) as i32);
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(sender) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __free_snd_seq_client(__slate_cleanup_ptr_1 as *mut _));
if sender.is_null() {
return 22i32.wrapping_neg();
}
let mut sport: *mut snd_seq_client_port = unsafe { snd_seq_port_use_ptr(sender, (*subs).sender.port as i32) };
let __slate_cleanup_ptr_2 = core::ptr::addr_of_mut!(sport) as *mut _;
let __slate_cleanup_2 = ::slate_support::Guard(&mut || __free_snd_seq_port(__slate_cleanup_ptr_2 as *mut _));
if sport.is_null() {
return 22i32.wrapping_neg();
}
let mut dport: *mut snd_seq_client_port = unsafe { snd_seq_port_use_ptr(receiver, (*subs).dest.port as i32) };
let __slate_cleanup_ptr_3 = core::ptr::addr_of_mut!(dport) as *mut _;
let __slate_cleanup_3 = ::slate_support::Guard(&mut || __free_snd_seq_port(__slate_cleanup_ptr_3 as *mut _));
if dport.is_null() {
return 22i32.wrapping_neg();
}
result = check_subscription_permission(client, sport, dport, subs);
if result < 0 {
return result;
}
// connect them
result = unsafe { snd_seq_port_connect(client, sender, sport, receiver, dport, subs) };
if result == 0 {
snd_seq_client_notify_subscription(254i32, 0i32, subs, 66i32);
}
// broadcast announce
result
}

/// remove from port's subscription list
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_ioctl_unsubscribe_port")]
extern "C" fn snd_seq_ioctl_unsubscribe_port(client: *mut snd_seq_client, arg: *mut ()) -> i32 {
let subs: *mut snd_seq_port_subscribe = arg as *mut snd_seq_port_subscribe;
let mut result: i32 = 0;
let mut receiver: *mut snd_seq_client = snd_seq_client_use_ptr((unsafe { (*subs).dest.client }) as i32);
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(receiver) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || __free_snd_seq_client(__slate_cleanup_ptr_0 as *mut _));
if receiver.is_null() {
return 6i32.wrapping_neg();
}
let mut sender: *mut snd_seq_client = snd_seq_client_use_ptr((unsafe { (*subs).sender.client }) as i32);
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(sender) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __free_snd_seq_client(__slate_cleanup_ptr_1 as *mut _));
if sender.is_null() {
return 6i32.wrapping_neg();
}
let mut sport: *mut snd_seq_client_port = unsafe { snd_seq_port_use_ptr(sender, (*subs).sender.port as i32) };
let __slate_cleanup_ptr_2 = core::ptr::addr_of_mut!(sport) as *mut _;
let __slate_cleanup_2 = ::slate_support::Guard(&mut || __free_snd_seq_port(__slate_cleanup_ptr_2 as *mut _));
if sport.is_null() {
return 6i32.wrapping_neg();
}
let mut dport: *mut snd_seq_client_port = unsafe { snd_seq_port_use_ptr(receiver, (*subs).dest.port as i32) };
let __slate_cleanup_ptr_3 = core::ptr::addr_of_mut!(dport) as *mut _;
let __slate_cleanup_3 = ::slate_support::Guard(&mut || __free_snd_seq_port(__slate_cleanup_ptr_3 as *mut _));
if dport.is_null() {
return 6i32.wrapping_neg();
}
result = check_subscription_permission(client, sport, dport, subs);
if result < 0 {
return result;
}
result = unsafe { snd_seq_port_disconnect(client, sender, sport, receiver, dport, subs) };
if result == 0 {
snd_seq_client_notify_subscription(254i32, 0i32, subs, 67i32);
}
// broadcast announce
result
}

/// CREATE_QUEUE ioctl()
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_ioctl_create_queue")]
extern "C" fn snd_seq_ioctl_create_queue(client: *mut snd_seq_client, arg: *mut ()) -> i32 {
let mut q: *mut snd_seq_queue = unsafe { snd_seq_queue_alloc((*client).number, (*(arg as *mut snd_seq_queue_info)).__slate_bits_0.__get_locked() as i32, (*(arg as *mut snd_seq_queue_info)).flags) };
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(q) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || __free_snd_seq_queue(__slate_cleanup_ptr_0 as *mut _));
if IS_ERR(q as *const ()) {
return PTR_ERR(q as *const ()) as i32;
}
unsafe {
(*(arg as *mut snd_seq_queue_info)).queue = (*q).queue;
}
unsafe {
let __t0 = (*q).locked as u32;
(*(arg as *mut snd_seq_queue_info)).__slate_bits_0.__set_locked(__t0);
}
unsafe {
(*(arg as *mut snd_seq_queue_info)).owner = (*q).owner;
}
// set queue name
if (unsafe { *((*(arg as *mut snd_seq_queue_info)).name.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) }) == 0 {
unsafe { snprintf((*(arg as *mut snd_seq_queue_info)).name.as_mut_ptr() as *mut u8, core::mem::size_of::<[u8; 64]>() as u64, c"Queue-%d".as_ptr() as *const u8, (*q).queue) };
}
unsafe { sized_strscpy((*q).name.as_mut_ptr() as *mut u8, ((*(arg as *mut snd_seq_queue_info)).name.as_mut_ptr() as *mut u8) as *const u8, (core::mem::size_of::<[u8; 64]>() as u64).wrapping_add(__must_be_cstr__68!() as u64).wrapping_add(__must_be_cstr__75!() as u64)) };
0
}

/// DELETE_QUEUE ioctl()
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_ioctl_delete_queue")]
extern "C" fn snd_seq_ioctl_delete_queue(client: *mut snd_seq_client, arg: *mut ()) -> i32 {
unsafe { snd_seq_queue_delete((*client).number, (*(arg as *mut snd_seq_queue_info)).queue) }
}

/// GET_QUEUE_INFO ioctl()
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_ioctl_get_queue_info")]
extern "C" fn snd_seq_ioctl_get_queue_info(client: *mut snd_seq_client, arg: *mut ()) -> i32 {
let info: *mut snd_seq_queue_info = arg as *mut snd_seq_queue_info;
let mut q: *mut snd_seq_queue = unsafe { queueptr((*info).queue) };
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(q) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || __free_snd_seq_queue(__slate_cleanup_ptr_0 as *mut _));
if q.is_null() {
return 22i32.wrapping_neg();
}
unsafe { memset(info as *mut (), 0i32, core::mem::size_of::<snd_seq_queue_info>() as u64) };
unsafe {
(*info).queue = (*q).queue;
}
unsafe {
(*info).owner = (*q).owner;
}
unsafe {
let __t0 = (*q).locked as u32;
(*info).__slate_bits_0.__set_locked(__t0);
}
unsafe { sized_strscpy((*info).name.as_mut_ptr() as *mut u8, ((*q).name.as_mut_ptr() as *mut u8) as *const u8, (core::mem::size_of::<[u8; 64]>() as u64).wrapping_add(__must_be_cstr__6!() as u64).wrapping_add(__must_be_cstr__83!() as u64)) };
0
}

/// SET_QUEUE_INFO ioctl()
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_ioctl_set_queue_info")]
extern "C" fn snd_seq_ioctl_set_queue_info(client: *mut snd_seq_client, arg: *mut ()) -> i32 {
if (unsafe { (*(arg as *mut snd_seq_queue_info)).owner }) != unsafe { (*client).number } {
return 22i32.wrapping_neg();
}
// change owner/locked permission
if (unsafe { snd_seq_queue_check_access((*(arg as *mut snd_seq_queue_info)).queue, (*client).number) }) != 0 {
if (unsafe { snd_seq_queue_set_owner((*(arg as *mut snd_seq_queue_info)).queue, (*client).number, (*(arg as *mut snd_seq_queue_info)).__slate_bits_0.__get_locked() as i32) }) < 0 {
return 1i32.wrapping_neg();
}
if ((unsafe { (*(arg as *mut snd_seq_queue_info)).__slate_bits_0.__get_locked() }) as i32) != 0 {
unsafe { snd_seq_queue_use((*(arg as *mut snd_seq_queue_info)).queue, (*client).number, 1i32) };
}
} else {
return 1i32.wrapping_neg();
}
let mut q: *mut snd_seq_queue = unsafe { queueptr((*(arg as *mut snd_seq_queue_info)).queue) };
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(q) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || __free_snd_seq_queue(__slate_cleanup_ptr_0 as *mut _));
if q.is_null() {
return 22i32.wrapping_neg();
}
if (unsafe { (*q).owner }) != unsafe { (*client).number } {
return 1i32.wrapping_neg();
}
unsafe { sized_strscpy((*q).name.as_mut_ptr() as *mut u8, ((*(arg as *mut snd_seq_queue_info)).name.as_mut_ptr() as *mut u8) as *const u8, (core::mem::size_of::<[u8; 64]>() as u64).wrapping_add(__must_be_cstr__56!() as u64).wrapping_add(__must_be_cstr__89!() as u64)) };
0
}

/// GET_NAMED_QUEUE ioctl()
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_ioctl_get_named_queue")]
extern "C" fn snd_seq_ioctl_get_named_queue(client: *mut snd_seq_client, arg: *mut ()) -> i32 {
let mut q: *mut snd_seq_queue = unsafe { snd_seq_queue_find_name((*(arg as *mut snd_seq_queue_info)).name.as_mut_ptr() as *mut u8) };
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(q) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || __free_snd_seq_queue(__slate_cleanup_ptr_0 as *mut _));
if q.is_null() {
return 22i32.wrapping_neg();
}
unsafe {
(*(arg as *mut snd_seq_queue_info)).queue = (*q).queue;
}
unsafe {
(*(arg as *mut snd_seq_queue_info)).owner = (*q).owner;
}
unsafe {
let __t0 = (*q).locked as u32;
(*(arg as *mut snd_seq_queue_info)).__slate_bits_0.__set_locked(__t0);
}
0
}

/// GET_QUEUE_STATUS ioctl()
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_ioctl_get_queue_status")]
extern "C" fn snd_seq_ioctl_get_queue_status(client: *mut snd_seq_client, arg: *mut ()) -> i32 {
let status: *mut snd_seq_queue_status = arg as *mut snd_seq_queue_status;
let mut queue: *mut snd_seq_queue = unsafe { queueptr((*status).queue) };
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(queue) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || __free_snd_seq_queue(__slate_cleanup_ptr_0 as *mut _));
if queue.is_null() {
return 22i32.wrapping_neg();
}
unsafe { memset(status as *mut (), 0i32, core::mem::size_of::<snd_seq_queue_status>() as u64) };
unsafe {
(*status).queue = (*queue).queue;
}
let tmr: *mut snd_seq_timer = unsafe { (*queue).timer };
unsafe {
(*status).events = unsafe { (*(*queue).tickq).cells }.wrapping_add((*(*queue).timeq).cells);
}
unsafe {
(*status).time = snd_seq_timer_get_cur_time(tmr, true);
}
unsafe {
(*status).tick = snd_seq_timer_get_cur_tick(tmr);
}
unsafe {
(*status).running = (*tmr).__slate_bits_0.__get_running() as i32;
}
unsafe {
(*status).flags = (*queue).flags as i32;
}
0
}

/// GET_QUEUE_TEMPO ioctl()
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_ioctl_get_queue_tempo")]
extern "C" fn snd_seq_ioctl_get_queue_tempo(client: *mut snd_seq_client, arg: *mut ()) -> i32 {
let tempo: *mut snd_seq_queue_tempo = arg as *mut snd_seq_queue_tempo;
let mut queue: *mut snd_seq_queue = unsafe { queueptr((*tempo).queue) };
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(queue) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || __free_snd_seq_queue(__slate_cleanup_ptr_0 as *mut _));
if queue.is_null() {
return 22i32.wrapping_neg();
}
unsafe { memset(tempo as *mut (), 0i32, core::mem::size_of::<snd_seq_queue_tempo>() as u64) };
unsafe {
(*tempo).queue = (*queue).queue;
}
let tmr: *mut snd_seq_timer = unsafe { (*queue).timer };
unsafe {
(*tempo).tempo = (*tmr).tempo;
}
unsafe {
(*tempo).ppq = (*tmr).ppq;
}
unsafe {
(*tempo).skew_value = (*tmr).skew;
}
unsafe {
(*tempo).skew_base = (*tmr).skew_base;
}
if (unsafe { (*client).user_pversion }) >= (SNDRV_PROTOCOL_VERSION!(1i32, 0i32, 4i32) as u32) {
unsafe {
(*tempo).tempo_base = (*tmr).tempo_base as u16;
}
}
0
}

/// SET_QUEUE_TEMPO ioctl()
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_set_queue_tempo")]
extern "C" fn snd_seq_set_queue_tempo(client: i32, tempo: *mut snd_seq_queue_tempo) -> i32 {
if (unsafe { snd_seq_queue_check_access((*tempo).queue, client) }) == 0 {
return 1i32.wrapping_neg();
}
unsafe { snd_seq_queue_timer_set_tempo((*tempo).queue, client, tempo) }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_seq_set_queue_tempo_496: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(i32, *mut snd_seq_queue_tempo) -> i32>, *mut ()>(Some(snd_seq_set_queue_tempo)) };

#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_ioctl_set_queue_tempo")]
extern "C" fn snd_seq_ioctl_set_queue_tempo(client: *mut snd_seq_client, arg: *mut ()) -> i32 {
let tempo: *mut snd_seq_queue_tempo = arg as *mut snd_seq_queue_tempo;
let mut result: i32 = 0;
if (unsafe { (*client).user_pversion }) < (SNDRV_PROTOCOL_VERSION!(1i32, 0i32, 4i32) as u32) {
unsafe {
(*tempo).tempo_base = 0;
}
}
result = snd_seq_set_queue_tempo(unsafe { (*client).number }, tempo);
if result < 0 { result } else { 0 }
}

/// GET_QUEUE_TIMER ioctl()
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_ioctl_get_queue_timer")]
extern "C" fn snd_seq_ioctl_get_queue_timer(client: *mut snd_seq_client, arg: *mut ()) -> i32 {
let timer: *mut snd_seq_queue_timer = arg as *mut snd_seq_queue_timer;
let mut queue: *mut snd_seq_queue = unsafe { queueptr((*timer).queue) };
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(queue) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || __free_snd_seq_queue(__slate_cleanup_ptr_0 as *mut _));
if queue.is_null() {
return 22i32.wrapping_neg();
}
let mut __UNIQUE_ID_guard_497: __SlateRecord_mutex_h_253 = class_mutex_constructor(unsafe { core::ptr::addr_of_mut!((*queue).timer_mutex) });
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_497) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || class_mutex_destructor(__slate_cleanup_ptr_1 as *mut _));
let mut __UNIQUE_ID_unlock_498: *mut __SlateRecord_mutex_h_253 = ((unsafe { core::ptr::addr_of_mut!((*queue).timer_mutex) }) as u64) as *mut __SlateRecord_mutex_h_253;
let __slate_cleanup_ptr_2 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_498) as *mut _;
let __slate_cleanup_2 = ::slate_support::Guard(&mut || __class_mutex_cleanup_ctx(__slate_cleanup_ptr_2 as *mut _));
let tmr: *mut snd_seq_timer = unsafe { (*queue).timer };
unsafe { memset(timer as *mut (), 0i32, core::mem::size_of::<snd_seq_queue_timer>() as u64) };
unsafe {
(*timer).queue = (*queue).queue;
}
unsafe {
(*timer).r#type = (*tmr).r#type;
}
if (unsafe { (*tmr).r#type }) == 0 {
unsafe {
(*timer).u.alsa.id = (*tmr).alsa_id;
}
unsafe {
(*timer).u.alsa.resolution = (*tmr).preferred_resolution as u32;
}
}
0
}

/// SET_QUEUE_TIMER ioctl()
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_ioctl_set_queue_timer")]
extern "C" fn snd_seq_ioctl_set_queue_timer(client: *mut snd_seq_client, arg: *mut ()) -> i32 {
let mut result: i32 = 0;
if (unsafe { (*(arg as *mut snd_seq_queue_timer)).r#type }) != 0 {
return 22i32.wrapping_neg();
}
if (unsafe { snd_seq_queue_check_access((*(arg as *mut snd_seq_queue_timer)).queue, (*client).number) }) != 0 {
let mut q: *mut snd_seq_queue = unsafe { queueptr((*(arg as *mut snd_seq_queue_timer)).queue) };
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(q) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || __free_snd_seq_queue(__slate_cleanup_ptr_0 as *mut _));
if q.is_null() {
return 6i32.wrapping_neg();
}
let mut __UNIQUE_ID_guard_499: __SlateRecord_mutex_h_253 = class_mutex_constructor(unsafe { core::ptr::addr_of_mut!((*q).timer_mutex) });
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_499) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || class_mutex_destructor(__slate_cleanup_ptr_1 as *mut _));
let mut __UNIQUE_ID_unlock_500: *mut __SlateRecord_mutex_h_253 = ((unsafe { core::ptr::addr_of_mut!((*q).timer_mutex) }) as u64) as *mut __SlateRecord_mutex_h_253;
let __slate_cleanup_ptr_2 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_500) as *mut _;
let __slate_cleanup_2 = ::slate_support::Guard(&mut || __class_mutex_cleanup_ctx(__slate_cleanup_ptr_2 as *mut _));
let tmr: *mut snd_seq_timer = unsafe { (*q).timer };
unsafe { snd_seq_queue_timer_close((*(arg as *mut snd_seq_queue_timer)).queue) };
unsafe {
(*tmr).r#type = (*(arg as *mut snd_seq_queue_timer)).r#type;
}
if (unsafe { (*tmr).r#type }) == 0 {
unsafe {
(*tmr).alsa_id = (*(arg as *mut snd_seq_queue_timer)).u.alsa.id;
}
unsafe {
(*tmr).preferred_resolution = (*(arg as *mut snd_seq_queue_timer)).u.alsa.resolution as u64;
}
}
result = unsafe { snd_seq_queue_timer_open((*(arg as *mut snd_seq_queue_timer)).queue) };
} else {
return 1i32.wrapping_neg();
}
result
}

/// GET_QUEUE_CLIENT ioctl()
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_ioctl_get_queue_client")]
extern "C" fn snd_seq_ioctl_get_queue_client(client: *mut snd_seq_client, arg: *mut ()) -> i32 {
let used: i32 = unsafe { snd_seq_queue_is_used((*(arg as *mut snd_seq_queue_client)).queue, (*client).number) };
if used < 0 {
return 22i32.wrapping_neg();
}
unsafe {
(*(arg as *mut snd_seq_queue_client)).used = used;
}
unsafe {
(*(arg as *mut snd_seq_queue_client)).client = (*client).number;
}
0
}

/// SET_QUEUE_CLIENT ioctl()
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_ioctl_set_queue_client")]
extern "C" fn snd_seq_ioctl_set_queue_client(client: *mut snd_seq_client, arg: *mut ()) -> i32 {
let mut err: i32 = 0;
if (unsafe { (*(arg as *mut snd_seq_queue_client)).used }) >= 0 {
err = unsafe { snd_seq_queue_use((*(arg as *mut snd_seq_queue_client)).queue, (*client).number, (*(arg as *mut snd_seq_queue_client)).used) };
if err < 0 {
return err;
}
}
snd_seq_ioctl_get_queue_client(client, arg)
}

/// GET_CLIENT_POOL ioctl()
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_ioctl_get_client_pool")]
extern "C" fn snd_seq_ioctl_get_client_pool(client: *mut snd_seq_client, arg: *mut ()) -> i32 {
let info: *mut snd_seq_client_pool = arg as *mut snd_seq_client_pool;
let mut cptr: *mut snd_seq_client = client_load_and_use_ptr(unsafe { (*info).client });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(cptr) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || __free_snd_seq_client(__slate_cleanup_ptr_0 as *mut _));
if cptr.is_null() {
return 2i32.wrapping_neg();
}
unsafe { memset(info as *mut (), 0i32, core::mem::size_of::<snd_seq_client_pool>() as u64) };
unsafe {
(*info).client = (*cptr).number;
}
unsafe {
(*info).output_pool = (*(*cptr).pool).size;
}
unsafe {
(*info).output_room = (*(*cptr).pool).room;
}
unsafe {
(*info).output_free = (*info).output_pool;
}
unsafe {
(*info).output_free = snd_seq_unused_cells((*cptr).pool);
}
if (unsafe { (*cptr).r#type }) == 1 {
unsafe {
(*info).input_pool = (*cptr).data.user.fifo_pool_size;
}
unsafe {
(*info).input_free = (*info).input_pool;
}
unsafe {
(*info).input_free = snd_seq_fifo_unused_cells((*cptr).data.user.fifo);
}
} else {
unsafe {
(*info).input_pool = 0;
}
unsafe {
(*info).input_free = 0;
}
}
0
}

/// SET_CLIENT_POOL ioctl()
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_ioctl_set_client_pool")]
extern "C" fn snd_seq_ioctl_set_client_pool(client: *mut snd_seq_client, arg: *mut ()) -> i32 {
let mut rc: i32 = 0;
if (unsafe { (*client).number }) != unsafe { (*(arg as *mut snd_seq_client_pool)).client } {
return 22i32.wrapping_neg();
}
// can't change other clients
let __v0: bool = (unsafe { (*(arg as *mut snd_seq_client_pool)).output_pool }) >= 1 && (unsafe { (*(arg as *mut snd_seq_client_pool)).output_pool }) <= 2000 && (snd_seq_write_pool_allocated(client) == 0 || (unsafe { (*(arg as *mut snd_seq_client_pool)).output_pool }) != unsafe { (*(*client).pool).size });
if __v0 {
if snd_seq_write_pool_allocated(client) != 0 {
// is the pool in use?
if atomic_read((unsafe { core::ptr::addr_of_mut!((*(*client).pool).counter) }) as *const __SlateRecord_types_h_190) != 0 {
return 16i32.wrapping_neg();
}
// remove all existing cells
unsafe { snd_seq_pool_mark_closing((*client).pool) };
unsafe { snd_seq_pool_done((*client).pool) };
}
unsafe {
(*(*client).pool).size = (*(arg as *mut snd_seq_client_pool)).output_pool;
}
rc = unsafe { snd_seq_pool_init((*client).pool) };
if rc < 0 {
return rc;
}
}
if (unsafe { (*client).r#type }) == 1 && !unsafe { (*client).data.user.fifo }.is_null() && (unsafe { (*(arg as *mut snd_seq_client_pool)).input_pool }) >= 1 && (unsafe { (*(arg as *mut snd_seq_client_pool)).input_pool }) <= 2000 && (unsafe { (*(arg as *mut snd_seq_client_pool)).input_pool }) != unsafe { (*client).data.user.fifo_pool_size } {
// change pool size
rc = unsafe { snd_seq_fifo_resize((*client).data.user.fifo, (*(arg as *mut snd_seq_client_pool)).input_pool) };
if rc < 0 {
return rc;
}
unsafe {
(*client).data.user.fifo_pool_size = (*(arg as *mut snd_seq_client_pool)).input_pool;
}
}
if (unsafe { (*(arg as *mut snd_seq_client_pool)).output_room }) >= 1 && (unsafe { (*(arg as *mut snd_seq_client_pool)).output_room }) <= unsafe { (*(*client).pool).size } {
unsafe {
(*(*client).pool).room = (*(arg as *mut snd_seq_client_pool)).output_room;
}
}
snd_seq_ioctl_get_client_pool(client, arg)
}

/// REMOVE_EVENTS ioctl()
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_ioctl_remove_events")]
extern "C" fn snd_seq_ioctl_remove_events(client: *mut snd_seq_client, arg: *mut ()) -> i32 {
let info: *mut snd_seq_remove_events = arg as *mut snd_seq_remove_events;
// Input mostly not implemented XXX.
if (unsafe { (*info).remove_mode }) & 1u32 != 0 {
// No restrictions so for a user client we can clear
// the whole fifo
if (unsafe { (*client).r#type }) == 1 && !unsafe { (*client).data.user.fifo }.is_null() {
unsafe { snd_seq_fifo_clear((*client).data.user.fifo) };
}
}
if (unsafe { (*info).remove_mode }) & 2u32 != 0 {
unsafe { snd_seq_queue_remove_cells((*client).number, info) };
}
0
}

/// get subscription info
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_ioctl_get_subscription")]
extern "C" fn snd_seq_ioctl_get_subscription(client: *mut snd_seq_client, arg: *mut ()) -> i32 {
let subs: *mut snd_seq_port_subscribe = arg as *mut snd_seq_port_subscribe;
let mut sender: *mut snd_seq_client = client_load_and_use_ptr((unsafe { (*subs).sender.client }) as i32);
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(sender) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || __free_snd_seq_client(__slate_cleanup_ptr_0 as *mut _));
if sender.is_null() {
return 22i32.wrapping_neg();
}
let mut sport: *mut snd_seq_client_port = unsafe { snd_seq_port_use_ptr(sender, (*subs).sender.port as i32) };
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(sport) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __free_snd_seq_port(__slate_cleanup_ptr_1 as *mut _));
if sport.is_null() {
return 22i32.wrapping_neg();
}
unsafe { snd_seq_port_get_subscription(core::ptr::addr_of_mut!((*sport).c_src), core::ptr::addr_of_mut!((*subs).dest), subs) }
}

/// get subscription info - check only its presence
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_ioctl_query_subs")]
extern "C" fn snd_seq_ioctl_query_subs(client: *mut snd_seq_client, arg: *mut ()) -> i32 {
let mut group: *mut snd_seq_port_subs_info = unsafe { core::mem::zeroed() };
let mut p: *mut hlist_node = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
let mut cptr: *mut snd_seq_client = client_load_and_use_ptr((unsafe { (*(arg as *mut snd_seq_query_subs)).root.client }) as i32);
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(cptr) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || __free_snd_seq_client(__slate_cleanup_ptr_0 as *mut _));
if cptr.is_null() {
return 6i32.wrapping_neg();
}
let mut port: *mut snd_seq_client_port = unsafe { snd_seq_port_use_ptr(cptr, (*(arg as *mut snd_seq_query_subs)).root.port as i32) };
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(port) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __free_snd_seq_port(__slate_cleanup_ptr_1 as *mut _));
if port.is_null() {
return 6i32.wrapping_neg();
}
match unsafe { (*(arg as *mut snd_seq_query_subs)).r#type } {
0 => {
group = unsafe { core::ptr::addr_of_mut!((*port).c_src) };
}
1 => {
group = unsafe { core::ptr::addr_of_mut!((*port).c_dest) };
}
_ => {
return 6i32.wrapping_neg();
}
}
let mut __UNIQUE_ID_guard_501: __SlateRecord_rwsem_h_259 = class_rwsem_read_constructor(unsafe { core::ptr::addr_of_mut!((*group).list_mutex) });
let __slate_cleanup_ptr_3 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_501) as *mut _;
let __slate_cleanup_3 = ::slate_support::Guard(&mut || class_rwsem_read_destructor(__slate_cleanup_ptr_3 as *mut _));
let mut __UNIQUE_ID_unlock_502: *mut __SlateRecord_rwsem_h_259 = ((unsafe { core::ptr::addr_of_mut!((*group).list_mutex) }) as u64) as *mut __SlateRecord_rwsem_h_259;
let __slate_cleanup_ptr_4 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_502) as *mut _;
let __slate_cleanup_4 = ::slate_support::Guard(&mut || __class_rwsem_read_cleanup_ctx(__slate_cleanup_ptr_4 as *mut _));
// search for the subscriber
unsafe {
(*(arg as *mut snd_seq_query_subs)).num_subs = (*group).count as i32;
}
p = unsafe { (*core::ptr::addr_of_mut!((*group).list_head)).first };
while !p.is_null() {
let __v5: i32 = i;
i = __v5.wrapping_add(1);
if __v5 == unsafe { (*(arg as *mut snd_seq_query_subs)).index } {
// found!
let mut s: *mut snd_seq_subscribers = unsafe { core::mem::zeroed() };
if (unsafe { (*(arg as *mut snd_seq_query_subs)).r#type }) == 0 {
s = hlist_entry!(p, snd_seq_subscribers);
unsafe {
(*(arg as *mut snd_seq_query_subs)).addr = (*s).info.dest;
}
} else {
s = hlist_entry__3!(p, snd_seq_subscribers);
unsafe {
(*(arg as *mut snd_seq_query_subs)).addr = (*s).info.sender;
}
}
unsafe {
(*(arg as *mut snd_seq_query_subs)).flags = (*s).info.flags;
}
unsafe {
(*(arg as *mut snd_seq_query_subs)).queue = (*s).info.queue;
}
return 0;
}
p = unsafe { (*p).next };
}
2i32.wrapping_neg()
}

/// query next client
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_ioctl_query_next_client")]
extern "C" fn snd_seq_ioctl_query_next_client(client: *mut snd_seq_client, arg: *mut ()) -> i32 {
let info: *mut snd_seq_client_info = arg as *mut snd_seq_client_info;
// search for next client
if (unsafe { (*info).client }) < 2147483647i32 {
let __v1: i32 = unsafe { (*info).client };
unsafe {
(*info).client = __v1.wrapping_add(1);
}
}
if (unsafe { (*info).client }) < 0 {
unsafe {
(*info).client = 0;
}
}
while (unsafe { (*info).client }) < 192 {
{
let mut cptr: *mut snd_seq_client = client_load_and_use_ptr(unsafe { (*info).client });
let __slate_cleanup_ptr_3 = core::ptr::addr_of_mut!(cptr) as *mut _;
let __slate_cleanup_3 = ::slate_support::Guard(&mut || __free_snd_seq_client(__slate_cleanup_ptr_3 as *mut _));
if !cptr.is_null() {
get_client_info(cptr, info);
return 0; // found
}
}
let __v6: i32 = unsafe { (*info).client };
unsafe {
(*info).client = __v6.wrapping_add(1);
}
}
2i32.wrapping_neg()
}

/// query next port
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_ioctl_query_next_port")]
extern "C" fn snd_seq_ioctl_query_next_port(client: *mut snd_seq_client, arg: *mut ()) -> i32 {
let info: *mut snd_seq_port_info = arg as *mut snd_seq_port_info;
let mut cptr: *mut snd_seq_client = client_load_and_use_ptr((unsafe { (*info).addr.client }) as i32);
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(cptr) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || __free_snd_seq_client(__slate_cleanup_ptr_0 as *mut _));
if cptr.is_null() {
return 6i32.wrapping_neg();
}
// search for next port
unsafe {
(*info).addr.port = ((*info).addr.port as i32).wrapping_add(1) as u8;
}
let mut port: *mut snd_seq_client_port = unsafe { snd_seq_port_query_nearest(cptr, info) };
let __slate_cleanup_ptr_4 = core::ptr::addr_of_mut!(port) as *mut _;
let __slate_cleanup_4 = ::slate_support::Guard(&mut || __free_snd_seq_port(__slate_cleanup_ptr_4 as *mut _));
if port.is_null() {
return 2i32.wrapping_neg();
}
// get port info
unsafe {
(*info).addr = (*port).addr;
}
unsafe { snd_seq_get_port_info(port, info) };
0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ioctl_handler {
    cmd: u32,
    func: Option<unsafe extern "C" fn(*mut snd_seq_client, *mut ()) -> i32>,
}


static ioctl_handlers: __SlateAlign16<[ioctl_handler; 33]> = __SlateAlign16([{
let mut __t0: ioctl_handler = unsafe { core::mem::zeroed() };
__t0.cmd = (2147504896u64 | (core::mem::size_of::<i32>() as u64) << 16i32) as u32;
__t0.func = Some(snd_seq_ioctl_pversion);
__t0
}, {
let mut __t1: ioctl_handler = unsafe { core::mem::zeroed() };
__t1.cmd = (1073763076u64 | (core::mem::size_of::<i32>() as u64) << 16i32) as u32;
__t1.func = Some(snd_seq_ioctl_user_pversion);
__t1
}, {
let mut __t2: ioctl_handler = unsafe { core::mem::zeroed() };
__t2.cmd = (2147504897u64 | (core::mem::size_of::<i32>() as u64) << 16i32) as u32;
__t2.func = Some(snd_seq_ioctl_client_id);
__t2
}, {
let mut __t3: ioctl_handler = unsafe { core::mem::zeroed() };
__t3.cmd = (3221246722u64 | (core::mem::size_of::<snd_seq_system_info>() as u64) << 16i32) as u32;
__t3.func = Some(snd_seq_ioctl_system_info);
__t3
}, {
let mut __t4: ioctl_handler = unsafe { core::mem::zeroed() };
__t4.cmd = (3221246723u64 | (core::mem::size_of::<snd_seq_running_info>() as u64) << 16i32) as u32;
__t4.func = Some(snd_seq_ioctl_running_mode);
__t4
}, {
let mut __t5: ioctl_handler = unsafe { core::mem::zeroed() };
__t5.cmd = (3221246736u64 | (core::mem::size_of::<snd_seq_client_info>() as u64) << 16i32) as u32;
__t5.func = Some(snd_seq_ioctl_get_client_info);
__t5
}, {
let mut __t6: ioctl_handler = unsafe { core::mem::zeroed() };
__t6.cmd = (1073763089u64 | (core::mem::size_of::<snd_seq_client_info>() as u64) << 16i32) as u32;
__t6.func = Some(snd_seq_ioctl_set_client_info);
__t6
}, {
let mut __t7: ioctl_handler = unsafe { core::mem::zeroed() };
__t7.cmd = (3221246752u64 | (core::mem::size_of::<snd_seq_port_info>() as u64) << 16i32) as u32;
__t7.func = Some(snd_seq_ioctl_create_port);
__t7
}, {
let mut __t8: ioctl_handler = unsafe { core::mem::zeroed() };
__t8.cmd = (1073763105u64 | (core::mem::size_of::<snd_seq_port_info>() as u64) << 16i32) as u32;
__t8.func = Some(snd_seq_ioctl_delete_port);
__t8
}, {
let mut __t9: ioctl_handler = unsafe { core::mem::zeroed() };
__t9.cmd = (3221246754u64 | (core::mem::size_of::<snd_seq_port_info>() as u64) << 16i32) as u32;
__t9.func = Some(snd_seq_ioctl_get_port_info);
__t9
}, {
let mut __t10: ioctl_handler = unsafe { core::mem::zeroed() };
__t10.cmd = (1073763107u64 | (core::mem::size_of::<snd_seq_port_info>() as u64) << 16i32) as u32;
__t10.func = Some(snd_seq_ioctl_set_port_info);
__t10
}, {
let mut __t11: ioctl_handler = unsafe { core::mem::zeroed() };
__t11.cmd = (1073763120u64 | (core::mem::size_of::<snd_seq_port_subscribe>() as u64) << 16i32) as u32;
__t11.func = Some(snd_seq_ioctl_subscribe_port);
__t11
}, {
let mut __t12: ioctl_handler = unsafe { core::mem::zeroed() };
__t12.cmd = (1073763121u64 | (core::mem::size_of::<snd_seq_port_subscribe>() as u64) << 16i32) as u32;
__t12.func = Some(snd_seq_ioctl_unsubscribe_port);
__t12
}, {
let mut __t13: ioctl_handler = unsafe { core::mem::zeroed() };
__t13.cmd = (3221246770u64 | (core::mem::size_of::<snd_seq_queue_info>() as u64) << 16i32) as u32;
__t13.func = Some(snd_seq_ioctl_create_queue);
__t13
}, {
let mut __t14: ioctl_handler = unsafe { core::mem::zeroed() };
__t14.cmd = (1073763123u64 | (core::mem::size_of::<snd_seq_queue_info>() as u64) << 16i32) as u32;
__t14.func = Some(snd_seq_ioctl_delete_queue);
__t14
}, {
let mut __t15: ioctl_handler = unsafe { core::mem::zeroed() };
__t15.cmd = (3221246772u64 | (core::mem::size_of::<snd_seq_queue_info>() as u64) << 16i32) as u32;
__t15.func = Some(snd_seq_ioctl_get_queue_info);
__t15
}, {
let mut __t16: ioctl_handler = unsafe { core::mem::zeroed() };
__t16.cmd = (3221246773u64 | (core::mem::size_of::<snd_seq_queue_info>() as u64) << 16i32) as u32;
__t16.func = Some(snd_seq_ioctl_set_queue_info);
__t16
}, {
let mut __t17: ioctl_handler = unsafe { core::mem::zeroed() };
__t17.cmd = (3221246774u64 | (core::mem::size_of::<snd_seq_queue_info>() as u64) << 16i32) as u32;
__t17.func = Some(snd_seq_ioctl_get_named_queue);
__t17
}, {
let mut __t18: ioctl_handler = unsafe { core::mem::zeroed() };
__t18.cmd = (3221246784u64 | (core::mem::size_of::<snd_seq_queue_status>() as u64) << 16i32) as u32;
__t18.func = Some(snd_seq_ioctl_get_queue_status);
__t18
}, {
let mut __t19: ioctl_handler = unsafe { core::mem::zeroed() };
__t19.cmd = (3221246785u64 | (core::mem::size_of::<snd_seq_queue_tempo>() as u64) << 16i32) as u32;
__t19.func = Some(snd_seq_ioctl_get_queue_tempo);
__t19
}, {
let mut __t20: ioctl_handler = unsafe { core::mem::zeroed() };
__t20.cmd = (1073763138u64 | (core::mem::size_of::<snd_seq_queue_tempo>() as u64) << 16i32) as u32;
__t20.func = Some(snd_seq_ioctl_set_queue_tempo);
__t20
}, {
let mut __t21: ioctl_handler = unsafe { core::mem::zeroed() };
__t21.cmd = (3221246789u64 | (core::mem::size_of::<snd_seq_queue_timer>() as u64) << 16i32) as u32;
__t21.func = Some(snd_seq_ioctl_get_queue_timer);
__t21
}, {
let mut __t22: ioctl_handler = unsafe { core::mem::zeroed() };
__t22.cmd = (1073763142u64 | (core::mem::size_of::<snd_seq_queue_timer>() as u64) << 16i32) as u32;
__t22.func = Some(snd_seq_ioctl_set_queue_timer);
__t22
}, {
let mut __t23: ioctl_handler = unsafe { core::mem::zeroed() };
__t23.cmd = (3221246793u64 | (core::mem::size_of::<snd_seq_queue_client>() as u64) << 16i32) as u32;
__t23.func = Some(snd_seq_ioctl_get_queue_client);
__t23
}, {
let mut __t24: ioctl_handler = unsafe { core::mem::zeroed() };
__t24.cmd = (1073763146u64 | (core::mem::size_of::<snd_seq_queue_client>() as u64) << 16i32) as u32;
__t24.func = Some(snd_seq_ioctl_set_queue_client);
__t24
}, {
let mut __t25: ioctl_handler = unsafe { core::mem::zeroed() };
__t25.cmd = (3221246795u64 | (core::mem::size_of::<snd_seq_client_pool>() as u64) << 16i32) as u32;
__t25.func = Some(snd_seq_ioctl_get_client_pool);
__t25
}, {
let mut __t26: ioctl_handler = unsafe { core::mem::zeroed() };
__t26.cmd = (1073763148u64 | (core::mem::size_of::<snd_seq_client_pool>() as u64) << 16i32) as u32;
__t26.func = Some(snd_seq_ioctl_set_client_pool);
__t26
}, {
let mut __t27: ioctl_handler = unsafe { core::mem::zeroed() };
__t27.cmd = (3221246800u64 | (core::mem::size_of::<snd_seq_port_subscribe>() as u64) << 16i32) as u32;
__t27.func = Some(snd_seq_ioctl_get_subscription);
__t27
}, {
let mut __t28: ioctl_handler = unsafe { core::mem::zeroed() };
__t28.cmd = (3221246801u64 | (core::mem::size_of::<snd_seq_client_info>() as u64) << 16i32) as u32;
__t28.func = Some(snd_seq_ioctl_query_next_client);
__t28
}, {
let mut __t29: ioctl_handler = unsafe { core::mem::zeroed() };
__t29.cmd = (3221246802u64 | (core::mem::size_of::<snd_seq_port_info>() as u64) << 16i32) as u32;
__t29.func = Some(snd_seq_ioctl_query_next_port);
__t29
}, {
let mut __t30: ioctl_handler = unsafe { core::mem::zeroed() };
__t30.cmd = (1073763150u64 | (core::mem::size_of::<snd_seq_remove_events>() as u64) << 16i32) as u32;
__t30.func = Some(snd_seq_ioctl_remove_events);
__t30
}, {
let mut __t31: ioctl_handler = unsafe { core::mem::zeroed() };
__t31.cmd = (3221246799u64 | (core::mem::size_of::<snd_seq_query_subs>() as u64) << 16i32) as u32;
__t31.func = Some(snd_seq_ioctl_query_subs);
__t31
}, {
let mut __t32: ioctl_handler = unsafe { core::mem::zeroed() };
__t32.cmd = 0u32;
__t32.func = None;
__t32
}]);

#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_ioctl")]
extern "C" fn snd_seq_ioctl(file: *mut file, cmd: u32, arg: u64) -> i64 {
let mut __UNIQUE_ID_unlock_504: core::mem::MaybeUninit<*mut __SlateRecord_mutex_h_253> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut __SlateRecord_mutex_h_253 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_504) as *mut *mut __SlateRecord_mutex_h_253;
let mut scope: core::mem::MaybeUninit<__SlateRecord_mutex_h_253> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut __SlateRecord_mutex_h_253 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_mutex_h_253;
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut size: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut u64 = core::ptr::addr_of_mut!(size) as *mut u64;
let mut handler: core::mem::MaybeUninit<*const ioctl_handler> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *const ioctl_handler = core::ptr::addr_of_mut!(handler) as *mut *const ioctl_handler;
// To use kernel stack for ioctl data.
let mut buf: core::mem::MaybeUninit<__SlateRecord_seq_clientmgr_c_2260> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut __SlateRecord_seq_clientmgr_c_2260 = core::ptr::addr_of_mut!(buf) as *mut __SlateRecord_seq_clientmgr_c_2260;
let mut client: *mut snd_seq_client = unsafe { core::mem::zeroed() };
let __slate_cleanup_ptr_1 = __slate_slot_1;
let __slate_cleanup_armed_1 = core::cell::Cell::new(false);
let __slate_cleanup_1 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_1.get() {
unsafe { class_mutex_destructor(__slate_cleanup_ptr_1 as *mut _) };
}
});
let __slate_cleanup_ptr_0 = __slate_slot_0;
let __slate_cleanup_armed_0 = core::cell::Cell::new(false);
let __slate_cleanup_0 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_0.get() {
unsafe { __class_mutex_cleanup_ctx(__slate_cleanup_ptr_0 as *mut _) };
}
});
unsafe {
client = (*file).private_data as *mut snd_seq_client;
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 192usize);
let __slate_auto_bytes_1 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if snd_BUG_ON!(client.is_null()) != 0 {
return 6i32.wrapping_neg() as i64;
} else {
*__slate_slot_6 = core::ptr::addr_of!(ioctl_handlers.0) as *const ioctl_handler;
while (*(*__slate_slot_6)).cmd > 0 {
if (*(*__slate_slot_6)).cmd == cmd {
break;
} else {
let __v3: *const ioctl_handler = *__slate_slot_6;
*__slate_slot_6 = __v3.wrapping_offset(1 as isize);
}
}
if (*(*__slate_slot_6)).cmd == 0 {
return 25i32.wrapping_neg() as i64;
} else {
memset(core::ptr::addr_of_mut!(*__slate_slot_7) as *mut (), 0i32, core::mem::size_of::<__SlateRecord_seq_clientmgr_c_2260>() as u64);
// All of ioctl commands for ALSA sequencer get an argument of size
// within 13 bits. We can safely pick up the size from the command.
*__slate_slot_5 = _IOC_SIZE!(unsafe { (*(*__slate_slot_6)).cmd }) as u64;
if (*(*__slate_slot_6)).cmd & 1073741824u32 != 0 {
if copy_from_user(core::ptr::addr_of_mut!(*__slate_slot_7) as *mut (), arg as *const (), *__slate_slot_5) != 0 {
return 14i32.wrapping_neg() as i64;
}
}
core::ptr::write(__slate_slot_1, class_mutex_constructor(core::ptr::addr_of_mut!((*client).ioctl_mutex)));
__slate_cleanup_armed_1.set(true);
core::ptr::write(__slate_slot_0, (core::ptr::addr_of_mut!((*client).ioctl_mutex) as u64) as *mut __SlateRecord_mutex_h_253);
__slate_cleanup_armed_0.set(true);
if !class_mutex_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_1))).is_null() || !class_mutex_is_conditional {
if true {
*__slate_slot_4 = unsafe { (*(*__slate_slot_6)).func }.unwrap()(client, core::ptr::addr_of_mut!(*__slate_slot_7) as *mut ());
}
}
__slate_cleanup_armed_0.set(false);
__class_mutex_cleanup_ctx(__slate_slot_0 as *mut _);
__slate_cleanup_armed_1.set(false);
class_mutex_destructor(__slate_slot_1 as *mut _);
if *__slate_slot_4 >= 0 {
// Some commands includes a bug in 'dir' field.
if ((*(*__slate_slot_6)).cmd as u64) == _IOW!(83i32, 74i32, snd_seq_queue_client) || ((*(*__slate_slot_6)).cmd as u64) == _IOW!(83i32, 76i32, snd_seq_client_pool) || (*(*__slate_slot_6)).cmd & 2147483648u32 != 0 {
if copy_to_user(arg as *mut (), core::ptr::addr_of_mut!(*__slate_slot_7) as *const (), *__slate_slot_5) != 0 {
return 14i32.wrapping_neg() as i64;
}
}
}
return *__slate_slot_4 as i64;
}
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct snd_seq_port_info32 {
    addr: snd_seq_addr,
    name: [u8; 64],
    capability: u32,
    r#type: u32,
    midi_channels: i32,
    midi_voices: i32,
    synth_voices: i32,
    read_use: i32,
    write_use: i32,
    kernel: u32,
    flags: u32,
    time_queue: u8,
    reserved: [u8; 59],
}


fn snd_seq_call_port_info_ioctl(client: *mut snd_seq_client, cmd: u32, mut data32: *mut snd_seq_port_info32) -> i32 {
let mut __slate_compound_40: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_42: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __x: u8 = unsafe { core::mem::zeroed() };
let mut __val_pu: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut u8 = core::ptr::addr_of_mut!(__val_pu) as *mut u8;
let mut __ptr_pu: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut () = core::ptr::addr_of_mut!(__ptr_pu) as *mut *mut ();
let mut __ret_pu: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(__ret_pu) as *mut i32;
let mut __v7: bool = unsafe { core::mem::zeroed() };
let mut __x_1: u32 = unsafe { core::mem::zeroed() };
let mut __val_pu: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut u32 = core::ptr::addr_of_mut!(__val_pu) as *mut u32;
let mut __ptr_pu: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut () = core::ptr::addr_of_mut!(__ptr_pu) as *mut *mut ();
let mut __ret_pu: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut i32 = core::ptr::addr_of_mut!(__ret_pu) as *mut i32;
let mut __v15: bool = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_unlock_506: core::mem::MaybeUninit<*mut __SlateRecord_mutex_h_253> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut *mut __SlateRecord_mutex_h_253 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_506) as *mut *mut __SlateRecord_mutex_h_253;
let mut scope: core::mem::MaybeUninit<__SlateRecord_mutex_h_253> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut __SlateRecord_mutex_h_253 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_mutex_h_253;
let mut __val_gu: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut u8 = core::ptr::addr_of_mut!(__val_gu) as *mut u8;
let mut __ret_gu: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut i32 = core::ptr::addr_of_mut!(__ret_gu) as *mut i32;
let mut __v22: bool = unsafe { core::mem::zeroed() };
let mut __val_gu: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut u32 = core::ptr::addr_of_mut!(__val_gu) as *mut u32;
let mut __ret_gu: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut i32 = core::ptr::addr_of_mut!(__ret_gu) as *mut i32;
let mut __v27: bool = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_28: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_29: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut data: core::mem::MaybeUninit<*mut snd_seq_port_info> = core::mem::MaybeUninit::uninit();
let __slate_slot_34: *mut *mut snd_seq_port_info = core::ptr::addr_of_mut!(data) as *mut *mut snd_seq_port_info;
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_35: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let __slate_cleanup_ptr_34 = __slate_slot_34;
let __slate_cleanup_armed_34 = core::cell::Cell::new(false);
let __slate_cleanup_34 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_34.get() {
unsafe { __free_kfree(__slate_cleanup_ptr_34 as *mut _) };
}
});
let __slate_cleanup_ptr_17 = __slate_slot_17;
let __slate_cleanup_armed_17 = core::cell::Cell::new(false);
let __slate_cleanup_17 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_17.get() {
unsafe { class_mutex_destructor(__slate_cleanup_ptr_17 as *mut _) };
}
});
let __slate_cleanup_ptr_16 = __slate_slot_16;
let __slate_cleanup_armed_16 = core::cell::Cell::new(false);
let __slate_cleanup_16 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_16.get() {
unsafe { __class_mutex_cleanup_ctx(__slate_cleanup_ptr_16 as *mut _) };
}
});
unsafe {
let __slate_auto_bytes_6 = __slate_slot_35 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
__slate_cleanup_armed_34.set(true);
__obj_size = size_mul(core::mem::size_of::<snd_seq_port_info>() as u64, 1u64);
let __slate_auto_bytes_7 = __slate_slot_29 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_28 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_28 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_29 = _kmalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_40 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_40)
});
alloc_tag_restore!();
} else {
*__slate_slot_29 = _kmalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_42 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_42)
});
}
*__slate_slot_34 = *__slate_slot_29 as *mut snd_seq_port_info;
if (*__slate_slot_34).is_null() {
return 12i32.wrapping_neg();
} else {
if copy_from_user(*__slate_slot_34 as *mut (), data32 as *const (), core::mem::size_of::<snd_seq_port_info32>() as u64) != 0 {
__v27 = true;
} else {
might_fault();
let __slate_auto_bytes_5 = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_23 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
__chk_user_ptr!();
let __t4: u64;
let __t5: u32;
core::arch::asm!("call __get_user_{0}", const 4 as u64, inlateout("ax") core::ptr::addr_of_mut!((*data32).flags) as u64 => __t4, lateout("dx") __t5, options(att_syntax));
*__slate_slot_24 = unsafe { core::mem::transmute::<u32, i32>(__t4 as u32) };
*__slate_slot_23 = unsafe { core::mem::transmute::<u32, u32>(__t5 as u32) };
instrument_get_user!(*__slate_slot_23);
(*(*__slate_slot_34)).flags = *__slate_slot_23;
__v27 = (*__slate_slot_24 as i64) != 0;
}
if __v27 {
__v22 = true;
} else {
might_fault();
let __slate_auto_bytes_4 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
__chk_user_ptr!();
let __t2: u64;
let __t3: u32;
core::arch::asm!("call __get_user_{0}", const 1 as u64, inlateout("ax") core::ptr::addr_of_mut!((*data32).time_queue) as u64 => __t2, lateout("dx") __t3, options(att_syntax));
*__slate_slot_19 = unsafe { core::mem::transmute::<u32, i32>(__t2 as u32) };
*__slate_slot_18 = unsafe { core::mem::transmute::<u8, u8>(__t3 as u8) };
instrument_get_user__3!(*__slate_slot_18);
(*(*__slate_slot_34)).time_queue = *__slate_slot_18;
__v22 = (*__slate_slot_19 as i64) != 0;
}
if __v22 {
return 14i32.wrapping_neg();
} else {
(*(*__slate_slot_34)).kernel = core::ptr::null_mut::<()>();
core::ptr::write(__slate_slot_17, class_mutex_constructor(core::ptr::addr_of_mut!((*client).ioctl_mutex)));
__slate_cleanup_armed_17.set(true);
core::ptr::write(__slate_slot_16, (core::ptr::addr_of_mut!((*client).ioctl_mutex) as u64) as *mut __SlateRecord_mutex_h_253);
__slate_cleanup_armed_16.set(true);
if !class_mutex_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_17))).is_null() || !class_mutex_is_conditional {
if true {
*__slate_slot_35 = snd_seq_kernel_client_ctl((*client).number, cmd, *__slate_slot_34 as *mut ());
}
}
__slate_cleanup_armed_16.set(false);
__class_mutex_cleanup_ctx(__slate_slot_16 as *mut _);
__slate_cleanup_armed_17.set(false);
class_mutex_destructor(__slate_slot_17 as *mut _);
if *__slate_slot_35 < 0 {
return *__slate_slot_35;
} else {
if copy_to_user(data32 as *mut (), *__slate_slot_34 as *const (), core::mem::size_of::<snd_seq_port_info32>() as u64) != 0 {
__v15 = true;
} else {
might_fault();
let __slate_auto_bytes_2 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
__x_1 = (*(*__slate_slot_34)).flags;
let __ptr: *mut u32 = core::ptr::addr_of_mut!((*data32).flags);
__chk_user_ptr!();
*__slate_slot_11 = __ptr as *mut ();
*__slate_slot_10 = __x_1;
let __t1: u64;
core::arch::asm!("mov %rbx, {1:r}\ncall __put_user_{0}\nmov {1:r}, %rbx", const 4 as u64, out(reg) _, inlateout("cx") *__slate_slot_11 as u64 => __t1, in("ax") *__slate_slot_10 as u32, options(att_syntax));
*__slate_slot_12 = unsafe { core::mem::transmute::<u32, i32>(__t1 as u32) };
instrument_put_user__2!(__x_1, __ptr);
__v15 = (*__slate_slot_12 as i64) != 0;
}
if __v15 {
__v7 = true;
} else {
might_fault();
let __slate_auto_bytes_8 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
__x = (*(*__slate_slot_34)).time_queue;
let __ptr: *mut u8 = core::ptr::addr_of_mut!((*data32).time_queue);
__chk_user_ptr!();
*__slate_slot_3 = __ptr as *mut ();
*__slate_slot_2 = __x;
let __t0: u64;
core::arch::asm!("mov %rbx, {1:r}\ncall __put_user_{0}\nmov {1:r}, %rbx", const 1 as u64, out(reg) _, inlateout("cx") *__slate_slot_3 as u64 => __t0, in("ax") (*__slate_slot_2 as u8) as u32, options(att_syntax));
*__slate_slot_4 = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
instrument_put_user__5!(__x, __ptr);
__v7 = (*__slate_slot_4 as i64) != 0;
}
if __v7 {
return 14i32.wrapping_neg();
} else {
return *__slate_slot_35;
}
}
}
}
}
}

#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_ioctl_compat")]
extern "C" fn snd_seq_ioctl_compat(file: *mut file, cmd: u32, arg: u64) -> i64 {
let client: *mut snd_seq_client = (unsafe { (*file).private_data }) as *mut snd_seq_client;
let argp: *mut () = compat_ptr(arg as u32);
if snd_BUG_ON!(client.is_null()) != 0 {
return 6i32.wrapping_neg() as i64;
}
match cmd {
1074025220 | 1076646722 | 1077957454 | 1078743882 | 1079006000 | 1079006001 | 1079530316 | 1080054598 | 1082938163 | 1086083857 | 2147767040 | 2147767041 | 3222295299 | 3224130369 | 3224392450 | 3226227529 | 3226489680 | 3227013963 | 3227013967 | 3227276096 | 3227538245 | 3230421810 | 3230421812..=3230421814 | 3233567504 | 3233567569 | 3255325458 | 3255325459 => {
return snd_seq_ioctl(file, cmd, arg);
}
3231994656 => {
return snd_seq_call_port_info_ioctl(client, _IOWR!(83i32, 32i32, snd_seq_port_info) as u32, argp as *mut snd_seq_port_info32) as i64;
}
1084511009 => {
return snd_seq_call_port_info_ioctl(client, _IOW!(83i32, 33i32, snd_seq_port_info) as u32, argp as *mut snd_seq_port_info32) as i64;
}
3231994658 => {
return snd_seq_call_port_info_ioctl(client, _IOWR!(83i32, 34i32, snd_seq_port_info) as u32, argp as *mut snd_seq_port_info32) as i64;
}
1084511011 => {
return snd_seq_call_port_info_ioctl(client, _IOW!(83i32, 35i32, snd_seq_port_info) as u32, argp as *mut snd_seq_port_info32) as i64;
}
3231994706 => {
return snd_seq_call_port_info_ioctl(client, _IOWR!(83i32, 82i32, snd_seq_port_info) as u32, argp as *mut snd_seq_port_info32) as i64;
}
_ => {
}
}
515i32.wrapping_neg() as i64
}

/// exported to kernel modules
#[unsafe(no_mangle)]
unsafe extern "C" fn snd_seq_create_kernel_client(card: *mut snd_card, mut client_index: i32, name_fmt: *const u8, mut __va_args: ...) -> i32 {
let mut __UNIQUE_ID_unlock_508: core::mem::MaybeUninit<*mut __SlateRecord_mutex_h_253> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut __SlateRecord_mutex_h_253 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_508) as *mut *mut __SlateRecord_mutex_h_253;
let mut scope: core::mem::MaybeUninit<__SlateRecord_mutex_h_253> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut __SlateRecord_mutex_h_253 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_mutex_h_253;
let mut args: core::mem::MaybeUninit<core::ffi::VaList<'static>> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut core::ffi::VaList<'static> = core::ptr::addr_of_mut!(args) as *mut core::ffi::VaList<'static>;
let mut client: core::mem::MaybeUninit<*mut snd_seq_client> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut snd_seq_client = core::ptr::addr_of_mut!(client) as *mut *mut snd_seq_client;
let __slate_cleanup_ptr_3 = __slate_slot_3;
let __slate_cleanup_armed_3 = core::cell::Cell::new(false);
let __slate_cleanup_3 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_3.get() {
unsafe { class_mutex_destructor(__slate_cleanup_ptr_3 as *mut _) };
}
});
let __slate_cleanup_ptr_2 = __slate_slot_2;
let __slate_cleanup_armed_2 = core::cell::Cell::new(false);
let __slate_cleanup_2 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_2.get() {
unsafe { __class_mutex_cleanup_ctx(__slate_cleanup_ptr_2 as *mut _) };
}
});
unsafe {
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 24usize);
if snd_BUG_ON__4!(in_interrupt!()) != 0 {
return 16i32.wrapping_neg();
} else {
if !card.is_null() && client_index >= 4 {
return 22i32.wrapping_neg();
} else {
if card.is_null() && client_index >= 16 {
return 22i32.wrapping_neg();
} else {
core::ptr::write(__slate_slot_3, class_mutex_constructor(core::ptr::addr_of_mut!(register_mutex)));
__slate_cleanup_armed_3.set(true);
core::ptr::write(__slate_slot_2, (core::ptr::addr_of_mut!(register_mutex) as u64) as *mut __SlateRecord_mutex_h_253);
__slate_cleanup_armed_2.set(true);
if !class_mutex_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_3))).is_null() || !class_mutex_is_conditional {
if true {
if !card.is_null() {
client_index = client_index.wrapping_add(16i32.wrapping_add(unsafe { (*card).number }.wrapping_mul(4)));
if client_index >= 128 {
client_index = 1i32.wrapping_neg();
}
}
// empty write queue as default
*__slate_slot_5 = seq_create_client1(client_index, 0i32);
if (*__slate_slot_5).is_null() {
return 16i32.wrapping_neg();
} else {
// failure code
usage_alloc(core::ptr::addr_of_mut!(client_usage), 1i32);
(*(*__slate_slot_5)).__slate_bits_0.__set_accept_input(1u32);
(*(*__slate_slot_5)).__slate_bits_0.__set_accept_output(1u32);
(*(*__slate_slot_5)).data.kernel.card = card;
(*(*__slate_slot_5)).user_pversion = SNDRV_PROTOCOL_VERSION!(1i32, 0i32, 5i32) as u32;
*__slate_slot_4 = core::mem::transmute::<core::ffi::VaList<'_>, core::ffi::VaList<'static>>(__va_args.clone());
vsnprintf((*(*__slate_slot_5)).name.as_mut_ptr() as *mut u8, core::mem::size_of::<[u8; 64]>() as u64, name_fmt, (*__slate_slot_4).clone());
va_end!();
(*(*__slate_slot_5)).r#type = 2;
}
}
}
__slate_cleanup_armed_2.set(false);
__class_mutex_cleanup_ctx(__slate_slot_2 as *mut _);
__slate_cleanup_armed_3.set(false);
class_mutex_destructor(__slate_slot_3 as *mut _);
// make others aware this new client
snd_seq_system_client_ev_client_start!(unsafe { (*(*__slate_slot_5)).number });
// return client number to caller
return (*(*__slate_slot_5)).number;
}
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_seq_create_kernel_client_509: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_card, i32, *const u8, ...) -> i32>, *mut ()>(Some(snd_seq_create_kernel_client)) };

/// exported to kernel modules
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_delete_kernel_client")]
extern "C" fn snd_seq_delete_kernel_client(client: i32) -> i32 {
if snd_BUG_ON__4!(in_interrupt!()) != 0 {
return 16i32.wrapping_neg();
}
let ptr: *mut snd_seq_client = clientptr(client);
if ptr.is_null() {
return 22i32.wrapping_neg();
}
seq_free_client(ptr);
unsafe { kfree(ptr as *const ()) };
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_seq_delete_kernel_client_510: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(i32) -> i32>, *mut ()>(Some(snd_seq_delete_kernel_client)) };

/// exported, called by kernel clients to enqueue events (w/o blocking)
///
/// RETURN VALUE: zero if succeed, negative if error
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_kernel_client_enqueue")]
extern "C" fn snd_seq_kernel_client_enqueue(client: i32, ev: *mut snd_seq_event, file: *mut file, blocking: bool) -> i32 {
if snd_BUG_ON!(ev.is_null()) != 0 {
return 22i32.wrapping_neg();
}
if !snd_seq_ev_is_ump!(ev) {
if ((unsafe { (*ev).r#type }) as i32) == 255 {
return 0;
}
// ignore this
if ((unsafe { (*ev).r#type }) as i32) == 150 {
return 22i32.wrapping_neg();
}
// quoted events can't be enqueued
}
// fill in client number
unsafe {
(*ev).source.client = client as u8;
}
if check_event_type_and_length(ev) != 0 {
return 22i32.wrapping_neg();
}
let mut cptr: *mut snd_seq_client = client_load_and_use_ptr(client);
let __slate_cleanup_ptr_2 = core::ptr::addr_of_mut!(cptr) as *mut _;
let __slate_cleanup_2 = ::slate_support::Guard(&mut || __free_snd_seq_client(__slate_cleanup_ptr_2 as *mut _));
if cptr.is_null() {
return 22i32.wrapping_neg();
}
if ((unsafe { (*cptr).__slate_bits_0.__get_accept_output() }) as i32) == 0 {
return 1i32.wrapping_neg();
} else {
// send it
let mut __UNIQUE_ID_guard_511: __SlateRecord_mutex_h_253 = class_mutex_constructor(unsafe { core::ptr::addr_of_mut!((*cptr).ioctl_mutex) });
let __slate_cleanup_ptr_3 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_511) as *mut _;
let __slate_cleanup_3 = ::slate_support::Guard(&mut || class_mutex_destructor(__slate_cleanup_ptr_3 as *mut _));
let mut __UNIQUE_ID_unlock_512: *mut __SlateRecord_mutex_h_253 = ((unsafe { core::ptr::addr_of_mut!((*cptr).ioctl_mutex) }) as u64) as *mut __SlateRecord_mutex_h_253;
let __slate_cleanup_ptr_4 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_512) as *mut _;
let __slate_cleanup_4 = ::slate_support::Guard(&mut || __class_mutex_cleanup_ctx(__slate_cleanup_ptr_4 as *mut _));
return snd_seq_client_enqueue_event(cptr, ev, file, blocking as i32, 0i32, 0i32, unsafe { core::ptr::addr_of_mut!((*cptr).ioctl_mutex) });
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_seq_kernel_client_enqueue_513: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(i32, *mut snd_seq_event, *mut file, bool) -> i32>, *mut ()>(Some(snd_seq_kernel_client_enqueue)) };

/// exported, called by kernel clients to dispatch events directly to other
/// clients, bypassing the queues.  Event time-stamp will be updated.
///
/// RETURN VALUE: negative = delivery failed,
/// 		 zero, or positive: the number of delivered events
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_kernel_client_dispatch")]
extern "C" fn snd_seq_kernel_client_dispatch(client: i32, ev: *mut snd_seq_event, atomic: i32, hop: i32) -> i32 {
if snd_BUG_ON!(ev.is_null()) != 0 {
return 22i32.wrapping_neg();
}
// fill in client number
unsafe {
(*ev).queue = 253;
}
unsafe {
(*ev).source.client = client as u8;
}
if check_event_type_and_length(ev) != 0 {
return 22i32.wrapping_neg();
}
let mut cptr: *mut snd_seq_client = snd_seq_client_use_ptr(client);
let __slate_cleanup_ptr_2 = core::ptr::addr_of_mut!(cptr) as *mut _;
let __slate_cleanup_2 = ::slate_support::Guard(&mut || __free_snd_seq_client(__slate_cleanup_ptr_2 as *mut _));
if cptr.is_null() {
return 22i32.wrapping_neg();
}
if ((unsafe { (*cptr).__slate_bits_0.__get_accept_output() }) as i32) == 0 {
return 1i32.wrapping_neg();
} else {
return snd_seq_deliver_event(cptr, ev, atomic, hop);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_seq_kernel_client_dispatch_514: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(i32, *mut snd_seq_event, i32, i32) -> i32>, *mut ()>(Some(snd_seq_kernel_client_dispatch)) };

fn call_seq_client_ctl(client: *mut snd_seq_client, cmd: u32, arg: *mut ()) -> i32 {
let mut handler: *const ioctl_handler = unsafe { core::mem::zeroed() };
handler = unsafe { core::ptr::addr_of!(ioctl_handlers.0) as *const ioctl_handler };
while (unsafe { (*handler).cmd }) > 0 {
if (unsafe { (*handler).cmd }) == cmd {
return unsafe { unsafe { (*handler).func }.unwrap()(client, arg) };
}
let __v1: *const ioctl_handler = handler;
handler = __v1.wrapping_offset(1 as isize);
}
let mut __v3: i32;
if false {
unsafe { _printk(c"\x017ALSA: seq unknown ioctl() 0x%x (type='%c', number=0x%02x)\n".as_ptr() as *const u8, cmd, _IOC_TYPE!(cmd), _IOC_NR!(cmd)) };
}
__v3 = 0;
25i32.wrapping_neg()
}

/// snd_seq_kernel_client_ctl - operate a command for a client with data in
/// 			       kernel space.
/// @clientid:	A numerical ID for a client.
/// @cmd:	An ioctl(2) command for ALSA sequencer operation.
/// @arg:	A pointer to data in kernel space.
///
/// Against its name, both kernel/application client can be handled by this
/// kernel API. A pointer of 'arg' argument should be in kernel space.
///
/// Return: 0 at success. Negative error code at failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_kernel_client_ctl")]
extern "C" fn snd_seq_kernel_client_ctl(clientid: i32, cmd: u32, arg: *mut ()) -> i32 {
let client: *mut snd_seq_client = clientptr(clientid);
if client.is_null() {
return 6i32.wrapping_neg();
}
call_seq_client_ctl(client, cmd, arg)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_seq_kernel_client_ctl_515: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(i32, u32, *mut ()) -> i32>, *mut ()>(Some(snd_seq_kernel_client_ctl)) };

/// a similar like above but taking locks; used only from OSS sequencer layer
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_kernel_client_ioctl")]
extern "C" fn snd_seq_kernel_client_ioctl(clientid: i32, cmd: u32, arg: *mut ()) -> i32 {
let mut client: *mut snd_seq_client = client_load_and_use_ptr(clientid);
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(client) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || __free_snd_seq_client(__slate_cleanup_ptr_0 as *mut _));
if client.is_null() {
return 6i32.wrapping_neg();
}
let mut __UNIQUE_ID_guard_516: __SlateRecord_mutex_h_253 = class_mutex_constructor(unsafe { core::ptr::addr_of_mut!((*client).ioctl_mutex) });
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_516) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || class_mutex_destructor(__slate_cleanup_ptr_1 as *mut _));
let mut __UNIQUE_ID_unlock_517: *mut __SlateRecord_mutex_h_253 = ((unsafe { core::ptr::addr_of_mut!((*client).ioctl_mutex) }) as u64) as *mut __SlateRecord_mutex_h_253;
let __slate_cleanup_ptr_2 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_517) as *mut _;
let __slate_cleanup_2 = ::slate_support::Guard(&mut || __class_mutex_cleanup_ctx(__slate_cleanup_ptr_2 as *mut _));
call_seq_client_ctl(client, cmd, arg)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_seq_kernel_client_ioctl_518: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(i32, u32, *mut ()) -> i32>, *mut ()>(Some(snd_seq_kernel_client_ioctl)) };

/// exported (for OSS emulator)
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_kernel_client_write_poll")]
extern "C" fn snd_seq_kernel_client_write_poll(clientid: i32, file: *mut file, wait: *mut poll_table_struct) -> i32 {
let client: *mut snd_seq_client = clientptr(clientid);
if client.is_null() {
return 6i32.wrapping_neg();
}
if (unsafe { snd_seq_pool_poll_wait((*client).pool, file, wait) }) != 0 {
return 1;
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_seq_kernel_client_write_poll_519: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(i32, *mut file, *mut poll_table_struct) -> i32>, *mut ()>(Some(snd_seq_kernel_client_write_poll)) };

/// get a sequencer client object; for internal use from a kernel client
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_kernel_client_get")]
extern "C" fn snd_seq_kernel_client_get(id: i32) -> *mut snd_seq_client {
snd_seq_client_use_ptr(id)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_seq_kernel_client_get_520: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(i32) -> *mut snd_seq_client>, *mut ()>(Some(snd_seq_kernel_client_get)) };

/// put a sequencer client object; for internal use from a kernel client
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_kernel_client_put")]
extern "C" fn snd_seq_kernel_client_put(cptr: *mut snd_seq_client) {
if !cptr.is_null() {
snd_seq_client_unref(cptr);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_snd_seq_kernel_client_put_521: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut snd_seq_client)>, *mut ()>(Some(snd_seq_kernel_client_put)) };

/// /proc interface
fn snd_seq_info_dump_subscribers(buffer: *mut snd_info_buffer, group: *mut snd_seq_port_subs_info, is_src: i32, msg: *mut u8) {
let mut p: *mut hlist_node = unsafe { core::mem::zeroed() };
let mut s: *mut snd_seq_subscribers = unsafe { core::mem::zeroed() };
let mut count: i32 = 0;
let mut __UNIQUE_ID_guard_522: __SlateRecord_rwsem_h_259 = class_rwsem_read_constructor(unsafe { core::ptr::addr_of_mut!((*group).list_mutex) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_522) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_rwsem_read_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_523: *mut __SlateRecord_rwsem_h_259 = ((unsafe { core::ptr::addr_of_mut!((*group).list_mutex) }) as u64) as *mut __SlateRecord_rwsem_h_259;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_523) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_rwsem_read_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
if hlist_empty((unsafe { core::ptr::addr_of_mut!((*group).list_head) }) as *const hlist_head) != 0 {
return;
}
unsafe { seq_printf((*buffer).buffer as *mut seq_file, msg as *const u8) };
p = unsafe { (*core::ptr::addr_of_mut!((*group).list_head)).first };
while !p.is_null() {
if is_src != 0 {
s = hlist_entry!(p, snd_seq_subscribers);
} else {
s = hlist_entry__3!(p, snd_seq_subscribers);
}
let __v4: i32 = count;
count = __v4.wrapping_add(1);
if __v4 != 0 {
unsafe { seq_printf((*buffer).buffer as *mut seq_file, c", ".as_ptr() as *const u8) };
}
unsafe { seq_printf((*buffer).buffer as *mut seq_file, c"%d:%d".as_ptr() as *const u8, if is_src != 0 { (*s).info.dest.client as i32 } else { (*s).info.sender.client as i32 }, if is_src != 0 { (*s).info.dest.port as i32 } else { (*s).info.sender.port as i32 }) };
if (unsafe { (*s).info.flags }) & 2u32 != 0 {
unsafe { seq_printf((*buffer).buffer as *mut seq_file, c"[%c:%d]".as_ptr() as *const u8, if (*s).info.flags & 4u32 != 0 { 114i32 } else { 116i32 }, (*s).info.queue as i32) };
}
if ((unsafe { (*group).__slate_bits_0.__get_exclusive() }) as i32) != 0 {
unsafe { seq_printf((*buffer).buffer as *mut seq_file, c"[ex]".as_ptr() as *const u8) };
}
p = unsafe { (*p).next };
}
unsafe { seq_printf((*buffer).buffer as *mut seq_file, c"\n".as_ptr() as *const u8) };
}

fn port_direction_name(dir: u8) -> *const u8 {
if (dir as i32) > 3 {
return c"Invalid".as_ptr() as *const u8;
}
unsafe { names.0[(dir as i32) as usize] }
}

static mut names: __SlateAlign16<[*const u8; 4]> = __SlateAlign16([(b"-\0".as_ptr() as *mut u8) as *const u8, (b"In\0".as_ptr() as *mut u8) as *const u8, (b"Out\0".as_ptr() as *mut u8) as *const u8, (b"In/Out\0".as_ptr() as *mut u8) as *const u8]);

fn snd_seq_info_dump_ports(buffer: *mut snd_info_buffer, client: *mut snd_seq_client) {
let mut p: *mut snd_seq_client_port = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_guard_524: __SlateRecord_mutex_h_253 = class_mutex_constructor(unsafe { core::ptr::addr_of_mut!((*client).ports_mutex) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_524) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_mutex_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_525: *mut __SlateRecord_mutex_h_253 = ((unsafe { core::ptr::addr_of_mut!((*client).ports_mutex) }) as u64) as *mut __SlateRecord_mutex_h_253;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_525) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_mutex_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
p = list_first_entry__9!(unsafe { core::ptr::addr_of_mut!((*client).ports_list_head) });
while list_entry_is_head!(p, unsafe { core::ptr::addr_of_mut!((*client).ports_list_head) }, list) == 0 {
if (unsafe { (*p).capability }) & 256u32 == 0 {
unsafe { seq_printf((*buffer).buffer as *mut seq_file, c"  Port %3d : \"%s\" (%c%c%c%c) [%s]".as_ptr() as *const u8, (*p).addr.port as i32, (*p).name.as_mut_ptr() as *mut u8, FLAG_PERM_RD!(unsafe { (*p).capability }), FLAG_PERM_WR!(unsafe { (*p).capability }), FLAG_PERM_EX!(unsafe { (*p).capability }), FLAG_PERM_DUPLEX!(unsafe { (*p).capability }), port_direction_name((*p).direction)) };
unsafe { seq_printf((*buffer).buffer as *mut seq_file, c"\n".as_ptr() as *const u8) };
snd_seq_info_dump_subscribers(buffer, unsafe { core::ptr::addr_of_mut!((*p).c_src) }, 1i32, c"    Connecting To: ".as_ptr() as *mut u8);
snd_seq_info_dump_subscribers(buffer, unsafe { core::ptr::addr_of_mut!((*p).c_dest) }, 0i32, c"    Connected From: ".as_ptr() as *mut u8);
}
p = list_next_entry__6!(p, list);
}
}

fn midi_version_string(version: u32) -> *const u8 {
match version {
0 => {
return c"Legacy".as_ptr() as *const u8;
}
1 => {
return c"UMP MIDI1".as_ptr() as *const u8;
}
2 => {
return c"UMP MIDI2".as_ptr() as *const u8;
}
_ => {
return c"Unknown".as_ptr() as *const u8;
}
}
unsafe { core::mem::zeroed() }
}

/// exported to seq_info.c
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.seq_clientmgr.snd_seq_info_clients_read")]
extern "C" fn snd_seq_info_clients_read(entry: *mut snd_info_entry, buffer: *mut snd_info_buffer) {
unsafe { seq_printf((*buffer).buffer as *mut seq_file, c"Client info\n".as_ptr() as *const u8) };
unsafe { seq_printf((*buffer).buffer as *mut seq_file, c"  cur  clients : %d\n".as_ptr() as *const u8, client_usage.cur) };
unsafe { seq_printf((*buffer).buffer as *mut seq_file, c"  peak clients : %d\n".as_ptr() as *const u8, client_usage.peak) };
unsafe { seq_printf((*buffer).buffer as *mut seq_file, c"  max  clients : %d\n".as_ptr() as *const u8, 192i32) };
unsafe { seq_printf((*buffer).buffer as *mut seq_file, c"\n".as_ptr() as *const u8) };
// list the client table
for c in 0i32..192i32 {
let mut client: *mut snd_seq_client = client_load_and_use_ptr(c);
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(client) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || __free_snd_seq_client(__slate_cleanup_ptr_0 as *mut _));
if !client.is_null() {
if (unsafe { (*client).r#type }) != 0 {
let mut __UNIQUE_ID_guard_526: __SlateRecord_mutex_h_253 = class_mutex_constructor(unsafe { core::ptr::addr_of_mut!((*client).ioctl_mutex) });
let __slate_cleanup_ptr_2 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_526) as *mut _;
let __slate_cleanup_2 = ::slate_support::Guard(&mut || class_mutex_destructor(__slate_cleanup_ptr_2 as *mut _));
let mut __UNIQUE_ID_unlock_527: *mut __SlateRecord_mutex_h_253 = ((unsafe { core::ptr::addr_of_mut!((*client).ioctl_mutex) }) as u64) as *mut __SlateRecord_mutex_h_253;
let __slate_cleanup_ptr_3 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_527) as *mut _;
let __slate_cleanup_3 = ::slate_support::Guard(&mut || __class_mutex_cleanup_ctx(__slate_cleanup_ptr_3 as *mut _));
unsafe { seq_printf((*buffer).buffer as *mut seq_file, c"Client %3d : \"%s\" [%s %s]\n".as_ptr() as *const u8, c, (*client).name.as_mut_ptr() as *mut u8, if (*client).r#type == 1 { c"User".as_ptr() as *mut u8 } else { c"Kernel".as_ptr() as *mut u8 }, midi_version_string((*client).midi_version)) };
snd_seq_info_dump_ports(buffer, client);
if snd_seq_write_pool_allocated(client) != 0 {
unsafe { seq_printf((*buffer).buffer as *mut seq_file, c"  Output pool :\n".as_ptr() as *const u8) };
unsafe { snd_seq_info_pool(buffer, (*client).pool, c"    ".as_ptr() as *mut u8) };
}
if (unsafe { (*client).r#type }) == 1 && !unsafe { (*client).data.user.fifo }.is_null() && !unsafe { (*(*client).data.user.fifo).pool }.is_null() {
unsafe { seq_printf((*buffer).buffer as *mut seq_file, c"  Input pool :\n".as_ptr() as *const u8) };
unsafe { snd_seq_info_pool(buffer, (*(*client).data.user.fifo).pool, c"    ".as_ptr() as *mut u8) };
}
}
}
}
}

// REGISTRATION PART
static snd_seq_f_ops: __SlateConst<file_operations> = __SlateConst({
let mut __t0: file_operations = unsafe { core::mem::zeroed() };
__t0.owner = core::ptr::null_mut::<module>();
__t0.read = Some(snd_seq_read);
__t0.write = Some(snd_seq_write);
__t0.poll = Some(snd_seq_poll);
__t0.unlocked_ioctl = Some(snd_seq_ioctl);
__t0.compat_ioctl = Some(snd_seq_ioctl_compat);
__t0.open = Some(snd_seq_open);
__t0.release = Some(snd_seq_release);
__t0
});

static mut seq_dev: *mut device = unsafe { core::mem::zeroed() };

/// register sequencer device
#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn snd_sequencer_device_init() -> i32 {
let mut __UNIQUE_ID_unlock_529: core::mem::MaybeUninit<*mut __SlateRecord_mutex_h_253> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut __SlateRecord_mutex_h_253 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_529) as *mut *mut __SlateRecord_mutex_h_253;
let mut scope: core::mem::MaybeUninit<__SlateRecord_mutex_h_253> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut __SlateRecord_mutex_h_253 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_mutex_h_253;
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let __slate_cleanup_ptr_1 = __slate_slot_1;
let __slate_cleanup_armed_1 = core::cell::Cell::new(false);
let __slate_cleanup_1 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_1.get() {
unsafe { class_mutex_destructor(__slate_cleanup_ptr_1 as *mut _) };
}
});
let __slate_cleanup_ptr_0 = __slate_slot_0;
let __slate_cleanup_armed_0 = core::cell::Cell::new(false);
let __slate_cleanup_0 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_0.get() {
unsafe { __class_mutex_cleanup_ctx(__slate_cleanup_ptr_0 as *mut _) };
}
});
unsafe {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_2 = snd_device_alloc(core::ptr::addr_of_mut!(seq_dev), core::ptr::null_mut::<snd_card>());
if *__slate_slot_2 < 0 {
return *__slate_slot_2;
} else {
dev_set_name(seq_dev, c"seq".as_ptr() as *const u8);
core::ptr::write(__slate_slot_1, class_mutex_constructor(core::ptr::addr_of_mut!(register_mutex)));
__slate_cleanup_armed_1.set(true);
core::ptr::write(__slate_slot_0, (core::ptr::addr_of_mut!(register_mutex) as u64) as *mut __SlateRecord_mutex_h_253);
__slate_cleanup_armed_0.set(true);
if !class_mutex_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_1))).is_null() || !class_mutex_is_conditional {
if true {
*__slate_slot_2 = snd_register_device(1i32, core::ptr::null_mut::<snd_card>(), 0i32, core::ptr::addr_of!(snd_seq_f_ops.0), core::ptr::null_mut::<()>(), seq_dev);
}
}
__slate_cleanup_armed_0.set(false);
__class_mutex_cleanup_ctx(__slate_slot_0 as *mut _);
__slate_cleanup_armed_1.set(false);
class_mutex_destructor(__slate_slot_1 as *mut _);
if *__slate_slot_2 < 0 {
put_device(seq_dev);
return *__slate_slot_2;
} else {
return 0;
}
}
}
}

/// unregister sequencer device
#[unsafe(no_mangle)]
extern "C" fn snd_sequencer_device_done() {
unsafe { snd_unregister_device(seq_dev) };
unsafe { put_device(seq_dev) };
}

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_check_copy_size_1: [u8; 1] = *b"\0";

static __slate_asm_string_check_copy_size_2: [u8; 55] = *b"/home/takashi/c-corpus/linux/include/linux/ucopysize.h\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

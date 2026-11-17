
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
//!
//! ALPS touchpad PS/2 mouse driver
//!
//! Copyright (c) 2003 Neil Brown <neilb@cse.unsw.edu.au>
//! Copyright (c) 2003-2005 Peter Osterlund <petero2@telia.com>
//! Copyright (c) 2004 Dmitry Torokhov <dtor@mail.ru>
//! Copyright (c) 2005 Vojtech Pavlik <vojtech@suse.cz>
//! Copyright (c) 2009 Sebastian Kapfer <sebastian_kapfer@gmx.net>
//!
//! ALPS detection, tap switching and status querying info is taken from
//! tpconfig utility (by C. Scott Ananian and Bruce Kall).
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
//! See linux/instrumentation.h
//! Begin/end of an instrumentation safe region
//! 64 bit
//! unused, was: #define EX_TYPE_COPY	 4
//! Optimization barrier
//!
//! The "volatile" is due to gcc bugs
//! Yes, this permits 64-bit accesses on 32-bit architectures. These will
//! actually be atomic in some cases (namely Armv7 + LPAE), but for others we
//! rely on the access being split into 2x32-bit accesses for a 32-bit quantity
//! (e.g. a virtual address) and a strong prevailing wind.
//! inhibit tail-call optimization
//! Call a function once. Similar to DO_ONCE(), but does not use jump label
//! patching via static keys.
//! Force a compilation error if condition is true, but also produce a
//! result (of value 0 and type int), so the expression can be used
//! e.g. in a structure initializer (or where-ever else comma expressions
//! aren't permitted).
//!
//! Take an error message as an optional second argument. If omitted,
//! default to the stringification of the tested expression.
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
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! issue num suppressed message on exit
//! strlen("ratelimit") + 1
//! Operation not permitted
//! Value too large for defined data type
//! Kernel pointers have redundant information, so we can use a
//! scheme where we can return either an error code or a normal
//! pointer with the same return value.
//!
//! This should be a per-architecture thing, to allow different
//! error and pointer decisions.
//! This counts to 15. Any more, it will return 16th argument.
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
//! label
//! Defines x86 CPU feature bits
//! An exchange-type operation, which takes a value and a pointer, and
//! returns the old value.
//! Prevent speculative execution past this barrier.
//! @a is a power of 2 value
//! Check at compile time that something is of a particular type.
//! Always evaluates to 1 so you may use it easily in comparisons.
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
//! restart by calling sys_restart_syscall
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
//! PAGE_SHIFT determines the page size.
//!
//! Note: This definition is required because PAGE_SHIFT is used
//! in several places throughout the codebase.
//! for_each_cpu - iterate over every cpu in a mask
//! @cpu: the (optionally unsigned) integer iterator
//! @mask: the cpumask pointer
//!
//! After the loop, cpu is >= nr_cpu_ids.
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
//! needs a pointer
//! Implementation details, userspace should not care about these
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
//! Input
//! pointer device, e.g. trackpad
//! caller sets number of bytes to receive
//! t3btl t4s
//! Sent for device identification
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
// __swahw32 - return a word-swapped 32-bit value
// @x: value to wordswap
//
// __swahw32(0x12340000) is 0x00001234
// __swahb32 - return a high and low byte-swapped 32-bit value
// @x: value to byteswap
//
// __swahb32(0x12345678) is 0x34127856
// Some subsystems have their own custom printk that applies a va_format to a
// generic format, for example, to include a device number or other metadata
// alongside the format supplied by the caller.
//
// In order to store these in the way they would be emitted by the printk
// infrastructure, the subsystem provides us with the start, fixed string, and
// any subsequent text in the format string.
//
// We take a variable argument list as pr_fmt/dev_fmt/etc are sometimes passed
// as multiple arguments (eg: `"%s: ", "blah"`), and we must only take the
// first one.
//
// subsys_fmt_prefix must be known at compile time, or compilation will fail
// (since this is a mistake). If fmt or level is not known at compile time, no
// index entry will be made (since this can legitimately happen).
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
// No such file or directory
// I/O error
// Argument list too long
// Try again
// Out of memory
// Bad address
// Device or resource busy
// No such device
// Invalid argument
// Text file busy
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
// Check that the bitops prototypes are sane
// assign_bit - Assign value to a bit in memory
// @nr: the bit to set
// @addr: the address to start counting from
// @value: the value to assign
// Operation is not supported
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
// disable sas during sighandling
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
// has button(s) under pad
// is a pointing stick
// Five fingers on trackpad
// Four fingers on trackpad
// MT slot being modified
// Center X touch position
// Center Y touch position
// Unique ID of initiated contact
// Need to take variadic arguments even though we don't use them, as dev_fmt()
// may only just have been expanded and may result in multiple arguments.
// Dummy dev_printk for disabled debugging statements to use whilst maintaining
// gcc's format checking.
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
/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__29 {
() => {
((core::mem::size_of::<[alps_model_info; 22]>() as u64) / (core::mem::size_of::<alps_model_info>() as u64)).wrapping_add(__must_be_array__19!() as u64)
};
}

macro_rules! BIT_MASK__2 {
($nr:expr) => {
UL!() << ($nr) % 64
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

macro_rules! BIT_WORD__3 {
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

macro_rules! IS_SS4PLUS_DEV {
($_b:expr) => {
((unsafe { *($_b.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) }) as i32) == 115 && ((unsafe { *($_b.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) }) as i32) == 3 && ((unsafe { *($_b.as_mut_ptr() as *mut u8).wrapping_offset(2 as isize) }) as i32) == 40
};
}

macro_rules! MAKE_PS2_CMD {
($params:expr, $results:expr, $cmd:expr) => {
($params) << 12i32 | ($results) << 8i32 | $cmd
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

macro_rules! SS4_1F_X_V2 {
($_b:expr) => {
((unsafe { *$_b.wrapping_offset(0 as isize) }) as i32) & 7 | ((unsafe { *$_b.wrapping_offset(1 as isize) }) as i32) << 3i32 & 120 | ((unsafe { *$_b.wrapping_offset(1 as isize) }) as i32) << 2i32 & 896 | ((unsafe { *$_b.wrapping_offset(2 as isize) }) as i32) << 5i32 & 7168
};
}

macro_rules! SS4_1F_Y_V2 {
($_b:expr) => {
((unsafe { *$_b.wrapping_offset(2 as isize) }) as i32) & 15 | ((unsafe { *$_b.wrapping_offset(3 as isize) }) as i32) >> 2i32 & 48 | ((unsafe { *$_b.wrapping_offset(4 as isize) }) as i32) << 6i32 & 960 | ((unsafe { *$_b.wrapping_offset(4 as isize) }) as i32) << 5i32 & 3072
};
}

macro_rules! SS4_1F_Z_V2 {
($_b:expr) => {
((unsafe { *$_b.wrapping_offset(5 as isize) }) as i32) & 15 | ((unsafe { *$_b.wrapping_offset(5 as isize) }) as i32) >> 1i32 & 112 | ((unsafe { *$_b.wrapping_offset(4 as isize) }) as i32) & 128
};
}

macro_rules! SS4_BTL_MF_X_V2 {
($_b:expr, $_i:expr) => {
SS4_STD_MF_X_V2!($_b, $_i) | ((unsafe { *$_b.wrapping_offset(0i32.wrapping_add($_i.wrapping_mul(3)) as isize) }) as i32) >> 3i32 & 16
};
}

macro_rules! SS4_BTL_MF_Y_V2 {
($_b:expr, $_i:expr) => {
SS4_STD_MF_Y_V2!($_b, $_i) | ((unsafe { *$_b.wrapping_offset(0i32.wrapping_add($_i.wrapping_mul(3)) as isize) }) as i32) >> 3i32 & 8
};
}

macro_rules! SS4_BTN_V2 {
($_b:expr) => {
((unsafe { *$_b.wrapping_offset(0 as isize) }) as i32) >> 5i32 & 7
};
}

macro_rules! SS4_IS_5F_DETECTED {
($_b:expr) => {
((unsafe { *$_b.wrapping_offset(2 as isize) }) as i32) & 16 == 16
};
}

macro_rules! SS4_IS_IDLE_V2 {
($_b:expr) => {
((unsafe { *$_b.wrapping_offset(0 as isize) }) as i32) == 24 && ((unsafe { *$_b.wrapping_offset(1 as isize) }) as i32) == 16 && ((unsafe { *$_b.wrapping_offset(2 as isize) }) as i32) == 0 && ((unsafe { *$_b.wrapping_offset(3 as isize) }) as i32) & 136 == 8 && ((unsafe { *$_b.wrapping_offset(4 as isize) }) as i32) == 16 && ((unsafe { *$_b.wrapping_offset(5 as isize) }) as i32) == 0
};
}

macro_rules! SS4_IS_MF_CONTINUE {
($_b:expr) => {
((unsafe { *$_b.wrapping_offset(2 as isize) }) as i32) & 16 == 16
};
}

macro_rules! SS4_MF_Z_V2 {
($_b:expr, $_i:expr) => {
((unsafe { *$_b.wrapping_offset(1i32.wrapping_add($_i.wrapping_mul(3)) as isize) }) as i32) & 1 | ((unsafe { *$_b.wrapping_offset(1i32.wrapping_add($_i.wrapping_mul(3)) as isize) }) as i32) >> 1i32 & 2
};
}

macro_rules! SS4_PLUS_BTL_MF_X_V2 {
($_b:expr, $_i:expr) => {
SS4_PLUS_STD_MF_X_V2!($_b, $_i) | ((unsafe { *$_b.wrapping_offset(0i32.wrapping_add($_i.wrapping_mul(3)) as isize) }) as i32) >> 4i32 & 8
};
}

macro_rules! SS4_PLUS_STD_MF_X_V2 {
($_b:expr, $_i:expr) => {
((unsafe { *$_b.wrapping_offset(0i32.wrapping_add($_i.wrapping_mul(3)) as isize) }) as i32) << 4i32 & 112 | ((unsafe { *$_b.wrapping_offset(1i32.wrapping_add($_i.wrapping_mul(3)) as isize) }) as i32) << 4i32 & 3968
};
}

macro_rules! SS4_STD_MF_X_V2 {
($_b:expr, $_i:expr) => {
((unsafe { *$_b.wrapping_offset(0i32.wrapping_add($_i.wrapping_mul(3)) as isize) }) as i32) << 5i32 & 224 | ((unsafe { *$_b.wrapping_offset(1i32.wrapping_add($_i.wrapping_mul(3)) as isize) }) as i32) << 5i32 & 7936
};
}

macro_rules! SS4_STD_MF_Y_V2 {
($_b:expr, $_i:expr) => {
((unsafe { *$_b.wrapping_offset(1i32.wrapping_add($_i.wrapping_mul(3)) as isize) }) as i32) << 3i32 & 16 | ((unsafe { *$_b.wrapping_offset(2i32.wrapping_add($_i.wrapping_mul(3)) as isize) }) as i32) << 5i32 & 480 | ((unsafe { *$_b.wrapping_offset(2i32.wrapping_add($_i.wrapping_mul(3)) as isize) }) as i32) << 4i32 & 3584
};
}

macro_rules! SS4_TS_X_V2 {
($_b:expr) => {
((((unsafe { *$_b.wrapping_offset(0 as isize) }) as i32) & 1) << 7i32 | ((unsafe { *$_b.wrapping_offset(1 as isize) }) as i32) & 127) as i8
};
}

macro_rules! SS4_TS_Y_V2 {
($_b:expr) => {
((((((unsafe { *$_b.wrapping_offset(3 as isize) }) as i32) & 1) << 7i32 | ((unsafe { *$_b.wrapping_offset(2 as isize) }) as i32) & 127) as i8) as i32).wrapping_neg()
};
}

macro_rules! SS4_TS_Z_V2 {
($_b:expr) => {
(((unsafe { *$_b.wrapping_offset(4 as isize) }) as i32) & 127) as i8
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

macro_rules! __clear_bit {
($nr:expr, $addr:expr) => {
if true && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
generic___clear_bit(($nr) as u64, ($addr) as *mut u64);
} else {
___clear_bit(($nr) as u64, ($addr) as *mut u64);
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

macro_rules! __is_nonneg {
($ux:expr) => {
statically_true!((($ux) as i64) >= 0)
};
}

macro_rules! __must_be_array__19 {
() => {
core::mem::size_of::<__SlateRecord_alps_c_2859>() as i32
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__2 {
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

/// The following macros are non-atomic versions of their non-underscored
/// counterparts.
macro_rules! __set_bit {
($nr:expr, $addr:expr) => {
if true && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
generic___set_bit(($nr) as u64, ($addr) as *mut u64);
} else {
___set_bit(($nr) as u64, ($addr) as *mut u64);
}
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
if is_signed_type__3!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul(((core::mem::size_of::<i32>() as u64) < 4) as i32)) }
};
}

macro_rules! __timer_init {
($_timer:expr, $_fn:expr, $_flags:expr) => {
unsafe { timer_init_key($_timer, $_fn, ($_flags) as u32, core::ptr::null::<u8>(), core::ptr::null_mut::<lock_class_key>()) };
};
}

macro_rules! __types_ok__2 {
($ux:expr, $uy:expr) => {
__sign_use!($ux) & __sign_use!($uy)
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
macro_rules! container_of__38 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__132!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__40 {
($ptr:expr) => {
{
let mut __v: *mut alps_data;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__116!() as isize)) }) as *mut ()) as *mut alps_data;
__v
}
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type__3 {
() => {
1i32.wrapping_neg() < 1
};
}

macro_rules! kcsan_check_atomic_read {
($ptr:expr, $size:expr) => {
kcsan_check_access($ptr, $size, 4i32);
};
}

macro_rules! kcsan_check_atomic_write {
($ptr:expr, $size:expr) => {
kcsan_check_access(absolute_pointer!($ptr) as *const (), $size, 5i32);
};
}

/// kcsan_check_write - check regular write access for races
///
/// @ptr: address of access
/// @size: size of access
macro_rules! kcsan_check_write {
($ptr:expr, $size:expr) => {
kcsan_check_access(absolute_pointer!($ptr) as *const (), $size, 1i32);
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

macro_rules! offsetof__116 {
() => {
core::mem::offset_of!(alps_data, timer) as u64
};
}

macro_rules! offsetof__132 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, dev3_register_work) as u64
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

macro_rules! test_bit {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if true && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
const_test_bit(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit(($nr) as u64, ($addr) as *const u64) as i32
};
__v
}
};
}

macro_rules! timer_container_of__6 {
($callback_timer:expr) => {
container_of__40!($callback_timer)
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
    static mut kmsg_fops: file_operations;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut jiffies: u64;
    static mut system_percpu_wq: *mut workqueue_struct;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn __bad_size_call_parameter();
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn memcmp(arg0: *const (), arg1: *const (), arg2: u64) -> i32;
    fn scnprintf(arg0: *mut u8, arg1: u64, arg2: *const u8, ...) -> i32;
    fn __compiletime_assert_16();
    fn __compiletime_assert_17();
    fn __compiletime_assert_62();
    fn _raw_spin_lock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irq(arg0: *mut raw_spinlock);
    fn __msecs_to_jiffies(arg0: u32) -> u64;
    fn timer_init_key(arg0: *mut timer_list, arg1: Option<unsafe extern "C" fn(*mut timer_list)>, arg2: u32, arg3: *const u8, arg4: *mut lock_class_key);
    fn mod_timer(arg0: *mut timer_list, arg1: u64) -> i32;
    fn timer_delete(arg0: *mut timer_list) -> i32;
    fn timer_shutdown_sync(arg0: *mut timer_list) -> i32;
    fn queue_work_on(arg0: i32, arg1: *mut workqueue_struct, arg2: *mut work_struct) -> bool;
    fn disable_work_sync(arg0: *mut work_struct) -> bool;
    fn mutex_lock(arg0: *mut mutex);
    fn mutex_unlock(arg0: *mut mutex);
    fn kfree(arg0: *const ());
    fn __compiletime_assert_208();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn _dev_printk(arg0: *const u8, arg1: *const device, arg2: *const u8, ...);
    fn _dev_err(arg0: *const device, arg1: *const u8, ...);
    fn _dev_warn(arg0: *const device, arg1: *const u8, ...);
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_395();
    fn __compiletime_assert_416();
    fn input_allocate_device() -> *mut input_dev;
    fn input_free_device(arg0: *mut input_dev);
    fn input_register_device(arg0: *mut input_dev) -> i32;
    fn input_unregister_device(arg0: *mut input_dev);
    fn input_event(arg0: *mut input_dev, arg1: u32, arg2: u32, arg3: i32);
    fn input_set_capability(arg0: *mut input_dev, arg1: u32, arg2: u32);
    fn input_alloc_absinfo(arg0: *mut input_dev);
    fn input_set_abs_params(arg0: *mut input_dev, arg1: u32, arg2: i32, arg3: i32, arg4: i32, arg5: i32);
    fn input_mt_init_slots(arg0: *mut input_dev, arg1: u32, arg2: u32) -> i32;
    fn input_mt_report_slot_state(arg0: *mut input_dev, arg1: u32, arg2: bool) -> bool;
    fn input_mt_report_finger_count(arg0: *mut input_dev, arg1: i32);
    fn input_mt_sync_frame(arg0: *mut input_dev);
    fn input_mt_assign_slots(arg0: *mut input_dev, arg1: *mut i32, arg2: *const input_mt_pos, arg3: i32, arg4: i32) -> i32;
    fn ps2_drain(arg0: *mut ps2dev, arg1: u64, arg2: u32);
    fn ps2_command(arg0: *mut ps2dev, arg1: *mut u8, arg2: u32) -> i32;
    fn dmi_check_system(arg0: *const dmi_system_id) -> i32;
    fn psmouse_reset(arg0: *mut psmouse) -> i32;
    fn psmouse_report_standard_buttons(arg0: *mut input_dev, arg1: u8);
    fn psmouse_report_standard_motion(arg0: *mut input_dev, arg1: *mut u8);
    fn __compiletime_assert_461();
    fn __compiletime_assert_464();
    fn __compiletime_assert_467();
    fn __compiletime_assert_470();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_1c5364fcd9da51fe(arg0: u32) -> bool;
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
    #[link_name = "llvm.is.constant.i64"]
    fn __slate_intrinsic_e1c548fa55baa574(arg0: i64) -> bool;
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


#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
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


#[inline(always)]
fn ERR_PTR(error: i64) -> *mut () {
error as *mut ()
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
fn instrument_write(v: *const (), size: u64) {
kasan_check_write(v, size as u32);
kcsan_check_write!(v, size);
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

#[inline(always)]
fn instrument_atomic_write(v: *const (), size: u64) {
kasan_check_write(v, size as u32);
kcsan_check_atomic_write!(v, size);
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
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


#[inline(always)]
fn generic___set_bit(nr: u64, addr: *mut u64) {
let mask: u64 = BIT_MASK!(nr);
let p: *mut u64 = (addr as *mut u64).wrapping_offset(BIT_WORD!(nr) as isize);
unsafe {
*p = *p | mask;
}
}

#[inline(always)]
fn generic___clear_bit(nr: u64, addr: *mut u64) {
let mask: u64 = BIT_MASK!(nr);
let p: *mut u64 = (addr as *mut u64).wrapping_offset(BIT_WORD!(nr) as isize);
let __v2: u64 = (unsafe { *p }) & !mask;
unsafe {
*p = __v2;
}
}

// const_*() definitions provide good compile-time optimizations when
// the passed arguments can be resolved at compile time.
#[inline(always)]
fn const_test_bit(nr: u64, addr: *const u64) -> bool {
let p: *const u64 = (addr as *const u64).wrapping_offset(BIT_WORD!(nr) as isize);
let mask: u64 = BIT_MASK!(nr);
let val: u64 = unsafe { *p };
val & mask != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pollfd {
}


#[inline(always)]
fn arch_set_bit(mut nr: i64, mut addr: *mut u64) {
if unsafe { __slate_intrinsic_e1c548fa55baa574(nr) } {
unsafe {
core::arch::asm!("lock orb {1:l},({0:r})", in(reg) ((addr as *const u8).wrapping_offset((nr >> (3 as i32)) as isize) as *mut ()) as *mut u8, in(reg) ((1 as i32) << (nr & ((7 as i32) as i64))) as u32, options(nostack, att_syntax));
}
} else {
unsafe {
core::arch::asm!("lock  btsq  {1:r},({0:r})", in(reg) addr as *mut i64, in(reg) nr as u64, options(nostack, att_syntax));
}
}
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
fn arch___set_bit(mut nr: u64, mut addr: *mut u64) {
unsafe {
core::arch::asm!(" btsq  {1:r},({0:r})", in(reg) addr as *mut i64, in(reg) nr as u64, options(nostack, att_syntax));
}
}

#[inline(always)]
fn arch_clear_bit(mut nr: i64, mut addr: *mut u64) {
if unsafe { __slate_intrinsic_e1c548fa55baa574(nr) } {
unsafe {
core::arch::asm!("lock andb {1:l},({0:r})", in(reg) ((addr as *const u8).wrapping_offset((nr >> (3 as i32)) as isize) as *mut ()) as *mut u8, in(reg) !((1 as i32) << (nr & ((7 as i32) as i64))) as u32, options(nostack, att_syntax));
}
} else {
unsafe {
core::arch::asm!("lock  btrq  {1:r},({0:r})", in(reg) addr as *mut i64, in(reg) nr as u64, options(nostack, att_syntax));
}
}
}

#[inline(always)]
fn arch___clear_bit(mut nr: u64, mut addr: *mut u64) {
unsafe {
core::arch::asm!(" btrq  {1:r},({0:r})", in(reg) addr as *mut i64, in(reg) nr as u64, options(nostack, att_syntax));
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct io_bitmap {
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
fn set_bit(nr: i64, addr: *mut u64) {
instrument_atomic_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_set_bit(nr, addr);
}

#[inline(always)]
fn clear_bit(nr: i64, addr: *mut u64) {
instrument_atomic_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_clear_bit(nr, addr);
}

#[inline(always)]
fn ___set_bit(nr: u64, addr: *mut u64) {
instrument_write(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch___set_bit(nr, addr);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
}


#[inline(always)]
fn ___clear_bit(nr: u64, addr: *mut u64) {
instrument_write(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch___clear_bit(nr, addr);
}

#[inline(always)]
fn _test_bit(nr: u64, addr: *const u64) -> bool {
instrument_atomic_read(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_bit(nr, addr)
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
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
}


fn INIT_LIST_HEAD(list: *mut list_head) {
loop {
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
struct llist_head {
    first: *mut llist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct llist_node {
    next: *mut llist_node,
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
struct futex_private_hash {
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


#[inline(always)]
fn get_current() -> *mut task_struct {
if IS_ENABLED!() != 0 {
let mut __v0: *mut task_struct;
loop {
if true {
unsafe { __compiletime_assert_62() };
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
struct pid_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sem_undo_list {
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
    __slate_bits_0: __slate_bits::__SlateBits126U0,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct orc_entry {
    sp_offset: i16,
    bp_offset: i16,
    __slate_bits_0: __slate_bits::__SlateBits127U0,
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
    __slate_bits_0: __slate_bits::__SlateBits128U0,
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
struct hrtimer_clock_base {
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
struct seccomp_filter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwlock {
    raw_lock: qrwlock,
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
struct futex_mm_data {
    phash: futex_mm_phash,
    unlock: futex_unlock_cs_ranges,
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
struct workqueue_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct refcount_struct {
    refs: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rlimit {
    rlim_cur: u64,
    rlim_max: u64,
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
struct __SlateRecord_signal_h_25 {
    sig: [u64; 1],
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
struct k_sigaction {
    sa: sigaction,
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
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
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
struct posix_cputimer_base {
    nextevt: u64,
    tqhead: timerqueue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seq_file {
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
struct rq_flags {
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
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
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
union unwind_task_id {
    __slate_anon_0: __SlateRecord_unwind_deferred_types_h_28,
    id: u64,
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
struct unwind_task_info {
    unwind_mask: __SlateRecord_types_h_197,
    cache: *mut unwind_cache,
    work: callback_head,
    id: unwind_task_id,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gendisk {
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
struct vm_struct {
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
    __slate_bits_0: __slate_bits::__SlateBits259U0,
    dl_timer: hrtimer,
    inactive_timer: hrtimer,
    rq: *mut rq,
    server_pick_task: Option<unsafe extern "C" fn(*mut sched_dl_entity, *mut rq_flags) -> *mut task_struct>,
    pi_se: *mut sched_dl_entity,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
union rcu_special {
    b: __SlateRecord_sched_h_791,
    s: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
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
struct uprobe {
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
struct old_timespec32 {
    tv_sec: i32,
    tv_nsec: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe_xol_ops {
}


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
fn _msecs_to_jiffies(m: u32) -> u64 {
((m as i64).wrapping_add(1000 / 1000).wrapping_sub(1) / (1000 / 1000)) as u64
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


#[inline(always)]
fn msecs_to_jiffies(m: u32) -> u64 {
if unsafe { __slate_intrinsic_1c5364fcd9da51fe(m) } {
if (m as i32) < 0 {
return 4611686018427387902u64;
}
return _msecs_to_jiffies(m);
} else {
return unsafe { __msecs_to_jiffies(m) };
}
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
struct rcu_gp_seq {
    norm: u64,
    exp: u64,
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
struct delayed_work {
    work: work_struct,
    timer: timer_list,
    wq: *mut workqueue_struct,
    cpu: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
}


fn __init_work(work: *mut work_struct, onstack: i32) {
}

fn queue_work(wq: *mut workqueue_struct, work: *mut work_struct) -> bool {
unsafe { queue_work_on(64i32, wq, work) }
}

fn schedule_work(work: *mut work_struct) -> bool {
queue_work(unsafe { system_percpu_wq }, work)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ldt_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wait_queue_head {
    lock: spinlock,
    head: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mutex_h_253 {
    lock: *mut mutex,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mem_cgroup {
}


#[inline(always)]
fn class_mutex_destructor(_T: *mut __SlateRecord_mutex_h_253) {
unsafe { mutex_unlock((*_T).lock) };
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


#[inline(always)]
fn class_mutex_constructor(l: *mut mutex) -> __SlateRecord_mutex_h_253 {
let mut _t: __SlateRecord_mutex_h_253 = __SlateRecord_mutex_h_253 { lock: l };
let _T: *mut __SlateRecord_mutex_h_253 = core::ptr::addr_of_mut!(_t);
unsafe { mutex_lock((*_T).lock) };
return _t;
}

#[inline(always)]
fn __class_mutex_cleanup_ctx(_T: *mut *mut __SlateRecord_mutex_h_253) {
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
struct swait_queue_head {
    lock: raw_spinlock,
    task_list: list_head,
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
struct completion {
    done: u32,
    wait: swait_queue_head,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct srcu_ctr {
    srcu_locks: __SlateRecord_types_h_197,
    srcu_unlocks: __SlateRecord_types_h_197,
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
union __SlateRecord_input_h_480 {
    constant: ff_constant_effect,
    ramp: ff_ramp_effect,
    periodic: ff_periodic_effect,
    condition: [ff_condition_effect; 2],
    rumble: ff_rumble_effect,
    haptic: ff_haptic_effect,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_288 {
    val: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct user_namespace {
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
struct ns_common {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_userfaultfd_ctx {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct poll_table_struct {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct pfnmap_track_ctx {
    kref: kref,
    pfn: u64,
    size: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mmap_action {
    __slate_anon_0: __SlateRecord_mm_types_h_830,
    r#type: u32,
    error_override: i32,
    __slate_bits_0: __slate_bits::__SlateBits382U0,
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
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
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
struct sock {
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
struct input_id {
    bustype: u16,
    vendor: u16,
    product: u16,
    version: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct input_absinfo {
    value: i32,
    minimum: i32,
    maximum: i32,
    fuzz: i32,
    flat: i32,
    resolution: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct input_keymap_entry {
    flags: u8,
    len: u8,
    index: u16,
    keycode: u32,
    scancode: [u8; 32],
}


// MT_TOOL types
#[repr(C)]
#[derive(Clone, Copy)]
struct ff_replay {
    length: u16,
    delay: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ff_trigger {
    button: u16,
    interval: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ff_envelope {
    attack_length: u16,
    attack_level: u16,
    fade_length: u16,
    fade_level: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ff_constant_effect {
    level: i16,
    envelope: ff_envelope,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ff_ramp_effect {
    start_level: i16,
    end_level: i16,
    envelope: ff_envelope,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wake_irq {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ff_condition_effect {
    right_saturation: u16,
    left_saturation: u16,
    right_coeff: i16,
    left_coeff: i16,
    deadband: u16,
    center: i16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ff_periodic_effect {
    waveform: u16,
    period: u16,
    magnitude: i16,
    offset: i16,
    phase: u16,
    envelope: ff_envelope,
    custom_len: u32,
    custom_data: *mut i16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_qos {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ff_rumble_effect {
    strong_magnitude: u16,
    weak_magnitude: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ff_haptic_effect {
    hid_usage: u16,
    vendor_id: u16,
    vendor_waveform_page: u8,
    intensity: u16,
    repeat_count: u16,
    retrigger_period: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fwnode_handle {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ff_effect {
    r#type: u16,
    id: i16,
    direction: u16,
    trigger: ff_trigger,
    replay: ff_replay,
    u: __SlateRecord_input_h_480,
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
struct folio_queue {
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
    __slate_bits_0: __slate_bits::__SlateBits445U0,
    vm_ops: *const vm_operations_struct,
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
    __slate_bits_0: __slate_bits::__SlateBits447U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mnt_idmap {
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
struct assoc_array_ptr {
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
struct key_type {
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
struct pm_message {
    event: i32,
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
struct pm_subsys_data {
    lock: spinlock,
    refcount: u32,
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
struct dev_pm_info {
    power_state: pm_message,
    __slate_bits_0: __slate_bits::__SlateBits472U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits472U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits472U2,
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
    __slate_bits_3: __slate_bits::__SlateBits472U3,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pid_h_62 {
    ino: u64,
    pidfs_hash: rhash_head,
    stashed: *mut dentry,
    attr: *mut pidfs_attr,
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
struct rhash_head {
    next: *mut rhash_head,
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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
struct backing_dev_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct block_device {
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


fn hrtimer_get_expires(timer: *const hrtimer) -> i64 {
unsafe { (*timer).node.expires }
}

fn hrtimer_get_softexpires(timer: *const hrtimer) -> i64 {
unsafe { (*timer)._softexpires }
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


fn hrtimer_start_expires(timer: *mut hrtimer, mode: u32) {
let soft: i64 = hrtimer_get_softexpires(timer as *const hrtimer);
let hard: i64 = hrtimer_get_expires(timer as *const hrtimer);
let delta: u64 = ktime_to_ns(ktime_sub!(hard, soft)) as u64;
unsafe { hrtimer_start_range_ns(timer, soft, delta, mode) };
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


fn hrtimer_restart(timer: *mut hrtimer) {
hrtimer_start_expires(timer, 0u32);
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
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
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
struct thread_group_cputimer {
    cputime_atomic: task_cputime_atomic,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_acl {
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
    __slate_bits_0: __slate_bits::__SlateBits262U0,
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


fn rlimit(limit: u32) -> u64 {
task_rlimit(get_current() as *const task_struct, limit)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_cmd {
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
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
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
struct module_param_attrs {
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
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct input_dev_poller {
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
struct __SlateRecord_alps_c_2859 {
}


#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

mod __slate_bits {
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits259U0 {
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
    pub struct __SlateBits262U0 {
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
    pub struct __SlateBits126U0 {
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
    pub struct __SlateBits128U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits445U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits447U0 {
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
    pub struct __SlateBits127U0 {
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
    pub struct __SlateBits382U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits713U0 {
        #[bits(7)]
        pub slot: u8,
        #[bits(1)]
        pub exact_match: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits472U0 {
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
    pub struct __SlateBits469U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits472U1 {
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
    pub struct __SlateBits472U2 {
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
    pub struct __SlateBits472U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits728U0 {
        #[bits(1)]
        pub first_mp: u32,
        #[bits(1)]
        pub is_mp: u32,
        #[bits(1)]
        pub left: u32,
        #[bits(1)]
        pub right: u32,
        #[bits(1)]
        pub middle: u32,
        #[bits(1)]
        pub ts_left: u32,
        #[bits(1)]
        pub ts_right: u32,
        #[bits(1)]
        pub ts_middle: u32,
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
    __slate_bits_0: __slate_bits::__SlateBits469U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct input_device_id {
    flags: u64,
    bustype: u16,
    vendor: u16,
    product: u16,
    version: u16,
    evbit: [u64; 1],
    keybit: [u64; 12],
    relbit: [u64; 1],
    absbit: [u64; 1],
    mscbit: [u64; 1],
    ledbit: [u64; 1],
    sndbit: [u64; 1],
    ffbit: [u64; 2],
    swbit: [u64; 1],
    propbit: [u64; 1],
    driver_info: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct input_value {
    r#type: u16,
    code: u16,
    value: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct input_dev {
    name: *const u8,
    phys: *const u8,
    uniq: *const u8,
    id: input_id,
    propbit: [u64; 1],
    evbit: [u64; 1],
    keybit: [u64; 12],
    relbit: [u64; 1],
    absbit: [u64; 1],
    mscbit: [u64; 1],
    ledbit: [u64; 1],
    sndbit: [u64; 1],
    ffbit: [u64; 2],
    swbit: [u64; 1],
    hint_events_per_packet: u32,
    keycodemax: u32,
    keycodesize: u32,
    keycode: *mut (),
    setkeycode: Option<unsafe extern "C" fn(*mut input_dev, *const input_keymap_entry, *mut u32) -> i32>,
    getkeycode: Option<unsafe extern "C" fn(*mut input_dev, *mut input_keymap_entry) -> i32>,
    ff: *mut ff_device,
    poller: *mut input_dev_poller,
    repeat_key: u32,
    timer: timer_list,
    rep: [i32; 2],
    mt: *mut input_mt,
    absinfo: *mut input_absinfo,
    key: [u64; 12],
    led: [u64; 1],
    snd: [u64; 1],
    sw: [u64; 1],
    open: Option<unsafe extern "C" fn(*mut input_dev) -> i32>,
    close: Option<unsafe extern "C" fn(*mut input_dev)>,
    flush: Option<unsafe extern "C" fn(*mut input_dev, *mut file) -> i32>,
    event: Option<unsafe extern "C" fn(*mut input_dev, u32, u32, i32) -> i32>,
    grab: *mut input_handle,
    event_lock: spinlock,
    mutex: mutex,
    users: u32,
    going_away: bool,
    dev: device,
    h_list: list_head,
    node: list_head,
    num_vals: u32,
    max_vals: u32,
    vals: *mut input_value,
    devres_managed: bool,
    timestamp: [i64; 3],
    inhibited: bool,
    ready: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct input_handler {
    private: *mut (),
    event: Option<unsafe extern "C" fn(*mut input_handle, u32, u32, i32)>,
    events: Option<unsafe extern "C" fn(*mut input_handle, *mut input_value, u32) -> u32>,
    filter: Option<unsafe extern "C" fn(*mut input_handle, u32, u32, i32) -> bool>,
    r#match: Option<unsafe extern "C" fn(*mut input_handler, *mut input_dev) -> bool>,
    connect: Option<unsafe extern "C" fn(*mut input_handler, *mut input_dev, *const input_device_id) -> i32>,
    disconnect: Option<unsafe extern "C" fn(*mut input_handle)>,
    start: Option<unsafe extern "C" fn(*mut input_handle)>,
    passive_observer: bool,
    legacy_minors: bool,
    minor: i32,
    name: *const u8,
    id_table: *const input_device_id,
    h_list: list_head,
    node: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct input_handle {
    private: *mut (),
    open: i32,
    name: *const u8,
    dev: *mut input_dev,
    handler: *mut input_handler,
    handle_events: Option<unsafe extern "C" fn(*mut input_handle, *mut input_value, u32) -> u32>,
    d_node: list_head,
    h_node: list_head,
}


fn input_report_key(dev: *mut input_dev, code: u32, value: i32) {
unsafe { input_event(dev, 1u32, code, (value != 0) as i32) };
}

fn input_report_rel(dev: *mut input_dev, code: u32, value: i32) {
unsafe { input_event(dev, 2u32, code, value) };
}

fn input_report_abs(dev: *mut input_dev, code: u32, value: i32) {
unsafe { input_event(dev, 3u32, code, value) };
}

fn input_sync(dev: *mut input_dev) {
unsafe { input_event(dev, 0u32, 0u32, 0i32) };
}

fn input_abs_set_res(dev: *mut input_dev, axis: u32, val: i32) {
unsafe { input_alloc_absinfo(dev) };
if !unsafe { (*dev).absinfo }.is_null() {
unsafe {
(*unsafe { (*dev).absinfo }.wrapping_offset(axis as isize)).resolution = val;
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ff_device {
    upload: Option<unsafe extern "C" fn(*mut input_dev, *mut ff_effect, *mut ff_effect) -> i32>,
    erase: Option<unsafe extern "C" fn(*mut input_dev, i32) -> i32>,
    playback: Option<unsafe extern "C" fn(*mut input_dev, i32, i32) -> i32>,
    set_gain: Option<unsafe extern "C" fn(*mut input_dev, u16)>,
    set_autocenter: Option<unsafe extern "C" fn(*mut input_dev, u16)>,
    destroy: Option<unsafe extern "C" fn(*mut ff_device)>,
    stop: Option<unsafe extern "C" fn(*mut ff_device)>,
    private: *mut (),
    ffbit: [u64; 2],
    mutex: mutex,
    max_effects: i32,
    effects: *mut ff_effect,
    effect_owners: [*mut file; 0],
}


// drop contacts not seen in frame
// use in-kernel tracking
// semi-mt device, finger count handled manually
#[repr(C)]
#[derive(Clone, Copy)]
struct input_mt_slot {
    abs: [i32; 14],
    frame: u32,
    key: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct input_mt {
    trkid: i32,
    num_slots: i32,
    slot: i32,
    flags: u32,
    frame: u32,
    red: *mut i32,
    slots: [input_mt_slot; 0],
}


fn input_mt_slot(dev: *mut input_dev, slot: i32) {
unsafe { input_event(dev, 3u32, 47u32, slot) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct input_mt_pos {
    x: i16,
    y: i16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct serio_device_id {
    r#type: u8,
    extra: u8,
    id: u8,
    proto: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct serio {
    port_data: *mut (),
    name: [u8; 32],
    phys: [u8; 32],
    firmware_id: [u8; 128],
    manual_bind: bool,
    id: serio_device_id,
    lock: spinlock,
    write: Option<unsafe extern "C" fn(*mut serio, u8) -> i32>,
    open: Option<unsafe extern "C" fn(*mut serio) -> i32>,
    close: Option<unsafe extern "C" fn(*mut serio)>,
    start: Option<unsafe extern "C" fn(*mut serio) -> i32>,
    stop: Option<unsafe extern "C" fn(*mut serio)>,
    parent: *mut serio,
    child_node: list_head,
    children: list_head,
    depth: u32,
    drv: *mut serio_driver,
    drv_mutex: mutex,
    dev: device,
    node: list_head,
    ps2_cmd_mutex: *mut mutex,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct serio_driver {
    description: *const u8,
    id_table: *const serio_device_id,
    manual_bind: bool,
    write_wakeup: Option<unsafe extern "C" fn(*mut serio)>,
    interrupt: Option<unsafe extern "C" fn(*mut serio, u8, u32) -> u32>,
    connect: Option<unsafe extern "C" fn(*mut serio, *mut serio_driver) -> i32>,
    reconnect: Option<unsafe extern "C" fn(*mut serio) -> i32>,
    fast_reconnect: Option<unsafe extern "C" fn(*mut serio) -> i32>,
    disconnect: Option<unsafe extern "C" fn(*mut serio)>,
    cleanup: Option<unsafe extern "C" fn(*mut serio)>,
    driver: device_driver,
}


fn serio_pause_rx(serio: *mut serio) {
spin_lock_irq(unsafe { core::ptr::addr_of_mut!((*serio).lock) });
}

fn serio_continue_rx(serio: *mut serio) {
spin_unlock_irq(unsafe { core::ptr::addr_of_mut!((*serio).lock) });
}

#[inline(always)]
fn class_serio_pause_rx_destructor(p: *mut *mut serio) {
let _T: *mut serio = unsafe { *p };
serio_continue_rx(_T);
}

#[inline(always)]
fn class_serio_pause_rx_constructor(_T: *mut serio) -> *mut serio {
let mut __v0: *mut serio;
serio_pause_rx(_T);
__v0 = _T;
let t: *mut serio = __v0;
t
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ps2dev {
    serio: *mut serio,
    cmd_mutex: mutex,
    wait: wait_queue_head,
    flags: u64,
    cmdbuf: [u8; 8],
    cmdcnt: u8,
    nak: u8,
    pre_receive_handler: Option<unsafe extern "C" fn(*mut ps2dev, u8, u32) -> u32>,
    receive_handler: Option<unsafe extern "C" fn(*mut ps2dev, u8)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dmi_strmatch {
    __slate_bits_0: __slate_bits::__SlateBits713U0,
    substr: [u8; 79],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dmi_system_id {
    callback: Option<unsafe extern "C" fn(*const dmi_system_id) -> i32>,
    ident: *const u8,
    matches: [dmi_strmatch; 4],
    driver_data: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct psmouse_protocol {
    r#type: u32,
    maxproto: bool,
    ignore_parity: bool,
    try_passthru: bool,
    smbus_companion: bool,
    name: *const u8,
    alias: *const u8,
    detect: Option<unsafe extern "C" fn(*mut psmouse, bool) -> i32>,
    init: Option<unsafe extern "C" fn(*mut psmouse) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct psmouse {
    private: *mut (),
    dev: *mut input_dev,
    ps2dev: ps2dev,
    resync_work: work_struct,
    vendor: *const u8,
    name: *const u8,
    protocol: *const psmouse_protocol,
    packet: [u8; 8],
    badbyte: u8,
    pktcnt: u8,
    pktsize: u8,
    oob_data_type: u8,
    extra_buttons: u8,
    acks_disable_command: bool,
    model: u32,
    last: u64,
    out_of_sync_cnt: u64,
    num_resyncs: u64,
    state: u32,
    devname: [u8; 64],
    phys: [u8; 32],
    rate: u32,
    resolution: u32,
    resetafter: u32,
    resync_time: u32,
    smartscroll: bool,
    protocol_handler: Option<unsafe extern "C" fn(*mut psmouse) -> u32>,
    set_rate: Option<unsafe extern "C" fn(*mut psmouse, u32)>,
    set_resolution: Option<unsafe extern "C" fn(*mut psmouse, u32)>,
    set_scale: Option<unsafe extern "C" fn(*mut psmouse, u32)>,
    reconnect: Option<unsafe extern "C" fn(*mut psmouse) -> i32>,
    fast_reconnect: Option<unsafe extern "C" fn(*mut psmouse) -> i32>,
    disconnect: Option<unsafe extern "C" fn(*mut psmouse)>,
    cleanup: Option<unsafe extern "C" fn(*mut psmouse)>,
    poll: Option<unsafe extern "C" fn(*mut psmouse) -> i32>,
    pt_activate: Option<unsafe extern "C" fn(*mut psmouse)>,
    pt_deactivate: Option<unsafe extern "C" fn(*mut psmouse)>,
}


// SS4btl SS4s
// ss3btl
// x-electrode offset
// y-electrode offset
// X-Coordinate value
// Y-Coordinate value
// Buttonless X-Coord value
// Buttonless Y-Coord value
// SS4 PLUS, X
// Buttonless SS4 PLUS, X
#[repr(C)]
#[derive(Clone, Copy)]
struct alps_protocol_info {
    version: u16,
    byte0: u8,
    mask0: u8,
    flags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct alps_model_info {
    signature: [u8; 3],
    protocol_info: alps_protocol_info,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct alps_nibble_commands {
    command: i32,
    data: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct alps_bitmap_point {
    start_bit: i32,
    num_bits: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct alps_fields {
    x_map: u32,
    y_map: u32,
    fingers: u32,
    pressure: i32,
    st: input_mt_pos,
    mt: [input_mt_pos; 4],
    __slate_bits_0: __slate_bits::__SlateBits728U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct alps_data {
    psmouse: *mut psmouse,
    dev2: *mut input_dev,
    dev3: *mut input_dev,
    phys2: [u8; 32],
    phys3: [u8; 32],
    dev3_register_work: work_struct,
    nibble_commands: *const alps_nibble_commands,
    addr_command: i32,
    proto_version: u16,
    byte0: u8,
    mask0: u8,
    dev_id: [u8; 3],
    fw_ver: [u8; 3],
    flags: i32,
    x_max: i32,
    y_max: i32,
    x_bits: i32,
    y_bits: i32,
    x_res: u32,
    y_res: u32,
    hw_init: Option<unsafe extern "C" fn(*mut psmouse) -> i32>,
    process_packet: Option<unsafe extern "C" fn(*mut psmouse)>,
    decode_fields: Option<unsafe extern "C" fn(*mut alps_fields, *mut u8, *mut psmouse) -> i32>,
    set_abs_params: Option<unsafe extern "C" fn(*mut alps_data, *mut input_dev)>,
    prev_fin: i32,
    multi_packet: i32,
    second_touch: i32,
    multi_data: [u8; 6],
    f: alps_fields,
    quirks: u8,
    timer: timer_list,
}


// trakcstick buttons in trackstick packet
// Definitions for ALPS version 3 and 4 command mode protocol
/// 0
/// 1
/// 2
/// 3
/// 4
/// 5
/// 6
/// 7
/// 8
/// 9
/// a
/// b
/// c
/// d
/// e
/// f
static alps_v3_nibble_commands: __SlateAlign16<[alps_nibble_commands; 16]> = __SlateAlign16([{
let mut __t0: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t0.command = 240i32;
__t0.data = 0u8;
__t0
}, {
let mut __t1: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t1.command = 246i32;
__t1.data = 0u8;
__t1
}, {
let mut __t2: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t2.command = 231i32;
__t2.data = 0u8;
__t2
}, {
let mut __t3: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t3.command = 4339i32;
__t3.data = 10u8;
__t3
}, {
let mut __t4: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t4.command = 4339i32;
__t4.data = 20u8;
__t4
}, {
let mut __t5: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t5.command = 4339i32;
__t5.data = 40u8;
__t5
}, {
let mut __t6: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t6.command = 4339i32;
__t6.data = 60u8;
__t6
}, {
let mut __t7: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t7.command = 4339i32;
__t7.data = 80u8;
__t7
}, {
let mut __t8: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t8.command = 4339i32;
__t8.data = 100u8;
__t8
}, {
let mut __t9: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t9.command = 4339i32;
__t9.data = 200u8;
__t9
}, {
let mut __t10: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t10.command = 498i32;
__t10.data = 0u8;
__t10
}, {
let mut __t11: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t11.command = 4328i32;
__t11.data = 0u8;
__t11
}, {
let mut __t12: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t12.command = 4328i32;
__t12.data = 1u8;
__t12
}, {
let mut __t13: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t13.command = 4328i32;
__t13.data = 2u8;
__t13
}, {
let mut __t14: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t14.command = 4328i32;
__t14.data = 3u8;
__t14
}, {
let mut __t15: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t15.command = 230i32;
__t15.data = 0u8;
__t15
}]);

/// 0
/// 1
/// 2
/// 3
/// 4
/// 5
/// 6
/// 7
/// 8
/// 9
/// a
/// b
/// c
/// d
/// e
/// f
static alps_v4_nibble_commands: __SlateAlign16<[alps_nibble_commands; 16]> = __SlateAlign16([{
let mut __t0: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t0.command = 244i32;
__t0.data = 0u8;
__t0
}, {
let mut __t1: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t1.command = 246i32;
__t1.data = 0u8;
__t1
}, {
let mut __t2: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t2.command = 231i32;
__t2.data = 0u8;
__t2
}, {
let mut __t3: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t3.command = 4339i32;
__t3.data = 10u8;
__t3
}, {
let mut __t4: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t4.command = 4339i32;
__t4.data = 20u8;
__t4
}, {
let mut __t5: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t5.command = 4339i32;
__t5.data = 40u8;
__t5
}, {
let mut __t6: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t6.command = 4339i32;
__t6.data = 60u8;
__t6
}, {
let mut __t7: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t7.command = 4339i32;
__t7.data = 80u8;
__t7
}, {
let mut __t8: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t8.command = 4339i32;
__t8.data = 100u8;
__t8
}, {
let mut __t9: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t9.command = 4339i32;
__t9.data = 200u8;
__t9
}, {
let mut __t10: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t10.command = 498i32;
__t10.data = 0u8;
__t10
}, {
let mut __t11: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t11.command = 4328i32;
__t11.data = 0u8;
__t11
}, {
let mut __t12: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t12.command = 4328i32;
__t12.data = 1u8;
__t12
}, {
let mut __t13: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t13.command = 4328i32;
__t13.data = 2u8;
__t13
}, {
let mut __t14: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t14.command = 4328i32;
__t14.data = 3u8;
__t14
}, {
let mut __t15: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t15.command = 230i32;
__t15.data = 0u8;
__t15
}]);

/// 0
/// 1
/// 2
/// 3
/// 4
/// 5
/// 6
/// 7
/// 8
/// 9
/// a
/// b
/// c
/// d
/// e
/// f
static alps_v6_nibble_commands: __SlateAlign16<[alps_nibble_commands; 16]> = __SlateAlign16([{
let mut __t0: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t0.command = 244i32;
__t0.data = 0u8;
__t0
}, {
let mut __t1: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t1.command = 4339i32;
__t1.data = 10u8;
__t1
}, {
let mut __t2: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t2.command = 4339i32;
__t2.data = 20u8;
__t2
}, {
let mut __t3: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t3.command = 4339i32;
__t3.data = 40u8;
__t3
}, {
let mut __t4: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t4.command = 4339i32;
__t4.data = 60u8;
__t4
}, {
let mut __t5: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t5.command = 4339i32;
__t5.data = 80u8;
__t5
}, {
let mut __t6: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t6.command = 4339i32;
__t6.data = 100u8;
__t6
}, {
let mut __t7: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t7.command = 4339i32;
__t7.data = 200u8;
__t7
}, {
let mut __t8: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t8.command = 754i32;
__t8.data = 0u8;
__t8
}, {
let mut __t9: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t9.command = 1001i32;
__t9.data = 0u8;
__t9
}, {
let mut __t10: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t10.command = 4328i32;
__t10.data = 0u8;
__t10
}, {
let mut __t11: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t11.command = 4328i32;
__t11.data = 1u8;
__t11
}, {
let mut __t12: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t12.command = 4328i32;
__t12.data = 2u8;
__t12
}, {
let mut __t13: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t13.command = 4328i32;
__t13.data = 3u8;
__t13
}, {
let mut __t14: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t14.command = 231i32;
__t14.data = 0u8;
__t14
}, {
let mut __t15: alps_nibble_commands = unsafe { core::mem::zeroed() };
__t15.command = 230i32;
__t15.data = 0u8;
__t15
}]);

// touchpad has trackstick
// device has a pass-through port
// hardware wheel present
// front & back buttons present
// front & back buttons present
// 4 direction button present
// 3-byte PS/2 packet interleaved with
// 6-byte ALPS packet
// separate stick button bits
// device is a clickpad
// device can report trackpoint pressure
/// XXX This entry is suspicious. First byte has zero lower nibble,
/// which is what a normal mouse would report. Also, the value 0x0e
/// isn't valid per PS/2 spec.
/// Dell Latitude D600
/// Toshiba Salellite Pro M10
/// UMAX-530T
/// Toshiba Tecra A11-11L
/// HP ze1115
/// Dell Latitude E5500, E6400, E6500, Precision M4400
/// Fujitsu Siemens S6010
/// Toshiba Satellite S2400-103
/// NEC Versa L320
/// Dell Latitude D800
/// ThinkPad R61 8918-5QG
/// Dell XT2
/// Ahtec Laptop
/// Dell Vostro 1400
static alps_model_data: __SlateAlign16<[alps_model_info; 22]> = __SlateAlign16([{
let mut __t0: alps_model_info = unsafe { core::mem::zeroed() };
__t0.signature = [32u8, 2u8, 14u8];
__t0.protocol_info = alps_protocol_info { version: 512u16, byte0: 248u8, mask0: 248u8, flags: 6u32 };
__t0
}, {
let mut __t1: alps_model_info = unsafe { core::mem::zeroed() };
__t1.signature = [34u8, 2u8, 10u8];
__t1.protocol_info = alps_protocol_info { version: 512u16, byte0: 248u8, mask0: 248u8, flags: 6u32 };
__t1
}, {
let mut __t2: alps_model_info = unsafe { core::mem::zeroed() };
__t2.signature = [34u8, 2u8, 20u8];
__t2.protocol_info = alps_protocol_info { version: 512u16, byte0: 255u8, mask0: 255u8, flags: 6u32 };
__t2
}, {
let mut __t3: alps_model_info = unsafe { core::mem::zeroed() };
__t3.signature = [50u8, 2u8, 20u8];
__t3.protocol_info = alps_protocol_info { version: 512u16, byte0: 248u8, mask0: 248u8, flags: 6u32 };
__t3
}, {
let mut __t4: alps_model_info = unsafe { core::mem::zeroed() };
__t4.signature = [51u8, 2u8, 10u8];
__t4.protocol_info = alps_protocol_info { version: 256u16, byte0: 136u8, mask0: 248u8, flags: 0u32 };
__t4
}, {
let mut __t5: alps_model_info = unsafe { core::mem::zeroed() };
__t5.signature = [82u8, 1u8, 20u8];
__t5.protocol_info = alps_protocol_info { version: 512u16, byte0: 255u8, mask0: 255u8, flags: 134u32 };
__t5
}, {
let mut __t6: alps_model_info = unsafe { core::mem::zeroed() };
__t6.signature = [83u8, 2u8, 10u8];
__t6.protocol_info = alps_protocol_info { version: 512u16, byte0: 248u8, mask0: 248u8, flags: 0u32 };
__t6
}, {
let mut __t7: alps_model_info = unsafe { core::mem::zeroed() };
__t7.signature = [83u8, 2u8, 20u8];
__t7.protocol_info = alps_protocol_info { version: 512u16, byte0: 248u8, mask0: 248u8, flags: 0u32 };
__t7
}, {
let mut __t8: alps_model_info = unsafe { core::mem::zeroed() };
__t8.signature = [96u8, 3u8, 200u8];
__t8.protocol_info = alps_protocol_info { version: 512u16, byte0: 248u8, mask0: 248u8, flags: 0u32 };
__t8
}, {
let mut __t9: alps_model_info = unsafe { core::mem::zeroed() };
__t9.signature = [98u8, 2u8, 20u8];
__t9.protocol_info = alps_protocol_info { version: 512u16, byte0: 207u8, mask0: 207u8, flags: 134u32 };
__t9
}, {
let mut __t10: alps_model_info = unsafe { core::mem::zeroed() };
__t10.signature = [99u8, 2u8, 10u8];
__t10.protocol_info = alps_protocol_info { version: 512u16, byte0: 248u8, mask0: 248u8, flags: 0u32 };
__t10
}, {
let mut __t11: alps_model_info = unsafe { core::mem::zeroed() };
__t11.signature = [99u8, 2u8, 20u8];
__t11.protocol_info = alps_protocol_info { version: 512u16, byte0: 248u8, mask0: 248u8, flags: 0u32 };
__t11
}, {
let mut __t12: alps_model_info = unsafe { core::mem::zeroed() };
__t12.signature = [99u8, 2u8, 40u8];
__t12.protocol_info = alps_protocol_info { version: 512u16, byte0: 248u8, mask0: 248u8, flags: 32u32 };
__t12
}, {
let mut __t13: alps_model_info = unsafe { core::mem::zeroed() };
__t13.signature = [99u8, 2u8, 60u8];
__t13.protocol_info = alps_protocol_info { version: 512u16, byte0: 143u8, mask0: 143u8, flags: 8u32 };
__t13
}, {
let mut __t14: alps_model_info = unsafe { core::mem::zeroed() };
__t14.signature = [99u8, 2u8, 80u8];
__t14.protocol_info = alps_protocol_info { version: 512u16, byte0: 239u8, mask0: 239u8, flags: 16u32 };
__t14
}, {
let mut __t15: alps_model_info = unsafe { core::mem::zeroed() };
__t15.signature = [99u8, 2u8, 100u8];
__t15.protocol_info = alps_protocol_info { version: 512u16, byte0: 248u8, mask0: 248u8, flags: 0u32 };
__t15
}, {
let mut __t16: alps_model_info = unsafe { core::mem::zeroed() };
__t16.signature = [99u8, 3u8, 200u8];
__t16.protocol_info = alps_protocol_info { version: 512u16, byte0: 248u8, mask0: 248u8, flags: 6u32 };
__t16
}, {
let mut __t17: alps_model_info = unsafe { core::mem::zeroed() };
__t17.signature = [115u8, 0u8, 10u8];
__t17.protocol_info = alps_protocol_info { version: 512u16, byte0: 248u8, mask0: 248u8, flags: 2u32 };
__t17
}, {
let mut __t18: alps_model_info = unsafe { core::mem::zeroed() };
__t18.signature = [115u8, 0u8, 20u8];
__t18.protocol_info = alps_protocol_info { version: 1536u16, byte0: 255u8, mask0: 255u8, flags: 2u32 };
__t18
}, {
let mut __t19: alps_model_info = unsafe { core::mem::zeroed() };
__t19.signature = [115u8, 2u8, 10u8];
__t19.protocol_info = alps_protocol_info { version: 512u16, byte0: 248u8, mask0: 248u8, flags: 0u32 };
__t19
}, {
let mut __t20: alps_model_info = unsafe { core::mem::zeroed() };
__t20.signature = [115u8, 2u8, 20u8];
__t20.protocol_info = alps_protocol_info { version: 512u16, byte0: 248u8, mask0: 248u8, flags: 32u32 };
__t20
}, {
let mut __t21: alps_model_info = unsafe { core::mem::zeroed() };
__t21.signature = [115u8, 2u8, 80u8];
__t21.protocol_info = alps_protocol_info { version: 512u16, byte0: 207u8, mask0: 207u8, flags: 64u32 };
__t21
}]);

static alps_v3_protocol_data: alps_protocol_info = alps_protocol_info { version: 768u16, byte0: 143u8, mask0: 143u8, flags: 1026u32 };

static alps_v3_rushmore_data: alps_protocol_info = alps_protocol_info { version: 784u16, byte0: 143u8, mask0: 143u8, flags: 1026u32 };

static alps_v4_protocol_data: alps_protocol_info = alps_protocol_info { version: 1024u16, byte0: 143u8, mask0: 143u8, flags: 0u32 };

static alps_v5_protocol_data: alps_protocol_info = alps_protocol_info { version: 1280u16, byte0: 200u8, mask0: 216u8, flags: 0u32 };

static alps_v7_protocol_data: alps_protocol_info = alps_protocol_info { version: 1792u16, byte0: 72u8, mask0: 72u8, flags: 1026u32 };

static alps_v8_protocol_data: alps_protocol_info = alps_protocol_info { version: 2048u16, byte0: 24u8, mask0: 24u8, flags: 0u32 };

static alps_v9_protocol_data: alps_protocol_info = alps_protocol_info { version: 2304u16, byte0: 200u8, mask0: 200u8, flags: 0u32 };

/// Some v2 models report the stick buttons in separate bits
/// Extrapolated from other entries
/// Reported-by: Hans de Bruin <jmdebruin@xmsnet.nl>
/// Reported-by: Hans de Goede <hdegoede@redhat.com>
/// Extrapolated from other entries
static alps_dmi_has_separate_stick_buttons: __SlateAlign16<[dmi_system_id; 5]> = __SlateAlign16([{
let mut __t0: dmi_system_id = unsafe { core::mem::zeroed() };
__t0.matches = {
let mut __t1: [dmi_strmatch; 4] = unsafe { core::mem::zeroed() };
__t1[0] = {
let mut __t2: dmi_strmatch = unsafe { core::mem::zeroed() };
__t2.__slate_bits_0.__set_slot(6u8);
__t2.substr = [68u8, 101u8, 108u8, 108u8, 32u8, 73u8, 110u8, 99u8, 46u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t2
};
__t1[1] = {
let mut __t3: dmi_strmatch = unsafe { core::mem::zeroed() };
__t3.__slate_bits_0.__set_slot(7u8);
__t3.substr = [76u8, 97u8, 116u8, 105u8, 116u8, 117u8, 100u8, 101u8, 32u8, 68u8, 52u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t3
};
__t1
};
__t0
}, {
let mut __t4: dmi_system_id = unsafe { core::mem::zeroed() };
__t4.matches = {
let mut __t5: [dmi_strmatch; 4] = unsafe { core::mem::zeroed() };
__t5[0] = {
let mut __t6: dmi_strmatch = unsafe { core::mem::zeroed() };
__t6.__slate_bits_0.__set_slot(6u8);
__t6.substr = [68u8, 101u8, 108u8, 108u8, 32u8, 73u8, 110u8, 99u8, 46u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t6
};
__t5[1] = {
let mut __t7: dmi_strmatch = unsafe { core::mem::zeroed() };
__t7.__slate_bits_0.__set_slot(7u8);
__t7.substr = [76u8, 97u8, 116u8, 105u8, 116u8, 117u8, 100u8, 101u8, 32u8, 68u8, 52u8, 51u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t7
};
__t5
};
__t4
}, {
let mut __t8: dmi_system_id = unsafe { core::mem::zeroed() };
__t8.matches = {
let mut __t9: [dmi_strmatch; 4] = unsafe { core::mem::zeroed() };
__t9[0] = {
let mut __t10: dmi_strmatch = unsafe { core::mem::zeroed() };
__t10.__slate_bits_0.__set_slot(6u8);
__t10.substr = [68u8, 101u8, 108u8, 108u8, 32u8, 73u8, 110u8, 99u8, 46u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t10
};
__t9[1] = {
let mut __t11: dmi_strmatch = unsafe { core::mem::zeroed() };
__t11.__slate_bits_0.__set_slot(7u8);
__t11.substr = [76u8, 97u8, 116u8, 105u8, 116u8, 117u8, 100u8, 101u8, 32u8, 68u8, 54u8, 50u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t11
};
__t9
};
__t8
}, {
let mut __t12: dmi_system_id = unsafe { core::mem::zeroed() };
__t12.matches = {
let mut __t13: [dmi_strmatch; 4] = unsafe { core::mem::zeroed() };
__t13[0] = {
let mut __t14: dmi_strmatch = unsafe { core::mem::zeroed() };
__t14.__slate_bits_0.__set_slot(6u8);
__t14.substr = [68u8, 101u8, 108u8, 108u8, 32u8, 73u8, 110u8, 99u8, 46u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t14
};
__t13[1] = {
let mut __t15: dmi_strmatch = unsafe { core::mem::zeroed() };
__t15.__slate_bits_0.__set_slot(7u8);
__t15.substr = [76u8, 97u8, 116u8, 105u8, 116u8, 117u8, 100u8, 101u8, 32u8, 68u8, 54u8, 51u8, 48u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
__t15
};
__t13
};
__t12
}, {
let mut __t16: dmi_system_id = unsafe { core::mem::zeroed() };
__t16
}]);

// Packet formats are described in Documentation/input/devices/alps.rst
fn alps_is_valid_first_byte(r#priv: *mut alps_data, data: u8) -> bool {
(data as i32) & ((unsafe { (*r#priv).mask0 }) as i32) == ((unsafe { (*r#priv).byte0 }) as i32)
}

fn alps_report_buttons(dev1: *mut input_dev, dev2: *mut input_dev, left: i32, right: i32, middle: i32) {
let mut dev: *mut input_dev = unsafe { core::mem::zeroed() };
// If shared button has already been reported on the
// other device (dev2) then this event should be also
// sent through that device.
let __v0: bool = !dev2.is_null() && test_bit!(272i32, unsafe { (*dev2).key.as_mut_ptr() as *mut u64 }) != 0;
dev = if __v0 { dev2 } else { dev1 };
input_report_key(dev, 272u32, left);
let __v2: bool = !dev2.is_null() && test_bit!(273i32, unsafe { (*dev2).key.as_mut_ptr() as *mut u64 }) != 0;
dev = if __v2 { dev2 } else { dev1 };
input_report_key(dev, 273u32, right);
let __v4: bool = !dev2.is_null() && test_bit!(274i32, unsafe { (*dev2).key.as_mut_ptr() as *mut u64 }) != 0;
dev = if __v4 { dev2 } else { dev1 };
input_report_key(dev, 274u32, middle);
// Sync the _other_ device now, we'll do the first
// device later once we report the rest of the events.
if !dev2.is_null() {
input_sync(dev2);
}
}

#[unsafe(link_section = ".text.slate_distinct.alps.alps_process_packet_v1_v2")]
extern "C" fn alps_process_packet_v1_v2(psmouse: *mut psmouse) {
let r#priv: *mut alps_data = (unsafe { (*psmouse).private }) as *mut alps_data;
let packet: *mut u8 = unsafe { (*psmouse).packet.as_mut_ptr() as *mut u8 };
let dev: *mut input_dev = unsafe { (*psmouse).dev };
let dev2: *mut input_dev = unsafe { (*r#priv).dev2 };
let mut x: i32 = 0;
let mut y: i32 = 0;
let mut z: i32 = 0;
let mut ges: i32 = 0;
let mut fin: i32 = 0;
let mut left: i32 = 0;
let mut right: i32 = 0;
let mut middle: i32 = 0;
let mut back: i32 = 0;
let mut forward: i32 = 0;
if ((unsafe { (*r#priv).proto_version }) as i32) == 256 {
left = ((unsafe { *packet.wrapping_offset(2 as isize) }) as i32) & 16;
right = ((unsafe { *packet.wrapping_offset(2 as isize) }) as i32) & 8;
middle = 0;
x = ((unsafe { *packet.wrapping_offset(1 as isize) }) as i32) | (((unsafe { *packet.wrapping_offset(0 as isize) }) as i32) & 7) << 7i32;
y = ((unsafe { *packet.wrapping_offset(4 as isize) }) as i32) | (((unsafe { *packet.wrapping_offset(3 as isize) }) as i32) & 7) << 7i32;
z = (unsafe { *packet.wrapping_offset(5 as isize) }) as i32;
} else {
left = ((unsafe { *packet.wrapping_offset(3 as isize) }) as i32) & 1;
right = ((unsafe { *packet.wrapping_offset(3 as isize) }) as i32) & 2;
middle = ((unsafe { *packet.wrapping_offset(3 as isize) }) as i32) & 4;
x = ((unsafe { *packet.wrapping_offset(1 as isize) }) as i32) | (((unsafe { *packet.wrapping_offset(2 as isize) }) as i32) & 120) << 4i32;
y = ((unsafe { *packet.wrapping_offset(4 as isize) }) as i32) | (((unsafe { *packet.wrapping_offset(3 as isize) }) as i32) & 112) << 3i32;
z = (unsafe { *packet.wrapping_offset(5 as isize) }) as i32;
}
if (unsafe { (*r#priv).flags }) & 16 != 0 {
back = ((unsafe { *packet.wrapping_offset(0 as isize) }) as i32) & 16;
forward = ((unsafe { *packet.wrapping_offset(2 as isize) }) as i32) & 4;
}
if (unsafe { (*r#priv).flags }) & 32 != 0 {
back = ((unsafe { *packet.wrapping_offset(3 as isize) }) as i32) & 4;
forward = ((unsafe { *packet.wrapping_offset(2 as isize) }) as i32) & 4;
let __v0: i32 = (forward != 0 && back != 0) as i32;
middle = __v0;
if __v0 != 0 {
back = 0;
forward = 0;
}
}
ges = ((unsafe { *packet.wrapping_offset(2 as isize) }) as i32) & 1;
fin = ((unsafe { *packet.wrapping_offset(2 as isize) }) as i32) & 2;
if (unsafe { (*r#priv).flags }) & 2 != 0 && z == 127 {
input_report_rel(dev2, 0u32, if x > 383 { x.wrapping_sub(768) } else { x });
input_report_rel(dev2, 1u32, if y > 255 { y.wrapping_sub(512) } else { y }.wrapping_neg());
alps_report_buttons(dev2, dev, left, right, middle);
input_sync(dev2);
return;
}
// Some models have separate stick button bits
if (unsafe { (*r#priv).flags }) & 256 != 0 {
left |= ((unsafe { *packet.wrapping_offset(0 as isize) }) as i32) & 1;
right |= ((unsafe { *packet.wrapping_offset(0 as isize) }) as i32) & 2;
middle |= ((unsafe { *packet.wrapping_offset(0 as isize) }) as i32) & 4;
}
alps_report_buttons(dev, dev2, left, right, middle);
// Convert hardware tap to a reasonable Z value
if ges != 0 && fin == 0 {
z = 40;
}
// A "tap and drag" operation is reported by the hardware as a transition
// from (!fin && ges) to (fin && ges). This should be translated to the
// sequence Z>0, Z==0, Z>0, so the Z==0 event has to be generated manually.
if ges != 0 && fin != 0 && (unsafe { (*r#priv).prev_fin }) == 0 {
input_report_abs(dev, 0u32, x);
input_report_abs(dev, 1u32, y);
input_report_abs(dev, 24u32, 0i32);
input_report_key(dev, 325u32, 0i32);
input_sync(dev);
}
unsafe {
(*r#priv).prev_fin = fin;
}
if z > 30 {
input_report_key(dev, 330u32, 1i32);
}
if z < 25 {
input_report_key(dev, 330u32, 0i32);
}
if z > 0 {
input_report_abs(dev, 0u32, x);
input_report_abs(dev, 1u32, y);
}
input_report_abs(dev, 24u32, z);
input_report_key(dev, 325u32, (z > 0) as i32);
if (unsafe { (*r#priv).flags }) & 8 != 0 {
input_report_rel(dev, 8u32, (((unsafe { *packet.wrapping_offset(2 as isize) }) as i32) << 1i32 & 8).wrapping_sub(((unsafe { *packet.wrapping_offset(0 as isize) }) as i32) >> 4i32 & 7));
}
if (unsafe { (*r#priv).flags }) & (16 | 32) != 0 {
input_report_key(dev, 277u32, forward);
input_report_key(dev, 278u32, back);
}
if (unsafe { (*r#priv).flags }) & 64 != 0 {
input_report_key(dev, 256u32, ((unsafe { *packet.wrapping_offset(2 as isize) }) as i32) & 4);
input_report_key(dev, 257u32, ((unsafe { *packet.wrapping_offset(0 as isize) }) as i32) & 16);
input_report_key(dev, 258u32, ((unsafe { *packet.wrapping_offset(3 as isize) }) as i32) & 4);
input_report_key(dev, 259u32, ((unsafe { *packet.wrapping_offset(0 as isize) }) as i32) & 32);
}
input_sync(dev);
}

fn alps_get_bitmap_points(mut map: u32, low: &mut alps_bitmap_point, high: &alps_bitmap_point, fingers: &mut i32) {
let mut point: *mut alps_bitmap_point = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
let mut bit: i32 = 0;
let mut prev_bit: i32 = 0;
point = low as *mut alps_bitmap_point;
while map != 0 {
bit = (map & 1) as i32;
if bit != 0 {
if prev_bit == 0 {
unsafe {
(*point).start_bit = i;
}
unsafe {
(*point).num_bits = 0;
}
let __v1: i32 = unsafe { *fingers };
unsafe {
*fingers = __v1.wrapping_add(1);
}
}
let __v4: i32 = unsafe { (*point).num_bits };
unsafe {
(*point).num_bits = __v4.wrapping_add(1);
}
} else {
if prev_bit != 0 {
point = (high as *const alps_bitmap_point) as *mut alps_bitmap_point;
}
}
prev_bit = bit;
i = i.wrapping_add(1);
map >>= 1i32;
}
}

// Process bitmap data from semi-mt protocols. Returns the number of
// fingers detected. A return value of 0 means at least one of the
// bitmaps was empty.
//
// The bitmaps don't have enough data to track fingers, so this function
// only generates points representing a bounding box of all contacts.
// These points are returned in fields->mt when the return value
// is greater than 0.
fn alps_process_bitmap(r#priv: *mut alps_data, fields: *mut alps_fields) -> i32 {
let mut i: i32 = 0;
let mut fingers_x: i32 = 0;
let mut fingers_y: i32 = 0;
let mut fingers: i32 = 0;
let mut closest: i32 = 0;
let mut x_low: alps_bitmap_point = {
let mut __t0: alps_bitmap_point = unsafe { core::mem::zeroed() };
__t0.start_bit = 0i32;
__t0
};
let mut x_high: alps_bitmap_point = {
let mut __t1: alps_bitmap_point = unsafe { core::mem::zeroed() };
__t1.start_bit = 0i32;
__t1
};
let mut y_low: alps_bitmap_point = {
let mut __t2: alps_bitmap_point = unsafe { core::mem::zeroed() };
__t2.start_bit = 0i32;
__t2
};
let mut y_high: alps_bitmap_point = {
let mut __t3: alps_bitmap_point = unsafe { core::mem::zeroed() };
__t3.start_bit = 0i32;
__t3
};
let mut corner: __SlateAlign16<[input_mt_pos; 4]> = __SlateAlign16(unsafe { core::mem::zeroed() });
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(corner.0) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
if (unsafe { (*fields).x_map }) == 0 || (unsafe { (*fields).y_map }) == 0 {
return 0;
}
alps_get_bitmap_points(unsafe { (*fields).x_map }, unsafe { &mut (*core::ptr::addr_of_mut!(x_low)) }, unsafe { &(*core::ptr::addr_of_mut!(x_high)) }, unsafe { &mut (*core::ptr::addr_of_mut!(fingers_x)) });
alps_get_bitmap_points(unsafe { (*fields).y_map }, unsafe { &mut (*core::ptr::addr_of_mut!(y_low)) }, unsafe { &(*core::ptr::addr_of_mut!(y_high)) }, unsafe { &mut (*core::ptr::addr_of_mut!(fingers_y)) });
// Fingers can overlap, so we use the maximum count of fingers
// on either axis as the finger count.
let mut __v0: i32;
let __UNIQUE_ID_x__459: i32 = fingers_x;
let __UNIQUE_ID_y__460: i32 = fingers_y;
loop {
if __types_ok__2!(__UNIQUE_ID_x__459, __UNIQUE_ID_y__460) == 0 {
unsafe { __compiletime_assert_461() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__459 > __UNIQUE_ID_y__460 { __UNIQUE_ID_x__459 } else { __UNIQUE_ID_y__460 };
fingers = __v0;
// If an axis reports only a single contact, we have overlapping or
// adjacent fingers. Divide the single contact between the two points.
if fingers_x == 1 {
i = x_low.num_bits.wrapping_sub(1) / 2;
x_low.num_bits = x_low.num_bits.wrapping_sub(i);
x_high.start_bit = x_low.start_bit.wrapping_add(i);
let mut __v2: i32;
let __UNIQUE_ID_x__462: i32 = i;
let __UNIQUE_ID_y__463: i32 = 1;
loop {
if __types_ok__2!(__UNIQUE_ID_x__462, __UNIQUE_ID_y__463) == 0 {
unsafe { __compiletime_assert_464() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = if __UNIQUE_ID_x__462 > __UNIQUE_ID_y__463 { __UNIQUE_ID_x__462 } else { __UNIQUE_ID_y__463 };
x_high.num_bits = __v2;
}
if fingers_y == 1 {
i = y_low.num_bits.wrapping_sub(1) / 2;
y_low.num_bits = y_low.num_bits.wrapping_sub(i);
y_high.start_bit = y_low.start_bit.wrapping_add(i);
let mut __v4: i32;
let __UNIQUE_ID_x__465: i32 = i;
let __UNIQUE_ID_y__466: i32 = 1;
loop {
if __types_ok__2!(__UNIQUE_ID_x__465, __UNIQUE_ID_y__466) == 0 {
unsafe { __compiletime_assert_467() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v4 = if __UNIQUE_ID_x__465 > __UNIQUE_ID_y__466 { __UNIQUE_ID_x__465 } else { __UNIQUE_ID_y__466 };
y_high.num_bits = __v4;
}
// top-left corner
corner.0[0].x = (unsafe { (*r#priv).x_max }.wrapping_mul(2i32.wrapping_mul(x_low.start_bit).wrapping_add(x_low.num_bits).wrapping_sub(1)) / 2i32.wrapping_mul(unsafe { (*r#priv).x_bits }.wrapping_sub(1))) as i16;
corner.0[0].y = (unsafe { (*r#priv).y_max }.wrapping_mul(2i32.wrapping_mul(y_low.start_bit).wrapping_add(y_low.num_bits).wrapping_sub(1)) / 2i32.wrapping_mul(unsafe { (*r#priv).y_bits }.wrapping_sub(1))) as i16;
// top-right corner
corner.0[1].x = (unsafe { (*r#priv).x_max }.wrapping_mul(2i32.wrapping_mul(x_high.start_bit).wrapping_add(x_high.num_bits).wrapping_sub(1)) / 2i32.wrapping_mul(unsafe { (*r#priv).x_bits }.wrapping_sub(1))) as i16;
corner.0[1].y = (unsafe { (*r#priv).y_max }.wrapping_mul(2i32.wrapping_mul(y_low.start_bit).wrapping_add(y_low.num_bits).wrapping_sub(1)) / 2i32.wrapping_mul(unsafe { (*r#priv).y_bits }.wrapping_sub(1))) as i16;
// bottom-right corner
corner.0[2].x = (unsafe { (*r#priv).x_max }.wrapping_mul(2i32.wrapping_mul(x_high.start_bit).wrapping_add(x_high.num_bits).wrapping_sub(1)) / 2i32.wrapping_mul(unsafe { (*r#priv).x_bits }.wrapping_sub(1))) as i16;
corner.0[2].y = (unsafe { (*r#priv).y_max }.wrapping_mul(2i32.wrapping_mul(y_high.start_bit).wrapping_add(y_high.num_bits).wrapping_sub(1)) / 2i32.wrapping_mul(unsafe { (*r#priv).y_bits }.wrapping_sub(1))) as i16;
// bottom-left corner
corner.0[3].x = (unsafe { (*r#priv).x_max }.wrapping_mul(2i32.wrapping_mul(x_low.start_bit).wrapping_add(x_low.num_bits).wrapping_sub(1)) / 2i32.wrapping_mul(unsafe { (*r#priv).x_bits }.wrapping_sub(1))) as i16;
corner.0[3].y = (unsafe { (*r#priv).y_max }.wrapping_mul(2i32.wrapping_mul(y_high.start_bit).wrapping_add(y_high.num_bits).wrapping_sub(1)) / 2i32.wrapping_mul(unsafe { (*r#priv).y_bits }.wrapping_sub(1))) as i16;
// x-bitmap order is reversed on v5 touchpads
if ((unsafe { (*r#priv).proto_version }) as i32) == 1280 {
i = 0;
while i < 4 {
corner.0[i as usize].x = unsafe { (*r#priv).x_max }.wrapping_sub(corner.0[i as usize].x as i32) as i16;
i = i.wrapping_add(1);
}
}
// y-bitmap order is reversed on v3 and v4 touchpads
if ((unsafe { (*r#priv).proto_version }) as i32) == 768 || ((unsafe { (*r#priv).proto_version }) as i32) == 1024 {
i = 0;
while i < 4 {
corner.0[i as usize].y = unsafe { (*r#priv).y_max }.wrapping_sub(corner.0[i as usize].y as i32) as i16;
i = i.wrapping_add(1);
}
}
// We only select a corner for the second touch once per 2 finger
// touch sequence to avoid the chosen corner (and thus the coordinates)
// jumping around when the first touch is in the middle.
if (unsafe { (*r#priv).second_touch }) == 1i32.wrapping_neg() {
// Find corner closest to our st coordinates
closest = 2147483647;
i = 0;
while i < 4 {
let dx: i32 = ((unsafe { (*fields).st.x }) as i32).wrapping_sub(corner.0[i as usize].x as i32);
let dy: i32 = ((unsafe { (*fields).st.y }) as i32).wrapping_sub(corner.0[i as usize].y as i32);
let distance: i32 = dx.wrapping_mul(dx).wrapping_add(dy.wrapping_mul(dy));
if distance < closest {
unsafe {
(*r#priv).second_touch = i;
}
closest = distance;
}
i = i.wrapping_add(1);
}
// And select the opposite corner to use for the 2nd touch
unsafe {
(*r#priv).second_touch = unsafe { (*r#priv).second_touch }.wrapping_add(2) % 4;
}
}
unsafe {
*((*fields).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(0 as isize) = (*fields).st;
}
unsafe {
*((*fields).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(1 as isize) = corner.0[(*r#priv).second_touch as usize];
}
fingers
}

fn alps_set_slot(dev: *mut input_dev, slot: i32, x: i32, y: i32) {
input_mt_slot(dev, slot);
unsafe { input_mt_report_slot_state(dev, 0u32, true) };
input_report_abs(dev, 53u32, x);
input_report_abs(dev, 54u32, y);
}

fn alps_report_mt_data(psmouse: *mut psmouse, n: i32) {
let r#priv: *mut alps_data = (unsafe { (*psmouse).private }) as *mut alps_data;
let dev: *mut input_dev = unsafe { (*psmouse).dev };
let f: *mut alps_fields = unsafe { core::ptr::addr_of_mut!((*r#priv).f) };
let mut slot: __SlateAlign16<[i32; 4]> = __SlateAlign16([0i32; 4]);
unsafe { input_mt_assign_slots(dev, slot.0.as_mut_ptr() as *mut i32, ((*f).mt.as_mut_ptr() as *mut input_mt_pos) as *const input_mt_pos, n, 0i32) };
for i in 0..n {
alps_set_slot(dev, slot.0[i as usize], (unsafe { (*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(i as isize)).x }) as i32, (unsafe { (*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(i as isize)).y }) as i32);
}
unsafe { input_mt_sync_frame(dev) };
}

fn alps_report_semi_mt_data(psmouse: *mut psmouse, mut fingers: i32) {
let r#priv: *mut alps_data = (unsafe { (*psmouse).private }) as *mut alps_data;
let dev: *mut input_dev = unsafe { (*psmouse).dev };
let f: *mut alps_fields = unsafe { core::ptr::addr_of_mut!((*r#priv).f) };
// Use st data when we don't have mt data
if fingers < 2 {
unsafe {
(*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(0 as isize)).x = (*f).st.x;
}
unsafe {
(*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(0 as isize)).y = (*f).st.y;
}
fingers = if (unsafe { (*f).pressure }) > 0 { 1 } else { 0 };
unsafe {
(*r#priv).second_touch = 1i32.wrapping_neg();
}
}
if fingers >= 1 {
alps_set_slot(dev, 0i32, (unsafe { (*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(0 as isize)).x }) as i32, (unsafe { (*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(0 as isize)).y }) as i32);
}
if fingers >= 2 {
alps_set_slot(dev, 1i32, (unsafe { (*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(1 as isize)).x }) as i32, (unsafe { (*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(1 as isize)).y }) as i32);
}
unsafe { input_mt_sync_frame(dev) };
unsafe { input_mt_report_finger_count(dev, fingers) };
input_report_key(dev, 272u32, (unsafe { (*f).__slate_bits_0.__get_left() }) as i32);
input_report_key(dev, 273u32, (unsafe { (*f).__slate_bits_0.__get_right() }) as i32);
input_report_key(dev, 274u32, (unsafe { (*f).__slate_bits_0.__get_middle() }) as i32);
input_report_abs(dev, 24u32, unsafe { (*f).pressure });
input_sync(dev);
}

fn alps_process_trackstick_packet_v3(psmouse: *mut psmouse) {
let r#priv: *mut alps_data = (unsafe { (*psmouse).private }) as *mut alps_data;
let packet: *mut u8 = unsafe { (*psmouse).packet.as_mut_ptr() as *mut u8 };
let dev: *mut input_dev = unsafe { (*r#priv).dev2 };
let mut x: i32 = 0;
let mut y: i32 = 0;
let mut z: i32 = 0;
let mut left: i32 = 0;
let mut right: i32 = 0;
let mut middle: i32 = 0;
// It should be a DualPoint when received trackstick packet
if (unsafe { (*r#priv).flags }) & 2 == 0 {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*(*psmouse).ps2dev.serio).dev) as *const device, c"alps: Rejected trackstick packet from non DualPoint device".as_ptr() as *const u8) };
return;
}
// Sanity check packet
if ((unsafe { *packet.wrapping_offset(0 as isize) }) as i32) & 64 == 0 {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*(*psmouse).ps2dev.serio).dev) as *const device, c"alps: Bad trackstick packet, discarding\n".as_ptr() as *const u8) };
}
return;
}
// There's a special packet that seems to indicate the end
// of a stream of trackstick data. Filter these out.
if ((unsafe { *packet.wrapping_offset(1 as isize) }) as i32) == 127 && ((unsafe { *packet.wrapping_offset(2 as isize) }) as i32) == 127 && ((unsafe { *packet.wrapping_offset(4 as isize) }) as i32) == 127 {
return;
}
x = (((((unsafe { *packet.wrapping_offset(0 as isize) }) as i32) & 32) << 2i32 | ((unsafe { *packet.wrapping_offset(1 as isize) }) as i32) & 127) as i8) as i32;
y = (((((unsafe { *packet.wrapping_offset(0 as isize) }) as i32) & 16) << 3i32 | ((unsafe { *packet.wrapping_offset(2 as isize) }) as i32) & 127) as i8) as i32;
z = ((unsafe { *packet.wrapping_offset(4 as isize) }) as i32) & 127;
// The x and y values tend to be quite large, and when used
// alone the trackstick is difficult to use. Scale them down
// to compensate.
x /= 8;
y /= 8;
input_report_rel(dev, 0u32, x);
input_report_rel(dev, 1u32, y.wrapping_neg());
input_report_abs(dev, 24u32, z);
// Most ALPS models report the trackstick buttons in the touchpad
// packets, but a few report them here. No reliable way has been
// found to differentiate between the models upfront, so we enable
// the quirk in response to seeing a button press in the trackstick
// packet.
left = ((unsafe { *packet.wrapping_offset(3 as isize) }) as i32) & 1;
right = ((unsafe { *packet.wrapping_offset(3 as isize) }) as i32) & 2;
middle = ((unsafe { *packet.wrapping_offset(3 as isize) }) as i32) & 4;
if ((unsafe { (*r#priv).quirks }) as i32) & 1 == 0 && (left != 0 || right != 0 || middle != 0) {
unsafe {
(*r#priv).quirks = (((*r#priv).quirks as i32) | 1) as u8;
}
}
if ((unsafe { (*r#priv).quirks }) as i32) & 1 != 0 {
input_report_key(dev, 272u32, left);
input_report_key(dev, 273u32, right);
input_report_key(dev, 274u32, middle);
}
input_sync(dev);
return;
}

fn alps_decode_buttons_v3(f: *mut alps_fields, p: *mut u8) {
unsafe {
let __t0 = ((*p.wrapping_offset(3 as isize) as i32) & 1 != 0) as u32;
(*f).__slate_bits_0.__set_left(__t0);
}
unsafe {
let __t1 = ((*p.wrapping_offset(3 as isize) as i32) & 2 != 0) as u32;
(*f).__slate_bits_0.__set_right(__t1);
}
unsafe {
let __t2 = ((*p.wrapping_offset(3 as isize) as i32) & 4 != 0) as u32;
(*f).__slate_bits_0.__set_middle(__t2);
}
unsafe {
let __t3 = ((*p.wrapping_offset(3 as isize) as i32) & 16 != 0) as u32;
(*f).__slate_bits_0.__set_ts_left(__t3);
}
unsafe {
let __t4 = ((*p.wrapping_offset(3 as isize) as i32) & 32 != 0) as u32;
(*f).__slate_bits_0.__set_ts_right(__t4);
}
unsafe {
let __t5 = ((*p.wrapping_offset(3 as isize) as i32) & 64 != 0) as u32;
(*f).__slate_bits_0.__set_ts_middle(__t5);
}
}

#[unsafe(link_section = ".text.slate_distinct.alps.alps_decode_pinnacle")]
extern "C" fn alps_decode_pinnacle(f: *mut alps_fields, p: *mut u8, psmouse: *mut psmouse) -> i32 {
unsafe {
let __t0 = ((*p.wrapping_offset(4 as isize) as i32) & 64 != 0) as u32;
(*f).__slate_bits_0.__set_first_mp(__t0);
}
unsafe {
let __t1 = ((*p.wrapping_offset(0 as isize) as i32) & 64 != 0) as u32;
(*f).__slate_bits_0.__set_is_mp(__t1);
}
if ((unsafe { (*f).__slate_bits_0.__get_is_mp() }) as i32) != 0 {
unsafe {
(*f).fingers = ((*p.wrapping_offset(5 as isize) as i32) & 3).wrapping_add(1) as u32;
}
unsafe {
(*f).x_map = (((*p.wrapping_offset(4 as isize) as i32) & 126) << 8i32 | ((*p.wrapping_offset(1 as isize) as i32) & 127) << 2i32 | ((*p.wrapping_offset(0 as isize) as i32) & 48) >> 4i32) as u32;
}
unsafe {
(*f).y_map = (((*p.wrapping_offset(3 as isize) as i32) & 112) << 4i32 | ((*p.wrapping_offset(2 as isize) as i32) & 127) << 1i32 | (*p.wrapping_offset(4 as isize) as i32) & 1) as u32;
}
} else {
unsafe {
(*f).st.x = (((*p.wrapping_offset(1 as isize) as i32) & 127) << 4i32 | ((*p.wrapping_offset(4 as isize) as i32) & 48) >> 2i32 | ((*p.wrapping_offset(0 as isize) as i32) & 48) >> 4i32) as i16;
}
unsafe {
(*f).st.y = (((*p.wrapping_offset(2 as isize) as i32) & 127) << 4i32 | (*p.wrapping_offset(4 as isize) as i32) & 15) as i16;
}
unsafe {
(*f).pressure = (*p.wrapping_offset(5 as isize) as i32) & 127;
}
alps_decode_buttons_v3(f, p);
}
0
}

#[unsafe(link_section = ".text.slate_distinct.alps.alps_decode_rushmore")]
extern "C" fn alps_decode_rushmore(f: *mut alps_fields, p: *mut u8, psmouse: *mut psmouse) -> i32 {
unsafe {
let __t0 = ((*p.wrapping_offset(4 as isize) as i32) & 64 != 0) as u32;
(*f).__slate_bits_0.__set_first_mp(__t0);
}
unsafe {
let __t1 = ((*p.wrapping_offset(5 as isize) as i32) & 64 != 0) as u32;
(*f).__slate_bits_0.__set_is_mp(__t1);
}
if ((unsafe { (*f).__slate_bits_0.__get_is_mp() }) as i32) != 0 {
let mut __v0: i32;
let __UNIQUE_ID_x__468: i32 = ((unsafe { *p.wrapping_offset(5 as isize) }) as i32) & 3;
let __UNIQUE_ID_y__469: i32 = ((unsafe { *p.wrapping_offset(5 as isize) }) as i32) >> 2i32 & 3;
loop {
if __types_ok__2!(__UNIQUE_ID_x__468, __UNIQUE_ID_y__469) == 0 {
unsafe { __compiletime_assert_470() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__468 > __UNIQUE_ID_y__469 { __UNIQUE_ID_x__468 } else { __UNIQUE_ID_y__469 };
unsafe {
(*f).fingers = __v0.wrapping_add(1) as u32;
}
unsafe {
(*f).x_map = (((*p.wrapping_offset(5 as isize) as i32) & 16) << 11i32 | ((*p.wrapping_offset(4 as isize) as i32) & 126) << 8i32 | ((*p.wrapping_offset(1 as isize) as i32) & 127) << 2i32 | ((*p.wrapping_offset(0 as isize) as i32) & 48) >> 4i32) as u32;
}
unsafe {
(*f).y_map = (((*p.wrapping_offset(5 as isize) as i32) & 32) << 6i32 | ((*p.wrapping_offset(3 as isize) as i32) & 112) << 4i32 | ((*p.wrapping_offset(2 as isize) as i32) & 127) << 1i32 | (*p.wrapping_offset(4 as isize) as i32) & 1) as u32;
}
} else {
unsafe {
(*f).st.x = (((*p.wrapping_offset(1 as isize) as i32) & 127) << 4i32 | ((*p.wrapping_offset(4 as isize) as i32) & 48) >> 2i32 | ((*p.wrapping_offset(0 as isize) as i32) & 48) >> 4i32) as i16;
}
unsafe {
(*f).st.y = (((*p.wrapping_offset(2 as isize) as i32) & 127) << 4i32 | (*p.wrapping_offset(4 as isize) as i32) & 15) as i16;
}
unsafe {
(*f).pressure = (*p.wrapping_offset(5 as isize) as i32) & 127;
}
alps_decode_buttons_v3(f, p);
}
0
}

#[unsafe(link_section = ".text.slate_distinct.alps.alps_decode_dolphin")]
extern "C" fn alps_decode_dolphin(f: *mut alps_fields, p: *mut u8, psmouse: *mut psmouse) -> i32 {
let mut palm_data: u64 = 0;
let r#priv: *mut alps_data = (unsafe { (*psmouse).private }) as *mut alps_data;
unsafe {
let __t0 = ((*p.wrapping_offset(0 as isize) as i32) & 2 != 0) as u32;
(*f).__slate_bits_0.__set_first_mp(__t0);
}
unsafe {
let __t1 = ((*p.wrapping_offset(0 as isize) as i32) & 32 != 0) as u32;
(*f).__slate_bits_0.__set_is_mp(__t1);
}
if ((unsafe { (*f).__slate_bits_0.__get_is_mp() }) as i32) == 0 {
unsafe {
(*f).st.x = ((*p.wrapping_offset(1 as isize) as i32) & 127 | ((*p.wrapping_offset(4 as isize) as i32) & 15) << 7i32) as i16;
}
unsafe {
(*f).st.y = ((*p.wrapping_offset(2 as isize) as i32) & 127 | ((*p.wrapping_offset(4 as isize) as i32) & 240) << 3i32) as i16;
}
unsafe {
(*f).pressure = if (*p.wrapping_offset(0 as isize) as i32) & 4 != 0 { 0 } else { (*p.wrapping_offset(5 as isize) as i32) & 127 };
}
alps_decode_buttons_v3(f, p);
} else {
unsafe {
(*f).fingers = (((*p.wrapping_offset(0 as isize) as i32) & 6) >> 1i32 | ((*p.wrapping_offset(0 as isize) as i32) & 16) >> 2i32) as u32;
}
palm_data = ((((unsafe { *p.wrapping_offset(1 as isize) }) as i32) & 127 | (((unsafe { *p.wrapping_offset(2 as isize) }) as i32) & 127) << 7i32 | (((unsafe { *p.wrapping_offset(4 as isize) }) as i32) & 127) << 14i32 | (((unsafe { *p.wrapping_offset(5 as isize) }) as i32) & 127) << 21i32 | (((unsafe { *p.wrapping_offset(3 as isize) }) as i32) & 7) << 28i32) as u64) | (((unsafe { *p.wrapping_offset(3 as isize) }) as u64) & 112) << 27i32 | (((unsafe { *p.wrapping_offset(0 as isize) }) as u64) & 1) << 34i32;
// Y-profile is stored in P(0) to p(n-1), n = y_bits;
unsafe {
(*f).y_map = (palm_data & BIT!(unsafe { (*r#priv).y_bits }).wrapping_sub(1)) as u32;
}
// X-profile is stored in p(n) to p(n+m-1), m = x_bits;
unsafe {
(*f).x_map = (palm_data >> (*r#priv).y_bits & BIT!(unsafe { (*r#priv).x_bits }).wrapping_sub(1)) as u32;
}
}
0
}

#[unsafe(link_section = ".text.slate_distinct.alps.alps_process_touchpad_packet_v3_v5")]
extern "C" fn alps_process_touchpad_packet_v3_v5(psmouse: *mut psmouse) {
let r#priv: *mut alps_data = (unsafe { (*psmouse).private }) as *mut alps_data;
let packet: *mut u8 = unsafe { (*psmouse).packet.as_mut_ptr() as *mut u8 };
let dev2: *mut input_dev = unsafe { (*r#priv).dev2 };
let f: *mut alps_fields = unsafe { core::ptr::addr_of_mut!((*r#priv).f) };
let mut fingers: i32 = 0;
unsafe { memset(f as *mut (), 0i32, core::mem::size_of::<alps_fields>() as u64) };
unsafe { unsafe { (*r#priv).decode_fields }.unwrap()(f, packet, psmouse) };
// There's no single feature of touchpad position and bitmap packets
// that can be used to distinguish between them. We rely on the fact
// that a bitmap packet should always follow a position packet with
// bit 6 of packet[4] set.
if (unsafe { (*r#priv).multi_packet }) != 0 {
// Sometimes a position packet will indicate a multi-packet
// sequence, but then what follows is another position
// packet. Check for this, and when it happens process the
// position packet as usual.
if ((unsafe { (*f).__slate_bits_0.__get_is_mp() }) as i32) != 0 {
fingers = (unsafe { (*f).fingers }) as i32;
// Bitmap processing uses position packet's coordinate
// data, so we need to do decode it first.
unsafe { unsafe { (*r#priv).decode_fields }.unwrap()(f, (*r#priv).multi_data.as_mut_ptr() as *mut u8, psmouse) };
if alps_process_bitmap(r#priv, f) == 0 {
fingers = 0;
}
// Use st data
} else {
unsafe {
(*r#priv).multi_packet = 0;
}
}
}
// Bit 6 of byte 0 is not usually set in position packets. The only
// times it seems to be set is in situations where the data is
// suspect anyway, e.g. a palm resting flat on the touchpad. Given
// this combined with the fact that this bit is useful for filtering
// out misidentified bitmap packets, we reject anything with this
// bit set.
if ((unsafe { (*f).__slate_bits_0.__get_is_mp() }) as i32) != 0 {
return;
}
if (unsafe { (*r#priv).multi_packet }) == 0 && ((unsafe { (*f).__slate_bits_0.__get_first_mp() }) as i32) != 0 {
unsafe {
(*r#priv).multi_packet = 1;
}
unsafe { memcpy(((*r#priv).multi_data.as_mut_ptr() as *mut u8) as *mut (), packet as *const (), core::mem::size_of::<[u8; 6]>() as u64) };
return;
}
unsafe {
(*r#priv).multi_packet = 0;
}
// Sometimes the hardware sends a single packet with z = 0
// in the middle of a stream. Real releases generate packets
// with x, y, and z all zero, so these seem to be flukes.
// Ignore them.
if (unsafe { (*f).st.x }) != 0 && (unsafe { (*f).st.y }) != 0 && (unsafe { (*f).pressure }) == 0 {
return;
}
alps_report_semi_mt_data(psmouse, fingers);
if (unsafe { (*r#priv).flags }) & 2 != 0 && ((unsafe { (*r#priv).quirks }) as i32) & 1 == 0 {
input_report_key(dev2, 272u32, (unsafe { (*f).__slate_bits_0.__get_ts_left() }) as i32);
input_report_key(dev2, 273u32, (unsafe { (*f).__slate_bits_0.__get_ts_right() }) as i32);
input_report_key(dev2, 274u32, (unsafe { (*f).__slate_bits_0.__get_ts_middle() }) as i32);
input_sync(dev2);
}
}

#[unsafe(link_section = ".text.slate_distinct.alps.alps_process_packet_v3")]
extern "C" fn alps_process_packet_v3(psmouse: *mut psmouse) {
let packet: *mut u8 = unsafe { (*psmouse).packet.as_mut_ptr() as *mut u8 };
// v3 protocol packets come in three types, two representing
// touchpad data and one representing trackstick data.
// Trackstick packets seem to be distinguished by always
// having 0x3f in the last byte. This value has never been
// observed in the last byte of either of the other types
// of packets.
if ((unsafe { *packet.wrapping_offset(5 as isize) }) as i32) == 63 {
alps_process_trackstick_packet_v3(psmouse);
return;
}
alps_process_touchpad_packet_v3_v5(psmouse);
}

#[unsafe(link_section = ".text.slate_distinct.alps.alps_process_packet_v6")]
extern "C" fn alps_process_packet_v6(psmouse: *mut psmouse) {
let r#priv: *mut alps_data = (unsafe { (*psmouse).private }) as *mut alps_data;
let packet: *mut u8 = unsafe { (*psmouse).packet.as_mut_ptr() as *mut u8 };
let dev: *mut input_dev = unsafe { (*psmouse).dev };
let dev2: *mut input_dev = unsafe { (*r#priv).dev2 };
let mut x: i32 = 0;
let mut y: i32 = 0;
let mut z: i32 = 0;
// We can use Byte5 to distinguish if the packet is from Touchpad
// or Trackpoint.
// Touchpad:	0 - 0x7E
// Trackpoint:	0x7F
if ((unsafe { *packet.wrapping_offset(5 as isize) }) as i32) == 127 {
// It should be a DualPoint when received Trackpoint packet
if (unsafe { (*r#priv).flags }) & 2 == 0 {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*(*psmouse).ps2dev.serio).dev) as *const device, c"alps: Rejected trackstick packet from non DualPoint device".as_ptr() as *const u8) };
return;
}
// Trackpoint packet
x = ((unsafe { *packet.wrapping_offset(1 as isize) }) as i32) | (((unsafe { *packet.wrapping_offset(3 as isize) }) as i32) & 32) << 2i32;
y = ((unsafe { *packet.wrapping_offset(2 as isize) }) as i32) | (((unsafe { *packet.wrapping_offset(3 as isize) }) as i32) & 64) << 1i32;
z = (unsafe { *packet.wrapping_offset(4 as isize) }) as i32;
// To prevent the cursor jump when finger lifted
if x == 127 && y == 127 && z == 127 {
z = 0;
y = 0;
x = 0;
}
// Divide 4 since trackpoint's speed is too fast
input_report_rel(dev2, 0u32, ((x as i8) as i32) / 4);
input_report_rel(dev2, 1u32, (((y as i8) as i32) / 4).wrapping_neg());
unsafe { psmouse_report_standard_buttons(dev2, *packet.wrapping_offset(3 as isize)) };
input_sync(dev2);
return;
}
// Touchpad packet
x = ((unsafe { *packet.wrapping_offset(1 as isize) }) as i32) | (((unsafe { *packet.wrapping_offset(3 as isize) }) as i32) & 120) << 4i32;
y = ((unsafe { *packet.wrapping_offset(2 as isize) }) as i32) | (((unsafe { *packet.wrapping_offset(4 as isize) }) as i32) & 120) << 4i32;
z = (unsafe { *packet.wrapping_offset(5 as isize) }) as i32;
if z > 30 {
input_report_key(dev, 330u32, 1i32);
}
if z < 25 {
input_report_key(dev, 330u32, 0i32);
}
if z > 0 {
input_report_abs(dev, 0u32, x);
input_report_abs(dev, 1u32, y);
}
input_report_abs(dev, 24u32, z);
input_report_key(dev, 325u32, (z > 0) as i32);
// v6 touchpad does not have middle button
let __v1: *mut u8 = packet.wrapping_offset(3 as isize);
let __v2: u8 = unsafe { *__v1 };
let __v3: u8 = (((__v2 as i32) as u64) & !BIT!(2i32)) as u8;
unsafe {
*__v1 = __v3;
}
unsafe { psmouse_report_standard_buttons(dev2, *packet.wrapping_offset(3 as isize)) };
input_sync(dev);
}

#[unsafe(link_section = ".text.slate_distinct.alps.alps_process_packet_v4")]
extern "C" fn alps_process_packet_v4(psmouse: *mut psmouse) {
let r#priv: *mut alps_data = (unsafe { (*psmouse).private }) as *mut alps_data;
let packet: *mut u8 = unsafe { (*psmouse).packet.as_mut_ptr() as *mut u8 };
let f: *mut alps_fields = unsafe { core::ptr::addr_of_mut!((*r#priv).f) };
let mut offset: i32 = 0;
// v4 has a 6-byte encoding for bitmap data, but this data is
// broken up between 3 normal packets. Use priv->multi_packet to
// track our position in the bitmap packet.
if ((unsafe { *packet.wrapping_offset(6 as isize) }) as i32) & 64 != 0 {
// sync, reset position
unsafe {
(*r#priv).multi_packet = 0;
}
}
let mut __v0: i64;
let __ret_warn_on: i32 = ((unsafe { (*r#priv).multi_packet }) > 2) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("471: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 471b - ., 3; .popsection\n# {0}", const 471 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_alps_process_packet_v4_1, sym __slate_asm_string_alps_process_packet_v4_2, const 907 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("472: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 472b - ., 4; .popsection\n# {0}", const 472 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
return;
}
offset = 2i32.wrapping_mul(unsafe { (*r#priv).multi_packet });
unsafe {
*((*r#priv).multi_data.as_mut_ptr() as *mut u8).wrapping_offset(offset as isize) = *packet.wrapping_offset(6 as isize);
}
unsafe {
*((*r#priv).multi_data.as_mut_ptr() as *mut u8).wrapping_offset(offset.wrapping_add(1) as isize) = *packet.wrapping_offset(7 as isize);
}
unsafe {
let __t0 = ((*packet.wrapping_offset(4 as isize) as i32) & 1 != 0) as u32;
(*f).__slate_bits_0.__set_left(__t0);
}
unsafe {
let __t1 = ((*packet.wrapping_offset(4 as isize) as i32) & 2 != 0) as u32;
(*f).__slate_bits_0.__set_right(__t1);
}
unsafe {
(*f).st.x = (((*packet.wrapping_offset(1 as isize) as i32) & 127) << 4i32 | ((*packet.wrapping_offset(3 as isize) as i32) & 48) >> 2i32 | ((*packet.wrapping_offset(0 as isize) as i32) & 48) >> 4i32) as i16;
}
unsafe {
(*f).st.y = (((*packet.wrapping_offset(2 as isize) as i32) & 127) << 4i32 | (*packet.wrapping_offset(3 as isize) as i32) & 15) as i16;
}
unsafe {
(*f).pressure = (*packet.wrapping_offset(5 as isize) as i32) & 127;
}
let __v6: i32 = unsafe { (*r#priv).multi_packet };
let __v7: i32 = __v6.wrapping_add(1);
unsafe {
(*r#priv).multi_packet = __v7;
}
if __v7 > 2 {
unsafe {
(*r#priv).multi_packet = 0;
}
unsafe {
(*f).x_map = (((*((*r#priv).multi_data.as_mut_ptr() as *mut u8).wrapping_offset(2 as isize) as i32) & 31) << 10i32 | ((*((*r#priv).multi_data.as_mut_ptr() as *mut u8).wrapping_offset(3 as isize) as i32) & 96) << 3i32 | ((*((*r#priv).multi_data.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) as i32) & 63) << 2i32 | ((*((*r#priv).multi_data.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) as i32) & 96) >> 5i32) as u32;
}
unsafe {
(*f).y_map = (((*((*r#priv).multi_data.as_mut_ptr() as *mut u8).wrapping_offset(5 as isize) as i32) & 1) << 10i32 | ((*((*r#priv).multi_data.as_mut_ptr() as *mut u8).wrapping_offset(3 as isize) as i32) & 31) << 5i32 | (*((*r#priv).multi_data.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) as i32) & 31) as u32;
}
unsafe {
(*f).fingers = alps_process_bitmap(r#priv, f) as u32;
}
}
alps_report_semi_mt_data(psmouse, (unsafe { (*f).fingers }) as i32);
}

fn alps_is_valid_package_v7(psmouse: *mut psmouse) -> bool {
match (unsafe { (*psmouse).pktcnt }) as i32 {
3 => {
return ((unsafe { *((*psmouse).packet.as_mut_ptr() as *mut u8).wrapping_offset(2 as isize) }) as i32) & 64 == 64;
}
4 => {
return ((unsafe { *((*psmouse).packet.as_mut_ptr() as *mut u8).wrapping_offset(3 as isize) }) as i32) & 72 == 72;
}
6 => {
return ((unsafe { *((*psmouse).packet.as_mut_ptr() as *mut u8).wrapping_offset(5 as isize) }) as i32) & 64 == 0;
}
_ => {
}
}
true
}

fn alps_get_packet_id_v7(byte: *mut u8) -> u8 {
let mut packet_id: u8 = 0;
if ((unsafe { *byte.wrapping_offset(4 as isize) }) as i32) & 64 != 0 {
packet_id = 1;
} else {
if ((unsafe { *byte.wrapping_offset(4 as isize) }) as i32) & 1 != 0 {
packet_id = 2;
} else {
if ((unsafe { *byte.wrapping_offset(0 as isize) }) as i32) & 16 != 0 && ((unsafe { *byte.wrapping_offset(4 as isize) }) as i32) & 67 == 0 {
packet_id = 3;
} else {
if ((unsafe { *byte.wrapping_offset(1 as isize) }) as i32) == 0 && ((unsafe { *byte.wrapping_offset(4 as isize) }) as i32) == 0 {
packet_id = 0;
} else {
packet_id = 4;
}
}
}
}
packet_id
}

fn alps_get_finger_coordinate_v7(mt: *mut input_mt_pos, pkt: *mut u8, pkt_id: u8) {
unsafe {
(*mt.wrapping_offset(0 as isize)).x = (((*pkt.wrapping_offset(2 as isize) as i32) & 128) << 4i32) as i16;
}
let __v0: *mut input_mt_pos = mt.wrapping_offset(0 as isize);
unsafe {
(*__v0).x = (((*__v0).x as i32) | ((*pkt.wrapping_offset(2 as isize) as i32) & 63) << 5i32) as i16;
}
let __v3: *mut input_mt_pos = mt.wrapping_offset(0 as isize);
unsafe {
(*__v3).x = (((*__v3).x as i32) | ((*pkt.wrapping_offset(3 as isize) as i32) & 48) >> 1i32) as i16;
}
let __v6: *mut input_mt_pos = mt.wrapping_offset(0 as isize);
unsafe {
(*__v6).x = (((*__v6).x as i32) | (*pkt.wrapping_offset(3 as isize) as i32) & 7) as i16;
}
unsafe {
(*mt.wrapping_offset(0 as isize)).y = ((*pkt.wrapping_offset(1 as isize) as i32) << 3i32 | (*pkt.wrapping_offset(0 as isize) as i32) & 7) as i16;
}
unsafe {
(*mt.wrapping_offset(1 as isize)).x = (((*pkt.wrapping_offset(3 as isize) as i32) & 128) << 4i32) as i16;
}
let __v9: *mut input_mt_pos = mt.wrapping_offset(1 as isize);
unsafe {
(*__v9).x = (((*__v9).x as i32) | ((*pkt.wrapping_offset(4 as isize) as i32) & 128) << 3i32) as i16;
}
let __v12: *mut input_mt_pos = mt.wrapping_offset(1 as isize);
unsafe {
(*__v12).x = (((*__v12).x as i32) | ((*pkt.wrapping_offset(4 as isize) as i32) & 63) << 4i32) as i16;
}
unsafe {
(*mt.wrapping_offset(1 as isize)).y = (((*pkt.wrapping_offset(5 as isize) as i32) & 128) << 3i32) as i16;
}
let __v15: *mut input_mt_pos = mt.wrapping_offset(1 as isize);
unsafe {
(*__v15).y = (((*__v15).y as i32) | ((*pkt.wrapping_offset(5 as isize) as i32) & 63) << 4i32) as i16;
}
match pkt_id as i32 {
1 => {
let __v18: *mut input_mt_pos = mt.wrapping_offset(1 as isize);
let __v20: i16 = (((unsafe { (*__v18).x }) as i32) & !15) as i16;
unsafe {
(*__v18).x = __v20;
}
let __v21: *mut input_mt_pos = mt.wrapping_offset(1 as isize);
unsafe {
(*__v21).y = (((*__v21).y as i32) | 15) as i16;
}
// Detect false-positive touches where x & y report max value
if ((unsafe { (*mt.wrapping_offset(1 as isize)).y }) as i32) == 2047 && ((unsafe { (*mt.wrapping_offset(1 as isize)).x }) as i32) == 4080 {
unsafe {
(*mt.wrapping_offset(1 as isize)).x = 0;
}
// y gets set to 0 at the end of this function
}
}
2 => {
let __v25: *mut input_mt_pos = mt.wrapping_offset(1 as isize);
let __v27: i16 = (((unsafe { (*__v25).x }) as i32) & !63) as i16;
unsafe {
(*__v25).x = __v27;
}
let __v28: *mut input_mt_pos = mt.wrapping_offset(1 as isize);
let __v30: i16 = (((unsafe { (*__v28).y }) as i32) & !32) as i16;
unsafe {
(*__v28).y = __v30;
}
let __v31: *mut input_mt_pos = mt.wrapping_offset(1 as isize);
unsafe {
(*__v31).y = (((*__v31).y as i32) | ((*pkt.wrapping_offset(4 as isize) as i32) & 2) << 4i32) as i16;
}
let __v34: *mut input_mt_pos = mt.wrapping_offset(1 as isize);
unsafe {
(*__v34).y = (((*__v34).y as i32) | 31) as i16;
}
}
3 => {
let __v37: *mut input_mt_pos = mt.wrapping_offset(1 as isize);
let __v39: i16 = (((unsafe { (*__v37).x }) as i32) & !63) as i16;
unsafe {
(*__v37).x = __v39;
}
let __v40: *mut input_mt_pos = mt.wrapping_offset(1 as isize);
unsafe {
(*__v40).x = (((*__v40).x as i32) | (*pkt.wrapping_offset(0 as isize) as i32) & 32) as i16;
}
let __v43: *mut input_mt_pos = mt.wrapping_offset(1 as isize);
unsafe {
(*__v43).y = (((*__v43).y as i32) | 15) as i16;
}
}
_ => {
}
}
unsafe {
(*mt.wrapping_offset(0 as isize)).y = 2047i32.wrapping_sub((*mt.wrapping_offset(0 as isize)).y as i32) as i16;
}
unsafe {
(*mt.wrapping_offset(1 as isize)).y = 2047i32.wrapping_sub((*mt.wrapping_offset(1 as isize)).y as i32) as i16;
}
}

fn alps_get_mt_count(mt: *mut input_mt_pos) -> i32 {
let mut fingers: i32 = 0;
for i in 0i32..4i32 {
if ((unsafe { (*mt.wrapping_offset(i as isize)).x }) as i32) != 0 || ((unsafe { (*mt.wrapping_offset(i as isize)).y }) as i32) != 0 {
fingers = fingers.wrapping_add(1);
}
}
fingers
}

#[unsafe(link_section = ".text.slate_distinct.alps.alps_decode_packet_v7")]
extern "C" fn alps_decode_packet_v7(f: *mut alps_fields, p: *mut u8, psmouse: *mut psmouse) -> i32 {
let r#priv: *mut alps_data = (unsafe { (*psmouse).private }) as *mut alps_data;
let pkt_id: u8 = alps_get_packet_id_v7(p as *mut u8);
if (pkt_id as i32) == 0 {
return 0;
}
if (pkt_id as i32) == 4 {
return 1i32.wrapping_neg();
}
// NEW packets are send to indicate a discontinuity in the finger
// coordinate reporting. Specifically a finger may have moved from
// slot 0 to 1 or vice versa. INPUT_MT_TRACK takes care of this for
// us.
//
// NEW packets have 3 problems:
// 1) They do not contain middle / right button info (on non clickpads)
//    this can be worked around by preserving the old button state
// 2) They do not contain an accurate fingercount, and they are
//    typically send when the number of fingers changes. We cannot use
//    the old finger count as that may mismatch with the amount of
//    touch coordinates we've available in the NEW packet
// 3) Their x data for the second touch is inaccurate leading to
//    a possible jump of the x coordinate by 16 units when the first
//    non NEW packet comes in
// Since problems 2 & 3 cannot be worked around, just ignore them.
if (pkt_id as i32) == 3 {
return 1;
}
alps_get_finger_coordinate_v7(unsafe { (*f).mt.as_mut_ptr() as *mut input_mt_pos }, p, pkt_id);
if (pkt_id as i32) == 1 {
unsafe {
(*f).fingers = alps_get_mt_count((*f).mt.as_mut_ptr() as *mut input_mt_pos) as u32;
}
} else {
unsafe {
(*f).fingers = 3i32.wrapping_add((*p.wrapping_offset(5 as isize) as i32) & 3) as u32;
}
}
// pkt_id == V7_PACKET_ID_MULTI
unsafe {
let __t0 = (((*p.wrapping_offset(0 as isize) as i32) & 128) >> 7i32) as u32;
(*f).__slate_bits_0.__set_left(__t0);
}
if (unsafe { (*r#priv).flags }) & 512 != 0 {
if ((unsafe { *p.wrapping_offset(0 as isize) }) as i32) & 32 != 0 {
let __v1: u32 = unsafe { (*f).fingers };
unsafe {
(*f).fingers = __v1.wrapping_add(1);
}
}
if ((unsafe { *p.wrapping_offset(0 as isize) }) as i32) & 16 != 0 {
let __v4: u32 = unsafe { (*f).fingers };
unsafe {
(*f).fingers = __v4.wrapping_add(1);
}
}
} else {
unsafe {
let __t1 = (((*p.wrapping_offset(0 as isize) as i32) & 32) >> 5i32) as u32;
(*f).__slate_bits_0.__set_right(__t1);
}
unsafe {
let __t2 = (((*p.wrapping_offset(0 as isize) as i32) & 16) >> 4i32) as u32;
(*f).__slate_bits_0.__set_middle(__t2);
}
}
// Sometimes a single touch is reported in mt[1] rather then mt[0]
if (unsafe { (*f).fingers }) == 1 && ((unsafe { (*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(0 as isize)).x }) as i32) == 0 && ((unsafe { (*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(0 as isize)).y }) as i32) == 0 {
unsafe {
(*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(0 as isize)).x = (*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(1 as isize)).x;
}
unsafe {
(*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(0 as isize)).y = (*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(1 as isize)).y;
}
unsafe {
(*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(1 as isize)).x = 0;
}
unsafe {
(*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(1 as isize)).y = 0;
}
}
0
}

fn alps_process_trackstick_packet_v7(psmouse: *mut psmouse) {
let r#priv: *mut alps_data = (unsafe { (*psmouse).private }) as *mut alps_data;
let packet: *mut u8 = unsafe { (*psmouse).packet.as_mut_ptr() as *mut u8 };
let dev2: *mut input_dev = unsafe { (*r#priv).dev2 };
let mut x: i32 = 0;
let mut y: i32 = 0;
let mut z: i32 = 0;
// It should be a DualPoint when received trackstick packet
if (unsafe { (*r#priv).flags }) & 2 == 0 {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*(*psmouse).ps2dev.serio).dev) as *const device, c"alps: Rejected trackstick packet from non DualPoint device".as_ptr() as *const u8) };
return;
}
x = ((unsafe { *packet.wrapping_offset(2 as isize) }) as i32) & 191 | (((unsafe { *packet.wrapping_offset(3 as isize) }) as i32) & 16) << 2i32;
y = ((unsafe { *packet.wrapping_offset(3 as isize) }) as i32) & 7 | ((unsafe { *packet.wrapping_offset(4 as isize) }) as i32) & 184 | (((unsafe { *packet.wrapping_offset(3 as isize) }) as i32) & 32) << 1i32;
z = ((unsafe { *packet.wrapping_offset(5 as isize) }) as i32) & 63 | (((unsafe { *packet.wrapping_offset(3 as isize) }) as i32) & 128) >> 1i32;
input_report_rel(dev2, 0u32, (x as i8) as i32);
input_report_rel(dev2, 1u32, ((y as i8) as i32).wrapping_neg());
input_report_abs(dev2, 24u32, z);
unsafe { psmouse_report_standard_buttons(dev2, *packet.wrapping_offset(1 as isize)) };
input_sync(dev2);
}

fn alps_process_touchpad_packet_v7(psmouse: *mut psmouse) {
let r#priv: *mut alps_data = (unsafe { (*psmouse).private }) as *mut alps_data;
let dev: *mut input_dev = unsafe { (*psmouse).dev };
let f: *mut alps_fields = unsafe { core::ptr::addr_of_mut!((*r#priv).f) };
unsafe { memset(f as *mut (), 0i32, core::mem::size_of::<alps_fields>() as u64) };
if (unsafe { unsafe { (*r#priv).decode_fields }.unwrap()(f, (*psmouse).packet.as_mut_ptr() as *mut u8, psmouse) }) != 0 {
return;
}
alps_report_mt_data(psmouse, alps_get_mt_count(unsafe { (*f).mt.as_mut_ptr() as *mut input_mt_pos }));
unsafe { input_mt_report_finger_count(dev, (*f).fingers as i32) };
input_report_key(dev, 272u32, (unsafe { (*f).__slate_bits_0.__get_left() }) as i32);
input_report_key(dev, 273u32, (unsafe { (*f).__slate_bits_0.__get_right() }) as i32);
input_report_key(dev, 274u32, (unsafe { (*f).__slate_bits_0.__get_middle() }) as i32);
input_sync(dev);
}

#[unsafe(link_section = ".text.slate_distinct.alps.alps_process_packet_v7")]
extern "C" fn alps_process_packet_v7(psmouse: *mut psmouse) {
let packet: *mut u8 = unsafe { (*psmouse).packet.as_mut_ptr() as *mut u8 };
if ((unsafe { *packet.wrapping_offset(0 as isize) }) as i32) == 72 && ((unsafe { *packet.wrapping_offset(4 as isize) }) as i32) & 71 == 6 {
alps_process_trackstick_packet_v7(psmouse);
} else {
alps_process_touchpad_packet_v7(psmouse);
}
}

fn alps_get_pkt_id_ss4_v2(mut byte: *mut u8) -> u32 {
let mut pkt_id: u32 = 0;
match ((unsafe { *byte.wrapping_offset(3 as isize) }) as i32) & 48 {
0 => {
if SS4_IS_IDLE_V2!(byte) {
pkt_id = 0;
} else {
pkt_id = 1;
}
}
16 => {
pkt_id = 2; // two-finger finger positions
}
32 => {
pkt_id = 4; // stick pointer
}
48 => {
pkt_id = 3; // third and fourth finger positions
}
_ => {
}
}
pkt_id
}

#[unsafe(link_section = ".text.slate_distinct.alps.alps_decode_ss4_v2")]
extern "C" fn alps_decode_ss4_v2(f: *mut alps_fields, mut p: *mut u8, psmouse: *mut psmouse) -> i32 {
let r#priv: *mut alps_data = (unsafe { (*psmouse).private }) as *mut alps_data;
let pkt_id: u32 = alps_get_pkt_id_ss4_v2(p);
let mut no_data_x: u32 = 0;
let mut no_data_y: u32 = 0;
// Current packet is 1Finger coordinate packet
match pkt_id as u32 {
1 => {
unsafe {
(*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(0 as isize)).x = SS4_1F_X_V2!(p) as i16;
}
unsafe {
(*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(0 as isize)).y = SS4_1F_Y_V2!(p) as i16;
}
unsafe {
(*f).pressure = SS4_1F_Z_V2!(p).wrapping_mul(2) & 127;
}
// When a button is held the device will give us events
// with x, y, and pressure of 0. This causes annoying jumps
// if a touch is released while the button is held.
// Handle this by claiming zero contacts.
unsafe {
(*f).fingers = (if (*f).pressure > 0 { 1i32 } else { 0i32 }) as u32;
}
unsafe {
(*f).__slate_bits_0.__set_first_mp(0u32);
}
unsafe {
(*f).__slate_bits_0.__set_is_mp(0u32);
}
}
2 => {
if (unsafe { (*r#priv).flags }) & 512 != 0 {
if IS_SS4PLUS_DEV!((*r#priv).dev_id) {
unsafe {
(*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(0 as isize)).x = SS4_PLUS_BTL_MF_X_V2!(p, 0i32) as i16;
}
unsafe {
(*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(1 as isize)).x = SS4_PLUS_BTL_MF_X_V2!(p, 1i32) as i16;
}
} else {
unsafe {
(*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(0 as isize)).x = SS4_BTL_MF_X_V2!(p, 0i32) as i16;
}
unsafe {
(*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(1 as isize)).x = SS4_BTL_MF_X_V2!(p, 1i32) as i16;
}
}
unsafe {
(*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(0 as isize)).y = SS4_BTL_MF_Y_V2!(p, 0i32) as i16;
}
unsafe {
(*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(1 as isize)).y = SS4_BTL_MF_Y_V2!(p, 1i32) as i16;
}
} else {
if IS_SS4PLUS_DEV!((*r#priv).dev_id) {
unsafe {
(*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(0 as isize)).x = SS4_PLUS_STD_MF_X_V2!(p, 0i32) as i16;
}
unsafe {
(*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(1 as isize)).x = SS4_PLUS_STD_MF_X_V2!(p, 1i32) as i16;
}
} else {
unsafe {
(*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(0 as isize)).x = SS4_STD_MF_X_V2!(p, 0i32) as i16;
}
unsafe {
(*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(1 as isize)).x = SS4_STD_MF_X_V2!(p, 1i32) as i16;
}
}
unsafe {
(*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(0 as isize)).y = SS4_STD_MF_Y_V2!(p, 0i32) as i16;
}
unsafe {
(*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(1 as isize)).y = SS4_STD_MF_Y_V2!(p, 1i32) as i16;
}
}
unsafe {
(*f).pressure = if SS4_MF_Z_V2!(p, 0i32) != 0 { 48 } else { 0 };
}
if SS4_IS_MF_CONTINUE!(p) {
unsafe {
(*f).__slate_bits_0.__set_first_mp(1u32);
}
} else {
unsafe {
(*f).fingers = 2;
}
unsafe {
(*f).__slate_bits_0.__set_first_mp(0u32);
}
}
unsafe {
(*f).__slate_bits_0.__set_is_mp(0u32);
}
}
3 => {
if (unsafe { (*r#priv).flags }) & 512 != 0 {
if IS_SS4PLUS_DEV!((*r#priv).dev_id) {
unsafe {
(*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(2 as isize)).x = SS4_PLUS_BTL_MF_X_V2!(p, 0i32) as i16;
}
unsafe {
(*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(3 as isize)).x = SS4_PLUS_BTL_MF_X_V2!(p, 1i32) as i16;
}
no_data_x = 4088;
} else {
unsafe {
(*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(2 as isize)).x = SS4_BTL_MF_X_V2!(p, 0i32) as i16;
}
unsafe {
(*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(3 as isize)).x = SS4_BTL_MF_X_V2!(p, 1i32) as i16;
}
no_data_x = 8176;
}
no_data_y = 4088;
unsafe {
(*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(2 as isize)).y = SS4_BTL_MF_Y_V2!(p, 0i32) as i16;
}
unsafe {
(*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(3 as isize)).y = SS4_BTL_MF_Y_V2!(p, 1i32) as i16;
}
} else {
if IS_SS4PLUS_DEV!((*r#priv).dev_id) {
unsafe {
(*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(2 as isize)).x = SS4_PLUS_STD_MF_X_V2!(p, 0i32) as i16;
}
unsafe {
(*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(3 as isize)).x = SS4_PLUS_STD_MF_X_V2!(p, 1i32) as i16;
}
no_data_x = 4080;
} else {
unsafe {
(*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(2 as isize)).x = SS4_STD_MF_X_V2!(p, 0i32) as i16;
}
unsafe {
(*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(3 as isize)).x = SS4_STD_MF_X_V2!(p, 1i32) as i16;
}
no_data_x = 8160;
}
no_data_y = 4080;
unsafe {
(*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(2 as isize)).y = SS4_STD_MF_Y_V2!(p, 0i32) as i16;
}
unsafe {
(*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(3 as isize)).y = SS4_STD_MF_Y_V2!(p, 1i32) as i16;
}
}
unsafe {
(*f).__slate_bits_0.__set_first_mp(0u32);
}
unsafe {
(*f).__slate_bits_0.__set_is_mp(1u32);
}
if SS4_IS_5F_DETECTED!(p) {
unsafe {
(*f).fingers = 5;
}
} else {
if ((unsafe { (*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(3 as isize)).x }) as u32) == no_data_x && ((unsafe { (*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(3 as isize)).y }) as u32) == no_data_y {
unsafe {
(*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(3 as isize)).x = 0;
}
unsafe {
(*((*f).mt.as_mut_ptr() as *mut input_mt_pos).wrapping_offset(3 as isize)).y = 0;
}
unsafe {
(*f).fingers = 3;
}
} else {
unsafe {
(*f).fingers = 4;
}
}
}
}
4 => {
unsafe {
(*f).__slate_bits_0.__set_first_mp(0u32);
}
// x, y, and pressure are decoded in
// alps_process_packet_ss4_v2()
unsafe {
(*f).__slate_bits_0.__set_is_mp(0u32);
}
}
_ => {
unsafe { memset(f as *mut (), 0i32, core::mem::size_of::<alps_fields>() as u64) };
}
}
// handle buttons
if (pkt_id as u32) == 4 {
unsafe {
let __t0 = (SS4_BTN_V2!(p) & 1 != 0) as u32;
(*f).__slate_bits_0.__set_ts_left(__t0);
}
unsafe {
let __t1 = (SS4_BTN_V2!(p) & 2 != 0) as u32;
(*f).__slate_bits_0.__set_ts_right(__t1);
}
unsafe {
let __t2 = (SS4_BTN_V2!(p) & 4 != 0) as u32;
(*f).__slate_bits_0.__set_ts_middle(__t2);
}
} else {
unsafe {
let __t3 = (SS4_BTN_V2!(p) & 1 != 0) as u32;
(*f).__slate_bits_0.__set_left(__t3);
}
if (unsafe { (*r#priv).flags }) & 512 == 0 {
unsafe {
let __t4 = (SS4_BTN_V2!(p) & 2 != 0) as u32;
(*f).__slate_bits_0.__set_right(__t4);
}
unsafe {
let __t5 = (SS4_BTN_V2!(p) & 4 != 0) as u32;
(*f).__slate_bits_0.__set_middle(__t5);
}
}
}
0
}

#[unsafe(link_section = ".text.slate_distinct.alps.alps_process_packet_ss4_v2")]
extern "C" fn alps_process_packet_ss4_v2(psmouse: *mut psmouse) {
let r#priv: *mut alps_data = (unsafe { (*psmouse).private }) as *mut alps_data;
let mut packet: *mut u8 = unsafe { (*psmouse).packet.as_mut_ptr() as *mut u8 };
let dev: *mut input_dev = unsafe { (*psmouse).dev };
let dev2: *mut input_dev = unsafe { (*r#priv).dev2 };
let f: *mut alps_fields = unsafe { core::ptr::addr_of_mut!((*r#priv).f) };
unsafe { memset(f as *mut (), 0i32, core::mem::size_of::<alps_fields>() as u64) };
unsafe { unsafe { (*r#priv).decode_fields }.unwrap()(f, packet, psmouse) };
if (unsafe { (*r#priv).multi_packet }) != 0 {
// Sometimes the first packet will indicate a multi-packet
// sequence, but sometimes the next multi-packet would not
// come. Check for this, and when it happens process the
// position packet as usual.
if ((unsafe { (*f).__slate_bits_0.__get_is_mp() }) as i32) != 0 {
// Now process the 1st packet
unsafe { unsafe { (*r#priv).decode_fields }.unwrap()(f, (*r#priv).multi_data.as_mut_ptr() as *mut u8, psmouse) };
} else {
unsafe {
(*r#priv).multi_packet = 0;
}
}
}
// "f.is_mp" would always be '0' after merging the 1st and 2nd packet.
// When it is set, it means 2nd packet comes without 1st packet come.
if ((unsafe { (*f).__slate_bits_0.__get_is_mp() }) as i32) != 0 {
return;
}
// Save the first packet
if (unsafe { (*r#priv).multi_packet }) == 0 && ((unsafe { (*f).__slate_bits_0.__get_first_mp() }) as i32) != 0 {
unsafe {
(*r#priv).multi_packet = 1;
}
unsafe { memcpy(((*r#priv).multi_data.as_mut_ptr() as *mut u8) as *mut (), packet as *const (), core::mem::size_of::<[u8; 6]>() as u64) };
return;
}
unsafe {
(*r#priv).multi_packet = 0;
}
// Report trackstick
if (alps_get_pkt_id_ss4_v2(packet) as u32) == 4 {
if (unsafe { (*r#priv).flags }) & 2 == 0 {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*(*psmouse).ps2dev.serio).dev) as *const device, c"alps: Rejected trackstick packet from non DualPoint device".as_ptr() as *const u8) };
return;
}
input_report_rel(dev2, 0u32, SS4_TS_X_V2!(packet) as i32);
input_report_rel(dev2, 1u32, SS4_TS_Y_V2!(packet));
input_report_abs(dev2, 24u32, SS4_TS_Z_V2!(packet) as i32);
input_report_key(dev2, 272u32, (unsafe { (*f).__slate_bits_0.__get_ts_left() }) as i32);
input_report_key(dev2, 273u32, (unsafe { (*f).__slate_bits_0.__get_ts_right() }) as i32);
input_report_key(dev2, 274u32, (unsafe { (*f).__slate_bits_0.__get_ts_middle() }) as i32);
input_sync(dev2);
return;
}
// Report touchpad
alps_report_mt_data(psmouse, (if (unsafe { (*f).fingers }) <= 4 { unsafe { (*f).fingers } } else { 4u32 }) as i32);
unsafe { input_mt_report_finger_count(dev, (*f).fingers as i32) };
input_report_key(dev, 272u32, (unsafe { (*f).__slate_bits_0.__get_left() }) as i32);
input_report_key(dev, 273u32, (unsafe { (*f).__slate_bits_0.__get_right() }) as i32);
input_report_key(dev, 274u32, (unsafe { (*f).__slate_bits_0.__get_middle() }) as i32);
input_report_abs(dev, 24u32, unsafe { (*f).pressure });
input_sync(dev);
}

fn alps_is_valid_package_ss4_v2(psmouse: *mut psmouse) -> bool {
if ((unsafe { (*psmouse).pktcnt }) as i32) == 4 && ((unsafe { *((*psmouse).packet.as_mut_ptr() as *mut u8).wrapping_offset(3 as isize) }) as i32) & 8 != 8 {
return false;
}
if ((unsafe { (*psmouse).pktcnt }) as i32) == 6 && ((unsafe { *((*psmouse).packet.as_mut_ptr() as *mut u8).wrapping_offset(5 as isize) }) as i32) & 16 != 0 {
return false;
}
true
}

static mut alps_mutex: mutex = {
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

fn alps_do_register_bare_ps2_mouse(r#priv: *mut alps_data) -> i32 {
let mut error: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(error) as *mut i32;
let mut dev3: core::mem::MaybeUninit<*mut input_dev> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut input_dev = core::ptr::addr_of_mut!(dev3) as *mut *mut input_dev;
let mut psmouse: *mut psmouse = unsafe { core::mem::zeroed() };
unsafe {
psmouse = (*r#priv).psmouse;
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_1 = input_allocate_device();
if (*__slate_slot_1).is_null() {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*(*psmouse).ps2dev.serio).dev) as *const device, c"alps: failed to allocate secondary device\n".as_ptr() as *const u8);
return 12i32.wrapping_neg();
} else {
scnprintf((*r#priv).phys3.as_mut_ptr() as *mut u8, core::mem::size_of::<[u8; 32]>() as u64, c"%s/%s".as_ptr() as *const u8, (*(*psmouse).ps2dev.serio).phys.as_mut_ptr() as *mut u8, if !unsafe { (*r#priv).dev2 }.is_null() { c"input2".as_ptr() as *mut u8 } else { c"input1".as_ptr() as *mut u8 });
(*(*__slate_slot_1)).phys = ((*r#priv).phys3.as_mut_ptr() as *mut u8) as *const u8;
// format of input device name is: "protocol vendor name"
// see function psmouse_switch_protocol() in psmouse-base.c
(*(*__slate_slot_1)).name = c"PS/2 ALPS Mouse".as_ptr() as *const u8;
(*(*__slate_slot_1)).id.bustype = 17;
(*(*__slate_slot_1)).id.vendor = 2;
(*(*__slate_slot_1)).id.product = 1;
(*(*__slate_slot_1)).id.version = 0;
(*(*__slate_slot_1)).dev.parent = core::ptr::addr_of_mut!((*(*psmouse).ps2dev.serio).dev);
input_set_capability(*__slate_slot_1, 2u32, 0u32);
input_set_capability(*__slate_slot_1, 2u32, 1u32);
input_set_capability(*__slate_slot_1, 1u32, 272u32);
input_set_capability(*__slate_slot_1, 1u32, 273u32);
input_set_capability(*__slate_slot_1, 1u32, 274u32);
__set_bit!(0i32, (*(*__slate_slot_1)).propbit.as_mut_ptr() as *mut u64);
*__slate_slot_0 = input_register_device(*__slate_slot_1);
if *__slate_slot_0 != 0 {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*(*psmouse).ps2dev.serio).dev) as *const device, c"alps: failed to register secondary device: %d\n".as_ptr() as *const u8, *__slate_slot_0);
input_free_device(*__slate_slot_1);
return *__slate_slot_0;
} else {
(*r#priv).dev3 = *__slate_slot_1;
return 0;
}
}
}
}

#[unsafe(link_section = ".text.slate_distinct.alps.alps_register_bare_ps2_mouse")]
extern "C" fn alps_register_bare_ps2_mouse(work: *mut work_struct) {
let r#priv: *mut alps_data = container_of__38!(work, alps_data);
let mut error: i32 = 0;
let mut __UNIQUE_ID_guard_473: __SlateRecord_mutex_h_253 = class_mutex_constructor(unsafe { core::ptr::addr_of_mut!(alps_mutex) });
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_473) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || class_mutex_destructor(__slate_cleanup_ptr_1 as *mut _));
let mut __UNIQUE_ID_unlock_474: *mut __SlateRecord_mutex_h_253 = ((unsafe { core::ptr::addr_of_mut!(alps_mutex) }) as u64) as *mut __SlateRecord_mutex_h_253;
let __slate_cleanup_ptr_2 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_474) as *mut _;
let __slate_cleanup_2 = ::slate_support::Guard(&mut || __class_mutex_cleanup_ctx(__slate_cleanup_ptr_2 as *mut _));
if unsafe { (*r#priv).dev3 }.is_null() {
error = alps_do_register_bare_ps2_mouse(r#priv);
if error != 0 {
// Save the error code so that we can detect that we
// already tried to create the device.
unsafe {
(*r#priv).dev3 = ERR_PTR(error as i64) as *mut input_dev;
}
}
}
}

fn alps_report_bare_ps2_packet(psmouse: *mut psmouse, packet: *mut u8, report_buttons: bool) {
let r#priv: *mut alps_data = (unsafe { (*psmouse).private }) as *mut alps_data;
let mut dev: *mut input_dev = unsafe { core::mem::zeroed() };
let mut dev2: *mut input_dev = core::ptr::null_mut::<input_dev>();
// Figure out which device to use to report the bare packet
if ((unsafe { (*r#priv).proto_version }) as i32) == 512 && (unsafe { (*r#priv).flags }) & 2 != 0 {
// On V2 devices the DualPoint Stick reports bare packets
dev = unsafe { (*r#priv).dev2 };
dev2 = unsafe { (*psmouse).dev };
} else {
if unlikely!(IS_ERR_OR_NULL((unsafe { (*r#priv).dev3 }) as *const ())) != 0 {
// Register dev3 mouse if we received PS/2 packet first time
if !IS_ERR((unsafe { (*r#priv).dev3 }) as *const ()) {
schedule_work(unsafe { core::ptr::addr_of_mut!((*r#priv).dev3_register_work) });
}
return;
} else {
dev = unsafe { (*r#priv).dev3 };
}
}
if report_buttons {
alps_report_buttons(dev, dev2, ((unsafe { *packet.wrapping_offset(0 as isize) }) as i32) & 1, ((unsafe { *packet.wrapping_offset(0 as isize) }) as i32) & 2, ((unsafe { *packet.wrapping_offset(0 as isize) }) as i32) & 4);
}
unsafe { psmouse_report_standard_motion(dev, packet) };
input_sync(dev);
}

fn alps_handle_interleaved_ps2(psmouse: *mut psmouse) -> u32 {
let r#priv: *mut alps_data = (unsafe { (*psmouse).private }) as *mut alps_data;
if ((unsafe { (*psmouse).pktcnt }) as i32) < 6 {
return 1;
}
if ((unsafe { (*psmouse).pktcnt }) as i32) == 6 {
// Start a timer to flush the packet if it ends up last
// 6-byte packet in the stream. Timer needs to fire
// psmouse core times out itself. 20 ms should be enough
// to decide if we are getting more data or not.
unsafe { mod_timer(core::ptr::addr_of_mut!((*r#priv).timer), core::ptr::read_volatile(core::ptr::addr_of!(jiffies)).wrapping_add(msecs_to_jiffies(20u32))) };
return 1;
}
unsafe { timer_delete(core::ptr::addr_of_mut!((*r#priv).timer)) };
if ((unsafe { *((*psmouse).packet.as_mut_ptr() as *mut u8).wrapping_offset(6 as isize) }) as i32) & 128 != 0 {
// Highest bit is set - that means we either had
// complete ALPS packet and this is start of the
// next packet or we got garbage.
let __v0: bool = (((unsafe { *((*psmouse).packet.as_mut_ptr() as *mut u8).wrapping_offset(3 as isize) }) as i32) | ((unsafe { *((*psmouse).packet.as_mut_ptr() as *mut u8).wrapping_offset(4 as isize) }) as i32) | ((unsafe { *((*psmouse).packet.as_mut_ptr() as *mut u8).wrapping_offset(5 as isize) }) as i32)) & 128 != 0 || !alps_is_valid_first_byte(r#priv, unsafe { *((*psmouse).packet.as_mut_ptr() as *mut u8).wrapping_offset(6 as isize) });
if __v0 {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*(*psmouse).ps2dev.serio).dev) as *const device, c"alps: refusing packet %4ph (suspected interleaved ps/2)\n".as_ptr() as *const u8, ((*psmouse).packet.as_mut_ptr() as *mut u8).wrapping_offset(3 as isize)) };
}
return 0;
}
unsafe { unsafe { (*r#priv).process_packet }.unwrap()(psmouse) };
// Continue with the next packet
unsafe {
*((*psmouse).packet.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) = *((*psmouse).packet.as_mut_ptr() as *mut u8).wrapping_offset(6 as isize);
}
unsafe {
(*psmouse).pktcnt = 1;
}
} else {
// High bit is 0 - that means that we indeed got a PS/2
// packet in the middle of ALPS packet.
//
// There is also possibility that we got 6-byte ALPS
// packet followed  by 3-byte packet from trackpoint. We
// can not distinguish between these 2 scenarios but
// because the latter is unlikely to happen in course of
// normal operation (user would need to press all
// buttons on the pad and start moving trackpoint
// without touching the pad surface) we assume former.
// Even if we are wrong the wost thing that would happen
// the cursor would jump but we should not get protocol
// de-synchronization.
alps_report_bare_ps2_packet(psmouse, unsafe { (*psmouse).packet.as_mut_ptr() as *mut u8 }.wrapping_offset(3 as isize), false);
// Continue with the standard ALPS protocol handling,
// but make sure we won't process it as an interleaved
// packet again, which may happen if all buttons are
// pressed. To avoid this let's reset the 4th bit which
// is normally 1.
unsafe {
*((*psmouse).packet.as_mut_ptr() as *mut u8).wrapping_offset(3 as isize) = ((*((*psmouse).packet.as_mut_ptr() as *mut u8).wrapping_offset(6 as isize) as i32) & 247) as u8;
}
unsafe {
(*psmouse).pktcnt = 4;
}
}
1
}

#[unsafe(link_section = ".text.slate_distinct.alps.alps_flush_packet")]
extern "C" fn alps_flush_packet(t: *mut timer_list) {
let r#priv: *mut alps_data = timer_container_of__6!(t);
let psmouse: *mut psmouse = unsafe { (*r#priv).psmouse };
let mut __UNIQUE_ID_guard_475: *mut serio = class_serio_pause_rx_constructor(unsafe { (*psmouse).ps2dev.serio });
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_475) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || class_serio_pause_rx_destructor(__slate_cleanup_ptr_1 as *mut _));
if ((unsafe { (*psmouse).pktcnt }) as i32) == ((unsafe { (*psmouse).pktsize }) as i32) {
// We did not any more data in reasonable amount of time.
// Validate the last 3 bytes and process as a standard
// ALPS packet.
if (((unsafe { *((*psmouse).packet.as_mut_ptr() as *mut u8).wrapping_offset(3 as isize) }) as i32) | ((unsafe { *((*psmouse).packet.as_mut_ptr() as *mut u8).wrapping_offset(4 as isize) }) as i32) | ((unsafe { *((*psmouse).packet.as_mut_ptr() as *mut u8).wrapping_offset(5 as isize) }) as i32)) & 128 != 0 {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*(*psmouse).ps2dev.serio).dev) as *const device, c"alps: refusing packet %3ph (suspected interleaved ps/2)\n".as_ptr() as *const u8, ((*psmouse).packet.as_mut_ptr() as *mut u8).wrapping_offset(3 as isize)) };
}
} else {
unsafe { unsafe { (*r#priv).process_packet }.unwrap()(psmouse) };
}
unsafe {
(*psmouse).pktcnt = 0;
}
}
}

#[unsafe(link_section = ".text.slate_distinct.alps.alps_process_byte")]
extern "C" fn alps_process_byte(psmouse: *mut psmouse) -> u32 {
let r#priv: *mut alps_data = (unsafe { (*psmouse).private }) as *mut alps_data;
// Check if we are dealing with a bare PS/2 packet, presumably from
// a device connected to the external PS/2 port. Because bare PS/2
// protocol does not have enough constant bits to self-synchronize
// properly we only do this if the device is fully synchronized.
// Can not distinguish V8's first byte from PS/2 packet's
if ((unsafe { (*r#priv).proto_version }) as i32) != 2048 && (unsafe { (*psmouse).out_of_sync_cnt }) == 0 && ((unsafe { *((*psmouse).packet.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) }) as i32) & 200 == 8 {
if ((unsafe { (*psmouse).pktcnt }) as i32) == 3 {
alps_report_bare_ps2_packet(psmouse, unsafe { (*psmouse).packet.as_mut_ptr() as *mut u8 }, true);
return 2;
}
return 1;
}
// Check for PS/2 packet stuffed in the middle of ALPS packet.
if (unsafe { (*r#priv).flags }) & 128 != 0 && ((unsafe { (*psmouse).pktcnt }) as i32) >= 4 && ((unsafe { *((*psmouse).packet.as_mut_ptr() as *mut u8).wrapping_offset(3 as isize) }) as i32) & 15 == 15 {
return alps_handle_interleaved_ps2(psmouse);
}
if !alps_is_valid_first_byte(r#priv, unsafe { *((*psmouse).packet.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) }) {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*(*psmouse).ps2dev.serio).dev) as *const device, c"alps: refusing packet[0] = %x (mask0 = %x, byte0 = %x)\n".as_ptr() as *const u8, *((*psmouse).packet.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) as i32, (*r#priv).mask0 as i32, (*r#priv).byte0 as i32) };
}
return 0;
}
// Bytes 2 - pktsize should have 0 in the highest bit
if ((unsafe { (*r#priv).proto_version }) as i32) < 1280 && ((unsafe { (*psmouse).pktcnt }) as i32) >= 2 && ((unsafe { (*psmouse).pktcnt }) as i32) <= ((unsafe { (*psmouse).pktsize }) as i32) && ((unsafe { *((*psmouse).packet.as_mut_ptr() as *mut u8).wrapping_offset(((*psmouse).pktcnt as i32).wrapping_sub(1) as isize) }) as i32) & 128 != 0 {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*(*psmouse).ps2dev.serio).dev) as *const device, c"alps: refusing packet[%i] = %x\n".as_ptr() as *const u8, ((*psmouse).pktcnt as i32).wrapping_sub(1), *((*psmouse).packet.as_mut_ptr() as *mut u8).wrapping_offset(((*psmouse).pktcnt as i32).wrapping_sub(1) as isize) as i32) };
}
if ((unsafe { (*r#priv).proto_version }) as i32) == 784 && ((unsafe { (*psmouse).pktcnt }) as i32) == ((unsafe { (*psmouse).pktsize }) as i32) {
// Some Dell boxes, such as Latitude E6440 or E7440
// with closed lid, quite often smash last byte of
// otherwise valid packet with 0xff. Given that the
// next packet is very likely to be valid let's
// report PSMOUSE_FULL_PACKET but not process data,
// rather than reporting PSMOUSE_BAD_DATA and
// filling the logs.
return 2;
}
return 0;
}
let __v0: bool = ((unsafe { (*r#priv).proto_version }) as i32) == 1792 && !alps_is_valid_package_v7(psmouse);
let __v1: bool = __v0 || ((unsafe { (*r#priv).proto_version }) as i32) == 2048 && !alps_is_valid_package_ss4_v2(psmouse);
if __v1 {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*(*psmouse).ps2dev.serio).dev) as *const device, c"alps: refusing packet[%i] = %x\n".as_ptr() as *const u8, ((*psmouse).pktcnt as i32).wrapping_sub(1), *((*psmouse).packet.as_mut_ptr() as *mut u8).wrapping_offset(((*psmouse).pktcnt as i32).wrapping_sub(1) as isize) as i32) };
}
return 0;
}
if ((unsafe { (*psmouse).pktcnt }) as i32) == ((unsafe { (*psmouse).pktsize }) as i32) {
unsafe { unsafe { (*r#priv).process_packet }.unwrap()(psmouse) };
return 2;
}
1
}

fn alps_command_mode_send_nibble(psmouse: *mut psmouse, nibble: i32) -> i32 {
let ps2dev: *mut ps2dev = unsafe { core::ptr::addr_of_mut!((*psmouse).ps2dev) };
let r#priv: *mut alps_data = (unsafe { (*psmouse).private }) as *mut alps_data;
let mut command: i32 = 0;
let mut dummy: [u8; 4] = [0u8; 4];
loop {
if unlikely!(nibble > 15) != 0 {
loop {
unsafe {
core::arch::asm!("476: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 476b - ., 3; .popsection\n# {0}", const 476 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_alps_command_mode_send_nibble_0, sym __slate_asm_string_alps_command_mode_send_nibble_1, const 1698 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
command = unsafe { (*unsafe { (*r#priv).nibble_commands }.wrapping_offset(nibble as isize)).command };
let param: *mut u8 = if command & 3840 != 0 { dummy.as_mut_ptr() as *mut u8 } else { (unsafe { core::ptr::addr_of!((*unsafe { (*r#priv).nibble_commands }.wrapping_offset(nibble as isize)).data) }) as *mut u8 };
if (unsafe { ps2_command(ps2dev, param, command as u32) }) != 0 {
return 1i32.wrapping_neg();
}
0
}

fn alps_command_mode_set_addr(psmouse: *mut psmouse, addr: i32) -> i32 {
let ps2dev: *mut ps2dev = unsafe { core::ptr::addr_of_mut!((*psmouse).ps2dev) };
let r#priv: *mut alps_data = (unsafe { (*psmouse).private }) as *mut alps_data;
let mut i: i32 = 12;
let mut nibble: i32 = 0;
if (unsafe { ps2_command(ps2dev, core::ptr::null_mut::<u8>(), (*r#priv).addr_command as u32) }) != 0 {
return 1i32.wrapping_neg();
}
while i >= 0 {
nibble = addr >> i & 15;
if alps_command_mode_send_nibble(psmouse, nibble) != 0 {
return 1i32.wrapping_neg();
}
i = i.wrapping_sub(4);
}
0
}

fn __alps_command_mode_read_reg(psmouse: *mut psmouse, addr: i32) -> i32 {
let ps2dev: *mut ps2dev = unsafe { core::ptr::addr_of_mut!((*psmouse).ps2dev) };
let mut param: [u8; 4] = [0u8; 4];
if (unsafe { ps2_command(ps2dev, param.as_mut_ptr() as *mut u8, 1001u32) }) != 0 {
return 1i32.wrapping_neg();
}
// The address being read is returned in the first two bytes
// of the result. Check that this address matches the expected
// address.
if addr != (param[0] as i32) << 8i32 | (param[1] as i32) {
return 1i32.wrapping_neg();
}
param[2] as i32
}

fn alps_command_mode_read_reg(psmouse: *mut psmouse, addr: i32) -> i32 {
if alps_command_mode_set_addr(psmouse, addr) != 0 {
return 1i32.wrapping_neg();
}
__alps_command_mode_read_reg(psmouse, addr)
}

fn __alps_command_mode_write_reg(psmouse: *mut psmouse, value: u8) -> i32 {
if alps_command_mode_send_nibble(psmouse, (value as i32) >> 4i32 & 15) != 0 {
return 1i32.wrapping_neg();
}
if alps_command_mode_send_nibble(psmouse, (value as i32) & 15) != 0 {
return 1i32.wrapping_neg();
}
0
}

fn alps_command_mode_write_reg(psmouse: *mut psmouse, addr: i32, value: u8) -> i32 {
if alps_command_mode_set_addr(psmouse, addr) != 0 {
return 1i32.wrapping_neg();
}
__alps_command_mode_write_reg(psmouse, value)
}

fn alps_rpt_cmd(psmouse: *mut psmouse, init_command: i32, repeated_command: i32, param: *mut u8) -> i32 {
let ps2dev: *mut ps2dev = unsafe { core::ptr::addr_of_mut!((*psmouse).ps2dev) };
unsafe {
*param.wrapping_offset(0 as isize) = 0;
}
let __v0: bool = init_command != 0 && (unsafe { ps2_command(ps2dev, param, init_command as u32) }) != 0;
if __v0 {
return 5i32.wrapping_neg();
}
let __v1: bool = (unsafe { ps2_command(ps2dev, core::ptr::null_mut::<u8>(), repeated_command as u32) }) != 0 || (unsafe { ps2_command(ps2dev, core::ptr::null_mut::<u8>(), repeated_command as u32) }) != 0;
let __v2: bool = __v1 || (unsafe { ps2_command(ps2dev, core::ptr::null_mut::<u8>(), repeated_command as u32) }) != 0;
if __v2 {
return 5i32.wrapping_neg();
}
unsafe {
*param.wrapping_offset(2 as isize) = 255;
}
unsafe {
*param.wrapping_offset(1 as isize) = 255;
}
unsafe {
*param.wrapping_offset(0 as isize) = 255;
}
if (unsafe { ps2_command(ps2dev, param, 1001u32) }) != 0 {
return 5i32.wrapping_neg();
}
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*(*psmouse).ps2dev.serio).dev) as *const device, c"alps: %2.2X report: %3ph\n".as_ptr() as *const u8, repeated_command, param) };
}
0
}

fn alps_check_valid_firmware_id(id: *mut u8) -> bool {
if ((unsafe { *id.wrapping_offset(0 as isize) }) as i32) == 115 {
return true;
}
if ((unsafe { *id.wrapping_offset(0 as isize) }) as i32) == 136 && (((unsafe { *id.wrapping_offset(1 as isize) }) as i32) == 7 || ((unsafe { *id.wrapping_offset(1 as isize) }) as i32) == 8 || ((unsafe { *id.wrapping_offset(1 as isize) }) as i32) & 240 == 176 || ((unsafe { *id.wrapping_offset(1 as isize) }) as i32) & 240 == 192) {
return true;
}
false
}

fn alps_enter_command_mode(psmouse: *mut psmouse) -> i32 {
let mut param: [u8; 4] = [0u8; 4];
if alps_rpt_cmd(psmouse, 0i32, 236i32, param.as_mut_ptr() as *mut u8) != 0 {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*(*psmouse).ps2dev.serio).dev) as *const device, c"alps: failed to enter command mode\n".as_ptr() as *const u8) };
return 1i32.wrapping_neg();
}
if !alps_check_valid_firmware_id(param.as_mut_ptr() as *mut u8) {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*(*psmouse).ps2dev.serio).dev) as *const device, c"alps: unknown response while entering command mode\n".as_ptr() as *const u8) };
}
return 1i32.wrapping_neg();
}
0
}

fn alps_exit_command_mode(psmouse: *mut psmouse) -> i32 {
let ps2dev: *mut ps2dev = unsafe { core::ptr::addr_of_mut!((*psmouse).ps2dev) };
if (unsafe { ps2_command(ps2dev, core::ptr::null_mut::<u8>(), 234u32) }) != 0 {
return 1i32.wrapping_neg();
}
0
}

/// For DualPoint devices select the device that should respond to
/// subsequent commands. It looks like glidepad is behind stickpointer,
/// I'd thought it would be other way around...
fn alps_passthrough_mode_v2(psmouse: *mut psmouse, enable: bool) -> i32 {
let ps2dev: *mut ps2dev = unsafe { core::ptr::addr_of_mut!((*psmouse).ps2dev) };
let cmd: i32 = if enable { 231 } else { 230 };
let __v0: bool = (unsafe { ps2_command(ps2dev, core::ptr::null_mut::<u8>(), cmd as u32) }) != 0 || (unsafe { ps2_command(ps2dev, core::ptr::null_mut::<u8>(), cmd as u32) }) != 0;
let __v1: bool = __v0 || (unsafe { ps2_command(ps2dev, core::ptr::null_mut::<u8>(), cmd as u32) }) != 0;
let __v2: bool = __v1 || (unsafe { ps2_command(ps2dev, core::ptr::null_mut::<u8>(), 245u32) }) != 0;
if __v2 {
return 1i32.wrapping_neg();
}
// we may get 3 more bytes, just ignore them
unsafe { ps2_drain(ps2dev, 3u64, 100u32) };
0
}

fn alps_absolute_mode_v1_v2(psmouse: *mut psmouse) -> i32 {
let ps2dev: *mut ps2dev = unsafe { core::ptr::addr_of_mut!((*psmouse).ps2dev) };
// Try ALPS magic knock - 4 disable before enable
let __v0: bool = (unsafe { ps2_command(ps2dev, core::ptr::null_mut::<u8>(), 245u32) }) != 0 || (unsafe { ps2_command(ps2dev, core::ptr::null_mut::<u8>(), 245u32) }) != 0;
let __v1: bool = __v0 || (unsafe { ps2_command(ps2dev, core::ptr::null_mut::<u8>(), 245u32) }) != 0;
let __v2: bool = __v1 || (unsafe { ps2_command(ps2dev, core::ptr::null_mut::<u8>(), 245u32) }) != 0;
let __v3: bool = __v2 || (unsafe { ps2_command(ps2dev, core::ptr::null_mut::<u8>(), 244u32) }) != 0;
if __v3 {
return 1i32.wrapping_neg();
}
// Switch mouse to poll (remote) mode so motion data will not
// get in our way
unsafe { ps2_command(ps2dev, core::ptr::null_mut::<u8>(), 240u32) }
}

fn alps_monitor_mode_send_word(psmouse: *mut psmouse, word: u16) -> i32 {
// b0-b11 are valid bits, send sequence is inverse.
// e.g. when word = 0x0123, nibble send sequence is 3, 2, 1
let mut i: i32 = 0;
let mut nibble: i32 = 0;
while i <= 8 {
nibble = (word as i32) >> i & 15;
if alps_command_mode_send_nibble(psmouse, nibble) != 0 {
return 1i32.wrapping_neg();
}
i = i.wrapping_add(4);
}
0
}

fn alps_monitor_mode_write_reg(psmouse: *mut psmouse, addr: u16, value: u16) -> i32 {
let ps2dev: *mut ps2dev = unsafe { core::ptr::addr_of_mut!((*psmouse).ps2dev) };
// 0x0A0 is the command to write the word
let __v0: bool = (unsafe { ps2_command(ps2dev, core::ptr::null_mut::<u8>(), 244u32) }) != 0 || alps_monitor_mode_send_word(psmouse, 160u16) != 0;
let __v1: bool = __v0 || alps_monitor_mode_send_word(psmouse, addr) != 0;
let __v2: bool = __v1 || alps_monitor_mode_send_word(psmouse, value) != 0;
let __v3: bool = __v2 || (unsafe { ps2_command(ps2dev, core::ptr::null_mut::<u8>(), 245u32) }) != 0;
if __v3 {
return 1i32.wrapping_neg();
}
0
}

fn alps_monitor_mode(psmouse: *mut psmouse, enable: bool) -> i32 {
let ps2dev: *mut ps2dev = unsafe { core::ptr::addr_of_mut!((*psmouse).ps2dev) };
if enable {
// EC E9 F5 F5 E7 E6 E7 E9 to enter monitor mode
let __v0: bool = (unsafe { ps2_command(ps2dev, core::ptr::null_mut::<u8>(), 236u32) }) != 0 || (unsafe { ps2_command(ps2dev, core::ptr::null_mut::<u8>(), 1001u32) }) != 0;
let __v1: bool = __v0 || (unsafe { ps2_command(ps2dev, core::ptr::null_mut::<u8>(), 245u32) }) != 0;
let __v2: bool = __v1 || (unsafe { ps2_command(ps2dev, core::ptr::null_mut::<u8>(), 245u32) }) != 0;
let __v3: bool = __v2 || (unsafe { ps2_command(ps2dev, core::ptr::null_mut::<u8>(), 231u32) }) != 0;
let __v4: bool = __v3 || (unsafe { ps2_command(ps2dev, core::ptr::null_mut::<u8>(), 230u32) }) != 0;
let __v5: bool = __v4 || (unsafe { ps2_command(ps2dev, core::ptr::null_mut::<u8>(), 231u32) }) != 0;
let __v6: bool = __v5 || (unsafe { ps2_command(ps2dev, core::ptr::null_mut::<u8>(), 1001u32) }) != 0;
if __v6 {
return 1i32.wrapping_neg();
}
} else {
// EC to exit monitor mode
if (unsafe { ps2_command(ps2dev, core::ptr::null_mut::<u8>(), 236u32) }) != 0 {
return 1i32.wrapping_neg();
}
}
0
}

fn alps_absolute_mode_v6(psmouse: *mut psmouse) -> i32 {
let reg_val: u16 = 385;
let mut ret: i32 = 0;
// enter monitor mode, to write the register
if alps_monitor_mode(psmouse, true) != 0 {
return 1i32.wrapping_neg();
}
ret = alps_monitor_mode_write_reg(psmouse, 0u16, reg_val);
if alps_monitor_mode(psmouse, false) != 0 {
ret = 1i32.wrapping_neg();
}
ret
}

fn alps_get_status(psmouse: *mut psmouse, param: *mut u8) -> i32 {
// Get status: 0xF5 0xF5 0xF5 0xE9
if alps_rpt_cmd(psmouse, 0i32, 245i32, param as *mut u8) != 0 {
return 1i32.wrapping_neg();
}
0
}

/// Turn touchpad tapping on or off. The sequences are:
/// 0xE9 0xF5 0xF5 0xF3 0x0A to enable,
/// 0xE9 0xF5 0xF5 0xE8 0x00 to disable.
/// My guess that 0xE9 (GetInfo) is here as a sync point.
/// For models that also have stickpointer (DualPoints) its tapping
/// is controlled separately (0xE6 0xE6 0xE6 0xF3 0x14|0x0A) but
/// we don't fiddle with it.
fn alps_tap_mode(psmouse: *mut psmouse, enable: i32) -> i32 {
let ps2dev: *mut ps2dev = unsafe { core::ptr::addr_of_mut!((*psmouse).ps2dev) };
let cmd: i32 = if enable != 0 { 4339 } else { 4328 };
let mut tap_arg: u8 = (if enable != 0 { 10i32 } else { 0i32 }) as u8;
let mut param: [u8; 4] = [0u8; 4];
let __v0: bool = (unsafe { ps2_command(ps2dev, param.as_mut_ptr() as *mut u8, 1001u32) }) != 0 || (unsafe { ps2_command(ps2dev, core::ptr::null_mut::<u8>(), 245u32) }) != 0;
let __v1: bool = __v0 || (unsafe { ps2_command(ps2dev, core::ptr::null_mut::<u8>(), 245u32) }) != 0;
let __v2: bool = __v1 || (unsafe { ps2_command(ps2dev, core::ptr::addr_of_mut!(tap_arg), cmd as u32) }) != 0;
if __v2 {
return 1i32.wrapping_neg();
}
if alps_get_status(psmouse, param.as_mut_ptr() as *mut u8) != 0 {
return 1i32.wrapping_neg();
}
0
}

/// alps_poll() - poll the touchpad for current motion packet.
/// Used in resync.
#[unsafe(link_section = ".text.slate_distinct.alps.alps_poll")]
extern "C" fn alps_poll(psmouse: *mut psmouse) -> i32 {
let r#priv: *mut alps_data = (unsafe { (*psmouse).private }) as *mut alps_data;
let mut buf: [u8; 8] = [0u8; 8];
let mut poll_failed: bool = false;
if (unsafe { (*r#priv).flags }) & 4 != 0 {
alps_passthrough_mode_v2(psmouse, true);
}
poll_failed = (unsafe { ps2_command(core::ptr::addr_of_mut!((*psmouse).ps2dev), buf.as_mut_ptr() as *mut u8, (235 | ((*psmouse).pktsize as i32) << 8i32) as u32) }) < 0;
if (unsafe { (*r#priv).flags }) & 4 != 0 {
alps_passthrough_mode_v2(psmouse, false);
}
if poll_failed || (buf[0] as i32) & ((unsafe { (*r#priv).mask0 }) as i32) != ((unsafe { (*r#priv).byte0 }) as i32) {
return 1i32.wrapping_neg();
}
if ((unsafe { (*psmouse).badbyte }) as i32) & 200 == 8 {
// Poll the track stick ...
if (unsafe { ps2_command(core::ptr::addr_of_mut!((*psmouse).ps2dev), buf.as_mut_ptr() as *mut u8, 1003u32) }) != 0 {
return 1i32.wrapping_neg();
}
}
unsafe { memcpy(((*psmouse).packet.as_mut_ptr() as *mut u8) as *mut (), buf.as_mut_ptr() as *const (), core::mem::size_of::<[u8; 8]>() as u64) };
0
}

#[unsafe(link_section = ".text.slate_distinct.alps.alps_hw_init_v1_v2")]
extern "C" fn alps_hw_init_v1_v2(psmouse: *mut psmouse) -> i32 {
let r#priv: *mut alps_data = (unsafe { (*psmouse).private }) as *mut alps_data;
let __v0: bool = (unsafe { (*r#priv).flags }) & 4 != 0 && alps_passthrough_mode_v2(psmouse, true) != 0;
if __v0 {
return 1i32.wrapping_neg();
}
if alps_tap_mode(psmouse, 1i32) != 0 {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*(*psmouse).ps2dev.serio).dev) as *const device, c"alps: Failed to enable hardware tapping\n".as_ptr() as *const u8) };
return 1i32.wrapping_neg();
}
if alps_absolute_mode_v1_v2(psmouse) != 0 {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*(*psmouse).ps2dev.serio).dev) as *const device, c"alps: Failed to enable absolute mode\n".as_ptr() as *const u8) };
return 1i32.wrapping_neg();
}
let __v3: bool = (unsafe { (*r#priv).flags }) & 4 != 0 && alps_passthrough_mode_v2(psmouse, false) != 0;
if __v3 {
return 1i32.wrapping_neg();
}
// ALPS needs stream mode, otherwise it won't report any data
if (unsafe { ps2_command(core::ptr::addr_of_mut!((*psmouse).ps2dev), core::ptr::null_mut::<u8>(), 234u32) }) != 0 {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*(*psmouse).ps2dev.serio).dev) as *const device, c"alps: Failed to enable stream mode\n".as_ptr() as *const u8) };
return 1i32.wrapping_neg();
}
0
}

/// Must be in passthrough mode when calling this function
fn alps_trackstick_enter_extended_mode_v3_v6(psmouse: *mut psmouse) -> i32 {
let mut param: [u8; 2] = [200, 20];
let __v0: bool = (unsafe { ps2_command(core::ptr::addr_of_mut!((*psmouse).ps2dev), core::ptr::null_mut::<u8>(), 230u32) }) != 0 || (unsafe { ps2_command(core::ptr::addr_of_mut!((*psmouse).ps2dev), core::ptr::null_mut::<u8>(), 230u32) }) != 0;
let __v1: bool = __v0 || (unsafe { ps2_command(core::ptr::addr_of_mut!((*psmouse).ps2dev), core::ptr::null_mut::<u8>(), 230u32) }) != 0;
let __v2: bool = __v1 || (unsafe { ps2_command(core::ptr::addr_of_mut!((*psmouse).ps2dev), (param.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize), 4339u32) }) != 0;
let __v3: bool = __v2 || (unsafe { ps2_command(core::ptr::addr_of_mut!((*psmouse).ps2dev), (param.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize), 4339u32) }) != 0;
if __v3 {
return 1i32.wrapping_neg();
}
0
}

#[unsafe(link_section = ".text.slate_distinct.alps.alps_hw_init_v6")]
extern "C" fn alps_hw_init_v6(psmouse: *mut psmouse) -> i32 {
let mut ret: i32 = 0;
// Enter passthrough mode to let trackpoint enter 6byte raw mode
if alps_passthrough_mode_v2(psmouse, true) != 0 {
return 1i32.wrapping_neg();
}
ret = alps_trackstick_enter_extended_mode_v3_v6(psmouse);
if alps_passthrough_mode_v2(psmouse, false) != 0 {
return 1i32.wrapping_neg();
}
if ret != 0 {
return ret;
}
if alps_absolute_mode_v6(psmouse) != 0 {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*(*psmouse).ps2dev.serio).dev) as *const device, c"alps: Failed to enable absolute mode\n".as_ptr() as *const u8) };
return 1i32.wrapping_neg();
}
0
}

/// Enable or disable passthrough mode to the trackstick.
fn alps_passthrough_mode_v3(psmouse: *mut psmouse, reg_base: i32, enable: bool) -> i32 {
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut reg_val: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i32 = core::ptr::addr_of_mut!(reg_val) as *mut i32;
unsafe {
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
ret = 1i32.wrapping_neg();
if alps_enter_command_mode(psmouse) != 0 {
return 1i32.wrapping_neg();
} else {
*__slate_slot_5 = alps_command_mode_read_reg(psmouse, reg_base.wrapping_add(8));
if *__slate_slot_5 != 1i32.wrapping_neg() {
if enable {
*__slate_slot_5 |= 1;
} else {
let __v2: i32 = *__slate_slot_5 & !1;
*__slate_slot_5 = __v2;
}
ret = __alps_command_mode_write_reg(psmouse, *__slate_slot_5 as u8);
}
if alps_exit_command_mode(psmouse) != 0 {
ret = 1i32.wrapping_neg();
}
return ret;
}
}
}

/// Must be in command mode when calling this function
fn alps_absolute_mode_v3(psmouse: *mut psmouse) -> i32 {
let mut reg_val: i32 = alps_command_mode_read_reg(psmouse, 4i32);
if reg_val == 1i32.wrapping_neg() {
return 1i32.wrapping_neg();
}
reg_val |= 6;
if __alps_command_mode_write_reg(psmouse, reg_val as u8) != 0 {
return 1i32.wrapping_neg();
}
0
}

fn alps_probe_trackstick_v3_v7(psmouse: *mut psmouse, reg_base: i32) -> i32 {
let mut reg_val: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(reg_val) as *mut i32;
let mut ret: i32 = unsafe { core::mem::zeroed() };
unsafe {
ret = 5i32.wrapping_neg();
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if alps_enter_command_mode(psmouse) == 0 {
*__slate_slot_0 = alps_command_mode_read_reg(psmouse, reg_base.wrapping_add(8));
if *__slate_slot_0 != 1i32.wrapping_neg() {
// bit 7: trackstick is present
ret = if *__slate_slot_0 & 128 != 0 { 0 } else { 19i32.wrapping_neg() };
}
}
alps_exit_command_mode(psmouse);
return ret;
}
}

fn alps_setup_trackstick_v3(psmouse: *mut psmouse, reg_base: i32) -> i32 {
let mut ret: i32 = 0;
let mut reg_val: i32 = 0;
let mut param: [u8; 4] = [0u8; 4];
// We need to configure trackstick to report data for touchpad in
// extended format. And also we need to tell touchpad to expect data
// from trackstick in extended format. Without this configuration
// trackstick packets sent from touchpad are in basic format which is
// different from what we expect.
if alps_passthrough_mode_v3(psmouse, reg_base, true) != 0 {
return 5i32.wrapping_neg();
}
// E7 report for the trackstick
//
// There have been reports of failures to seem to trace back
// to the above trackstick check failing. When these occur
// this E7 report fails, so when that happens we continue
// with the assumption that there isn't a trackstick after
// all.
if alps_rpt_cmd(psmouse, 0i32, 231i32, param.as_mut_ptr() as *mut u8) != 0 {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*(*psmouse).ps2dev.serio).dev) as *const device, c"alps: Failed to initialize trackstick (E7 report failed)\n".as_ptr() as *const u8) };
ret = 19i32.wrapping_neg();
} else {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*(*psmouse).ps2dev.serio).dev) as *const device, c"alps: trackstick E7 report: %3ph\n".as_ptr() as *const u8, param.as_mut_ptr() as *mut u8) };
}
if alps_trackstick_enter_extended_mode_v3_v6(psmouse) != 0 {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*(*psmouse).ps2dev.serio).dev) as *const device, c"alps: Failed to enter into trackstick extended mode\n".as_ptr() as *const u8) };
ret = 5i32.wrapping_neg();
}
}
if alps_passthrough_mode_v3(psmouse, reg_base, false) != 0 {
return 5i32.wrapping_neg();
}
if ret != 0 {
return ret;
}
if alps_enter_command_mode(psmouse) != 0 {
return 5i32.wrapping_neg();
}
reg_val = alps_command_mode_read_reg(psmouse, reg_base.wrapping_add(8));
if reg_val == 1i32.wrapping_neg() {
ret = 5i32.wrapping_neg();
} else {
// Tell touchpad that trackstick is now in extended mode.
// If bit 1 isn't set the packet format is different.
let __v2: i32 = reg_val;
reg_val = ((__v2 as u64) | BIT!(1i32)) as i32;
if __alps_command_mode_write_reg(psmouse, reg_val as u8) != 0 {
ret = 5i32.wrapping_neg();
}
}
if alps_exit_command_mode(psmouse) != 0 {
return 5i32.wrapping_neg();
}
ret
}

#[unsafe(link_section = ".text.slate_distinct.alps.alps_hw_init_v3")]
extern "C" fn alps_hw_init_v3(psmouse: *mut psmouse) -> i32 {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut __v2: bool = unsafe { core::mem::zeroed() };
let mut param: core::mem::MaybeUninit<[u8; 4]> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut [u8; 4] = core::ptr::addr_of_mut!(param) as *mut [u8; 4];
let mut reg_val: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(reg_val) as *mut i32;
let mut ps2dev: *mut ps2dev = unsafe { core::mem::zeroed() };
unsafe {
let r#priv: *mut alps_data = (*psmouse).private as *mut alps_data;
ps2dev = core::ptr::addr_of_mut!((*psmouse).ps2dev);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if (*r#priv).flags & 2 != 0 {
__v2 = alps_setup_trackstick_v3(psmouse, 0i32) == 5i32.wrapping_neg();
} else {
__v2 = false;
}
if !__v2 {
if alps_enter_command_mode(psmouse) != 0 {
__v1 = true;
} else {
__v1 = alps_absolute_mode_v3(psmouse) != 0;
}
if __v1 {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*(*psmouse).ps2dev.serio).dev) as *const device, c"alps: Failed to enter absolute mode\n".as_ptr() as *const u8);
} else {
*__slate_slot_4 = alps_command_mode_read_reg(psmouse, 6i32);
if *__slate_slot_4 != 1i32.wrapping_neg() {
if __alps_command_mode_write_reg(psmouse, (*__slate_slot_4 | 1) as u8) == 0 {
*__slate_slot_4 = alps_command_mode_read_reg(psmouse, 7i32);
if *__slate_slot_4 != 1i32.wrapping_neg() {
if __alps_command_mode_write_reg(psmouse, (*__slate_slot_4 | 1) as u8) == 0 {
if alps_command_mode_read_reg(psmouse, 324i32) != 1i32.wrapping_neg() {
if __alps_command_mode_write_reg(psmouse, 4u8) == 0 {
if alps_command_mode_read_reg(psmouse, 345i32) != 1i32.wrapping_neg() {
if __alps_command_mode_write_reg(psmouse, 3u8) == 0 {
if alps_command_mode_read_reg(psmouse, 355i32) != 1i32.wrapping_neg() {
if alps_command_mode_write_reg(psmouse, 355i32, 3u8) == 0 {
if alps_command_mode_read_reg(psmouse, 354i32) != 1i32.wrapping_neg() {
if alps_command_mode_write_reg(psmouse, 354i32, 4u8) == 0 {
alps_exit_command_mode(psmouse);
// Set rate and enable data reporting
(*__slate_slot_3)[0] = 100;
if ps2_command(ps2dev, (*__slate_slot_3).as_mut_ptr() as *mut u8, 4339u32) != 0 {
__v0 = true;
} else {
__v0 = ps2_command(ps2dev, core::ptr::null_mut::<u8>(), 244u32) != 0;
}
if __v0 {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*(*psmouse).ps2dev.serio).dev) as *const device, c"alps: Failed to enable data reporting\n".as_ptr() as *const u8);
return 1i32.wrapping_neg();
} else {
return 0;
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
alps_exit_command_mode(psmouse);
// Leaving the touchpad in command mode will essentially render
// it unusable until the machine reboots, so exit it here just
// to be safe
return 1i32.wrapping_neg();
}
}

fn alps_get_v3_v7_resolution(psmouse: *mut psmouse, reg_pitch: i32) -> i32 {
let mut reg: i32 = 0;
let mut x_pitch: i32 = 0;
let mut y_pitch: i32 = 0;
let mut x_electrode: i32 = 0;
let mut y_electrode: i32 = 0;
let mut x_phys: i32 = 0;
let mut y_phys: i32 = 0;
let r#priv: *mut alps_data = (unsafe { (*psmouse).private }) as *mut alps_data;
reg = alps_command_mode_read_reg(psmouse, reg_pitch);
if reg < 0 {
return reg;
}
x_pitch = (((reg << 4i32) as i8) as i32) >> 4i32; // sign extend lower 4 bits
x_pitch = 50i32.wrapping_add(2i32.wrapping_mul(x_pitch)); // In 0.1 mm units
y_pitch = ((reg as i8) as i32) >> 4i32; // sign extend upper 4 bits
y_pitch = 36i32.wrapping_add(2i32.wrapping_mul(y_pitch)); // In 0.1 mm units
reg = alps_command_mode_read_reg(psmouse, reg_pitch.wrapping_add(1));
if reg < 0 {
return reg;
}
x_electrode = (((reg << 4i32) as i8) as i32) >> 4i32; // sign extend lower 4 bits
x_electrode = 17i32.wrapping_add(x_electrode);
y_electrode = ((reg as i8) as i32) >> 4i32; // sign extend upper 4 bits
y_electrode = 13i32.wrapping_add(y_electrode);
x_phys = x_pitch.wrapping_mul(x_electrode.wrapping_sub(1)); // In 0.1 mm units
y_phys = y_pitch.wrapping_mul(y_electrode.wrapping_sub(1)); // In 0.1 mm units
unsafe {
(*r#priv).x_res = (unsafe { (*r#priv).x_max }.wrapping_mul(10) / x_phys) as u32;
}
// units / mm
unsafe {
(*r#priv).y_res = (unsafe { (*r#priv).y_max }.wrapping_mul(10) / y_phys) as u32;
}
// units / mm
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*(*psmouse).ps2dev.serio).dev) as *const device, c"alps: pitch %dx%d num-electrodes %dx%d physical size %dx%d mm res %dx%d\n".as_ptr() as *const u8, x_pitch, y_pitch, x_electrode, y_electrode, x_phys / 10, y_phys / 10, (*r#priv).x_res, (*r#priv).y_res) };
}
0
}

#[unsafe(link_section = ".text.slate_distinct.alps.alps_hw_init_rushmore_v3")]
extern "C" fn alps_hw_init_rushmore_v3(psmouse: *mut psmouse) -> i32 {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut reg_val: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut i32 = core::ptr::addr_of_mut!(reg_val) as *mut i32;
let mut ps2dev: *mut ps2dev = unsafe { core::mem::zeroed() };
unsafe {
'__join_0: {
let r#priv: *mut alps_data = (*psmouse).private as *mut alps_data;
ps2dev = core::ptr::addr_of_mut!((*psmouse).ps2dev);
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
ret = 1i32.wrapping_neg();
if (*r#priv).flags & 2 != 0 {
*__slate_slot_3 = alps_setup_trackstick_v3(psmouse, 49856i32);
if *__slate_slot_3 == 5i32.wrapping_neg() {
break '__join_0;
}
}
if alps_enter_command_mode(psmouse) != 0 {
__v1 = true;
} else {
__v1 = alps_command_mode_read_reg(psmouse, 49881i32) == 1i32.wrapping_neg();
}
if __v1 {
__v0 = true;
} else {
__v0 = alps_command_mode_write_reg(psmouse, 49867i32, 0u8) != 0;
}
if !__v0 {
if alps_get_v3_v7_resolution(psmouse, 49882i32) == 0 {
*__slate_slot_3 = alps_command_mode_read_reg(psmouse, 49862i32);
if *__slate_slot_3 != 1i32.wrapping_neg() {
if __alps_command_mode_write_reg(psmouse, (*__slate_slot_3 & 253) as u8) == 0 {
if alps_command_mode_write_reg(psmouse, 49865i32, 100u8) == 0 {
// enter absolute mode
*__slate_slot_3 = alps_command_mode_read_reg(psmouse, 49860i32);
if *__slate_slot_3 != 1i32.wrapping_neg() {
if __alps_command_mode_write_reg(psmouse, (*__slate_slot_3 | 2) as u8) == 0 {
alps_exit_command_mode(psmouse);
return ps2_command(ps2dev, core::ptr::null_mut::<u8>(), 244u32);
}
}
}
}
}
}
}
}
alps_exit_command_mode(psmouse);
return ret;
}
}

/// Must be in command mode when calling this function
fn alps_absolute_mode_v4(psmouse: *mut psmouse) -> i32 {
let mut reg_val: i32 = alps_command_mode_read_reg(psmouse, 4i32);
if reg_val == 1i32.wrapping_neg() {
return 1i32.wrapping_neg();
}
reg_val |= 2;
if __alps_command_mode_write_reg(psmouse, reg_val as u8) != 0 {
return 1i32.wrapping_neg();
}
0
}

#[unsafe(link_section = ".text.slate_distinct.alps.alps_hw_init_v4")]
extern "C" fn alps_hw_init_v4(psmouse: *mut psmouse) -> i32 {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut __v2: bool = unsafe { core::mem::zeroed() };
let mut __v3: bool = unsafe { core::mem::zeroed() };
let mut param: core::mem::MaybeUninit<[u8; 4]> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut [u8; 4] = core::ptr::addr_of_mut!(param) as *mut [u8; 4];
let mut ps2dev: *mut ps2dev = unsafe { core::mem::zeroed() };
unsafe {
ps2dev = core::ptr::addr_of_mut!((*psmouse).ps2dev);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if alps_enter_command_mode(psmouse) == 0 {
if alps_absolute_mode_v4(psmouse) != 0 {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*(*psmouse).ps2dev.serio).dev) as *const device, c"alps: Failed to enter absolute mode\n".as_ptr() as *const u8);
} else {
if alps_command_mode_write_reg(psmouse, 7i32, 140u8) == 0 {
if alps_command_mode_write_reg(psmouse, 329i32, 3u8) == 0 {
if alps_command_mode_write_reg(psmouse, 352i32, 3u8) == 0 {
if alps_command_mode_write_reg(psmouse, 383i32, 21u8) == 0 {
if alps_command_mode_write_reg(psmouse, 337i32, 1u8) == 0 {
if alps_command_mode_write_reg(psmouse, 360i32, 3u8) == 0 {
if alps_command_mode_write_reg(psmouse, 330i32, 3u8) == 0 {
if alps_command_mode_write_reg(psmouse, 353i32, 3u8) == 0 {
alps_exit_command_mode(psmouse);
// This sequence changes the output from a 9-byte to an
// 8-byte format. All the same data seems to be present,
// just in a more compact format.
(*__slate_slot_4)[0] = 200;
(*__slate_slot_4)[1] = 100;
(*__slate_slot_4)[2] = 80;
if ps2_command(ps2dev, ((*__slate_slot_4).as_mut_ptr() as *mut u8).wrapping_offset(0 as isize), 4339u32) != 0 {
__v3 = true;
} else {
__v3 = ps2_command(ps2dev, ((*__slate_slot_4).as_mut_ptr() as *mut u8).wrapping_offset(1 as isize), 4339u32) != 0;
}
if __v3 {
__v2 = true;
} else {
__v2 = ps2_command(ps2dev, ((*__slate_slot_4).as_mut_ptr() as *mut u8).wrapping_offset(2 as isize), 4339u32) != 0;
}
if __v2 {
__v1 = true;
} else {
__v1 = ps2_command(ps2dev, (*__slate_slot_4).as_mut_ptr() as *mut u8, 754u32) != 0;
}
if __v1 {
return 1i32.wrapping_neg();
} else {
// Set rate and enable data reporting
(*__slate_slot_4)[0] = 100;
if ps2_command(ps2dev, (*__slate_slot_4).as_mut_ptr() as *mut u8, 4339u32) != 0 {
__v0 = true;
} else {
__v0 = ps2_command(ps2dev, core::ptr::null_mut::<u8>(), 244u32) != 0;
}
if __v0 {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*(*psmouse).ps2dev.serio).dev) as *const device, c"alps: Failed to enable data reporting\n".as_ptr() as *const u8);
return 1i32.wrapping_neg();
} else {
return 0;
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
alps_exit_command_mode(psmouse);
// Leaving the touchpad in command mode will essentially render
// it unusable until the machine reboots, so exit it here just
// to be safe
return 1i32.wrapping_neg();
}
}

fn alps_get_otp_values_ss4_v2(psmouse: *mut psmouse, index: u8, otp: *mut u8) -> i32 {
let ps2dev: *mut ps2dev = unsafe { core::ptr::addr_of_mut!((*psmouse).ps2dev) };
match index as i32 {
0 => {
let __v0: bool = (unsafe { ps2_command(ps2dev, core::ptr::null_mut::<u8>(), 234u32) }) != 0 || (unsafe { ps2_command(ps2dev, core::ptr::null_mut::<u8>(), 234u32) }) != 0;
let __v1: bool = __v0 || (unsafe { ps2_command(ps2dev, otp, 1001u32) }) != 0;
if __v1 {
return 1i32.wrapping_neg();
}
}
1 => {
let __v3: bool = (unsafe { ps2_command(ps2dev, core::ptr::null_mut::<u8>(), 240u32) }) != 0 || (unsafe { ps2_command(ps2dev, core::ptr::null_mut::<u8>(), 240u32) }) != 0;
let __v4: bool = __v3 || (unsafe { ps2_command(ps2dev, otp, 1001u32) }) != 0;
if __v4 {
return 1i32.wrapping_neg();
}
}
_ => {
}
}
0
}

fn alps_update_device_area_ss4_v2(otp: *mut [u8; 4], r#priv: *mut alps_data) -> i32 {
let mut num_x_electrode: i32 = 0;
let mut num_y_electrode: i32 = 0;
let mut x_pitch: i32 = 0;
let mut y_pitch: i32 = 0;
let mut x_phys: i32 = 0;
let mut y_phys: i32 = 0;
if IS_SS4PLUS_DEV!((*r#priv).dev_id) {
num_x_electrode = 16i32.wrapping_add(((unsafe { *((*otp.wrapping_offset(0 as isize)).as_mut_ptr() as *mut u8).wrapping_offset(2 as isize) }) as i32) & 15);
num_y_electrode = 5i32.wrapping_add(((unsafe { *((*otp.wrapping_offset(0 as isize)).as_mut_ptr() as *mut u8).wrapping_offset(2 as isize) }) as i32) >> 4i32 & 15);
unsafe {
(*r#priv).x_max = num_x_electrode.wrapping_sub(1).wrapping_mul(128);
}
unsafe {
(*r#priv).y_max = num_y_electrode.wrapping_sub(1).wrapping_mul(128);
}
x_pitch = (((unsafe { *((*otp.wrapping_offset(0 as isize)).as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) }) as i32) & 15).wrapping_add(37);
y_pitch = (((unsafe { *((*otp.wrapping_offset(0 as isize)).as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) }) as i32) >> 4i32 & 15).wrapping_add(37);
} else {
num_x_electrode = 7i32.wrapping_add(((unsafe { *((*otp.wrapping_offset(1 as isize)).as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) }) as i32) & 15);
num_y_electrode = 7i32.wrapping_add(((unsafe { *((*otp.wrapping_offset(1 as isize)).as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) }) as i32) >> 4i32 & 15);
unsafe {
(*r#priv).x_max = num_x_electrode.wrapping_sub(1).wrapping_mul(256);
}
unsafe {
(*r#priv).y_max = num_y_electrode.wrapping_sub(1).wrapping_mul(256);
}
x_pitch = (((unsafe { *((*otp.wrapping_offset(1 as isize)).as_mut_ptr() as *mut u8).wrapping_offset(2 as isize) }) as i32) >> 2i32 & 7).wrapping_add(50);
y_pitch = (((unsafe { *((*otp.wrapping_offset(1 as isize)).as_mut_ptr() as *mut u8).wrapping_offset(2 as isize) }) as i32) >> 5i32 & 7).wrapping_add(50);
}
x_phys = x_pitch.wrapping_mul(num_x_electrode.wrapping_sub(1)); // In 0.1 mm units
y_phys = y_pitch.wrapping_mul(num_y_electrode.wrapping_sub(1)); // In 0.1 mm units
unsafe {
(*r#priv).x_res = (unsafe { (*r#priv).x_max }.wrapping_mul(10) / x_phys) as u32;
}
// units / mm
unsafe {
(*r#priv).y_res = (unsafe { (*r#priv).y_max }.wrapping_mul(10) / y_phys) as u32;
}
// units / mm
0
}

fn alps_update_btn_info_ss4_v2(otp: *mut [u8; 4], r#priv: *mut alps_data) -> i32 {
let is_btnless: u8 = if IS_SS4PLUS_DEV!((*r#priv).dev_id) {
(((unsafe { *((*otp.wrapping_offset(1 as isize)).as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) }) as i32) >> 1i32 & 1) as u8
} else {
(((unsafe { *((*otp.wrapping_offset(1 as isize)).as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) }) as i32) >> 3i32 & 1) as u8
};
if is_btnless != 0 {
unsafe {
(*r#priv).flags = (*r#priv).flags | 512;
}
}
0
}

fn alps_update_dual_info_ss4_v2(otp: *mut [u8; 4], r#priv: *mut alps_data, psmouse: *mut psmouse) -> i32 {
let mut is_dual: bool = false;
let mut reg_val: i32 = 0;
let ps2dev: *mut ps2dev = unsafe { core::ptr::addr_of_mut!((*psmouse).ps2dev) };
if IS_SS4PLUS_DEV!((*r#priv).dev_id) {
is_dual = ((unsafe { *((*otp.wrapping_offset(0 as isize)).as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) }) as i32) >> 4i32 & 1 != 0;
if !is_dual {
// For support TrackStick of Thinkpad L/E series
let __v0: bool = alps_exit_command_mode(psmouse) == 0 && alps_enter_command_mode(psmouse) == 0;
if __v0 {
reg_val = alps_command_mode_read_reg(psmouse, 215i32);
}
alps_exit_command_mode(psmouse);
unsafe { ps2_command(ps2dev, core::ptr::null_mut::<u8>(), 244u32) };
if reg_val == 12 || reg_val == 29 {
is_dual = true;
}
}
}
if is_dual {
unsafe {
(*r#priv).flags = (*r#priv).flags | (2 | 1024);
}
}
0
}

fn alps_set_defaults_ss4_v2(psmouse: *mut psmouse, r#priv: *mut alps_data) -> i32 {
let mut otp: [[u8; 4]; 2] = unsafe { core::mem::zeroed() };
unsafe { memset(otp.as_mut_ptr() as *mut (), 0i32, core::mem::size_of::<[[u8; 4]; 2]>() as u64) };
let __v0: bool = alps_get_otp_values_ss4_v2(psmouse, 1u8, (otp[1].as_mut_ptr() as *mut u8).wrapping_offset(0 as isize)) != 0 || alps_get_otp_values_ss4_v2(psmouse, 0u8, (otp[0].as_mut_ptr() as *mut u8).wrapping_offset(0 as isize)) != 0;
if __v0 {
return 1i32.wrapping_neg();
}
alps_update_device_area_ss4_v2(otp.as_mut_ptr() as *mut [u8; 4], r#priv);
alps_update_btn_info_ss4_v2(otp.as_mut_ptr() as *mut [u8; 4], r#priv);
alps_update_dual_info_ss4_v2(otp.as_mut_ptr() as *mut [u8; 4], r#priv, psmouse);
0
}

fn alps_dolphin_get_device_area(psmouse: *mut psmouse, r#priv: *mut alps_data) -> i32 {
let ps2dev: *mut ps2dev = unsafe { core::ptr::addr_of_mut!((*psmouse).ps2dev) };
let mut param: [u8; 4] = {
let mut __t0: [u8; 4] = unsafe { core::mem::zeroed() };
__t0[0] = 0u8;
__t0
};
let mut num_x_electrode: i32 = 0;
let mut num_y_electrode: i32 = 0;
if alps_enter_command_mode(psmouse) != 0 {
return 1i32.wrapping_neg();
}
param[0] = 10;
let __v0: bool = (unsafe { ps2_command(ps2dev, core::ptr::null_mut::<u8>(), 236u32) }) != 0 || (unsafe { ps2_command(ps2dev, core::ptr::null_mut::<u8>(), 240u32) }) != 0;
let __v1: bool = __v0 || (unsafe { ps2_command(ps2dev, core::ptr::null_mut::<u8>(), 240u32) }) != 0;
let __v2: bool = __v1 || (unsafe { ps2_command(ps2dev, (param.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize), 4339u32) }) != 0;
let __v3: bool = __v2 || (unsafe { ps2_command(ps2dev, (param.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize), 4339u32) }) != 0;
if __v3 {
return 1i32.wrapping_neg();
}
if (unsafe { ps2_command(ps2dev, param.as_mut_ptr() as *mut u8, 1001u32) }) != 0 {
return 1i32.wrapping_neg();
}
// Dolphin's sensor line number is not fixed. It can be calculated
// by adding the device's register value with DOLPHIN_PROFILE_X/YOFFSET.
// Further more, we can get device's x_max and y_max by multiplying
// sensor line number with DOLPHIN_COUNT_PER_ELECTRODE.
//
// e.g. When we get register's sensor_x = 11 & sensor_y = 8,
// 	real sensor line number X = 11 + 8 = 19, and
// 	real sensor line number Y = 8 + 1 = 9.
// 	So, x_max = (19 - 1) * 64 = 1152, and
// 	    y_max = (9 - 1) * 64 = 512.
num_x_electrode = 8i32.wrapping_add((param[2] as i32) & 15);
num_y_electrode = 1i32.wrapping_add((param[2] as i32) >> 4i32 & 15);
unsafe {
(*r#priv).x_bits = num_x_electrode;
}
unsafe {
(*r#priv).y_bits = num_y_electrode;
}
unsafe {
(*r#priv).x_max = num_x_electrode.wrapping_sub(1).wrapping_mul(64);
}
unsafe {
(*r#priv).y_max = num_y_electrode.wrapping_sub(1).wrapping_mul(64);
}
if alps_exit_command_mode(psmouse) != 0 {
return 1i32.wrapping_neg();
}
0
}

#[unsafe(link_section = ".text.slate_distinct.alps.alps_hw_init_dolphin_v1")]
extern "C" fn alps_hw_init_dolphin_v1(psmouse: *mut psmouse) -> i32 {
let ps2dev: *mut ps2dev = unsafe { core::ptr::addr_of_mut!((*psmouse).ps2dev) };
let mut param: [u8; 2] = [0u8; 2];
// This is dolphin "v1" as empirically defined by florin9doi
param[0] = 100;
param[1] = 40;
let __v0: bool = (unsafe { ps2_command(ps2dev, core::ptr::null_mut::<u8>(), 234u32) }) != 0 || (unsafe { ps2_command(ps2dev, (param.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize), 4339u32) }) != 0;
let __v1: bool = __v0 || (unsafe { ps2_command(ps2dev, (param.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize), 4339u32) }) != 0;
if __v1 {
return 1i32.wrapping_neg();
}
0
}

#[unsafe(link_section = ".text.slate_distinct.alps.alps_hw_init_v7")]
extern "C" fn alps_hw_init_v7(psmouse: *mut psmouse) -> i32 {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut reg_val: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut i32 = core::ptr::addr_of_mut!(reg_val) as *mut i32;
let mut ps2dev: *mut ps2dev = unsafe { core::mem::zeroed() };
unsafe {
ps2dev = core::ptr::addr_of_mut!((*psmouse).ps2dev);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
ret = 1i32.wrapping_neg();
if alps_enter_command_mode(psmouse) != 0 {
__v0 = true;
} else {
__v0 = alps_command_mode_read_reg(psmouse, 49881i32) == 1i32.wrapping_neg();
}
if !__v0 {
if alps_get_v3_v7_resolution(psmouse, 50071i32) == 0 {
if alps_command_mode_write_reg(psmouse, 49865i32, 100u8) == 0 {
*__slate_slot_2 = alps_command_mode_read_reg(psmouse, 49860i32);
if *__slate_slot_2 != 1i32.wrapping_neg() {
if __alps_command_mode_write_reg(psmouse, (*__slate_slot_2 | 2) as u8) == 0 {
alps_exit_command_mode(psmouse);
return ps2_command(ps2dev, core::ptr::null_mut::<u8>(), 244u32);
}
}
}
}
}
alps_exit_command_mode(psmouse);
return ret;
}
}

#[unsafe(link_section = ".text.slate_distinct.alps.alps_hw_init_ss4_v2")]
extern "C" fn alps_hw_init_ss4_v2(psmouse: *mut psmouse) -> i32 {
let mut __v0: bool = unsafe { core::mem::zeroed() };
// T.B.D. Decread noise packet number, delete in the future
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut __v2: bool = unsafe { core::mem::zeroed() };
let mut __v3: bool = unsafe { core::mem::zeroed() };
// enter absolute mode
let mut __v4: bool = unsafe { core::mem::zeroed() };
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut param: [u8; 2] = unsafe { core::mem::zeroed() };
let mut ps2dev: *mut ps2dev = unsafe { core::mem::zeroed() };
unsafe {
ps2dev = core::ptr::addr_of_mut!((*psmouse).ps2dev);
param = [100, 40];
ret = 1i32.wrapping_neg();
if ps2_command(ps2dev, core::ptr::null_mut::<u8>(), 234u32) != 0 {
__v4 = true;
} else {
__v4 = ps2_command(ps2dev, core::ptr::null_mut::<u8>(), 234u32) != 0;
}
if __v4 {
__v3 = true;
} else {
__v3 = ps2_command(ps2dev, (param.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) as *mut u8, 4339u32) != 0;
}
if __v3 {
__v2 = true;
} else {
__v2 = ps2_command(ps2dev, (param.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) as *mut u8, 4339u32) != 0;
}
if !__v2 {
if alps_exit_command_mode(psmouse) != 0 {
__v1 = true;
} else {
__v1 = alps_enter_command_mode(psmouse) != 0;
}
if __v1 {
__v0 = true;
} else {
__v0 = alps_command_mode_write_reg(psmouse, 29i32, 32u8) != 0;
}
if !__v0 {
alps_exit_command_mode(psmouse);
return ps2_command(ps2dev, core::ptr::null_mut::<u8>(), 244u32);
}
}
alps_exit_command_mode(psmouse);
return ret;
}
}

fn alps_set_protocol(psmouse: *mut psmouse, r#priv: *mut alps_data, protocol: *const alps_protocol_info) -> i32 {
unsafe {
(*psmouse).private = r#priv as *mut ();
}
timer_setup!(unsafe { core::ptr::addr_of_mut!((*r#priv).timer) }, Some(alps_flush_packet), 0i32);
unsafe {
(*r#priv).proto_version = (*protocol).version;
}
unsafe {
(*r#priv).byte0 = (*protocol).byte0;
}
unsafe {
(*r#priv).mask0 = (*protocol).mask0;
}
unsafe {
(*r#priv).flags = (*protocol).flags as i32;
}
unsafe {
(*r#priv).x_max = 2000;
}
unsafe {
(*r#priv).y_max = 1400;
}
unsafe {
(*r#priv).x_bits = 15;
}
unsafe {
(*r#priv).y_bits = 11;
}
match (unsafe { (*r#priv).proto_version }) as i32 {
256 | 512 => {
unsafe {
(*r#priv).hw_init = Some(alps_hw_init_v1_v2);
}
unsafe {
(*r#priv).process_packet = Some(alps_process_packet_v1_v2);
}
unsafe {
(*r#priv).set_abs_params = Some(alps_set_abs_params_st);
}
unsafe {
(*r#priv).x_max = 1023;
}
unsafe {
(*r#priv).y_max = 767;
}
if (unsafe { dmi_check_system(core::ptr::addr_of!(alps_dmi_has_separate_stick_buttons.0) as *const dmi_system_id) }) != 0 {
unsafe {
(*r#priv).flags = (*r#priv).flags | 256;
}
}
}
768 => {
unsafe {
(*r#priv).hw_init = Some(alps_hw_init_v3);
}
unsafe {
(*r#priv).process_packet = Some(alps_process_packet_v3);
}
unsafe {
(*r#priv).set_abs_params = Some(alps_set_abs_params_semi_mt);
}
unsafe {
(*r#priv).decode_fields = Some(alps_decode_pinnacle);
}
unsafe {
(*r#priv).nibble_commands = core::ptr::addr_of!(alps_v3_nibble_commands.0) as *const alps_nibble_commands;
}
unsafe {
(*r#priv).addr_command = 236;
}
if alps_probe_trackstick_v3_v7(psmouse, 0i32) < 0 {
let __v6: i32 = (unsafe { (*r#priv).flags }) & !2;
unsafe {
(*r#priv).flags = __v6;
}
}
}
784 => {
unsafe {
(*r#priv).hw_init = Some(alps_hw_init_rushmore_v3);
}
unsafe {
(*r#priv).process_packet = Some(alps_process_packet_v3);
}
unsafe {
(*r#priv).set_abs_params = Some(alps_set_abs_params_semi_mt);
}
unsafe {
(*r#priv).decode_fields = Some(alps_decode_rushmore);
}
unsafe {
(*r#priv).nibble_commands = core::ptr::addr_of!(alps_v3_nibble_commands.0) as *const alps_nibble_commands;
}
unsafe {
(*r#priv).addr_command = 236;
}
unsafe {
(*r#priv).x_bits = 16;
}
unsafe {
(*r#priv).y_bits = 12;
}
if alps_probe_trackstick_v3_v7(psmouse, 49856i32) < 0 {
let __v9: i32 = (unsafe { (*r#priv).flags }) & !2;
unsafe {
(*r#priv).flags = __v9;
}
}
}
1024 => {
unsafe {
(*r#priv).hw_init = Some(alps_hw_init_v4);
}
unsafe {
(*r#priv).process_packet = Some(alps_process_packet_v4);
}
unsafe {
(*r#priv).set_abs_params = Some(alps_set_abs_params_semi_mt);
}
unsafe {
(*r#priv).nibble_commands = core::ptr::addr_of!(alps_v4_nibble_commands.0) as *const alps_nibble_commands;
}
unsafe {
(*r#priv).addr_command = 245;
}
}
1280 => {
unsafe {
(*r#priv).hw_init = Some(alps_hw_init_dolphin_v1);
}
unsafe {
(*r#priv).process_packet = Some(alps_process_touchpad_packet_v3_v5);
}
unsafe {
(*r#priv).decode_fields = Some(alps_decode_dolphin);
}
unsafe {
(*r#priv).set_abs_params = Some(alps_set_abs_params_semi_mt);
}
unsafe {
(*r#priv).nibble_commands = core::ptr::addr_of!(alps_v3_nibble_commands.0) as *const alps_nibble_commands;
}
unsafe {
(*r#priv).addr_command = 236;
}
unsafe {
(*r#priv).x_bits = 23;
}
unsafe {
(*r#priv).y_bits = 12;
}
if alps_dolphin_get_device_area(psmouse, r#priv) != 0 {
return 5i32.wrapping_neg();
}
}
1536 => {
unsafe {
(*r#priv).hw_init = Some(alps_hw_init_v6);
}
unsafe {
(*r#priv).process_packet = Some(alps_process_packet_v6);
}
unsafe {
(*r#priv).set_abs_params = Some(alps_set_abs_params_st);
}
unsafe {
(*r#priv).nibble_commands = core::ptr::addr_of!(alps_v6_nibble_commands.0) as *const alps_nibble_commands;
}
unsafe {
(*r#priv).x_max = 2047;
}
unsafe {
(*r#priv).y_max = 1535;
}
}
1792 => {
unsafe {
(*r#priv).hw_init = Some(alps_hw_init_v7);
}
unsafe {
(*r#priv).process_packet = Some(alps_process_packet_v7);
}
unsafe {
(*r#priv).decode_fields = Some(alps_decode_packet_v7);
}
unsafe {
(*r#priv).set_abs_params = Some(alps_set_abs_params_v7);
}
unsafe {
(*r#priv).nibble_commands = core::ptr::addr_of!(alps_v3_nibble_commands.0) as *const alps_nibble_commands;
}
unsafe {
(*r#priv).addr_command = 236;
}
unsafe {
(*r#priv).x_max = 4095;
}
unsafe {
(*r#priv).y_max = 2047;
}
if ((unsafe { *((*r#priv).fw_ver.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) }) as i32) != 186 {
unsafe {
(*r#priv).flags = (*r#priv).flags | 512;
}
}
if alps_probe_trackstick_v3_v7(psmouse, 49856i32) < 0 {
let __v15: i32 = (unsafe { (*r#priv).flags }) & !2;
unsafe {
(*r#priv).flags = __v15;
}
}
}
2048 => {
unsafe {
(*r#priv).hw_init = Some(alps_hw_init_ss4_v2);
}
unsafe {
(*r#priv).process_packet = Some(alps_process_packet_ss4_v2);
}
unsafe {
(*r#priv).decode_fields = Some(alps_decode_ss4_v2);
}
unsafe {
(*r#priv).set_abs_params = Some(alps_set_abs_params_ss4_v2);
}
unsafe {
(*r#priv).nibble_commands = core::ptr::addr_of!(alps_v3_nibble_commands.0) as *const alps_nibble_commands;
}
unsafe {
(*r#priv).addr_command = 236;
}
if alps_set_defaults_ss4_v2(psmouse, r#priv) != 0 {
return 5i32.wrapping_neg();
}
}
_ => {
}
}
0
}

fn alps_match_table(e7: *mut u8, ec: *mut u8) -> *const alps_protocol_info {
let mut model: *const alps_model_info = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
while (i as u64) < ARRAY_SIZE__29!() {
model = unsafe { core::ptr::addr_of!(alps_model_data.0) as *const alps_model_info }.wrapping_offset(i as isize);
if (unsafe { memcmp(e7 as *const (), ((*model).signature.as_ptr() as *const u8) as *const (), core::mem::size_of::<[u8; 3]>() as u64) }) == 0 {
return unsafe { core::ptr::addr_of!((*model).protocol_info) };
}
i = i.wrapping_add(1);
}
core::ptr::null::<alps_protocol_info>()
}

fn alps_is_cs19_trackpoint(psmouse: *mut psmouse) -> bool {
let mut param: [u8; 2] = {
let mut __t0: [u8; 2] = unsafe { core::mem::zeroed() };
__t0[0] = 0u8;
__t0
};
if (unsafe { ps2_command(core::ptr::addr_of_mut!((*psmouse).ps2dev), param.as_mut_ptr() as *mut u8, MAKE_PS2_CMD!(0i32, 2i32, 225i32) as u32) }) != 0 {
return false;
}
// param[0] contains the trackpoint device variant_id while
// param[1] contains the firmware_id. So far all alps
// trackpoint-only devices have their variant_ids equal
// TP_VARIANT_ALPS and their firmware_ids are in 0x20~0x2f range.
(param[0] as i32) == 2 && (param[1] as i32) & 240 == 32
}

fn alps_identify(psmouse: *mut psmouse, r#priv: *mut alps_data) -> i32 {
let mut protocol: *const alps_protocol_info = unsafe { core::mem::zeroed() };
let mut e6: [u8; 4] = [0u8; 4];
let mut e7: [u8; 4] = [0u8; 4];
let mut ec: [u8; 4] = [0u8; 4];
let mut error: i32 = 0;
// First try "E6 report".
// ALPS should return 0,0,10 or 0,0,100 if no buttons are pressed.
// The bits 0-2 of the first byte will be 1s if some buttons are
// pressed.
if alps_rpt_cmd(psmouse, 4328i32, 230i32, e6.as_mut_ptr() as *mut u8) != 0 {
return 5i32.wrapping_neg();
}
if (e6[0] as i32) & 248 != 0 || (e6[1] as i32) != 0 || (e6[2] as i32) != 10 && (e6[2] as i32) != 100 {
return 22i32.wrapping_neg();
}
// Now get the "E7" and "EC" reports.  These will uniquely identify
// most ALPS touchpads.
let __v0: bool = alps_rpt_cmd(psmouse, 4328i32, 231i32, e7.as_mut_ptr() as *mut u8) != 0 || alps_rpt_cmd(psmouse, 4328i32, 236i32, ec.as_mut_ptr() as *mut u8) != 0;
let __v1: bool = __v0 || alps_exit_command_mode(psmouse) != 0;
if __v1 {
return 5i32.wrapping_neg();
}
protocol = alps_match_table(e7.as_mut_ptr() as *mut u8, ec.as_mut_ptr() as *mut u8);
if protocol.is_null() {
if (e7[0] as i32) == 115 && (e7[1] as i32) == 2 && (e7[2] as i32) == 100 && (ec[2] as i32) == 138 {
protocol = unsafe { core::ptr::addr_of!(alps_v4_protocol_data) };
} else {
if (e7[0] as i32) == 115 && (e7[1] as i32) == 3 && (e7[2] as i32) == 80 && (ec[0] as i32) == 115 && ((ec[1] as i32) == 1 || (ec[1] as i32) == 2) {
protocol = unsafe { core::ptr::addr_of!(alps_v5_protocol_data) };
} else {
if (ec[0] as i32) == 136 && ((ec[1] as i32) & 240 == 176 || (ec[1] as i32) & 240 == 192) {
protocol = unsafe { core::ptr::addr_of!(alps_v7_protocol_data) };
} else {
if (ec[0] as i32) == 136 && (ec[1] as i32) == 8 {
protocol = unsafe { core::ptr::addr_of!(alps_v3_rushmore_data) };
} else {
if (ec[0] as i32) == 136 && (ec[1] as i32) == 7 && (ec[2] as i32) >= 144 && (ec[2] as i32) <= 157 {
protocol = unsafe { core::ptr::addr_of!(alps_v3_protocol_data) };
} else {
if (e7[0] as i32) == 115 && (e7[1] as i32) == 3 && ((e7[2] as i32) == 20 || (e7[2] as i32) == 40) {
protocol = unsafe { core::ptr::addr_of!(alps_v8_protocol_data) };
} else {
if (e7[0] as i32) == 115 && (e7[1] as i32) == 3 && (e7[2] as i32) == 200 {
protocol = unsafe { core::ptr::addr_of!(alps_v9_protocol_data) };
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*(*psmouse).ps2dev.serio).dev) as *const device, c"alps: Unsupported ALPS V9 touchpad: E7=%3ph, EC=%3ph\n".as_ptr() as *const u8, e7.as_mut_ptr() as *mut u8, ec.as_mut_ptr() as *mut u8) };
return 22i32.wrapping_neg();
} else {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*(*psmouse).ps2dev.serio).dev) as *const device, c"alps: Likely not an ALPS touchpad: E7=%3ph, EC=%3ph\n".as_ptr() as *const u8, e7.as_mut_ptr() as *mut u8, ec.as_mut_ptr() as *mut u8) };
}
return 22i32.wrapping_neg();
}
}
}
}
}
}
}
}
if !r#priv.is_null() {
// Save Device ID and Firmware version
unsafe { memcpy(((*r#priv).dev_id.as_mut_ptr() as *mut u8) as *mut (), e7.as_mut_ptr() as *const (), 3u64) };
unsafe { memcpy(((*r#priv).fw_ver.as_mut_ptr() as *mut u8) as *mut (), ec.as_mut_ptr() as *const (), 3u64) };
error = alps_set_protocol(psmouse, r#priv, protocol);
if error != 0 {
return error;
}
}
0
}

#[unsafe(link_section = ".text.slate_distinct.alps.alps_reconnect")]
extern "C" fn alps_reconnect(psmouse: *mut psmouse) -> i32 {
let r#priv: *mut alps_data = (unsafe { (*psmouse).private }) as *mut alps_data;
unsafe { psmouse_reset(psmouse) };
if alps_identify(psmouse, r#priv) < 0 {
return 1i32.wrapping_neg();
}
unsafe { unsafe { (*r#priv).hw_init }.unwrap()(psmouse) }
}

#[unsafe(link_section = ".text.slate_distinct.alps.alps_disconnect")]
extern "C" fn alps_disconnect(psmouse: *mut psmouse) {
let r#priv: *mut alps_data = (unsafe { (*psmouse).private }) as *mut alps_data;
unsafe { psmouse_reset(psmouse) };
unsafe { timer_shutdown_sync(core::ptr::addr_of_mut!((*r#priv).timer)) };
unsafe { disable_work_sync(core::ptr::addr_of_mut!((*r#priv).dev3_register_work)) };
if !unsafe { (*r#priv).dev2 }.is_null() {
unsafe { input_unregister_device((*r#priv).dev2) };
}
if !IS_ERR_OR_NULL((unsafe { (*r#priv).dev3 }) as *const ()) {
unsafe { input_unregister_device((*r#priv).dev3) };
}
unsafe { kfree(r#priv as *const ()) };
}

#[unsafe(link_section = ".text.slate_distinct.alps.alps_set_abs_params_st")]
extern "C" fn alps_set_abs_params_st(r#priv: *mut alps_data, dev1: *mut input_dev) {
unsafe { input_set_abs_params(dev1, 0u32, 0i32, (*r#priv).x_max, 0i32, 0i32) };
unsafe { input_set_abs_params(dev1, 1u32, 0i32, (*r#priv).y_max, 0i32, 0i32) };
unsafe { input_set_abs_params(dev1, 24u32, 0i32, 127i32, 0i32, 0i32) };
}

fn alps_set_abs_params_mt_common(r#priv: *mut alps_data, dev1: *mut input_dev) {
unsafe { input_set_abs_params(dev1, 53u32, 0i32, (*r#priv).x_max, 0i32, 0i32) };
unsafe { input_set_abs_params(dev1, 54u32, 0i32, (*r#priv).y_max, 0i32, 0i32) };
input_abs_set_res(dev1, 53u32, (unsafe { (*r#priv).x_res }) as i32);
input_abs_set_res(dev1, 54u32, (unsafe { (*r#priv).y_res }) as i32);
set_bit(334i64, (unsafe { (*dev1).keybit.as_mut_ptr() as *mut u64 }) as *mut u64);
set_bit(335i64, (unsafe { (*dev1).keybit.as_mut_ptr() as *mut u64 }) as *mut u64);
}

#[unsafe(link_section = ".text.slate_distinct.alps.alps_set_abs_params_semi_mt")]
extern "C" fn alps_set_abs_params_semi_mt(r#priv: *mut alps_data, dev1: *mut input_dev) {
alps_set_abs_params_mt_common(r#priv, dev1);
unsafe { input_set_abs_params(dev1, 24u32, 0i32, 127i32, 0i32, 0i32) };
unsafe { input_mt_init_slots(dev1, 4u32, 21u32) };
}

#[unsafe(link_section = ".text.slate_distinct.alps.alps_set_abs_params_v7")]
extern "C" fn alps_set_abs_params_v7(r#priv: *mut alps_data, dev1: *mut input_dev) {
alps_set_abs_params_mt_common(r#priv, dev1);
set_bit(328i64, (unsafe { (*dev1).keybit.as_mut_ptr() as *mut u64 }) as *mut u64);
unsafe { input_mt_init_slots(dev1, 4u32, 13u32) };
set_bit(328i64, (unsafe { (*dev1).keybit.as_mut_ptr() as *mut u64 }) as *mut u64);
}

#[unsafe(link_section = ".text.slate_distinct.alps.alps_set_abs_params_ss4_v2")]
extern "C" fn alps_set_abs_params_ss4_v2(r#priv: *mut alps_data, dev1: *mut input_dev) {
alps_set_abs_params_mt_common(r#priv, dev1);
unsafe { input_set_abs_params(dev1, 24u32, 0i32, 127i32, 0i32, 0i32) };
set_bit(328i64, (unsafe { (*dev1).keybit.as_mut_ptr() as *mut u64 }) as *mut u64);
unsafe { input_mt_init_slots(dev1, 4u32, 13u32) };
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.alps.alps_init")]
extern "C" fn alps_init(psmouse: *mut psmouse) -> i32 {
let mut __slate_compound_46: __SlateRecord_types_h_197 = unsafe { core::mem::zeroed() };
let mut dev2: core::mem::MaybeUninit<*mut input_dev> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut input_dev = core::ptr::addr_of_mut!(dev2) as *mut *mut input_dev;
let mut error: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_43: *mut i32 = core::ptr::addr_of_mut!(error) as *mut i32;
let mut dev1: *mut input_dev = unsafe { core::mem::zeroed() };
let mut r#priv: *mut alps_data = unsafe { core::mem::zeroed() };
unsafe {
'__join_0: {
r#priv = (*psmouse).private as *mut alps_data;
dev1 = (*psmouse).dev;
let __slate_auto_bytes_1 = __slate_slot_43 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
*__slate_slot_43 = unsafe { (*r#priv).hw_init }.unwrap()(psmouse);
if *__slate_slot_43 == 0 {
// Undo part of setup done for us by psmouse core since touchpad
// is not a relative device.
__clear_bit!(2i32, (*dev1).evbit.as_mut_ptr() as *mut u64);
__clear_bit!(0i32, (*dev1).relbit.as_mut_ptr() as *mut u64);
__clear_bit!(1i32, (*dev1).relbit.as_mut_ptr() as *mut u64);
// Now set up our capabilities.
let __v42: *mut u64 = ((*dev1).evbit.as_mut_ptr() as *mut u64).wrapping_offset(BIT_WORD__3!(1i32) as isize);
let __v41: u64 = *__v42;
let __v40: u64 = __v41 | BIT_MASK__2!(1i32);
*__v42 = __v40;
let __v39: *mut u64 = ((*dev1).keybit.as_mut_ptr() as *mut u64).wrapping_offset(BIT_WORD__3!(330i32) as isize);
let __v38: u64 = *__v39;
let __v37: u64 = __v38 | BIT_MASK__2!(330i32);
*__v39 = __v37;
let __v36: *mut u64 = ((*dev1).keybit.as_mut_ptr() as *mut u64).wrapping_offset(BIT_WORD__3!(325i32) as isize);
let __v35: u64 = *__v36;
let __v34: u64 = __v35 | BIT_MASK__2!(325i32);
*__v36 = __v34;
let __v33: *mut u64 = ((*dev1).keybit.as_mut_ptr() as *mut u64).wrapping_offset(BIT_WORD__3!(272i32) as isize);
let __v32: u64 = *__v33;
let __v31: u64 = __v32 | (BIT_MASK__2!(272i32) | BIT_MASK__2!(273i32));
*__v33 = __v31;
let __v30: *mut u64 = ((*dev1).evbit.as_mut_ptr() as *mut u64).wrapping_offset(BIT_WORD__3!(3i32) as isize);
let __v29: u64 = *__v30;
let __v28: u64 = __v29 | BIT_MASK__2!(3i32);
*__v30 = __v28;
unsafe { (*r#priv).set_abs_params }.unwrap()(r#priv, dev1);
if (*r#priv).flags & 8 != 0 {
let __v27: *mut u64 = ((*dev1).evbit.as_mut_ptr() as *mut u64).wrapping_offset(BIT_WORD__3!(2i32) as isize);
let __v26: u64 = *__v27;
let __v25: u64 = __v26 | BIT_MASK__2!(2i32);
*__v27 = __v25;
let __v24: *mut u64 = ((*dev1).relbit.as_mut_ptr() as *mut u64).wrapping_offset(BIT_WORD__3!(8i32) as isize);
let __v23: u64 = *__v24;
let __v22: u64 = __v23 | BIT_MASK__2!(8i32);
*__v24 = __v22;
}
if (*r#priv).flags & (16 | 32) != 0 {
let __v21: *mut u64 = ((*dev1).keybit.as_mut_ptr() as *mut u64).wrapping_offset(BIT_WORD__3!(277i32) as isize);
let __v20: u64 = *__v21;
let __v19: u64 = __v20 | BIT_MASK__2!(277i32);
*__v21 = __v19;
let __v18: *mut u64 = ((*dev1).keybit.as_mut_ptr() as *mut u64).wrapping_offset(BIT_WORD__3!(278i32) as isize);
let __v17: u64 = *__v18;
let __v16: u64 = __v17 | BIT_MASK__2!(278i32);
*__v18 = __v16;
}
if (*r#priv).flags & 64 != 0 {
let __v12: *mut u64 = ((*dev1).keybit.as_mut_ptr() as *mut u64).wrapping_offset(BIT_WORD__3!(256i32) as isize);
let __v11: u64 = *__v12;
let __v10: u64 = __v11 | BIT_MASK__2!(256i32);
*__v12 = __v10;
let __v9: *mut u64 = ((*dev1).keybit.as_mut_ptr() as *mut u64).wrapping_offset(BIT_WORD__3!(257i32) as isize);
let __v8: u64 = *__v9;
let __v7: u64 = __v8 | BIT_MASK__2!(257i32);
*__v9 = __v7;
let __v6: *mut u64 = ((*dev1).keybit.as_mut_ptr() as *mut u64).wrapping_offset(BIT_WORD__3!(258i32) as isize);
let __v5: u64 = *__v6;
let __v4: u64 = __v5 | BIT_MASK__2!(258i32);
*__v6 = __v4;
let __v3: *mut u64 = ((*dev1).keybit.as_mut_ptr() as *mut u64).wrapping_offset(BIT_WORD__3!(259i32) as isize);
let __v2: u64 = *__v3;
let __v1: u64 = __v2 | BIT_MASK__2!(259i32);
*__v3 = __v1;
} else {
if (*r#priv).flags & 512 != 0 {
set_bit(2i64, ((*dev1).propbit.as_mut_ptr() as *mut u64) as *mut u64);
clear_bit(273i64, ((*dev1).keybit.as_mut_ptr() as *mut u64) as *mut u64);
} else {
let __v15: *mut u64 = ((*dev1).keybit.as_mut_ptr() as *mut u64).wrapping_offset(BIT_WORD__3!(274i32) as isize);
let __v14: u64 = *__v15;
let __v13: u64 = __v14 | BIT_MASK__2!(274i32);
*__v15 = __v13;
}
}
if (*r#priv).flags & 2 != 0 {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_0 = input_allocate_device();
if (*__slate_slot_0).is_null() {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*(*psmouse).ps2dev.serio).dev) as *const device, c"alps: failed to allocate trackstick device\n".as_ptr() as *const u8);
*__slate_slot_43 = 12i32.wrapping_neg();
break '__join_0;
} else {
scnprintf((*r#priv).phys2.as_mut_ptr() as *mut u8, core::mem::size_of::<[u8; 32]>() as u64, c"%s/input1".as_ptr() as *const u8, (*(*psmouse).ps2dev.serio).phys.as_mut_ptr() as *mut u8);
(*(*__slate_slot_0)).phys = ((*r#priv).phys2.as_mut_ptr() as *mut u8) as *const u8;
// format of input device name is: "protocol vendor name"
// see function psmouse_switch_protocol() in psmouse-base.c
(*(*__slate_slot_0)).name = c"AlpsPS/2 ALPS DualPoint Stick".as_ptr() as *const u8;
(*(*__slate_slot_0)).id.bustype = 17;
(*(*__slate_slot_0)).id.vendor = 2;
(*(*__slate_slot_0)).id.product = 8;
(*(*__slate_slot_0)).id.version = (*r#priv).proto_version;
(*(*__slate_slot_0)).dev.parent = core::ptr::addr_of_mut!((*(*psmouse).ps2dev.serio).dev);
input_set_capability(*__slate_slot_0, 2u32, 0u32);
input_set_capability(*__slate_slot_0, 2u32, 1u32);
if (*r#priv).flags & 1024 != 0 {
input_set_capability(*__slate_slot_0, 3u32, 24u32);
input_set_abs_params(*__slate_slot_0, 24u32, 0i32, 127i32, 0i32, 0i32);
}
input_set_capability(*__slate_slot_0, 1u32, 272u32);
input_set_capability(*__slate_slot_0, 1u32, 273u32);
input_set_capability(*__slate_slot_0, 1u32, 274u32);
__set_bit!(0i32, (*(*__slate_slot_0)).propbit.as_mut_ptr() as *mut u64);
__set_bit!(5i32, (*(*__slate_slot_0)).propbit.as_mut_ptr() as *mut u64);
*__slate_slot_43 = input_register_device(*__slate_slot_0);
if *__slate_slot_43 != 0 {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*(*psmouse).ps2dev.serio).dev) as *const device, c"alps: failed to register trackstick device: %d\n".as_ptr() as *const u8, *__slate_slot_43);
input_free_device(*__slate_slot_0);
break '__join_0;
} else {
(*r#priv).dev2 = *__slate_slot_0;
}
}
}
(*r#priv).psmouse = psmouse;
loop {
__init_work(core::ptr::addr_of_mut!((*r#priv).dev3_register_work), 0i32);
(*core::ptr::addr_of_mut!((*r#priv).dev3_register_work)).data = *{
__slate_compound_46 = __SlateRecord_types_h_197 { counter: 4503599625273344i64 };
core::ptr::addr_of_mut!(__slate_compound_46)
};
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*r#priv).dev3_register_work)).entry));
(*core::ptr::addr_of_mut!((*r#priv).dev3_register_work)).func = Some(alps_register_bare_ps2_mouse);
if true {
if true {
break;
}
}
}
(*psmouse).protocol_handler = Some(alps_process_byte);
(*psmouse).poll = Some(alps_poll);
(*psmouse).disconnect = Some(alps_disconnect);
(*psmouse).reconnect = Some(alps_reconnect);
(*psmouse).pktsize = (if ((*r#priv).proto_version as i32) == 1024 { 8i32 } else { 6i32 }) as u8;
// We are having trouble resyncing ALPS touchpads so disable it for now
(*psmouse).resync_time = 0;
// Allow 2 invalid packets without resetting device
(*psmouse).resetafter = ((*psmouse).pktsize as i32).wrapping_mul(2) as u32;
return 0;
}
}
psmouse_reset(psmouse);
// Even though we did not allocate psmouse->private we do free
// it here.
kfree((*psmouse).private as *const ());
(*psmouse).private = core::ptr::null_mut::<()>();
return *__slate_slot_43;
}
}

static mut __key: lock_class_key = unsafe { core::mem::zeroed() };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.alps.alps_detect")]
extern "C" fn alps_detect(psmouse: *mut psmouse, set_properties: bool) -> i32 {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut error: i32 = alps_identify(psmouse, core::ptr::null_mut::<alps_data>());
if error != 0 {
return error;
}
// ALPS cs19 is a trackpoint-only device, and uses different
// protocol than DualPoint ones, so we return -EINVAL here and let
// trackpoint.c drive this device. If the trackpoint driver is not
// enabled, the device will fall back to a bare PS/2 mouse.
// If ps2_command() fails here, we depend on the immediately
// followed psmouse_reset() to reset the device to normal state.
if alps_is_cs19_trackpoint(psmouse) {
if false {
unsafe { _dev_printk(c"\x017".as_ptr() as *const u8, core::ptr::addr_of_mut!((*(*psmouse).ps2dev.serio).dev) as *const device, c"alps: ALPS CS19 trackpoint-only device detected, ignoring\n".as_ptr() as *const u8) };
}
return 22i32.wrapping_neg();
}
// Reset the device to make sure it is fully operational:
// on some laptops, like certain Dell Latitudes, we may
// fail to properly detect presence of trackstick if device
// has not been reset.
unsafe { psmouse_reset(psmouse) };
let mut __v0: *mut alps_data;
let __obj_size: u64 = size_mul(core::mem::size_of::<alps_data>() as u64, 1u64);
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
__v0 = __v1 as *mut alps_data;
let r#priv: *mut alps_data = __v0;
if r#priv.is_null() {
return 12i32.wrapping_neg();
}
error = alps_identify(psmouse, r#priv);
if error != 0 {
unsafe { kfree(r#priv as *const ()) };
return error;
}
if set_properties {
unsafe {
(*psmouse).vendor = c"ALPS".as_ptr() as *const u8;
}
unsafe {
(*psmouse).name = (if (*r#priv).flags & 2 != 0 { c"DualPoint TouchPad".as_ptr() as *mut u8 } else { c"GlidePoint".as_ptr() as *mut u8 }) as *const u8;
}
unsafe {
(*psmouse).model = (*r#priv).proto_version as u32;
}
} else {
// Destroy alps_data structure we allocated earlier since
// this was just a "trial run". Otherwise we'll keep it
// to be used by alps_init() which has to be called if
// we succeed and set_properties is true.
unsafe { kfree(r#priv as *const ()) };
unsafe {
(*psmouse).private = core::ptr::null_mut::<()>();
}
}
0
}

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_alps_command_mode_send_nibble_0: [u8; 1] = *b"\0";

static __slate_asm_string_alps_command_mode_send_nibble_1: [u8; 56] = *b"/home/takashi/c-corpus/linux/drivers/input/mouse/alps.c\0";

static __slate_asm_string_alps_process_packet_v4_1: [u8; 1] = *b"\0";

static __slate_asm_string_alps_process_packet_v4_2: [u8; 56] = *b"/home/takashi/c-corpus/linux/drivers/input/mouse/alps.c\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

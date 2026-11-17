
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
//! (c) 1999 Andreas Gal		<gal@cs.uni-magdeburg.de>
//! (c) 2000-2001 Vojtech Pavlik	<vojtech@ucw.cz>
//! (c) 2007-2009 Jiri Kosina
//!
//! HID debugging support
//! small_const_nbits(n) is true precisely when it is known at compile-time
//! that BITMAP_SIZE(n) is 1, i.e. 1 <= n <= BITS_PER_LONG. This allows
//! various bit/bitmap APIs to provide a fast inline implementation. Bitmaps
//! of size 0 are very rare, and a compile-time-known-size 0 is most likely
//! a sign of error. They will be handled correctly by the bit/bitmap APIs,
//! but using the out-of-line functions, so that the inline implementations
//! can unconditionally dereference the pointer(s).
//! __DECLARE_FLEX_ARRAY() - Declare a flexible array usable in a union
//!
//! @TYPE: The type of each flexible array element
//! @NAME: The name of the flexible array member
//!
//! In order to have a flexible array member in a union or alone in a
//! struct, it needs to be wrapped in an anonymous struct with at least 1
//! named member, but that member can be empty.
//! aligned_u64 should be used in defining kernel<->userspace ABIs to avoid
//! common 32/64-bit compat problems.
//! 64-bit values align to 4-byte boundaries on x86_32 (and possibly other
//! architectures) and to 8-byte boundaries on 64-bit architectures.  The new
//! aligned_64 type enforces 8-byte alignment so that structs containing
//! aligned_64 values have the same alignment on 32-bit and 64-bit architectures.
//! No conversions are necessary between 32-bit user-space and a 64-bit kernel.
//! These are the special 64-bit data types that are 8-byte aligned
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
//! Internal kernel timer frequency
//! Sign Flag
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
//! inhibit tail-call optimization
//! Call a function once. Similar to DO_ONCE(), but does not use jump label
//! patching via static keys.
//! warning conditions
//! # chars in a path name including nul
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
//! virt_to_page(kaddr) returns a valid pointer if and only if
//! virt_addr_valid(kaddr) returns true.
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
//! Compile time versions of __arch_hweightN()
//! ilog2 - log base 2 of 32-bit or a 64-bit unsigned value
//! @n: parameter
//!
//! constant-capable log of base 2 calculation
//! - this can be used to initialise global variables from constant data, hence
//! the massive ternary operator construction
//!
//! selects the appropriately-sized optimised version depending on sizeof(n)
//! writeable
//! Operation not permitted
//! Value too large for defined data type
//! These should never be seen by user programs.  To return one of ERESTART*
//! codes, signal_pending() MUST be set.  Note that ptrace can observe these
//! at syscall exit tracing, but they will never be left for the debugged user
//! process to see.
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
//! u64_to_user_ptr - cast a pointer passed as u64 from user space to void __user *
//! @x: The u64 value from user space, usually via IOCTL
//!
//! u64_to_user_ptr() simply casts a pointer passed as u64 from user space to void
//! __user * correctly. Using this lets us get rid of all the tiresome casts.
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
//! Instruction encoding provided for binutils backwards compatibility.
//! Defines VMX CPU feature bits
//! This is the default CPU features testing macro to use in code.
//!
//! It is for detection of features which need kernel infrastructure to be
//! used.  It may *not* directly test the CPU itself.  Use the cpu_has() family
//! if you want true runtime testing of CPU features, like in hypervisor code
//! where you are supporting a possible guest feature where host support for it
//! is not relevant.
//! We use the MSB for PREEMPT_NEED_RESCHED mostly because it is available.
//! Macros to retrieve the current execution context:
//!
//! in_nmi()		- We're in NMI context
//! in_hardirq()		- We're in hard IRQ context
//! in_serving_softirq()	- We're in softirq context
//! in_task()		- We're in task context
//! signal pending
//! New user namespace
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
//! Remapping spinlock architecture specific functions to the corresponding
//! queued spinlock functions.
//! A writer holds the lock
//! Define the various spin_lock methods.  Note we define these
//! regardless of whether CONFIG_SMP or CONFIG_PREEMPTION are set. The
//! various methods are defined as nops in the case they are not
//! required.
//! Used in tsk->__state:
//! Instruction opcode for WRMSRNS supported in binutils >= 2.40
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
//! These are specified by iBCS2
//! Epoll event masks
//! SC System Power Down
//! Need to take variadic arguments even though we don't use them, as dev_fmt()
//! may only just have been expanded and may result in multiple arguments.
//! number of env pointers
//! Generate SD flag indexes
//! Resource type
//! Desired design of maximum size and alignment (see RFC2553)
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
//! ld/ldx fields
//! Must be a compile-time constant, so implement it as a macro
//! Bits set in the VMA until the stack is in its final location
//! Define a list of socket types which can be the argument for
//! skc_to_*_sock() helpers. All these sockets should have
//! sock_common as the first argument in its memory layout.
//! BTF_KIND_INT is followed by a u32 and the following
//! is the 32 bits arrangement:
//! Currently, the max observed size in the kernel code is
//! JUMP_LABEL_NOP_SIZE/RELATIVEJUMP_SIZE, which are 5.
//! Raise it if needed.
//! should be >= TASK_COMM_LEN
//! in linux/sched.h
//! hlist_nulls_first_rcu - returns the first element of the hash list.
//! @head: the head of the list.
//! define the enumeration of all cgroup subsystems
//! To avoid confusing the compiler (and generating warnings) with code
//! that attempts to access what would be a 0-element array (i.e. sized
//! to a potentially empty array when CGROUP_SUBSYS_COUNT == 0), this
//! constant expression can be added.
//! kunmap_local - Unmap a page mapped via kmap_local_page().
//! @__addr: An address within the page mapped
//!
//! @__addr can be any address within the mapped page.  Commonly it is the
//! address return from kmap_local_page(), but it can also include offsets.
//!
//! Unmapping should be done in the reverse order of the mapping.  See
//! kmap_local_page() for details.
//! 31 pointers + header align the folio_batch structure to a power of two
//! modes for vm.memfd_noexec sysctl
//! These values are stored in task->ptrace_message
//! by ptrace_stop to describe the current syscall-stop.
//! SEIZE used, enable new behavior
//! Choose 31 as it makes rqspinlock_held cacheline-aligned.
//! bpf_type_flag contains a set of flags that are applicable to the values of
//! arg_type, ret_type and reg_type. For example, a pointer value may be null,
//! or a memory is read-only. We classify types into two categories: base types
//! and extended types. Extended types are base types combined with a type flag.
//!
//! Currently there are no more than 32 base types in arg_type, ret_type and
//! reg_types.
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
// DECLARE_FLEX_ARRAY() - Declare a flexible array usable in a union
//
// @TYPE: The type of each flexible array element
// @NAME: The name of the flexible array member
//
// In order to have a flexible array member in a union or alone in a
// struct, it needs to be wrapped in an anonymous struct with at least 1
// named member, but that member can be empty.
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
// Overflow Flag
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
// "fsrm" Fast Short Rep Mov
// "serialize" SERIALIZE instruction
// LFENCE always serializing / synchronizes RDTSC
// BUG word(s)
// "l1tf" CPU is affected by L1 Terminal Fault
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
// Atomic operations are already serializing on x86
// The following macros are non-atomic versions of their non-underscored
// counterparts.
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
// No such file or directory
// I/O error
// No such device or address
// Argument list too long
// Try again
// Out of memory
// Permission denied
// Bad address
// Device or resource busy
// No such device
// Invalid argument
// Text file busy
// No space left on device
// Math result not representable
// Operation not supported on transport endpoint
// Memory page has hardware error
// restart by calling sys_restart_syscall
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
// min_t - return minimum of two values, using the specified type
// @type: data type to use
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
// swap - swap values of @a and @b
// @a: first value
// @b: second value
// Use these carefully: no type checking, and uses the arguments
// multiple times. Use for obvious constants only.
// net/core/page_pool.c
//
// page_pool uses additional free bits within this value to store data, see the
// definition of PP_DMA_INDEX_MASK in mm.h
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
// signal notifications exist
// rescheduling necessary
// idle is polling for TIF_NEED_RESCHED
// re-arm the timer
// SCHED_ISO: reserved but not implemented yet
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
// folio_page - Return a page from a folio.
// @folio: The folio.
// @n: The page number to return.
//
// @n is relative to the start of the folio.  This function does not
// check that the page number lies within @folio; the caller is presumed
// to have a reference to the page.
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
// Currently stuck as a macro due to indirect forward reference to
// linux/mmzone.h's __section_mem_map_addr() definition:
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
// AL Compiz Scale (Expose)
// AC Stop
// AC Properties
// AC Undo
// AC Copy
// AC Open
// AC Paste
// AC Search
// AC Cut
// AL Integrated Help Center
// Menu (show menu)
// AL Calculator
// SC System Sleep
// System Wake Up
// AL Local Machine Browser
// AL Internet Browser
// AL Terminal Lock/Screensaver
// Display orientation for e.g. tablets
// AC Bookmarks
// AC Back
// AC Forward
// Media Select Telephone
// AL Consumer Control Configuration
// AC Home
// AC Refresh
// AC Exit
// AC New
// AC Redo/Repeat
// AC Desktop Show All Applications
// AC Close
// AC Print
// AL Checkbook/Finance
// AC Cancel
// Cycle between available video
// outputs (Monitor/LCD/TV-out/etc)
// AC Send
// AC Reply
// AC Forward Msg
// AC Save
// drive next video source
// drive previous video source
// brightness up, after max is min
// Set Auto Brightness: manual
// brightness control is off,
// rely on ambient
// display device to off state
// Wireless WAN (LTE, UMTS, GSM, etc.)
// Key that controls all radios
// Mute / unmute the microphone
// Five fingers on trackpad
// Four fingers on trackpad
// AL OEM Features/Tips/Tutorial
// Media Select Program Guide
// Media Select Home
// AC View Toggle
// HUTRR37: Aspect
// Media Select Computer
// Media Select TV
// Media Select Cable
// Media Select VCR
// VCR Plus
// Media Select Satellite
// Media Select CD
// Media Select Tape
// Media Select Tuner
// Media Select DVD
// AL Audio Browser
// AL Movie Browser
// Media Select Messages
// Channel Increment
// Channel Decrement
// Recall Last
// Media Select Video Phone
// Media Select Games
// AC Zoom In
// AC Zoom Out
// AC Zoom
// AL Word Processor
// AL Text Editor
// AL Spreadsheet
// AL Graphics Editor
// AL Presentation App
// AL Database App
// AL Newsreader
// AL Voicemail
// AL Contacts/Address Book
// AL Instant Messaging
// Turn display (LCD) on and off
// AL Spell Check
// AL Logoff
// Consumer - transport controls
// GenDesc - system context menu
// Consumer - transport control
// 10 channels up (10+)
// 10 channels down (10-)
// AL Image Browser
// Show/hide the notification center
// Answer incoming call
// Decline incoming call
// used by phones, remote controls,
// and other keypads
// Phone key A - HUT Telephony 0xb9
// WiFi Protected Setup key
// Request switch touchpad on or off
// Attendant call on or off
// Reading light on or off
// Ambient light sensor
// Display rotation lock
// AL Button Configuration
// AL Task/Project Manager
// AL Log/Journal/Timecard
// AL Control Panel
// AL Select Task/Application
// AL Screen Saver
// Listening Voice Command
// AL Context-aware desktop assistant
// AC Next Keyboard Layout Select
// Show/hide emoji picker (HUTRR101)
// Start or Stop Voice Dictation Session (HUTRR99)
// Enables programmatic access to camera devices. (HUTRR72)
// Disables programmatic access to camera devices. (HUTRR72)
// Toggles the current state of the camera access control. (HUTRR72)
// Toggles the system bound accessibility UI/command (HUTRR116)
// Toggles the system-wide "Do Not Disturb" control (HUTRR94)
// Set Brightness to Minimum
// Set Brightness to Maximum
// AL Action on Selection (HUTRR119)
// AL Contextual Insertion (HUTRR119)
// AL Contextual Query (HUTRR119)
// Diagonal movement keys
// Show Device's Root Menu
// Show Top Menu of the Media (e.g. DVD)
// Toggle Audio Description: refers to an audio service that helps blind and
// visually impaired consumers understand the action in a program. Note: in
// some countries this is referred to as "Video Description".
// Video on Demand
// Control a data application associated with the currently viewed channel,
// e.g. teletext or data broadcast application (MHEG, MHP, HbbTV, etc.)
// Electronic privacy screen control
// Select an area of screen to be copied
// Move the focus to the next or previous user controllable element within a UI container
// Toggle Autopilot engagement
// Shortcut Keys
// Some keyboards have keys which do not have a defined meaning, these keys
// are intended to be programmed / bound to macros by the user. For most
// keyboards with these macro-keys the key-sequence to inject, or action to
// take, is all handled by software on the host side. So from the kernel's
// point of view these are just normal keys.
//
// The KEY_MACRO# codes below are intended for such keys, which may be labeled
// e.g. G1-G18, or S1 - S30. The KEY_MACRO# codes MUST NOT be used for keys
// where the marking on the key does indicate a defined meaning / purpose.
//
// The KEY_MACRO# codes MUST also NOT be used as fallback for when no existing
// KEY_FOO define matches the marking / purpose. In this case a new KEY_FOO
// define MUST be added.
// Some keyboards with the macro-keys described above have some extra keys
// for controlling the host-side software responsible for the macro handling:
// -A macro recording start/stop key. Note that not all keyboards which emit
//  KEY_MACRO_RECORD_START will also emit KEY_MACRO_RECORD_STOP if
//  KEY_MACRO_RECORD_STOP is not advertised, then KEY_MACRO_RECORD_START
//  should be interpreted as a recording start/stop toggle;
// -Keys for switching between different macro (pre)sets, either a key for
//  cycling through the configured presets or keys to directly select a preset.
// Some keyboards have a buildin LCD panel where the contents are controlled
// by the host. Often these have a number of keys directly below the LCD
// intended for controlling a menu shown on the LCD. These keys often don't
// have any labeling so we just name them KEY_KBD_LCD_MENU#
// MT slot being modified
// Major axis of touching ellipse
// Minor axis (omit if circular)
// Major axis of approaching ellipse
// Minor axis (omit if circular)
// Ellipse orientation
// Center X touch position
// Center Y touch position
// Type of touching device
// Group a set of packets as a blob
// Unique ID of initiated contact
// Pressure on contact area
// Contact hover distance
// Center X tool position
// Center Y tool position
// set = lid shut
// set = tablet mode
// set = inserted
// rfkill master switch, type "any"
// set = radio enabled
// set = inserted
// set = plugged into dock
// set = inserted
// set = mechanical switch set
// set = inserted
// set = lens covered
// set = keypad slide out
// set = front proximity sensor active
// set = rotate locked/disabled
// set = inserted
// set = device disabled
// set = pen inserted
// set = cover closed
// Generate SD flag bits
// Resource extended types
// No address assigned yet
// Implementation specific max size
// double word (64-bit)
// 1 << 13
// insn[0].src_reg:  BPF_PSEUDO_FUNC
// insn[0].imm:      insn offset to the func
// insn[1].imm:      0
// insn[0].off:      0
// insn[1].off:      0
// ldimm64 rewrite:  address of the function
// verifier type:    PTR_TO_FUNC.
// The description below is an attempt at providing documentation to eBPF
// developers about the multiple available eBPF helper functions. It can be
// parsed and used to produce a manual page. The workflow is the following,
// and requires the rst2man utility:
//
//     $ ./scripts/bpf_doc.py  *             --filename include/uapi/linux/bpf.h > /tmp/bpf-helpers.rst
//     $ rst2man /tmp/bpf-helpers.rst > /tmp/bpf-helpers.7
//     $ man /tmp/bpf-helpers.7
//
// Note that in order to produce this external documentation, some RST
// formatting is used in the descriptions to get "bold" and "italics" in
// manual pages. Also note that the few trailing white spaces are
// intentional, removing them would break paragraphs for rst2man.
//
// Start of BPF helper function descriptions:
//
// void *bpf_map_lookup_elem(struct bpf_map *map, const void *key)
// 	Description
// 		Perform a lookup in *map* for an entry associated to *key*.
// 	Return
// 		Map value associated to *key*, or **NULL** if no entry was
// 		found.
//
// long bpf_map_update_elem(struct bpf_map *map, const void *key, const void *value, u64 flags)
// 	Description
// 		Add or update the value of the entry associated to *key* in
// 		*map* with *value*. *flags* is one of:
//
// 		**BPF_NOEXIST**
// 			The entry for *key* must not exist in the map.
// 		**BPF_EXIST**
// 			The entry for *key* must already exist in the map.
// 		**BPF_ANY**
// 			No condition on the existence of the entry for *key*.
//
// 		Flag value **BPF_NOEXIST** cannot be used for maps of types
// 		**BPF_MAP_TYPE_ARRAY** or **BPF_MAP_TYPE_PERCPU_ARRAY**  (all
// 		elements always exist), the helper would return an error.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_map_delete_elem(struct bpf_map *map, const void *key)
// 	Description
// 		Delete entry with *key* from *map*.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_probe_read(void *dst, u32 size, const void *unsafe_ptr)
// 	Description
// 		For tracing programs, safely attempt to read *size* bytes from
// 		kernel space address *unsafe_ptr* and store the data in *dst*.
//
// 		Generally, use **bpf_probe_read_user**\ () or
// 		**bpf_probe_read_kernel**\ () instead.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// u64 bpf_ktime_get_ns(void)
// 	Description
// 		Return the time elapsed since system boot, in nanoseconds.
// 		Does not include time the system was suspended.
// 		See: **clock_gettime**\ (**CLOCK_MONOTONIC**)
// 	Return
// 		Current *ktime*.
//
// long bpf_trace_printk(const char *fmt, u32 fmt_size, ...)
// 	Description
// 		This helper is a "printk()-like" facility for debugging. It
// 		prints a message defined by format *fmt* (of size *fmt_size*)
// 		to file *\/sys/kernel/tracing/trace* from TraceFS, if
// 		available. It can take up to three additional **u64**
// 		arguments (as an eBPF helpers, the total number of arguments is
// 		limited to five).
//
// 		Each time the helper is called, it appends a line to the trace.
// 		Lines are discarded while *\/sys/kernel/tracing/trace* is
// 		open, use *\/sys/kernel/tracing/trace_pipe* to avoid this.
// 		The format of the trace is customizable, and the exact output
// 		one will get depends on the options set in
// 		*\/sys/kernel/tracing/trace_options* (see also the
// 		*README* file under the same directory). However, it usually
// 		defaults to something like:
//
// 		::
//
// 			telnet-470   [001] .N.. 419421.045894: 0x00000001: <formatted msg>
//
// 		In the above:
//
// 			* ``telnet`` is the name of the current task.
// 			* ``470`` is the PID of the current task.
// 			* ``001`` is the CPU number on which the task is
// 			  running.
// 			* In ``.N..``, each character refers to a set of
// 			  options (whether irqs are enabled, scheduling
// 			  options, whether hard/softirqs are running, level of
// 			  preempt_disabled respectively). **N** means that
// 			  **TIF_NEED_RESCHED** and **PREEMPT_NEED_RESCHED**
// 			  are set.
// 			* ``419421.045894`` is a timestamp.
// 			* ``0x00000001`` is a fake value used by BPF for the
// 			  instruction pointer register.
// 			* ``<formatted msg>`` is the message formatted with
// 			  *fmt*.
//
// 		The conversion specifiers supported by *fmt* are similar, but
// 		more limited than for printk(). They are **%d**, **%i**,
// 		**%u**, **%x**, **%ld**, **%li**, **%lu**, **%lx**, **%lld**,
// 		**%lli**, **%llu**, **%llx**, **%p**, **%s**. No modifier (size
// 		of field, padding with zeroes, etc.) is available, and the
// 		helper will return **-EINVAL** (but print nothing) if it
// 		encounters an unknown specifier.
//
// 		Also, note that **bpf_trace_printk**\ () is slow, and should
// 		only be used for debugging purposes. For this reason, a notice
// 		block (spanning several lines) is printed to kernel logs and
// 		states that the helper should not be used "for production use"
// 		the first time this helper is used (or more precisely, when
// 		**trace_printk**\ () buffers are allocated). For passing values
// 		to user space, perf events should be preferred.
// 	Return
// 		The number of bytes written to the buffer, or a negative error
// 		in case of failure.
//
// u32 bpf_get_prandom_u32(void)
// 	Description
// 		Get a pseudo-random number.
//
// 		From a security point of view, this helper uses its own
// 		pseudo-random internal state, and cannot be used to infer the
// 		seed of other random functions in the kernel. However, it is
// 		essential to note that the generator used by the helper is not
// 		cryptographically secure.
// 	Return
// 		A random 32-bit unsigned value.
//
// u32 bpf_get_smp_processor_id(void)
// 	Description
// 		Get the SMP (symmetric multiprocessing) processor id. Note that
// 		all programs run with migration disabled, which means that the
// 		SMP processor id is stable during all the execution of the
// 		program.
// 	Return
// 		The SMP id of the processor running the program.
// 	Attributes
// 		__bpf_fastcall
//
// long bpf_skb_store_bytes(struct sk_buff *skb, u32 offset, const void *from, u32 len, u64 flags)
// 	Description
// 		Store *len* bytes from address *from* into the packet
// 		associated to *skb*, at *offset*. The *flags* are a combination
// 		of the following values:
//
// 		**BPF_F_RECOMPUTE_CSUM**
// 			Automatically update *skb*\ **->csum** after storing the
// 			bytes.
// 		**BPF_F_INVALIDATE_HASH**
// 			Set *skb*\ **->hash**, *skb*\ **->swhash** and *skb* * 			**->l4hash** to 0.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_l3_csum_replace(struct sk_buff *skb, u32 offset, u64 from, u64 to, u64 size)
// 	Description
// 		Recompute the layer 3 (e.g. IP) checksum for the packet
// 		associated to *skb*. Computation is incremental, so the helper
// 		must know the former value of the header field that was
// 		modified (*from*), the new value of this field (*to*), and the
// 		number of bytes (2 or 4) for this field, stored in *size*.
// 		Alternatively, it is possible to store the difference between
// 		the previous and the new values of the header field in *to*, by
// 		setting *from* and *size* to 0. For both methods, *offset*
// 		indicates the location of the IP checksum within the packet.
//
// 		This helper works in combination with **bpf_csum_diff**\ (),
// 		which does not update the checksum in-place, but offers more
// 		flexibility and can handle sizes larger than 2 or 4 for the
// 		checksum to update.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_l4_csum_replace(struct sk_buff *skb, u32 offset, u64 from, u64 to, u64 flags)
// 	Description
// 		Recompute the layer 4 (e.g. TCP, UDP or ICMP) checksum for the
// 		packet associated to *skb*. Computation is incremental, so the
// 		helper must know the former value of the header field that was
// 		modified (*from*), the new value of this field (*to*), and the
// 		number of bytes (2 or 4) for this field, stored on the lowest
// 		four bits of *flags*. Alternatively, it is possible to store
// 		the difference between the previous and the new values of the
// 		header field in *to*, by setting *from* and the four lowest
// 		bits of *flags* to 0. For both methods, *offset* indicates the
// 		location of the IP checksum within the packet. In addition to
// 		the size of the field, *flags* can be added (bitwise OR) actual
// 		flags. With **BPF_F_MARK_MANGLED_0**, a null checksum is left
// 		untouched (unless **BPF_F_MARK_ENFORCE** is added as well), and
// 		for updates resulting in a null checksum the value is set to
// 		**CSUM_MANGLED_0** instead. Flag **BPF_F_PSEUDO_HDR** indicates
// 		that the modified header field is part of the pseudo-header.
// 		Flag **BPF_F_IPV6** should be set for IPv6 packets.
//
// 		This helper works in combination with **bpf_csum_diff**\ (),
// 		which does not update the checksum in-place, but offers more
// 		flexibility and can handle sizes larger than 2 or 4 for the
// 		checksum to update.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_tail_call(void *ctx, struct bpf_map *prog_array_map, u32 index)
// 	Description
// 		This special helper is used to trigger a "tail call", or in
// 		other words, to jump into another eBPF program. The same stack
// 		frame is used (but values on stack and in registers for the
// 		caller are not accessible to the callee). This mechanism allows
// 		for program chaining, either for raising the maximum number of
// 		available eBPF instructions, or to execute given programs in
// 		conditional blocks. For security reasons, there is an upper
// 		limit to the number of successive tail calls that can be
// 		performed.
//
// 		Upon call of this helper, the program attempts to jump into a
// 		program referenced at index *index* in *prog_array_map*, a
// 		special map of type **BPF_MAP_TYPE_PROG_ARRAY**, and passes
// 		*ctx*, a pointer to the context.
//
// 		If the call succeeds, the kernel immediately runs the first
// 		instruction of the new program. This is not a function call,
// 		and it never returns to the previous program. If the call
// 		fails, then the helper has no effect, and the caller continues
// 		to run its subsequent instructions. A call can fail if the
// 		destination program for the jump does not exist (i.e. *index*
// 		is superior to the number of entries in *prog_array_map*), or
// 		if the maximum number of tail calls has been reached for this
// 		chain of programs. This limit is defined in the kernel by the
// 		macro **MAX_TAIL_CALL_CNT** (not accessible to user space),
// 		which is currently set to 33.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_clone_redirect(struct sk_buff *skb, u32 ifindex, u64 flags)
// 	Description
// 		Clone and redirect the packet associated to *skb* to another
// 		net device of index *ifindex*. Both ingress and egress
// 		interfaces can be used for redirection. The **BPF_F_INGRESS**
// 		value in *flags* is used to make the distinction (ingress path
// 		is selected if the flag is present, egress path otherwise).
// 		This is the only flag supported for now.
//
// 		In comparison with **bpf_redirect**\ () helper,
// 		**bpf_clone_redirect**\ () has the associated cost of
// 		duplicating the packet buffer, but this can be executed out of
// 		the eBPF program. Conversely, **bpf_redirect**\ () is more
// 		efficient, but it is handled through an action code where the
// 		redirection happens only after the eBPF program has returned.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure. Positive
// 		error indicates a potential drop or congestion in the target
// 		device. The particular positive error codes are not defined.
//
// u64 bpf_get_current_pid_tgid(void)
// 	Description
// 		Get the current pid and tgid.
// 	Return
// 		A 64-bit integer containing the current tgid and pid, and
// 		created as such:
// 		*current_task*\ **->tgid << 32 \|**
// 		*current_task*\ **->pid**.
//
// u64 bpf_get_current_uid_gid(void)
// 	Description
// 		Get the current uid and gid.
// 	Return
// 		A 64-bit integer containing the current GID and UID, and
// 		created as such: *current_gid* **<< 32 \|** *current_uid*.
//
// long bpf_get_current_comm(void *buf, u32 size_of_buf)
// 	Description
// 		Copy the **comm** attribute of the current task into *buf* of
// 		*size_of_buf*. The **comm** attribute contains the name of
// 		the executable (excluding the path) for the current task. The
// 		*size_of_buf* must be strictly positive. On success, the
// 		helper makes sure that the *buf* is NUL-terminated. On failure,
// 		it is filled with zeroes.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// u32 bpf_get_cgroup_classid(struct sk_buff *skb)
// 	Description
// 		Retrieve the classid for the current task, i.e. for the net_cls
// 		cgroup to which *skb* belongs.
//
// 		This helper can be used on TC egress path, but not on ingress.
//
// 		The net_cls cgroup provides an interface to tag network packets
// 		based on a user-provided identifier for all traffic coming from
// 		the tasks belonging to the related cgroup. See also the related
// 		kernel documentation, available from the Linux sources in file
// 		*Documentation/admin-guide/cgroup-v1/net_cls.rst*.
//
// 		The Linux kernel has two versions for cgroups: there are
// 		cgroups v1 and cgroups v2. Both are available to users, who can
// 		use a mixture of them, but note that the net_cls cgroup is for
// 		cgroup v1 only. This makes it incompatible with BPF programs
// 		run on cgroups, which is a cgroup-v2-only feature (a socket can
// 		only hold data for one version of cgroups at a time).
//
// 		This helper is only available is the kernel was compiled with
// 		the **CONFIG_CGROUP_NET_CLASSID** configuration option set to
// 		"**y**" or to "**m**".
// 	Return
// 		The classid, or 0 for the default unconfigured classid.
//
// long bpf_skb_vlan_push(struct sk_buff *skb, __be16 vlan_proto, u16 vlan_tci)
// 	Description
// 		Push a *vlan_tci* (VLAN tag control information) of protocol
// 		*vlan_proto* to the packet associated to *skb*, then update
// 		the checksum. Note that if *vlan_proto* is different from
// 		**ETH_P_8021Q** and **ETH_P_8021AD**, it is considered to
// 		be **ETH_P_8021Q**.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_vlan_pop(struct sk_buff *skb)
// 	Description
// 		Pop a VLAN header from the packet associated to *skb*.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_get_tunnel_key(struct sk_buff *skb, struct bpf_tunnel_key *key, u32 size, u64 flags)
// 	Description
// 		Get tunnel metadata. This helper takes a pointer *key* to an
// 		empty **struct bpf_tunnel_key** of **size**, that will be
// 		filled with tunnel metadata for the packet associated to *skb*.
// 		The *flags* can be set to **BPF_F_TUNINFO_IPV6**, which
// 		indicates that the tunnel is based on IPv6 protocol instead of
// 		IPv4.
//
// 		The **struct bpf_tunnel_key** is an object that generalizes the
// 		principal parameters used by various tunneling protocols into a
// 		single struct. This way, it can be used to easily make a
// 		decision based on the contents of the encapsulation header,
// 		"summarized" in this struct. In particular, it holds the IP
// 		address of the remote end (IPv4 or IPv6, depending on the case)
// 		in *key*\ **->remote_ipv4** or *key*\ **->remote_ipv6**. Also,
// 		this struct exposes the *key*\ **->tunnel_id**, which is
// 		generally mapped to a VNI (Virtual Network Identifier), making
// 		it programmable together with the **bpf_skb_set_tunnel_key** * 		() helper.
//
// 		Let's imagine that the following code is part of a program
// 		attached to the TC ingress interface, on one end of a GRE
// 		tunnel, and is supposed to filter out all messages coming from
// 		remote ends with IPv4 address other than 10.0.0.1:
//
// 		::
//
// 			int ret;
// 			struct bpf_tunnel_key key = {};
//
// 			ret = bpf_skb_get_tunnel_key(skb, &key, sizeof(key), 0);
// 			if (ret < 0)
// 				return TC_ACT_SHOT;	// drop packet
//
// 			if (key.remote_ipv4 != 0x0a000001)
// 				return TC_ACT_SHOT;	// drop packet
//
// 			return TC_ACT_OK;		// accept packet
//
// 		This interface can also be used with all encapsulation devices
// 		that can operate in "collect metadata" mode: instead of having
// 		one network device per specific configuration, the "collect
// 		metadata" mode only requires a single device where the
// 		configuration can be extracted from this helper.
//
// 		This can be used together with various tunnels such as VXLan,
// 		Geneve, GRE or IP in IP (IPIP).
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_set_tunnel_key(struct sk_buff *skb, struct bpf_tunnel_key *key, u32 size, u64 flags)
// 	Description
// 		Populate tunnel metadata for packet associated to *skb.* The
// 		tunnel metadata is set to the contents of *key*, of *size*. The
// 		*flags* can be set to a combination of the following values:
//
// 		**BPF_F_TUNINFO_IPV6**
// 			Indicate that the tunnel is based on IPv6 protocol
// 			instead of IPv4.
// 		**BPF_F_ZERO_CSUM_TX**
// 			For IPv4 packets, add a flag to tunnel metadata
// 			indicating that checksum computation should be skipped
// 			and checksum set to zeroes.
// 		**BPF_F_DONT_FRAGMENT**
// 			Add a flag to tunnel metadata indicating that the
// 			packet should not be fragmented.
// 		**BPF_F_SEQ_NUMBER**
// 			Add a flag to tunnel metadata indicating that a
// 			sequence number should be added to tunnel header before
// 			sending the packet. This flag was added for GRE
// 			encapsulation, but might be used with other protocols
// 			as well in the future.
// 		**BPF_F_NO_TUNNEL_KEY**
// 			Add a flag to tunnel metadata indicating that no tunnel
// 			key should be set in the resulting tunnel header.
//
// 		Here is a typical usage on the transmit path:
//
// 		::
//
// 			struct bpf_tunnel_key key;
// 			     populate key ...
// 			bpf_skb_set_tunnel_key(skb, &key, sizeof(key), 0);
// 			bpf_clone_redirect(skb, vxlan_dev_ifindex, 0);
//
// 		See also the description of the **bpf_skb_get_tunnel_key**\ ()
// 		helper for additional information.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// u64 bpf_perf_event_read(struct bpf_map *map, u64 flags)
// 	Description
// 		Read the value of a perf event counter. This helper relies on a
// 		*map* of type **BPF_MAP_TYPE_PERF_EVENT_ARRAY**. The nature of
// 		the perf event counter is selected when *map* is updated with
// 		perf event file descriptors. The *map* is an array whose size
// 		is the number of available CPUs, and each cell contains a value
// 		relative to one CPU. The value to retrieve is indicated by
// 		*flags*, that contains the index of the CPU to look up, masked
// 		with **BPF_F_INDEX_MASK**. Alternatively, *flags* can be set to
// 		**BPF_F_CURRENT_CPU** to indicate that the value for the
// 		current CPU should be retrieved.
//
// 		Note that before Linux 4.13, only hardware perf event can be
// 		retrieved.
//
// 		Also, be aware that the newer helper
// 		**bpf_perf_event_read_value**\ () is recommended over
// 		**bpf_perf_event_read**\ () in general. The latter has some ABI
// 		quirks where error and counter value are used as a return code
// 		(which is wrong to do since ranges may overlap). This issue is
// 		fixed with **bpf_perf_event_read_value**\ (), which at the same
// 		time provides more features over the **bpf_perf_event_read** * 		() interface. Please refer to the description of
// 		**bpf_perf_event_read_value**\ () for details.
// 	Return
// 		The value of the perf event counter read from the map, or a
// 		negative error code in case of failure.
//
// long bpf_redirect(u32 ifindex, u64 flags)
// 	Description
// 		Redirect the packet to another net device of index *ifindex*.
// 		This helper is somewhat similar to **bpf_clone_redirect** * 		(), except that the packet is not cloned, which provides
// 		increased performance.
//
// 		Except for XDP, both ingress and egress interfaces can be used
// 		for redirection. The **BPF_F_INGRESS** value in *flags* is used
// 		to make the distinction (ingress path is selected if the flag
// 		is present, egress path otherwise). Currently, XDP only
// 		supports redirection to the egress interface, and accepts no
// 		flag at all.
//
// 		The same effect can also be attained with the more generic
// 		**bpf_redirect_map**\ (), which uses a BPF map to store the
// 		redirect target instead of providing it directly to the helper.
// 	Return
// 		For XDP, the helper returns **XDP_REDIRECT** on success or
// 		**XDP_ABORTED** on error. For other program types, the values
// 		are **TC_ACT_REDIRECT** on success or **TC_ACT_SHOT** on
// 		error.
//
// u32 bpf_get_route_realm(struct sk_buff *skb)
// 	Description
// 		Retrieve the realm or the route, that is to say the
// 		**tclassid** field of the destination for the *skb*. The
// 		identifier retrieved is a user-provided tag, similar to the
// 		one used with the net_cls cgroup (see description for
// 		**bpf_get_cgroup_classid**\ () helper), but here this tag is
// 		held by a route (a destination entry), not by a task.
//
// 		Retrieving this identifier works with the clsact TC egress hook
// 		(see also **tc-bpf(8)**), or alternatively on conventional
// 		classful egress qdiscs, but not on TC ingress path. In case of
// 		clsact TC egress hook, this has the advantage that, internally,
// 		the destination entry has not been dropped yet in the transmit
// 		path. Therefore, the destination entry does not need to be
// 		artificially held via **netif_keep_dst**\ () for a classful
// 		qdisc until the *skb* is freed.
//
// 		This helper is available only if the kernel was compiled with
// 		**CONFIG_IP_ROUTE_CLASSID** configuration option.
// 	Return
// 		The realm of the route for the packet associated to *skb*, or 0
// 		if none was found.
//
// long bpf_perf_event_output(void *ctx, struct bpf_map *map, u64 flags, void *data, u64 size)
// 	Description
// 		Write raw *data* blob into a special BPF perf event held by
// 		*map* of type **BPF_MAP_TYPE_PERF_EVENT_ARRAY**. This perf
// 		event must have the following attributes: **PERF_SAMPLE_RAW**
// 		as **sample_type**, **PERF_TYPE_SOFTWARE** as **type**, and
// 		**PERF_COUNT_SW_BPF_OUTPUT** as **config**.
//
// 		The *flags* are used to indicate the index in *map* for which
// 		the value must be put, masked with **BPF_F_INDEX_MASK**.
// 		Alternatively, *flags* can be set to **BPF_F_CURRENT_CPU**
// 		to indicate that the index of the current CPU core should be
// 		used.
//
// 		The value to write, of *size*, is passed through eBPF stack and
// 		pointed by *data*.
//
// 		The context of the program *ctx* needs also be passed to the
// 		helper.
//
// 		On user space, a program willing to read the values needs to
// 		call **perf_event_open**\ () on the perf event (either for
// 		one or for all CPUs) and to store the file descriptor into the
// 		*map*. This must be done before the eBPF program can send data
// 		into it. An example is available in file
// 		*samples/bpf/trace_output_user.c* in the Linux kernel source
// 		tree (the eBPF program counterpart is in
// 		*samples/bpf/trace_output.bpf.c*).
//
// 		**bpf_perf_event_output**\ () achieves better performance
// 		than **bpf_trace_printk**\ () for sharing data with user
// 		space, and is much better suitable for streaming data from eBPF
// 		programs.
//
// 		Note that this helper is not restricted to tracing use cases
// 		and can be used with programs attached to TC or XDP as well,
// 		where it allows for passing data to user space listeners. Data
// 		can be:
//
// 		* Only custom structs,
// 		* Only the packet payload, or
// 		* A combination of both.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_load_bytes(const void *skb, u32 offset, void *to, u32 len)
// 	Description
// 		This helper was provided as an easy way to load data from a
// 		packet. It can be used to load *len* bytes from *offset* from
// 		the packet associated to *skb*, into the buffer pointed by
// 		*to*.
//
// 		Since Linux 4.7, usage of this helper has mostly been replaced
// 		by "direct packet access", enabling packet data to be
// 		manipulated with *skb*\ **->data** and *skb*\ **->data_end**
// 		pointing respectively to the first byte of packet data and to
// 		the byte after the last byte of packet data. However, it
// 		remains useful if one wishes to read large quantities of data
// 		at once from a packet into the eBPF stack.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_get_stackid(void *ctx, struct bpf_map *map, u64 flags)
// 	Description
// 		Walk a user or a kernel stack and return its id. To achieve
// 		this, the helper needs *ctx*, which is a pointer to the context
// 		on which the tracing program is executed, and a pointer to a
// 		*map* of type **BPF_MAP_TYPE_STACK_TRACE**.
//
// 		The last argument, *flags*, holds the number of stack frames to
// 		skip (from 0 to 255), masked with
// 		**BPF_F_SKIP_FIELD_MASK**. The next bits can be used to set
// 		a combination of the following flags:
//
// 		**BPF_F_USER_STACK**
// 			Collect a user space stack instead of a kernel stack.
// 		**BPF_F_FAST_STACK_CMP**
// 			Compare stacks by hash only.
// 		**BPF_F_REUSE_STACKID**
// 			If two different stacks hash into the same *stackid*,
// 			discard the old one.
//
// 		The stack id retrieved is a 32 bit long integer handle which
// 		can be further combined with other data (including other stack
// 		ids) and used as a key into maps. This can be useful for
// 		generating a variety of graphs (such as flame graphs or off-cpu
// 		graphs).
//
// 		For walking a stack, this helper is an improvement over
// 		**bpf_probe_read**\ (), which can be used with unrolled loops
// 		but is not efficient and consumes a lot of eBPF instructions.
// 		Instead, **bpf_get_stackid**\ () can collect up to
// 		**PERF_MAX_STACK_DEPTH** both kernel and user frames. Note that
// 		this limit can be controlled with the **sysctl** program, and
// 		that it should be manually increased in order to profile long
// 		user stacks (such as stacks for Java programs). To do so, use:
//
// 		::
//
// 			# sysctl kernel.perf_event_max_stack=<new value>
// 	Return
// 		The positive or null stack id on success, or a negative error
// 		in case of failure.
//
// s64 bpf_csum_diff(__be32 *from, u32 from_size, __be32 *to, u32 to_size, __wsum seed)
// 	Description
// 		Compute a checksum difference, from the raw buffer pointed by
// 		*from*, of length *from_size* (that must be a multiple of 4),
// 		towards the raw buffer pointed by *to*, of size *to_size*
// 		(same remark). An optional *seed* can be added to the value
// 		(this can be cascaded, the seed may come from a previous call
// 		to the helper).
//
// 		This is flexible enough to be used in several ways:
//
// 		* With *from_size* == 0, *to_size* > 0 and *seed* set to
// 		  checksum, it can be used when pushing new data.
// 		* With *from_size* > 0, *to_size* == 0 and *seed* set to
// 		  checksum, it can be used when removing data from a packet.
// 		* With *from_size* > 0, *to_size* > 0 and *seed* set to 0, it
// 		  can be used to compute a diff. Note that *from_size* and
// 		  *to_size* do not need to be equal.
//
// 		This helper can be used in combination with
// 		**bpf_l3_csum_replace**\ () and **bpf_l4_csum_replace**\ (), to
// 		which one can feed in the difference computed with
// 		**bpf_csum_diff**\ ().
// 	Return
// 		The checksum result, or a negative error code in case of
// 		failure.
//
// long bpf_skb_get_tunnel_opt(struct sk_buff *skb, void *opt, u32 size)
// 	Description
// 		Retrieve tunnel options metadata for the packet associated to
// 		*skb*, and store the raw tunnel option data to the buffer *opt*
// 		of *size*.
//
// 		This helper can be used with encapsulation devices that can
// 		operate in "collect metadata" mode (please refer to the related
// 		note in the description of **bpf_skb_get_tunnel_key**\ () for
// 		more details). A particular example where this can be used is
// 		in combination with the Geneve encapsulation protocol, where it
// 		allows for pushing (with **bpf_skb_get_tunnel_opt**\ () helper)
// 		and retrieving arbitrary TLVs (Type-Length-Value headers) from
// 		the eBPF program. This allows for full customization of these
// 		headers.
// 	Return
// 		The size of the option data retrieved.
//
// long bpf_skb_set_tunnel_opt(struct sk_buff *skb, void *opt, u32 size)
// 	Description
// 		Set tunnel options metadata for the packet associated to *skb*
// 		to the option data contained in the raw buffer *opt* of *size*.
//
// 		See also the description of the **bpf_skb_get_tunnel_opt**\ ()
// 		helper for additional information.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_change_proto(struct sk_buff *skb, __be16 proto, u64 flags)
// 	Description
// 		Change the protocol of the *skb* to *proto*. Currently
// 		supported are transition from IPv4 to IPv6, and from IPv6 to
// 		IPv4. The helper takes care of the groundwork for the
// 		transition, including resizing the socket buffer. The eBPF
// 		program is expected to fill the new headers, if any, via
// 		**skb_store_bytes**\ () and to recompute the checksums with
// 		**bpf_l3_csum_replace**\ () and **bpf_l4_csum_replace** * 		(). The main case for this helper is to perform NAT64
// 		operations out of an eBPF program.
//
// 		Internally, the GSO type is marked as dodgy so that headers are
// 		checked and segments are recalculated by the GSO/GRO engine.
// 		The size for GSO target is adapted as well.
//
// 		All values for *flags* are reserved for future usage, and must
// 		be left at zero.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_change_type(struct sk_buff *skb, u32 type)
// 	Description
// 		Change the packet type for the packet associated to *skb*. This
// 		comes down to setting *skb*\ **->pkt_type** to *type*, except
// 		the eBPF program does not have a write access to *skb* * 		**->pkt_type** beside this helper. Using a helper here allows
// 		for graceful handling of errors.
//
// 		The major use case is to change incoming *skb*s to
// 		**PACKET_HOST** in a programmatic way instead of having to
// 		recirculate via **redirect**\ (..., **BPF_F_INGRESS**), for
// 		example.
//
// 		Note that *type* only allows certain values. At this time, they
// 		are:
//
// 		**PACKET_HOST**
// 			Packet is for us.
// 		**PACKET_BROADCAST**
// 			Send packet to all.
// 		**PACKET_MULTICAST**
// 			Send packet to group.
// 		**PACKET_OTHERHOST**
// 			Send packet to someone else.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_under_cgroup(struct sk_buff *skb, struct bpf_map *map, u32 index)
// 	Description
// 		Check whether *skb* is a descendant of the cgroup2 held by
// 		*map* of type **BPF_MAP_TYPE_CGROUP_ARRAY**, at *index*.
// 	Return
// 		The return value depends on the result of the test, and can be:
//
// 		* 0, if the *skb* failed the cgroup2 descendant test.
// 		* 1, if the *skb* succeeded the cgroup2 descendant test.
// 		* A negative error code, if an error occurred.
//
// u32 bpf_get_hash_recalc(struct sk_buff *skb)
// 	Description
// 		Retrieve the hash of the packet, *skb*\ **->hash**. If it is
// 		not set, in particular if the hash was cleared due to mangling,
// 		recompute this hash. Later accesses to the hash can be done
// 		directly with *skb*\ **->hash**.
//
// 		Calling **bpf_set_hash_invalid**\ (), changing a packet
// 		prototype with **bpf_skb_change_proto**\ (), or calling
// 		**bpf_skb_store_bytes**\ () with the
// 		**BPF_F_INVALIDATE_HASH** are actions susceptible to clear
// 		the hash and to trigger a new computation for the next call to
// 		**bpf_get_hash_recalc**\ ().
// 	Return
// 		The 32-bit hash.
//
// u64 bpf_get_current_task(void)
// 	Description
// 		Get the current task.
// 	Return
// 		A pointer to the current task struct.
//
// long bpf_probe_write_user(void *dst, const void *src, u32 len)
// 	Description
// 		Attempt in a safe way to write *len* bytes from the buffer
// 		*src* to *dst* in memory. It only works for threads that are in
// 		user context, and *dst* must be a valid user space address.
//
// 		This helper should not be used to implement any kind of
// 		security mechanism because of TOC-TOU attacks, but rather to
// 		debug, divert, and manipulate execution of semi-cooperative
// 		processes.
//
// 		Keep in mind that this feature is meant for experiments, and it
// 		has a risk of crashing the system and running programs.
// 		Therefore, when an eBPF program using this helper is attached,
// 		a warning including PID and process name is printed to kernel
// 		logs.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_current_task_under_cgroup(struct bpf_map *map, u32 index)
// 	Description
// 		Check whether the probe is being run is the context of a given
// 		subset of the cgroup2 hierarchy. The cgroup2 to test is held by
// 		*map* of type **BPF_MAP_TYPE_CGROUP_ARRAY**, at *index*.
// 	Return
// 		The return value depends on the result of the test, and can be:
//
// 		* 1, if current task belongs to the cgroup2.
// 		* 0, if current task does not belong to the cgroup2.
// 		* A negative error code, if an error occurred.
//
// long bpf_skb_change_tail(struct sk_buff *skb, u32 len, u64 flags)
// 	Description
// 		Resize (trim or grow) the packet associated to *skb* to the
// 		new *len*. The *flags* are reserved for future usage, and must
// 		be left at zero.
//
// 		The basic idea is that the helper performs the needed work to
// 		change the size of the packet, then the eBPF program rewrites
// 		the rest via helpers like **bpf_skb_store_bytes**\ (),
// 		**bpf_l3_csum_replace**\ (), **bpf_l3_csum_replace**\ ()
// 		and others. This helper is a slow path utility intended for
// 		replies with control messages. And because it is targeted for
// 		slow path, the helper itself can afford to be slow: it
// 		implicitly linearizes, unclones and drops offloads from the
// 		*skb*.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_pull_data(struct sk_buff *skb, u32 len)
// 	Description
// 		Pull in non-linear data in case the *skb* is non-linear and not
// 		all of *len* are part of the linear section. Make *len* bytes
// 		from *skb* readable and writable. If a zero value is passed for
// 		*len*, then all bytes in the linear part of *skb* will be made
// 		readable and writable.
//
// 		This helper is only needed for reading and writing with direct
// 		packet access.
//
// 		For direct packet access, testing that offsets to access
// 		are within packet boundaries (test on *skb*\ **->data_end**) is
// 		susceptible to fail if offsets are invalid, or if the requested
// 		data is in non-linear parts of the *skb*. On failure the
// 		program can just bail out, or in the case of a non-linear
// 		buffer, use a helper to make the data available. The
// 		**bpf_skb_load_bytes**\ () helper is a first solution to access
// 		the data. Another one consists in using **bpf_skb_pull_data**
// 		to pull in once the non-linear parts, then retesting and
// 		eventually access the data.
//
// 		At the same time, this also makes sure the *skb* is uncloned,
// 		which is a necessary condition for direct write. As this needs
// 		to be an invariant for the write part only, the verifier
// 		detects writes and adds a prologue that is calling
// 		**bpf_skb_pull_data()** to effectively unclone the *skb* from
// 		the very beginning in case it is indeed cloned.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// s64 bpf_csum_update(struct sk_buff *skb, __wsum csum)
// 	Description
// 		Add the checksum *csum* into *skb*\ **->csum** in case the
// 		driver has supplied a checksum for the entire packet into that
// 		field. Return an error otherwise. This helper is intended to be
// 		used in combination with **bpf_csum_diff**\ (), in particular
// 		when the checksum needs to be updated after data has been
// 		written into the packet through direct packet access.
// 	Return
// 		The checksum on success, or a negative error code in case of
// 		failure.
//
// void bpf_set_hash_invalid(struct sk_buff *skb)
// 	Description
// 		Invalidate the current *skb*\ **->hash**. It can be used after
// 		mangling on headers through direct packet access, in order to
// 		indicate that the hash is outdated and to trigger a
// 		recalculation the next time the kernel tries to access this
// 		hash or when the **bpf_get_hash_recalc**\ () helper is called.
// 	Return
// 		void.
//
// long bpf_get_numa_node_id(void)
// 	Description
// 		Return the id of the current NUMA node. The primary use case
// 		for this helper is the selection of sockets for the local NUMA
// 		node, when the program is attached to sockets using the
// 		**SO_ATTACH_REUSEPORT_EBPF** option (see also **socket(7)**),
// 		but the helper is also available to other eBPF program types,
// 		similarly to **bpf_get_smp_processor_id**\ ().
// 	Return
// 		The id of current NUMA node.
//
// long bpf_skb_change_head(struct sk_buff *skb, u32 len, u64 flags)
// 	Description
// 		Grows headroom of packet associated to *skb* and adjusts the
// 		offset of the MAC header accordingly, adding *len* bytes of
// 		space. It automatically extends and reallocates memory as
// 		required.
//
// 		This helper can be used on a layer 3 *skb* to push a MAC header
// 		for redirection into a layer 2 device.
//
// 		All values for *flags* are reserved for future usage, and must
// 		be left at zero.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_xdp_adjust_head(struct xdp_buff *xdp_md, int delta)
// 	Description
// 		Adjust (move) *xdp_md*\ **->data** by *delta* bytes. Note that
// 		it is possible to use a negative value for *delta*. This helper
// 		can be used to prepare the packet for pushing or popping
// 		headers.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_probe_read_str(void *dst, u32 size, const void *unsafe_ptr)
// 	Description
// 		Copy a NUL terminated string from an unsafe kernel address
// 		*unsafe_ptr* to *dst*. See **bpf_probe_read_kernel_str**\ () for
// 		more details.
//
// 		Generally, use **bpf_probe_read_user_str**\ () or
// 		**bpf_probe_read_kernel_str**\ () instead.
// 	Return
// 		On success, the strictly positive length of the string,
// 		including the trailing NUL character. On error, a negative
// 		value.
//
// u64 bpf_get_socket_cookie(struct sk_buff *skb)
// 	Description
// 		If the **struct sk_buff** pointed by *skb* has a known socket,
// 		retrieve the cookie (generated by the kernel) of this socket.
// 		If no cookie has been set yet, generate a new cookie. Once
// 		generated, the socket cookie remains stable for the life of the
// 		socket. This helper can be useful for monitoring per socket
// 		networking traffic statistics as it provides a global socket
// 		identifier that can be assumed unique.
// 	Return
// 		A 8-byte long unique number on success, or 0 if the socket
// 		field is missing inside *skb*.
//
// u64 bpf_get_socket_cookie(struct bpf_sock_addr *ctx)
// 	Description
// 		Equivalent to bpf_get_socket_cookie() helper that accepts
// 		*skb*, but gets socket from **struct bpf_sock_addr** context.
// 	Return
// 		A 8-byte long unique number.
//
// u64 bpf_get_socket_cookie(struct bpf_sock_ops *ctx)
// 	Description
// 		Equivalent to **bpf_get_socket_cookie**\ () helper that accepts
// 		*skb*, but gets socket from **struct bpf_sock_ops** context.
// 	Return
// 		A 8-byte long unique number.
//
// u64 bpf_get_socket_cookie(struct sock *sk)
// 	Description
// 		Equivalent to **bpf_get_socket_cookie**\ () helper that accepts
// 		*sk*, but gets socket from a BTF **struct sock**. This helper
// 		also works for sleepable programs.
// 	Return
// 		A 8-byte long unique number or 0 if *sk* is NULL.
//
// u32 bpf_get_socket_uid(struct sk_buff *skb)
// 	Description
// 		Get the owner UID of the socked associated to *skb*.
// 	Return
// 		The owner UID of the socket associated to *skb*. If the socket
// 		is **NULL**, or if it is not a full socket (i.e. if it is a
// 		time-wait or a request socket instead), **overflowuid** value
// 		is returned (note that **overflowuid** might also be the actual
// 		UID value for the socket).
//
// long bpf_set_hash(struct sk_buff *skb, u32 hash)
// 	Description
// 		Set the full hash for *skb* (set the field *skb*\ **->hash**)
// 		to value *hash*.
// 	Return
// 		0
//
// long bpf_setsockopt(void *bpf_socket, int level, int optname, void *optval, int optlen)
// 	Description
// 		Emulate a call to **setsockopt()** on the socket associated to
// 		*bpf_socket*, which must be a full socket. The *level* at
// 		which the option resides and the name *optname* of the option
// 		must be specified, see **setsockopt(2)** for more information.
// 		The option value of length *optlen* is pointed by *optval*.
//
// 		*bpf_socket* should be one of the following:
//
// 		* **struct bpf_sock_ops** for **BPF_PROG_TYPE_SOCK_OPS**.
// 		* **struct bpf_sock_addr** for **BPF_CGROUP_INET4_CONNECT**,
// 		  **BPF_CGROUP_INET6_CONNECT** and **BPF_CGROUP_UNIX_CONNECT**.
//
// 		This helper actually implements a subset of **setsockopt()**.
// 		It supports the following *level*\ s:
//
// 		* **SOL_SOCKET**, which supports the following *optname*\ s:
// 		  **SO_RCVBUF**, **SO_SNDBUF**, **SO_MAX_PACING_RATE**,
// 		  **SO_PRIORITY**, **SO_RCVLOWAT**, **SO_MARK**,
// 		  **SO_BINDTODEVICE**, **SO_KEEPALIVE**, **SO_REUSEADDR**,
// 		  **SO_REUSEPORT**, **SO_BINDTOIFINDEX**, **SO_TXREHASH**.
// 		* **IPPROTO_TCP**, which supports the following *optname*\ s:
// 		  **TCP_CONGESTION**, **TCP_BPF_IW**,
// 		  **TCP_BPF_SNDCWND_CLAMP**, **TCP_SAVE_SYN**,
// 		  **TCP_KEEPIDLE**, **TCP_KEEPINTVL**, **TCP_KEEPCNT**,
// 		  **TCP_SYNCNT**, **TCP_USER_TIMEOUT**, **TCP_NOTSENT_LOWAT**,
// 		  **TCP_NODELAY**, **TCP_MAXSEG**, **TCP_WINDOW_CLAMP**,
// 		  **TCP_THIN_LINEAR_TIMEOUTS**, **TCP_BPF_DELACK_MAX**,
// 		  **TCP_BPF_RTO_MIN**, **TCP_BPF_SOCK_OPS_CB_FLAGS**.
// 		* **IPPROTO_IP**, which supports *optname* **IP_TOS**.
// 		* **IPPROTO_IPV6**, which supports the following *optname*\ s:
// 		  **IPV6_TCLASS**, **IPV6_AUTOFLOWLABEL**.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_adjust_room(struct sk_buff *skb, s32 len_diff, u32 mode, u64 flags)
// 	Description
// 		Grow or shrink the room for data in the packet associated to
// 		*skb* by *len_diff*, and according to the selected *mode*.
//
// 		By default, the helper will reset any offloaded checksum
// 		indicator of the skb to CHECKSUM_NONE. This can be avoided
// 		by the following flag:
//
// 		* **BPF_F_ADJ_ROOM_NO_CSUM_RESET**: Do not reset offloaded
// 		  checksum data of the skb to CHECKSUM_NONE.
//
// 		There are two supported modes at this time:
//
// 		* **BPF_ADJ_ROOM_MAC**: Adjust room at the mac layer
// 		  (room space is added or removed between the layer 2 and
// 		  layer 3 headers).
//
// 		* **BPF_ADJ_ROOM_NET**: Adjust room at the network layer
// 		  (room space is added or removed between the layer 3 and
// 		  layer 4 headers).
//
// 		The following flags are supported at this time:
//
// 		* **BPF_F_ADJ_ROOM_FIXED_GSO**: Do not adjust gso_size.
// 		  Adjusting mss in this way is not allowed for datagrams.
//
// 		* **BPF_F_ADJ_ROOM_ENCAP_L3_IPV4**,
// 		  **BPF_F_ADJ_ROOM_ENCAP_L3_IPV6**:
// 		  Any new space is reserved to hold a tunnel header.
// 		  Configure skb offsets and other fields accordingly.
//
// 		* **BPF_F_ADJ_ROOM_ENCAP_L4_GRE**,
// 		  **BPF_F_ADJ_ROOM_ENCAP_L4_UDP**:
// 		  Use with ENCAP_L3 flags to further specify the tunnel type.
//
// 		* **BPF_F_ADJ_ROOM_ENCAP_L2**\ (*len*):
// 		  Use with ENCAP_L3/L4 flags to further specify the tunnel
// 		  type; *len* is the length of the inner MAC header.
//
// 		* **BPF_F_ADJ_ROOM_ENCAP_L2_ETH**:
// 		  Use with BPF_F_ADJ_ROOM_ENCAP_L2 flag to further specify the
// 		  L2 type as Ethernet.
//
// 		* **BPF_F_ADJ_ROOM_DECAP_L3_IPV4**,
// 		  **BPF_F_ADJ_ROOM_DECAP_L3_IPV6**:
// 		  Indicate the new IP header version after decapsulating the
// 		  outer IP header. Used when the inner and outer IP versions
// 		  are different. These flags only trigger a protocol change
// 		  without clearing any tunnel-specific GSO flags.
//
// 		* **BPF_F_ADJ_ROOM_DECAP_L4_GRE**:
// 		  Clear GRE tunnel GSO flags (SKB_GSO_GRE and SKB_GSO_GRE_CSUM)
// 		  when decapsulating a GRE tunnel.
//
// 		* **BPF_F_ADJ_ROOM_DECAP_L4_UDP**:
// 		  Clear UDP tunnel GSO flags (SKB_GSO_UDP_TUNNEL and
// 		  SKB_GSO_UDP_TUNNEL_CSUM) when decapsulating a UDP tunnel.
//
// 		* **BPF_F_ADJ_ROOM_DECAP_IPXIP4**:
// 		  Clear IPIP/SIT tunnel GSO flag (SKB_GSO_IPXIP4) when decapsulating
// 		  a tunnel with an outer IPv4 header (IPv4-in-IPv4 or IPv6-in-IPv4).
//
// 		* **BPF_F_ADJ_ROOM_DECAP_IPXIP6**:
// 		  Clear IPv6 encapsulation tunnel GSO flag (SKB_GSO_IPXIP6) when
// 		  decapsulating a tunnel with an outer IPv6 header (IPv6-in-IPv6
// 		  or IPv4-in-IPv6).
//
// 		When using the decapsulation flags above, the skb->encapsulation
// 		flag is automatically cleared if all tunnel-specific GSO flags
// 		(SKB_GSO_UDP_TUNNEL, SKB_GSO_UDP_TUNNEL_CSUM, SKB_GSO_GRE,
// 		SKB_GSO_GRE_CSUM, SKB_GSO_IPXIP4, SKB_GSO_IPXIP6) have been
// 		removed from the packet. This handles cases where all tunnel
// 		layers have been decapsulated.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_redirect_map(struct bpf_map *map, u64 key, u64 flags)
// 	Description
// 		Redirect the packet to the endpoint referenced by *map* at
// 		index *key*. Depending on its type, this *map* can contain
// 		references to net devices (for forwarding packets through other
// 		ports), or to CPUs (for redirecting XDP frames to another CPU;
// 		but this is only implemented for native XDP (with driver
// 		support) as of this writing).
//
// 		The lower two bits of *flags* are used as the return code if
// 		the map lookup fails. This is so that the return value can be
// 		one of the XDP program return codes up to **XDP_TX**, as chosen
// 		by the caller. The higher bits of *flags* can be set to
// 		BPF_F_BROADCAST or BPF_F_EXCLUDE_INGRESS as defined below.
//
// 		With BPF_F_BROADCAST the packet will be broadcasted to all the
// 		interfaces in the map, with BPF_F_EXCLUDE_INGRESS the ingress
// 		interface will be excluded when do broadcasting.
//
// 		See also **bpf_redirect**\ (), which only supports redirecting
// 		to an ifindex, but doesn't require a map to do so.
// 	Return
// 		**XDP_REDIRECT** on success, or the value of the two lower bits
// 		of the *flags* argument on error.
//
// long bpf_sk_redirect_map(struct sk_buff *skb, struct bpf_map *map, u32 key, u64 flags)
// 	Description
// 		Redirect the packet to the socket referenced by *map* (of type
// 		**BPF_MAP_TYPE_SOCKMAP**) at index *key*. Both ingress and
// 		egress interfaces can be used for redirection. The
// 		**BPF_F_INGRESS** value in *flags* is used to make the
// 		distinction (ingress path is selected if the flag is present,
// 		egress path otherwise). This is the only flag supported for now.
// 	Return
// 		**SK_PASS** on success, or **SK_DROP** on error.
//
// long bpf_sock_map_update(struct bpf_sock_ops *skops, struct bpf_map *map, void *key, u64 flags)
// 	Description
// 		Add an entry to, or update a *map* referencing sockets. The
// 		*skops* is used as a new value for the entry associated to
// 		*key*. *flags* is one of:
//
// 		**BPF_NOEXIST**
// 			The entry for *key* must not exist in the map.
// 		**BPF_EXIST**
// 			The entry for *key* must already exist in the map.
// 		**BPF_ANY**
// 			No condition on the existence of the entry for *key*.
//
// 		If the *map* has eBPF programs (parser and verdict), those will
// 		be inherited by the socket being added. If the socket is
// 		already attached to eBPF programs, this results in an error.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_xdp_adjust_meta(struct xdp_buff *xdp_md, int delta)
// 	Description
// 		Adjust the address pointed by *xdp_md*\ **->data_meta** by
// 		*delta* (which can be positive or negative). Note that this
// 		operation modifies the address stored in *xdp_md*\ **->data**,
// 		so the latter must be loaded only after the helper has been
// 		called.
//
// 		The use of *xdp_md*\ **->data_meta** is optional and programs
// 		are not required to use it. The rationale is that when the
// 		packet is processed with XDP (e.g. as DoS filter), it is
// 		possible to push further meta data along with it before passing
// 		to the stack, and to give the guarantee that an ingress eBPF
// 		program attached as a TC classifier on the same device can pick
// 		this up for further post-processing. Since TC works with socket
// 		buffers, it remains possible to set from XDP the **mark** or
// 		**priority** pointers, or other pointers for the socket buffer.
// 		Having this scratch space generic and programmable allows for
// 		more flexibility as the user is free to store whatever meta
// 		data they need.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_perf_event_read_value(struct bpf_map *map, u64 flags, struct bpf_perf_event_value *buf, u32 buf_size)
// 	Description
// 		Read the value of a perf event counter, and store it into *buf*
// 		of size *buf_size*. This helper relies on a *map* of type
// 		**BPF_MAP_TYPE_PERF_EVENT_ARRAY**. The nature of the perf event
// 		counter is selected when *map* is updated with perf event file
// 		descriptors. The *map* is an array whose size is the number of
// 		available CPUs, and each cell contains a value relative to one
// 		CPU. The value to retrieve is indicated by *flags*, that
// 		contains the index of the CPU to look up, masked with
// 		**BPF_F_INDEX_MASK**. Alternatively, *flags* can be set to
// 		**BPF_F_CURRENT_CPU** to indicate that the value for the
// 		current CPU should be retrieved.
//
// 		This helper behaves in a way close to
// 		**bpf_perf_event_read**\ () helper, save that instead of
// 		just returning the value observed, it fills the *buf*
// 		structure. This allows for additional data to be retrieved: in
// 		particular, the enabled and running times (in *buf* * 		**->enabled** and *buf*\ **->running**, respectively) are
// 		copied. In general, **bpf_perf_event_read_value**\ () is
// 		recommended over **bpf_perf_event_read**\ (), which has some
// 		ABI issues and provides fewer functionalities.
//
// 		These values are interesting, because hardware PMU (Performance
// 		Monitoring Unit) counters are limited resources. When there are
// 		more PMU based perf events opened than available counters,
// 		kernel will multiplex these events so each event gets certain
// 		percentage (but not all) of the PMU time. In case that
// 		multiplexing happens, the number of samples or counter value
// 		will not reflect the case compared to when no multiplexing
// 		occurs. This makes comparison between different runs difficult.
// 		Typically, the counter value should be normalized before
// 		comparing to other experiments. The usual normalization is done
// 		as follows.
//
// 		::
//
// 			normalized_counter = counter * t_enabled / t_running
//
// 		Where t_enabled is the time enabled for event and t_running is
// 		the time running for event since last normalization. The
// 		enabled and running times are accumulated since the perf event
// 		open. To achieve scaling factor between two invocations of an
// 		eBPF program, users can use CPU id as the key (which is
// 		typical for perf array usage model) to remember the previous
// 		value and do the calculation inside the eBPF program.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_perf_prog_read_value(struct bpf_perf_event_data *ctx, struct bpf_perf_event_value *buf, u32 buf_size)
// 	Description
// 		For an eBPF program attached to a perf event, retrieve the
// 		value of the event counter associated to *ctx* and store it in
// 		the structure pointed by *buf* and of size *buf_size*. Enabled
// 		and running times are also stored in the structure (see
// 		description of helper **bpf_perf_event_read_value**\ () for
// 		more details).
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_getsockopt(void *bpf_socket, int level, int optname, void *optval, int optlen)
// 	Description
// 		Emulate a call to **getsockopt()** on the socket associated to
// 		*bpf_socket*, which must be a full socket. The *level* at
// 		which the option resides and the name *optname* of the option
// 		must be specified, see **getsockopt(2)** for more information.
// 		The retrieved value is stored in the structure pointed by
// 		*opval* and of length *optlen*.
//
// 		*bpf_socket* should be one of the following:
//
// 		* **struct bpf_sock_ops** for **BPF_PROG_TYPE_SOCK_OPS**.
// 		* **struct bpf_sock_addr** for **BPF_CGROUP_INET4_CONNECT**,
// 		  **BPF_CGROUP_INET6_CONNECT** and **BPF_CGROUP_UNIX_CONNECT**.
//
// 		This helper actually implements a subset of **getsockopt()**.
// 		It supports the same set of *optname*\ s that is supported by
// 		the **bpf_setsockopt**\ () helper.  The exceptions are
// 		**TCP_BPF_*** is **bpf_setsockopt**\ () only and
// 		**TCP_SAVED_SYN** is **bpf_getsockopt**\ () only.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_override_return(struct pt_regs *regs, u64 rc)
// 	Description
// 		Used for error injection, this helper uses kprobes to override
// 		the return value of the probed function, and to set it to *rc*.
// 		The first argument is the context *regs* on which the kprobe
// 		works.
//
// 		This helper works by setting the PC (program counter)
// 		to an override function which is run in place of the original
// 		probed function. This means the probed function is not run at
// 		all. The replacement function just returns with the required
// 		value.
//
// 		This helper has security implications, and thus is subject to
// 		restrictions. It is only available if the kernel was compiled
// 		with the **CONFIG_BPF_KPROBE_OVERRIDE** configuration
// 		option, and in this case it only works on functions tagged with
// 		**ALLOW_ERROR_INJECTION** in the kernel code.
// 	Return
// 		0
//
// long bpf_sock_ops_cb_flags_set(struct bpf_sock_ops *bpf_sock, int argval)
// 	Description
// 		Attempt to set the value of the **bpf_sock_ops_cb_flags** field
// 		for the full TCP socket associated to *bpf_sock_ops* to
// 		*argval*.
//
// 		The primary use of this field is to determine if there should
// 		be calls to eBPF programs of type
// 		**BPF_PROG_TYPE_SOCK_OPS** at various points in the TCP
// 		code. A program of the same type can change its value, per
// 		connection and as necessary, when the connection is
// 		established. This field is directly accessible for reading, but
// 		this helper must be used for updates in order to return an
// 		error if an eBPF program tries to set a callback that is not
// 		supported in the current kernel.
//
// 		*argval* is a flag array which can combine these flags:
//
// 		* **BPF_SOCK_OPS_RTO_CB_FLAG** (retransmission time out)
// 		* **BPF_SOCK_OPS_RETRANS_CB_FLAG** (retransmission)
// 		* **BPF_SOCK_OPS_STATE_CB_FLAG** (TCP state change)
// 		* **BPF_SOCK_OPS_RTT_CB_FLAG** (every RTT)
//
// 		Therefore, this function can be used to clear a callback flag by
// 		setting the appropriate bit to zero. e.g. to disable the RTO
// 		callback:
//
// 		**bpf_sock_ops_cb_flags_set(bpf_sock,**
// 			**bpf_sock->bpf_sock_ops_cb_flags & ~BPF_SOCK_OPS_RTO_CB_FLAG)**
//
// 		Here are some examples of where one could call such eBPF
// 		program:
//
// 		* When RTO fires.
// 		* When a packet is retransmitted.
// 		* When the connection terminates.
// 		* When a packet is sent.
// 		* When a packet is received.
// 	Return
// 		Code **-EINVAL** if the socket is not a full TCP socket;
// 		otherwise, a positive number containing the bits that could not
// 		be set is returned (which comes down to 0 if all bits were set
// 		as required).
//
// long bpf_msg_redirect_map(struct sk_msg_buff *msg, struct bpf_map *map, u32 key, u64 flags)
// 	Description
// 		This helper is used in programs implementing policies at the
// 		socket level. If the message *msg* is allowed to pass (i.e. if
// 		the verdict eBPF program returns **SK_PASS**), redirect it to
// 		the socket referenced by *map* (of type
// 		**BPF_MAP_TYPE_SOCKMAP**) at index *key*. Both ingress and
// 		egress interfaces can be used for redirection. The
// 		**BPF_F_INGRESS** value in *flags* is used to make the
// 		distinction (ingress path is selected if the flag is present,
// 		egress path otherwise). This is the only flag supported for now.
// 	Return
// 		**SK_PASS** on success, or **SK_DROP** on error.
//
// long bpf_msg_apply_bytes(struct sk_msg_buff *msg, u32 bytes)
// 	Description
// 		For socket policies, apply the verdict of the eBPF program to
// 		the next *bytes* (number of bytes) of message *msg*.
//
// 		For example, this helper can be used in the following cases:
//
// 		* A single **sendmsg**\ () or **sendfile**\ () system call
// 		  contains multiple logical messages that the eBPF program is
// 		  supposed to read and for which it should apply a verdict.
// 		* An eBPF program only cares to read the first *bytes* of a
// 		  *msg*. If the message has a large payload, then setting up
// 		  and calling the eBPF program repeatedly for all bytes, even
// 		  though the verdict is already known, would create unnecessary
// 		  overhead.
//
// 		When called from within an eBPF program, the helper sets a
// 		counter internal to the BPF infrastructure, that is used to
// 		apply the last verdict to the next *bytes*. If *bytes* is
// 		smaller than the current data being processed from a
// 		**sendmsg**\ () or **sendfile**\ () system call, the first
// 		*bytes* will be sent and the eBPF program will be re-run with
// 		the pointer for start of data pointing to byte number *bytes*
// 		**+ 1**. If *bytes* is larger than the current data being
// 		processed, then the eBPF verdict will be applied to multiple
// 		**sendmsg**\ () or **sendfile**\ () calls until *bytes* are
// 		consumed.
//
// 		Note that if a socket closes with the internal counter holding
// 		a non-zero value, this is not a problem because data is not
// 		being buffered for *bytes* and is sent as it is received.
// 	Return
// 		0
//
// long bpf_msg_cork_bytes(struct sk_msg_buff *msg, u32 bytes)
// 	Description
// 		For socket policies, prevent the execution of the verdict eBPF
// 		program for message *msg* until *bytes* (byte number) have been
// 		accumulated.
//
// 		This can be used when one needs a specific number of bytes
// 		before a verdict can be assigned, even if the data spans
// 		multiple **sendmsg**\ () or **sendfile**\ () calls. The extreme
// 		case would be a user calling **sendmsg**\ () repeatedly with
// 		1-byte long message segments. Obviously, this is bad for
// 		performance, but it is still valid. If the eBPF program needs
// 		*bytes* bytes to validate a header, this helper can be used to
// 		prevent the eBPF program to be called again until *bytes* have
// 		been accumulated.
// 	Return
// 		0
//
// long bpf_msg_pull_data(struct sk_msg_buff *msg, u32 start, u32 end, u64 flags)
// 	Description
// 		For socket policies, pull in non-linear data from user space
// 		for *msg* and set pointers *msg*\ **->data** and *msg* * 		**->data_end** to *start* and *end* bytes offsets into *msg*,
// 		respectively.
//
// 		If a program of type **BPF_PROG_TYPE_SK_MSG** is run on a
// 		*msg* it can only parse data that the (**data**, **data_end**)
// 		pointers have already consumed. For **sendmsg**\ () hooks this
// 		is likely the first scatterlist element. But for calls relying
// 		on the **sendpage** handler (e.g. **sendfile**\ ()) this will
// 		be the range (**0**, **0**) because the data is shared with
// 		user space and by default the objective is to avoid allowing
// 		user space to modify data while (or after) eBPF verdict is
// 		being decided. This helper can be used to pull in data and to
// 		set the start and end pointer to given values. Data will be
// 		copied if necessary (i.e. if data was not linear and if start
// 		and end pointers do not point to the same chunk).
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
//
// 		All values for *flags* are reserved for future usage, and must
// 		be left at zero.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_bind(struct bpf_sock_addr *ctx, struct sockaddr *addr, int addr_len)
// 	Description
// 		Bind the socket associated to *ctx* to the address pointed by
// 		*addr*, of length *addr_len*. This allows for making outgoing
// 		connection from the desired IP address, which can be useful for
// 		example when all processes inside a cgroup should use one
// 		single IP address on a host that has multiple IP configured.
//
// 		This helper works for IPv4 and IPv6, TCP and UDP sockets. The
// 		domain (*addr*\ **->sa_family**) must be **AF_INET** (or
// 		**AF_INET6**). It's advised to pass zero port (**sin_port**
// 		or **sin6_port**) which triggers IP_BIND_ADDRESS_NO_PORT-like
// 		behavior and lets the kernel efficiently pick up an unused
// 		port as long as 4-tuple is unique. Passing non-zero port might
// 		lead to degraded performance.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_xdp_adjust_tail(struct xdp_buff *xdp_md, int delta)
// 	Description
// 		Adjust (move) *xdp_md*\ **->data_end** by *delta* bytes. It is
// 		possible to both shrink and grow the packet tail.
// 		Shrink done via *delta* being a negative integer.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_skb_get_xfrm_state(struct sk_buff *skb, u32 index, struct bpf_xfrm_state *xfrm_state, u32 size, u64 flags)
// 	Description
// 		Retrieve the XFRM state (IP transform framework, see also
// 		**ip-xfrm(8)**) at *index* in XFRM "security path" for *skb*.
//
// 		The retrieved value is stored in the **struct bpf_xfrm_state**
// 		pointed by *xfrm_state* and of length *size*.
//
// 		All values for *flags* are reserved for future usage, and must
// 		be left at zero.
//
// 		This helper is available only if the kernel was compiled with
// 		**CONFIG_XFRM** configuration option.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_get_stack(void *ctx, void *buf, u32 size, u64 flags)
// 	Description
// 		Return a user or a kernel stack in bpf program provided buffer.
// 		To achieve this, the helper needs *ctx*, which is a pointer
// 		to the context on which the tracing program is executed.
// 		To store the stacktrace, the bpf program provides *buf* with
// 		a nonnegative *size*.
//
// 		The last argument, *flags*, holds the number of stack frames to
// 		skip (from 0 to 255), masked with
// 		**BPF_F_SKIP_FIELD_MASK**. The next bits can be used to set
// 		the following flags:
//
// 		**BPF_F_USER_STACK**
// 			Collect a user space stack instead of a kernel stack.
// 		**BPF_F_USER_BUILD_ID**
// 			Collect (build_id, file_offset) instead of ips for user
// 			stack, only valid if **BPF_F_USER_STACK** is also
// 			specified.
//
// 			*file_offset* is an offset relative to the beginning
// 			of the executable or shared object file backing the vma
// 			which the *ip* falls in. It is *not* an offset relative
// 			to that object's base address. Accordingly, it must be
// 			adjusted by adding (sh_addr - sh_offset), where
// 			sh_{addr,offset} correspond to the executable section
// 			containing *file_offset* in the object, for comparisons
// 			to symbols' st_value to be valid.
//
// 		**bpf_get_stack**\ () can collect up to
// 		**PERF_MAX_STACK_DEPTH** both kernel and user frames, subject
// 		to sufficient large buffer size. Note that
// 		this limit can be controlled with the **sysctl** program, and
// 		that it should be manually increased in order to profile long
// 		user stacks (such as stacks for Java programs). To do so, use:
//
// 		::
//
// 			# sysctl kernel.perf_event_max_stack=<new value>
// 	Return
// 		The non-negative copied *buf* length equal to or less than
// 		*size* on success, or a negative error in case of failure.
//
// long bpf_skb_load_bytes_relative(const void *skb, u32 offset, void *to, u32 len, u32 start_header)
// 	Description
// 		This helper is similar to **bpf_skb_load_bytes**\ () in that
// 		it provides an easy way to load *len* bytes from *offset*
// 		from the packet associated to *skb*, into the buffer pointed
// 		by *to*. The difference to **bpf_skb_load_bytes**\ () is that
// 		a fifth argument *start_header* exists in order to select a
// 		base offset to start from. *start_header* can be one of:
//
// 		**BPF_HDR_START_MAC**
// 			Base offset to load data from is *skb*'s mac header.
// 		**BPF_HDR_START_NET**
// 			Base offset to load data from is *skb*'s network header.
//
// 		In general, "direct packet access" is the preferred method to
// 		access packet data, however, this helper is in particular useful
// 		in socket filters where *skb*\ **->data** does not always point
// 		to the start of the mac header and where "direct packet access"
// 		is not available.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_fib_lookup(void *ctx, struct bpf_fib_lookup *params, int plen, u32 flags)
// 	Description
// 		Do FIB lookup in kernel tables using parameters in *params*.
// 		If lookup is successful and result shows packet is to be
// 		forwarded, the neighbor tables are searched for the nexthop.
// 		If successful (ie., FIB lookup shows forwarding and nexthop
// 		is resolved), the nexthop address is returned in ipv4_dst
// 		or ipv6_dst based on family, smac is set to mac address of
// 		egress device, dmac is set to nexthop mac address, rt_metric
// 		is set to metric from route (IPv4/IPv6 only), and ifindex
// 		is set to the device index of the nexthop from the FIB lookup.
//
// 		*plen* argument is the size of the passed in struct.
// 		*flags* argument can be a combination of one or more of the
// 		following values:
//
// 		**BPF_FIB_LOOKUP_DIRECT**
// 			Do a direct table lookup vs full lookup using FIB
// 			rules.
// 		**BPF_FIB_LOOKUP_TBID**
// 			Used with BPF_FIB_LOOKUP_DIRECT.
// 			Use the routing table ID present in *params*->tbid
// 			for the fib lookup.
// 		**BPF_FIB_LOOKUP_OUTPUT**
// 			Perform lookup from an egress perspective (default is
// 			ingress).
// 		**BPF_FIB_LOOKUP_SKIP_NEIGH**
// 			Skip the neighbour table lookup. *params*->dmac
// 			and *params*->smac will not be set as output. A common
// 			use case is to call **bpf_redirect_neigh**\ () after
// 			doing **bpf_fib_lookup**\ ().
// 		**BPF_FIB_LOOKUP_SRC**
// 			Derive and set source IP addr in *params*->ipv{4,6}_src
// 			for the nexthop. If the src addr cannot be derived,
// 			**BPF_FIB_LKUP_RET_NO_SRC_ADDR** is returned. In this
// 			case, *params*->dmac and *params*->smac are not set either.
// 		**BPF_FIB_LOOKUP_MARK**
// 			Use the mark present in *params*->mark for the fib lookup.
// 			This option should not be used with BPF_FIB_LOOKUP_DIRECT,
// 			as it only has meaning for full lookups.
// 		**BPF_FIB_LOOKUP_VLAN**
// 			If the fib lookup resolves to a VLAN device whose
// 			parent is a real (non-VLAN) device, set
// 			*params*->h_vlan_proto and *params*->h_vlan_TCI from
// 			the VLAN device and replace *params*->ifindex with the
// 			parent's ifindex. *params*->h_vlan_TCI carries the VID
// 			only, with PCP and DEI bits zero; a consumer wanting to
// 			set egress priority writes PCP itself. *params*->smac is
// 			the VLAN device's own address, which can differ from the
// 			parent's. Only the immediate parent is resolved; if it
// 			is itself a VLAN device (QinQ) or in another namespace,
// 			the egress cannot be reduced to a physical device plus
// 			one tag and the lookup returns
// 			**BPF_FIB_LKUP_RET_VLAN_FAILURE** with *params*->ifindex
// 			left at the input. To obtain the VLAN device's own
// 			ifindex, repeat the lookup without
// 			**BPF_FIB_LOOKUP_VLAN**, re-initializing *params*
// 			first: output fields overwrite the inputs they share
// 			storage with. The swap and the vlan fields
// 			are written only on success; other output fields keep
// 			the helper's existing behaviour, so a frag-needed result
// 			still reports the route mtu in *params*->mtu_result.
// 			This flag is only valid for XDP programs; tc programs
// 			receive -EINVAL since they can redirect to the VLAN
// 			device directly.
// 		**BPF_FIB_LOOKUP_VLAN_INPUT**
// 			Treat *params*->h_vlan_proto and *params*->h_vlan_TCI
// 			as an input VLAN tag and run the lookup as if ingress
// 			had happened on the VLAN subinterface carrying that tag
// 			on *params*->ifindex. The VID is the low 12 bits of
// 			*params*->h_vlan_TCI; *params*->h_vlan_proto must be
// 			ETH_P_8021Q or ETH_P_8021AD in network byte order, else
// 			**-EINVAL**. If *params*->ifindex is itself a VLAN
// 			device, its inner (QinQ) subinterface is matched; for a
// 			bond or team, pass the master's ifindex. An unmatched
// 			tag, a down device, or one in another namespace returns
// 			**BPF_FIB_LKUP_RET_NOT_FWDED**, mirroring real ingress.
// 			A VID of 0 is looked up literally, so do not set this
// 			flag for priority-tagged frames. Cannot be combined with
// 			**BPF_FIB_LOOKUP_TBID** or **BPF_FIB_LOOKUP_OUTPUT**
// 			(returns **-EINVAL**).
//
// 		*ctx* is either **struct xdp_md** for XDP programs or
// 		**struct sk_buff** tc cls_act programs.
// 	Return
// 		* < 0 if any input argument is invalid
// 		*   0 on success (packet is forwarded, nexthop neighbor exists)
// 		* > 0 one of **BPF_FIB_LKUP_RET_** codes explaining why the
// 		  packet is not forwarded or needs assist from full stack
//
// 		If lookup fails with BPF_FIB_LKUP_RET_FRAG_NEEDED, then the MTU
// 		was exceeded and output params->mtu_result contains the MTU.
//
// long bpf_sock_hash_update(struct bpf_sock_ops *skops, struct bpf_map *map, void *key, u64 flags)
// 	Description
// 		Add an entry to, or update a sockhash *map* referencing sockets.
// 		The *skops* is used as a new value for the entry associated to
// 		*key*. *flags* is one of:
//
// 		**BPF_NOEXIST**
// 			The entry for *key* must not exist in the map.
// 		**BPF_EXIST**
// 			The entry for *key* must already exist in the map.
// 		**BPF_ANY**
// 			No condition on the existence of the entry for *key*.
//
// 		If the *map* has eBPF programs (parser and verdict), those will
// 		be inherited by the socket being added. If the socket is
// 		already attached to eBPF programs, this results in an error.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_msg_redirect_hash(struct sk_msg_buff *msg, struct bpf_map *map, void *key, u64 flags)
// 	Description
// 		This helper is used in programs implementing policies at the
// 		socket level. If the message *msg* is allowed to pass (i.e. if
// 		the verdict eBPF program returns **SK_PASS**), redirect it to
// 		the socket referenced by *map* (of type
// 		**BPF_MAP_TYPE_SOCKHASH**) using hash *key*. Both ingress and
// 		egress interfaces can be used for redirection. The
// 		**BPF_F_INGRESS** value in *flags* is used to make the
// 		distinction (ingress path is selected if the flag is present,
// 		egress path otherwise). This is the only flag supported for now.
// 	Return
// 		**SK_PASS** on success, or **SK_DROP** on error.
//
// long bpf_sk_redirect_hash(struct sk_buff *skb, struct bpf_map *map, void *key, u64 flags)
// 	Description
// 		This helper is used in programs implementing policies at the
// 		skb socket level. If the sk_buff *skb* is allowed to pass (i.e.
// 		if the verdict eBPF program returns **SK_PASS**), redirect it
// 		to the socket referenced by *map* (of type
// 		**BPF_MAP_TYPE_SOCKHASH**) using hash *key*. Both ingress and
// 		egress interfaces can be used for redirection. The
// 		**BPF_F_INGRESS** value in *flags* is used to make the
// 		distinction (ingress path is selected if the flag is present,
// 		egress otherwise). This is the only flag supported for now.
// 	Return
// 		**SK_PASS** on success, or **SK_DROP** on error.
//
// long bpf_lwt_push_encap(struct sk_buff *skb, u32 type, void *hdr, u32 len)
// 	Description
// 		Encapsulate the packet associated to *skb* within a Layer 3
// 		protocol header. This header is provided in the buffer at
// 		address *hdr*, with *len* its size in bytes. *type* indicates
// 		the protocol of the header and can be one of:
//
// 		**BPF_LWT_ENCAP_SEG6**
// 			IPv6 encapsulation with Segment Routing Header
// 			(**struct ipv6_sr_hdr**). *hdr* only contains the SRH,
// 			the IPv6 header is computed by the kernel.
// 		**BPF_LWT_ENCAP_SEG6_INLINE**
// 			Only works if *skb* contains an IPv6 packet. Insert a
// 			Segment Routing Header (**struct ipv6_sr_hdr**) inside
// 			the IPv6 header.
// 		**BPF_LWT_ENCAP_IP**
// 			IP encapsulation (GRE/GUE/IPIP/etc). The outer header
// 			must be IPv4 or IPv6, followed by zero or more
// 			additional headers, up to **LWT_BPF_MAX_HEADROOM**
// 			total bytes in all prepended headers. Please note that
// 			if **skb_is_gso**\ (*skb*) is true, no more than two
// 			headers can be prepended, and the inner header, if
// 			present, should be either GRE or UDP/GUE.
//
// 		**BPF_LWT_ENCAP_SEG6**\ \* types can be called by BPF programs
// 		of type **BPF_PROG_TYPE_LWT_IN**; **BPF_LWT_ENCAP_IP** type can
// 		be called by bpf programs of types **BPF_PROG_TYPE_LWT_IN** and
// 		**BPF_PROG_TYPE_LWT_XMIT**.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_lwt_seg6_store_bytes(struct sk_buff *skb, u32 offset, const void *from, u32 len)
// 	Description
// 		Store *len* bytes from address *from* into the packet
// 		associated to *skb*, at *offset*. Only the flags, tag and TLVs
// 		inside the outermost IPv6 Segment Routing Header can be
// 		modified through this helper.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_lwt_seg6_adjust_srh(struct sk_buff *skb, u32 offset, s32 delta)
// 	Description
// 		Adjust the size allocated to TLVs in the outermost IPv6
// 		Segment Routing Header contained in the packet associated to
// 		*skb*, at position *offset* by *delta* bytes. Only offsets
// 		after the segments are accepted. *delta* can be as well
// 		positive (growing) as negative (shrinking).
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_lwt_seg6_action(struct sk_buff *skb, u32 action, void *param, u32 param_len)
// 	Description
// 		Apply an IPv6 Segment Routing action of type *action* to the
// 		packet associated to *skb*. Each action takes a parameter
// 		contained at address *param*, and of length *param_len* bytes.
// 		*action* can be one of:
//
// 		**SEG6_LOCAL_ACTION_END_X**
// 			End.X action: Endpoint with Layer-3 cross-connect.
// 			Type of *param*: **struct in6_addr**.
// 		**SEG6_LOCAL_ACTION_END_T**
// 			End.T action: Endpoint with specific IPv6 table lookup.
// 			Type of *param*: **int**.
// 		**SEG6_LOCAL_ACTION_END_B6**
// 			End.B6 action: Endpoint bound to an SRv6 policy.
// 			Type of *param*: **struct ipv6_sr_hdr**.
// 		**SEG6_LOCAL_ACTION_END_B6_ENCAP**
// 			End.B6.Encap action: Endpoint bound to an SRv6
// 			encapsulation policy.
// 			Type of *param*: **struct ipv6_sr_hdr**.
//
// 		A call to this helper is susceptible to change the underlying
// 		packet buffer. Therefore, at load time, all checks on pointers
// 		previously done by the verifier are invalidated and must be
// 		performed again, if the helper is used in combination with
// 		direct packet access.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_rc_repeat(void *ctx)
// 	Description
// 		This helper is used in programs implementing IR decoding, to
// 		report a successfully decoded repeat key message. This delays
// 		the generation of a key up event for previously generated
// 		key down event.
//
// 		Some IR protocols like NEC have a special IR message for
// 		repeating last button, for when a button is held down.
//
// 		The *ctx* should point to the lirc sample as passed into
// 		the program.
//
// 		This helper is only available is the kernel was compiled with
// 		the **CONFIG_BPF_LIRC_MODE2** configuration option set to
// 		"**y**".
// 	Return
// 		0
//
// long bpf_rc_keydown(void *ctx, u32 protocol, u64 scancode, u32 toggle)
// 	Description
// 		This helper is used in programs implementing IR decoding, to
// 		report a successfully decoded key press with *scancode*,
// 		*toggle* value in the given *protocol*. The scancode will be
// 		translated to a keycode using the rc keymap, and reported as
// 		an input key down event. After a period a key up event is
// 		generated. This period can be extended by calling either
// 		**bpf_rc_keydown**\ () again with the same values, or calling
// 		**bpf_rc_repeat**\ ().
//
// 		Some protocols include a toggle bit, in case the button was
// 		released and pressed again between consecutive scancodes.
//
// 		The *ctx* should point to the lirc sample as passed into
// 		the program.
//
// 		The *protocol* is the decoded protocol number (see
// 		**enum rc_proto** for some predefined values).
//
// 		This helper is only available is the kernel was compiled with
// 		the **CONFIG_BPF_LIRC_MODE2** configuration option set to
// 		"**y**".
// 	Return
// 		0
//
// u64 bpf_skb_cgroup_id(struct sk_buff *skb)
// 	Description
// 		Return the cgroup v2 id of the socket associated with the *skb*.
// 		This is roughly similar to the **bpf_get_cgroup_classid**\ ()
// 		helper for cgroup v1 by providing a tag resp. identifier that
// 		can be matched on or used for map lookups e.g. to implement
// 		policy. The cgroup v2 id of a given path in the hierarchy is
// 		exposed in user space through the f_handle API in order to get
// 		to the same 64-bit id.
//
// 		This helper can be used on TC egress path, but not on ingress,
// 		and is available only if the kernel was compiled with the
// 		**CONFIG_SOCK_CGROUP_DATA** configuration option.
// 	Return
// 		The id is returned or 0 in case the id could not be retrieved.
//
// u64 bpf_get_current_cgroup_id(void)
// 	Description
// 		Get the current cgroup id based on the cgroup within which
// 		the current task is running.
// 	Return
// 		A 64-bit integer containing the current cgroup id based
// 		on the cgroup within which the current task is running.
//
// void *bpf_get_local_storage(void *map, u64 flags)
// 	Description
// 		Get the pointer to the local storage area.
// 		The type and the size of the local storage is defined
// 		by the *map* argument.
// 		The *flags* meaning is specific for each map type,
// 		and has to be 0 for cgroup local storage.
//
// 		Depending on the BPF program type, a local storage area
// 		can be shared between multiple instances of the BPF program,
// 		running simultaneously.
//
// 		A user should care about the synchronization by himself.
// 		For example, by using the **BPF_ATOMIC** instructions to alter
// 		the shared data.
// 	Return
// 		A pointer to the local storage area.
//
// long bpf_sk_select_reuseport(struct sk_reuseport_md *reuse, struct bpf_map *map, void *key, u64 flags)
// 	Description
// 		Select a **SO_REUSEPORT** socket from a
// 		**BPF_MAP_TYPE_REUSEPORT_SOCKARRAY** *map*.
// 		It checks the selected socket is matching the incoming
// 		request in the socket buffer.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// u64 bpf_skb_ancestor_cgroup_id(struct sk_buff *skb, int ancestor_level)
// 	Description
// 		Return id of cgroup v2 that is ancestor of cgroup associated
// 		with the *skb* at the *ancestor_level*.  The root cgroup is at
// 		*ancestor_level* zero and each step down the hierarchy
// 		increments the level. If *ancestor_level* == level of cgroup
// 		associated with *skb*, then return value will be same as that
// 		of **bpf_skb_cgroup_id**\ ().
//
// 		The helper is useful to implement policies based on cgroups
// 		that are upper in hierarchy than immediate cgroup associated
// 		with *skb*.
//
// 		The format of returned id and helper limitations are same as in
// 		**bpf_skb_cgroup_id**\ ().
// 	Return
// 		The id is returned or 0 in case the id could not be retrieved.
//
// struct bpf_sock *bpf_sk_lookup_tcp(void *ctx, struct bpf_sock_tuple *tuple, u32 tuple_size, u64 netns, u64 flags)
// 	Description
// 		Look for TCP socket matching *tuple*, optionally in a child
// 		network namespace *netns*. The return value must be checked,
// 		and if non-**NULL**, released via **bpf_sk_release**\ ().
//
// 		The *ctx* should point to the context of the program, such as
// 		the skb or socket (depending on the hook in use). This is used
// 		to determine the base network namespace for the lookup.
//
// 		*tuple_size* must be one of:
//
// 		**sizeof**\ (*tuple*\ **->ipv4**)
// 			Look for an IPv4 socket.
// 		**sizeof**\ (*tuple*\ **->ipv6**)
// 			Look for an IPv6 socket.
//
// 		If the *netns* is a negative signed 32-bit integer, then the
// 		socket lookup table in the netns associated with the *ctx*
// 		will be used. For the TC hooks, this is the netns of the device
// 		in the skb. For socket hooks, this is the netns of the socket.
// 		If *netns* is any other signed 32-bit value greater than or
// 		equal to zero then it specifies the ID of the netns relative to
// 		the netns associated with the *ctx*. *netns* values beyond the
// 		range of 32-bit integers are reserved for future use.
//
// 		All values for *flags* are reserved for future usage, and must
// 		be left at zero.
//
// 		This helper is available only if the kernel was compiled with
// 		**CONFIG_NET** configuration option.
// 	Return
// 		Pointer to **struct bpf_sock**, or **NULL** in case of failure.
// 		For sockets with reuseport option, the **struct bpf_sock**
// 		result is from *reuse*\ **->socks**\ [] using the hash of the
// 		tuple.
//
// struct bpf_sock *bpf_sk_lookup_udp(void *ctx, struct bpf_sock_tuple *tuple, u32 tuple_size, u64 netns, u64 flags)
// 	Description
// 		Look for UDP socket matching *tuple*, optionally in a child
// 		network namespace *netns*. The return value must be checked,
// 		and if non-**NULL**, released via **bpf_sk_release**\ ().
//
// 		The *ctx* should point to the context of the program, such as
// 		the skb or socket (depending on the hook in use). This is used
// 		to determine the base network namespace for the lookup.
//
// 		*tuple_size* must be one of:
//
// 		**sizeof**\ (*tuple*\ **->ipv4**)
// 			Look for an IPv4 socket.
// 		**sizeof**\ (*tuple*\ **->ipv6**)
// 			Look for an IPv6 socket.
//
// 		If the *netns* is a negative signed 32-bit integer, then the
// 		socket lookup table in the netns associated with the *ctx*
// 		will be used. For the TC hooks, this is the netns of the device
// 		in the skb. For socket hooks, this is the netns of the socket.
// 		If *netns* is any other signed 32-bit value greater than or
// 		equal to zero then it specifies the ID of the netns relative to
// 		the netns associated with the *ctx*. *netns* values beyond the
// 		range of 32-bit integers are reserved for future use.
//
// 		All values for *flags* are reserved for future usage, and must
// 		be left at zero.
//
// 		This helper is available only if the kernel was compiled with
// 		**CONFIG_NET** configuration option.
// 	Return
// 		Pointer to **struct bpf_sock**, or **NULL** in case of failure.
// 		For sockets with reuseport option, the **struct bpf_sock**
// 		result is from *reuse*\ **->socks**\ [] using the hash of the
// 		tuple.
//
// long bpf_sk_release(void *sock)
// 	Description
// 		Release the reference held by *sock*. *sock* must be a
// 		non-**NULL** pointer that was returned from
// 		**bpf_sk_lookup_xxx**\ ().
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_map_push_elem(struct bpf_map *map, const void *value, u64 flags)
// 	Description
// 		Push an element *value* in *map*. *flags* is one of:
//
// 		**BPF_EXIST**
// 			If the queue/stack is full, the oldest element is
// 			removed to make room for this.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_map_pop_elem(struct bpf_map *map, void *value)
// 	Description
// 		Pop an element from *map*.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_map_peek_elem(struct bpf_map *map, void *value)
// 	Description
// 		Get an element from *map* without removing it.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_msg_push_data(struct sk_msg_buff *msg, u32 start, u32 len, u64 flags)
// 	Description
// 		For socket policies, insert *len* bytes into *msg* at offset
// 		*start*.
//
// 		If a program of type **BPF_PROG_TYPE_SK_MSG** is run on a
// 		*msg* it may want to insert metadata or options into the *msg*.
// 		This can later be read and used by any of the lower layer BPF
// 		hooks.
//
// 		This helper may fail if under memory pressure (a malloc
// 		fails) in these cases BPF programs will get an appropriate
// 		error and BPF programs will need to handle them.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_msg_pop_data(struct sk_msg_buff *msg, u32 start, u32 len, u64 flags)
// 	Description
// 		Will remove *len* bytes from a *msg* starting at byte *start*.
// 		This may result in **ENOMEM** errors under certain situations if
// 		an allocation and copy are required due to a full ring buffer.
// 		However, the helper will try to avoid doing the allocation
// 		if possible. Other errors can occur if input parameters are
// 		invalid either due to *start* byte not being valid part of *msg*
// 		payload and/or *pop* value being to large.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_rc_pointer_rel(void *ctx, s32 rel_x, s32 rel_y)
// 	Description
// 		This helper is used in programs implementing IR decoding, to
// 		report a successfully decoded pointer movement.
//
// 		The *ctx* should point to the lirc sample as passed into
// 		the program.
//
// 		This helper is only available is the kernel was compiled with
// 		the **CONFIG_BPF_LIRC_MODE2** configuration option set to
// 		"**y**".
// 	Return
// 		0
//
// long bpf_spin_lock(struct bpf_spin_lock *lock)
// 	Description
// 		Acquire a spinlock represented by the pointer *lock*, which is
// 		stored as part of a value of a map. Taking the lock allows to
// 		safely update the rest of the fields in that value. The
// 		spinlock can (and must) later be released with a call to
// 		**bpf_spin_unlock**\ (\ *lock*\ ).
//
// 		Spinlocks in BPF programs come with a number of restrictions
// 		and constraints:
//
// 		* **bpf_spin_lock** objects are only allowed inside maps of
// 		  types **BPF_MAP_TYPE_HASH** and **BPF_MAP_TYPE_ARRAY** (this
// 		  list could be extended in the future).
// 		* BTF description of the map is mandatory.
// 		* The BPF program can take ONE lock at a time, since taking two
// 		  or more could cause dead locks.
// 		* Only one **struct bpf_spin_lock** is allowed per map element.
// 		* When the lock is taken, calls (either BPF to BPF or helpers)
// 		  are not allowed.
// 		* The **BPF_LD_ABS** and **BPF_LD_IND** instructions are not
// 		  allowed inside a spinlock-ed region.
// 		* The BPF program MUST call **bpf_spin_unlock**\ () to release
// 		  the lock, on all execution paths, before it returns.
// 		* The BPF program can access **struct bpf_spin_lock** only via
// 		  the **bpf_spin_lock**\ () and **bpf_spin_unlock**\ ()
// 		  helpers. Loading or storing data into the **struct
// 		  bpf_spin_lock** *lock*\ **;** field of a map is not allowed.
// 		* To use the **bpf_spin_lock**\ () helper, the BTF description
// 		  of the map value must be a struct and have **struct
// 		  bpf_spin_lock** *anyname*\ **;** field at the top level.
// 		  Nested lock inside another struct is not allowed.
// 		* The **struct bpf_spin_lock** *lock* field in a map value must
// 		  be aligned on a multiple of 4 bytes in that value.
// 		* Syscall with command **BPF_MAP_LOOKUP_ELEM** does not copy
// 		  the **bpf_spin_lock** field to user space.
// 		* Syscall with command **BPF_MAP_UPDATE_ELEM**, or update from
// 		  a BPF program, do not update the **bpf_spin_lock** field.
// 		* **bpf_spin_lock** cannot be on the stack or inside a
// 		  networking packet (it can only be inside of a map values).
// 		* **bpf_spin_lock** is available to root only.
// 		* Tracing programs and socket filter programs cannot use
// 		  **bpf_spin_lock**\ () due to insufficient preemption checks
// 		  (but this may change in the future).
// 		* **bpf_spin_lock** is not allowed in inner maps of map-in-map.
// 	Return
// 		0
//
// long bpf_spin_unlock(struct bpf_spin_lock *lock)
// 	Description
// 		Release the *lock* previously locked by a call to
// 		**bpf_spin_lock**\ (\ *lock*\ ).
// 	Return
// 		0
//
// struct bpf_sock *bpf_sk_fullsock(struct bpf_sock *sk)
// 	Description
// 		This helper gets a **struct bpf_sock** pointer such
// 		that all the fields in this **bpf_sock** can be accessed.
// 	Return
// 		A **struct bpf_sock** pointer on success, or **NULL** in
// 		case of failure.
//
// struct bpf_tcp_sock *bpf_tcp_sock(struct bpf_sock *sk)
// 	Description
// 		This helper gets a **struct bpf_tcp_sock** pointer from a
// 		**struct bpf_sock** pointer.
// 	Return
// 		A **struct bpf_tcp_sock** pointer on success, or **NULL** in
// 		case of failure.
//
// long bpf_skb_ecn_set_ce(struct sk_buff *skb)
// 	Description
// 		Set ECN (Explicit Congestion Notification) field of IP header
// 		to **CE** (Congestion Encountered) if current value is **ECT**
// 		(ECN Capable Transport). Otherwise, do nothing. Works with IPv6
// 		and IPv4.
// 	Return
// 		1 if the **CE** flag is set (either by the current helper call
// 		or because it was already present), 0 if it is not set.
//
// struct bpf_sock *bpf_get_listener_sock(struct bpf_sock *sk)
// 	Description
// 		Return a **struct bpf_sock** pointer in **TCP_LISTEN** state.
// 		**bpf_sk_release**\ () is unnecessary and not allowed.
// 	Return
// 		A **struct bpf_sock** pointer on success, or **NULL** in
// 		case of failure.
//
// struct bpf_sock *bpf_skc_lookup_tcp(void *ctx, struct bpf_sock_tuple *tuple, u32 tuple_size, u64 netns, u64 flags)
// 	Description
// 		Look for TCP socket matching *tuple*, optionally in a child
// 		network namespace *netns*. The return value must be checked,
// 		and if non-**NULL**, released via **bpf_sk_release**\ ().
//
// 		This function is identical to **bpf_sk_lookup_tcp**\ (), except
// 		that it also returns timewait or request sockets. Use
// 		**bpf_sk_fullsock**\ () or **bpf_tcp_sock**\ () to access the
// 		full structure.
//
// 		This helper is available only if the kernel was compiled with
// 		**CONFIG_NET** configuration option.
// 	Return
// 		Pointer to **struct bpf_sock**, or **NULL** in case of failure.
// 		For sockets with reuseport option, the **struct bpf_sock**
// 		result is from *reuse*\ **->socks**\ [] using the hash of the
// 		tuple.
//
// long bpf_tcp_check_syncookie(void *sk, void *iph, u32 iph_len, struct tcphdr *th, u32 th_len)
// 	Description
// 		Check whether *iph* and *th* contain a valid SYN cookie ACK for
// 		the listening socket in *sk*.
//
// 		*iph* points to the start of the IPv4 or IPv6 header, while
// 		*iph_len* contains **sizeof**\ (**struct iphdr**) or
// 		**sizeof**\ (**struct ipv6hdr**).
//
// 		*th* points to the start of the TCP header, while *th_len*
// 		contains the length of the TCP header (at least
// 		**sizeof**\ (**struct tcphdr**)).
// 	Return
// 		0 if *iph* and *th* are a valid SYN cookie ACK, or a negative
// 		error otherwise.
//
// long bpf_sysctl_get_name(struct bpf_sysctl *ctx, char *buf, size_t buf_len, u64 flags)
// 	Description
// 		Get name of sysctl in /proc/sys/ and copy it into provided by
// 		program buffer *buf* of size *buf_len*.
//
// 		The buffer is always NUL terminated, unless it's zero-sized.
//
// 		If *flags* is zero, full name (e.g. "net/ipv4/tcp_mem") is
// 		copied. Use **BPF_F_SYSCTL_BASE_NAME** flag to copy base name
// 		only (e.g. "tcp_mem").
// 	Return
// 		Number of character copied (not including the trailing NUL).
//
// 		**-E2BIG** if the buffer wasn't big enough (*buf* will contain
// 		truncated name in this case).
//
// long bpf_sysctl_get_current_value(struct bpf_sysctl *ctx, char *buf, size_t buf_len)
// 	Description
// 		Get current value of sysctl as it is presented in /proc/sys
// 		(incl. newline, etc), and copy it as a string into provided
// 		by program buffer *buf* of size *buf_len*.
//
// 		The whole value is copied, no matter what file position user
// 		space issued e.g. sys_read at.
//
// 		The buffer is always NUL terminated, unless it's zero-sized.
// 	Return
// 		Number of character copied (not including the trailing NUL).
//
// 		**-E2BIG** if the buffer wasn't big enough (*buf* will contain
// 		truncated name in this case).
//
// 		**-EINVAL** if current value was unavailable, e.g. because
// 		sysctl is uninitialized and read returns -EIO for it.
//
// long bpf_sysctl_get_new_value(struct bpf_sysctl *ctx, char *buf, size_t buf_len)
// 	Description
// 		Get new value being written by user space to sysctl (before
// 		the actual write happens) and copy it as a string into
// 		provided by program buffer *buf* of size *buf_len*.
//
// 		User space may write new value at file position > 0.
//
// 		The buffer is always NUL terminated, unless it's zero-sized.
// 	Return
// 		Number of character copied (not including the trailing NUL).
//
// 		**-E2BIG** if the buffer wasn't big enough (*buf* will contain
// 		truncated name in this case).
//
// 		**-EINVAL** if sysctl is being read.
//
// long bpf_sysctl_set_new_value(struct bpf_sysctl *ctx, const char *buf, size_t buf_len)
// 	Description
// 		Override new value being written by user space to sysctl with
// 		value provided by program in buffer *buf* of size *buf_len*.
//
// 		*buf* should contain a string in same form as provided by user
// 		space on sysctl write.
//
// 		User space may write new value at file position > 0. To override
// 		the whole sysctl value file position should be set to zero.
// 	Return
// 		0 on success.
//
// 		**-E2BIG** if the *buf_len* is too big.
//
// 		**-EINVAL** if sysctl is being read.
//
// long bpf_strtol(const char *buf, size_t buf_len, u64 flags, long *res)
// 	Description
// 		Convert the initial part of the string from buffer *buf* of
// 		size *buf_len* to a long integer according to the given base
// 		and save the result in *res*.
//
// 		The string may begin with an arbitrary amount of white space
// 		(as determined by **isspace**\ (3)) followed by a single
// 		optional '**-**' sign.
//
// 		Five least significant bits of *flags* encode base, other bits
// 		are currently unused.
//
// 		Base must be either 8, 10, 16 or 0 to detect it automatically
// 		similar to user space **strtol**\ (3).
// 	Return
// 		Number of characters consumed on success. Must be positive but
// 		no more than *buf_len*.
//
// 		**-EINVAL** if no valid digits were found or unsupported base
// 		was provided.
//
// 		**-ERANGE** if resulting value was out of range.
//
// long bpf_strtoul(const char *buf, size_t buf_len, u64 flags, unsigned long *res)
// 	Description
// 		Convert the initial part of the string from buffer *buf* of
// 		size *buf_len* to an unsigned long integer according to the
// 		given base and save the result in *res*.
//
// 		The string may begin with an arbitrary amount of white space
// 		(as determined by **isspace**\ (3)).
//
// 		Five least significant bits of *flags* encode base, other bits
// 		are currently unused.
//
// 		Base must be either 8, 10, 16 or 0 to detect it automatically
// 		similar to user space **strtoul**\ (3).
// 	Return
// 		Number of characters consumed on success. Must be positive but
// 		no more than *buf_len*.
//
// 		**-EINVAL** if no valid digits were found or unsupported base
// 		was provided.
//
// 		**-ERANGE** if resulting value was out of range.
//
// void *bpf_sk_storage_get(struct bpf_map *map, void *sk, void *value, u64 flags)
// 	Description
// 		Get a bpf-local-storage from a *sk*.
//
// 		Logically, it could be thought of getting the value from
// 		a *map* with *sk* as the **key**.  From this
// 		perspective,  the usage is not much different from
// 		**bpf_map_lookup_elem**\ (*map*, **&**\ *sk*) except this
// 		helper enforces the key must be a full socket and the map must
// 		be a **BPF_MAP_TYPE_SK_STORAGE** also.
//
// 		Underneath, the value is stored locally at *sk* instead of
// 		the *map*.  The *map* is used as the bpf-local-storage
// 		"type". The bpf-local-storage "type" (i.e. the *map*) is
// 		searched against all bpf-local-storages residing at *sk*.
//
// 		*sk* is a kernel **struct sock** pointer for LSM program.
// 		*sk* is a **struct bpf_sock** pointer for other program types.
//
// 		An optional *flags* (**BPF_SK_STORAGE_GET_F_CREATE**) can be
// 		used such that a new bpf-local-storage will be
// 		created if one does not exist.  *value* can be used
// 		together with **BPF_SK_STORAGE_GET_F_CREATE** to specify
// 		the initial value of a bpf-local-storage.  If *value* is
// 		**NULL**, the new bpf-local-storage will be zero initialized.
// 	Return
// 		A bpf-local-storage pointer is returned on success.
//
// 		**NULL** if not found or there was an error in adding
// 		a new bpf-local-storage.
//
// long bpf_sk_storage_delete(struct bpf_map *map, void *sk)
// 	Description
// 		Delete a bpf-local-storage from a *sk*.
// 	Return
// 		0 on success.
//
// 		**-ENOENT** if the bpf-local-storage cannot be found.
// 		**-EINVAL** if sk is not a fullsock (e.g. a request_sock).
//
// long bpf_send_signal(u32 sig)
// 	Description
// 		Send signal *sig* to the process of the current task.
// 		The signal may be delivered to any of this process's threads.
// 	Return
// 		0 on success or successfully queued.
//
// 		**-EBUSY** if work queue under nmi is full.
//
// 		**-EINVAL** if *sig* is invalid.
//
// 		**-EPERM** if no permission to send the *sig*.
//
// 		**-EAGAIN** if bpf program can try again.
//
// s64 bpf_tcp_gen_syncookie(void *sk, void *iph, u32 iph_len, struct tcphdr *th, u32 th_len)
// 	Description
// 		Try to issue a SYN cookie for the packet with corresponding
// 		IP/TCP headers, *iph* and *th*, on the listening socket in *sk*.
//
// 		*iph* points to the start of the IPv4 or IPv6 header, while
// 		*iph_len* contains **sizeof**\ (**struct iphdr**) or
// 		**sizeof**\ (**struct ipv6hdr**).
//
// 		*th* points to the start of the TCP header, while *th_len*
// 		contains the length of the TCP header with options (at least
// 		**sizeof**\ (**struct tcphdr**)).
// 	Return
// 		On success, lower 32 bits hold the generated SYN cookie in
// 		followed by 16 bits which hold the MSS value for that cookie,
// 		and the top 16 bits are unused.
//
// 		On failure, the returned value is one of the following:
//
// 		**-EINVAL** SYN cookie cannot be issued due to error
//
// 		**-ENOENT** SYN cookie should not be issued (no SYN flood)
//
// 		**-EOPNOTSUPP** kernel configuration does not enable SYN cookies
//
// 		**-EPROTONOSUPPORT** IP packet version is not 4 or 6
//
// long bpf_skb_output(void *ctx, struct bpf_map *map, u64 flags, void *data, u64 size)
// 	Description
// 		Write raw *data* blob into a special BPF perf event held by
// 		*map* of type **BPF_MAP_TYPE_PERF_EVENT_ARRAY**. This perf
// 		event must have the following attributes: **PERF_SAMPLE_RAW**
// 		as **sample_type**, **PERF_TYPE_SOFTWARE** as **type**, and
// 		**PERF_COUNT_SW_BPF_OUTPUT** as **config**.
//
// 		The *flags* are used to indicate the index in *map* for which
// 		the value must be put, masked with **BPF_F_INDEX_MASK**.
// 		Alternatively, *flags* can be set to **BPF_F_CURRENT_CPU**
// 		to indicate that the index of the current CPU core should be
// 		used.
//
// 		The value to write, of *size*, is passed through eBPF stack and
// 		pointed by *data*.
//
// 		*ctx* is a pointer to in-kernel struct sk_buff.
//
// 		This helper is similar to **bpf_perf_event_output**\ () but
// 		restricted to raw_tracepoint bpf programs.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_probe_read_user(void *dst, u32 size, const void *unsafe_ptr)
// 	Description
// 		Safely attempt to read *size* bytes from user space address
// 		*unsafe_ptr* and store the data in *dst*.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_probe_read_kernel(void *dst, u32 size, const void *unsafe_ptr)
// 	Description
// 		Safely attempt to read *size* bytes from kernel space address
// 		*unsafe_ptr* and store the data in *dst*.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_probe_read_user_str(void *dst, u32 size, const void *unsafe_ptr)
// 	Description
// 		Copy a NUL terminated string from an unsafe user address
// 		*unsafe_ptr* to *dst*. The *size* should include the
// 		terminating NUL byte. In case the string length is smaller than
// 		*size*, the target is not padded with further NUL bytes. If the
// 		string length is larger than *size*, just *size*-1 bytes are
// 		copied and the last byte is set to NUL.
//
// 		On success, returns the number of bytes that were written,
// 		including the terminal NUL. This makes this helper useful in
// 		tracing programs for reading strings, and more importantly to
// 		get its length at runtime. See the following snippet:
//
// 		::
//
// 			SEC("kprobe/sys_open")
// 			void bpf_sys_open(struct pt_regs *ctx)
// 			{
// 			        char buf[PATHLEN]; // PATHLEN is defined to 256
// 			        int res = bpf_probe_read_user_str(buf, sizeof(buf),
// 				                                  ctx->di);
//
// 				// Consume buf, for example push it to
// 				// userspace via bpf_perf_event_output(); we
// 				// can use res (the string length) as event
// 				// size, after checking its boundaries.
// 			}
//
// 		In comparison, using **bpf_probe_read_user**\ () helper here
// 		instead to read the string would require to estimate the length
// 		at compile time, and would often result in copying more memory
// 		than necessary.
//
// 		Another useful use case is when parsing individual process
// 		arguments or individual environment variables navigating
// 		*current*\ **->mm->arg_start** and *current* * 		**->mm->env_start**: using this helper and the return value,
// 		one can quickly iterate at the right offset of the memory area.
// 	Return
// 		On success, the strictly positive length of the output string,
// 		including the trailing NUL character. On error, a negative
// 		value.
//
// long bpf_probe_read_kernel_str(void *dst, u32 size, const void *unsafe_ptr)
// 	Description
// 		Copy a NUL terminated string from an unsafe kernel address *unsafe_ptr*
// 		to *dst*. Same semantics as with **bpf_probe_read_user_str**\ () apply.
// 	Return
// 		On success, the strictly positive length of the string, including
// 		the trailing NUL character. On error, a negative value.
//
// long bpf_tcp_send_ack(void *tp, u32 rcv_nxt)
// 	Description
// 		Send out a tcp-ack. *tp* is the in-kernel struct **tcp_sock**.
// 		*rcv_nxt* is the ack_seq to be sent out.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_send_signal_thread(u32 sig)
// 	Description
// 		Send signal *sig* to the thread corresponding to the current task.
// 	Return
// 		0 on success or successfully queued.
//
// 		**-EBUSY** if work queue under nmi is full.
//
// 		**-EINVAL** if *sig* is invalid.
//
// 		**-EPERM** if no permission to send the *sig*.
//
// 		**-EAGAIN** if bpf program can try again.
//
// u64 bpf_jiffies64(void)
// 	Description
// 		Obtain the 64bit jiffies
// 	Return
// 		The 64 bit jiffies
//
// long bpf_read_branch_records(struct bpf_perf_event_data *ctx, void *buf, u32 size, u64 flags)
// 	Description
// 		For an eBPF program attached to a perf event, retrieve the
// 		branch records (**struct perf_branch_entry**) associated to *ctx*
// 		and store it in the buffer pointed by *buf* up to size
// 		*size* bytes.
// 	Return
// 		On success, number of bytes written to *buf*. On error, a
// 		negative value.
//
// 		The *flags* can be set to **BPF_F_GET_BRANCH_RECORDS_SIZE** to
// 		instead return the number of bytes required to store all the
// 		branch entries. If this flag is set, *buf* may be NULL.
//
// 		**-EINVAL** if arguments invalid or **size** not a multiple
// 		of **sizeof**\ (**struct perf_branch_entry**\ ).
//
// 		**-ENOENT** if architecture does not support branch records.
//
// long bpf_get_ns_current_pid_tgid(u64 dev, u64 ino, struct bpf_pidns_info *nsdata, u32 size)
// 	Description
// 		Returns 0 on success, values for *pid* and *tgid* as seen from the current
// 		*namespace* will be returned in *nsdata*.
// 	Return
// 		0 on success, or one of the following in case of failure:
//
// 		**-EINVAL** if dev and inum supplied don't match dev_t and inode number
//              with nsfs of current task, or if dev conversion to dev_t lost high bits.
//
// 		**-ENOENT** if pidns does not exists for the current task.
//
// long bpf_xdp_output(void *ctx, struct bpf_map *map, u64 flags, void *data, u64 size)
// 	Description
// 		Write raw *data* blob into a special BPF perf event held by
// 		*map* of type **BPF_MAP_TYPE_PERF_EVENT_ARRAY**. This perf
// 		event must have the following attributes: **PERF_SAMPLE_RAW**
// 		as **sample_type**, **PERF_TYPE_SOFTWARE** as **type**, and
// 		**PERF_COUNT_SW_BPF_OUTPUT** as **config**.
//
// 		The *flags* are used to indicate the index in *map* for which
// 		the value must be put, masked with **BPF_F_INDEX_MASK**.
// 		Alternatively, *flags* can be set to **BPF_F_CURRENT_CPU**
// 		to indicate that the index of the current CPU core should be
// 		used.
//
// 		The value to write, of *size*, is passed through eBPF stack and
// 		pointed by *data*.
//
// 		*ctx* is a pointer to in-kernel struct xdp_buff.
//
// 		This helper is similar to **bpf_perf_eventoutput**\ () but
// 		restricted to raw_tracepoint bpf programs.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// u64 bpf_get_netns_cookie(void *ctx)
// 	Description
// 		Retrieve the cookie (generated by the kernel) of the network
// 		namespace the input *ctx* is associated with. The network
// 		namespace cookie remains stable for its lifetime and provides
// 		a global identifier that can be assumed unique. If *ctx* is
// 		NULL, then the helper returns the cookie for the initial
// 		network namespace. The cookie itself is very similar to that
// 		of **bpf_get_socket_cookie**\ () helper, but for network
// 		namespaces instead of sockets.
// 	Return
// 		A 8-byte long opaque number.
//
// u64 bpf_get_current_ancestor_cgroup_id(int ancestor_level)
// 	Description
// 		Return id of cgroup v2 that is ancestor of the cgroup associated
// 		with the current task at the *ancestor_level*. The root cgroup
// 		is at *ancestor_level* zero and each step down the hierarchy
// 		increments the level. If *ancestor_level* == level of cgroup
// 		associated with the current task, then return value will be the
// 		same as that of **bpf_get_current_cgroup_id**\ ().
//
// 		The helper is useful to implement policies based on cgroups
// 		that are upper in hierarchy than immediate cgroup associated
// 		with the current task.
//
// 		The format of returned id and helper limitations are same as in
// 		**bpf_get_current_cgroup_id**\ ().
// 	Return
// 		The id is returned or 0 in case the id could not be retrieved.
//
// long bpf_sk_assign(struct sk_buff *skb, void *sk, u64 flags)
// 	Description
// 		Helper is overloaded depending on BPF program type. This
// 		description applies to **BPF_PROG_TYPE_SCHED_CLS** and
// 		**BPF_PROG_TYPE_SCHED_ACT** programs.
//
// 		Assign the *sk* to the *skb*. When combined with appropriate
// 		routing configuration to receive the packet towards the socket,
// 		will cause *skb* to be delivered to the specified socket.
// 		Subsequent redirection of *skb* via  **bpf_redirect**\ (),
// 		**bpf_clone_redirect**\ () or other methods outside of BPF may
// 		interfere with successful delivery to the socket.
//
// 		This operation is only valid from TC ingress path.
//
// 		The *flags* argument must be zero.
// 	Return
// 		0 on success, or a negative error in case of failure:
//
// 		**-EINVAL** if specified *flags* are not supported.
//
// 		**-ENOENT** if the socket is unavailable for assignment.
//
// 		**-ENETUNREACH** if the socket is unreachable (wrong netns).
//
// 		**-EOPNOTSUPP** if the operation is not supported, for example
// 		a call from outside of TC ingress.
//
// long bpf_sk_assign(struct bpf_sk_lookup *ctx, struct bpf_sock *sk, u64 flags)
// 	Description
// 		Helper is overloaded depending on BPF program type. This
// 		description applies to **BPF_PROG_TYPE_SK_LOOKUP** programs.
//
// 		Select the *sk* as a result of a socket lookup.
//
// 		For the operation to succeed passed socket must be compatible
// 		with the packet description provided by the *ctx* object.
//
// 		L4 protocol (**IPPROTO_TCP** or **IPPROTO_UDP**) must
// 		be an exact match. While IP family (**AF_INET** or
// 		**AF_INET6**) must be compatible, that is IPv6 sockets
// 		that are not v6-only can be selected for IPv4 packets.
//
// 		Only TCP listeners and UDP unconnected sockets can be
// 		selected. *sk* can also be NULL to reset any previous
// 		selection.
//
// 		*flags* argument can combination of following values:
//
// 		* **BPF_SK_LOOKUP_F_REPLACE** to override the previous
// 		  socket selection, potentially done by a BPF program
// 		  that ran before us.
//
// 		* **BPF_SK_LOOKUP_F_NO_REUSEPORT** to skip
// 		  load-balancing within reuseport group for the socket
// 		  being selected.
//
// 		On success *ctx->sk* will point to the selected socket.
//
// 	Return
// 		0 on success, or a negative errno in case of failure.
//
// 		* **-EAFNOSUPPORT** if socket family (*sk->family*) is
// 		  not compatible with packet family (*ctx->family*).
//
// 		* **-EEXIST** if socket has been already selected,
// 		  potentially by another program, and
// 		  **BPF_SK_LOOKUP_F_REPLACE** flag was not specified.
//
// 		* **-EINVAL** if unsupported flags were specified.
//
// 		* **-EPROTOTYPE** if socket L4 protocol
// 		  (*sk->protocol*) doesn't match packet protocol
// 		  (*ctx->protocol*).
//
// 		* **-ESOCKTNOSUPPORT** if socket is not in allowed
// 		  state (TCP listening or UDP unconnected).
//
// u64 bpf_ktime_get_boot_ns(void)
// 	Description
// 		Return the time elapsed since system boot, in nanoseconds.
// 		Does include the time the system was suspended.
// 		See: **clock_gettime**\ (**CLOCK_BOOTTIME**)
// 	Return
// 		Current *ktime*.
//
// long bpf_seq_printf(struct seq_file *m, const char *fmt, u32 fmt_size, const void *data, u32 data_len)
// 	Description
// 		**bpf_seq_printf**\ () uses seq_file **seq_printf**\ () to print
// 		out the format string.
// 		The *m* represents the seq_file. The *fmt* and *fmt_size* are for
// 		the format string itself. The *data* and *data_len* are format string
// 		arguments. The *data* are a **u64** array and corresponding format string
// 		values are stored in the array. For strings and pointers where pointees
// 		are accessed, only the pointer values are stored in the *data* array.
// 		The *data_len* is the size of *data* in bytes - must be a multiple of 8.
//
// 		Formats **%s**, **%p{i,I}{4,6}** requires to read kernel memory.
// 		Reading kernel memory may fail due to either invalid address or
// 		valid address but requiring a major memory fault. If reading kernel memory
// 		fails, the string for **%s** will be an empty string, and the ip
// 		address for **%p{i,I}{4,6}** will be 0. Not returning error to
// 		bpf program is consistent with what **bpf_trace_printk**\ () does for now.
// 	Return
// 		0 on success, or a negative error in case of failure:
//
// 		**-EBUSY** if per-CPU memory copy buffer is busy, can try again
// 		by returning 1 from bpf program.
//
// 		**-EINVAL** if arguments are invalid, or if *fmt* is invalid/unsupported.
//
// 		**-E2BIG** if *fmt* contains too many format specifiers.
//
// 		**-EOVERFLOW** if an overflow happened: The same object will be tried again.
//
// long bpf_seq_write(struct seq_file *m, const void *data, u32 len)
// 	Description
// 		**bpf_seq_write**\ () uses seq_file **seq_write**\ () to write the data.
// 		The *m* represents the seq_file. The *data* and *len* represent the
// 		data to write in bytes.
// 	Return
// 		0 on success, or a negative error in case of failure:
//
// 		**-EOVERFLOW** if an overflow happened: The same object will be tried again.
//
// u64 bpf_sk_cgroup_id(void *sk)
// 	Description
// 		Return the cgroup v2 id of the socket *sk*.
//
// 		*sk* must be a non-**NULL** pointer to a socket, e.g. one
// 		returned from **bpf_sk_lookup_xxx**\ (),
// 		**bpf_sk_fullsock**\ (), etc. The format of returned id is
// 		same as in **bpf_skb_cgroup_id**\ ().
//
// 		This helper is available only if the kernel was compiled with
// 		the **CONFIG_SOCK_CGROUP_DATA** configuration option.
// 	Return
// 		The id is returned or 0 in case the id could not be retrieved.
//
// u64 bpf_sk_ancestor_cgroup_id(void *sk, int ancestor_level)
// 	Description
// 		Return id of cgroup v2 that is ancestor of cgroup associated
// 		with the *sk* at the *ancestor_level*.  The root cgroup is at
// 		*ancestor_level* zero and each step down the hierarchy
// 		increments the level. If *ancestor_level* == level of cgroup
// 		associated with *sk*, then return value will be same as that
// 		of **bpf_sk_cgroup_id**\ ().
//
// 		The helper is useful to implement policies based on cgroups
// 		that are upper in hierarchy than immediate cgroup associated
// 		with *sk*.
//
// 		The format of returned id and helper limitations are same as in
// 		**bpf_sk_cgroup_id**\ ().
// 	Return
// 		The id is returned or 0 in case the id could not be retrieved.
//
// long bpf_ringbuf_output(void *ringbuf, void *data, u64 size, u64 flags)
// 	Description
// 		Copy *size* bytes from *data* into a ring buffer *ringbuf*.
// 		If **BPF_RB_NO_WAKEUP** is specified in *flags*, no notification
// 		of new data availability is sent.
// 		If **BPF_RB_FORCE_WAKEUP** is specified in *flags*, notification
// 		of new data availability is sent unconditionally.
// 		If **0** is specified in *flags*, an adaptive notification
// 		of new data availability is sent.
//
// 		An adaptive notification is a notification sent whenever the user-space
// 		process has caught up and consumed all available payloads. In case the user-space
// 		process is still processing a previous payload, then no notification is needed
// 		as it will process the newly added payload automatically.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// void *bpf_ringbuf_reserve(void *ringbuf, u64 size, u64 flags)
// 	Description
// 		Reserve *size* bytes of payload in a ring buffer *ringbuf*.
// 		*flags* must be 0.
// 	Return
// 		Valid pointer with *size* bytes of memory available; NULL,
// 		otherwise.
//
// void bpf_ringbuf_submit(void *data, u64 flags)
// 	Description
// 		Submit reserved ring buffer sample, pointed to by *data*.
// 		If **BPF_RB_NO_WAKEUP** is specified in *flags*, no notification
// 		of new data availability is sent.
// 		If **BPF_RB_FORCE_WAKEUP** is specified in *flags*, notification
// 		of new data availability is sent unconditionally.
// 		If **0** is specified in *flags*, an adaptive notification
// 		of new data availability is sent.
//
// 		See 'bpf_ringbuf_output()' for the definition of adaptive notification.
// 	Return
// 		Nothing. Always succeeds.
//
// void bpf_ringbuf_discard(void *data, u64 flags)
// 	Description
// 		Discard reserved ring buffer sample, pointed to by *data*.
// 		If **BPF_RB_NO_WAKEUP** is specified in *flags*, no notification
// 		of new data availability is sent. Discarded records remain in
// 		the ring buffer until consumed by user space, so a later submit
// 		using adaptive wakeup might not wake up the consumer.
// 		If **BPF_RB_FORCE_WAKEUP** is specified in *flags*, notification
// 		of new data availability is sent unconditionally.
// 		If **0** is specified in *flags*, an adaptive notification
// 		of new data availability is sent.
//
// 		See 'bpf_ringbuf_output()' for the definition of adaptive notification.
// 	Return
// 		Nothing. Always succeeds.
//
// u64 bpf_ringbuf_query(void *ringbuf, u64 flags)
// 	Description
// 		Query various characteristics of provided ring buffer. What
// 		exactly is queries is determined by *flags*:
//
// 		* **BPF_RB_AVAIL_DATA**: Amount of data not yet consumed.
// 		* **BPF_RB_RING_SIZE**: The size of ring buffer.
// 		* **BPF_RB_CONS_POS**: Consumer position (can wrap around).
// 		* **BPF_RB_PROD_POS**: Producer(s) position (can wrap around).
// 		* **BPF_RB_OVERWRITE_POS**: Overwrite position (can wrap around).
//
// 		Data returned is just a momentary snapshot of actual values
// 		and could be inaccurate, so this facility should be used to
// 		power heuristics and for reporting, not to make 100% correct
// 		calculation.
// 	Return
// 		Requested value, or 0, if *flags* are not recognized.
//
// long bpf_csum_level(struct sk_buff *skb, u64 level)
// 	Description
// 		Change the skbs checksum level by one layer up or down, or
// 		reset it entirely to none in order to have the stack perform
// 		checksum validation. The level is applicable to the following
// 		protocols: TCP, UDP, GRE, SCTP, FCOE. For example, a decap of
// 		| ETH | IP | UDP | GUE | IP | TCP | into | ETH | IP | TCP |
// 		through **bpf_skb_adjust_room**\ () helper with passing in
// 		**BPF_F_ADJ_ROOM_NO_CSUM_RESET** flag would require one	call
// 		to **bpf_csum_level**\ () with **BPF_CSUM_LEVEL_DEC** since
// 		the UDP header is removed. Similarly, an encap of the latter
// 		into the former could be accompanied by a helper call to
// 		**bpf_csum_level**\ () with **BPF_CSUM_LEVEL_INC** if the
// 		skb is still intended to be processed in higher layers of the
// 		stack instead of just egressing at tc.
//
// 		There are three supported level settings at this time:
//
// 		* **BPF_CSUM_LEVEL_INC**: Increases skb->csum_level for skbs
// 		  with CHECKSUM_UNNECESSARY.
// 		* **BPF_CSUM_LEVEL_DEC**: Decreases skb->csum_level for skbs
// 		  with CHECKSUM_UNNECESSARY.
// 		* **BPF_CSUM_LEVEL_RESET**: Resets skb->csum_level to 0 and
// 		  sets CHECKSUM_NONE to force checksum validation by the stack.
// 		* **BPF_CSUM_LEVEL_QUERY**: No-op, returns the current
// 		  skb->csum_level.
// 	Return
// 		0 on success, or a negative error in case of failure. In the
// 		case of **BPF_CSUM_LEVEL_QUERY**, the current skb->csum_level
// 		is returned or the error code -EACCES in case the skb is not
// 		subject to CHECKSUM_UNNECESSARY.
//
// struct tcp6_sock *bpf_skc_to_tcp6_sock(void *sk)
// 	Description
// 		Dynamically cast a *sk* pointer to a *tcp6_sock* pointer.
// 	Return
// 		*sk* if casting is valid, or **NULL** otherwise.
//
// struct tcp_sock *bpf_skc_to_tcp_sock(void *sk)
// 	Description
// 		Dynamically cast a *sk* pointer to a *tcp_sock* pointer.
// 	Return
// 		*sk* if casting is valid, or **NULL** otherwise.
//
// struct tcp_timewait_sock *bpf_skc_to_tcp_timewait_sock(void *sk)
// 	Description
// 		Dynamically cast a *sk* pointer to a *tcp_timewait_sock* pointer.
// 	Return
// 		*sk* if casting is valid, or **NULL** otherwise.
//
// struct tcp_request_sock *bpf_skc_to_tcp_request_sock(void *sk)
// 	Description
// 		Dynamically cast a *sk* pointer to a *tcp_request_sock* pointer.
// 	Return
// 		*sk* if casting is valid, or **NULL** otherwise.
//
// struct udp6_sock *bpf_skc_to_udp6_sock(void *sk)
// 	Description
// 		Dynamically cast a *sk* pointer to a *udp6_sock* pointer.
// 	Return
// 		*sk* if casting is valid, or **NULL** otherwise.
//
// long bpf_get_task_stack(struct task_struct *task, void *buf, u32 size, u64 flags)
// 	Description
// 		Return a user or a kernel stack in bpf program provided buffer.
// 		Note: the user stack will only be populated if the *task* is
// 		the current task; all other tasks will return -EOPNOTSUPP.
// 		To achieve this, the helper needs *task*, which is a valid
// 		pointer to **struct task_struct**. To store the stacktrace, the
// 		bpf program provides *buf* with a nonnegative *size*.
//
// 		The last argument, *flags*, holds the number of stack frames to
// 		skip (from 0 to 255), masked with
// 		**BPF_F_SKIP_FIELD_MASK**. The next bits can be used to set
// 		the following flags:
//
// 		**BPF_F_USER_STACK**
// 			Collect a user space stack instead of a kernel stack.
// 			The *task* must be the current task.
// 		**BPF_F_USER_BUILD_ID**
// 			Collect buildid+offset instead of ips for user stack,
// 			only valid if **BPF_F_USER_STACK** is also specified.
//
// 		**bpf_get_task_stack**\ () can collect up to
// 		**PERF_MAX_STACK_DEPTH** both kernel and user frames, subject
// 		to sufficient large buffer size. Note that
// 		this limit can be controlled with the **sysctl** program, and
// 		that it should be manually increased in order to profile long
// 		user stacks (such as stacks for Java programs). To do so, use:
//
// 		::
//
// 			# sysctl kernel.perf_event_max_stack=<new value>
// 	Return
// 		The non-negative copied *buf* length equal to or less than
// 		*size* on success, or a negative error in case of failure.
//
// long bpf_load_hdr_opt(struct bpf_sock_ops *skops, void *searchby_res, u32 len, u64 flags)
// 	Description
// 		Load header option.  Support reading a particular TCP header
// 		option for bpf program (**BPF_PROG_TYPE_SOCK_OPS**).
//
// 		If *flags* is 0, it will search the option from the
// 		*skops*\ **->skb_data**.  The comment in **struct bpf_sock_ops**
// 		has details on what skb_data contains under different
// 		*skops*\ **->op**.
//
// 		The first byte of the *searchby_res* specifies the
// 		kind that it wants to search.
//
// 		If the searching kind is an experimental kind
// 		(i.e. 253 or 254 according to RFC6994).  It also
// 		needs to specify the "magic" which is either
// 		2 bytes or 4 bytes.  It then also needs to
// 		specify the size of the magic by using
// 		the 2nd byte which is "kind-length" of a TCP
// 		header option and the "kind-length" also
// 		includes the first 2 bytes "kind" and "kind-length"
// 		itself as a normal TCP header option also does.
//
// 		For example, to search experimental kind 254 with
// 		2 byte magic 0xeB9F, the searchby_res should be
// 		[ 254, 4, 0xeB, 0x9F, 0, 0, .... 0 ].
//
// 		To search for the standard window scale option (3),
// 		the *searchby_res* should be [ 3, 0, 0, .... 0 ].
// 		Note, kind-length must be 0 for regular option.
//
// 		Searching for No-Op (0) and End-of-Option-List (1) are
// 		not supported.
//
// 		*len* must be at least 2 bytes which is the minimal size
// 		of a header option.
//
// 		Supported flags:
//
// 		* **BPF_LOAD_HDR_OPT_TCP_SYN** to search from the
// 		  saved_syn packet or the just-received syn packet.
//
// 	Return
// 		> 0 when found, the header option is copied to *searchby_res*.
// 		The return value is the total length copied. On failure, a
// 		negative error code is returned:
//
// 		**-EINVAL** if a parameter is invalid.
//
// 		**-ENOMSG** if the option is not found.
//
// 		**-ENOENT** if no syn packet is available when
// 		**BPF_LOAD_HDR_OPT_TCP_SYN** is used.
//
// 		**-ENOSPC** if there is not enough space.  Only *len* number of
// 		bytes are copied.
//
// 		**-EFAULT** on failure to parse the header options in the
// 		packet.
//
// 		**-EPERM** if the helper cannot be used under the current
// 		*skops*\ **->op**.
//
// long bpf_store_hdr_opt(struct bpf_sock_ops *skops, const void *from, u32 len, u64 flags)
// 	Description
// 		Store header option.  The data will be copied
// 		from buffer *from* with length *len* to the TCP header.
//
// 		The buffer *from* should have the whole option that
// 		includes the kind, kind-length, and the actual
// 		option data.  The *len* must be at least kind-length
// 		long.  The kind-length does not have to be 4 byte
// 		aligned.  The kernel will take care of the padding
// 		and setting the 4 bytes aligned value to th->doff.
//
// 		This helper will check for duplicated option
// 		by searching the same option in the outgoing skb.
//
// 		This helper can only be called during
// 		**BPF_SOCK_OPS_WRITE_HDR_OPT_CB**.
//
// 	Return
// 		0 on success, or negative error in case of failure:
//
// 		**-EINVAL** If param is invalid.
//
// 		**-ENOSPC** if there is not enough space in the header.
// 		Nothing has been written
//
// 		**-EEXIST** if the option already exists.
//
// 		**-EFAULT** on failure to parse the existing header options.
//
// 		**-EPERM** if the helper cannot be used under the current
// 		*skops*\ **->op**.
//
// long bpf_reserve_hdr_opt(struct bpf_sock_ops *skops, u32 len, u64 flags)
// 	Description
// 		Reserve *len* bytes for the bpf header option.  The
// 		space will be used by **bpf_store_hdr_opt**\ () later in
// 		**BPF_SOCK_OPS_WRITE_HDR_OPT_CB**.
//
// 		If **bpf_reserve_hdr_opt**\ () is called multiple times,
// 		the total number of bytes will be reserved.
//
// 		This helper can only be called during
// 		**BPF_SOCK_OPS_HDR_OPT_LEN_CB**.
//
// 	Return
// 		0 on success, or negative error in case of failure:
//
// 		**-EINVAL** if a parameter is invalid.
//
// 		**-ENOSPC** if there is not enough space in the header.
//
// 		**-EPERM** if the helper cannot be used under the current
// 		*skops*\ **->op**.
//
// void *bpf_inode_storage_get(struct bpf_map *map, void *inode, void *value, u64 flags)
// 	Description
// 		Get a bpf_local_storage from an *inode*.
//
// 		Logically, it could be thought of as getting the value from
// 		a *map* with *inode* as the **key**.  From this
// 		perspective,  the usage is not much different from
// 		**bpf_map_lookup_elem**\ (*map*, **&**\ *inode*) except this
// 		helper enforces the key must be an inode and the map must also
// 		be a **BPF_MAP_TYPE_INODE_STORAGE**.
//
// 		Underneath, the value is stored locally at *inode* instead of
// 		the *map*.  The *map* is used as the bpf-local-storage
// 		"type". The bpf-local-storage "type" (i.e. the *map*) is
// 		searched against all bpf_local_storage residing at *inode*.
//
// 		An optional *flags* (**BPF_LOCAL_STORAGE_GET_F_CREATE**) can be
// 		used such that a new bpf_local_storage will be
// 		created if one does not exist.  *value* can be used
// 		together with **BPF_LOCAL_STORAGE_GET_F_CREATE** to specify
// 		the initial value of a bpf_local_storage.  If *value* is
// 		**NULL**, the new bpf_local_storage will be zero initialized.
// 	Return
// 		A bpf_local_storage pointer is returned on success.
//
// 		**NULL** if not found or there was an error in adding
// 		a new bpf_local_storage.
//
// int bpf_inode_storage_delete(struct bpf_map *map, void *inode)
// 	Description
// 		Delete a bpf_local_storage from an *inode*.
// 	Return
// 		0 on success.
//
// 		**-ENOENT** if the bpf_local_storage cannot be found.
//
// long bpf_d_path(const struct path *path, char *buf, u32 sz)
// 	Description
// 		Return full path for given **struct path** object, which
// 		needs to be the kernel BTF *path* object. The path is
// 		returned in the provided buffer *buf* of size *sz* and
// 		is zero terminated.
//
// 	Return
// 		On success, the strictly positive length of the string,
// 		including the trailing NUL character. On error, a negative
// 		value.
//
// long bpf_copy_from_user(void *dst, u32 size, const void *user_ptr)
// 	Description
// 		Read *size* bytes from user space address *user_ptr* and store
// 		the data in *dst*. This is a wrapper of **copy_from_user**\ ().
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_snprintf_btf(char *str, u32 str_size, struct btf_ptr *ptr, u32 btf_ptr_size, u64 flags)
// 	Description
// 		Use BTF to store a string representation of *ptr*->ptr in *str*,
// 		using *ptr*->type_id.  This value should specify the type
// 		that *ptr*->ptr points to. LLVM __builtin_btf_type_id(type, 1)
// 		can be used to look up vmlinux BTF type ids. Traversing the
// 		data structure using BTF, the type information and values are
// 		stored in the first *str_size* - 1 bytes of *str*.  Safe copy of
// 		the pointer data is carried out to avoid kernel crashes during
// 		operation.  Smaller types can use string space on the stack;
// 		larger programs can use map data to store the string
// 		representation.
//
// 		The string can be subsequently shared with userspace via
// 		bpf_perf_event_output() or ring buffer interfaces.
// 		bpf_trace_printk() is to be avoided as it places too small
// 		a limit on string size to be useful.
//
// 		*flags* is a combination of
//
// 		**BTF_F_COMPACT**
// 			no formatting around type information
// 		**BTF_F_NONAME**
// 			no struct/union member names/types
// 		**BTF_F_PTR_RAW**
// 			show raw (unobfuscated) pointer values;
// 			equivalent to printk specifier %px.
// 		**BTF_F_ZERO**
// 			show zero-valued struct/union members; they
// 			are not displayed by default
//
// 	Return
// 		The number of bytes that were written (or would have been
// 		written if output had to be truncated due to string size),
// 		or a negative error in cases of failure.
//
// long bpf_seq_printf_btf(struct seq_file *m, struct btf_ptr *ptr, u32 ptr_size, u64 flags)
// 	Description
// 		Use BTF to write to seq_write a string representation of
// 		*ptr*->ptr, using *ptr*->type_id as per bpf_snprintf_btf().
// 		*flags* are identical to those used for bpf_snprintf_btf.
// 	Return
// 		0 on success or a negative error in case of failure.
//
// u64 bpf_skb_cgroup_classid(struct sk_buff *skb)
// 	Description
// 		See **bpf_get_cgroup_classid**\ () for the main description.
// 		This helper differs from **bpf_get_cgroup_classid**\ () in that
// 		the cgroup v1 net_cls class is retrieved only from the *skb*'s
// 		associated socket instead of the current process.
// 	Return
// 		The id is returned or 0 in case the id could not be retrieved.
//
// long bpf_redirect_neigh(u32 ifindex, struct bpf_redir_neigh *params, int plen, u64 flags)
// 	Description
// 		Redirect the packet to another net device of index *ifindex*
// 		and fill in L2 addresses from neighboring subsystem. This helper
// 		is somewhat similar to **bpf_redirect**\ (), except that it
// 		populates L2 addresses as well, meaning, internally, the helper
// 		relies on the neighbor lookup for the L2 address of the nexthop.
//
// 		The helper will perform a FIB lookup based on the skb's
// 		networking header to get the address of the next hop, unless
// 		this is supplied by the caller in the *params* argument. The
// 		*plen* argument indicates the len of *params* and should be set
// 		to 0 if *params* is NULL.
//
// 		The *flags* argument is reserved and must be 0. The helper is
// 		currently only supported for tc BPF program types, and enabled
// 		for IPv4 and IPv6 protocols.
// 	Return
// 		The helper returns **TC_ACT_REDIRECT** on success or
// 		**TC_ACT_SHOT** on error.
//
// void *bpf_per_cpu_ptr(const void *percpu_ptr, u32 cpu)
//     Description
//             Take a pointer to a percpu ksym, *percpu_ptr*, and return a
//             pointer to the percpu kernel variable on *cpu*. A ksym is an
//             extern variable decorated with '__ksym'. For ksym, there is a
//             global var (either static or global) defined of the same name
//             in the kernel. The ksym is percpu if the global var is percpu.
//             The returned pointer points to the global percpu var on *cpu*.
//
//             bpf_per_cpu_ptr() has the same semantic as per_cpu_ptr() in the
//             kernel, except that bpf_per_cpu_ptr() may return NULL. This
//             happens if *cpu* is larger than nr_cpu_ids. The caller of
//             bpf_per_cpu_ptr() must check the returned value.
//     Return
//             A pointer pointing to the kernel percpu variable on *cpu*, or
//             NULL, if *cpu* is invalid.
//
// void *bpf_this_cpu_ptr(const void *percpu_ptr)
// 	Description
// 		Take a pointer to a percpu ksym, *percpu_ptr*, and return a
// 		pointer to the percpu kernel variable on this cpu. See the
// 		description of 'ksym' in **bpf_per_cpu_ptr**\ ().
//
// 		bpf_this_cpu_ptr() has the same semantic as this_cpu_ptr() in
// 		the kernel. Different from **bpf_per_cpu_ptr**\ (), it would
// 		never return NULL.
// 	Return
// 		A pointer pointing to the kernel percpu variable on this cpu.
//
// long bpf_redirect_peer(u32 ifindex, u64 flags)
// 	Description
// 		Redirect the packet to another net device of index *ifindex*.
// 		This helper is somewhat similar to **bpf_redirect**\ (), except
// 		that the redirection happens to the *ifindex*' peer device. If
// 		*flags* is 0, the netns switch takes place from ingress to
// 		ingress without going through the CPU's backlog queue. If the
// 		**BPF_F_EGRESS** flag is provided then redirection happens in
// 		the egress direction of the peer device.
//
// 		*skb*\ **->mark** and *skb*\ **->tstamp** are not cleared during
// 		the netns switch.
//
// 		If the *flags* argument is 0, the helper is currently only
// 		supported for tc BPF program types at the ingress hook and for
// 		veth and netkit target device types. The peer device must reside
// 		in a different network namespace.
// 	Return
// 		The helper returns **TC_ACT_REDIRECT** on success or
// 		**TC_ACT_SHOT** on error.
//
// void *bpf_task_storage_get(struct bpf_map *map, struct task_struct *task, void *value, u64 flags)
// 	Description
// 		Get a bpf_local_storage from the *task*.
//
// 		Logically, it could be thought of as getting the value from
// 		a *map* with *task* as the **key**.  From this
// 		perspective,  the usage is not much different from
// 		**bpf_map_lookup_elem**\ (*map*, **&**\ *task*) except this
// 		helper enforces the key must be a task_struct and the map must also
// 		be a **BPF_MAP_TYPE_TASK_STORAGE**.
//
// 		Underneath, the value is stored locally at *task* instead of
// 		the *map*.  The *map* is used as the bpf-local-storage
// 		"type". The bpf-local-storage "type" (i.e. the *map*) is
// 		searched against all bpf_local_storage residing at *task*.
//
// 		An optional *flags* (**BPF_LOCAL_STORAGE_GET_F_CREATE**) can be
// 		used such that a new bpf_local_storage will be
// 		created if one does not exist.  *value* can be used
// 		together with **BPF_LOCAL_STORAGE_GET_F_CREATE** to specify
// 		the initial value of a bpf_local_storage.  If *value* is
// 		**NULL**, the new bpf_local_storage will be zero initialized.
// 	Return
// 		A bpf_local_storage pointer is returned on success.
//
// 		**NULL** if not found or there was an error in adding
// 		a new bpf_local_storage.
//
// long bpf_task_storage_delete(struct bpf_map *map, struct task_struct *task)
// 	Description
// 		Delete a bpf_local_storage from a *task*.
// 	Return
// 		0 on success.
//
// 		**-ENOENT** if the bpf_local_storage cannot be found.
//
// struct task_struct *bpf_get_current_task_btf(void)
// 	Description
// 		Return a BTF pointer to the "current" task.
// 		This pointer can also be used in helpers that accept an
// 		*ARG_PTR_TO_BTF_ID* of type *task_struct*.
// 	Return
// 		Pointer to the current task.
//
// long bpf_bprm_opts_set(struct linux_binprm *bprm, u64 flags)
// 	Description
// 		Set or clear certain options on *bprm*:
//
// 		**BPF_F_BPRM_SECUREEXEC** Set the secureexec bit
// 		which sets the **AT_SECURE** auxv for glibc. The bit
// 		is cleared if the flag is not specified.
// 	Return
// 		**-EINVAL** if invalid *flags* are passed, zero otherwise.
//
// u64 bpf_ktime_get_coarse_ns(void)
// 	Description
// 		Return a coarse-grained version of the time elapsed since
// 		system boot, in nanoseconds. Does not include time the system
// 		was suspended.
//
// 		See: **clock_gettime**\ (**CLOCK_MONOTONIC_COARSE**)
// 	Return
// 		Current *ktime*.
//
// long bpf_ima_inode_hash(struct inode *inode, void *dst, u32 size)
// 	Description
// 		Returns the stored IMA hash of the *inode* (if it's available).
// 		If the hash is larger than *size*, then only *size*
// 		bytes will be copied to *dst*
// 	Return
// 		The **hash_algo** is returned on success,
// 		**-EOPNOTSUPP** if IMA is disabled or **-EINVAL** if
// 		invalid arguments are passed.
//
// struct socket *bpf_sock_from_file(struct file *file)
// 	Description
// 		If the given file represents a socket, returns the associated
// 		socket.
// 	Return
// 		A pointer to a struct socket on success or NULL if the file is
// 		not a socket.
//
// long bpf_check_mtu(void *ctx, u32 ifindex, u32 *mtu_len, s32 len_diff, u64 flags)
// 	Description
// 		Check packet size against exceeding MTU of net device (based
// 		on *ifindex*).  This helper will likely be used in combination
// 		with helpers that adjust/change the packet size.
//
// 		The argument *len_diff* can be used for querying with a planned
// 		size change. This allows to check MTU prior to changing packet
// 		ctx. Providing a *len_diff* adjustment that is larger than the
// 		actual packet size (resulting in negative packet size) will in
// 		principle not exceed the MTU, which is why it is not considered
// 		a failure.  Other BPF helpers are needed for performing the
// 		planned size change; therefore the responsibility for catching
// 		a negative packet size belongs in those helpers.
//
// 		Specifying *ifindex* zero means the MTU check is performed
// 		against the current net device.  This is practical if this isn't
// 		used prior to redirect.
//
// 		On input *mtu_len* must be a valid pointer, else verifier will
// 		reject BPF program.  If the value *mtu_len* is initialized to
// 		zero then the ctx packet size is use.  When value *mtu_len* is
// 		provided as input this specify the L3 length that the MTU check
// 		is done against. Remember XDP and TC length operate at L2, but
// 		this value is L3 as this correlate to MTU and IP-header tot_len
// 		values which are L3 (similar behavior as bpf_fib_lookup).
//
// 		The Linux kernel route table can configure MTUs on a more
// 		specific per route level, which is not provided by this helper.
// 		For route level MTU checks use the **bpf_fib_lookup**\ ()
// 		helper.
//
// 		*ctx* is either **struct xdp_md** for XDP programs or
// 		**struct sk_buff** for tc cls_act programs.
//
// 		The *flags* argument can be a combination of one or more of the
// 		following values:
//
// 		**BPF_MTU_CHK_SEGS**
// 			This flag will only works for *ctx* **struct sk_buff**.
// 			If packet context contains extra packet segment buffers
// 			(often knows as GSO skb), then MTU check is harder to
// 			check at this point, because in transmit path it is
// 			possible for the skb packet to get re-segmented
// 			(depending on net device features).  This could still be
// 			a MTU violation, so this flag enables performing MTU
// 			check against segments, with a different violation
// 			return code to tell it apart. Check cannot use len_diff.
//
// 		On return *mtu_len* pointer contains the MTU value of the net
// 		device.  Remember the net device configured MTU is the L3 size,
// 		which is returned here and XDP and TC length operate at L2.
// 		Helper take this into account for you, but remember when using
// 		MTU value in your BPF-code.
//
// 	Return
// 		* 0 on success, and populate MTU value in *mtu_len* pointer.
//
// 		* < 0 if any input argument is invalid (*mtu_len* not updated)
//
// 		MTU violations return positive values, but also populate MTU
// 		value in *mtu_len* pointer, as this can be needed for
// 		implementing PMTU handing:
//
// 		* **BPF_MTU_CHK_RET_FRAG_NEEDED**
// 		* **BPF_MTU_CHK_RET_SEGS_TOOBIG**
//
// long bpf_for_each_map_elem(struct bpf_map *map, void *callback_fn, void *callback_ctx, u64 flags)
// 	Description
// 		For each element in **map**, call **callback_fn** function with
// 		**map**, **callback_ctx** and other map-specific parameters.
// 		The **callback_fn** should be a static function and
// 		the **callback_ctx** should be a pointer to the stack.
// 		The **flags** is used to control certain aspects of the helper.
// 		Currently, the **flags** must be 0.
//
// 		The following are a list of supported map types and their
// 		respective expected callback signatures:
//
// 		BPF_MAP_TYPE_HASH, BPF_MAP_TYPE_PERCPU_HASH,
// 		BPF_MAP_TYPE_LRU_HASH, BPF_MAP_TYPE_LRU_PERCPU_HASH,
// 		BPF_MAP_TYPE_ARRAY, BPF_MAP_TYPE_PERCPU_ARRAY
//
// 		long (\*callback_fn)(struct bpf_map \*map, const void \*key, void \*value, void \*ctx);
//
// 		For per_cpu maps, the map_value is the value on the cpu where the
// 		bpf_prog is running.
//
// 		If **callback_fn** return 0, the helper will continue to the next
// 		element. If return value is 1, the helper will skip the rest of
// 		elements and return. Other return values are not used now.
//
// 	Return
// 		The number of traversed map elements for success, **-EINVAL** for
// 		invalid **flags**.
//
// long bpf_snprintf(char *str, u32 str_size, const char *fmt, u64 *data, u32 data_len)
// 	Description
// 		Outputs a string into the **str** buffer of size **str_size**
// 		based on a format string stored in a read-only map pointed by
// 		**fmt**.
//
// 		Each format specifier in **fmt** corresponds to one u64 element
// 		in the **data** array. For strings and pointers where pointees
// 		are accessed, only the pointer values are stored in the *data*
// 		array. The *data_len* is the size of *data* in bytes - must be
// 		a multiple of 8.
//
// 		Formats **%s** and **%p{i,I}{4,6}** require to read kernel
// 		memory. Reading kernel memory may fail due to either invalid
// 		address or valid address but requiring a major memory fault. If
// 		reading kernel memory fails, the string for **%s** will be an
// 		empty string, and the ip address for **%p{i,I}{4,6}** will be 0.
// 		Not returning error to bpf program is consistent with what
// 		**bpf_trace_printk**\ () does for now.
//
// 	Return
// 		The strictly positive length of the formatted string, including
// 		the trailing zero character. If the return value is greater than
// 		**str_size**, **str** contains a truncated string, guaranteed to
// 		be zero-terminated except when **str_size** is 0.
//
// 		Or **-EBUSY** if the per-CPU memory copy buffer is busy.
//
// long bpf_sys_bpf(u32 cmd, void *attr, u32 attr_size)
// 	Description
// 		Execute bpf syscall with given arguments.
// 	Return
// 		A syscall result.
//
// long bpf_btf_find_by_name_kind(char *name, int name_sz, u32 kind, int flags)
// 	Description
// 		Find BTF type with given name and kind in vmlinux BTF or in module's BTFs.
// 	Return
// 		Returns btf_id and btf_obj_fd in lower and upper 32 bits.
//
// long bpf_sys_close(u32 fd)
// 	Description
// 		Execute close syscall for given FD.
// 	Return
// 		A syscall result.
//
// long bpf_timer_init(struct bpf_timer *timer, struct bpf_map *map, u64 flags)
// 	Description
// 		Initialize the timer.
// 		First 4 bits of *flags* specify clockid.
// 		Only CLOCK_MONOTONIC, CLOCK_REALTIME, CLOCK_BOOTTIME are allowed.
// 		All other bits of *flags* are reserved.
// 		The verifier will reject the program if *timer* is not from
// 		the same *map*.
// 	Return
// 		0 on success.
// 		**-EBUSY** if *timer* is already initialized.
// 		**-EINVAL** if invalid *flags* are passed.
// 		**-EPERM** if *timer* is in a map that doesn't have any user references.
// 		The user space should either hold a file descriptor to a map with timers
// 		or pin such map in bpffs. When map is unpinned or file descriptor is
// 		closed all timers in the map will be cancelled and freed.
//
// long bpf_timer_set_callback(struct bpf_timer *timer, void *callback_fn)
// 	Description
// 		Configure the timer to call *callback_fn* static function.
// 	Return
// 		0 on success.
// 		**-EINVAL** if *timer* was not initialized with bpf_timer_init() earlier.
// 		**-EPERM** if *timer* is in a map that doesn't have any user references.
// 		The user space should either hold a file descriptor to a map with timers
// 		or pin such map in bpffs. When map is unpinned or file descriptor is
// 		closed all timers in the map will be cancelled and freed.
//
// long bpf_timer_start(struct bpf_timer *timer, u64 nsecs, u64 flags)
// 	Description
// 		Set timer expiration N nanoseconds from the current time. The
// 		configured callback will be invoked in soft irq context on some cpu
// 		and will not repeat unless another bpf_timer_start() is made.
// 		In such case the next invocation can migrate to a different cpu.
// 		Since struct bpf_timer is a field inside map element the map
// 		owns the timer. The bpf_timer_set_callback() will increment refcnt
// 		of BPF program to make sure that callback_fn code stays valid.
// 		When user space reference to a map reaches zero all timers
// 		in a map are cancelled and corresponding program's refcnts are
// 		decremented. This is done to make sure that Ctrl-C of a user
// 		process doesn't leave any timers running. If map is pinned in
// 		bpffs the callback_fn can re-arm itself indefinitely.
// 		bpf_map_update/delete_elem() helpers and user space sys_bpf commands
// 		cancel and free the timer in the given map element.
// 		The map can contain timers that invoke callback_fn-s from different
// 		programs. The same callback_fn can serve different timers from
// 		different maps if key/value layout matches across maps.
// 		Every bpf_timer_set_callback() can have different callback_fn.
//
// 		*flags* can be one of:
//
// 		**BPF_F_TIMER_ABS**
// 			Start the timer in absolute expire value instead of the
// 			default relative one.
// 		**BPF_F_TIMER_CPU_PIN**
// 			Timer will be pinned to the CPU of the caller.
//
// 	Return
// 		0 on success.
// 		**-EINVAL** if *timer* was not initialized with bpf_timer_init() earlier
// 		or invalid *flags* are passed.
//
// long bpf_timer_cancel(struct bpf_timer *timer)
// 	Description
// 		Cancel the timer and wait for callback_fn to finish if it was running.
// 	Return
// 		0 if the timer was not active.
// 		1 if the timer was active.
// 		**-EINVAL** if *timer* was not initialized with bpf_timer_init() earlier.
// 		**-EDEADLK** if callback_fn tried to call bpf_timer_cancel() on its
// 		own timer which would have led to a deadlock otherwise.
//
// u64 bpf_get_func_ip(void *ctx)
// 	Description
// 		Get address of the traced function (for tracing and kprobe programs).
//
// 		When called for kprobe program attached as uprobe it returns
// 		probe address for both entry and return uprobe.
//
// 	Return
// 		Address of the traced function for kprobe.
// 		0 for kprobes placed within the function (not at the entry).
// 		Address of the probe for uprobe and return uprobe.
//
// u64 bpf_get_attach_cookie(void *ctx)
// 	Description
// 		Get bpf_cookie value provided (optionally) during the program
// 		attachment. It might be different for each individual
// 		attachment, even if BPF program itself is the same.
// 		Expects BPF program context *ctx* as a first argument.
//
// 		Supported for the following program types:
// 			- kprobe/uprobe;
// 			- tracepoint;
// 			- perf_event.
// 	Return
// 		Value specified by user at BPF link creation/attachment time
// 		or 0, if it was not specified.
//
// long bpf_task_pt_regs(struct task_struct *task)
// 	Description
// 		Get the struct pt_regs associated with **task**.
// 	Return
// 		A pointer to struct pt_regs.
//
// long bpf_get_branch_snapshot(void *entries, u32 size, u64 flags)
// 	Description
// 		Get branch trace from hardware engines like Intel LBR. The
// 		hardware engine is stopped shortly after the helper is
// 		called. Therefore, the user need to filter branch entries
// 		based on the actual use case. To capture branch trace
// 		before the trigger point of the BPF program, the helper
// 		should be called at the beginning of the BPF program.
//
// 		The data is stored as struct perf_branch_entry into output
// 		buffer *entries*. *size* is the size of *entries* in bytes.
// 		*flags* is reserved for now and must be zero.
//
// 	Return
// 		On success, number of bytes written to *buf*. On error, a
// 		negative value.
//
// 		**-EINVAL** if *flags* is not zero.
//
// 		**-ENOENT** if architecture does not support branch records.
//
// long bpf_trace_vprintk(const char *fmt, u32 fmt_size, const void *data, u32 data_len)
// 	Description
// 		Behaves like **bpf_trace_printk**\ () helper, but takes an array of u64
// 		to format and can handle more format args as a result.
//
// 		Arguments are to be used as in **bpf_seq_printf**\ () helper.
// 	Return
// 		The number of bytes written to the buffer, or a negative error
// 		in case of failure.
//
// struct unix_sock *bpf_skc_to_unix_sock(void *sk)
// 	Description
// 		Dynamically cast a *sk* pointer to a *unix_sock* pointer.
// 	Return
// 		*sk* if casting is valid, or **NULL** otherwise.
//
// long bpf_kallsyms_lookup_name(const char *name, int name_sz, int flags, u64 *res)
// 	Description
// 		Get the address of a kernel symbol, returned in *res*. *res* is
// 		set to 0 if the symbol is not found.
// 	Return
// 		On success, zero. On error, a negative value.
//
// 		**-EINVAL** if *flags* is not zero.
//
// 		**-EINVAL** if string *name* is not the same size as *name_sz*.
//
// 		**-ENOENT** if symbol is not found.
//
// 		**-EPERM** if caller does not have permission to obtain kernel address.
//
// long bpf_find_vma(struct task_struct *task, u64 addr, void *callback_fn, void *callback_ctx, u64 flags)
// 	Description
// 		Find vma of *task* that contains *addr*, call *callback_fn*
// 		function with *task*, *vma*, and *callback_ctx*.
// 		The *callback_fn* should be a static function and
// 		the *callback_ctx* should be a pointer to the stack.
// 		The *flags* is used to control certain aspects of the helper.
// 		Currently, the *flags* must be 0.
//
// 		The expected callback signature is
//
// 		long (\*callback_fn)(struct task_struct \*task, struct vm_area_struct \*vma, void \*callback_ctx);
//
// 	Return
// 		0 on success.
// 		**-ENOENT** if *task->mm* is NULL, or no vma contains *addr*.
// 		**-EBUSY** if failed to try lock mmap_lock.
// 		**-EINVAL** for invalid **flags**.
//
// long bpf_loop(u32 nr_loops, void *callback_fn, void *callback_ctx, u64 flags)
// 	Description
// 		For **nr_loops**, call **callback_fn** function
// 		with **callback_ctx** as the context parameter.
// 		The **callback_fn** should be a static function and
// 		the **callback_ctx** should be a pointer to the stack.
// 		The **flags** is used to control certain aspects of the helper.
// 		Currently, the **flags** must be 0. Currently, nr_loops is
// 		limited to 1 << 23 (~8 million) loops.
//
// 		long (\*callback_fn)(u64 index, void \*ctx);
//
// 		where **index** is the current index in the loop. The index
// 		is zero-indexed.
//
// 		If **callback_fn** returns 0, the helper will continue to the next
// 		loop. If return value is 1, the helper will skip the rest of
// 		the loops and return. Other return values are not used now,
// 		and will be rejected by the verifier.
//
// 	Return
// 		The number of loops performed, **-EINVAL** for invalid **flags**,
// 		**-E2BIG** if **nr_loops** exceeds the maximum number of loops.
//
// long bpf_strncmp(const char *s1, u32 s1_sz, const char *s2)
// 	Description
// 		Do strncmp() between **s1** and **s2**. **s1** doesn't need
// 		to be null-terminated and **s1_sz** is the maximum storage
// 		size of **s1**. **s2** must be a read-only string.
// 	Return
// 		An integer less than, equal to, or greater than zero
// 		if the first **s1_sz** bytes of **s1** is found to be
// 		less than, to match, or be greater than **s2**.
//
// long bpf_get_func_arg(void *ctx, u32 n, u64 *value)
// 	Description
// 		Get **n**-th argument register (zero based) of the traced function (for tracing programs)
// 		returned in **value**.
//
// 	Return
// 		0 on success.
// 		**-EINVAL** if n >= argument register count of traced function.
//
// long bpf_get_func_ret(void *ctx, u64 *value)
// 	Description
// 		Get return value of the traced function (for tracing programs)
// 		in **value**.
//
// 	Return
// 		0 on success.
// 		**-EOPNOTSUPP** for tracing programs other than BPF_TRACE_FEXIT or BPF_MODIFY_RETURN.
//
// long bpf_get_func_arg_cnt(void *ctx)
// 	Description
// 		Get number of registers of the traced function (for tracing programs) where
// 		function arguments are stored in these registers.
//
// 	Return
// 		The number of argument registers of the traced function.
//
// int bpf_get_retval(void)
// 	Description
// 		Get the BPF program's return value that will be returned to the upper layers.
//
// 		This helper is currently supported by cgroup programs and only by the hooks
// 		where BPF program's return value is returned to the userspace via errno.
// 	Return
// 		The BPF program's return value.
//
// int bpf_set_retval(int retval)
// 	Description
// 		Set the BPF program's return value that will be returned to the upper layers.
//
// 		This helper is currently supported by cgroup programs and only by the hooks
// 		where BPF program's return value is returned to the userspace via errno.
//
// 		Note that there is the following corner case where the program exports an error
// 		via bpf_set_retval but signals success via 'return 1':
//
// 			bpf_set_retval(-EPERM);
// 			return 1;
//
// 		In this case, the BPF program's return value will use helper's -EPERM. This
// 		still holds true for cgroup/bind{4,6} which supports extra 'return 3' success case.
//
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// u64 bpf_xdp_get_buff_len(struct xdp_buff *xdp_md)
// 	Description
// 		Get the total size of a given xdp buff (linear and paged area)
// 	Return
// 		The total size of a given xdp buffer.
//
// long bpf_xdp_load_bytes(struct xdp_buff *xdp_md, u32 offset, void *buf, u32 len)
// 	Description
// 		This helper is provided as an easy way to load data from a
// 		xdp buffer. It can be used to load *len* bytes from *offset* from
// 		the frame associated to *xdp_md*, into the buffer pointed by
// 		*buf*.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_xdp_store_bytes(struct xdp_buff *xdp_md, u32 offset, void *buf, u32 len)
// 	Description
// 		Store *len* bytes from buffer *buf* into the frame
// 		associated to *xdp_md*, at *offset*.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// long bpf_copy_from_user_task(void *dst, u32 size, const void *user_ptr, struct task_struct *tsk, u64 flags)
// 	Description
// 		Read *size* bytes from user space address *user_ptr* in *tsk*'s
// 		address space, and stores the data in *dst*. *flags* is not
// 		used yet and is provided for future extensibility. This helper
// 		can only be used by sleepable programs.
// 	Return
// 		0 on success, or a negative error in case of failure. On error
// 		*dst* buffer is zeroed out.
//
// long bpf_skb_set_tstamp(struct sk_buff *skb, u64 tstamp, u32 tstamp_type)
// 	Description
// 		Change the __sk_buff->tstamp_type to *tstamp_type*
// 		and set *tstamp* to the __sk_buff->tstamp together.
//
// 		If there is no need to change the __sk_buff->tstamp_type,
// 		the tstamp value can be directly written to __sk_buff->tstamp
// 		instead.
//
// 		BPF_SKB_TSTAMP_DELIVERY_MONO is the only tstamp that
// 		will be kept during bpf_redirect_*().  A non zero
// 		*tstamp* must be used with the BPF_SKB_TSTAMP_DELIVERY_MONO
// 		*tstamp_type*.
//
// 		A BPF_SKB_TSTAMP_UNSPEC *tstamp_type* can only be used
// 		with a zero *tstamp*.
//
// 		Only IPv4 and IPv6 skb->protocol are supported.
//
// 		This function is most useful when it needs to set a
// 		mono delivery time to __sk_buff->tstamp and then
// 		bpf_redirect_*() to the egress of an iface.  For example,
// 		changing the (rcv) timestamp in __sk_buff->tstamp at
// 		ingress to a mono delivery time and then bpf_redirect_*()
// 		to sch_fq@phy-dev.
// 	Return
// 		0 on success.
// 		**-EINVAL** for invalid input
// 		**-EOPNOTSUPP** for unsupported protocol
//
// long bpf_ima_file_hash(struct file *file, void *dst, u32 size)
// 	Description
// 		Returns a calculated IMA hash of the *file*.
// 		If the hash is larger than *size*, then only *size*
// 		bytes will be copied to *dst*
// 	Return
// 		The **hash_algo** is returned on success,
// 		**-EOPNOTSUPP** if the hash calculation failed or **-EINVAL** if
// 		invalid arguments are passed.
//
// void *bpf_kptr_xchg(void *dst, void *ptr)
// 	Description
// 		Exchange kptr at pointer *dst* with *ptr*, and return the old value.
// 		*dst* can be map value or local kptr. *ptr* can be NULL, otherwise
// 		it must be a referenced pointer which will be released when this helper
// 		is called.
// 	Return
// 		The old value of kptr (which can be NULL). The returned pointer
// 		if not NULL, is a reference which must be released using its
// 		corresponding release function, or moved into a BPF map before
// 		program exit.
//
// void *bpf_map_lookup_percpu_elem(struct bpf_map *map, const void *key, u32 cpu)
// 	Description
// 		Perform a lookup in *percpu map* for an entry associated to
// 		*key* on *cpu*.
// 	Return
// 		Map value associated to *key* on *cpu*, or **NULL** if no entry
// 		was found or *cpu* is invalid.
//
// struct mptcp_sock *bpf_skc_to_mptcp_sock(void *sk)
// 	Description
// 		Dynamically cast a *sk* pointer to a *mptcp_sock* pointer.
// 	Return
// 		*sk* if casting is valid, or **NULL** otherwise.
//
// long bpf_dynptr_from_mem(void *data, u64 size, u64 flags, struct bpf_dynptr *ptr)
// 	Description
// 		Get a dynptr to local memory *data*.
//
// 		*data* must be a ptr to a map value.
// 		The maximum *size* supported is DYNPTR_MAX_SIZE.
// 		*flags* is currently unused.
// 	Return
// 		0 on success, -E2BIG if the size exceeds DYNPTR_MAX_SIZE,
// 		-EINVAL if flags is not 0.
//
// long bpf_ringbuf_reserve_dynptr(void *ringbuf, u32 size, u64 flags, struct bpf_dynptr *ptr)
// 	Description
// 		Reserve *size* bytes of payload in a ring buffer *ringbuf*
// 		through the dynptr interface. *flags* must be 0.
//
// 		Please note that a corresponding bpf_ringbuf_submit_dynptr or
// 		bpf_ringbuf_discard_dynptr must be called on *ptr*, even if the
// 		reservation fails. This is enforced by the verifier.
// 	Return
// 		0 on success, or a negative error in case of failure.
//
// void bpf_ringbuf_submit_dynptr(struct bpf_dynptr *ptr, u64 flags)
// 	Description
// 		Submit reserved ring buffer sample, pointed to by *data*,
// 		through the dynptr interface. This is a no-op if the dynptr is
// 		invalid/null.
//
// 		For more information on *flags*, please see
// 		'bpf_ringbuf_submit'.
// 	Return
// 		Nothing. Always succeeds.
//
// void bpf_ringbuf_discard_dynptr(struct bpf_dynptr *ptr, u64 flags)
// 	Description
// 		Discard reserved ring buffer sample through the dynptr
// 		interface. This is a no-op if the dynptr is invalid/null.
//
// 		For more information on *flags*, please see
// 		'bpf_ringbuf_discard'.
// 	Return
// 		Nothing. Always succeeds.
//
// long bpf_dynptr_read(void *dst, u64 len, const struct bpf_dynptr *src, u64 offset, u64 flags)
// 	Description
// 		Read *len* bytes from *src* into *dst*, starting from *offset*
// 		into *src*.
// 		*flags* is currently unused.
// 	Return
// 		0 on success, -E2BIG if *offset* + *len* exceeds the length
// 		of *src*'s data, -EINVAL if *src* is an invalid dynptr or if
// 		*flags* is not 0.
//
// long bpf_dynptr_write(const struct bpf_dynptr *dst, u64 offset, void *src, u64 len, u64 flags)
// 	Description
// 		Write *len* bytes from *src* into *dst*, starting from *offset*
// 		into *dst*.
//
// 		*flags* must be 0 except for skb-type dynptrs.
//
// 		For skb-type dynptrs:
// 		    *  All data slices of the dynptr are automatically
// 		       invalidated after **bpf_dynptr_write**\ (). This is
// 		       because writing may pull the skb and change the
// 		       underlying packet buffer.
//
// 		    *  For *flags*, please see the flags accepted by
// 		       **bpf_skb_store_bytes**\ ().
// 	Return
// 		0 on success, -E2BIG if *offset* + *len* exceeds the length
// 		of *dst*'s data, -EINVAL if *dst* is an invalid dynptr or if *dst*
// 		is a read-only dynptr or if *flags* is not correct. For skb-type dynptrs,
// 		other errors correspond to errors returned by **bpf_skb_store_bytes**\ ().
//
// void *bpf_dynptr_data(const struct bpf_dynptr *ptr, u64 offset, u64 len)
// 	Description
// 		Get a pointer to the underlying dynptr data.
//
// 		*len* must be a statically known value. The returned data slice
// 		is invalidated whenever the dynptr is invalidated.
//
// 		skb and xdp type dynptrs may not use bpf_dynptr_data. They should
// 		instead use bpf_dynptr_slice and bpf_dynptr_slice_rdwr.
// 	Return
// 		Pointer to the underlying dynptr data, NULL if the dynptr is
// 		read-only, if the dynptr is invalid, or if the offset and length
// 		is out of bounds.
//
// s64 bpf_tcp_raw_gen_syncookie_ipv4(struct iphdr *iph, struct tcphdr *th, u32 th_len)
// 	Description
// 		Try to issue a SYN cookie for the packet with corresponding
// 		IPv4/TCP headers, *iph* and *th*, without depending on a
// 		listening socket.
//
// 		*iph* points to the IPv4 header.
//
// 		*th* points to the start of the TCP header, while *th_len*
// 		contains the length of the TCP header (at least
// 		**sizeof**\ (**struct tcphdr**)).
// 	Return
// 		On success, lower 32 bits hold the generated SYN cookie in
// 		followed by 16 bits which hold the MSS value for that cookie,
// 		and the top 16 bits are unused.
//
// 		On failure, the returned value is one of the following:
//
// 		**-EINVAL** if *th_len* is invalid.
//
// s64 bpf_tcp_raw_gen_syncookie_ipv6(struct ipv6hdr *iph, struct tcphdr *th, u32 th_len)
// 	Description
// 		Try to issue a SYN cookie for the packet with corresponding
// 		IPv6/TCP headers, *iph* and *th*, without depending on a
// 		listening socket.
//
// 		*iph* points to the IPv6 header.
//
// 		*th* points to the start of the TCP header, while *th_len*
// 		contains the length of the TCP header (at least
// 		**sizeof**\ (**struct tcphdr**)).
// 	Return
// 		On success, lower 32 bits hold the generated SYN cookie in
// 		followed by 16 bits which hold the MSS value for that cookie,
// 		and the top 16 bits are unused.
//
// 		On failure, the returned value is one of the following:
//
// 		**-EINVAL** if *th_len* is invalid.
//
// 		**-EPROTONOSUPPORT** if CONFIG_IPV6 is not builtin.
//
// long bpf_tcp_raw_check_syncookie_ipv4(struct iphdr *iph, struct tcphdr *th)
// 	Description
// 		Check whether *iph* and *th* contain a valid SYN cookie ACK
// 		without depending on a listening socket.
//
// 		*iph* points to the IPv4 header.
//
// 		*th* points to the TCP header.
// 	Return
// 		0 if *iph* and *th* are a valid SYN cookie ACK.
//
// 		On failure, the returned value is one of the following:
//
// 		**-EACCES** if the SYN cookie is not valid.
//
// long bpf_tcp_raw_check_syncookie_ipv6(struct ipv6hdr *iph, struct tcphdr *th)
// 	Description
// 		Check whether *iph* and *th* contain a valid SYN cookie ACK
// 		without depending on a listening socket.
//
// 		*iph* points to the IPv6 header.
//
// 		*th* points to the TCP header.
// 	Return
// 		0 if *iph* and *th* are a valid SYN cookie ACK.
//
// 		On failure, the returned value is one of the following:
//
// 		**-EACCES** if the SYN cookie is not valid.
//
// 		**-EPROTONOSUPPORT** if CONFIG_IPV6 is not builtin.
//
// u64 bpf_ktime_get_tai_ns(void)
// 	Description
// 		A nonsettable system-wide clock derived from wall-clock time but
// 		ignoring leap seconds.  This clock does not experience
// 		discontinuities and backwards jumps caused by NTP inserting leap
// 		seconds as CLOCK_REALTIME does.
//
// 		See: **clock_gettime**\ (**CLOCK_TAI**)
// 	Return
// 		Current *ktime*.
//
// long bpf_user_ringbuf_drain(struct bpf_map *map, void *callback_fn, void *ctx, u64 flags)
// 	Description
// 		Drain samples from the specified user ring buffer, and invoke
// 		the provided callback for each such sample:
//
// 		long (\*callback_fn)(const struct bpf_dynptr \*dynptr, void \*ctx);
//
// 		If **callback_fn** returns 0, the helper will continue to try
// 		and drain the next sample, up to a maximum of
// 		BPF_MAX_USER_RINGBUF_SAMPLES samples. If the return value is 1,
// 		the helper will skip the rest of the samples and return. Other
// 		return values are not used now, and will be rejected by the
// 		verifier.
// 	Return
// 		The number of drained samples if no error was encountered while
// 		draining samples, or 0 if no samples were present in the ring
// 		buffer. If a user-space producer was epoll-waiting on this map,
// 		and at least one sample was drained, they will receive an event
// 		notification notifying them of available space in the ring
// 		buffer. If the BPF_RB_NO_WAKEUP flag is passed to this
// 		function, no wakeup notification will be sent. If the
// 		BPF_RB_FORCE_WAKEUP flag is passed, a wakeup notification will
// 		be sent even if no sample was drained.
//
// 		On failure, the returned value is one of the following:
//
// 		**-EBUSY** if the ring buffer is contended, and another calling
// 		context was concurrently draining the ring buffer.
//
// 		**-EINVAL** if user-space is not properly tracking the ring
// 		buffer due to the producer position not being aligned to 8
// 		bytes, a sample not being aligned to 8 bytes, or the producer
// 		position not matching the advertised length of a sample.
//
// 		**-E2BIG** if user-space has tried to publish a sample which is
// 		larger than the size of the ring buffer, or which cannot fit
// 		within a struct bpf_dynptr.
//
// void *bpf_cgrp_storage_get(struct bpf_map *map, struct cgroup *cgroup, void *value, u64 flags)
// 	Description
// 		Get a bpf_local_storage from the *cgroup*.
//
// 		Logically, it could be thought of as getting the value from
// 		a *map* with *cgroup* as the **key**.  From this
// 		perspective,  the usage is not much different from
// 		**bpf_map_lookup_elem**\ (*map*, **&**\ *cgroup*) except this
// 		helper enforces the key must be a cgroup struct and the map must also
// 		be a **BPF_MAP_TYPE_CGRP_STORAGE**.
//
// 		In reality, the local-storage value is embedded directly inside of the
// 		*cgroup* object itself, rather than being located in the
// 		**BPF_MAP_TYPE_CGRP_STORAGE** map. When the local-storage value is
// 		queried for some *map* on a *cgroup* object, the kernel will perform an
// 		O(n) iteration over all of the live local-storage values for that
// 		*cgroup* object until the local-storage value for the *map* is found.
//
// 		An optional *flags* (**BPF_LOCAL_STORAGE_GET_F_CREATE**) can be
// 		used such that a new bpf_local_storage will be
// 		created if one does not exist.  *value* can be used
// 		together with **BPF_LOCAL_STORAGE_GET_F_CREATE** to specify
// 		the initial value of a bpf_local_storage.  If *value* is
// 		**NULL**, the new bpf_local_storage will be zero initialized.
// 	Return
// 		A bpf_local_storage pointer is returned on success.
//
// 		**NULL** if not found or there was an error in adding
// 		a new bpf_local_storage.
//
// long bpf_cgrp_storage_delete(struct bpf_map *map, struct cgroup *cgroup)
// 	Description
// 		Delete a bpf_local_storage from a *cgroup*.
// 	Return
// 		0 on success.
//
// 		**-ENOENT** if the bpf_local_storage cannot be found.
// This helper list is effectively frozen. If you are trying to		 * add a new helper, you should add a kfunc instead which has		 * less stability guarantees. See Documentation/bpf/kfuncs.rst
// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
// Attributes stored in the BTF_INT_ENCODING
// If the struct/union type info kind_flag is set, the
// following two macros are used to access bitfield_size
// and bit_offset from btf_member.offset.
// hlist_nulls_next_rcu - returns the element of the list after @node.
// @node: element of the list.
// hlist_nulls_pprev_rcu - returns the dereferenced pprev of @node.
// @node: element of the list.
// task_css_check - obtain css for (task, subsys) w/ extra access conds
// @task: the target task
// @subsys_id: the target subsystem ID
// @__c: extra condition expression to be passed to rcu_dereference_check()
//
// Return the cgroup_subsys_state for the (@task, @subsys_id) pair.  The
// synchronization rules are the same as task_css_set_check().
// PT_TRACE_* event enable flags
// Max number of base types.
// Max number of all types.
// The longest tracepoint has 12 args.
// See include/trace/bpf_probe.h
//
// Also reuse this macro for maximum number of arguments a BPF function
// or a kfunc can have. Args 1-5 are passed in registers, args 6-12 via
// stack arg slots. The JIT may map some stack arg slots to registers based
// on the native calling convention (e.g., arg 6 to R9 on x86-64).
// Fits in 2048B
// Per-cpu temp buffers used by printf-like helpers to store the bprintf binary
// arguments representation.
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

macro_rules! _AC__2 {
() => {
16045481047390945280u64
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

macro_rules! __WRITE_ONCE__5 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut u32, ($val) as u32) };
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

/// helper macro to distinguish between real in place fifo where the fifo
/// array is a part of the structure and the fifo type where the array is
/// outside of the fifo structure.
macro_rules! __is_kfifo_ptr {
() => {
(core::mem::size_of::<__SlateRecord_hid_debug_h_31>() as u64) == (core::mem::size_of::<__SlateRecord_hid_debug_c_3695>() as u64)
};
}

/// helper macro to distinguish between real in place fifo where the fifo
/// array is a part of the structure and the fifo type where the array is
/// outside of the fifo structure.
macro_rules! __is_kfifo_ptr__2 {
() => {
((const { core::mem::size_of::<__SlateRecord_hid_debug_h_31>() }) as u64) == ((const { core::mem::size_of::<__SlateRecord_hid_debug_c_3793>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word {
() => {
((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__4 {
() => {
((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__3 {
() => {
((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
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

macro_rules! alloc_tag_restore {
() => {
while false {
}
};
}

macro_rules! arch_atomic_fetch_sub {
($i:expr, $v:expr) => {
arch_atomic_fetch_add($i.wrapping_neg(), $v)
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
macro_rules! container_of__2 {
($ptr:expr) => {
{
let mut __v: *mut hid_report;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__7!() as isize)) }) as *mut ()) as *mut hid_report;
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
macro_rules! container_of__74 {
($ptr:expr) => {
{
let mut __v: *mut hid_debug_list;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__111!() as isize)) }) as *mut ()) as *mut hid_debug_list;
__v
}
};
}

macro_rules! debug_normal_state_change {
() => {
while false {
}
};
}

/// debugfs_create_file - create a file in the debugfs filesystem
/// @name: a pointer to a string containing the name of the file to create.
/// @mode: the permission that the file should have.
/// @parent: a pointer to the parent dentry for this file.  This should be a
///          directory dentry if set.  If this parameter is NULL, then the
///          file will be created in the root of the debugfs filesystem.
/// @data: a pointer to something that the caller will want to get to later
///        on.  The inode.i_private pointer will point to this value on
///        the open() call.
/// @fops: a pointer to a struct file_operations or struct debugfs_short_fops that
///        should be used for this file.
///
/// This is the basic "create a file" function for debugfs.  It allows for a
/// wide range of flexibility in creating a file, or a directory (if you want
/// to create a directory, the debugfs_create_dir() function is
/// recommended to be used instead.)
///
/// This function will return a pointer to a dentry if it succeeds.  This
/// pointer must be passed to the debugfs_remove() function when the file is
/// to be removed (no automatic cleanup happens if your module is unloaded,
/// you are responsible here.)  If an error occurs, ERR_PTR(-ERROR) will be
/// returned.
///
/// If debugfs is not enabled in the kernel, the value -%ENODEV will be
/// returned.
///
/// If fops points to a struct debugfs_short_fops, then simple_open() will be
/// used for the open, and only read/write/llseek are supported and are proxied,
/// so no module reference or release are needed.
///
/// NOTE: it's expected that most callers should _ignore_ the errors returned
/// by this function. Other debugfs functions handle the fact that the "dentry"
/// passed to them could be an error and they don't crash in that case.
/// Drivers should generally work fine even if debugfs fails to init anyway.
macro_rules! debugfs_create_file {
($name:expr, $mode:expr, $parent:expr, $data:expr, $fops:expr) => {
unsafe { debugfs_create_file_full(($name) as *const u8, ($mode) as u16, $parent, ($data) as *mut (), core::ptr::null::<()>(), $fops) }
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

/// kfifo_alloc - dynamically allocates a new fifo buffer
/// @fifo: pointer to the fifo
/// @size: the number of elements in the fifo, this must be a power of 2
/// @gfp_mask: get_free_pages mask, passed to kmalloc()
///
/// This macro dynamically allocates a new fifo buffer.
///
/// The number of elements will be rounded-up to a power of 2.
/// The fifo will be release with kfifo_free().
/// Return 0 if no error, otherwise an error code.
macro_rules! kfifo_alloc {
($fifo:expr, $size:expr, $gfp_mask:expr) => {
{
let mut __v: i32;
let __v2: *mut __SlateRecord_hid_debug_h_31 = $fifo;
let __v3: *mut __kfifo = unsafe { core::ptr::addr_of_mut!((*__v2).__slate_anon_0.kfifo) };
let __v4: i32 = if __is_kfifo_ptr!() {
__kfifo_alloc(__v3, ($size) as u32, core::mem::size_of::<u8>() as u64, $gfp_mask)
} else {
22i32.wrapping_neg()
};
__v = __v4;
__kfifo_int_must_check_helper(__v)
}
};
}

/// kfifo_free - frees the fifo
/// @fifo: the fifo to be freed
macro_rules! kfifo_free {
($fifo:expr) => {
let __tmp: *mut __SlateRecord_hid_debug_h_31 = $fifo;
let __kfifo: *mut __kfifo = unsafe { core::ptr::addr_of_mut!((*__tmp).__slate_anon_0.kfifo) };
if __is_kfifo_ptr__2!() {
unsafe { __kfifo_free(__kfifo) };
}
};
}

/// kfifo_in - put data into the fifo
/// @fifo: address of the fifo to be used
/// @buf: the data to be added
/// @n: number of elements to be added
///
/// This macro copies the given buffer into the fifo and returns the
/// number of copied elements.
///
/// Note that with only one concurrent reader and one concurrent
/// writer, you don't need extra locking to use these macro.
macro_rules! kfifo_in {
($fifo:expr, $buf:expr, $n:expr) => {
let mut __v: u32;
let __tmp: *mut __SlateRecord_hid_debug_h_31 = $fifo;
let __buf: *const u8 = ($buf) as *const u8;
let __n: u64 = $n;
let __recsize: u64 = core::mem::size_of::<[u8; 0]>() as u64;
let __kfifo: *mut __kfifo = unsafe { core::ptr::addr_of_mut!((*__tmp).__slate_anon_0.kfifo) };
let __v2: u32 = if __recsize != 0 {
unsafe { __kfifo_in_r(__kfifo, __buf as *const (), __n as u32, __recsize) }
} else {
unsafe { __kfifo_in(__kfifo, __buf as *const (), __n as u32) }
};
__v = __v2;
};
}

/// kfifo_is_empty - returns true if the fifo is empty
/// @fifo: address of the fifo to be used
macro_rules! kfifo_is_empty {
($fifo:expr) => {
{
let mut __v: bool;
let __v2: *mut __SlateRecord_hid_debug_h_31 = $fifo;
__v = (unsafe { (*__v2).__slate_anon_0.kfifo.r#in }) == unsafe { (*__v2).__slate_anon_0.kfifo.out };
__v
}
};
}

/// kfifo_to_user - copies data from the fifo into user space
/// @fifo: address of the fifo to be used
/// @to: where the data must be copied
/// @len: the size of the destination buffer
/// @copied: pointer to output variable to store the number of copied bytes
///
/// This macro copies at most @len bytes from the fifo into the
/// @to buffer and returns -EFAULT/0.
///
/// Note that with only one concurrent reader and one concurrent
/// writer, you don't need extra locking to use these macro.
macro_rules! kfifo_to_user {
($fifo:expr, $to:expr, $len:expr, $copied:expr) => {
{
let mut __v: i32;
let __v2: *mut __SlateRecord_hid_debug_h_31 = $fifo;
let __v3: *mut () = ($to) as *mut ();
let __v4: u32 = ($len) as u32;
let __v5: *mut u32 = ($copied) as *mut u32;
let __v7: *mut __kfifo = unsafe { core::ptr::addr_of_mut!((*__v2).__slate_anon_0.kfifo) };
__v = unsafe { __kfifo_to_user(__v7, __v3, __v4 as u64, __v5) };
__kfifo_int_must_check_helper(__v)
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
macro_rules! list_entry__37 {
($ptr:expr) => {
container_of__74!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__2 {
($ptr:expr) => {
container_of__2!($ptr)
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
macro_rules! list_first_entry__39 {
($ptr:expr) => {
list_entry__37!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry {
($ptr:expr) => {
list_entry__2!(unsafe { (*$ptr).next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry {
($pos:expr, $member:ident) => {
list_entry__2!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__39 {
($pos:expr, $member:ident) => {
list_entry__37!(unsafe { (*$pos).$member.next })
};
}

macro_rules! offsetof__7 {
() => {
core::mem::offset_of!(hid_report, list) as u64
};
}

macro_rules! offsetof__111 {
() => {
core::mem::offset_of!(hid_debug_list, node) as u64
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

macro_rules! spin_lock_irqsave__2 {
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

macro_rules! static_branch_unlikely__4 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const4(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__3 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const3(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
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

macro_rules! task_thread_info {
($task:expr) => {
unsafe { core::ptr::addr_of_mut!((*$task).thread_info) }
};
}

macro_rules! test_bit__3 {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if (unsafe { __slate_intrinsic_a39e427d3b6c2df4($nr) }) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
const_test_bit(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit(($nr) as u64, ($addr) as *const u64) as i32
};
__v
}
};
}

macro_rules! trace_set_current_state__3 {
($state_value:expr) => {
loop {
if static_branch_unlikely__3!(unsafe { core::ptr::addr_of_mut!(__tracepoint_sched_set_state_tp.key) }) != 0 {
unsafe { __trace_set_current_state($state_value) };
}
if true {
break;
}
}
};
}

macro_rules! trace_set_current_state {
($state_value:expr) => {
loop {
if static_branch_unlikely!(unsafe { core::ptr::addr_of_mut!(__tracepoint_sched_set_state_tp.key) }) != 0 {
unsafe { __trace_set_current_state($state_value) };
}
if true {
break;
}
}
};
}

macro_rules! trace_set_current_state__2 {
($state_value:expr) => {
loop {
if static_branch_unlikely__2!(unsafe { core::ptr::addr_of_mut!(__tracepoint_sched_set_state_tp.key) }) != 0 {
unsafe { __trace_set_current_state($state_value) };
}
if true {
break;
}
}
};
}

macro_rules! trace_set_current_state__4 {
($state_value:expr) => {
loop {
if static_branch_unlikely__4!(unsafe { core::ptr::addr_of_mut!(__tracepoint_sched_set_state_tp.key) }) != 0 {
unsafe { __trace_set_current_state($state_value) };
}
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

macro_rules! wake_up_interruptible {
($x:expr) => {
unsafe { __wake_up($x, 1u32, 1i32, core::ptr::null_mut::<()>()) };
};
}

unsafe extern "C" {
    static mut kmsg_fops: file_operations;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut cpuinfo_op: seq_operations;
    static mut __tracepoint_sched_set_state_tp: tracepoint;
    static mut sighand_cachep: *mut kmem_cache;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut kstat: kernel_stat;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn _printk(arg0: *const u8, ...) -> i32;
    fn __bad_size_call_parameter();
    fn __compiletime_assert_0();
    fn ____wrong_branch_error() -> bool;
    fn __xchg_wrong_size();
    fn __xadd_wrong_size();
    fn strlen(arg0: *const u8) -> u64;
    fn snprintf(arg0: *mut u8, arg1: u64, arg2: *const u8, ...) -> i32;
    fn scnprintf(arg0: *mut u8, arg1: u64, arg2: *const u8, ...) -> i32;
    fn __compiletime_assert_16();
    fn __compiletime_assert_19();
    fn _raw_spin_lock_irqsave(arg0: *mut raw_spinlock) -> u64;
    fn _raw_spin_unlock_irqrestore(arg0: *mut raw_spinlock, arg1: u64);
    fn schedule();
    fn __trace_set_current_state(arg0: i32);
    fn refcount_warn_saturate(arg0: *mut refcount_struct, arg1: u32);
    fn default_wake_function(arg0: *mut wait_queue_entry, arg1: u32, arg2: i32, arg3: *mut ()) -> i32;
    fn add_wait_queue(arg0: *mut wait_queue_head, arg1: *mut wait_queue_entry);
    fn remove_wait_queue(arg0: *mut wait_queue_head, arg1: *mut wait_queue_entry);
    fn __wake_up(arg0: *mut wait_queue_head, arg1: u32, arg2: i32, arg3: *mut ()) -> i32;
    fn mutex_init_generic(arg0: *mut mutex);
    fn mutex_lock(arg0: *mut mutex);
    fn mutex_unlock(arg0: *mut mutex);
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_359();
    fn down_interruptible(arg0: *mut semaphore) -> i32;
    fn up(arg0: *mut semaphore);
    fn __compiletime_assert_384();
    fn kfree(arg0: *const ());
    fn __compiletime_assert_390();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn noop_llseek(arg0: *mut file, arg1: i64, arg2: i32) -> i64;
    fn seq_read(arg0: *mut file, arg1: *mut u8, arg2: u64, arg3: *mut i64) -> i64;
    fn seq_lseek(arg0: *mut file, arg1: i64, arg2: i32) -> i64;
    fn seq_printf(arg0: *mut seq_file, arg1: *const u8, ...);
    fn single_open(arg0: *mut file, arg1: Option<unsafe extern "C" fn(*mut seq_file, *mut ()) -> i32>, arg2: *mut ()) -> i32;
    fn single_release(arg0: *mut inode, arg1: *mut file) -> i32;
    fn debugfs_create_file_full(arg0: *const u8, arg1: u16, arg2: *mut dentry, arg3: *mut (), arg4: *const (), arg5: *const file_operations) -> *mut dentry;
    fn debugfs_create_file_short(arg0: *const u8, arg1: u16, arg2: *mut dentry, arg3: *mut (), arg4: *const (), arg5: *const debugfs_short_fops) -> *mut dentry;
    fn debugfs_create_dir(arg0: *const u8, arg1: *mut dentry) -> *mut dentry;
    fn debugfs_remove(arg0: *mut dentry);
    fn __kfifo_alloc_node(arg0: *mut __kfifo, arg1: u32, arg2: u64, arg3: u32, arg4: i32) -> i32;
    fn __kfifo_free(arg0: *mut __kfifo);
    fn __kfifo_in(arg0: *mut __kfifo, arg1: *const (), arg2: u32) -> u32;
    fn __kfifo_to_user(arg0: *mut __kfifo, arg1: *mut (), arg2: u64, arg3: *mut u32) -> i32;
    fn __kfifo_in_r(arg0: *mut __kfifo, arg1: *const (), arg2: u32, arg3: u64) -> u32;
    fn __kfifo_to_user_r(arg0: *mut __kfifo, arg1: *mut (), arg2: u64, arg3: *mut u32, arg4: u64) -> i32;
    fn hiddev_free(arg0: *mut kref);
    fn __compiletime_assert_638();
    fn __compiletime_assert_639();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
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
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct cacheline_padding {
    x: [u8; 0],
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
struct io_bitmap {
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
fn __must_check_overflow(overflow: bool) -> bool {
unlikely!(overflow) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_static_call_types_h_65 {
    r#type: u64,
    mods: *mut static_call_mod,
    sites: *mut static_call_site,
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
struct qrwlock {
    __slate_anon_0: __SlateRecord_qrwlock_types_h_27,
    wait_lock: qspinlock,
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
struct range {
    start: u64,
    end: u64,
}


#[inline(always)]
fn const_test_bit(nr: u64, addr: *const u64) -> bool {
let p: *const u64 = (addr as *const u64).wrapping_offset(BIT_WORD!(nr) as isize);
let mask: u64 = BIT_MASK!(nr);
let val: u64 = unsafe { *p };
val & mask != 0
}

#[inline(always)]
fn constant_test_bit(nr: i64, addr: *const u64) -> bool {
1u64 << (nr & 63i64) & unsafe { core::ptr::read_volatile(addr.wrapping_offset((nr >> 6i32) as isize)) } != 0
}

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
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
#[inline(always)]
fn _test_bit(nr: u64, addr: *const u64) -> bool {
instrument_atomic_read(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_bit(nr, addr)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_64_types_h_21 {
    pte: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_64_types_h_22 {
    pmd: u64,
}


// 4th level page in 5-level paging case
// 3rd level page
// PMD_SHIFT determines the size of the area a middle-level
// page table can map
// entries per page directory level
// We borrow bit 3 to remember PG_anon_exclusive.
#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_jump_label_h_95 {
    r#type: u64,
    entries: *mut jump_entry,
    next: *mut static_key_mod,
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
struct static_key_mod {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_types_h_295 {
    pgd: u64,
}


// No need to mask any bits for !PAE
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_types_h_368 {
    pud: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits117U0,
}


#[inline(always)]
fn ERR_PTR(error: i64) -> *mut () {
error as *mut ()
}

#[inline(always)]
fn IS_ERR(ptr: *const ()) -> bool {
IS_ERR_VALUE!(ptr as u64) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct static_key {
    enabled: __SlateRecord_types_h_190,
    __slate_anon_1: __SlateRecord_jump_label_h_95,
}


#[inline(always)]
fn arch_static_branch__const1(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_sched_set_state_tp, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
}

#[inline(always)]
fn arch_static_branch__const2(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_sched_set_state_tp, const 0 as u8, label {
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
struct perf_event {
}


#[inline(always)]
fn arch_static_branch__const3(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_sched_set_state_tp, const 0 as u8, label {
__slate_asm_target = true;
}, options(nostack, att_syntax));
if __slate_asm_target {
return true;
} else {
return false;
}
}
}

#[inline(always)]
fn arch_static_branch__const4(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_sched_set_state_tp, const 0 as u8, label {
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
struct static_key_false {
    key: static_key,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct orc_entry {
    sp_offset: i16,
    bp_offset: i16,
    __slate_bits_0: __slate_bits::__SlateBits124U0,
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
fn raw_atomic_fetch_add_relaxed(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
arch_atomic_fetch_add(i, v)
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
fn raw_atomic_fetch_sub_release(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
arch_atomic_fetch_sub!(i, v)
}

#[inline(always)]
fn atomic_fetch_add_relaxed(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_fetch_add_relaxed(i, v)
}

#[inline(always)]
fn atomic_fetch_sub_release(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
kcsan_release!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_fetch_sub_release(i, v)
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


fn __list_add_valid(new: *mut list_head, prev: *mut list_head, next: *mut list_head) -> bool {
true
}

fn __list_del_entry_valid(entry: *mut list_head) -> bool {
true
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
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
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_16() };
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

#[repr(C)]
#[derive(Clone, Copy)]
struct futex_pi_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct robust_list_head {
}


fn list_add_tail(new: *mut list_head, head: *mut list_head) {
__list_add(new, unsafe { (*head).prev }, head);
}

fn __list_del(prev: *mut list_head, next: *mut list_head) {
unsafe {
(*next).prev = prev;
}
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_19() };
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

#[repr(C)]
#[derive(Clone, Copy)]
struct futex_private_hash {
}


fn __list_del_entry(entry: *mut list_head) {
if !__list_del_entry_valid(entry) {
return;
}
__list_del(unsafe { (*entry).prev }, unsafe { (*entry).next });
}

fn list_del(entry: *mut list_head) {
__list_del_entry(entry);
unsafe {
(*entry).next = ((256i32 as *const u8).wrapping_offset(_AC__2!() as isize) as *mut ()) as *mut list_head;
}
unsafe {
(*entry).prev = ((290i32 as *const u8).wrapping_offset(_AC__2!() as isize) as *mut ()) as *mut list_head;
}
}

fn list_is_head(list: *const list_head, head: *const list_head) -> i32 {
(list == head) as i32
}

// list_for_each	-	iterate over a list
// @pos:	the &struct list_head to use as a loop cursor.
// @head:	the head for your list.
// list_for_each_entry	-	iterate over list of given type
// @pos:	the type * to use as a loop cursor.
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
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
    __slate_bits_0: __slate_bits::__SlateBits125U0,
    pkru: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sem_undo_list {
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
fn test_ti_thread_flag(ti: *mut thread_info, flag: i32) -> i32 {
test_bit__3!(flag, unsafe { core::ptr::addr_of_mut!((*ti).flags) })
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


#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer_clock_base {
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
struct seccomp_filter {
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
struct task_cputime {
    stime: u64,
    utime: u64,
    sum_exec_runtime: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_signal_h_25 {
    sig: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct workqueue_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
union sigval {
    sival_int: i32,
    sival_ptr: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
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
struct kernel_siginfo {
    __slate_anon_0: __SlateRecord_signal_types_h_14,
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
    __slate_bits_0: __slate_bits::__SlateBits220U0,
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
// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
#[inline(always)]
fn spinlock_check(lock: *mut spinlock) -> *mut raw_spinlock {
unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) }
}

#[inline(always)]
fn spin_unlock_irqrestore(lock: *mut spinlock, flags: u64) {
raw_spin_unlock_irqrestore!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) }, flags);
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
struct posix_cputimer_base {
    nextevt: u64,
    tqhead: timerqueue_head,
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
struct rq_flags {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqcount_spinlock {
    seqcount: seqcount,
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
struct seqlock {
    seqcount: seqcount_spinlock,
    lock: spinlock,
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


// Convenience macros for the sake of set_current_state:
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
struct gendisk {
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
struct vm_struct {
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
    __slate_bits_0: __slate_bits::__SlateBits283U0,
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
struct mem_cgroup {
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
fn test_tsk_thread_flag(tsk: *mut task_struct, flag: i32) -> i32 {
test_ti_thread_flag(task_thread_info!(tsk), flag)
}

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
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
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
struct rwsem_waiter {
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
struct uprobe {
}


fn __refcount_inc(r: *mut refcount_struct, oldp: *mut i32) {
__refcount_add(1i32, r, oldp);
}

fn refcount_inc(r: *mut refcount_struct) {
__refcount_inc(r, core::ptr::null_mut::<i32>());
}

fn __refcount_sub_and_test(i: i32, r: *mut refcount_struct, oldp: *mut i32) -> bool {
let old: i32 = atomic_fetch_sub_release(i, unsafe { core::ptr::addr_of_mut!((*r).refs) });
if !oldp.is_null() {
unsafe {
*oldp = old;
}
}
if old > 0 && old == i {
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
return true;
}
if unlikely!(old <= 0 || old.wrapping_sub(i) < 0) != 0 {
unsafe { refcount_warn_saturate(r, 3u32) };
}
false
}

fn __refcount_dec_and_test(r: *mut refcount_struct, oldp: *mut i32) -> bool {
__refcount_sub_and_test(1i32, r, oldp)
}

fn refcount_dec_and_test(r: *mut refcount_struct) -> bool {
__refcount_dec_and_test(r, core::ptr::null_mut::<i32>())
}

// wait_queue_entry::flags
#[repr(C)]
#[derive(Clone, Copy)]
struct wait_queue_entry {
    flags: u32,
    private: *mut (),
    func: Option<unsafe extern "C" fn(*mut wait_queue_entry, u32, i32, *mut ()) -> i32>,
    entry: list_head,
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
struct uprobe_xol_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct completion {
    done: u32,
    wait: swait_queue_head,
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
struct hlist_nulls_node {
    next: *mut hlist_nulls_node,
    pprev: *mut *mut hlist_nulls_node,
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


fn __mutex_init(lock: *mut mutex, name: *const u8, key: *mut lock_class_key) {
unsafe { mutex_init_generic(lock) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kref {
    refcount: refcount_struct,
}


fn kref_get(kref: *mut kref) {
refcount_inc(unsafe { core::ptr::addr_of_mut!((*kref).refcount) });
}

fn kref_put(kref: *mut kref, release: Option<unsafe extern "C" fn(*mut kref)>) -> i32 {
if refcount_dec_and_test(unsafe { core::ptr::addr_of_mut!((*kref).refcount) }) {
unsafe { release.unwrap()(kref) };
return 1;
}
0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ldt_struct {
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
struct rcu_work {
    work: work_struct,
    rcu: callback_head,
    wq: *mut workqueue_struct,
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
struct arch_uprobe_task {
    saved_scratch_register: u64,
    saved_trap_nr: u32,
    saved_tf: u32,
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
struct device_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache {
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
struct mnt_idmap {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_userfaultfd_ctx {
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
    __slate_bits_0: __slate_bits::__SlateBits415U0,
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
struct assoc_array_ptr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct net {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mm_struct {
    __slate_anon_0: __SlateRecord_mm_types_h_1176,
    flexible_array: [u8; 0],
}


// Clear the access bit
// Clear the dirty bit
// MMF_HAS_PINNED: Whether this mm has pinned any pages.  This can be either
// replaced in the future by mm.pinned_vm when it becomes stable, or grow into
// a counter on its own. We're aggresive on this bit for now: even if the
// pinned pages were unpinned later on, we'll still keep this bit set for the
// lifecycle of this mm, just for simplicity.
// FOLL_PIN has run, never cleared
#[repr(C)]
#[derive(Clone, Copy)]
struct key_type {
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
struct hlist_bl_node {
    next: *mut hlist_bl_node,
    pprev: *mut *mut hlist_bl_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_cred_h_147 {
    non_rcu: i32,
    rcu: callback_head,
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
struct pidfs_attr {
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
struct __SlateRecord_pid_h_62 {
    ino: u64,
    pidfs_hash: rhash_head,
    stashed: *mut dentry,
    attr: *mut pidfs_attr,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_capability_h_24 {
    val: u64,
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
struct tty_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ctl_table_set {
    is_seen: Option<unsafe extern "C" fn(*mut ctl_table_set) -> i32>,
    dir: ctl_dir,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_audit_buf {
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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
struct fserror_event {
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
    __slate_bits_0: __slate_bits::__SlateBits286U0,
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
fn task_sigpending(p: *mut task_struct) -> i32 {
unlikely__2!(test_tsk_thread_flag(p, 1i32)) as i32
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


fn signal_pending(p: *mut task_struct) -> i32 {
// TIF_NOTIFY_SIGNAL isn't really a signal, but it requires the same
// behavior in terms of ensuring that we break out of wait loops
// so that notify signal callbacks can be processed.
if unlikely__2!(test_tsk_thread_flag(p, 2i32)) != 0 {
return 1;
}
task_sigpending(p)
}

fn task_rlimit(task: *const task_struct, limit: u32) -> u64 {
let mut __v0: u64;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_359() };
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


fn rlimit(limit: u32) -> u64 {
task_rlimit(get_current() as *const task_struct, limit)
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
struct rcu_sync {
    gp_state: i32,
    gp_count: i32,
    gp_wait: wait_queue_head,
    cb_head: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct semaphore_waiter {
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
struct request_queue {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_context {
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
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
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
struct swap_info_struct {
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
struct quota_format_type {
    qf_fmt_id: i32,
    qf_ops: *const quota_format_ops,
    qf_owner: *mut module,
    qf_next: *mut quota_format_type,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_acl {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_cmd {
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
struct semaphore {
    lock: raw_spinlock,
    count: u32,
    first_waiter: *mut semaphore_waiter,
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
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_384() };
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
struct percpu_ref_data {
    count: __SlateRecord_types_h_197,
    release: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    confirm_switch: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    __slate_bits_0: __slate_bits::__SlateBits575U0,
    rcu: callback_head,
    r#ref: *mut percpu_ref,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_ref {
    percpu_count_ptr: u64,
    data: *mut percpu_ref_data,
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
unsafe { __compiletime_assert_390() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
} else {
loop {
unsafe {
core::arch::asm!("391: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 391b - ., 3; .popsection\n# {0}", const 391 as i32, options(nostack, att_syntax));
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


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_open_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_iattrs {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
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


// XArray tags, for tagging dirty and writeback pages in the pagecache.
#[repr(C)]
#[derive(Clone, Copy)]
struct inode_state_flags {
    __state: u32,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct sock {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct fasync_struct {
    fa_lock: rwlock,
    magic: i32,
    fa_fd: i32,
    fa_next: *mut fasync_struct,
    fa_file: *mut file,
    fa_rcu: callback_head,
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
struct debugfs_short_fops {
    read: Option<unsafe extern "C" fn(*mut file, *mut u8, u64, *mut i64) -> i64>,
    write: Option<unsafe extern "C" fn(*mut file, *const u8, u64, *mut i64) -> i64>,
    llseek: Option<unsafe extern "C" fn(*mut file, i64, i32) -> i64>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wake_irq {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __kfifo {
    r#in: u32,
    out: u32,
    mask: u32,
    esize: u32,
    data: *mut (),
}


// DECLARE_KFIFO_PTR - macro to declare a fifo pointer object
// @fifo: name of the declared fifo
// @type: type of the fifo elements
fn __kfifo_int_must_check_helper(val: i32) -> i32 {
val
}

#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_qos {
}


fn __kfifo_alloc(fifo: *mut __kfifo, size: u32, esize: u64, gfp_mask: u32) -> i32 {
unsafe { __kfifo_alloc_node(fifo, size, esize, gfp_mask, 1i32.wrapping_neg()) }
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
struct fwnode_handle {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct poll_table_struct {
    _qproc: Option<unsafe extern "C" fn(*mut file, *mut wait_queue_head, *mut poll_table_struct)>,
    _key: u32,
}


fn poll_wait(filp: *mut file, wait_address: *mut wait_queue_head, p: *mut poll_table_struct) {
if !p.is_null() && (unsafe { (*p)._qproc }) != None {
unsafe { unsafe { (*p)._qproc }.unwrap()(filp, wait_address, p) };
// This memory barrier is paired in the wq_has_sleeper().
// See the comment above prepare_to_wait(), we need to
// ensure that subsequent tests in this thread can't be
// reordered with __add_wait_queue() in _qproc() paths.
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
}

#[repr(C)]
#[derive(Clone, Copy)]
struct hid_device_id {
    bus: u16,
    group: u16,
    vendor: u32,
    product: u32,
    driver_data: u64,
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
struct folio_queue {
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
struct input_keymap_entry {
    flags: u8,
    len: u8,
    index: u16,
    keycode: u32,
    scancode: [u8; 32],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_uio_h_78 {
    nr_segs: u64,
    folioq_slot: u8,
    xarray_start: i64,
}


// Values describing the status of a force-feedback effect
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
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
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
struct module_param_attrs {
}


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
    __slate_bits_0: __slate_bits::__SlateBits657U0,
    vm_ops: *const vm_operations_struct,
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
struct sysfs_ops {
    show: Option<unsafe extern "C" fn(*mut kobject, *mut attribute, *mut u8) -> i64>,
    store: Option<unsafe extern "C" fn(*mut kobject, *mut attribute, *const u8, u64) -> i64>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct input_dev_poller {
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
    __slate_bits_0: __slate_bits::__SlateBits587U0,
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
struct input_mt {
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
struct pm_message {
    event: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_memremap_h_143 {
    range: range,
    __slate_anon_1: __SlateRecord_memremap_h_145,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_memremap_h_145 {
    __empty_ranges: __SlateRecord_memremap_h_145_2,
    ranges: [range; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_memremap_h_145_2 {
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
struct __SlateRecord_mm_h_754 {
    vma: *mut vm_area_struct,
    gfp_mask: u32,
    pgoff: u64,
    address: u64,
    real_address: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_mm_h_768 {
    orig_pte: __SlateRecord_pgtable_64_types_h_21,
    orig_pmd: __SlateRecord_pgtable_64_types_h_22,
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
    __slate_bits_0: __slate_bits::__SlateBits682U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits682U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits682U2,
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
    __slate_bits_3: __slate_bits::__SlateBits682U3,
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
struct ipc_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mnt_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct proc_ns_operations {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct time_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uts_namespace {
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct __SlateRecord_ns_common_types_h_112 {
    __ns_ref: refcount_struct,
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
struct cgroup_taskset {
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
union __SlateRecord_cgroup_defs_h_644 {
    __slate_anon_0: __SlateRecord_cgroup_defs_h_645,
    __slate_anon_1: __SlateRecord_cgroup_defs_h_646,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cgroup_defs_h_645 {
    __empty_ancestors: __SlateRecord_cgroup_defs_h_645_2,
    ancestors: [*mut cgroup; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cgroup_defs_h_645_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cgroup_defs_h_646 {
    _root_ancestor: *mut cgroup,
    __slate_anon_1: __SlateRecord_cgroup_defs_h_648,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cgroup_defs_h_648 {
    __empty__low_ancestors: __SlateRecord_cgroup_defs_h_648_2,
    _low_ancestors: [*mut cgroup; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_cgroup_defs_h_648_2 {
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
struct disk_stats {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_holder_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct partition_meta_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_pin {
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
struct __SlateRecord_hid_debug_h_31 {
    __slate_anon_0: __SlateRecord_hid_debug_h_31_2,
    buf: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_hid_debug_h_31_2 {
    kfifo: __kfifo,
    r#type: *mut u8,
    const_type: *const u8,
    rectype: *mut [u8; 0],
    ptr: *mut u8,
    ptr_const: *const u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_msi_info {
    domain: *mut irq_domain,
    data: *mut msi_device_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_hid_debug_c_3695 {
    __slate_anon_0: __SlateRecord_hid_debug_c_3695_2,
    buf: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_hid_debug_c_3695_2 {
    kfifo: __kfifo,
    r#type: *mut u8,
    const_type: *const u8,
    rectype: *mut [u8; 0],
    ptr: *mut u8,
    ptr_const: *const u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_hid_debug_c_3793 {
    __slate_anon_0: __SlateRecord_hid_debug_c_3793_2,
    buf: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_hid_debug_c_3793_2 {
    kfifo: __kfifo,
    r#type: *mut u8,
    const_type: *const u8,
    rectype: *mut [u8; 0],
    ptr: *mut u8,
    ptr_const: *const u8,
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
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits541U0 {
        #[bits(1)]
        pub for_kupdate: u32,
        #[bits(1)]
        pub for_background: u32,
        #[bits(1)]
        pub tagged_writepages: u32,
        #[bits(1)]
        pub range_cyclic: u32,
        #[bits(1)]
        pub for_sync: u32,
        #[bits(1)]
        pub unpinned_netfs_wb: u32,
        #[bits(1)]
        pub no_cgroup_owner: u32,
        #[bits(1, access = na)]
        pub __slate_pad_7: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits575U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits283U0 {
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
    pub struct __SlateBits657U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits799U0 {
        #[bits(1)]
        pub early_init: bool,
        #[bits(1)]
        pub implicit_on_dfl: bool,
        #[bits(1)]
        pub threaded: bool,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits286U0 {
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
    pub struct __SlateBits117U0 {
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
    pub struct __SlateBits125U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits682U0 {
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
    pub struct __SlateBits679U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits682U1 {
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
    pub struct __SlateBits682U2 {
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
    pub struct __SlateBits682U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits587U0 {
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
    pub struct __SlateBits124U0 {
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
    pub struct __SlateBits415U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits220U0 {
        #[bits(1)]
        pub faultable: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
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
    __slate_bits_0: __slate_bits::__SlateBits679U0,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct vmem_altmap {
    base_pfn: u64,
    end_pfn: u64,
    reserve: u64,
    free: u64,
    align: u64,
    alloc: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pagemap_ops {
    folio_free: Option<unsafe extern "C" fn(*mut folio)>,
    migrate_to_ram: Option<unsafe extern "C" fn(*mut vm_fault) -> u32>,
    memory_failure: Option<unsafe extern "C" fn(*mut dev_pagemap, u64, u64, i32) -> i32>,
    folio_split: Option<unsafe extern "C" fn(*mut folio, *mut folio)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pagemap {
    altmap: vmem_altmap,
    r#ref: percpu_ref,
    done: completion,
    r#type: u32,
    flags: u32,
    vmemmap_shift: u64,
    ops: *const dev_pagemap_ops,
    owner: *mut (),
    nr_range: i32,
    __slate_anon_9: __SlateRecord_memremap_h_143,
}


// VMA basic access permission flags
// These flags can be updated atomically via VMA/mmap read lock.
#[repr(C)]
#[derive(Clone, Copy)]
struct vm_fault {
    __slate_anon_0: __SlateRecord_mm_h_754,
    flags: u32,
    pmd: *mut __SlateRecord_pgtable_64_types_h_22,
    pud: *mut __SlateRecord_pgtable_types_h_368,
    __slate_anon_4: __SlateRecord_mm_h_768,
    cow_page: *mut page,
    page: *mut page,
    pte: *mut __SlateRecord_pgtable_64_types_h_21,
    ptl: *mut spinlock,
    prealloc_pte: *mut page,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_operations_struct {
    open: Option<unsafe extern "C" fn(*mut vm_area_struct)>,
    close: Option<unsafe extern "C" fn(*mut vm_area_struct)>,
    mapped: Option<unsafe extern "C" fn(u64, u64, u64, *const file, *mut *mut ()) -> i32>,
    may_split: Option<unsafe extern "C" fn(*mut vm_area_struct, u64) -> i32>,
    mremap: Option<unsafe extern "C" fn(*mut vm_area_struct) -> i32>,
    mprotect: Option<unsafe extern "C" fn(*mut vm_area_struct, u64, u64, u64) -> i32>,
    fault: Option<unsafe extern "C" fn(*mut vm_fault) -> u32>,
    huge_fault: Option<unsafe extern "C" fn(*mut vm_fault, u32) -> u32>,
    map_pages: Option<unsafe extern "C" fn(*mut vm_fault, u64, u64) -> u32>,
    pagesize: Option<unsafe extern "C" fn(*mut vm_area_struct) -> u64>,
    page_mkwrite: Option<unsafe extern "C" fn(*mut vm_fault) -> u32>,
    pfn_mkwrite: Option<unsafe extern "C" fn(*mut vm_fault) -> u32>,
    access: Option<unsafe extern "C" fn(*mut vm_area_struct, u64, *mut (), i32, i32) -> i32>,
    name: Option<unsafe extern "C" fn(*mut vm_area_struct) -> *const u8>,
    set_policy: Option<unsafe extern "C" fn(*mut vm_area_struct, *mut mempolicy) -> i32>,
    get_policy: Option<unsafe extern "C" fn(*mut vm_area_struct, u64, *mut u64) -> *mut mempolicy>,
}


// Helper macro which bitwise-or combines the specified input flags into a
// vma_flags_t bitmap value. E.g.:
//
// vma_flags_t flags = mk_vma_flags(VMA_IO_BIT, VMA_PFNMAP_BIT,
//              VMA_DONTEXPAND_BIT, VMA_DONTDUMP_BIT);
//
// The compiler cleverly optimises away all of the work and this ends up being
// equivalent to aggregating the values manually.
// Test whether ALL specified VMA flags are set, e.g.:
//
// if (vma_flags_test_all(flags, VMA_READ_BIT, VMA_MAYREAD_BIT)) { ... }
// 127: arbitrary random number, small enough to assemble well
// GUP_PIN_COUNTING_BIAS, and the associated functions that use it, overload
// the page's refcount so that two separate items are tracked: the original page
// reference count, and also a new count of how many pin_user_pages() calls were
// made against the page. ("gup-pinned" is another term for the latter).
//
// With this scheme, pin_user_pages() becomes special: such pages are marked as
// distinct from normal pages. As such, the unpin_user_page() call (and its
// variants) must be used in order to release gup-pinned pages.
//
// Choice of value:
//
// By making GUP_PIN_COUNTING_BIAS a power of two, debugging of page reference
// counts with respect to pin_user_pages() and unpin_user_page() becomes
// simpler, due to the fact that adding an even power of two to the page
// refcount has the effect of using only the upper N bits, for the code that
// counts up using the bias value. This means that the lower bits are left for
// the exclusive use of the original code that increments and decrements by one
// (or at least, by much smaller values than the bias value).
//
// Of course, once the lower bits overflow into the upper bits (and this is
// OK, because subtraction recovers the original values), then visual inspection
// no longer suffices to directly view the separate counts. However, for normal
// applications that don't have huge page reference counts, this won't be an
// issue.
//
// Locking: the lockless algorithm described in folio_try_get_rcu()
// provides safe operation for get_user_pages(), folio_mkclean() and
// other calls that race to set up page table entries.
// Mask used for checking in page_pool_page_is_pp() below. page->pp_magic is
// OR'ed with PP_SIGNATURE after the allocation in order to preserve bit 0 for
// the head page of compound page and bit 1 for pfmemalloc page, as well as the
// bits used for the DMA index. page_is_pfmemalloc() is checked in
// __page_pool_put_page() to avoid recycling the pfmemalloc page.
#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_mod {
    next: *mut static_call_mod,
    r#mod: *mut module,
    sites: *mut static_call_site,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct taskstats {
    version: u16,
    ac_exitcode: u32,
    ac_flag: u8,
    ac_nice: u8,
    cpu_count: u64,
    cpu_delay_total: u64,
    blkio_count: u64,
    blkio_delay_total: u64,
    swapin_count: u64,
    swapin_delay_total: u64,
    cpu_run_real_total: u64,
    cpu_run_virtual_total: u64,
    ac_comm: [u8; 32],
    ac_sched: u8,
    ac_pad: [u8; 3],
    __slate_pad_15: [u8; 4],
    ac_uid: u32,
    ac_gid: u32,
    ac_pid: u32,
    ac_ppid: u32,
    ac_btime: u32,
    ac_etime: u64,
    ac_utime: u64,
    ac_stime: u64,
    ac_minflt: u64,
    ac_majflt: u64,
    coremem: u64,
    virtmem: u64,
    hiwater_rss: u64,
    hiwater_vm: u64,
    read_char: u64,
    write_char: u64,
    read_syscalls: u64,
    write_syscalls: u64,
    read_bytes: u64,
    write_bytes: u64,
    cancelled_write_bytes: u64,
    nvcsw: u64,
    nivcsw: u64,
    ac_utimescaled: u64,
    ac_stimescaled: u64,
    cpu_scaled_run_real_total: u64,
    freepages_count: u64,
    freepages_delay_total: u64,
    thrashing_count: u64,
    thrashing_delay_total: u64,
    ac_btime64: u64,
    compact_count: u64,
    compact_delay_total: u64,
    ac_tgid: u32,
    ac_tgetime: u64,
    ac_exe_dev: u64,
    ac_exe_inode: u64,
    wpcopy_count: u64,
    wpcopy_delay_total: u64,
    irq_count: u64,
    irq_delay_total: u64,
    cpu_delay_max: u64,
    cpu_delay_min: u64,
    blkio_delay_max: u64,
    blkio_delay_min: u64,
    swapin_delay_max: u64,
    swapin_delay_min: u64,
    freepages_delay_max: u64,
    freepages_delay_min: u64,
    thrashing_delay_max: u64,
    thrashing_delay_min: u64,
    compact_delay_max: u64,
    compact_delay_min: u64,
    wpcopy_delay_max: u64,
    wpcopy_delay_min: u64,
    irq_delay_max: u64,
    irq_delay_min: u64,
    cpu_delay_max_ts: __kernel_timespec,
    blkio_delay_max_ts: __kernel_timespec,
    swapin_delay_max_ts: __kernel_timespec,
    freepages_delay_max_ts: __kernel_timespec,
    thrashing_delay_max_ts: __kernel_timespec,
    compact_delay_max_ts: __kernel_timespec,
    wpcopy_delay_max_ts: __kernel_timespec,
    irq_delay_max_ts: __kernel_timespec,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct nsproxy {
    count: refcount_struct,
    uts_ns: *mut uts_namespace,
    ipc_ns: *mut ipc_namespace,
    mnt_ns: *mut mnt_namespace,
    pid_ns_for_children: *mut pid_namespace,
    net_ns: *mut net,
    time_ns: *mut time_namespace,
    time_ns_for_children: *mut time_namespace,
    cgroup_ns: *mut cgroup_namespace,
}


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


#[repr(C)]
#[derive(Clone, Copy)]
struct kernel_stat {
    irqs_sum: u64,
    softirqs: [u32; 10],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct u64_stats_sync {
}


core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_hid_resolv_usage: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_hid_dump_field: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_hid_dump_device: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_hid_debug_event: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_hid_dump_report: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_hid_dump_input: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous", sym hid_resolv_usage, sym hid_dump_field, sym hid_dump_device, sym hid_debug_event, sym hid_dump_report, sym hid_dump_input, options(att_syntax));

#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_bpf {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct psi_group {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_file {
    kn: *mut kernfs_node,
    notified_at: u64,
    notify_timer: timer_list,
    lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_subsys_state {
    cgroup: *mut cgroup,
    ss: *mut cgroup_subsys,
    refcnt: percpu_ref,
    rstat_cpu: *mut css_rstat_cpu,
    sibling: list_head,
    children: list_head,
    id: i32,
    flags: u32,
    serial_nr: u64,
    online_cnt: __SlateRecord_types_h_190,
    destroy_work: work_struct,
    destroy_rwork: rcu_work,
    parent: *mut cgroup_subsys_state,
    nr_descendants: i32,
    nr_populated_csets: i32,
    nr_populated_children: i32,
    kill_finish_work: work_struct,
    rstat_flush_next: *mut cgroup_subsys_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct css_set {
    subsys: [*mut cgroup_subsys_state; 14],
    refcount: refcount_struct,
    dom_cset: *mut css_set,
    dfl_cgrp: *mut cgroup,
    nr_tasks: i32,
    tasks: list_head,
    mg_tasks: list_head,
    dying_tasks: list_head,
    task_iters: list_head,
    e_cset_node: [list_head; 14],
    threaded_csets: list_head,
    threaded_csets_node: list_head,
    hlist: hlist_node,
    cgrp_links: list_head,
    mg_src_preload_node: list_head,
    mg_dst_preload_node: list_head,
    mg_node: list_head,
    mg_src_cgrp: *mut cgroup,
    mg_dst_cgrp: *mut cgroup,
    mg_dst_cset: *mut css_set,
    dead: bool,
    callback_head: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_base_stat {
    cputime: task_cputime,
    ntime: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct css_rstat_cpu {
    updated_children: *mut cgroup_subsys_state,
    updated_next: *mut cgroup_subsys_state,
    lnode: llist_node,
    owner: *mut cgroup_subsys_state,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_rstat_base_cpu {
    bsync: u64_stats_sync,
    bstat: cgroup_base_stat,
    last_bstat: cgroup_base_stat,
    subtree_bstat: cgroup_base_stat,
    last_subtree_bstat: cgroup_base_stat,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_freezer_state {
    freeze: bool,
    e_freeze: bool,
    nr_frozen_descendants: i32,
    nr_frozen_tasks: i32,
    freeze_seq: seqcount_spinlock,
    freeze_start_nsec: u64,
    frozen_nsec: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup {
    self_: cgroup_subsys_state,
    flags: u64,
    level: i32,
    max_depth: i32,
    nr_descendants: i32,
    nr_dying_descendants: i32,
    max_descendants: i32,
    nr_populated_domain_children: i32,
    nr_populated_threaded_children: i32,
    nr_threaded_children: i32,
    kill_seq: u32,
    kn: *mut kernfs_node,
    procs_file: cgroup_file,
    events_file: cgroup_file,
    psi_files: [cgroup_file; 0],
    subtree_control: u32,
    subtree_ss_mask: u32,
    old_subtree_control: u32,
    old_subtree_ss_mask: u32,
    subsys: [*mut cgroup_subsys_state; 14],
    nr_dying_subsys: [i32; 14],
    root: *mut cgroup_root,
    cset_links: list_head,
    e_csets: [list_head; 14],
    dom_cgrp: *mut cgroup,
    old_dom_cgrp: *mut cgroup,
    rstat_base_cpu: *mut cgroup_rstat_base_cpu,
    _pad_: cacheline_padding,
    last_bstat: cgroup_base_stat,
    bstat: cgroup_base_stat,
    prev_cputime: prev_cputime,
    pidlists: list_head,
    pidlist_mutex: mutex,
    offline_waitq: wait_queue_head,
    release_agent_work: work_struct,
    psi: *mut psi_group,
    bpf: cgroup_bpf,
    freezer: cgroup_freezer_state,
    __slate_anon_38: __SlateRecord_cgroup_defs_h_644,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_root {
    kf_root: *mut kernfs_root,
    subsys_mask: u32,
    hierarchy_id: i32,
    root_list: list_head,
    rcu: callback_head,
    nr_cgrps: __SlateRecord_types_h_190,
    flags: u32,
    release_agent_path: [u8; 4096],
    name: [u8; 64],
    cgrp: cgroup,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cftype {
    name: [u8; 64],
    private: u64,
    max_write_len: u64,
    flags: u32,
    file_offset: u32,
    ss: *mut cgroup_subsys,
    node: list_head,
    kf_ops: *mut kernfs_ops,
    open: Option<unsafe extern "C" fn(*mut kernfs_open_file) -> i32>,
    release: Option<unsafe extern "C" fn(*mut kernfs_open_file)>,
    read_u64: Option<unsafe extern "C" fn(*mut cgroup_subsys_state, *mut cftype) -> u64>,
    read_s64: Option<unsafe extern "C" fn(*mut cgroup_subsys_state, *mut cftype) -> i64>,
    seq_show: Option<unsafe extern "C" fn(*mut seq_file, *mut ()) -> i32>,
    seq_start: Option<unsafe extern "C" fn(*mut seq_file, *mut i64) -> *mut ()>,
    seq_next: Option<unsafe extern "C" fn(*mut seq_file, *mut (), *mut i64) -> *mut ()>,
    seq_stop: Option<unsafe extern "C" fn(*mut seq_file, *mut ())>,
    write_u64: Option<unsafe extern "C" fn(*mut cgroup_subsys_state, *mut cftype, u64) -> i32>,
    write_s64: Option<unsafe extern "C" fn(*mut cgroup_subsys_state, *mut cftype, i64) -> i32>,
    write: Option<unsafe extern "C" fn(*mut kernfs_open_file, *mut u8, u64, i64) -> i64>,
    poll: Option<unsafe extern "C" fn(*mut kernfs_open_file, *mut poll_table_struct) -> u32>,
    lockdep_key: lock_class_key,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_subsys {
    css_alloc: Option<unsafe extern "C" fn(*mut cgroup_subsys_state) -> *mut cgroup_subsys_state>,
    css_online: Option<unsafe extern "C" fn(*mut cgroup_subsys_state) -> i32>,
    css_offline: Option<unsafe extern "C" fn(*mut cgroup_subsys_state)>,
    css_released: Option<unsafe extern "C" fn(*mut cgroup_subsys_state)>,
    css_free: Option<unsafe extern "C" fn(*mut cgroup_subsys_state)>,
    css_reset: Option<unsafe extern "C" fn(*mut cgroup_subsys_state)>,
    css_killed: Option<unsafe extern "C" fn(*mut cgroup_subsys_state)>,
    css_rstat_flush: Option<unsafe extern "C" fn(*mut cgroup_subsys_state, i32)>,
    css_extra_stat_show: Option<unsafe extern "C" fn(*mut seq_file, *mut cgroup_subsys_state) -> i32>,
    css_local_stat_show: Option<unsafe extern "C" fn(*mut seq_file, *mut cgroup_subsys_state) -> i32>,
    can_attach: Option<unsafe extern "C" fn(*mut cgroup_taskset) -> i32>,
    cancel_attach: Option<unsafe extern "C" fn(*mut cgroup_taskset)>,
    attach: Option<unsafe extern "C" fn(*mut cgroup_taskset)>,
    can_fork: Option<unsafe extern "C" fn(*mut task_struct, *mut css_set) -> i32>,
    cancel_fork: Option<unsafe extern "C" fn(*mut task_struct, *mut css_set)>,
    fork: Option<unsafe extern "C" fn(*mut task_struct)>,
    exit: Option<unsafe extern "C" fn(*mut task_struct)>,
    release: Option<unsafe extern "C" fn(*mut task_struct)>,
    bind: Option<unsafe extern "C" fn(*mut cgroup_subsys_state)>,
    __slate_bits_0: __slate_bits::__SlateBits799U0,
    id: i32,
    name: *const u8,
    legacy_name: *const u8,
    root: *mut cgroup_root,
    css_idr: idr,
    cfts: list_head,
    dfl_cftypes: *mut cftype,
    legacy_cftypes: *mut cftype,
    depends_on: u32,
    rstat_ss_lock: spinlock,
    lhead: *mut llist_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_namespace {
    ns: ns_common,
    user_ns: *mut user_namespace,
    ucounts: *mut ucounts,
    root_cset: *mut css_set,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fprop_local_percpu {
    events: percpu_counter,
    period: u32,
    lock: raw_spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bdi_writeback {
    bdi: *mut backing_dev_info,
    state: u64,
    last_old_flush: u64,
    b_dirty: list_head,
    b_io: list_head,
    b_more_io: list_head,
    b_dirty_time: list_head,
    list_lock: spinlock,
    writeback_inodes: __SlateRecord_types_h_190,
    stat: [percpu_counter; 5],
    bw_time_stamp: u64,
    dirtied_stamp: u64,
    written_stamp: u64,
    write_bandwidth: u64,
    avg_write_bandwidth: u64,
    dirty_ratelimit: u64,
    balanced_dirty_ratelimit: u64,
    completions: fprop_local_percpu,
    dirty_exceeded: i32,
    start_all_reason: u32,
    work_lock: spinlock,
    work_list: list_head,
    dwork: delayed_work,
    bw_dwork: delayed_work,
    bdi_node: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct backing_dev_info {
    id: u64,
    rb_node: rb_node,
    bdi_list: list_head,
    ra_pages: u64,
    io_pages: u64,
    refcnt: kref,
    capabilities: u32,
    min_ratio: u32,
    max_ratio: u32,
    max_prop_frac: u32,
    tot_write_bandwidth: __SlateRecord_types_h_197,
    last_bdp_sleep: u64,
    wb: bdi_writeback,
    wb_list: list_head,
    wb_waitq: wait_queue_head,
    dev: *mut device,
    dev_name: [u8; 64],
    owner: *mut device,
    debug_dir: *mut dentry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bio_vec {
    bv_page: *mut page,
    bv_len: u32,
    bv_offset: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct block_device {
    bd_start_sect: u64,
    bd_nr_sectors: u64,
    bd_disk: *mut gendisk,
    bd_queue: *mut request_queue,
    bd_stats: *mut disk_stats,
    bd_stamp: u64,
    __bd_flags: __SlateRecord_types_h_190,
    bd_dev: u32,
    bd_mapping: *mut address_space,
    bd_openers: __SlateRecord_types_h_190,
    bd_size_lock: spinlock,
    bd_claiming: *mut (),
    bd_holder: *mut (),
    bd_holder_ops: *const blk_holder_ops,
    bd_holder_lock: mutex,
    bd_holders: i32,
    bd_holder_dir: *mut kobject,
    bd_fsfreeze_count: __SlateRecord_types_h_190,
    bd_fsfreeze_mutex: mutex,
    bd_meta_info: *mut partition_meta_info,
    bd_writers: i32,
    bd_security: *mut (),
    bd_device: device,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct folio_batch {
    nr: u8,
    i: u8,
    percpu_pvec_drained: bool,
    folios: [*mut folio; 31],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct writeback_control {
    nr_to_write: i64,
    pages_skipped: i64,
    range_start: i64,
    range_end: i64,
    sync_mode: u32,
    __slate_bits_0: __slate_bits::__SlateBits541U0,
    fbatch: folio_batch,
    index: u64,
    saved_err: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xattr_handler {
    name: *const u8,
    prefix: *const u8,
    flags: i32,
    list: Option<unsafe extern "C" fn(*mut dentry) -> bool>,
    get: Option<unsafe extern "C" fn(*const xattr_handler, *mut dentry, *mut inode, *const u8, *mut (), u64) -> i32>,
    set: Option<unsafe extern "C" fn(*const xattr_handler, *mut mnt_idmap, *mut dentry, *mut inode, *const u8, *const (), u64, i32) -> i32>,
}


// MFD_EXEC implied if unset
#[repr(C)]
#[derive(Clone, Copy)]
struct pid_namespace {
    idr: idr,
    rcu: callback_head,
    pid_allocated: u32,
    memfd_noexec_scope: i32,
    set: ctl_table_set,
    sysctls: *mut ctl_table_header,
    child_reaper: *mut task_struct,
    pid_cachep: *mut kmem_cache,
    level: u32,
    pid_max: i32,
    parent: *mut pid_namespace,
    bacct: *mut fs_pin,
    user_ns: *mut user_namespace,
    ucounts: *mut ucounts,
    reboot: i32,
    ns: ns_common,
    work: work_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hid_collection {
    parent_idx: i32,
    r#type: u32,
    usage: u32,
    level: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hid_usage {
    hid: u32,
    collection_index: u32,
    usage_index: u32,
    resolution_multiplier: i8,
    wheel_factor: i8,
    code: u16,
    r#type: u8,
    hat_min: i16,
    hat_max: i16,
    hat_dir: i16,
    wheel_accumulated: i16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hid_field {
    physical: u32,
    logical: u32,
    application: u32,
    usage: *mut hid_usage,
    maxusage: u32,
    flags: u32,
    report_offset: u32,
    report_size: u32,
    report_count: u32,
    report_type: u32,
    value: *mut i32,
    new_value: *mut i32,
    usages_priorities: *mut i32,
    logical_minimum: i32,
    logical_maximum: i32,
    physical_minimum: i32,
    physical_maximum: i32,
    unit_exponent: i32,
    unit: u32,
    ignored: bool,
    report: *mut hid_report,
    index: u32,
    hidinput: *mut hid_input,
    dpad: u16,
    slot_idx: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hid_field_entry {
    list: list_head,
    field: *mut hid_field,
    index: u32,
    priority: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hid_report {
    list: list_head,
    hidinput_list: list_head,
    field_entry_list: list_head,
    id: u32,
    r#type: u32,
    application: u32,
    field: [*mut hid_field; 256],
    field_entries: *mut hid_field_entry,
    maxfield: u32,
    size: u32,
    device: *mut hid_device,
    tool_active: bool,
    tool: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hid_report_enum {
    numbered: u32,
    report_list: list_head,
    report_id_hash: [*mut hid_report; 256],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hid_input {
    list: list_head,
    report: *mut hid_report,
    input: *mut input_dev,
    name: *const u8,
    reports: list_head,
    application: u32,
    registered: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hid_device {
    dev_rdesc: *const u8,
    bpf_rdesc: *const u8,
    rdesc: *const u8,
    dev_rsize: u32,
    bpf_rsize: u32,
    rsize: u32,
    collection_size: u32,
    collection: *mut hid_collection,
    maxcollection: u32,
    maxapplication: u32,
    bus: u16,
    group: u16,
    vendor: u32,
    product: u32,
    version: u32,
    r#type: u32,
    country: u32,
    report_enum: [hid_report_enum; 3],
    led_work: work_struct,
    driver_input_lock: semaphore,
    dev: device,
    driver: *mut hid_driver,
    devres_group_id: *mut (),
    ll_driver: *const hid_ll_driver,
    ll_open_lock: mutex,
    ll_open_count: u32,
    status: u64,
    claimed: u32,
    quirks: u32,
    initial_quirks: u32,
    io_started: bool,
    inputs: list_head,
    hiddev: *mut (),
    hidraw: *mut (),
    name: [u8; 128],
    phys: [u8; 64],
    uniq: [u8; 64],
    firmware_version: u64,
    driver_data: *mut (),
    ff_init: Option<unsafe extern "C" fn(*mut hid_device) -> i32>,
    hiddev_connect: Option<unsafe extern "C" fn(*mut hid_device, u32) -> i32>,
    hiddev_disconnect: Option<unsafe extern "C" fn(*mut hid_device)>,
    hiddev_hid_event: Option<unsafe extern "C" fn(*mut hid_device, *mut hid_field, *mut hid_usage, i32)>,
    hiddev_report_event: Option<unsafe extern "C" fn(*mut hid_device, *mut hid_report)>,
    debug: u16,
    debug_dir: *mut dentry,
    debug_rdesc: *mut dentry,
    debug_events: *mut dentry,
    debug_list: list_head,
    debug_list_lock: spinlock,
    debug_wait: wait_queue_head,
    r#ref: kref,
    id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hid_report_id {
    report_type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hid_usage_id {
    usage_hid: u32,
    usage_type: u32,
    usage_code: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hid_driver {
    name: *const u8,
    id_table: *const hid_device_id,
    dyn_list: list_head,
    dyn_lock: spinlock,
    r#match: Option<unsafe extern "C" fn(*mut hid_device, bool) -> bool>,
    probe: Option<unsafe extern "C" fn(*mut hid_device, *const hid_device_id) -> i32>,
    remove: Option<unsafe extern "C" fn(*mut hid_device)>,
    report_table: *const hid_report_id,
    raw_event: Option<unsafe extern "C" fn(*mut hid_device, *mut hid_report, *mut u8, i32) -> i32>,
    usage_table: *const hid_usage_id,
    event: Option<unsafe extern "C" fn(*mut hid_device, *mut hid_field, *mut hid_usage, i32) -> i32>,
    report: Option<unsafe extern "C" fn(*mut hid_device, *mut hid_report)>,
    report_fixup: Option<unsafe extern "C" fn(*mut hid_device, *mut u8, *mut u32) -> *const u8>,
    input_mapping: Option<unsafe extern "C" fn(*mut hid_device, *mut hid_input, *mut hid_field, *mut hid_usage, *mut *mut u64, *mut i32) -> i32>,
    input_mapped: Option<unsafe extern "C" fn(*mut hid_device, *mut hid_input, *mut hid_field, *mut hid_usage, *mut *mut u64, *mut i32) -> i32>,
    input_configured: Option<unsafe extern "C" fn(*mut hid_device, *mut hid_input) -> i32>,
    feature_mapping: Option<unsafe extern "C" fn(*mut hid_device, *mut hid_field, *mut hid_usage)>,
    suspend: Option<unsafe extern "C" fn(*mut hid_device, pm_message) -> i32>,
    resume: Option<unsafe extern "C" fn(*mut hid_device) -> i32>,
    reset_resume: Option<unsafe extern "C" fn(*mut hid_device) -> i32>,
    on_hid_hw_open: Option<unsafe extern "C" fn(*mut hid_device)>,
    on_hid_hw_close: Option<unsafe extern "C" fn(*mut hid_device)>,
    driver: device_driver,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hid_ll_driver {
    start: Option<unsafe extern "C" fn(*mut hid_device) -> i32>,
    stop: Option<unsafe extern "C" fn(*mut hid_device)>,
    open: Option<unsafe extern "C" fn(*mut hid_device) -> i32>,
    close: Option<unsafe extern "C" fn(*mut hid_device)>,
    power: Option<unsafe extern "C" fn(*mut hid_device, i32) -> i32>,
    parse: Option<unsafe extern "C" fn(*mut hid_device) -> i32>,
    request: Option<unsafe extern "C" fn(*mut hid_device, *mut hid_report, i32)>,
    wait: Option<unsafe extern "C" fn(*mut hid_device) -> i32>,
    raw_request: Option<unsafe extern "C" fn(*mut hid_device, u8, *mut u8, u64, u8, i32) -> i32>,
    output_report: Option<unsafe extern "C" fn(*mut hid_device, *mut u8, u64) -> i32>,
    idle: Option<unsafe extern "C" fn(*mut hid_device, i32, i32, i32) -> i32>,
    may_wakeup: Option<unsafe extern "C" fn(*mut hid_device) -> bool>,
    max_buffer_size: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hid_debug_list {
    hid_debug_fifo: __SlateRecord_hid_debug_h_31,
    fasync: *mut fasync_struct,
    hdev: *mut hid_device,
    node: list_head,
    read_mutex: mutex,
}


// Should you need to contact me, the author, you can do so either by
// e-mail - mail your message to <vojtech@ucw.cz>, or by paper mail:
// Vojtech Pavlik, Simunkova 1594, Prague 8, 182 00 Czech Republic
static mut hid_debug_root: *mut dentry = unsafe { core::mem::zeroed() };

#[repr(C)]
#[derive(Clone, Copy)]
struct hid_usage_entry {
    page: u32,
    usage: u32,
    description: *const u8,
}


/// pages 0xff00 to 0xffff are vendor-specific
static hid_usage_table: __SlateAlign16<[hid_usage_entry; 2804]> = __SlateAlign16([hid_usage_entry { page: 0u32, usage: 0u32, description: (b"Undefined\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 0u32, description: (b"GenericDesktop\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 1u32, description: (b"Pointer\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 2u32, description: (b"Mouse\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 4u32, description: (b"Joystick\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 5u32, description: (b"Gamepad\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 6u32, description: (b"Keyboard\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 7u32, description: (b"Keypad\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 8u32, description: (b"MultiaxisController\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 9u32, description: (b"TabletPCSystemControls\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 10u32, description: (b"WaterCoolingDevice\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 11u32, description: (b"ComputerChassisDevice\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 12u32, description: (b"WirelessRadioControls\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 13u32, description: (b"PortableDeviceControl\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 14u32, description: (b"SystemMultiAxisController\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 15u32, description: (b"SpatialController\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 16u32, description: (b"AssistiveControl\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 17u32, description: (b"DeviceDock\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 18u32, description: (b"DockableDevice\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 19u32, description: (b"CallStateManagementControl\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 48u32, description: (b"X\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 49u32, description: (b"Y\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 50u32, description: (b"Z\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 51u32, description: (b"Rx\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 52u32, description: (b"Ry\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 53u32, description: (b"Rz\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 54u32, description: (b"Slider\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 55u32, description: (b"Dial\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 56u32, description: (b"Wheel\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 57u32, description: (b"HatSwitch\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 58u32, description: (b"CountedBuffer\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 59u32, description: (b"ByteCount\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 60u32, description: (b"MotionWakeup\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 61u32, description: (b"Start\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 62u32, description: (b"Select\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 64u32, description: (b"Vx\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 65u32, description: (b"Vy\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 66u32, description: (b"Vz\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 67u32, description: (b"Vbrx\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 68u32, description: (b"Vbry\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 69u32, description: (b"Vbrz\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 70u32, description: (b"Vno\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 71u32, description: (b"FeatureNotification\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 72u32, description: (b"ResolutionMultiplier\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 73u32, description: (b"Qx\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 74u32, description: (b"Qy\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 75u32, description: (b"Qz\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 76u32, description: (b"Qw\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 128u32, description: (b"SystemControl\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 129u32, description: (b"SystemPowerDown\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 130u32, description: (b"SystemSleep\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 131u32, description: (b"SystemWakeUp\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 132u32, description: (b"SystemContextMenu\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 133u32, description: (b"SystemMainMenu\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 134u32, description: (b"SystemAppMenu\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 135u32, description: (b"SystemMenuHelp\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 136u32, description: (b"SystemMenuExit\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 137u32, description: (b"SystemMenuSelect\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 138u32, description: (b"SystemMenuRight\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 139u32, description: (b"SystemMenuLeft\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 140u32, description: (b"SystemMenuUp\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 141u32, description: (b"SystemMenuDown\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 142u32, description: (b"SystemColdRestart\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 143u32, description: (b"SystemWarmRestart\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 144u32, description: (b"DpadUp\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 145u32, description: (b"DpadDown\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 146u32, description: (b"DpadRight\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 147u32, description: (b"DpadLeft\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 148u32, description: (b"IndexTrigger\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 149u32, description: (b"PalmTrigger\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 150u32, description: (b"Thumbstick\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 151u32, description: (b"SystemFunctionShift\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 152u32, description: (b"SystemFunctionShiftLock\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 153u32, description: (b"SystemFunctionShiftLockIndicator\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 154u32, description: (b"SystemDismissNotification\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 155u32, description: (b"SystemDoNotDisturb\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 160u32, description: (b"SystemDock\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 161u32, description: (b"SystemUndock\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 162u32, description: (b"SystemSetup\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 163u32, description: (b"SystemBreak\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 164u32, description: (b"SystemDebuggerBreak\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 165u32, description: (b"ApplicationBreak\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 166u32, description: (b"ApplicationDebuggerBreak\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 167u32, description: (b"SystemSpeakerMute\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 168u32, description: (b"SystemHibernate\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 169u32, description: (b"SystemMicrophoneMute\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 176u32, description: (b"SystemDisplayInvert\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 177u32, description: (b"SystemDisplayInternal\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 178u32, description: (b"SystemDisplayExternal\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 179u32, description: (b"SystemDisplayBoth\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 180u32, description: (b"SystemDisplayDual\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 181u32, description: (b"SystemDisplayToggleIntExtMode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 182u32, description: (b"SystemDisplaySwapPrimarySecondary\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 183u32, description: (b"SystemDisplayToggleLCDAutoscale\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 192u32, description: (b"SensorZone\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 193u32, description: (b"RPM\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 194u32, description: (b"CoolantLevel\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 195u32, description: (b"CoolantCriticalLevel\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 196u32, description: (b"CoolantPump\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 197u32, description: (b"ChassisEnclosure\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 198u32, description: (b"WirelessRadioButton\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 199u32, description: (b"WirelessRadioLED\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 200u32, description: (b"WirelessRadioSliderSwitch\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 201u32, description: (b"SystemDisplayRotationLockButton\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 202u32, description: (b"SystemDisplayRotationLockSliderSwitch\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 203u32, description: (b"ControlEnable\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 208u32, description: (b"DockableDeviceUniqueID\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 209u32, description: (b"DockableDeviceVendorID\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 210u32, description: (b"DockableDevicePrimaryUsagePage\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 211u32, description: (b"DockableDevicePrimaryUsageID\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 212u32, description: (b"DockableDeviceDockingState\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 213u32, description: (b"DockableDeviceDisplayOcclusion\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 214u32, description: (b"DockableDeviceObjectType\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 224u32, description: (b"CallActiveLED\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 225u32, description: (b"CallMuteToggle\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 1u32, usage: 226u32, description: (b"CallMuteLED\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 0u32, description: (b"SimulationControls\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 1u32, description: (b"FlightSimulationDevice\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 2u32, description: (b"AutomobileSimulationDevice\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 3u32, description: (b"TankSimulationDevice\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 4u32, description: (b"SpaceshipSimulationDevice\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 5u32, description: (b"SubmarineSimulationDevice\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 6u32, description: (b"SailingSimulationDevice\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 7u32, description: (b"MotorcycleSimulationDevice\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 8u32, description: (b"SportsSimulationDevice\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 9u32, description: (b"AirplaneSimulationDevice\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 10u32, description: (b"HelicopterSimulationDevice\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 11u32, description: (b"MagicCarpetSimulationDevice\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 12u32, description: (b"BicycleSimulationDevice\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 32u32, description: (b"FlightControlStick\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 33u32, description: (b"FlightStick\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 34u32, description: (b"CyclicControl\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 35u32, description: (b"CyclicTrim\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 36u32, description: (b"FlightYoke\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 37u32, description: (b"TrackControl\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 176u32, description: (b"Aileron\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 177u32, description: (b"AileronTrim\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 178u32, description: (b"AntiTorqueControl\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 179u32, description: (b"AutopilotEnable\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 180u32, description: (b"ChaffRelease\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 181u32, description: (b"CollectiveControl\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 182u32, description: (b"DiveBrake\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 183u32, description: (b"ElectronicCountermeasures\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 184u32, description: (b"Elevator\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 185u32, description: (b"ElevatorTrim\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 186u32, description: (b"Rudder\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 187u32, description: (b"Throttle\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 188u32, description: (b"FlightCommunications\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 189u32, description: (b"FlareRelease\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 190u32, description: (b"LandingGear\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 191u32, description: (b"ToeBrake\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 192u32, description: (b"Trigger\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 193u32, description: (b"WeaponsArm\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 194u32, description: (b"WeaponsSelect\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 195u32, description: (b"WingFlaps\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 196u32, description: (b"Accelerator\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 197u32, description: (b"Brake\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 198u32, description: (b"Clutch\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 199u32, description: (b"Shifter\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 200u32, description: (b"Steering\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 201u32, description: (b"TurretDirection\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 202u32, description: (b"BarrelElevation\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 203u32, description: (b"DivePlane\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 204u32, description: (b"Ballast\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 205u32, description: (b"BicycleCrank\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 206u32, description: (b"HandleBars\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 207u32, description: (b"FrontBrake\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 2u32, usage: 208u32, description: (b"RearBrake\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 3u32, usage: 0u32, description: (b"VRControls\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 3u32, usage: 1u32, description: (b"Belt\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 3u32, usage: 2u32, description: (b"BodySuit\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 3u32, usage: 3u32, description: (b"Flexor\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 3u32, usage: 4u32, description: (b"Glove\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 3u32, usage: 5u32, description: (b"HeadTracker\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 3u32, usage: 6u32, description: (b"HeadMountedDisplay\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 3u32, usage: 7u32, description: (b"HandTracker\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 3u32, usage: 8u32, description: (b"Oculometer\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 3u32, usage: 9u32, description: (b"Vest\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 3u32, usage: 10u32, description: (b"AnimatronicDevice\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 3u32, usage: 32u32, description: (b"StereoEnable\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 3u32, usage: 33u32, description: (b"DisplayEnable\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 4u32, usage: 0u32, description: (b"SportControls\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 4u32, usage: 1u32, description: (b"BaseballBat\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 4u32, usage: 2u32, description: (b"GolfClub\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 4u32, usage: 3u32, description: (b"RowingMachine\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 4u32, usage: 4u32, description: (b"Treadmill\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 4u32, usage: 48u32, description: (b"Oar\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 4u32, usage: 49u32, description: (b"Slope\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 4u32, usage: 50u32, description: (b"Rate\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 4u32, usage: 51u32, description: (b"StickSpeed\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 4u32, usage: 52u32, description: (b"StickFaceAngle\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 4u32, usage: 53u32, description: (b"StickHeelToe\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 4u32, usage: 54u32, description: (b"StickFollowThrough\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 4u32, usage: 55u32, description: (b"StickTempo\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 4u32, usage: 56u32, description: (b"StickType\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 4u32, usage: 57u32, description: (b"StickHeight\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 4u32, usage: 80u32, description: (b"Putter\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 4u32, usage: 81u32, description: (b"1Iron\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 4u32, usage: 82u32, description: (b"2Iron\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 4u32, usage: 83u32, description: (b"3Iron\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 4u32, usage: 84u32, description: (b"4Iron\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 4u32, usage: 85u32, description: (b"5Iron\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 4u32, usage: 86u32, description: (b"6Iron\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 4u32, usage: 87u32, description: (b"7Iron\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 4u32, usage: 88u32, description: (b"8Iron\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 4u32, usage: 89u32, description: (b"9Iron\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 4u32, usage: 90u32, description: (b"10Iron\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 4u32, usage: 91u32, description: (b"11Iron\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 4u32, usage: 92u32, description: (b"SandWedge\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 4u32, usage: 93u32, description: (b"LoftWedge\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 4u32, usage: 94u32, description: (b"PowerWedge\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 4u32, usage: 95u32, description: (b"1Wood\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 4u32, usage: 96u32, description: (b"3Wood\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 4u32, usage: 97u32, description: (b"5Wood\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 4u32, usage: 98u32, description: (b"7Wood\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 4u32, usage: 99u32, description: (b"9Wood\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 5u32, usage: 0u32, description: (b"GameControls\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 5u32, usage: 1u32, description: (b"3DGameController\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 5u32, usage: 2u32, description: (b"PinballDevice\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 5u32, usage: 3u32, description: (b"GunDevice\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 5u32, usage: 32u32, description: (b"PointofView\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 5u32, usage: 33u32, description: (b"TurnRightLeft\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 5u32, usage: 34u32, description: (b"PitchForwardBackward\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 5u32, usage: 35u32, description: (b"RollRightLeft\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 5u32, usage: 36u32, description: (b"MoveRightLeft\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 5u32, usage: 37u32, description: (b"MoveForwardBackward\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 5u32, usage: 38u32, description: (b"MoveUpDown\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 5u32, usage: 39u32, description: (b"LeanRightLeft\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 5u32, usage: 40u32, description: (b"LeanForwardBackward\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 5u32, usage: 41u32, description: (b"HeightofPOV\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 5u32, usage: 42u32, description: (b"Flipper\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 5u32, usage: 43u32, description: (b"SecondaryFlipper\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 5u32, usage: 44u32, description: (b"Bump\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 5u32, usage: 45u32, description: (b"NewGame\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 5u32, usage: 46u32, description: (b"ShootBall\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 5u32, usage: 47u32, description: (b"Player\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 5u32, usage: 48u32, description: (b"GunBolt\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 5u32, usage: 49u32, description: (b"GunClip\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 5u32, usage: 50u32, description: (b"GunSelector\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 5u32, usage: 51u32, description: (b"GunSingleShot\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 5u32, usage: 52u32, description: (b"GunBurst\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 5u32, usage: 53u32, description: (b"GunAutomatic\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 5u32, usage: 54u32, description: (b"GunSafety\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 5u32, usage: 55u32, description: (b"GamepadFireJump\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 5u32, usage: 57u32, description: (b"GamepadTrigger\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 5u32, usage: 58u32, description: (b"FormfittingGamepad\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 6u32, usage: 0u32, description: (b"GenericDeviceControls\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 6u32, usage: 1u32, description: (b"BackgroundNonuserControls\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 6u32, usage: 32u32, description: (b"BatteryStrength\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 6u32, usage: 33u32, description: (b"WirelessChannel\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 6u32, usage: 34u32, description: (b"WirelessID\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 6u32, usage: 35u32, description: (b"DiscoverWirelessControl\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 6u32, usage: 36u32, description: (b"SecurityCodeCharacterEntered\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 6u32, usage: 37u32, description: (b"SecurityCodeCharacterErased\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 6u32, usage: 38u32, description: (b"SecurityCodeCleared\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 6u32, usage: 39u32, description: (b"SequenceID\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 6u32, usage: 40u32, description: (b"SequenceIDReset\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 6u32, usage: 41u32, description: (b"RFSignalStrength\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 6u32, usage: 42u32, description: (b"SoftwareVersion\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 6u32, usage: 43u32, description: (b"ProtocolVersion\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 6u32, usage: 44u32, description: (b"HardwareVersion\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 6u32, usage: 45u32, description: (b"Major\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 6u32, usage: 46u32, description: (b"Minor\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 6u32, usage: 47u32, description: (b"Revision\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 6u32, usage: 48u32, description: (b"Handedness\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 6u32, usage: 49u32, description: (b"EitherHand\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 6u32, usage: 50u32, description: (b"LeftHand\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 6u32, usage: 51u32, description: (b"RightHand\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 6u32, usage: 52u32, description: (b"BothHands\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 6u32, usage: 64u32, description: (b"GripPoseOffset\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 6u32, usage: 65u32, description: (b"PointerPoseOffset\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 0u32, description: (b"KeyboardKeypad\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 1u32, description: (b"ErrorRollOver\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 2u32, description: (b"POSTFail\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 3u32, description: (b"ErrorUndefined\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 4u32, description: (b"KeyboardA\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 5u32, description: (b"KeyboardB\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 6u32, description: (b"KeyboardC\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 7u32, description: (b"KeyboardD\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 8u32, description: (b"KeyboardE\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 9u32, description: (b"KeyboardF\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 10u32, description: (b"KeyboardG\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 11u32, description: (b"KeyboardH\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 12u32, description: (b"KeyboardI\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 13u32, description: (b"KeyboardJ\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 14u32, description: (b"KeyboardK\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 15u32, description: (b"KeyboardL\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 16u32, description: (b"KeyboardM\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 17u32, description: (b"KeyboardN\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 18u32, description: (b"KeyboardO\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 19u32, description: (b"KeyboardP\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 20u32, description: (b"KeyboardQ\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 21u32, description: (b"KeyboardR\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 22u32, description: (b"KeyboardS\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 23u32, description: (b"KeyboardT\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 24u32, description: (b"KeyboardU\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 25u32, description: (b"KeyboardV\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 26u32, description: (b"KeyboardW\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 27u32, description: (b"KeyboardX\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 28u32, description: (b"KeyboardY\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 29u32, description: (b"KeyboardZ\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 30u32, description: (b"Keyboard1andBang\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 31u32, description: (b"Keyboard2andAt\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 32u32, description: (b"Keyboard3andHash\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 33u32, description: (b"Keyboard4andDollar\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 34u32, description: (b"Keyboard5andPercent\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 35u32, description: (b"Keyboard6andCaret\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 36u32, description: (b"Keyboard7andAmpersand\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 37u32, description: (b"Keyboard8andStar\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 38u32, description: (b"Keyboard9andLeftBracket\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 39u32, description: (b"Keyboard0andRightBracket\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 40u32, description: (b"KeyboardReturnEnter\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 41u32, description: (b"KeyboardEscape\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 42u32, description: (b"KeyboardDelete\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 43u32, description: (b"KeyboardTab\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 44u32, description: (b"KeyboardSpacebar\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 45u32, description: (b"KeyboardDashandUnderscore\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 46u32, description: (b"KeyboardEqualsandPlus\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 47u32, description: (b"KeyboardLeftBrace\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 48u32, description: (b"KeyboardRightBrace\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 49u32, description: (b"KeyboardBackslashandPipe\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 50u32, description: (b"KeyboardNonUSHashandTilde\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 51u32, description: (b"KeyboardSemiColonandColon\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 52u32, description: (b"KeyboardLeftAposandDouble\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 53u32, description: (b"KeyboardGraveAccentandTilde\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 54u32, description: (b"KeyboardCommaandLessThan\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 55u32, description: (b"KeyboardPeriodandGreaterThan\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 56u32, description: (b"KeyboardForwardSlashandQuestionMark\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 57u32, description: (b"KeyboardCapsLock\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 58u32, description: (b"KeyboardF1\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 59u32, description: (b"KeyboardF2\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 60u32, description: (b"KeyboardF3\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 61u32, description: (b"KeyboardF4\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 62u32, description: (b"KeyboardF5\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 63u32, description: (b"KeyboardF6\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 64u32, description: (b"KeyboardF7\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 65u32, description: (b"KeyboardF8\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 66u32, description: (b"KeyboardF9\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 67u32, description: (b"KeyboardF10\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 68u32, description: (b"KeyboardF11\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 69u32, description: (b"KeyboardF12\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 70u32, description: (b"KeyboardPrintScreen\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 71u32, description: (b"KeyboardScrollLock\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 72u32, description: (b"KeyboardPause\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 73u32, description: (b"KeyboardInsert\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 74u32, description: (b"KeyboardHome\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 75u32, description: (b"KeyboardPageUp\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 76u32, description: (b"KeyboardDeleteForward\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 77u32, description: (b"KeyboardEnd\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 78u32, description: (b"KeyboardPageDown\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 79u32, description: (b"KeyboardRightArrow\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 80u32, description: (b"KeyboardLeftArrow\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 81u32, description: (b"KeyboardDownArrow\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 82u32, description: (b"KeyboardUpArrow\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 83u32, description: (b"KeypadNumLockandClear\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 84u32, description: (b"KeypadForwardSlash\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 85u32, description: (b"KeypadStar\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 86u32, description: (b"KeypadDash\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 87u32, description: (b"KeypadPlus\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 88u32, description: (b"KeypadENTER\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 89u32, description: (b"Keypad1andEnd\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 90u32, description: (b"Keypad2andDownArrow\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 91u32, description: (b"Keypad3andPageDn\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 92u32, description: (b"Keypad4andLeftArrow\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 93u32, description: (b"Keypad5\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 94u32, description: (b"Keypad6andRightArrow\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 95u32, description: (b"Keypad7andHome\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 96u32, description: (b"Keypad8andUpArrow\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 97u32, description: (b"Keypad9andPageUp\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 98u32, description: (b"Keypad0andInsert\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 99u32, description: (b"KeypadPeriodandDelete\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 100u32, description: (b"KeyboardNonUSBackslashandPipe\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 101u32, description: (b"KeyboardApplication\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 102u32, description: (b"KeyboardPower\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 103u32, description: (b"KeypadEquals\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 104u32, description: (b"KeyboardF13\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 105u32, description: (b"KeyboardF14\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 106u32, description: (b"KeyboardF15\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 107u32, description: (b"KeyboardF16\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 108u32, description: (b"KeyboardF17\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 109u32, description: (b"KeyboardF18\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 110u32, description: (b"KeyboardF19\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 111u32, description: (b"KeyboardF20\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 112u32, description: (b"KeyboardF21\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 113u32, description: (b"KeyboardF22\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 114u32, description: (b"KeyboardF23\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 115u32, description: (b"KeyboardF24\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 116u32, description: (b"KeyboardExecute\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 117u32, description: (b"KeyboardHelp\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 118u32, description: (b"KeyboardMenu\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 119u32, description: (b"KeyboardSelect\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 120u32, description: (b"KeyboardStop\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 121u32, description: (b"KeyboardAgain\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 122u32, description: (b"KeyboardUndo\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 123u32, description: (b"KeyboardCut\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 124u32, description: (b"KeyboardCopy\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 125u32, description: (b"KeyboardPaste\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 126u32, description: (b"KeyboardFind\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 127u32, description: (b"KeyboardMute\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 128u32, description: (b"KeyboardVolumeUp\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 129u32, description: (b"KeyboardVolumeDown\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 130u32, description: (b"KeyboardLockingCapsLock\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 131u32, description: (b"KeyboardLockingNumLock\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 132u32, description: (b"KeyboardLockingScrollLock\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 133u32, description: (b"KeypadComma\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 134u32, description: (b"KeypadEqualSign\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 135u32, description: (b"KeyboardInternational1\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 136u32, description: (b"KeyboardInternational2\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 137u32, description: (b"KeyboardInternational3\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 138u32, description: (b"KeyboardInternational4\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 139u32, description: (b"KeyboardInternational5\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 140u32, description: (b"KeyboardInternational6\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 141u32, description: (b"KeyboardInternational7\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 142u32, description: (b"KeyboardInternational8\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 143u32, description: (b"KeyboardInternational9\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 144u32, description: (b"KeyboardLANG1\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 145u32, description: (b"KeyboardLANG2\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 146u32, description: (b"KeyboardLANG3\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 147u32, description: (b"KeyboardLANG4\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 148u32, description: (b"KeyboardLANG5\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 149u32, description: (b"KeyboardLANG6\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 150u32, description: (b"KeyboardLANG7\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 151u32, description: (b"KeyboardLANG8\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 152u32, description: (b"KeyboardLANG9\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 153u32, description: (b"KeyboardAlternateErase\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 154u32, description: (b"KeyboardSysReqAttention\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 155u32, description: (b"KeyboardCancel\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 156u32, description: (b"KeyboardClear\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 157u32, description: (b"KeyboardPrior\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 158u32, description: (b"KeyboardReturn\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 159u32, description: (b"KeyboardSeparator\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 160u32, description: (b"KeyboardOut\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 161u32, description: (b"KeyboardOper\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 162u32, description: (b"KeyboardClearAgain\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 163u32, description: (b"KeyboardCrSelProps\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 164u32, description: (b"KeyboardExSel\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 176u32, description: (b"KeypadDouble0\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 177u32, description: (b"KeypadTriple0\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 178u32, description: (b"ThousandsSeparator\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 179u32, description: (b"DecimalSeparator\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 180u32, description: (b"CurrencyUnit\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 181u32, description: (b"CurrencySubunit\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 182u32, description: (b"KeypadLeftBracket\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 183u32, description: (b"KeypadRightBracket\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 184u32, description: (b"KeypadLeftBrace\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 185u32, description: (b"KeypadRightBrace\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 186u32, description: (b"KeypadTab\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 187u32, description: (b"KeypadBackspace\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 188u32, description: (b"KeypadA\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 189u32, description: (b"KeypadB\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 190u32, description: (b"KeypadC\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 191u32, description: (b"KeypadD\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 192u32, description: (b"KeypadE\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 193u32, description: (b"KeypadF\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 194u32, description: (b"KeypadXOR\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 195u32, description: (b"KeypadCaret\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 196u32, description: (b"KeypadPercentage\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 197u32, description: (b"KeypadLess\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 198u32, description: (b"KeypadGreater\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 199u32, description: (b"KeypadAmpersand\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 200u32, description: (b"KeypadDoubleAmpersand\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 201u32, description: (b"KeypadBar\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 202u32, description: (b"KeypadDoubleBar\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 203u32, description: (b"KeypadColon\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 204u32, description: (b"KeypadHash\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 205u32, description: (b"KeypadSpace\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 206u32, description: (b"KeypadAt\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 207u32, description: (b"KeypadBang\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 208u32, description: (b"KeypadMemoryStore\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 209u32, description: (b"KeypadMemoryRecall\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 210u32, description: (b"KeypadMemoryClear\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 211u32, description: (b"KeypadMemoryAdd\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 212u32, description: (b"KeypadMemorySubtract\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 213u32, description: (b"KeypadMemoryMultiply\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 214u32, description: (b"KeypadMemoryDivide\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 215u32, description: (b"KeypadPlusMinus\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 216u32, description: (b"KeypadClear\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 217u32, description: (b"KeypadClearEntry\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 218u32, description: (b"KeypadBinary\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 219u32, description: (b"KeypadOctal\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 220u32, description: (b"KeypadDecimal\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 221u32, description: (b"KeypadHexadecimal\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 224u32, description: (b"KeyboardLeftControl\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 225u32, description: (b"KeyboardLeftShift\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 226u32, description: (b"KeyboardLeftAlt\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 227u32, description: (b"KeyboardLeftGUI\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 228u32, description: (b"KeyboardRightControl\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 229u32, description: (b"KeyboardRightShift\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 230u32, description: (b"KeyboardRightAlt\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 7u32, usage: 231u32, description: (b"KeyboardRightGUI\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 0u32, description: (b"LED\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 1u32, description: (b"NumLock\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 2u32, description: (b"CapsLock\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 3u32, description: (b"ScrollLock\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 4u32, description: (b"Compose\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 5u32, description: (b"Kana\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 6u32, description: (b"Power\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 7u32, description: (b"Shift\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 8u32, description: (b"DoNotDisturb\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 9u32, description: (b"Mute\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 10u32, description: (b"ToneEnable\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 11u32, description: (b"HighCutFilter\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 12u32, description: (b"LowCutFilter\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 13u32, description: (b"EqualizerEnable\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 14u32, description: (b"SoundFieldOn\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 15u32, description: (b"SurroundOn\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 16u32, description: (b"Repeat\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 17u32, description: (b"Stereo\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 18u32, description: (b"SamplingRateDetect\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 19u32, description: (b"Spinning\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 20u32, description: (b"CAV\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 21u32, description: (b"CLV\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 22u32, description: (b"RecordingFormatDetect\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 23u32, description: (b"OffHook\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 24u32, description: (b"Ring\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 25u32, description: (b"MessageWaiting\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 26u32, description: (b"DataMode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 27u32, description: (b"BatteryOperation\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 28u32, description: (b"BatteryOK\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 29u32, description: (b"BatteryLow\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 30u32, description: (b"Speaker\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 31u32, description: (b"Headset\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 32u32, description: (b"Hold\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 33u32, description: (b"Microphone\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 34u32, description: (b"Coverage\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 35u32, description: (b"NightMode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 36u32, description: (b"SendCalls\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 37u32, description: (b"CallPickup\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 38u32, description: (b"Conference\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 39u32, description: (b"Standby\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 40u32, description: (b"CameraOn\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 41u32, description: (b"CameraOff\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 42u32, description: (b"OnLine\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 43u32, description: (b"OffLine\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 44u32, description: (b"Busy\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 45u32, description: (b"Ready\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 46u32, description: (b"PaperOut\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 47u32, description: (b"PaperJam\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 48u32, description: (b"Remote\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 49u32, description: (b"Forward\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 50u32, description: (b"Reverse\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 51u32, description: (b"Stop\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 52u32, description: (b"Rewind\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 53u32, description: (b"FastForward\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 54u32, description: (b"Play\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 55u32, description: (b"Pause\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 56u32, description: (b"Record\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 57u32, description: (b"Error\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 58u32, description: (b"UsageSelectedIndicator\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 59u32, description: (b"UsageInUseIndicator\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 60u32, description: (b"UsageMultiModeIndicator\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 61u32, description: (b"IndicatorOn\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 62u32, description: (b"IndicatorFlash\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 63u32, description: (b"IndicatorSlowBlink\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 64u32, description: (b"IndicatorFastBlink\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 65u32, description: (b"IndicatorOff\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 66u32, description: (b"FlashOnTime\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 67u32, description: (b"SlowBlinkOnTime\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 68u32, description: (b"SlowBlinkOffTime\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 69u32, description: (b"FastBlinkOnTime\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 70u32, description: (b"FastBlinkOffTime\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 71u32, description: (b"UsageIndicatorColor\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 72u32, description: (b"IndicatorRed\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 73u32, description: (b"IndicatorGreen\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 74u32, description: (b"IndicatorAmber\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 75u32, description: (b"GenericIndicator\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 76u32, description: (b"SystemSuspend\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 77u32, description: (b"ExternalPowerConnected\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 78u32, description: (b"IndicatorBlue\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 79u32, description: (b"IndicatorOrange\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 80u32, description: (b"GoodStatus\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 81u32, description: (b"WarningStatus\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 82u32, description: (b"RGBLED\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 83u32, description: (b"RedLEDChannel\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 84u32, description: (b"BlueLEDChannel\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 85u32, description: (b"GreenLEDChannel\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 86u32, description: (b"LEDIntensity\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 87u32, description: (b"SystemMicrophoneMute\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 96u32, description: (b"PlayerIndicator\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 97u32, description: (b"Player1\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 98u32, description: (b"Player2\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 99u32, description: (b"Player3\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 100u32, description: (b"Player4\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 101u32, description: (b"Player5\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 102u32, description: (b"Player6\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 103u32, description: (b"Player7\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 8u32, usage: 104u32, description: (b"Player8\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 9u32, usage: 0u32, description: (b"Button\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 10u32, usage: 0u32, description: (b"Ordinal\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 0u32, description: (b"TelephonyDevice\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 1u32, description: (b"Phone\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 2u32, description: (b"AnsweringMachine\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 3u32, description: (b"MessageControls\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 4u32, description: (b"Handset\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 5u32, description: (b"Headset\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 6u32, description: (b"TelephonyKeyPad\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 7u32, description: (b"ProgrammableButton\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 32u32, description: (b"HookSwitch\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 33u32, description: (b"Flash\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 34u32, description: (b"Feature\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 35u32, description: (b"Hold\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 36u32, description: (b"Redial\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 37u32, description: (b"Transfer\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 38u32, description: (b"Drop\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 39u32, description: (b"Park\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 40u32, description: (b"ForwardCalls\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 41u32, description: (b"AlternateFunction\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 42u32, description: (b"Line\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 43u32, description: (b"SpeakerPhone\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 44u32, description: (b"Conference\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 45u32, description: (b"RingEnable\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 46u32, description: (b"RingSelect\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 47u32, description: (b"PhoneMute\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 48u32, description: (b"CallerID\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 49u32, description: (b"Send\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 80u32, description: (b"SpeedDial\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 81u32, description: (b"StoreNumber\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 82u32, description: (b"RecallNumber\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 83u32, description: (b"PhoneDirectory\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 112u32, description: (b"VoiceMail\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 113u32, description: (b"ScreenCalls\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 114u32, description: (b"DoNotDisturb\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 115u32, description: (b"Message\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 116u32, description: (b"AnswerOnOff\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 144u32, description: (b"InsideDialTone\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 145u32, description: (b"OutsideDialTone\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 146u32, description: (b"InsideRingTone\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 147u32, description: (b"OutsideRingTone\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 148u32, description: (b"PriorityRingTone\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 149u32, description: (b"InsideRingback\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 150u32, description: (b"PriorityRingback\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 151u32, description: (b"LineBusyTone\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 152u32, description: (b"ReorderTone\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 153u32, description: (b"CallWaitingTone\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 154u32, description: (b"ConfirmationTone1\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 155u32, description: (b"ConfirmationTone2\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 156u32, description: (b"TonesOff\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 157u32, description: (b"OutsideRingback\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 158u32, description: (b"Ringer\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 176u32, description: (b"PhoneKey0\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 177u32, description: (b"PhoneKey1\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 178u32, description: (b"PhoneKey2\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 179u32, description: (b"PhoneKey3\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 180u32, description: (b"PhoneKey4\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 181u32, description: (b"PhoneKey5\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 182u32, description: (b"PhoneKey6\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 183u32, description: (b"PhoneKey7\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 184u32, description: (b"PhoneKey8\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 185u32, description: (b"PhoneKey9\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 186u32, description: (b"PhoneKeyStar\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 187u32, description: (b"PhoneKeyPound\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 188u32, description: (b"PhoneKeyA\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 189u32, description: (b"PhoneKeyB\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 190u32, description: (b"PhoneKeyC\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 191u32, description: (b"PhoneKeyD\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 192u32, description: (b"PhoneCallHistoryKey\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 193u32, description: (b"PhoneCallerIDKey\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 194u32, description: (b"PhoneSettingsKey\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 240u32, description: (b"HostControl\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 241u32, description: (b"HostAvailable\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 242u32, description: (b"HostCallActive\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 243u32, description: (b"ActivateHandsetAudio\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 244u32, description: (b"RingType\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 245u32, description: (b"RedialablePhoneNumber\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 248u32, description: (b"StopRingTone\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 249u32, description: (b"PSTNRingTone\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 250u32, description: (b"HostRingTone\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 251u32, description: (b"AlertSoundError\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 252u32, description: (b"AlertSoundConfirm\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 253u32, description: (b"AlertSoundNotification\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 254u32, description: (b"SilentRing\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 264u32, description: (b"EmailMessageWaiting\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 265u32, description: (b"VoicemailMessageWaiting\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 266u32, description: (b"HostHold\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 272u32, description: (b"IncomingCallHistoryCount\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 273u32, description: (b"OutgoingCallHistoryCount\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 274u32, description: (b"IncomingCallHistory\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 275u32, description: (b"OutgoingCallHistory\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 276u32, description: (b"PhoneLocale\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 320u32, description: (b"PhoneTimeSecond\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 321u32, description: (b"PhoneTimeMinute\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 322u32, description: (b"PhoneTimeHour\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 323u32, description: (b"PhoneDateDay\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 324u32, description: (b"PhoneDateMonth\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 325u32, description: (b"PhoneDateYear\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 326u32, description: (b"HandsetNickname\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 327u32, description: (b"AddressBookID\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 330u32, description: (b"CallDuration\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 11u32, usage: 331u32, description: (b"DualModePhone\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 0u32, description: (b"Consumer\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 1u32, description: (b"ConsumerControl\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 2u32, description: (b"NumericKeyPad\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 3u32, description: (b"ProgrammableButtons\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 4u32, description: (b"Microphone\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 5u32, description: (b"Headphone\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 6u32, description: (b"GraphicEqualizer\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 32u32, description: (b"10\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 33u32, description: (b"100\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 34u32, description: (b"AMPM\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 48u32, description: (b"Power\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 49u32, description: (b"Reset\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 50u32, description: (b"Sleep\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 51u32, description: (b"SleepAfter\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 52u32, description: (b"SleepMode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 53u32, description: (b"Illumination\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 54u32, description: (b"FunctionButtons\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 64u32, description: (b"Menu\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 65u32, description: (b"MenuPick\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 66u32, description: (b"MenuUp\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 67u32, description: (b"MenuDown\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 68u32, description: (b"MenuLeft\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 69u32, description: (b"MenuRight\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 70u32, description: (b"MenuEscape\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 71u32, description: (b"MenuValueIncrease\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 72u32, description: (b"MenuValueDecrease\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 96u32, description: (b"DataOnScreen\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 97u32, description: (b"ClosedCaption\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 98u32, description: (b"ClosedCaptionSelect\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 99u32, description: (b"VCRTV\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 100u32, description: (b"BroadcastMode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 101u32, description: (b"Snapshot\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 102u32, description: (b"Still\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 103u32, description: (b"PictureinPictureToggle\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 104u32, description: (b"PictureinPictureSwap\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 105u32, description: (b"RedMenuButton\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 106u32, description: (b"GreenMenuButton\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 107u32, description: (b"BlueMenuButton\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 108u32, description: (b"YellowMenuButton\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 109u32, description: (b"Aspect\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 110u32, description: (b"3DModeSelect\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 111u32, description: (b"DisplayBrightnessIncrement\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 112u32, description: (b"DisplayBrightnessDecrement\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 113u32, description: (b"DisplayBrightness\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 114u32, description: (b"DisplayBacklightToggle\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 115u32, description: (b"DisplaySetBrightnesstoMinimum\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 116u32, description: (b"DisplaySetBrightnesstoMaximum\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 117u32, description: (b"DisplaySetAutoBrightness\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 118u32, description: (b"CameraAccessEnabled\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 119u32, description: (b"CameraAccessDisabled\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 120u32, description: (b"CameraAccessToggle\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 121u32, description: (b"KeyboardBrightnessIncrement\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 122u32, description: (b"KeyboardBrightnessDecrement\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 123u32, description: (b"KeyboardBacklightSetLevel\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 124u32, description: (b"KeyboardBacklightOOC\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 125u32, description: (b"KeyboardBacklightSetMinimum\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 126u32, description: (b"KeyboardBacklightSetMaximum\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 127u32, description: (b"KeyboardBacklightAuto\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 128u32, description: (b"Selection\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 129u32, description: (b"AssignSelection\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 130u32, description: (b"ModeStep\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 131u32, description: (b"RecallLast\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 132u32, description: (b"EnterChannel\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 133u32, description: (b"OrderMovie\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 134u32, description: (b"Channel\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 135u32, description: (b"MediaSelection\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 136u32, description: (b"MediaSelectComputer\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 137u32, description: (b"MediaSelectTV\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 138u32, description: (b"MediaSelectWWW\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 139u32, description: (b"MediaSelectDVD\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 140u32, description: (b"MediaSelectTelephone\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 141u32, description: (b"MediaSelectProgramGuide\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 142u32, description: (b"MediaSelectVideoPhone\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 143u32, description: (b"MediaSelectGames\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 144u32, description: (b"MediaSelectMessages\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 145u32, description: (b"MediaSelectCD\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 146u32, description: (b"MediaSelectVCR\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 147u32, description: (b"MediaSelectTuner\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 148u32, description: (b"Quit\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 149u32, description: (b"Help\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 150u32, description: (b"MediaSelectTape\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 151u32, description: (b"MediaSelectCable\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 152u32, description: (b"MediaSelectSatellite\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 153u32, description: (b"MediaSelectSecurity\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 154u32, description: (b"MediaSelectHome\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 155u32, description: (b"MediaSelectCall\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 156u32, description: (b"ChannelIncrement\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 157u32, description: (b"ChannelDecrement\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 158u32, description: (b"MediaSelectSAP\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 160u32, description: (b"VCRPlus\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 161u32, description: (b"Once\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 162u32, description: (b"Daily\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 163u32, description: (b"Weekly\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 164u32, description: (b"Monthly\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 176u32, description: (b"Play\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 177u32, description: (b"Pause\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 178u32, description: (b"Record\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 179u32, description: (b"FastForward\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 180u32, description: (b"Rewind\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 181u32, description: (b"ScanNextTrack\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 182u32, description: (b"ScanPreviousTrack\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 183u32, description: (b"Stop\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 184u32, description: (b"Eject\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 185u32, description: (b"RandomPlay\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 186u32, description: (b"SelectDisc\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 187u32, description: (b"EnterDisc\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 188u32, description: (b"Repeat\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 189u32, description: (b"Tracking\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 190u32, description: (b"TrackNormal\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 191u32, description: (b"SlowTracking\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 192u32, description: (b"FrameForward\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 193u32, description: (b"FrameBack\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 194u32, description: (b"Mark\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 195u32, description: (b"ClearMark\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 196u32, description: (b"RepeatFromMark\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 197u32, description: (b"ReturnToMark\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 198u32, description: (b"SearchMarkForward\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 199u32, description: (b"SearchMarkBackwards\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 200u32, description: (b"CounterReset\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 201u32, description: (b"ShowCounter\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 202u32, description: (b"TrackingIncrement\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 203u32, description: (b"TrackingDecrement\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 204u32, description: (b"StopEject\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 205u32, description: (b"PlayPause\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 206u32, description: (b"PlaySkip\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 207u32, description: (b"VoiceCommand\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 208u32, description: (b"InvokeCaptureInterface\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 209u32, description: (b"StartorStopGameRecording\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 210u32, description: (b"HistoricalGameCapture\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 211u32, description: (b"CaptureGameScreenshot\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 212u32, description: (b"ShoworHideRecordingIndicator\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 213u32, description: (b"StartorStopMicrophoneCapture\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 214u32, description: (b"StartorStopCameraCapture\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 215u32, description: (b"StartorStopGameBroadcast\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 216u32, description: (b"StartorStopVoiceDictationSession\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 217u32, description: (b"InvokeDismissEmojiPicker\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 224u32, description: (b"Volume\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 225u32, description: (b"Balance\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 226u32, description: (b"Mute\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 227u32, description: (b"Bass\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 228u32, description: (b"Treble\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 229u32, description: (b"BassBoost\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 230u32, description: (b"SurroundMode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 231u32, description: (b"Loudness\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 232u32, description: (b"MPX\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 233u32, description: (b"VolumeIncrement\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 234u32, description: (b"VolumeDecrement\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 240u32, description: (b"SpeedSelect\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 241u32, description: (b"PlaybackSpeed\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 242u32, description: (b"StandardPlay\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 243u32, description: (b"LongPlay\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 244u32, description: (b"ExtendedPlay\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 245u32, description: (b"Slow\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 256u32, description: (b"FanEnable\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 257u32, description: (b"FanSpeed\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 258u32, description: (b"LightEnable\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 259u32, description: (b"LightIlluminationLevel\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 260u32, description: (b"ClimateControlEnable\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 261u32, description: (b"RoomTemperature\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 262u32, description: (b"SecurityEnable\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 263u32, description: (b"FireAlarm\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 264u32, description: (b"PoliceAlarm\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 265u32, description: (b"Proximity\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 266u32, description: (b"Motion\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 267u32, description: (b"DuressAlarm\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 268u32, description: (b"HoldupAlarm\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 269u32, description: (b"MedicalAlarm\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 336u32, description: (b"BalanceRight\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 337u32, description: (b"BalanceLeft\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 338u32, description: (b"BassIncrement\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 339u32, description: (b"BassDecrement\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 340u32, description: (b"TrebleIncrement\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 341u32, description: (b"TrebleDecrement\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 352u32, description: (b"SpeakerSystem\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 353u32, description: (b"ChannelLeft\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 354u32, description: (b"ChannelRight\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 355u32, description: (b"ChannelCenter\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 356u32, description: (b"ChannelFront\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 357u32, description: (b"ChannelCenterFront\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 358u32, description: (b"ChannelSide\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 359u32, description: (b"ChannelSurround\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 360u32, description: (b"ChannelLowFrequencyEnhancement\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 361u32, description: (b"ChannelTop\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 362u32, description: (b"ChannelUnknown\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 368u32, description: (b"Subchannel\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 369u32, description: (b"SubchannelIncrement\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 370u32, description: (b"SubchannelDecrement\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 371u32, description: (b"AlternateAudioIncrement\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 372u32, description: (b"AlternateAudioDecrement\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 384u32, description: (b"ApplicationLaunchButtons\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 385u32, description: (b"ALLaunchButtonConfigurationTool\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 386u32, description: (b"ALProgrammableButtonConfiguration\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 387u32, description: (b"ALConsumerControlConfiguration\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 388u32, description: (b"ALWordProcessor\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 389u32, description: (b"ALTextEditor\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 390u32, description: (b"ALSpreadsheet\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 391u32, description: (b"ALGraphicsEditor\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 392u32, description: (b"ALPresentationApp\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 393u32, description: (b"ALDatabaseApp\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 394u32, description: (b"ALEmailReader\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 395u32, description: (b"ALNewsreader\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 396u32, description: (b"ALVoicemail\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 397u32, description: (b"ALContactsAddressBook\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 398u32, description: (b"ALCalendarSchedule\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 399u32, description: (b"ALTaskProjectManager\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 400u32, description: (b"ALLogJournalTimecard\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 401u32, description: (b"ALCheckbookFinance\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 402u32, description: (b"ALCalculator\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 403u32, description: (b"ALAVCapturePlayback\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 404u32, description: (b"ALLocalMachineBrowser\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 405u32, description: (b"ALLANWANBrowser\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 406u32, description: (b"ALInternetBrowser\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 407u32, description: (b"ALRemoteNetworkingISPConnect\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 408u32, description: (b"ALNetworkConference\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 409u32, description: (b"ALNetworkChat\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 410u32, description: (b"ALTelephonyDialer\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 411u32, description: (b"ALLogon\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 412u32, description: (b"ALLogoff\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 413u32, description: (b"ALLogonLogoff\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 414u32, description: (b"ALTerminalLockScreensaver\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 415u32, description: (b"ALControlPanel\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 416u32, description: (b"ALCommandLineProcessorRun\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 417u32, description: (b"ALProcessTaskManager\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 418u32, description: (b"ALSelectTaskApplication\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 419u32, description: (b"ALNextTaskApplication\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 420u32, description: (b"ALPreviousTaskApplication\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 421u32, description: (b"ALPreemptiveHaltTaskApplication\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 422u32, description: (b"ALIntegratedHelpCenter\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 423u32, description: (b"ALDocuments\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 424u32, description: (b"ALThesaurus\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 425u32, description: (b"ALDictionary\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 426u32, description: (b"ALDesktop\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 427u32, description: (b"ALSpellCheck\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 428u32, description: (b"ALGrammarCheck\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 429u32, description: (b"ALWirelessStatus\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 430u32, description: (b"ALKeyboardLayout\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 431u32, description: (b"ALVirusProtection\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 432u32, description: (b"ALEncryption\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 433u32, description: (b"ALScreenSaver\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 434u32, description: (b"ALAlarms\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 435u32, description: (b"ALClock\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 436u32, description: (b"ALFileBrowser\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 437u32, description: (b"ALPowerStatus\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 438u32, description: (b"ALImageBrowser\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 439u32, description: (b"ALAudioBrowser\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 440u32, description: (b"ALMovieBrowser\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 441u32, description: (b"ALDigitalRightsManager\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 442u32, description: (b"ALDigitalWallet\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 444u32, description: (b"ALInstantMessaging\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 445u32, description: (b"ALOEMFeaturesTipsTutorialBrowser\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 446u32, description: (b"ALOEMHelp\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 447u32, description: (b"ALOnlineCommunity\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 448u32, description: (b"ALEntertainmentContentBrowser\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 449u32, description: (b"ALOnlineShoppingBrowser\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 450u32, description: (b"ALSmartCardInformationHelp\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 451u32, description: (b"ALMarketMonitorFinanceBrowser\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 452u32, description: (b"ALCustomizedCorporateNewsBrowser\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 453u32, description: (b"ALOnlineActivityBrowser\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 454u32, description: (b"ALResearchSearchBrowser\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 455u32, description: (b"ALAudioPlayer\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 456u32, description: (b"ALMessageStatus\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 457u32, description: (b"ALContactSync\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 458u32, description: (b"ALNavigation\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 459u32, description: (b"ALContextawareDesktopAssistant\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 460u32, description: (b"ALActionOnSelection\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 461u32, description: (b"ALContextualInsertion\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 462u32, description: (b"ALContextualQuery\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 512u32, description: (b"GenericGUIApplicationControls\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 513u32, description: (b"ACNew\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 514u32, description: (b"ACOpen\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 515u32, description: (b"ACClose\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 516u32, description: (b"ACExit\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 517u32, description: (b"ACMaximize\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 518u32, description: (b"ACMinimize\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 519u32, description: (b"ACSave\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 520u32, description: (b"ACPrint\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 521u32, description: (b"ACProperties\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 538u32, description: (b"ACUndo\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 539u32, description: (b"ACCopy\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 540u32, description: (b"ACCut\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 541u32, description: (b"ACPaste\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 542u32, description: (b"ACSelectAll\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 543u32, description: (b"ACFind\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 544u32, description: (b"ACFindandReplace\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 545u32, description: (b"ACSearch\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 546u32, description: (b"ACGoTo\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 547u32, description: (b"ACHome\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 548u32, description: (b"ACBack\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 549u32, description: (b"ACForward\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 550u32, description: (b"ACStop\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 551u32, description: (b"ACRefresh\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 552u32, description: (b"ACPreviousLink\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 553u32, description: (b"ACNextLink\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 554u32, description: (b"ACBookmarks\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 555u32, description: (b"ACHistory\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 556u32, description: (b"ACSubscriptions\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 557u32, description: (b"ACZoomIn\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 558u32, description: (b"ACZoomOut\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 559u32, description: (b"ACZoom\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 560u32, description: (b"ACFullScreenView\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 561u32, description: (b"ACNormalView\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 562u32, description: (b"ACViewToggle\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 563u32, description: (b"ACScrollUp\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 564u32, description: (b"ACScrollDown\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 565u32, description: (b"ACScroll\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 566u32, description: (b"ACPanLeft\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 567u32, description: (b"ACPanRight\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 568u32, description: (b"ACPan\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 569u32, description: (b"ACNewWindow\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 570u32, description: (b"ACTileHorizontally\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 571u32, description: (b"ACTileVertically\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 572u32, description: (b"ACFormat\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 573u32, description: (b"ACEdit\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 574u32, description: (b"ACBold\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 575u32, description: (b"ACItalics\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 576u32, description: (b"ACUnderline\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 577u32, description: (b"ACStrikethrough\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 578u32, description: (b"ACSubscript\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 579u32, description: (b"ACSuperscript\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 580u32, description: (b"ACAllCaps\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 581u32, description: (b"ACRotate\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 582u32, description: (b"ACResize\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 583u32, description: (b"ACFlipHorizontal\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 584u32, description: (b"ACFlipVertical\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 585u32, description: (b"ACMirrorHorizontal\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 586u32, description: (b"ACMirrorVertical\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 587u32, description: (b"ACFontSelect\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 588u32, description: (b"ACFontColor\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 589u32, description: (b"ACFontSize\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 590u32, description: (b"ACJustifyLeft\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 591u32, description: (b"ACJustifyCenterH\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 592u32, description: (b"ACJustifyRight\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 593u32, description: (b"ACJustifyBlockH\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 594u32, description: (b"ACJustifyTop\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 595u32, description: (b"ACJustifyCenterV\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 596u32, description: (b"ACJustifyBottom\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 597u32, description: (b"ACJustifyBlockV\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 598u32, description: (b"ACIndentDecrease\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 599u32, description: (b"ACIndentIncrease\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 600u32, description: (b"ACNumberedList\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 601u32, description: (b"ACRestartNumbering\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 602u32, description: (b"ACBulletedList\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 603u32, description: (b"ACPromote\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 604u32, description: (b"ACDemote\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 605u32, description: (b"ACYes\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 606u32, description: (b"ACNo\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 607u32, description: (b"ACCancel\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 608u32, description: (b"ACCatalog\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 609u32, description: (b"ACBuyCheckout\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 610u32, description: (b"ACAddtoCart\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 611u32, description: (b"ACExpand\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 612u32, description: (b"ACExpandAll\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 613u32, description: (b"ACCollapse\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 614u32, description: (b"ACCollapseAll\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 615u32, description: (b"ACPrintPreview\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 616u32, description: (b"ACPasteSpecial\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 617u32, description: (b"ACInsertMode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 618u32, description: (b"ACDelete\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 619u32, description: (b"ACLock\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 620u32, description: (b"ACUnlock\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 621u32, description: (b"ACProtect\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 622u32, description: (b"ACUnprotect\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 623u32, description: (b"ACAttachComment\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 624u32, description: (b"ACDeleteComment\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 625u32, description: (b"ACViewComment\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 626u32, description: (b"ACSelectWord\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 627u32, description: (b"ACSelectSentence\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 628u32, description: (b"ACSelectParagraph\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 629u32, description: (b"ACSelectColumn\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 630u32, description: (b"ACSelectRow\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 631u32, description: (b"ACSelectTable\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 632u32, description: (b"ACSelectObject\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 633u32, description: (b"ACRedoRepeat\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 634u32, description: (b"ACSort\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 635u32, description: (b"ACSortAscending\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 636u32, description: (b"ACSortDescending\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 637u32, description: (b"ACFilter\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 638u32, description: (b"ACSetClock\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 639u32, description: (b"ACViewClock\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 640u32, description: (b"ACSelectTimeZone\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 641u32, description: (b"ACEditTimeZones\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 642u32, description: (b"ACSetAlarm\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 643u32, description: (b"ACClearAlarm\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 644u32, description: (b"ACSnoozeAlarm\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 645u32, description: (b"ACResetAlarm\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 646u32, description: (b"ACSynchronize\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 647u32, description: (b"ACSendReceive\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 648u32, description: (b"ACSendTo\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 649u32, description: (b"ACReply\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 650u32, description: (b"ACReplyAll\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 651u32, description: (b"ACForwardMsg\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 652u32, description: (b"ACSend\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 653u32, description: (b"ACAttachFile\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 654u32, description: (b"ACUpload\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 655u32, description: (b"ACDownloadSaveTargetAs\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 656u32, description: (b"ACSetBorders\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 657u32, description: (b"ACInsertRow\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 658u32, description: (b"ACInsertColumn\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 659u32, description: (b"ACInsertFile\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 660u32, description: (b"ACInsertPicture\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 661u32, description: (b"ACInsertObject\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 662u32, description: (b"ACInsertSymbol\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 663u32, description: (b"ACSaveandClose\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 664u32, description: (b"ACRename\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 665u32, description: (b"ACMerge\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 666u32, description: (b"ACSplit\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 667u32, description: (b"ACDisributeHorizontally\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 668u32, description: (b"ACDistributeVertically\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 669u32, description: (b"ACNextKeyboardLayoutSelect\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 670u32, description: (b"ACNavigationGuidance\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 671u32, description: (b"ACDesktopShowAllWindows\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 672u32, description: (b"ACSoftKeyLeft\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 673u32, description: (b"ACSoftKeyRight\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 674u32, description: (b"ACDesktopShowAllApplications\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 688u32, description: (b"ACIdleKeepAlive\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 704u32, description: (b"ExtendedKeyboardAttributesCollection\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 705u32, description: (b"KeyboardFormFactor\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 706u32, description: (b"KeyboardKeyType\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 707u32, description: (b"KeyboardPhysicalLayout\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 708u32, description: (b"VendorSpecificKeyboardPhysicalLayout\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 709u32, description: (b"KeyboardIETFLanguageTagIndex\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 710u32, description: (b"ImplementedKeyboardInputAssistControls\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 711u32, description: (b"KeyboardInputAssistPrevious\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 712u32, description: (b"KeyboardInputAssistNext\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 713u32, description: (b"KeyboardInputAssistPreviousGroup\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 714u32, description: (b"KeyboardInputAssistNextGroup\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 715u32, description: (b"KeyboardInputAssistAccept\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 716u32, description: (b"KeyboardInputAssistCancel\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 720u32, description: (b"PrivacyScreenToggle\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 721u32, description: (b"PrivacyScreenLevelDecrement\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 722u32, description: (b"PrivacyScreenLevelIncrement\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 723u32, description: (b"PrivacyScreenLevelMinimum\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 724u32, description: (b"PrivacyScreenLevelMaximum\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 1280u32, description: (b"ContactEdited\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 1281u32, description: (b"ContactAdded\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 1282u32, description: (b"ContactRecordActive\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 1283u32, description: (b"ContactIndex\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 1284u32, description: (b"ContactNickname\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 1285u32, description: (b"ContactFirstName\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 1286u32, description: (b"ContactLastName\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 1287u32, description: (b"ContactFullName\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 1288u32, description: (b"ContactPhoneNumberPersonal\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 1289u32, description: (b"ContactPhoneNumberBusiness\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 1290u32, description: (b"ContactPhoneNumberMobile\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 1291u32, description: (b"ContactPhoneNumberPager\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 1292u32, description: (b"ContactPhoneNumberFax\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 1293u32, description: (b"ContactPhoneNumberOther\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 1294u32, description: (b"ContactEmailPersonal\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 1295u32, description: (b"ContactEmailBusiness\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 1296u32, description: (b"ContactEmailOther\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 1297u32, description: (b"ContactEmailMain\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 1298u32, description: (b"ContactSpeedDialNumber\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 1299u32, description: (b"ContactStatusFlag\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 12u32, usage: 1300u32, description: (b"ContactMisc\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 0u32, description: (b"Digitizers\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 1u32, description: (b"Digitizer\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 2u32, description: (b"Pen\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 3u32, description: (b"LightPen\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 4u32, description: (b"TouchScreen\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 5u32, description: (b"TouchPad\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 6u32, description: (b"Whiteboard\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 7u32, description: (b"CoordinateMeasuringMachine\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 8u32, description: (b"3DDigitizer\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 9u32, description: (b"StereoPlotter\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 10u32, description: (b"ArticulatedArm\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 11u32, description: (b"Armature\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 12u32, description: (b"MultiplePointDigitizer\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 13u32, description: (b"FreeSpaceWand\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 14u32, description: (b"DeviceConfiguration\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 15u32, description: (b"CapacitiveHeatMapDigitizer\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 32u32, description: (b"Stylus\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 33u32, description: (b"Puck\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 34u32, description: (b"Finger\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 35u32, description: (b"Devicesettings\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 36u32, description: (b"CharacterGesture\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 48u32, description: (b"TipPressure\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 49u32, description: (b"BarrelPressure\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 50u32, description: (b"InRange\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 51u32, description: (b"Touch\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 52u32, description: (b"Untouch\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 53u32, description: (b"Tap\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 54u32, description: (b"Quality\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 55u32, description: (b"DataValid\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 56u32, description: (b"TransducerIndex\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 57u32, description: (b"TabletFunctionKeys\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 58u32, description: (b"ProgramChangeKeys\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 59u32, description: (b"BatteryStrength\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 60u32, description: (b"Invert\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 61u32, description: (b"XTilt\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 62u32, description: (b"YTilt\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 63u32, description: (b"Azimuth\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 64u32, description: (b"Altitude\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 65u32, description: (b"Twist\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 66u32, description: (b"TipSwitch\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 67u32, description: (b"SecondaryTipSwitch\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 68u32, description: (b"BarrelSwitch\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 69u32, description: (b"Eraser\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 70u32, description: (b"TabletPick\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 71u32, description: (b"TouchValid\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 72u32, description: (b"Width\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 73u32, description: (b"Height\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 81u32, description: (b"ContactIdentifier\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 82u32, description: (b"DeviceMode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 83u32, description: (b"DeviceIdentifier\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 84u32, description: (b"ContactCount\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 85u32, description: (b"ContactCountMaximum\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 86u32, description: (b"ScanTime\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 87u32, description: (b"SurfaceSwitch\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 88u32, description: (b"ButtonSwitch\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 89u32, description: (b"PadType\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 90u32, description: (b"SecondaryBarrelSwitch\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 91u32, description: (b"TransducerSerialNumber\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 92u32, description: (b"PreferredColor\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 93u32, description: (b"PreferredColorisLocked\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 94u32, description: (b"PreferredLineWidth\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 95u32, description: (b"PreferredLineWidthisLocked\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 96u32, description: (b"LatencyMode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 97u32, description: (b"GestureCharacterQuality\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 98u32, description: (b"CharacterGestureDataLength\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 99u32, description: (b"CharacterGestureData\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 100u32, description: (b"GestureCharacterEncoding\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 101u32, description: (b"UTF8CharacterGestureEncoding\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 102u32, description: (b"UTF16LittleEndianCharacterGestureEncoding\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 103u32, description: (b"UTF16BigEndianCharacterGestureEncoding\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 104u32, description: (b"UTF32LittleEndianCharacterGestureEncoding\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 105u32, description: (b"UTF32BigEndianCharacterGestureEncoding\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 106u32, description: (b"CapacitiveHeatMapProtocolVendorID\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 107u32, description: (b"CapacitiveHeatMapProtocolVersion\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 108u32, description: (b"CapacitiveHeatMapFrameData\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 109u32, description: (b"GestureCharacterEnable\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 110u32, description: (b"TransducerSerialNumberPart2\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 111u32, description: (b"NoPreferredColor\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 112u32, description: (b"PreferredLineStyle\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 113u32, description: (b"PreferredLineStyleisLocked\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 114u32, description: (b"Ink\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 115u32, description: (b"Pencil\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 116u32, description: (b"Highlighter\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 117u32, description: (b"ChiselMarker\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 118u32, description: (b"Brush\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 119u32, description: (b"NoPreference\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 128u32, description: (b"DigitizerDiagnostic\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 129u32, description: (b"DigitizerError\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 130u32, description: (b"ErrNormalStatus\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 131u32, description: (b"ErrTransducersExceeded\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 132u32, description: (b"ErrFullTransFeaturesUnavailable\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 133u32, description: (b"ErrChargeLow\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 144u32, description: (b"TransducerSoftwareInfo\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 145u32, description: (b"TransducerVendorId\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 146u32, description: (b"TransducerProductId\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 147u32, description: (b"DeviceSupportedProtocols\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 148u32, description: (b"TransducerSupportedProtocols\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 149u32, description: (b"NoProtocol\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 150u32, description: (b"WacomAESProtocol\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 151u32, description: (b"USIProtocol\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 152u32, description: (b"MicrosoftPenProtocol\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 160u32, description: (b"SupportedReportRates\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 161u32, description: (b"ReportRate\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 162u32, description: (b"TransducerConnected\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 163u32, description: (b"SwitchDisabled\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 164u32, description: (b"SwitchUnimplemented\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 165u32, description: (b"TransducerSwitches\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 166u32, description: (b"TransducerIndexSelector\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 13u32, usage: 176u32, description: (b"ButtonPressThreshold\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 14u32, usage: 0u32, description: (b"Haptics\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 14u32, usage: 1u32, description: (b"SimpleHapticController\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 14u32, usage: 16u32, description: (b"WaveformList\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 14u32, usage: 17u32, description: (b"DurationList\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 14u32, usage: 32u32, description: (b"AutoTrigger\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 14u32, usage: 33u32, description: (b"ManualTrigger\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 14u32, usage: 34u32, description: (b"AutoTriggerAssociatedControl\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 14u32, usage: 35u32, description: (b"Intensity\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 14u32, usage: 36u32, description: (b"RepeatCount\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 14u32, usage: 37u32, description: (b"RetriggerPeriod\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 14u32, usage: 38u32, description: (b"WaveformVendorPage\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 14u32, usage: 39u32, description: (b"WaveformVendorID\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 14u32, usage: 40u32, description: (b"WaveformCutoffTime\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 14u32, usage: 4097u32, description: (b"WaveformNone\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 14u32, usage: 4098u32, description: (b"WaveformStop\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 14u32, usage: 4099u32, description: (b"WaveformClick\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 14u32, usage: 4100u32, description: (b"WaveformBuzzContinuous\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 14u32, usage: 4101u32, description: (b"WaveformRumbleContinuous\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 14u32, usage: 4102u32, description: (b"WaveformPress\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 14u32, usage: 4103u32, description: (b"WaveformRelease\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 14u32, usage: 4104u32, description: (b"WaveformHover\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 14u32, usage: 4105u32, description: (b"WaveformSuccess\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 14u32, usage: 4106u32, description: (b"WaveformError\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 14u32, usage: 4107u32, description: (b"WaveformInkContinuous\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 14u32, usage: 4108u32, description: (b"WaveformPencilContinuous\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 14u32, usage: 4109u32, description: (b"WaveformMarkerContinuous\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 14u32, usage: 4110u32, description: (b"WaveformChiselMarkerContinuous\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 14u32, usage: 4111u32, description: (b"WaveformBrushContinuous\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 14u32, usage: 4112u32, description: (b"WaveformEraserContinuous\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 14u32, usage: 4113u32, description: (b"WaveformSparkleContinuous\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 0u32, description: (b"PhysicalInputDevice\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 1u32, description: (b"PhysicalInputDevice\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 32u32, description: (b"Normal\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 33u32, description: (b"SetEffectReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 34u32, description: (b"EffectParameterBlockIndex\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 35u32, description: (b"ParameterBlockOffset\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 36u32, description: (b"ROMFlag\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 37u32, description: (b"EffectType\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 38u32, description: (b"ETConstantForce\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 39u32, description: (b"ETRamp\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 40u32, description: (b"ETCustomForce\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 48u32, description: (b"ETSquare\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 49u32, description: (b"ETSine\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 50u32, description: (b"ETTriangle\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 51u32, description: (b"ETSawtoothUp\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 52u32, description: (b"ETSawtoothDown\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 64u32, description: (b"ETSpring\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 65u32, description: (b"ETDamper\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 66u32, description: (b"ETInertia\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 67u32, description: (b"ETFriction\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 80u32, description: (b"Duration\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 81u32, description: (b"SamplePeriod\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 82u32, description: (b"Gain\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 83u32, description: (b"TriggerButton\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 84u32, description: (b"TriggerRepeatInterval\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 85u32, description: (b"AxesEnable\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 86u32, description: (b"DirectionEnable\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 87u32, description: (b"Direction\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 88u32, description: (b"TypeSpecificBlockOffset\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 89u32, description: (b"BlockType\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 90u32, description: (b"SetEnvelopeReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 91u32, description: (b"AttackLevel\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 92u32, description: (b"AttackTime\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 93u32, description: (b"FadeLevel\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 94u32, description: (b"FadeTime\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 95u32, description: (b"SetConditionReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 96u32, description: (b"CenterPointOffset\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 97u32, description: (b"PositiveCoefficient\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 98u32, description: (b"NegativeCoefficient\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 99u32, description: (b"PositiveSaturation\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 100u32, description: (b"NegativeSaturation\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 101u32, description: (b"DeadBand\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 102u32, description: (b"DownloadForceSample\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 103u32, description: (b"IsochCustomForceEnable\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 104u32, description: (b"CustomForceDataReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 105u32, description: (b"CustomForceData\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 106u32, description: (b"CustomForceVendorDefinedData\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 107u32, description: (b"SetCustomForceReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 108u32, description: (b"CustomForceDataOffset\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 109u32, description: (b"SampleCount\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 110u32, description: (b"SetPeriodicReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 111u32, description: (b"Offset\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 112u32, description: (b"Magnitude\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 113u32, description: (b"Phase\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 114u32, description: (b"Period\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 115u32, description: (b"SetConstantForceReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 116u32, description: (b"SetRampForceReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 117u32, description: (b"RampStart\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 118u32, description: (b"RampEnd\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 119u32, description: (b"EffectOperationReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 120u32, description: (b"EffectOperation\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 121u32, description: (b"OpEffectStart\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 122u32, description: (b"OpEffectStartSolo\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 123u32, description: (b"OpEffectStop\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 124u32, description: (b"LoopCount\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 125u32, description: (b"DeviceGainReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 126u32, description: (b"DeviceGain\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 127u32, description: (b"ParameterBlockPoolsReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 128u32, description: (b"RAMPoolSize\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 129u32, description: (b"ROMPoolSize\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 130u32, description: (b"ROMEffectBlockCount\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 131u32, description: (b"SimultaneousEffectsMax\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 132u32, description: (b"PoolAlignment\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 133u32, description: (b"ParameterBlockMoveReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 134u32, description: (b"MoveSource\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 135u32, description: (b"MoveDestination\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 136u32, description: (b"MoveLength\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 137u32, description: (b"EffectParameterBlockLoadReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 139u32, description: (b"EffectParameterBlockLoadStatus\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 140u32, description: (b"BlockLoadSuccess\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 141u32, description: (b"BlockLoadFull\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 142u32, description: (b"BlockLoadError\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 143u32, description: (b"BlockHandle\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 144u32, description: (b"EffectParameterBlockFreeReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 145u32, description: (b"TypeSpecificBlockHandle\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 146u32, description: (b"PIDStateReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 148u32, description: (b"EffectPlaying\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 149u32, description: (b"PIDDeviceControlReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 150u32, description: (b"PIDDeviceControl\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 151u32, description: (b"DCEnableActuators\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 152u32, description: (b"DCDisableActuators\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 153u32, description: (b"DCStopAllEffects\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 154u32, description: (b"DCReset\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 155u32, description: (b"DCPause\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 156u32, description: (b"DCContinue\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 159u32, description: (b"DevicePaused\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 160u32, description: (b"ActuatorsEnabled\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 164u32, description: (b"SafetySwitch\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 165u32, description: (b"ActuatorOverrideSwitch\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 166u32, description: (b"ActuatorPower\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 167u32, description: (b"StartDelay\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 168u32, description: (b"ParameterBlockSize\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 169u32, description: (b"DeviceManagedPool\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 170u32, description: (b"SharedParameterBlocks\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 171u32, description: (b"CreateNewEffectParameterBlockReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 15u32, usage: 172u32, description: (b"RAMPoolAvailable\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 17u32, usage: 0u32, description: (b"SoC\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 17u32, usage: 1u32, description: (b"SocControl\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 17u32, usage: 2u32, description: (b"FirmwareTransfer\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 17u32, usage: 3u32, description: (b"FirmwareFileId\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 17u32, usage: 4u32, description: (b"FileOffsetInBytes\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 17u32, usage: 5u32, description: (b"FileTransferSizeMaxInBytes\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 17u32, usage: 6u32, description: (b"FilePayload\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 17u32, usage: 7u32, description: (b"FilePayloadSizeInBytes\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 17u32, usage: 8u32, description: (b"FilePayloadContainsLastBytes\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 17u32, usage: 9u32, description: (b"FileTransferStop\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 17u32, usage: 10u32, description: (b"FileTransferTillEnd\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 18u32, usage: 0u32, description: (b"EyeandHeadTrackers\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 18u32, usage: 1u32, description: (b"EyeTracker\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 18u32, usage: 2u32, description: (b"HeadTracker\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 18u32, usage: 16u32, description: (b"TrackingData\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 18u32, usage: 17u32, description: (b"Capabilities\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 18u32, usage: 18u32, description: (b"Configuration\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 18u32, usage: 19u32, description: (b"Status\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 18u32, usage: 20u32, description: (b"Control\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 18u32, usage: 32u32, description: (b"SensorTimestamp\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 18u32, usage: 33u32, description: (b"PositionX\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 18u32, usage: 34u32, description: (b"PositionY\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 18u32, usage: 35u32, description: (b"PositionZ\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 18u32, usage: 36u32, description: (b"GazePoint\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 18u32, usage: 37u32, description: (b"LeftEyePosition\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 18u32, usage: 38u32, description: (b"RightEyePosition\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 18u32, usage: 39u32, description: (b"HeadPosition\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 18u32, usage: 40u32, description: (b"HeadDirectionPoint\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 18u32, usage: 41u32, description: (b"RotationaboutXaxis\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 18u32, usage: 42u32, description: (b"RotationaboutYaxis\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 18u32, usage: 43u32, description: (b"RotationaboutZaxis\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 18u32, usage: 256u32, description: (b"TrackerQuality\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 18u32, usage: 257u32, description: (b"MinimumTrackingDistance\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 18u32, usage: 258u32, description: (b"OptimumTrackingDistance\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 18u32, usage: 259u32, description: (b"MaximumTrackingDistance\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 18u32, usage: 260u32, description: (b"MaximumScreenPlaneWidth\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 18u32, usage: 261u32, description: (b"MaximumScreenPlaneHeight\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 18u32, usage: 512u32, description: (b"DisplayManufacturerID\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 18u32, usage: 513u32, description: (b"DisplayProductID\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 18u32, usage: 514u32, description: (b"DisplaySerialNumber\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 18u32, usage: 515u32, description: (b"DisplayManufacturerDate\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 18u32, usage: 516u32, description: (b"CalibratedScreenWidth\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 18u32, usage: 517u32, description: (b"CalibratedScreenHeight\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 18u32, usage: 768u32, description: (b"SamplingFrequency\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 18u32, usage: 769u32, description: (b"ConfigurationStatus\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 18u32, usage: 1024u32, description: (b"DeviceModeRequest\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 0u32, description: (b"AuxiliaryDisplay\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 1u32, description: (b"AlphanumericDisplay\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 2u32, description: (b"AuxiliaryDisplay\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 32u32, description: (b"DisplayAttributesReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 33u32, description: (b"ASCIICharacterSet\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 34u32, description: (b"DataReadBack\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 35u32, description: (b"FontReadBack\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 36u32, description: (b"DisplayControlReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 37u32, description: (b"ClearDisplay\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 38u32, description: (b"DisplayEnable\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 39u32, description: (b"ScreenSaverDelay\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 40u32, description: (b"ScreenSaverEnable\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 41u32, description: (b"VerticalScroll\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 42u32, description: (b"HorizontalScroll\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 43u32, description: (b"CharacterReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 44u32, description: (b"DisplayData\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 45u32, description: (b"DisplayStatus\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 46u32, description: (b"StatNotReady\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 47u32, description: (b"StatReady\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 48u32, description: (b"ErrNotaloadablecharacter\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 49u32, description: (b"ErrFontdatacannotberead\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 50u32, description: (b"CursorPositionReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 51u32, description: (b"Row\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 52u32, description: (b"Column\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 53u32, description: (b"Rows\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 54u32, description: (b"Columns\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 55u32, description: (b"CursorPixelPositioning\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 56u32, description: (b"CursorMode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 57u32, description: (b"CursorEnable\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 58u32, description: (b"CursorBlink\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 59u32, description: (b"FontReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 60u32, description: (b"FontData\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 61u32, description: (b"CharacterWidth\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 62u32, description: (b"CharacterHeight\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 63u32, description: (b"CharacterSpacingHorizontal\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 64u32, description: (b"CharacterSpacingVertical\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 65u32, description: (b"UnicodeCharacterSet\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 66u32, description: (b"Font7Segment\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 67u32, description: (b"7SegmentDirectMap\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 68u32, description: (b"Font14Segment\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 69u32, description: (b"14SegmentDirectMap\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 70u32, description: (b"DisplayBrightness\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 71u32, description: (b"DisplayContrast\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 72u32, description: (b"CharacterAttribute\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 73u32, description: (b"AttributeReadback\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 74u32, description: (b"AttributeData\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 75u32, description: (b"CharAttrEnhance\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 76u32, description: (b"CharAttrUnderline\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 77u32, description: (b"CharAttrBlink\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 128u32, description: (b"BitmapSizeX\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 129u32, description: (b"BitmapSizeY\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 130u32, description: (b"MaxBlitSize\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 131u32, description: (b"BitDepthFormat\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 132u32, description: (b"DisplayOrientation\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 133u32, description: (b"PaletteReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 134u32, description: (b"PaletteDataSize\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 135u32, description: (b"PaletteDataOffset\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 136u32, description: (b"PaletteData\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 138u32, description: (b"BlitReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 139u32, description: (b"BlitRectangleX1\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 140u32, description: (b"BlitRectangleY1\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 141u32, description: (b"BlitRectangleX2\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 142u32, description: (b"BlitRectangleY2\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 143u32, description: (b"BlitData\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 144u32, description: (b"SoftButton\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 145u32, description: (b"SoftButtonID\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 146u32, description: (b"SoftButtonSide\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 147u32, description: (b"SoftButtonOffset1\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 148u32, description: (b"SoftButtonOffset2\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 149u32, description: (b"SoftButtonReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 194u32, description: (b"SoftKeys\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 204u32, description: (b"DisplayDataExtensions\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 207u32, description: (b"CharacterMapping\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 221u32, description: (b"UnicodeEquivalent\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 223u32, description: (b"CharacterPageMapping\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 20u32, usage: 255u32, description: (b"RequestReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 0u32, description: (b"Sensors\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1u32, description: (b"Sensor\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 16u32, description: (b"Biometric\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 17u32, description: (b"BiometricHumanPresence\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 18u32, description: (b"BiometricHumanProximity\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 19u32, description: (b"BiometricHumanTouch\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 20u32, description: (b"BiometricBloodPressure\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 21u32, description: (b"BiometricBodyTemperature\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 22u32, description: (b"BiometricHeartRate\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 23u32, description: (b"BiometricHeartRateVariability\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 24u32, description: (b"BiometricPeripheralOxygenSaturation\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 25u32, description: (b"BiometricRespiratoryRate\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 32u32, description: (b"Electrical\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 33u32, description: (b"ElectricalCapacitance\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 34u32, description: (b"ElectricalCurrent\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 35u32, description: (b"ElectricalPower\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 36u32, description: (b"ElectricalInductance\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 37u32, description: (b"ElectricalResistance\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 38u32, description: (b"ElectricalVoltage\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 39u32, description: (b"ElectricalPotentiometer\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 40u32, description: (b"ElectricalFrequency\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 41u32, description: (b"ElectricalPeriod\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 48u32, description: (b"Environmental\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 49u32, description: (b"EnvironmentalAtmosphericPressure\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 50u32, description: (b"EnvironmentalHumidity\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 51u32, description: (b"EnvironmentalTemperature\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 52u32, description: (b"EnvironmentalWindDirection\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 53u32, description: (b"EnvironmentalWindSpeed\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 54u32, description: (b"EnvironmentalAirQuality\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 55u32, description: (b"EnvironmentalHeatIndex\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 56u32, description: (b"EnvironmentalSurfaceTemperature\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 57u32, description: (b"EnvironmentalVolatileOrganicCompounds\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 58u32, description: (b"EnvironmentalObjectPresence\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 59u32, description: (b"EnvironmentalObjectProximity\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 64u32, description: (b"Light\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 65u32, description: (b"LightAmbientLight\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 66u32, description: (b"LightConsumerInfrared\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 67u32, description: (b"LightInfraredLight\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 68u32, description: (b"LightVisibleLight\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 69u32, description: (b"LightUltravioletLight\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 80u32, description: (b"Location\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 81u32, description: (b"LocationBroadcast\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 82u32, description: (b"LocationDeadReckoning\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 83u32, description: (b"LocationGPSGlobalPositioningSystem\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 84u32, description: (b"LocationLookup\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 85u32, description: (b"LocationOther\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 86u32, description: (b"LocationStatic\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 87u32, description: (b"LocationTriangulation\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 96u32, description: (b"Mechanical\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 97u32, description: (b"MechanicalBooleanSwitch\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 98u32, description: (b"MechanicalBooleanSwitchArray\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 99u32, description: (b"MechanicalMultivalueSwitch\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 100u32, description: (b"MechanicalForce\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 101u32, description: (b"MechanicalPressure\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 102u32, description: (b"MechanicalStrain\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 103u32, description: (b"MechanicalWeight\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 104u32, description: (b"MechanicalHapticVibrator\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 105u32, description: (b"MechanicalHallEffectSwitch\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 112u32, description: (b"Motion\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 113u32, description: (b"MotionAccelerometer1D\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 114u32, description: (b"MotionAccelerometer2D\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 115u32, description: (b"MotionAccelerometer3D\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 116u32, description: (b"MotionGyrometer1D\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 117u32, description: (b"MotionGyrometer2D\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 118u32, description: (b"MotionGyrometer3D\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 119u32, description: (b"MotionMotionDetector\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 120u32, description: (b"MotionSpeedometer\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 121u32, description: (b"MotionAccelerometer\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 122u32, description: (b"MotionGyrometer\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 123u32, description: (b"MotionGravityVector\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 124u32, description: (b"MotionLinearAccelerometer\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 128u32, description: (b"Orientation\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 129u32, description: (b"OrientationCompass1D\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 130u32, description: (b"OrientationCompass2D\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 131u32, description: (b"OrientationCompass3D\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 132u32, description: (b"OrientationInclinometer1D\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 133u32, description: (b"OrientationInclinometer2D\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 134u32, description: (b"OrientationInclinometer3D\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 135u32, description: (b"OrientationDistance1D\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 136u32, description: (b"OrientationDistance2D\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 137u32, description: (b"OrientationDistance3D\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 138u32, description: (b"OrientationDeviceOrientation\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 139u32, description: (b"OrientationCompass\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 140u32, description: (b"OrientationInclinometer\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 141u32, description: (b"OrientationDistance\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 142u32, description: (b"OrientationRelativeOrientation\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 143u32, description: (b"OrientationSimpleOrientation\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 144u32, description: (b"Scanner\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 145u32, description: (b"ScannerBarcode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 146u32, description: (b"ScannerRFID\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 147u32, description: (b"ScannerNFC\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 160u32, description: (b"Time\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 161u32, description: (b"TimeAlarmTimer\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 162u32, description: (b"TimeRealTimeClock\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 176u32, description: (b"PersonalActivity\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 177u32, description: (b"PersonalActivityActivityDetection\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 178u32, description: (b"PersonalActivityDevicePosition\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 179u32, description: (b"PersonalActivityFloorTracker\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 180u32, description: (b"PersonalActivityPedometer\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 181u32, description: (b"PersonalActivityStepDetection\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 192u32, description: (b"OrientationExtended\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 193u32, description: (b"OrientationExtendedGeomagneticOrientation\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 194u32, description: (b"OrientationExtendedMagnetometer\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 208u32, description: (b"Gesture\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 209u32, description: (b"GestureChassisFlipGesture\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 210u32, description: (b"GestureHingeFoldGesture\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 224u32, description: (b"Other\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 225u32, description: (b"OtherCustom\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 226u32, description: (b"OtherGeneric\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 227u32, description: (b"OtherGenericEnumerator\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 228u32, description: (b"OtherHingeAngle\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 240u32, description: (b"VendorReserved1\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 241u32, description: (b"VendorReserved2\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 242u32, description: (b"VendorReserved3\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 243u32, description: (b"VendorReserved4\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 244u32, description: (b"VendorReserved5\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 245u32, description: (b"VendorReserved6\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 246u32, description: (b"VendorReserved7\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 247u32, description: (b"VendorReserved8\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 248u32, description: (b"VendorReserved9\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 249u32, description: (b"VendorReserved10\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 250u32, description: (b"VendorReserved11\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 251u32, description: (b"VendorReserved12\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 252u32, description: (b"VendorReserved13\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 253u32, description: (b"VendorReserved14\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 254u32, description: (b"VendorReserved15\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 255u32, description: (b"VendorReserved16\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 512u32, description: (b"Event\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 513u32, description: (b"EventSensorState\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 514u32, description: (b"EventSensorEvent\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 768u32, description: (b"Property\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 769u32, description: (b"PropertyFriendlyName\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 770u32, description: (b"PropertyPersistentUniqueID\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 771u32, description: (b"PropertySensorStatus\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 772u32, description: (b"PropertyMinimumReportInterval\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 773u32, description: (b"PropertySensorManufacturer\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 774u32, description: (b"PropertySensorModel\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 775u32, description: (b"PropertySensorSerialNumber\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 776u32, description: (b"PropertySensorDescription\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 777u32, description: (b"PropertySensorConnectionType\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 778u32, description: (b"PropertySensorDevicePath\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 779u32, description: (b"PropertyHardwareRevision\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 780u32, description: (b"PropertyFirmwareVersion\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 781u32, description: (b"PropertyReleaseDate\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 782u32, description: (b"PropertyReportInterval\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 783u32, description: (b"PropertyChangeSensitivityAbsolute\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 784u32, description: (b"PropertyChangeSensitivityPercentofRange\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 785u32, description: (b"PropertyChangeSensitivityPercentRelative\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 786u32, description: (b"PropertyAccuracy\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 787u32, description: (b"PropertyResolution\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 788u32, description: (b"PropertyMaximum\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 789u32, description: (b"PropertyMinimum\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 790u32, description: (b"PropertyReportingState\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 791u32, description: (b"PropertySamplingRate\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 792u32, description: (b"PropertyResponseCurve\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 793u32, description: (b"PropertyPowerState\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 794u32, description: (b"PropertyMaximumFIFOEvents\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 795u32, description: (b"PropertyReportLatency\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 796u32, description: (b"PropertyFlushFIFOEvents\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 797u32, description: (b"PropertyMaximumPowerConsumption\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 798u32, description: (b"PropertyIsPrimary\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 799u32, description: (b"PropertyHumanPresenceDetectionType\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1024u32, description: (b"DataFieldLocation\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1026u32, description: (b"DataFieldAltitudeAntennaSeaLevel\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1027u32, description: (b"DataFieldDifferentialReferenceStationID\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1028u32, description: (b"DataFieldAltitudeEllipsoidError\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1029u32, description: (b"DataFieldAltitudeEllipsoid\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1030u32, description: (b"DataFieldAltitudeSeaLevelError\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1031u32, description: (b"DataFieldAltitudeSeaLevel\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1032u32, description: (b"DataFieldDifferentialGPSDataAge\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1033u32, description: (b"DataFieldErrorRadius\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1034u32, description: (b"DataFieldFixQuality\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1035u32, description: (b"DataFieldFixType\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1036u32, description: (b"DataFieldGeoidalSeparation\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1037u32, description: (b"DataFieldGPSOperationMode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1038u32, description: (b"DataFieldGPSSelectionMode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1039u32, description: (b"DataFieldGPSStatus\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1040u32, description: (b"DataFieldPositionDilutionofPrecision\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1041u32, description: (b"DataFieldHorizontalDilutionofPrecision\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1042u32, description: (b"DataFieldVerticalDilutionofPrecision\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1043u32, description: (b"DataFieldLatitude\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1044u32, description: (b"DataFieldLongitude\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1045u32, description: (b"DataFieldTrueHeading\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1046u32, description: (b"DataFieldMagneticHeading\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1047u32, description: (b"DataFieldMagneticVariation\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1048u32, description: (b"DataFieldSpeed\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1049u32, description: (b"DataFieldSatellitesinView\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1050u32, description: (b"DataFieldSatellitesinViewAzimuth\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1051u32, description: (b"DataFieldSatellitesinViewElevation\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1052u32, description: (b"DataFieldSatellitesinViewIDs\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1053u32, description: (b"DataFieldSatellitesinViewPRNs\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1054u32, description: (b"DataFieldSatellitesinViewSNRatios\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1055u32, description: (b"DataFieldSatellitesUsedCount\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1056u32, description: (b"DataFieldSatellitesUsedPRNs\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1057u32, description: (b"DataFieldNMEASentence\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1058u32, description: (b"DataFieldAddressLine1\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1059u32, description: (b"DataFieldAddressLine2\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1060u32, description: (b"DataFieldCity\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1061u32, description: (b"DataFieldStateorProvince\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1062u32, description: (b"DataFieldCountryorRegion\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1063u32, description: (b"DataFieldPostalCode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1066u32, description: (b"PropertyLocation\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1067u32, description: (b"PropertyLocationDesiredAccuracy\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1072u32, description: (b"DataFieldEnvironmental\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1073u32, description: (b"DataFieldAtmosphericPressure\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1075u32, description: (b"DataFieldRelativeHumidity\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1076u32, description: (b"DataFieldTemperature\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1077u32, description: (b"DataFieldWindDirection\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1078u32, description: (b"DataFieldWindSpeed\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1079u32, description: (b"DataFieldAirQualityIndex\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1080u32, description: (b"DataFieldEquivalentCO2\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1081u32, description: (b"DataFieldVolatileOrganicCompoundConcentration\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1082u32, description: (b"DataFieldObjectPresence\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1083u32, description: (b"DataFieldObjectProximityRange\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1084u32, description: (b"DataFieldObjectProximityOutofRange\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1088u32, description: (b"PropertyEnvironmental\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1089u32, description: (b"PropertyReferencePressure\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1104u32, description: (b"DataFieldMotion\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1105u32, description: (b"DataFieldMotionState\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1106u32, description: (b"DataFieldAcceleration\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1107u32, description: (b"DataFieldAccelerationAxisX\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1108u32, description: (b"DataFieldAccelerationAxisY\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1109u32, description: (b"DataFieldAccelerationAxisZ\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1110u32, description: (b"DataFieldAngularVelocity\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1111u32, description: (b"DataFieldAngularVelocityaboutXAxis\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1112u32, description: (b"DataFieldAngularVelocityaboutYAxis\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1113u32, description: (b"DataFieldAngularVelocityaboutZAxis\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1114u32, description: (b"DataFieldAngularPosition\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1115u32, description: (b"DataFieldAngularPositionaboutXAxis\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1116u32, description: (b"DataFieldAngularPositionaboutYAxis\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1117u32, description: (b"DataFieldAngularPositionaboutZAxis\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1118u32, description: (b"DataFieldMotionSpeed\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1119u32, description: (b"DataFieldMotionIntensity\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1136u32, description: (b"DataFieldOrientation\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1137u32, description: (b"DataFieldHeading\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1138u32, description: (b"DataFieldHeadingXAxis\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1139u32, description: (b"DataFieldHeadingYAxis\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1140u32, description: (b"DataFieldHeadingZAxis\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1141u32, description: (b"DataFieldHeadingCompensatedMagneticNorth\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1142u32, description: (b"DataFieldHeadingCompensatedTrueNorth\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1143u32, description: (b"DataFieldHeadingMagneticNorth\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1144u32, description: (b"DataFieldHeadingTrueNorth\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1145u32, description: (b"DataFieldDistance\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1146u32, description: (b"DataFieldDistanceXAxis\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1147u32, description: (b"DataFieldDistanceYAxis\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1148u32, description: (b"DataFieldDistanceZAxis\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1149u32, description: (b"DataFieldDistanceOutofRange\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1150u32, description: (b"DataFieldTilt\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1151u32, description: (b"DataFieldTiltXAxis\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1152u32, description: (b"DataFieldTiltYAxis\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1153u32, description: (b"DataFieldTiltZAxis\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1154u32, description: (b"DataFieldRotationMatrix\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1155u32, description: (b"DataFieldQuaternion\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1156u32, description: (b"DataFieldMagneticFlux\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1157u32, description: (b"DataFieldMagneticFluxXAxis\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1158u32, description: (b"DataFieldMagneticFluxYAxis\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1159u32, description: (b"DataFieldMagneticFluxZAxis\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1160u32, description: (b"DataFieldMagnetometerAccuracy\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1161u32, description: (b"DataFieldSimpleOrientationDirection\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1168u32, description: (b"DataFieldMechanical\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1169u32, description: (b"DataFieldBooleanSwitchState\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1170u32, description: (b"DataFieldBooleanSwitchArrayStates\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1171u32, description: (b"DataFieldMultivalueSwitchValue\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1172u32, description: (b"DataFieldForce\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1173u32, description: (b"DataFieldAbsolutePressure\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1174u32, description: (b"DataFieldGaugePressure\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1175u32, description: (b"DataFieldStrain\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1176u32, description: (b"DataFieldWeight\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1184u32, description: (b"PropertyMechanical\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1185u32, description: (b"PropertyVibrationState\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1186u32, description: (b"PropertyForwardVibrationSpeed\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1187u32, description: (b"PropertyBackwardVibrationSpeed\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1200u32, description: (b"DataFieldBiometric\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1201u32, description: (b"DataFieldHumanPresence\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1202u32, description: (b"DataFieldHumanProximityRange\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1203u32, description: (b"DataFieldHumanProximityOutofRange\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1204u32, description: (b"DataFieldHumanTouchState\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1205u32, description: (b"DataFieldBloodPressure\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1206u32, description: (b"DataFieldBloodPressureDiastolic\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1207u32, description: (b"DataFieldBloodPressureSystolic\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1208u32, description: (b"DataFieldHeartRate\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1209u32, description: (b"DataFieldRestingHeartRate\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1210u32, description: (b"DataFieldHeartbeatInterval\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1211u32, description: (b"DataFieldRespiratoryRate\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1212u32, description: (b"DataFieldSpO2\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1213u32, description: (b"DataFieldHumanAttentionDetected\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1214u32, description: (b"DataFieldHumanHeadAzimuth\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1215u32, description: (b"DataFieldHumanHeadAltitude\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1216u32, description: (b"DataFieldHumanHeadRoll\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1217u32, description: (b"DataFieldHumanHeadPitch\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1218u32, description: (b"DataFieldHumanHeadYaw\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1219u32, description: (b"DataFieldHumanCorrelationId\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1232u32, description: (b"DataFieldLight\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1233u32, description: (b"DataFieldIlluminance\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1234u32, description: (b"DataFieldColorTemperature\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1235u32, description: (b"DataFieldChromaticity\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1236u32, description: (b"DataFieldChromaticityX\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1237u32, description: (b"DataFieldChromaticityY\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1238u32, description: (b"DataFieldConsumerIRSentenceReceive\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1239u32, description: (b"DataFieldInfraredLight\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1240u32, description: (b"DataFieldRedLight\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1241u32, description: (b"DataFieldGreenLight\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1242u32, description: (b"DataFieldBlueLight\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1243u32, description: (b"DataFieldUltravioletALight\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1244u32, description: (b"DataFieldUltravioletBLight\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1245u32, description: (b"DataFieldUltravioletIndex\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1246u32, description: (b"DataFieldNearInfraredLight\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1247u32, description: (b"PropertyLight\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1248u32, description: (b"PropertyConsumerIRSentenceSend\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1250u32, description: (b"PropertyAutoBrightnessPreferred\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1251u32, description: (b"PropertyAutoColorPreferred\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1264u32, description: (b"DataFieldScanner\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1265u32, description: (b"DataFieldRFIDTag40Bit\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1266u32, description: (b"DataFieldNFCSentenceReceive\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1272u32, description: (b"PropertyScanner\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1273u32, description: (b"PropertyNFCSentenceSend\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1280u32, description: (b"DataFieldElectrical\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1281u32, description: (b"DataFieldCapacitance\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1282u32, description: (b"DataFieldCurrent\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1283u32, description: (b"DataFieldElectricalPower\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1284u32, description: (b"DataFieldInductance\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1285u32, description: (b"DataFieldResistance\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1286u32, description: (b"DataFieldVoltage\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1287u32, description: (b"DataFieldFrequency\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1288u32, description: (b"DataFieldPeriod\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1289u32, description: (b"DataFieldPercentofRange\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1312u32, description: (b"DataFieldTime\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1313u32, description: (b"DataFieldYear\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1314u32, description: (b"DataFieldMonth\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1315u32, description: (b"DataFieldDay\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1316u32, description: (b"DataFieldDayofWeek\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1317u32, description: (b"DataFieldHour\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1318u32, description: (b"DataFieldMinute\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1319u32, description: (b"DataFieldSecond\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1320u32, description: (b"DataFieldMillisecond\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1321u32, description: (b"DataFieldTimestamp\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1322u32, description: (b"DataFieldJulianDayofYear\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1323u32, description: (b"DataFieldTimeSinceSystemBoot\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1328u32, description: (b"PropertyTime\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1329u32, description: (b"PropertyTimeZoneOffsetfromUTC\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1330u32, description: (b"PropertyTimeZoneName\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1331u32, description: (b"PropertyDaylightSavingsTimeObserved\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1332u32, description: (b"PropertyTimeTrimAdjustment\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1333u32, description: (b"PropertyArmAlarm\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1344u32, description: (b"DataFieldCustom\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1345u32, description: (b"DataFieldCustomUsage\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1346u32, description: (b"DataFieldCustomBooleanArray\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1347u32, description: (b"DataFieldCustomValue\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1348u32, description: (b"DataFieldCustomValue1\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1349u32, description: (b"DataFieldCustomValue2\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1350u32, description: (b"DataFieldCustomValue3\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1351u32, description: (b"DataFieldCustomValue4\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1352u32, description: (b"DataFieldCustomValue5\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1353u32, description: (b"DataFieldCustomValue6\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1354u32, description: (b"DataFieldCustomValue7\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1355u32, description: (b"DataFieldCustomValue8\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1356u32, description: (b"DataFieldCustomValue9\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1357u32, description: (b"DataFieldCustomValue10\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1358u32, description: (b"DataFieldCustomValue11\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1359u32, description: (b"DataFieldCustomValue12\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1360u32, description: (b"DataFieldCustomValue13\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1361u32, description: (b"DataFieldCustomValue14\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1362u32, description: (b"DataFieldCustomValue15\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1363u32, description: (b"DataFieldCustomValue16\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1364u32, description: (b"DataFieldCustomValue17\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1365u32, description: (b"DataFieldCustomValue18\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1366u32, description: (b"DataFieldCustomValue19\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1367u32, description: (b"DataFieldCustomValue20\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1368u32, description: (b"DataFieldCustomValue21\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1369u32, description: (b"DataFieldCustomValue22\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1370u32, description: (b"DataFieldCustomValue23\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1371u32, description: (b"DataFieldCustomValue24\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1372u32, description: (b"DataFieldCustomValue25\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1373u32, description: (b"DataFieldCustomValue26\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1374u32, description: (b"DataFieldCustomValue27\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1375u32, description: (b"DataFieldCustomValue28\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1376u32, description: (b"DataFieldGeneric\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1377u32, description: (b"DataFieldGenericGUIDorPROPERTYKEY\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1378u32, description: (b"DataFieldGenericCategoryGUID\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1379u32, description: (b"DataFieldGenericTypeGUID\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1380u32, description: (b"DataFieldGenericEventPROPERTYKEY\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1381u32, description: (b"DataFieldGenericPropertyPROPERTYKEY\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1382u32, description: (b"DataFieldGenericDataFieldPROPERTYKEY\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1383u32, description: (b"DataFieldGenericEvent\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1384u32, description: (b"DataFieldGenericProperty\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1385u32, description: (b"DataFieldGenericDataField\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1386u32, description: (b"DataFieldEnumeratorTableRowIndex\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1387u32, description: (b"DataFieldEnumeratorTableRowCount\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1388u32, description: (b"DataFieldGenericGUIDorPROPERTYKEYkind\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1389u32, description: (b"DataFieldGenericGUID\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1390u32, description: (b"DataFieldGenericPROPERTYKEY\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1391u32, description: (b"DataFieldGenericTopLevelCollectionID\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1392u32, description: (b"DataFieldGenericReportID\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1393u32, description: (b"DataFieldGenericReportItemPositionIndex\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1394u32, description: (b"DataFieldGenericFirmwareVARTYPE\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1395u32, description: (b"DataFieldGenericUnitofMeasure\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1396u32, description: (b"DataFieldGenericUnitExponent\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1397u32, description: (b"DataFieldGenericReportSize\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1398u32, description: (b"DataFieldGenericReportCount\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1408u32, description: (b"PropertyGeneric\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1409u32, description: (b"PropertyEnumeratorTableRowIndex\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1410u32, description: (b"PropertyEnumeratorTableRowCount\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1424u32, description: (b"DataFieldPersonalActivity\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1425u32, description: (b"DataFieldActivityType\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1426u32, description: (b"DataFieldActivityState\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1427u32, description: (b"DataFieldDevicePosition\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1428u32, description: (b"DataFieldStepCount\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1429u32, description: (b"DataFieldStepCountReset\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1430u32, description: (b"DataFieldStepDuration\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1431u32, description: (b"DataFieldStepType\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1440u32, description: (b"PropertyMinimumActivityDetectionInterval\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1441u32, description: (b"PropertySupportedActivityTypes\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1442u32, description: (b"PropertySubscribedActivityTypes\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1443u32, description: (b"PropertySupportedStepTypes\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1444u32, description: (b"PropertySubscribedStepTypes\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1445u32, description: (b"PropertyFloorHeight\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1456u32, description: (b"DataFieldCustomTypeID\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1472u32, description: (b"PropertyCustom\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1473u32, description: (b"PropertyCustomValue1\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1474u32, description: (b"PropertyCustomValue2\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1475u32, description: (b"PropertyCustomValue3\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1476u32, description: (b"PropertyCustomValue4\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1477u32, description: (b"PropertyCustomValue5\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1478u32, description: (b"PropertyCustomValue6\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1479u32, description: (b"PropertyCustomValue7\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1480u32, description: (b"PropertyCustomValue8\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1481u32, description: (b"PropertyCustomValue9\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1482u32, description: (b"PropertyCustomValue10\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1483u32, description: (b"PropertyCustomValue11\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1484u32, description: (b"PropertyCustomValue12\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1485u32, description: (b"PropertyCustomValue13\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1486u32, description: (b"PropertyCustomValue14\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1487u32, description: (b"PropertyCustomValue15\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1488u32, description: (b"PropertyCustomValue16\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1504u32, description: (b"DataFieldHinge\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1505u32, description: (b"DataFieldHingeAngle\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1520u32, description: (b"DataFieldGestureSensor\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1521u32, description: (b"DataFieldGestureState\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1522u32, description: (b"DataFieldHingeFoldInitialAngle\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1523u32, description: (b"DataFieldHingeFoldFinalAngle\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1524u32, description: (b"DataFieldHingeFoldContributingPanel\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 1525u32, description: (b"DataFieldHingeFoldType\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2048u32, description: (b"SensorStateUndefined\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2049u32, description: (b"SensorStateReady\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2050u32, description: (b"SensorStateNotAvailable\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2051u32, description: (b"SensorStateNoData\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2052u32, description: (b"SensorStateInitializing\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2053u32, description: (b"SensorStateAccessDenied\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2054u32, description: (b"SensorStateError\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2064u32, description: (b"SensorEventUnknown\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2065u32, description: (b"SensorEventStateChanged\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2066u32, description: (b"SensorEventPropertyChanged\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2067u32, description: (b"SensorEventDataUpdated\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2068u32, description: (b"SensorEventPollResponse\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2069u32, description: (b"SensorEventChangeSensitivity\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2070u32, description: (b"SensorEventRangeMaximumReached\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2071u32, description: (b"SensorEventRangeMinimumReached\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2072u32, description: (b"SensorEventHighThresholdCrossUpward\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2073u32, description: (b"SensorEventHighThresholdCrossDownward\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2074u32, description: (b"SensorEventLowThresholdCrossUpward\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2075u32, description: (b"SensorEventLowThresholdCrossDownward\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2076u32, description: (b"SensorEventZeroThresholdCrossUpward\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2077u32, description: (b"SensorEventZeroThresholdCrossDownward\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2078u32, description: (b"SensorEventPeriodExceeded\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2079u32, description: (b"SensorEventFrequencyExceeded\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2080u32, description: (b"SensorEventComplexTrigger\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2096u32, description: (b"ConnectionTypePCIntegrated\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2097u32, description: (b"ConnectionTypePCAttached\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2098u32, description: (b"ConnectionTypePCExternal\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2112u32, description: (b"ReportingStateReportNoEvents\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2113u32, description: (b"ReportingStateReportAllEvents\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2114u32, description: (b"ReportingStateReportThresholdEvents\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2115u32, description: (b"ReportingStateWakeOnNoEvents\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2116u32, description: (b"ReportingStateWakeOnAllEvents\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2117u32, description: (b"ReportingStateWakeOnThresholdEvents\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2118u32, description: (b"ReportingStateAnytime\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2128u32, description: (b"PowerStateUndefined\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2129u32, description: (b"PowerStateD0FullPower\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2130u32, description: (b"PowerStateD1LowPower\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2131u32, description: (b"PowerStateD2StandbyPowerwithWakeup\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2132u32, description: (b"PowerStateD3SleepwithWakeup\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2133u32, description: (b"PowerStateD4PowerOff\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2144u32, description: (b"AccuracyDefault\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2145u32, description: (b"AccuracyHigh\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2146u32, description: (b"AccuracyMedium\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2147u32, description: (b"AccuracyLow\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2160u32, description: (b"FixQualityNoFix\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2161u32, description: (b"FixQualityGPS\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2162u32, description: (b"FixQualityDGPS\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2176u32, description: (b"FixTypeNoFix\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2177u32, description: (b"FixTypeGPSSPSModeFixValid\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2178u32, description: (b"FixTypeDGPSSPSModeFixValid\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2179u32, description: (b"FixTypeGPSPPSModeFixValid\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2180u32, description: (b"FixTypeRealTimeKinematic\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2181u32, description: (b"FixTypeFloatRTK\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2182u32, description: (b"FixTypeEstimateddeadreckoned\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2183u32, description: (b"FixTypeManualInputMode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2184u32, description: (b"FixTypeSimulatorMode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2192u32, description: (b"GPSOperationModeManual\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2193u32, description: (b"GPSOperationModeAutomatic\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2208u32, description: (b"GPSSelectionModeAutonomous\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2209u32, description: (b"GPSSelectionModeDGPS\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2210u32, description: (b"GPSSelectionModeEstimateddeadreckoned\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2211u32, description: (b"GPSSelectionModeManualInput\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2212u32, description: (b"GPSSelectionModeSimulator\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2213u32, description: (b"GPSSelectionModeDataNotValid\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2224u32, description: (b"GPSStatusDataValid\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2225u32, description: (b"GPSStatusDataNotValid\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2240u32, description: (b"DayofWeekSunday\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2241u32, description: (b"DayofWeekMonday\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2242u32, description: (b"DayofWeekTuesday\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2243u32, description: (b"DayofWeekWednesday\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2244u32, description: (b"DayofWeekThursday\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2245u32, description: (b"DayofWeekFriday\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2246u32, description: (b"DayofWeekSaturday\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2256u32, description: (b"KindCategory\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2257u32, description: (b"KindType\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2258u32, description: (b"KindEvent\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2259u32, description: (b"KindProperty\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2260u32, description: (b"KindDataField\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2272u32, description: (b"MagnetometerAccuracyLow\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2273u32, description: (b"MagnetometerAccuracyMedium\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2274u32, description: (b"MagnetometerAccuracyHigh\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2288u32, description: (b"SimpleOrientationDirectionNotRotated\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2289u32, description: (b"SimpleOrientationDirectionRotated90DegreesCCW\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2290u32, description: (b"SimpleOrientationDirectionRotated180DegreesCCW\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2291u32, description: (b"SimpleOrientationDirectionRotated270DegreesCCW\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2292u32, description: (b"SimpleOrientationDirectionFaceUp\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2293u32, description: (b"SimpleOrientationDirectionFaceDown\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2304u32, description: (b"VT_NULL\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2305u32, description: (b"VT_BOOL\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2306u32, description: (b"VT_UI1\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2307u32, description: (b"VT_I1\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2308u32, description: (b"VT_UI2\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2309u32, description: (b"VT_I2\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2310u32, description: (b"VT_UI4\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2311u32, description: (b"VT_I4\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2312u32, description: (b"VT_UI8\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2313u32, description: (b"VT_I8\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2314u32, description: (b"VT_R4\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2315u32, description: (b"VT_R8\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2316u32, description: (b"VT_WSTR\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2317u32, description: (b"VT_STR\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2318u32, description: (b"VT_CLSID\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2319u32, description: (b"VT_VECTORVT_UI1\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2320u32, description: (b"VT_F16E0\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2321u32, description: (b"VT_F16E1\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2322u32, description: (b"VT_F16E2\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2323u32, description: (b"VT_F16E3\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2324u32, description: (b"VT_F16E4\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2325u32, description: (b"VT_F16E5\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2326u32, description: (b"VT_F16E6\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2327u32, description: (b"VT_F16E7\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2328u32, description: (b"VT_F16E8\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2329u32, description: (b"VT_F16E9\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2330u32, description: (b"VT_F16EA\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2331u32, description: (b"VT_F16EB\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2332u32, description: (b"VT_F16EC\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2333u32, description: (b"VT_F16ED\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2334u32, description: (b"VT_F16EE\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2335u32, description: (b"VT_F16EF\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2336u32, description: (b"VT_F32E0\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2337u32, description: (b"VT_F32E1\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2338u32, description: (b"VT_F32E2\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2339u32, description: (b"VT_F32E3\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2340u32, description: (b"VT_F32E4\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2341u32, description: (b"VT_F32E5\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2342u32, description: (b"VT_F32E6\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2343u32, description: (b"VT_F32E7\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2344u32, description: (b"VT_F32E8\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2345u32, description: (b"VT_F32E9\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2346u32, description: (b"VT_F32EA\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2347u32, description: (b"VT_F32EB\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2348u32, description: (b"VT_F32EC\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2349u32, description: (b"VT_F32ED\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2350u32, description: (b"VT_F32EE\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2351u32, description: (b"VT_F32EF\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2352u32, description: (b"ActivityTypeUnknown\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2353u32, description: (b"ActivityTypeStationary\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2354u32, description: (b"ActivityTypeFidgeting\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2355u32, description: (b"ActivityTypeWalking\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2356u32, description: (b"ActivityTypeRunning\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2357u32, description: (b"ActivityTypeInVehicle\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2358u32, description: (b"ActivityTypeBiking\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2359u32, description: (b"ActivityTypeIdle\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2368u32, description: (b"UnitNotSpecified\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2369u32, description: (b"UnitLux\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2370u32, description: (b"UnitDegreesKelvin\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2371u32, description: (b"UnitDegreesCelsius\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2372u32, description: (b"UnitPascal\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2373u32, description: (b"UnitNewton\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2374u32, description: (b"UnitMetersSecond\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2375u32, description: (b"UnitKilogram\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2376u32, description: (b"UnitMeter\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2377u32, description: (b"UnitMetersSecondSecond\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2378u32, description: (b"UnitFarad\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2379u32, description: (b"UnitAmpere\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2380u32, description: (b"UnitWatt\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2381u32, description: (b"UnitHenry\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2382u32, description: (b"UnitOhm\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2383u32, description: (b"UnitVolt\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2384u32, description: (b"UnitHertz\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2385u32, description: (b"UnitBar\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2386u32, description: (b"UnitDegreesAnticlockwise\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2387u32, description: (b"UnitDegreesClockwise\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2388u32, description: (b"UnitDegrees\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2389u32, description: (b"UnitDegreesSecond\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2390u32, description: (b"UnitDegreesSecondSecond\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2391u32, description: (b"UnitKnot\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2392u32, description: (b"UnitPercent\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2393u32, description: (b"UnitSecond\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2394u32, description: (b"UnitMillisecond\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2395u32, description: (b"UnitG\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2396u32, description: (b"UnitBytes\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2397u32, description: (b"UnitMilligauss\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2398u32, description: (b"UnitBits\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2400u32, description: (b"ActivityStateNoStateChange\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2401u32, description: (b"ActivityStateStartActivity\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2402u32, description: (b"ActivityStateEndActivity\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2416u32, description: (b"Exponent0\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2417u32, description: (b"Exponent1\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2418u32, description: (b"Exponent2\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2419u32, description: (b"Exponent3\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2420u32, description: (b"Exponent4\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2421u32, description: (b"Exponent5\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2422u32, description: (b"Exponent6\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2423u32, description: (b"Exponent7\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2424u32, description: (b"Exponent8\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2425u32, description: (b"Exponent9\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2426u32, description: (b"ExponentA\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2427u32, description: (b"ExponentB\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2428u32, description: (b"ExponentC\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2429u32, description: (b"ExponentD\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2430u32, description: (b"ExponentE\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2431u32, description: (b"ExponentF\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2432u32, description: (b"DevicePositionUnknown\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2433u32, description: (b"DevicePositionUnchanged\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2434u32, description: (b"DevicePositionOnDesk\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2435u32, description: (b"DevicePositionInHand\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2436u32, description: (b"DevicePositionMovinginBag\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2437u32, description: (b"DevicePositionStationaryinBag\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2448u32, description: (b"StepTypeUnknown\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2449u32, description: (b"StepTypeWalking\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2450u32, description: (b"StepTypeRunning\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2464u32, description: (b"GestureStateUnknown\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2465u32, description: (b"GestureStateStarted\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2466u32, description: (b"GestureStateCompleted\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2467u32, description: (b"GestureStateCancelled\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2480u32, description: (b"HingeFoldContributingPanelUnknown\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2481u32, description: (b"HingeFoldContributingPanelPanel1\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2482u32, description: (b"HingeFoldContributingPanelPanel2\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2483u32, description: (b"HingeFoldContributingPanelBoth\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2484u32, description: (b"HingeFoldTypeUnknown\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2485u32, description: (b"HingeFoldTypeIncreasing\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2486u32, description: (b"HingeFoldTypeDecreasing\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2496u32, description: (b"HumanPresenceDetectionTypeVendorDefinedNonBiometric\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2497u32, description: (b"HumanPresenceDetectionTypeVendorDefinedBiometric\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2498u32, description: (b"HumanPresenceDetectionTypeFacialBiometric\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 2499u32, description: (b"HumanPresenceDetectionTypeAudioBiometric\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 4096u32, description: (b"ModifierChangeSensitivityAbsolute\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 8192u32, description: (b"ModifierMaximum\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 12288u32, description: (b"ModifierMinimum\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 16384u32, description: (b"ModifierAccuracy\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 20480u32, description: (b"ModifierResolution\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 24576u32, description: (b"ModifierThresholdHigh\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 28672u32, description: (b"ModifierThresholdLow\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 32768u32, description: (b"ModifierCalibrationOffset\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 36864u32, description: (b"ModifierCalibrationMultiplier\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 40960u32, description: (b"ModifierReportInterval\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 45056u32, description: (b"ModifierFrequencyMax\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 49152u32, description: (b"ModifierPeriodMax\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 53248u32, description: (b"ModifierChangeSensitivityPercentofRange\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 57344u32, description: (b"ModifierChangeSensitivityPercentRelative\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 32u32, usage: 61440u32, description: (b"ModifierVendorReserved\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 64u32, usage: 0u32, description: (b"MedicalInstrument\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 64u32, usage: 1u32, description: (b"MedicalUltrasound\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 64u32, usage: 32u32, description: (b"VCRAcquisition\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 64u32, usage: 33u32, description: (b"FreezeThaw\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 64u32, usage: 34u32, description: (b"ClipStore\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 64u32, usage: 35u32, description: (b"Update\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 64u32, usage: 36u32, description: (b"Next\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 64u32, usage: 37u32, description: (b"Save\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 64u32, usage: 38u32, description: (b"Print\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 64u32, usage: 39u32, description: (b"MicrophoneEnable\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 64u32, usage: 64u32, description: (b"Cine\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 64u32, usage: 65u32, description: (b"TransmitPower\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 64u32, usage: 66u32, description: (b"Volume\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 64u32, usage: 67u32, description: (b"Focus\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 64u32, usage: 68u32, description: (b"Depth\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 64u32, usage: 96u32, description: (b"SoftStepPrimary\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 64u32, usage: 97u32, description: (b"SoftStepSecondary\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 64u32, usage: 112u32, description: (b"DepthGainCompensation\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 64u32, usage: 128u32, description: (b"ZoomSelect\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 64u32, usage: 129u32, description: (b"ZoomAdjust\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 64u32, usage: 130u32, description: (b"SpectralDopplerModeSelect\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 64u32, usage: 131u32, description: (b"SpectralDopplerAdjust\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 64u32, usage: 132u32, description: (b"ColorDopplerModeSelect\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 64u32, usage: 133u32, description: (b"ColorDopplerAdjust\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 64u32, usage: 134u32, description: (b"MotionModeSelect\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 64u32, usage: 135u32, description: (b"MotionModeAdjust\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 64u32, usage: 136u32, description: (b"2DModeSelect\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 64u32, usage: 137u32, description: (b"2DModeAdjust\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 64u32, usage: 160u32, description: (b"SoftControlSelect\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 64u32, usage: 161u32, description: (b"SoftControlAdjust\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 0u32, description: (b"BrailleDisplay\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 1u32, description: (b"BrailleDisplay\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 2u32, description: (b"BrailleRow\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 3u32, description: (b"8DotBrailleCell\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 4u32, description: (b"6DotBrailleCell\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 5u32, description: (b"NumberofBrailleCells\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 6u32, description: (b"ScreenReaderControl\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 7u32, description: (b"ScreenReaderIdentifier\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 250u32, description: (b"RouterSet1\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 251u32, description: (b"RouterSet2\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 252u32, description: (b"RouterSet3\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 256u32, description: (b"RouterKey\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 257u32, description: (b"RowRouterKey\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 512u32, description: (b"BrailleButtons\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 513u32, description: (b"BrailleKeyboardDot1\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 514u32, description: (b"BrailleKeyboardDot2\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 515u32, description: (b"BrailleKeyboardDot3\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 516u32, description: (b"BrailleKeyboardDot4\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 517u32, description: (b"BrailleKeyboardDot5\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 518u32, description: (b"BrailleKeyboardDot6\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 519u32, description: (b"BrailleKeyboardDot7\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 520u32, description: (b"BrailleKeyboardDot8\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 521u32, description: (b"BrailleKeyboardSpace\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 522u32, description: (b"BrailleKeyboardLeftSpace\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 523u32, description: (b"BrailleKeyboardRightSpace\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 524u32, description: (b"BrailleFaceControls\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 525u32, description: (b"BrailleLeftControls\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 526u32, description: (b"BrailleRightControls\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 527u32, description: (b"BrailleTopControls\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 528u32, description: (b"BrailleJoystickCenter\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 529u32, description: (b"BrailleJoystickUp\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 530u32, description: (b"BrailleJoystickDown\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 531u32, description: (b"BrailleJoystickLeft\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 532u32, description: (b"BrailleJoystickRight\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 533u32, description: (b"BrailleDPadCenter\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 534u32, description: (b"BrailleDPadUp\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 535u32, description: (b"BrailleDPadDown\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 536u32, description: (b"BrailleDPadLeft\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 537u32, description: (b"BrailleDPadRight\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 538u32, description: (b"BraillePanLeft\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 539u32, description: (b"BraillePanRight\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 540u32, description: (b"BrailleRockerUp\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 541u32, description: (b"BrailleRockerDown\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65u32, usage: 542u32, description: (b"BrailleRockerPress\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 89u32, usage: 0u32, description: (b"LightingAndIllumination\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 89u32, usage: 1u32, description: (b"LampArray\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 89u32, usage: 2u32, description: (b"LampArrayAttributesReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 89u32, usage: 3u32, description: (b"LampCount\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 89u32, usage: 4u32, description: (b"BoundingBoxWidthInMicrometers\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 89u32, usage: 5u32, description: (b"BoundingBoxHeightInMicrometers\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 89u32, usage: 6u32, description: (b"BoundingBoxDepthInMicrometers\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 89u32, usage: 7u32, description: (b"LampArrayKind\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 89u32, usage: 8u32, description: (b"MinUpdateIntervalInMicroseconds\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 89u32, usage: 32u32, description: (b"LampAttributesRequestReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 89u32, usage: 33u32, description: (b"LampId\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 89u32, usage: 34u32, description: (b"LampAttributesResponseReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 89u32, usage: 35u32, description: (b"PositionXInMicrometers\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 89u32, usage: 36u32, description: (b"PositionYInMicrometers\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 89u32, usage: 37u32, description: (b"PositionZInMicrometers\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 89u32, usage: 38u32, description: (b"LampPurposes\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 89u32, usage: 39u32, description: (b"UpdateLatencyInMicroseconds\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 89u32, usage: 40u32, description: (b"RedLevelCount\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 89u32, usage: 41u32, description: (b"GreenLevelCount\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 89u32, usage: 42u32, description: (b"BlueLevelCount\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 89u32, usage: 43u32, description: (b"IntensityLevelCount\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 89u32, usage: 44u32, description: (b"IsProgrammable\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 89u32, usage: 45u32, description: (b"InputBinding\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 89u32, usage: 80u32, description: (b"LampMultiUpdateReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 89u32, usage: 81u32, description: (b"RedUpdateChannel\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 89u32, usage: 82u32, description: (b"GreenUpdateChannel\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 89u32, usage: 83u32, description: (b"BlueUpdateChannel\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 89u32, usage: 84u32, description: (b"IntensityUpdateChannel\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 89u32, usage: 85u32, description: (b"LampUpdateFlags\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 89u32, usage: 96u32, description: (b"LampRangeUpdateReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 89u32, usage: 97u32, description: (b"LampIdStart\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 89u32, usage: 98u32, description: (b"LampIdEnd\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 89u32, usage: 112u32, description: (b"LampArrayControlReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 89u32, usage: 113u32, description: (b"AutonomousMode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 128u32, usage: 0u32, description: (b"Monitor\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 128u32, usage: 1u32, description: (b"MonitorControl\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 128u32, usage: 2u32, description: (b"EDIDInformation\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 128u32, usage: 3u32, description: (b"VDIFInformation\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 128u32, usage: 4u32, description: (b"VESAVersion\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 129u32, usage: 0u32, description: (b"MonitorEnumerated\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 0u32, description: (b"VESAVirtualControls\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 1u32, description: (b"Degauss\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 16u32, description: (b"Brightness\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 18u32, description: (b"Contrast\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 22u32, description: (b"RedVideoGain\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 24u32, description: (b"GreenVideoGain\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 26u32, description: (b"BlueVideoGain\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 28u32, description: (b"Focus\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 32u32, description: (b"HorizontalPosition\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 34u32, description: (b"HorizontalSize\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 36u32, description: (b"HorizontalPincushion\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 38u32, description: (b"HorizontalPincushionBalance\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 40u32, description: (b"HorizontalMisconvergence\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 42u32, description: (b"HorizontalLinearity\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 44u32, description: (b"HorizontalLinearityBalance\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 48u32, description: (b"VerticalPosition\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 50u32, description: (b"VerticalSize\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 52u32, description: (b"VerticalPincushion\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 54u32, description: (b"VerticalPincushionBalance\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 56u32, description: (b"VerticalMisconvergence\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 58u32, description: (b"VerticalLinearity\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 60u32, description: (b"VerticalLinearityBalance\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 64u32, description: (b"ParallelogramDistortionKeyBalance\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 66u32, description: (b"TrapezoidalDistortionKey\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 68u32, description: (b"TiltRotation\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 70u32, description: (b"TopCornerDistortionControl\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 72u32, description: (b"TopCornerDistortionBalance\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 74u32, description: (b"BottomCornerDistortionControl\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 76u32, description: (b"BottomCornerDistortionBalance\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 86u32, description: (b"HorizontalMoire\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 88u32, description: (b"VerticalMoire\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 94u32, description: (b"InputLevelSelect\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 96u32, description: (b"InputSourceSelect\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 108u32, description: (b"RedVideoBlackLevel\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 110u32, description: (b"GreenVideoBlackLevel\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 112u32, description: (b"BlueVideoBlackLevel\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 162u32, description: (b"AutoSizeCenter\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 164u32, description: (b"PolarityHorizontalSynchronization\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 166u32, description: (b"PolarityVerticalSynchronization\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 168u32, description: (b"SynchronizationType\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 170u32, description: (b"ScreenOrientation\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 172u32, description: (b"HorizontalFrequency\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 174u32, description: (b"VerticalFrequency\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 176u32, description: (b"Settings\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 202u32, description: (b"OnScreenDisplay\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 130u32, usage: 212u32, description: (b"StereoMode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 0u32, description: (b"Power\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 1u32, description: (b"iName\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 2u32, description: (b"PresentStatus\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 3u32, description: (b"ChangedStatus\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 4u32, description: (b"UPS\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 5u32, description: (b"PowerSupply\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 16u32, description: (b"BatterySystem\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 17u32, description: (b"BatterySystemId\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 18u32, description: (b"Battery\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 19u32, description: (b"BatteryId\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 20u32, description: (b"Charger\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 21u32, description: (b"ChargerId\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 22u32, description: (b"PowerConverter\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 23u32, description: (b"PowerConverterId\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 24u32, description: (b"OutletSystem\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 25u32, description: (b"OutletSystemId\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 26u32, description: (b"Input\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 27u32, description: (b"InputId\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 28u32, description: (b"Output\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 29u32, description: (b"OutputId\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 30u32, description: (b"Flow\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 31u32, description: (b"FlowId\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 32u32, description: (b"Outlet\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 33u32, description: (b"OutletId\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 34u32, description: (b"Gang\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 35u32, description: (b"GangId\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 36u32, description: (b"PowerSummary\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 37u32, description: (b"PowerSummaryId\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 48u32, description: (b"Voltage\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 49u32, description: (b"Current\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 50u32, description: (b"Frequency\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 51u32, description: (b"ApparentPower\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 52u32, description: (b"ActivePower\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 53u32, description: (b"PercentLoad\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 54u32, description: (b"Temperature\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 55u32, description: (b"Humidity\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 56u32, description: (b"BadCount\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 64u32, description: (b"ConfigVoltage\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 65u32, description: (b"ConfigCurrent\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 66u32, description: (b"ConfigFrequency\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 67u32, description: (b"ConfigApparentPower\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 68u32, description: (b"ConfigActivePower\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 69u32, description: (b"ConfigPercentLoad\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 70u32, description: (b"ConfigTemperature\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 71u32, description: (b"ConfigHumidity\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 80u32, description: (b"SwitchOnControl\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 81u32, description: (b"SwitchOffControl\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 82u32, description: (b"ToggleControl\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 83u32, description: (b"LowVoltageTransfer\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 84u32, description: (b"HighVoltageTransfer\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 85u32, description: (b"DelayBeforeReboot\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 86u32, description: (b"DelayBeforeStartup\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 87u32, description: (b"DelayBeforeShutdown\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 88u32, description: (b"Test\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 89u32, description: (b"ModuleReset\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 90u32, description: (b"AudibleAlarmControl\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 96u32, description: (b"Present\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 97u32, description: (b"Good\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 98u32, description: (b"InternalFailure\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 99u32, description: (b"VoltagOutOfRange\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 100u32, description: (b"FrequencyOutOfRange\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 101u32, description: (b"Overload\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 102u32, description: (b"OverCharged\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 103u32, description: (b"OverTemperature\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 104u32, description: (b"ShutdownRequested\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 105u32, description: (b"ShutdownImminent\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 107u32, description: (b"SwitchOnOff\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 108u32, description: (b"Switchable\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 109u32, description: (b"Used\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 110u32, description: (b"Boost\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 111u32, description: (b"Buck\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 112u32, description: (b"Initialized\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 113u32, description: (b"Tested\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 114u32, description: (b"AwaitingPower\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 115u32, description: (b"CommunicationLost\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 253u32, description: (b"iManufacturer\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 254u32, description: (b"iProduct\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 132u32, usage: 255u32, description: (b"iSerialNumber\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 0u32, description: (b"BatterySystem\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 1u32, description: (b"SmartBatteryBatteryMode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 2u32, description: (b"SmartBatteryBatteryStatus\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 3u32, description: (b"SmartBatteryAlarmWarning\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 4u32, description: (b"SmartBatteryChargerMode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 5u32, description: (b"SmartBatteryChargerStatus\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 6u32, description: (b"SmartBatteryChargerSpecInfo\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 7u32, description: (b"SmartBatterySelectorState\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 8u32, description: (b"SmartBatterySelectorPresets\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 9u32, description: (b"SmartBatterySelectorInfo\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 16u32, description: (b"OptionalMfgFunction1\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 17u32, description: (b"OptionalMfgFunction2\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 18u32, description: (b"OptionalMfgFunction3\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 19u32, description: (b"OptionalMfgFunction4\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 20u32, description: (b"OptionalMfgFunction5\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 21u32, description: (b"ConnectionToSMBus\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 22u32, description: (b"OutputConnection\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 23u32, description: (b"ChargerConnection\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 24u32, description: (b"BatteryInsertion\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 25u32, description: (b"UseNext\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 26u32, description: (b"OKToUse\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 27u32, description: (b"BatterySupported\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 28u32, description: (b"SelectorRevision\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 29u32, description: (b"ChargingIndicator\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 40u32, description: (b"ManufacturerAccess\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 41u32, description: (b"RemainingCapacityLimit\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 42u32, description: (b"RemainingTimeLimit\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 43u32, description: (b"AtRate\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 44u32, description: (b"CapacityMode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 45u32, description: (b"BroadcastToCharger\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 46u32, description: (b"PrimaryBattery\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 47u32, description: (b"ChargeController\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 64u32, description: (b"TerminateCharge\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 65u32, description: (b"TerminateDischarge\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 66u32, description: (b"BelowRemainingCapacityLimit\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 67u32, description: (b"RemainingTimeLimitExpired\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 68u32, description: (b"Charging\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 69u32, description: (b"Discharging\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 70u32, description: (b"FullyCharged\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 71u32, description: (b"FullyDischarged\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 72u32, description: (b"ConditioningFlag\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 73u32, description: (b"AtRateOK\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 74u32, description: (b"SmartBatteryErrorCode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 75u32, description: (b"NeedReplacement\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 96u32, description: (b"AtRateTimeToFull\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 97u32, description: (b"AtRateTimeToEmpty\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 98u32, description: (b"AverageCurrent\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 99u32, description: (b"MaxError\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 100u32, description: (b"RelativeStateOfCharge\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 101u32, description: (b"AbsoluteStateOfCharge\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 102u32, description: (b"RemainingCapacity\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 103u32, description: (b"FullChargeCapacity\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 104u32, description: (b"RunTimeToEmpty\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 105u32, description: (b"AverageTimeToEmpty\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 106u32, description: (b"AverageTimeToFull\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 107u32, description: (b"CycleCount\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 128u32, description: (b"BatteryPackModelLevel\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 129u32, description: (b"InternalChargeController\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 130u32, description: (b"PrimaryBatterySupport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 131u32, description: (b"DesignCapacity\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 132u32, description: (b"SpecificationInfo\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 133u32, description: (b"ManufactureDate\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 134u32, description: (b"SerialNumber\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 135u32, description: (b"iManufacturerName\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 136u32, description: (b"iDeviceName\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 137u32, description: (b"iDeviceChemistry\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 138u32, description: (b"ManufacturerData\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 139u32, description: (b"Rechargeable\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 140u32, description: (b"WarningCapacityLimit\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 141u32, description: (b"CapacityGranularity1\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 142u32, description: (b"CapacityGranularity2\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 143u32, description: (b"iOEMInformation\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 192u32, description: (b"InhibitCharge\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 193u32, description: (b"EnablePolling\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 194u32, description: (b"ResetToZero\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 208u32, description: (b"ACPresent\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 209u32, description: (b"BatteryPresent\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 210u32, description: (b"PowerFail\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 211u32, description: (b"AlarmInhibited\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 212u32, description: (b"ThermistorUnderRange\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 213u32, description: (b"ThermistorHot\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 214u32, description: (b"ThermistorCold\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 215u32, description: (b"ThermistorOverRange\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 216u32, description: (b"VoltageOutOfRange\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 217u32, description: (b"CurrentOutOfRange\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 218u32, description: (b"CurrentNotRegulated\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 219u32, description: (b"VoltageNotRegulated\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 220u32, description: (b"MasterMode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 240u32, description: (b"ChargerSelectorSupport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 241u32, description: (b"ChargerSpec\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 242u32, description: (b"Level2\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 133u32, usage: 243u32, description: (b"Level3\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 0u32, description: (b"BarcodeScanner\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 1u32, description: (b"BarcodeBadgeReader\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 2u32, description: (b"BarcodeScanner\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 3u32, description: (b"DumbBarCodeScanner\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 4u32, description: (b"CordlessScannerBase\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 5u32, description: (b"BarCodeScannerCradle\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 16u32, description: (b"AttributeReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 17u32, description: (b"SettingsReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 18u32, description: (b"ScannedDataReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 19u32, description: (b"RawScannedDataReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 20u32, description: (b"TriggerReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 21u32, description: (b"StatusReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 22u32, description: (b"UPCEANControlReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 23u32, description: (b"EAN23LabelControlReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 24u32, description: (b"Code39ControlReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 25u32, description: (b"Interleaved2of5ControlReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 26u32, description: (b"Standard2of5ControlReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 27u32, description: (b"MSIPlesseyControlReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 28u32, description: (b"CodabarControlReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 29u32, description: (b"Code128ControlReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 30u32, description: (b"Misc1DControlReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 31u32, description: (b"2DControlReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 48u32, description: (b"AimingPointerMode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 49u32, description: (b"BarCodePresentSensor\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 50u32, description: (b"Class1ALaser\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 51u32, description: (b"Class2Laser\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 52u32, description: (b"HeaterPresent\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 53u32, description: (b"ContactScanner\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 54u32, description: (b"ElectronicArticleSurveillanceNotification\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 55u32, description: (b"ConstantElectronicArticleSurveillance\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 56u32, description: (b"ErrorIndication\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 57u32, description: (b"FixedBeeper\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 58u32, description: (b"GoodDecodeIndication\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 59u32, description: (b"HandsFreeScanning\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 60u32, description: (b"IntrinsicallySafe\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 61u32, description: (b"KlasseEinsLaser\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 62u32, description: (b"LongRangeScanner\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 63u32, description: (b"MirrorSpeedControl\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 64u32, description: (b"NotOnFileIndication\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 65u32, description: (b"ProgrammableBeeper\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 66u32, description: (b"Triggerless\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 67u32, description: (b"Wand\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 68u32, description: (b"WaterResistant\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 69u32, description: (b"MultiRangeScanner\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 70u32, description: (b"ProximitySensor\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 77u32, description: (b"FragmentDecoding\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 78u32, description: (b"ScannerReadConfidence\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 79u32, description: (b"DataPrefix\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 80u32, description: (b"PrefixAIMI\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 81u32, description: (b"PrefixNone\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 82u32, description: (b"PrefixProprietary\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 85u32, description: (b"ActiveTime\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 86u32, description: (b"AimingLaserPattern\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 87u32, description: (b"BarCodePresent\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 88u32, description: (b"BeeperState\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 89u32, description: (b"LaserOnTime\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 90u32, description: (b"LaserState\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 91u32, description: (b"LockoutTime\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 92u32, description: (b"MotorState\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 93u32, description: (b"MotorTimeout\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 94u32, description: (b"PowerOnResetScanner\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 95u32, description: (b"PreventReadofBarcodes\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 96u32, description: (b"InitiateBarcodeRead\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 97u32, description: (b"TriggerState\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 98u32, description: (b"TriggerMode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 99u32, description: (b"TriggerModeBlinkingLaserOn\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 100u32, description: (b"TriggerModeContinuousLaserOn\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 101u32, description: (b"TriggerModeLaseronwhilePulled\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 102u32, description: (b"TriggerModeLaserstaysonafterrelease\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 109u32, description: (b"CommitParameterstoNVM\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 110u32, description: (b"ParameterScanning\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 111u32, description: (b"ParametersChanged\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 112u32, description: (b"Setparameterdefaultvalues\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 117u32, description: (b"ScannerInCradle\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 118u32, description: (b"ScannerInRange\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 122u32, description: (b"AimDuration\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 123u32, description: (b"GoodReadLampDuration\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 124u32, description: (b"GoodReadLampIntensity\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 125u32, description: (b"GoodReadLED\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 126u32, description: (b"GoodReadToneFrequency\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 127u32, description: (b"GoodReadToneLength\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 128u32, description: (b"GoodReadToneVolume\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 130u32, description: (b"NoReadMessage\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 131u32, description: (b"NotonFileVolume\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 132u32, description: (b"PowerupBeep\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 133u32, description: (b"SoundErrorBeep\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 134u32, description: (b"SoundGoodReadBeep\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 135u32, description: (b"SoundNotOnFileBeep\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 136u32, description: (b"GoodReadWhentoWrite\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 137u32, description: (b"GRWTIAfterDecode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 138u32, description: (b"GRWTIBeepLampaftertransmit\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 139u32, description: (b"GRWTINoBeepLampuseatall\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 145u32, description: (b"BooklandEAN\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 146u32, description: (b"ConvertEAN8to13Type\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 147u32, description: (b"ConvertUPCAtoEAN13\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 148u32, description: (b"ConvertUPCEtoA\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 149u32, description: (b"EAN13\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 150u32, description: (b"EAN8\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 151u32, description: (b"EAN99128Mandatory\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 152u32, description: (b"EAN99P5128Optional\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 153u32, description: (b"EnableEANTwoLabel\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 154u32, description: (b"UPCEAN\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 155u32, description: (b"UPCEANCouponCode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 156u32, description: (b"UPCEANPeriodicals\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 157u32, description: (b"UPCA\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 158u32, description: (b"UPCAwith128Mandatory\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 159u32, description: (b"UPCAwith128Optional\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 160u32, description: (b"UPCAwithP5Optional\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 161u32, description: (b"UPCE\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 162u32, description: (b"UPCE1\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 169u32, description: (b"Periodical\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 170u32, description: (b"PeriodicalAutoDiscriminate2\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 171u32, description: (b"PeriodicalOnlyDecodewith2\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 172u32, description: (b"PeriodicalIgnore2\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 173u32, description: (b"PeriodicalAutoDiscriminate5\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 174u32, description: (b"PeriodicalOnlyDecodewith5\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 175u32, description: (b"PeriodicalIgnore5\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 176u32, description: (b"Check\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 177u32, description: (b"CheckDisablePrice\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 178u32, description: (b"CheckEnable4digitPrice\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 179u32, description: (b"CheckEnable5digitPrice\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 180u32, description: (b"CheckEnableEuropean4digitPrice\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 181u32, description: (b"CheckEnableEuropean5digitPrice\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 183u32, description: (b"EANTwoLabel\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 184u32, description: (b"EANThreeLabel\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 185u32, description: (b"EAN8FlagDigit1\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 186u32, description: (b"EAN8FlagDigit2\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 187u32, description: (b"EAN8FlagDigit3\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 188u32, description: (b"EAN13FlagDigit1\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 189u32, description: (b"EAN13FlagDigit2\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 190u32, description: (b"EAN13FlagDigit3\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 191u32, description: (b"AddEAN23LabelDefinition\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 192u32, description: (b"ClearallEAN23LabelDefinitions\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 195u32, description: (b"Codabar\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 196u32, description: (b"Code128\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 199u32, description: (b"Code39\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 200u32, description: (b"Code93\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 201u32, description: (b"FullASCIIConversion\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 202u32, description: (b"Interleaved2of5\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 203u32, description: (b"ItalianPharmacyCode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 204u32, description: (b"MSIPlessey\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 205u32, description: (b"Standard2of5IATA\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 206u32, description: (b"Standard2of5\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 211u32, description: (b"TransmitStartStop\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 212u32, description: (b"TriOptic\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 213u32, description: (b"UCCEAN128\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 214u32, description: (b"CheckDigit\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 215u32, description: (b"CheckDigitDisable\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 216u32, description: (b"CheckDigitEnableInterleaved2of5OPCC\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 217u32, description: (b"CheckDigitEnableInterleaved2of5USS\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 218u32, description: (b"CheckDigitEnableStandard2of5OPCC\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 219u32, description: (b"CheckDigitEnableStandard2of5USS\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 220u32, description: (b"CheckDigitEnableOneMSIPlessey\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 221u32, description: (b"CheckDigitEnableTwoMSIPlessey\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 222u32, description: (b"CheckDigitCodabarEnable\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 223u32, description: (b"CheckDigitCode39Enable\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 240u32, description: (b"TransmitCheckDigit\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 241u32, description: (b"DisableCheckDigitTransmit\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 242u32, description: (b"EnableCheckDigitTransmit\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 251u32, description: (b"SymbologyIdentifier1\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 252u32, description: (b"SymbologyIdentifier2\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 253u32, description: (b"SymbologyIdentifier3\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 254u32, description: (b"DecodedData\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 255u32, description: (b"DecodeDataContinued\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 256u32, description: (b"BarSpaceData\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 257u32, description: (b"ScannerDataAccuracy\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 258u32, description: (b"RawDataPolarity\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 259u32, description: (b"PolarityInvertedBarCode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 260u32, description: (b"PolarityNormalBarCode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 262u32, description: (b"MinimumLengthtoDecode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 263u32, description: (b"MaximumLengthtoDecode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 264u32, description: (b"DiscreteLengthtoDecode1\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 265u32, description: (b"DiscreteLengthtoDecode2\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 266u32, description: (b"DataLengthMethod\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 267u32, description: (b"DLMethodReadany\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 268u32, description: (b"DLMethodCheckinRange\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 269u32, description: (b"DLMethodCheckforDiscrete\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 272u32, description: (b"AztecCode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 273u32, description: (b"BC412\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 274u32, description: (b"ChannelCode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 275u32, description: (b"Code16\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 276u32, description: (b"Code32\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 277u32, description: (b"Code49\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 278u32, description: (b"CodeOne\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 279u32, description: (b"Colorcode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 280u32, description: (b"DataMatrix\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 281u32, description: (b"MaxiCode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 282u32, description: (b"MicroPDF\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 283u32, description: (b"PDF417\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 284u32, description: (b"PosiCode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 285u32, description: (b"QRCode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 286u32, description: (b"SuperCode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 287u32, description: (b"UltraCode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 288u32, description: (b"USD5SlugCode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 140u32, usage: 289u32, description: (b"VeriCode\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 0u32, description: (b"Scales\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 1u32, description: (b"Scales\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 32u32, description: (b"ScaleDevice\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 33u32, description: (b"ScaleClass\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 34u32, description: (b"ScaleClassIMetric\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 35u32, description: (b"ScaleClassIIMetric\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 36u32, description: (b"ScaleClassIIIMetric\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 37u32, description: (b"ScaleClassIIILMetric\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 38u32, description: (b"ScaleClassIVMetric\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 39u32, description: (b"ScaleClassIIIEnglish\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 40u32, description: (b"ScaleClassIIILEnglish\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 41u32, description: (b"ScaleClassIVEnglish\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 42u32, description: (b"ScaleClassGeneric\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 48u32, description: (b"ScaleAttributeReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 49u32, description: (b"ScaleControlReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 50u32, description: (b"ScaleDataReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 51u32, description: (b"ScaleStatusReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 52u32, description: (b"ScaleWeightLimitReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 53u32, description: (b"ScaleStatisticsReport\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 64u32, description: (b"DataWeight\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 65u32, description: (b"DataScaling\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 80u32, description: (b"WeightUnit\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 81u32, description: (b"WeightUnitMilligram\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 82u32, description: (b"WeightUnitGram\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 83u32, description: (b"WeightUnitKilogram\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 84u32, description: (b"WeightUnitCarats\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 85u32, description: (b"WeightUnitTaels\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 86u32, description: (b"WeightUnitGrains\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 87u32, description: (b"WeightUnitPennyweights\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 88u32, description: (b"WeightUnitMetricTon\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 89u32, description: (b"WeightUnitAvoirTon\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 90u32, description: (b"WeightUnitTroyOunce\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 91u32, description: (b"WeightUnitOunce\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 92u32, description: (b"WeightUnitPound\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 96u32, description: (b"CalibrationCount\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 97u32, description: (b"ReZeroCount\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 112u32, description: (b"ScaleStatus\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 113u32, description: (b"ScaleStatusFault\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 114u32, description: (b"ScaleStatusStableatCenterofZero\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 115u32, description: (b"ScaleStatusInMotion\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 116u32, description: (b"ScaleStatusWeightStable\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 117u32, description: (b"ScaleStatusUnderZero\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 118u32, description: (b"ScaleStatusOverWeightLimit\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 119u32, description: (b"ScaleStatusRequiresCalibration\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 120u32, description: (b"ScaleStatusRequiresRezeroing\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 128u32, description: (b"ZeroScale\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 141u32, usage: 129u32, description: (b"EnforcedZeroReturn\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 142u32, usage: 0u32, description: (b"MagneticStripeReader\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 142u32, usage: 1u32, description: (b"MSRDeviceReadOnly\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 142u32, usage: 17u32, description: (b"Track1Length\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 142u32, usage: 18u32, description: (b"Track2Length\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 142u32, usage: 19u32, description: (b"Track3Length\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 142u32, usage: 20u32, description: (b"TrackJISLength\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 142u32, usage: 32u32, description: (b"TrackData\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 142u32, usage: 33u32, description: (b"Track1Data\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 142u32, usage: 34u32, description: (b"Track2Data\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 142u32, usage: 35u32, description: (b"Track3Data\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 142u32, usage: 36u32, description: (b"TrackJISData\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 144u32, usage: 0u32, description: (b"CameraControl\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 144u32, usage: 32u32, description: (b"CameraAutofocus\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 144u32, usage: 33u32, description: (b"CameraShutter\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 145u32, usage: 0u32, description: (b"Arcade\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 145u32, usage: 1u32, description: (b"GeneralPurposeIOCard\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 145u32, usage: 2u32, description: (b"CoinDoor\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 145u32, usage: 3u32, description: (b"WatchdogTimer\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 145u32, usage: 48u32, description: (b"GeneralPurposeAnalogInputState\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 145u32, usage: 49u32, description: (b"GeneralPurposeDigitalInputState\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 145u32, usage: 50u32, description: (b"GeneralPurposeOpticalInputState\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 145u32, usage: 51u32, description: (b"GeneralPurposeDigitalOutputState\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 145u32, usage: 52u32, description: (b"NumberofCoinDoors\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 145u32, usage: 53u32, description: (b"CoinDrawerDropCount\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 145u32, usage: 54u32, description: (b"CoinDrawerStart\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 145u32, usage: 55u32, description: (b"CoinDrawerService\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 145u32, usage: 56u32, description: (b"CoinDrawerTilt\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 145u32, usage: 57u32, description: (b"CoinDoorTest\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 145u32, usage: 64u32, description: (b"CoinDoorLockout\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 145u32, usage: 65u32, description: (b"WatchdogTimeout\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 145u32, usage: 66u32, description: (b"WatchdogAction\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 145u32, usage: 67u32, description: (b"WatchdogReboot\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 145u32, usage: 68u32, description: (b"WatchdogRestart\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 145u32, usage: 69u32, description: (b"AlarmInput\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 145u32, usage: 70u32, description: (b"CoinDoorCounter\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 145u32, usage: 71u32, description: (b"IODirectionMapping\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 145u32, usage: 72u32, description: (b"SetIODirectionMapping\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 145u32, usage: 73u32, description: (b"ExtendedOpticalInputState\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 145u32, usage: 74u32, description: (b"PinPadInputState\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 145u32, usage: 75u32, description: (b"PinPadStatus\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 145u32, usage: 76u32, description: (b"PinPadOutput\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 145u32, usage: 77u32, description: (b"PinPadCommand\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 61904u32, usage: 0u32, description: (b"FIDOAlliance\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 61904u32, usage: 1u32, description: (b"U2FAuthenticatorDevice\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 61904u32, usage: 32u32, description: (b"InputReportData\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 61904u32, usage: 33u32, description: (b"OutputReportData\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 65535u32, usage: 0u32, description: (b"Vendor-specific-FF\0".as_ptr() as *mut u8) as *const u8 }, hid_usage_entry { page: 0u32, usage: 0u32, description: core::ptr::null::<u8>() }]);

/// Either output directly into simple seq_file, or (if f == NULL)
/// allocate a separate buffer that will then be passed to the 'events'
/// ringbuffer.
///
/// This is because these functions can be called both for "one-shot"
/// "rdesc" while resolving, or for blocking "events".
///
/// This holds both for resolv_usage_page() and hid_resolv_usage().
fn resolv_usage_page(page: u32, f: *mut seq_file) -> *mut u8 {
let mut __slate_compound_2: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_4: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut p: *const hid_usage_entry = unsafe { core::mem::zeroed() };
let mut buf: *mut u8 = core::ptr::null_mut::<u8>();
if f.is_null() {
let mut __v0: *mut ();
let mut __v1: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(512u64, 2080u32, unsafe { *{
__slate_compound_2 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_2)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(512u64, 2080u32, unsafe { *{
__slate_compound_4 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_4)
} });
}
__v1 = _res;
__v0 = __v1;
buf = __v0 as *mut u8;
if buf.is_null() {
return ERR_PTR(12i32.wrapping_neg() as i64) as *mut u8;
}
}
p = unsafe { core::ptr::addr_of!(hid_usage_table.0) as *const hid_usage_entry };
while !unsafe { (*p).description }.is_null() {
if (unsafe { (*p).page }) == page {
if f.is_null() {
unsafe { snprintf(buf, 512u64, c"%s".as_ptr() as *const u8, (*p).description) };
return buf;
} else {
unsafe { seq_printf(f, c"%s".as_ptr() as *const u8, (*p).description) };
return core::ptr::null_mut::<u8>();
}
}
let __v6: *const hid_usage_entry = p;
p = __v6.wrapping_offset(1 as isize);
}
if f.is_null() {
unsafe { snprintf(buf, 512u64, c"%04x".as_ptr() as *const u8, page) };
} else {
unsafe { seq_printf(f, c"%04x".as_ptr() as *const u8, page) };
}
buf
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.hid_debug.hid_resolv_usage")]
extern "C" fn hid_resolv_usage(usage: u32, f: *mut seq_file) -> *mut u8 {
let mut p: *const hid_usage_entry = unsafe { core::mem::zeroed() };
let mut m: *const hid_usage_entry = unsafe { core::mem::zeroed() };
let mut buf: *mut u8 = core::ptr::null_mut::<u8>();
let mut len: i32 = 0;
let mut modifier: *const u8 = core::ptr::null::<u8>();
let usage_modifier: u32 = usage & 61440;
let mut usage_actual: u32 = usage & 65535;
buf = resolv_usage_page(usage >> 16i32, f);
if IS_ERR(buf as *const ()) {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x013hid: error allocating HID debug buffer\n".as_ptr() as *const u8) };
return core::ptr::null_mut::<u8>();
}
if f.is_null() {
len = (unsafe { strlen(buf as *const u8) }) as i32;
let __v2: i32 = unsafe { scnprintf(buf.wrapping_offset(len as isize), 512i32.wrapping_sub(len) as u64, c".".as_ptr() as *const u8) };
len = len.wrapping_add(__v2);
} else {
unsafe { seq_printf(f, c".".as_ptr() as *const u8) };
}
p = unsafe { core::ptr::addr_of!(hid_usage_table.0) as *const hid_usage_entry };
while !unsafe { (*p).description }.is_null() {
if (unsafe { (*p).page }) == usage >> 16i32 {
if (unsafe { (*p).page }) == 32 && usage_modifier != 0 {
m = p;
while !unsafe { (*m).description }.is_null() {
if (unsafe { (*p).page }) == unsafe { (*m).page } && (unsafe { (*m).usage }) == usage_modifier {
modifier = unsafe { (*m).description };
break;
}
let __v6: *const hid_usage_entry = m;
m = __v6.wrapping_offset(1 as isize);
}
if !modifier.is_null() {
usage_actual &= 4095;
}
}
if modifier.is_null() {
modifier = c"".as_ptr() as *const u8;
}
let __v8: *const hid_usage_entry = p;
p = __v8.wrapping_offset(1 as isize);
while !unsafe { (*p).description }.is_null() && (unsafe { (*p).usage }) != 0 {
if (unsafe { (*p).usage }) == usage_actual {
if f.is_null() {
unsafe { snprintf(buf.wrapping_offset(len as isize), 512i32.wrapping_sub(len) as u64, c"%s%s".as_ptr() as *const u8, (*p).description, modifier) };
} else {
unsafe { seq_printf(f, c"%s%s".as_ptr() as *const u8, (*p).description, modifier) };
}
return buf;
}
let __v11: *const hid_usage_entry = p;
p = __v11.wrapping_offset(1 as isize);
}
break;
}
let __v14: *const hid_usage_entry = p;
p = __v14.wrapping_offset(1 as isize);
}
if f.is_null() {
unsafe { snprintf(buf.wrapping_offset(len as isize), 512i32.wrapping_sub(len) as u64, c"%04x".as_ptr() as *const u8, usage & 65535) };
} else {
unsafe { seq_printf(f, c"%04x".as_ptr() as *const u8, usage & 65535) };
}
buf
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_hid_resolv_usage_632: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u32, *mut seq_file) -> *mut u8>, *mut ()>(Some(hid_resolv_usage)) };

fn tab(n: i32, f: *mut seq_file) {
unsafe { seq_printf(f, c"%*s".as_ptr() as *const u8, n, c"".as_ptr() as *mut u8) };
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.hid_debug.hid_dump_field")]
extern "C" fn hid_dump_field(field: *mut hid_field, n: i32, f: *mut seq_file) {
let mut j: i32 = 0;
if (unsafe { (*field).physical }) != 0 {
tab(n, f);
unsafe { seq_printf(f, c"Physical(".as_ptr() as *const u8) };
hid_resolv_usage(unsafe { (*field).physical }, f);
unsafe { seq_printf(f, c")\n".as_ptr() as *const u8) };
}
if (unsafe { (*field).logical }) != 0 {
tab(n, f);
unsafe { seq_printf(f, c"Logical(".as_ptr() as *const u8) };
hid_resolv_usage(unsafe { (*field).logical }, f);
unsafe { seq_printf(f, c")\n".as_ptr() as *const u8) };
}
if (unsafe { (*field).application }) != 0 {
tab(n, f);
unsafe { seq_printf(f, c"Application(".as_ptr() as *const u8) };
hid_resolv_usage(unsafe { (*field).application }, f);
unsafe { seq_printf(f, c")\n".as_ptr() as *const u8) };
}
tab(n, f);
unsafe { seq_printf(f, c"Usage(%d)\n".as_ptr() as *const u8, (*field).maxusage) };
while (j as u32) < unsafe { (*field).maxusage } {
tab(n.wrapping_add(2), f);
hid_resolv_usage(unsafe { (*unsafe { (*field).usage }.wrapping_offset(j as isize)).hid }, f);
unsafe { seq_printf(f, c"\n".as_ptr() as *const u8) };
j = j.wrapping_add(1);
}
if (unsafe { (*field).logical_minimum }) != unsafe { (*field).logical_maximum } {
tab(n, f);
unsafe { seq_printf(f, c"Logical Minimum(%d)\n".as_ptr() as *const u8, (*field).logical_minimum) };
tab(n, f);
unsafe { seq_printf(f, c"Logical Maximum(%d)\n".as_ptr() as *const u8, (*field).logical_maximum) };
}
if (unsafe { (*field).physical_minimum }) != unsafe { (*field).physical_maximum } {
tab(n, f);
unsafe { seq_printf(f, c"Physical Minimum(%d)\n".as_ptr() as *const u8, (*field).physical_minimum) };
tab(n, f);
unsafe { seq_printf(f, c"Physical Maximum(%d)\n".as_ptr() as *const u8, (*field).physical_maximum) };
}
if (unsafe { (*field).unit_exponent }) != 0 {
tab(n, f);
unsafe { seq_printf(f, c"Unit Exponent(%d)\n".as_ptr() as *const u8, (*field).unit_exponent) };
}
if (unsafe { (*field).unit }) != 0 {
let mut i: i32 = 0;
let mut sys: i32 = 0;
let mut data: u32 = unsafe { (*field).unit };
// First nibble tells us which system we're in.
sys = (data & 15) as i32;
data >>= 4i32;
if sys > 4 {
tab(n, f);
unsafe { seq_printf(f, c"Unit(Invalid)\n".as_ptr() as *const u8) };
} else {
let mut earlier_unit: i32 = 0;
tab(n, f);
unsafe { seq_printf(f, c"Unit(%s : ".as_ptr() as *const u8, systems.0[sys as usize]) };
i = 1;
while (i as u64) < (core::mem::size_of::<u32>() as u64).wrapping_mul(2) {
let nibble: u8 = (data & 15) as u8;
data >>= 4i32;
if (nibble as i32) != 0 {
let __v7: i32 = earlier_unit;
earlier_unit = __v7.wrapping_add(1);
if __v7 > 0 {
unsafe { seq_printf(f, c"*".as_ptr() as *const u8) };
}
unsafe { seq_printf(f, c"%s".as_ptr() as *const u8, units.0[sys as usize][i as usize]) };
if (nibble as i32) != 1 {
// This is a _signed_ nibble(!)
let mut val: i32 = (nibble as i32) & 7;
if (nibble as i32) & 8 != 0 {
val = (7 & !val).wrapping_add(1).wrapping_neg();
}
unsafe { seq_printf(f, c"^%d".as_ptr() as *const u8, val) };
}
}
i = i.wrapping_add(1);
}
unsafe { seq_printf(f, c")\n".as_ptr() as *const u8) };
}
}
tab(n, f);
unsafe { seq_printf(f, c"Report Size(%u)\n".as_ptr() as *const u8, (*field).report_size) };
tab(n, f);
unsafe { seq_printf(f, c"Report Count(%u)\n".as_ptr() as *const u8, (*field).report_count) };
tab(n, f);
unsafe { seq_printf(f, c"Report Offset(%u)\n".as_ptr() as *const u8, (*field).report_offset) };
tab(n, f);
unsafe { seq_printf(f, c"Flags( ".as_ptr() as *const u8) };
j = (unsafe { (*field).flags }) as i32;
unsafe { seq_printf(f, c"%s".as_ptr() as *const u8, if 1 & j != 0 { c"Constant ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }) };
unsafe { seq_printf(f, c"%s".as_ptr() as *const u8, if 2 & j != 0 { c"Variable ".as_ptr() as *mut u8 } else { c"Array ".as_ptr() as *mut u8 }) };
unsafe { seq_printf(f, c"%s".as_ptr() as *const u8, if 4 & j != 0 { c"Relative ".as_ptr() as *mut u8 } else { c"Absolute ".as_ptr() as *mut u8 }) };
unsafe { seq_printf(f, c"%s".as_ptr() as *const u8, if 8 & j != 0 { c"Wrap ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }) };
unsafe { seq_printf(f, c"%s".as_ptr() as *const u8, if 16 & j != 0 { c"NonLinear ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }) };
unsafe { seq_printf(f, c"%s".as_ptr() as *const u8, if 32 & j != 0 { c"NoPreferredState ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }) };
unsafe { seq_printf(f, c"%s".as_ptr() as *const u8, if 64 & j != 0 { c"NullState ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }) };
unsafe { seq_printf(f, c"%s".as_ptr() as *const u8, if 128 & j != 0 { c"Volatile ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }) };
unsafe { seq_printf(f, c"%s".as_ptr() as *const u8, if 256 & j != 0 { c"BufferedByte ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 }) };
unsafe { seq_printf(f, c")\n".as_ptr() as *const u8) };
}

static mut systems: __SlateAlign16<[*const u8; 5]> = __SlateAlign16([(b"None\0".as_ptr() as *mut u8) as *const u8, (b"SI Linear\0".as_ptr() as *mut u8) as *const u8, (b"SI Rotation\0".as_ptr() as *mut u8) as *const u8, (b"English Linear\0".as_ptr() as *mut u8) as *const u8, (b"English Rotation\0".as_ptr() as *mut u8) as *const u8]);

static mut units: __SlateAlign16<[[*const u8; 8]; 5]> = __SlateAlign16([[(b"None\0".as_ptr() as *mut u8) as *const u8, (b"None\0".as_ptr() as *mut u8) as *const u8, (b"None\0".as_ptr() as *mut u8) as *const u8, (b"None\0".as_ptr() as *mut u8) as *const u8, (b"None\0".as_ptr() as *mut u8) as *const u8, (b"None\0".as_ptr() as *mut u8) as *const u8, (b"None\0".as_ptr() as *mut u8) as *const u8, (b"None\0".as_ptr() as *mut u8) as *const u8], [(b"None\0".as_ptr() as *mut u8) as *const u8, (b"Centimeter\0".as_ptr() as *mut u8) as *const u8, (b"Gram\0".as_ptr() as *mut u8) as *const u8, (b"Seconds\0".as_ptr() as *mut u8) as *const u8, (b"Kelvin\0".as_ptr() as *mut u8) as *const u8, (b"Ampere\0".as_ptr() as *mut u8) as *const u8, (b"Candela\0".as_ptr() as *mut u8) as *const u8, (b"None\0".as_ptr() as *mut u8) as *const u8], [(b"None\0".as_ptr() as *mut u8) as *const u8, (b"Radians\0".as_ptr() as *mut u8) as *const u8, (b"Gram\0".as_ptr() as *mut u8) as *const u8, (b"Seconds\0".as_ptr() as *mut u8) as *const u8, (b"Kelvin\0".as_ptr() as *mut u8) as *const u8, (b"Ampere\0".as_ptr() as *mut u8) as *const u8, (b"Candela\0".as_ptr() as *mut u8) as *const u8, (b"None\0".as_ptr() as *mut u8) as *const u8], [(b"None\0".as_ptr() as *mut u8) as *const u8, (b"Inch\0".as_ptr() as *mut u8) as *const u8, (b"Slug\0".as_ptr() as *mut u8) as *const u8, (b"Seconds\0".as_ptr() as *mut u8) as *const u8, (b"Fahrenheit\0".as_ptr() as *mut u8) as *const u8, (b"Ampere\0".as_ptr() as *mut u8) as *const u8, (b"Candela\0".as_ptr() as *mut u8) as *const u8, (b"None\0".as_ptr() as *mut u8) as *const u8], [(b"None\0".as_ptr() as *mut u8) as *const u8, (b"Degrees\0".as_ptr() as *mut u8) as *const u8, (b"Slug\0".as_ptr() as *mut u8) as *const u8, (b"Seconds\0".as_ptr() as *mut u8) as *const u8, (b"Fahrenheit\0".as_ptr() as *mut u8) as *const u8, (b"Ampere\0".as_ptr() as *mut u8) as *const u8, (b"Candela\0".as_ptr() as *mut u8) as *const u8, (b"None\0".as_ptr() as *mut u8) as *const u8]]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_hid_dump_field_633: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hid_field, i32, *mut seq_file)>, *mut ()>(Some(hid_dump_field)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.hid_debug.hid_dump_device")]
extern "C" fn hid_dump_device(device: *mut hid_device, f: *mut seq_file) {
let mut report_enum: *mut hid_report_enum = unsafe { core::mem::zeroed() };
let mut list: *mut list_head = unsafe { core::mem::zeroed() };
let mut k: u32 = 0;
for i in 0u32..3u32 {
report_enum = unsafe { (*device).report_enum.as_mut_ptr() as *mut hid_report_enum }.wrapping_offset(i as isize);
list = unsafe { (*report_enum).report_list.next };
while list != unsafe { core::ptr::addr_of_mut!((*report_enum).report_list) } {
tab(2i32, f);
unsafe { seq_printf(f, c"%s".as_ptr() as *const u8, table.0[i as usize]) };
if (unsafe { (*(list as *mut hid_report)).id }) != 0 {
unsafe { seq_printf(f, c"(%d)".as_ptr() as *const u8, (*(list as *mut hid_report)).id) };
}
unsafe { seq_printf(f, c"[%s]".as_ptr() as *const u8, table.0[((*(list as *mut hid_report)).r#type as u32) as usize]) };
unsafe { seq_printf(f, c"\n".as_ptr() as *const u8) };
k = 0;
while k < unsafe { (*(list as *mut hid_report)).maxfield } {
tab(4i32, f);
unsafe { seq_printf(f, c"Field(%d)\n".as_ptr() as *const u8, k) };
hid_dump_field(unsafe { *((*(list as *mut hid_report)).field.as_mut_ptr() as *mut *mut hid_field).wrapping_offset(k as isize) }, 6i32, f);
k = k.wrapping_add(1);
}
list = unsafe { (*list).next };
}
}
}

static mut table: __SlateAlign16<[*const u8; 3]> = __SlateAlign16([(b"INPUT\0".as_ptr() as *mut u8) as *const u8, (b"OUTPUT\0".as_ptr() as *mut u8) as *const u8, (b"FEATURE\0".as_ptr() as *mut u8) as *const u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_hid_dump_device_634: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hid_device, *mut seq_file)>, *mut ()>(Some(hid_dump_device)) };

/// enqueue string to 'events' ring buffer
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.hid_debug.hid_debug_event")]
extern "C" fn hid_debug_event(hdev: *mut hid_device, buf: *mut u8) {
let mut list: *mut hid_debug_list = unsafe { core::mem::zeroed() };
let mut flags: u64 = 0;
spin_lock_irqsave__2!(unsafe { core::ptr::addr_of_mut!((*hdev).debug_list_lock) }, flags);
list = list_first_entry__39!(unsafe { core::ptr::addr_of_mut!((*hdev).debug_list) });
while list_entry_is_head!(list, unsafe { core::ptr::addr_of_mut!((*hdev).debug_list) }, node) == 0 {
kfifo_in!(unsafe { core::ptr::addr_of_mut!((*list).hid_debug_fifo) }, buf, unsafe { strlen(buf as *const u8) });
list = list_next_entry__39!(list, node);
}
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*hdev).debug_list_lock) }, flags);
wake_up_interruptible!(unsafe { core::ptr::addr_of_mut!((*hdev).debug_wait) });
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_hid_debug_event_635: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hid_device, *mut u8)>, *mut ()>(Some(hid_debug_event)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.hid_debug.hid_dump_report")]
extern "C" fn hid_dump_report(hid: *mut hid_device, r#type: i32, data: *mut u8, size: i32) {
let mut __slate_compound_2: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_4: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __v0: *mut ();
let mut __v1: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_noprof(512u64, 2080u32, unsafe { *{
__slate_compound_2 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_2)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_noprof(512u64, 2080u32, unsafe { *{
__slate_compound_4 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_4)
} });
}
__v1 = _res;
__v0 = __v1;
let buf: *mut u8 = __v0 as *mut u8;
if buf.is_null() {
return;
}
let report_enum: *mut hid_report_enum = unsafe { (*hid).report_enum.as_mut_ptr() as *mut hid_report_enum }.wrapping_offset(r#type as isize);
// dump the report
unsafe { snprintf(buf, 511u64, c"\nreport (size %u) (%snumbered) = ".as_ptr() as *const u8, size, if (*report_enum).numbered != 0 { c"".as_ptr() as *mut u8 } else { c"un".as_ptr() as *mut u8 }) };
hid_debug_event(hid, buf);
for i in 0..size as u32 {
unsafe { snprintf(buf, 511u64, c" %02x".as_ptr() as *const u8, *data.wrapping_offset(i as isize) as i32) };
hid_debug_event(hid, buf);
}
hid_debug_event(hid, c"\n".as_ptr() as *mut u8);
unsafe { kfree(buf as *const ()) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_hid_dump_report_636: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hid_device, i32, *mut u8, i32)>, *mut ()>(Some(hid_dump_report)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.hid_debug.hid_dump_input")]
extern "C" fn hid_dump_input(hdev: *mut hid_device, usage: *mut hid_usage, value: i32) {
let mut len: i32 = 0;
let buf: *mut u8 = hid_resolv_usage(unsafe { (*usage).hid }, core::ptr::null_mut::<seq_file>());
if buf.is_null() {
return;
}
len = (unsafe { strlen(buf as *const u8) }) as i32;
unsafe { snprintf(buf.wrapping_offset(len as isize), 512i32.wrapping_sub(len).wrapping_sub(1) as u64, c" = %d\n".as_ptr() as *const u8, value) };
hid_debug_event(hdev, buf);
unsafe { kfree(buf as *const ()) };
wake_up_interruptible!(unsafe { core::ptr::addr_of_mut!((*hdev).debug_wait) });
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_hid_dump_input_637: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut hid_device, *mut hid_usage, i32)>, *mut ()>(Some(hid_dump_input)) };

static mut events: __SlateAlign16<[*const u8; 32]> = __SlateAlign16({
let mut __t0: [*const u8; 32] = unsafe { core::mem::zeroed() };
__t0[0] = (b"Sync\0".as_ptr() as *mut u8) as *const u8;
__t0[1] = (b"Key\0".as_ptr() as *mut u8) as *const u8;
__t0[2] = (b"Relative\0".as_ptr() as *mut u8) as *const u8;
__t0[3] = (b"Absolute\0".as_ptr() as *mut u8) as *const u8;
__t0[4] = (b"Misc\0".as_ptr() as *mut u8) as *const u8;
__t0[5] = (b"Software\0".as_ptr() as *mut u8) as *const u8;
__t0[17] = (b"LED\0".as_ptr() as *mut u8) as *const u8;
__t0[18] = (b"Sound\0".as_ptr() as *mut u8) as *const u8;
__t0[20] = (b"Repeat\0".as_ptr() as *mut u8) as *const u8;
__t0[21] = (b"ForceFeedback\0".as_ptr() as *mut u8) as *const u8;
__t0[22] = (b"Power\0".as_ptr() as *mut u8) as *const u8;
__t0[23] = (b"ForceFeedbackStatus\0".as_ptr() as *mut u8) as *const u8;
__t0
});

static mut syncs: __SlateAlign16<[*const u8; 16]> = __SlateAlign16({
let mut __t0: [*const u8; 16] = unsafe { core::mem::zeroed() };
__t0[0] = (b"Report\0".as_ptr() as *mut u8) as *const u8;
__t0[1] = (b"Config\0".as_ptr() as *mut u8) as *const u8;
__t0[2] = (b"MT Report\0".as_ptr() as *mut u8) as *const u8;
__t0[3] = (b"Dropped\0".as_ptr() as *mut u8) as *const u8;
__t0
});

static mut keys: __SlateAlign16<[*const u8; 768]> = __SlateAlign16({
let mut __t0: [*const u8; 768] = unsafe { core::mem::zeroed() };
__t0[0] = (b"Reserved\0".as_ptr() as *mut u8) as *const u8;
__t0[1] = (b"Esc\0".as_ptr() as *mut u8) as *const u8;
__t0[2] = (b"1\0".as_ptr() as *mut u8) as *const u8;
__t0[3] = (b"2\0".as_ptr() as *mut u8) as *const u8;
__t0[4] = (b"3\0".as_ptr() as *mut u8) as *const u8;
__t0[5] = (b"4\0".as_ptr() as *mut u8) as *const u8;
__t0[6] = (b"5\0".as_ptr() as *mut u8) as *const u8;
__t0[7] = (b"6\0".as_ptr() as *mut u8) as *const u8;
__t0[8] = (b"7\0".as_ptr() as *mut u8) as *const u8;
__t0[9] = (b"8\0".as_ptr() as *mut u8) as *const u8;
__t0[10] = (b"9\0".as_ptr() as *mut u8) as *const u8;
__t0[11] = (b"0\0".as_ptr() as *mut u8) as *const u8;
__t0[12] = (b"Minus\0".as_ptr() as *mut u8) as *const u8;
__t0[13] = (b"Equal\0".as_ptr() as *mut u8) as *const u8;
__t0[14] = (b"Backspace\0".as_ptr() as *mut u8) as *const u8;
__t0[15] = (b"Tab\0".as_ptr() as *mut u8) as *const u8;
__t0[16] = (b"Q\0".as_ptr() as *mut u8) as *const u8;
__t0[17] = (b"W\0".as_ptr() as *mut u8) as *const u8;
__t0[18] = (b"E\0".as_ptr() as *mut u8) as *const u8;
__t0[19] = (b"R\0".as_ptr() as *mut u8) as *const u8;
__t0[20] = (b"T\0".as_ptr() as *mut u8) as *const u8;
__t0[21] = (b"Y\0".as_ptr() as *mut u8) as *const u8;
__t0[22] = (b"U\0".as_ptr() as *mut u8) as *const u8;
__t0[23] = (b"I\0".as_ptr() as *mut u8) as *const u8;
__t0[24] = (b"O\0".as_ptr() as *mut u8) as *const u8;
__t0[25] = (b"P\0".as_ptr() as *mut u8) as *const u8;
__t0[26] = (b"LeftBrace\0".as_ptr() as *mut u8) as *const u8;
__t0[27] = (b"RightBrace\0".as_ptr() as *mut u8) as *const u8;
__t0[28] = (b"Enter\0".as_ptr() as *mut u8) as *const u8;
__t0[29] = (b"LeftControl\0".as_ptr() as *mut u8) as *const u8;
__t0[30] = (b"A\0".as_ptr() as *mut u8) as *const u8;
__t0[31] = (b"S\0".as_ptr() as *mut u8) as *const u8;
__t0[32] = (b"D\0".as_ptr() as *mut u8) as *const u8;
__t0[33] = (b"F\0".as_ptr() as *mut u8) as *const u8;
__t0[34] = (b"G\0".as_ptr() as *mut u8) as *const u8;
__t0[35] = (b"H\0".as_ptr() as *mut u8) as *const u8;
__t0[36] = (b"J\0".as_ptr() as *mut u8) as *const u8;
__t0[37] = (b"K\0".as_ptr() as *mut u8) as *const u8;
__t0[38] = (b"L\0".as_ptr() as *mut u8) as *const u8;
__t0[39] = (b"Semicolon\0".as_ptr() as *mut u8) as *const u8;
__t0[40] = (b"Apostrophe\0".as_ptr() as *mut u8) as *const u8;
__t0[41] = (b"Grave\0".as_ptr() as *mut u8) as *const u8;
__t0[42] = (b"LeftShift\0".as_ptr() as *mut u8) as *const u8;
__t0[43] = (b"BackSlash\0".as_ptr() as *mut u8) as *const u8;
__t0[44] = (b"Z\0".as_ptr() as *mut u8) as *const u8;
__t0[45] = (b"X\0".as_ptr() as *mut u8) as *const u8;
__t0[46] = (b"C\0".as_ptr() as *mut u8) as *const u8;
__t0[47] = (b"V\0".as_ptr() as *mut u8) as *const u8;
__t0[48] = (b"B\0".as_ptr() as *mut u8) as *const u8;
__t0[49] = (b"N\0".as_ptr() as *mut u8) as *const u8;
__t0[50] = (b"M\0".as_ptr() as *mut u8) as *const u8;
__t0[51] = (b"Comma\0".as_ptr() as *mut u8) as *const u8;
__t0[52] = (b"Dot\0".as_ptr() as *mut u8) as *const u8;
__t0[53] = (b"Slash\0".as_ptr() as *mut u8) as *const u8;
__t0[54] = (b"RightShift\0".as_ptr() as *mut u8) as *const u8;
__t0[55] = (b"KPAsterisk\0".as_ptr() as *mut u8) as *const u8;
__t0[56] = (b"LeftAlt\0".as_ptr() as *mut u8) as *const u8;
__t0[57] = (b"Space\0".as_ptr() as *mut u8) as *const u8;
__t0[58] = (b"CapsLock\0".as_ptr() as *mut u8) as *const u8;
__t0[59] = (b"F1\0".as_ptr() as *mut u8) as *const u8;
__t0[60] = (b"F2\0".as_ptr() as *mut u8) as *const u8;
__t0[61] = (b"F3\0".as_ptr() as *mut u8) as *const u8;
__t0[62] = (b"F4\0".as_ptr() as *mut u8) as *const u8;
__t0[63] = (b"F5\0".as_ptr() as *mut u8) as *const u8;
__t0[64] = (b"F6\0".as_ptr() as *mut u8) as *const u8;
__t0[65] = (b"F7\0".as_ptr() as *mut u8) as *const u8;
__t0[66] = (b"F8\0".as_ptr() as *mut u8) as *const u8;
__t0[67] = (b"F9\0".as_ptr() as *mut u8) as *const u8;
__t0[68] = (b"F10\0".as_ptr() as *mut u8) as *const u8;
__t0[69] = (b"NumLock\0".as_ptr() as *mut u8) as *const u8;
__t0[70] = (b"ScrollLock\0".as_ptr() as *mut u8) as *const u8;
__t0[71] = (b"KP7\0".as_ptr() as *mut u8) as *const u8;
__t0[72] = (b"KP8\0".as_ptr() as *mut u8) as *const u8;
__t0[73] = (b"KP9\0".as_ptr() as *mut u8) as *const u8;
__t0[74] = (b"KPMinus\0".as_ptr() as *mut u8) as *const u8;
__t0[75] = (b"KP4\0".as_ptr() as *mut u8) as *const u8;
__t0[76] = (b"KP5\0".as_ptr() as *mut u8) as *const u8;
__t0[77] = (b"KP6\0".as_ptr() as *mut u8) as *const u8;
__t0[78] = (b"KPPlus\0".as_ptr() as *mut u8) as *const u8;
__t0[79] = (b"KP1\0".as_ptr() as *mut u8) as *const u8;
__t0[80] = (b"KP2\0".as_ptr() as *mut u8) as *const u8;
__t0[81] = (b"KP3\0".as_ptr() as *mut u8) as *const u8;
__t0[82] = (b"KP0\0".as_ptr() as *mut u8) as *const u8;
__t0[83] = (b"KPDot\0".as_ptr() as *mut u8) as *const u8;
__t0[85] = (b"Zenkaku/Hankaku\0".as_ptr() as *mut u8) as *const u8;
__t0[86] = (b"102nd\0".as_ptr() as *mut u8) as *const u8;
__t0[87] = (b"F11\0".as_ptr() as *mut u8) as *const u8;
__t0[88] = (b"F12\0".as_ptr() as *mut u8) as *const u8;
__t0[89] = (b"RO\0".as_ptr() as *mut u8) as *const u8;
__t0[90] = (b"Katakana\0".as_ptr() as *mut u8) as *const u8;
__t0[91] = (b"HIRAGANA\0".as_ptr() as *mut u8) as *const u8;
__t0[92] = (b"Henkan\0".as_ptr() as *mut u8) as *const u8;
__t0[93] = (b"Katakana/Hiragana\0".as_ptr() as *mut u8) as *const u8;
__t0[94] = (b"Muhenkan\0".as_ptr() as *mut u8) as *const u8;
__t0[95] = (b"KPJpComma\0".as_ptr() as *mut u8) as *const u8;
__t0[96] = (b"KPEnter\0".as_ptr() as *mut u8) as *const u8;
__t0[97] = (b"RightCtrl\0".as_ptr() as *mut u8) as *const u8;
__t0[98] = (b"KPSlash\0".as_ptr() as *mut u8) as *const u8;
__t0[99] = (b"SysRq\0".as_ptr() as *mut u8) as *const u8;
__t0[100] = (b"RightAlt\0".as_ptr() as *mut u8) as *const u8;
__t0[101] = (b"LineFeed\0".as_ptr() as *mut u8) as *const u8;
__t0[102] = (b"Home\0".as_ptr() as *mut u8) as *const u8;
__t0[103] = (b"Up\0".as_ptr() as *mut u8) as *const u8;
__t0[104] = (b"PageUp\0".as_ptr() as *mut u8) as *const u8;
__t0[105] = (b"Left\0".as_ptr() as *mut u8) as *const u8;
__t0[106] = (b"Right\0".as_ptr() as *mut u8) as *const u8;
__t0[107] = (b"End\0".as_ptr() as *mut u8) as *const u8;
__t0[108] = (b"Down\0".as_ptr() as *mut u8) as *const u8;
__t0[109] = (b"PageDown\0".as_ptr() as *mut u8) as *const u8;
__t0[110] = (b"Insert\0".as_ptr() as *mut u8) as *const u8;
__t0[111] = (b"Delete\0".as_ptr() as *mut u8) as *const u8;
__t0[112] = (b"Macro\0".as_ptr() as *mut u8) as *const u8;
__t0[113] = (b"Mute\0".as_ptr() as *mut u8) as *const u8;
__t0[114] = (b"VolumeDown\0".as_ptr() as *mut u8) as *const u8;
__t0[115] = (b"VolumeUp\0".as_ptr() as *mut u8) as *const u8;
__t0[116] = (b"Power\0".as_ptr() as *mut u8) as *const u8;
__t0[117] = (b"KPEqual\0".as_ptr() as *mut u8) as *const u8;
__t0[118] = (b"KPPlusMinus\0".as_ptr() as *mut u8) as *const u8;
__t0[119] = (b"Pause\0".as_ptr() as *mut u8) as *const u8;
__t0[120] = (b"Scale\0".as_ptr() as *mut u8) as *const u8;
__t0[121] = (b"KPComma\0".as_ptr() as *mut u8) as *const u8;
__t0[122] = (b"Hangeul\0".as_ptr() as *mut u8) as *const u8;
__t0[123] = (b"Hanja\0".as_ptr() as *mut u8) as *const u8;
__t0[124] = (b"Yen\0".as_ptr() as *mut u8) as *const u8;
__t0[125] = (b"LeftMeta\0".as_ptr() as *mut u8) as *const u8;
__t0[126] = (b"RightMeta\0".as_ptr() as *mut u8) as *const u8;
__t0[127] = (b"Compose\0".as_ptr() as *mut u8) as *const u8;
__t0[128] = (b"Stop\0".as_ptr() as *mut u8) as *const u8;
__t0[129] = (b"Again\0".as_ptr() as *mut u8) as *const u8;
__t0[130] = (b"Props\0".as_ptr() as *mut u8) as *const u8;
__t0[131] = (b"Undo\0".as_ptr() as *mut u8) as *const u8;
__t0[132] = (b"Front\0".as_ptr() as *mut u8) as *const u8;
__t0[133] = (b"Copy\0".as_ptr() as *mut u8) as *const u8;
__t0[134] = (b"Open\0".as_ptr() as *mut u8) as *const u8;
__t0[135] = (b"Paste\0".as_ptr() as *mut u8) as *const u8;
__t0[136] = (b"Find\0".as_ptr() as *mut u8) as *const u8;
__t0[137] = (b"Cut\0".as_ptr() as *mut u8) as *const u8;
__t0[138] = (b"Help\0".as_ptr() as *mut u8) as *const u8;
__t0[139] = (b"Menu\0".as_ptr() as *mut u8) as *const u8;
__t0[140] = (b"Calc\0".as_ptr() as *mut u8) as *const u8;
__t0[141] = (b"Setup\0".as_ptr() as *mut u8) as *const u8;
__t0[142] = (b"Sleep\0".as_ptr() as *mut u8) as *const u8;
__t0[143] = (b"WakeUp\0".as_ptr() as *mut u8) as *const u8;
__t0[144] = (b"File\0".as_ptr() as *mut u8) as *const u8;
__t0[145] = (b"SendFile\0".as_ptr() as *mut u8) as *const u8;
__t0[146] = (b"DeleteFile\0".as_ptr() as *mut u8) as *const u8;
__t0[147] = (b"X-fer\0".as_ptr() as *mut u8) as *const u8;
__t0[148] = (b"Prog1\0".as_ptr() as *mut u8) as *const u8;
__t0[149] = (b"Prog2\0".as_ptr() as *mut u8) as *const u8;
__t0[150] = (b"WWW\0".as_ptr() as *mut u8) as *const u8;
__t0[151] = (b"MSDOS\0".as_ptr() as *mut u8) as *const u8;
__t0[152] = (b"Coffee\0".as_ptr() as *mut u8) as *const u8;
__t0[153] = (b"RotateDisplay\0".as_ptr() as *mut u8) as *const u8;
__t0[154] = (b"CycleWindows\0".as_ptr() as *mut u8) as *const u8;
__t0[155] = (b"Mail\0".as_ptr() as *mut u8) as *const u8;
__t0[156] = (b"Bookmarks\0".as_ptr() as *mut u8) as *const u8;
__t0[157] = (b"Computer\0".as_ptr() as *mut u8) as *const u8;
__t0[158] = (b"Back\0".as_ptr() as *mut u8) as *const u8;
__t0[159] = (b"Forward\0".as_ptr() as *mut u8) as *const u8;
__t0[160] = (b"CloseCD\0".as_ptr() as *mut u8) as *const u8;
__t0[161] = (b"EjectCD\0".as_ptr() as *mut u8) as *const u8;
__t0[162] = (b"EjectCloseCD\0".as_ptr() as *mut u8) as *const u8;
__t0[163] = (b"NextSong\0".as_ptr() as *mut u8) as *const u8;
__t0[164] = (b"PlayPause\0".as_ptr() as *mut u8) as *const u8;
__t0[165] = (b"PreviousSong\0".as_ptr() as *mut u8) as *const u8;
__t0[166] = (b"StopCD\0".as_ptr() as *mut u8) as *const u8;
__t0[167] = (b"Record\0".as_ptr() as *mut u8) as *const u8;
__t0[168] = (b"Rewind\0".as_ptr() as *mut u8) as *const u8;
__t0[169] = (b"Phone\0".as_ptr() as *mut u8) as *const u8;
__t0[170] = (b"ISOKey\0".as_ptr() as *mut u8) as *const u8;
__t0[171] = (b"Config\0".as_ptr() as *mut u8) as *const u8;
__t0[172] = (b"HomePage\0".as_ptr() as *mut u8) as *const u8;
__t0[173] = (b"Refresh\0".as_ptr() as *mut u8) as *const u8;
__t0[174] = (b"Exit\0".as_ptr() as *mut u8) as *const u8;
__t0[175] = (b"Move\0".as_ptr() as *mut u8) as *const u8;
__t0[176] = (b"Edit\0".as_ptr() as *mut u8) as *const u8;
__t0[177] = (b"ScrollUp\0".as_ptr() as *mut u8) as *const u8;
__t0[178] = (b"ScrollDown\0".as_ptr() as *mut u8) as *const u8;
__t0[179] = (b"KPLeftParenthesis\0".as_ptr() as *mut u8) as *const u8;
__t0[180] = (b"KPRightParenthesis\0".as_ptr() as *mut u8) as *const u8;
__t0[181] = (b"New\0".as_ptr() as *mut u8) as *const u8;
__t0[182] = (b"Redo\0".as_ptr() as *mut u8) as *const u8;
__t0[183] = (b"F13\0".as_ptr() as *mut u8) as *const u8;
__t0[184] = (b"F14\0".as_ptr() as *mut u8) as *const u8;
__t0[185] = (b"F15\0".as_ptr() as *mut u8) as *const u8;
__t0[186] = (b"F16\0".as_ptr() as *mut u8) as *const u8;
__t0[187] = (b"F17\0".as_ptr() as *mut u8) as *const u8;
__t0[188] = (b"F18\0".as_ptr() as *mut u8) as *const u8;
__t0[189] = (b"F19\0".as_ptr() as *mut u8) as *const u8;
__t0[190] = (b"F20\0".as_ptr() as *mut u8) as *const u8;
__t0[191] = (b"F21\0".as_ptr() as *mut u8) as *const u8;
__t0[192] = (b"F22\0".as_ptr() as *mut u8) as *const u8;
__t0[193] = (b"F23\0".as_ptr() as *mut u8) as *const u8;
__t0[194] = (b"F24\0".as_ptr() as *mut u8) as *const u8;
__t0[200] = (b"PlayCD\0".as_ptr() as *mut u8) as *const u8;
__t0[201] = (b"PauseCD\0".as_ptr() as *mut u8) as *const u8;
__t0[202] = (b"Prog3\0".as_ptr() as *mut u8) as *const u8;
__t0[203] = (b"Prog4\0".as_ptr() as *mut u8) as *const u8;
__t0[204] = (b"AllApplications\0".as_ptr() as *mut u8) as *const u8;
__t0[205] = (b"Suspend\0".as_ptr() as *mut u8) as *const u8;
__t0[206] = (b"Close\0".as_ptr() as *mut u8) as *const u8;
__t0[207] = (b"Play\0".as_ptr() as *mut u8) as *const u8;
__t0[208] = (b"FastForward\0".as_ptr() as *mut u8) as *const u8;
__t0[209] = (b"BassBoost\0".as_ptr() as *mut u8) as *const u8;
__t0[210] = (b"Print\0".as_ptr() as *mut u8) as *const u8;
__t0[211] = (b"HP\0".as_ptr() as *mut u8) as *const u8;
__t0[212] = (b"Camera\0".as_ptr() as *mut u8) as *const u8;
__t0[213] = (b"Sound\0".as_ptr() as *mut u8) as *const u8;
__t0[214] = (b"Question\0".as_ptr() as *mut u8) as *const u8;
__t0[215] = (b"Email\0".as_ptr() as *mut u8) as *const u8;
__t0[216] = (b"Chat\0".as_ptr() as *mut u8) as *const u8;
__t0[217] = (b"Search\0".as_ptr() as *mut u8) as *const u8;
__t0[218] = (b"Connect\0".as_ptr() as *mut u8) as *const u8;
__t0[219] = (b"Finance\0".as_ptr() as *mut u8) as *const u8;
__t0[220] = (b"Sport\0".as_ptr() as *mut u8) as *const u8;
__t0[221] = (b"Shop\0".as_ptr() as *mut u8) as *const u8;
__t0[222] = (b"AlternateErase\0".as_ptr() as *mut u8) as *const u8;
__t0[223] = (b"Cancel\0".as_ptr() as *mut u8) as *const u8;
__t0[224] = (b"BrightnessDown\0".as_ptr() as *mut u8) as *const u8;
__t0[225] = (b"BrightnessUp\0".as_ptr() as *mut u8) as *const u8;
__t0[226] = (b"Media\0".as_ptr() as *mut u8) as *const u8;
__t0[227] = (b"SwitchVideoMode\0".as_ptr() as *mut u8) as *const u8;
__t0[228] = (b"KbdIlluminationToggle\0".as_ptr() as *mut u8) as *const u8;
__t0[229] = (b"KbdIlluminationDown\0".as_ptr() as *mut u8) as *const u8;
__t0[230] = (b"KbdIlluminationUp\0".as_ptr() as *mut u8) as *const u8;
__t0[231] = (b"Send\0".as_ptr() as *mut u8) as *const u8;
__t0[232] = (b"Reply\0".as_ptr() as *mut u8) as *const u8;
__t0[233] = (b"ForwardMail\0".as_ptr() as *mut u8) as *const u8;
__t0[234] = (b"Save\0".as_ptr() as *mut u8) as *const u8;
__t0[235] = (b"Documents\0".as_ptr() as *mut u8) as *const u8;
__t0[236] = (b"Battery\0".as_ptr() as *mut u8) as *const u8;
__t0[237] = (b"BlueTooth\0".as_ptr() as *mut u8) as *const u8;
__t0[238] = (b"WLAN\0".as_ptr() as *mut u8) as *const u8;
__t0[239] = (b"UWB\0".as_ptr() as *mut u8) as *const u8;
__t0[240] = (b"Unknown\0".as_ptr() as *mut u8) as *const u8;
__t0[241] = (b"VideoNext\0".as_ptr() as *mut u8) as *const u8;
__t0[242] = (b"VideoPrev\0".as_ptr() as *mut u8) as *const u8;
__t0[243] = (b"BrightnessCycle\0".as_ptr() as *mut u8) as *const u8;
__t0[244] = (b"BrightnessAuto\0".as_ptr() as *mut u8) as *const u8;
__t0[245] = (b"DisplayOff\0".as_ptr() as *mut u8) as *const u8;
__t0[246] = (b"WWAN\0".as_ptr() as *mut u8) as *const u8;
__t0[247] = (b"RFKill\0".as_ptr() as *mut u8) as *const u8;
__t0[248] = (b"MicrophoneMute\0".as_ptr() as *mut u8) as *const u8;
__t0[256] = (b"Btn0\0".as_ptr() as *mut u8) as *const u8;
__t0[257] = (b"Btn1\0".as_ptr() as *mut u8) as *const u8;
__t0[258] = (b"Btn2\0".as_ptr() as *mut u8) as *const u8;
__t0[259] = (b"Btn3\0".as_ptr() as *mut u8) as *const u8;
__t0[260] = (b"Btn4\0".as_ptr() as *mut u8) as *const u8;
__t0[261] = (b"Btn5\0".as_ptr() as *mut u8) as *const u8;
__t0[262] = (b"Btn6\0".as_ptr() as *mut u8) as *const u8;
__t0[263] = (b"Btn7\0".as_ptr() as *mut u8) as *const u8;
__t0[264] = (b"Btn8\0".as_ptr() as *mut u8) as *const u8;
__t0[265] = (b"Btn9\0".as_ptr() as *mut u8) as *const u8;
__t0[272] = (b"LeftBtn\0".as_ptr() as *mut u8) as *const u8;
__t0[273] = (b"RightBtn\0".as_ptr() as *mut u8) as *const u8;
__t0[274] = (b"MiddleBtn\0".as_ptr() as *mut u8) as *const u8;
__t0[275] = (b"SideBtn\0".as_ptr() as *mut u8) as *const u8;
__t0[276] = (b"ExtraBtn\0".as_ptr() as *mut u8) as *const u8;
__t0[277] = (b"ForwardBtn\0".as_ptr() as *mut u8) as *const u8;
__t0[278] = (b"BackBtn\0".as_ptr() as *mut u8) as *const u8;
__t0[279] = (b"TaskBtn\0".as_ptr() as *mut u8) as *const u8;
__t0[288] = (b"Trigger\0".as_ptr() as *mut u8) as *const u8;
__t0[289] = (b"ThumbBtn\0".as_ptr() as *mut u8) as *const u8;
__t0[290] = (b"ThumbBtn2\0".as_ptr() as *mut u8) as *const u8;
__t0[291] = (b"TopBtn\0".as_ptr() as *mut u8) as *const u8;
__t0[292] = (b"TopBtn2\0".as_ptr() as *mut u8) as *const u8;
__t0[293] = (b"PinkieBtn\0".as_ptr() as *mut u8) as *const u8;
__t0[294] = (b"BaseBtn\0".as_ptr() as *mut u8) as *const u8;
__t0[295] = (b"BaseBtn2\0".as_ptr() as *mut u8) as *const u8;
__t0[296] = (b"BaseBtn3\0".as_ptr() as *mut u8) as *const u8;
__t0[297] = (b"BaseBtn4\0".as_ptr() as *mut u8) as *const u8;
__t0[298] = (b"BaseBtn5\0".as_ptr() as *mut u8) as *const u8;
__t0[299] = (b"BaseBtn6\0".as_ptr() as *mut u8) as *const u8;
__t0[303] = (b"BtnDead\0".as_ptr() as *mut u8) as *const u8;
__t0[304] = (b"BtnA\0".as_ptr() as *mut u8) as *const u8;
__t0[305] = (b"BtnB\0".as_ptr() as *mut u8) as *const u8;
__t0[306] = (b"BtnC\0".as_ptr() as *mut u8) as *const u8;
__t0[307] = (b"BtnX\0".as_ptr() as *mut u8) as *const u8;
__t0[308] = (b"BtnY\0".as_ptr() as *mut u8) as *const u8;
__t0[309] = (b"BtnZ\0".as_ptr() as *mut u8) as *const u8;
__t0[310] = (b"BtnTL\0".as_ptr() as *mut u8) as *const u8;
__t0[311] = (b"BtnTR\0".as_ptr() as *mut u8) as *const u8;
__t0[312] = (b"BtnTL2\0".as_ptr() as *mut u8) as *const u8;
__t0[313] = (b"BtnTR2\0".as_ptr() as *mut u8) as *const u8;
__t0[314] = (b"BtnSelect\0".as_ptr() as *mut u8) as *const u8;
__t0[315] = (b"BtnStart\0".as_ptr() as *mut u8) as *const u8;
__t0[316] = (b"BtnMode\0".as_ptr() as *mut u8) as *const u8;
__t0[317] = (b"BtnThumbL\0".as_ptr() as *mut u8) as *const u8;
__t0[318] = (b"BtnThumbR\0".as_ptr() as *mut u8) as *const u8;
__t0[320] = (b"ToolPen\0".as_ptr() as *mut u8) as *const u8;
__t0[321] = (b"ToolRubber\0".as_ptr() as *mut u8) as *const u8;
__t0[322] = (b"ToolBrush\0".as_ptr() as *mut u8) as *const u8;
__t0[323] = (b"ToolPencil\0".as_ptr() as *mut u8) as *const u8;
__t0[324] = (b"ToolAirbrush\0".as_ptr() as *mut u8) as *const u8;
__t0[325] = (b"ToolFinger\0".as_ptr() as *mut u8) as *const u8;
__t0[326] = (b"ToolMouse\0".as_ptr() as *mut u8) as *const u8;
__t0[327] = (b"ToolLens\0".as_ptr() as *mut u8) as *const u8;
__t0[328] = (b"ToolQuintTap\0".as_ptr() as *mut u8) as *const u8;
__t0[329] = (b"Stylus3\0".as_ptr() as *mut u8) as *const u8;
__t0[330] = (b"Touch\0".as_ptr() as *mut u8) as *const u8;
__t0[331] = (b"Stylus\0".as_ptr() as *mut u8) as *const u8;
__t0[332] = (b"Stylus2\0".as_ptr() as *mut u8) as *const u8;
__t0[333] = (b"ToolDoubleTap\0".as_ptr() as *mut u8) as *const u8;
__t0[334] = (b"ToolTripleTap\0".as_ptr() as *mut u8) as *const u8;
__t0[335] = (b"ToolQuadrupleTap\0".as_ptr() as *mut u8) as *const u8;
__t0[336] = (b"BtnGearDown\0".as_ptr() as *mut u8) as *const u8;
__t0[337] = (b"BtnGearUp\0".as_ptr() as *mut u8) as *const u8;
__t0[352] = (b"Ok\0".as_ptr() as *mut u8) as *const u8;
__t0[353] = (b"Select\0".as_ptr() as *mut u8) as *const u8;
__t0[354] = (b"Goto\0".as_ptr() as *mut u8) as *const u8;
__t0[355] = (b"Clear\0".as_ptr() as *mut u8) as *const u8;
__t0[356] = (b"Power2\0".as_ptr() as *mut u8) as *const u8;
__t0[357] = (b"Option\0".as_ptr() as *mut u8) as *const u8;
__t0[358] = (b"Info\0".as_ptr() as *mut u8) as *const u8;
__t0[359] = (b"Time\0".as_ptr() as *mut u8) as *const u8;
__t0[360] = (b"Vendor\0".as_ptr() as *mut u8) as *const u8;
__t0[361] = (b"Archive\0".as_ptr() as *mut u8) as *const u8;
__t0[362] = (b"Program\0".as_ptr() as *mut u8) as *const u8;
__t0[363] = (b"Channel\0".as_ptr() as *mut u8) as *const u8;
__t0[364] = (b"Favorites\0".as_ptr() as *mut u8) as *const u8;
__t0[365] = (b"EPG\0".as_ptr() as *mut u8) as *const u8;
__t0[366] = (b"PVR\0".as_ptr() as *mut u8) as *const u8;
__t0[367] = (b"MHP\0".as_ptr() as *mut u8) as *const u8;
__t0[368] = (b"Language\0".as_ptr() as *mut u8) as *const u8;
__t0[369] = (b"Title\0".as_ptr() as *mut u8) as *const u8;
__t0[370] = (b"Subtitle\0".as_ptr() as *mut u8) as *const u8;
__t0[371] = (b"Angle\0".as_ptr() as *mut u8) as *const u8;
__t0[372] = (b"FullScreen\0".as_ptr() as *mut u8) as *const u8;
__t0[373] = (b"Mode\0".as_ptr() as *mut u8) as *const u8;
__t0[374] = (b"Keyboard\0".as_ptr() as *mut u8) as *const u8;
__t0[375] = (b"AspectRatio\0".as_ptr() as *mut u8) as *const u8;
__t0[376] = (b"PC\0".as_ptr() as *mut u8) as *const u8;
__t0[377] = (b"TV\0".as_ptr() as *mut u8) as *const u8;
__t0[378] = (b"TV2\0".as_ptr() as *mut u8) as *const u8;
__t0[379] = (b"VCR\0".as_ptr() as *mut u8) as *const u8;
__t0[380] = (b"VCR2\0".as_ptr() as *mut u8) as *const u8;
__t0[381] = (b"Sat\0".as_ptr() as *mut u8) as *const u8;
__t0[382] = (b"Sat2\0".as_ptr() as *mut u8) as *const u8;
__t0[383] = (b"CD\0".as_ptr() as *mut u8) as *const u8;
__t0[384] = (b"Tape\0".as_ptr() as *mut u8) as *const u8;
__t0[385] = (b"Radio\0".as_ptr() as *mut u8) as *const u8;
__t0[386] = (b"Tuner\0".as_ptr() as *mut u8) as *const u8;
__t0[387] = (b"Player\0".as_ptr() as *mut u8) as *const u8;
__t0[388] = (b"Text\0".as_ptr() as *mut u8) as *const u8;
__t0[389] = (b"DVD\0".as_ptr() as *mut u8) as *const u8;
__t0[390] = (b"Aux\0".as_ptr() as *mut u8) as *const u8;
__t0[391] = (b"MP3\0".as_ptr() as *mut u8) as *const u8;
__t0[392] = (b"Audio\0".as_ptr() as *mut u8) as *const u8;
__t0[393] = (b"Video\0".as_ptr() as *mut u8) as *const u8;
__t0[394] = (b"Directory\0".as_ptr() as *mut u8) as *const u8;
__t0[395] = (b"List\0".as_ptr() as *mut u8) as *const u8;
__t0[396] = (b"Memo\0".as_ptr() as *mut u8) as *const u8;
__t0[397] = (b"Calendar\0".as_ptr() as *mut u8) as *const u8;
__t0[398] = (b"Red\0".as_ptr() as *mut u8) as *const u8;
__t0[399] = (b"Green\0".as_ptr() as *mut u8) as *const u8;
__t0[400] = (b"Yellow\0".as_ptr() as *mut u8) as *const u8;
__t0[401] = (b"Blue\0".as_ptr() as *mut u8) as *const u8;
__t0[402] = (b"ChannelUp\0".as_ptr() as *mut u8) as *const u8;
__t0[403] = (b"ChannelDown\0".as_ptr() as *mut u8) as *const u8;
__t0[404] = (b"First\0".as_ptr() as *mut u8) as *const u8;
__t0[405] = (b"Last\0".as_ptr() as *mut u8) as *const u8;
__t0[406] = (b"AB\0".as_ptr() as *mut u8) as *const u8;
__t0[407] = (b"Next\0".as_ptr() as *mut u8) as *const u8;
__t0[408] = (b"Restart\0".as_ptr() as *mut u8) as *const u8;
__t0[409] = (b"Slow\0".as_ptr() as *mut u8) as *const u8;
__t0[410] = (b"Shuffle\0".as_ptr() as *mut u8) as *const u8;
__t0[411] = (b"Break\0".as_ptr() as *mut u8) as *const u8;
__t0[412] = (b"Previous\0".as_ptr() as *mut u8) as *const u8;
__t0[413] = (b"Digits\0".as_ptr() as *mut u8) as *const u8;
__t0[414] = (b"TEEN\0".as_ptr() as *mut u8) as *const u8;
__t0[415] = (b"TWEN\0".as_ptr() as *mut u8) as *const u8;
__t0[416] = (b"VideoPhone\0".as_ptr() as *mut u8) as *const u8;
__t0[417] = (b"Games\0".as_ptr() as *mut u8) as *const u8;
__t0[418] = (b"ZoomIn\0".as_ptr() as *mut u8) as *const u8;
__t0[419] = (b"ZoomOut\0".as_ptr() as *mut u8) as *const u8;
__t0[420] = (b"ZoomReset\0".as_ptr() as *mut u8) as *const u8;
__t0[421] = (b"WordProcessor\0".as_ptr() as *mut u8) as *const u8;
__t0[422] = (b"Editor\0".as_ptr() as *mut u8) as *const u8;
__t0[423] = (b"SpreadSheet\0".as_ptr() as *mut u8) as *const u8;
__t0[424] = (b"GraphicsEditor\0".as_ptr() as *mut u8) as *const u8;
__t0[425] = (b"Presentation\0".as_ptr() as *mut u8) as *const u8;
__t0[426] = (b"DataBase\0".as_ptr() as *mut u8) as *const u8;
__t0[427] = (b"News\0".as_ptr() as *mut u8) as *const u8;
__t0[428] = (b"VoiceMail\0".as_ptr() as *mut u8) as *const u8;
__t0[429] = (b"Addressbook\0".as_ptr() as *mut u8) as *const u8;
__t0[430] = (b"Messenger\0".as_ptr() as *mut u8) as *const u8;
__t0[431] = (b"DisplayToggle\0".as_ptr() as *mut u8) as *const u8;
__t0[432] = (b"SpellCheck\0".as_ptr() as *mut u8) as *const u8;
__t0[433] = (b"Logoff\0".as_ptr() as *mut u8) as *const u8;
__t0[434] = (b"Dollar\0".as_ptr() as *mut u8) as *const u8;
__t0[435] = (b"Euro\0".as_ptr() as *mut u8) as *const u8;
__t0[436] = (b"FrameBack\0".as_ptr() as *mut u8) as *const u8;
__t0[437] = (b"FrameForward\0".as_ptr() as *mut u8) as *const u8;
__t0[438] = (b"ContextMenu\0".as_ptr() as *mut u8) as *const u8;
__t0[439] = (b"MediaRepeat\0".as_ptr() as *mut u8) as *const u8;
__t0[440] = (b"10ChannelsUp\0".as_ptr() as *mut u8) as *const u8;
__t0[441] = (b"10ChannelsDown\0".as_ptr() as *mut u8) as *const u8;
__t0[442] = (b"Images\0".as_ptr() as *mut u8) as *const u8;
__t0[444] = (b"NotificationCenter\0".as_ptr() as *mut u8) as *const u8;
__t0[445] = (b"PickUpPhone\0".as_ptr() as *mut u8) as *const u8;
__t0[446] = (b"HangUpPhone\0".as_ptr() as *mut u8) as *const u8;
__t0[448] = (b"DeleteEOL\0".as_ptr() as *mut u8) as *const u8;
__t0[449] = (b"DeleteEOS\0".as_ptr() as *mut u8) as *const u8;
__t0[450] = (b"InsertLine\0".as_ptr() as *mut u8) as *const u8;
__t0[451] = (b"DeleteLine\0".as_ptr() as *mut u8) as *const u8;
__t0[464] = (b"Fn\0".as_ptr() as *mut u8) as *const u8;
__t0[465] = (b"Fn+ESC\0".as_ptr() as *mut u8) as *const u8;
__t0[466] = (b"Fn+F1\0".as_ptr() as *mut u8) as *const u8;
__t0[467] = (b"Fn+F2\0".as_ptr() as *mut u8) as *const u8;
__t0[468] = (b"Fn+F3\0".as_ptr() as *mut u8) as *const u8;
__t0[469] = (b"Fn+F4\0".as_ptr() as *mut u8) as *const u8;
__t0[470] = (b"Fn+F5\0".as_ptr() as *mut u8) as *const u8;
__t0[471] = (b"Fn+F6\0".as_ptr() as *mut u8) as *const u8;
__t0[472] = (b"Fn+F7\0".as_ptr() as *mut u8) as *const u8;
__t0[473] = (b"Fn+F8\0".as_ptr() as *mut u8) as *const u8;
__t0[474] = (b"Fn+F9\0".as_ptr() as *mut u8) as *const u8;
__t0[475] = (b"Fn+F10\0".as_ptr() as *mut u8) as *const u8;
__t0[476] = (b"Fn+F11\0".as_ptr() as *mut u8) as *const u8;
__t0[477] = (b"Fn+F12\0".as_ptr() as *mut u8) as *const u8;
__t0[478] = (b"Fn+1\0".as_ptr() as *mut u8) as *const u8;
__t0[479] = (b"Fn+2\0".as_ptr() as *mut u8) as *const u8;
__t0[480] = (b"Fn+D\0".as_ptr() as *mut u8) as *const u8;
__t0[481] = (b"Fn+E\0".as_ptr() as *mut u8) as *const u8;
__t0[482] = (b"Fn+F\0".as_ptr() as *mut u8) as *const u8;
__t0[483] = (b"Fn+S\0".as_ptr() as *mut u8) as *const u8;
__t0[484] = (b"Fn+B\0".as_ptr() as *mut u8) as *const u8;
__t0[485] = (b"FnRightShift\0".as_ptr() as *mut u8) as *const u8;
__t0[497] = (b"BrlDot1\0".as_ptr() as *mut u8) as *const u8;
__t0[498] = (b"BrlDot2\0".as_ptr() as *mut u8) as *const u8;
__t0[499] = (b"BrlDot3\0".as_ptr() as *mut u8) as *const u8;
__t0[500] = (b"BrlDot4\0".as_ptr() as *mut u8) as *const u8;
__t0[501] = (b"BrlDot5\0".as_ptr() as *mut u8) as *const u8;
__t0[502] = (b"BrlDot6\0".as_ptr() as *mut u8) as *const u8;
__t0[503] = (b"BrlDot7\0".as_ptr() as *mut u8) as *const u8;
__t0[504] = (b"BrlDot8\0".as_ptr() as *mut u8) as *const u8;
__t0[505] = (b"BrlDot9\0".as_ptr() as *mut u8) as *const u8;
__t0[506] = (b"BrlDot10\0".as_ptr() as *mut u8) as *const u8;
__t0[512] = (b"Numeric0\0".as_ptr() as *mut u8) as *const u8;
__t0[513] = (b"Numeric1\0".as_ptr() as *mut u8) as *const u8;
__t0[514] = (b"Numeric2\0".as_ptr() as *mut u8) as *const u8;
__t0[515] = (b"Numeric3\0".as_ptr() as *mut u8) as *const u8;
__t0[516] = (b"Numeric4\0".as_ptr() as *mut u8) as *const u8;
__t0[517] = (b"Numeric5\0".as_ptr() as *mut u8) as *const u8;
__t0[518] = (b"Numeric6\0".as_ptr() as *mut u8) as *const u8;
__t0[519] = (b"Numeric7\0".as_ptr() as *mut u8) as *const u8;
__t0[520] = (b"Numeric8\0".as_ptr() as *mut u8) as *const u8;
__t0[521] = (b"Numeric9\0".as_ptr() as *mut u8) as *const u8;
__t0[522] = (b"NumericStar\0".as_ptr() as *mut u8) as *const u8;
__t0[523] = (b"NumericPound\0".as_ptr() as *mut u8) as *const u8;
__t0[524] = (b"NumericA\0".as_ptr() as *mut u8) as *const u8;
__t0[525] = (b"NumericB\0".as_ptr() as *mut u8) as *const u8;
__t0[526] = (b"NumericC\0".as_ptr() as *mut u8) as *const u8;
__t0[527] = (b"NumericD\0".as_ptr() as *mut u8) as *const u8;
__t0[528] = (b"CameraFocus\0".as_ptr() as *mut u8) as *const u8;
__t0[529] = (b"WPSButton\0".as_ptr() as *mut u8) as *const u8;
__t0[530] = (b"TouchPadToggle\0".as_ptr() as *mut u8) as *const u8;
__t0[531] = (b"TouchPadOn\0".as_ptr() as *mut u8) as *const u8;
__t0[532] = (b"TouchPadOff\0".as_ptr() as *mut u8) as *const u8;
__t0[533] = (b"CameraZoomIn\0".as_ptr() as *mut u8) as *const u8;
__t0[534] = (b"CameraZoomOut\0".as_ptr() as *mut u8) as *const u8;
__t0[535] = (b"CameraUp\0".as_ptr() as *mut u8) as *const u8;
__t0[536] = (b"CameraDown\0".as_ptr() as *mut u8) as *const u8;
__t0[537] = (b"CameraLeft\0".as_ptr() as *mut u8) as *const u8;
__t0[538] = (b"CameraRight\0".as_ptr() as *mut u8) as *const u8;
__t0[539] = (b"AttendantOn\0".as_ptr() as *mut u8) as *const u8;
__t0[540] = (b"AttendantOff\0".as_ptr() as *mut u8) as *const u8;
__t0[541] = (b"AttendantToggle\0".as_ptr() as *mut u8) as *const u8;
__t0[542] = (b"LightToggle\0".as_ptr() as *mut u8) as *const u8;
__t0[544] = (b"BtnDPadUp\0".as_ptr() as *mut u8) as *const u8;
__t0[545] = (b"BtnDPadDown\0".as_ptr() as *mut u8) as *const u8;
__t0[546] = (b"BtnDPadLeft\0".as_ptr() as *mut u8) as *const u8;
__t0[547] = (b"BtnDPadRight\0".as_ptr() as *mut u8) as *const u8;
__t0[548] = (b"BtnGripL\0".as_ptr() as *mut u8) as *const u8;
__t0[549] = (b"BtnGripR\0".as_ptr() as *mut u8) as *const u8;
__t0[550] = (b"BtnGripL2\0".as_ptr() as *mut u8) as *const u8;
__t0[551] = (b"BtnGripR2\0".as_ptr() as *mut u8) as *const u8;
__t0[560] = (b"ALSToggle\0".as_ptr() as *mut u8) as *const u8;
__t0[561] = (b"RotateLockToggle\0".as_ptr() as *mut u8) as *const u8;
__t0[576] = (b"ButtonConfig\0".as_ptr() as *mut u8) as *const u8;
__t0[577] = (b"TaskManager\0".as_ptr() as *mut u8) as *const u8;
__t0[578] = (b"Journal\0".as_ptr() as *mut u8) as *const u8;
__t0[579] = (b"ControlPanel\0".as_ptr() as *mut u8) as *const u8;
__t0[580] = (b"AppSelect\0".as_ptr() as *mut u8) as *const u8;
__t0[581] = (b"ScreenSaver\0".as_ptr() as *mut u8) as *const u8;
__t0[582] = (b"VoiceCommand\0".as_ptr() as *mut u8) as *const u8;
__t0[583] = (b"Assistant\0".as_ptr() as *mut u8) as *const u8;
__t0[584] = (b"KbdLayoutNext\0".as_ptr() as *mut u8) as *const u8;
__t0[585] = (b"EmojiPicker\0".as_ptr() as *mut u8) as *const u8;
__t0[586] = (b"Dictate\0".as_ptr() as *mut u8) as *const u8;
__t0[587] = (b"CameraAccessEnable\0".as_ptr() as *mut u8) as *const u8;
__t0[588] = (b"CameraAccessDisable\0".as_ptr() as *mut u8) as *const u8;
__t0[589] = (b"CameraAccessToggle\0".as_ptr() as *mut u8) as *const u8;
__t0[590] = (b"Accessibility\0".as_ptr() as *mut u8) as *const u8;
__t0[591] = (b"DoNotDisturb\0".as_ptr() as *mut u8) as *const u8;
__t0[592] = (b"BrightnessMin\0".as_ptr() as *mut u8) as *const u8;
__t0[593] = (b"BrightnessMax\0".as_ptr() as *mut u8) as *const u8;
__t0[596] = (b"ActionOnSelection\0".as_ptr() as *mut u8) as *const u8;
__t0[597] = (b"ContextualInsert\0".as_ptr() as *mut u8) as *const u8;
__t0[598] = (b"ContextualQuery\0".as_ptr() as *mut u8) as *const u8;
__t0[608] = (b"KbdInputAssistPrev\0".as_ptr() as *mut u8) as *const u8;
__t0[609] = (b"KbdInputAssistNext\0".as_ptr() as *mut u8) as *const u8;
__t0[610] = (b"KbdInputAssistPrevGroup\0".as_ptr() as *mut u8) as *const u8;
__t0[611] = (b"KbdInputAssistNextGroup\0".as_ptr() as *mut u8) as *const u8;
__t0[612] = (b"KbdInputAssistAccept\0".as_ptr() as *mut u8) as *const u8;
__t0[613] = (b"KbdInputAssistCancel\0".as_ptr() as *mut u8) as *const u8;
__t0[614] = (b"RightUp\0".as_ptr() as *mut u8) as *const u8;
__t0[615] = (b"RightDown\0".as_ptr() as *mut u8) as *const u8;
__t0[616] = (b"LeftUp\0".as_ptr() as *mut u8) as *const u8;
__t0[617] = (b"LeftDown\0".as_ptr() as *mut u8) as *const u8;
__t0[618] = (b"RootMenu\0".as_ptr() as *mut u8) as *const u8;
__t0[619] = (b"MediaTopMenu\0".as_ptr() as *mut u8) as *const u8;
__t0[620] = (b"Numceric11\0".as_ptr() as *mut u8) as *const u8;
__t0[621] = (b"Numeric12\0".as_ptr() as *mut u8) as *const u8;
__t0[622] = (b"AudioDesc\0".as_ptr() as *mut u8) as *const u8;
__t0[623] = (b"3DMode\0".as_ptr() as *mut u8) as *const u8;
__t0[624] = (b"NextFavorite\0".as_ptr() as *mut u8) as *const u8;
__t0[625] = (b"StopRecord\0".as_ptr() as *mut u8) as *const u8;
__t0[626] = (b"PauseRecord\0".as_ptr() as *mut u8) as *const u8;
__t0[627] = (b"VOD\0".as_ptr() as *mut u8) as *const u8;
__t0[628] = (b"Unmute\0".as_ptr() as *mut u8) as *const u8;
__t0[629] = (b"FastReverse\0".as_ptr() as *mut u8) as *const u8;
__t0[630] = (b"SlowReverse\0".as_ptr() as *mut u8) as *const u8;
__t0[631] = (b"Data\0".as_ptr() as *mut u8) as *const u8;
__t0[632] = (b"OnScreenKeyBoard\0".as_ptr() as *mut u8) as *const u8;
__t0[633] = (b"PrivacyScreenToggle\0".as_ptr() as *mut u8) as *const u8;
__t0[634] = (b"SelectiveScreenshot\0".as_ptr() as *mut u8) as *const u8;
__t0[635] = (b"NextElement\0".as_ptr() as *mut u8) as *const u8;
__t0[636] = (b"PreviousElement\0".as_ptr() as *mut u8) as *const u8;
__t0[637] = (b"AutoPiloteEngage\0".as_ptr() as *mut u8) as *const u8;
__t0[638] = (b"MarkWayPoint\0".as_ptr() as *mut u8) as *const u8;
__t0[639] = (b"SOS\0".as_ptr() as *mut u8) as *const u8;
__t0[640] = (b"NavChar\0".as_ptr() as *mut u8) as *const u8;
__t0[641] = (b"FishingChart\0".as_ptr() as *mut u8) as *const u8;
__t0[642] = (b"SingleRangeRadar\0".as_ptr() as *mut u8) as *const u8;
__t0[643] = (b"DualRangeRadat\0".as_ptr() as *mut u8) as *const u8;
__t0[644] = (b"RadarOverLay\0".as_ptr() as *mut u8) as *const u8;
__t0[645] = (b"TraditionalSonar\0".as_ptr() as *mut u8) as *const u8;
__t0[646] = (b"ClearVUSonar\0".as_ptr() as *mut u8) as *const u8;
__t0[647] = (b"SideVUSonar\0".as_ptr() as *mut u8) as *const u8;
__t0[648] = (b"NavInfo\0".as_ptr() as *mut u8) as *const u8;
__t0[649] = (b"BrightnessMenu\0".as_ptr() as *mut u8) as *const u8;
__t0[656] = (b"Macro1\0".as_ptr() as *mut u8) as *const u8;
__t0[657] = (b"Macro2\0".as_ptr() as *mut u8) as *const u8;
__t0[658] = (b"Macro3\0".as_ptr() as *mut u8) as *const u8;
__t0[659] = (b"Macro4\0".as_ptr() as *mut u8) as *const u8;
__t0[660] = (b"Macro5\0".as_ptr() as *mut u8) as *const u8;
__t0[661] = (b"Macro6\0".as_ptr() as *mut u8) as *const u8;
__t0[662] = (b"Macro7\0".as_ptr() as *mut u8) as *const u8;
__t0[663] = (b"Macro8\0".as_ptr() as *mut u8) as *const u8;
__t0[664] = (b"Macro9\0".as_ptr() as *mut u8) as *const u8;
__t0[665] = (b"Macro10\0".as_ptr() as *mut u8) as *const u8;
__t0[666] = (b"Macro11\0".as_ptr() as *mut u8) as *const u8;
__t0[667] = (b"Macro12\0".as_ptr() as *mut u8) as *const u8;
__t0[668] = (b"Macro13\0".as_ptr() as *mut u8) as *const u8;
__t0[669] = (b"Macro14\0".as_ptr() as *mut u8) as *const u8;
__t0[670] = (b"Macro15\0".as_ptr() as *mut u8) as *const u8;
__t0[671] = (b"Macro16\0".as_ptr() as *mut u8) as *const u8;
__t0[672] = (b"Macro17\0".as_ptr() as *mut u8) as *const u8;
__t0[673] = (b"Macro18\0".as_ptr() as *mut u8) as *const u8;
__t0[674] = (b"Macro19\0".as_ptr() as *mut u8) as *const u8;
__t0[675] = (b"Macro20\0".as_ptr() as *mut u8) as *const u8;
__t0[676] = (b"Macro21\0".as_ptr() as *mut u8) as *const u8;
__t0[677] = (b"Macro22\0".as_ptr() as *mut u8) as *const u8;
__t0[678] = (b"Macro23\0".as_ptr() as *mut u8) as *const u8;
__t0[679] = (b"Macro24\0".as_ptr() as *mut u8) as *const u8;
__t0[680] = (b"Macro25\0".as_ptr() as *mut u8) as *const u8;
__t0[681] = (b"Macro26\0".as_ptr() as *mut u8) as *const u8;
__t0[682] = (b"Macro27\0".as_ptr() as *mut u8) as *const u8;
__t0[683] = (b"Macro28\0".as_ptr() as *mut u8) as *const u8;
__t0[684] = (b"Macro29\0".as_ptr() as *mut u8) as *const u8;
__t0[685] = (b"Macro30\0".as_ptr() as *mut u8) as *const u8;
__t0[688] = (b"MacroRecordStart\0".as_ptr() as *mut u8) as *const u8;
__t0[689] = (b"MacroRecordStop\0".as_ptr() as *mut u8) as *const u8;
__t0[690] = (b"MacroPresetCycle\0".as_ptr() as *mut u8) as *const u8;
__t0[691] = (b"MacroPreset1\0".as_ptr() as *mut u8) as *const u8;
__t0[692] = (b"MacroPreset2\0".as_ptr() as *mut u8) as *const u8;
__t0[693] = (b"MacroPrest3\0".as_ptr() as *mut u8) as *const u8;
__t0[696] = (b"KbdLcdMenu1\0".as_ptr() as *mut u8) as *const u8;
__t0[697] = (b"KbdLcdMenu2\0".as_ptr() as *mut u8) as *const u8;
__t0[698] = (b"KbdLcdMenu3\0".as_ptr() as *mut u8) as *const u8;
__t0[699] = (b"KbdLcdMenu4\0".as_ptr() as *mut u8) as *const u8;
__t0[700] = (b"KbdLcdMenu5\0".as_ptr() as *mut u8) as *const u8;
__t0[704] = (b"TriggerHappy1\0".as_ptr() as *mut u8) as *const u8;
__t0[705] = (b"TriggerHappy2\0".as_ptr() as *mut u8) as *const u8;
__t0[706] = (b"TriggerHappy3\0".as_ptr() as *mut u8) as *const u8;
__t0[707] = (b"TriggerHappy4\0".as_ptr() as *mut u8) as *const u8;
__t0[708] = (b"TriggerHappy5\0".as_ptr() as *mut u8) as *const u8;
__t0[709] = (b"TriggerHappy6\0".as_ptr() as *mut u8) as *const u8;
__t0[710] = (b"TriggerHappy7\0".as_ptr() as *mut u8) as *const u8;
__t0[711] = (b"TriggerHappy8\0".as_ptr() as *mut u8) as *const u8;
__t0[712] = (b"TriggerHappy9\0".as_ptr() as *mut u8) as *const u8;
__t0[713] = (b"TriggerHappy10\0".as_ptr() as *mut u8) as *const u8;
__t0[714] = (b"TriggerHappy11\0".as_ptr() as *mut u8) as *const u8;
__t0[715] = (b"TriggerHappy12\0".as_ptr() as *mut u8) as *const u8;
__t0[716] = (b"TriggerHappy13\0".as_ptr() as *mut u8) as *const u8;
__t0[717] = (b"TriggerHappy14\0".as_ptr() as *mut u8) as *const u8;
__t0[718] = (b"TriggerHappy15\0".as_ptr() as *mut u8) as *const u8;
__t0[719] = (b"TriggerHappy16\0".as_ptr() as *mut u8) as *const u8;
__t0[720] = (b"TriggerHappy17\0".as_ptr() as *mut u8) as *const u8;
__t0[721] = (b"TriggerHappy18\0".as_ptr() as *mut u8) as *const u8;
__t0[722] = (b"TriggerHappy19\0".as_ptr() as *mut u8) as *const u8;
__t0[723] = (b"TriggerHappy20\0".as_ptr() as *mut u8) as *const u8;
__t0[724] = (b"TriggerHappy21\0".as_ptr() as *mut u8) as *const u8;
__t0[725] = (b"TriggerHappy22\0".as_ptr() as *mut u8) as *const u8;
__t0[726] = (b"TriggerHappy23\0".as_ptr() as *mut u8) as *const u8;
__t0[727] = (b"TriggerHappy24\0".as_ptr() as *mut u8) as *const u8;
__t0[728] = (b"TriggerHappy25\0".as_ptr() as *mut u8) as *const u8;
__t0[729] = (b"TriggerHappy26\0".as_ptr() as *mut u8) as *const u8;
__t0[730] = (b"TriggerHappy27\0".as_ptr() as *mut u8) as *const u8;
__t0[731] = (b"TriggerHappy28\0".as_ptr() as *mut u8) as *const u8;
__t0[732] = (b"TriggerHappy29\0".as_ptr() as *mut u8) as *const u8;
__t0[733] = (b"TriggerHappy30\0".as_ptr() as *mut u8) as *const u8;
__t0[734] = (b"TriggerHappy31\0".as_ptr() as *mut u8) as *const u8;
__t0[735] = (b"TriggerHappy32\0".as_ptr() as *mut u8) as *const u8;
__t0[736] = (b"TriggerHappy33\0".as_ptr() as *mut u8) as *const u8;
__t0[737] = (b"TriggerHappy34\0".as_ptr() as *mut u8) as *const u8;
__t0[738] = (b"TriggerHappy35\0".as_ptr() as *mut u8) as *const u8;
__t0[739] = (b"TriggerHappy36\0".as_ptr() as *mut u8) as *const u8;
__t0[740] = (b"TriggerHappy37\0".as_ptr() as *mut u8) as *const u8;
__t0[741] = (b"TriggerHappy38\0".as_ptr() as *mut u8) as *const u8;
__t0[742] = (b"TriggerHappy39\0".as_ptr() as *mut u8) as *const u8;
__t0[743] = (b"TriggerHappy40\0".as_ptr() as *mut u8) as *const u8;
__t0
});

static mut relatives: __SlateAlign16<[*const u8; 16]> = __SlateAlign16({
let mut __t0: [*const u8; 16] = unsafe { core::mem::zeroed() };
__t0[0] = (b"X\0".as_ptr() as *mut u8) as *const u8;
__t0[1] = (b"Y\0".as_ptr() as *mut u8) as *const u8;
__t0[2] = (b"Z\0".as_ptr() as *mut u8) as *const u8;
__t0[3] = (b"Rx\0".as_ptr() as *mut u8) as *const u8;
__t0[4] = (b"Ry\0".as_ptr() as *mut u8) as *const u8;
__t0[5] = (b"Rz\0".as_ptr() as *mut u8) as *const u8;
__t0[6] = (b"HWheel\0".as_ptr() as *mut u8) as *const u8;
__t0[7] = (b"Dial\0".as_ptr() as *mut u8) as *const u8;
__t0[8] = (b"Wheel\0".as_ptr() as *mut u8) as *const u8;
__t0[9] = (b"Misc\0".as_ptr() as *mut u8) as *const u8;
__t0[11] = (b"WheelHiRes\0".as_ptr() as *mut u8) as *const u8;
__t0[12] = (b"HWheelHiRes\0".as_ptr() as *mut u8) as *const u8;
__t0
});

static mut absolutes: __SlateAlign16<[*const u8; 64]> = __SlateAlign16({
let mut __t0: [*const u8; 64] = unsafe { core::mem::zeroed() };
__t0[0] = (b"X\0".as_ptr() as *mut u8) as *const u8;
__t0[1] = (b"Y\0".as_ptr() as *mut u8) as *const u8;
__t0[2] = (b"Z\0".as_ptr() as *mut u8) as *const u8;
__t0[3] = (b"Rx\0".as_ptr() as *mut u8) as *const u8;
__t0[4] = (b"Ry\0".as_ptr() as *mut u8) as *const u8;
__t0[5] = (b"Rz\0".as_ptr() as *mut u8) as *const u8;
__t0[6] = (b"Throttle\0".as_ptr() as *mut u8) as *const u8;
__t0[7] = (b"Rudder\0".as_ptr() as *mut u8) as *const u8;
__t0[8] = (b"Wheel\0".as_ptr() as *mut u8) as *const u8;
__t0[9] = (b"Gas\0".as_ptr() as *mut u8) as *const u8;
__t0[10] = (b"Brake\0".as_ptr() as *mut u8) as *const u8;
__t0[16] = (b"Hat0X\0".as_ptr() as *mut u8) as *const u8;
__t0[17] = (b"Hat0Y\0".as_ptr() as *mut u8) as *const u8;
__t0[18] = (b"Hat1X\0".as_ptr() as *mut u8) as *const u8;
__t0[19] = (b"Hat1Y\0".as_ptr() as *mut u8) as *const u8;
__t0[20] = (b"Hat2X\0".as_ptr() as *mut u8) as *const u8;
__t0[21] = (b"Hat2Y\0".as_ptr() as *mut u8) as *const u8;
__t0[22] = (b"Hat3X\0".as_ptr() as *mut u8) as *const u8;
__t0[23] = (b"Hat 3Y\0".as_ptr() as *mut u8) as *const u8;
__t0[24] = (b"Pressure\0".as_ptr() as *mut u8) as *const u8;
__t0[25] = (b"Distance\0".as_ptr() as *mut u8) as *const u8;
__t0[26] = (b"XTilt\0".as_ptr() as *mut u8) as *const u8;
__t0[27] = (b"YTilt\0".as_ptr() as *mut u8) as *const u8;
__t0[28] = (b"ToolWidth\0".as_ptr() as *mut u8) as *const u8;
__t0[32] = (b"Volume\0".as_ptr() as *mut u8) as *const u8;
__t0[33] = (b"Profile\0".as_ptr() as *mut u8) as *const u8;
__t0[34] = (b"SoundProfile\0".as_ptr() as *mut u8) as *const u8;
__t0[40] = (b"Misc\0".as_ptr() as *mut u8) as *const u8;
__t0[47] = (b"MTSlot\0".as_ptr() as *mut u8) as *const u8;
__t0[48] = (b"MTMajor\0".as_ptr() as *mut u8) as *const u8;
__t0[49] = (b"MTMinor\0".as_ptr() as *mut u8) as *const u8;
__t0[50] = (b"MTMajorW\0".as_ptr() as *mut u8) as *const u8;
__t0[51] = (b"MTMinorW\0".as_ptr() as *mut u8) as *const u8;
__t0[52] = (b"MTOrientation\0".as_ptr() as *mut u8) as *const u8;
__t0[53] = (b"MTPositionX\0".as_ptr() as *mut u8) as *const u8;
__t0[54] = (b"MTPositionY\0".as_ptr() as *mut u8) as *const u8;
__t0[55] = (b"MTToolType\0".as_ptr() as *mut u8) as *const u8;
__t0[56] = (b"MTBlobID\0".as_ptr() as *mut u8) as *const u8;
__t0[57] = (b"MTTrackingID\0".as_ptr() as *mut u8) as *const u8;
__t0[58] = (b"MTPressure\0".as_ptr() as *mut u8) as *const u8;
__t0[59] = (b"MTDistance\0".as_ptr() as *mut u8) as *const u8;
__t0[60] = (b"MTToolX\0".as_ptr() as *mut u8) as *const u8;
__t0[61] = (b"MTToolY\0".as_ptr() as *mut u8) as *const u8;
__t0
});

static mut misc: __SlateAlign16<[*const u8; 8]> = __SlateAlign16({
let mut __t0: [*const u8; 8] = unsafe { core::mem::zeroed() };
__t0[0] = (b"Serial\0".as_ptr() as *mut u8) as *const u8;
__t0[1] = (b"Pulseled\0".as_ptr() as *mut u8) as *const u8;
__t0[2] = (b"Gesture\0".as_ptr() as *mut u8) as *const u8;
__t0[3] = (b"RawData\0".as_ptr() as *mut u8) as *const u8;
__t0[4] = (b"Scan\0".as_ptr() as *mut u8) as *const u8;
__t0[5] = (b"TimeStamp\0".as_ptr() as *mut u8) as *const u8;
__t0
});

static mut leds: __SlateAlign16<[*const u8; 16]> = __SlateAlign16({
let mut __t0: [*const u8; 16] = unsafe { core::mem::zeroed() };
__t0[0] = (b"NumLock\0".as_ptr() as *mut u8) as *const u8;
__t0[1] = (b"CapsLock\0".as_ptr() as *mut u8) as *const u8;
__t0[2] = (b"ScrollLock\0".as_ptr() as *mut u8) as *const u8;
__t0[3] = (b"Compose\0".as_ptr() as *mut u8) as *const u8;
__t0[4] = (b"Kana\0".as_ptr() as *mut u8) as *const u8;
__t0[5] = (b"Sleep\0".as_ptr() as *mut u8) as *const u8;
__t0[6] = (b"Suspend\0".as_ptr() as *mut u8) as *const u8;
__t0[7] = (b"Mute\0".as_ptr() as *mut u8) as *const u8;
__t0[8] = (b"Misc\0".as_ptr() as *mut u8) as *const u8;
__t0[9] = (b"Mail\0".as_ptr() as *mut u8) as *const u8;
__t0[10] = (b"Charging\0".as_ptr() as *mut u8) as *const u8;
__t0
});

static mut repeats: __SlateAlign16<[*const u8; 2]> = __SlateAlign16([(b"Delay\0".as_ptr() as *mut u8) as *const u8, (b"Period\0".as_ptr() as *mut u8) as *const u8]);

static mut sounds: __SlateAlign16<[*const u8; 8]> = __SlateAlign16({
let mut __t0: [*const u8; 8] = unsafe { core::mem::zeroed() };
__t0[0] = (b"Click\0".as_ptr() as *mut u8) as *const u8;
__t0[1] = (b"Bell\0".as_ptr() as *mut u8) as *const u8;
__t0[2] = (b"Tone\0".as_ptr() as *mut u8) as *const u8;
__t0
});

static mut software: __SlateAlign16<[*const u8; 18]> = __SlateAlign16({
let mut __t0: [*const u8; 18] = unsafe { core::mem::zeroed() };
__t0[0] = (b"Lid\0".as_ptr() as *mut u8) as *const u8;
__t0[1] = (b"TabletMode\0".as_ptr() as *mut u8) as *const u8;
__t0[2] = (b"HeadPhoneInsert\0".as_ptr() as *mut u8) as *const u8;
__t0[3] = (b"RFKillAll\0".as_ptr() as *mut u8) as *const u8;
__t0[4] = (b"MicrophoneInsert\0".as_ptr() as *mut u8) as *const u8;
__t0[5] = (b"Dock\0".as_ptr() as *mut u8) as *const u8;
__t0[6] = (b"LineOutInsert\0".as_ptr() as *mut u8) as *const u8;
__t0[7] = (b"JackPhysicalInsert\0".as_ptr() as *mut u8) as *const u8;
__t0[8] = (b"VideoOutInsert\0".as_ptr() as *mut u8) as *const u8;
__t0[9] = (b"CameraLensCover\0".as_ptr() as *mut u8) as *const u8;
__t0[10] = (b"KeyPadSlide\0".as_ptr() as *mut u8) as *const u8;
__t0[11] = (b"FrontProximity\0".as_ptr() as *mut u8) as *const u8;
__t0[12] = (b"RotateLock\0".as_ptr() as *mut u8) as *const u8;
__t0[13] = (b"LineInInsert\0".as_ptr() as *mut u8) as *const u8;
__t0[14] = (b"MuteDevice\0".as_ptr() as *mut u8) as *const u8;
__t0[15] = (b"PenInserted\0".as_ptr() as *mut u8) as *const u8;
__t0[16] = (b"MachineCover\0".as_ptr() as *mut u8) as *const u8;
__t0
});

static mut force: __SlateAlign16<[*const u8; 128]> = __SlateAlign16({
let mut __t0: [*const u8; 128] = unsafe { core::mem::zeroed() };
__t0[80] = (b"FF_RUMBLE\0".as_ptr() as *mut u8) as *const u8;
__t0[81] = (b"FF_PERIODIC\0".as_ptr() as *mut u8) as *const u8;
__t0[82] = (b"FF_CONSTANT\0".as_ptr() as *mut u8) as *const u8;
__t0[83] = (b"FF_SPRING\0".as_ptr() as *mut u8) as *const u8;
__t0[84] = (b"FF_FRICTION\0".as_ptr() as *mut u8) as *const u8;
__t0[85] = (b"FF_DAMPER\0".as_ptr() as *mut u8) as *const u8;
__t0[86] = (b"FF_INERTIA\0".as_ptr() as *mut u8) as *const u8;
__t0[87] = (b"FF_RAMP\0".as_ptr() as *mut u8) as *const u8;
__t0[88] = (b"FF_SQUARE\0".as_ptr() as *mut u8) as *const u8;
__t0[89] = (b"FF_TRIANGLE\0".as_ptr() as *mut u8) as *const u8;
__t0[90] = (b"FF_SINE\0".as_ptr() as *mut u8) as *const u8;
__t0[91] = (b"FF_SAW_UP\0".as_ptr() as *mut u8) as *const u8;
__t0[92] = (b"FF_SAW_DOWN\0".as_ptr() as *mut u8) as *const u8;
__t0[93] = (b"FF_CUSTOM\0".as_ptr() as *mut u8) as *const u8;
__t0[96] = (b"FF_GAIN\0".as_ptr() as *mut u8) as *const u8;
__t0[97] = (b"FF_AUTOCENTER\0".as_ptr() as *mut u8) as *const u8;
__t0[127] = (b"FF_MAX\0".as_ptr() as *mut u8) as *const u8;
__t0
});

static mut force_status: __SlateAlign16<[*const u8; 2]> = __SlateAlign16([(b"FF_STATUS_STOPPED\0".as_ptr() as *mut u8) as *const u8, (b"FF_STATUS_PLAYING\0".as_ptr() as *mut u8) as *const u8]);

static mut names: __SlateAlign16<[*mut *const u8; 32]> = __SlateAlign16({
let mut __t0: [*mut *const u8; 32] = unsafe { core::mem::zeroed() };
__t0[0] = unsafe { core::ptr::addr_of_mut!(syncs.0) as *mut *const u8 };
__t0[1] = unsafe { core::ptr::addr_of_mut!(keys.0) as *mut *const u8 };
__t0[2] = unsafe { core::ptr::addr_of_mut!(relatives.0) as *mut *const u8 };
__t0[3] = unsafe { core::ptr::addr_of_mut!(absolutes.0) as *mut *const u8 };
__t0[4] = unsafe { core::ptr::addr_of_mut!(misc.0) as *mut *const u8 };
__t0[5] = unsafe { core::ptr::addr_of_mut!(software.0) as *mut *const u8 };
__t0[17] = unsafe { core::ptr::addr_of_mut!(leds.0) as *mut *const u8 };
__t0[18] = unsafe { core::ptr::addr_of_mut!(sounds.0) as *mut *const u8 };
__t0[20] = unsafe { core::ptr::addr_of_mut!(repeats.0) as *mut *const u8 };
__t0[21] = unsafe { core::ptr::addr_of_mut!(force.0) as *mut *const u8 };
__t0[23] = unsafe { core::ptr::addr_of_mut!(force_status.0) as *mut *const u8 };
__t0
});

fn hid_resolv_event(r#type: u8, code: u16, f: *mut seq_file) {
if !unsafe { events.0[(r#type as i32) as usize] }.is_null() {
unsafe { seq_printf(f, c"%s.".as_ptr() as *const u8, events.0[(r#type as i32) as usize]) };
} else {
unsafe { seq_printf(f, c"%02x.".as_ptr() as *const u8, r#type as i32) };
}
if !unsafe { names.0[(r#type as i32) as usize] }.is_null() && !unsafe { *unsafe { names.0[(r#type as i32) as usize] }.wrapping_offset((code as i32) as isize) }.is_null() {
unsafe { seq_printf(f, c"%s".as_ptr() as *const u8, *unsafe { names.0[(r#type as i32) as usize] }.wrapping_offset((code as i32) as isize)) };
} else {
unsafe { seq_printf(f, c"%04x".as_ptr() as *const u8, code as i32) };
}
}

fn hid_dump_input_mapping(hid: *mut hid_device, f: *mut seq_file) {
let mut i: i32 = 0;
let mut j: i32 = 0;
let mut k: i32 = 0;
let mut report: *mut hid_report = unsafe { core::mem::zeroed() };
while k <= 1 {
report = list_first_entry!(unsafe { core::ptr::addr_of_mut!((*((*hid).report_enum.as_mut_ptr() as *mut hid_report_enum).wrapping_offset(k as isize)).report_list) });
while list_entry_is_head!(report, unsafe { core::ptr::addr_of_mut!((*((*hid).report_enum.as_mut_ptr() as *mut hid_report_enum).wrapping_offset(k as isize)).report_list) }, list) == 0 {
i = 0;
while (i as u32) < unsafe { (*report).maxfield } {
j = 0;
while (j as u32) < unsafe { (*(*((*report).field.as_mut_ptr() as *mut *mut hid_field).wrapping_offset(i as isize))).maxusage } {
let usage: *mut hid_usage = unsafe { (*(*((*report).field.as_mut_ptr() as *mut *mut hid_field).wrapping_offset(i as isize))).usage }.wrapping_offset(j as isize);
hid_resolv_usage(unsafe { (*usage).hid }, f);
unsafe { seq_printf(f, c" ---> ".as_ptr() as *const u8) };
hid_resolv_event(unsafe { (*usage).r#type }, unsafe { (*usage).code }, f);
unsafe { seq_printf(f, c"\n".as_ptr() as *const u8) };
j = j.wrapping_add(1);
}
i = i.wrapping_add(1);
}
report = list_next_entry!(report, list);
}
k = k.wrapping_add(1);
}
}

#[unsafe(link_section = ".text.slate_distinct.hid_debug.hid_debug_rdesc_show")]
extern "C" fn hid_debug_rdesc_show(f: *mut seq_file, p: *mut ()) -> i32 {
let hdev: *mut hid_device = (unsafe { (*f).private }) as *mut hid_device;
let mut rdesc: *const u8 = unsafe { (*hdev).rdesc };
let mut rsize: u32 = unsafe { (*hdev).rsize };
// dump HID report descriptor
let mut i: i32 = 0;
if rdesc.is_null() {
rdesc = unsafe { (*hdev).dev_rdesc };
rsize = unsafe { (*hdev).dev_rsize };
}
while (i as u32) < rsize {
unsafe { seq_printf(f, c"%02x ".as_ptr() as *const u8, *rdesc.wrapping_offset(i as isize) as i32) };
i = i.wrapping_add(1);
}
unsafe { seq_printf(f, c"\n\n".as_ptr() as *const u8) };
// dump parsed data and input mappings
if (unsafe { down_interruptible(core::ptr::addr_of_mut!((*hdev).driver_input_lock)) }) != 0 {
return 0;
}
hid_dump_device(hdev, f);
unsafe { seq_printf(f, c"\n".as_ptr() as *const u8) };
hid_dump_input_mapping(hdev, f);
unsafe { up(core::ptr::addr_of_mut!((*hdev).driver_input_lock)) };
0
}

#[unsafe(link_section = ".text.slate_distinct.hid_debug.hid_debug_events_open")]
extern "C" fn hid_debug_events_open(inode: *mut inode, file: *mut file) -> i32 {
let mut __slate_compound_19: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_21: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
let mut list: core::mem::MaybeUninit<*mut hid_debug_list> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut hid_debug_list = core::ptr::addr_of_mut!(list) as *mut *mut hid_debug_list;
let mut err: i32 = unsafe { core::mem::zeroed() };
unsafe {
err = 0;
let __slate_auto_bytes_1 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
__obj_size = size_mul(core::mem::size_of::<hid_debug_list>() as u64, 1u64);
let __slate_auto_bytes_3 = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_1 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_2 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_19 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_19)
});
alloc_tag_restore!();
} else {
*__slate_slot_2 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_21 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_21)
});
}
let __v0: *mut hid_debug_list = *__slate_slot_2 as *mut hid_debug_list;
*__slate_slot_8 = __v0;
if __v0.is_null() {
err = 12i32.wrapping_neg();
} else {
err = kfifo_alloc!(core::ptr::addr_of_mut!((*(*__slate_slot_8)).hid_debug_fifo), 512i32, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32));
if err != 0 {
kfree(*__slate_slot_8 as *const ());
} else {
(*(*__slate_slot_8)).hdev = (*inode).i_private as *mut hid_device;
kref_get(core::ptr::addr_of_mut!((*(*(*__slate_slot_8)).hdev).r#ref));
(*file).private_data = *__slate_slot_8 as *mut ();
loop {
__mutex_init(core::ptr::addr_of_mut!((*(*__slate_slot_8)).read_mutex), c"&list->read_mutex".as_ptr() as *const u8, core::ptr::addr_of_mut!(__key));
if true {
break;
}
}
spin_lock_irqsave__2!(core::ptr::addr_of_mut!((*(*(*__slate_slot_8)).hdev).debug_list_lock), *__slate_slot_7);
list_add_tail(core::ptr::addr_of_mut!((*(*__slate_slot_8)).node), core::ptr::addr_of_mut!((*(*(*__slate_slot_8)).hdev).debug_list));
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*(*(*__slate_slot_8)).hdev).debug_list_lock), *__slate_slot_7);
}
}
return err;
}
}

static mut __key: lock_class_key = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".text.slate_distinct.hid_debug.hid_debug_events_read")]
extern "C" fn hid_debug_events_read(file: *mut file, buffer: *mut u8, count: u64, ppos: *mut i64) -> i64 {
let mut __ret: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut u32 = core::ptr::addr_of_mut!(__ret) as *mut u32;
let mut __ai_ptr: core::mem::MaybeUninit<*mut u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut u32 = core::ptr::addr_of_mut!(__ai_ptr) as *mut *mut u32;
let mut __ret: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u32 = core::ptr::addr_of_mut!(__ret) as *mut u32;
let mut __ai_ptr: core::mem::MaybeUninit<*mut u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut u32 = core::ptr::addr_of_mut!(__ai_ptr) as *mut *mut u32;
let mut wait: core::mem::MaybeUninit<wait_queue_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut wait_queue_entry = core::ptr::addr_of_mut!(wait) as *mut wait_queue_entry;
let mut copied: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut i32 = core::ptr::addr_of_mut!(copied) as *mut i32;
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut list: *mut hid_debug_list = unsafe { core::mem::zeroed() };
unsafe {
'__join_0: {
list = (*file).private_data as *mut hid_debug_list;
ret = 0;
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
core::ptr::write(__slate_slot_8, {
let mut __t2: wait_queue_entry = core::mem::zeroed();
__t2.private = get_current() as *mut ();
__t2.func = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut wait_queue_entry, u32, i32, *mut ()) -> i32>>(default_wake_function as *const ()) };
__t2.entry = list_head { next: core::ptr::null_mut::<list_head>(), prev: core::ptr::null_mut::<list_head>() };
__t2
});
mutex_lock(core::ptr::addr_of_mut!((*list).read_mutex));
if kfifo_is_empty!(core::ptr::addr_of_mut!((*list).hid_debug_fifo)) {
add_wait_queue(core::ptr::addr_of_mut!((*(*list).hdev).debug_wait), core::ptr::addr_of_mut!(*__slate_slot_8));
loop {
debug_normal_state_change!();
trace_set_current_state!(1i32);
loop {
kcsan_mb!();
loop {
core::ptr::write(__slate_slot_6, core::ptr::addr_of_mut!((*get_current()).__state));
kcsan_mb!();
instrument_atomic_read_write(*__slate_slot_6 as *const (), core::mem::size_of::<u32>() as u64);
core::ptr::write(__slate_slot_4, 1u32);
let __t1: u32;
core::arch::asm!("xchgl {0:e}, ({1:r})", inlateout(reg) *__slate_slot_4 as u32 => __t1, in(reg) *__slate_slot_6, options(nostack, att_syntax));
*__slate_slot_4 = unsafe { core::mem::transmute::<u32, u32>(__t1 as u32) };
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
'__join_7: {
'__join_23: {
'__join_21: {
'__loop_8: loop {
if kfifo_is_empty!(core::ptr::addr_of_mut!((*list).hid_debug_fifo)) {
if signal_pending(get_current()) != 0 {
break '__join_23;
} else {
// if list->hdev is NULL we cannot remove_wait_queue().
// if list->hdev->debug is 0 then hid_debug_unregister()
// was already called and list->hdev is being destroyed.
// if we add remove_wait_queue() here we can hit a race.
if unsafe { (*list).hdev }.is_null() || (*(*list).hdev).debug == 0 {
break '__join_21;
} else {
if (*file).f_flags & 2048u32 != 0 {
break;
} else {
// allow O_NONBLOCK from other threads
mutex_unlock(core::ptr::addr_of_mut!((*list).read_mutex));
schedule();
mutex_lock(core::ptr::addr_of_mut!((*list).read_mutex));
loop {
debug_normal_state_change!();
trace_set_current_state__3!(1i32);
loop {
kcsan_mb!();
loop {
core::ptr::write(__slate_slot_2, core::ptr::addr_of_mut!((*get_current()).__state));
kcsan_mb!();
instrument_atomic_read_write(*__slate_slot_2 as *const (), core::mem::size_of::<u32>() as u64);
core::ptr::write(__slate_slot_0, 1u32);
let __t0: u32;
core::arch::asm!("xchgl {0:e}, ({1:r})", inlateout(reg) *__slate_slot_0 as u32 => __t0, in(reg) *__slate_slot_2, options(nostack, att_syntax));
*__slate_slot_0 = unsafe { core::mem::transmute::<u32, u32>(__t0 as u32) };
if true {
break;
}
}
if true {
break;
}
}
if true {
continue '__loop_8;
}
}
}
}
}
} else {
break '__join_7;
}
}
ret = 11i32.wrapping_neg();
break '__join_7;
}
ret = 5i32.wrapping_neg();
loop {
debug_normal_state_change!();
trace_set_current_state__2!(0i32);
loop {
if true {
__WRITE_ONCE__5!((*get_current()).__state, 0i32);
if true {
break;
}
}
}
if true {
break '__join_0;
}
}
}
ret = 512i32.wrapping_neg();
}
loop {
debug_normal_state_change!();
trace_set_current_state__4!(0i32);
loop {
if true {
__WRITE_ONCE__5!((*get_current()).__state, 0i32);
if true {
break;
}
}
}
if true {
break;
}
}
remove_wait_queue(core::ptr::addr_of_mut!((*(*list).hdev).debug_wait), core::ptr::addr_of_mut!(*__slate_slot_8));
if ret != 0 {
break '__join_0;
}
}
// pass the fifo content to userspace, locking is not needed with only
// one concurrent reader and one concurrent writer
ret = kfifo_to_user!(core::ptr::addr_of_mut!((*list).hid_debug_fifo), buffer, count, core::ptr::addr_of_mut!(*__slate_slot_9));
if ret == 0 {
ret = *__slate_slot_9;
}
}
mutex_unlock(core::ptr::addr_of_mut!((*list).read_mutex));
return ret as i64;
}
}

#[unsafe(link_section = ".text.slate_distinct.hid_debug.hid_debug_events_poll")]
extern "C" fn hid_debug_events_poll(file: *mut file, wait: *mut poll_table_struct) -> u32 {
let list: *mut hid_debug_list = (unsafe { (*file).private_data }) as *mut hid_debug_list;
poll_wait(file, unsafe { core::ptr::addr_of_mut!((*(*list).hdev).debug_wait) }, wait);
if !kfifo_is_empty!(unsafe { core::ptr::addr_of_mut!((*list).hid_debug_fifo) }) {
return 1 | 64;
}
if (unsafe { (*(*list).hdev).debug }) == 0 {
return 8 | 16;
}
0
}

#[unsafe(link_section = ".text.slate_distinct.hid_debug.hid_debug_events_release")]
extern "C" fn hid_debug_events_release(inode: *mut inode, file: *mut file) -> i32 {
let list: *mut hid_debug_list = (unsafe { (*file).private_data }) as *mut hid_debug_list;
let mut flags: u64 = 0;
spin_lock_irqsave__2!(unsafe { core::ptr::addr_of_mut!((*(*list).hdev).debug_list_lock) }, flags);
list_del(unsafe { core::ptr::addr_of_mut!((*list).node) });
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*(*list).hdev).debug_list_lock) }, flags);
kfifo_free!(unsafe { core::ptr::addr_of_mut!((*list).hid_debug_fifo) });
kref_put(unsafe { core::ptr::addr_of_mut!((*(*list).hdev).r#ref) }, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut kref)>>(hiddev_free as *const ()) });
unsafe { kfree(list as *const ()) };
0
}

#[unsafe(link_section = ".text.slate_distinct.hid_debug.hid_debug_rdesc_open")]
extern "C" fn hid_debug_rdesc_open(inode: *mut inode, file: *mut file) -> i32 {
unsafe { single_open(file, Some(hid_debug_rdesc_show), (*inode).i_private) }
}

static hid_debug_rdesc_fops: __SlateConst<file_operations> = __SlateConst({
let mut __t0: file_operations = unsafe { core::mem::zeroed() };
__t0.owner = core::ptr::null_mut::<module>();
__t0.llseek = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut file, i64, i32) -> i64>>(seq_lseek as *const ()) };
__t0.read = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut file, *mut u8, u64, *mut i64) -> i64>>(seq_read as *const ()) };
__t0.open = Some(hid_debug_rdesc_open);
__t0.release = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut inode, *mut file) -> i32>>(single_release as *const ()) };
__t0
});

static hid_debug_events_fops: __SlateConst<file_operations> = __SlateConst({
let mut __t0: file_operations = unsafe { core::mem::zeroed() };
__t0.owner = core::ptr::null_mut::<module>();
__t0.llseek = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut file, i64, i32) -> i64>>(noop_llseek as *const ()) };
__t0.read = Some(hid_debug_events_read);
__t0.poll = Some(hid_debug_events_poll);
__t0.open = Some(hid_debug_events_open);
__t0.release = Some(hid_debug_events_release);
__t0
});

#[unsafe(no_mangle)]
extern "C" fn hid_debug_register(hdev: *mut hid_device, name: *const u8) {
unsafe {
(*hdev).debug_dir = debugfs_create_dir(name, hid_debug_root);
}
unsafe {
(*hdev).debug_rdesc = debugfs_create_file!(c"rdesc".as_ptr() as *mut u8, 256i32, unsafe { (*hdev).debug_dir }, hdev, core::ptr::addr_of!(hid_debug_rdesc_fops.0));
}
unsafe {
(*hdev).debug_events = debugfs_create_file!(c"events".as_ptr() as *mut u8, 256i32, unsafe { (*hdev).debug_dir }, hdev, core::ptr::addr_of!(hid_debug_events_fops.0));
}
unsafe {
(*hdev).debug = 1;
}
}

#[unsafe(no_mangle)]
extern "C" fn hid_debug_unregister(hdev: *mut hid_device) {
unsafe {
(*hdev).debug = 0;
}
wake_up_interruptible!(unsafe { core::ptr::addr_of_mut!((*hdev).debug_wait) });
unsafe { debugfs_remove((*hdev).debug_rdesc) };
unsafe { debugfs_remove((*hdev).debug_events) };
unsafe { debugfs_remove((*hdev).debug_dir) };
}

#[unsafe(no_mangle)]
extern "C" fn hid_debug_init() {
unsafe {
hid_debug_root = debugfs_create_dir(c"hid".as_ptr() as *const u8, core::ptr::null_mut::<dentry>());
}
}

#[unsafe(no_mangle)]
extern "C" fn hid_debug_exit() {
unsafe { debugfs_remove(hid_debug_root) };
}

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

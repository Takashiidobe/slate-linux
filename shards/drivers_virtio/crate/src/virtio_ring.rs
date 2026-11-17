
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
//! Virtio ring implementation.
//!
//! Copyright 2007 Rusty Russell IBM Corporation
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
//! This counts to 15. Any more, it will return 16th argument.
//! Optimization barrier
//!
//! The "volatile" is due to gcc bugs
//! Yes, this permits 64-bit accesses on 32-bit architectures. These will
//! actually be atomic in some cases (namely Armv7 + LPAE), but for others we
//! rely on the access being split into 2x32-bit accesses for a 32-bit quantity
//! (e.g. a virtual address) and a strong prevailing wind.
//! Operation not permitted
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
//! Begin/end of an instrumentation safe region
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
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
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
//! Check at compile time that something is of a particular type.
//! Always evaluates to 1 so you may use it easily in comparisons.
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
//! label
//! Defines x86 CPU feature bits
//! 64bit FS base
//! Bit size and mask of CPU number stored in the per CPU data (and TSC_AUX)
//! PAGE_SHIFT determines the page size.
//!
//! Note: This definition is required because PAGE_SHIFT is used
//! in several places throughout the codebase.
//! writeable
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
//! Barriers for virtual machine guests when talking to an SMP host
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
//! Must be a compile-time constant, so implement it as a macro
//! Resource type
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
//! return the ->next pointer of a list_head in an rcu safe
//! way, we must not access it directly
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
//! Keep unconverted code working
//! used to be O_SYNC, see below
//! Valid hint values for F_{GET,SET}_RW_HINT. 0 is "not set", or can be
//! used to clear any hints previously set.
//! Gives us 8 prio classes with 13-bits of data for each class
//! per-IO O_SYNC
//! File supports atomic writes
//! The IDR API does not expose the tagging functionality of the radix tree
//! to users.  Use tag 0 to track whether a node has free space below it.
//! number of env pointers
//! See the comment in asm-generic/io.h about ioremap_np().
//! Generate SD flag indexes
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
//! Generic info of form tag = "info"
//! DMA_ATTR_WRITE_COMBINE: Specifies that writes to the mapping may be
//! buffered to improve performance.
//! This marks a buffer as continuing via the next field.
//! Driver has used its parts of the config, and is happy
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
// DECLARE_FLEX_ARRAY() - Declare a flexible array usable in a union
//
// @TYPE: The type of each flexible array element
// @NAME: The name of the flexible array member
//
// In order to have a flexible array member in a union or alone in a
// struct, it needs to be wrapped in an anonymous struct with at least 1
// named member, but that member can be empty.
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
// No such file or directory
// I/O error
// No such device or address
// Argument list too long
// Try again
// Out of memory
// Bad address
// Device or resource busy
// Invalid argument
// Text file busy
// No space left on device
// Operation not supported on transport endpoint
// Memory page has hardware error
// IS_ERR_VALUE - Detect an error pointer.
// @x: The pointer to check.
//
// Like IS_ERR(), but does not generate a compiler warning if result is unused.
// Operation is not supported
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
// Dummy printk for disabled debugging statements to use whilst maintaining
// gcc's format checking.
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
// min - return minimum of two values of the same or compatible types
// @x: first value
// @y: second value
// max - return maximum of two values of the same or compatible types
// @x: first value
// @y: second value
// Use these carefully: no type checking, and uses the arguments
// multiple times. Use for obvious constants only.
// container_of_const - cast a member of a structure out to the containing
// 			structure and preserve the const-ness of the pointer
// @ptr:		the pointer to the member
// @type:		the type of the container struct this is embedded in.
// @member:		the name of the member within the struct.
//
// Always prefer container_of_const() instead of container_of() in new code.
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
// Currently stuck as a macro due to indirect forward reference to
// linux/mmzone.h's __section_mem_map_addr() definition:
// All top-level MITIGATION_PAGE_TABLE_ISOLATION page tables are order-1 pages
// (8k-aligned and 8k in size).  The kernel one is at the beginning 4k and
// the user one is in the last 4k.  To switch between them, you
// just need to flip the 12th bit in their addresses.
// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
// Resource extended types
// No address assigned yet
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
// Set the IDR flag and the IDR_FREE tag
// Need to take variadic arguments even though we don't use them, as dev_fmt()
// may only just have been expanded and may result in multiple arguments.
// Generate SD flag bits
// Implementation specific max size
// Virtio feature bits VIRTIO_TRANSPORT_F_START through
// VIRTIO_TRANSPORT_F_END are reserved for the transport
// being used (e.g. virtio_ring, virtio_pci etc.), the
// rest are per-device feature bits.
// v1.0 compliant.
// If clear - device has the platform DMA (e.g. IOMMU) bypass quirk feature.
// If set - use platform DMA tools to access the memory.
//
// Note the reverse polarity (compared to most other features),
// this is for compatibility with legacy systems.
// This feature indicates support for the packed virtqueue layout.
// Inorder feature indicates that all buffers are used by the device
// in the same order in which they have been made available.
// This feature indicates that memory accesses by the driver and the
// device are ordered in a way described by the platform.
// This feature indicates that the driver passes extra data (besides
// identifying the virtqueue) in its device notifications.
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
/// @a is a power of 2 value
macro_rules! ALIGN {
($x:expr, $a:expr) => {
__ALIGN_KERNEL!($x, $a)
};
}

macro_rules! BIT_MASK {
($nr:expr) => {
UL!() << ($nr) % 64
};
}

macro_rules! BIT_ULL {
($nr:expr) => {
ULL!() << $nr
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

/// to align the pointer to the (next) page boundary
macro_rules! PAGE_ALIGN {
($addr:expr) => {
ALIGN!($addr, 4096u64)
};
}

macro_rules! PASS_TOKEN_PARAMS {
($_size:expr) => {
$_size
};
}

macro_rules! PFN_PHYS {
($x:expr) => {
(($x) as u64) << 12i32
};
}

macro_rules! ULL {
() => {
_ULL!()
};
}

macro_rules! UL {
() => {
_UL!()
};
}

macro_rules! VIRTIO_BIT {
($b:expr) => {
BIT_ULL!(($b) & 63)
};
}

macro_rules! VIRTIO_U64 {
($b:expr) => {
($b) >> 6i32
};
}

macro_rules! _AC {
() => {
1u64
};
}

macro_rules! _AC__3 {
() => {
18446744071562067968u64
};
}

macro_rules! _AC__2 {
() => {
16045481047390945280u64
};
}

macro_rules! _ULL {
() => {
_AC!()
};
}

macro_rules! _UL {
() => {
_AC!()
};
}

macro_rules! __ALIGN_KERNEL_MASK {
($x:expr, $mask:expr) => {
$x.wrapping_add($mask) & !$mask
};
}

macro_rules! __ALIGN_KERNEL {
($x:expr, $a:expr) => {
__ALIGN_KERNEL_MASK!($x, (($a) as u64).wrapping_sub(1))
};
}

macro_rules! __WRITE_ONCE__3 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut bool, $val) };
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
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut u16, $val) };
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

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__2 {
() => {
((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__5 {
() => {
((const { core::mem::size_of::<bool>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<bool>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<bool>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<bool>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__3 {
() => {
((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__6 {
() => {
((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word {
() => {
((const { core::mem::size_of::<u16>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u16>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u16>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u16>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__4 {
() => {
((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

macro_rules! __or {
() => {
____or!()
};
}

macro_rules! __pa {
($x:expr) => {
__phys_addr!(($x) as u64)
};
}

macro_rules! __phys_addr {
($x:expr) => {
__phys_addr_nodebug($x)
};
}

macro_rules! __swab16 {
($x:expr) => {
unsafe { __slate_intrinsic_1a0c079c71348b00(($x) as u16) }
};
}

macro_rules! __swab32 {
($x:expr) => {
unsafe { __slate_intrinsic_e94e9cf815168b51(($x) as u32) }
};
}

macro_rules! __swab64 {
($x:expr) => {
unsafe { __slate_intrinsic_5c27152fac4a123f(($x) as u64) }
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
let mut __v: *mut virtqueue;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__3!() as isize)) }) as *mut ()) as *mut virtqueue;
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

macro_rules! kcsan_mb {
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

macro_rules! kcsan_wmb {
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
macro_rules! list_entry {
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
macro_rules! list_first_entry {
($ptr:expr) => {
list_entry!(unsafe { (*$ptr).next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry {
($pos:expr, $member:ident) => {
list_entry!(unsafe { (*$pos).$member.next })
};
}

macro_rules! offset_in_page {
($p:expr) => {
(($p) as u64) & !(!(_AC!() << 12i32).wrapping_sub(1))
};
}

macro_rules! offsetof__3 {
() => {
core::mem::offset_of!(virtqueue, list) as u64
};
}

macro_rules! offsetof__4 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, vq) as u64
};
}

macro_rules! page_to_phys {
($page:expr) => {
PFN_PHYS!(((unsafe { $page.offset_from(vmemmap_base as *mut page) }) as i64) as u64)
};
}

macro_rules! raw_spin_lock {
($lock:expr) => {
unsafe { _raw_spin_lock($lock) };
};
}

macro_rules! raw_spin_unlock {
($lock:expr) => {
unsafe { _raw_spin_unlock($lock) };
};
}

/// These macros should be used after a dma_map_sg call has been done
/// to get bus addresses of each of the SG entries and their lengths.
/// You should only work with the number of sg entries dma_map_sg
/// returns, or alternatively stop on the first sg_dma_len(sg) which
/// is 0.
macro_rules! sg_dma_address {
($sg:expr) => {
unsafe { (*$sg).dma_address }
};
}

macro_rules! sg_dma_len {
($sg:expr) => {
unsafe { (*$sg).dma_length }
};
}

macro_rules! test_bit__3 {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if true && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*$addr) } {
const_test_bit(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit(($nr) as u64, ($addr) as *const u64) as i32
};
__v
}
};
}

macro_rules! to_vvq__2 {
($_vq:expr) => {
{
let mut __v: *mut vring_virtqueue;
__v = ((unsafe { ((($_vq) as *mut ()) as *const u8).wrapping_offset(-(offsetof__4!(vring_virtqueue) as isize)) }) as *mut ()) as *mut vring_virtqueue;
__v as *const vring_virtqueue
}
};
}

macro_rules! to_vvq {
($_vq:expr) => {
{
let mut __v: *mut vring_virtqueue;
__v = ((unsafe { ((($_vq) as *mut ()) as *const u8).wrapping_offset(-(offsetof__4!(vring_virtqueue) as isize)) }) as *mut ()) as *mut vring_virtqueue;
__v
}
};
}

macro_rules! to_vvq__3 {
($_vq:expr) => {
{
let mut __v: *mut vring_virtqueue;
__v = ((unsafe { (($_vq) as *const u8).wrapping_offset(-(offsetof__4!(vring_virtqueue) as isize)) }) as *mut ()) as *mut vring_virtqueue;
__v
}
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

/// virt_to_page(kaddr) returns a valid pointer if and only if
/// virt_addr_valid(kaddr) returns true.
macro_rules! virt_to_page {
($kaddr:expr) => {
((unsafe { vmemmap_base }) as *mut page).wrapping_offset((__pa!($kaddr) >> 12i32) as isize)
};
}

macro_rules! xen_domain {
() => {
false
};
}

unsafe extern "C" {
    static mut __SCK__WARN_trap: static_call_key;
    static mut kmsg_fops: file_operations;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut phys_base: u64;
    static mut page_offset_base: u64;
    static mut vmemmap_base: u64;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn _printk(arg0: *const u8, ...) -> i32;
    fn __bad_size_call_parameter();
    fn __xchg_wrong_size();
    fn __compiletime_assert_18();
    fn __compiletime_assert_21();
    fn __compiletime_assert_57();
    fn _raw_spin_lock(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock(arg0: *mut raw_spinlock);
    fn alloc_pages_exact_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn free_pages_exact(arg0: *mut (), arg1: u64);
    fn kfree(arg0: *const ());
    fn __compiletime_assert_257();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn cache_get_priv_group(arg0: *mut cacheinfo) -> *const attribute_group;
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_376();
    fn __compiletime_assert_419();
    fn is_vmalloc_addr(arg0: *const ()) -> bool;
    fn _dev_err(arg0: *const device, arg1: *const u8, ...);
    fn _dev_warn(arg0: *const device, arg1: *const u8, ...);
    fn dev_driver_string(arg0: *const device) -> *const u8;
    fn dma_map_page_attrs(arg0: *mut device, arg1: *mut page, arg2: u64, arg3: u64, arg4: u32, arg5: u64) -> u64;
    fn dma_unmap_page_attrs(arg0: *mut device, arg1: u64, arg2: u64, arg3: u32, arg4: u64);
    fn dma_alloc_attrs(arg0: *mut device, arg1: u64, arg2: *mut u64, arg3: u32, arg4: u64) -> *mut ();
    fn dma_free_attrs(arg0: *mut device, arg1: u64, arg2: *mut (), arg3: u64, arg4: u64);
    fn dma_max_mapping_size(arg0: *mut device) -> u64;
    fn __dma_sync_single_for_cpu(arg0: *mut device, arg1: u64, arg2: u64, arg3: u32);
    fn __dma_sync_single_for_device(arg0: *mut device, arg1: u64, arg2: u64, arg3: u32);
    fn __dma_need_sync(arg0: *mut device, arg1: u64) -> bool;
    fn __compiletime_assert_511();
    fn virtio_check_driver_offered_feature(arg0: *const virtio_device, arg1: u32);
    fn __compiletime_assert_517();
    fn __compiletime_assert_518();
    fn __compiletime_assert_519();
    fn __compiletime_assert_520();
    fn __compiletime_assert_521();
    fn __compiletime_assert_522();
    fn __compiletime_assert_523();
    fn __compiletime_assert_534();
    fn __compiletime_assert_535();
    fn __compiletime_assert_536();
    fn __compiletime_assert_550();
    fn __compiletime_assert_551();
    fn __compiletime_assert_552();
    fn __compiletime_assert_553();
    fn __compiletime_assert_554();
    fn __compiletime_assert_555();
    fn __compiletime_assert_556();
    fn __compiletime_assert_598();
    fn __compiletime_assert_600();
    fn __compiletime_assert_602();
    fn __compiletime_assert_604();
    fn __compiletime_assert_606();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.bswap.i16"]
    fn __slate_intrinsic_1a0c079c71348b00(arg0: u16) -> u16;
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_1c5364fcd9da51fe(arg0: u32) -> bool;
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
    #[link_name = "llvm.bswap.i64"]
    fn __slate_intrinsic_5c27152fac4a123f(arg0: u64) -> u64;
    #[link_name = "llvm.bswap.i32"]
    fn __slate_intrinsic_e94e9cf815168b51(arg0: u32) -> u32;
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


#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_mod {
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
struct jump_entry {
    code: i32,
    target: i32,
    key: i64,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_site {
    addr: i32,
    key: i32,
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


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct orc_entry {
    sp_offset: i16,
    bp_offset: i16,
    __slate_bits_0: __slate_bits::__SlateBits103U0,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits104U0,
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
// 4th level page in 5-level paging case
// 3rd level page
// PMD_SHIFT determines the size of the area a middle-level
// page table can map
// entries per page directory level
// We borrow bit 3 to remember PG_anon_exclusive.
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
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_types_h_368 {
    pud: u64,
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

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
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

#[repr(C)]
#[derive(Clone, Copy)]
struct io_bitmap {
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

#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
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
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_nodemask_types_h_17 {
    bits: [u64; 1],
}


#[inline(always)]
fn is_power_of_2(n: u64) -> bool {
n.wrapping_sub(1) < n ^ n.wrapping_sub(1)
}

// ilog2 - log base 2 of 32-bit or a 64-bit unsigned value
// @n: parameter
//
// constant-capable log of base 2 calculation
// - this can be used to initialise global variables from constant data, hence
// the massive ternary operator construction
//
// selects the appropriately-sized optimised version depending on sizeof(n)
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
fn __list_add_valid(new: *mut list_head, prev: *mut list_head, next: *mut list_head) -> bool {
true
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pollfd {
}


fn __list_del_entry_valid(entry: *mut list_head) -> bool {
true
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
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_18() };
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

fn list_add_tail(new: *mut list_head, head: *mut list_head) {
__list_add(new, unsafe { (*head).prev }, head);
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
}


fn __list_del(prev: *mut list_head, next: *mut list_head) {
unsafe {
(*next).prev = prev;
}
loop {
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_21() };
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
struct mutex_waiter {
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

#[inline(always)]
fn __phys_addr_nodebug(mut x: u64) -> u64 {
let y: u64 = x.wrapping_sub(_AC__3!());
// use the carry flag to determine if x was < __START_KERNEL_map
x = y.wrapping_add(if x > y { unsafe { phys_base } } else { _AC__3!().wrapping_sub(unsafe { page_offset_base }) });
x
}

#[repr(C)]
#[derive(Clone, Copy)]
struct range {
    start: u64,
    end: u64,
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
    __slate_bits_0: __slate_bits::__SlateBits122U0,
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
struct futex_pi_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct robust_list_head {
}


// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
#[inline(always)]
fn spin_lock(lock: *mut spinlock) {
raw_spin_lock!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[inline(always)]
fn spin_unlock(lock: *mut spinlock) {
raw_spin_unlock!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[repr(C)]
#[derive(Clone, Copy)]
struct futex_private_hash {
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
struct hrtimer_clock_base {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct seccomp_filter {
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
struct rb_root_cached {
    rb_root: rb_root,
    rb_leftmost: *mut rb_node,
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
struct workqueue_struct {
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
struct rseq {
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
struct seccomp {
    mode: i32,
    filter_count: __SlateRecord_types_h_190,
    filter: *mut seccomp_filter,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
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
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
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
    __slate_bits_0: __slate_bits::__SlateBits280U0,
    dl_timer: hrtimer,
    inactive_timer: hrtimer,
    rq: *mut rq,
    server_pick_task: Option<unsafe extern "C" fn(*mut sched_dl_entity, *mut rq_flags) -> *mut task_struct>,
    pi_se: *mut sched_dl_entity,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
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
struct uprobe {
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
    __slate_bits_0: __slate_bits::__SlateBits72U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits72U1,
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
    __slate_bits_2: __slate_bits::__SlateBits72U2,
    mce_kill_me: callback_head,
    mce_count: i32,
    kretprobe_instances: llist_head,
    rethooks: llist_head,
    l1d_flush_kill: callback_head,
    unwind_info: unwind_task_info,
    thread: thread_struct,
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


fn mem_alloc_profiling_enabled() -> bool {
false
}

#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
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
struct ldt_struct {
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
struct mem_cgroup {
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
struct srcu_struct {
    srcu_ctrp: *mut srcu_ctr,
    sda: *mut srcu_data,
    srcu_reader_flavor: u8,
    dep_map: lockdep_map,
    srcu_sup: *mut srcu_usage,
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
struct percpu_counter {
    lock: raw_spinlock,
    count: i64,
    list: list_head,
    counters: *mut i32,
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
struct device_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain {
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
struct kmem_cache {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_userfaultfd_ctx {
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
    __slate_bits_0: __slate_bits::__SlateBits398U0,
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


// Opaque type representing current mm_struct flag state. Must be accessed via
// mm_flags_xxx() helper functions.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_1171 {
    __mm_flags: [u64; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct user_namespace {
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
struct percpu_ref_data {
    count: __SlateRecord_types_h_197,
    release: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    confirm_switch: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    __slate_bits_0: __slate_bits::__SlateBits429U0,
    rcu: callback_head,
    r#ref: *mut percpu_ref,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mnt_idmap {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_ref {
    percpu_count_ptr: u64,
    data: *mut percpu_ref_data,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array_ptr {
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
struct key_type {
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
struct __SlateRecord_slab_h_532 {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_cred_h_147 {
    non_rcu: i32,
    rcu: callback_head,
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

#[repr(C)]
#[derive(Clone, Copy)]
struct pidfs_attr {
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
unsafe { __compiletime_assert_257() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
} else {
loop {
unsafe {
core::arch::asm!("258: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 258b - ., 3; .popsection\n# {0}", const 258 as i32, options(nostack, att_syntax));
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

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pid_h_62 {
    ino: u64,
    pidfs_hash: rhash_head,
    stashed: *mut dentry,
    attr: *mut pidfs_attr,
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
// kmalloc_objs - Allocate an array of the given type
// @VAR_OR_TYPE: Variable or type to allocate an array of.
// @COUNT: How many elements in the array.
// @...: optional GFP flags for the allocation (GFP_KERNEL when not specified).
//
// Returns: newly allocated pointer to array of @VAR_OR_TYPE on success,
// or NULL on failure.
#[repr(C)]
#[derive(Clone, Copy)]
struct cacheinfo {
    id: u32,
    r#type: u32,
    level: u32,
    coherency_line_size: u32,
    number_of_sets: u32,
    ways_of_associativity: u32,
    physical_line_partition: u32,
    size: u32,
    shared_cpu_map: cpumask,
    attributes: u32,
    fw_token: *mut (),
    disable_sysfs: bool,
    r#priv: *mut (),
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

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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
    __slate_bits_0: __slate_bits::__SlateBits283U0,
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
if !(__native_word__4!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_376() };
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
struct backing_dev_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct block_device {
}


// Bits set in the VMA until the stack is in its final location
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
struct xarray {
    xa_lock: spinlock,
    xa_flags: u32,
    xa_head: *mut (),
}


// We encode errnos in the xas->xa_node.  If an error has happened, we need to
// drop the lock to fix it, and once we've done so the xa_state is invalid.
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
struct list_lru {
    node: *mut list_lru_node,
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
struct hlist_bl_node {
    next: *mut hlist_bl_node,
    pprev: *mut *mut hlist_bl_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_context {
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
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
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
struct poll_table_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_info_struct {
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
struct posix_acl {
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
struct qc_state {
    s_incoredqs: u32,
    s_state: [qc_type_state; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_cmd {
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
struct kernfs_root {
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
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
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
struct sock {
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
if !(__native_word__3!() || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_419() };
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
struct wake_irq {
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
struct dev_pm_qos {
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
struct folio_queue {
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
struct kernfs_elem_dir {
    subdirs: u64,
    children: rb_root,
    root: *mut kernfs_root,
    rev: u64,
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
    __slate_bits_0: __slate_bits::__SlateBits614U0,
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
struct module_param_attrs {
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
    __slate_bits_0: __slate_bits::__SlateBits566U0,
}


fn kobject_name(kobj: *const kobject) -> *const u8 {
unsafe { (*kobj).name }
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


fn virt_to_phys(address: *mut ()) -> u64 {
__pa!(address)
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
struct fwnode_handle {
    secondary: *mut fwnode_handle,
    ops: *const fwnode_operations,
    dev: *mut device,
    suppliers: list_head,
    consumers: list_head,
    flags: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fwnode_endpoint {
    port: u32,
    id: u32,
    local_fwnode: *const fwnode_handle,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fwnode_reference_args {
    fwnode: *mut fwnode_handle,
    nargs: u32,
    args: [u64; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vduse_vq_group {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fwnode_operations {
    get: Option<unsafe extern "C" fn(*mut fwnode_handle) -> *mut fwnode_handle>,
    put: Option<unsafe extern "C" fn(*mut fwnode_handle)>,
    device_is_available: Option<unsafe extern "C" fn(*const fwnode_handle) -> bool>,
    device_get_match_data: Option<unsafe extern "C" fn(*const fwnode_handle, *const device) -> *const ()>,
    device_dma_supported: Option<unsafe extern "C" fn(*const fwnode_handle) -> bool>,
    device_get_dma_attr: Option<unsafe extern "C" fn(*const fwnode_handle) -> u32>,
    property_present: Option<unsafe extern "C" fn(*const fwnode_handle, *const u8) -> bool>,
    property_read_bool: Option<unsafe extern "C" fn(*const fwnode_handle, *const u8) -> bool>,
    property_read_int_array: Option<unsafe extern "C" fn(*const fwnode_handle, *const u8, u32, *mut (), u64) -> i32>,
    property_read_string_array: Option<unsafe extern "C" fn(*const fwnode_handle, *const u8, *mut *const u8, u64) -> i32>,
    get_name: Option<unsafe extern "C" fn(*const fwnode_handle) -> *const u8>,
    get_name_prefix: Option<unsafe extern "C" fn(*const fwnode_handle) -> *const u8>,
    get_parent: Option<unsafe extern "C" fn(*const fwnode_handle) -> *mut fwnode_handle>,
    get_next_child_node: Option<unsafe extern "C" fn(*const fwnode_handle, *mut fwnode_handle) -> *mut fwnode_handle>,
    get_named_child_node: Option<unsafe extern "C" fn(*const fwnode_handle, *const u8) -> *mut fwnode_handle>,
    get_reference_args: Option<unsafe extern "C" fn(*const fwnode_handle, *const u8, *const u8, u32, u32, *mut fwnode_reference_args) -> i32>,
    graph_get_next_endpoint: Option<unsafe extern "C" fn(*const fwnode_handle, *mut fwnode_handle) -> *mut fwnode_handle>,
    graph_get_remote_endpoint: Option<unsafe extern "C" fn(*const fwnode_handle) -> *mut fwnode_handle>,
    graph_get_port_parent: Option<unsafe extern "C" fn(*mut fwnode_handle) -> *mut fwnode_handle>,
    graph_parse_endpoint: Option<unsafe extern "C" fn(*const fwnode_handle, *mut fwnode_endpoint) -> i32>,
    iomap: Option<unsafe extern "C" fn(*mut fwnode_handle, i32) -> *mut ()>,
    irq_get: Option<unsafe extern "C" fn(*const fwnode_handle, u32) -> i32>,
    add_links: Option<unsafe extern "C" fn(*mut fwnode_handle) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct scatterlist {
    page_link: u64,
    offset: u32,
    length: u32,
    dma_address: u64,
    dma_length: u32,
    dma_flags: u32,
}


// We overload the LSB of the page pointer to indicate whether it's
// a valid sg entry, or whether it points to the start of a new scatterlist.
// Those low bits are there for everyone! (thanks mason :-)
fn __sg_flags(sg: *mut scatterlist) -> u32 {
((unsafe { (*sg).page_link }) & (1 | 2)) as u32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct vringh_config_ops {
}


fn sg_chain_ptr(sg: *mut scatterlist) -> *mut scatterlist {
((unsafe { (*sg).page_link }) & !(1 | 2)) as *mut scatterlist
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_virtio_h_184 {
    features: u64,
    features_array: [u64; 2],
}


fn sg_is_chain(sg: *mut scatterlist) -> bool {
(__sg_flags(sg) as u64) & 1 != 0
}

fn sg_is_last(sg: *mut scatterlist) -> bool {
(__sg_flags(sg) as u64) & 2 != 0
}

fn sg_next(mut sg: *mut scatterlist) -> *mut scatterlist {
if sg_is_last(sg) {
return core::ptr::null_mut::<scatterlist>();
}
let __v0: *mut scatterlist = sg;
sg = __v0.wrapping_offset(1 as isize);
if unlikely!(sg_is_chain(sg)) != 0 {
sg = sg_chain_ptr(sg);
}
sg
}

fn sg_page(sg: *mut scatterlist) -> *mut page {
((unsafe { (*sg).page_link }) & !(1 | 2)) as *mut page
}

fn sg_phys(sg: *mut scatterlist) -> u64 {
page_to_phys!(sg_page(sg)).wrapping_add((unsafe { (*sg).offset }) as u64)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pm_message {
    event: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_affinity {
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
    __slate_bits_0: __slate_bits::__SlateBits644U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits644U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits644U2,
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
    __slate_bits_3: __slate_bits::__SlateBits644U3,
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
struct iovec {
    iov_base: *mut (),
    iov_len: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_virtio_ring_c_137 {
    num: u32,
    desc: *mut vring_packed_desc,
    driver: *mut vring_packed_desc_event,
    device: *mut vring_packed_desc_event,
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
struct used_entry {
    id: u32,
    len: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_virtio_ring_c_249 {
    /// Available for split ring
    split: vring_virtqueue_split,
    /// Available for packed ring
    packed: vring_virtqueue_packed,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_virtio_ring_c_2012 {
    __slate_anon_0: __SlateRecord_virtio_ring_c_2012_2,
    u32: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_virtio_ring_c_2012_2 {
    off_wrap: u16,
    flags: u16,
}


#[repr(transparent)]
struct __SlateConst<T>(T);

unsafe impl<T> Sync for __SlateConst<T> {
}

mod __slate_bits {
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits280U0 {
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
    pub struct __SlateBits72U0 {
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
    pub struct __SlateBits72U1 {
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
    pub struct __SlateBits283U0 {
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
    pub struct __SlateBits72U2 {
        #[bits(1)]
        pub mce_ripv: u64,
        #[bits(1)]
        pub mce_whole_page: u64,
        #[bits(62)]
        pub __mce_reserved: u64,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits104U0 {
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
    pub struct __SlateBits122U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits429U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits614U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits566U0 {
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
    pub struct __SlateBits103U0 {
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
    pub struct __SlateBits398U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits644U0 {
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
    pub struct __SlateBits641U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits644U1 {
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
    pub struct __SlateBits644U2 {
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
    pub struct __SlateBits644U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
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


// The following license idents are currently accepted as indicating free
// software modules
//
// 	"GPL"				[GNU Public License v2]
// 	"GPL v2"			[GNU Public License v2]
// 	"GPL and additional rights"	[GNU Public License v2 rights and more]
// 	"Dual BSD/GPL"			[GNU Public License v2
// 					 or BSD license choice]
// 	"Dual MIT/GPL"			[GNU Public License v2
// 					 or MIT license choice]
// 	"Dual MPL/GPL"			[GNU Public License v2
// 					 or Mozilla license choice]
//
// The following other idents are available
//
// 	"Proprietary"			[Non free products]
//
// Both "GPL v2" and "GPL" (the latter also in dual licensed strings) are
// merely stating that the module is licensed under the GPL v2, but are not
// telling whether "GPL v2 only" or "GPL v2 or later". The reason why there
// are two variants is a historic and failed attempt to convey more
// information in the MODULE_LICENSE string. For module loading the
// "only/or later" distinction is completely irrelevant and does neither
// replace the proper license identifiers in the corresponding source file
// nor amends them in any way. The sole purpose is to make the
// 'Proprietary' flagging work and to refuse to bind symbols which are
// exported with EXPORT_SYMBOL_GPL when a non free module is loaded.
//
// In the same way "BSD" is not a clear license information. It merely
// states, that the module is licensed under one of the compatible BSD
// license variants. The detailed and correct license information is again
// to be found in the corresponding source files.
//
// There are dual licensed components, but when running with Linux it is the
// GPL that is relevant so this is a non issue. Similarly LGPL linked with GPL
// is a GPL combined work.
//
// This exists for several reasons
// 1.	So modinfo can show license info for users wanting to vet their setup
// 	is free
// 2.	So the community can ignore bug reports including proprietary modules
// 3.	So vendors can do likewise based on their own policies
// What your module does.
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


fn dev_dma_skip_sync(dev: *const device) -> bool {
test_bit__3!(3i32, unsafe { (*dev).flags.as_ptr() as *const u64 }) != 0
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
    __slate_bits_0: __slate_bits::__SlateBits641U0,
}


fn dev_name(dev: *const device) -> *const u8 {
// Use the init name until the kobject becomes available
if !unsafe { (*dev).init_name }.is_null() {
return unsafe { (*dev).init_name };
}
kobject_name(unsafe { core::ptr::addr_of!((*dev).kobj) })
}

#[repr(C)]
#[derive(Clone, Copy)]
struct virtio_device_id {
    device: u32,
    vendor: u32,
}


// DMA_ATTR_NO_WARN: This tells the DMA-mapping subsystem to suppress
// allocation failure reports (similarly to __GFP_NOWARN).
// DMA_ATTR_DEBUGGING_IGNORE_CACHELINES: Indicates the CPU cache line can be
// overlapped. All mappings sharing a cacheline must have this attribute for
// this to be considered safe.
// A dma_addr_t can hold any valid DMA or bus address for the platform.  It can
// be given to a device to use as a DMA source or target.  It is specific to a
// given device and there may be a translation between the CPU physical address
// space and the bus address space.
//
// DMA_MAPPING_ERROR is the magic error code if a mapping failed.  It should not
// be used directly in drivers, but checked for using dma_mapping_error()
// instead.
// Use the high bit to mark if we used swiotlb for one or more ranges.
fn debug_dma_mapping_error(dev: *mut device, dma_addr: u64) {
}

fn dma_mapping_error(dev: *mut device, dma_addr: u64) -> i32 {
debug_dma_mapping_error(dev, dma_addr);
if unlikely!(dma_addr == !0) != 0 {
return 12i32.wrapping_neg();
}
0
}

fn dma_dev_need_sync(dev: *const device) -> bool {
// Always call DMA sync operations when debugging is enabled
!dev_dma_skip_sync(dev) || IS_ENABLED!() != 0
}

fn dma_sync_single_for_cpu(dev: *mut device, addr: u64, size: u64, dir: u32) {
if dma_dev_need_sync(dev as *const device) {
unsafe { __dma_sync_single_for_cpu(dev, addr, size, dir) };
}
}

fn dma_sync_single_for_device(dev: *mut device, addr: u64, size: u64, dir: u32) {
if dma_dev_need_sync(dev as *const device) {
unsafe { __dma_sync_single_for_device(dev, addr, size, dir) };
}
}

fn dma_need_sync(dev: *mut device, dma_addr: u64) -> bool {
let __v0: i32 = if dma_dev_need_sync(dev as *const device) {
(unsafe { __dma_need_sync(dev, dma_addr) }) as i32
} else {
0
};
__v0 != 0
}

fn dma_sync_single_range_for_cpu(dev: *mut device, addr: u64, offset: u64, size: u64, dir: u32) {
dma_sync_single_for_cpu(dev, addr.wrapping_add(offset), size, dir);
return;
}

fn dma_sync_single_range_for_device(dev: *mut device, addr: u64, offset: u64, size: u64, dir: u32) {
dma_sync_single_for_device(dev, addr.wrapping_add(offset), size, dir);
return;
}

fn dma_alloc_coherent(dev: *mut device, size: u64, dma_handle: *mut u64, gfp: u32) -> *mut () {
unsafe { dma_alloc_attrs(dev, size, dma_handle, gfp, if gfp & (BIT!(13i32) as u32) != 0 { 256u64 } else { 0u64 }) }
}

fn dma_free_coherent(dev: *mut device, size: u64, cpu_addr: *mut (), dma_handle: u64) {
unsafe { dma_free_attrs(dev, size, cpu_addr, dma_handle, 0u64) };
return;
}

fn virtio_features_chk_bit(bit: u32) -> bool {
if unsafe { __slate_intrinsic_1c5364fcd9da51fe(bit) } {
// Don't care returning the correct value: the build
// will fail before any bad features access
loop {
if bit >= 128u32 {
unsafe { __compiletime_assert_511() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
} else {
let mut __v1: i64;
let __ret_warn_on: i32 = (bit >= 128u32) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("512: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 512b - ., 3; .popsection\n# {0}", const 512 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_virtio_features_chk_bit_2, sym __slate_asm_string_virtio_features_chk_bit_3, const 30 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("513: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 513b - ., 4; .popsection\n# {0}", const 513 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
if __v1 != 0 {
return false;
}
}
true
}

fn virtio_features_test_bit(features: *const u64, bit: u32) -> bool {
virtio_features_chk_bit(bit) && (unsafe { *features.wrapping_offset(VIRTIO_U64!(bit) as isize) }) & VIRTIO_BIT!(bit) != 0
}

fn virtio_features_clear_bit(features: *mut u64, bit: u32) {
if virtio_features_chk_bit(bit) {
let __v0: *mut u64 = features.wrapping_offset(VIRTIO_U64!(bit) as isize);
let __v1: u64 = unsafe { *__v0 };
let __v2: u64 = __v1 & !VIRTIO_BIT!(bit);
unsafe {
*__v0 = __v2;
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct virtqueue {
    list: list_head,
    callback: Option<unsafe extern "C" fn(*mut virtqueue)>,
    name: *const u8,
    vdev: *mut virtio_device,
    index: u32,
    num_free: u32,
    num_max: u32,
    reset: bool,
    r#priv: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
union virtio_map {
    dma_dev: *mut device,
    group: *mut vduse_vq_group,
}


#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_is_broken")]
extern "C" fn virtqueue_is_broken(_vq: *const virtqueue) -> bool {
let vq: *const vring_virtqueue = to_vvq__2!(_vq);
let mut __v1: bool;
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<bool>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_602() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*vq).broken) as *const bool) };
__v1
}

#[repr(C)]
#[derive(Clone, Copy)]
struct virtio_device {
    index: i32,
    failed: bool,
    config_core_enabled: bool,
    config_driver_disabled: bool,
    config_change_pending: bool,
    config_lock: spinlock,
    vqs_list_lock: spinlock,
    dev: device,
    id: virtio_device_id,
    config: *const virtio_config_ops,
    vringh_config: *const vringh_config_ops,
    map: *const virtio_map_ops,
    vqs: list_head,
    __slate_anon_13: __SlateRecord_virtio_h_184,
    r#priv: *mut (),
    vmap: virtio_map,
}


#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtio_break_device")]
extern "C" fn virtio_break_device(dev: *mut virtio_device) {
let mut _vq: *mut virtqueue = unsafe { core::mem::zeroed() };
spin_lock(unsafe { core::ptr::addr_of_mut!((*dev).vqs_list_lock) });
_vq = list_first_entry!(unsafe { core::ptr::addr_of_mut!((*dev).vqs) });
while list_entry_is_head!(_vq, unsafe { core::ptr::addr_of_mut!((*dev).vqs) }, list) == 0 {
let vq: *mut vring_virtqueue = to_vvq!(_vq);
// Pairs with READ_ONCE() in virtqueue_is_broken().
loop {
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<bool>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_604() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__3!((*vq).broken, true);
if true {
break;
}
}
_vq = list_next_entry!(_vq, list);
}
spin_unlock(unsafe { core::ptr::addr_of_mut!((*dev).vqs_list_lock) });
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.__virtio_unbreak_device")]
extern "C" fn __virtio_unbreak_device(dev: *mut virtio_device) {
let mut _vq: *mut virtqueue = unsafe { core::mem::zeroed() };
spin_lock(unsafe { core::ptr::addr_of_mut!((*dev).vqs_list_lock) });
_vq = list_first_entry!(unsafe { core::ptr::addr_of_mut!((*dev).vqs) });
while list_entry_is_head!(_vq, unsafe { core::ptr::addr_of_mut!((*dev).vqs) }, list) == 0 {
let vq: *mut vring_virtqueue = to_vvq!(_vq);
// Pairs with READ_ONCE() in virtqueue_is_broken().
loop {
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<bool>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_606() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__3!((*vq).broken, false);
if true {
break;
}
}
_vq = list_next_entry!(_vq, list);
}
spin_unlock(unsafe { core::ptr::addr_of_mut!((*dev).vqs_list_lock) });
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.__virtqueue_break")]
extern "C" fn __virtqueue_break(_vq: *mut virtqueue) {
let vq: *mut vring_virtqueue = to_vvq!(_vq);
// Pairs with READ_ONCE() in virtqueue_is_broken().
loop {
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<bool>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_598() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__3!((*vq).broken, true);
if true {
break;
}
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.__virtqueue_unbreak")]
extern "C" fn __virtqueue_unbreak(_vq: *mut virtqueue) {
let vq: *mut vring_virtqueue = to_vvq!(_vq);
// Pairs with READ_ONCE() in virtqueue_is_broken().
loop {
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<bool>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_600() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__3!((*vq).broken, false);
if true {
break;
}
}
}

// This marks a buffer as write-only (otherwise read-only).
// This means the buffer contains a list of buffer descriptors.
// Mark a descriptor as available or used in packed ring.
// Notice: they are defined as shifts instead of shifted values.
// The Host uses this in used->flags to advise the Guest: don't kick me when
// you add a buffer.  It's unreliable, so it's simply an optimization.  Guest
// will still kick if it's out of buffers.
// The Guest uses this in avail->flags to advise the Host: don't interrupt me
// when you consume a buffer.  It's unreliable, so it's simply an
// optimization.
// Enable events in packed ring.
// Disable events in packed ring.
// Enable events for a specific descriptor in packed ring.
// (as specified by Descriptor Ring Change Event Offset/Wrap Counter).
// Only valid if VIRTIO_RING_F_EVENT_IDX has been negotiated.
// Wrap counter bit shift in event suppression structure
// of packed ring.
// We support indirect buffer descriptors
// The Guest publishes the used index for which it expects an interrupt
// at the end of the avail ring. Host should ignore the avail->flags field.
//
// The Host publishes the avail index for which it expects a kick
// at the end of the used ring. Guest should ignore the used->flags field.
// Alignment requirements for vring elements.
// When using pre-virtio 1.0 layout, these fall out naturally.
#[repr(C)]
#[derive(Clone, Copy)]
struct vring_desc {
    addr: u64,
    len: u32,
    flags: u16,
    next: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vring_avail {
    flags: u16,
    idx: u16,
    ring: [u16; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vring_used_elem {
    id: u32,
    len: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vring_used {
    flags: u16,
    idx: u16,
    ring: [vring_used_elem; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vring {
    num: u32,
    desc: *mut vring_desc,
    avail: *mut vring_avail,
    used: *mut vring_used,
}


core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_virtio_max_dma_size: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_virtqueue_add_sgs: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_virtqueue_add_outbuf: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_virtqueue_add_outbuf_premapped: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_virtqueue_add_inbuf: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_virtqueue_add_inbuf_cache_clean: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_virtqueue_add_inbuf_ctx: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_virtqueue_add_inbuf_premapped: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {7} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_virtqueue_dma_dev: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {8} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_virtqueue_kick_prepare: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {9} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_virtqueue_notify: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {10} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_virtqueue_kick: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {11} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_virtqueue_get_buf_ctx: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {12} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_virtqueue_get_buf: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {13} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_virtqueue_disable_cb: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {14} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_virtqueue_enable_cb_prepare: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {15} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_virtqueue_poll: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {16} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_virtqueue_enable_cb: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {17} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_virtqueue_enable_cb_delayed: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {18} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_virtqueue_detach_unused_buf: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {19} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_vring_interrupt: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {20} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_vring_create_virtqueue: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {21} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_vring_create_virtqueue_map: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {22} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_virtqueue_resize: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {23} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_virtqueue_reset: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {24} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_vring_new_virtqueue: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {25} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_vring_del_virtqueue: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {26} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_vring_notification_data: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {27} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_vring_transport_features: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {28} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_virtqueue_get_vring_size: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {29} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol___virtqueue_break: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {30} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol___virtqueue_unbreak: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {31} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_virtqueue_is_broken: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {32} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_virtio_break_device: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {33} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol___virtio_unbreak_device: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {34} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_virtqueue_get_desc_addr: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {35} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_virtqueue_get_avail_addr: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {36} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_virtqueue_get_used_addr: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {37} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_virtqueue_get_vring: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {38} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_virtqueue_map_alloc_coherent: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {39} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_virtqueue_map_free_coherent: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {40} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_virtqueue_map_page_attrs: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {41} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_virtqueue_unmap_page_attrs: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {42} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_virtqueue_map_single_attrs: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {43} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_virtqueue_unmap_single_attrs: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {44} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_virtqueue_map_mapping_error: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {45} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_virtqueue_map_need_sync: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {46} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_virtqueue_map_sync_single_range_for_cpu: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {47} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_virtqueue_map_sync_single_range_for_device: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {48} ; .previous", sym virtio_max_dma_size, sym virtqueue_add_sgs, sym virtqueue_add_outbuf, sym virtqueue_add_outbuf_premapped, sym virtqueue_add_inbuf, sym virtqueue_add_inbuf_cache_clean, sym virtqueue_add_inbuf_ctx, sym virtqueue_add_inbuf_premapped, sym virtqueue_dma_dev, sym virtqueue_kick_prepare, sym virtqueue_notify, sym virtqueue_kick, sym virtqueue_get_buf_ctx, sym virtqueue_get_buf, sym virtqueue_disable_cb, sym virtqueue_enable_cb_prepare, sym virtqueue_poll, sym virtqueue_enable_cb, sym virtqueue_enable_cb_delayed, sym virtqueue_detach_unused_buf, sym vring_interrupt, sym vring_create_virtqueue, sym vring_create_virtqueue_map, sym virtqueue_resize, sym virtqueue_reset, sym vring_new_virtqueue, sym vring_del_virtqueue, sym vring_notification_data, sym vring_transport_features, sym virtqueue_get_vring_size, sym __virtqueue_break, sym __virtqueue_unbreak, sym virtqueue_is_broken, sym virtio_break_device, sym __virtio_unbreak_device, sym virtqueue_get_desc_addr, sym virtqueue_get_avail_addr, sym virtqueue_get_used_addr, sym virtqueue_get_vring, sym virtqueue_map_alloc_coherent, sym virtqueue_map_free_coherent, sym virtqueue_map_page_attrs, sym virtqueue_unmap_page_attrs, sym virtqueue_map_single_attrs, sym virtqueue_unmap_single_attrs, sym virtqueue_map_mapping_error, sym virtqueue_map_need_sync, sym virtqueue_map_sync_single_range_for_cpu, sym virtqueue_map_sync_single_range_for_device, options(att_syntax));

// The standard layout for the ring is a continuous chunk of memory which looks
// like this.  We assume num is a power of 2.
//
// struct vring
// {
// 	// The actual descriptors (16 bytes each)
// 	struct vring_desc desc[num];
//
// 	// A ring of available descriptor heads with free-running index.
// 	__virtio16 avail_flags;
// 	__virtio16 avail_idx;
// 	__virtio16 available[num];
// 	__virtio16 used_event_idx;
//
// 	// Padding to the next align boundary.
// 	char pad[];
//
// 	// A ring of used descriptor heads with free-running index.
// 	__virtio16 used_flags;
// 	__virtio16 used_idx;
// 	struct vring_used_elem used[num];
// 	__virtio16 avail_event_idx;
// };
//
// We publish the used event index at the end of the available ring, and vice
// versa. They are at the end for backwards compatibility.
fn vring_init(vr: *mut vring, num: u32, p: *mut (), align: u64) {
unsafe {
(*vr).num = num;
}
unsafe {
(*vr).desc = p as *mut vring_desc;
}
unsafe {
(*vr).avail = (p as *mut u8).wrapping_offset((num as u64).wrapping_mul(core::mem::size_of::<vring_desc>() as u64) as isize) as *mut vring_avail;
}
unsafe {
(*vr).used = ((((core::ptr::addr_of_mut!((*(*vr).avail).ring) as *mut u16).wrapping_offset(num as isize) as u64).wrapping_add(core::mem::size_of::<u16>() as u64).wrapping_add(align).wrapping_sub(1) & !align.wrapping_sub(1)) as *mut ()) as *mut vring_used;
}
}

fn vring_size(num: u32, align: u64) -> u32 {
((core::mem::size_of::<vring_desc>() as u64).wrapping_mul(num as u64).wrapping_add((core::mem::size_of::<u16>() as u64).wrapping_mul(3u32.wrapping_add(num) as u64)).wrapping_add(align).wrapping_sub(1) & !align.wrapping_sub(1)).wrapping_add((core::mem::size_of::<u16>() as u64).wrapping_mul(3)).wrapping_add((core::mem::size_of::<vring_used_elem>() as u64).wrapping_mul(num as u64)) as u32
}

fn vring_need_event(event_idx: u16, new_idx: u16, old: u16) -> i32 {
// Note: Xen has similar logic for notification hold-off
// in include/xen/interface/io/ring.h with req_event and req_prod
// corresponding to event_idx + 1 and new_idx respectively.
// Note also that req_event and req_prod in Xen start at 1,
// event indexes in virtio start at 0.
((((new_idx as i32).wrapping_sub(event_idx as i32).wrapping_sub(1) as u16) as i32) < (((new_idx as i32).wrapping_sub(old as i32) as u16) as i32)) as i32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct vring_packed_desc_event {
    off_wrap: u16,
    flags: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vring_packed_desc {
    addr: u64,
    len: u32,
    id: u16,
    flags: u16,
}


fn virtio_mb(weak_barriers: bool) {
if weak_barriers {
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
loop {
kcsan_mb!();
unsafe {
core::arch::asm!("mfence", options(nostack, att_syntax));
}
if true {
break;
}
}
}
}

fn virtio_rmb(weak_barriers: bool) {
if weak_barriers {
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
} else {
loop {
kcsan_rmb!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
}
}

fn virtio_wmb(weak_barriers: bool) {
if weak_barriers {
loop {
kcsan_wmb!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
} else {
loop {
kcsan_wmb!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
}
}

fn virtio_legacy_is_little_endian() -> bool {
true
}

fn __virtio16_to_cpu(little_endian: bool, val: u16) -> u16 {
if little_endian {
return val;
} else {
return __swab16!(val);
}
}

fn __cpu_to_virtio16(little_endian: bool, val: u16) -> u16 {
if little_endian {
return val;
} else {
return __swab16!(val);
}
}

fn __virtio32_to_cpu(little_endian: bool, val: u32) -> u32 {
if little_endian {
return val;
} else {
return __swab32!(val);
}
}

fn __cpu_to_virtio32(little_endian: bool, val: u32) -> u32 {
if little_endian {
return val;
} else {
return __swab32!(val);
}
}

fn __cpu_to_virtio64(little_endian: bool, val: u64) -> u64 {
if little_endian {
return val;
} else {
return __swab64!(val);
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct virtio_shm_region {
    addr: u64,
    len: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct virtqueue_info {
    name: *const u8,
    callback: Option<unsafe extern "C" fn(*mut virtqueue)>,
    ctx: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct virtio_config_ops {
    get: Option<unsafe extern "C" fn(*mut virtio_device, u32, *mut (), u32)>,
    set: Option<unsafe extern "C" fn(*mut virtio_device, u32, *const (), u32)>,
    generation: Option<unsafe extern "C" fn(*mut virtio_device) -> u32>,
    get_status: Option<unsafe extern "C" fn(*mut virtio_device) -> u8>,
    set_status: Option<unsafe extern "C" fn(*mut virtio_device, u8)>,
    reset: Option<unsafe extern "C" fn(*mut virtio_device)>,
    find_vqs: Option<unsafe extern "C" fn(*mut virtio_device, u32, *mut *mut virtqueue, *mut virtqueue_info, *mut irq_affinity) -> i32>,
    del_vqs: Option<unsafe extern "C" fn(*mut virtio_device)>,
    synchronize_cbs: Option<unsafe extern "C" fn(*mut virtio_device)>,
    get_features: Option<unsafe extern "C" fn(*mut virtio_device) -> u64>,
    get_extended_features: Option<unsafe extern "C" fn(*mut virtio_device, *mut u64)>,
    finalize_features: Option<unsafe extern "C" fn(*mut virtio_device) -> i32>,
    bus_name: Option<unsafe extern "C" fn(*mut virtio_device) -> *const u8>,
    set_vq_affinity: Option<unsafe extern "C" fn(*mut virtqueue, *const cpumask) -> i32>,
    get_vq_affinity: Option<unsafe extern "C" fn(*mut virtio_device, i32) -> *const cpumask>,
    get_shm_region: Option<unsafe extern "C" fn(*mut virtio_device, *mut virtio_shm_region, u8) -> bool>,
    disable_vq_and_reset: Option<unsafe extern "C" fn(*mut virtqueue) -> i32>,
    enable_vq_after_reset: Option<unsafe extern "C" fn(*mut virtqueue) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct virtio_map_ops {
    map_page: Option<unsafe extern "C" fn(virtio_map, *mut page, u64, u64, u32, u64) -> u64>,
    unmap_page: Option<unsafe extern "C" fn(virtio_map, u64, u64, u32, u64)>,
    sync_single_for_cpu: Option<unsafe extern "C" fn(virtio_map, u64, u64, u32)>,
    sync_single_for_device: Option<unsafe extern "C" fn(virtio_map, u64, u64, u32)>,
    alloc: Option<unsafe extern "C" fn(virtio_map, u64, *mut u64, u32) -> *mut ()>,
    free: Option<unsafe extern "C" fn(virtio_map, u64, *mut (), u64, u64)>,
    need_sync: Option<unsafe extern "C" fn(virtio_map, u64) -> bool>,
    mapping_error: Option<unsafe extern "C" fn(virtio_map, u64) -> i32>,
    max_mapping_size: Option<unsafe extern "C" fn(virtio_map) -> u64>,
}


fn __virtio_test_bit(vdev: *const virtio_device, fbit: u32) -> bool {
virtio_features_test_bit(unsafe { (*vdev).__slate_anon_13.features_array.as_ptr() as *const u64 }, fbit)
}

fn __virtio_clear_bit(vdev: *mut virtio_device, fbit: u32) {
virtio_features_clear_bit(unsafe { (*vdev).__slate_anon_13.features_array.as_mut_ptr() as *mut u64 }, fbit);
}

fn virtio_has_feature(vdev: *const virtio_device, fbit: u32) -> bool {
if fbit < 28 {
unsafe { virtio_check_driver_offered_feature(vdev, fbit) };
}
__virtio_test_bit(vdev, fbit)
}

fn virtio_has_dma_quirk(vdev: *const virtio_device) -> bool {
// Note the reverse polarity of the quirk feature (compared to most
// other features), this is for compatibility with legacy systems.
!virtio_has_feature(vdev, 33u32)
}

fn virtio_is_little_endian(vdev: *mut virtio_device) -> bool {
virtio_has_feature(vdev as *const virtio_device, 32u32) || virtio_legacy_is_little_endian()
}

fn virtio16_to_cpu(vdev: *mut virtio_device, val: u16) -> u16 {
__virtio16_to_cpu(virtio_is_little_endian(vdev), val)
}

fn cpu_to_virtio16(vdev: *mut virtio_device, val: u16) -> u16 {
__cpu_to_virtio16(virtio_is_little_endian(vdev), val)
}

fn virtio32_to_cpu(vdev: *mut virtio_device, val: u32) -> u32 {
__virtio32_to_cpu(virtio_is_little_endian(vdev), val)
}

fn cpu_to_virtio32(vdev: *mut virtio_device, val: u32) -> u32 {
__cpu_to_virtio32(virtio_is_little_endian(vdev), val)
}

fn cpu_to_virtio64(vdev: *mut virtio_device, val: u64) -> u64 {
__cpu_to_virtio64(virtio_is_little_endian(vdev), val)
}

fn kmsan_handle_dma(phys: u64, size: u64, dir: u32) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct vring_desc_state_split {
    /// Data for callback.
    data: *mut (),
    /// Indirect desc table and extra table, if any. These two will be
    /// allocated together. So we won't stress more to the memory allocator.
    indir_desc: *mut vring_desc,
    total_in_len: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vring_desc_state_packed {
    /// Data for callback.
    data: *mut (),
    /// Indirect desc table and extra table, if any. These two will be
    /// allocated together. So we won't stress more to the memory allocator.
    indir_desc: *mut vring_packed_desc,
    /// Descriptor list length.
    num: u16,
    /// The last desc state in a list.
    last: u16,
    /// In length for the skipped buffer.
    total_in_len: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vring_desc_extra {
    /// Descriptor DMA addr.
    addr: u64,
    /// Descriptor length.
    len: u32,
    /// Descriptor flags.
    flags: u16,
    /// The next desc state in a list.
    next: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vring_virtqueue_split {
    /// Actual memory layout for this queue.
    vring: vring,
    /// Last written value to avail->flags
    avail_flags_shadow: u16,
    /// Last written value to avail->idx in
    /// guest byte order.
    avail_idx_shadow: u16,
    /// Per-descriptor state.
    desc_state: *mut vring_desc_state_split,
    desc_extra: *mut vring_desc_extra,
    /// DMA address and size information
    queue_dma_addr: u64,
    queue_size_in_bytes: u64,
    /// The parameters for creating vrings are reserved for creating new
    /// vring.
    vring_align: u32,
    may_reduce_num: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vring_virtqueue_packed {
    /// Actual memory layout for this queue.
    vring: __SlateRecord_virtio_ring_c_137,
    /// Driver ring wrap counter.
    avail_wrap_counter: bool,
    /// Avail used flags.
    avail_used_flags: u16,
    /// Index of the next avail descriptor.
    next_avail_idx: u16,
    /// Last written value to driver->flags in
    /// guest byte order.
    event_flags_shadow: u16,
    /// Per-descriptor state.
    desc_state: *mut vring_desc_state_packed,
    desc_extra: *mut vring_desc_extra,
    /// DMA address and size information
    ring_dma_addr: u64,
    driver_event_dma_addr: u64,
    device_event_dma_addr: u64,
    ring_size_in_bytes: u64,
    event_size_in_bytes: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct virtqueue_ops {
    add: Option<unsafe extern "C" fn(*mut vring_virtqueue, *mut *mut scatterlist, u32, u32, u32, *mut (), *mut (), bool, u32, u64) -> i32>,
    get: Option<unsafe extern "C" fn(*mut vring_virtqueue, *mut u32, *mut *mut ()) -> *mut ()>,
    kick_prepare: Option<unsafe extern "C" fn(*mut vring_virtqueue) -> bool>,
    disable_cb: Option<unsafe extern "C" fn(*mut vring_virtqueue)>,
    enable_cb_delayed: Option<unsafe extern "C" fn(*mut vring_virtqueue) -> bool>,
    enable_cb_prepare: Option<unsafe extern "C" fn(*mut vring_virtqueue) -> u32>,
    poll: Option<unsafe extern "C" fn(*const vring_virtqueue, u32) -> bool>,
    detach_unused_buf: Option<unsafe extern "C" fn(*mut vring_virtqueue) -> *mut ()>,
    more_used: Option<unsafe extern "C" fn(*const vring_virtqueue) -> bool>,
    resize: Option<unsafe extern "C" fn(*mut vring_virtqueue, u32) -> i32>,
    reset: Option<unsafe extern "C" fn(*mut vring_virtqueue)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vring_virtqueue {
    vq: virtqueue,
    /// Is DMA API used?
    use_map_api: bool,
    /// Can we use weak barriers?
    weak_barriers: bool,
    /// Other side has made a mess, don't try any more.
    broken: bool,
    /// Host supports indirect buffers
    indirect: bool,
    /// Host publishes avail event idx
    event: bool,
    layout: u32,
    /// Without IN_ORDER it's the head of free buffer list. With
    /// IN_ORDER and SPLIT, it's the next available buffer
    /// index. With IN_ORDER and PACKED, it's unused.
    free_head: u32,
    /// With IN_ORDER, once we see an in-order batch, this stores
    /// this last entry, and until we return the last buffer.
    /// After this, id is set to UINT_MAX to mark it invalid.
    /// Unused without IN_ORDER.
    batch_last: used_entry,
    /// Number we've added since last sync.
    num_added: u32,
    /// Last used index  we've seen.
    /// for split ring, it just contains last used index
    /// for packed ring:
    /// bits up to VRING_PACKED_EVENT_F_WRAP_CTR include the last used index.
    /// bits from VRING_PACKED_EVENT_F_WRAP_CTR include the used wrap counter.
    last_used_idx: u16,
    /// With IN_ORDER and SPLIT, last descriptor id we used to
    /// detach buffer.
    last_used: u16,
    /// Hint for event idx: already triggered no need to disable.
    event_triggered: bool,
    __slate_anon_13: __SlateRecord_virtio_ring_c_249,
    /// How to notify other side. FIXME: commonalize hcalls!
    notify: Option<unsafe extern "C" fn(*mut virtqueue) -> bool>,
    /// DMA, allocation, and size information
    we_own_ring: bool,
    map: virtio_map,
}


// Accessors for device-writable fields in virtio rings.
// These fields are concurrently written by the device and read by the driver.
// Use READ_ONCE() to prevent compiler optimizations, document the
// intentional data race and prevent KCSAN warnings.
fn vring_read_split_used_idx(vq: *const vring_virtqueue) -> u16 {
let mut __v0: u16;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u16>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_517() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*(*vq).__slate_anon_13.split.vring.used).idx) as *const u16) };
virtio16_to_cpu(unsafe { (*vq).vq.vdev }, __v0)
}

fn vring_read_split_used_id(vq: *const vring_virtqueue, idx: u16) -> u32 {
let mut __v0: u32;
loop {
if !(__native_word__6!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_518() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*(core::ptr::addr_of_mut!((*(*vq).__slate_anon_13.split.vring.used).ring) as *mut vring_used_elem).wrapping_offset((idx as i32) as isize)).id) as *const u32) };
virtio32_to_cpu(unsafe { (*vq).vq.vdev }, __v0)
}

fn vring_read_split_used_len(vq: *const vring_virtqueue, idx: u16) -> u32 {
let mut __v0: u32;
loop {
if !(__native_word__6!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_519() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*(core::ptr::addr_of_mut!((*(*vq).__slate_anon_13.split.vring.used).ring) as *mut vring_used_elem).wrapping_offset((idx as i32) as isize)).len) as *const u32) };
virtio32_to_cpu(unsafe { (*vq).vq.vdev }, __v0)
}

fn vring_read_split_avail_event(vq: *const vring_virtqueue) -> u16 {
let mut __v0: u16;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u16>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_520() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(((core::ptr::addr_of_mut!((*(*core::ptr::addr_of!((*vq).__slate_anon_13.split.vring)).used).ring) as *mut vring_used_elem).wrapping_offset((*core::ptr::addr_of!((*vq).__slate_anon_13.split.vring)).num as isize) as *mut u16) as *const u16) };
virtio16_to_cpu(unsafe { (*vq).vq.vdev }, __v0)
}

fn vring_read_packed_desc_flags(vq: *const vring_virtqueue, idx: u16) -> u16 {
let mut __v0: u16;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u16>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_521() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*unsafe { (*vq).__slate_anon_13.packed.vring.desc }.wrapping_offset((idx as i32) as isize)).flags) as *const u16) };
__v0
}

fn vring_read_packed_desc_id(vq: *const vring_virtqueue, idx: u16) -> u16 {
let mut __v0: u16;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u16>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_522() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*unsafe { (*vq).__slate_anon_13.packed.vring.desc }.wrapping_offset((idx as i32) as isize)).id) as *const u16) };
__v0
}

fn vring_read_packed_desc_len(vq: *const vring_virtqueue, idx: u16) -> u32 {
let mut __v0: u32;
loop {
if !(__native_word__6!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_523() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*unsafe { (*vq).__slate_anon_13.packed.vring.desc }.wrapping_offset((idx as i32) as isize)).len) as *const u32) };
__v0
}

// Helpers.
fn virtqueue_is_packed(vq: *const vring_virtqueue) -> bool {
((unsafe { (*vq).layout }) as u32) == 1 || ((unsafe { (*vq).layout }) as u32) == 3
}

fn virtqueue_is_in_order(vq: *const vring_virtqueue) -> bool {
((unsafe { (*vq).layout }) as u32) == 2 || ((unsafe { (*vq).layout }) as u32) == 3
}

fn virtqueue_use_indirect(vq: *const vring_virtqueue, total_sg: u32) -> bool {
// If the host supports indirect descriptor tables, and we have multiple
// buffers, then go indirect. FIXME: tune this threshold
(unsafe { (*vq).indirect }) && total_sg > 1 && (unsafe { (*vq).vq.num_free }) != 0
}

// Modern virtio devices have feature bits to specify whether they need a
// quirk and bypass the IOMMU. If not there, just use the DMA API.
//
// If there, the interaction between virtio and DMA API is messy.
//
// On most systems with virtio, physical addresses match bus addresses,
// and it doesn't particularly matter whether we use the DMA API.
//
// On some systems, including Xen and any system with a physical device
// that speaks virtio behind a physical IOMMU, we must use the DMA API
// for virtio DMA to work at all.
//
// On other systems, including SPARC and PPC64, virtio-pci devices are
// enumerated as though they are behind an IOMMU, but the virtio host
// ignores the IOMMU, so we must either pretend that the IOMMU isn't
// there or somehow map everything as the identity.
//
// For the time being, we preserve historic behavior and bypass the DMA
// API.
//
// TODO: install a per-device DMA ops structure that does the right thing
// taking into account all the above quirks, and use the DMA API
// unconditionally on data path.
fn vring_use_map_api(vdev: *const virtio_device) -> bool {
if !virtio_has_dma_quirk(vdev) {
return true;
}
// Otherwise, we are left to guess.
//
// In theory, it's possible to have a buggy QEMU-supposed
// emulated Q35 IOMMU and Xen enabled at the same time.  On
// such a configuration, virtio has never worked and will
// not work without an even larger kludge.  Instead, enable
// the DMA API if we're a Xen guest, which at least allows
// all of the sensible Xen configurations to work correctly.
if xen_domain!() {
return true;
}
false
}

fn vring_need_unmap_buffer(vring: *const vring_virtqueue, extra: *const vring_desc_extra) -> bool {
(unsafe { (*vring).use_map_api }) && (unsafe { (*extra).addr }) != !0
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtio_max_dma_size")]
extern "C" fn virtio_max_dma_size(vdev: *const virtio_device) -> u64 {
let mut max_segment_size: u64 = !0;
if vring_use_map_api(vdev) {
if !unsafe { (*vdev).map }.is_null() {
max_segment_size = unsafe { unsafe { (*(*vdev).map).max_mapping_size }.unwrap()((*vdev).vmap) };
} else {
max_segment_size = unsafe { dma_max_mapping_size((*vdev).dev.parent) };
}
}
max_segment_size
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_virtio_max_dma_size_524: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const virtio_device) -> u64>, *mut ()>(Some(virtio_max_dma_size)) };

fn vring_alloc_queue(vdev: *mut virtio_device, size: u64, map_handle: *mut u64, flag: u32, map: virtio_map) -> *mut () {
if vring_use_map_api(vdev as *const virtio_device) {
return virtqueue_map_alloc_coherent(vdev, map, size, map_handle, flag);
} else {
let queue: *mut () = alloc_hooks!(unsafe { alloc_pages_exact_noprof(size.wrapping_add(4095u64) & 18446744073709547520u64, flag) });
if !queue.is_null() {
let phys_addr: u64 = virt_to_phys(queue as *mut ());
unsafe {
*map_handle = phys_addr;
}
// Sanity check: make sure we dind't truncate
// the address.  The only arches I can find that
// have 64-bit phys_addr_t but 32-bit dma_addr_t
// are certain non-highmem MIPS and x86
// configurations, but these configurations
// should never allocate physical pages above 32
// bits, so this is fine.  Just in case, throw a
// warning and abort if we end up with an
// unrepresentable address.
let mut __v5: i64;
let __ret_warn_on: i32 = ((unsafe { *map_handle }) != phys_addr) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("525: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 525b - ., 3; .popsection\n# {0}", const 525 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_vring_alloc_queue_6, sym __slate_asm_string_vring_alloc_queue_7, const 450 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("526: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 526b - ., 4; .popsection\n# {0}", const 526 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v5 = (__ret_warn_on != 0) as i64;
if __v5 != 0 {
unsafe { free_pages_exact(queue, PAGE_ALIGN!(size)) };
return core::ptr::null_mut::<()>();
}
}
return queue;
}
}

fn vring_free_queue(vdev: *mut virtio_device, size: u64, queue: *mut (), map_handle: u64, map: virtio_map) {
if vring_use_map_api(vdev as *const virtio_device) {
virtqueue_map_free_coherent(vdev, map, size, queue, map_handle);
} else {
unsafe { free_pages_exact(queue, PAGE_ALIGN!(size)) };
}
}

/// The DMA ops on various arches are rather gnarly right now, and
/// making all of the arch DMA ops work on the vring device itself
/// is a mess.
fn vring_dma_dev(vq: *const vring_virtqueue) -> *mut device {
unsafe { (*vq).map.dma_dev }
}

fn vring_mapping_error(vq: *const vring_virtqueue, addr: u64) -> i32 {
let vdev: *mut virtio_device = unsafe { (*vq).vq.vdev };
if !unsafe { (*vq).use_map_api } {
return 0;
}
if !unsafe { (*vdev).map }.is_null() {
return unsafe { unsafe { (*(*vdev).map).mapping_error }.unwrap()((*vq).map, addr) };
} else {
return dma_mapping_error(vring_dma_dev(vq), addr);
}
}

/// Map one sg entry.
fn vring_map_one_sg(vq: *const vring_virtqueue, sg: *mut scatterlist, direction: u32, addr: &mut u64, len: &mut u32, premapped: bool, attr: u64) -> i32 {
if premapped {
unsafe {
*(addr as *mut u64) = sg_dma_address!(sg);
}
unsafe {
*(len as *mut u32) = sg_dma_len!(sg);
}
return 0;
}
unsafe {
*(len as *mut u32) = (*sg).length;
}
if !unsafe { (*vq).use_map_api } {
// If DMA is not used, KMSAN doesn't know that the scatterlist
// is initialized by the hardware. Explicitly check/unpoison it
// depending on the direction.
kmsan_handle_dma(sg_phys(sg), (unsafe { (*sg).length }) as u64, direction);
unsafe {
*(addr as *mut u64) = sg_phys(sg);
}
return 0;
}
// We can't use dma_map_sg, because we don't use scatterlists in
// the way it expects (we don't guarantee that the scatterlist
// will exist for the lifetime of the mapping).
unsafe {
*(addr as *mut u64) = virtqueue_map_page_attrs(core::ptr::addr_of!((*vq).vq), sg_page(sg), (*sg).offset as u64, (*sg).length as u64, direction, attr);
}
if vring_mapping_error(vq, unsafe { *(addr as *mut u64) }) != 0 {
return 12i32.wrapping_neg();
}
0
}

fn vring_map_single(vq: *const vring_virtqueue, cpu_addr: *mut (), size: u64, direction: u32) -> u64 {
if !unsafe { (*vq).use_map_api } {
return virt_to_phys(cpu_addr as *mut ());
}
virtqueue_map_single_attrs(unsafe { core::ptr::addr_of!((*vq).vq) }, cpu_addr, size, direction, 0u64)
}

fn virtqueue_init(vq: *mut vring_virtqueue, num: u32) {
unsafe {
(*vq).vq.num_free = num;
}
if virtqueue_is_packed(vq as *const vring_virtqueue) {
unsafe {
(*vq).last_used_idx = 32768u16;
}
} else {
unsafe {
(*vq).last_used_idx = 0;
}
}
unsafe {
(*vq).last_used = 0;
}
unsafe {
(*vq).event_triggered = false;
}
unsafe {
(*vq).num_added = 0;
}
}

// Split ring specific functions - *_split().
fn vring_unmap_one_split(vq: *const vring_virtqueue, extra: *mut vring_desc_extra) -> u32 {
let mut flags: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut u16 = core::ptr::addr_of_mut!(flags) as *mut u16;
unsafe {
'__join_0: {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 2usize);
*__slate_slot_0 = (*extra).flags;
if (*__slate_slot_0 as i32) & 4 != 0 {
if !(*vq).use_map_api {
break '__join_0;
}
} else {
if !vring_need_unmap_buffer(vq, extra as *const vring_desc_extra) {
break '__join_0;
}
}
virtqueue_unmap_page_attrs(core::ptr::addr_of!((*vq).vq), (*extra).addr, (*extra).len as u64, (if (*__slate_slot_0 as i32) & 2 != 0 { 2i32 } else { 1i32 }) as u32, 0u64);
}
return (*extra).next as u32;
}
}

fn alloc_indirect_split(vq: *mut vring_virtqueue, total_sg: u32, mut gfp: u32) -> *mut vring_desc {
let mut __slate_compound_4: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_6: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut size: u32 = 0;
// We require lowmem mappings for the descriptors because
// otherwise virt_to_phys will give us bogus addresses in the
// virtqueue.
let __v0: u32 = gfp;
gfp = __v0 & !(BIT!(1i32) as u32);
size = (core::mem::size_of::<vring_desc>() as u64).wrapping_mul(total_sg as u64).wrapping_add((core::mem::size_of::<vring_desc_extra>() as u64).wrapping_mul(total_sg as u64)) as u32;
let mut __v2: *mut ();
let mut __v3: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_noprof(size as u64, gfp, unsafe { *{
__slate_compound_4 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_4)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_noprof(size as u64, gfp, unsafe { *{
__slate_compound_6 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_6)
} });
}
__v3 = _res;
__v2 = __v3;
let desc: *mut vring_desc = __v2 as *mut vring_desc;
if desc.is_null() {
return core::ptr::null_mut::<vring_desc>();
}
let extra: *mut vring_desc_extra = desc.wrapping_offset(total_sg as isize) as *mut vring_desc_extra;
for i in 0..total_sg {
unsafe {
(*extra.wrapping_offset(i as isize)).next = i.wrapping_add(1) as u16;
}
}
desc
}

fn virtqueue_add_desc_split(vq: *mut vring_virtqueue, desc: *mut vring_desc, extra: *mut vring_desc_extra, i: u32, addr: u64, len: u32, flags: u16, premapped: bool) -> u32 {
let vdev: *mut virtio_device = unsafe { (*vq).vq.vdev };
let mut next: u16 = 0;
unsafe {
(*desc.wrapping_offset(i as isize)).flags = cpu_to_virtio16(vdev, flags);
}
unsafe {
(*desc.wrapping_offset(i as isize)).addr = cpu_to_virtio64(vdev, addr);
}
unsafe {
(*desc.wrapping_offset(i as isize)).len = cpu_to_virtio32(vdev, len);
}
unsafe {
(*extra.wrapping_offset(i as isize)).addr = if premapped { !0 } else { addr };
}
unsafe {
(*extra.wrapping_offset(i as isize)).len = len;
}
unsafe {
(*extra.wrapping_offset(i as isize)).flags = flags;
}
next = unsafe { (*extra.wrapping_offset(i as isize)).next };
unsafe {
(*desc.wrapping_offset(i as isize)).next = cpu_to_virtio16(vdev, next);
}
next as u32
}

#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_add_split")]
extern "C" fn virtqueue_add_split(vq: *mut vring_virtqueue, sgs: *mut *mut scatterlist, total_sg: u32, out_sgs: u32, in_sgs: u32, data: *mut (), ctx: *mut (), premapped: bool, gfp: u32, attr: u64) -> i32 {
// Now that the indirect table is filled in, map it.
let mut addr: u64 = unsafe { core::mem::zeroed() };
let mut flags: u16 = unsafe { core::mem::zeroed() };
let mut len: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_28: *mut u32 = core::ptr::addr_of_mut!(len) as *mut u32;
let mut addr_1: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_29: *mut u64 = core::ptr::addr_of_mut!(addr_1) as *mut u64;
let mut flags_1: u16 = unsafe { core::mem::zeroed() };
let mut len: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_37: *mut u32 = core::ptr::addr_of_mut!(len) as *mut u32;
let mut addr_2: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_38: *mut u64 = core::ptr::addr_of_mut!(addr_2) as *mut u64;
let mut indirect: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_43: *mut bool = core::ptr::addr_of_mut!(indirect) as *mut bool;
let mut head: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_44: *mut i32 = core::ptr::addr_of_mut!(head) as *mut i32;
// Total length for in-order
let mut total_in_len: u32 = unsafe { core::mem::zeroed() };
let mut sg_count: u32 = unsafe { core::mem::zeroed() };
let mut err_idx: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_47: *mut u32 = core::ptr::addr_of_mut!(err_idx) as *mut u32;
let mut descs_used: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_48: *mut u32 = core::ptr::addr_of_mut!(descs_used) as *mut u32;
let mut avail: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_49: *mut u32 = core::ptr::addr_of_mut!(avail) as *mut u32;
let mut n: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_50: *mut u32 = core::ptr::addr_of_mut!(n) as *mut u32;
let mut i: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_51: *mut u32 = core::ptr::addr_of_mut!(i) as *mut u32;
let mut desc: core::mem::MaybeUninit<*mut vring_desc> = core::mem::MaybeUninit::uninit();
let __slate_slot_52: *mut *mut vring_desc = core::ptr::addr_of_mut!(desc) as *mut *mut vring_desc;
let mut sg: core::mem::MaybeUninit<*mut scatterlist> = core::mem::MaybeUninit::uninit();
let __slate_slot_53: *mut *mut scatterlist = core::ptr::addr_of_mut!(sg) as *mut *mut scatterlist;
let mut extra: core::mem::MaybeUninit<*mut vring_desc_extra> = core::mem::MaybeUninit::uninit();
let __slate_slot_54: *mut *mut vring_desc_extra = core::ptr::addr_of_mut!(extra) as *mut *mut vring_desc_extra;
unsafe {
let __slate_auto_bytes_2 = __slate_slot_54 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_53 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_52 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_51 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_6 = __slate_slot_50 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_7 = __slate_slot_49 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_8 = __slate_slot_48 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_9 = __slate_slot_47 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 4usize);
sg_count = 0;
total_in_len = 0;
let __slate_auto_bytes_10 = __slate_slot_44 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_10.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_11 = __slate_slot_43 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_11.add(0usize), 0u8, 1usize);
'__join_72: {
loop {
if unlikely!(data.is_null()) != 0 {
break '__join_72;
} else {
if true {
break;
}
}
}
'__join_68: {
loop {
if unlikely!(!ctx.is_null() && (*vq).indirect) != 0 {
break '__join_68;
} else {
if true {
break;
}
}
}
if unlikely!(unsafe { (*vq).broken }) != 0 {
return 5i32.wrapping_neg();
} else {
'__join_61: {
loop {
if unlikely!(total_sg == 0) != 0 {
break '__join_61;
} else {
if true {
break;
}
}
}
*__slate_slot_44 = (*vq).free_head as i32;
if virtqueue_use_indirect(vq as *const vring_virtqueue, total_sg) {
*__slate_slot_52 = alloc_indirect_split(vq, total_sg, gfp);
} else {
*__slate_slot_52 = core::ptr::null_mut::<vring_desc>();
let __ret_warn_on: i32 = (total_sg > (*vq).__slate_anon_13.split.vring.num && !(*vq).indirect) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("530: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 530b - ., 3; .popsection\n# {0}", const 530 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_virtqueue_add_split_55, sym __slate_asm_string_virtqueue_add_split_56, const 688 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("531: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 531b - ., 4; .popsection\n# {0}", const 531 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
}
if !(*__slate_slot_52).is_null() {
// Use a single buffer which doesn't continue
*__slate_slot_43 = true;
// Set up rest to use this indirect table.
*__slate_slot_51 = 0;
*__slate_slot_48 = 1;
*__slate_slot_54 = (*__slate_slot_52).wrapping_offset(total_sg as isize) as *mut vring_desc_extra;
} else {
*__slate_slot_43 = false;
*__slate_slot_52 = (*vq).__slate_anon_13.split.vring.desc;
*__slate_slot_54 = (*vq).__slate_anon_13.split.desc_extra;
*__slate_slot_51 = *__slate_slot_44 as u32;
*__slate_slot_48 = total_sg;
}
if unlikely!((*vq).vq.num_free < *__slate_slot_48) != 0 {
// FIXME: for historical reasons, we force a notify here if
// there are outgoing parts to the buffer.  Presumably the
// host should service the ring ASAP.
if out_sgs != 0 {
unsafe { (*vq).notify }.unwrap()(core::ptr::addr_of_mut!((*vq).vq));
}
if *__slate_slot_43 {
kfree(*__slate_slot_52 as *const ());
}
return 28i32.wrapping_neg();
} else {
*__slate_slot_50 = 0;
'__join_9: {
while *__slate_slot_50 < out_sgs {
*__slate_slot_53 = *sgs.wrapping_offset(*__slate_slot_50 as isize);
while !(*__slate_slot_53).is_null() {
let __slate_auto_bytes_1 = __slate_slot_38 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_37 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
flags_1 = 0;
let __v34: u32 = sg_count.wrapping_add(1);
sg_count = __v34;
if __v34 != total_sg {
flags_1 = ((flags_1 as i32) | 1) as u16;
}
if vring_map_one_sg(vq as *const vring_virtqueue, *__slate_slot_53, 1u32, &mut (*core::ptr::addr_of_mut!(*__slate_slot_38)), &mut (*core::ptr::addr_of_mut!(*__slate_slot_37)), premapped, attr) != 0 {
break '__join_9;
} else {
// Note that we trust indirect descriptor
// table since it use stream DMA mapping.
*__slate_slot_51 = virtqueue_add_desc_split(vq, *__slate_slot_52, *__slate_slot_54, *__slate_slot_51, *__slate_slot_38, *__slate_slot_37, flags_1, premapped);
*__slate_slot_53 = sg_next(*__slate_slot_53);
}
}
let __v31: u32 = *__slate_slot_50;
*__slate_slot_50 = __v31.wrapping_add(1);
}
while *__slate_slot_50 < out_sgs.wrapping_add(in_sgs) {
*__slate_slot_53 = *sgs.wrapping_offset(*__slate_slot_50 as isize);
while !(*__slate_slot_53).is_null() {
let __slate_auto_bytes_12 = __slate_slot_29 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_12.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_28 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
flags = 2;
let __v25: u32 = sg_count.wrapping_add(1);
sg_count = __v25;
if __v25 != total_sg {
flags = ((flags as i32) | 1) as u16;
}
if vring_map_one_sg(vq as *const vring_virtqueue, *__slate_slot_53, 2u32, &mut (*core::ptr::addr_of_mut!(*__slate_slot_29)), &mut (*core::ptr::addr_of_mut!(*__slate_slot_28)), premapped, attr) != 0 {
break '__join_9;
} else {
// Note that we trust indirect descriptor
// table since it use stream DMA mapping.
*__slate_slot_51 = virtqueue_add_desc_split(vq, *__slate_slot_52, *__slate_slot_54, *__slate_slot_51, *__slate_slot_29, *__slate_slot_28, flags, premapped);
total_in_len = total_in_len.wrapping_add(*__slate_slot_28);
*__slate_slot_53 = sg_next(*__slate_slot_53);
}
}
let __v20: u32 = *__slate_slot_50;
*__slate_slot_50 = __v20.wrapping_add(1);
}
if *__slate_slot_43 {
addr = vring_map_single(vq as *const vring_virtqueue, *__slate_slot_52 as *mut (), (total_sg as u64).wrapping_mul(core::mem::size_of::<vring_desc>() as u64), 1u32);
if vring_mapping_error(vq as *const vring_virtqueue, addr) != 0 {
break '__join_9;
} else {
virtqueue_add_desc_split(vq, (*vq).__slate_anon_13.split.vring.desc, (*vq).__slate_anon_13.split.desc_extra, *__slate_slot_44 as u32, addr, (total_sg as u64).wrapping_mul(core::mem::size_of::<vring_desc>() as u64) as u32, 4u16, false);
}
}
// We're using some buffers from the free list.
let __v16: u32 = (*vq).vq.num_free;
(*vq).vq.num_free = __v16.wrapping_sub(*__slate_slot_48);
// Update free pointer
if virtqueue_is_in_order(vq as *const vring_virtqueue) {
let __v13: u32 = (*vq).free_head;
(*vq).free_head = __v13.wrapping_add(*__slate_slot_48);
if (*vq).free_head >= (*vq).__slate_anon_13.split.vring.num {
let __v10: u32 = (*vq).free_head;
(*vq).free_head = __v10.wrapping_sub((*vq).__slate_anon_13.split.vring.num);
}
(*unsafe { (*vq).__slate_anon_13.split.desc_state }.wrapping_offset(*__slate_slot_44 as isize)).total_in_len = total_in_len;
} else {
if *__slate_slot_43 {
(*vq).free_head = (*unsafe { (*vq).__slate_anon_13.split.desc_extra }.wrapping_offset(*__slate_slot_44 as isize)).next as u32;
} else {
(*vq).free_head = *__slate_slot_51;
}
}
// Store token and indirect buffer state.
(*unsafe { (*vq).__slate_anon_13.split.desc_state }.wrapping_offset(*__slate_slot_44 as isize)).data = data;
if *__slate_slot_43 {
(*unsafe { (*vq).__slate_anon_13.split.desc_state }.wrapping_offset(*__slate_slot_44 as isize)).indir_desc = *__slate_slot_52;
} else {
(*unsafe { (*vq).__slate_anon_13.split.desc_state }.wrapping_offset(*__slate_slot_44 as isize)).indir_desc = ctx as *mut vring_desc;
}
// Put entry in available array (but don't update avail->idx until they
// do sync).
*__slate_slot_49 = (((*vq).__slate_anon_13.split.avail_idx_shadow as i32) as u32) & unsafe { (*vq).__slate_anon_13.split.vring.num }.wrapping_sub(1);
*(core::ptr::addr_of_mut!((*(*vq).__slate_anon_13.split.vring.avail).ring) as *mut u16).wrapping_offset(*__slate_slot_49 as isize) = cpu_to_virtio16((*vq).vq.vdev, *__slate_slot_44 as u16);
// Descriptors and available array need to be set before we expose the
// new available array entries.
virtio_wmb((*vq).weak_barriers);
(*vq).__slate_anon_13.split.avail_idx_shadow = ((*vq).__slate_anon_13.split.avail_idx_shadow as i32).wrapping_add(1) as u16;
(*(*vq).__slate_anon_13.split.vring.avail).idx = cpu_to_virtio16((*vq).vq.vdev, (*vq).__slate_anon_13.split.avail_idx_shadow);
let __v4: u32 = (*vq).num_added;
(*vq).num_added = __v4.wrapping_add(1);
// This is very unlikely, but theoretically possible.  Kick
// just in case.
if unlikely!((*vq).num_added == 65535u32) != 0 {
virtqueue_kick(core::ptr::addr_of_mut!((*vq).vq));
}
return 0;
}
*__slate_slot_47 = *__slate_slot_51;
if *__slate_slot_43 {
*__slate_slot_51 = 0;
} else {
*__slate_slot_51 = *__slate_slot_44 as u32;
}
*__slate_slot_50 = 0;
while *__slate_slot_50 < total_sg {
if *__slate_slot_51 == *__slate_slot_47 {
break;
} else {
*__slate_slot_51 = vring_unmap_one_split(vq as *const vring_virtqueue, (*__slate_slot_54).wrapping_offset(*__slate_slot_51 as isize));
let __v1: u32 = *__slate_slot_50;
*__slate_slot_50 = __v1.wrapping_add(1);
}
}
if *__slate_slot_43 {
kfree(*__slate_slot_52 as *const ());
}
return 12i32.wrapping_neg();
}
}
core::arch::asm!("529: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 529b - ., 3; .popsection\n# {0}", const 529 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_virtqueue_add_split_57, sym __slate_asm_string_virtqueue_add_split_58, const 680 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
}
core::arch::asm!("528: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 528b - ., 3; .popsection\n# {0}", const 528 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_virtqueue_add_split_59, sym __slate_asm_string_virtqueue_add_split_60, const 671 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
core::arch::asm!("527: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 527b - ., 3; .popsection\n# {0}", const 527 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_virtqueue_add_split_61, sym __slate_asm_string_virtqueue_add_split_62, const 670 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
unsafe { core::mem::zeroed() }
}

#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_kick_prepare_split")]
extern "C" fn virtqueue_kick_prepare_split(vq: *mut vring_virtqueue) -> bool {
let mut new: u16 = 0;
let mut old: u16 = 0;
let mut needs_kick: bool = false;
// We need to expose available array entries before checking avail
// event.
virtio_mb(unsafe { (*vq).weak_barriers });
old = (((unsafe { (*vq).__slate_anon_13.split.avail_idx_shadow }) as i32) as u32).wrapping_sub(unsafe { (*vq).num_added }) as u16;
new = unsafe { (*vq).__slate_anon_13.split.avail_idx_shadow };
unsafe {
(*vq).num_added = 0;
}
if unsafe { (*vq).event } {
needs_kick = vring_need_event(vring_read_split_avail_event(vq as *const vring_virtqueue), new, old) != 0;
} else {
needs_kick = ((unsafe { (*(*vq).__slate_anon_13.split.vring.used).flags }) as i32) & (cpu_to_virtio16(unsafe { (*vq).vq.vdev }, 1u16) as i32) == 0;
}
needs_kick
}

fn detach_indirect_split(vq: *mut vring_virtqueue, head: u32) {
let mut extra: *mut vring_desc_extra = unsafe { (*vq).__slate_anon_13.split.desc_extra };
let indir_desc: *mut vring_desc = unsafe { (*unsafe { (*vq).__slate_anon_13.split.desc_state }.wrapping_offset(head as isize)).indir_desc };
let mut len: u32 = 0;
let mut num: u32 = 0;
// Free the indirect table, if any, now that it's unmapped.
if indir_desc.is_null() {
return;
}
len = unsafe { (*unsafe { (*vq).__slate_anon_13.split.desc_extra }.wrapping_offset(head as isize)).len };
loop {
if unlikely!(((unsafe { (*unsafe { (*vq).__slate_anon_13.split.desc_extra }.wrapping_offset(head as isize)).flags }) as i32) & 4 == 0) != 0 {
loop {
unsafe {
core::arch::asm!("532: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 532b - ., 3; .popsection\n# {0}", const 532 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_detach_indirect_split_0, sym __slate_asm_string_detach_indirect_split_1, const 886 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
if unlikely!(len == 0 || (len as u64) % (core::mem::size_of::<vring_desc>() as u64) != 0) != 0 {
loop {
unsafe {
core::arch::asm!("533: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 533b - ., 3; .popsection\n# {0}", const 533 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_detach_indirect_split_5, sym __slate_asm_string_detach_indirect_split_6, const 887 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
num = ((len as u64) / (core::mem::size_of::<vring_desc>() as u64)) as u32;
extra = indir_desc.wrapping_offset(num as isize) as *mut vring_desc_extra;
if unsafe { (*vq).use_map_api } {
for j in 0..num {
vring_unmap_one_split(vq as *const vring_virtqueue, extra.wrapping_offset(j as isize));
}
}
unsafe { kfree(indir_desc as *const ()) };
unsafe {
(*unsafe { (*vq).__slate_anon_13.split.desc_state }.wrapping_offset(head as isize)).indir_desc = core::ptr::null_mut::<vring_desc>();
}
}

fn detach_buf_split_in_order(vq: *mut vring_virtqueue, head: u32, ctx: *mut *mut ()) -> u32 {
let mut i: u32 = 0;
let nextflag: u16 = cpu_to_virtio16(unsafe { (*vq).vq.vdev }, 1u16);
// Clear data ptr.
unsafe {
(*unsafe { (*vq).__slate_anon_13.split.desc_state }.wrapping_offset(head as isize)).data = core::ptr::null_mut::<()>();
}
let extra: *mut vring_desc_extra = unsafe { (*vq).__slate_anon_13.split.desc_extra };
// Put back on free list: unmap first-level descriptors and find end
i = head;
while ((unsafe { (*unsafe { (*vq).__slate_anon_13.split.vring.desc }.wrapping_offset(i as isize)).flags }) as i32) & (nextflag as i32) != 0 {
i = vring_unmap_one_split(vq as *const vring_virtqueue, extra.wrapping_offset(i as isize));
let __v1: u32 = unsafe { (*vq).vq.num_free };
unsafe {
(*vq).vq.num_free = __v1.wrapping_add(1);
}
}
vring_unmap_one_split(vq as *const vring_virtqueue, extra.wrapping_offset(i as isize));
// Plus final descriptor
let __v5: u32 = unsafe { (*vq).vq.num_free };
unsafe {
(*vq).vq.num_free = __v5.wrapping_add(1);
}
if unsafe { (*vq).indirect } {
detach_indirect_split(vq, head);
} else {
if !ctx.is_null() {
unsafe {
*ctx = (*unsafe { (*vq).__slate_anon_13.split.desc_state }.wrapping_offset(head as isize)).indir_desc as *mut ();
}
}
}
i
}

fn detach_buf_split(vq: *mut vring_virtqueue, head: u32, ctx: *mut *mut ()) {
let i: u32 = detach_buf_split_in_order(vq, head, ctx);
unsafe {
(*unsafe { (*vq).__slate_anon_13.split.desc_extra }.wrapping_offset(i as isize)).next = (*vq).free_head as u16;
}
unsafe {
(*vq).free_head = head;
}
}

#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_poll_split")]
extern "C" fn virtqueue_poll_split(vq: *const vring_virtqueue, last_used_idx: u32) -> bool {
((last_used_idx as u16) as i32) != (vring_read_split_used_idx(vq) as i32)
}

#[unsafe(link_section = ".text.slate_distinct.virtio_ring.more_used_split")]
extern "C" fn more_used_split(vq: *const vring_virtqueue) -> bool {
virtqueue_poll_split(vq, (unsafe { (*vq).last_used_idx }) as u32)
}

#[unsafe(link_section = ".text.slate_distinct.virtio_ring.more_used_split_in_order")]
extern "C" fn more_used_split_in_order(vq: *const vring_virtqueue) -> bool {
if (unsafe { (*vq).batch_last.id }) != !0 {
return true;
}
virtqueue_poll_split(vq, (unsafe { (*vq).last_used_idx }) as u32)
}

#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_get_buf_ctx_split")]
extern "C" fn virtqueue_get_buf_ctx_split(vq: *mut vring_virtqueue, len: *mut u32, ctx: *mut *mut ()) -> *mut () {
let mut i: u32 = 0;
let mut last_used: u16 = 0;
if unlikely!(unsafe { (*vq).broken }) != 0 {
return core::ptr::null_mut::<()>();
}
if !more_used_split(vq as *const vring_virtqueue) {
let mut __v0: i32;
if false {
unsafe { _printk(c"\x017No more buffers in queue\n".as_ptr() as *const u8) };
}
__v0 = 0;
return core::ptr::null_mut::<()>();
}
// Only get used array entries after they have been exposed by host.
virtio_rmb(unsafe { (*vq).weak_barriers });
last_used = ((((unsafe { (*vq).last_used_idx }) as i32) as u32) & unsafe { (*vq).__slate_anon_13.split.vring.num }.wrapping_sub(1)) as u16;
i = vring_read_split_used_id(vq as *const vring_virtqueue, last_used);
unsafe {
*len = vring_read_split_used_len(vq as *const vring_virtqueue, last_used);
}
if unlikely!(i >= unsafe { (*vq).__slate_anon_13.split.vring.num }) != 0 {
loop {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*(*vq).vq.vdev).dev) as *const device, c"%s:id %u out of range\n".as_ptr() as *const u8, (*vq).vq.name, i) };
unsafe {
(*vq).broken = true;
}
if true {
break;
}
}
return core::ptr::null_mut::<()>();
}
if unlikely!(unsafe { (*unsafe { (*vq).__slate_anon_13.split.desc_state }.wrapping_offset(i as isize)).data }.is_null()) != 0 {
loop {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*(*vq).vq.vdev).dev) as *const device, c"%s:id %u is not a head!\n".as_ptr() as *const u8, (*vq).vq.name, i) };
unsafe {
(*vq).broken = true;
}
if true {
break;
}
}
return core::ptr::null_mut::<()>();
}
// detach_buf_split clears data, so grab it now.
let ret: *mut () = unsafe { (*unsafe { (*vq).__slate_anon_13.split.desc_state }.wrapping_offset(i as isize)).data };
detach_buf_split(vq, i, ctx);
unsafe {
(*vq).last_used_idx = ((*vq).last_used_idx as i32).wrapping_add(1) as u16;
}
// If we expect an interrupt for the next entry, tell host
// by writing event index and flush out the write before
// the read in the next get_buf call.
if ((unsafe { (*vq).__slate_anon_13.split.avail_flags_shadow }) as i32) & 1 == 0 {
loop {
if unsafe { (*vq).weak_barriers } {
loop {
kcsan_mb!();
loop {
let mut __v9: u16;
let mut __ai_ptr: *mut u16 = unsafe { core::ptr::addr_of_mut!((*(*core::ptr::addr_of_mut!((*vq).__slate_anon_13.split.vring)).avail).ring) as *mut u16 }.wrapping_offset((unsafe { (*core::ptr::addr_of_mut!((*vq).__slate_anon_13.split.vring)).num }) as isize);
kcsan_mb!();
instrument_atomic_read_write(__ai_ptr as *const (), core::mem::size_of::<u16>() as u64);
let mut __v11: u16;
let mut __ret: u16 = cpu_to_virtio16(unsafe { (*vq).vq.vdev }, unsafe { (*vq).last_used_idx });
match (const { core::mem::size_of::<u16>() }) as u64 {
1 => {
unsafe {
let __t0: u16;
core::arch::asm!("xchgb {0:l}, ({1:r})", inlateout(reg) __ret as u16 => __t0, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u16, u16>(__t0 as u16) };
}
}
2 => {
unsafe {
let __t1: u16;
core::arch::asm!("xchgw {0:x}, ({1:r})", inlateout(reg) __ret as u16 => __t1, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
}
4 => {
unsafe {
let __t2: u16;
core::arch::asm!("xchgl {0:x}, ({1:r})", inlateout(reg) __ret as u16 => __t2, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u16, u16>(__t2 as u16) };
}
}
8 => {
unsafe {
let __t3: u16;
core::arch::asm!("xchgq {0:r}, ({1:r})", inlateout(reg) __ret as u16 => __t3, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u16, u16>(__t3 as u16) };
}
}
_ => {
unsafe { __xchg_wrong_size() };
}
}
__v11 = __ret;
__v9 = __v11;
if true {
break;
}
}
if true {
break;
}
}
} else {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u16>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_534() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__2!(*unsafe { core::ptr::addr_of_mut!((*(*core::ptr::addr_of_mut!((*vq).__slate_anon_13.split.vring)).avail).ring) as *mut u16 }.wrapping_offset((unsafe { (*core::ptr::addr_of_mut!((*vq).__slate_anon_13.split.vring)).num }) as isize), cpu_to_virtio16(unsafe { (*vq).vq.vdev }, unsafe { (*vq).last_used_idx }));
if true {
break;
}
}
loop {
kcsan_mb!();
unsafe {
core::arch::asm!("mfence", options(nostack, att_syntax));
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
ret
}

#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_get_buf_ctx_split_in_order")]
extern "C" fn virtqueue_get_buf_ctx_split_in_order(vq: *mut vring_virtqueue, len: *mut u32, ctx: *mut *mut ()) -> *mut () {
let num: u32 = unsafe { (*vq).__slate_anon_13.split.vring.num };
let num_free: u32 = unsafe { (*vq).vq.num_free };
let mut last_used: u16 = 0;
let mut last_used_idx: u16 = 0;
if unlikely!(unsafe { (*vq).broken }) != 0 {
return core::ptr::null_mut::<()>();
}
last_used = ((((unsafe { (*vq).last_used }) as i32) as u32) & num.wrapping_sub(1)) as u16;
last_used_idx = ((((unsafe { (*vq).last_used_idx }) as i32) as u32) & num.wrapping_sub(1)) as u16;
if (unsafe { (*vq).batch_last.id }) == !0 {
if !more_used_split_in_order(vq as *const vring_virtqueue) {
let mut __v0: i32;
if false {
unsafe { _printk(c"\x017No more buffers in queue\n".as_ptr() as *const u8) };
}
__v0 = 0;
return core::ptr::null_mut::<()>();
}
// Only get used array entries after they have been
// exposed by host.
virtio_rmb(unsafe { (*vq).weak_barriers });
unsafe {
(*vq).batch_last.id = vring_read_split_used_id(vq as *const vring_virtqueue, last_used_idx);
}
unsafe {
(*vq).batch_last.len = vring_read_split_used_len(vq as *const vring_virtqueue, last_used_idx);
}
}
if (unsafe { (*vq).batch_last.id }) == ((last_used as i32) as u32) {
unsafe {
(*vq).batch_last.id = !0;
}
unsafe {
*len = (*vq).batch_last.len;
}
} else {
unsafe {
*len = (*unsafe { (*vq).__slate_anon_13.split.desc_state }.wrapping_offset((last_used as i32) as isize)).total_in_len;
}
}
if unlikely!(unsafe { (*unsafe { (*vq).__slate_anon_13.split.desc_state }.wrapping_offset((last_used as i32) as isize)).data }.is_null()) != 0 {
loop {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*(*vq).vq.vdev).dev) as *const device, c"%s:id %u is not a head!\n".as_ptr() as *const u8, (*vq).vq.name, last_used as i32) };
unsafe {
(*vq).broken = true;
}
if true {
break;
}
}
return core::ptr::null_mut::<()>();
}
// detach_buf_split clears data, so grab it now.
let ret: *mut () = unsafe { (*unsafe { (*vq).__slate_anon_13.split.desc_state }.wrapping_offset((last_used as i32) as isize)).data };
detach_buf_split_in_order(vq, last_used as u32, ctx);
unsafe {
(*vq).last_used_idx = ((*vq).last_used_idx as i32).wrapping_add(1) as u16;
}
unsafe {
(*vq).last_used = (((*vq).last_used as i32) as u32).wrapping_add(unsafe { (*vq).vq.num_free }.wrapping_sub(num_free)) as u16;
}
// If we expect an interrupt for the next entry, tell host
// by writing event index and flush out the write before
// the read in the next get_buf call.
if ((unsafe { (*vq).__slate_anon_13.split.avail_flags_shadow }) as i32) & 1 == 0 {
loop {
if unsafe { (*vq).weak_barriers } {
loop {
kcsan_mb!();
loop {
let mut __v10: u16;
let mut __ai_ptr: *mut u16 = unsafe { core::ptr::addr_of_mut!((*(*core::ptr::addr_of_mut!((*vq).__slate_anon_13.split.vring)).avail).ring) as *mut u16 }.wrapping_offset((unsafe { (*core::ptr::addr_of_mut!((*vq).__slate_anon_13.split.vring)).num }) as isize);
kcsan_mb!();
instrument_atomic_read_write(__ai_ptr as *const (), core::mem::size_of::<u16>() as u64);
let mut __v12: u16;
let mut __ret: u16 = cpu_to_virtio16(unsafe { (*vq).vq.vdev }, unsafe { (*vq).last_used_idx });
match (const { core::mem::size_of::<u16>() }) as u64 {
1 => {
unsafe {
let __t0: u16;
core::arch::asm!("xchgb {0:l}, ({1:r})", inlateout(reg) __ret as u16 => __t0, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u16, u16>(__t0 as u16) };
}
}
2 => {
unsafe {
let __t1: u16;
core::arch::asm!("xchgw {0:x}, ({1:r})", inlateout(reg) __ret as u16 => __t1, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
}
4 => {
unsafe {
let __t2: u16;
core::arch::asm!("xchgl {0:x}, ({1:r})", inlateout(reg) __ret as u16 => __t2, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u16, u16>(__t2 as u16) };
}
}
8 => {
unsafe {
let __t3: u16;
core::arch::asm!("xchgq {0:r}, ({1:r})", inlateout(reg) __ret as u16 => __t3, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u16, u16>(__t3 as u16) };
}
}
_ => {
unsafe { __xchg_wrong_size() };
}
}
__v12 = __ret;
__v10 = __v12;
if true {
break;
}
}
if true {
break;
}
}
} else {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u16>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_535() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__2!(*unsafe { core::ptr::addr_of_mut!((*(*core::ptr::addr_of_mut!((*vq).__slate_anon_13.split.vring)).avail).ring) as *mut u16 }.wrapping_offset((unsafe { (*core::ptr::addr_of_mut!((*vq).__slate_anon_13.split.vring)).num }) as isize), cpu_to_virtio16(unsafe { (*vq).vq.vdev }, unsafe { (*vq).last_used_idx }));
if true {
break;
}
}
loop {
kcsan_mb!();
unsafe {
core::arch::asm!("mfence", options(nostack, att_syntax));
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
ret
}

#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_disable_cb_split")]
extern "C" fn virtqueue_disable_cb_split(vq: *mut vring_virtqueue) {
if ((unsafe { (*vq).__slate_anon_13.split.avail_flags_shadow }) as i32) & 1 == 0 {
unsafe {
(*vq).__slate_anon_13.split.avail_flags_shadow = (((*vq).__slate_anon_13.split.avail_flags_shadow as i32) | 1) as u16;
}
// If device triggered an event already it won't trigger one again:
// no need to disable.
if unsafe { (*vq).event_triggered } {
return;
}
if unsafe { (*vq).event } {
unsafe {
*(core::ptr::addr_of_mut!((*(*core::ptr::addr_of_mut!((*vq).__slate_anon_13.split.vring)).avail).ring) as *mut u16).wrapping_offset((*core::ptr::addr_of_mut!((*vq).__slate_anon_13.split.vring)).num as isize) = 0;
}
} else {
unsafe {
(*(*vq).__slate_anon_13.split.vring.avail).flags = cpu_to_virtio16((*vq).vq.vdev, (*vq).__slate_anon_13.split.avail_flags_shadow);
}
}
// TODO: this is a hack. Figure out a cleaner value to write.
}
}

#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_enable_cb_prepare_split")]
extern "C" fn virtqueue_enable_cb_prepare_split(vq: *mut vring_virtqueue) -> u32 {
let mut last_used_idx: u16 = 0;
// We optimistically turn back on interrupts, then check if there was
// more to do.
//
// Depending on the VIRTIO_RING_F_EVENT_IDX feature, we need to
// either clear the flags bit or point the event index at the next
// entry. Always do both to keep code simple.
if ((unsafe { (*vq).__slate_anon_13.split.avail_flags_shadow }) as i32) & 1 != 0 {
let __v2: u16 = (((unsafe { (*vq).__slate_anon_13.split.avail_flags_shadow }) as i32) & !1) as u16;
unsafe {
(*vq).__slate_anon_13.split.avail_flags_shadow = __v2;
}
if !unsafe { (*vq).event } {
unsafe {
(*(*vq).__slate_anon_13.split.vring.avail).flags = cpu_to_virtio16((*vq).vq.vdev, (*vq).__slate_anon_13.split.avail_flags_shadow);
}
}
}
let __v3: u16 = unsafe { (*vq).last_used_idx };
last_used_idx = __v3;
unsafe {
*(core::ptr::addr_of_mut!((*(*core::ptr::addr_of_mut!((*vq).__slate_anon_13.split.vring)).avail).ring) as *mut u16).wrapping_offset((*core::ptr::addr_of_mut!((*vq).__slate_anon_13.split.vring)).num as isize) = cpu_to_virtio16((*vq).vq.vdev, __v3);
}
last_used_idx as u32
}

#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_enable_cb_delayed_split")]
extern "C" fn virtqueue_enable_cb_delayed_split(vq: *mut vring_virtqueue) -> bool {
let mut bufs: u16 = 0;
// We optimistically turn back on interrupts, then check if there was
// more to do.
//
// Depending on the VIRTIO_RING_F_USED_EVENT_IDX feature, we need to
// either clear the flags bit or point the event index at the next
// entry. Always update the event index to keep code simple.
if ((unsafe { (*vq).__slate_anon_13.split.avail_flags_shadow }) as i32) & 1 != 0 {
let __v2: u16 = (((unsafe { (*vq).__slate_anon_13.split.avail_flags_shadow }) as i32) & !1) as u16;
unsafe {
(*vq).__slate_anon_13.split.avail_flags_shadow = __v2;
}
if !unsafe { (*vq).event } {
unsafe {
(*(*vq).__slate_anon_13.split.vring.avail).flags = cpu_to_virtio16((*vq).vq.vdev, (*vq).__slate_anon_13.split.avail_flags_shadow);
}
}
}
// TODO: tune this threshold
bufs = (((((unsafe { (*vq).__slate_anon_13.split.avail_idx_shadow }) as i32).wrapping_sub((unsafe { (*vq).last_used_idx }) as i32) as u16) as i32).wrapping_mul(3) / 4) as u16;
loop {
if unsafe { (*vq).weak_barriers } {
loop {
kcsan_mb!();
loop {
let mut __v4: u16;
let mut __ai_ptr: *mut u16 = unsafe { core::ptr::addr_of_mut!((*(*core::ptr::addr_of_mut!((*vq).__slate_anon_13.split.vring)).avail).ring) as *mut u16 }.wrapping_offset((unsafe { (*core::ptr::addr_of_mut!((*vq).__slate_anon_13.split.vring)).num }) as isize);
kcsan_mb!();
instrument_atomic_read_write(__ai_ptr as *const (), core::mem::size_of::<u16>() as u64);
let mut __v6: u16;
let mut __ret: u16 = cpu_to_virtio16(unsafe { (*vq).vq.vdev }, ((unsafe { (*vq).last_used_idx }) as i32).wrapping_add(bufs as i32) as u16);
match (const { core::mem::size_of::<u16>() }) as u64 {
1 => {
unsafe {
let __t0: u16;
core::arch::asm!("xchgb {0:l}, ({1:r})", inlateout(reg) __ret as u16 => __t0, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u16, u16>(__t0 as u16) };
}
}
2 => {
unsafe {
let __t1: u16;
core::arch::asm!("xchgw {0:x}, ({1:r})", inlateout(reg) __ret as u16 => __t1, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
}
4 => {
unsafe {
let __t2: u16;
core::arch::asm!("xchgl {0:x}, ({1:r})", inlateout(reg) __ret as u16 => __t2, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u16, u16>(__t2 as u16) };
}
}
8 => {
unsafe {
let __t3: u16;
core::arch::asm!("xchgq {0:r}, ({1:r})", inlateout(reg) __ret as u16 => __t3, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u16, u16>(__t3 as u16) };
}
}
_ => {
unsafe { __xchg_wrong_size() };
}
}
__v6 = __ret;
__v4 = __v6;
if true {
break;
}
}
if true {
break;
}
}
} else {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u16>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_536() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__2!(*unsafe { core::ptr::addr_of_mut!((*(*core::ptr::addr_of_mut!((*vq).__slate_anon_13.split.vring)).avail).ring) as *mut u16 }.wrapping_offset((unsafe { (*core::ptr::addr_of_mut!((*vq).__slate_anon_13.split.vring)).num }) as isize), cpu_to_virtio16(unsafe { (*vq).vq.vdev }, ((unsafe { (*vq).last_used_idx }) as i32).wrapping_add(bufs as i32) as u16));
if true {
break;
}
}
loop {
kcsan_mb!();
unsafe {
core::arch::asm!("mfence", options(nostack, att_syntax));
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
if unlikely!((((vring_read_split_used_idx(vq as *const vring_virtqueue) as i32).wrapping_sub((unsafe { (*vq).last_used_idx }) as i32) as u16) as i32) > (bufs as i32)) != 0 {
return false;
}
true
}

#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_detach_unused_buf_split")]
extern "C" fn virtqueue_detach_unused_buf_split(vq: *mut vring_virtqueue) -> *mut () {
let mut i: u32 = 0;
while i < unsafe { (*vq).__slate_anon_13.split.vring.num } {
if !unsafe { (*unsafe { (*vq).__slate_anon_13.split.desc_state }.wrapping_offset(i as isize)).data }.is_null() {
// detach_buf_split clears data, so grab it now.
let buf: *mut () = unsafe { (*unsafe { (*vq).__slate_anon_13.split.desc_state }.wrapping_offset(i as isize)).data };
if virtqueue_is_in_order(vq as *const vring_virtqueue) {
detach_buf_split_in_order(vq, i, core::ptr::null_mut::<*mut ()>());
} else {
detach_buf_split(vq, i, core::ptr::null_mut::<*mut ()>());
}
unsafe {
(*vq).__slate_anon_13.split.avail_idx_shadow = ((*vq).__slate_anon_13.split.avail_idx_shadow as i32).wrapping_sub(1) as u16;
}
unsafe {
(*(*vq).__slate_anon_13.split.vring.avail).idx = cpu_to_virtio16((*vq).vq.vdev, (*vq).__slate_anon_13.split.avail_idx_shadow);
}
return buf;
}
i = i.wrapping_add(1);
}
// That should have freed everything.
loop {
if unlikely!((unsafe { (*vq).vq.num_free }) != unsafe { (*vq).__slate_anon_13.split.vring.num }) != 0 {
loop {
unsafe {
core::arch::asm!("537: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 537b - ., 3; .popsection\n# {0}", const 537 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_virtqueue_detach_unused_buf_split_6, sym __slate_asm_string_virtqueue_detach_unused_buf_split_7, const 1191 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
core::ptr::null_mut::<()>()
}

fn virtqueue_vring_init_split(vring_split: *mut vring_virtqueue_split, vq: *mut vring_virtqueue) {
let vdev: *mut virtio_device = unsafe { (*vq).vq.vdev };
unsafe {
(*vring_split).avail_flags_shadow = 0;
}
unsafe {
(*vring_split).avail_idx_shadow = 0;
}
// No callback?  Tell other side not to bother us.
if (unsafe { (*vq).vq.callback }) == None {
unsafe {
(*vring_split).avail_flags_shadow = (((*vring_split).avail_flags_shadow as i32) | 1) as u16;
}
if !unsafe { (*vq).event } {
unsafe {
(*(*vring_split).vring.avail).flags = cpu_to_virtio16(vdev, (*vring_split).avail_flags_shadow);
}
}
}
}

#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_reset_split")]
extern "C" fn virtqueue_reset_split(vq: *mut vring_virtqueue) {
let num: i32 = (unsafe { (*vq).__slate_anon_13.split.vring.num }) as i32;
unsafe {
(*(*vq).__slate_anon_13.split.vring.avail).flags = 0;
}
unsafe {
(*(*vq).__slate_anon_13.split.vring.avail).idx = 0;
}
// reset avail event
unsafe {
*(core::ptr::addr_of_mut!((*(*vq).__slate_anon_13.split.vring.avail).ring) as *mut u16).wrapping_offset(num as isize) = 0;
}
unsafe {
(*(*vq).__slate_anon_13.split.vring.used).flags = 0;
}
unsafe {
(*(*vq).__slate_anon_13.split.vring.used).idx = 0;
}
// reset used event
unsafe {
*((core::ptr::addr_of_mut!((*(*vq).__slate_anon_13.split.vring.used).ring) as *mut vring_used_elem).wrapping_offset(num as isize) as *mut u16) = 0;
}
virtqueue_init(vq, num as u32);
virtqueue_vring_init_split(unsafe { core::ptr::addr_of_mut!((*vq).__slate_anon_13.split) }, vq);
}

fn virtqueue_vring_attach_split(vq: *mut vring_virtqueue, vring_split: *mut vring_virtqueue_split) {
unsafe {
(*vq).__slate_anon_13.split = *vring_split;
}
// Put everything in free lists.
unsafe {
(*vq).free_head = 0;
}
unsafe {
(*vq).batch_last.id = !0;
}
}

fn vring_alloc_state_extra_split(vring_split: *mut vring_virtqueue_split) -> i32 {
let mut __slate_compound_9: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_11: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut num: u32 = unsafe { core::mem::zeroed() };
let mut extra: core::mem::MaybeUninit<*mut vring_desc_extra> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut vring_desc_extra = core::ptr::addr_of_mut!(extra) as *mut *mut vring_desc_extra;
let mut state: core::mem::MaybeUninit<*mut vring_desc_state_split> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut vring_desc_state_split = core::ptr::addr_of_mut!(state) as *mut *mut vring_desc_state_split;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
num = (*vring_split).vring.num;
__obj_size = size_mul(core::mem::size_of::<vring_desc_state_split>() as u64, num as u64);
let __slate_auto_bytes_3 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_0 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_1 = _kmalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_9 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_9)
});
alloc_tag_restore!();
} else {
*__slate_slot_1 = _kmalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_11 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_11)
});
}
*__slate_slot_8 = *__slate_slot_1 as *mut vring_desc_state_split;
if !(*__slate_slot_8).is_null() {
*__slate_slot_7 = vring_alloc_desc_extra(num);
if (*__slate_slot_7).is_null() {
kfree(*__slate_slot_8 as *const ());
} else {
memset(*__slate_slot_8 as *mut (), 0i32, (num as u64).wrapping_mul(core::mem::size_of::<vring_desc_state_split>() as u64));
(*vring_split).desc_state = *__slate_slot_8;
(*vring_split).desc_extra = *__slate_slot_7;
return 0;
}
}
return 12i32.wrapping_neg();
}
}

fn vring_free_split(vring_split: &vring_virtqueue_split, vdev: *mut virtio_device, map: virtio_map) {
vring_free_queue(vdev, unsafe { (*(vring_split as *const vring_virtqueue_split)).queue_size_in_bytes }, (unsafe { (*(vring_split as *const vring_virtqueue_split)).vring.desc }) as *mut (), unsafe { (*(vring_split as *const vring_virtqueue_split)).queue_dma_addr }, map);
unsafe { kfree((*(vring_split as *const vring_virtqueue_split)).desc_state as *const ()) };
unsafe { kfree((*(vring_split as *const vring_virtqueue_split)).desc_extra as *const ()) };
}

fn vring_alloc_queue_split(vring_split: &mut vring_virtqueue_split, vdev: *mut virtio_device, mut num: u32, vring_align: u32, may_reduce_num: bool, map: virtio_map) -> i32 {
let mut queue: *mut () = core::ptr::null_mut::<()>();
let mut dma_addr: u64 = 0;
// We assume num is a power of 2.
if !is_power_of_2(num as u64) {
while false {
}
unsafe { _dev_warn(core::ptr::addr_of_mut!((*vdev).dev) as *const device, c"Bad virtqueue length %u\n".as_ptr() as *const u8, num) };
return 22i32.wrapping_neg();
}
// TODO: allocate each queue chunk individually
loop {
let __v2: bool = num != 0 && (vring_size(num, vring_align as u64) as u64) > _AC!() << 12i32;
if !__v2 {
break;
}
queue = vring_alloc_queue(vdev, vring_size(num, vring_align as u64) as u64, core::ptr::addr_of_mut!(dma_addr), ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32) | (BIT!(13i32) as u32) | (BIT!(8i32) as u32), map);
if !queue.is_null() {
break;
}
if !may_reduce_num {
return 12i32.wrapping_neg();
}
num /= 2;
}
if num == 0 {
return 12i32.wrapping_neg();
}
if queue.is_null() {
// Try to get a single page. You are my only hope!
queue = vring_alloc_queue(vdev, vring_size(num, vring_align as u64) as u64, core::ptr::addr_of_mut!(dma_addr), ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32) | (BIT!(8i32) as u32), map);
}
if queue.is_null() {
return 12i32.wrapping_neg();
}
vring_init(unsafe { core::ptr::addr_of_mut!((*(vring_split as *mut vring_virtqueue_split)).vring) }, num, queue, vring_align as u64);
unsafe {
(*(vring_split as *mut vring_virtqueue_split)).queue_dma_addr = dma_addr;
}
unsafe {
(*(vring_split as *mut vring_virtqueue_split)).queue_size_in_bytes = vring_size(num, vring_align as u64) as u64;
}
unsafe {
(*(vring_split as *mut vring_virtqueue_split)).vring_align = vring_align;
}
unsafe {
(*(vring_split as *mut vring_virtqueue_split)).may_reduce_num = may_reduce_num;
}
0
}

fn __vring_new_virtqueue_split(index: u32, vring_split: *mut vring_virtqueue_split, vdev: *mut virtio_device, weak_barriers: bool, context: bool, notify: Option<unsafe extern "C" fn(*mut virtqueue) -> bool>, callback: Option<unsafe extern "C" fn(*mut virtqueue)>, name: *const u8, map: virtio_map) -> *mut virtqueue {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut vq: *mut vring_virtqueue = unsafe { core::mem::zeroed() };
let mut err: i32 = 0;
let mut __v0: *mut vring_virtqueue;
let __obj_size: u64 = size_mul(core::mem::size_of::<vring_virtqueue>() as u64, 1u64);
let mut __v1: *mut ();
let mut __v2: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_3 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_3)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_5 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_5)
} });
}
__v2 = _res;
__v1 = __v2;
__v0 = __v1 as *mut vring_virtqueue;
vq = __v0;
if vq.is_null() {
return core::ptr::null_mut::<virtqueue>();
}
unsafe {
(*vq).vq.callback = callback;
}
unsafe {
(*vq).vq.vdev = vdev;
}
unsafe {
(*vq).vq.name = name;
}
unsafe {
(*vq).vq.index = index;
}
unsafe {
(*vq).vq.reset = false;
}
unsafe {
(*vq).we_own_ring = false;
}
unsafe {
(*vq).notify = notify;
}
unsafe {
(*vq).weak_barriers = weak_barriers;
}
unsafe {
(*vq).broken = false;
}
unsafe {
(*vq).map = map;
}
unsafe {
(*vq).use_map_api = vring_use_map_api(vdev as *const virtio_device);
}
unsafe {
(*vq).indirect = virtio_has_feature(vdev as *const virtio_device, 28u32) && !context;
}
unsafe {
(*vq).event = virtio_has_feature(vdev as *const virtio_device, 29u32);
}
unsafe {
(*vq).layout = (if virtio_has_feature(vdev as *const virtio_device, 35u32) { 2i32 } else { 0i32 }) as u32;
}
if virtio_has_feature(vdev as *const virtio_device, 36u32) {
unsafe {
(*vq).weak_barriers = false;
}
}
err = vring_alloc_state_extra_split(vring_split);
if err != 0 {
unsafe { kfree(vq as *const ()) };
return core::ptr::null_mut::<virtqueue>();
}
virtqueue_vring_init_split(vring_split, vq);
virtqueue_init(vq, unsafe { (*vring_split).vring.num });
virtqueue_vring_attach_split(vq, vring_split);
spin_lock(unsafe { core::ptr::addr_of_mut!((*vdev).vqs_list_lock) });
list_add_tail(unsafe { core::ptr::addr_of_mut!((*vq).vq.list) }, unsafe { core::ptr::addr_of_mut!((*vdev).vqs) });
spin_unlock(unsafe { core::ptr::addr_of_mut!((*vdev).vqs_list_lock) });
unsafe { core::ptr::addr_of_mut!((*vq).vq) }
}

fn vring_create_virtqueue_split(index: u32, num: u32, vring_align: u32, vdev: *mut virtio_device, weak_barriers: bool, may_reduce_num: bool, context: bool, notify: Option<unsafe extern "C" fn(*mut virtqueue) -> bool>, callback: Option<unsafe extern "C" fn(*mut virtqueue)>, name: *const u8, map: virtio_map) -> *mut virtqueue {
let mut vring_split: vring_virtqueue_split = {
let mut __t0: vring_virtqueue_split = unsafe { core::mem::zeroed() };
__t0
};
let err: i32 = vring_alloc_queue_split(unsafe { &mut (*core::ptr::addr_of_mut!(vring_split)) }, vdev, num, vring_align, may_reduce_num, map);
if err != 0 {
return core::ptr::null_mut::<virtqueue>();
}
let vq: *mut virtqueue = __vring_new_virtqueue_split(index, core::ptr::addr_of_mut!(vring_split), vdev, weak_barriers, context, notify, callback, name, map);
if vq.is_null() {
vring_free_split(unsafe { &(*core::ptr::addr_of_mut!(vring_split)) }, vdev, map);
return core::ptr::null_mut::<virtqueue>();
}
unsafe {
(*to_vvq!(vq)).we_own_ring = true;
}
vq
}

#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_resize_split")]
extern "C" fn virtqueue_resize_split(vq: *mut vring_virtqueue, num: u32) -> i32 {
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut vdev: *mut virtio_device = unsafe { core::mem::zeroed() };
let mut vring_split: core::mem::MaybeUninit<vring_virtqueue_split> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut vring_virtqueue_split = core::ptr::addr_of_mut!(vring_split) as *mut vring_virtqueue_split;
unsafe {
core::ptr::write(__slate_slot_2, {
let mut __t0: vring_virtqueue_split = core::mem::zeroed();
__t0
});
vdev = (*vq).vq.vdev;
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_0 = vring_alloc_queue_split(&mut (*core::ptr::addr_of_mut!(*__slate_slot_2)), vdev, num, (*vq).__slate_anon_13.split.vring_align, (*vq).__slate_anon_13.split.may_reduce_num, (*vq).map);
if *__slate_slot_0 == 0 {
*__slate_slot_0 = vring_alloc_state_extra_split(core::ptr::addr_of_mut!(*__slate_slot_2));
if *__slate_slot_0 != 0 {
vring_free_split(&(*core::ptr::addr_of_mut!(*__slate_slot_2)), vdev, (*vq).map);
} else {
vring_free(core::ptr::addr_of_mut!((*vq).vq));
virtqueue_vring_init_split(core::ptr::addr_of_mut!(*__slate_slot_2), vq);
virtqueue_init(vq, (*__slate_slot_2).vring.num);
virtqueue_vring_attach_split(vq, core::ptr::addr_of_mut!(*__slate_slot_2));
return 0;
}
}
virtqueue_reset_split(vq);
return 12i32.wrapping_neg();
}
}

/// Packed ring specific functions - *_packed().
fn packed_used_wrap_counter(last_used_idx: u16) -> bool {
(last_used_idx as i32) & 32768i32 != 0
}

fn packed_last_used(last_used_idx: u16) -> u16 {
((last_used_idx as i32) & !32768i32.wrapping_neg()) as u16
}

fn vring_unmap_extra_packed(vq: *const vring_virtqueue, extra: *const vring_desc_extra) {
let flags: u16 = unsafe { (*extra).flags };
if (flags as i32) & 4 != 0 {
if !unsafe { (*vq).use_map_api } {
return;
}
} else {
if !vring_need_unmap_buffer(vq, extra) {
return;
}
}
virtqueue_unmap_page_attrs(unsafe { core::ptr::addr_of!((*vq).vq) }, unsafe { (*extra).addr }, (unsafe { (*extra).len }) as u64, (if (flags as i32) & 2 != 0 { 2i32 } else { 1i32 }) as u32, 0u64);
}

fn alloc_indirect_packed(total_sg: u32, mut gfp: u32) -> *mut vring_packed_desc {
let mut __slate_compound_4: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_6: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
let mut size: i32 = 0;
// We require lowmem mappings for the descriptors because
// otherwise virt_to_phys will give us bogus addresses in the
// virtqueue.
let __v0: u32 = gfp;
gfp = __v0 & !(BIT!(1i32) as u32);
size = (core::mem::size_of::<vring_packed_desc>() as u64).wrapping_add(core::mem::size_of::<vring_desc_extra>() as u64).wrapping_mul(total_sg as u64) as i32;
let mut __v2: *mut ();
let mut __v3: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_noprof(size as u64, gfp, unsafe { *{
__slate_compound_4 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_4)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_noprof(size as u64, gfp, unsafe { *{
__slate_compound_6 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_6)
} });
}
__v3 = _res;
__v2 = __v3;
let desc: *mut vring_packed_desc = __v2 as *mut vring_packed_desc;
if desc.is_null() {
return core::ptr::null_mut::<vring_packed_desc>();
}
let extra: *mut vring_desc_extra = desc.wrapping_offset(total_sg as isize) as *mut vring_desc_extra;
while (i as u32) < total_sg {
unsafe {
(*extra.wrapping_offset(i as isize)).next = i.wrapping_add(1) as u16;
}
i = i.wrapping_add(1);
}
desc
}

fn virtqueue_add_indirect_packed(vq: *mut vring_virtqueue, sgs: *mut *mut scatterlist, total_sg: u32, out_sgs: u32, in_sgs: u32, data: *mut (), premapped: bool, gfp: u32, id: u16, attr: u64) -> i32 {
let mut addr: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut u64 = core::ptr::addr_of_mut!(addr) as *mut u64;
let mut head: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut u16 = core::ptr::addr_of_mut!(head) as *mut u16;
let mut total_in_len: u32 = unsafe { core::mem::zeroed() };
let mut len: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_25: *mut u32 = core::ptr::addr_of_mut!(len) as *mut u32;
let mut err_idx: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_26: *mut u32 = core::ptr::addr_of_mut!(err_idx) as *mut u32;
let mut n: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_27: *mut u32 = core::ptr::addr_of_mut!(n) as *mut u32;
let mut i: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_28: *mut u32 = core::ptr::addr_of_mut!(i) as *mut u32;
let mut sg: core::mem::MaybeUninit<*mut scatterlist> = core::mem::MaybeUninit::uninit();
let __slate_slot_29: *mut *mut scatterlist = core::ptr::addr_of_mut!(sg) as *mut *mut scatterlist;
let mut desc: core::mem::MaybeUninit<*mut vring_packed_desc> = core::mem::MaybeUninit::uninit();
let __slate_slot_30: *mut *mut vring_packed_desc = core::ptr::addr_of_mut!(desc) as *mut *mut vring_packed_desc;
let mut extra: core::mem::MaybeUninit<*mut vring_desc_extra> = core::mem::MaybeUninit::uninit();
let __slate_slot_31: *mut *mut vring_desc_extra = core::ptr::addr_of_mut!(extra) as *mut *mut vring_desc_extra;
unsafe {
let __slate_auto_bytes = __slate_slot_31 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_30 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_29 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_28 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_27 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_26 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_6 = __slate_slot_25 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
total_in_len = 0;
let __slate_auto_bytes_7 = __slate_slot_23 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 2usize);
let __slate_auto_bytes = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_23 = (*vq).__slate_anon_13.packed.next_avail_idx;
*__slate_slot_30 = alloc_indirect_packed(total_sg, gfp);
if (*__slate_slot_30).is_null() {
return 12i32.wrapping_neg();
} else {
*__slate_slot_31 = (*__slate_slot_30).wrapping_offset(total_sg as isize) as *mut vring_desc_extra;
if unlikely!((*vq).vq.num_free < 1) != 0 {
kfree(*__slate_slot_30 as *const ());
return 28i32.wrapping_neg();
} else {
*__slate_slot_28 = 0;
*__slate_slot_27 = 0;
'__join_3: {
while *__slate_slot_27 < out_sgs.wrapping_add(in_sgs) {
*__slate_slot_29 = *sgs.wrapping_offset(*__slate_slot_27 as isize);
while !(*__slate_slot_29).is_null() {
if vring_map_one_sg(vq as *const vring_virtqueue, *__slate_slot_29, (if *__slate_slot_27 < out_sgs { 1i32 } else { 2i32 }) as u32, &mut (*core::ptr::addr_of_mut!(*__slate_slot_22)), &mut (*core::ptr::addr_of_mut!(*__slate_slot_25)), premapped, attr) != 0 {
break '__join_3;
} else {
(*(*__slate_slot_30).wrapping_offset(*__slate_slot_28 as isize)).flags = (if *__slate_slot_27 < out_sgs { 0i32 } else { 2i32 }) as u16;
(*(*__slate_slot_30).wrapping_offset(*__slate_slot_28 as isize)).addr = *__slate_slot_22;
(*(*__slate_slot_30).wrapping_offset(*__slate_slot_28 as isize)).len = *__slate_slot_25;
if unlikely!(unsafe { (*vq).use_map_api }) != 0 {
(*(*__slate_slot_31).wrapping_offset(*__slate_slot_28 as isize)).addr = if premapped { !0 } else { *__slate_slot_22 };
(*(*__slate_slot_31).wrapping_offset(*__slate_slot_28 as isize)).len = *__slate_slot_25;
(*(*__slate_slot_31).wrapping_offset(*__slate_slot_28 as isize)).flags = (if *__slate_slot_27 < out_sgs { 0i32 } else { 2i32 }) as u16;
}
if *__slate_slot_27 >= out_sgs {
total_in_len = total_in_len.wrapping_add(*__slate_slot_25);
}
let __v18: u32 = *__slate_slot_28;
*__slate_slot_28 = __v18.wrapping_add(1);
*__slate_slot_29 = sg_next(*__slate_slot_29);
}
}
let __v16: u32 = *__slate_slot_27;
*__slate_slot_27 = __v16.wrapping_add(1);
}
// Now that the indirect table is filled in, map it.
*__slate_slot_22 = vring_map_single(vq as *const vring_virtqueue, *__slate_slot_30 as *mut (), (total_sg as u64).wrapping_mul(core::mem::size_of::<vring_packed_desc>() as u64), 1u32);
if vring_mapping_error(vq as *const vring_virtqueue, *__slate_slot_22) == 0 {
(*unsafe { (*vq).__slate_anon_13.packed.vring.desc }.wrapping_offset((*__slate_slot_23 as i32) as isize)).addr = *__slate_slot_22;
(*unsafe { (*vq).__slate_anon_13.packed.vring.desc }.wrapping_offset((*__slate_slot_23 as i32) as isize)).len = (total_sg as u64).wrapping_mul(core::mem::size_of::<vring_packed_desc>() as u64) as u32;
(*unsafe { (*vq).__slate_anon_13.packed.vring.desc }.wrapping_offset((*__slate_slot_23 as i32) as isize)).id = id;
if (*vq).use_map_api {
(*unsafe { (*vq).__slate_anon_13.packed.desc_extra }.wrapping_offset((id as i32) as isize)).addr = *__slate_slot_22;
(*unsafe { (*vq).__slate_anon_13.packed.desc_extra }.wrapping_offset((id as i32) as isize)).len = (total_sg as u64).wrapping_mul(core::mem::size_of::<vring_packed_desc>() as u64) as u32;
(*unsafe { (*vq).__slate_anon_13.packed.desc_extra }.wrapping_offset((id as i32) as isize)).flags = (4 | ((*vq).__slate_anon_13.packed.avail_used_flags as i32)) as u16;
}
// A driver MUST NOT make the first descriptor in the list
// available before all subsequent descriptors comprising
// the list are made available.
virtio_wmb((*vq).weak_barriers);
(*unsafe { (*vq).__slate_anon_13.packed.vring.desc }.wrapping_offset((*__slate_slot_23 as i32) as isize)).flags = (4 | ((*vq).__slate_anon_13.packed.avail_used_flags as i32)) as u16;
// We're using some buffers from the free list.
let __v13: u32 = (*vq).vq.num_free;
(*vq).vq.num_free = __v13.wrapping_sub(1);
// Update free pointer
*__slate_slot_27 = (*__slate_slot_23 as i32).wrapping_add(1) as u32;
if *__slate_slot_27 >= (*vq).__slate_anon_13.packed.vring.num {
*__slate_slot_27 = 0;
(*vq).__slate_anon_13.packed.avail_wrap_counter = ((*vq).__slate_anon_13.packed.avail_wrap_counter as i32) ^ 1 != 0;
(*vq).__slate_anon_13.packed.avail_used_flags = (((*vq).__slate_anon_13.packed.avail_used_flags as i32) ^ 32896i32) as u16;
}
(*vq).__slate_anon_13.packed.next_avail_idx = *__slate_slot_27 as u16;
if !virtqueue_is_in_order(vq as *const vring_virtqueue) {
(*vq).free_head = (*unsafe { (*vq).__slate_anon_13.packed.desc_extra }.wrapping_offset((id as i32) as isize)).next as u32;
}
// Store token and indirect buffer state.
(*unsafe { (*vq).__slate_anon_13.packed.desc_state }.wrapping_offset((id as i32) as isize)).num = 1;
(*unsafe { (*vq).__slate_anon_13.packed.desc_state }.wrapping_offset((id as i32) as isize)).data = data;
(*unsafe { (*vq).__slate_anon_13.packed.desc_state }.wrapping_offset((id as i32) as isize)).indir_desc = *__slate_slot_30;
(*unsafe { (*vq).__slate_anon_13.packed.desc_state }.wrapping_offset((id as i32) as isize)).last = id;
(*unsafe { (*vq).__slate_anon_13.packed.desc_state }.wrapping_offset((id as i32) as isize)).total_in_len = total_in_len;
let __v4: u32 = (*vq).num_added;
(*vq).num_added = __v4.wrapping_add(1);
return 0;
}
}
*__slate_slot_26 = *__slate_slot_28;
*__slate_slot_28 = 0;
while *__slate_slot_28 < *__slate_slot_26 {
vring_unmap_extra_packed(vq as *const vring_virtqueue, (*__slate_slot_31).wrapping_offset(*__slate_slot_28 as isize) as *const vring_desc_extra);
let __v1: u32 = *__slate_slot_28;
*__slate_slot_28 = __v1.wrapping_add(1);
}
kfree(*__slate_slot_30 as *const ());
return 12i32.wrapping_neg();
}
}
}
}

#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_add_packed")]
extern "C" fn virtqueue_add_packed(vq: *mut vring_virtqueue, sgs: *mut *mut scatterlist, total_sg: u32, out_sgs: u32, in_sgs: u32, data: *mut (), ctx: *mut (), premapped: bool, gfp: u32, attr: u64) -> i32 {
let mut addr: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_25: *mut u64 = core::ptr::addr_of_mut!(addr) as *mut u64;
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_30: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut unpub_flags: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_31: *mut u16 = core::ptr::addr_of_mut!(unpub_flags) as *mut u16;
let mut avail_used_flags: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_32: *mut u16 = core::ptr::addr_of_mut!(avail_used_flags) as *mut u16;
let mut curr: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_33: *mut u16 = core::ptr::addr_of_mut!(curr) as *mut u16;
let mut prev: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_34: *mut u16 = core::ptr::addr_of_mut!(prev) as *mut u16;
let mut id: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_35: *mut u16 = core::ptr::addr_of_mut!(id) as *mut u16;
let mut head: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_36: *mut u16 = core::ptr::addr_of_mut!(head) as *mut u16;
let mut flags: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_37: *mut u16 = core::ptr::addr_of_mut!(flags) as *mut u16;
let mut head_flags: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_38: *mut u16 = core::ptr::addr_of_mut!(head_flags) as *mut u16;
let mut len: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_39: *mut u32 = core::ptr::addr_of_mut!(len) as *mut u32;
let mut err_idx: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_40: *mut u32 = core::ptr::addr_of_mut!(err_idx) as *mut u32;
let mut descs_used: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_41: *mut u32 = core::ptr::addr_of_mut!(descs_used) as *mut u32;
let mut c: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_42: *mut u32 = core::ptr::addr_of_mut!(c) as *mut u32;
let mut n: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_43: *mut u32 = core::ptr::addr_of_mut!(n) as *mut u32;
let mut i: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_44: *mut u32 = core::ptr::addr_of_mut!(i) as *mut u32;
let mut sg: core::mem::MaybeUninit<*mut scatterlist> = core::mem::MaybeUninit::uninit();
let __slate_slot_45: *mut *mut scatterlist = core::ptr::addr_of_mut!(sg) as *mut *mut scatterlist;
let mut desc: core::mem::MaybeUninit<*mut vring_packed_desc> = core::mem::MaybeUninit::uninit();
let __slate_slot_46: *mut *mut vring_packed_desc = core::ptr::addr_of_mut!(desc) as *mut *mut vring_packed_desc;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_46 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_45 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_44 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_43 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_42 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_6 = __slate_slot_41 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_7 = __slate_slot_40 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_8 = __slate_slot_39 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_9 = __slate_slot_38 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 2usize);
let __slate_auto_bytes_10 = __slate_slot_37 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_10.add(0usize), 0u8, 2usize);
let __slate_auto_bytes_11 = __slate_slot_36 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_11.add(0usize), 0u8, 2usize);
let __slate_auto_bytes_12 = __slate_slot_35 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_12.add(0usize), 0u8, 2usize);
let __slate_auto_bytes_13 = __slate_slot_34 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_13.add(0usize), 0u8, 2usize);
let __slate_auto_bytes_14 = __slate_slot_33 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_14.add(0usize), 0u8, 2usize);
let __slate_auto_bytes_15 = __slate_slot_32 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_15.add(0usize), 0u8, 2usize);
let __slate_auto_bytes_16 = __slate_slot_31 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_16.add(0usize), 0u8, 2usize);
let __slate_auto_bytes_17 = __slate_slot_30 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_17.add(0usize), 0u8, 4usize);
'__join_59: {
loop {
if unlikely!(data.is_null()) != 0 {
break '__join_59;
} else {
if true {
break;
}
}
}
'__join_55: {
loop {
if unlikely!(!ctx.is_null() && (*vq).indirect) != 0 {
break '__join_55;
} else {
if true {
break;
}
}
}
if unlikely!(unsafe { (*vq).broken }) != 0 {
return 5i32.wrapping_neg();
} else {
'__join_48: {
loop {
if unlikely!(total_sg == 0) != 0 {
break '__join_48;
} else {
if true {
break;
}
}
}
'__join_36: {
if virtqueue_use_indirect(vq as *const vring_virtqueue, total_sg) {
*__slate_slot_35 = (*vq).free_head as u16;
'__join_42: {
loop {
if unlikely!(((*__slate_slot_35 as i32) as u32) == (*vq).__slate_anon_13.packed.vring.num) != 0 {
break '__join_42;
} else {
if true {
break;
}
}
}
*__slate_slot_30 = virtqueue_add_indirect_packed(vq, sgs, total_sg, out_sgs, in_sgs, data, premapped, gfp, *__slate_slot_35, attr);
if *__slate_slot_30 != 12i32.wrapping_neg() {
return *__slate_slot_30;
} else {
// fall back on direct
break '__join_36;
}
}
core::arch::asm!("541: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 541b - ., 3; .popsection\n# {0}", const 541 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_virtqueue_add_packed_51, sym __slate_asm_string_virtqueue_add_packed_52, const 1692 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
}
*__slate_slot_36 = (*vq).__slate_anon_13.packed.next_avail_idx;
*__slate_slot_32 = (*vq).__slate_anon_13.packed.avail_used_flags;
let __ret_warn_on: i32 = (total_sg > (*vq).__slate_anon_13.packed.vring.num && !(*vq).indirect) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("542: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 542b - ., 3; .popsection\n# {0}", const 542 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_virtqueue_add_packed_49, sym __slate_asm_string_virtqueue_add_packed_50, const 1707 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("543: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 543b - ., 4; .popsection\n# {0}", const 543 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
*__slate_slot_46 = (*vq).__slate_anon_13.packed.vring.desc;
*__slate_slot_44 = *__slate_slot_36 as u32;
*__slate_slot_41 = total_sg;
if unlikely!((*vq).vq.num_free < *__slate_slot_41) != 0 {
return 28i32.wrapping_neg();
} else {
*__slate_slot_35 = (*vq).free_head as u16;
'__join_28: {
loop {
if unlikely!(((*__slate_slot_35 as i32) as u32) == (*vq).__slate_anon_13.packed.vring.num) != 0 {
break '__join_28;
} else {
if true {
break;
}
}
}
*__slate_slot_33 = *__slate_slot_35;
*__slate_slot_42 = 0;
*__slate_slot_43 = 0;
'__join_8: {
while *__slate_slot_43 < out_sgs.wrapping_add(in_sgs) {
*__slate_slot_45 = *sgs.wrapping_offset(*__slate_slot_43 as isize);
while !(*__slate_slot_45).is_null() {
let __slate_auto_bytes = __slate_slot_25 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
if vring_map_one_sg(vq as *const vring_virtqueue, *__slate_slot_45, (if *__slate_slot_43 < out_sgs { 1i32 } else { 2i32 }) as u32, &mut (*core::ptr::addr_of_mut!(*__slate_slot_25)), &mut (*core::ptr::addr_of_mut!(*__slate_slot_39)), premapped, attr) != 0 {
break '__join_8;
} else {
let __v24: u32 = *__slate_slot_42;
let __v23: u32 = __v24.wrapping_add(1);
*__slate_slot_42 = __v23;
*__slate_slot_37 = (((*vq).__slate_anon_13.packed.avail_used_flags as i32) | if __v23 == total_sg { 0 } else { 1 } | if *__slate_slot_43 < out_sgs { 0 } else { 2 }) as u16;
if *__slate_slot_44 == ((*__slate_slot_36 as i32) as u32) {
*__slate_slot_38 = *__slate_slot_37;
} else {
(*(*__slate_slot_46).wrapping_offset(*__slate_slot_44 as isize)).flags = *__slate_slot_37;
}
(*(*__slate_slot_46).wrapping_offset(*__slate_slot_44 as isize)).addr = *__slate_slot_25;
(*(*__slate_slot_46).wrapping_offset(*__slate_slot_44 as isize)).len = *__slate_slot_39;
(*(*__slate_slot_46).wrapping_offset(*__slate_slot_44 as isize)).id = *__slate_slot_35;
if unlikely!(unsafe { (*vq).use_map_api }) != 0 {
(*unsafe { (*vq).__slate_anon_13.packed.desc_extra }.wrapping_offset((*__slate_slot_33 as i32) as isize)).addr = if premapped { !0 } else { *__slate_slot_25 };
(*unsafe { (*vq).__slate_anon_13.packed.desc_extra }.wrapping_offset((*__slate_slot_33 as i32) as isize)).len = *__slate_slot_39;
(*unsafe { (*vq).__slate_anon_13.packed.desc_extra }.wrapping_offset((*__slate_slot_33 as i32) as isize)).flags = *__slate_slot_37;
}
*__slate_slot_34 = *__slate_slot_33;
*__slate_slot_33 = (*unsafe { (*vq).__slate_anon_13.packed.desc_extra }.wrapping_offset((*__slate_slot_33 as i32) as isize)).next;
let __v22: u32 = *__slate_slot_44;
let __v21: u32 = __v22.wrapping_add(1);
*__slate_slot_44 = __v21;
if __v21 >= (*vq).__slate_anon_13.packed.vring.num {
*__slate_slot_44 = 0;
(*vq).__slate_anon_13.packed.avail_used_flags = (((*vq).__slate_anon_13.packed.avail_used_flags as i32) ^ 32896i32) as u16;
}
*__slate_slot_45 = sg_next(*__slate_slot_45);
}
}
let __v17: u32 = *__slate_slot_43;
*__slate_slot_43 = __v17.wrapping_add(1);
}
if *__slate_slot_44 <= ((*__slate_slot_36 as i32) as u32) {
(*vq).__slate_anon_13.packed.avail_wrap_counter = ((*vq).__slate_anon_13.packed.avail_wrap_counter as i32) ^ 1 != 0;
}
// We're using some buffers from the free list.
let __v11: u32 = (*vq).vq.num_free;
(*vq).vq.num_free = __v11.wrapping_sub(*__slate_slot_41);
// Update free pointer
(*vq).__slate_anon_13.packed.next_avail_idx = *__slate_slot_44 as u16;
(*vq).free_head = *__slate_slot_33 as u32;
// Store token.
(*unsafe { (*vq).__slate_anon_13.packed.desc_state }.wrapping_offset((*__slate_slot_35 as i32) as isize)).num = *__slate_slot_41 as u16;
(*unsafe { (*vq).__slate_anon_13.packed.desc_state }.wrapping_offset((*__slate_slot_35 as i32) as isize)).data = data;
(*unsafe { (*vq).__slate_anon_13.packed.desc_state }.wrapping_offset((*__slate_slot_35 as i32) as isize)).indir_desc = ctx as *mut vring_packed_desc;
(*unsafe { (*vq).__slate_anon_13.packed.desc_state }.wrapping_offset((*__slate_slot_35 as i32) as isize)).last = *__slate_slot_34;
// A driver MUST NOT make the first descriptor in the list
// available before all subsequent descriptors comprising
// the list are made available.
virtio_wmb((*vq).weak_barriers);
(*unsafe { (*vq).__slate_anon_13.packed.vring.desc }.wrapping_offset((*__slate_slot_36 as i32) as isize)).flags = *__slate_slot_38;
let __v8: u32 = (*vq).num_added;
(*vq).num_added = __v8.wrapping_add(*__slate_slot_41);
return 0;
}
*__slate_slot_40 = *__slate_slot_44;
*__slate_slot_44 = *__slate_slot_36 as u32;
*__slate_slot_33 = (*vq).free_head as u16;
(*vq).__slate_anon_13.packed.avail_used_flags = *__slate_slot_32;
*__slate_slot_31 = ((*__slate_slot_32 as i32) ^ 32896i32) as u16;
*__slate_slot_43 = 0;
while *__slate_slot_43 < total_sg {
if *__slate_slot_44 == *__slate_slot_40 {
break;
} else {
// The mapping loop made every descriptor but the head
// available. Stamp the previous wrap counter's AVAIL and USED
// bits on those, so that a later and shorter chain at this head
// does not leave one of them available beyond its own last
// descriptor. Marking them used instead would hand
// is_used_desc_packed() a completion we never made.
if *__slate_slot_44 != ((*__slate_slot_36 as i32) as u32) {
(*(*__slate_slot_46).wrapping_offset(*__slate_slot_44 as isize)).flags = *__slate_slot_31;
}
vring_unmap_extra_packed(vq as *const vring_virtqueue, unsafe { (*vq).__slate_anon_13.packed.desc_extra }.wrapping_offset((*__slate_slot_33 as i32) as isize) as *const vring_desc_extra);
*__slate_slot_33 = (*unsafe { (*vq).__slate_anon_13.packed.desc_extra }.wrapping_offset((*__slate_slot_33 as i32) as isize)).next;
let __v5: u32 = *__slate_slot_44;
*__slate_slot_44 = __v5.wrapping_add(1);
if *__slate_slot_44 >= (*vq).__slate_anon_13.packed.vring.num {
*__slate_slot_44 = 0;
*__slate_slot_31 = ((*__slate_slot_31 as i32) ^ 32896i32) as u16;
}
let __v1: u32 = *__slate_slot_43;
*__slate_slot_43 = __v1.wrapping_add(1);
}
}
return 5i32.wrapping_neg();
}
core::arch::asm!("544: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 544b - ., 3; .popsection\n# {0}", const 544 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_virtqueue_add_packed_47, sym __slate_asm_string_virtqueue_add_packed_48, const 1721 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
}
core::arch::asm!("540: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 540b - ., 3; .popsection\n# {0}", const 540 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_virtqueue_add_packed_53, sym __slate_asm_string_virtqueue_add_packed_54, const 1688 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
}
core::arch::asm!("539: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 539b - ., 3; .popsection\n# {0}", const 539 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_virtqueue_add_packed_55, sym __slate_asm_string_virtqueue_add_packed_56, const 1679 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
core::arch::asm!("538: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 538b - ., 3; .popsection\n# {0}", const 538 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_virtqueue_add_packed_57, sym __slate_asm_string_virtqueue_add_packed_58, const 1678 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
unsafe { core::mem::zeroed() }
}

#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_add_packed_in_order")]
extern "C" fn virtqueue_add_packed_in_order(vq: *mut vring_virtqueue, sgs: *mut *mut scatterlist, total_sg: u32, out_sgs: u32, in_sgs: u32, data: *mut (), ctx: *mut (), premapped: bool, gfp: u32, attr: u64) -> i32 {
let mut len: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_33: *mut u32 = core::ptr::addr_of_mut!(len) as *mut u32;
let mut addr: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_34: *mut u64 = core::ptr::addr_of_mut!(addr) as *mut u64;
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_39: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut avail_wrap_counter: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_40: *mut bool = core::ptr::addr_of_mut!(avail_wrap_counter) as *mut bool;
let mut unpub_flags: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_41: *mut u16 = core::ptr::addr_of_mut!(unpub_flags) as *mut u16;
let mut avail_used_flags: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_42: *mut u16 = core::ptr::addr_of_mut!(avail_used_flags) as *mut u16;
let mut head: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_43: *mut u16 = core::ptr::addr_of_mut!(head) as *mut u16;
let mut flags: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_44: *mut u16 = core::ptr::addr_of_mut!(flags) as *mut u16;
let mut head_flags: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_45: *mut u16 = core::ptr::addr_of_mut!(head_flags) as *mut u16;
let mut total_in_len: u32 = unsafe { core::mem::zeroed() };
let mut err_idx: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_47: *mut u32 = core::ptr::addr_of_mut!(err_idx) as *mut u32;
let mut sg_count: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_48: *mut u32 = core::ptr::addr_of_mut!(sg_count) as *mut u32;
let mut n: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_49: *mut u32 = core::ptr::addr_of_mut!(n) as *mut u32;
let mut i: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_50: *mut u32 = core::ptr::addr_of_mut!(i) as *mut u32;
let mut sg: core::mem::MaybeUninit<*mut scatterlist> = core::mem::MaybeUninit::uninit();
let __slate_slot_51: *mut *mut scatterlist = core::ptr::addr_of_mut!(sg) as *mut *mut scatterlist;
let mut desc: core::mem::MaybeUninit<*mut vring_packed_desc> = core::mem::MaybeUninit::uninit();
let __slate_slot_52: *mut *mut vring_packed_desc = core::ptr::addr_of_mut!(desc) as *mut *mut vring_packed_desc;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_52 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_51 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_50 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_49 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_48 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_6 = __slate_slot_47 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
total_in_len = 0;
let __slate_auto_bytes_7 = __slate_slot_45 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 2usize);
let __slate_auto_bytes_8 = __slate_slot_44 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 2usize);
let __slate_auto_bytes_9 = __slate_slot_43 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 2usize);
let __slate_auto_bytes_10 = __slate_slot_42 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_10.add(0usize), 0u8, 2usize);
let __slate_auto_bytes_11 = __slate_slot_41 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_11.add(0usize), 0u8, 2usize);
let __slate_auto_bytes_12 = __slate_slot_40 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_12.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_13 = __slate_slot_39 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_13.add(0usize), 0u8, 4usize);
'__join_53: {
loop {
if unlikely!(data.is_null()) != 0 {
break '__join_53;
} else {
if true {
break;
}
}
}
'__join_49: {
loop {
if unlikely!(!ctx.is_null() && (*vq).indirect) != 0 {
break '__join_49;
} else {
if true {
break;
}
}
}
if unlikely!(unsafe { (*vq).broken }) != 0 {
return 5i32.wrapping_neg();
} else {
'__join_42: {
loop {
if unlikely!(total_sg == 0) != 0 {
break '__join_42;
} else {
if true {
break;
}
}
}
if virtqueue_use_indirect(vq as *const vring_virtqueue, total_sg) {
*__slate_slot_39 = virtqueue_add_indirect_packed(vq, sgs, total_sg, out_sgs, in_sgs, data, premapped, gfp, (*vq).__slate_anon_13.packed.next_avail_idx, attr);
if *__slate_slot_39 != 12i32.wrapping_neg() {
return *__slate_slot_39;
}
// fall back on direct
}
*__slate_slot_43 = (*vq).__slate_anon_13.packed.next_avail_idx;
*__slate_slot_42 = (*vq).__slate_anon_13.packed.avail_used_flags;
*__slate_slot_40 = (*vq).__slate_anon_13.packed.avail_wrap_counter;
let __ret_warn_on: i32 = (total_sg > (*vq).__slate_anon_13.packed.vring.num && !(*vq).indirect) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("548: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 548b - ., 3; .popsection\n# {0}", const 548 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_virtqueue_add_packed_in_order_53, sym __slate_asm_string_virtqueue_add_packed_in_order_54, const 1881 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("549: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 549b - ., 4; .popsection\n# {0}", const 549 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
*__slate_slot_52 = (*vq).__slate_anon_13.packed.vring.desc;
*__slate_slot_50 = *__slate_slot_43 as u32;
if unlikely!((*vq).vq.num_free < total_sg) != 0 {
return 28i32.wrapping_neg();
} else {
*__slate_slot_48 = 0;
*__slate_slot_49 = 0;
'__join_8: {
while *__slate_slot_49 < out_sgs.wrapping_add(in_sgs) {
*__slate_slot_51 = *sgs.wrapping_offset(*__slate_slot_49 as isize);
while !(*__slate_slot_51).is_null() {
let __slate_auto_bytes = __slate_slot_34 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_33 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_44 = 0;
let __v32: u32 = *__slate_slot_48;
let __v31: u32 = __v32.wrapping_add(1);
*__slate_slot_48 = __v31;
if __v31 != total_sg {
*__slate_slot_44 = ((*__slate_slot_44 as i32) | 1) as u16;
}
if *__slate_slot_49 >= out_sgs {
*__slate_slot_44 = ((*__slate_slot_44 as i32) | 2) as u16;
}
if vring_map_one_sg(vq as *const vring_virtqueue, *__slate_slot_51, (if *__slate_slot_49 < out_sgs { 1i32 } else { 2i32 }) as u32, &mut (*core::ptr::addr_of_mut!(*__slate_slot_34)), &mut (*core::ptr::addr_of_mut!(*__slate_slot_33)), premapped, attr) != 0 {
break '__join_8;
} else {
*__slate_slot_44 = ((*__slate_slot_44 as i32) | ((*vq).__slate_anon_13.packed.avail_used_flags as i32)) as u16;
if *__slate_slot_50 == ((*__slate_slot_43 as i32) as u32) {
*__slate_slot_45 = *__slate_slot_44;
} else {
(*(*__slate_slot_52).wrapping_offset(*__slate_slot_50 as isize)).flags = *__slate_slot_44;
}
(*(*__slate_slot_52).wrapping_offset(*__slate_slot_50 as isize)).addr = *__slate_slot_34;
(*(*__slate_slot_52).wrapping_offset(*__slate_slot_50 as isize)).len = *__slate_slot_33;
(*(*__slate_slot_52).wrapping_offset(*__slate_slot_50 as isize)).id = *__slate_slot_43;
if unlikely!(unsafe { (*vq).use_map_api }) != 0 {
(*unsafe { (*vq).__slate_anon_13.packed.desc_extra }.wrapping_offset(*__slate_slot_50 as isize)).addr = if premapped { !0 } else { *__slate_slot_34 };
(*unsafe { (*vq).__slate_anon_13.packed.desc_extra }.wrapping_offset(*__slate_slot_50 as isize)).len = *__slate_slot_33;
(*unsafe { (*vq).__slate_anon_13.packed.desc_extra }.wrapping_offset(*__slate_slot_50 as isize)).flags = *__slate_slot_44;
}
let __v24: u32 = *__slate_slot_50;
let __v23: u32 = __v24.wrapping_add(1);
*__slate_slot_50 = __v23;
if __v23 >= (*vq).__slate_anon_13.packed.vring.num {
*__slate_slot_50 = 0;
(*vq).__slate_anon_13.packed.avail_used_flags = (((*vq).__slate_anon_13.packed.avail_used_flags as i32) ^ 32896i32) as u16;
(*vq).__slate_anon_13.packed.avail_wrap_counter = ((*vq).__slate_anon_13.packed.avail_wrap_counter as i32) ^ 1 != 0;
}
if *__slate_slot_49 >= out_sgs {
total_in_len = total_in_len.wrapping_add(*__slate_slot_33);
}
*__slate_slot_51 = sg_next(*__slate_slot_51);
}
}
let __v14: u32 = *__slate_slot_49;
*__slate_slot_49 = __v14.wrapping_add(1);
}
// We're using some buffers from the free list.
let __v11: u32 = (*vq).vq.num_free;
(*vq).vq.num_free = __v11.wrapping_sub(total_sg);
// Update free pointer
(*vq).__slate_anon_13.packed.next_avail_idx = *__slate_slot_50 as u16;
// Store token.
(*unsafe { (*vq).__slate_anon_13.packed.desc_state }.wrapping_offset((*__slate_slot_43 as i32) as isize)).num = total_sg as u16;
(*unsafe { (*vq).__slate_anon_13.packed.desc_state }.wrapping_offset((*__slate_slot_43 as i32) as isize)).data = data;
(*unsafe { (*vq).__slate_anon_13.packed.desc_state }.wrapping_offset((*__slate_slot_43 as i32) as isize)).indir_desc = ctx as *mut vring_packed_desc;
(*unsafe { (*vq).__slate_anon_13.packed.desc_state }.wrapping_offset((*__slate_slot_43 as i32) as isize)).total_in_len = total_in_len;
// A driver MUST NOT make the first descriptor in the list
// available before all subsequent descriptors comprising
// the list are made available.
virtio_wmb((*vq).weak_barriers);
(*unsafe { (*vq).__slate_anon_13.packed.vring.desc }.wrapping_offset((*__slate_slot_43 as i32) as isize)).flags = *__slate_slot_45;
let __v8: u32 = (*vq).num_added;
(*vq).num_added = __v8.wrapping_add(total_sg);
return 0;
}
*__slate_slot_47 = *__slate_slot_50;
*__slate_slot_50 = *__slate_slot_43 as u32;
(*vq).__slate_anon_13.packed.avail_used_flags = *__slate_slot_42;
(*vq).__slate_anon_13.packed.avail_wrap_counter = *__slate_slot_40;
*__slate_slot_41 = ((*__slate_slot_42 as i32) ^ 32896i32) as u16;
*__slate_slot_49 = 0;
while *__slate_slot_49 < total_sg {
if *__slate_slot_50 == *__slate_slot_47 {
break;
} else {
// The mapping loop made every descriptor but the head
// available. Stamp the previous wrap counter's AVAIL and USED
// bits on those, so that a later and shorter chain at this head
// does not leave one of them available beyond its own last
// descriptor. Marking them used instead would hand
// is_used_desc_packed() a completion we never made.
if *__slate_slot_50 != ((*__slate_slot_43 as i32) as u32) {
(*(*__slate_slot_52).wrapping_offset(*__slate_slot_50 as isize)).flags = *__slate_slot_41;
}
vring_unmap_extra_packed(vq as *const vring_virtqueue, unsafe { (*vq).__slate_anon_13.packed.desc_extra }.wrapping_offset(*__slate_slot_50 as isize) as *const vring_desc_extra);
let __v5: u32 = *__slate_slot_50;
*__slate_slot_50 = __v5.wrapping_add(1);
if *__slate_slot_50 >= (*vq).__slate_anon_13.packed.vring.num {
*__slate_slot_50 = 0;
*__slate_slot_41 = ((*__slate_slot_41 as i32) ^ 32896i32) as u16;
}
let __v1: u32 = *__slate_slot_49;
*__slate_slot_49 = __v1.wrapping_add(1);
}
}
return 5i32.wrapping_neg();
}
}
core::arch::asm!("547: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 547b - ., 3; .popsection\n# {0}", const 547 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_virtqueue_add_packed_in_order_55, sym __slate_asm_string_virtqueue_add_packed_in_order_56, const 1862 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
}
core::arch::asm!("546: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 546b - ., 3; .popsection\n# {0}", const 546 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_virtqueue_add_packed_in_order_57, sym __slate_asm_string_virtqueue_add_packed_in_order_58, const 1853 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
core::arch::asm!("545: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 545b - ., 3; .popsection\n# {0}", const 545 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_virtqueue_add_packed_in_order_59, sym __slate_asm_string_virtqueue_add_packed_in_order_60, const 1852 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
unsafe { core::mem::zeroed() }
}

#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_kick_prepare_packed")]
extern "C" fn virtqueue_kick_prepare_packed(vq: *mut vring_virtqueue) -> bool {
let mut snapshot: core::mem::MaybeUninit<__SlateRecord_virtio_ring_c_2012> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut __SlateRecord_virtio_ring_c_2012 = core::ptr::addr_of_mut!(snapshot) as *mut __SlateRecord_virtio_ring_c_2012;
let mut needs_kick: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut bool = core::ptr::addr_of_mut!(needs_kick) as *mut bool;
let mut event_idx: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u16 = core::ptr::addr_of_mut!(event_idx) as *mut u16;
let mut wrap_counter: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut u16 = core::ptr::addr_of_mut!(wrap_counter) as *mut u16;
let mut flags: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u16 = core::ptr::addr_of_mut!(flags) as *mut u16;
let mut off_wrap: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u16 = core::ptr::addr_of_mut!(off_wrap) as *mut u16;
let mut old: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut u16 = core::ptr::addr_of_mut!(old) as *mut u16;
let mut new: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut u16 = core::ptr::addr_of_mut!(new) as *mut u16;
unsafe {
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 2usize);
let __slate_auto_bytes_1 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 2usize);
let __slate_auto_bytes_2 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 2usize);
let __slate_auto_bytes_3 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 2usize);
let __slate_auto_bytes_4 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 2usize);
let __slate_auto_bytes_5 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 2usize);
let __slate_auto_bytes_6 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 1usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
// We need to expose the new flags value before checking notification
// suppressions.
virtio_mb((*vq).weak_barriers);
*__slate_slot_8 = (((*vq).__slate_anon_13.packed.next_avail_idx as i32) as u32).wrapping_sub((*vq).num_added) as u16;
*__slate_slot_9 = (*vq).__slate_anon_13.packed.next_avail_idx;
(*vq).num_added = 0;
(*__slate_slot_2).u32 = *((*vq).__slate_anon_13.packed.vring.device as *mut u32);
*__slate_slot_6 = (*__slate_slot_2).__slate_anon_0.flags;
if (*__slate_slot_6 as i32) != 2 {
*__slate_slot_3 = (*__slate_slot_6 as i32) != 1;
} else {
*__slate_slot_7 = (*__slate_slot_2).__slate_anon_0.off_wrap;
*__slate_slot_5 = ((*__slate_slot_7 as i32) >> 15i32) as u16;
*__slate_slot_4 = ((*__slate_slot_7 as i32) & -32769i32) as u16;
if (*__slate_slot_5 as i32) != ((*vq).__slate_anon_13.packed.avail_wrap_counter as i32) {
*__slate_slot_4 = ((*__slate_slot_4 as i32) as u32).wrapping_sub((*vq).__slate_anon_13.packed.vring.num) as u16;
}
*__slate_slot_3 = vring_need_event(*__slate_slot_4, *__slate_slot_9, *__slate_slot_8) != 0;
}
return *__slate_slot_3;
}
}

fn detach_buf_packed_in_order(vq: *mut vring_virtqueue, id: u32, ctx: *mut *mut ()) {
let mut state: *mut vring_desc_state_packed = core::ptr::null_mut::<vring_desc_state_packed>();
let mut i: u32 = 0;
let mut curr: u32 = 0;
state = unsafe { (*vq).__slate_anon_13.packed.desc_state }.wrapping_offset(id as isize);
// Clear data ptr.
unsafe {
(*state).data = core::ptr::null_mut::<()>();
}
let __v1: u32 = unsafe { (*vq).vq.num_free };
unsafe {
(*vq).vq.num_free = __v1.wrapping_add(((*state).num as i32) as u32);
}
if unlikely!(unsafe { (*vq).use_map_api }) != 0 {
curr = id;
i = 0;
while i < (((unsafe { (*state).num }) as i32) as u32) {
vring_unmap_extra_packed(vq as *const vring_virtqueue, unsafe { (*vq).__slate_anon_13.packed.desc_extra }.wrapping_offset(curr as isize) as *const vring_desc_extra);
curr = (unsafe { (*unsafe { (*vq).__slate_anon_13.packed.desc_extra }.wrapping_offset(curr as isize)).next }) as u32;
i = i.wrapping_add(1);
}
}
if unsafe { (*vq).indirect } {
let mut len: u32 = 0;
let mut num: u32 = 0;
// Free the indirect table, if any, now that it's unmapped.
let desc: *mut vring_packed_desc = unsafe { (*state).indir_desc };
if desc.is_null() {
return;
}
if unsafe { (*vq).use_map_api } {
len = unsafe { (*unsafe { (*vq).__slate_anon_13.packed.desc_extra }.wrapping_offset(id as isize)).len };
num = ((len as u64) / (core::mem::size_of::<vring_packed_desc>() as u64)) as u32;
let extra: *mut vring_desc_extra = desc.wrapping_offset(num as isize) as *mut vring_desc_extra;
i = 0;
while i < num {
vring_unmap_extra_packed(vq as *const vring_virtqueue, extra.wrapping_offset(i as isize) as *const vring_desc_extra);
i = i.wrapping_add(1);
}
}
unsafe { kfree(desc as *const ()) };
unsafe {
(*state).indir_desc = core::ptr::null_mut::<vring_packed_desc>();
}
} else {
if !ctx.is_null() {
unsafe {
*ctx = (*state).indir_desc as *mut ();
}
}
}
}

fn detach_buf_packed(vq: *mut vring_virtqueue, id: u32, ctx: *mut *mut ()) {
let state: *mut vring_desc_state_packed = unsafe { (*vq).__slate_anon_13.packed.desc_state }.wrapping_offset(id as isize);
unsafe {
(*unsafe { (*vq).__slate_anon_13.packed.desc_extra }.wrapping_offset(((*state).last as i32) as isize)).next = (*vq).free_head as u16;
}
unsafe {
(*vq).free_head = id;
}
detach_buf_packed_in_order(vq, id, ctx);
}

fn is_used_desc_packed(vq: *const vring_virtqueue, idx: u16, used_wrap_counter: bool) -> bool {
let flags: u16 = vring_read_packed_desc_flags(vq, idx);
let mut avail: bool = false;
let mut used: bool = false;
avail = (flags as i32) & 128i32 != 0;
used = (flags as i32) & 32768i32 != 0;
(avail as i32) == (used as i32) && (used as i32) == (used_wrap_counter as i32)
}

#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_poll_packed")]
extern "C" fn virtqueue_poll_packed(vq: *const vring_virtqueue, off_wrap: u32) -> bool {
let mut wrap_counter: bool = false;
let mut used_idx: u16 = 0;
wrap_counter = off_wrap >> 15i32 != 0;
used_idx = (off_wrap & 4294934527u32) as u16;
is_used_desc_packed(vq, used_idx, wrap_counter)
}

#[unsafe(link_section = ".text.slate_distinct.virtio_ring.more_used_packed")]
extern "C" fn more_used_packed(vq: *const vring_virtqueue) -> bool {
let mut __v0: u16;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u16>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_550() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*vq).last_used_idx) as *const u16) };
virtqueue_poll_packed(vq, __v0 as u32)
}

fn update_last_used_idx_packed(vq: *mut vring_virtqueue, id: u16, mut last_used: u16, mut used_wrap_counter: u16) {
last_used = (last_used as i32).wrapping_add((unsafe { (*unsafe { (*vq).__slate_anon_13.packed.desc_state }.wrapping_offset((id as i32) as isize)).num }) as i32) as u16;
if unlikely!(((last_used as i32) as u32) >= unsafe { (*vq).__slate_anon_13.packed.vring.num }) != 0 {
last_used = ((last_used as i32) as u32).wrapping_sub(unsafe { (*vq).__slate_anon_13.packed.vring.num }) as u16;
used_wrap_counter = ((used_wrap_counter as i32) ^ 1) as u16;
}
last_used = ((last_used as i32) | (used_wrap_counter as i32) << 15i32) as u16;
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u16>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_551() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__2!((*vq).last_used_idx, last_used);
if true {
break;
}
}
// If we expect an interrupt for the next entry, tell host
// by writing event index and flush out the write before
// the read in the next get_buf call.
if ((unsafe { (*vq).__slate_anon_13.packed.event_flags_shadow }) as i32) == 2 {
loop {
if unsafe { (*vq).weak_barriers } {
loop {
kcsan_mb!();
loop {
let mut __v10: u16;
let mut __ai_ptr: *mut u16 = unsafe { core::ptr::addr_of_mut!((*(*vq).__slate_anon_13.packed.vring.driver).off_wrap) };
kcsan_mb!();
instrument_atomic_read_write(__ai_ptr as *const (), core::mem::size_of::<u16>() as u64);
let mut __v12: u16;
let mut __ret: u16 = unsafe { (*vq).last_used_idx };
match (const { core::mem::size_of::<u16>() }) as u64 {
1 => {
unsafe {
let __t0: u16;
core::arch::asm!("xchgb {0:l}, ({1:r})", inlateout(reg) __ret as u16 => __t0, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u16, u16>(__t0 as u16) };
}
}
2 => {
unsafe {
let __t1: u16;
core::arch::asm!("xchgw {0:x}, ({1:r})", inlateout(reg) __ret as u16 => __t1, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u16, u16>(__t1 as u16) };
}
}
4 => {
unsafe {
let __t2: u16;
core::arch::asm!("xchgl {0:x}, ({1:r})", inlateout(reg) __ret as u16 => __t2, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u16, u16>(__t2 as u16) };
}
}
8 => {
unsafe {
let __t3: u16;
core::arch::asm!("xchgq {0:r}, ({1:r})", inlateout(reg) __ret as u16 => __t3, in(reg) __ai_ptr, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u16, u16>(__t3 as u16) };
}
}
_ => {
unsafe { __xchg_wrong_size() };
}
}
__v12 = __ret;
__v10 = __v12;
if true {
break;
}
}
if true {
break;
}
}
} else {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u16>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_552() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__2!(*unsafe { core::ptr::addr_of_mut!((*(*vq).__slate_anon_13.packed.vring.driver).off_wrap) }, unsafe { (*vq).last_used_idx });
if true {
break;
}
}
loop {
kcsan_mb!();
unsafe {
core::arch::asm!("mfence", options(nostack, att_syntax));
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

#[unsafe(link_section = ".text.slate_distinct.virtio_ring.more_used_packed_in_order")]
extern "C" fn more_used_packed_in_order(vq: *const vring_virtqueue) -> bool {
if (unsafe { (*vq).batch_last.id }) != !0 {
return true;
}
let mut __v0: u16;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u16>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_553() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*vq).last_used_idx) as *const u16) };
virtqueue_poll_packed(vq, __v0 as u32)
}

#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_get_buf_ctx_packed_in_order")]
extern "C" fn virtqueue_get_buf_ctx_packed_in_order(vq: *mut vring_virtqueue, len: *mut u32, ctx: *mut *mut ()) -> *mut () {
let num: u32 = unsafe { (*vq).__slate_anon_13.packed.vring.num };
let mut last_used: u16 = 0;
let mut last_used_idx: u16 = 0;
let mut used_wrap_counter: bool = false;
if unlikely!(unsafe { (*vq).broken }) != 0 {
return core::ptr::null_mut::<()>();
}
last_used_idx = unsafe { (*vq).last_used_idx };
used_wrap_counter = packed_used_wrap_counter(last_used_idx);
last_used = packed_last_used(last_used_idx);
if (unsafe { (*vq).batch_last.id }) == !0 {
if !more_used_packed_in_order(vq as *const vring_virtqueue) {
let mut __v0: i32;
if false {
unsafe { _printk(c"\x017No more buffers in queue\n".as_ptr() as *const u8) };
}
__v0 = 0;
return core::ptr::null_mut::<()>();
}
// Only get used elements after they have been exposed by host.
virtio_rmb(unsafe { (*vq).weak_barriers });
unsafe {
(*vq).batch_last.id = (*unsafe { (*vq).__slate_anon_13.packed.vring.desc }.wrapping_offset((last_used as i32) as isize)).id as u32;
}
unsafe {
(*vq).batch_last.len = (*unsafe { (*vq).__slate_anon_13.packed.vring.desc }.wrapping_offset((last_used as i32) as isize)).len;
}
}
if (unsafe { (*vq).batch_last.id }) == ((last_used as i32) as u32) {
unsafe {
(*vq).batch_last.id = !0;
}
unsafe {
*len = (*vq).batch_last.len;
}
} else {
unsafe {
*len = (*unsafe { (*vq).__slate_anon_13.packed.desc_state }.wrapping_offset((last_used as i32) as isize)).total_in_len;
}
}
if unlikely!(((last_used as i32) as u32) >= num) != 0 {
loop {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*(*vq).vq.vdev).dev) as *const device, c"%s:id %u out of range\n".as_ptr() as *const u8, (*vq).vq.name, last_used as i32) };
unsafe {
(*vq).broken = true;
}
if true {
break;
}
}
return core::ptr::null_mut::<()>();
}
if unlikely!(unsafe { (*unsafe { (*vq).__slate_anon_13.packed.desc_state }.wrapping_offset((last_used as i32) as isize)).data }.is_null()) != 0 {
loop {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*(*vq).vq.vdev).dev) as *const device, c"%s:id %u is not a head!\n".as_ptr() as *const u8, (*vq).vq.name, last_used as i32) };
unsafe {
(*vq).broken = true;
}
if true {
break;
}
}
return core::ptr::null_mut::<()>();
}
// detach_buf_packed clears data, so grab it now.
let ret: *mut () = unsafe { (*unsafe { (*vq).__slate_anon_13.packed.desc_state }.wrapping_offset((last_used as i32) as isize)).data };
detach_buf_packed_in_order(vq, last_used as u32, ctx);
update_last_used_idx_packed(vq, last_used, last_used, used_wrap_counter as u16);
ret
}

#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_get_buf_ctx_packed")]
extern "C" fn virtqueue_get_buf_ctx_packed(vq: *mut vring_virtqueue, len: *mut u32, ctx: *mut *mut ()) -> *mut () {
let num: u32 = unsafe { (*vq).__slate_anon_13.packed.vring.num };
let mut last_used: u16 = 0;
let mut id: u16 = 0;
let mut last_used_idx: u16 = 0;
let mut used_wrap_counter: bool = false;
if unlikely!(unsafe { (*vq).broken }) != 0 {
return core::ptr::null_mut::<()>();
}
if !more_used_packed(vq as *const vring_virtqueue) {
let mut __v0: i32;
if false {
unsafe { _printk(c"\x017No more buffers in queue\n".as_ptr() as *const u8) };
}
__v0 = 0;
return core::ptr::null_mut::<()>();
}
// Only get used elements after they have been exposed by host.
virtio_rmb(unsafe { (*vq).weak_barriers });
let mut __v1: u16;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u16>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_554() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*vq).last_used_idx) as *const u16) };
last_used_idx = __v1;
used_wrap_counter = packed_used_wrap_counter(last_used_idx);
last_used = packed_last_used(last_used_idx);
id = vring_read_packed_desc_id(vq as *const vring_virtqueue, last_used);
unsafe {
*len = vring_read_packed_desc_len(vq as *const vring_virtqueue, last_used);
}
if unlikely!(((id as i32) as u32) >= num) != 0 {
loop {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*(*vq).vq.vdev).dev) as *const device, c"%s:id %u out of range\n".as_ptr() as *const u8, (*vq).vq.name, id as i32) };
unsafe {
(*vq).broken = true;
}
if true {
break;
}
}
return core::ptr::null_mut::<()>();
}
if unlikely!(unsafe { (*unsafe { (*vq).__slate_anon_13.packed.desc_state }.wrapping_offset((id as i32) as isize)).data }.is_null()) != 0 {
loop {
while false {
}
unsafe { _dev_err(core::ptr::addr_of_mut!((*(*vq).vq.vdev).dev) as *const device, c"%s:id %u is not a head!\n".as_ptr() as *const u8, (*vq).vq.name, id as i32) };
unsafe {
(*vq).broken = true;
}
if true {
break;
}
}
return core::ptr::null_mut::<()>();
}
// detach_buf_packed clears data, so grab it now.
let ret: *mut () = unsafe { (*unsafe { (*vq).__slate_anon_13.packed.desc_state }.wrapping_offset((id as i32) as isize)).data };
detach_buf_packed(vq, id as u32, ctx);
update_last_used_idx_packed(vq, id, last_used, used_wrap_counter as u16);
ret
}

#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_disable_cb_packed")]
extern "C" fn virtqueue_disable_cb_packed(vq: *mut vring_virtqueue) {
if ((unsafe { (*vq).__slate_anon_13.packed.event_flags_shadow }) as i32) != 1 {
unsafe {
(*vq).__slate_anon_13.packed.event_flags_shadow = 1;
}
// If device triggered an event already it won't trigger one again:
// no need to disable.
if unsafe { (*vq).event_triggered } {
return;
}
unsafe {
(*(*vq).__slate_anon_13.packed.vring.driver).flags = (*vq).__slate_anon_13.packed.event_flags_shadow;
}
}
}

#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_enable_cb_prepare_packed")]
extern "C" fn virtqueue_enable_cb_prepare_packed(vq: *mut vring_virtqueue) -> u32 {
// We optimistically turn back on interrupts, then check if there was
// more to do.
if unsafe { (*vq).event } {
unsafe {
(*(*vq).__slate_anon_13.packed.vring.driver).off_wrap = (*vq).last_used_idx;
}
// We need to update event offset and event wrap
// counter first before updating event flags.
virtio_wmb(unsafe { (*vq).weak_barriers });
}
if ((unsafe { (*vq).__slate_anon_13.packed.event_flags_shadow }) as i32) == 1 {
unsafe {
(*vq).__slate_anon_13.packed.event_flags_shadow = (if (*vq).event { 2i32 } else { 0i32 }) as u16;
}
unsafe {
(*(*vq).__slate_anon_13.packed.vring.driver).flags = (*vq).__slate_anon_13.packed.event_flags_shadow;
}
}
(unsafe { (*vq).last_used_idx }) as u32
}

#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_enable_cb_delayed_packed")]
extern "C" fn virtqueue_enable_cb_delayed_packed(vq: *mut vring_virtqueue) -> bool {
let mut used_idx: u16 = 0;
let mut wrap_counter: u16 = 0;
let mut last_used_idx: u16 = 0;
let mut bufs: u16 = 0;
// We optimistically turn back on interrupts, then check if there was
// more to do.
if unsafe { (*vq).event } {
// TODO: tune this threshold
bufs = (unsafe { (*vq).__slate_anon_13.packed.vring.num }.wrapping_sub(unsafe { (*vq).vq.num_free }).wrapping_mul(3) / 4) as u16;
let mut __v0: u16;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u16>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_555() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*vq).last_used_idx) as *const u16) };
last_used_idx = __v0;
wrap_counter = packed_used_wrap_counter(last_used_idx) as u16;
used_idx = (packed_last_used(last_used_idx) as i32).wrapping_add(bufs as i32) as u16;
if ((used_idx as i32) as u32) >= unsafe { (*vq).__slate_anon_13.packed.vring.num } {
used_idx = ((used_idx as i32) as u32).wrapping_sub(unsafe { (*vq).__slate_anon_13.packed.vring.num }) as u16;
wrap_counter = ((wrap_counter as i32) ^ 1) as u16;
}
unsafe {
(*(*vq).__slate_anon_13.packed.vring.driver).off_wrap = ((used_idx as i32) | (wrap_counter as i32) << 15i32) as u16;
}
// We need to update event offset and event wrap
// counter first before updating event flags.
virtio_wmb(unsafe { (*vq).weak_barriers });
}
if ((unsafe { (*vq).__slate_anon_13.packed.event_flags_shadow }) as i32) == 1 {
unsafe {
(*vq).__slate_anon_13.packed.event_flags_shadow = (if (*vq).event { 2i32 } else { 0i32 }) as u16;
}
unsafe {
(*(*vq).__slate_anon_13.packed.vring.driver).flags = (*vq).__slate_anon_13.packed.event_flags_shadow;
}
}
// We need to update event suppression structure first
// before re-checking for more used buffers.
virtio_mb(unsafe { (*vq).weak_barriers });
let mut __v6: u16;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u16>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_556() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v6 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*vq).last_used_idx) as *const u16) };
last_used_idx = __v6;
wrap_counter = packed_used_wrap_counter(last_used_idx) as u16;
used_idx = packed_last_used(last_used_idx);
if is_used_desc_packed(vq as *const vring_virtqueue, used_idx, wrap_counter != 0) {
return false;
}
true
}

#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_detach_unused_buf_packed")]
extern "C" fn virtqueue_detach_unused_buf_packed(vq: *mut vring_virtqueue) -> *mut () {
let mut i: u32 = 0;
while i < unsafe { (*vq).__slate_anon_13.packed.vring.num } {
if !unsafe { (*unsafe { (*vq).__slate_anon_13.packed.desc_state }.wrapping_offset(i as isize)).data }.is_null() {
// detach_buf clears data, so grab it now.
let buf: *mut () = unsafe { (*unsafe { (*vq).__slate_anon_13.packed.desc_state }.wrapping_offset(i as isize)).data };
if virtqueue_is_in_order(vq as *const vring_virtqueue) {
detach_buf_packed_in_order(vq, i, core::ptr::null_mut::<*mut ()>());
} else {
detach_buf_packed(vq, i, core::ptr::null_mut::<*mut ()>());
}
return buf;
}
i = i.wrapping_add(1);
}
// That should have freed everything.
loop {
if unlikely!((unsafe { (*vq).vq.num_free }) != unsafe { (*vq).__slate_anon_13.packed.vring.num }) != 0 {
loop {
unsafe {
core::arch::asm!("557: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 557b - ., 3; .popsection\n# {0}", const 557 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_virtqueue_detach_unused_buf_packed_3, sym __slate_asm_string_virtqueue_detach_unused_buf_packed_4, const 2414 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
core::ptr::null_mut::<()>()
}

fn vring_alloc_desc_extra(num: u32) -> *mut vring_desc_extra {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut i: u32 = 0;
let mut __v0: *mut vring_desc_extra;
let __obj_size: u64 = size_mul(core::mem::size_of::<vring_desc_extra>() as u64, num as u64);
let mut __v1: *mut ();
let mut __v2: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_3 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_3)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_5 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_5)
} });
}
__v2 = _res;
__v1 = __v2;
__v0 = __v1 as *mut vring_desc_extra;
let desc_extra: *mut vring_desc_extra = __v0;
if desc_extra.is_null() {
return core::ptr::null_mut::<vring_desc_extra>();
}
unsafe { memset(desc_extra as *mut (), 0i32, (num as u64).wrapping_mul(core::mem::size_of::<vring_desc_extra>() as u64)) };
while i < num.wrapping_sub(1) {
unsafe {
(*desc_extra.wrapping_offset(i as isize)).next = i.wrapping_add(1) as u16;
}
i = i.wrapping_add(1);
}
unsafe {
(*desc_extra.wrapping_offset(num.wrapping_sub(1) as isize)).next = 0;
}
desc_extra
}

fn vring_free_packed(vring_packed: &mut vring_virtqueue_packed, vdev: *mut virtio_device, map: virtio_map) {
if !unsafe { (*(vring_packed as *mut vring_virtqueue_packed)).vring.desc }.is_null() {
vring_free_queue(vdev, unsafe { (*(vring_packed as *mut vring_virtqueue_packed)).ring_size_in_bytes }, (unsafe { (*(vring_packed as *mut vring_virtqueue_packed)).vring.desc }) as *mut (), unsafe { (*(vring_packed as *mut vring_virtqueue_packed)).ring_dma_addr }, map);
}
if !unsafe { (*(vring_packed as *mut vring_virtqueue_packed)).vring.driver }.is_null() {
vring_free_queue(vdev, unsafe { (*(vring_packed as *mut vring_virtqueue_packed)).event_size_in_bytes }, (unsafe { (*(vring_packed as *mut vring_virtqueue_packed)).vring.driver }) as *mut (), unsafe { (*(vring_packed as *mut vring_virtqueue_packed)).driver_event_dma_addr }, map);
}
if !unsafe { (*(vring_packed as *mut vring_virtqueue_packed)).vring.device }.is_null() {
vring_free_queue(vdev, unsafe { (*(vring_packed as *mut vring_virtqueue_packed)).event_size_in_bytes }, (unsafe { (*(vring_packed as *mut vring_virtqueue_packed)).vring.device }) as *mut (), unsafe { (*(vring_packed as *mut vring_virtqueue_packed)).device_event_dma_addr }, map);
}
unsafe { kfree((*(vring_packed as *mut vring_virtqueue_packed)).desc_state as *const ()) };
unsafe { kfree((*(vring_packed as *mut vring_virtqueue_packed)).desc_extra as *const ()) };
}

fn vring_alloc_queue_packed(vring_packed: &mut vring_virtqueue_packed, vdev: *mut virtio_device, num: u32, map: virtio_map) -> i32 {
let mut event_size_in_bytes: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut u64 = core::ptr::addr_of_mut!(event_size_in_bytes) as *mut u64;
let mut ring_size_in_bytes: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut u64 = core::ptr::addr_of_mut!(ring_size_in_bytes) as *mut u64;
let mut device_event_dma_addr: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut u64 = core::ptr::addr_of_mut!(device_event_dma_addr) as *mut u64;
let mut driver_event_dma_addr: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut u64 = core::ptr::addr_of_mut!(driver_event_dma_addr) as *mut u64;
let mut ring_dma_addr: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u64 = core::ptr::addr_of_mut!(ring_dma_addr) as *mut u64;
let mut device: core::mem::MaybeUninit<*mut vring_packed_desc_event> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut vring_packed_desc_event = core::ptr::addr_of_mut!(device) as *mut *mut vring_packed_desc_event;
let mut driver: core::mem::MaybeUninit<*mut vring_packed_desc_event> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut vring_packed_desc_event = core::ptr::addr_of_mut!(driver) as *mut *mut vring_packed_desc_event;
let mut ring: core::mem::MaybeUninit<*mut vring_packed_desc> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut vring_packed_desc = core::ptr::addr_of_mut!(ring) as *mut *mut vring_packed_desc;
unsafe {
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_1 = (num as u64).wrapping_mul(core::mem::size_of::<vring_packed_desc>() as u64);
*__slate_slot_7 = vring_alloc_queue(vdev, *__slate_slot_1, core::ptr::addr_of_mut!(*__slate_slot_4), ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32) | (BIT!(13i32) as u32) | (BIT!(8i32) as u32), map) as *mut vring_packed_desc;
if !(*__slate_slot_7).is_null() {
(*(vring_packed as *mut vring_virtqueue_packed)).vring.desc = *__slate_slot_7;
(*(vring_packed as *mut vring_virtqueue_packed)).ring_dma_addr = *__slate_slot_4;
(*(vring_packed as *mut vring_virtqueue_packed)).ring_size_in_bytes = *__slate_slot_1;
*__slate_slot_0 = core::mem::size_of::<vring_packed_desc_event>() as u64;
*__slate_slot_6 = vring_alloc_queue(vdev, *__slate_slot_0, core::ptr::addr_of_mut!(*__slate_slot_3), ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32) | (BIT!(13i32) as u32) | (BIT!(8i32) as u32), map) as *mut vring_packed_desc_event;
if !(*__slate_slot_6).is_null() {
(*(vring_packed as *mut vring_virtqueue_packed)).vring.driver = *__slate_slot_6;
(*(vring_packed as *mut vring_virtqueue_packed)).event_size_in_bytes = *__slate_slot_0;
(*(vring_packed as *mut vring_virtqueue_packed)).driver_event_dma_addr = *__slate_slot_3;
*__slate_slot_5 = vring_alloc_queue(vdev, *__slate_slot_0, core::ptr::addr_of_mut!(*__slate_slot_2), ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32) | (BIT!(13i32) as u32) | (BIT!(8i32) as u32), map) as *mut vring_packed_desc_event;
if !(*__slate_slot_5).is_null() {
(*(vring_packed as *mut vring_virtqueue_packed)).vring.device = *__slate_slot_5;
(*(vring_packed as *mut vring_virtqueue_packed)).device_event_dma_addr = *__slate_slot_2;
(*(vring_packed as *mut vring_virtqueue_packed)).vring.num = num;
return 0;
}
}
}
vring_free_packed(vring_packed, vdev, map);
return 12i32.wrapping_neg();
}
}

fn vring_alloc_state_extra_packed(vring_packed: &mut vring_virtqueue_packed) -> i32 {
let mut __slate_compound_9: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_11: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut num: u32 = unsafe { core::mem::zeroed() };
let mut extra: core::mem::MaybeUninit<*mut vring_desc_extra> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut vring_desc_extra = core::ptr::addr_of_mut!(extra) as *mut *mut vring_desc_extra;
let mut state: core::mem::MaybeUninit<*mut vring_desc_state_packed> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut vring_desc_state_packed = core::ptr::addr_of_mut!(state) as *mut *mut vring_desc_state_packed;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
num = (*(vring_packed as *mut vring_virtqueue_packed)).vring.num;
__obj_size = size_mul(core::mem::size_of::<vring_desc_state_packed>() as u64, num as u64);
let __slate_auto_bytes_3 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_0 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_1 = _kmalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_9 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_9)
});
alloc_tag_restore!();
} else {
*__slate_slot_1 = _kmalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_11 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_11)
});
}
*__slate_slot_8 = *__slate_slot_1 as *mut vring_desc_state_packed;
if !(*__slate_slot_8).is_null() {
memset(*__slate_slot_8 as *mut (), 0i32, (num as u64).wrapping_mul(core::mem::size_of::<vring_desc_state_packed>() as u64));
*__slate_slot_7 = vring_alloc_desc_extra(num);
if (*__slate_slot_7).is_null() {
kfree(*__slate_slot_8 as *const ());
} else {
(*(vring_packed as *mut vring_virtqueue_packed)).desc_state = *__slate_slot_8;
(*(vring_packed as *mut vring_virtqueue_packed)).desc_extra = *__slate_slot_7;
return 0;
}
}
return 12i32.wrapping_neg();
}
}

fn virtqueue_vring_init_packed(vring_packed: *mut vring_virtqueue_packed, callback: bool) {
unsafe {
(*vring_packed).next_avail_idx = 0;
}
unsafe {
(*vring_packed).avail_wrap_counter = true;
}
unsafe {
(*vring_packed).event_flags_shadow = 0;
}
unsafe {
(*vring_packed).avail_used_flags = 128u16;
}
// No callback?  Tell other side not to bother us.
if !callback {
unsafe {
(*vring_packed).event_flags_shadow = 1;
}
unsafe {
(*(*vring_packed).vring.driver).flags = (*vring_packed).event_flags_shadow;
}
}
}

fn virtqueue_vring_attach_packed(vq: *mut vring_virtqueue, vring_packed: &mut vring_virtqueue_packed) {
unsafe {
(*vq).__slate_anon_13.packed = *(vring_packed as *mut vring_virtqueue_packed);
}
if virtqueue_is_in_order(vq as *const vring_virtqueue) {
unsafe {
(*vq).batch_last.id = !0;
}
} else {
// Put everything in free lists. Note that
// next_avail_idx is sufficient with IN_ORDER so
// free_head is unused.
unsafe {
(*vq).free_head = 0;
}
}
}

#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_reset_packed")]
extern "C" fn virtqueue_reset_packed(vq: *mut vring_virtqueue) {
unsafe { memset((*vq).__slate_anon_13.packed.vring.device as *mut (), 0i32, (*vq).__slate_anon_13.packed.event_size_in_bytes) };
unsafe { memset((*vq).__slate_anon_13.packed.vring.driver as *mut (), 0i32, (*vq).__slate_anon_13.packed.event_size_in_bytes) };
// we need to reset the desc.flags. For more, see is_used_desc_packed()
unsafe { memset((*vq).__slate_anon_13.packed.vring.desc as *mut (), 0i32, (*vq).__slate_anon_13.packed.ring_size_in_bytes) };
virtqueue_init(vq, unsafe { (*vq).__slate_anon_13.packed.vring.num });
virtqueue_vring_init_packed(unsafe { core::ptr::addr_of_mut!((*vq).__slate_anon_13.packed) }, (unsafe { (*vq).vq.callback }) != None);
}

fn __vring_new_virtqueue_packed(index: u32, vring_packed: &mut vring_virtqueue_packed, vdev: *mut virtio_device, weak_barriers: bool, context: bool, notify: Option<unsafe extern "C" fn(*mut virtqueue) -> bool>, callback: Option<unsafe extern "C" fn(*mut virtqueue)>, name: *const u8, map: virtio_map) -> *mut virtqueue {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut vq: *mut vring_virtqueue = unsafe { core::mem::zeroed() };
let mut err: i32 = 0;
let mut __v0: *mut vring_virtqueue;
let __obj_size: u64 = size_mul(core::mem::size_of::<vring_virtqueue>() as u64, 1u64);
let mut __v1: *mut ();
let mut __v2: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_3 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_3)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_noprof(__obj_size, 3264u32, unsafe { *{
__slate_compound_5 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_5)
} });
}
__v2 = _res;
__v1 = __v2;
__v0 = __v1 as *mut vring_virtqueue;
vq = __v0;
if vq.is_null() {
return core::ptr::null_mut::<virtqueue>();
}
unsafe {
(*vq).vq.callback = callback;
}
unsafe {
(*vq).vq.vdev = vdev;
}
unsafe {
(*vq).vq.name = name;
}
unsafe {
(*vq).vq.index = index;
}
unsafe {
(*vq).vq.reset = false;
}
unsafe {
(*vq).we_own_ring = false;
}
unsafe {
(*vq).notify = notify;
}
unsafe {
(*vq).weak_barriers = weak_barriers;
}
unsafe {
(*vq).broken = false;
}
unsafe {
(*vq).map = map;
}
unsafe {
(*vq).use_map_api = vring_use_map_api(vdev as *const virtio_device);
}
unsafe {
(*vq).indirect = virtio_has_feature(vdev as *const virtio_device, 28u32) && !context;
}
unsafe {
(*vq).event = virtio_has_feature(vdev as *const virtio_device, 29u32);
}
unsafe {
(*vq).layout = (if virtio_has_feature(vdev as *const virtio_device, 35u32) { 3i32 } else { 1i32 }) as u32;
}
if virtio_has_feature(vdev as *const virtio_device, 36u32) {
unsafe {
(*vq).weak_barriers = false;
}
}
err = vring_alloc_state_extra_packed(vring_packed);
if err != 0 {
unsafe { kfree(vq as *const ()) };
return core::ptr::null_mut::<virtqueue>();
}
virtqueue_vring_init_packed(vring_packed as *mut vring_virtqueue_packed, callback != None);
virtqueue_init(vq, unsafe { (*(vring_packed as *mut vring_virtqueue_packed)).vring.num });
virtqueue_vring_attach_packed(vq, vring_packed);
spin_lock(unsafe { core::ptr::addr_of_mut!((*vdev).vqs_list_lock) });
list_add_tail(unsafe { core::ptr::addr_of_mut!((*vq).vq.list) }, unsafe { core::ptr::addr_of_mut!((*vdev).vqs) });
spin_unlock(unsafe { core::ptr::addr_of_mut!((*vdev).vqs_list_lock) });
unsafe { core::ptr::addr_of_mut!((*vq).vq) }
}

fn vring_create_virtqueue_packed(index: u32, num: u32, vring_align: u32, vdev: *mut virtio_device, weak_barriers: bool, may_reduce_num: bool, context: bool, notify: Option<unsafe extern "C" fn(*mut virtqueue) -> bool>, callback: Option<unsafe extern "C" fn(*mut virtqueue)>, name: *const u8, map: virtio_map) -> *mut virtqueue {
let mut vring_packed: vring_virtqueue_packed = {
let mut __t0: vring_virtqueue_packed = unsafe { core::mem::zeroed() };
__t0
};
if vring_alloc_queue_packed(unsafe { &mut (*core::ptr::addr_of_mut!(vring_packed)) }, vdev, num, map) != 0 {
return core::ptr::null_mut::<virtqueue>();
}
let vq: *mut virtqueue = __vring_new_virtqueue_packed(index, unsafe { &mut (*core::ptr::addr_of_mut!(vring_packed)) }, vdev, weak_barriers, context, notify, callback, name, map);
if vq.is_null() {
vring_free_packed(unsafe { &mut (*core::ptr::addr_of_mut!(vring_packed)) }, vdev, map);
return core::ptr::null_mut::<virtqueue>();
}
unsafe {
(*to_vvq!(vq)).we_own_ring = true;
}
vq
}

#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_resize_packed")]
extern "C" fn virtqueue_resize_packed(vq: *mut vring_virtqueue, num: u32) -> i32 {
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut vdev: *mut virtio_device = unsafe { core::mem::zeroed() };
let mut vring_packed: core::mem::MaybeUninit<vring_virtqueue_packed> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut vring_virtqueue_packed = core::ptr::addr_of_mut!(vring_packed) as *mut vring_virtqueue_packed;
unsafe {
core::ptr::write(__slate_slot_2, {
let mut __t0: vring_virtqueue_packed = core::mem::zeroed();
__t0
});
vdev = (*vq).vq.vdev;
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if vring_alloc_queue_packed(&mut (*core::ptr::addr_of_mut!(*__slate_slot_2)), vdev, num, (*vq).map) == 0 {
*__slate_slot_0 = vring_alloc_state_extra_packed(&mut (*core::ptr::addr_of_mut!(*__slate_slot_2)));
if *__slate_slot_0 != 0 {
vring_free_packed(&mut (*core::ptr::addr_of_mut!(*__slate_slot_2)), vdev, (*vq).map);
} else {
vring_free(core::ptr::addr_of_mut!((*vq).vq));
virtqueue_vring_init_packed(core::ptr::addr_of_mut!(*__slate_slot_2), (*vq).vq.callback != None);
virtqueue_init(vq, (*__slate_slot_2).vring.num);
virtqueue_vring_attach_packed(vq, &mut (*core::ptr::addr_of_mut!(*__slate_slot_2)));
return 0;
}
}
virtqueue_reset_packed(vq);
return 12i32.wrapping_neg();
}
}

static split_ops: __SlateConst<virtqueue_ops> = __SlateConst(virtqueue_ops { add: Some(virtqueue_add_split), get: Some(virtqueue_get_buf_ctx_split), kick_prepare: Some(virtqueue_kick_prepare_split), disable_cb: Some(virtqueue_disable_cb_split), enable_cb_delayed: Some(virtqueue_enable_cb_delayed_split), enable_cb_prepare: Some(virtqueue_enable_cb_prepare_split), poll: Some(virtqueue_poll_split), detach_unused_buf: Some(virtqueue_detach_unused_buf_split), more_used: Some(more_used_split), resize: Some(virtqueue_resize_split), reset: Some(virtqueue_reset_split) });

static packed_ops: __SlateConst<virtqueue_ops> = __SlateConst(virtqueue_ops { add: Some(virtqueue_add_packed), get: Some(virtqueue_get_buf_ctx_packed), kick_prepare: Some(virtqueue_kick_prepare_packed), disable_cb: Some(virtqueue_disable_cb_packed), enable_cb_delayed: Some(virtqueue_enable_cb_delayed_packed), enable_cb_prepare: Some(virtqueue_enable_cb_prepare_packed), poll: Some(virtqueue_poll_packed), detach_unused_buf: Some(virtqueue_detach_unused_buf_packed), more_used: Some(more_used_packed), resize: Some(virtqueue_resize_packed), reset: Some(virtqueue_reset_packed) });

static split_in_order_ops: __SlateConst<virtqueue_ops> = __SlateConst(virtqueue_ops { add: Some(virtqueue_add_split), get: Some(virtqueue_get_buf_ctx_split_in_order), kick_prepare: Some(virtqueue_kick_prepare_split), disable_cb: Some(virtqueue_disable_cb_split), enable_cb_delayed: Some(virtqueue_enable_cb_delayed_split), enable_cb_prepare: Some(virtqueue_enable_cb_prepare_split), poll: Some(virtqueue_poll_split), detach_unused_buf: Some(virtqueue_detach_unused_buf_split), more_used: Some(more_used_split_in_order), resize: Some(virtqueue_resize_split), reset: Some(virtqueue_reset_split) });

static packed_in_order_ops: __SlateConst<virtqueue_ops> = __SlateConst(virtqueue_ops { add: Some(virtqueue_add_packed_in_order), get: Some(virtqueue_get_buf_ctx_packed_in_order), kick_prepare: Some(virtqueue_kick_prepare_packed), disable_cb: Some(virtqueue_disable_cb_packed), enable_cb_delayed: Some(virtqueue_enable_cb_delayed_packed), enable_cb_prepare: Some(virtqueue_enable_cb_prepare_packed), poll: Some(virtqueue_poll_packed), detach_unused_buf: Some(virtqueue_detach_unused_buf_packed), more_used: Some(more_used_packed_in_order), resize: Some(virtqueue_resize_packed), reset: Some(virtqueue_reset_packed) });

fn virtqueue_disable_and_recycle(_vq: *mut virtqueue, recycle: Option<unsafe extern "C" fn(*mut virtqueue, *mut ())>) -> i32 {
let vq: *mut vring_virtqueue = to_vvq!(_vq);
let vdev: *mut virtio_device = unsafe { (*vq).vq.vdev };
let mut err: i32 = 0;
if !unsafe { (*vq).we_own_ring } {
return 1i32.wrapping_neg();
}
if (unsafe { (*(*vdev).config).disable_vq_and_reset }) == None {
return 2i32.wrapping_neg();
}
if (unsafe { (*(*vdev).config).enable_vq_after_reset }) == None {
return 2i32.wrapping_neg();
}
err = unsafe { unsafe { (*(*vdev).config).disable_vq_and_reset }.unwrap()(_vq) };
if err != 0 {
return err;
}
loop {
let __v1: *mut () = virtqueue_detach_unused_buf(_vq);
let buf: *mut () = __v1;
if __v1.is_null() {
break;
}
unsafe { recycle.unwrap()(_vq, buf) };
}
0
}

fn virtqueue_enable_after_reset(_vq: *mut virtqueue) -> i32 {
let vq: *mut vring_virtqueue = to_vvq!(_vq);
let vdev: *mut virtio_device = unsafe { (*vq).vq.vdev };
if (unsafe { unsafe { (*(*vdev).config).enable_vq_after_reset }.unwrap()(_vq) }) != 0 {
return 16i32.wrapping_neg();
}
0
}

// Generic functions and exported symbols.
fn virtqueue_add(_vq: *mut virtqueue, sgs: *mut *mut scatterlist, total_sg: u32, out_sgs: u32, in_sgs: u32, data: *mut (), ctx: *mut (), premapped: bool, gfp: u32, attr: u64) -> i32 {
let vq: *mut vring_virtqueue = to_vvq!(_vq);
let mut __v1: i32;
let __VIRTQUEUE_CALL_vq: *mut vring_virtqueue = vq;
let mut ret: i32 = 0;
match (unsafe { (*__VIRTQUEUE_CALL_vq).layout }) as u32 {
0 => {
ret = unsafe { unsafe { split_ops.0.add }.unwrap()(__VIRTQUEUE_CALL_vq, sgs, total_sg, out_sgs, in_sgs, data, ctx, premapped, gfp, attr) };
}
1 => {
ret = unsafe { unsafe { packed_ops.0.add }.unwrap()(__VIRTQUEUE_CALL_vq, sgs, total_sg, out_sgs, in_sgs, data, ctx, premapped, gfp, attr) };
}
2 => {
ret = unsafe { unsafe { split_in_order_ops.0.add }.unwrap()(vq, sgs, total_sg, out_sgs, in_sgs, data, ctx, premapped, gfp, attr) };
}
3 => {
ret = unsafe { unsafe { packed_in_order_ops.0.add }.unwrap()(vq, sgs, total_sg, out_sgs, in_sgs, data, ctx, premapped, gfp, attr) };
}
_ => {
loop {
unsafe {
core::arch::asm!("558: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 558b - ., 3; .popsection\n# {0}", const 558 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_virtqueue_add_3, sym __slate_asm_string_virtqueue_add_4, const 2871 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
}
__v1 = ret;
__v1
}

/// virtqueue_add_sgs - expose buffers to other end
/// @_vq: the struct virtqueue we're talking about.
/// @sgs: array of terminated scatterlists.
/// @out_sgs: the number of scatterlists readable by other side
/// @in_sgs: the number of scatterlists which are writable (after readable ones)
/// @data: the token identifying the buffer.
/// @gfp: how to do memory allocations (if necessary).
///
/// Caller must ensure we don't call this with other virtqueue operations
/// at the same time (except where noted).
///
/// Returns zero or a negative error (ie. ENOSPC, ENOMEM, EIO).
///
/// NB: ENOSPC is a special code that is only returned on an attempt to add a
/// buffer to a full VQ. It indicates that some buffers are outstanding and that
/// the operation can be retried after some buffers have been used.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_add_sgs")]
extern "C" fn virtqueue_add_sgs(_vq: *mut virtqueue, sgs: *mut *mut scatterlist, out_sgs: u32, in_sgs: u32, data: *mut (), gfp: u32) -> i32 {
// Count them first.
let mut i: u32 = 0;
let mut total_sg: u32 = 0;
while i < out_sgs.wrapping_add(in_sgs) {
let mut sg: *mut scatterlist = unsafe { core::mem::zeroed() };
sg = unsafe { *sgs.wrapping_offset(i as isize) };
while !sg.is_null() {
total_sg = total_sg.wrapping_add(1);
sg = sg_next(sg);
}
i = i.wrapping_add(1);
}
virtqueue_add(_vq, sgs, total_sg, out_sgs, in_sgs, data, core::ptr::null_mut::<()>(), false, gfp, 0u64)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_virtqueue_add_sgs_559: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut virtqueue, *mut *mut scatterlist, u32, u32, *mut (), u32) -> i32>, *mut ()>(Some(virtqueue_add_sgs)) };

/// virtqueue_add_outbuf - expose output buffers to other end
/// @vq: the struct virtqueue we're talking about.
/// @sg: scatterlist (must be well-formed and terminated!)
/// @num: the number of entries in @sg readable by other side
/// @data: the token identifying the buffer.
/// @gfp: how to do memory allocations (if necessary).
///
/// Caller must ensure we don't call this with other virtqueue operations
/// at the same time (except where noted).
///
/// Returns zero or a negative error (ie. ENOSPC, ENOMEM, EIO).
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_add_outbuf")]
extern "C" fn virtqueue_add_outbuf(vq: *mut virtqueue, mut sg: *mut scatterlist, num: u32, data: *mut (), gfp: u32) -> i32 {
virtqueue_add(vq, core::ptr::addr_of_mut!(sg), num, 1u32, 0u32, data, core::ptr::null_mut::<()>(), false, gfp, 0u64)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_virtqueue_add_outbuf_560: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut virtqueue, *mut scatterlist, u32, *mut (), u32) -> i32>, *mut ()>(Some(virtqueue_add_outbuf)) };

/// virtqueue_add_outbuf_premapped - expose output buffers to other end
/// @vq: the struct virtqueue we're talking about.
/// @sg: scatterlist (must be well-formed and terminated!)
/// @num: the number of entries in @sg readable by other side
/// @data: the token identifying the buffer.
/// @gfp: how to do memory allocations (if necessary).
///
/// Caller must ensure we don't call this with other virtqueue operations
/// at the same time (except where noted).
///
/// Return:
/// Returns zero or a negative error (ie. ENOSPC, ENOMEM, EIO).
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_add_outbuf_premapped")]
extern "C" fn virtqueue_add_outbuf_premapped(vq: *mut virtqueue, mut sg: *mut scatterlist, num: u32, data: *mut (), gfp: u32) -> i32 {
virtqueue_add(vq, core::ptr::addr_of_mut!(sg), num, 1u32, 0u32, data, core::ptr::null_mut::<()>(), true, gfp, 0u64)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_virtqueue_add_outbuf_premapped_561: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut virtqueue, *mut scatterlist, u32, *mut (), u32) -> i32>, *mut ()>(Some(virtqueue_add_outbuf_premapped)) };

/// virtqueue_add_inbuf - expose input buffers to other end
/// @vq: the struct virtqueue we're talking about.
/// @sg: scatterlist (must be well-formed and terminated!)
/// @num: the number of entries in @sg writable by other side
/// @data: the token identifying the buffer.
/// @gfp: how to do memory allocations (if necessary).
///
/// Caller must ensure we don't call this with other virtqueue operations
/// at the same time (except where noted).
///
/// Returns zero or a negative error (ie. ENOSPC, ENOMEM, EIO).
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_add_inbuf")]
extern "C" fn virtqueue_add_inbuf(vq: *mut virtqueue, mut sg: *mut scatterlist, num: u32, data: *mut (), gfp: u32) -> i32 {
virtqueue_add(vq, core::ptr::addr_of_mut!(sg), num, 0u32, 1u32, data, core::ptr::null_mut::<()>(), false, gfp, 0u64)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_virtqueue_add_inbuf_562: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut virtqueue, *mut scatterlist, u32, *mut (), u32) -> i32>, *mut ()>(Some(virtqueue_add_inbuf)) };

/// virtqueue_add_inbuf_cache_clean - expose input buffers with cache clean
/// @vq: the struct virtqueue we're talking about.
/// @sg: scatterlist (must be well-formed and terminated!)
/// @num: the number of entries in @sg writable by other side
/// @data: the token identifying the buffer.
/// @gfp: how to do memory allocations (if necessary).
///
/// Same as virtqueue_add_inbuf but passes DMA_ATTR_DEBUGGING_IGNORE_CACHELINES
/// to indicate that the CPU will not dirty any cacheline overlapping this buffer
/// while it is available, and to suppress overlapping cacheline warnings in DMA
/// debug builds.
///
/// Caller must ensure we don't call this with other virtqueue operations
/// at the same time (except where noted).
///
/// Returns zero or a negative error (ie. ENOSPC, ENOMEM, EIO).
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_add_inbuf_cache_clean")]
extern "C" fn virtqueue_add_inbuf_cache_clean(vq: *mut virtqueue, mut sg: *mut scatterlist, num: u32, data: *mut (), gfp: u32) -> i32 {
virtqueue_add(vq, core::ptr::addr_of_mut!(sg), num, 0u32, 1u32, data, core::ptr::null_mut::<()>(), false, gfp, 2048u64)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_virtqueue_add_inbuf_cache_clean_563: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut virtqueue, *mut scatterlist, u32, *mut (), u32) -> i32>, *mut ()>(Some(virtqueue_add_inbuf_cache_clean)) };

/// virtqueue_add_inbuf_ctx - expose input buffers to other end
/// @vq: the struct virtqueue we're talking about.
/// @sg: scatterlist (must be well-formed and terminated!)
/// @num: the number of entries in @sg writable by other side
/// @data: the token identifying the buffer.
/// @ctx: extra context for the token
/// @gfp: how to do memory allocations (if necessary).
///
/// Caller must ensure we don't call this with other virtqueue operations
/// at the same time (except where noted).
///
/// Returns zero or a negative error (ie. ENOSPC, ENOMEM, EIO).
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_add_inbuf_ctx")]
extern "C" fn virtqueue_add_inbuf_ctx(vq: *mut virtqueue, mut sg: *mut scatterlist, num: u32, data: *mut (), ctx: *mut (), gfp: u32) -> i32 {
virtqueue_add(vq, core::ptr::addr_of_mut!(sg), num, 0u32, 1u32, data, ctx, false, gfp, 0u64)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_virtqueue_add_inbuf_ctx_564: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut virtqueue, *mut scatterlist, u32, *mut (), *mut (), u32) -> i32>, *mut ()>(Some(virtqueue_add_inbuf_ctx)) };

/// virtqueue_add_inbuf_premapped - expose input buffers to other end
/// @vq: the struct virtqueue we're talking about.
/// @sg: scatterlist (must be well-formed and terminated!)
/// @num: the number of entries in @sg writable by other side
/// @data: the token identifying the buffer.
/// @ctx: extra context for the token
/// @gfp: how to do memory allocations (if necessary).
///
/// Caller must ensure we don't call this with other virtqueue operations
/// at the same time (except where noted).
///
/// Return:
/// Returns zero or a negative error (ie. ENOSPC, ENOMEM, EIO).
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_add_inbuf_premapped")]
extern "C" fn virtqueue_add_inbuf_premapped(vq: *mut virtqueue, mut sg: *mut scatterlist, num: u32, data: *mut (), ctx: *mut (), gfp: u32) -> i32 {
virtqueue_add(vq, core::ptr::addr_of_mut!(sg), num, 0u32, 1u32, data, ctx, true, gfp, 0u64)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_virtqueue_add_inbuf_premapped_565: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut virtqueue, *mut scatterlist, u32, *mut (), *mut (), u32) -> i32>, *mut ()>(Some(virtqueue_add_inbuf_premapped)) };

/// virtqueue_dma_dev - get the dma dev
/// @_vq: the struct virtqueue we're talking about.
///
/// Returns the dma dev. That can been used for dma api.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_dma_dev")]
extern "C" fn virtqueue_dma_dev(_vq: *mut virtqueue) -> *mut device {
let vq: *mut vring_virtqueue = to_vvq!(_vq);
if (unsafe { (*vq).use_map_api }) && unsafe { (*(*_vq).vdev).map }.is_null() {
return unsafe { (*vq).map.dma_dev };
} else {
return core::ptr::null_mut::<device>();
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_virtqueue_dma_dev_566: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut virtqueue) -> *mut device>, *mut ()>(Some(virtqueue_dma_dev)) };

/// virtqueue_kick_prepare - first half of split virtqueue_kick call.
/// @_vq: the struct virtqueue
///
/// Instead of virtqueue_kick(), you can do:
/// 	if (virtqueue_kick_prepare(vq))
/// 		virtqueue_notify(vq);
///
/// This is sometimes useful because the virtqueue_kick_prepare() needs
/// to be serialized, but the actual virtqueue_notify() call does not.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_kick_prepare")]
extern "C" fn virtqueue_kick_prepare(_vq: *mut virtqueue) -> bool {
let vq: *mut vring_virtqueue = to_vvq!(_vq);
let mut __v1: bool;
let __VIRTQUEUE_CALL_vq: *mut vring_virtqueue = vq;
let mut ret: bool = false;
match (unsafe { (*__VIRTQUEUE_CALL_vq).layout }) as u32 {
0 => {
ret = unsafe { unsafe { split_ops.0.kick_prepare }.unwrap()(__VIRTQUEUE_CALL_vq) };
}
1 => {
ret = unsafe { unsafe { packed_ops.0.kick_prepare }.unwrap()(__VIRTQUEUE_CALL_vq) };
}
2 => {
ret = unsafe { unsafe { split_in_order_ops.0.kick_prepare }.unwrap()(vq) };
}
3 => {
ret = unsafe { unsafe { packed_in_order_ops.0.kick_prepare }.unwrap()(vq) };
}
_ => {
loop {
unsafe {
core::arch::asm!("567: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 567b - ., 3; .popsection\n# {0}", const 567 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_virtqueue_kick_prepare_3, sym __slate_asm_string_virtqueue_kick_prepare_4, const 3089 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
}
__v1 = ret;
__v1
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_virtqueue_kick_prepare_568: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut virtqueue) -> bool>, *mut ()>(Some(virtqueue_kick_prepare)) };

/// virtqueue_notify - second half of split virtqueue_kick call.
/// @_vq: the struct virtqueue
///
/// This does not need to be serialized.
///
/// Returns false if host notify failed or queue is broken, otherwise true.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_notify")]
extern "C" fn virtqueue_notify(_vq: *mut virtqueue) -> bool {
let vq: *mut vring_virtqueue = to_vvq!(_vq);
if unlikely!(unsafe { (*vq).broken }) != 0 {
return false;
}
// Prod other side to tell it about changes.
if !unsafe { unsafe { (*vq).notify }.unwrap()(_vq) } {
unsafe {
(*vq).broken = true;
}
return false;
}
true
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_virtqueue_notify_569: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut virtqueue) -> bool>, *mut ()>(Some(virtqueue_notify)) };

/// virtqueue_kick - update after add_buf
/// @vq: the struct virtqueue
///
/// After one or more virtqueue_add_* calls, invoke this to kick
/// the other side.
///
/// Caller must ensure we don't call this with other virtqueue
/// operations at the same time (except where noted).
///
/// Returns false if kick failed, otherwise true.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_kick")]
extern "C" fn virtqueue_kick(vq: *mut virtqueue) -> bool {
if virtqueue_kick_prepare(vq) {
return virtqueue_notify(vq);
}
true
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_virtqueue_kick_570: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut virtqueue) -> bool>, *mut ()>(Some(virtqueue_kick)) };

/// virtqueue_get_buf_ctx - get the next used buffer
/// @_vq: the struct virtqueue we're talking about.
/// @len: the length written into the buffer
/// @ctx: extra context for the token
///
/// If the device wrote data into the buffer, @len will be set to the
/// amount written.  This means you don't need to clear the buffer
/// beforehand to ensure there's no data leakage in the case of short
/// writes.
///
/// Caller must ensure we don't call this with other virtqueue
/// operations at the same time (except where noted).
///
/// Returns NULL if there are no used buffers, or the "data" token
/// handed to virtqueue_add_*().
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_get_buf_ctx")]
extern "C" fn virtqueue_get_buf_ctx(_vq: *mut virtqueue, len: *mut u32, ctx: *mut *mut ()) -> *mut () {
let vq: *mut vring_virtqueue = to_vvq!(_vq);
let mut __v1: *mut ();
let __VIRTQUEUE_CALL_vq: *mut vring_virtqueue = vq;
let mut ret: *mut () = unsafe { core::mem::zeroed() };
match (unsafe { (*__VIRTQUEUE_CALL_vq).layout }) as u32 {
0 => {
ret = unsafe { unsafe { split_ops.0.get }.unwrap()(__VIRTQUEUE_CALL_vq, len, ctx) };
}
1 => {
ret = unsafe { unsafe { packed_ops.0.get }.unwrap()(__VIRTQUEUE_CALL_vq, len, ctx) };
}
2 => {
ret = unsafe { unsafe { split_in_order_ops.0.get }.unwrap()(vq, len, ctx) };
}
3 => {
ret = unsafe { unsafe { packed_in_order_ops.0.get }.unwrap()(vq, len, ctx) };
}
_ => {
loop {
unsafe {
core::arch::asm!("571: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 571b - ., 3; .popsection\n# {0}", const 571 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_virtqueue_get_buf_ctx_3, sym __slate_asm_string_virtqueue_get_buf_ctx_4, const 3159 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
}
__v1 = ret;
__v1
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_virtqueue_get_buf_ctx_572: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut virtqueue, *mut u32, *mut *mut ()) -> *mut ()>, *mut ()>(Some(virtqueue_get_buf_ctx)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_get_buf")]
extern "C" fn virtqueue_get_buf(_vq: *mut virtqueue, len: *mut u32) -> *mut () {
virtqueue_get_buf_ctx(_vq, len, core::ptr::null_mut::<*mut ()>())
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_virtqueue_get_buf_573: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut virtqueue, *mut u32) -> *mut ()>, *mut ()>(Some(virtqueue_get_buf)) };

/// virtqueue_disable_cb - disable callbacks
/// @_vq: the struct virtqueue we're talking about.
///
/// Note that this is not necessarily synchronous, hence unreliable and only
/// useful as an optimization.
///
/// Unlike other operations, this need not be serialized.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_disable_cb")]
extern "C" fn virtqueue_disable_cb(_vq: *mut virtqueue) {
let vq: *mut vring_virtqueue = to_vvq!(_vq);
let __VIRTQUEUE_CALL_vq: *mut vring_virtqueue = vq;
match (unsafe { (*__VIRTQUEUE_CALL_vq).layout }) as u32 {
0 => {
unsafe { unsafe { split_ops.0.disable_cb }.unwrap()(__VIRTQUEUE_CALL_vq) };
}
1 => {
unsafe { unsafe { packed_ops.0.disable_cb }.unwrap()(__VIRTQUEUE_CALL_vq) };
}
2 => {
unsafe { unsafe { split_in_order_ops.0.disable_cb }.unwrap()(vq) };
}
3 => {
unsafe { unsafe { packed_in_order_ops.0.disable_cb }.unwrap()(vq) };
}
_ => {
loop {
unsafe {
core::arch::asm!("574: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 574b - ., 3; .popsection\n# {0}", const 574 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_virtqueue_disable_cb_2, sym __slate_asm_string_virtqueue_disable_cb_3, const 3181 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_virtqueue_disable_cb_575: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut virtqueue)>, *mut ()>(Some(virtqueue_disable_cb)) };

/// virtqueue_enable_cb_prepare - restart callbacks after disable_cb
/// @_vq: the struct virtqueue we're talking about.
///
/// This re-enables callbacks; it returns current queue state
/// in an opaque unsigned value. This value should be later tested by
/// virtqueue_poll, to detect a possible race between the driver checking for
/// more work, and enabling callbacks.
///
/// Caller must ensure we don't call this with other virtqueue
/// operations at the same time (except where noted).
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_enable_cb_prepare")]
extern "C" fn virtqueue_enable_cb_prepare(_vq: *mut virtqueue) -> u32 {
let vq: *mut vring_virtqueue = to_vvq!(_vq);
if unsafe { (*vq).event_triggered } {
unsafe {
(*vq).event_triggered = false;
}
}
let mut __v1: u32;
let __VIRTQUEUE_CALL_vq: *mut vring_virtqueue = vq;
let mut ret: u32 = 0;
match (unsafe { (*__VIRTQUEUE_CALL_vq).layout }) as u32 {
0 => {
ret = unsafe { unsafe { split_ops.0.enable_cb_prepare }.unwrap()(__VIRTQUEUE_CALL_vq) };
}
1 => {
ret = unsafe { unsafe { packed_ops.0.enable_cb_prepare }.unwrap()(__VIRTQUEUE_CALL_vq) };
}
2 => {
ret = unsafe { unsafe { split_in_order_ops.0.enable_cb_prepare }.unwrap()(vq) };
}
3 => {
ret = unsafe { unsafe { packed_in_order_ops.0.enable_cb_prepare }.unwrap()(vq) };
}
_ => {
loop {
unsafe {
core::arch::asm!("576: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 576b - ., 3; .popsection\n# {0}", const 576 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_virtqueue_enable_cb_prepare_3, sym __slate_asm_string_virtqueue_enable_cb_prepare_4, const 3204 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
}
__v1 = ret;
__v1
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_virtqueue_enable_cb_prepare_577: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut virtqueue) -> u32>, *mut ()>(Some(virtqueue_enable_cb_prepare)) };

/// virtqueue_poll - query pending used buffers
/// @_vq: the struct virtqueue we're talking about.
/// @last_used_idx: virtqueue state (from call to virtqueue_enable_cb_prepare).
///
/// Returns "true" if there are pending used buffers in the queue.
///
/// This does not need to be serialized.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_poll")]
extern "C" fn virtqueue_poll(_vq: *mut virtqueue, last_used_idx: u32) -> bool {
let vq: *mut vring_virtqueue = to_vvq!(_vq);
if unlikely!(unsafe { (*vq).broken }) != 0 {
return false;
}
virtio_mb(unsafe { (*vq).weak_barriers });
let mut __v1: bool;
let __VIRTQUEUE_CALL_vq: *mut vring_virtqueue = vq;
let mut ret: bool = false;
match (unsafe { (*__VIRTQUEUE_CALL_vq).layout }) as u32 {
0 => {
ret = unsafe { unsafe { split_ops.0.poll }.unwrap()(__VIRTQUEUE_CALL_vq as *const vring_virtqueue, last_used_idx) };
}
1 => {
ret = unsafe { unsafe { packed_ops.0.poll }.unwrap()(__VIRTQUEUE_CALL_vq as *const vring_virtqueue, last_used_idx) };
}
2 => {
ret = unsafe { unsafe { split_in_order_ops.0.poll }.unwrap()(vq as *const vring_virtqueue, last_used_idx) };
}
3 => {
ret = unsafe { unsafe { packed_in_order_ops.0.poll }.unwrap()(vq as *const vring_virtqueue, last_used_idx) };
}
_ => {
loop {
unsafe {
core::arch::asm!("578: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 578b - ., 3; .popsection\n# {0}", const 578 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_virtqueue_poll_3, sym __slate_asm_string_virtqueue_poll_4, const 3226 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
}
__v1 = ret;
__v1
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_virtqueue_poll_579: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut virtqueue, u32) -> bool>, *mut ()>(Some(virtqueue_poll)) };

/// virtqueue_enable_cb - restart callbacks after disable_cb.
/// @_vq: the struct virtqueue we're talking about.
///
/// This re-enables callbacks; it returns "false" if there are pending
/// buffers in the queue, to detect a possible race between the driver
/// checking for more work, and enabling callbacks.
///
/// Caller must ensure we don't call this with other virtqueue
/// operations at the same time (except where noted).
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_enable_cb")]
extern "C" fn virtqueue_enable_cb(_vq: *mut virtqueue) -> bool {
let last_used_idx: u32 = virtqueue_enable_cb_prepare(_vq);
!virtqueue_poll(_vq, last_used_idx)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_virtqueue_enable_cb_580: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut virtqueue) -> bool>, *mut ()>(Some(virtqueue_enable_cb)) };

/// virtqueue_enable_cb_delayed - restart callbacks after disable_cb.
/// @_vq: the struct virtqueue we're talking about.
///
/// This re-enables callbacks but hints to the other side to delay
/// interrupts until most of the available buffers have been processed;
/// it returns "false" if there are many pending buffers in the queue,
/// to detect a possible race between the driver checking for more work,
/// and enabling callbacks.
///
/// Caller must ensure we don't call this with other virtqueue
/// operations at the same time (except where noted).
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_enable_cb_delayed")]
extern "C" fn virtqueue_enable_cb_delayed(_vq: *mut virtqueue) -> bool {
let vq: *mut vring_virtqueue = to_vvq!(_vq);
// When the device is broken there is no point in polling used->idx,
// the backend will never update it. Return true to let callers
// exit their cleanup loops instead of spinning forever.
if unlikely!(unsafe { (*vq).broken }) != 0 {
return true;
}
if unsafe { (*vq).event_triggered } {
let mut __v1: bool;
__kcsan_disable_current();
let mut __v: bool = false;
let __v2: bool = false;
unsafe {
(*vq).event_triggered = __v2;
}
__v = __v2;
__kcsan_enable_current();
__v1 = __v;
}
let mut __v3: bool;
let __VIRTQUEUE_CALL_vq: *mut vring_virtqueue = vq;
let mut ret: bool = false;
match (unsafe { (*__VIRTQUEUE_CALL_vq).layout }) as u32 {
0 => {
ret = unsafe { unsafe { split_ops.0.enable_cb_delayed }.unwrap()(__VIRTQUEUE_CALL_vq) };
}
1 => {
ret = unsafe { unsafe { packed_ops.0.enable_cb_delayed }.unwrap()(__VIRTQUEUE_CALL_vq) };
}
2 => {
ret = unsafe { unsafe { split_in_order_ops.0.enable_cb_delayed }.unwrap()(vq) };
}
3 => {
ret = unsafe { unsafe { packed_in_order_ops.0.enable_cb_delayed }.unwrap()(vq) };
}
_ => {
loop {
unsafe {
core::arch::asm!("581: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 581b - ., 3; .popsection\n# {0}", const 581 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_virtqueue_enable_cb_delayed_5, sym __slate_asm_string_virtqueue_enable_cb_delayed_6, const 3277 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
}
__v3 = ret;
__v3
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_virtqueue_enable_cb_delayed_582: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut virtqueue) -> bool>, *mut ()>(Some(virtqueue_enable_cb_delayed)) };

/// virtqueue_detach_unused_buf - detach first unused buffer
/// @_vq: the struct virtqueue we're talking about.
///
/// Returns NULL or the "data" token handed to virtqueue_add_*().
/// This is not valid on an active queue; it is useful for device
/// shutdown or the reset queue.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_detach_unused_buf")]
extern "C" fn virtqueue_detach_unused_buf(_vq: *mut virtqueue) -> *mut () {
let vq: *mut vring_virtqueue = to_vvq!(_vq);
let mut __v1: *mut ();
let __VIRTQUEUE_CALL_vq: *mut vring_virtqueue = vq;
let mut ret: *mut () = unsafe { core::mem::zeroed() };
match (unsafe { (*__VIRTQUEUE_CALL_vq).layout }) as u32 {
0 => {
ret = unsafe { unsafe { split_ops.0.detach_unused_buf }.unwrap()(__VIRTQUEUE_CALL_vq) };
}
1 => {
ret = unsafe { unsafe { packed_ops.0.detach_unused_buf }.unwrap()(__VIRTQUEUE_CALL_vq) };
}
2 => {
ret = unsafe { unsafe { split_in_order_ops.0.detach_unused_buf }.unwrap()(vq) };
}
3 => {
ret = unsafe { unsafe { packed_in_order_ops.0.detach_unused_buf }.unwrap()(vq) };
}
_ => {
loop {
unsafe {
core::arch::asm!("583: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 583b - ., 3; .popsection\n# {0}", const 583 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_virtqueue_detach_unused_buf_3, sym __slate_asm_string_virtqueue_detach_unused_buf_4, const 3293 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
}
__v1 = ret;
__v1
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_virtqueue_detach_unused_buf_584: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut virtqueue) -> *mut ()>, *mut ()>(Some(virtqueue_detach_unused_buf)) };

fn more_used(vq: *const vring_virtqueue) -> bool {
let mut __v0: bool;
let __VIRTQUEUE_CALL_vq: *const vring_virtqueue = vq;
let mut ret: bool = false;
match (unsafe { (*__VIRTQUEUE_CALL_vq).layout }) as u32 {
0 => {
ret = unsafe { unsafe { split_ops.0.more_used }.unwrap()(__VIRTQUEUE_CALL_vq) };
}
1 => {
ret = unsafe { unsafe { packed_ops.0.more_used }.unwrap()(__VIRTQUEUE_CALL_vq) };
}
2 => {
ret = unsafe { unsafe { split_in_order_ops.0.more_used }.unwrap()(vq) };
}
3 => {
ret = unsafe { unsafe { packed_in_order_ops.0.more_used }.unwrap()(vq) };
}
_ => {
loop {
unsafe {
core::arch::asm!("585: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 585b - ., 3; .popsection\n# {0}", const 585 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_more_used_2, sym __slate_asm_string_more_used_3, const 3299 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
}
__v0 = ret;
__v0
}

/// vring_interrupt - notify a virtqueue on an interrupt
/// @irq: the IRQ number (ignored)
/// @_vq: the struct virtqueue to notify
///
/// Calls the callback function of @_vq to process the virtqueue
/// notification.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.vring_interrupt")]
extern "C" fn vring_interrupt(irq: i32, _vq: *mut ()) -> u32 {
let vq: *mut vring_virtqueue = to_vvq__3!(_vq);
if !more_used(vq as *const vring_virtqueue) {
let mut __v1: i32;
if false {
unsafe { _printk(c"\x017virtqueue interrupt with no work for %p\n".as_ptr() as *const u8, vq) };
}
__v1 = 0;
return 0;
}
if unlikely!(unsafe { (*vq).broken }) != 0 {
return 1;
}
// Just a hint for performance: so it's ok that this can be racy!
if unsafe { (*vq).event } {
let mut __v2: bool;
__kcsan_disable_current();
let mut __v: bool = false;
let __v3: bool = true;
unsafe {
(*vq).event_triggered = __v3;
}
__v = __v3;
__kcsan_enable_current();
__v2 = __v;
}
let mut __v4: i32;
if false {
unsafe { _printk(c"\x017virtqueue callback for %p (%p)\n".as_ptr() as *const u8, vq, unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut virtqueue)>, *const u8>((*vq).vq.callback) }) };
}
__v4 = 0;
if (unsafe { (*vq).vq.callback }) != None {
unsafe { unsafe { (*vq).vq.callback }.unwrap()(core::ptr::addr_of_mut!((*vq).vq)) };
}
1
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_vring_interrupt_586: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(i32, *mut ()) -> u32>, *mut ()>(Some(vring_interrupt)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.vring_create_virtqueue")]
extern "C" fn vring_create_virtqueue(index: u32, num: u32, vring_align: u32, vdev: *mut virtio_device, weak_barriers: bool, may_reduce_num: bool, context: bool, notify: Option<unsafe extern "C" fn(*mut virtqueue) -> bool>, callback: Option<unsafe extern "C" fn(*mut virtqueue)>, name: *const u8) -> *mut virtqueue {
let map: virtio_map = virtio_map { dma_dev: unsafe { (*vdev).dev.parent } };
if virtio_has_feature(vdev as *const virtio_device, 34u32) {
return vring_create_virtqueue_packed(index, num, vring_align, vdev, weak_barriers, may_reduce_num, context, notify, callback, name, map);
}
vring_create_virtqueue_split(index, num, vring_align, vdev, weak_barriers, may_reduce_num, context, notify, callback, name, map)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_vring_create_virtqueue_587: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u32, u32, u32, *mut virtio_device, bool, bool, bool, Option<unsafe extern "C" fn(*mut virtqueue) -> bool>, Option<unsafe extern "C" fn(*mut virtqueue)>, *const u8) -> *mut virtqueue>, *mut ()>(Some(vring_create_virtqueue)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.vring_create_virtqueue_map")]
extern "C" fn vring_create_virtqueue_map(index: u32, num: u32, vring_align: u32, vdev: *mut virtio_device, weak_barriers: bool, may_reduce_num: bool, context: bool, notify: Option<unsafe extern "C" fn(*mut virtqueue) -> bool>, callback: Option<unsafe extern "C" fn(*mut virtqueue)>, name: *const u8, map: virtio_map) -> *mut virtqueue {
if virtio_has_feature(vdev as *const virtio_device, 34u32) {
return vring_create_virtqueue_packed(index, num, vring_align, vdev, weak_barriers, may_reduce_num, context, notify, callback, name, map);
}
vring_create_virtqueue_split(index, num, vring_align, vdev, weak_barriers, may_reduce_num, context, notify, callback, name, map)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_vring_create_virtqueue_map_588: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u32, u32, u32, *mut virtio_device, bool, bool, bool, Option<unsafe extern "C" fn(*mut virtqueue) -> bool>, Option<unsafe extern "C" fn(*mut virtqueue)>, *const u8, virtio_map) -> *mut virtqueue>, *mut ()>(Some(vring_create_virtqueue_map)) };

/// virtqueue_resize - resize the vring of vq
/// @_vq: the struct virtqueue we're talking about.
/// @num: new ring num
/// @recycle: callback to recycle unused buffers
/// @recycle_done: callback to be invoked when recycle for all unused buffers done
///
/// When it is really necessary to create a new vring, it will set the current vq
/// into the reset state. Then call the passed callback to recycle the buffer
/// that is no longer used. Only after the new vring is successfully created, the
/// old vring will be released.
///
/// Caller must ensure we don't call this with other virtqueue operations
/// at the same time (except where noted).
///
/// Returns zero or a negative error.
/// 0: success.
/// -ENOMEM: Failed to allocate a new ring, fall back to the original ring size.
///  vq can still work normally
/// -EBUSY: Failed to sync with device, vq may not work properly
/// -ENOENT: Transport or device not supported
/// -E2BIG/-EINVAL: num error
/// -EPERM: Operation not permitted
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_resize")]
extern "C" fn virtqueue_resize(_vq: *mut virtqueue, num: u32, recycle: Option<unsafe extern "C" fn(*mut virtqueue, *mut ())>, recycle_done: Option<unsafe extern "C" fn(*mut virtqueue)>) -> i32 {
let vq: *mut vring_virtqueue = to_vvq!(_vq);
let mut err: i32 = 0;
let mut err_reset: i32 = 0;
if num > unsafe { (*vq).vq.num_max } {
return 7i32.wrapping_neg();
}
if num == 0 {
return 22i32.wrapping_neg();
}
if virtqueue_get_vring_size(_vq as *const virtqueue) == num {
return 0;
}
err = virtqueue_disable_and_recycle(_vq, recycle);
if err != 0 {
return err;
}
if recycle_done != None {
unsafe { recycle_done.unwrap()(_vq) };
}
let mut __v1: i32;
let __VIRTQUEUE_CALL_vq: *mut vring_virtqueue = vq;
let mut ret: i32 = 0;
match (unsafe { (*__VIRTQUEUE_CALL_vq).layout }) as u32 {
0 => {
ret = unsafe { unsafe { split_ops.0.resize }.unwrap()(__VIRTQUEUE_CALL_vq, num) };
}
1 => {
ret = unsafe { unsafe { packed_ops.0.resize }.unwrap()(__VIRTQUEUE_CALL_vq, num) };
}
2 => {
ret = unsafe { unsafe { split_in_order_ops.0.resize }.unwrap()(vq, num) };
}
3 => {
ret = unsafe { unsafe { packed_in_order_ops.0.resize }.unwrap()(vq, num) };
}
_ => {
loop {
unsafe {
core::arch::asm!("589: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 589b - ., 3; .popsection\n# {0}", const 589 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_virtqueue_resize_3, sym __slate_asm_string_virtqueue_resize_4, const 3438 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
}
__v1 = ret;
err = __v1;
err_reset = virtqueue_enable_after_reset(_vq);
if err_reset != 0 {
return err_reset;
}
err
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_virtqueue_resize_590: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut virtqueue, u32, Option<unsafe extern "C" fn(*mut virtqueue, *mut ())>, Option<unsafe extern "C" fn(*mut virtqueue)>) -> i32>, *mut ()>(Some(virtqueue_resize)) };

/// virtqueue_reset - detach and recycle all unused buffers
/// @_vq: the struct virtqueue we're talking about.
/// @recycle: callback to recycle unused buffers
/// @recycle_done: callback to be invoked when recycle for all unused buffers done
///
/// Caller must ensure we don't call this with other virtqueue operations
/// at the same time (except where noted).
///
/// Returns zero or a negative error.
/// 0: success.
/// -EBUSY: Failed to sync with device, vq may not work properly
/// -ENOENT: Transport or device not supported
/// -EPERM: Operation not permitted
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_reset")]
extern "C" fn virtqueue_reset(_vq: *mut virtqueue, recycle: Option<unsafe extern "C" fn(*mut virtqueue, *mut ())>, recycle_done: Option<unsafe extern "C" fn(*mut virtqueue)>) -> i32 {
let vq: *mut vring_virtqueue = to_vvq!(_vq);
let err: i32 = virtqueue_disable_and_recycle(_vq, recycle);
if err != 0 {
return err;
}
if recycle_done != None {
unsafe { recycle_done.unwrap()(_vq) };
}
let __VIRTQUEUE_CALL_vq: *mut vring_virtqueue = vq;
match (unsafe { (*__VIRTQUEUE_CALL_vq).layout }) as u32 {
0 => {
unsafe { unsafe { split_ops.0.reset }.unwrap()(__VIRTQUEUE_CALL_vq) };
}
1 => {
unsafe { unsafe { packed_ops.0.reset }.unwrap()(__VIRTQUEUE_CALL_vq) };
}
2 => {
unsafe { unsafe { split_in_order_ops.0.reset }.unwrap()(vq) };
}
3 => {
unsafe { unsafe { packed_in_order_ops.0.reset }.unwrap()(vq) };
}
_ => {
loop {
unsafe {
core::arch::asm!("591: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 591b - ., 3; .popsection\n# {0}", const 591 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_virtqueue_reset_2, sym __slate_asm_string_virtqueue_reset_3, const 3476 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
}
virtqueue_enable_after_reset(_vq)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_virtqueue_reset_592: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut virtqueue, Option<unsafe extern "C" fn(*mut virtqueue, *mut ())>, Option<unsafe extern "C" fn(*mut virtqueue)>) -> i32>, *mut ()>(Some(virtqueue_reset)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.vring_new_virtqueue")]
extern "C" fn vring_new_virtqueue(index: u32, num: u32, vring_align: u32, vdev: *mut virtio_device, weak_barriers: bool, context: bool, pages: *mut (), notify: Option<unsafe extern "C" fn(*mut virtqueue) -> bool>, callback: Option<unsafe extern "C" fn(*mut virtqueue)>, name: *const u8) -> *mut virtqueue {
let mut vring_split: vring_virtqueue_split = {
let mut __t0: vring_virtqueue_split = unsafe { core::mem::zeroed() };
__t0
};
let map: virtio_map = virtio_map { dma_dev: unsafe { (*vdev).dev.parent } };
if virtio_has_feature(vdev as *const virtio_device, 34u32) {
let mut vring_packed: vring_virtqueue_packed = {
let mut __t1: vring_virtqueue_packed = unsafe { core::mem::zeroed() };
__t1
};
vring_packed.vring.num = num;
vring_packed.vring.desc = pages as *mut vring_packed_desc;
return __vring_new_virtqueue_packed(index, unsafe { &mut (*core::ptr::addr_of_mut!(vring_packed)) }, vdev, weak_barriers, context, notify, callback, name, map);
}
vring_init(core::ptr::addr_of_mut!(vring_split.vring), num, pages, vring_align as u64);
__vring_new_virtqueue_split(index, core::ptr::addr_of_mut!(vring_split), vdev, weak_barriers, context, notify, callback, name, map)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_vring_new_virtqueue_593: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u32, u32, u32, *mut virtio_device, bool, bool, *mut (), Option<unsafe extern "C" fn(*mut virtqueue) -> bool>, Option<unsafe extern "C" fn(*mut virtqueue)>, *const u8) -> *mut virtqueue>, *mut ()>(Some(vring_new_virtqueue)) };

fn vring_free(_vq: *mut virtqueue) {
let vq: *mut vring_virtqueue = to_vvq!(_vq);
if unsafe { (*vq).we_own_ring } {
if virtqueue_is_packed(vq as *const vring_virtqueue) {
vring_free_queue(unsafe { (*vq).vq.vdev }, unsafe { (*vq).__slate_anon_13.packed.ring_size_in_bytes }, (unsafe { (*vq).__slate_anon_13.packed.vring.desc }) as *mut (), unsafe { (*vq).__slate_anon_13.packed.ring_dma_addr }, unsafe { (*vq).map });
vring_free_queue(unsafe { (*vq).vq.vdev }, unsafe { (*vq).__slate_anon_13.packed.event_size_in_bytes }, (unsafe { (*vq).__slate_anon_13.packed.vring.driver }) as *mut (), unsafe { (*vq).__slate_anon_13.packed.driver_event_dma_addr }, unsafe { (*vq).map });
vring_free_queue(unsafe { (*vq).vq.vdev }, unsafe { (*vq).__slate_anon_13.packed.event_size_in_bytes }, (unsafe { (*vq).__slate_anon_13.packed.vring.device }) as *mut (), unsafe { (*vq).__slate_anon_13.packed.device_event_dma_addr }, unsafe { (*vq).map });
unsafe { kfree((*vq).__slate_anon_13.packed.desc_state as *const ()) };
unsafe { kfree((*vq).__slate_anon_13.packed.desc_extra as *const ()) };
} else {
vring_free_queue(unsafe { (*vq).vq.vdev }, unsafe { (*vq).__slate_anon_13.split.queue_size_in_bytes }, (unsafe { (*vq).__slate_anon_13.split.vring.desc }) as *mut (), unsafe { (*vq).__slate_anon_13.split.queue_dma_addr }, unsafe { (*vq).map });
}
}
if !virtqueue_is_packed(vq as *const vring_virtqueue) {
unsafe { kfree((*vq).__slate_anon_13.split.desc_state as *const ()) };
unsafe { kfree((*vq).__slate_anon_13.split.desc_extra as *const ()) };
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.vring_del_virtqueue")]
extern "C" fn vring_del_virtqueue(_vq: *mut virtqueue) {
let vq: *mut vring_virtqueue = to_vvq!(_vq);
spin_lock(unsafe { core::ptr::addr_of_mut!((*(*vq).vq.vdev).vqs_list_lock) });
list_del(unsafe { core::ptr::addr_of_mut!((*_vq).list) });
spin_unlock(unsafe { core::ptr::addr_of_mut!((*(*vq).vq.vdev).vqs_list_lock) });
vring_free(_vq);
unsafe { kfree(vq as *const ()) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_vring_del_virtqueue_594: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut virtqueue)>, *mut ()>(Some(vring_del_virtqueue)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.vring_notification_data")]
extern "C" fn vring_notification_data(_vq: *mut virtqueue) -> u32 {
let vq: *mut vring_virtqueue = to_vvq!(_vq);
let next: u16 = if virtqueue_is_packed(vq as *const vring_virtqueue) {
(((unsafe { (*vq).__slate_anon_13.packed.next_avail_idx }) as i32) & !32768i32.wrapping_neg() | ((unsafe { (*vq).__slate_anon_13.packed.avail_wrap_counter }) as i32) << 15i32) as u16
} else {
unsafe { (*vq).__slate_anon_13.split.avail_idx_shadow }
};
(((next as i32) << 16i32) as u32) | unsafe { (*_vq).index }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_vring_notification_data_595: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut virtqueue) -> u32>, *mut ()>(Some(vring_notification_data)) };

/// Manipulates transport-specific feature bits.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.vring_transport_features")]
extern "C" fn vring_transport_features(vdev: *mut virtio_device) {
for i in 28u32..42u32 {
match i {
28 => {
}
29 => {
}
32 => {
}
33 => {
}
34 => {
}
36 => {
}
38 => {
}
35 => {
}
_ => {
__virtio_clear_bit(vdev, i); // We don't understand this bit.
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_vring_transport_features_596: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut virtio_device)>, *mut ()>(Some(vring_transport_features)) };

/// virtqueue_get_vring_size - return the size of the virtqueue's vring
/// @_vq: the struct virtqueue containing the vring of interest.
///
/// Returns the size of the vring.  This is mainly used for boasting to
/// userspace.  Unlike other operations, this need not be serialized.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_get_vring_size")]
extern "C" fn virtqueue_get_vring_size(_vq: *const virtqueue) -> u32 {
let vq: *const vring_virtqueue = to_vvq__2!(_vq);
if virtqueue_is_packed(vq) { unsafe { (*vq).__slate_anon_13.packed.vring.num } } else { unsafe { (*vq).__slate_anon_13.split.vring.num } }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_virtqueue_get_vring_size_597: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const virtqueue) -> u32>, *mut ()>(Some(virtqueue_get_vring_size)) };

// This function should only be called by the core, not directly by the driver.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___virtqueue_break_599: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut virtqueue)>, *mut ()>(Some(__virtqueue_break)) };

// This function should only be called by the core, not directly by the driver.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___virtqueue_unbreak_601: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut virtqueue)>, *mut ()>(Some(__virtqueue_unbreak)) };

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_virtqueue_is_broken_603: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const virtqueue) -> bool>, *mut ()>(Some(virtqueue_is_broken)) };

// This should prevent the device from being used, allowing drivers to
// recover.  You may need to grab appropriate locks to flush.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_virtio_break_device_605: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut virtio_device)>, *mut ()>(Some(virtio_break_device)) };

// This should allow the device to be used by the driver. You may
// need to grab appropriate locks to flush the write to
// vq->broken. This should only be used in some specific case e.g
// (probing and restoring). This function should only be called by the
// core, not directly by the driver.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___virtio_unbreak_device_607: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut virtio_device)>, *mut ()>(Some(__virtio_unbreak_device)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_get_desc_addr")]
extern "C" fn virtqueue_get_desc_addr(_vq: *const virtqueue) -> u64 {
let vq: *const vring_virtqueue = to_vvq__2!(_vq);
loop {
if unlikely!(!unsafe { (*vq).we_own_ring }) != 0 {
loop {
unsafe {
core::arch::asm!("608: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 608b - ., 3; .popsection\n# {0}", const 608 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_virtqueue_get_desc_addr_1, sym __slate_asm_string_virtqueue_get_desc_addr_2, const 3710 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
if virtqueue_is_packed(vq) {
return unsafe { (*vq).__slate_anon_13.packed.ring_dma_addr };
}
unsafe { (*vq).__slate_anon_13.split.queue_dma_addr }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_virtqueue_get_desc_addr_609: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const virtqueue) -> u64>, *mut ()>(Some(virtqueue_get_desc_addr)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_get_avail_addr")]
extern "C" fn virtqueue_get_avail_addr(_vq: *const virtqueue) -> u64 {
let vq: *const vring_virtqueue = to_vvq__2!(_vq);
loop {
if unlikely!(!unsafe { (*vq).we_own_ring }) != 0 {
loop {
unsafe {
core::arch::asm!("610: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 610b - ., 3; .popsection\n# {0}", const 610 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_virtqueue_get_avail_addr_1, sym __slate_asm_string_virtqueue_get_avail_addr_2, const 3723 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
if virtqueue_is_packed(vq) {
return unsafe { (*vq).__slate_anon_13.packed.driver_event_dma_addr };
}
unsafe { (*vq).__slate_anon_13.split.queue_dma_addr }.wrapping_add(((unsafe { ((*vq).__slate_anon_13.split.vring.avail as *mut u8).offset_from((*vq).__slate_anon_13.split.vring.desc as *mut u8) }) as i64) as u64)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_virtqueue_get_avail_addr_611: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const virtqueue) -> u64>, *mut ()>(Some(virtqueue_get_avail_addr)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_get_used_addr")]
extern "C" fn virtqueue_get_used_addr(_vq: *const virtqueue) -> u64 {
let vq: *const vring_virtqueue = to_vvq__2!(_vq);
loop {
if unlikely!(!unsafe { (*vq).we_own_ring }) != 0 {
loop {
unsafe {
core::arch::asm!("612: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 612b - ., 3; .popsection\n# {0}", const 612 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_virtqueue_get_used_addr_1, sym __slate_asm_string_virtqueue_get_used_addr_2, const 3737 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
if virtqueue_is_packed(vq) {
return unsafe { (*vq).__slate_anon_13.packed.device_event_dma_addr };
}
unsafe { (*vq).__slate_anon_13.split.queue_dma_addr }.wrapping_add(((unsafe { ((*vq).__slate_anon_13.split.vring.used as *mut u8).offset_from((*vq).__slate_anon_13.split.vring.desc as *mut u8) }) as i64) as u64)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_virtqueue_get_used_addr_613: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const virtqueue) -> u64>, *mut ()>(Some(virtqueue_get_used_addr)) };

/// Only available for split ring
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_get_vring")]
extern "C" fn virtqueue_get_vring(vq: *const virtqueue) -> *const vring {
unsafe { core::ptr::addr_of!((*to_vvq__2!(vq)).__slate_anon_13.split.vring) }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_virtqueue_get_vring_614: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const virtqueue) -> *const vring>, *mut ()>(Some(virtqueue_get_vring)) };

/// virtqueue_map_alloc_coherent - alloc coherent mapping
/// @vdev: the virtio device we are talking to
/// @map: metadata for performing mapping
/// @size: the size of the buffer
/// @map_handle: the pointer to the mapped address
/// @gfp: allocation flag (GFP_XXX)
///
/// return virtual address or NULL on error
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_map_alloc_coherent")]
extern "C" fn virtqueue_map_alloc_coherent(vdev: *mut virtio_device, map: virtio_map, size: u64, map_handle: *mut u64, gfp: u32) -> *mut () {
if !unsafe { (*vdev).map }.is_null() {
return unsafe { unsafe { (*(*vdev).map).alloc }.unwrap()(map, size, map_handle, gfp) };
} else {
return dma_alloc_coherent(unsafe { map.dma_dev }, size, map_handle, gfp);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_virtqueue_map_alloc_coherent_615: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut virtio_device, virtio_map, u64, *mut u64, u32) -> *mut ()>, *mut ()>(Some(virtqueue_map_alloc_coherent)) };

/// virtqueue_map_free_coherent - free coherent mapping
/// @vdev: the virtio device we are talking to
/// @map: metadata for performing mapping
/// @size: the size of the buffer
/// @vaddr: the virtual address that needs to be freed
/// @map_handle: the mapped address that needs to be freed
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_map_free_coherent")]
extern "C" fn virtqueue_map_free_coherent(vdev: *mut virtio_device, map: virtio_map, size: u64, vaddr: *mut (), map_handle: u64) {
if !unsafe { (*vdev).map }.is_null() {
unsafe { unsafe { (*(*vdev).map).free }.unwrap()(map, size, vaddr, map_handle, 0u64) };
} else {
dma_free_coherent(unsafe { map.dma_dev }, size, vaddr, map_handle);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_virtqueue_map_free_coherent_616: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut virtio_device, virtio_map, u64, *mut (), u64)>, *mut ()>(Some(virtqueue_map_free_coherent)) };

/// virtqueue_map_page_attrs - map a page to the device
/// @_vq: the virtqueue we are talking to
/// @page: the page that will be mapped by the device
/// @offset: the offset in the page for a buffer
/// @size: the buffer size
/// @dir: mapping direction
/// @attrs: mapping attributes
///
/// Returns mapped address. Caller should check that by virtqueue_map_mapping_error().
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_map_page_attrs")]
extern "C" fn virtqueue_map_page_attrs(_vq: *const virtqueue, page: *mut page, offset: u64, size: u64, dir: u32, attrs: u64) -> u64 {
let vq: *const vring_virtqueue = to_vvq__2!(_vq);
let vdev: *mut virtio_device = unsafe { (*_vq).vdev };
if !unsafe { (*vdev).map }.is_null() {
return unsafe { unsafe { (*(*vdev).map).map_page }.unwrap()((*vq).map, page, offset, size, dir, attrs) };
}
unsafe { dma_map_page_attrs(vring_dma_dev(vq), page, offset, size, dir, attrs) }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_virtqueue_map_page_attrs_617: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const virtqueue, *mut page, u64, u64, u32, u64) -> u64>, *mut ()>(Some(virtqueue_map_page_attrs)) };

/// virtqueue_unmap_page_attrs - map a page to the device
/// @_vq: the virtqueue we are talking to
/// @map_handle: the mapped address
/// @size: the buffer size
/// @dir: mapping direction
/// @attrs: unmapping attributes
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_unmap_page_attrs")]
extern "C" fn virtqueue_unmap_page_attrs(_vq: *const virtqueue, map_handle: u64, size: u64, dir: u32, attrs: u64) {
let vq: *const vring_virtqueue = to_vvq__2!(_vq);
let vdev: *mut virtio_device = unsafe { (*_vq).vdev };
if !unsafe { (*vdev).map }.is_null() {
unsafe { unsafe { (*(*vdev).map).unmap_page }.unwrap()((*vq).map, map_handle, size, dir, attrs) };
} else {
unsafe { dma_unmap_page_attrs(vring_dma_dev(vq), map_handle, size, dir, attrs) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_virtqueue_unmap_page_attrs_618: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const virtqueue, u64, u64, u32, u64)>, *mut ()>(Some(virtqueue_unmap_page_attrs)) };

/// virtqueue_map_single_attrs - map DMA for _vq
/// @_vq: the struct virtqueue we're talking about.
/// @ptr: the pointer of the buffer to do dma
/// @size: the size of the buffer to do dma
/// @dir: DMA direction
/// @attrs: DMA Attrs
///
/// The caller calls this to do dma mapping in advance. The DMA address can be
/// passed to this _vq when it is in pre-mapped mode.
///
/// return mapped address. Caller should check that by virtqueue_map_mapping_error().
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_map_single_attrs")]
extern "C" fn virtqueue_map_single_attrs(_vq: *const virtqueue, ptr: *mut (), size: u64, dir: u32, attrs: u64) -> u64 {
let vq: *const vring_virtqueue = to_vvq__2!(_vq);
if !unsafe { (*vq).use_map_api } {
kmsan_handle_dma(virt_to_phys(ptr as *mut ()), size, dir);
return virt_to_phys(ptr as *mut ());
}
// DMA must never operate on areas that might be remapped.
let mut __v1: i32;
let __ret_warn_on: i32 = !(!unsafe { is_vmalloc_addr(ptr as *const ()) }) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
2i32 | BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s %s: rejecting DMA map of vmalloc memory\n".as_ptr() as *const u8, dev_driver_string(core::ptr::addr_of_mut!((*(*_vq).vdev).dev) as *const device), dev_name(core::ptr::addr_of_mut!((*(*_vq).vdev).dev) as *const device)) };
let mut __v2: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v2 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v3: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_virtqueue_map_single_attrs_4, sym __slate_asm_string_virtqueue_map_single_attrs_5, const 3883 as i32, const 2323 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v3 = bug;
unsafe { __v2.unwrap()(__v3, dev_driver_string(core::ptr::addr_of_mut!((*(*_vq).vdev).dev) as *const device), dev_name(core::ptr::addr_of_mut!((*(*_vq).vdev).dev) as *const device)) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v1 = __ret_warn_on;
if __v1 != 0 {
return !0;
}
virtqueue_map_page_attrs(unsafe { core::ptr::addr_of!((*vq).vq) }, virt_to_page!(ptr), offset_in_page!(ptr), size, dir, attrs)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_619: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_virtqueue_map_single_attrs_620: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const virtqueue, *mut (), u64, u32, u64) -> u64>, *mut ()>(Some(virtqueue_map_single_attrs)) };

/// virtqueue_unmap_single_attrs - unmap map for _vq
/// @_vq: the struct virtqueue we're talking about.
/// @addr: the dma address to unmap
/// @size: the size of the buffer
/// @dir: DMA direction
/// @attrs: DMA Attrs
///
/// Unmap the address that is mapped by the virtqueue_map_* APIs.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_unmap_single_attrs")]
extern "C" fn virtqueue_unmap_single_attrs(_vq: *const virtqueue, addr: u64, size: u64, dir: u32, attrs: u64) {
let vq: *const vring_virtqueue = to_vvq__2!(_vq);
if !unsafe { (*vq).use_map_api } {
return;
}
virtqueue_unmap_page_attrs(_vq, addr, size, dir, attrs);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_virtqueue_unmap_single_attrs_621: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const virtqueue, u64, u64, u32, u64)>, *mut ()>(Some(virtqueue_unmap_single_attrs)) };

/// virtqueue_map_mapping_error - check dma address
/// @_vq: the struct virtqueue we're talking about.
/// @addr: DMA address
///
/// Returns 0 means dma valid. Other means invalid dma address.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_map_mapping_error")]
extern "C" fn virtqueue_map_mapping_error(_vq: *const virtqueue, addr: u64) -> i32 {
let vq: *const vring_virtqueue = to_vvq__2!(_vq);
vring_mapping_error(vq, addr)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_virtqueue_map_mapping_error_622: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const virtqueue, u64) -> i32>, *mut ()>(Some(virtqueue_map_mapping_error)) };

/// virtqueue_map_need_sync - check a dma address needs sync
/// @_vq: the struct virtqueue we're talking about.
/// @addr: DMA address
///
/// Check if the dma address mapped by the virtqueue_map_* APIs needs to be
/// synchronized
///
/// return bool
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_map_need_sync")]
extern "C" fn virtqueue_map_need_sync(_vq: *const virtqueue, addr: u64) -> bool {
let vq: *const vring_virtqueue = to_vvq__2!(_vq);
let vdev: *mut virtio_device = unsafe { (*_vq).vdev };
if !unsafe { (*vq).use_map_api } {
return false;
}
if !unsafe { (*vdev).map }.is_null() {
return unsafe { unsafe { (*(*vdev).map).need_sync }.unwrap()((*vq).map, addr) };
} else {
return dma_need_sync(vring_dma_dev(vq), addr);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_virtqueue_map_need_sync_623: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const virtqueue, u64) -> bool>, *mut ()>(Some(virtqueue_map_need_sync)) };

/// virtqueue_map_sync_single_range_for_cpu - map sync for cpu
/// @_vq: the struct virtqueue we're talking about.
/// @addr: DMA address
/// @offset: DMA address offset
/// @size: buf size for sync
/// @dir: DMA direction
///
/// Before calling this function, use virtqueue_map_need_sync() to confirm that
/// the DMA address really needs to be synchronized
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_map_sync_single_range_for_cpu")]
extern "C" fn virtqueue_map_sync_single_range_for_cpu(_vq: *const virtqueue, addr: u64, offset: u64, size: u64, dir: u32) {
let vq: *const vring_virtqueue = to_vvq__2!(_vq);
let vdev: *mut virtio_device = unsafe { (*_vq).vdev };
if !unsafe { (*vq).use_map_api } {
return;
}
if !unsafe { (*vdev).map }.is_null() {
unsafe { unsafe { (*(*vdev).map).sync_single_for_cpu }.unwrap()((*vq).map, addr.wrapping_add(offset), size, dir) };
} else {
dma_sync_single_range_for_cpu(vring_dma_dev(vq), addr, offset, size, dir);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_virtqueue_map_sync_single_range_for_cpu_624: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const virtqueue, u64, u64, u64, u32)>, *mut ()>(Some(virtqueue_map_sync_single_range_for_cpu)) };

/// virtqueue_map_sync_single_range_for_device - map sync for device
/// @_vq: the struct virtqueue we're talking about.
/// @addr: DMA address
/// @offset: DMA address offset
/// @size: buf size for sync
/// @dir: DMA direction
///
/// Before calling this function, use virtqueue_map_need_sync() to confirm that
/// the DMA address really needs to be synchronized
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.virtio_ring.virtqueue_map_sync_single_range_for_device")]
extern "C" fn virtqueue_map_sync_single_range_for_device(_vq: *const virtqueue, addr: u64, offset: u64, size: u64, dir: u32) {
let vq: *const vring_virtqueue = to_vvq__2!(_vq);
let vdev: *mut virtio_device = unsafe { (*_vq).vdev };
if !unsafe { (*vq).use_map_api } {
return;
}
if !unsafe { (*vdev).map }.is_null() {
unsafe { unsafe { (*(*vdev).map).sync_single_for_device }.unwrap()((*vq).map, addr.wrapping_add(offset), size, dir) };
} else {
dma_sync_single_range_for_device(vring_dma_dev(vq), addr, offset, size, dir);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_virtqueue_map_sync_single_range_for_device_625: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const virtqueue, u64, u64, u64, u32)>, *mut ()>(Some(virtqueue_map_sync_single_range_for_device)) };

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_626: [u8; 51] = [118, 105, 114, 116, 105, 111, 95, 114, 105, 110, 103, 46, 100, 101, 115, 99, 114, 105, 112, 116, 105, 111, 110, 61, 86, 105, 114, 116, 105, 111, 32, 114, 105, 110, 103, 32, 105, 109, 112, 108, 101, 109, 101, 110, 116, 97, 116, 105, 111, 110, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_627: [u8; 44] = [118, 105, 114, 116, 105, 111, 95, 114, 105, 110, 103, 46, 102, 105, 108, 101, 61, 100, 114, 105, 118, 101, 114, 115, 47, 118, 105, 114, 116, 105, 111, 47, 118, 105, 114, 116, 105, 111, 95, 114, 105, 110, 103, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_628: [u8; 24] = [118, 105, 114, 116, 105, 111, 95, 114, 105, 110, 103, 46, 108, 105, 99, 101, 110, 115, 101, 61, 71, 80, 76, 0];

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_detach_indirect_split_0: [u8; 1] = *b"\0";

static __slate_asm_string_detach_indirect_split_1: [u8; 58] = *b"/home/takashi/c-corpus/linux/drivers/virtio/virtio_ring.c\0";

static __slate_asm_string_detach_indirect_split_5: [u8; 1] = *b"\0";

static __slate_asm_string_detach_indirect_split_6: [u8; 58] = *b"/home/takashi/c-corpus/linux/drivers/virtio/virtio_ring.c\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

static __slate_asm_string_more_used_2: [u8; 1] = *b"\0";

static __slate_asm_string_more_used_3: [u8; 58] = *b"/home/takashi/c-corpus/linux/drivers/virtio/virtio_ring.c\0";

static __slate_asm_string_virtio_features_chk_bit_2: [u8; 1] = *b"\0";

static __slate_asm_string_virtio_features_chk_bit_3: [u8; 61] = *b"/home/takashi/c-corpus/linux/include/linux/virtio_features.h\0";

static __slate_asm_string_virtqueue_add_3: [u8; 1] = *b"\0";

static __slate_asm_string_virtqueue_add_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/drivers/virtio/virtio_ring.c\0";

static __slate_asm_string_virtqueue_add_packed_47: [u8; 1] = *b"\0";

static __slate_asm_string_virtqueue_add_packed_48: [u8; 58] = *b"/home/takashi/c-corpus/linux/drivers/virtio/virtio_ring.c\0";

static __slate_asm_string_virtqueue_add_packed_49: [u8; 1] = *b"\0";

static __slate_asm_string_virtqueue_add_packed_50: [u8; 58] = *b"/home/takashi/c-corpus/linux/drivers/virtio/virtio_ring.c\0";

static __slate_asm_string_virtqueue_add_packed_51: [u8; 1] = *b"\0";

static __slate_asm_string_virtqueue_add_packed_52: [u8; 58] = *b"/home/takashi/c-corpus/linux/drivers/virtio/virtio_ring.c\0";

static __slate_asm_string_virtqueue_add_packed_53: [u8; 1] = *b"\0";

static __slate_asm_string_virtqueue_add_packed_54: [u8; 58] = *b"/home/takashi/c-corpus/linux/drivers/virtio/virtio_ring.c\0";

static __slate_asm_string_virtqueue_add_packed_55: [u8; 1] = *b"\0";

static __slate_asm_string_virtqueue_add_packed_56: [u8; 58] = *b"/home/takashi/c-corpus/linux/drivers/virtio/virtio_ring.c\0";

static __slate_asm_string_virtqueue_add_packed_57: [u8; 1] = *b"\0";

static __slate_asm_string_virtqueue_add_packed_58: [u8; 58] = *b"/home/takashi/c-corpus/linux/drivers/virtio/virtio_ring.c\0";

static __slate_asm_string_virtqueue_add_packed_in_order_53: [u8; 1] = *b"\0";

static __slate_asm_string_virtqueue_add_packed_in_order_54: [u8; 58] = *b"/home/takashi/c-corpus/linux/drivers/virtio/virtio_ring.c\0";

static __slate_asm_string_virtqueue_add_packed_in_order_55: [u8; 1] = *b"\0";

static __slate_asm_string_virtqueue_add_packed_in_order_56: [u8; 58] = *b"/home/takashi/c-corpus/linux/drivers/virtio/virtio_ring.c\0";

static __slate_asm_string_virtqueue_add_packed_in_order_57: [u8; 1] = *b"\0";

static __slate_asm_string_virtqueue_add_packed_in_order_58: [u8; 58] = *b"/home/takashi/c-corpus/linux/drivers/virtio/virtio_ring.c\0";

static __slate_asm_string_virtqueue_add_packed_in_order_59: [u8; 1] = *b"\0";

static __slate_asm_string_virtqueue_add_packed_in_order_60: [u8; 58] = *b"/home/takashi/c-corpus/linux/drivers/virtio/virtio_ring.c\0";

static __slate_asm_string_virtqueue_add_split_55: [u8; 1] = *b"\0";

static __slate_asm_string_virtqueue_add_split_56: [u8; 58] = *b"/home/takashi/c-corpus/linux/drivers/virtio/virtio_ring.c\0";

static __slate_asm_string_virtqueue_add_split_57: [u8; 1] = *b"\0";

static __slate_asm_string_virtqueue_add_split_58: [u8; 58] = *b"/home/takashi/c-corpus/linux/drivers/virtio/virtio_ring.c\0";

static __slate_asm_string_virtqueue_add_split_59: [u8; 1] = *b"\0";

static __slate_asm_string_virtqueue_add_split_60: [u8; 58] = *b"/home/takashi/c-corpus/linux/drivers/virtio/virtio_ring.c\0";

static __slate_asm_string_virtqueue_add_split_61: [u8; 1] = *b"\0";

static __slate_asm_string_virtqueue_add_split_62: [u8; 58] = *b"/home/takashi/c-corpus/linux/drivers/virtio/virtio_ring.c\0";

static __slate_asm_string_virtqueue_detach_unused_buf_3: [u8; 1] = *b"\0";

static __slate_asm_string_virtqueue_detach_unused_buf_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/drivers/virtio/virtio_ring.c\0";

static __slate_asm_string_virtqueue_detach_unused_buf_packed_3: [u8; 1] = *b"\0";

static __slate_asm_string_virtqueue_detach_unused_buf_packed_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/drivers/virtio/virtio_ring.c\0";

static __slate_asm_string_virtqueue_detach_unused_buf_split_6: [u8; 1] = *b"\0";

static __slate_asm_string_virtqueue_detach_unused_buf_split_7: [u8; 58] = *b"/home/takashi/c-corpus/linux/drivers/virtio/virtio_ring.c\0";

static __slate_asm_string_virtqueue_disable_cb_2: [u8; 1] = *b"\0";

static __slate_asm_string_virtqueue_disable_cb_3: [u8; 58] = *b"/home/takashi/c-corpus/linux/drivers/virtio/virtio_ring.c\0";

static __slate_asm_string_virtqueue_enable_cb_delayed_5: [u8; 1] = *b"\0";

static __slate_asm_string_virtqueue_enable_cb_delayed_6: [u8; 58] = *b"/home/takashi/c-corpus/linux/drivers/virtio/virtio_ring.c\0";

static __slate_asm_string_virtqueue_enable_cb_prepare_3: [u8; 1] = *b"\0";

static __slate_asm_string_virtqueue_enable_cb_prepare_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/drivers/virtio/virtio_ring.c\0";

static __slate_asm_string_virtqueue_get_avail_addr_1: [u8; 1] = *b"\0";

static __slate_asm_string_virtqueue_get_avail_addr_2: [u8; 58] = *b"/home/takashi/c-corpus/linux/drivers/virtio/virtio_ring.c\0";

static __slate_asm_string_virtqueue_get_buf_ctx_3: [u8; 1] = *b"\0";

static __slate_asm_string_virtqueue_get_buf_ctx_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/drivers/virtio/virtio_ring.c\0";

static __slate_asm_string_virtqueue_get_desc_addr_1: [u8; 1] = *b"\0";

static __slate_asm_string_virtqueue_get_desc_addr_2: [u8; 58] = *b"/home/takashi/c-corpus/linux/drivers/virtio/virtio_ring.c\0";

static __slate_asm_string_virtqueue_get_used_addr_1: [u8; 1] = *b"\0";

static __slate_asm_string_virtqueue_get_used_addr_2: [u8; 58] = *b"/home/takashi/c-corpus/linux/drivers/virtio/virtio_ring.c\0";

static __slate_asm_string_virtqueue_kick_prepare_3: [u8; 1] = *b"\0";

static __slate_asm_string_virtqueue_kick_prepare_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/drivers/virtio/virtio_ring.c\0";

static __slate_asm_string_virtqueue_map_single_attrs_4: [u8; 44] = *b"%s %s: rejecting DMA map of vmalloc memory\n\0";

static __slate_asm_string_virtqueue_map_single_attrs_5: [u8; 58] = *b"/home/takashi/c-corpus/linux/drivers/virtio/virtio_ring.c\0";

static __slate_asm_string_virtqueue_poll_3: [u8; 1] = *b"\0";

static __slate_asm_string_virtqueue_poll_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/drivers/virtio/virtio_ring.c\0";

static __slate_asm_string_virtqueue_reset_2: [u8; 1] = *b"\0";

static __slate_asm_string_virtqueue_reset_3: [u8; 58] = *b"/home/takashi/c-corpus/linux/drivers/virtio/virtio_ring.c\0";

static __slate_asm_string_virtqueue_resize_3: [u8; 1] = *b"\0";

static __slate_asm_string_virtqueue_resize_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/drivers/virtio/virtio_ring.c\0";

static __slate_asm_string_vring_alloc_queue_6: [u8; 1] = *b"\0";

static __slate_asm_string_vring_alloc_queue_7: [u8; 58] = *b"/home/takashi/c-corpus/linux/drivers/virtio/virtio_ring.c\0";
